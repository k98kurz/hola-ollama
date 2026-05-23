"""
Session class for Hola session management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from hola.types import Message as OMessage, Role
from hola.session.exceptions import (
    SessionDeleteError,
    SessionError,
    TempFileNotFoundError,
)
from hola.session.models import (
    Message as DBMessage,
    Memory,
    Session as DBSession,
    Task,
    TaskStatus,
)
from hola.session.storage import _get_session_temp_dir, initialize, _get_config
from hola.session.fsm import (
    FSM,
    WorkflowFSM,
    WorkflowState,
    WorkflowEvent,
)
from hola.permissions import Permission
import packify
import os
import fnmatch


def _match_path_with_glob(path: str, pattern: str) -> bool:
    """Match path against glob pattern, supporting ** for recursive matching."""
    path_parts = path.split('/')
    pattern_parts = pattern.split('/')

    i = 0
    j = 0

    while i < len(path_parts) and j < len(pattern_parts):
        if pattern_parts[j] == '**':
            if j + 1 == len(pattern_parts):
                return True
            while i < len(path_parts):
                if _match_segment(
                        path_parts[i], pattern_parts[j + 1]
                    ):
                    break
                i += 1
            if i == len(path_parts):
                return False
            i += 1
            j += 2
        elif _match_segment(path_parts[i], pattern_parts[j]):
            i += 1
            j += 1
        else:
            return False

    if j == len(pattern_parts) - 1 and pattern_parts[j] == '**':
        return True

    return i == len(path_parts) and j == len(pattern_parts)


def _match_segment(segment: str, pattern: str) -> bool:
    """Match a single path segment against a pattern."""
    return fnmatch.fnmatch(segment, pattern)


@dataclass(slots=True)
class Session:
    """Agent session with state management and persistence."""
    session_id: str
    portable: bool
    dbsession: DBSession = field(default=None)
    sequence: int = field(default=0)
    state: FSM | None = field(default=None)
    permissions_hook: Callable[[str, int, int], tuple[bool, list[str]]] | None = (
        field(default=None)
    )
    task_sequences: dict[str, int] = field(default_factory=dict)

    @property
    def cwd(self) -> str:
        """Current working directory."""
        return self.dbsession.cwd

    def __post_init__(self) -> None:
        initialize(self.portable)

        self.dbsession = DBSession.find(self.session_id) or DBSession.insert({
            'id': self.session_id,
            'cwd': os.getcwd(),
        })
        last_m_list = self.dbsession.messages().query().order_by(
            'sequence', 'desc'
        ).select(['sequence']).take(1)
        if last_m_list:
            last_m = last_m_list[0]
            if last_m.sequence:
                self.sequence = last_m.sequence + 1

        if not self.dbsession.state:
            self.state = WorkflowFSM()
            auto_save = self._create_fsm_autosave_hook()
            for transition in self.state.rules:
                self.state.add_transition_hook(transition, auto_save)

        phase_tasks = Task.query().equal(
            'session_id', self.session_id
        ).group('phase').select(['phase']).get()

        for task in phase_tasks:
            phase = task.data.get('phase') if hasattr(task, 'data') else task.phase
            last_task = Task.query().equal('session_id', self.session_id).equal(
                'phase', phase
            ).order_by('sequence', 'desc').first()
            self.task_sequences[phase] = last_task.sequence if last_task else 0

    def add_message(self, model: str, message: OMessage) -> None:
        """Add a message to conversation history."""
        self.sequence += 1
        m = DBMessage({
            'session_id': self.session_id,
            'sequence': self.sequence,
            'role': message.role.value,
            'model': model,
            'content': message.content,
            'thinking': message.thinking,
        })
        m.details = {
            'tool_name': message.tool_name,
            'tool_calls': message.tool_calls,
        }
        m.save()

    def get_history(
            self, model: str | None = None, limit: int = 10,
        ) -> list[tuple[str, OMessage]]:
        """Get conversation history, optionally filtered by model."""
        sqb = self.dbsession.messages().query().order_by('sequence', 'desc')
        if model is not None:
            sqb.equal('model', model)

        msgs = [
            (m.model, OMessage(m.role, m.content, thinking=m.thinking))
            for m in sqb.take(limit)
        ]
        msgs.reverse()
        return msgs

    def clear_history(self) -> None:
        """Clear conversation history."""
        sqb = self.dbsession.messages().query()
        sqb.delete()
        self.sequence = 0

    def add_tool_output(self, call_id: str, result: object) -> None:
        """Set tool output by call ID. Stored as Message with Role.TOOL."""
        m = DBMessage({
            'session_id': self.session_id,
            'role': Role.TOOL.value,
            'content': '',
            'call_id': call_id,
        })
        m.details = {'result': result}
        m.save()

    def get_tool_output(self, call_id: str) -> object | None:
        """Get tool output by call ID."""
        sqb = self.dbsession.messages().query().equal('call_id', call_id)
        call = sqb.first()
        if call:
            return call.details.get('result')
        return None

    def list_tool_outputs(self) -> dict[str, object]:
        """List all tool outputs."""
        outputs = {}
        sqb = self.dbsession.messages().query().equal('role', Role.TOOL.value)
        for m in sqb.get():
            call_id = m.call_id
            result = m.details.get('result')
            outputs[call_id] = result

        return outputs

    def _create_fsm_autosave_hook(self) -> Callable:
        session = self
        def auto_save(transition, context, data):
            if session.dbsession is None:
                return
            session.dbsession.state = session.state.pack()
            session.dbsession.save()
        return auto_save

    def set_fsm(self, fsm: FSM) -> None:
        """Replace the current FSM instance with a new one."""
        self.state = fsm

        auto_save = self._create_fsm_autosave_hook()
        for transition in self.state.rules:
            self.state.add_transition_hook(transition, auto_save)

        self.dbsession.state = self.state.pack()
        self.dbsession.save()

    def load_fsm_data(self, fsm_class: type[FSM], inject: dict = {}) -> None:
        """Load FSM from storage with provided inject dependencies."""
        if not self.dbsession.state:
            raise SessionError("No FSM data found in session")
        self.state = fsm_class.unpack(self.dbsession.state, inject=inject)

        auto_save = self._create_fsm_autosave_hook()
        for transition in self.state.rules:
            self.state.add_transition_hook(transition, auto_save)

    def save_fsm_data(self) -> None:
        """Save FSM state to storage."""
        if not self.state:
            raise SessionError("No FSM instance to save")
        self.dbsession.state = self.state.pack()
        self.dbsession.save()

    def set_metadata(self, key: str, value: object) -> None:
        """Set metadata key-value pair."""
        metadata = self.dbsession.metadata
        metadata[key] = value
        self.dbsession.metadata = metadata
        self.dbsession.save()

    def get_metadata(
            self, key: str, default: object = None
        ) -> object:
        """Get metadata value by key."""
        return self.dbsession.metadata.get(key, default)

    def list_metadata(self) -> dict[str, object]:
        """List all metadata."""
        return self.dbsession.metadata

    def memorize(self, key: str, value: object) -> None:
        """Memorize a key-value pair."""
        memory = self.dbsession.memories().query().equal('name', key).first()

        if memory:
            memory.content = value
            memory.save()
        else:
            memory = Memory({
                'id': Memory.generate_id(),
                'session_id': self.session_id,
                'name': key,
            })
            memory.content = value
            memory.save()

    def recall(self, key: str) -> object:
        """Recall a memorized value by key."""
        memory = self.dbsession.memories().query().equal('name', key).first()
        if memory:
            return memory.content
        return None

    def forget(self, key: str) -> None:
        """Forget a memorized key-value pair."""
        memory = self.dbsession.memories().query().equal('name', key).first()
        if memory:
            memory.delete()

    def list_memories(self) -> dict[str, object]:
        """List all memorized key-value pairs."""
        return {
            mem.name: mem.content
            for mem in self.dbsession.memories().query().get()
        }

    def create_temp_file(
            self, filename: str, content: bytes | str
        ) -> Path:
        """Create a temporary file in the session's temp directory.
            Returns the full path to the created file.
        """
        temp_dir = _get_session_temp_dir(self.session_id, self.portable)
        temp_dir.mkdir(parents=True, exist_ok=True)

        file_path = temp_dir / filename
        try:
            if isinstance(content, str):
                file_path.write_text(content)
            else:
                file_path.write_bytes(content)
        except OSError as e:
            raise SessionError(f"Failed to create temp file: {e}")

        return file_path

    def read_temp_file(self, filename: str) -> bytes:
        """Read a temporary file as bytes."""
        temp_dir = _get_session_temp_dir(self.session_id, self.portable)
        file_path = temp_dir / filename

        if not file_path.exists():
            raise TempFileNotFoundError(f"Temp file not found: {filename}")

        try:
            return file_path.read_bytes()
        except OSError as e:
            raise SessionError(f"Failed to read temp file: {e}")

    def delete_temp_file(self, filename: str) -> None:
        """Delete a temporary file."""
        temp_dir = _get_session_temp_dir(self.session_id, self.portable)
        file_path = temp_dir / filename

        if not file_path.exists():
            raise TempFileNotFoundError(f"Temp file not found: {filename}")

        try:
            file_path.unlink()
        except OSError as e:
            raise SessionError(f"Failed to delete temp file: {e}")

    def list_temp_files(self) -> list[str]:
        """List all temporary files in the session's temp directory."""
        temp_dir = _get_session_temp_dir(self.session_id, self.portable)

        if not temp_dir.exists():
            return []

        return [f.name for f in temp_dir.iterdir() if f.is_file()]

    def delete(self) -> None:
        """Delete session and its storage."""
        from hola.session.storage import delete_session
        delete_session(self.session_id, self.portable)
        self.dbsession = None

    def get_file_permission(self, path: Path | str) -> Permission:
        """Get effective permission for a path. Checks exact matches and
            glob patterns, applies inheritance, uses defaults, calls
            permission hook if access would be denied.
        """
        path_obj = Path(path).absolute()
        path_str = str(path_obj)
        perms = self.dbsession.permissions
        files = perms.get('files', {})
        temp_dir = _get_config(self.portable).path(
            f"sessions/{self.session_id}/temp"
        )
        temp_pattern = str(temp_dir / "**")

        def check_permission(requested_bit: int) -> tuple[bool, list[str]]:
            """Check if requested permission is granted. Returns
                (allowed, denial_reasons).
            """
            # Check exact matches first
            if path_str in files:
                perm = files[path_str]
                if (perm.value & requested_bit) != 0:
                    return (True, [])
                else:
                    reasons = [
                        f"Path {path_str} does not have {requested_bit} "
                        "permission"
                    ]
                    return (False, reasons)

            # Check glob patterns (more specific patterns first, first match wins)
            sorted_patterns = sorted(
                files.items(),
                key=lambda x: (-len(x[0]), x[0])
            )
            for pattern, perm in sorted_patterns:
                if _match_path_with_glob(path_str, pattern):
                    if (perm.value & requested_bit) != 0:
                        return (True, [])
                    else:
                        reasons = [
                            f"Pattern {pattern} denies {requested_bit} "
                            "permission"
                        ]
                        return (False, reasons)

            # Check inheritance (parent directories)
            check_path = path_obj.parent
            while check_path != check_path.parent:
                check_str = str(check_path)
                if check_str in files:
                    perm = files[check_str]
                    if (perm.value & requested_bit) != 0:
                        return (True, [])
                    else:
                        reasons = [
                            f"Parent {check_str} denies "
                            f"{requested_bit} permission"
                        ]
                        return (False, reasons)
                sorted_patterns = sorted(
                    files.items(),
                    key=lambda x: (-len(x[0]), x[0])
                )
                for pattern, perm in sorted_patterns:
                    if _match_path_with_glob(check_str, pattern):
                        if (perm.value & requested_bit) != 0:
                            return (True, [])
                        else:
                            reasons = [
                                f"Pattern {pattern} on parent denies "
                                f"{requested_bit} permission"
                            ]
                            return (False, reasons)
                check_path = check_path.parent

            # Apply default permissions
            temp_dir_str = str(temp_dir)
            if path_str.startswith(temp_dir_str):
                if (7 & requested_bit) != 0:
                    return (True, [])
                else:
                    reasons = [
                        f"Default temp permissions deny {requested_bit}"
                    ]
                    return (False, reasons)
            elif self.dbsession.cwd and path_str.startswith(
                    self.dbsession.cwd
                ):
                if (4 & requested_bit) != 0:
                    return (True, [])
                else:
                    reasons = [
                        f"Default CWD permissions deny {requested_bit}"
                    ]
                    return (False, reasons)
            else:
                reasons = [f"No permissions granted for {path_str}"]
                return (False, reasons)

        # Check all permission bits and build Permission object
        read_allowed, _ = check_permission(4)
        write_allowed, _ = check_permission(2)
        execute_allowed, _ = check_permission(1)

        if read_allowed or write_allowed or execute_allowed:
            return Permission.from_bits(read_allowed, write_allowed, execute_allowed)

        # Call permission hook if access is denied
        if self.permissions_hook is not None:
            requested = 7
            current = 0
            allowed, denial_reasons = self.permissions_hook(
                path_str, requested, current
            )
            if allowed:
                return Permission(7)

        # Default: no permissions
        return Permission(0)

    def set_permission(self, path: str, permission: Permission) -> None:
        """Add or update permission for a path pattern."""
        normalized_path = str(Path(path).absolute())
        perms = self.dbsession.permissions
        if 'files' not in perms:
            perms['files'] = {}
        perms['files'][normalized_path] = permission
        self.dbsession.permissions = perms
        self.dbsession.save()

    def remove_permission(self, path: str) -> None:
        """Remove permission entry for a path pattern."""
        normalized_path = str(Path(path).absolute())
        perms = self.dbsession.permissions
        if 'files' in perms and normalized_path in perms['files']:
            del perms['files'][normalized_path]
            self.dbsession.permissions = perms
            self.dbsession.save()

    def set_permission_hook(
            self,
            func: Callable[[str, int, int], tuple[bool, list[str]]] | None
        ) -> None:
        """Register or clear the permission hook."""
        self.permissions_hook = func

    @property
    def file_permissions(self) -> dict[str, dict[str, bool]]:
        """Get all file permissions as a simple dict format."""
        perms = self.dbsession.permissions.get('files', {})
        return {
            path: {'read': bool(perm.value & 4), 'write': bool(perm.value & 2)}
            for path, perm in perms.items()
        }

    def get_allowed_modules(self) -> list[str]:
        """Return list of allowed Python modules."""
        perms = self.dbsession.permissions
        return perms.get('modules', [])

    def set_allowed_modules(self, mods: list[str]) -> None:
        """Store allowed Python modules."""
        perms = self.dbsession.permissions
        perms['modules'] = mods
        self.dbsession.permissions = perms
        self.dbsession.save()

    def add_task(
            self, phase: str, description: str, notes: str | None = None
        ) -> Task:
        """Create a new task with pending status. Returns the created task."""
        phase = phase.upper()
        if not description or not description.strip():
            raise ValueError("Task description cannot be empty")
        self.task_sequences[phase] = self.task_sequences.get(phase, 0) + 1
        sequence = self.task_sequences[phase]

        memory_key = None
        if notes is not None:
            memory_key = f"task_{phase}_{sequence}_notes"
            self.memorize(memory_key, notes)

        task = Task({
            'id': Task.generate_id(),
            'session_id': self.session_id,
            'sequence': sequence,
            'status': TaskStatus.PENDING.value,
            'phase': phase,
            'description': description,
            'memory_key': memory_key,
        })
        task.save()
        return task

    def get_task(self, phase: str, sequence: int) -> Task | None:
        """Get a task by phase and sequence number."""
        phase = phase.upper()
        return Task.query().equal('session_id', self.session_id).equal(
            'phase', phase
        ).equal('sequence', sequence).first()

    def list_tasks(
            self, phase: str | None = None, status: str | None = None
        ) -> list[Task]:
        """List tasks, optionally filtered by phase and/or status."""
        qb = Task.query().equal('session_id', self.session_id)
        if phase:
            qb.equal('phase', phase.upper())
        if status:
            qb.equal('status', status.upper())
        return qb.order_by('sequence', 'asc').get()

    def move_task(
            self, phase: str, old_sequence: int, new_sequence: int
        ) -> list[str]:
        """Move a task to a new sequence number, renumbering affected tasks."""
        phase = phase.upper()
        if old_sequence == new_sequence:
            return []

        tasks = Task.query().equal('session_id', self.session_id).equal(
            'phase', phase
        ).order_by('sequence', 'asc').get()

        task_map = {t.sequence: t for t in tasks}
        if old_sequence not in task_map:
            return []

        moved_task = task_map[old_sequence]
        changes = []
        descriptions = {t.sequence: t.description for t in tasks}

        if new_sequence < old_sequence:
            for seq in range(new_sequence, old_sequence):
                task = task_map.get(seq)
                if task:
                    old_seq = task.sequence
                    task.sequence = seq + 1
                    task.save()
                    changes.append(f"{descriptions[old_seq]}: {old_seq}->{seq + 1}")
            moved_task.sequence = new_sequence
            moved_task.save()
            changes.append(
                f"{descriptions[old_sequence]}: {old_sequence}->{new_sequence}"
            )
        else:
            for seq in range(old_sequence + 1, new_sequence + 1):
                task = task_map.get(seq)
                if task:
                    old_seq = task.sequence
                    task.sequence = seq - 1
                    task.save()
                    changes.append(f"{descriptions[old_seq]}: {old_seq}->{seq - 1}")
            moved_task.sequence = new_sequence
            moved_task.save()
            changes.append(
                f"{descriptions[old_sequence]}: {old_sequence}->{new_sequence}"
            )

        return changes

    def delete_task(self, phase: str, sequence: int) -> None:
        """Delete a task by phase and sequence number."""
        phase = phase.upper()
        task = self.get_task(phase, sequence)
        if task:
            if task.memory_key:
                self.forget(task.memory_key)
            task.delete()

    def complete_task(
            self, phase: str, sequence: int, result: str | None = None
        ) -> None:
        """Mark a task as completed, optionally storing result in memory."""
        phase = phase.upper()
        task = self.get_task(phase, sequence)
        if task:
            task.status = TaskStatus.COMPLETED
            task.save()
            if result is not None:
                memory_key = f"task_{phase}_{sequence}_result"
                self.memorize(memory_key, result)

    def reject_task(self, phase: str, sequence: int, reason: str) -> None:
        """Mark a task as rejected and store reason in memory."""
        phase = phase.upper()
        task = self.get_task(phase, sequence)
        if task:
            task.status = TaskStatus.REJECTED
            task.save()
            memory_key = f"task_{phase}_{sequence}_rejection"
            self.memorize(memory_key, reason)

    def accept_task(self, phase: str, sequence: int) -> None:
        """Mark a task as accepted."""
        phase = phase.upper()
        task = self.get_task(phase, sequence)
        if task:
            task.status = TaskStatus.ACCEPTED
            task.save()
