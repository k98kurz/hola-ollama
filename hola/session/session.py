"""
Session class for Hola session management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from hola.context import estimate_tokens
from hola.errors import type_assert, value_assert
from hola.helpers import get_memory_key
from hola.permissions import Permission
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
    AsyncFSM as FSM,
    WorkflowFSM,
    WorkflowState,
    WorkflowEvent,
)
from hola.types import Message as OMessage, Role
import packify
import os
import fnmatch


MAX_TASK_DESCRIPTION_LENGTH = 200


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
                self.sequence = last_m.sequence

        if not self.dbsession.state:
            self.state = WorkflowFSM()
            auto_save = self._create_fsm_autosave_hook()
            for transition in self.state.rules:
                self.state.add_transition_hook(transition, auto_save)

        task_phases = Task.query().equal(
            'session_id', self.session_id
        ).group('phase').select(['phase']).get()

        for row in task_phases:
            last_task = Task.query().equal('session_id', self.session_id).equal(
                'phase', row.data['phase']
            ).order_by('sequence', 'desc').first()
            self.task_sequences[row.data['phase']] = 0
            if last_task:
                self.task_sequences[row.data['phase']] = last_task.sequence

    def add_message(
            self, model: str, phase: str, message: OMessage,
            is_compaction: bool = False,
        ) -> None:
        """Add a message to conversation history."""
        self.sequence += 2
        m = DBMessage({
            'session_id': self.session_id,
            'sequence': self.sequence,
            'role': message.role.value,
            'model': model,
            'phase': phase,
            'content': message.content,
            'thinking': message.thinking,
            'is_compaction': is_compaction,
        })
        m.details = {'tool_calls': message.tool_calls}
        m.estimated_tokens = estimate_tokens(m)
        m.save()

    def ripple_renumber(self, start_sequence: int, increment: int = 2) -> None:
        """Pulls all messages with a sequence greater than or equal to
            `start_sequence`, adds `increment` to each sequence number,
            then saves them back to the database. Useful for resolving
            collisions during compactions. Updates the cached
            `self.sequence` so later appends cannot collide with the
            shifted messages.
        """
        type_assert(type(start_sequence) is int, 'start_sequence must be int')
        type_assert(type(increment) is int, 'increment must be int')
        messages = self.dbsession.messages().query().greater_or_equal(
            'sequence', start_sequence
        ).order_by('sequence', 'desc').get()
        for m in messages:
            m.sequence += increment
            m.save()
        self.sequence = max(
            self.sequence, max((m.sequence for m in messages), default=0)
        )

    def insert_compaction(
            self, model: str, phase: str, content: str,
            message_sequences: list[int],
        ) -> str:
        """Create a compaction message and mark the given message
            sequences as compacted by it. Returns the compaction
            message ID. The selection must contain at least one
            non-compaction message of the phase; otherwise a
            `ValueError` is raised. When the natural slot after the
            newest selected message is occupied, the messages at and
            above it are rippled forward by 2 to make room.
        """
        value_assert(
            message_sequences, 'message_sequences cannot be empty'
        )
        non_compaction = self.dbsession.messages().query().equal(
            'phase', phase
        ).is_in(
            'sequence', message_sequences
        ).equal(
            'is_compaction', False
        ).first()
        value_assert(
            non_compaction is not None,
            f"cannot compact only compaction messages for phase "
            f"'{phase}': {message_sequences}"
        )
        candidate = max(message_sequences) + 1
        collision = self.dbsession.messages().query().equal(
            'sequence', candidate
        ).first()
        if collision is not None:
            self.ripple_renumber(candidate)
        m = DBMessage({
            'session_id': self.session_id,
            'sequence': candidate,
            'role': Role.SYSTEM.value,
            'model': model,
            'phase': phase,
            'content': content,
            'is_compaction': True,
        })
        m.estimated_tokens = estimate_tokens(m)
        m.save()

        self.dbsession.messages().query().equal(
            'phase', phase
        ).is_in(
            'sequence', message_sequences
        ).update({'compacted_by': m.id})

        return m.id

    def get_history(
            self, phase: str | None = None, model: str | None = None,
            limit: int | None = None, only_since_last_compaction: bool = True
        ) -> list[OMessage]:
        """Get conversation history, optionally filtered by phase and
            model. `only_since_last_compaction` is only effective if
            `phase` is provided; by default, if `phase` is provided,
            `only_since_last_compaction` should be True and should
            return the most recent compaction for the phase and the
            messages that followed.
        """
        sqb = self.dbsession.messages().query().order_by('sequence', 'desc')
        last_compaction = None
        if model is not None:
            sqb.equal('model', model)
        if phase is not None:
            sqb.equal('phase', phase)
            if only_since_last_compaction:
                sqb2 = self.dbsession.messages().query()
                sqb2.equal('phase', phase).order_by('sequence', 'desc')
                if model is not None:
                    sqb2.equal('model', model)
                last_compaction = sqb2.equal('is_compaction', True).first()
                if last_compaction:
                    sqb.greater('sequence', last_compaction.sequence)

        results = sqb.take(limit) if limit else sqb.get()
        if last_compaction:
            results.append(last_compaction)

        msgs = [
            OMessage(
                m.role, m.content, thinking=m.thinking,
                tool_calls=m.details.get('tool_calls'),
                tokens=m.tokens, estimated_tokens=m.estimated_tokens,
                model=m.model or '', sequence=m.sequence,
                id=m.id, is_compaction=m.is_compaction,
                compacted_by=m.compacted_by, call_id=m.call_id,
            )
            for m in results
        ]
        msgs.reverse()
        return msgs

    def clear_history(
            self, phase: str | None = None, after_sequence: int | None = None
        ) -> None:
        """Clear conversation history, optionally only for a specific
            phase or beyond a specific sequence number.
        """
        sqb = self.dbsession.messages().query()
        if phase:
            sqb.equal('phase', phase)
        if after_sequence is not None:
            sqb.greater('sequence', after_sequence)
        sqb.delete()
        if phase is None and after_sequence is None:
            self.sequence = 0
        elif phase is None:
            self.sequence = after_sequence

    def add_tool_output(
            self, call_id: str, result: object, phase: str | None = None
        ) -> None:
        """Set tool output by call ID. Stored as Message with Role.TOOL.
            Phase is optional; when omitted the output is phase-less and
            excluded from phase-filtered history.
        """
        self.sequence += 2
        m = DBMessage({
            'session_id': self.session_id,
            'sequence': self.sequence,
            'role': Role.TOOL.value,
            'content': '',
            'call_id': call_id,
            'phase': phase,
        })
        m.details = {'result': result}
        m.estimated_tokens = estimate_tokens(m)
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

    def load_fsm_data(
            self, fsm_class: type[FSM] = WorkflowFSM, inject: dict = {}
        ) -> None:
        """Load FSM from storage with provided inject dependencies."""
        if not self.dbsession.state:
            raise SessionError("No FSM data found in session")
        self.state = fsm_class.unpack(
            self.dbsession.state, inject={**globals(), **inject}
        )

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

    def unset_metadata(self, *keys: str) -> None:
        """Unset the specified metadata keys if they exist."""
        if not any([k in self.dbsession.metadata for k in keys]):
            return
        metadata = self.dbsession.metadata
        for k in set(keys):
            if k in metadata:
                del metadata[k]
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

    @property
    def config(self) -> dict[str, object]:
        """The config dict for the session."""
        return self.dbsession.config

    @config.setter
    def config(self, val: dict) -> None:
        config = self.dbsession.config
        for k, v in val.items():
            config[k] = v
        self.dbsession.config = config
        self.dbsession.save()

    def unset_config(self, *keys: str) -> None:
        """Unset the given config keys if they exist."""
        if not any([k in self.dbsession.config for k in keys]):
            return
        config = self.dbsession.config
        for k in set(keys):
            if k in config:
                del config[k]
        self.dbsession.config = config
        self.dbsession.save()

    def set_token_baseline(self, phase: str, tokens: int) -> None:
        """Sets the token baseline for a given phase. Used in context
            management.
        """
        if 'token_baselines' not in self.state.context:
            self.state.context['token_baselines'] = {}
        self.state.context['token_baselines'][phase] = tokens
        self.save_fsm_data()

    def get_token_baseline(self, phase: str) -> int:
        """Gets the token baseline for a given phase. Used in context
            management.
        """
        token_baselines = self.state.context.get('token_baselines', {})
        return token_baselines.get(phase, 0)

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
        if len(description) > MAX_TASK_DESCRIPTION_LENGTH:
            raise ValueError(
                f"Task description cannot exceed "
                f"{MAX_TASK_DESCRIPTION_LENGTH} characters "
                f"(got {len(description)}). Shorten the description, "
                f"move details into the notes, and call add_task again."
            )
        self.task_sequences[phase] = self.task_sequences.get(phase, 0) + 1
        sequence = self.task_sequences[phase]

        memory_key = None
        if notes is not None and notes.strip():
            memory_key = get_memory_key(f"{phase} {sequence}")
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

        assignments = []
        if new_sequence < old_sequence:
            for seq in range(new_sequence, old_sequence):
                task = task_map.get(seq)
                if task:
                    old_seq = task.sequence
                    assignments.append((task, seq + 1))
                    changes.append(
                        f"{descriptions[old_seq]}: {old_seq}->{seq + 1}"
                    )
            assignments.append((moved_task, new_sequence))
            changes.append(
                f"{descriptions[old_sequence]}: "
                f"{old_sequence}->{new_sequence}"
            )
        else:
            for seq in range(old_sequence + 1, new_sequence + 1):
                task = task_map.get(seq)
                if task:
                    old_seq = task.sequence
                    assignments.append((task, seq - 1))
                    changes.append(
                        f"{descriptions[old_seq]}: {old_seq}->{seq - 1}"
                    )
            assignments.append((moved_task, new_sequence))
            changes.append(
                f"{descriptions[old_sequence]}: "
                f"{old_sequence}->{new_sequence}"
            )

        self._renumber_tasks(phase, assignments)

        self.task_sequences[phase] = max(max(task_map), new_sequence)

        return changes

    def _renumber_tasks(
            self, phase: str, assignments: list[tuple[Task, int]]
        ) -> None:
        """Apply new sequence numbers to tasks, migrating each task's
            memo entry to the key for its new sequence. Reads all memo
            values first so the shifts do not clobber one another, then
            writes all new keys before forgetting old ones so an
            interruption leaves orphaned duplicates, not lost data.
        """
        memos = {
            task.memory_key: self.recall(task.memory_key)
            for task, _ in assignments if task.memory_key
        }
        new_keys = set()
        for task, new_seq in assignments:
            if task.memory_key:
                old_key = task.memory_key
                value = memos[old_key]
                task.memory_key = get_memory_key(f"{phase} {new_seq}")
                new_keys.add(task.memory_key)
                if value is not None:
                    self.memorize(task.memory_key, value)
            task.sequence = new_seq
            task.save()
        for old_key in memos:
            if old_key not in new_keys:
                self.forget(old_key)

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
            if result is not None and result.strip():
                memory_key = get_memory_key(f"{phase} {sequence}")
                task.memory_key = memory_key
                memo = self.recall(memory_key)
                if not memo:
                    memo = {}
                if not isinstance(memo, dict):
                    memo = {"notes": memo}
                memo["completion"] = result
                self.memorize(memory_key, memo)
            task.save()

    def accept_task(self, phase: str, sequence: int) -> None:
        """Mark a task as accepted."""
        phase = phase.upper()
        task = self.get_task(phase, sequence)
        if task:
            task.status = TaskStatus.ACCEPTED
            task.save()

    def reject_task(self, phase: str, sequence: int, reason: str) -> None:
        """Mark a task as rejected and store reason in memory."""
        phase = phase.upper()
        task = self.get_task(phase, sequence)
        if task:
            task.status = TaskStatus.REJECTED
            if reason is not None and reason.strip():
                memory_key = get_memory_key(f"{phase} {sequence}")
                task.memory_key = memory_key
                memo = self.recall(memory_key)
                if not memo:
                    memo = {}
                if not isinstance(memo, dict):
                    memo = {"notes": memo}
                memo["rejection"] = reason
                self.memorize(memory_key, memo)
            task.save()

    def accept(self) -> None:
        """Mark the workflow as accepted by the reviewer agent."""
        self.state.context['status'] = 'accepted'
        if 'reject_reason' in self.state.context:
            del self.state.context['reject_reason']
        self.save_fsm_data()

    def reject(self, reason: str) -> None:
        """Mark the workflow as rejected with a reason by the review agent."""
        self.state.context['status'] = 'rejected'
        self.state.context['reject_reason'] = reason
        self.save_fsm_data()
