"""
Tests for orchestration tools task management functionality.
"""

from pathlib import Path
from hola.session import Session
from hola.session.storage import initialize
from hola.session.models import Task, TaskStatus
from hola.tools.core import list_tools
import shutil
import unittest


class TestOrchestrationToolsRegistration(unittest.TestCase):
    """Test cases for orchestration tools registration."""
    def test_tools_registered(self):
        """Test all orchestration tools are registered."""
        tools = list_tools()
        tool_names = [t.name for t in tools]

        expected_tools = [
            'view_tasks', 'add_task', 'move_task', 'delete_task',
            'complete_task', 'reject_task', 'accept_task'
        ]

        for tool_name in expected_tools:
            assert tool_name in tool_names, f"{tool_name} not registered"

    def test_tools_schema_valid(self):
        """Test orchestration tools have valid schema without session parameter."""
        tools = list_tools()
        tool_dict = {t.name: t for t in tools}

        expected_tools = [
            'view_tasks', 'add_task', 'move_task', 'delete_task',
            'complete_task', 'reject_task', 'accept_task'
        ]

        for tool_name in expected_tools:
            tool = tool_dict[tool_name]
            assert tool is not None, f"{tool_name} tool not found"

            params = tool.parameters.get('properties', {})
            required = tool.parameters.get('required', [])

            assert 'session' not in params, f"{tool_name} should not have session param"
            assert 'session' not in required, f"{tool_name} should not require session param"


class TestTaskModel(unittest.TestCase):
    """Test cases for Task database model."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("task_model_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_task_creation(self):
        """Test creating a task with required fields."""
        task = Task({
            'session_id': self.session.session_id,
            'sequence': 1,
            'status': 'PENDING',
            'phase': 'PLANNING',
            'description': 'Test task description',
        })
        task.save()
        assert task.id is not None, task.id
        assert task.status == TaskStatus.PENDING, task.status

    def test_task_status_enum_property(self):
        """Test task status property handles enum encoding/decoding."""
        task = Task({
            'session_id': self.session.session_id,
            'sequence': 1,
            'status': 'PENDING',
            'phase': 'PLANNING',
            'description': 'Test task',
        })
        task.save()
        assert task.status == TaskStatus.PENDING, task.status

        task.status = TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED, task.status

        task.status = 'REJECTED'
        assert task.status == TaskStatus.REJECTED, task.status

    def test_task_relation_to_session(self):
        """Test task belongs_to session relation."""
        task = Task({
            'session_id': self.session.session_id,
            'sequence': 1,
            'status': 'PENDING',
            'phase': 'REVIEWING',
            'description': 'Test task',
        })
        task.save()
        self.session.dbsession.tasks().reload()
        assert self.session.dbsession.tasks, self.session.dbsession.tasks
        assert self.session.dbsession.tasks[0].id == task.id
        assert task.session.id == self.session.session_id


class TestTaskSequenceManagement(unittest.TestCase):
    """Test cases for task sequence number management."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("sequence_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_sequence_numbers_start_at_one(self):
        """Test first task in phase gets sequence 1."""
        task = self.session.add_task('PLANNING', 'First task')
        assert task.sequence == 1, task.sequence

    def test_sequence_numbers_increment_per_phase(self):
        """Test sequence numbers increment monotonically per phase."""
        task1 = self.session.add_task('PLANNING', 'Task 1')
        assert task1.sequence == 1, task1.sequence
        assert self.session.task_sequences['PLANNING'] == 1

        task2 = self.session.add_task('PLANNING', 'Task 2')
        assert task2.sequence == 2, task2.sequence
        assert self.session.task_sequences['PLANNING'] == 2

        task3 = self.session.add_task('PLANNING', 'Task 3')
        assert task3.sequence == 3, task3.sequence
        assert self.session.task_sequences['PLANNING'] == 3

    def test_sequence_numbers_independent_per_phase(self):
        """Test each phase has independent sequence numbering."""
        planning_task = self.session.add_task('PLANNING', 'Plan task')
        executing_task1 = self.session.add_task('EXECUTING', 'Execute 1')
        executing_task2 = self.session.add_task('EXECUTING', 'Execute 2')
        reviewing_task = self.session.add_task('REVIEWING', 'Review task')

        assert planning_task.sequence == 1, planning_task.sequence
        assert executing_task1.sequence == 1, executing_task1.sequence
        assert executing_task2.sequence == 2, executing_task2.sequence
        assert reviewing_task.sequence == 1, reviewing_task.sequence

    def test_sequence_numbers_persist_across_session_reload(self):
        """Test sequence numbers reload correctly from database."""
        session_id = self.session.session_id
        self.session.add_task('PLANNING', 'Task 1')
        self.session.add_task('PLANNING', 'Task 2')
        self.session.add_task('EXECUTING', 'Exec 1')

        reloaded = Session(session_id, portable=True)
        assert reloaded.task_sequences
        assert 'PLANNING' in reloaded.task_sequences
        assert reloaded.task_sequences['PLANNING'] == 2
        assert 'EXECUTING' in reloaded.task_sequences
        assert reloaded.task_sequences['EXECUTING'] == 1

        task = reloaded.add_task('PLANNING', 'Task 3')
        assert task.sequence == 3, task.sequence
        assert reloaded.task_sequences['PLANNING'] == 3

        exec_task = reloaded.add_task('EXECUTING', 'Exec 2')
        assert exec_task.sequence == 2, exec_task.sequence
        assert reloaded.task_sequences['EXECUTING'] == 2


class TestTaskQueryMethods(unittest.TestCase):
    """Test cases for Session task query methods."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("query_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_get_task(self):
        """Test retrieving a task by phase and sequence."""
        self.session.add_task('PLANNING', 'Target task')
        task = self.session.get_task('PLANNING', 1)
        assert task is not None, task
        assert task.description == 'Target task', task.description

    def test_get_task_not_found(self):
        """Test get_task returns None for non-existent task."""
        task = self.session.get_task('PLANNING', 999)
        assert task is None, task

    def test_list_tasks_all(self):
        """Test listing all tasks without filters."""
        self.session.add_task('PLANNING', 'Plan 1')
        self.session.add_task('PLANNING', 'Plan 2')
        self.session.add_task('EXECUTING', 'Exec 1')
        self.session.add_task('REVIEWING', 'Review 1')

        tasks = self.session.list_tasks()
        assert len(tasks) == 4, len(tasks)

    def test_list_tasks_filtered_by_phase(self):
        """Test listing tasks filtered by phase."""
        self.session.add_task('PLANNING', 'Plan 1')
        self.session.add_task('PLANNING', 'Plan 2')
        self.session.add_task('EXECUTING', 'Exec 1')

        planning_tasks = self.session.list_tasks(phase='PLANNING')
        assert len(planning_tasks) == 2, len(planning_tasks)

        executing_tasks = self.session.list_tasks(phase='EXECUTING')
        assert len(executing_tasks) == 1, len(executing_tasks)

    def test_list_tasks_filtered_by_status(self):
        """Test listing tasks filtered by status."""
        self.session.add_task('PLANNING', 'Pending task')
        task = self.session.add_task('PLANNING', 'Completed task')
        task.status = TaskStatus.COMPLETED
        task.save()

        pending_tasks = self.session.list_tasks(status='PENDING')
        assert len(pending_tasks) == 1, len(pending_tasks)

        completed_tasks = self.session.list_tasks(status='COMPLETED')
        assert len(completed_tasks) == 1, len(completed_tasks)

    def test_list_tasks_filtered_by_phase_and_status(self):
        """Test listing tasks filtered by both phase and status."""
        self.session.add_task('PLANNING', 'Pending plan')
        task1 = self.session.add_task('PLANNING', 'Completed plan')
        task1.status = TaskStatus.COMPLETED
        task1.save()
        self.session.add_task('EXECUTING', 'Pending exec')
        task2 = self.session.add_task('EXECUTING', 'Completed exec')
        task2.status = TaskStatus.COMPLETED
        task2.save()

        pending_planning = self.session.list_tasks(
            phase='PLANNING', status='PENDING'
        )
        assert len(pending_planning) == 1, len(pending_planning)

        completed_executing = self.session.list_tasks(
            phase='EXECUTING', status='COMPLETED'
        )
        assert len(completed_executing) == 1, len(completed_executing)


class TestTaskLifecycle(unittest.TestCase):
    """Test cases for task lifecycle operations."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("lifecycle_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_add_task_with_notes(self):
        """Test adding task stores notes in memory."""
        task = self.session.add_task('PLANNING', 'Task with notes', 'Detailed notes here')
        assert task.memory_key is not None, task.memory_key
        assert task.memory_key == 'task_PLANNING_1_notes', task.memory_key

        notes = self.session.recall(task.memory_key)
        assert notes == 'Detailed notes here', notes

    def test_add_task_without_notes(self):
        """Test adding task without notes has no memory_key."""
        task = self.session.add_task('PLANNING', 'Simple task')
        assert task.memory_key is None, task.memory_key

    def test_add_task_empty_description_raises_error(self):
        """Test adding task with empty description raises ValueError."""
        with self.assertRaises(ValueError) as e:
            self.session.add_task('PLANNING', '')
        assert "cannot be empty" in str(e.exception).lower()

    def test_add_task_whitespace_description_raises_error(self):
        """Test adding task with whitespace-only description raises ValueError."""
        with self.assertRaises(ValueError) as e:
            self.session.add_task('PLANNING', '   ')
        assert "cannot be empty" in str(e.exception).lower()

    def test_complete_task_marks_completed(self):
        """Test completing task marks status as COMPLETED."""
        task = self.session.add_task('EXECUTING', 'Task to complete')
        self.session.complete_task('EXECUTING', 1, 'Result: success')

        updated = self.session.get_task('EXECUTING', 1)
        assert updated.status == TaskStatus.COMPLETED, updated.status

    def test_complete_task_stores_result_in_memory(self):
        """Test completing task stores result in memory."""
        task = self.session.add_task('EXECUTING', 'Task to complete')
        result = 'Final result data'
        self.session.complete_task('EXECUTING', 1, result)

        stored = self.session.recall('task_EXECUTING_1_result')
        assert stored == result, stored

    def test_reject_task_marks_rejected(self):
        """Test rejecting task marks status as REJECTED."""
        task = self.session.add_task('REVIEWING', 'Task to reject')
        self.session.reject_task('REVIEWING', 1, 'Invalid approach')

        updated = self.session.get_task('REVIEWING', 1)
        assert updated.status == TaskStatus.REJECTED, updated.status

    def test_reject_task_stores_reason_in_memory(self):
        """Test rejecting task stores reason in memory."""
        task = self.session.add_task('REVIEWING', 'Task to reject')
        reason = 'Implementation does not meet requirements'
        self.session.reject_task('REVIEWING', 1, reason)

        stored = self.session.recall('task_REVIEWING_1_rejection')
        assert stored == reason, stored

    def test_accept_task_marks_accepted(self):
        """Test accepting task marks status as ACCEPTED."""
        task = self.session.add_task('REVIEWING', 'Task to accept')
        self.session.accept_task('REVIEWING', 1)

        updated = self.session.get_task('REVIEWING', 1)
        assert updated.status == TaskStatus.ACCEPTED, updated.status

    def test_delete_task_removes_from_database(self):
        """Test deleting task removes it from database."""
        task = self.session.add_task('PLANNING', 'Task to delete')
        self.session.delete_task('PLANNING', 1)

        deleted = self.session.get_task('PLANNING', 1)
        assert deleted is None, deleted

    def test_delete_task_removes_memory_entry(self):
        """Test deleting task removes associated memory entry."""
        task = self.session.add_task('PLANNING', 'Task with notes', 'Important notes')
        memory_key = task.memory_key
        self.session.delete_task('PLANNING', 1)

        notes = self.session.recall(memory_key)
        assert notes is None, notes


class TestTaskMoveOperations(unittest.TestCase):
    """Test cases for task move and renumbering operations."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("move_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_move_task_backward(self):
        """Test moving task to earlier sequence."""
        self.session.add_task('PLANNING', 'Task 1')
        self.session.add_task('PLANNING', 'Task 2')
        task3 = self.session.add_task('PLANNING', 'Task 3')

        changes = self.session.move_task('PLANNING', 3, 1)
        assert len(changes) == 3, len(changes)

        reloaded1 = self.session.get_task('PLANNING', 1)
        reloaded2 = self.session.get_task('PLANNING', 2)
        reloaded3 = self.session.get_task('PLANNING', 3)

        assert reloaded1.description == 'Task 3', reloaded1.description
        assert reloaded2.description == 'Task 1', reloaded2.description
        assert reloaded3.description == 'Task 2', reloaded3.description

    def test_move_task_forward(self):
        """Test moving task to later sequence."""
        self.session.add_task('PLANNING', 'Task 1')
        task2 = self.session.add_task('PLANNING', 'Task 2')
        task3 = self.session.add_task('PLANNING', 'Task 3')

        changes = self.session.move_task('PLANNING', 2, 3)
        assert len(changes) == 2, len(changes)

        reloaded2 = self.session.get_task('PLANNING', 2)
        reloaded3 = self.session.get_task('PLANNING', 3)

        assert reloaded2.description == 'Task 3', reloaded2.description
        assert reloaded3.description == 'Task 2', reloaded3.description

    def test_move_task_to_same_position_no_change(self):
        """Test moving task to same position has no effect."""
        task = self.session.add_task('PLANNING', 'Task 1')

        changes = self.session.move_task('PLANNING', 1, 1)
        assert len(changes) == 0, len(changes)

        unchanged = self.session.get_task('PLANNING', 1)
        assert unchanged.description == 'Task 1', unchanged.description

    def test_move_task_returns_descriptive_changes(self):
        """Test move_task returns list of descriptive renumbering strings."""
        self.session.add_task('PLANNING', 'Task 1')
        self.session.add_task('PLANNING', 'Task 2')
        self.session.add_task('PLANNING', 'Task 3')

        changes = self.session.move_task('PLANNING', 3, 1)
        assert 'Task 3: 3->1' in changes, changes
        assert 'Task 1: 1->2' in changes, changes
        assert 'Task 2: 2->3' in changes, changes


class TestTaskPersistence(unittest.TestCase):
    """Test cases for task persistence across session reloads."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session_id = "persistence_test"
        self.session = Session(self.session_id, portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_tasks_persist_across_session_reload(self):
        """Test tasks are saved and retrieved correctly."""
        self.session.add_task('PLANNING', 'Persistent task 1')
        self.session.add_task('PLANNING', 'Persistent task 2')
        self.session.add_task('EXECUTING', 'Persistent exec')

        reloaded = Session(self.session_id, portable=True)

        tasks = reloaded.list_tasks()
        assert len(tasks) == 3, len(tasks)

        planning_tasks = reloaded.list_tasks(phase='PLANNING')
        assert len(planning_tasks) == 2, len(planning_tasks)

        task1 = reloaded.get_task('PLANNING', 1)
        assert task1.description == 'Persistent task 1', task1.description

    def test_task_status_persists(self):
        """Test task status changes persist across reload."""
        task = self.session.add_task('PLANNING', 'Status test')
        task.status = TaskStatus.COMPLETED
        task.save()

        reloaded = Session(self.session_id, portable=True)

        loaded_task = reloaded.get_task('PLANNING', 1)
        assert loaded_task.status == TaskStatus.COMPLETED, loaded_task.status

    def test_memory_entries_persist(self):
        """Test task-related memory entries persist."""
        task = self.session.add_task('PLANNING', 'Memory test', 'Important notes')
        memory_key = task.memory_key

        reloaded = Session(self.session_id, portable=True)

        notes = reloaded.recall(memory_key)
        assert notes == 'Important notes', notes


if __name__ == "__main__":
    unittest.main()
