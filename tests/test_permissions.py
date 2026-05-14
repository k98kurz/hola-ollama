"""
Tests for Permission system (Permission class and Session permission methods).
"""

from pathlib import Path
from hola.session import Session
from hola.session.storage import initialize
from hola.permissions import Permission
import os
import shutil
import unittest


class TestSessionPermissionSetup(unittest.TestCase):
    """Test cases for session permission setup."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("perm_test", portable=True)
        self.test_dir = Path.cwd() / ".hola"

    def tearDown(self):
        """Clean up after tests."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)


class TestPermissionCore(unittest.TestCase):
    """Test cases for Permission class core functionality."""
    def test_permission_creation_and_properties(self):
        """Test permission creation, properties, and from_bits."""
        perm_default = Permission()
        assert perm_default.value == 0, perm_default.value

        perm_value = Permission(7)
        assert perm_value.value == 7, perm_value.value

        perm_zero = Permission(0)
        assert perm_zero.read is False, perm_zero.read
        assert perm_zero.write is False, perm_zero.write
        assert perm_zero.execute is False, perm_zero.execute

        perm_read = Permission(4)
        assert perm_read.read is True, perm_read.read
        assert perm_read.write is False, perm_read.write
        assert perm_read.execute is False, perm_read.execute

        perm_write = Permission(2)
        assert perm_write.read is False, perm_write.read
        assert perm_write.write is True, perm_write.write
        assert perm_write.execute is False, perm_write.execute

        perm_execute = Permission(1)
        assert perm_execute.read is False, perm_execute.read
        assert perm_execute.write is False, perm_execute.write
        assert perm_execute.execute is True, perm_execute.execute

        perm_all = Permission(7)
        assert perm_all.read is True, perm_all.read
        assert perm_all.write is True, perm_all.write
        assert perm_all.execute is True, perm_all.execute

        perm_default_bits = Permission.from_bits()
        assert perm_default_bits.value == 0, perm_default_bits.value

        perm_read_bits = Permission.from_bits(read=True)
        assert perm_read_bits.value == 4, perm_read_bits.value

        perm_write_bits = Permission.from_bits(write=True)
        assert perm_write_bits.value == 2, perm_write_bits.value

        perm_execute_bits = Permission.from_bits(execute=True)
        assert perm_execute_bits.value == 1, perm_execute_bits.value

        perm_combo = Permission.from_bits(read=True, write=True, execute=True)
        assert perm_combo.value == 7, perm_combo.value

    def test_permission_serialization(self):
        """Test pack/unpack serialization works correctly."""
        for value in range(8):
            perm1 = Permission(value)
            packed = perm1.pack()
            assert isinstance(packed, bytes), type(packed)

            perm2 = Permission.unpack(packed)
            assert perm1.value == perm2.value, perm2.value
            assert perm1.read == perm2.read, perm2.read
            assert perm1.write == perm2.write, perm2.write
            assert perm1.execute == perm2.execute, perm2.execute


class TestGetFilePermission(TestSessionPermissionSetup):
    """Test cases for get_file_permission with various patterns."""
    def test_exact_matches_and_defaults(self):
        """Test exact path matches and default permissions."""
        path = "/tmp/test.txt"
        self.session.set_permission(path, Permission(4))
        perm = self.session.get_file_permission(path)
        assert perm.value == 4, perm.value

        path2 = "/tmp/test2.txt"
        self.session.set_permission(path2, Permission(7))
        perm2 = self.session.get_file_permission(path2)
        assert perm2.value == 7, perm2.value

        path3 = "/tmp/test3.txt"
        self.session.set_permission(path3, Permission(2))
        perm3 = self.session.get_file_permission(path3)
        assert perm3.value == 2, perm3.value

        perm_default = self.session.get_file_permission("/unconfigured/path.txt")
        assert perm_default.value == 0, perm_default.value

        temp_dir = Path.cwd() / ".hola" / "sessions" / "perm_test" / "temp"
        temp_file = temp_dir / "test.txt"
        perm_temp = self.session.get_file_permission(str(temp_file))
        assert perm_temp.value == 7, perm_temp.value

        perm_cwd = self.session.get_file_permission(os.getcwd())
        assert perm_cwd.value == 4, perm_cwd.value

        perm_external = self.session.get_file_permission("/external/path/file.txt")
        assert perm_external.value == 0, perm_external.value

    def test_glob_patterns(self):
        """Test glob pattern matching, specificity, and wildcards."""
        pattern = "/tmp/**/*.txt"
        self.session.set_permission(pattern, Permission(7))

        perm_match = self.session.get_file_permission("/tmp/subdir/file.txt")
        assert perm_match.value == 7, perm_match.value

        perm_nomatch = self.session.get_file_permission("/tmp/subdir/file.md")
        assert perm_nomatch.value == 0, perm_nomatch.value

        pattern1 = "/tmp/**/*.txt"
        pattern2 = "/tmp/**/*"
        self.session.set_permission(pattern1, Permission(4))
        self.session.set_permission(pattern2, Permission(2))

        perm_specific = self.session.get_file_permission("/tmp/file.txt")
        assert perm_specific.value == 4, perm_specific.value

        self.session.set_permission("/var/test.*", Permission(6))
        perm_wildcard = self.session.get_file_permission("/var/test.txt")
        assert perm_wildcard.value == 6, perm_wildcard.value

        self.session.set_permission("/var/file?.txt", Permission(5))
        perm_question = self.session.get_file_permission("/var/file1.txt")
        assert perm_question.value == 5, perm_question.value

        exact_path = "/var/exact.txt"
        glob_pattern = "/var/**/*.txt"
        self.session.set_permission(glob_pattern, Permission(4))
        self.session.set_permission(exact_path, Permission(7))

        perm_exact = self.session.get_file_permission(exact_path)
        assert perm_exact.value == 7, perm_exact.value

    def test_permission_inheritance(self):
        """Test permission inheritance from parent directories."""
        parent = "/tmp/test_dir"
        self.session.set_permission(parent, Permission(6))
        perm = self.session.get_file_permission("/tmp/test_dir/file.txt")
        assert perm.value == 6, perm.value

        grandparent = "/tmp"
        self.session.set_permission(grandparent, Permission(7))
        self.session.set_permission(parent, Permission(4))

        perm_closest = self.session.get_file_permission("/tmp/test_dir/file.txt")
        assert perm_closest.value == 4, perm_closest.value

        self.session.set_permission("/tmp/**", Permission(6))
        perm_glob = self.session.get_file_permission("/tmp/subdir/file.txt")
        assert perm_glob.value == 6, perm_glob.value


class TestPermissionManagement(TestSessionPermissionSetup):
    """Test cases for set/remove permission and persistence."""
    def test_set_and_remove_permissions(self):
        """Test setting, updating, and removing permissions."""
        path = "/tmp/test.txt"
        self.session.set_permission(path, Permission(7))
        perm = self.session.get_file_permission(path)
        assert perm.value == 7, perm.value

        self.session.set_permission(path, Permission(4))
        perm2 = self.session.get_file_permission(path)
        assert perm2.value == 4, perm2.value

        self.session.remove_permission(path)
        perm3 = self.session.get_file_permission(path)
        assert perm3.value == 0, perm3.value

        pattern = "/tmp/**/*.py"
        self.session.set_permission(pattern, Permission(5))
        perm_glob = self.session.get_file_permission("/tmp/test/script.py")
        assert perm_glob.value == 5, perm_glob.value

        self.session.remove_permission(pattern)
        perm_glob2 = self.session.get_file_permission("/tmp/test/script.py")
        assert perm_glob2.value == 0, perm_glob2.value

        paths = ["/tmp/file1.txt", "/tmp/file2.txt", "/tmp/file3.txt"]
        for i, path in enumerate(paths):
            self.session.set_permission(path, Permission(i))

        for i, path in enumerate(paths):
            perm_multi = self.session.get_file_permission(path)
            assert perm_multi.value == i, perm_multi.value

    def test_permissions_persist_across_sessions(self):
        """Test permissions persist when session is reloaded."""
        path = "/tmp/persist.txt"
        self.session.set_permission(path, Permission(6))

        from hola.session import get_session
        session2 = get_session("perm_test", portable=True)

        perm = session2.get_file_permission(path)
        assert perm.value == 6, perm.value

        modules = ["math", "json", "re"]
        self.session.set_allowed_modules(modules)

        session3 = get_session("perm_test", portable=True)

        retrieved = session3.get_allowed_modules()
        assert retrieved == modules, retrieved

        session3.delete()


class TestPermissionFeatures(TestSessionPermissionSetup):
    """Test cases for permission hooks and allowed modules."""
    def test_permission_hook(self):
        """Test permission hook allows/denies access and receives correct params."""
        hook_called = []

        def hook_func(path, requested, current):
            hook_called.append(True)
            return (True, [])

        self.session.set_permission_hook(hook_func)
        perm_allow = self.session.get_file_permission("/external/file.txt")

        assert len(hook_called) > 0, hook_called
        assert perm_allow.value == 7, perm_allow.value

        def hook_deny(path, requested, current):
            return (False, ["Hook denied access"])

        self.session.set_permission_hook(hook_deny)
        perm_deny = self.session.get_file_permission("/external/file2.txt")

        assert perm_deny.value == 0, perm_deny.value

        captured_params = []

        def hook_params(path, requested, current):
            captured_params.append((path, requested, current))
            return (False, [])

        self.session.set_permission_hook(hook_params)
        test_path = "/external/test.txt"
        self.session.get_file_permission(test_path)

        assert len(captured_params) == 1, captured_params
        path, requested, current = captured_params[0]
        assert path == test_path, path
        assert requested == 7, requested
        assert current == 0, current

        hook_not_called = []

        def hook_check(path, requested, current):
            hook_not_called.append(True)
            return (True, [])

        self.session.set_permission_hook(hook_check)
        self.session.set_permission("/tmp/test.txt", Permission(7))
        perm_skip = self.session.get_file_permission("/tmp/test.txt")

        assert len(hook_not_called) == 0, hook_not_called
        assert perm_skip.value == 7, perm_skip.value

        assert self.session.permissions_hook is not None
        self.session.set_permission_hook(None)
        assert self.session.permissions_hook is None

    def test_allowed_modules(self):
        """Test setting, retrieving, and persisting allowed modules."""
        modules_default = self.session.get_allowed_modules()
        assert isinstance(modules_default, list), type(modules_default)
        assert len(modules_default) == 0, len(modules_default)

        modules = ["math", "json", "re"]
        self.session.set_allowed_modules(modules)
        retrieved = self.session.get_allowed_modules()
        assert retrieved == modules, retrieved

        modules2 = ["sys", "os"]
        self.session.set_allowed_modules(modules2)
        retrieved2 = self.session.get_allowed_modules()
        assert retrieved2 == modules2, retrieved2

        perms = self.session.dbsession.permissions
        assert 'modules' in perms, perms
        assert perms['modules'] == modules2, perms['modules']


if __name__ == "__main__":
    unittest.main()
