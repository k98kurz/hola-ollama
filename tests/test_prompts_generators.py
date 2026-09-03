"""
Test prompt generation for different workflow phases.
"""

from hola.prompts.generators import (
    interpolate_template,
    generate_prompt,
    generate_initial_planning_prompt,
    generate_planning_task_prompt,
    generate_planning_finalization_prompt,
    generate_executing_prompt,
    generate_reviewing_task_prompt,
    generate_planning_task_remediation_prompt,
    generate_reviewing_final_prompt,
    generate_planning_remediation_prompt,
    generate_compaction_prompt,
    generate_done_prompt,
)
import unittest


class TestInterpolateTemplate(unittest.TestCase):
    """Test the template interpolation helper."""
    def test_basic_substitution(self):
        result = interpolate_template("Hello $name", name="World")
        assert result == "Hello World", result

    def test_missing_placeholder_raises_keyerror(self):
        with self.assertRaises(KeyError):
            interpolate_template("Hello $name, $age")

    def test_extra_kwargs_ignored(self):
        result = interpolate_template("Hello $name", name="World", extra="ignored")
        assert result == "Hello World", result


class TestPromptGenerators(unittest.TestCase):
    """Test prompt generation functions."""
    # --- PLANNING generators ---
    def test_initial_planning_prompt(self):
        result = generate_initial_planning_prompt(request="build a website")
        assert "build a website" in result, result
        assert "workflow planning agent" in result, result

        with self.assertRaises(ValueError):
            generate_initial_planning_prompt()

    def test_planning_task_prompt(self):
        result = generate_planning_task_prompt(
            request="build a website", task="design the layout"
        )
        assert "build a website" in result, result
        assert "design the layout" in result, result
        assert "assigned a task" in result, result

        with self.assertRaises(ValueError):
            generate_planning_task_prompt(request="req")

    def test_planning_finalization_prompt(self):
        result = generate_planning_finalization_prompt(request="build a website")
        assert "build a website" in result, result
        assert "Finalize the plan" in result, result

        with self.assertRaises(ValueError):
            generate_planning_finalization_prompt()

    # --- EXECUTING generator ---
    def test_executing_prompt(self):
        result = generate_executing_prompt(
            request="build a website", task="implement login"
        )
        assert "build a website" in result, result
        assert "implement login" in result, result
        assert "execution agent" in result, result

        with self.assertRaises(ValueError):
            generate_executing_prompt(request="req")

    # --- REVIEWING generators ---
    def test_reviewing_task_prompt(self):
        result = generate_reviewing_task_prompt(
            request="build a website", task="review login"
        )
        assert "build a website" in result, result
        assert "review login" in result, result
        assert "review agent" in result, result

        with self.assertRaises(ValueError):
            generate_reviewing_task_prompt(request="req")
        with self.assertRaises(ValueError):
            generate_reviewing_task_prompt(task="t")

    def test_reviewing_final_prompt(self):
        result = generate_reviewing_final_prompt(request="build a website")
        assert "build a website" in result, result
        assert "accept" in result, result
        assert "reject" in result, result

        with self.assertRaises(ValueError):
            generate_reviewing_final_prompt()

    # --- REMEDIATION generators ---
    def test_planning_task_remediation_prompt(self):
        result = generate_planning_task_remediation_prompt(
            request="build a website", task="fix login",
            reject_reason="incomplete implementation",
        )
        assert "build a website" in result, result
        assert "fix login" in result, result
        assert "remediation" in result.lower(), result.lower()
        assert "incomplete implementation" in result, result

        with self.assertRaises(ValueError):
            generate_planning_task_remediation_prompt(
                request="req", task="t"
            )
    def test_planning_remediation_prompt(self):
        result = generate_planning_remediation_prompt(
            request="build a website", review="missed requirements"
        )
        assert "build a website" in result, result
        assert "missed requirements" in result, result
        assert "final review agent rejected" in result, result

        with self.assertRaises(ValueError):
            generate_planning_remediation_prompt(request="req")

    # --- COMPACTION generator ---
    def test_compaction_prompt(self):
        result = generate_compaction_prompt(request="build a website")
        assert "build a website" in result, result
        assert "<current_task>" not in result, result
        assert "compaction agent" in result, result

        result = generate_compaction_prompt(
            request="build a website", task="current task"
        )
        assert "current task" in result, result
        assert "<current_task>" in result, result
        assert result.index("</original_request>") < \
            result.index("<current_task>"), result

        with self.assertRaises(ValueError):
            generate_compaction_prompt()

    # --- DONE generator ---
    def test_done_prompt(self):
        result = generate_done_prompt(request="build a website")
        assert "build a website" in result, result
        assert "final summary agent" in result, result

        with self.assertRaises(ValueError):
            generate_done_prompt()

    # --- DISPATCH ---
    def test_generate_prompt_dispatch(self):
        pairs = [
            ("PLANNING", "", {"request": "req"},
             "workflow planning agent"),
            ("PLANNING", "TASK", {"request": "req", "task": "t"},
             "assigned a task"),
            ("PLANNING", "FINALIZATION", {"request": "req"},
             "Finalize the plan"),
            ("EXECUTING", "", {"request": "req", "task": "t"},
             "execution agent"),
            ("REVIEWING", "TASK", {"request": "req", "task": "t"},
             "review agent"),
            ("PLANNING", "TASK_REMEDIATION",
             {"request": "req", "task": "t", "reject_reason": "bad"},
             "remediation"),
            ("REVIEWING", "FINAL", {"request": "req"},
             "accept"),
            ("PLANNING", "RECOVER", {"request": "req", "review": "bad"},
             "final review agent rejected"),
            ("COMPACTION", "", {"request": "req"},
             "compaction agent"),
            ("DONE", "", {"request": "req"},
             "final summary agent"),
        ]
        for phase, subphase, kwargs, marker in pairs:
            result = generate_prompt(phase, subphase, **kwargs)
            assert marker in result, (
                f"{phase}/{subphase} should contain {marker!r}")

    def test_generate_prompt_missing(self):
        with self.assertRaises(ValueError):
            generate_prompt("NONEXISTENT")

        with self.assertRaises(ValueError):
            generate_prompt("PLANNING", "NONEXISTENT")

    # --- ADDENDUM on all generators ---
    def test_addendum_all_generators(self):
        cases = [
            (generate_initial_planning_prompt,
             "initial_planning_prompt_addendum",
             {"request": "req"}),
            (generate_planning_task_prompt,
             "planning_task_prompt_addendum",
             {"request": "req", "task": "t"}),
            (generate_planning_finalization_prompt,
             "planning_finalization_prompt_addendum",
             {"request": "req"}),
            (generate_executing_prompt,
             "executing_prompt_addendum",
             {"request": "req", "task": "t"}),
            (generate_reviewing_task_prompt,
             "reviewing_task_prompt_addendum",
             {"request": "req", "task": "t"}),
            (generate_planning_task_remediation_prompt,
             "planning_task_remediation_prompt_addendum",
             {"request": "req", "task": "t", "reject_reason": "bad"}),
            (generate_reviewing_final_prompt,
             "reviewing_final_prompt_addendum",
             {"request": "req"}),
            (generate_planning_remediation_prompt,
             "planning_recover_final_rejection_prompt_addendum",
             {"request": "req", "review": "bad"}),
            (generate_compaction_prompt,
             "compaction_prompt_addendum",
             {"request": "req"}),
            (generate_done_prompt,
             "done_prompt_addendum",
             {"request": "req"}),
        ]
        for func, kwarg, base_kwargs in cases:
            result = func(**base_kwargs, **{kwarg: "trailing hint"})
            assert "trailing hint" in result, f"{func.__name__} addendum failed"

    # --- CUSTOM TEMPLATE ---
    def test_custom_template_override(self):
        result = generate_initial_planning_prompt(
            request="test",
            initial_planning_prompt_template="Custom: $request",
        )
        assert result == "Custom: test", result

        result = generate_executing_prompt(
            request="r", task="t",
            executing_prompt_template="Execute: $request -> $task",
        )
        assert result == "Execute: r -> t", result

        result = generate_done_prompt(
            request="test",
            done_prompt_template="Done: $request",
        )
        assert result == "Done: test", result

        result = generate_compaction_prompt(
            request="test",
            compaction_prompt_template="Compact: $request ($task)",
        )
        assert result == "Compact: test (N/A)", result

        result = generate_compaction_prompt(
            request="test", task="t",
            compaction_prompt_template="Compact: $request\n"
            "<current_task>\n$task\n</current_task>",
        )
        assert result.count("<current_task>") == 1, result


if __name__ == '__main__':
    unittest.main()
