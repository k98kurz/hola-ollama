# Overview

This project is a WIP agentic harness system for Ollama. The goal is to allow
package users to build robust, custom agents/workflows using LLMs served by a
local Ollama service. This is a library, not an application. 

# Setup

```bash
source venv/bin/activate  # REQUIRED before ALL Python commands
```

# Testing & Building Convention

- NEVER, under ANY CIRCUMSTANCES, run `pip install` on ANYTHING. NEVER.
- NEVER, under ANY CIRCUMSTANCES, run `git checkout` on a file. NEVER.
- NEVER, under ANY CIRCUMSTANCES, run `git stash`. NEVER.
- NEVER, under ANY CIRCUMSTANCES, run `git reset` or `git reset --hard`. NEVER.
- When writing ephemeral test code, use a minimum number of print statements to
  conserve output tokens.
- NEVER create sessions with `portable=False` in tests. ALWAYS use `portable=True`.
- NO MOCKS. We want to prove that things actually work, not that we can pretend that
  they work. I'm sick of stupid mocks obscuring broken code.
- Use the temp/ directory in the project for temporary files.
- When testing ollama anything, use granite4.1:3b as the model.
- For the Build agent: use a todo list to track your work for anything non-trivial
  - If you see only `todowrite` and not `todoread`, immediately halt and report
- If you encounter anything requiring a major change in plans, stop and discuss
- Do not worry about backwards compatibility; this project is under development,
  incomplete, and has never been used anywhere.
- Do NOT use pytest. This project uses unittest EXCLUSIVELY.
- Use bare assert statements where possible in tests, not `self.assertX`
  - Exception: use `with self.assertRaises(SomeErrorClass) as e:` when testing errors
- Do NOT worry about backwards compatibility. This project is WIP and unused. Just
  make the changes.
- Do NOT waste time calling bash or python just to print out summaries.
- Do NOT use `python -m unittest discover tests` when making small changes in
  files covered by specific test files. Just execute the correct test file. DO
  NOT RUN ALL TESTS ALL THE TIME.
- Tests that call Ollama are EXPENSIVE. They burn CPU, take real time, and
  grind the user's hardware. Treat every Ollama-calling test run as a costly
  operation, not a free one.
- Run an Ollama-calling test file ONCE, then read the full output. Do NOT
  re-run the same file over and over with grep suffixes, different flags, or
  minor variations. If you need something from the output, it is already
  captured -- search it, don't re-run it.
- NEVER re-run Ollama-calling tests in a loop: not after every small edit, not
  during review, not "just to check". Make your changes, then run the relevant
  tests ONE time. If you already ran them and made only doc/comment changes,
  do NOT run them again.
- Write concise, meaningful tests, not a thousand pointless tests. Test important
  details, not irrelevant shite. Don't write a dozen test classes with 2 cases each;
  test classes should bundle related test methods together.
- Tries ALWAYS equals retries + 1.
- Do NOT make unauthorized edits to tickets. Update ONLY the ticket status.

## Agent Memory

`docs/memory.md` contains important architectural decisions, patterns, and gotchas
that have been learned through development. Agents should read this file before
starting work to avoid rediscovering information or making the same mistakes.

Note: This file is NOT for agents to automatically write to. It contains curated
knowledge maintained by the user.

# Code Style

## Line length

- Soft max of 80 for normal lines
- Hard max of 85 for code
- Hard max of 72 for multi-line docstring lines
- Hard max of 80 for single-line docstrings
- NO dangling whitespace
- Use scripts/linechecker.py

## Annotations

- This project uses Python 3.10+ style annotations.
- Do NOT use `Union`: use `type1 | type2`
- Do NOT use `Optional`: use `sometype | None`
- Use built-in generic types: `list[type]`, `dict[key, value]`, `set[type]`,
`tuple[type, ...]` instead of `List[type]`, `Dict[key, value]`, etc. (no import
needed)

## Multi-line function/method signatures

- If a function/method signature is long enough to be more than 80 chars, break it
into a multi-line signature
- Parameters should be on their own line, two indentations in
- More than one parameter can be specified per line
- Parameters following a `*` (keyword only) must start on new line
- Final closing parenthesis and return annotation should be on own line one
indentation in (level with function/method body)

Example:
```python
def some_function_with_lots_of_params(
        param1: bytes, param2: str, ..., *,
        kwarg_only_1: bool = False, ...
    ) -> dict[str, int]:
    ...
```

## Multi-line compound conditionals

- Surround with parentheses, with opening parenthesis one indentation layer in
- First conditional should be on same line as opening parenthesis, indented in once
from the opening parenthesis
- Each additional condition should be on its own line, indented in twice, with the
combining word ("and"/"or") at the beginning of the condition
- Closing "):" should be indented once, level with the body of the block

Example:
```python
if  (   some_long_condition_goes >= here_first
        and some_other_condition <= goes_here
    ):
    ...
```

## Docstrings

- Start docstring on same line as openinig quotation marks
- Do not add an empty line before the first and subsequent lines
- Use additional lines in docstring for word wrapping
- Indent additional lines in one level
- Quote code (variable names, types, etc) within the docstring using backticks
- Closing quotation marks on own line except for short, one-line docstrings
- File-level multiline docstrings should be unindented and should start on own line
- Do NOT write docstrings that are improperly indented
- Do NOT add empty lines in docstrings
- Do NOT add "Arg: " and "Returns: " lists
- ONLY include information that is not obvious from the annotations, and write in
full sentences

Examples:
```python
"""
This file does things. It has several functions that do things. Blah
blah blah this is a docstring.
"""

def some_function(x: int, y: int) -> float:
    """This does something. Returns `x / y`."""
    ...

def some_other_function(x: int, y: int, z: int) -> list[float]:
    """This does something else. Returns a list of floats, e.g.
        `x / y`, `z / y`, etc.
    """
    ...
```

## Imports

- Imports must always be at the top of the file, never in function/method bodies
- Group all `from package import whatever` before all `import package` statements
- Then group imports: relative first, stdlib first, then external dependencies,
  then internal modules
- Order imports alphabetically within each group
- No blank lines between import statements
- Imports come after file-level docstrings

Example:
```python
"""
This is a file-level docstring example. Below, the inline comments are
to explain the import pattern to agents; agents should not replicate
the inline comments in actual code.
"""

from .SubModule import SomeThing # relative
from hashlib import sha256 # stdlib
from sys import argv # stdlib
from crossconfig import get_config # external package
from hola.ollama import OllamaClient # internal, non-relative
import json # stdlib
import os # stdlib
import packify # external package
```

## Exports

- Do NOT use `__all__ = [...]`.
- Only import into an `__init__.py` what is expected to be exported for that module
- If something has to be imported and used temporarily within an `__init__.py` file,
  use `del Thing` at the end of the file
