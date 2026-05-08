# Overview

This project is a WIP agentic harness system for Ollama. The goal is to allow
package users to build robust, custom agents/workflows using LLMs served by a
local Ollama service. This is a library, not an application. 

# Setup

```bash
source venv/bin/activate  # REQUIRED before all Python commands
pip install -e .          # Install package for most manual testing
```

# Testing & Building Convention

1. When writing ephemeral test code, use a minimum number of print statements to
  conserve output tokens.
2. When testing ollama anything, use granite4.1:3b as the model.
3. For the Build agent: use todo lists to track your work for anything non-trivial
4. If you encounter anything requiring a major change in plans, stop and discuss
5. Do not worry about backwards compatibility; this project is under development,
  incomplete, and has never been used anywhere.

# Code Style

## Line length

- Soft max of 80 for normal lines
- Hard max of 85 for code and 72 for docstrings
- NO dangling whitespace

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
- Then group imports: stdlib first, then external dependencies, then ideagen internal modules
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

from hashlib import sha256 # stdlib
from sys import argv # stdlib
from crossconfig import get_config # external package
from ideagen.ollama import run_ollama # internal module
import json # stdlib
import os # stdlib
import packify # external package
```

## Exports

- Do NOT use `__all__ = [...]`.
- Only import into an `__init__.py` what is expected to be exported for that module
- If something has to be imported and used temporarily within an `__init__.py` file,
  use `del Thing` at the end of the file
