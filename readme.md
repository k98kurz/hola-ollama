# Hola

An extensible harness for Ollama to enable development of robust, local AI agents
with deterministic execution guarantees and reasonable security measures.

## Features / Status

- [x] Finite State Machines for workflows
- [ ] Basic RAG prompt generation
- [ ] Vector db extension for RAG prompt generation
- [ ] Context gates definable at each workflow step
- Tool system
  - [x] Tool calling for tool-enabled models
  - [x] XML-based workaround for models that do not natively support tools
  - A few bundled tools:
    - [x] FS IO: grep, find, ls, read, write, edit, append, rm, mv, cp
    - [x] Context Management: memorize, recall
    - [ ] Arbitrary Computation: calculate, python (sandboxed)
    - [ ] Orchestration:
      - All phases: launch_subagent (launches a subagent with a prompt)
      - PLAN phase: view_tasks, add_task, move_task, delete_task
      - EXECUTE phase: complete_task
      - REVIEW phase: reject_task, accept_task
    - No bash/cmd tool. The goal is to enforce ACL so agents can't run wild, so
      there will be a handful of useful tools to obviate the need for bash/cmd.
  - [x] Monadic tool call support (i.e. chaining of tools), passing outputs as
    inputs, e.g.
    - `memorize("original-file", read("file.txt"))`
    - `write("backup.txt", recall("original-file"))`
  - [x] Tool system allows template interpolation of memorized values, e.g.
    - `python("python code ${original-file}")`
- [ ] Bundled top-level workflows: general and research
- [ ] Bundled dedicated subagents: execute and discover
- [ ] Benchmarking tools for testing models
- [ ] Ability to include multiple models within an agentic workflow
- [x] Rigid but extensible permission system
- [ ] Plugin system for tools
- [ ] Skill system supporting skill installation in hola-specific locations as well
  as .agents/, .claude/, .opencode/, .cursor/ within the CWD
- [ ] CLI tool for launching a session with a specified workflow/agent
- [ ] File tree snapshot system to enable rolling back changes
- [ ] Optional long-term, inter-session memory systems (e.g. soul.md)
- [ ] Optional daemon with scheduled workflows/agents
- [ ] Optional TUI

## Testing

First, clone the repo, set up the virtualenv, and install requirements.

```bash
git clone ...
python -m venv venv/
source venv/bin/activate
pip install -r requirements.txt
```

For windows, replace `source venv/bin/activate` with `source venv/Scripts/activate`.

Then run the test suite with the following:

```bash
find tests -name test_*.py -print -exec {} \;
```

or

```bash
python -m unittest
```

There are currently 179 tests.

## Contributing

Check out the [Pycelium discord server](https://discord.gg/b2QFEJDX69). If you
experience a problem, please discuss it on the Discord server. All suggestions
for improvement are also welcome, and the best place for that is also Discord.
If you experience a bug and do not use Discord, open an issue or discussion on
Github.

## Pycelium Reciprocal License

Copyright (c) 2026 Jonathan Voss (k98kurz) / The Pycelium Company

This software is licensed under the following conditions:

1. Free Use: This software is free to use without payment or fee.

2. Redistribution: This software may be distributed, redistributed, packaged,
and included in other software projects or products, whether as source code or
in any compiled or compressed form, so long as all copyright notices and license
terms are preserved and included. In particular, derivative works must attribute
the original project (and any forks if relevant) so that the end users/customers
of a derivative work or a service provided in connection with a derivative work
are made aware of the original project, applicable copyrights, and these license
terms.

3. Modifications: Any modifications or improvements made to the software source
code must be made available as source code and distributed with this license.

4. Contributions: Modifications and/or improvements may be contributed to the
project. The project maintainer(s) may evaluate such modifications and/or
improvements for fitness for potential integration into the original project.
Contributors maintain their copyright over such modifications and improvements,
and all such accepted contributions will be acknowledged via a contributor
credits section unless the contributor explicitly waives this right.

5. Relicensing: The maintainer(s) of the original project code repository may
relicense this software under different license terms so long as all copyright
holders and contributors, or their heirs or assigns, have been provided 90 days
notice and do not object.

6. Disclaimer: This software is provided "as is". The copyright holders and
contributors who license this software disclaim all warranties with regard to
this software including all implied warranties of merchantability and fitness.
In no event will the copyright holders and contributors be liable for any
special, direct, indirect, or consequential damages or any damages whatsoever
resulting from loss of use, data or profits, whether in an action of contract,
negligence or other tortious action, arising out of or in connection with the
use or performance of this software.
