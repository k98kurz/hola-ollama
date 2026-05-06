# Hola

An extensible harness for Ollama to enable development of robust, local AI agents.

## Features

- Finite State Machines for workflows
- Tool system
  - Tool calling for tool-enabled models
  - Tool workaround for models that do not natively support tools
  - A few bundled tools: grep, ls, read, write, edit, append, memorize, recall, python (sandboxed)
  - Tool system allows chaining of tools, passing outputs as inputs, e.g.
    - `memorize("original-file", read("file.txt"))`
    - `write("backup.txt", recall("original-file"))`
  - Tool system allows template interpolation of memorized values, e.g.
    - `python("python code ${original-file}")
- Benchmarking tools for testing models
- Ability to include multiple models within an agentic workflow
- Context gates definable at each workflow step

