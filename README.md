# Software-Testing-System
This repository contains a graph-based software testing system for Python programs. It is designed to automatically analyze Python code, construct behavioral models, and generate tests using constraint solving (via Z3). The system can test individual files or entire Python project directories.

### Features

1. Graph-Based Testing
- The system parses Python source code and builds control-flow graphs (CFGs).
- These graphs are used to reason about program structure, branching, and execution paths.
- Test cases are derived by exploring paths in the CFG.

2. Symbolic Execution + Z3 Solver
- Uses Z3 (SMT solver) to reason about symbolic constraints extracted from program paths.
- Generates feasible input values that satisfy those constraints, ensuring test coverage of complex scenarios.
- Helps detect unreachable paths, inconsistent conditions, and edge cases.

3. Works for Single Files and Directories
- Provide a single Python file to test a directory and the system will test all .py files inside it recursively.
