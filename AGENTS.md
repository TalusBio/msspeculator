# AGENTS.md

Python trains. Rust predicts and writes spectral libraries. New inference work goes in Rust, and
reaches Python through the pyo3 seam when training needs a prediction; see
[ADR 0001](docs/adr/0001-inference-targets-portable-rust.md).

Naming something, or picking between two words for the same thing: [CONTEXT.md](CONTEXT.md) is the
glossary and it is opinionated.

Environment, checks, and CI: [docs/development.md](docs/development.md). Package boundaries and
data contracts: [docs/design.md](docs/design.md).

## Gotchas

Each of these has cost someone a confusing failure.

- **`cargo` needs the project interpreter**: `PYO3_PYTHON="$PWD/.venv/bin/python"`. Otherwise cargo
  takes whatever `python` is on PATH, and pyo3 refuses a version it does not support with an error
  that never mentions the venv. `task lint` and `task test` already set it; a bare `cargo` command
  does not.
- **Run `task sync-rust` after changing Rust, before running Python tests.** Python imports the
  installed extension, so a new binding stays invisible until it is rebuilt. The symptom is
  `AttributeError` on a function you just wrote.
- **Docstrings under `src/` run as tests** (`--doctest-modules`). An example in a docstring is a
  claim that gets checked, which is why they are worth writing.
- **The pre-commit hook runs format and lint, not tests.** A commit that passed its hook says
  nothing about `task test`.
