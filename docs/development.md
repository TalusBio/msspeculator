# Development

The project combines Python training and ETL with Rust chemistry and production inference.

## Set up

Install the locked development environment:

```bash
uv sync --locked --extra torch-cpu --extra teacher --extra etl --extra tracking
uv run pre-commit install
```

Dependencies and package sources belong in `pyproject.toml`. Regenerate `uv.lock` with `uv lock`.

## Checks

```bash
task format
task lint
task test
```

The checks cover Ruff, ast-grep, Rust formatting and clippy, Python tests, cross-language parity,
and the Rust workspace. CI runs the same boundaries on Ubuntu with CPU Torch wheels.

Rust changes need the project Python interpreter when building the PyO3 extension:

```bash
PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --manifest-path rust/Cargo.toml --workspace
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for change guidelines and bug-report details. See the
[design document](design.md) for package boundaries and data contracts.
