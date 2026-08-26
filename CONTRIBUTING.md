# Contributing

pepdistill uses Python and Rust. Install Python 3.11, uv, Cargo, and Task, then create the
development environment:

```bash
uv sync --locked --extra torch-cpu --extra teacher --extra etl --extra tracking
uv run pre-commit install
```

Use the CPU extra for local work unless you specifically need CUDA. Dependencies and package
sources belong in `pyproject.toml`; regenerate `uv.lock` with `uv lock`. Do not add package installs
or version pins to CI scripts.

Before submitting a change:

```bash
task format
task lint
task test
```

`task format` edits files. `task lint` and `task test` match the repository's Python/Rust quality
and test boundaries. Keep documentation current with behavior, and update generated reports through
their named tool rather than editing measurements by hand.

Bug reports should include the command, configuration with credentials removed, platform, Python
and Rust versions, and the smallest reproducible input. Never attach private spectra, credentials,
or object-store URLs you are not authorized to share.

Contributions are accepted under the repository's [Apache License 2.0](LICENSE).
