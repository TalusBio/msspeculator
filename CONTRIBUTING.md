# Contributing

Start with the [development guide](docs/development.md) for the environment and checks.
Coding agents should read [AGENTS.md](AGENTS.md) first; it is short, and the gotchas in it are
ones humans hit too.

Dependencies and package sources belong in `pyproject.toml`; regenerate `uv.lock` with `uv lock`.
Keep documentation current with behavior, and update generated reports through their named tool
rather than editing measurements by hand.

Bug reports should include the command, configuration with credentials removed, platform, Python
and Rust versions, and the smallest reproducible input. Never attach private spectra, credentials,
or object-store URLs you are not authorized to share.

Contributions are accepted under the repository's [Apache License 2.0](LICENSE).
