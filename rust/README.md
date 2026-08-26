# Rust crates

The Rust workspace contains the shared implementation used by the Python extension, the reusable
inference library, and the standalone CLI.

- `core/` (`msspeculator-core`) owns chemistry, `Peptide`, ProForma parsing, tokenization, batch
  encoding, artifact loading, and model inference.
- `inference/` (`msspeculator-inference`) owns FASTA digestion, bounded worker queues, and
  spectral-library writers. Rust applications can call it without spawning the CLI.
- `src/lib.rs` (`msspeculator_rs`) is the PyO3 binding. It exposes core types and tensor helpers
  to Python; it contains no separate chemistry implementation.
- `cli/` (`msspeculator-cli`) parses command-line arguments and calls the inference library.

## Build the Python extension

The root `pyproject.toml` declares this workspace as an editable runtime dependency. From the
repository root:

```bash
uv sync --locked
```

After a Rust change, rebuild it with:

```bash
uv run maturin develop -m rust/Cargo.toml --release
```

The standalone crates can be tested and linted with:

```bash
cargo test --manifest-path rust/Cargo.toml --workspace
cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings
```

See the [Rust API guide](../docs/rust-api.md) for dependency usage. See the [design
document](../docs/design.md) for the boundaries between Rust, Python, and the CLI.
