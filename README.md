# msspeculator

[![CI](https://github.com/TalusBio/msspeculator/actions/workflows/ci.yml/badge.svg)](https://github.com/TalusBio/msspeculator/actions/workflows/ci.yml)

`msspeculator` predicts peptide retention time, ion mobility, and MS2 spectra, then writes
spectral libraries for search. Use the Python commands for quick inference or the Rust path for
high-throughput library generation.

## Install

From a checkout:

```bash
uv sync --locked --no-dev
```

This installs the Python package and its Rust extension. A Rust toolchain with `cargo` must be on
`PATH` for the first sync.

## Generate a library

The Rust CLI includes a small built-in model, so no checkpoint is needed for a first run:

```bash
cargo run --release -p msspeculator-cli -- \
  library --model builtin:small-v0 \
  --fasta proteome.fasta --out library.tsv
```

Use an exported checkpoint instead:

```bash
uv run msspeculator export-rust --model model.ckpt -o model.safetensors
cargo run --release -p msspeculator-cli -- \
  library --model model.safetensors \
  --fasta proteome.fasta --out library.tsv
```

The output is DIA-NN TSV by default. Use a `.mzspeclib.txt` suffix for mzSpecLib text and add
`.gz` to compress either format. The CLI also supports single-peptide JSON prediction and a
model-health report. Run `cargo run -p msspeculator-cli -- --help` for all options.

Rust applications can call the same length-batched, queued inference path without spawning a
process. See the [Rust API guide](docs/rust-api.md).

## Python inference

Use a trained checkpoint to write a Parquet library:

```bash
uv sync --locked --no-dev --extra torch-cpu
uv run msspeculator predict --model model.ckpt \
  --fasta proteome.fasta -o library.parquet --device auto
```

## More

- [Training guide](docs/training.md): teacher distillation, prepared experimental data, and
  context-aware fine-tuning.
- [Development guide](docs/development.md): contributor setup, tests, linting, and CI.
- [Local runbook](docs/runbook-full-run.md): a complete local workflow.
- [Talus infrastructure](docs/talus-infrastructure.md): corpus preparation and production jobs.
- [Documentation index](docs/README.md): design notes, model details, reports, and roadmap.

## License

Apache License 2.0. See [LICENSE](LICENSE).
