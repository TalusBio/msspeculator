# msspeculator

[![CI](https://github.com/TalusBio/msspeculator/actions/workflows/ci.yml/badge.svg)](https://github.com/TalusBio/msspeculator/actions/workflows/ci.yml)

`msspeculator` predicts peptide retention time, ion mobility, and MS2 spectra, then writes
spectral libraries for search. It includes Python training and data preparation code, plus a
standalone Rust inference path for production library generation.

The project supports two data sources. A fake teacher makes small, repeatable smoke runs. The
AlphaPeptDeep teacher supplies labels for FASTA digestion. Prepared experimental libraries can
then fine-tune the student with acquisition context.

## Install

```bash
uv sync --locked
uv sync --locked --extra torch-cpu --extra etl --extra tracking
```

The first command installs the core package and Rust extension. The second adds CPU training,
ETL, and experiment tracking. Use `--extra torch` for the CUDA-capable PyPI build on Linux and
`--extra teacher` when using AlphaPeptDeep. A Rust toolchain with `cargo` must be on `PATH`.

Contributor commands live in `Taskfile.yml`:

```bash
task format
task lint
task test
```

## Quick start

Create a small training config:

```toml
out = "runs/smoke"
preset = "flash"
device = "cpu"

[pretrain]
enabled = true
teacher = "fake"
passes = 1

[[pretrain.sources]]
fasta = "smoke.fasta"
enzyme = "trypsin"

[train]
enabled = false
```

Then run and export it:

```bash
uv run msspeculator run smoke.toml
uv run msspeculator export-rust --model runs/smoke/model.ckpt \
  -o runs/smoke/model.safetensors
```

For the complete local workflow, including prepared-data training, see the [local
runbook](docs/runbook-full-run.md). Talus deployment instructions are in the [infrastructure
runbook](docs/talus-infrastructure.md).

## Generate a library

The Rust CLI can use a built-in model or an exported checkpoint:

```bash
cargo run --release -p msspeculator-cli -- \
  library --model runs/smoke/model.safetensors \
  --fasta smoke.fasta --out library.tsv
```

Use `--out library.mzspeclib.txt.gz` for compressed mzSpecLib text. The CLI also has a
single-peptide `predict` command and `run-doctor` for the vendored iRT standards. Its full option
set is available with `msspeculator --help` and `cargo run -p msspeculator-cli -- --help`.

Rust applications can call the same high-throughput library writer without spawning the CLI. See
the [Rust API guide](docs/rust-api.md).

## Where things live

```text
src/msspeculator/data/       digestion, encoding, splits, prepared-data readers
src/msspeculator/etl/        restartable prepared-data conversion
src/msspeculator/teacher/    fake and AlphaPeptDeep teachers
src/msspeculator/models/     student architectures and checkpoints
src/msspeculator/distill/    training loops and pipeline configuration
src/msspeculator/predict/    Python inference
rust/core/                    chemistry, encoding, artifacts, Rust inference
rust/inference/               reusable FASTA orchestration and library writers
rust/cli/                     command-line wrapper
```

The [documentation index](docs/README.md) links to design notes, runbooks, generated reports, and
the roadmap. Contributions are described in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
