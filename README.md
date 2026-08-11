# pepdistill

Fast spectral-library generation by **distilling** AlphaPeptDeep into small,
hardware-friendly student models.

The teacher models (AlphaPeptDeep) are accurate but heavy. `pepdistill` trains a
compact student to mimic the teacher's MS2 fragment intensities, retention time, and
CCS. A FASTA is sufficient for teacher distillation; an optional second stage fine-tunes on
prepared experimental spectral libraries.

- **No LSTMs.** The maintained students are Transformer encoders, with a standalone
  safetensors-based Rust inference implementation for production library generation.
- **Deterministic splits.** Train/val/test is assigned by hashing the *stripped*
  sequence, so every mod-form and charge state of a peptide stays in one split
  (no leakage) and the split is stable across runs and dataset growth.
- **One config-driven pipeline.** A single `run` command walks toggleable Lightning stages
  — `pretrain` (teacher distill) → `train` (real spectra) → `export` → `bench` — from one
  TOML config. `predict` (library generation) is the separate inference command.
- **Learns from real spectra too.** Beyond teacher distillation, the student can fine-tune on
  real experimental libraries (PROSPECT) with per-run acquisition **context conditioning**:
  collision energy drives the MS2 context, chromatography drives the RT context.

## Install

```bash
uv sync                                  # inference + the pepdistill_rs Rust extension
uv sync --extra teacher --extra etl --extra tracking  # full cloud training + W&B tracking
uv sync --extra onnx                     # optional/deferred ONNX environment
```

Chemistry, the `Peptide` type, tokenization, and batch-encoding are single-sourced in Rust
(`rust/core`) and required at runtime — `pepdistill.chem` is just a re-export shim over the
`pepdistill_rs` extension. `pepdistill-rs` is declared as a `[tool.uv.sources]` path
dependency on `rust/` (build backend: maturin), so a plain `uv sync` compiles and installs
it automatically. This means **a Rust toolchain (`cargo`) must be on `PATH`** the first time
you sync; if `uv sync` ever can't drive the build in your environment, build the extension
manually into the venv instead:

```bash
uv run maturin develop -m rust/Cargo.toml --release
```

The `fake` teacher (deterministic, dependency-free) is always available for
development and tests; the real `alphapeptdeep` teacher needs the `teacher` extra.

## Documentation

Start with the [documentation index](docs/README.md) for the current design, model details,
full preparation/training runbook, and roadmap.

## Quick start

Describe the run in a TOML config, then run it:

```toml
# run.toml
out = "work"
preset = "small"
device = "auto"

[pretrain]                    # online teacher-distill warmup (digests streamed + labeled live)
enabled = true
teacher = "alphapeptdeep"     # or "fake"
nce_min = 20                  # collision-energy sweep (per-peptide) -> learned CE axis
nce_max = 40
passes  = 1                   # full enumerations of the digests
[[pretrain.sources]]
fasta = "proteome.fasta"

[train]                       # fine-tune on the prepared Parquet manifest
enabled = true
prepared_prefix = "s3://bucket/pepdistill-prepared/v1"
epochs = 60
num_workers = 0              # Polars decodes in-process with its native thread pool
model_threads = 4            # intra-op CPU threads used by the model

# [export] and [bench] are optional and off by default.
```

```bash
pepdistill run run.toml                 # -> work/{pretrain,latest,best,model}.ckpt + summary.json
pepdistill run run.toml --no-train      # pretrain only (disable any stage inline)
```

Real-spectrum training consumes immutable prepared Parquet chunks. The preparation workflow is
restartable and range-addressable; see [the full-run runbook](docs/runbook-full-run.md). The
trainer never downloads or extracts annotation archives.

For distillation only, set `[train] enabled = false` or pass `--no-train`. Omitting `[train]`
does **not** disable it: training is on by default and requires `prepared_prefix`. Any stage can
be turned off in the config or with its corresponding `--no-*` flag.

### Predict a library

Standalone Python inference from a trained checkpoint:

```bash
pepdistill predict --model work/model.ckpt  --fasta proteome.fasta -o library.parquet --device auto
```

The ONNX path remains available behind the `onnx` extra, but is deferred while the production
work focuses on the Rust library generator and search integration.

Condition MS2 on the run's acquisition context (torch runtime only, needs a checkpoint with a
saved `MSContextEncoder` — i.e. one trained with `[train]` or `[pretrain]` enabled). Give the
full factor grammar `INSTRUMENT::DETECTOR::FRAGMENTATION::ENERGY`, or just `--nce` as a
shorthand for "unknown instrument/detector/fragmentation, only the energy":

```bash
pepdistill predict --model work/model.ckpt --fasta proteome.fasta -o library.parquet \
  --ms-context "Lumos::FTMS::HCD::30"

pepdistill predict --model work/model.ckpt --fasta proteome.fasta -o library.parquet --nce 30
```

RT and CCS are always predicted context-free (RT through the runbook's neutral/iRT row); only
MS2 fragment intensities move with `--ms-context`/`--nce`.

### Generate a library or predict one peptide in Rust

A pure-Rust inference path mirrors the torch `predict` for a single peptide. Export the trained
checkpoint to a self-contained `.safetensors` artifact (weights + config/vocab/norm in the
metadata), then run the standalone binary. FASTA mode digests with trypsin, applies fixed
Carbamidomethyl@C plus up to one variable Oxidation@M by default, predicts charges 2–4, converts
predicted CCS to Bruker 1/K0, and writes a streaming DIA-NN TSV accepted by `timsseek`:

```bash
pepdistill export-rust --model work/model.ckpt -o work/model.safetensors
cargo run -q --release -p pepdistill-cli -- \
  --model work/model.safetensors --fasta proteome.fasta --out library.tsv \
  --ms-context "Lumos::FTMS::HCD::30"

timsseek --raw-inputs sample.d --speclib-uri library.tsv --output-uri search-results
```

Single-peptide JSON remains available:

```bash
cargo run -q --release -p pepdistill-cli -- \
  --model work/model.safetensors --peptide PEPTIDER --charge 2 \
  --ms-context "Lumos::FTMS::HCD::30"      # or --nce 30, or omit for base MS2
```

It prints one JSON object — precursor m/z, RT, CCS, and the fragment table (struct-of-arrays) —
to stdout. Transformer presets only; RT is the context-free iRT base unless `--chrom-context
NAME` selects a saved dataset. Parity with the torch path is measured by
`tests/test_rust_parity.py`.

`--peptide` accepts a modified sequence, not just a bare one — `PEPC[Carbamidomethyl@C]IDER`,
`[TMT6plex]PEPTIDER`, `PEPTIDER[Amidated]`, or a bare Dalton delta `PEP[+42.010565]TIDER`. A
named mod is encoded from its element composition, a `+`/`-` delta from its mass; the JSON's
`peptide` field echoes back how the input was parsed. Artifacts carry a `format_version` and
the reader refuses one it does not understand rather than guessing at missing tensors.

## Output

`library.parquet` is a tidy long-format spectral library — one row per retained
fragment:

| column | meaning |
|---|---|
| `modified_sequence`, `stripped_sequence`, `charge` | precursor identity |
| `precursor_mz`, `rt_pred`, `ccs_pred` | predicted precursor properties |
| `ion_type`, `fragment_charge`, `fragment_ordinal`, `fragment_mz` | fragment identity |
| `relative_intensity` | predicted intensity, normalized to the base peak |

## Package layout

```
pepdistill/
  chem.py      re-export shim over the Rust ext (pepdistill_rs): masses, m/z, fragment-ion
               series, and the Peptide class all live in Rust (rust/core); no chemistry
               logic left in Python
  data/        FASTA digest, precursor enumeration, deterministic split, tensor encoding,
               vendored PROSPECT catalog/shard index, prepared-data reader
  etl/         restartable Polars preparation, shard manifests, and finalization
  teacher/     Teacher ABC + FakeTeacher + PeptDeepTeacher (behind [teacher] extra)
  models/      student architectures (no LSTM) + presets + context conditioning + checkpoint I/O
  distill/     dataset, losses (MS2 cosine / RT+CCS MSE), Lightning regimes (distill +
               real-speclib context) and the config-driven pipeline
  predict/     library.py (reference) + fast.py (vectorized) + deferred ONNX export/runtime
  eval.py      val reduction (best example per library entry)
  util.py      device resolution (auto -> mps/cpu)
  cli.py       run/predict/export-rust plus prepare/prepare-status/prepare-finalize

rust/
  core/        chemistry + Peptide + tokenizer + student forward — the single source of truth,
               used by both the Python ext and the CLI
  cli/         pepdistill-cli: FASTA -> DIA-NN TSV and single-peptide JSON inference
  src/lib.rs   pepdistill_rs: pyo3 extension exposing core to Python (a hard dependency)
```

## Student presets & speed

| preset | backbone | parameters |
|---|---|--:|
| `flash` | Transformer 1L/1-head/d32 | 23,782 |
| `small-2h` | Transformer 2L/2-head/d64 | 132,358 |
| `small` | Transformer 2L/4-head/d64 | 132,358 |
| `base-4h` | Transformer 4L/4-head/d128 | 898,822 |
| `base`  | Transformer 4L/8-head/d128 | 898,822 |

Changing only the attention head count does not change the parameter count: it repartitions the
same model width. The paired presets are intended as controlled architecture sweeps.

The production Rust path is length-bucketed, uses tanh-GELU and optimized matrix
multiplication, and runs bounded parallel model workers feeding one writer thread. On the
documented 474,630-precursor / 16,650,774-transition fixture it improved from 168.62 seconds
single-core to 11.53–12.97 seconds with the worker pool (roughly 36.6k–41.2k precursors/s).
Treat these as workload-specific measurements, not forward-only model claims; reproduce them
on target hardware before capacity planning. Output order is unspecified.
