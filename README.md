# pepdistill

Fast spectral-library generation by **distilling** AlphaPeptDeep into small,
hardware-friendly student models.

The teacher models (AlphaPeptDeep) are accurate but heavy. `pepdistill` trains a
compact student to mimic the teacher's MS2 fragment intensities, retention time, and
CCS — using **only a FASTA file** as input. No experimental spectra required: the
teacher generates the training labels from in-silico digests.

- **No LSTMs.** The student is a Transformer-encoder or dilated-CNN — both parallelize
  well on CPU/GPU and export cleanly to ONNX.
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
uv sync                     # core (torch, pandas, typer) + the pepdistill_rs Rust ext
uv sync --extra teacher     # + AlphaPeptDeep teacher (peptdeep)
uv sync --extra onnx        # + ONNX export & onnxruntime inference
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

[train]                       # fine-tune on real spectra (PROSPECT pool, streamed)
enabled = true
meta   = "TUM_third_pool_meta_data.parquet"
zip    = "TUM_third_pool.zip"
shards = [0]
epochs = 60

[export]                      # ONNX
enabled = true

[bench]                       # library throughput
enabled = true
fasta   = "proteome.fasta"
```

```bash
pepdistill run run.toml                 # -> work/model.ckpt, work/model.onnx, work/summary.json
pepdistill run run.toml --no-train      # pretrain only (disable any stage inline)
```

Distillation-only (no real spectra): drop the `[train]` block or `enabled = false`. Any stage
can be turned off in the config or with `--no-pretrain` / `--no-train`.

### Predict a library

Standalone inference from a trained model (torch or ONNX):

```bash
pepdistill predict --model work/model.ckpt  --fasta proteome.fasta -o library.parquet --device auto
pepdistill predict --model work/model.onnx  --fasta proteome.fasta -o library.parquet --runtime onnx
```

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

### Predict one peptide in Rust (experimental)

A pure-Rust inference path mirrors the torch `predict` for a single peptide. Export the trained
checkpoint to a self-contained `.safetensors` artifact (weights + config/vocab/norm in the
metadata), then run the standalone binary:

```bash
pepdistill export-rust --model work/model.ckpt -o work/model.safetensors
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
               PROSPECT real-spectra source + tiered fsspec cache
  teacher/     Teacher ABC + FakeTeacher + PeptDeepTeacher (behind [teacher] extra)
  models/      student architectures (no LSTM) + presets + context conditioning + checkpoint I/O
  distill/     dataset, losses (MS2 cosine / RT+CCS MSE), Lightning regimes (distill +
               real-speclib context) and the config-driven pipeline
  predict/     library.py (reference) + fast.py (vectorized) + onnx.py (export/runtime)
  eval.py      val reduction (best example per library entry)
  util.py      device resolution (auto -> mps/cpu)
  cli.py       typer CLI (run + predict + export-rust)

rust/
  core/        chemistry + Peptide + tokenizer + student forward — the single source of truth,
               used by both the Python ext and the CLI
  cli/         pepdistill-cli: single-peptide predict binary (.safetensors -> JSON on stdout)
  src/lib.rs   pepdistill_rs: pyo3 extension exposing core to Python (a hard dependency)
```

## Student presets & speed

| preset | backbone | params | forward-only (torch CPU) | (torch MPS) |
|---|---|--:|--:|--:|
| `flash` | Transformer 1L/1-head/d32 | 19K | **~165k/s** | ~266k/s |
| `tiny`  | dilated CNN d48/2L | 35K | ~12k/s | ~214k/s |
| `small` | Transformer 2L/4-head/d64 | 119K | ~11k/s | ~89k/s |
| `base`  | Transformer 4L/8-head/d128 | 600K | — | — |

`flash` is the throughput pick on a no-GPU box: a single-head, single-layer transformer
clears 100k precursors/s on CPU alone. Swap or add presets in
`pepdistill/models/registry.py`.

**Inference is length-bucketed and fully vectorized** (`predict/fast.py`): precursors are
grouped by length into dense tensors (no padding), the model runs once per bucket, and
fragment m/z + the output table are built with pure numpy — no per-fragment Python. The
remaining end-to-end ceiling is the DataFrame/parquet assembly over millions of fragment
rows; that is the part a Rust runtime would take over. `forward_dense` (mask-free) is the
export/inference path and is what ONNX serializes.
