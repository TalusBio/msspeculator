# Runbook — full preparation and training run

**Updated 2026-08-10.** A "full run" = `pepdistill run <toml>` over the `RunConfig` stages
`pretrain → train → export → bench` (each toggleable), then predict. This runbook covers
pretrain + train (the two on-by-default stages), library generation, and a search-engine smoke
test.

## 0. Prerequisites

```bash
uv sync --extra teacher --extra etl  # teacher + Polars/S3 ETL; builds Rust (needs cargo)
```

Inputs:
- **Pretrain FASTA** — a proteome to digest and stream. The cloud staging helper downloads and
  caches the pinned E. coli K-12 reference proteome from UniProt when it is absent.
- **Prepared PROSPECT catalog** — the vendored Zenodo catalog and compressed shard index define
  the inputs. The configured S3 cache is only a read-through intermediate: a missing object is
  fetched from Zenodo and repopulates the cache. Deleting the cache must not change correctness.

## 1. Prepare the real-data assets

`runs/prepare-full.toml` selects all non-test archives from `prospect`, `tmt`, `multi_ptm`, and
`tmt_ptm`; the separately labelled `test_ptm` record is excluded. First inspect the global shard
count, then distribute any disjoint half-open ranges:

```bash
pepdistill prepare-status runs/prepare-full.toml --count-only
pepdistill prepare runs/prepare-full.toml --range 0:518
pepdistill prepare runs/prepare-full.toml --range 518:1036
# ...ranges may run on any number of workers...
pepdistill prepare-status runs/prepare-full.toml
pepdistill prepare-finalize runs/prepare-full.toml
```

The current catalog contains 5,174 shards. That number is descriptive, not a partitioning
contract: use the count reported by the command when scripting ranges. A completed shard is
skipped unless `--force` is passed. Finalization refuses to publish `manifest.json` until every
expected shard asset is complete.

For the Launchpad array wrapper used by this repository:

```bash
.venv/bin/python tools/prepare_launchpad_full_run.py
launchpad run tools/launchpad_prospect_etl.py \
  --stage .launchpad/full-run-stage --array-size 10 \
  --env PEPDISTILL_PREPARE_ARRAY_SIZE=10
```

Each array task derives a contiguous global range from its array index, balancing boundaries by
the vendored raw Parquet byte sizes rather than shard count. The prepared output is identical
whether it runs on one worker or thousands. Explicit `--range` remains available for exact retry
ranges; `pepdistill prepare ... --partition 1/10` exposes the same weighted planner directly.

To retry one interrupted range without launching another array, pass it explicitly through the
environment:

```bash
launchpad run tools/launchpad_prospect_etl.py --stage .launchpad/full-run-stage \
  --env PEPDISTILL_PREPARE_RANGE=517:1034
```

After all ranges succeed, finalize in the same staged environment:

```bash
launchpad run tools/launchpad_prepare_finalize.py --stage .launchpad/full-run-stage
```

## 2. Config — `run.toml`

```toml
out = "runs/full"
remote_output_prefix = "s3://bucket/pepdistill-training/full-v1/small"
preset = "small"                 # flash | small-2h | small | base-4h | base
device = "auto"                  # auto -> mps/cpu; "cuda" -> gpu
seed = 0

[pretrain]                       # online teacher-distill warmup
enabled = true
teacher = "alphapeptdeep"        # or "fake" (dependency-free SMOKE only, not a real model)
nce_min = 20.0                   # per-peptide collision-energy sweep -> learned CE axis
nce_max = 40.0
passes  = 1                      # full enumerations of the digests
chunk_size = 10000               # peptides per teacher call
checkpoint_every_steps = 500     # periodically mirror an inference-ready warm start
# instrument/detector/fragmentation default to Lumos/FTMS/HCD (peptdeep's acquisition)
[[pretrain.sources]]             # one block PER fasta; per-source digestion + charge knobs
fasta = "proteome.fasta"
enzyme = "trypsin"               # or "unspecific" -> immunopeptidome windows
missed = 2
min_len = 7
max_len = 30
min_charge = 2                   # <-- charge range lives HERE (per source), not on [pretrain]
max_charge = 4
max_var_mods = 1

[train]                          # fine-tune on the prepared PROSPECT manifest
enabled = true
prepared_prefix = "s3://bucket/pepdistill-prepared/v1"
epochs = 60
loss_weights = [1.0, 1.0, 1.0]   # (ms2, iRT, raw_rt)

# [export] / [bench] left off. ONNX is deferred; use export-rust for production inference.
```

### What the knobs mean
- **Pretrain sources**: each `[[pretrain.sources]]` is one FASTA + its own digestion/charge
  settings (`DigestSource`). That's why `fasta` and `min_charge`/`max_charge` are per-source, not
  stage-level — you can mix, e.g., a tryptic proteome and an `unspecific` immunopeptidome source.
- **Preparation ranges**: `--range START:STOP` addresses the flattened global shard catalog.
  The range is half-open and independent of worker count or scheduling. `prepare-status` reports
  completion; `prepare-finalize` refuses to write a training manifest while any shard is missing.
- **Validation set**: not a separate shard. The loaded shards are split train/val/test by a
  deterministic hash of the *stripped* sequence (`assign_split`), so every mod-form/charge of a
  peptide stays in one split (no leakage). Val is deduped to best-per-entry; train keeps every
  observation. Real-data val metrics are reported separately for every configured dataset as
  `val/<dataset>/spectral_angle`, `val/<dataset>/irt_mae`, `val/<dataset>/rawrt_mae`, and
  `val/<dataset>/n` (the number of deduplicated validation entries).

## 3. Run pretrain → train

```bash
pepdistill run run.toml
# -> runs/full/model.ckpt   (student + saved MSContextEncoder + ChromRunbook + dataset_index)
# -> runs/full/summary.json (per-stage metrics)
```

Pretraining logs loss and learning rate per epoch and saves `pretrain.ckpt` when the stage ends.
Real-data training saves `latest.ckpt` every epoch and `best.ckpt` whenever mean per-dataset
spectral agreement improves. Its early-stop line labels this aggregate as spectral agreement;
higher is better. Historical controlled 60-epoch references reached weighted spectral angles
0.7075 (exact GELU) and 0.7084 (tanh-GELU); use per-dataset metrics and downstream search results
for acceptance, not either pooled number alone.

When `remote_output_prefix` is set, `pretrain-latest.ckpt` is mirrored every configured number of
steps; `pretrain.ckpt`, `train_metrics.jsonl`, `latest.ckpt`, `best.ckpt`, `model.ckpt`, and
`summary.json` are mirrored whenever they change. An object-store write failure stops the job
instead of silently leaving it without a durable recovery artifact.

W&B is the experiment index, not the artifact store: it receives configuration, namespaced
pretrain/train metrics, learning rate, system/console logs, grouping, and each mirrored S3 URI in
the run summary. Checkpoint/model bytes remain only in S3 (`log_model = false`). Enable it with:

```toml
[tracking]
enabled = true
project = "pepdistill"
group = "full-v1"
name = "small"
tags = ["full-nontest", "small"]
```

Install with `uv sync --extra tracking`. For Launchpad, store the key in AWS Secrets Manager,
grant the Batch job role `secretsmanager:GetSecretValue`, and pass only the non-secret identifier
as `--env PEPDISTILL_WANDB_SECRET_ID=pepdistill/wandb-api-key`. The wrapper fetches the value into
the training subprocess without logging or writing it. Do **not** use `--env WANDB_API_KEY=...`:
Launchpad environment values are plaintext in AWS job metadata and CloudTrail. Locally, use the
normal `WANDB_API_KEY` environment variable. Use `mode = "offline"` for a local run that should be
synchronized later.

These checkpoints are **warm starts**, not exact training resumes: they contain model and context
weights, but not optimizer, scheduler, early-stop, epoch, or streaming-cursor state. Starting from
one therefore begins the configured stage again with the saved weights.

### Launch the five-preset cloud sweep

The staging helper generates one config and one S3 output prefix for each of `flash`, `small-2h`,
`small`, `base-4h`, and `base`. Launch all five by array index:

```bash
.venv/bin/python tools/prepare_launchpad_full_run.py
launchpad run tools/launchpad_prepared_train.py \
  --stage .launchpad/full-run-stage --array-size 5 \
  --env PEPDISTILL_TRAIN_PRESETS=flash,small-2h,small,base-4h,base \
  --env PEPDISTILL_WANDB_SECRET_ID=pepdistill/wandb-api-key
```

A standalone invocation defaults to `small`; select one explicitly with
`--env PEPDISTILL_TRAIN_PRESET=base`. To recover a terminated real-training job from its durable
warm start, skip pretraining and load that preset's latest checkpoint:

```bash
launchpad run tools/launchpad_prepared_train.py --stage .launchpad/full-run-stage \
  --env PEPDISTILL_TRAIN_PRESET=base \
  --env PEPDISTILL_MODEL_IN=s3://terraform-workstations-bucket/jspaezp/pepdistill-training/full-v1/base/latest.ckpt \
  --env PEPDISTILL_NO_PRETRAIN=1
```

## 4. Generate a library and search it

Export once, then generate DIA-NN TSV directly in Rust. CCS is retained in prediction results;
the DIA-NN adapter reports ion mobility as Bruker 1/K0.

```bash
pepdistill export-rust --model runs/full/model.ckpt -o runs/full/model.safetensors
cargo run -q --release -p pepdistill-cli -- \
  --model runs/full/model.safetensors --fasta proteome.fasta --out library.tsv \
  --ms-context "Lumos::FTMS::HCD::30"
timsseek --raw-inputs sample.d --speclib-uri library.tsv --output-uri search-results
```

### Predict one peptide

The trained `model.ckpt` predicts. For a **single peptide**, use the Rust CLI (the single-peptide
tool); export the checkpoint to a `.safetensors` artifact first:

```bash
cargo run -q --release -p pepdistill-cli -- \
  --model runs/full/model.safetensors --peptide PEPTIDER --charge 2 \
  --ms-context "Lumos::FTMS::HCD::30"     # or --nce 30, or omit for base MS2
# -> one JSON object: precursor_mz, rt, ccs, fragments{ion, ord, z, mz, rel}
```

Notes:
- Transformer presets only.
- `--peptide` takes a **modified sequence**: `PEPC[Carbamidomethyl@C]IDER` (side chain),
  `[TMT6plex]PEPTIDER` (N-term), `PEPTIDER[Amidated]` (C-term), or a bare Dalton delta
  `PEP[+42.010565]TIDER`. Named mods go through the compositional encoder, `+`/`-` deltas
  through the mass encoder. The `peptide` field of the JSON echoes how the string was read.
- The artifact is versioned: a `.safetensors` exported before the mod-representation-v2 work
  (`format_version` 1) is **rejected**, not read with defaults. Re-export from the checkpoint.
- `--ms-context INSTRUMENT::DETECTOR::FRAGMENTATION::ENERGY` conditions MS2; `--chrom-context NAME`
  routes RT through a saved dataset's runbook row (else RT is the context-free iRT base).

### Alternative: torch, via a one-line FASTA (whole library, not single-peptide)
`pepdistill predict` digests a FASTA / reads a precursor table and writes a parquet library — it
does not take a bare peptide, and it enumerates charges 2–4 by default:

```bash
printf '>p\nPEPTIDER\n' > one.fasta
pepdistill predict --model runs/full/model.ckpt --fasta one.fasta -o one.parquet \
  --ms-context "Lumos::FTMS::HCD::30"     # torch runtime only; needs a saved encoder
```

## Outputs recap
- `runs/full/model.ckpt` — trained student + context (encoder/runbook/dataset_index).
- `runs/full/pretrain.ckpt` — snapshot after teacher distillation.
- `runs/full/latest.ckpt`, `runs/full/best.ckpt` — rolling real-data checkpoints.
- `runs/full/pretrain-latest.ckpt` — periodic pretrain warm start, when the interval is reached.
- `runs/full/summary.json` — per-stage metrics.
- `runs/full/model.safetensors` — Rust-loadable artifact (from `export-rust`).
- Prediction: JSON on stdout (Rust CLI) or `one.parquet` (torch predict).
