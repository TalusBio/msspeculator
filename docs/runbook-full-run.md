# Local runbook

This is the public path from a source checkout to a generated spectral library. It requires no
Talus account or cloud credentials. The production-scale deployment uses the same commands with
object-store paths and distributed workers; those details live in the
[Talus infrastructure runbook](talus-infrastructure.md).

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- A Rust toolchain with `cargo`
- [Task](https://taskfile.dev/) for contributor commands

Install the core package:

```bash
uv sync --locked
uv run msspeculator --help
```

Training on a CPU uses the `torch-cpu` extra. The regular `torch` extra intentionally selects the
CUDA-capable PyPI build on Linux.

```bash
uv sync --locked --extra torch-cpu --extra etl --extra tracking
```

## Fast local smoke run

The fake teacher is deterministic and does not require AlphaPeptDeep. It verifies configuration,
digestion, training, checkpointing, and export without downloading the experimental corpus.

Create a small FASTA:

```bash
printf '>example\nMKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE\n' > smoke.fasta
```

Create `smoke.toml`:

```toml
out = "runs/smoke"
preset = "flash"
device = "cpu"
seed = 0

[pretrain]
enabled = true
teacher = "fake"
passes = 1
chunk_size = 32

[[pretrain.sources]]
fasta = "smoke.fasta"
enzyme = "trypsin"
missed = 1
min_len = 7
max_len = 30
min_charge = 2
max_charge = 3

[train]
enabled = false
```

Run it and export the checkpoint for standalone Rust inference:

```bash
uv run msspeculator run smoke.toml
uv run msspeculator export-rust --model runs/smoke/model.ckpt -o runs/smoke/model.safetensors
```

## Generate a library

The Rust CLI includes a small built-in model, so inference does not require Python training or a
downloaded checkpoint:

```bash
cargo run -q --release -p msspeculator-cli -- \
  library --fasta smoke.fasta --out library.mzspeclib.txt.gz \
  --ms-context "Lumos::FTMS::HCD::30"
```

Add `--decoys` to include pseudo-reversed target-decoy entries. The generator preserves the first
and last residues, skips sequence collisions with targets, and marks decoys in either output
format.
mzSpecLib entries also carry a shared `msspeculator:decoy_pair_id` on each target/decoy pair, one
per target peptidoform and charge. A collision-skipped pair keeps the ID on its target only. DIA-NN
has no column for that relationship.

The output suffix selects the format:

| suffix | format |
| --- | --- |
| `.mzspeclib.txt`, `.mzspeclib` | HUPO-PSI mzSpecLib text |
| anything else | DIA-NN TSV |

A trailing `.gz` compresses either format. mzSpecLib is preferred for published libraries because
its header carries the resolved model, FASTA digests, digestion rules, modifications, acquisition
context, and fragment settings. TSV output writes the same provenance to a `config.json` sidecar.

Regenerating a library in place reports what it is replacing, from whichever copy of the
provenance the old file has, and then rebuilds. A run over the same FASTA with the same settings
says so; one with a knob changed names the knob and both values. The report never stops the
rebuild.

Single-peptide prediction is also available:

```bash
cargo run -q --release -p msspeculator-cli -- \
  predict --peptide 'PEPC[UNIMOD:4]IDER' --charge 2 --nce 30
```

## Prepare experimental data

Preparation is range-addressable and restartable. Copy `runs/prepare-full.toml`, replace its
organization-specific cache and output locations, then inspect the current shard count before
choosing ranges:

```bash
uv run msspeculator prepare-status prepare.toml --count-only
uv run msspeculator prepare prepare.toml --range 0:100
uv run msspeculator prepare-status prepare.toml
uv run msspeculator prepare-finalize prepare.toml
```

Ranges are half-open and independent of worker count. A completed shard is skipped unless
`--force` is passed, and finalization refuses to publish a manifest until every expected shard is
present. When code changes what a shard contains, bump `PREPARE_POLICY_VERSION` in
`src/msspeculator/etl/config.py`; performance-only changes do not require a bump.

Prepared assets may live on a local filesystem or an fsspec-compatible object store. Object-store
credentials come from that provider's normal environment or instance identity; msspeculator does
not manage credentials.

## Train on prepared data

Use the full configuration grammar in the [design document](design.md) and the checked-in run
configurations as advanced examples. A minimal real-data stage is:

```toml
out = "runs/full"
preset = "small"
device = "cpu"
seed = 0

[pretrain]
enabled = false

[train]
enabled = true
prepared_prefix = "/path/to/prepared/v2"
epochs = 60
num_workers = 0
model_threads = 4
```

```bash
uv run msspeculator run run.toml
```

The stage writes `latest.ckpt`, `best.ckpt`, `model.ckpt`, metrics, and `summary.json` below
`out`. Checkpoints are warm starts rather than exact optimizer resumes. Validation is deduplicated
and reported per dataset; use those metrics and downstream search results instead of a pooled score
as the acceptance criterion.

For acquisition-context fitting, retention semantics, curation reports, and model internals, see
the [current design](design.md), [model v2](model-v2-design.md), and generated reports in the
[documentation index](README.md).
