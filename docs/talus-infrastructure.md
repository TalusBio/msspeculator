# Talus infrastructure runbook

This page records how the public pepdistill workflow is deployed on Talus infrastructure. It is
for maintainers with access to the relevant AWS account, object-store prefixes, Launchpad, and W&B
project. Public users should follow the [local runbook](runbook-full-run.md).

Concrete bucket names and credentials are intentionally omitted. Supply them through deployment
configuration and environment; never commit or paste credentials into a run file.

## Local AWS access

Polars reads `s3://` through its object-store client, which consumes environment credentials or an
instance role but not the AWS SSO cache directly. For an authorized local session:

```bash
aws sso login
set -a; eval "$(aws configure export-credentials --format env)"; set +a
```

Cloud workers use an instance role. W&B uses a short-lived service-account key supplied as
`WANDB_API_KEY` in job metadata; revoke it after the job and its retries are terminal. Use W&B
offline mode when remote telemetry is unnecessary.

## Prepare the corpus

`runs/prepare-full.toml` selects the non-test PROSPECT, TMT, multi-PTM, and TMT-PTM archives. The
vendored Zenodo catalog and shard index define source identity. S3 is a read-through cache and
prepared-asset destination.

Stage once, then launch no more than 40 array children:

```bash
.venv/bin/python tools/prepare_launchpad_full_run.py
aws s3 sync .launchpad/full-run-stage "$STAGE_URI"
launchpad run tools/launchpad_prospect_etl.py \
  --stage "$STAGE_URI" --array-size 40 \
  --env PEPDISTILL_PREPARE_ARRAY_SIZE=40
```

Each child receives a contiguous global range weighted by raw Parquet bytes. Forty children fit the
shared host disk and complete the current corpus in roughly 20 minutes; 80 children exhausted the
host while independently unpacking Python environments.

Retry a specific half-open range when needed:

```bash
launchpad run tools/launchpad_prospect_etl.py --stage "$STAGE_URI" \
  --env PEPDISTILL_PREPARE_RANGE=517:1034
```

Finalize only after every range succeeds:

```bash
launchpad run tools/launchpad_prepare_finalize.py --stage "$STAGE_URI"
```

## Audit the prepared corpus

Generate both the machine-readable records and their committed Markdown views:

```bash
.venv/bin/python tools/prepared_curation_report.py \
  --prepared "$PREPARED_URI" --out curation-summary.json \
  --markdown docs/prepared-curation.md

.venv/bin/python tools/teacher_yardstick.py \
  --prepared "$PREPARED_URI" --out teacher-yardstick.json \
  --markdown docs/teacher-yardstick.md
```

The committed Markdown intentionally omits the internal storage URI. The JSON record retains it for
authorized reproducibility and can re-render the report with `--render-from`.

## Train the preset sweep

The staging helper generates isolated configs for `flash`, `small-2h`, `small`, `base-4h`, and
`base`. The entrypoint requests 4 vCPUs and 16 GB; the prepared loader stays in-process and the
model uses four intra-op threads.

```bash
.venv/bin/python tools/prepare_launchpad_full_run.py
aws s3 sync .launchpad/full-run-stage "$STAGE_URI"
for preset in flash small-2h small base-4h base; do
  launchpad run tools/launchpad_prepared_train.py \
    --stage "$STAGE_URI" \
    --env PEPDISTILL_TRAIN_PRESET="$preset" \
    --env WANDB_API_KEY="$WANDB_API_KEY"
done
```

Durable checkpoints and metrics are mirrored to each run's configured object-store prefix. W&B is
the experiment index, not the artifact store; checkpoint bytes remain in object storage.

To warm-start a terminated training stage, name the preset, checkpoint URI, and pretraining skip:

```bash
launchpad run tools/launchpad_prepared_train.py --stage "$STAGE_URI" \
  --env PEPDISTILL_TRAIN_PRESET=base \
  --env PEPDISTILL_MODEL_IN="$CHECKPOINT_URI" \
  --env PEPDISTILL_NO_PRETRAIN=1
```

## Generate a whole-proteome library

The Launchpad image contains no compiler or download tools. Cross-compile the Rust CLI, then stage
the immutable binary, model, and exact FASTA bytes:

```bash
cross build --release --target x86_64-unknown-linux-musl \
  --manifest-path rust/Cargo.toml -p pepdistill-cli
mkdir -p speclib-stage
cp rust/target/x86_64-unknown-linux-musl/release/pepdistill-cli speclib-stage/
uv run pepdistill export-rust --model runs/full/best.ckpt \
  -o speclib-stage/model.safetensors
cp /path/to/pinned-proteome.fasta speclib-stage/proteome.fasta
aws s3 sync speclib-stage "$SPECLIB_STAGE_URI"

launchpad run tools/launchpad_speclib.py --stage "$SPECLIB_STAGE_URI" \
  --env PEPDISTILL_SPECLIB_OUT="$SPECLIB_OUTPUT_URI"
```

Stage the FASTA rather than fetching a moving release during the job. The generated library records
the model and FASTA digests plus every resolved inference setting. For large libraries, gzip in the
writer thread avoids materializing the much larger uncompressed output.
