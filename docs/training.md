# Training

Training is optional. Inference only needs the core install in the README.

## Install training dependencies

Use the CPU wheels for local work and CI:

```bash
uv sync --locked --extra torch-cpu --extra etl --extra tracking
```

Add `--extra teacher` to use AlphaPeptDeep. Use `--extra torch` instead of `torch-cpu` for the
CUDA-capable PyPI build on Linux. The extras are separate because preparation workers do not need
Torch.

## Run a teacher warmup

Create a TOML run configuration:

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

Run it and export the checkpoint for Rust inference:

```bash
uv run msspeculator run smoke.toml
uv run msspeculator export-rust --model runs/smoke/model.ckpt \
  -o runs/smoke/model.safetensors
```

The `fake` teacher is deterministic and has no extra dependency. The AlphaPeptDeep teacher needs
the `teacher` extra.

## Fine-tune on experimental spectra

Prepared PROSPECT Parquet shards are immutable and restartable. Set `[train] enabled = true` and
provide `prepared_prefix` in the run config. The trainer can condition MS2 on acquisition factors
and RT on a saved chromatography context.

The [full local runbook](runbook-full-run.md) has the preparation, training, and export commands.
The [Talus runbook](talus-infrastructure.md) covers the production workflow.
