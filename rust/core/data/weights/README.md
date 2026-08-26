# Bundled weights

Artifacts compiled into `pepdistill-cli` by `rust/core/src/builtin.rs`, so that a clean clone
builds a tool that can predict. Each is addressed as `--model builtin:<name>`; the name and the
file's blake2b-256 are what a generated library records as its model identity.

**The provenance is inside each file**, not in this note. Every artifact written by
`pepdistill export-rust` carries a `provenance` key in its safetensors `__metadata__`: the source
checkpoint's name and digest, plus the checkpoint's own training record: stage, epoch, global
step, and the per-dataset validation values behind the score. Read it with:

```bash
uv run python -c "
import json, struct, sys
with open(sys.argv[1], 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    print(json.dumps(json.loads(json.loads(f.read(n))['__metadata__']['pepdistill'])['provenance'],
                     indent=2))
" rust/core/data/weights/small-v0.safetensors
```

This file records what that metadata cannot: which run produced the checkpoint, since the
checkpoint itself is not in version control (`runs/*` is gitignored).

## `small-v0.safetensors`

| | |
| --- | --- |
| preset | `small`, transformer, `d_model` 64, 2 layers, 4 heads, 132,358 parameters |
| activation | `gelu_tanh` (production default) |
| score | 0.8054 mean per-dataset spectral agreement, 41 datasets |
| source checkpoint | `runs/full-local-decay2/best.ckpt`, epoch 33, step 2,108,705 |
| run config | `runs/full-local-resume2.toml` |
| corpus | prepared v2 manifest, 5,174 shards, 21,003,479 rows, 41 datasets |
| blake2b-256 | `8dc9edf567606df1ce2b98530d679ebce139f364021062b72a20d4eaca7162a3` |

`v0`, not `v1`: the five-preset sweep that picks a production model has not run, so this is a
working default rather than a blessed release.

### Regenerating it

Export from a checkpoint. This step is deterministic. The same checkpoint yields the same bytes,
which is what lets `builtin.rs` record a digest and a test assert it:

```bash
uv run pepdistill export-rust --model runs/full-local-decay2/best.ckpt \
  -o rust/core/data/weights/small-v0.safetensors
```

Producing the checkpoint is not deterministic. Training reads shards in an order that depends on
worker scheduling, so re-running the config reproduces the model's *behaviour* within run-to-run
variation, not its weights:

```bash
uv run pepdistill run --config runs/full-local-resume2.toml
```

Re-exporting changes the digest whenever the checkpoint or the export format changes. The recorded
digest in `rust/core/src/builtin.rs` has to be updated in the same commit, or
`every_bundled_artifact_matches_its_recorded_digest` fails. This prevents weights from being
swapped under a name without saying so.

### Size

`MAX_BUNDLED_BYTES` (4 MiB) caps what may be embedded, enforced by the same test. Larger weights
should be fetched at runtime and cached instead of committed. The `base` preset still fits; a
substantially larger model does not.
