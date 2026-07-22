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
- **Staged, cached pipeline.** `digest → label → distill → predict`, each writing a
  reusable artifact. The teacher runs once.

## Install

```bash
uv sync                     # core (torch, pandas, typer)
uv sync --extra teacher     # + AlphaPeptDeep teacher (peptdeep)
uv sync --extra onnx        # + ONNX export & onnxruntime inference
```

The `fake` teacher (deterministic, dependency-free) is always available for
development and tests; the real `alphapeptdeep` teacher needs the `teacher` extra.

## Quick start

One-shot, end to end into a working directory:

```bash
pepdistill pipeline proteome.fasta -w work/ --teacher alphapeptdeep --preset small --epochs 30
# -> work/precursors.parquet, work/labels/, work/model.ckpt, work/library.parquet
```

Or run the stages individually:

```bash
pepdistill digest  proteome.fasta -o precursors.parquet
pepdistill label   precursors.parquet -o labels/ --teacher alphapeptdeep
pepdistill distill --precursors precursors.parquet --labels labels/ -o model.ckpt --preset small
pepdistill predict --model model.ckpt --fasta proteome.fasta -o library.parquet --device auto
pepdistill benchmark --model model.ckpt --fasta proteome.fasta --device auto
```

### Streaming (online) distillation

Skip the label cache entirely: the teacher scores fresh sequences **live in the training
loop**. A curriculum warms up on a random-peptide generator, then switches to real FASTA
digests. Throughput is capped by the teacher — meant to run overnight.

```bash
pepdistill distill-stream -o model.ckpt --fasta proteome.fasta \
    --teacher alphapeptdeep --preset flash \
    --total-batches 20000 --warmup-batches 2000 --batch-size 256 --device auto
```

`--device auto` uses Apple MPS when present, else CPU. No FASTA → trains purely on random
sequences (the teacher generalizes to any peptide).

### ONNX export

```bash
pepdistill export --model model.ckpt -o model.onnx        # cnn presets export fully dynamic
pepdistill predict --model model.onnx --fasta p.fasta -o lib.parquet --runtime onnx
```

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
  chem.py      masses, m/z, fragment-ion series (pure numeric)
  data/        FASTA digest, precursor enumeration, deterministic split, tensor encoding,
               streaming sources (random generator + FASTA stream)
  teacher/     Teacher ABC + FakeTeacher + PeptDeepTeacher (behind [teacher] extra)
  models/      student architectures (no LSTM) + presets + checkpoint I/O
  distill/     dataset, losses (MS2 cosine / RT+CCS MSE), batch + streaming trainers
  predict/     library.py (reference) + fast.py (vectorized) + onnx.py (export/runtime)
  util.py      device resolution (auto -> mps/cpu)
  cli.py       typer CLI
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
