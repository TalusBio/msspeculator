# pepdistill — current design

## Goal

Generate search-engine-ready spectral libraries quickly with a compact student model. The
student is first distilled from AlphaPeptDeep over an in-silico FASTA digest, then can be
fine-tuned on prepared experimental libraries. It predicts MS2 fragment intensities, iRT/raw
RT, and CCS; the Rust library generator converts CCS to Bruker 1/K0 for DIA-NN output.

## Pipeline

```text
FASTA ──digest/enumerate──> AlphaPeptDeep labels ──online pretrain──┐
                                                                  ├──> checkpoint
Zenodo ──prepare shards──> immutable Parquet + manifest ──train───┘
                                                                       │
                                   export-rust ──> safetensors ──> DIA-NN TSV ──> search
```

`pepdistill run` controls the `pretrain`, `train`, `export`, and `bench` stages from one TOML
file. Pretraining streams teacher labels instead of materializing a label cache. Its OneCycle
learning-rate schedule is on by default. Real-data training reads only prepared Parquet assets;
it does not extract PROSPECT archives in the training process.

Preparation is a separate, restartable ETL:

- A vendored catalog and compressed shard index describe the upstream Zenodo records.
- An optional S3 prefix is a read-through intermediate cache, never the source of truth.
- Every global input shard maps to one immutable output asset and completion manifest.
- `--range START:STOP` distributes half-open global shard ranges independently of worker count.
- Finalization verifies all shard assets and publishes the worker-independent training manifest.

## Model and data contracts

- Rust (`rust/core`) is the single source of truth for peptide parsing, chemistry, tokenization,
  fragment m/z, tensor packing, and the standalone student forward pass. Python imports these
  contracts through `pepdistill_rs`.
- The production activation is tanh-approximated GELU. Transformer and dilated-CNN presets are
  available; the standalone Rust runtime currently supports transformer artifacts.
- The input combines residue/terminus tokens, compositional and mass-only modification features,
  position, precursor charge, and optional acquisition context.
- MS2 context uses instrument, detector, fragmentation, and collision energy. RT can select a
  saved chromatography context. CCS prediction itself is context-free.
- Deterministic hashing of stripped peptide sequence keeps all modification and charge forms in
  the same train/validation/test partition.
- Real-data validation is deduplicated to the best observation per library entry and reported
  per dataset. Early stopping follows the mean of the configured per-dataset spectral angles.

## Inference and output

The Python predictor writes long-format Parquet. The production Rust path loads a self-contained,
versioned `.safetensors` artifact, digests FASTA, batches equal-length precursors, and uses a
bounded worker pool feeding one writer thread. FASTA output is a streaming DIA-NN TSV; precursor
CCS is converted to ion mobility in 1/K0. Output row order is intentionally unspecified.

The DIA-NN adapter has been exercised end to end with `timsseek` against a Bruker timsTOF run.
Single-peptide JSON prediction remains available for parity checks and inspection.

## Package boundaries

```text
pepdistill/data/       digest, precursor enumeration, split, encoding, prepared-data reader
pepdistill/etl/        catalog discovery, archive conversion, shard manifests, finalization
pepdistill/teacher/    AlphaPeptDeep and deterministic fake teachers
pepdistill/models/     student architectures, presets, context encoders, checkpoint contracts
pepdistill/distill/    pretrain/train loops, validation, early stopping, pipeline configuration
pepdistill/predict/    reference/vectorized Python library generation and deferred ONNX path
rust/core/             shared chemistry, encoding, artifact reader, and student inference
rust/cli/              standalone safetensors-to-DIA-NN/JSON inference
```

## Deliberately deferred

ONNX remains available as an experimental export/runtime path, but it is not the current product
milestone and should not drive architecture work. The current optimization and integration target
is the Rust FASTA-to-library-to-search path. Additional search-engine adapters can follow measured
consumer needs.
