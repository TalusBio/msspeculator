# msspeculator: current design

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
                       export-rust ──> safetensors ──> DIA-NN TSV / mzSpecLib ──> search
```

`msspeculator run` controls the `pretrain`, `train`, `export`, and `bench` stages from one TOML
file. Pretraining streams teacher labels instead of materializing a label cache. Its OneCycle
learning-rate schedule is on by default. Real-data training reads only prepared Parquet assets;
it does not extract PROSPECT archives in the training process.

Preparation is a separate, restartable ETL:

- A vendored catalog and compressed shard index describe the upstream Zenodo records.
- An optional S3 prefix is a read-through cache. The vendored catalog and shard index remain the
  source records.
- Every global input shard maps to one immutable output asset and completion manifest.
- `--range START:STOP` distributes half-open global shard ranges independently of worker count.
- Finalization verifies all shard assets and publishes the worker-independent training manifest.

Prepared training partitions shuffled shards exactly once across persistent loader workers.
Within each worker, rows are carried across shard boundaries into exact-length batches, avoiding
transformer padding/masks and allowing Parquet decode and tensor collation to overlap model work.

## Model and data contracts

- Rust (`rust/core`) owns peptide parsing, chemistry, tokenization,
  fragment m/z, tensor packing, and the standalone student forward pass. Python imports these
  contracts through `msspeculator_rs`.

Training can apply chemistry-preserving residue substitution augmentation after collation. A
residue token is replaced while its original-minus-replacement elemental composition and mass
are attached at the same site. This leaves precursor/fragment targets invariant and teaches the
token and compositional-modification paths a shared chemical representation. Residue formulas,
like residue masses, are exported from the Rust chemistry authority.
- The production activation is tanh-approximated GELU. All maintained presets are transformers;
  the `small` and `base` families expose controlled head-count variants for training sweeps.
- The input combines residue/terminus tokens, compositional and mass-only modification fields,
  position, precursor charge, and optional acquisition context.
- MS2 context uses instrument, detector, fragmentation, and collision energy. RT can select a
  saved chromatography context. CCS prediction itself is context-free.
- Deterministic hashing of stripped peptide sequence keeps all modification and charge forms in
  the same train/validation/test partition.
- Real-data validation is deduplicated to the best observation per library entry and reported
  per dataset. Early stopping follows the mean of the configured per-dataset spectral angles.
- Optional longitudinal diagnostics reuse one teacher-labeled iRT reference panel and fixed PCA
  bases for the whole run. Initial/hourly/epoch/final renders can therefore be compared directly;
  plots are stored as run artifacts and logged to W&B without entering the hot training loop.

## Inference and output

The Python predictor writes long-format Parquet. The production Rust path loads a self-contained,
versioned `.safetensors` artifact from a path or from the copy vendored into the binary. A clean
clone therefore builds a tool that predicts. It digests FASTA, batches equal-length precursors, and uses a
bounded worker pool feeding one writer thread. FASTA output streams as either a DIA-NN TSV or an
mzSpecLib text library, selected by the `--out` suffix and optionally gzipped in that same writer
thread; precursor CCS is converted to ion mobility in 1/K0. Output row order is intentionally
unspecified. Every precursor is validated and capped once, before any format sees it, so the two
serializations cannot disagree about what the library contains. They differ only in how they spell it.
mzSpecLib also carries the resolved generation configuration in its header. This is the
same record the `config.json` sidecar holds.

The DIA-NN adapter has been exercised end to end with `timsseek` against a Bruker timsTOF run.
Single-peptide JSON prediction remains available for parity checks and inspection.

## Package boundaries

```text
msspeculator/data/       digest, precursor enumeration, split, encoding, prepared-data reader
msspeculator/etl/        catalog discovery, archive conversion, shard manifests, finalization
msspeculator/teacher/    AlphaPeptDeep and deterministic fake teachers
msspeculator/models/     student architectures, presets, context encoders, checkpoint contracts
msspeculator/distill/    pretrain/train loops, validation, early stopping, pipeline configuration
msspeculator/predict/    reference and vectorized Python library generation
rust/core/             shared chemistry, encoding, artifact reader, and student inference
rust/cli/              standalone safetensors-to-DIA-NN/mzSpecLib/JSON inference
```

## Deliberately deferred

We removed the ONNX export and runtime because nothing used them. The export also baked a sequence
length into the graph and rejected other lengths. The optimization and integration target is the Rust
FASTA-to-library-to-search path. Additional search-engine adapters can follow measured consumer
needs, and a portable export format can be reintroduced when a named consumer requires one.
