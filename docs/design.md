# Current design

## What the project does

`msspeculator` trains a compact student model and uses it to predict peptide RT, CCS, and MS2
fragment intensities. Python handles training and prepared-data ETL. Rust owns chemistry and the
production inference path.

## Data flow

```text
FASTA -> digest -> teacher labels -> pretrain --+
                                                +-> checkpoint -> Rust artifact -> library -> search
prepared shards -> train ----------------------+
```

Pretraining streams labels from a teacher. Real-spectrum training reads immutable Parquet shards.
Preparation is separate, restartable, and range-addressable. Its vendored catalog and shard index
define source identity. An optional S3 prefix caches source files and stores prepared assets. A
final manifest is published only after every expected shard is present.

Prepared training assigns shards deterministically across loader workers. Each worker carries rows
across shard boundaries to form exact-length batches, so the model does not need padding masks.

## Contracts

Rust is the authority for peptide parsing, chemistry constants, modification resolution,
tokenization, fragment m/z, tensor packing, and the portable safetensors artifact. Python imports
those values through the PyO3 extension.

The model input combines residue and terminus tokens, modification composition or mass fields,
position, charge, and optional acquisition context. MS2 context includes instrument, detector,
fragmentation, and collision energy. RT can use a saved chromatography context. CCS is
context-free.

Training can replace a residue with a chemistry-equivalent token while attaching the composition
and mass difference at that position. Precursor and fragment targets stay unchanged.

The train, validation, and test split hashes the stripped peptide sequence. Modification and charge
forms therefore stay in one split. Validation keeps the best observation per library entry and
reports each dataset separately.

## Inference and output

The Python predictor writes long-format Parquet. Rust loads a versioned, self-contained
`.safetensors` artifact, groups equal-length precursors into batches, and runs bounded workers
feeding a writer. Output order is unspecified. The writer emits DIA-NN TSV or mzSpecLib text, with
optional gzip compression. CCS is converted to Bruker 1/K0 for library output.
The optional decoy path pseudo-reverses internal residues, skips target-sequence collisions, and
marks generated entries in both output formats.

The `msspeculator-inference` crate owns this FASTA path. Its `write_library` function accepts
`LibraryOptions`, starts the producer and worker queues, and returns `LibraryStats`. The CLI passes
parsed arguments to that function. Rust applications can use the same path without reimplementing
thread management or invoking a process.

An application that wants the rows rather than a file implements `LibrarySink` and calls
`stream_library`, which takes the same options minus the output path and sidecar. The two entry
points share one implementation, so a caller-supplied sink sees exactly what the bundled DIA-NN and
mzSpecLib writers see.

Every precursor is validated and capped before serialization. TSV output includes a `config.json`
sidecar. mzSpecLib stores the same resolved generation settings in its header.

## Package boundaries

```text
src/msspeculator/data/       digestion, encoding, split, prepared-data reader
src/msspeculator/etl/        catalog, archive conversion, manifests, finalization
src/msspeculator/teacher/    AlphaPeptDeep and fake teachers
src/msspeculator/models/     architectures, presets, contexts, checkpoints
src/msspeculator/distill/    pretrain/train loops and validation
rust/core/                    shared chemistry, encoding, artifact, inference
rust/inference/               FASTA orchestration and library writers
rust/cli/                     command-line wrapper
```

ONNX support was removed because no current consumer used it and its graph fixed sequence length.
New output adapters should follow a measured consumer need.
