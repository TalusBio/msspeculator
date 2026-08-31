# msspeculator

Predicts peptide retention time, ion mobility, and MS2 spectra, then writes spectral libraries
for search. Training happens in Python; inference and library generation happen in Rust.

## Language

### Prediction and libraries

**Spectral library**:
A set of predicted precursors with their fragment spectra, serialized for a search engine to
consume.

**Spectrum row**:
One precursor and its kept peaks, validated and capped, ready for any serialization. The
precursor-grained record; a sink's only remaining job is spelling.
_Avoid_: library row, fragment row (both name a different grain)

**Peak**:
One kept fragment of a predicted spectrum: m/z, intensity, ion type, ordinal, charge. Named for
the PSI-MS controlled vocabulary that mzSpecLib output already emits.
_Avoid_: transition (the SRM/MRM lineage term DIA-NN uses; PSI-MS wins on collision)

**Library sink**:
A serialization of a spectral library: one header, then one spectrum row per precursor, in
prediction order. DIA-NN TSV and mzSpecLib text are the two adapters.
_Avoid_: writer, exporter, format handler

**Precursor**:
A peptidoform at a particular charge. The thing a spectrum is predicted for.

**Decoy**:
A pseudo-reversed peptidoform emitted alongside its target so a search engine can estimate false
discovery. Skipped when its stripped sequence collides with a real target.

### Model and context

**Student**:
The small model being trained and shipped. The thing that predicts.

**Teacher**:
An existing published predictor (AlphaPeptDeep) that labels peptides so the student can learn
from them before any real data is involved. Python-only, by dependency.

**Portable weights**:
A self-contained `.safetensors` export of a trained student, carrying everything Rust inference
needs and nothing a Python runtime would. The unit that crosses from training into inference.
Spelled `Artifact` in the Rust code, where the word has only this one sense and the struct is
genuinely file-shaped (it round-trips metadata keys the build does not model).
_Avoid_: artifact, in Python, where it already means any mirrored run output; checkpoint

**Training checkpoint**:
The torch-side `.ckpt` holding a student plus its acquisition context, written during a run and
readable only where Lightning and torch are installed.
_Avoid_: artifact, bare "checkpoint"

**Acquisition context**:
How a spectrum was acquired, as the model consumes it: instrument, detector, fragmentation and
collision energy, either spelled out as factors or addressed by the name of a fitted setup.

**Setup**:
A named acquisition context fitted into an artifact, addressable by label. What a library that
records neither instrument nor collision energy can still be predicted for.

**Chromatography context**:
The gradient a retention time is reported against. Present means `rt` is a dataset's gradient
time in minutes; absent means it is the context-free index.

**iRT**:
The context-free retention index: run-independent, and what the student predicts natively.

### Training

**Regime**:
One way of training the student against one kind of data, with its own configuration and its own
Lightning module. Stream-pretrain (teacher-labeled, enumerated live) and real-speclib (a prepared
corpus) are the two.
_Avoid_: stage, mode, phase

**Prepared corpus**:
Immutable, chunked Parquet training assets published under a prefix, with a manifest. Built once
by the preparation ETL; never rewritten.

**Reference panel**:
A fixed set of peptides and teacher spectra, saved once, re-rendered against changing student
weights so successive snapshots are comparable.
