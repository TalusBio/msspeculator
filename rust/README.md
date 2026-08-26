# msspeculator_rs

Rust core for `msspeculator`: chemistry (residue/mod masses, fragment m/z), the `Peptide`
type, tokenization, and batched tensor encoding (`collate`/`bucket_arrays`). This crate owns
chemistry and encoding. `msspeculator.chem` and
`msspeculator.data.encode` are thin Python re-export shims over it, not independent
implementations.

## Layout

- `core/` (`msspeculator-core`) is the authority for chemistry constants, `Peptide` (masses,
  precursor m/z, canonical/modified sequence), the tokenizer, and the batched bucket
  builder (fragment m/z matrix + tensor packing). Pure Rust, unit-tested in isolation
  (`cargo test -p msspeculator-core`).
- `src/lib.rs` (crate `msspeculator_rs`) is a thin pyo3 binding layer over `core`. It exposes the
  `Peptide` pyclass, vocab/chemistry constants, and numpy-returning functions
  (`fragment_mz`, `fragment_mz_matrix`, `ms2_target_shape`, `collate`, ...). It adds no
  chemistry or tokenization logic of its own.
- `cli/` (`msspeculator-cli`) is a standalone Rust binary with no Python dependency. It handles spectral-library
  generation and the Python-vs-Rust prediction-parity check (`tests/test_rust_parity.py`).
  `library --fasta ... --out library.tsv` digests and enumerates precursors, loads the `.safetensors`
  weights once, predicts in Rust, converts CCS to Bruker 1/K0, and writes DIA-NN TSV.
  `predict --peptide` takes a strict ProForma-compatible subset such as
  `[UNIMOD:737]-PEPC[UNIMOD:4]IDER`, `PEP[Formula:[13C2][12C-2]H2N]TIDER`, or a signed Dalton
  delta `PEP[+42.010565]TIDER`. Terminal placement requires the ProForma hyphen. Historical
  names remain internal to prepared training data and are not public parser input.
  The `.safetensors` artifact carries a `format_version`; `core/src/artifact.rs` rejects any
  version it does not read rather than filling in missing tensors.
  `run-doctor --model MODEL --out DIR` predicts the vendored Biognosys iRT standards, renders
  a compact terminal scatter, and writes `DIR/{irt-scatter.svg,report.txt,irt-predictions.tsv}`
  for model debugging.

FASTA inference groups precursors into equal-length batches of 64. A bounded worker pool clones
the immutable model per worker and feeds one writer thread; set `PEPDISTILL_WORKERS` to override
the default of `available_parallelism - 1`. Output order is unspecified. On the documented
474,630-precursor / 16,650,774-transition fixture, the worker pool completed in 11.53–12.97
seconds (roughly 36.6k–41.2k precursors/s); measure again on the deployment host because this is
end-to-end, workload- and hardware-specific throughput.

## Build into the project venv

This extension is a **hard runtime dependency** of `msspeculator`, not an optional
accelerator. The root `pyproject.toml` declares it via `[tool.uv.sources]` as an editable
path dependency on this directory, so a plain `uv sync` from the repo root builds and
installs it. A Rust toolchain with `cargo` must be on `PATH`:

```sh
uv sync
```

If you need to rebuild it directly (e.g. after editing Rust source without re-running
`uv sync`):

```sh
uv run maturin develop -m rust/Cargo.toml --release
```

After either, `import msspeculator_rs` succeeds in the project venv and
`tests/test_ext_smoke.py` / `tests/test_rust_parity.py` run instead of skipping.

## Contract

Rust owns chemistry: `mod_delta`/`mod_composition` (descriptor -> mass / element composition),
`RESIDUE_MASS`, `H2O`, `PROTON`, vocab constants (`AA_OFFSET`/`PAD_IDX`/`N_TOKENS`/terminus
indices), and ion-series layout (`ION_TYPES`) all live in `rust/core` and are exported through
`msspeculator_rs`. Python code never duplicates or re-derives these values. It imports them from
`msspeculator.chem` (re-exported from the ext).

Callers pass **ProForma descriptors** (e.g. `"UNIMOD:35"`, `"Formula:H2O"`), not pre-scaled
deltas. Rust resolves composition and mass internally against the vendored UNIMOD table. This
applies uniformly to `Peptide` construction, `collate`/`bucket_arrays`, and
`fragment_mz`/`fragment_mz_matrix`. There is no
Python-side chemistry to keep in parity with Rust; Rust unit tests
(`cargo test -p msspeculator-core`, see `core/src/chem.rs`, `core/src/peptide.rs`,
`core/src/tokenize.rs`, `core/src/bucket.rs`) are the authoritative coverage.
