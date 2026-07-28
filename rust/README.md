# pepdistill_rs

Rust core for `pepdistill`: chemistry (residue/mod masses, fragment m/z), the `Peptide`
type, tokenization, and batched tensor encoding (`collate`/`bucket_arrays`). This crate is
the **single source of truth** for chemistry and encoding — `pepdistill.chem` and
`pepdistill.data.encode` are thin Python re-export shims over it, not independent
implementations.

## Layout

- `core/` (`pepdistill-core`) — the authority: chemistry constants, `Peptide` (masses,
  precursor m/z, canonical/modified sequence), the tokenizer, and the batched bucket
  builder (fragment m/z matrix + tensor packing). Pure Rust, unit-tested in isolation
  (`cargo test -p pepdistill-core`).
- `src/lib.rs` (crate `pepdistill_rs`) — a thin pyo3 binding layer over `core`: exposes the
  `Peptide` pyclass, vocab/chemistry constants, and numpy-returning functions
  (`fragment_mz`, `fragment_mz_matrix`, `ms2_target_shape`, `collate`, ...). It adds no
  chemistry or tokenization logic of its own.
- `cli/` (`pepdistill-cli`) — a standalone Rust binary (no Python) used for the
  Python-vs-Rust prediction-parity check (`tests/test_rust_parity.py`).

## Build into the project venv

This extension is a **hard runtime dependency** of `pepdistill`, not an optional
accelerator. The root `pyproject.toml` declares it via `[tool.uv.sources]` as an editable
path dependency on this directory, so a plain `uv sync` from the repo root builds and
installs it (requires a Rust toolchain — `cargo` — on `PATH`):

```sh
uv sync
```

If you need to rebuild it directly (e.g. after editing Rust source without re-running
`uv sync`):

```sh
uv run maturin develop -m rust/Cargo.toml --release
```

After either, `import pepdistill_rs` succeeds in the project venv and
`tests/test_ext_smoke.py` / `tests/test_rust_parity.py` run instead of skipping.

## Contract

Rust owns chemistry: `MOD_DELTA` (mod name -> mass delta), `RESIDUE_MASS`, `H2O`, `PROTON`,
vocab constants (`AA_OFFSET`/`PAD_IDX`/`N_TOKENS`/terminus indices), and ion-series layout
(`ION_TYPES`) all live in `rust/core` and are exported through `pepdistill_rs`. Python code
never duplicates or re-derives these values — it imports them from `pepdistill.chem`
(re-exported from the ext).

Callers pass **modification names** (e.g. `"Oxidation@M"`), not pre-scaled deltas — Rust
looks up the mass shift internally via `MOD_DELTA`. This applies uniformly to `Peptide`
construction, `collate`/`bucket_arrays`, and `fragment_mz`/`fragment_mz_matrix`. There is no
Python-side chemistry to keep in parity with Rust; Rust unit tests
(`cargo test -p pepdistill-core`, see `core/src/chem.rs`, `core/src/peptide.rs`,
`core/src/tokenize.rs`, `core/src/bucket.rs`) are the authoritative coverage.
