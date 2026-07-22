# pepdistill_rs

Rust tokenizer/batcher. Optional accelerator for `pepdistill.data.encode.collate` — the
Python version stays the reference oracle; this packs the same numpy arrays in Rust.

## Build into the project venv

```sh
uv run maturin develop -m rust/Cargo.toml --release
```

Then `pepdistill.data.encode_rs.HAVE_RS` is `True` and `tests/test_encode_rs.py`
(the Python↔Rust parity guard) runs instead of skipping.

## Contract

Vocab constants are duplicated in `rust/src/lib.rs` (`AA_OFFSET/PAD_IDX/NTERM_IDX/CTERM_IDX`)
and `src/pepdistill/data/encode.py`. Soft/greenfield — change freely, but the parity test
fails if they drift. Chemistry (`MOD_DELTA`, `MOD_SCALE`) is NOT duplicated: the Python
caller passes already-scaled `(site, delta)` pairs, so `chem.py` stays single-source.
