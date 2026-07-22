//! Rust tokenizer/batcher for pepdistill.
//!
//! `collate` is a 1:1 port of `pepdistill.data.encode.collate`: it packs a list of
//! precursors into the six numpy arrays that back the Python `Batch`. The token id is
//! `ord(aa) - AA_OFFSET` (no lookup table) — the whole reason this ports so cleanly. The
//! Python side stays the reference oracle; a parity test asserts these arrays match it, so
//! the soft "contract" (these four constants) can't drift silently.
//!
//! Modification *values* are supplied by the caller (already `MOD_DELTA[name] / MOD_SCALE`)
//! so the chemistry constants stay single-sourced in Python's `chem.py`; Rust only places
//! them at their sites.

use ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// Vocab contract — must equal pepdistill.data.encode. Guarded by the parity test.
const AA_OFFSET: i64 = 65; // ord('A')
const PAD_IDX: i64 = 26;
const NTERM_IDX: i64 = 27;
const CTERM_IDX: i64 = 28;

/// Pack precursors into Batch arrays.
///
/// `mod_sites[i]` / `mod_deltas[i]` are parallel per-precursor lists: residue index (0-based
/// within the sequence) and the already-scaled delta to add at that site.
#[pyfunction]
fn collate<'py>(
    py: Python<'py>,
    seqs: Vec<String>,
    charges: Vec<i64>,
    mod_sites: Vec<Vec<usize>>,
    mod_deltas: Vec<Vec<f32>>,
    use_termini: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let b = seqs.len();
    let extra = if use_termini { 2 } else { 0 };
    let off = if use_termini { 1usize } else { 0usize };

    let lengths: Vec<i64> = seqs.iter().map(|s| s.len() as i64).collect();
    let max_len = lengths.iter().copied().max().unwrap_or(0) as usize;
    let tok_len = max_len + extra;
    let frag_w = tok_len.saturating_sub(1);

    let mut tokens = Array2::<i64>::from_elem((b, tok_len), PAD_IDX);
    let mut mod_delta = Array2::<f32>::zeros((b, tok_len));
    let mut pad_mask = Array2::<bool>::from_elem((b, tok_len), true);
    let mut frag_mask = Array2::<bool>::from_elem((b, frag_w), false);

    for i in 0..b {
        let s = seqs[i].as_bytes();
        let n = s.len();
        if use_termini {
            tokens[[i, 0]] = NTERM_IDX;
            tokens[[i, 1 + n]] = CTERM_IDX;
        }
        for j in 0..n {
            tokens[[i, off + j]] = s[j] as i64 - AA_OFFSET; // ord(aa) - ord('A')
        }
        for (k, &site) in mod_sites[i].iter().enumerate() {
            mod_delta[[i, off + site]] += mod_deltas[i][k];
        }
        for p in 0..(n + extra) {
            pad_mask[[i, p]] = false;
        }
        // Valid inter-residue fragment sites: adjacent-pool indices [off, off + n - 1).
        for p in off..(off + n.saturating_sub(1)) {
            frag_mask[[i, p]] = true;
        }
    }

    let d = PyDict::new_bound(py);
    d.set_item("tokens", tokens.into_pyarray_bound(py))?;
    d.set_item("mod_delta", mod_delta.into_pyarray_bound(py))?;
    d.set_item("charge", Array1::from(charges).into_pyarray_bound(py))?;
    d.set_item("lengths", Array1::from(lengths).into_pyarray_bound(py))?;
    d.set_item("pad_mask", pad_mask.into_pyarray_bound(py))?;
    d.set_item("frag_mask", frag_mask.into_pyarray_bound(py))?;
    Ok(d)
}

#[pymodule]
fn pepdistill_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(collate, m)?)?;
    m.add("AA_OFFSET", AA_OFFSET)?;
    m.add("PAD_IDX", PAD_IDX)?;
    m.add("NTERM_IDX", NTERM_IDX)?;
    m.add("CTERM_IDX", CTERM_IDX)?;
    Ok(())
}
