//! Core tokenizer — a pure-Rust port of `pepdistill.data.encode.collate`.
//!
//! Token id is `ord(aa) - AA_OFFSET` (no lookup table). Modification deltas are looked up via
//! `chem::mod_delta` (scaled by `MOD_SCALE`) so the chemistry constants stay single-sourced in
//! `chem.rs`.

use ndarray::{Array1, Array2};

use crate::chem;

// Vocab contract — single home. The pyo3 ext re-exports these.
pub const AA_OFFSET: i64 = 65; // ord('A')
pub const PAD_IDX: i64 = 26;
pub const NTERM_IDX: i64 = 27;
pub const CTERM_IDX: i64 = 28;
pub const N_TOKENS: i64 = 29;
pub const MOD_SCALE: f32 = 100.0;

pub struct CollateArrays {
    pub tokens: Array2<i64>,
    pub mod_delta: Array2<f32>,
    pub charge: Array1<i64>,
    pub lengths: Array1<i64>,
    pub pad_mask: Array2<bool>,
    pub frag_mask: Array2<bool>,
}

/// Pack precursors into `Batch` arrays. `mod_sites[i]` / `mod_names[i]` are parallel
/// per-precursor lists: residue index (0-based within the sequence) and the modification
/// name to look up via `chem::mod_delta`.
pub fn collate(
    seqs: &[String],
    charges: &[i64],
    mod_sites: &[Vec<usize>],
    mod_names: &[Vec<String>],
    use_termini: bool,
) -> anyhow::Result<CollateArrays> {
    let b = seqs.len();
    let extra = if use_termini { 2 } else { 0 };
    let off = if use_termini { 1usize } else { 0 };
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
            tokens[[i, off + j]] = s[j] as i64 - AA_OFFSET;
        }
        for (k, &site) in mod_sites[i].iter().enumerate() {
            let d = chem::mod_delta(&mod_names[i][k])
                .ok_or_else(|| anyhow::anyhow!("unknown modification {}", mod_names[i][k]))?;
            mod_delta[[i, off + site]] += (d as f32) / MOD_SCALE;
        }
        for p in 0..(n + extra) {
            pad_mask[[i, p]] = false;
        }
        for p in off..(off + n.saturating_sub(1)) {
            frag_mask[[i, p]] = true;
        }
    }
    Ok(CollateArrays {
        tokens,
        mod_delta,
        charge: Array1::from(charges.to_vec()),
        lengths: Array1::from(lengths),
        pad_mask,
        frag_mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collate_no_termini_shapes_and_tokens() {
        let a = collate(
            &["PEP".into(), "AC".into()],
            &[2, 3],
            &[vec![], vec![]],
            &[vec![], vec![]],
            false,
        )
        .unwrap();
        assert_eq!(a.tokens.shape(), &[2, 3]); // max_len=3, no termini
        assert_eq!(a.tokens[[0, 0]], (b'P' - b'A') as i64); // 15
        assert_eq!(a.tokens[[1, 2]], PAD_IDX); // "AC" padded to len 3
        assert!(a.pad_mask[[1, 2]]);
    }

    #[test]
    fn collate_with_termini_and_mods() {
        let a = collate(
            &["AC".into()],
            &[2],
            &[vec![1usize]],
            &[vec!["Carbamidomethyl@C".to_string()]],
            true,
        )
        .unwrap();
        // tok_len = 2 + 2 = 4: [NTERM, A, C, CTERM]
        assert_eq!(a.tokens.shape(), &[1, 4]);
        assert_eq!(a.tokens[[0, 0]], NTERM_IDX);
        assert_eq!(a.tokens[[0, 1]], (b'A' - b'A') as i64);
        assert_eq!(a.tokens[[0, 2]], (b'C' - b'A') as i64);
        assert_eq!(a.tokens[[0, 3]], CTERM_IDX);
        // mod site 1 (0-based within seq) lands at offset 1+1=2
        let expected = (chem::mod_delta("Carbamidomethyl@C").unwrap() as f32) / MOD_SCALE;
        assert!((a.mod_delta[[0, 2]] - expected).abs() < 1e-6);
        assert!(!a.pad_mask[[0, 0]]);
        assert!(!a.pad_mask[[0, 3]]);
    }

    #[test]
    fn collate_unknown_mod_errors() {
        let res = collate(
            &["AC".into()],
            &[2],
            &[vec![0usize]],
            &[vec!["NotAMod".to_string()]],
            false,
        );
        assert!(res.is_err());
    }
}
