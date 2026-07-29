//! Core tokenizer — a pure-Rust port of `pepdistill.data.encode.collate`.
//!
//! Token id is `ord(aa) - AA_OFFSET` (no lookup table). Modification deltas are looked up via
//! `chem::mod_delta` (scaled by `MOD_SCALE`) so the chemistry constants stay single-sourced in
//! `chem.rs`.

use ndarray::{Array1, Array2};

use crate::peptide::{Peptide, Site};

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

/// Pack precursors into `Batch` arrays. `peptides[i].mods` sites are mapped onto the token
/// grid: `Site::Residue(j)` lands at `off + j`; terminal mods fold into the boundary residue's
/// column (they get their own column in a later task).
pub fn collate(peptides: &[Peptide], charges: &[i64]) -> anyhow::Result<CollateArrays> {
    let b = peptides.len();
    // N/C-term tokens are mandatory: 2 extra columns, residues start at index 1.
    let extra = 2;
    let off = 1usize;
    let lengths: Vec<i64> = peptides.iter().map(|p| p.sequence.len() as i64).collect();
    let max_len = lengths.iter().copied().max().unwrap_or(0) as usize;
    let tok_len = max_len + extra;
    let frag_w = tok_len.saturating_sub(1);

    let mut tokens = Array2::<i64>::from_elem((b, tok_len), PAD_IDX);
    let mut mod_delta = Array2::<f32>::zeros((b, tok_len));
    let mut pad_mask = Array2::<bool>::from_elem((b, tok_len), true);
    let mut frag_mask = Array2::<bool>::from_elem((b, frag_w), false);

    for i in 0..b {
        let s = peptides[i].sequence.as_bytes();
        let n = s.len();
        tokens[[i, 0]] = NTERM_IDX;
        tokens[[i, 1 + n]] = CTERM_IDX;
        for j in 0..n {
            tokens[[i, off + j]] = s[j] as i64 - AA_OFFSET;
        }
        let last = n.saturating_sub(1);
        for (site, spec) in &peptides[i].mods {
            let d = spec.delta_mass()?;
            let idx = match site {
                Site::NTerm => 0,
                Site::CTerm => last,
                Site::Residue(j) => {
                    if *j > last {
                        anyhow::bail!("mod site {j} out of range for length {n}");
                    }
                    *j
                }
            };
            mod_delta[[i, off + idx]] += (d as f32) / MOD_SCALE;
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
    use crate::chem;
    use crate::peptide::ModSpec;

    #[test]
    fn collate_shapes_and_tokens() {
        let a = collate(
            &[Peptide::new("PEP".into(), vec![]), Peptide::new("AC".into(), vec![])],
            &[2, 3],
        )
        .unwrap();
        assert_eq!(a.tokens.shape(), &[2, 5]); // max_len=3 + mandatory termini
        assert_eq!(a.tokens[[0, 1]], (b'P' - b'A') as i64); // 15, offset by NTERM
        assert_eq!(a.tokens[[1, 3]], CTERM_IDX); // "AC" (n=2): CTERM lands at 1+n=3
        assert_eq!(a.tokens[[1, 4]], PAD_IDX); // trailing pad column (max_len=3 > n=2)
        assert!(a.pad_mask[[1, 4]]);
    }

    #[test]
    fn collate_with_termini_and_mods() {
        let a = collate(
            &[Peptide::new(
                "AC".into(),
                vec![(Site::Residue(1), ModSpec::Named("Carbamidomethyl@C".to_string()))],
            )],
            &[2],
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
            &[Peptide::new(
                "AC".into(),
                vec![(Site::Residue(0), ModSpec::Named("NotAMod".to_string()))],
            )],
            &[2],
        );
        assert!(res.is_err());
    }
}
