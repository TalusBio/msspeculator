//! Core tokenizer — a pure-Rust port of `pepdistill.data.encode.collate`.
//!
//! Token id is `ord(aa) - AA_OFFSET` (no lookup table). Modifications are exposed as four
//! channels (`mod_arrays`) — element composition, raw mass, and two boolean presence masks —
//! rather than one scaled scalar, so a later routing layer can send a modification through a
//! compositional or a mass-only encoder. Chemistry constants stay single-sourced in `chem.rs`.

use ndarray::{Array1, Array2, Array3};

use crate::composition::N_ELEMENTS;
use crate::peptide::{ModSpec, Peptide, Site};

// Vocab contract — single home. The pyo3 ext re-exports these.
pub const AA_OFFSET: i64 = 65; // ord('A')
pub const PAD_IDX: i64 = 26;
pub const NTERM_IDX: i64 = 27;
pub const CTERM_IDX: i64 = 28;
pub const N_TOKENS: i64 = 29;

pub struct CollateArrays {
    pub tokens: Array2<i64>,
    pub mod_comp: Array3<f32>,
    pub mod_mass: Array2<f32>,
    pub mod_present: Array2<bool>,
    pub mod_named: Array2<bool>,
    pub charge: Array1<i64>,
    pub lengths: Array1<i64>,
    pub pad_mask: Array2<bool>,
    pub frag_mask: Array2<bool>,
}

pub struct ModArrays {
    pub mod_comp: Array3<f32>,   // (B, T, N_ELEMENTS)
    pub mod_mass: Array2<f32>,   // (B, T), Daltons, unscaled
    pub mod_present: Array2<bool>,
    pub mod_named: Array2<bool>,
}

/// Column of the token grid a site occupies. Layout is always `[N] r1..rL [C]`, so residue `i`
/// sits at `1 + i` and the C-term token at `1 + len`.
fn site_column(site: &Site, seq_len: usize) -> usize {
    match site {
        Site::NTerm => 0,
        Site::Residue(i) => 1 + i,
        Site::CTerm => 1 + seq_len,
    }
}

/// Build the four modification channels. `tok_len` is the padded token width (`maxL + 2`).
///
/// Presence is tracked in its own boolean rather than inferred from `mod_mass != 0`: a float
/// comparison would silently mislabel a near-zero delta, and a wrong model input is worse than
/// a loud failure.
pub fn mod_arrays(peptides: &[Peptide], tok_len: usize) -> anyhow::Result<ModArrays> {
    let b = peptides.len();
    let mut mod_comp = Array3::<f32>::zeros((b, tok_len, N_ELEMENTS));
    let mut mod_mass = Array2::<f32>::zeros((b, tok_len));
    let mut mod_present = Array2::<bool>::from_elem((b, tok_len), false);
    let mut mod_named = Array2::<bool>::from_elem((b, tok_len), false);

    for (i, pep) in peptides.iter().enumerate() {
        let n = pep.sequence.len();
        for (site, spec) in &pep.mods {
            // Validate against this peptide's own length first. `tok_len` is the batch's
            // padded width (driven by the longest peptide), so a short peptide's out-of-range
            // residue index can still land inside the padded grid — checking only `col >=
            // tok_len` would silently write the mod into a padding column instead of erroring.
            if let Site::Residue(j) = site {
                if *j >= n {
                    anyhow::bail!("mod site {j} out of range for length {n}");
                }
            }
            let col = site_column(site, n);
            if col >= tok_len {
                anyhow::bail!("column {col} for mod site {site:?} exceeds token width {tok_len}");
            }
            mod_mass[[i, col]] += spec.delta_mass()? as f32;
            mod_present[[i, col]] = true;
            if let ModSpec::Named(name) = spec {
                let ec = crate::chem::mod_element_comp(name)?;
                for (k, v) in ec.iter().enumerate() {
                    mod_comp[[i, col, k]] += *v as f32;
                }
                mod_named[[i, col]] = true;
            }
        }
    }
    Ok(ModArrays { mod_comp, mod_mass, mod_present, mod_named })
}

/// Pack precursors into `Batch` arrays. `peptides[i].mods` sites are mapped onto the token
/// grid via `mod_arrays`: `Site::Residue(j)` lands at `off + j`; termini get their own column
/// (index 0 and `1 + len`).
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
        for p in 0..(n + extra) {
            pad_mask[[i, p]] = false;
        }
        for p in off..(off + n.saturating_sub(1)) {
            frag_mask[[i, p]] = true;
        }
    }
    let ma = mod_arrays(peptides, tok_len)?;
    Ok(CollateArrays {
        tokens,
        mod_comp: ma.mod_comp,
        mod_mass: ma.mod_mass,
        mod_present: ma.mod_present,
        mod_named: ma.mod_named,
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
        let expected = chem::mod_delta("Carbamidomethyl@C").unwrap() as f32;
        assert!((a.mod_mass[[0, 2]] - expected).abs() < 1e-6);
        assert!(a.mod_present[[0, 2]]);
        assert!(a.mod_named[[0, 2]]);
        assert!(!a.pad_mask[[0, 0]]);
        assert!(!a.pad_mask[[0, 3]]);
    }

    #[test]
    fn mod_arrays_mass_only_is_present_but_not_named() {
        let a = collate(
            &[Peptide::new("PEPTIDE".into(), vec![(Site::Residue(2), ModSpec::MassOnly(42.010565))])],
            &[2],
        )
        .unwrap();
        let col = 1 + 2;
        assert!(a.mod_present[[0, col]]);
        assert!(!a.mod_named[[0, col]]);
        assert!((a.mod_mass[[0, col]] - 42.010565).abs() < 1e-5);
        for k in 0..N_ELEMENTS {
            assert_eq!(a.mod_comp[[0, col, k]], 0.0);
        }
    }

    #[test]
    fn mod_site_out_of_range_for_its_own_peptide_errors_even_within_padded_width() {
        // "AC" (n=2) has a Residue(3) mod: out of range for itself (3 >= 2), but its column
        // (1+3=4) still fits inside the batch's padded tok_len (7+2=9, from "PEPTIDE"). Without
        // per-peptide validation this silently writes the mod into "AC"'s padding column instead
        // of erroring.
        let res = collate(
            &[
                Peptide::new("AC".into(), vec![(Site::Residue(3), ModSpec::MassOnly(1.0))]),
                Peptide::new("PEPTIDE".into(), vec![]),
            ],
            &[2, 2],
        );
        assert!(res.is_err(), "mod site out of range for its own peptide must error");
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
