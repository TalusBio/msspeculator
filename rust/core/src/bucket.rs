//! Batched, length-bucketed tokenizer + fragment m/z — a pure-Rust port of
//! `pepdistill.predict.fast._bucket_arrays` / `_fragment_mz`.
//!
//! All precursors passed to one call share `length` (the bucketing key), so tokens and
//! residue masses pack into dense rectangular arrays with no per-fragment Python/loop
//! equivalent — the fragment m/z math is fully vectorized over the batch.

use ndarray::{Array1, Array2, Array3};

use crate::chem::{self, H2O, ION_TYPES, PROTON};
use crate::peptide::{Peptide, Site};
use crate::tokenize::{self, AA_OFFSET, CTERM_IDX, NTERM_IDX, PAD_IDX};

pub struct BucketArrays {
    pub tokens: Array2<i64>,
    pub mod_comp: Array3<f32>,
    pub mod_mass: Array2<f32>,
    pub mod_present: Array2<bool>,
    pub mod_named: Array2<bool>,
    pub charge: Array1<i64>,
    pub residue_mass: Array2<f64>,
}

/// Dense token/mod/charge/residue-mass arrays for a same-length bucket.
///
/// Tokens are wrapped with N/C-term ids -> shape `(B, length+2)`. `residue_mass` stays
/// `(B, length)`: termini carry no mass and never enter m/z. The four mod channels come from
/// `tokenize::mod_arrays`; `residue_mass` folds terminal mods into the boundary residue
/// separately (N-term -> residue 0, C-term -> last), matching `Peptide::residue_masses`.
pub fn bucket_arrays(
    peptides: &[Peptide],
    charges: &[i64],
    length: usize,
) -> anyhow::Result<BucketArrays> {
    let b = peptides.len();
    // N/C-term tokens are mandatory: 2 extra columns, residues start at index 1.
    let extra = 2;
    let off = crate::tokenize::FRAG_OFFSET;
    let tok_len = length + extra;
    let mut tokens = Array2::<i64>::from_elem((b, tok_len), PAD_IDX);
    let mut residue_mass = Array2::<f64>::zeros((b, length));
    let last = length.saturating_sub(1);
    for i in 0..b {
        let s = peptides[i].sequence.as_bytes();
        tokens[[i, 0]] = NTERM_IDX;
        tokens[[i, 1 + length]] = CTERM_IDX;
        for j in 0..length {
            tokens[[i, off + j]] = s[j] as i64 - AA_OFFSET;
            residue_mass[[i, j]] = chem::residue_mass(s[j])
                .ok_or_else(|| anyhow::anyhow!("unsupported residue {:?}", s[j] as char))?;
        }
        for (site, spec) in &peptides[i].mods {
            let d = spec.delta_mass()?;
            let idx = match site {
                Site::NTerm => 0,
                Site::CTerm => last,
                Site::Residue(j) => {
                    if *j > last {
                        anyhow::bail!("mod site {j} out of range for length {length}");
                    }
                    *j
                }
            };
            residue_mass[[i, idx]] += d;
        }
    }
    let ma = tokenize::mod_arrays(peptides, tok_len)?;
    Ok(BucketArrays {
        tokens,
        mod_comp: ma.mod_comp,
        mod_mass: ma.mod_mass,
        mod_present: ma.mod_present,
        mod_named: ma.mod_named,
        charge: Array1::from(charges.to_vec()),
        residue_mass,
    })
}

/// Vectorized fragment m/z and precursor m/z for a same-length bucket.
///
/// Returns `(mz [B, L-1, n_ion], precursor_mz [B])`.
pub fn bucket_fragment_mz(
    residue_mass: &Array2<f64>,
    charge: &Array1<i64>,
) -> (Array3<f64>, Array1<f64>) {
    let b = residue_mass.nrows();
    let l = residue_mass.ncols();
    let n_ion = ION_TYPES.len();
    let mut mz = Array3::<f64>::zeros((b, l - 1, n_ion));
    let mut pmz = Array1::<f64>::zeros(b);
    for r in 0..b {
        // prefix sums
        let mut prefix = vec![0.0_f64; l];
        let mut acc = 0.0;
        for (p, m) in prefix.iter_mut().zip(residue_mass.row(r).iter()) {
            acc += *m;
            *p = acc;
        }
        let total = prefix[l - 1];
        for i in 0..l - 1 {
            let b_neutral = prefix[i];
            let y_neutral = total - prefix[i] + H2O;
            for (col, &(is_b, z)) in ION_TYPES.iter().enumerate() {
                let neutral = if is_b { b_neutral } else { y_neutral };
                mz[[r, i, col]] = (neutral + z as f64 * PROTON) / z as f64;
            }
        }
        pmz[r] = (total + H2O + charge[r] as f64 * PROTON) / charge[r] as f64;
    }
    (mz, pmz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peptide::ModSpec;

    #[test]
    fn bucket_fragment_mz_matches_scalar() {
        // one peptide "SAMPLER" (len 7), charge 2, no mods
        let ba = bucket_arrays(&[Peptide::new("SAMPLER".into(), vec![])], &[2], 7).unwrap();
        let (mz, pmz) = bucket_fragment_mz(&ba.residue_mass, &ba.charge);
        let rm = crate::chem::residue_masses(b"SAMPLER").unwrap();
        // column 0 is (b,1); position 0 -> b1
        let b1 = crate::chem::fragment_mz(&rm, "b", 1, 1).unwrap();
        assert!((mz[[0, 0, 0]] - b1).abs() < 1e-9);
        assert!((pmz[0] - crate::chem::precursor_mz(&rm, 2)).abs() < 1e-9);
    }

    #[test]
    fn bucket_arrays_termini_and_mods() {
        let ba = bucket_arrays(
            &[Peptide::new(
                "AC".into(),
                vec![(
                    Site::Residue(1),
                    ModSpec::Named("Carbamidomethyl@C".to_string()),
                )],
            )],
            &[2],
            2,
        )
        .unwrap();
        assert_eq!(ba.tokens.shape(), &[1, 4]);
        assert_eq!(ba.tokens[[0, 0]], NTERM_IDX);
        assert_eq!(ba.tokens[[0, 3]], CTERM_IDX);
        assert_eq!(ba.residue_mass.shape(), &[1, 2]);
        let base_c = chem::residue_mass(b'C').unwrap();
        let d = chem::mod_delta("Carbamidomethyl@C").unwrap();
        assert!((ba.residue_mass[[0, 1]] - (base_c + d)).abs() < 1e-9);
    }

    #[test]
    fn bucket_fragment_mz_batches_independently() {
        // Two same-length peptides in one batch; each row's mz should match its own scalar calc.
        let ba = bucket_arrays(
            &[
                Peptide::new("SAMPLER".into(), vec![]),
                Peptide::new("GAMPLEK".into(), vec![]),
            ],
            &[2, 1],
            7,
        )
        .unwrap();
        let (mz, pmz) = bucket_fragment_mz(&ba.residue_mass, &ba.charge);
        let rm1 = crate::chem::residue_masses(b"GAMPLEK").unwrap();
        let y1 = crate::chem::fragment_mz(&rm1, "y", 1, 1).unwrap();
        assert!((mz[[1, 5, 1]] - y1).abs() < 1e-9);
        assert!((pmz[1] - crate::chem::precursor_mz(&rm1, 1)).abs() < 1e-9);
    }
}
