//! Modified peptide — the data + mass/rendering behavior Python's `chem.Peptide` had.

use crate::chem;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Peptide {
    pub sequence: String,
    pub mods: Vec<(usize, String)>, // normalized: sorted for canonical eq/hash
}

impl Peptide {
    pub fn new(sequence: String, mut mods: Vec<(usize, String)>) -> Self {
        mods.sort();
        Self { sequence, mods }
    }

    pub fn length(&self) -> usize {
        self.sequence.len()
    }

    pub fn residue_masses(&self) -> anyhow::Result<Vec<f64>> {
        chem::residue_masses_mod(self.sequence.as_bytes(), &self.mods)
    }

    pub fn mono_mass(&self) -> anyhow::Result<f64> {
        Ok(chem::mono_mass(&self.residue_masses()?))
    }

    pub fn precursor_mz(&self, charge: i64) -> anyhow::Result<f64> {
        Ok((self.mono_mass()? + charge as f64 * chem::PROTON) / charge as f64)
    }

    /// e.g. "AC[Carbamidomethyl@C]DEM[Oxidation@M]K"
    pub fn modified_sequence(&self) -> String {
        let by_site: std::collections::HashMap<usize, &str> =
            self.mods.iter().map(|(s, n)| (*s, n.as_str())).collect();
        let mut out = String::new();
        for (i, aa) in self.sequence.chars().enumerate() {
            out.push(aa);
            if let Some(name) = by_site.get(&i) {
                out.push('[');
                out.push_str(name);
                out.push(']');
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precursor_mz_charge2() {
        let p = Peptide::new("PEPTIDE".into(), vec![]);
        let expected = (799.35997 + 2.0 * crate::chem::PROTON) / 2.0;
        assert!((p.precursor_mz(2).unwrap() - expected).abs() < 0.01);
    }

    #[test]
    fn modified_sequence_rendering() {
        let p = Peptide::new(
            "ACDEMK".into(),
            vec![(4, "Oxidation@M".into()), (1, "Carbamidomethyl@C".into())],
        );
        assert_eq!(p.modified_sequence(), "AC[Carbamidomethyl@C]DEM[Oxidation@M]K");
    }

    #[test]
    fn mods_sorted_canonical_for_eq_hash() {
        let a = Peptide::new("ACDEMK".into(), vec![(4, "Oxidation@M".into()), (1, "Carbamidomethyl@C".into())]);
        let b = Peptide::new("ACDEMK".into(), vec![(1, "Carbamidomethyl@C".into()), (4, "Oxidation@M".into())]);
        assert_eq!(a, b);
    }
}
