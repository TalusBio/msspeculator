//! Modified peptide — data plus mass/rendering behavior.

use crate::chem;

/// Where a modification sits. Ordering is deliberate: `derive(Ord)` uses declaration order, so
/// sorting a mods vector already produces N-term → residues → C-term without a custom compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Site {
    NTerm,
    Residue(usize),
    CTerm,
}

/// What the modification is. `Named` resolves to a composition and can drive the compositional
/// encoder; `MassOnly` carries a bare delta and can only drive the mass encoder. Keeping these
/// distinct at the type level is what stops a mass-only mod from acquiring a composition it
/// never had.
#[derive(Debug, Clone)]
pub enum ModSpec {
    Named(String),
    MassOnly(f64),
}

// PartialEq is hand-written (not derived) so it agrees bit-for-bit with Hash and Ord below,
// both of which compare `f64` via `to_bits()`. Derived PartialEq would use IEEE equality
// (-0.0 == 0.0, NaN != NaN), which disagrees with to_bits()-based Hash/Ord and would corrupt
// HashMap/HashSet lookups keyed on Peptide (Python uses Peptide as a dict key and dedup key).
impl PartialEq for ModSpec {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ModSpec::Named(a), ModSpec::Named(b)) => a == b,
            (ModSpec::MassOnly(a), ModSpec::MassOnly(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }
}

impl Eq for ModSpec {}

impl std::hash::Hash for ModSpec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ModSpec::Named(n) => (0u8, n).hash(state),
            ModSpec::MassOnly(m) => (1u8, m.to_bits()).hash(state),
        }
    }
}

impl PartialOrd for ModSpec {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ModSpec {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (ModSpec::Named(a), ModSpec::Named(b)) => a.cmp(b),
            (ModSpec::MassOnly(a), ModSpec::MassOnly(b)) => a.to_bits().cmp(&b.to_bits()),
            (ModSpec::Named(_), ModSpec::MassOnly(_)) => std::cmp::Ordering::Less,
            (ModSpec::MassOnly(_), ModSpec::Named(_)) => std::cmp::Ordering::Greater,
        }
    }
}

impl ModSpec {
    /// Monoisotopic delta in Da. Errors on an unresolvable name rather than contributing zero.
    pub fn delta_mass(&self) -> anyhow::Result<f64> {
        match self {
            ModSpec::MassOnly(m) => Ok(*m),
            ModSpec::Named(n) => {
                chem::mod_delta(n).ok_or_else(|| anyhow::anyhow!("unknown modification {n}"))
            }
        }
    }

    /// Rendered form inside a modified sequence.
    pub fn render(&self) -> String {
        match self {
            ModSpec::Named(n) => n.clone(),
            ModSpec::MassOnly(m) => format!("{m:+}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Peptide {
    pub sequence: String,
    pub mods: Vec<(Site, ModSpec)>, // normalized: sorted for canonical eq/hash
}

impl Peptide {
    pub fn new(sequence: String, mut mods: Vec<(Site, ModSpec)>) -> Self {
        mods.sort();
        Self { sequence, mods }
    }

    pub fn length(&self) -> usize {
        self.sequence.len()
    }

    /// Per-residue masses with every modification's delta folded in. Terminal mods add to the
    /// first / last residue for mass purposes — they occupy their own embedding column, but
    /// chemically they sit on the peptide's ends.
    pub fn residue_masses(&self) -> anyhow::Result<Vec<f64>> {
        let mut rm = chem::residue_masses(self.sequence.as_bytes())?;
        if rm.is_empty() {
            anyhow::bail!("empty peptide has no residue masses");
        }
        let last = rm.len() - 1;
        for (site, spec) in &self.mods {
            let d = spec.delta_mass()?;
            let idx = match site {
                Site::NTerm => 0,
                Site::CTerm => last,
                Site::Residue(i) => {
                    if *i > last {
                        anyhow::bail!("mod site {i} out of range for length {}", rm.len());
                    }
                    *i
                }
            };
            rm[idx] += d;
        }
        Ok(rm)
    }

    pub fn mono_mass(&self) -> anyhow::Result<f64> {
        Ok(chem::mono_mass(&self.residue_masses()?))
    }

    pub fn precursor_mz(&self, charge: i64) -> anyhow::Result<f64> {
        Ok((self.mono_mass()? + charge as f64 * chem::PROTON) / charge as f64)
    }

    /// e.g. "[TMT6plex]ET[Phospho]TLHLVLR", or "P[+79.96633]EPTIDE" for a mass-only mod.
    pub fn modified_sequence(&self) -> String {
        let mut out = String::new();
        for (site, spec) in &self.mods {
            if *site == Site::NTerm {
                out.push('[');
                out.push_str(&spec.render());
                out.push(']');
            }
        }
        for (i, aa) in self.sequence.chars().enumerate() {
            out.push(aa);
            for (site, spec) in &self.mods {
                if *site == Site::Residue(i) {
                    out.push('[');
                    out.push_str(&spec.render());
                    out.push(']');
                }
            }
        }
        for (site, spec) in &self.mods {
            if *site == Site::CTerm {
                out.push('[');
                out.push_str(&spec.render());
                out.push(']');
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn hash_of<T: Hash>(v: &T) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    #[test]
    fn mass_only_eq_ord_hash_agree_on_signed_zero() {
        // -0.0 == 0.0 under IEEE float equality, but they're different bit patterns. PartialEq
        // must agree with Ord/Hash (both to_bits()-based), so they must compare UNEQUAL
        // everywhere, not just in Ord/Hash — a derived (IEEE) PartialEq would disagree here.
        let neg = ModSpec::MassOnly(-0.0);
        let pos = ModSpec::MassOnly(0.0);
        assert_ne!(neg, pos);
        assert_ne!(neg.cmp(&pos), std::cmp::Ordering::Equal);
        assert_ne!(hash_of(&neg), hash_of(&pos));
    }

    #[test]
    fn mass_only_peptide_equals_itself() {
        // Reflexivity: in particular this must hold for NaN, where derived (IEEE) PartialEq
        // would report NaN != NaN and break the Eq contract.
        let p = Peptide::new("PEPTIDE".into(), vec![(Site::Residue(0), ModSpec::MassOnly(f64::NAN))]);
        let q = Peptide::new("PEPTIDE".into(), vec![(Site::Residue(0), ModSpec::MassOnly(f64::NAN))]);
        assert_eq!(p, p);
        assert_eq!(p, q);
        assert_eq!(hash_of(&p), hash_of(&q));
    }

    #[test]
    fn sites_sort_nterm_first_cterm_last() {
        let mut v = vec![Site::CTerm, Site::Residue(3), Site::NTerm, Site::Residue(1)];
        v.sort();
        assert_eq!(v, vec![Site::NTerm, Site::Residue(1), Site::Residue(3), Site::CTerm]);
    }

    #[test]
    fn modified_sequence_renders_termini_and_mass_only() {
        let p = Peptide::new(
            "ETTLHLVLR".into(),
            vec![
                (Site::Residue(1), ModSpec::Named("Phospho".into())),
                (Site::NTerm, ModSpec::Named("TMT6plex".into())),
            ],
        );
        assert_eq!(p.modified_sequence(), "[TMT6plex]ET[Phospho]TLHLVLR");

        let q = Peptide::new(
            "PEPTIDE".into(),
            vec![(Site::Residue(0), ModSpec::MassOnly(79.96633))],
        );
        assert_eq!(q.modified_sequence(), "P[+79.96633]EPTIDE");

        let r = Peptide::new("PEK".into(), vec![(Site::CTerm, ModSpec::Named("Phospho".into()))]);
        assert_eq!(r.modified_sequence(), "PEK[Phospho]");
    }

    #[test]
    fn terminal_and_side_chain_mods_coexist_on_residue_zero() {
        let p = Peptide::new(
            "KPEPTIDE".into(),
            vec![
                (Site::NTerm, ModSpec::Named("TMT6plex".into())),
                (Site::Residue(0), ModSpec::Named("TMT6plex".into())),
            ],
        );
        assert_eq!(p.mods.len(), 2, "two distinct sites, not one merged mod");
        let base = Peptide::new("KPEPTIDE".into(), vec![]).mono_mass().unwrap();
        let d = crate::chem::mod_delta("TMT6plex").unwrap();
        assert!((p.mono_mass().unwrap() - (base + 2.0 * d)).abs() < 1e-6);
    }

    #[test]
    fn mass_only_mod_contributes_its_delta() {
        let base = Peptide::new("PEPTIDE".into(), vec![]).mono_mass().unwrap();
        let p = Peptide::new(
            "PEPTIDE".into(),
            vec![(Site::Residue(2), ModSpec::MassOnly(42.010565))],
        );
        assert!((p.mono_mass().unwrap() - (base + 42.010565)).abs() < 1e-9);
    }

    #[test]
    fn unknown_named_mod_errors() {
        let p = Peptide::new("PEK".into(), vec![(Site::Residue(0), ModSpec::Named("Nope".into()))]);
        assert!(p.mono_mass().is_err());
    }

    #[test]
    fn precursor_mz_charge2() {
        let p = Peptide::new("PEPTIDE".into(), vec![]);
        let expected = (799.35997 + 2.0 * crate::chem::PROTON) / 2.0;
        assert!((p.precursor_mz(2).unwrap() - expected).abs() < 0.01);
    }

    #[test]
    fn mods_sorted_canonical_for_eq_hash() {
        let a = Peptide::new("ACDEMK".into(), vec![
            (Site::Residue(4), ModSpec::Named("Oxidation@M".into())),
            (Site::NTerm, ModSpec::Named("TMT6plex".into())),
        ]);
        let b = Peptide::new("ACDEMK".into(), vec![
            (Site::NTerm, ModSpec::Named("TMT6plex".into())),
            (Site::Residue(4), ModSpec::Named("Oxidation@M".into())),
        ]);
        assert_eq!(a, b);
    }
}
