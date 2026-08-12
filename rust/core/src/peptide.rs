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
            ModSpec::Named(n) => chem::mod_delta(n),
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

    /// Parse the form [`Peptide::modified_sequence`] renders: a leading `[mod]` is the
    /// N-terminus, a `[mod]` after the final residue is the C-terminus, and a body starting
    /// with `+`/`-` is a bare mass delta rather than a name.
    ///
    /// A trailing `[mod]` becomes [`Site::CTerm`], not a side-chain mod on the last residue —
    /// that is what makes the round trip exact, since `modified_sequence` renders both the same
    /// way. A caller wanting a side-chain mod on the final residue must build the `Peptide`
    /// directly.
    ///
    /// Known limitation: a mod NAME containing `]` cannot round-trip. Some UNIMOD titles do
    /// (e.g. `Xlink:DSS[156]`); `modified_sequence` renders them unquoted and `parse` then
    /// stops at the inner `]`. It fails loudly — the tail becomes an unexpected character or a
    /// bad site rather than a silently wrong mod — and no such title is reachable from
    /// PROSPECT's frozen accession set, so the delimiter stays unescaped. Escaping or quoting
    /// the body is the fix if a bracketed name ever has to be supported.
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        let mut sequence = String::new();
        let mut mods: Vec<(Site, ModSpec)> = Vec::new();
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '[' {
                let mut body = String::new();
                let mut closed = false;
                for c2 in chars.by_ref() {
                    if c2 == ']' {
                        closed = true;
                        break;
                    }
                    body.push(c2);
                }
                if !closed {
                    anyhow::bail!("unclosed '[' in peptide {s:?}");
                }
                let spec =
                    if body.starts_with('+') || body.starts_with('-') {
                        ModSpec::MassOnly(body.parse().map_err(|_| {
                            anyhow::anyhow!("bad mass delta {body:?} in peptide {s:?}")
                        })?)
                    } else {
                        ModSpec::Named(body)
                    };
                let site = if sequence.is_empty() {
                    Site::NTerm
                } else if chars.peek().is_none() {
                    Site::CTerm
                } else {
                    Site::Residue(sequence.len() - 1)
                };
                mods.push((site, spec));
            } else if c.is_ascii_uppercase() {
                sequence.push(c);
            } else {
                anyhow::bail!("unexpected character {c:?} in peptide {s:?}");
            }
        }
        if sequence.is_empty() {
            anyhow::bail!("peptide {s:?} has no residues");
        }
        Ok(Self::new(sequence, mods))
    }

    /// Reject a site that carries both a `Named` mod and a `MassOnly` delta.
    ///
    /// Several `Named` mods on one site are fine (their compositions accumulate) and so are
    /// several `MassOnly` deltas (their masses sum). One of each is not: the two route to
    /// different encoders, and `tokenize::mod_arrays` has exactly one boolean per column to
    /// choose with. Marking the column "named" would drop the mass-only delta from the model
    /// input while it still moves `mono_mass` and every fragment m/z; folding the delta into
    /// the mass channel instead would drop the named mod's composition. Neither is correct, so
    /// refuse — the standing rule here is errors over degradation.
    ///
    /// Checked on `Site`, not on the residue index `residue_masses` folds onto, so this agrees
    /// exactly with `mod_arrays`' notion of a column: an N-term mod and a `Residue(0)` mod are
    /// different sites even though both add to residue 0's mass.
    pub fn validate_mod_specs(&self) -> anyhow::Result<()> {
        for (i, (site, spec)) in self.mods.iter().enumerate() {
            for (other_site, other) in &self.mods[i + 1..] {
                if site != other_site {
                    continue;
                }
                let mixed = matches!(
                    (spec, other),
                    (ModSpec::Named(_), ModSpec::MassOnly(_))
                        | (ModSpec::MassOnly(_), ModSpec::Named(_))
                );
                if mixed {
                    anyhow::bail!(
                        "site {site:?} of {} carries both a named modification and a mass-only \
                         delta ([{}] and [{}]); named mods route through the composition \
                         encoder and mass-only deltas through the mass encoder, so a single \
                         site cannot carry one of each",
                        self.modified_sequence(),
                        spec.render(),
                        other.render()
                    );
                }
            }
        }
        Ok(())
    }

    /// Per-residue masses with every modification's delta folded in. Terminal mods add to the
    /// first / last residue for mass purposes — they occupy their own embedding column, but
    /// chemically they sit on the peptide's ends.
    ///
    /// Refuses the same mixed-spec sites [`Peptide::validate_mod_specs`] describes: a peptide
    /// the tokenizer cannot encode must not be quietly mass-computable here either, or a
    /// library row would carry m/z values for a molecule no model input ever described.
    pub fn residue_masses(&self) -> anyhow::Result<Vec<f64>> {
        self.validate_mod_specs()?;
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
        let p = Peptide::new(
            "PEPTIDE".into(),
            vec![(Site::Residue(0), ModSpec::MassOnly(f64::NAN))],
        );
        let q = Peptide::new(
            "PEPTIDE".into(),
            vec![(Site::Residue(0), ModSpec::MassOnly(f64::NAN))],
        );
        assert_eq!(p, p);
        assert_eq!(p, q);
        assert_eq!(hash_of(&p), hash_of(&q));
    }

    #[test]
    fn sites_sort_nterm_first_cterm_last() {
        let mut v = vec![Site::CTerm, Site::Residue(3), Site::NTerm, Site::Residue(1)];
        v.sort();
        assert_eq!(
            v,
            vec![Site::NTerm, Site::Residue(1), Site::Residue(3), Site::CTerm]
        );
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

        let r = Peptide::new(
            "PEK".into(),
            vec![(Site::CTerm, ModSpec::Named("Phospho".into()))],
        );
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
        let p = Peptide::new(
            "PEK".into(),
            vec![(Site::Residue(0), ModSpec::Named("Nope".into()))],
        );
        assert!(p.mono_mass().is_err());
    }

    #[test]
    fn precursor_mz_charge2() {
        let p = Peptide::new("PEPTIDE".into(), vec![]);
        let expected = (799.35997 + 2.0 * crate::chem::PROTON) / 2.0;
        assert!((p.precursor_mz(2).unwrap() - expected).abs() < 0.01);
    }

    #[test]
    fn parse_roundtrips_modified_sequence() {
        for s in [
            "PEPTIDE",
            "AC[Carbamidomethyl@C]DEM[Oxidation@M]K",
            "[TMT6plex]ET[Phospho]TLHLVLR",
            "PEK[Phospho]",
            "P[+79.96633]EPTIDE",
        ] {
            let p = Peptide::parse(s).unwrap();
            assert_eq!(p.modified_sequence(), s, "roundtrip failed for {s}");
        }
    }

    #[test]
    fn parse_rejects_unclosed_bracket() {
        assert!(Peptide::parse("PEP[Phospho").is_err());
    }

    #[test]
    fn parse_rejects_lowercase_residue() {
        assert!(Peptide::parse("pePTIDE").is_err());
    }

    #[test]
    fn parse_places_sites_where_collate_expects_them() {
        // The round-trip test cannot distinguish CTerm from Residue(last) — both render the
        // same. Pin the documented choice explicitly, because they occupy different embedding
        // columns.
        let p = Peptide::parse("PEK[Phospho]").unwrap();
        assert_eq!(
            p.mods,
            vec![(Site::CTerm, ModSpec::Named("Phospho".into()))]
        );
        let q = Peptide::parse("[TMT6plex]PEPC[Carbamidomethyl@C]IDER").unwrap();
        assert_eq!(
            q.mods,
            vec![
                (Site::NTerm, ModSpec::Named("TMT6plex".into())),
                (Site::Residue(3), ModSpec::Named("Carbamidomethyl@C".into())),
            ]
        );
    }

    #[test]
    fn parse_keeps_co_sited_mods_on_one_site() {
        // Two brackets after one residue are two mods on that residue, not on consecutive
        // ones. The runtime accumulates their compositions into a single encoder call, so the
        // site must be shared.
        let p = Peptide::parse("PEPC[Oxidation@M][Phospho]IDER").unwrap();
        assert_eq!(
            p.mods,
            vec![
                (Site::Residue(3), ModSpec::Named("Oxidation@M".into())),
                (Site::Residue(3), ModSpec::Named("Phospho".into())),
            ]
        );
        assert_eq!(p.modified_sequence(), "PEPC[Oxidation@M][Phospho]IDER");
    }

    #[test]
    fn parse_rejects_empty_and_residue_free_input() {
        assert!(Peptide::parse("").is_err());
        assert!(Peptide::parse("[TMT6plex]").is_err());
    }

    #[test]
    fn parse_rejects_a_bad_mass_delta() {
        assert!(Peptide::parse("PEP[+notanumber]TIDE").is_err());
    }

    #[test]
    fn mods_sorted_canonical_for_eq_hash() {
        let a = Peptide::new(
            "ACDEMK".into(),
            vec![
                (Site::Residue(4), ModSpec::Named("Oxidation@M".into())),
                (Site::NTerm, ModSpec::Named("TMT6plex".into())),
            ],
        );
        let b = Peptide::new(
            "ACDEMK".into(),
            vec![
                (Site::NTerm, ModSpec::Named("TMT6plex".into())),
                (Site::Residue(4), ModSpec::Named("Oxidation@M".into())),
            ],
        );
        assert_eq!(a, b);
    }
}
