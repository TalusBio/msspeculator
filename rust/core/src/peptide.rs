//! Modified peptide — data plus mass/rendering behavior.

use crate::{chem, composition::AtomicComposition};

/// Where a modification sits. Ordering is deliberate: `derive(Ord)` uses declaration order, so
/// sorting a mods vector already produces N-term → residues → C-term without a custom compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Site {
    NTerm,
    Residue(usize),
    CTerm,
}

/// Modification identity plus its resolved encoder route. Every modification retains an
/// unambiguous UniMod/formula identity, except `MassOnly`, which carries a bare delta rather than
/// inventing a composition for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EncodingRoute {
    Composition,
    Mass,
}

#[derive(Debug, Clone)]
pub enum ModSpec {
    /// Controlled-vocabulary identity, retaining its accession through output.
    Unimod {
        accession: u32,
        route: EncodingRoute,
    },
    /// Public elemental formula, retaining both its spelling and exact isotope composition.
    Formula {
        formula: String,
        composition: AtomicComposition,
        route: EncodingRoute,
    },
    MassOnly(f64),
}

// PartialEq is hand-written (not derived) so it agrees bit-for-bit with Hash and Ord below,
// both of which compare `f64` via `to_bits()`. Derived PartialEq would use IEEE equality
// (-0.0 == 0.0, NaN != NaN), which disagrees with to_bits()-based Hash/Ord and would corrupt
// HashMap/HashSet lookups keyed on Peptide (Python uses Peptide as a dict key and dedup key).
impl PartialEq for ModSpec {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                ModSpec::Unimod {
                    accession: aa,
                    route: ar,
                },
                ModSpec::Unimod {
                    accession: ba,
                    route: br,
                },
            ) => aa == ba && ar == br,
            (
                ModSpec::Formula {
                    formula: af,
                    composition: ac,
                    route: ar,
                },
                ModSpec::Formula {
                    formula: bf,
                    composition: bc,
                    route: br,
                },
            ) => af == bf && ac == bc && ar == br,
            (ModSpec::MassOnly(a), ModSpec::MassOnly(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }
}

impl Eq for ModSpec {}

impl std::hash::Hash for ModSpec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ModSpec::Unimod { accession, route } => (0u8, accession, route).hash(state),
            ModSpec::Formula {
                formula,
                composition,
                route,
            } => (1u8, formula, composition, route).hash(state),
            ModSpec::MassOnly(m) => (2u8, m.to_bits()).hash(state),
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
            (
                ModSpec::Unimod {
                    accession: aa,
                    route: ar,
                },
                ModSpec::Unimod {
                    accession: ba,
                    route: br,
                },
            ) => (aa, ar).cmp(&(ba, br)),
            (
                ModSpec::Formula {
                    formula: af,
                    composition: ac,
                    route: ar,
                },
                ModSpec::Formula {
                    formula: bf,
                    composition: bc,
                    route: br,
                },
            ) => (af, &ac.counts, ar).cmp(&(bf, &bc.counts, br)),
            (ModSpec::MassOnly(a), ModSpec::MassOnly(b)) => a.to_bits().cmp(&b.to_bits()),
            (a, b) => a.variant_order().cmp(&b.variant_order()),
        }
    }
}

impl ModSpec {
    fn variant_order(&self) -> u8 {
        match self {
            Self::Unimod { .. } => 0,
            Self::Formula { .. } => 1,
            Self::MassOnly(_) => 2,
        }
    }

    /// Monoisotopic delta in Da. Errors on an unresolvable name rather than contributing zero.
    pub fn delta_mass(&self) -> anyhow::Result<f64> {
        match self {
            ModSpec::MassOnly(m) => Ok(*m),
            ModSpec::Unimod { accession, .. } => {
                let entry = crate::unimod::by_accession(*accession)
                    .ok_or_else(|| anyhow::anyhow!("unknown UNIMOD accession {accession}"))?;
                entry.comp.mono_mass(crate::unimod::nuclide_masses())
            }
            ModSpec::Formula { composition, .. } => {
                composition.mono_mass(crate::unimod::nuclide_masses())
            }
        }
    }

    /// Composition input for the model, or `None` when this modification explicitly routes
    /// through the scalar mass encoder.
    pub fn element_comp(&self) -> anyhow::Result<Option<[i8; crate::composition::N_ELEMENTS]>> {
        match self {
            ModSpec::Unimod {
                accession,
                route: EncodingRoute::Composition,
            } => Ok(Some(
                crate::unimod::by_accession(*accession)
                    .ok_or_else(|| anyhow::anyhow!("unknown UNIMOD accession {accession}"))?
                    .comp
                    .element_comp()?,
            )),
            ModSpec::Formula {
                composition,
                route: EncodingRoute::Composition,
                ..
            } => Ok(Some(composition.element_comp()?)),
            ModSpec::Unimod {
                route: EncodingRoute::Mass,
                ..
            }
            | ModSpec::Formula {
                route: EncodingRoute::Mass,
                ..
            }
            | ModSpec::MassOnly(_) => Ok(None),
        }
    }

    /// Rendered form inside a modified sequence: always a ProForma descriptor, so every
    /// modified sequence we emit parses back through [`crate::proforma::parse_peptide`].
    ///
    /// A consumer that genuinely requires another notation converts at its own boundary rather
    /// than changing this — peptdeep needs alphabase `Name@Site` names, and that mapping lives in
    /// the teacher wrapper.
    pub fn render(&self) -> String {
        match self {
            ModSpec::Unimod { accession, .. } => format!("UNIMOD:{accession}"),
            ModSpec::Formula { formula, .. } => format!("Formula:{formula}"),
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

    /// Reject a site that mixes composition-routed and mass-routed modifications.
    ///
    /// Several composition-routed mods on one site are fine (their compositions accumulate),
    /// as are several mass-routed deltas (their masses sum). Mixing the routes is not: they go to
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
                let mixed = spec.element_comp()?.is_some() != other.element_comp()?.is_some();
                if mixed {
                    anyhow::bail!(
                        "site {site:?} of {} carries both a composition-routed modification and \
                         a mass-routed delta ([{}] and [{}]); a single site has one encoder \
                         route and therefore cannot carry one of each",
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

    /// Public rendering is ProForma-shaped and makes terminal placement unambiguous.
    pub fn modified_sequence(&self) -> String {
        let mut out = String::new();
        for (site, spec) in &self.mods {
            if *site == Site::NTerm {
                out.push('[');
                out.push_str(&spec.render());
                out.push(']');
                out.push('-');
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
                out.push('-');
                out.push('[');
                out.push_str(&spec.render());
                out.push(']');
            }
        }
        out
    }

    /// Parse the strict public modification language. Legacy named modifications remain
    /// constructible through `Peptide::new` for prepared training data, but are not accepted
    /// as user input.
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        crate::proforma::parse_peptide(s)
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
    fn modified_sequence_round_trips_through_our_own_grammar() {
        // The alias names are the dangerous case: `Carbamidomethyl@C` inside ProForma brackets
        // is not a descriptor our grammar accepts, and the PROSPECT reader silently reinterpreted
        // the name's capitals as residues. Emission must therefore be accession-based, and the
        // proof is that parsing our own output reconstructs the same peptide.
        for mods in [
            vec![(Site::Residue(1), crate::proforma::unimod_spec(4).unwrap())],
            vec![(Site::Residue(4), crate::proforma::unimod_spec(35).unwrap())],
            vec![(Site::NTerm, crate::proforma::unimod_spec(737).unwrap())],
            vec![
                (Site::NTerm, crate::proforma::unimod_spec(737).unwrap()),
                (Site::Residue(2), crate::proforma::unimod_spec(21).unwrap()),
            ],
        ] {
            let peptide = Peptide::new("ACDEMK".into(), mods);
            let rendered = peptide.modified_sequence();
            assert!(
                !rendered.contains('@'),
                "emitted {rendered:?} still carries an alphabase-style site suffix"
            );
            let reparsed = crate::proforma::parse_peptide(&rendered)
                .unwrap_or_else(|err| panic!("cannot reparse our own output {rendered:?}: {err}"));
            assert_eq!(reparsed.sequence, peptide.sequence);
            assert_eq!(reparsed.mono_mass().unwrap(), peptide.mono_mass().unwrap());
            assert_eq!(reparsed.modified_sequence(), rendered);
        }
    }

    #[test]
    fn modified_sequence_renders_termini_and_mass_only() {
        let p = Peptide::new(
            "ETTLHLVLR".into(),
            vec![
                (Site::Residue(1), crate::proforma::unimod_spec(21).unwrap()),
                (Site::NTerm, crate::proforma::unimod_spec(737).unwrap()),
            ],
        );
        // Internal names are read, never emitted: output carries accessions so it round-trips.
        assert_eq!(p.modified_sequence(), "[UNIMOD:737]-ET[UNIMOD:21]TLHLVLR");

        let q = Peptide::new(
            "PEPTIDE".into(),
            vec![(Site::Residue(0), ModSpec::MassOnly(79.96633))],
        );
        assert_eq!(q.modified_sequence(), "P[+79.96633]EPTIDE");

        let r = Peptide::new(
            "PEK".into(),
            vec![(Site::CTerm, crate::proforma::unimod_spec(21).unwrap())],
        );
        assert_eq!(r.modified_sequence(), "PEK-[UNIMOD:21]");
    }

    #[test]
    fn terminal_and_side_chain_mods_coexist_on_residue_zero() {
        let p = Peptide::new(
            "KPEPTIDE".into(),
            vec![
                (Site::NTerm, crate::proforma::unimod_spec(737).unwrap()),
                (Site::Residue(0), crate::proforma::unimod_spec(737).unwrap()),
            ],
        );
        assert_eq!(p.mods.len(), 2, "two distinct sites, not one merged mod");
        let base = Peptide::new("KPEPTIDE".into(), vec![]).mono_mass().unwrap();
        let d = crate::proforma::unimod_spec(737)
            .unwrap()
            .delta_mass()
            .unwrap();
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
    fn unknown_accession_errors() {
        // A spec can only be built from a real accession, but one constructed directly must still
        // fail loudly rather than contribute a zero delta to the precursor mass.
        let p = Peptide::new(
            "PEK".into(),
            vec![(
                Site::Residue(0),
                ModSpec::Unimod {
                    accession: 999_999,
                    route: EncodingRoute::Composition,
                },
            )],
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
            "AC[UNIMOD:4]DEM[UNIMOD:35]K",
            "[UNIMOD:737]-ET[UNIMOD:21]TLHLVLR",
            "PEK[UNIMOD:21]",
            "PEK-[UNIMOD:2]",
            "P[+79.96633]EPTIDE",
            "P[Formula:[13C2][12C-2]H2N]EPTIDE",
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
        let p = Peptide::parse("PEK-[UNIMOD:21]").unwrap();
        assert_eq!(
            p.mods,
            vec![(
                Site::CTerm,
                ModSpec::Unimod {
                    accession: 21,
                    route: EncodingRoute::Composition,
                },
            ),]
        );
        let q = Peptide::parse("[UNIMOD:737]-PEPC[UNIMOD:4]IDER").unwrap();
        assert_eq!(
            q.mods,
            vec![
                (
                    Site::NTerm,
                    ModSpec::Unimod {
                        accession: 737,
                        route: EncodingRoute::Composition,
                    },
                ),
                (
                    Site::Residue(3),
                    ModSpec::Unimod {
                        accession: 4,
                        route: EncodingRoute::Composition,
                    },
                ),
            ]
        );
    }

    #[test]
    fn parse_keeps_co_sited_mods_on_one_site() {
        // Two brackets after one residue are two mods on that residue, not on consecutive
        // ones. The runtime accumulates their compositions into a single encoder call, so the
        // site must be shared.
        let p = Peptide::parse("PEPC[UNIMOD:35][UNIMOD:21]IDER").unwrap();
        assert_eq!(
            p.mods,
            vec![
                (
                    Site::Residue(3),
                    ModSpec::Unimod {
                        accession: 21,
                        route: EncodingRoute::Composition,
                    },
                ),
                (
                    Site::Residue(3),
                    ModSpec::Unimod {
                        accession: 35,
                        route: EncodingRoute::Composition,
                    },
                ),
            ]
        );
        assert_eq!(p.modified_sequence(), "PEPC[UNIMOD:21][UNIMOD:35]IDER");
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
                (Site::Residue(4), crate::proforma::unimod_spec(35).unwrap()),
                (Site::NTerm, crate::proforma::unimod_spec(737).unwrap()),
            ],
        );
        let b = Peptide::new(
            "ACDEMK".into(),
            vec![
                (Site::NTerm, crate::proforma::unimod_spec(737).unwrap()),
                (Site::Residue(4), crate::proforma::unimod_spec(35).unwrap()),
            ],
        );
        assert_eq!(a, b);
    }
}
