//! Deliberately limited, ProForma-compatible modification grammar.
//!
//! Public input accepts only UNIMOD accessions, signed Dalton deltas, and `Formula:` elemental
//! formulas. Historical names remain an internal serialization detail; accepting them here
//! would make the public syntax depend on a mutable title/alias table.

use std::collections::BTreeSet;

use anyhow::{bail, Context, Result};
use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;

use crate::composition::AtomicComposition;
use crate::peptide::{EncodingRoute, ModSpec, Peptide, Site};

#[derive(Parser)]
#[grammar = "proforma.pest"]
struct ProFormaParser;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModificationTarget {
    Residues(BTreeSet<char>),
    PeptideNTerm,
    PeptideCTerm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModificationRule {
    pub target: ModificationTarget,
    pub spec: ModSpec,
}

fn positive_u32(pair: Pair<'_, Rule>, what: &str) -> Result<u32> {
    let value: u32 = pair
        .as_str()
        .parse()
        .with_context(|| format!("invalid {what} {:?}", pair.as_str()))?;
    if value == 0 {
        bail!("{what} must be positive, got 0");
    }
    Ok(value)
}

fn cardinality(pair: Option<Pair<'_, Rule>>) -> Result<i32> {
    let Some(pair) = pair else { return Ok(1) };
    let value: i32 = pair
        .as_str()
        .parse()
        .with_context(|| format!("invalid formula cardinality {:?}", pair.as_str()))?;
    if value == 0 {
        bail!("formula cardinality zero is not supported");
    }
    Ok(value)
}

fn parse_formula(pair: Pair<'_, Rule>) -> Result<(String, AtomicComposition)> {
    let text = pair
        .as_str()
        .strip_prefix("Formula:")
        .expect("formula grammar lost prefix")
        .to_string();
    let mut counts = Vec::new();
    for atom in pair.into_inner() {
        debug_assert_eq!(atom.as_rule(), Rule::formula_atom);
        let atom = atom.into_inner().next().expect("formula atom is empty");
        match atom.as_rule() {
            Rule::natural_atom => {
                let mut fields = atom.into_inner();
                let element = fields.next().expect("natural atom has no element");
                let count = cardinality(fields.next())?;
                counts.push((element.as_str().to_string(), count));
            }
            Rule::isotope_atom => {
                let mut fields = atom.into_inner();
                let isotope = positive_u32(
                    fields.next().expect("isotope atom has no isotope number"),
                    "isotope number",
                )?;
                let element = fields.next().expect("isotope atom has no element");
                let count = cardinality(fields.next())?;
                counts.push((format!("{isotope}{}", element.as_str()), count));
            }
            _ => unreachable!("grammar returned non-atom"),
        }
    }
    Ok((text, AtomicComposition { counts }))
}

fn parse_modification(pair: Pair<'_, Rule>) -> Result<ModSpec> {
    debug_assert_eq!(pair.as_rule(), Rule::modification);
    let descriptor = pair.into_inner().next().expect("modification is empty");
    match descriptor.as_rule() {
        Rule::unimod => {
            let accession: u32 = descriptor
                .as_str()
                .strip_prefix("UNIMOD:")
                .expect("UNIMOD grammar lost prefix")
                .parse()
                .context("invalid UNIMOD accession")?;
            let entry = crate::unimod::by_accession(accession)
                .with_context(|| format!("unknown UNIMOD accession {accession}"))?;
            let route = match entry.comp.element_comp() {
                Ok(_) => EncodingRoute::Composition,
                Err(reason) => {
                    eprintln!(
                        "warning: UNIMOD:{accession} ({}) cannot use the composition encoder: \
                         {reason}; using exact {:+.9} Da through the mass encoder",
                        entry.title, entry.mono_mass
                    );
                    EncodingRoute::Mass
                }
            };
            Ok(ModSpec::Unimod { accession, route })
        }
        Rule::mass_delta => {
            let mass: f64 = descriptor.as_str().parse().context("invalid mass delta")?;
            if !mass.is_finite() || mass == 0.0 {
                bail!("mass delta must be finite and nonzero");
            }
            Ok(ModSpec::MassOnly(mass))
        }
        Rule::formula => {
            let (formula, composition) = parse_formula(descriptor)?;
            let mass = composition
                .mono_mass(crate::unimod::nuclide_masses())
                .with_context(|| format!("cannot calculate Formula:{formula} mass"))?;
            let route = match composition.element_comp() {
                Ok(_) => EncodingRoute::Composition,
                Err(reason) => {
                    eprintln!(
                        "warning: Formula:{formula} cannot use the composition encoder: \
                         {reason}; using exact {mass:+.9} Da through the mass encoder"
                    );
                    EncodingRoute::Mass
                }
            };
            Ok(ModSpec::Formula {
                formula,
                composition,
                route,
            })
        }
        _ => unreachable!("grammar returned unsupported descriptor"),
    }
}

pub fn parse_peptide(input: &str) -> Result<Peptide> {
    let mut parsed = ProFormaParser::parse(Rule::peptide, input)
        .map_err(|err| anyhow::anyhow!("invalid modified peptide {input:?}: {err}"))?;
    let peptide = parsed.next().expect("successful peptide parse is empty");
    let mut sequence = String::new();
    let mut mods = Vec::new();
    for part in peptide.into_inner() {
        match part.as_rule() {
            Rule::n_term => {
                let modification = part.into_inner().next().expect("empty N-terminal rule");
                mods.push((Site::NTerm, parse_modification(modification)?));
            }
            Rule::c_term => {
                let modification = part.into_inner().next().expect("empty C-terminal rule");
                mods.push((Site::CTerm, parse_modification(modification)?));
            }
            Rule::residue => {
                let mut fields = part.into_inner();
                let aa = fields.next().expect("residue has no amino acid").as_str();
                let index = sequence.len();
                sequence.push_str(aa);
                for modification in fields {
                    mods.push((Site::Residue(index), parse_modification(modification)?));
                }
            }
            Rule::EOI => {}
            _ => unreachable!("grammar returned unsupported peptide part"),
        }
    }
    Ok(Peptide::new(sequence, mods))
}

pub fn parse_modification_rule(input: &str) -> Result<ModificationRule> {
    let mut parsed = ProFormaParser::parse(Rule::modification_rule, input)
        .map_err(|err| anyhow::anyhow!("invalid modification rule {input:?}: {err}"))?;
    let outer = parsed
        .next()
        .expect("successful modification rule parse is empty");
    let rule = outer
        .into_inner()
        .next()
        .expect("modification rule is empty");
    let (target, modification) = match rule.as_rule() {
        Rule::n_term_rule => (
            ModificationTarget::PeptideNTerm,
            rule.into_inner().next().expect("empty N-terminal rule"),
        ),
        Rule::c_term_rule => (
            ModificationTarget::PeptideCTerm,
            rule.into_inner().next().expect("empty C-terminal rule"),
        ),
        Rule::residue_rule => {
            let mut fields = rule.into_inner();
            let residues = fields
                .next()
                .expect("residue rule has no targets")
                .as_str()
                .chars()
                .collect();
            (
                ModificationTarget::Residues(residues),
                fields.next().expect("residue rule has no modification"),
            )
        }
        _ => unreachable!("grammar returned unsupported modification rule"),
    };
    Ok(ModificationRule {
        target,
        spec: parse_modification(modification)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_unimod_mass_formula_and_unambiguous_termini() {
        let peptide =
            parse_peptide("[UNIMOD:737]-AC[UNIMOD:2057]DM[+15.994915]K-[Formula:H-1N-1O]").unwrap();
        assert_eq!(peptide.sequence, "ACDMK");
        assert_eq!(peptide.mods[0].0, Site::NTerm);
        assert_eq!(peptide.mods[1].0, Site::Residue(1));
        assert_eq!(peptide.mods[2].0, Site::Residue(3));
        assert_eq!(peptide.mods[3].0, Site::CTerm);
    }

    #[test]
    fn isotope_formula_folds_elements_but_keeps_exact_mass() {
        let rule = parse_modification_rule("C[Formula:[13C2][12C-2]H2N]").unwrap();
        let ModSpec::Formula {
            composition, route, ..
        } = rule.spec
        else {
            panic!("expected formula")
        };
        assert_eq!(route, EncodingRoute::Composition);
        assert_eq!(composition.element_comp().unwrap(), [0, 2, 1, 0, 0, 0]);
        let mass = composition
            .mono_mass(crate::unimod::nuclide_masses())
            .unwrap();
        assert!(mass > 18.0);
    }

    #[test]
    fn cyspat_resolves_by_accession_to_composition_and_exact_mass() {
        let rule = parse_modification_rule("C[UNIMOD:2057]").unwrap();
        let ModSpec::Unimod { accession, route } = rule.spec else {
            panic!("expected UniMod modification")
        };
        assert_eq!(accession, 2057);
        assert_eq!(route, EncodingRoute::Composition);
        let entry = crate::unimod::by_accession(accession).unwrap();
        assert_eq!(entry.comp.element_comp().unwrap(), [8, 16, 1, 4, 0, 1]);
        assert!((entry.mono_mass - 221.081_695).abs() < 1e-6);
    }

    #[test]
    fn rejects_names_zero_cardinality_and_missing_terminal_hyphens() {
        assert!(parse_modification_rule("M[Oxidation]").is_err());
        assert!(parse_modification_rule("M[Formula:C0H2]").is_err());
        assert!(parse_peptide("[UNIMOD:737]PEPTIDE").is_err());
    }
}
