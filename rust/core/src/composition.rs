//! Isotope-aware elemental composition.
//!
//! Two views of one stored value. The **atomic composition** keeps isotopes distinct
//! (`13C(4) 15N`) and yields an exact monoisotopic mass. The **element composition** folds each
//! isotope onto its parent element and yields the fixed 6-vector the student consumes. They
//! They disagree for isotope-labeled mods. That is the point, not a defect.
//!
//! Nuclide masses are not defined here; they are vendored from UNIMOD (see `unimod.rs`) so no
//! chemistry constant is ever hand-typed.

use std::collections::HashMap;

/// The frozen model-input basis. Changing this changes the student's input contract and
/// invalidates every checkpoint. It is not a list that grows casually.
pub const ELEMENTS: [&str; 6] = ["C", "H", "N", "O", "S", "P"];
pub const N_ELEMENTS: usize = ELEMENTS.len();

/// Fold an isotope symbol onto its parent element. UNIMOD writes isotopes as
/// `<digits><Element>` without exception, so this is a prefix strip rather than a table.
pub fn parent_element(symbol: &str) -> &str {
    symbol.trim_start_matches(|c: char| c.is_ascii_digit())
}

/// Elemental composition delta, isotopes kept distinct. Counts may be negative (losses).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct AtomicComposition {
    pub counts: Vec<(String, i32)>,
}

impl AtomicComposition {
    /// Parse UNIMOD's composition notation: space-separated `Sym` or `Sym(count)` terms,
    /// e.g. `"H(20) C(8) 13C(4) N 15N O(2)"`. A bare symbol means count 1.
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        let mut counts = Vec::new();
        for term in s.split_whitespace() {
            let (sym, n) = match term.split_once('(') {
                None => (term, 1),
                Some((sym, rest)) => {
                    let digits = rest.strip_suffix(')').ok_or_else(|| {
                        anyhow::anyhow!("malformed composition term {term:?}: missing ')'")
                    })?;
                    let n: i32 = digits.parse().map_err(|_| {
                        anyhow::anyhow!("malformed composition term {term:?}: bad count")
                    })?;
                    (sym, n)
                }
            };
            if sym.is_empty() {
                anyhow::bail!("malformed composition term {term:?}: empty symbol");
            }
            counts.push((sym.to_string(), n));
        }
        Ok(Self { counts })
    }

    /// Exact monoisotopic mass. Errors (rather than skipping) on a nuclide the table lacks.
    pub fn mono_mass(&self, masses: &HashMap<String, f64>) -> anyhow::Result<f64> {
        let mut total = 0.0;
        for (sym, n) in &self.counts {
            let m = match masses.get(sym) {
                Some(m) => m,
                None => {
                    // UNIMOD's element table names the principal isotope by its bare element
                    // (`C`, not `12C`), while chemForma permits the explicit spelling `[12C]`.
                    // Accept that spelling only when its mass number is the rounded principal
                    // monoisotopic mass; an absent non-principal isotope remains a hard error.
                    let parent = parent_element(sym);
                    let isotope = sym
                        .strip_suffix(parent)
                        .filter(|prefix| !prefix.is_empty())
                        .and_then(|prefix| prefix.parse::<u32>().ok());
                    match (isotope, masses.get(parent)) {
                        (Some(a), Some(parent_mass)) if a == parent_mass.round() as u32 => {
                            parent_mass
                        }
                        _ => {
                            return Err(anyhow::anyhow!(
                                "no monoisotopic mass for nuclide {sym:?}"
                            ));
                        }
                    }
                }
            };
            total += m * (*n as f64);
        }
        Ok(total)
    }

    /// Project onto the fixed 6-element basis, folding isotopes onto their parents.
    ///
    /// Errors on any element outside the basis. Zero-filling instead would understate the
    /// modification in a model input. A silent wrong answer is worse than no answer.
    pub fn element_comp(&self) -> anyhow::Result<[i8; N_ELEMENTS]> {
        let mut out = [0i32; N_ELEMENTS];
        for (sym, n) in &self.counts {
            let parent = parent_element(sym);
            let idx = ELEMENTS.iter().position(|e| *e == parent).ok_or_else(|| {
                anyhow::anyhow!(
                    "element {parent:?} (from {sym:?}) is outside the model's \
                     {ELEMENTS:?} basis; this modification cannot be encoded"
                )
            })?;
            out[idx] += n;
        }
        let mut packed = [0i8; N_ELEMENTS];
        for (i, v) in out.iter().enumerate() {
            packed[i] = i8::try_from(*v).map_err(|_| {
                anyhow::anyhow!("element count {v} for {:?} exceeds i8", ELEMENTS[i])
            })?;
        }
        Ok(packed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn masses() -> HashMap<String, f64> {
        // A stand-in for the vendored elements.tsv; the real table is loaded in Task 2.
        [
            ("C", 12.0),
            ("13C", 13.00335483507),
            ("H", 1.00782503207),
            ("N", 14.0030740048),
            ("15N", 15.0001088982),
            ("O", 15.9949146196),
            ("P", 30.973761998),
            ("S", 31.97207100),
            ("Se", 79.9165213),
        ]
        .iter()
        .map(|(s, m)| (s.to_string(), *m))
        .collect()
    }

    #[test]
    fn parses_counts_implicit_and_explicit() {
        let c = AtomicComposition::parse("H(20) C(8) 13C(4) N 15N O(2)").unwrap();
        assert_eq!(c.counts.len(), 6);
        assert_eq!(c.counts[0], ("H".to_string(), 20));
        assert_eq!(c.counts[3], ("N".to_string(), 1)); // bare symbol means count 1
    }

    #[test]
    fn parses_negative_counts() {
        let c = AtomicComposition::parse("H(-2) O(-1)").unwrap();
        assert_eq!(c.counts, vec![("H".to_string(), -2), ("O".to_string(), -1)]);
    }

    #[test]
    fn tmt6plex_mass_is_exact() {
        let c = AtomicComposition::parse("H(20) C(8) 13C(4) N 15N O(2)").unwrap();
        assert!((c.mono_mass(&masses()).unwrap() - 229.162_932_1).abs() < 1e-5);
    }

    #[test]
    fn tmt6plex_element_comp_folds_isotopes() {
        let c = AtomicComposition::parse("H(20) C(8) 13C(4) N 15N O(2)").unwrap();
        // ELEMENTS order is C, H, N, O, S, P.
        assert_eq!(c.element_comp().unwrap(), [12, 20, 2, 2, 0, 0]);
    }

    #[test]
    fn phospho_element_comp_and_mass_agree() {
        let c = AtomicComposition::parse("H O(3) P").unwrap();
        assert_eq!(c.element_comp().unwrap(), [0, 1, 0, 3, 0, 1]);
        assert!((c.mono_mass(&masses()).unwrap() - 79.966_331_2).abs() < 1e-5);
    }

    #[test]
    fn out_of_basis_element_errors_naming_it() {
        let c = AtomicComposition::parse("Se C(2)").unwrap();
        let err = c.element_comp().unwrap_err().to_string();
        assert!(
            err.contains("Se"),
            "error must name the element, got: {err}"
        );
    }

    #[test]
    fn unknown_nuclide_mass_errors_naming_it() {
        let c = AtomicComposition::parse("Xx(2)").unwrap();
        let err = c.mono_mass(&masses()).unwrap_err().to_string();
        assert!(
            err.contains("Xx"),
            "error must name the nuclide, got: {err}"
        );
    }

    #[test]
    fn parent_element_strips_isotope_prefix() {
        assert_eq!(parent_element("13C"), "C");
        assert_eq!(parent_element("15N"), "N");
        assert_eq!(parent_element("2H"), "H");
        assert_eq!(parent_element("C"), "C");
    }
}
