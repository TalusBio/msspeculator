//! pyo3 bindings: thin shell exposing pepdistill-core to Python as `pepdistill_rs`.

// The `#[pyfunction]`/`#[pymethods]` macros generate wrapper code that converts the user's
// `PyResult<T>` into itself via `Into`; clippy flags that macro-generated no-op conversion at
// our function signatures even though there's nothing for us to change. Silencing it here
// (rather than per-item) keeps the real lint active for any future non-pyo3 code in this crate.
#![allow(clippy::useless_conversion)]

use ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyDict, PyList};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pepdistill_core::chem;
use pepdistill_core::peptide::{ModSpec, Peptide as CorePeptide, Site};
use pepdistill_core::{bucket, proforma, tokenize, unimod};

fn to_pyerr(e: anyhow::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Python-side `site` is `int | "n" | "c"`.
fn parse_site(obj: &Bound<'_, PyAny>) -> PyResult<Site> {
    if let Ok(s) = obj.extract::<&str>() {
        return match s {
            "n" => Ok(Site::NTerm),
            "c" => Ok(Site::CTerm),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "site must be an int, 'n', or 'c'; got {s:?}"
            ))),
        };
    }
    Ok(Site::Residue(obj.extract::<usize>()?))
}

/// Python-side `spec` is `str | float`. A `"UNIMOD:<accession>"` string is the canonical identity
/// and becomes a real accession spec; anything else remains a historical name.
fn parse_spec(obj: &Bound<'_, PyAny>) -> PyResult<ModSpec> {
    if let Ok(name) = obj.extract::<String>() {
        if let Some(digits) = name.strip_prefix("UNIMOD:") {
            let accession = digits.parse::<u32>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid UNIMOD accession {name:?}"
                ))
            })?;
            return proforma::unimod_spec(accession).map_err(to_pyerr);
        }
        return Ok(ModSpec::Named(name));
    }
    Ok(ModSpec::MassOnly(obj.extract::<f64>()?))
}

fn parse_mods(mods: &[(Bound<'_, PyAny>, Bound<'_, PyAny>)]) -> PyResult<Vec<(Site, ModSpec)>> {
    mods.iter()
        .map(|(s, m)| Ok((parse_site(s)?, parse_spec(m)?)))
        .collect()
}

/// Inverse of `parse_site`: `Site::NTerm -> "n"`, `CTerm -> "c"`, `Residue(i) -> i`.
fn site_to_py(site: &Site, py: Python<'_>) -> PyObject {
    match site {
        Site::NTerm => "n".into_py(py),
        Site::CTerm => "c".into_py(py),
        Site::Residue(i) => i.into_py(py),
    }
}

/// Inverse of `parse_spec`: `Named(n) -> n`, `MassOnly(m) -> m`.
fn spec_to_py(spec: &ModSpec, py: Python<'_>) -> PyObject {
    match spec {
        ModSpec::Named(n) => n.clone().into_py(py),
        ModSpec::Unimod { .. } | ModSpec::Formula { .. } => spec.render().into_py(py),
        ModSpec::MassOnly(m) => m.into_py(py),
    }
}

fn mods_to_py(mods: &[(Site, ModSpec)], py: Python<'_>) -> Vec<(PyObject, PyObject)> {
    mods.iter()
        .map(|(s, m)| (site_to_py(s, py), spec_to_py(m, py)))
        .collect()
}

/// `__reduce__` state: (class, (sequence, mods)) — kept as a named alias to satisfy
/// clippy's `type_complexity` lint.
type ReduceState = (PyObject, (String, Vec<(PyObject, PyObject)>));

#[pyclass(frozen, name = "Peptide")]
pub struct Peptide {
    inner: CorePeptide,
}

#[pymethods]
impl Peptide {
    #[new]
    #[pyo3(signature = (sequence, mods=Vec::new()))]
    fn new(sequence: String, mods: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)>) -> PyResult<Self> {
        Ok(Self {
            inner: CorePeptide::new(sequence, parse_mods(&mods)?),
        })
    }

    /// Parse a canonical ProForma modified sequence. Refuses anything else, including the
    /// PROSPECT spelling that omits the N-terminal separator -- use `from_prospect` for that,
    /// once, at ingest.
    #[staticmethod]
    fn from_string(modified_sequence: &str) -> PyResult<Self> {
        Ok(Self {
            inner: proforma::parse_peptide(modified_sequence).map_err(to_pyerr)?,
        })
    }

    /// Parse a PROSPECT modified sequence, tolerating its one deviation from ProForma. This is the
    /// only entry point that accepts a degenerate spelling; it exists to be called once, where
    /// source metadata is read, so that everything downstream holds the canonical form.
    #[staticmethod]
    fn from_prospect(modified_sequence: &str) -> PyResult<Self> {
        Ok(Self {
            inner: proforma::parse_prospect_peptide(modified_sequence).map_err(to_pyerr)?,
        })
    }

    #[getter]
    fn sequence(&self) -> &str {
        &self.inner.sequence
    }
    #[getter]
    fn mods(&self, py: Python<'_>) -> Vec<(PyObject, PyObject)> {
        mods_to_py(&self.inner.mods, py)
    }
    #[getter]
    fn length(&self) -> usize {
        self.inner.length()
    }
    fn residue_masses(&self) -> PyResult<Vec<f64>> {
        self.inner.residue_masses().map_err(to_pyerr)
    }
    fn mono_mass(&self) -> PyResult<f64> {
        self.inner.mono_mass().map_err(to_pyerr)
    }
    fn precursor_mz(&self, charge: i64) -> PyResult<f64> {
        self.inner.precursor_mz(charge).map_err(to_pyerr)
    }
    fn modified_sequence(&self) -> String {
        self.inner.modified_sequence()
    }

    fn __hash__(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.inner.hash(&mut h);
        h.finish()
    }
    fn __richcmp__(&self, other: &Peptide, op: CompareOp, py: Python<'_>) -> PyObject {
        match op {
            CompareOp::Eq => (self.inner == other.inner).into_py(py),
            CompareOp::Ne => (self.inner != other.inner).into_py(py),
            _ => py.NotImplemented(),
        }
    }
    // Defensive pickle support (not required by current code paths).
    fn __reduce__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<ReduceState> {
        let cls = py.get_type_bound::<Peptide>().into_py(py);
        let mods = mods_to_py(&slf.inner.mods, py);
        Ok((cls, (slf.inner.sequence.clone(), mods)))
    }
}

/// Parse a modification rule into `(targets, spec)`.
///
/// `"STY[UNIMOD:21]"` -> `(["S", "T", "Y"], "UNIMOD:21")`: one rule naming every residue it
/// applies to. `"[UNIMOD:737]-"` and `"-[UNIMOD:21]"` target the termini, reported as `"n"` and
/// `"c"` so they match the site vocabulary `Peptide` already uses.
///
/// Same grammar as a peptide's modifications, so a rule cannot express a modification a peptide
/// could not carry, and an unknown accession is rejected here rather than at first use.
#[pyfunction]
fn parse_modification_rule(rule: &str) -> PyResult<(Vec<String>, String)> {
    let parsed = proforma::parse_modification_rule(rule).map_err(to_pyerr)?;
    let targets = match parsed.target {
        proforma::ModificationTarget::Residues(residues) => {
            residues.into_iter().map(|r| r.to_string()).collect()
        }
        proforma::ModificationTarget::PeptideNTerm => vec!["n".to_string()],
        proforma::ModificationTarget::PeptideCTerm => vec!["c".to_string()],
    };
    Ok((targets, parsed.spec.render()))
}

#[pyfunction]
fn fragment_mz(rm: Vec<f64>, ion: &str, ordinal: usize, charge: i64) -> PyResult<f64> {
    chem::fragment_mz(&rm, ion, ordinal, charge).map_err(to_pyerr)
}

#[pyfunction]
fn fragment_mz_matrix(
    sequence: &str,
    mods: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)>,
) -> PyResult<Vec<Vec<f64>>> {
    let pep = CorePeptide::new(sequence.to_string(), parse_mods(&mods)?);
    let rm = pep.residue_masses().map_err(to_pyerr)?;
    Ok(chem::fragment_mz_matrix(&rm))
}

#[pyfunction]
fn ms2_target_shape(length: usize) -> (usize, usize) {
    chem::ms2_target_shape(length)
}

/// Canonical name for a UNIMOD accession: our alias if one exists, else the UNIMOD title.
/// `None` if the accession is not in the vendored table.
#[pyfunction]
fn unimod_name(accession: u32) -> Option<String> {
    let e = unimod::by_accession(accession)?;
    let alias = unimod::ALIASES
        .iter()
        .find(|(_, a)| *a == accession)
        .map(|(n, _)| n.to_string());
    Some(alias.unwrap_or_else(|| e.title.clone()))
}

/// UNIMOD accession for one of our aliases or a canonical UNIMOD title.
#[pyfunction]
fn unimod_accession(name: &str) -> Option<u32> {
    unimod::by_name(name).map(|entry| entry.accession)
}

/// 6-element composition delta for a named modification, in `composition::ELEMENTS` order.
/// Raises `ValueError` if the name is unknown or needs an element outside that basis.
#[pyfunction]
fn mod_element_comp(name: &str) -> PyResult<[i8; pepdistill_core::composition::N_ELEMENTS]> {
    chem::mod_element_comp(name).map_err(to_pyerr)
}

#[pyfunction]
fn collate<'py>(
    py: Python<'py>,
    peptides: Vec<PyRef<'py, Peptide>>,
    charges: Vec<i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let core_peptides: Vec<CorePeptide> = peptides.iter().map(|p| p.inner.clone()).collect();
    let a = tokenize::collate(&core_peptides, &charges).map_err(to_pyerr)?;
    let d = PyDict::new_bound(py);
    d.set_item("tokens", a.tokens.into_pyarray_bound(py))?;
    d.set_item("mod_comp", a.mod_comp.into_pyarray_bound(py))?;
    d.set_item("mod_mass", a.mod_mass.into_pyarray_bound(py))?;
    d.set_item("mod_present", a.mod_present.into_pyarray_bound(py))?;
    d.set_item("mod_named", a.mod_named.into_pyarray_bound(py))?;
    d.set_item("charge", a.charge.into_pyarray_bound(py))?;
    d.set_item("lengths", a.lengths.into_pyarray_bound(py))?;
    d.set_item("pad_mask", a.pad_mask.into_pyarray_bound(py))?;
    d.set_item("frag_mask", a.frag_mask.into_pyarray_bound(py))?;
    Ok(d)
}

/// Collate a prepared shard's canonical ProForma column into NumPy-backed model arrays.
///
/// One column, parsed by the strict grammar. The previous signature took a bare sequence plus a
/// `site:spec;...` string, which was a second serialization of the same peptide with its own
/// parser, its own failure modes, and no way to tell a mod on the last residue from one on the
/// C-terminus.
#[pyfunction]
fn collate_prepared<'py>(
    py: Python<'py>,
    proforma: Vec<String>,
    charges: Vec<i64>,
) -> PyResult<Bound<'py, PyDict>> {
    if proforma.len() != charges.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "prepared columns have unequal lengths: proforma={}, charges={}",
            proforma.len(),
            charges.len()
        )));
    }
    let peptides = proforma
        .iter()
        .map(|sequence| proforma::parse_peptide(sequence))
        .collect::<anyhow::Result<Vec<_>>>()
        .map_err(to_pyerr)?;
    let a = tokenize::collate(&peptides, &charges).map_err(to_pyerr)?;
    let d = PyDict::new_bound(py);
    d.set_item("tokens", a.tokens.into_pyarray_bound(py))?;
    d.set_item("mod_comp", a.mod_comp.into_pyarray_bound(py))?;
    d.set_item("mod_mass", a.mod_mass.into_pyarray_bound(py))?;
    d.set_item("mod_present", a.mod_present.into_pyarray_bound(py))?;
    d.set_item("mod_named", a.mod_named.into_pyarray_bound(py))?;
    d.set_item("charge", a.charge.into_pyarray_bound(py))?;
    d.set_item("lengths", a.lengths.into_pyarray_bound(py))?;
    d.set_item("pad_mask", a.pad_mask.into_pyarray_bound(py))?;
    d.set_item("frag_mask", a.frag_mask.into_pyarray_bound(py))?;
    Ok(d)
}

#[pyfunction]
fn bucket_arrays<'py>(
    py: Python<'py>,
    peptides: Vec<PyRef<'py, Peptide>>,
    charges: Vec<i64>,
    length: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let core_peptides: Vec<CorePeptide> = peptides.iter().map(|p| p.inner.clone()).collect();
    let a = bucket::bucket_arrays(&core_peptides, &charges, length).map_err(to_pyerr)?;
    let d = PyDict::new_bound(py);
    d.set_item("tokens", a.tokens.into_pyarray_bound(py))?;
    d.set_item("mod_comp", a.mod_comp.into_pyarray_bound(py))?;
    d.set_item("mod_mass", a.mod_mass.into_pyarray_bound(py))?;
    d.set_item("mod_present", a.mod_present.into_pyarray_bound(py))?;
    d.set_item("mod_named", a.mod_named.into_pyarray_bound(py))?;
    d.set_item("charge", a.charge.into_pyarray_bound(py))?;
    d.set_item("residue_mass", a.residue_mass.into_pyarray_bound(py))?;
    Ok(d)
}

/// (fragment m/z tensor, precursor m/z vector) — named alias to satisfy clippy's
/// `type_complexity` lint.
type FragmentMzResult<'py> = (
    Bound<'py, numpy::PyArray3<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
);

#[pyfunction]
fn bucket_fragment_mz<'py>(
    py: Python<'py>,
    residue_mass: numpy::PyReadonlyArray2<'py, f64>,
    charge: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<FragmentMzResult<'py>> {
    let rm: Array2<f64> = residue_mass.as_array().to_owned();
    let ch: Array1<i64> = charge.as_array().to_owned();
    let (mz, pmz) = bucket::bucket_fragment_mz(&rm, &ch);
    Ok((mz.into_pyarray_bound(py), pmz.into_pyarray_bound(py)))
}

#[pymodule]
fn pepdistill_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Peptide>()?;
    m.add_function(wrap_pyfunction!(fragment_mz, m)?)?;
    m.add_function(wrap_pyfunction!(fragment_mz_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(ms2_target_shape, m)?)?;
    m.add_function(wrap_pyfunction!(unimod_name, m)?)?;
    m.add_function(wrap_pyfunction!(parse_modification_rule, m)?)?;
    m.add_function(wrap_pyfunction!(unimod_accession, m)?)?;
    m.add_function(wrap_pyfunction!(mod_element_comp, m)?)?;
    m.add_function(wrap_pyfunction!(collate, m)?)?;
    m.add_function(wrap_pyfunction!(collate_prepared, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_fragment_mz, m)?)?;

    m.add("PROTON", chem::PROTON)?;
    m.add("H2O", chem::H2O)?;
    m.add("AA_OFFSET", tokenize::AA_OFFSET)?;
    m.add("PAD_IDX", tokenize::PAD_IDX)?;
    m.add("NTERM_IDX", tokenize::NTERM_IDX)?;
    m.add("CTERM_IDX", tokenize::CTERM_IDX)?;
    m.add("N_TOKENS", tokenize::N_TOKENS)?;
    m.add("FRAG_OFFSET", tokenize::FRAG_OFFSET)?;

    let py = m.py();
    let ion = PyList::empty_bound(py);
    for &(is_b, z) in chem::ION_TYPES.iter() {
        ion.append((if is_b { "b" } else { "y" }, z as i64))?;
    }
    m.add("ION_TYPES", ion)?;

    let mods = PyDict::new_bound(py);
    for &(name, _acc) in unimod::ALIASES.iter() {
        let delta = chem::mod_delta(name)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        mods.set_item(name, delta)?;
    }
    m.add("MOD_DELTA", mods)?;

    let residues = PyDict::new_bound(py);
    let residue_compositions = PyDict::new_bound(py);
    for aa in b'A'..=b'Z' {
        if let Some(mass) = chem::residue_mass(aa) {
            let name = (aa as char).to_string();
            residues.set_item(&name, mass)?;
            let comp = chem::residue_element_comp(aa).ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "residue {name} has a mass but no elemental composition"
                ))
            })?;
            residue_compositions.set_item(&name, comp)?;
        }
    }
    m.add("RESIDUE_MASS", residues)?;
    m.add("RESIDUE_COMP", residue_compositions)?;
    Ok(())
}
