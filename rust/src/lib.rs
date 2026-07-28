//! pyo3 bindings: thin shell exposing pepdistill-core to Python as `pepdistill_rs`.

use ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyDict, PyList};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pepdistill_core::chem;
use pepdistill_core::peptide::Peptide as CorePeptide;
use pepdistill_core::{bucket, tokenize};

fn to_pyerr(e: anyhow::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

#[pyclass(frozen, name = "Peptide")]
pub struct Peptide {
    inner: CorePeptide,
}

#[pymethods]
impl Peptide {
    #[new]
    #[pyo3(signature = (sequence, mods=Vec::new()))]
    fn new(sequence: String, mods: Vec<(usize, String)>) -> Self {
        Self { inner: CorePeptide::new(sequence, mods) }
    }
    #[getter]
    fn sequence(&self) -> &str { &self.inner.sequence }
    #[getter]
    fn mods(&self) -> Vec<(usize, String)> { self.inner.mods.clone() }
    #[getter]
    fn length(&self) -> usize { self.inner.length() }
    fn residue_masses(&self) -> PyResult<Vec<f64>> { self.inner.residue_masses().map_err(to_pyerr) }
    fn mono_mass(&self) -> PyResult<f64> { self.inner.mono_mass().map_err(to_pyerr) }
    fn precursor_mz(&self, charge: i64) -> PyResult<f64> { self.inner.precursor_mz(charge).map_err(to_pyerr) }
    fn modified_sequence(&self) -> String { self.inner.modified_sequence() }

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
    fn __reduce__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<(PyObject, (String, Vec<(usize, String)>))> {
        let cls = py.get_type_bound::<Peptide>().into_py(py);
        Ok((cls, (slf.inner.sequence.clone(), slf.inner.mods.clone())))
    }
}

#[pyfunction]
fn fragment_mz(rm: Vec<f64>, ion: &str, ordinal: usize, charge: i64) -> PyResult<f64> {
    chem::fragment_mz(&rm, ion, ordinal, charge).map_err(to_pyerr)
}

#[pyfunction]
fn fragment_mz_matrix(sequence: &str, mods: Vec<(usize, String)>) -> PyResult<Vec<Vec<f64>>> {
    let rm = chem::residue_masses_mod(sequence.as_bytes(), &mods).map_err(to_pyerr)?;
    Ok(chem::fragment_mz_matrix(&rm))
}

#[pyfunction]
fn ms2_target_shape(length: usize) -> (usize, usize) {
    chem::ms2_target_shape(length)
}

#[pyfunction]
fn collate<'py>(
    py: Python<'py>,
    seqs: Vec<String>, charges: Vec<i64>,
    mod_sites: Vec<Vec<usize>>, mod_names: Vec<Vec<String>>, use_termini: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let a = tokenize::collate(&seqs, &charges, &mod_sites, &mod_names, use_termini).map_err(to_pyerr)?;
    let d = PyDict::new_bound(py);
    d.set_item("tokens", a.tokens.into_pyarray_bound(py))?;
    d.set_item("mod_delta", a.mod_delta.into_pyarray_bound(py))?;
    d.set_item("charge", a.charge.into_pyarray_bound(py))?;
    d.set_item("lengths", a.lengths.into_pyarray_bound(py))?;
    d.set_item("pad_mask", a.pad_mask.into_pyarray_bound(py))?;
    d.set_item("frag_mask", a.frag_mask.into_pyarray_bound(py))?;
    Ok(d)
}

#[pyfunction]
fn bucket_arrays<'py>(
    py: Python<'py>,
    seqs: Vec<String>, charges: Vec<i64>,
    mod_sites: Vec<Vec<usize>>, mod_names: Vec<Vec<String>>, length: usize, use_termini: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let a = bucket::bucket_arrays(&seqs, &charges, &mod_sites, &mod_names, length, use_termini).map_err(to_pyerr)?;
    let d = PyDict::new_bound(py);
    d.set_item("tokens", a.tokens.into_pyarray_bound(py))?;
    d.set_item("mod_delta", a.mod_delta.into_pyarray_bound(py))?;
    d.set_item("charge", a.charge.into_pyarray_bound(py))?;
    d.set_item("residue_mass", a.residue_mass.into_pyarray_bound(py))?;
    Ok(d)
}

#[pyfunction]
fn bucket_fragment_mz<'py>(
    py: Python<'py>,
    residue_mass: numpy::PyReadonlyArray2<'py, f64>,
    charge: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, numpy::PyArray3<f64>>, Bound<'py, numpy::PyArray1<f64>>)> {
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
    m.add_function(wrap_pyfunction!(collate, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_fragment_mz, m)?)?;

    m.add("PROTON", chem::PROTON)?;
    m.add("H2O", chem::H2O)?;
    m.add("AA_OFFSET", tokenize::AA_OFFSET)?;
    m.add("PAD_IDX", tokenize::PAD_IDX)?;
    m.add("NTERM_IDX", tokenize::NTERM_IDX)?;
    m.add("CTERM_IDX", tokenize::CTERM_IDX)?;
    m.add("N_TOKENS", tokenize::N_TOKENS)?;
    m.add("MOD_SCALE", tokenize::MOD_SCALE as f64)?;

    let py = m.py();
    let ion = PyList::empty_bound(py);
    for &(is_b, z) in chem::ION_TYPES.iter() {
        ion.append((if is_b { "b" } else { "y" }, z as i64))?;
    }
    m.add("ION_TYPES", ion)?;

    let mods = PyDict::new_bound(py);
    for name in ["Carbamidomethyl@C", "Oxidation@M", "Phospho", "TMT6plex"] {
        mods.set_item(name, chem::mod_delta(name).unwrap())?;
    }
    m.add("MOD_DELTA", mods)?;

    let residues = PyDict::new_bound(py);
    for aa in b'A'..=b'Z' {
        if let Some(mass) = chem::residue_mass(aa) {
            residues.set_item((aa as char).to_string(), mass)?;
        }
    }
    m.add("RESIDUE_MASS", residues)?;
    Ok(())
}
