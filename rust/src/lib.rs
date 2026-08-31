//! pyo3 bindings: thin shell exposing msspeculator-core to Python as `msspeculator_rs`.

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

use msspeculator_core::chem;
use msspeculator_core::model::Predictor;
use msspeculator_core::peptide::{ModSpec, Peptide as CorePeptide, Site};
use msspeculator_core::split as speclib_split;
use msspeculator_core::{bucket, irt, proforma, speclib, tokenize, unimod, Artifact};

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

/// Python-side `spec` is `str | float`: one unbracketed ProForma descriptor (`"UNIMOD:35"`,
/// `"Formula:H2O"`), or a bare mass delta. A string that is no descriptor is refused here rather
/// than kept as a name, identity is a controlled vocabulary everywhere past ingest.
fn parse_spec(obj: &Bound<'_, PyAny>) -> PyResult<ModSpec> {
    if let Ok(descriptor) = obj.extract::<String>() {
        return proforma::parse_descriptor(&descriptor).map_err(to_pyerr);
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

/// Inverse of `parse_spec`: a descriptor string back out, or `MassOnly(m) -> m`.
fn spec_to_py(spec: &ModSpec, py: Python<'_>) -> PyObject {
    match spec {
        ModSpec::Unimod { .. } | ModSpec::Formula { .. } => spec.render().into_py(py),
        ModSpec::MassOnly(m) => m.into_py(py),
    }
}

fn mods_to_py(mods: &[(Site, ModSpec)], py: Python<'_>) -> Vec<(PyObject, PyObject)> {
    mods.iter()
        .map(|(s, m)| (site_to_py(s, py), spec_to_py(m, py)))
        .collect()
}

/// `__reduce__` state: (class, (sequence, mods)), kept as a named alias to satisfy
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
    /// PROSPECT spelling that omits the N-terminal separator; use `from_prospect` for that,
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

/// The UNIMOD title for an accession, `"Phospho"` for 21. `None` if the accession is unknown.
///
/// The bare title, with no site suffix: a consumer that needs alphabase's `Name@Site` spelling
/// appends the site it holds, which is the only thing that knows where the modification sits.
#[pyfunction]
fn unimod_title(accession: u32) -> Option<String> {
    unimod::by_accession(accession).map(|entry| entry.title.clone())
}

/// Mass delta in Daltons for one unbracketed ProForma descriptor, e.g. `"UNIMOD:35"`.
#[pyfunction]
fn mod_delta(descriptor: &str) -> PyResult<f64> {
    proforma::parse_descriptor(descriptor)
        .and_then(|spec| spec.delta_mass())
        .map_err(to_pyerr)
}

/// 6-element composition delta for one unbracketed ProForma descriptor, in
/// `composition::ELEMENTS` order. Raises `ValueError` for a descriptor with no composition,
/// a bare mass delta, or one whose elements fall outside that basis.
#[pyfunction]
fn mod_composition(descriptor: &str) -> PyResult<[i8; msspeculator_core::composition::N_ELEMENTS]> {
    let spec = proforma::parse_descriptor(descriptor).map_err(to_pyerr)?;
    spec.element_comp().map_err(to_pyerr)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "modification {descriptor:?} routes through the mass encoder and has no composition"
        ))
    })
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
    d.set_item(
        "mod_has_composition",
        a.mod_has_composition.into_pyarray_bound(py),
    )?;
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
    d.set_item(
        "mod_has_composition",
        a.mod_has_composition.into_pyarray_bound(py),
    )?;
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
    d.set_item(
        "mod_has_composition",
        a.mod_has_composition.into_pyarray_bound(py),
    )?;
    d.set_item("charge", a.charge.into_pyarray_bound(py))?;
    d.set_item("residue_mass", a.residue_mass.into_pyarray_bound(py))?;
    Ok(d)
}

/// (fragment m/z tensor, precursor m/z vector), named alias to satisfy clippy's
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

/// Cell a fragment occupies on the MS2 target grid: `(site, ion_column)`, or `None` if the ion
/// does not exist for a peptide of that length.
///
/// The mapping the preparation ETL fills the grid with. Exposed so a Python reader of a vendored
/// spectrum lands on the same cells the Rust doctor does; recomputing the site from an ordinal by
/// hand is how a panel ends up claiming a b1 ion.
#[pyfunction]
fn fragment_cell(ion: &str, ordinal: usize, charge: u8, length: usize) -> PyResult<(u16, u8)> {
    let kind = ion.chars().next().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("ion type must be 'b' or 'y', got an empty string")
    })?;
    speclib::fragment_cell(kind, ordinal, charge, length).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{ion}{ordinal}^{charge} is not a fragment of a {length}-mer"
        ))
    })
}

/// Least-squares agreement of predicted retention index against observed.
///
/// Exposed so the training panel and `msspeculator-cli doctor` report the same slope for the same
/// weights. A second implementation in numpy would drift, and "is the RT scale right?" is not a
/// question that can have two answers.
#[pyfunction]
fn irt_regression<'py>(
    py: Python<'py>,
    observed: Vec<f64>,
    predicted: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    // Checked here rather than in core: every Rust caller pairs a panel with its own predictions
    // and cannot mismatch them, whereas a Python caller assembles both lists itself.
    if observed.len() != predicted.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "observed and predicted must be the same length, got {} and {}",
            observed.len(),
            predicted.len()
        )));
    }
    if observed.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cannot regress an empty panel",
        ));
    }
    let summary = irt::summarize(&observed, &predicted);
    let d = PyDict::new_bound(py);
    d.set_item("n", summary.n)?;
    d.set_item("slope", summary.slope)?;
    d.set_item("intercept", summary.intercept)?;
    d.set_item("r_squared", summary.r_squared)?;
    d.set_item("mae", summary.mae)?;
    Ok(d)
}

/// A loaded set of portable weights, kept alive across calls.
///
/// The point of holding it is parity checking: comparing a torch forward against the Rust forward
/// on the same weights, in-process, without building a binary or shelling out. Loading per call
/// would make a per-epoch check cost more than the epoch.
#[pyclass]
struct PortableWeights {
    artifact: Artifact,
}

#[pymethods]
impl PortableWeights {
    /// Load a `.safetensors` export written by `msspeculator export-rust`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let artifact = Artifact::load(path).map_err(to_pyerr)?;
        Ok(Self { artifact })
    }

    /// Dense forward for one peptidoform: `(ms2 [L-1, n_ion], rt, ccs)`.
    ///
    /// Dense and unfiltered, unlike the CLI's prediction. The comparison this exists for is
    /// against torch's own dense output. Applying a base-peak normalization and an intensity
    /// floor first would hide the small disagreements it is looking for.
    ///
    /// Acquisition context arrives as either an already-encoded vector (what torch feeds its own
    /// model) or the name of a setup fitted into these weights. It is never a `::`-separated
    /// string here: that grammar belongs to the CLI, and parsing it in a second place is what
    /// this seam exists to avoid.
    #[pyo3(signature = (peptide, charge, ms_context=None, ms_setup=None, chrom_context=None))]
    fn forward<'py>(
        &self,
        py: Python<'py>,
        peptide: &str,
        charge: i64,
        ms_context: Option<numpy::PyReadonlyArray1<'py, f32>>,
        ms_setup: Option<&str>,
        chrom_context: Option<&str>,
    ) -> PyResult<(Bound<'py, numpy::PyArray2<f32>>, f32, f32)> {
        if ms_context.is_some() && ms_setup.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pass ms_context or ms_setup, not both",
            ));
        }
        let pep = CorePeptide::parse(peptide).map_err(to_pyerr)?;
        let predictor = Predictor::new(&self.artifact);

        // Both forms end as a context vector, which the weights then project into the shift the
        // heads consume. The projection belongs to the weights, not to the caller.
        let context: Option<Array1<f32>> = match (&ms_context, ms_setup) {
            (Some(vector), _) => Some(vector.as_array().to_owned()),
            (None, Some(setup)) => Some(predictor.named_ms_context(setup).map_err(to_pyerr)?),
            (None, None) => None,
        };
        let ms_shift = match &context {
            Some(vector) => Some(predictor.context_shift(vector.view()).map_err(to_pyerr)?),
            None => None,
        };

        let (chrom_shift, chrom_affine) = match chrom_context {
            Some(dataset) => (
                Some(predictor.chrom_context_shift(dataset).map_err(to_pyerr)?),
                Some(predictor.chrom_affine(dataset).map_err(to_pyerr)?),
            ),
            None => (None, None),
        };

        let (ms2, rt, ccs) = predictor
            .forward(
                &pep,
                charge,
                ms_shift.as_ref(),
                chrom_shift.as_ref(),
                chrom_affine,
            )
            .map_err(to_pyerr)?;
        Ok((ms2.into_pyarray_bound(py), rt, ccs))
    }
}

/// Assign a bare sequence to a split, from the Rust port of `msspeculator.data.split`.
///
/// Exposed so the two implementations can be compared directly. They have to agree exactly: the
/// corpus is split in Python and a library is split in Rust, and a disagreement would put a
/// peptide the model trained on into a held-out score without anything failing.
#[pyfunction]
#[pyo3(signature = (sequence, salt, train, val))]
fn assign_split(sequence: &str, salt: &str, train: f64, val: f64) -> PyResult<String> {
    let cfg = speclib_split::SplitConfig {
        train,
        val,
        test: 1.0 - train - val,
        salt: salt.to_string(),
    };
    Ok(match speclib_split::assign_split(sequence, &cfg) {
        speclib_split::Split::Train => "train",
        speclib_split::Split::Val => "val",
        speclib_split::Split::Test => "test",
    }
    .to_string())
}

/// Read a spectral library into columnar arrays.
///
/// Fragments come back in CSR form rather than a dense grid: a library reports roughly a dozen of
/// the fifty-odd cells, and the offsets are also exactly the record of which cells it reported.
/// The peptide crosses as a ProForma string so it re-enters through the same parser the prepared
/// corpus uses, instead of introducing a second peptide representation at the boundary.
#[pyfunction]
#[pyo3(signature = (path, spec))]
fn read_speclib<'py>(
    py: Python<'py>,
    path: &str,
    spec: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let text = |key: &str| -> PyResult<String> {
        spec.get_item(key)?
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("spec is missing {key:?}"))
            })?
            .extract::<String>()
    };
    let mut aliases = Vec::new();
    if let Some(items) = spec.get_item("aliases")? {
        for item in items.iter()? {
            let entry = item?;
            let mapping = entry
                .downcast::<PyDict>()
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err("each alias must be a dict"))?;
            let accession = mapping
                .get_item("accession")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("alias needs an accession"))?
                .extract::<u32>()?;
            let observed_mass = match mapping.get_item("observed_mass")? {
                Some(value) if !value.is_none() => Some(value.extract::<f64>()?),
                _ => None,
            };
            aliases.push(speclib::ModAlias {
                accession,
                observed_mass,
            });
        }
    }
    let retention = match spec.get_item("retention_column")? {
        Some(value) if !value.is_none() => {
            speclib::RetentionSource::Column(value.extract::<String>()?)
        }
        _ => speclib::RetentionSource::Normalized,
    };
    let parsed = speclib::LibrarySpec {
        context: text("context")?,
        instrument: text("instrument")?,
        detector: text("detector")?,
        fragmentation: text("fragmentation")?,
        aliases,
        retention,
        drop_excluded: match spec.get_item("drop_excluded")? {
            Some(value) if !value.is_none() => value.extract::<bool>()?,
            _ => false,
        },
    };
    let (precursors, stats) =
        speclib::read_speclib(std::path::Path::new(path), &parsed).map_err(to_pyerr)?;

    let mut proforma = Vec::with_capacity(precursors.len());
    let mut charge = Vec::with_capacity(precursors.len());
    let mut retention_out = Vec::with_capacity(precursors.len());
    let mut mobility = Vec::with_capacity(precursors.len());
    let mut offsets = Vec::with_capacity(precursors.len() + 1);
    let mut sites = Vec::new();
    let mut ions = Vec::new();
    let mut values = Vec::new();
    offsets.push(0_i64);
    for precursor in &precursors {
        proforma.push(precursor.peptide.modified_sequence());
        charge.push(precursor.charge as i64);
        retention_out.push(precursor.retention);
        mobility.push(precursor.ion_mobility.unwrap_or(f32::NAN));
        for fragment in &precursor.fragments {
            sites.push(fragment.site as i32);
            ions.push(fragment.ion as i8);
            values.push(fragment.intensity);
        }
        offsets.push(sites.len() as i64);
    }

    let stats_dict = PyDict::new_bound(py);
    stats_dict.set_item("rows", stats.rows)?;
    stats_dict.set_item("decoys", stats.decoys)?;
    stats_dict.set_item("excluded", stats.excluded)?;
    stats_dict.set_item("precursors", stats.precursors)?;
    stats_dict.set_item(
        "precursors_without_fragments",
        stats.precursors_without_fragments,
    )?;
    stats_dict.set_item("fragments_dropped", stats.fragments_dropped)?;
    stats_dict.set_item("unmapped_masses", stats.unmapped_masses)?;

    let d = PyDict::new_bound(py);
    d.set_item("proforma", proforma)?;
    d.set_item("charge", Array1::from(charge).into_pyarray_bound(py))?;
    d.set_item(
        "retention",
        Array1::from(retention_out).into_pyarray_bound(py),
    )?;
    d.set_item(
        "ion_mobility",
        Array1::from(mobility).into_pyarray_bound(py),
    )?;
    d.set_item("frag_offset", Array1::from(offsets).into_pyarray_bound(py))?;
    d.set_item("frag_site", Array1::from(sites).into_pyarray_bound(py))?;
    d.set_item("frag_ion", Array1::from(ions).into_pyarray_bound(py))?;
    d.set_item("frag_value", Array1::from(values).into_pyarray_bound(py))?;
    d.set_item("stats", stats_dict)?;
    Ok(d)
}

#[pymodule]
fn msspeculator_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Peptide>()?;
    m.add_class::<PortableWeights>()?;
    m.add_function(wrap_pyfunction!(fragment_mz, m)?)?;
    m.add_function(wrap_pyfunction!(fragment_mz_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(ms2_target_shape, m)?)?;
    m.add_function(wrap_pyfunction!(parse_modification_rule, m)?)?;
    m.add_function(wrap_pyfunction!(unimod_title, m)?)?;
    m.add_function(wrap_pyfunction!(mod_delta, m)?)?;
    m.add_function(wrap_pyfunction!(mod_composition, m)?)?;
    m.add_function(wrap_pyfunction!(collate, m)?)?;
    m.add_function(wrap_pyfunction!(collate_prepared, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_fragment_mz, m)?)?;
    m.add_function(wrap_pyfunction!(read_speclib, m)?)?;
    m.add_function(wrap_pyfunction!(assign_split, m)?)?;
    m.add_function(wrap_pyfunction!(irt_regression, m)?)?;
    m.add_function(wrap_pyfunction!(fragment_cell, m)?)?;

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
