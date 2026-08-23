//! Retention landmark peptides, and the fit that checks what scale a model's index is on.
//!
//! An artifact's retention head reports a dimensionless index. Which index is a property of the
//! corpus it trained on, so a library cannot simply assert one -- it has to measure it. Predicting
//! this set and regressing against the published values turns the claim into a number: a slope
//! near 1 after rescaling and a high r-squared mean the model's index is an affine image of the
//! reference scale, and the fit itself is the conversion a consumer needs.
//!
//! The set is 68 PROCAL and 11 Biognosys standards, collected in
//! <https://github.com/jspaezp/ms2ml/blob/main/ms2ml/landmarks.py>, with every value expressed on
//! the **Biognosys iRT** scale. That source notes they are landmarks and should not be trained on;
//! whether a given corpus honoured that is exactly what a low r-squared would reveal.

use anyhow::Result;

use crate::artifact::Artifact;
use crate::model::Predictor;
use crate::peptide::Peptide;

/// `(sequence, Biognosys-scale iRT, vendor)`, ordered by iRT.
pub const LANDMARKS: [(&str, f64, &str); 79] = [
    ("HEHISSDYAGK", -36.83, "procal"),
    ("IGYDHGHIEHK", -33.50, "procal"),
    ("TFAHTESHISK", -33.32, "procal"),
    ("LGGNEQVTR", -24.92, "biognosys"),
    ("ISLGEHEGGGK", -18.54, "procal"),
    ("YVGDSYDSSAK", -16.87, "procal"),
    ("FGTGTYAGGEK", -9.35, "procal"),
    ("LSSGYDGTSYK", -8.82, "procal"),
    ("TASGVGGFSTK", -4.18, "procal"),
    ("LTSGDFGEDSK", -3.76, "procal"),
    ("AGDEALGDTYK", -3.52, "procal"),
    ("GAGSSEPVTGLDAK", 0.00, "biognosys"),
    ("SYASDFGSSAK", 1.79, "procal"),
    ("LYSYYSSTESK", 6.39, "procal"),
    ("FASDTSDEAFK", 7.20, "procal"),
    ("LTDTFADDDTK", 8.25, "procal"),
    ("LYTGAGYDEVK", 10.53, "procal"),
    ("VEATFGVDESNAK", 12.39, "biognosys"),
    ("TLIAYDDSSTK", 14.98, "procal"),
    ("TASEFDSAIAQDK", 17.84, "procal"),
    ("YILAGVENSK", 19.79, "biognosys"),
    ("HDLDYGIDSYK", 19.86, "procal"),
    ("FLASSEGGFTK", 20.88, "procal"),
    ("HTAYSDFLSDK", 25.90, "procal"),
    ("FVGTEYDGLAK", 26.82, "procal"),
    ("TPVISGGPYEYR", 28.71, "biognosys"),
    ("YALDSYSLSSK", 32.00, "procal"),
    ("TPVITGAPYEYR", 33.38, "biognosys"),
    ("YYGTIEDTEFK", 33.73, "procal"),
    ("GFLDYESTGAK", 35.90, "procal"),
    ("HLTGLTFDTYK", 36.50, "procal"),
    ("YFGYTSDTFGK", 41.42, "procal"),
    ("HDTVFGSYLYK", 41.42, "procal"),
    ("DGLDAASYYAPVR", 42.26, "biognosys"),
    ("FSYDGFEEDYK", 44.22, "procal"),
    ("ALFSSITDSEK", 44.88, "procal"),
    ("LYLSEYDTIGK", 48.16, "procal"),
    ("HFALFSTDVTK", 50.41, "procal"),
    ("VSGFSDISIYK", 51.67, "procal"),
    ("GSGGFTEFDLK", 51.97, "procal"),
    ("TFTGTTDSFFK", 52.20, "procal"),
    ("TFGTETFDTFK", 54.53, "procal"),
    ("ADVTPADFSEWSK", 54.62, "biognosys"),
    ("YTSFYGAYFEK", 56.65, "procal"),
    ("LTDELLSEYYK", 57.66, "procal"),
    ("ASDLLSGYYIK", 57.68, "procal"),
    ("YGFSSEDIFTK", 57.77, "procal"),
    ("HTYDDEFFTFK", 58.44, "procal"),
    ("FLFTGYDTSVK", 61.07, "procal"),
    ("GLSDYLVSTVK", 61.34, "procal"),
    ("VYAETLSGFIK", 62.57, "procal"),
    ("GLFYGGYEFTK", 62.96, "procal"),
    ("GSTDDGFIILK", 63.07, "procal"),
    ("TSIDSFIDSYK", 63.51, "procal"),
    ("TLLLDAEGFEK", 65.49, "procal"),
    ("GFVIDDGLITK", 66.46, "procal"),
    ("GFEYSIDYFSK", 66.90, "procal"),
    ("GTFIIDPGGVIR", 70.52, "biognosys"),
    ("GIFGAFTDDYK", 71.49, "procal"),
    ("LEIYTDFDAIK", 71.99, "procal"),
    ("FTEGGILDLYK", 72.95, "procal"),
    ("LLFSYSSGFVK", 73.23, "procal"),
    ("STFFSFGDVGK", 74.29, "procal"),
    ("LTAYFEDLELK", 75.09, "procal"),
    ("VDTFLDGFSVK", 76.57, "procal"),
    ("GASDFLSFAVK", 77.42, "procal"),
    ("GEDLDFIYVVK", 79.62, "procal"),
    ("VSSIFFDTFDK", 82.28, "procal"),
    ("SILDYVSLVEKK", 83.05, "procal"),
    ("GTFIIDPAAVIR", 87.23, "biognosys"),
    ("VYGYELTSLFK", 87.89, "procal"),
    ("GGFFSFGDLTK", 88.04, "procal"),
    ("YDTAIDFGLFK", 89.40, "procal"),
    ("IVLFELEGITK", 94.97, "procal"),
    ("GIEDYYIFFAK", 95.37, "procal"),
    ("SILDYVSLVEK", 96.26, "procal"),
    ("AFSDEFSYFFK", 99.13, "procal"),
    ("AFLYEIIDIGK", 99.61, "procal"),
    ("LFLQFGAQGSPFLK", 100.00, "biognosys"),
];

/// Least-squares relation between an artifact's index and the reference scale.
#[derive(Debug, Clone, Copy)]
pub struct LandmarkFit {
    pub n: usize,
    /// `index = slope * reference + intercept`.
    pub slope: f64,
    pub intercept: f64,
    pub r2: f64,
    pub resid_sd: f64,
    pub max_abs_resid: f64,
}

impl LandmarkFit {
    /// Whether the index is affine-consistent with the reference scale.
    ///
    /// The threshold is on the fit, not on the residuals: a model with several index units of
    /// error still sits on the same line, while a model trained against a different scale departs
    /// from it systematically no matter how precise it is.
    pub fn is_consistent(&self) -> bool {
        self.n >= 20 && self.r2 >= 0.98 && self.slope > 0.0
    }

    /// The reverse mapping, as an expression a consumer can apply to a library's values.
    pub fn to_reference_expression(&self) -> String {
        format!(
            "reference_irt = (value - {:.6}) / {:.6}",
            self.intercept, self.slope
        )
    }
}

/// Predict every landmark with no acquisition context and fit the result against the reference.
///
/// Context-free on purpose: the index is what the model reports before any chromatography row is
/// applied, so a fit through a named dataset would measure that dataset instead of the scale.
pub fn landmark_fit(art: &Artifact) -> Result<LandmarkFit> {
    let predictor = Predictor::new(art);
    let mut reference = Vec::with_capacity(LANDMARKS.len());
    let mut predicted = Vec::with_capacity(LANDMARKS.len());
    for (sequence, irt, _vendor) in LANDMARKS {
        let peptide = Peptide::new(sequence.to_string(), Vec::new());
        let encoded = predictor.encode(&peptide)?;
        let rt = predictor.predict_rt(&encoded, None, None)?;
        if !rt.is_finite() {
            anyhow::bail!("landmark {sequence} predicted a non-finite retention index");
        }
        reference.push(irt);
        predicted.push(f64::from(rt));
    }
    Ok(least_squares(&reference, &predicted))
}

/// Ordinary least squares of `y` on `x`, with the residual spread the caller needs to judge it.
fn least_squares(x: &[f64], y: &[f64]) -> LandmarkFit {
    let n = x.len();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n as f64;
    let (x_mean, y_mean) = (mean(x), mean(y));
    let covariance: f64 = x
        .iter()
        .zip(y)
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum();
    let variance: f64 = x.iter().map(|x| (x - x_mean).powi(2)).sum();
    let slope = if variance > 0.0 {
        covariance / variance
    } else {
        0.0
    };
    let intercept = y_mean - slope * x_mean;
    let residuals: Vec<f64> = x
        .iter()
        .zip(y)
        .map(|(x, y)| y - (slope * x + intercept))
        .collect();
    let residual_ss: f64 = residuals.iter().map(|r| r * r).sum();
    let total_ss: f64 = y.iter().map(|y| (y - y_mean).powi(2)).sum();
    LandmarkFit {
        n,
        slope,
        intercept,
        r2: if total_ss > 0.0 {
            1.0 - residual_ss / total_ss
        } else {
            0.0
        },
        resid_sd: (residual_ss / n as f64).sqrt(),
        max_abs_resid: residuals.iter().fold(0.0f64, |worst, r| worst.max(r.abs())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_landmark_is_a_predictable_peptide() {
        // The table is a fixture the fit depends on: a sequence the encoder cannot take, or a
        // length outside the model's range, would silently shrink `n` instead of failing.
        for (sequence, _irt, vendor) in LANDMARKS {
            assert!(
                (7..=30).contains(&sequence.len()),
                "{sequence} is {} residues",
                sequence.len()
            );
            assert!(
                sequence
                    .chars()
                    .all(|aa| "GASPVTCLINDQKEMHFRYW".contains(aa)),
                "{sequence} has a residue the encoder does not know"
            );
            assert!(vendor == "procal" || vendor == "biognosys", "{vendor}");
        }
        let mut sequences: Vec<&str> = LANDMARKS.iter().map(|(s, _, _)| *s).collect();
        sequences.sort_unstable();
        let unique = sequences.len();
        sequences.dedup();
        assert_eq!(sequences.len(), unique, "duplicate landmark sequence");
    }

    #[test]
    fn a_rescaled_line_is_recovered_exactly() {
        let x: Vec<f64> = (0..40).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|x| 0.75 * x + 22.5).collect();
        let fit = least_squares(&x, &y);
        assert!((fit.slope - 0.75).abs() < 1e-9, "{}", fit.slope);
        assert!((fit.intercept - 22.5).abs() < 1e-9, "{}", fit.intercept);
        assert!((fit.r2 - 1.0).abs() < 1e-9, "{}", fit.r2);
        assert!(fit.is_consistent());
        // The published expression has to invert the fit it came from.
        assert_eq!(
            fit.to_reference_expression(),
            "reference_irt = (value - 22.500000) / 0.750000"
        );
    }

    #[test]
    fn noise_around_no_relation_is_not_consistent() {
        // A model whose index has nothing to do with the reference scale must not be labelled as
        // sitting on it, however tight its own numbers are.
        let x: Vec<f64> = (0..40).map(f64::from).collect();
        let y: Vec<f64> = (0..40).map(|i| f64::from((i * 37) % 11)).collect();
        let fit = least_squares(&x, &y);
        assert!(fit.r2 < 0.5, "{}", fit.r2);
        assert!(!fit.is_consistent());
    }
}
