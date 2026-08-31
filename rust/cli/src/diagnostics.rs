//! Dependency-free model diagnostics emitted by the Rust inference CLI.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use msspeculator_core::irt::{summarize, IrtSummary};
use msspeculator_core::peptide::Peptide;
use msspeculator_core::similarity::spectral_angle;
use msspeculator_core::speclib::fragment_cell;
use msspeculator_core::{chem, predict, Artifact, MsContext};

const IRT_STANDARDS_TSV: &str = include_str!("../../../data/reference_peptides/biognosys_irt.tsv");
const REFERENCE_SPECTRA_TSV: &str =
    include_str!("../../../data/reference_peptides/diagnostic_spectra.tsv");

/// One vendored experimental spectrum, and how well the model reproduced it.
#[derive(Debug)]
pub struct SpectrumScore {
    pub dataset: String,
    pub proforma: String,
    pub charge: i64,
    pub spectral_angle: f32,
}

/// Parse `y8^2` / `b2` into the cell it occupies for a peptide of this length.
///
/// Goes through [`fragment_cell`], the same mapping the preparation ETL used to fill the grid, so
/// the observed and predicted spectra cannot end up indexed against different conventions.
fn annotation_cell(annotation: &str, length: usize) -> Result<(u16, u8)> {
    let (head, charge) = match annotation.split_once('^') {
        Some((head, z)) => (
            head,
            z.parse::<u8>()
                .with_context(|| format!("fragment charge in {annotation:?}"))?,
        ),
        None => (annotation, 1u8),
    };
    let kind = head
        .chars()
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty fragment annotation"))?;
    let ordinal: usize = head[kind.len_utf8()..]
        .parse()
        .with_context(|| format!("fragment ordinal in {annotation:?}"))?;
    fragment_cell(kind, ordinal, charge, length)
        .ok_or_else(|| anyhow::anyhow!("{annotation:?} is not a fragment of a {length}-mer"))
}

/// Lay a `;`-separated annotation/intensity pair out on the dense fragment grid.
fn dense_observed(annotations: &str, intensities: &str, length: usize) -> Result<Vec<f32>> {
    let columns = chem::ION_TYPES.len();
    let mut grid = vec![0.0f32; (length - 1) * columns];
    let mut values = intensities.split(';');
    for annotation in annotations.split(';') {
        let intensity: f32 = values
            .next()
            .ok_or_else(|| anyhow::anyhow!("fewer intensities than annotations"))?
            .parse()
            .with_context(|| format!("intensity for {annotation:?}"))?;
        let (site, ion) = annotation_cell(annotation, length)?;
        grid[site as usize * columns + ion as usize] = intensity;
    }
    anyhow::ensure!(
        values.next().is_none(),
        "more intensities than annotations in the reference panel"
    );
    Ok(grid)
}

/// Score the model against every vendored reference spectrum.
///
/// Each is predicted under the acquisition context the spectrum was actually recorded with. A
/// context-free prediction compared against a spectrum from a real instrument would report the
/// missing context as a bad model.
fn score_reference_spectra(artifact: &Artifact) -> Result<Vec<SpectrumScore>> {
    let mut header = REFERENCE_SPECTRA_TSV.lines();
    let columns: Vec<&str> = header
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty reference spectra table"))?
        .split('\t')
        .collect();
    let index = |name: &str| -> Result<usize> {
        columns
            .iter()
            .position(|column| *column == name)
            .ok_or_else(|| anyhow::anyhow!("reference spectra table has no {name:?} column"))
    };
    let (i_dataset, i_proforma, i_charge) =
        (index("dataset")?, index("proforma")?, index("charge")?);
    let (i_instrument, i_detector) = (index("instrument")?, index("detector")?);
    let (i_fragmentation, i_energy) = (index("fragmentation")?, index("energy")?);
    let (i_annotations, i_intensity) = (index("annotations")?, index("relative_intensity")?);

    let mut scores = Vec::new();
    for (row, line) in header.enumerate() {
        let f: Vec<&str> = line.split('\t').collect();
        let proforma = f[i_proforma];
        let charge: i64 = f[i_charge]
            .parse()
            .with_context(|| format!("charge on reference row {}", row + 2))?;
        let length = Peptide::parse(proforma)?.sequence.len();
        let observed = dense_observed(f[i_annotations], f[i_intensity], length)?;

        let context = MsContext::Factors {
            instrument: f[i_instrument].to_string(),
            detector: f[i_detector].to_string(),
            fragmentation: f[i_fragmentation].to_string(),
            energy: Some(
                f[i_energy]
                    .parse()
                    .with_context(|| format!("energy on reference row {}", row + 2))?,
            ),
        };
        // No intensity floor: the comparison wants the whole grid, and a floor would silently
        // treat a fragment the model got wrong as a fragment it never predicted.
        let prediction = predict(artifact, proforma, charge, Some(&context), None, 0.0)?;

        let columns_count = chem::ION_TYPES.len();
        let mut predicted = vec![0.0f32; (length - 1) * columns_count];
        for i in 0..prediction.fragments.ion.len() {
            let kind = prediction.fragments.ion[i]
                .chars()
                .next()
                .ok_or_else(|| anyhow::anyhow!("prediction returned an unnamed ion"))?;
            if let Some((site, ion)) = fragment_cell(
                kind,
                prediction.fragments.ord[i] as usize,
                prediction.fragments.z[i] as u8,
                length,
            ) {
                predicted[site as usize * columns_count + ion as usize] =
                    prediction.fragments.rel[i] as f32;
            }
        }

        scores.push(SpectrumScore {
            dataset: f[i_dataset].to_string(),
            proforma: proforma.to_string(),
            charge,
            spectral_angle: spectral_angle(predicted.iter().copied(), observed.iter().copied()),
        });
    }
    anyhow::ensure!(
        !scores.is_empty(),
        "the vendored reference panel carries no spectra"
    );
    Ok(scores)
}

#[derive(Debug)]
struct IrtStandard<'a> {
    peptide: &'a str,
    charge: i64,
    irt: f64,
}

fn irt_standards() -> Result<Vec<IrtStandard<'static>>> {
    IRT_STANDARDS_TSV
        .lines()
        .skip(1)
        .enumerate()
        .map(|(index, line)| {
            let fields: Vec<&str> = line.split('\t').collect();
            anyhow::ensure!(fields.len() == 5, "bad vendored iRT row {}", index + 2);
            Ok(IrtStandard {
                peptide: fields[1],
                charge: fields[2]
                    .parse()
                    .with_context(|| format!("bad iRT charge on row {}", index + 2))?,
                irt: fields[4]
                    .parse()
                    .with_context(|| format!("bad iRT value on row {}", index + 2))?,
            })
        })
        .collect()
}

pub struct DoctorReport {
    pub summary: IrtSummary,
    pub spectra: Vec<SpectrumScore>,
    pub terminal_plot: String,
    pub svg_path: PathBuf,
    pub report_path: PathBuf,
    pub predictions_path: PathBuf,
}

impl DoctorReport {
    /// Mean spectral angle over the reference panel.
    pub fn mean_spectral_angle(&self) -> f32 {
        self.spectra
            .iter()
            .map(|score| score.spectral_angle)
            .sum::<f32>()
            / self.spectra.len() as f32
    }
}

fn bounds(observed: &[f64], predicted: &[f64]) -> (f64, f64) {
    let low = observed
        .iter()
        .chain(predicted)
        .copied()
        .fold(f64::INFINITY, f64::min);
    let high = observed
        .iter()
        .chain(predicted)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let margin = ((high - low) * 0.05).max(1.0);
    (low - margin, high + margin)
}

fn render_terminal(observed: &[f64], predicted: &[f64]) -> String {
    const WIDTH: usize = 61;
    const HEIGHT: usize = 19;
    let (low, high) = bounds(observed, predicted);
    let scale_x = |value: f64| ((value - low) / (high - low) * (WIDTH - 1) as f64).round() as usize;
    let scale_y = |value: f64| {
        HEIGHT - 1 - ((value - low) / (high - low) * (HEIGHT - 1) as f64).round() as usize
    };
    let mut grid = vec![vec![' '; WIDTH]; HEIGHT];
    let mut column = 0;
    while column < WIDTH {
        let value = low + (high - low) * column as f64 / (WIDTH - 1) as f64;
        grid[scale_y(value)][column] = '.';
        column += 1;
    }
    for (&x, &y) in observed.iter().zip(predicted) {
        let cell = &mut grid[scale_y(y)][scale_x(x)];
        *cell = if *cell == 'o' { '@' } else { 'o' };
    }
    let mut output = String::from("predicted iRT\n");
    for row in grid {
        output.push('|');
        output.extend(row);
        output.push('\n');
    }
    writeln!(output, "+{}", "-".repeat(WIDTH)).expect("writing to String cannot fail");
    writeln!(output, " {low:>6.1} observed iRT {:>6.1}", high)
        .expect("writing to String cannot fail");
    output.push_str(" o peptide   . identity   @ overlapping peptides");
    output
}

fn render_svg(
    standards: &[IrtStandard<'_>],
    observed: &[f64],
    predicted: &[f64],
    summary: &IrtSummary,
) -> Result<String> {
    const WIDTH: f64 = 820.0;
    const HEIGHT: f64 = 760.0;
    const LEFT: f64 = 82.0;
    const RIGHT: f64 = 28.0;
    const TOP: f64 = 54.0;
    const BOTTOM: f64 = 78.0;
    let (low, high) = bounds(observed, predicted);
    let plot_width = WIDTH - LEFT - RIGHT;
    let plot_height = HEIGHT - TOP - BOTTOM;
    let x = |value: f64| LEFT + (value - low) / (high - low) * plot_width;
    let y = |value: f64| TOP + (high - value) / (high - low) * plot_height;

    let mut svg = String::new();
    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">"#
    )?;
    writeln!(svg, r#"<rect width="100%" height="100%" fill="white"/>"#)?;
    writeln!(
        svg,
        r#"<text x="{}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">Built-in iRT model doctor</text>"#,
        WIDTH / 2.0
    )?;
    writeln!(
        svg,
        r##"<rect x="{LEFT}" y="{TOP}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#333"/>"##
    )?;
    for tick in 0..=5 {
        let value = low + (high - low) * tick as f64 / 5.0;
        let tx = x(value);
        let ty = y(value);
        writeln!(
            svg,
            r##"<line x1="{tx}" y1="{TOP}" x2="{tx}" y2="{}" stroke="#e5e5e5"/>"##,
            TOP + plot_height
        )?;
        writeln!(
            svg,
            r##"<line x1="{LEFT}" y1="{ty}" x2="{}" y2="{ty}" stroke="#e5e5e5"/>"##,
            LEFT + plot_width
        )?;
        writeln!(
            svg,
            r#"<text x="{tx}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.1}</text>"#,
            TOP + plot_height + 22.0
        )?;
        writeln!(
            svg,
            r#"<text x="{}" y="{}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1}</text>"#,
            LEFT - 8.0,
            ty + 4.0
        )?;
    }
    writeln!(
        svg,
        r##"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#444" stroke-dasharray="6 5" stroke-width="1.5"/>"##,
        x(low),
        y(low),
        x(high),
        y(high)
    )?;
    let fit_low = summary.slope * low + summary.intercept;
    let fit_high = summary.slope * high + summary.intercept;
    writeln!(
        svg,
        r##"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#d1495b" stroke-width="2"/>"##,
        x(low),
        y(fit_low),
        x(high),
        y(fit_high)
    )?;
    for ((&observed, &predicted), standard) in observed.iter().zip(predicted).zip(standards) {
        let peptide = standard.peptide;
        writeln!(
            svg,
            r##"<circle cx="{}" cy="{}" r="4" fill="#2878b5"><title>{peptide}: observed {observed:.2}, predicted {predicted:.2}</title></circle>"##,
            x(observed),
            y(predicted)
        )?;
    }
    writeln!(
        svg,
        r#"<text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="15">reference iRT</text>"#,
        LEFT + plot_width / 2.0,
        HEIGHT - 24.0
    )?;
    writeln!(
        svg,
        r#"<text x="20" y="{}" text-anchor="middle" transform="rotate(-90 20 {})" font-family="sans-serif" font-size="15">predicted context-free iRT</text>"#,
        TOP + plot_height / 2.0,
        TOP + plot_height / 2.0
    )?;
    writeln!(
        svg,
        r#"<text x="{}" y="{}" font-family="monospace" font-size="13">slope={:.3}  intercept={:.3}  R²={:.3}  MAE={:.3}  n={}</text>"#,
        LEFT + 12.0,
        TOP + 22.0,
        summary.slope,
        summary.intercept,
        summary.r_squared,
        summary.mae,
        summary.n
    )?;
    svg.push_str("</svg>\n");
    Ok(svg)
}

fn render_predictions(standards: &[IrtStandard<'_>], predicted: &[f64]) -> Result<String> {
    let mut output = String::from(
        "proforma_sequence\tprecursor_charge\treference_irt\tpredicted_irt\tresidual\n",
    );
    for (standard, prediction) in standards.iter().zip(predicted) {
        writeln!(
            output,
            "{}\t{}\t{:.9}\t{:.9}\t{:.9}",
            standard.peptide,
            standard.charge,
            standard.irt,
            prediction,
            prediction - standard.irt,
        )?;
    }
    Ok(output)
}

fn render_report(plot: &str, summary: &IrtSummary, spectra: &[SpectrumScore]) -> Result<String> {
    let mut output = plot.to_string();
    writeln!(output)?;
    writeln!(
        output,
        "retention  n={} slope={:.6} intercept={:.6} R2={:.6} MAE={:.6}",
        summary.n, summary.slope, summary.intercept, summary.r_squared, summary.mae,
    )?;
    writeln!(output)?;
    writeln!(
        output,
        "fragmentation, against vendored real spectra. Normalized spectral contrast angle in",
    )?;
    writeln!(
        output,
        "[0, 1] on the dense (length-1, n_ion) grid, the same metric a training run reports as",
    )?;
    writeln!(
        output,
        "val/<dataset>/spectral_angle. The panel drops peaks below 1% of its base peak.",
    )?;
    for score in spectra {
        writeln!(
            output,
            "  {:<26} {:<26} z={}  spectral_angle={:.4}",
            score.dataset, score.proforma, score.charge, score.spectral_angle,
        )?;
    }
    Ok(output)
}

pub fn run_doctor(artifact: &Artifact, output_dir: &str) -> Result<DoctorReport> {
    let standards = irt_standards()?;
    let observed: Vec<f64> = standards.iter().map(|standard| standard.irt).collect();
    let predicted: Vec<f64> = standards
        .iter()
        .map(|standard| {
            predict(artifact, standard.peptide, standard.charge, None, None, 0.0)
                .map(|prediction| f64::from(prediction.rt))
        })
        .collect::<Result<_>>()?;
    let summary = summarize(&observed, &predicted);
    let spectra = score_reference_spectra(artifact)?;
    let terminal_plot = render_terminal(&observed, &predicted);
    let svg_path = Path::new(output_dir).join("irt-scatter.svg");
    let report_path = Path::new(output_dir).join("report.txt");
    let predictions_path = Path::new(output_dir).join("irt-predictions.tsv");
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create model-doctor directory {output_dir}"))?;
    std::fs::write(
        &svg_path,
        render_svg(&standards, &observed, &predicted, &summary)?,
    )
    .with_context(|| format!("write iRT scatter {}", svg_path.display()))?;
    std::fs::write(
        &report_path,
        render_report(&terminal_plot, &summary, &spectra)?,
    )
    .with_context(|| format!("write doctor report {}", report_path.display()))?;
    std::fs::write(
        &predictions_path,
        render_predictions(&standards, &predicted)?,
    )
    .with_context(|| format!("write iRT predictions {}", predictions_path.display()))?;
    Ok(DoctorReport {
        summary,
        spectra,
        terminal_plot,
        svg_path,
        report_path,
        predictions_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_panel_is_ordered_and_complete() {
        let standards = irt_standards().unwrap();
        assert_eq!(standards.len(), 11);
        assert_eq!(standards[0].peptide, "LGGNEQVTR");
        assert_eq!(standards[0].irt, -24.916114);
        assert_eq!(standards[10].peptide, "LFLQFGAQGSPFLK");
        assert_eq!(standards[10].irt, 100.00282166666665);
    }

    #[test]
    fn linear_summary_recovers_known_line() {
        let summary = summarize(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
        assert!((summary.slope - 2.0).abs() < 1e-12);
        assert!((summary.intercept - 1.0).abs() < 1e-12);
        assert!((summary.r_squared - 1.0).abs() < 1e-12);
    }

    #[test]
    fn svg_and_terminal_plots_contain_points() {
        let observed = [0.0, 1.0];
        let predicted = [0.1, 0.9];
        let summary = summarize(&observed, &predicted);
        let standards = irt_standards().unwrap();
        let svg = render_svg(&standards[..2], &observed, &predicted, &summary).unwrap();
        assert!(svg.starts_with("<svg"));
        assert_eq!(svg.matches("<circle").count(), 2);
        assert!(svg.contains("reference iRT"));
        assert!(render_terminal(&observed, &predicted).contains("identity"));
        let predictions = render_predictions(&standards[..2], &predicted).unwrap();
        assert!(predictions.starts_with("proforma_sequence\tprecursor_charge"));
        assert!(predictions.contains("LGGNEQVTR\t2"));
        let scores = [SpectrumScore {
            dataset: "pool".into(),
            proforma: "PEPTIDEK".into(),
            charge: 2,
            spectral_angle: 0.87,
        }];
        let report = render_report("plot", &summary, &scores).unwrap();
        assert!(report.contains("slope="));
        assert!(report.contains("spectral_angle=0.8700"));
    }

    /// The panel is vendored, so a truncated or reordered file has to fail loudly here rather
    /// than quietly score a model against nothing.
    #[test]
    fn the_vendored_panel_parses_onto_the_fragment_grid() {
        let mut lines = REFERENCE_SPECTRA_TSV.lines();
        let columns: Vec<&str> = lines.next().unwrap().split('\t').collect();
        let annotations = columns.iter().position(|c| *c == "annotations").unwrap();
        let intensity = columns
            .iter()
            .position(|c| *c == "relative_intensity")
            .unwrap();
        let proforma = columns.iter().position(|c| *c == "proforma").unwrap();

        let mut rows = 0;
        for line in lines {
            let f: Vec<&str> = line.split('\t').collect();
            let length = Peptide::parse(f[proforma]).unwrap().sequence.len();
            let grid = dense_observed(f[annotations], f[intensity], length).unwrap();
            assert_eq!(grid.len(), (length - 1) * chem::ION_TYPES.len());
            // Intensities are relative to the base peak, so one cell has to be exactly 1.
            assert!(grid.iter().any(|value| (value - 1.0).abs() < 1e-6));
            rows += 1;
        }
        assert!(rows >= 3, "only {rows} reference spectra");
    }

    /// b1 ions are not observed in practice. A panel claiming one means the writer applied the
    /// model's padded-pool offset to the target grid, which shifts every ordinal by one.
    #[test]
    fn the_vendored_panel_claims_no_b1_ion() {
        for line in REFERENCE_SPECTRA_TSV.lines().skip(1) {
            for annotation in line
                .split('\t')
                .find(|f| f.contains(';'))
                .unwrap()
                .split(';')
            {
                assert_ne!(annotation, "b1", "reference panel claims a b1 ion");
            }
        }
    }
}
