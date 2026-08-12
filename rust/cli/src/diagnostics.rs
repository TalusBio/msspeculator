//! Dependency-free model diagnostics emitted by the Rust inference CLI.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pepdistill_core::{predict, Artifact};

const IRT_STANDARDS_TSV: &str = include_str!("../../../data/reference_peptides/biognosys_irt.tsv");

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

#[derive(Debug)]
pub struct IrtSummary {
    pub n: usize,
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub mae: f64,
}

pub struct DoctorReport {
    pub summary: IrtSummary,
    pub terminal_plot: String,
    pub svg_path: PathBuf,
}

fn summarize(observed: &[f64], predicted: &[f64]) -> IrtSummary {
    let n = observed.len();
    let mean_x = observed.iter().sum::<f64>() / n as f64;
    let mean_y = predicted.iter().sum::<f64>() / n as f64;
    let covariance = observed
        .iter()
        .zip(predicted)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let variance_x = observed.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
    let variance_y = predicted.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
    let slope = if variance_x > 0.0 {
        covariance / variance_x
    } else {
        0.0
    };
    let intercept = mean_y - slope * mean_x;
    let r_squared = if variance_x > 0.0 && variance_y > 0.0 {
        covariance.powi(2) / (variance_x * variance_y)
    } else {
        0.0
    };
    let mae = observed
        .iter()
        .zip(predicted)
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>()
        / n as f64;
    IrtSummary {
        n,
        slope,
        intercept,
        r_squared,
        mae,
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
    let terminal_plot = render_terminal(&observed, &predicted);
    let svg_path = Path::new(output_dir).join("irt-scatter.svg");
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create model-doctor directory {output_dir}"))?;
    std::fs::write(
        &svg_path,
        render_svg(&standards, &observed, &predicted, &summary)?,
    )
    .with_context(|| format!("write iRT scatter {}", svg_path.display()))?;
    Ok(DoctorReport {
        summary,
        terminal_plot,
        svg_path,
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
    }
}
