//! Monoisotopic mass and m/z arithmetic — a 1:1 port of `pepdistill.chem`.
//!
//! v1 supports the 20 standard amino acids and no modifications. Fragment ordering matches
//! `chem.ION_TYPES` = (b,1),(y,1),(b,2),(y,2).

pub const PROTON: f64 = 1.007_276_466_879;
pub const H2O: f64 = 18.010_564_684_25;

/// (ion kind is_b, fragment charge) columns, in the exact order of `chem.ION_TYPES`.
pub const ION_TYPES: [(bool, u8); 4] = [(true, 1), (false, 1), (true, 2), (false, 2)];

/// Monoisotopic residue mass (Da) for a standard amino acid, or `None` if unsupported.
pub fn residue_mass(aa: u8) -> Option<f64> {
    Some(match aa {
        b'G' => 57.021_463_723,
        b'A' => 71.037_113_787,
        b'S' => 87.032_028_409,
        b'P' => 97.052_763_851,
        b'V' => 99.068_413_915,
        b'T' => 101.047_678_473,
        b'C' => 103.009_184_477,
        b'L' => 113.084_064_015,
        b'I' => 113.084_064_015,
        b'N' => 114.042_927_446,
        b'D' => 115.026_943_031,
        b'Q' => 128.058_577_540,
        b'K' => 128.094_963_016,
        b'E' => 129.042_593_095,
        b'M' => 131.040_484_605,
        b'H' => 137.058_911_861,
        b'F' => 147.068_413_915,
        b'R' => 156.101_111_050,
        b'Y' => 163.063_328_575,
        b'W' => 186.079_312_952,
        _ => return None,
    })
}

/// Per-position residue masses for a bare (unmodified) peptide.
pub fn residue_masses(seq: &[u8]) -> anyhow::Result<Vec<f64>> {
    seq.iter()
        .map(|&aa| {
            residue_mass(aa).ok_or_else(|| anyhow::anyhow!("unsupported residue {:?}", aa as char))
        })
        .collect()
}

/// Precursor m/z for a peptide of the given residue masses at `charge`.
pub fn precursor_mz(rm: &[f64], charge: i64) -> f64 {
    let total: f64 = rm.iter().sum();
    (total + H2O + charge as f64 * PROTON) / charge as f64
}

/// Fragment m/z matrix, shape `(L-1, ION_TYPES.len())`, matching `chem.fragment_mz_matrix`.
///
/// Row `i` (0-based) is the site after residue `i+1`: b ordinal `i+1`, y ordinal `L-1-i`.
pub fn fragment_mz_matrix(rm: &[f64]) -> Vec<Vec<f64>> {
    let n = rm.len();
    // prefix[i] = sum of residues 0..=i (cumulative), matching fast.py's cumsum.
    let mut prefix = vec![0.0_f64; n];
    let mut acc = 0.0;
    for (p, &m) in prefix.iter_mut().zip(rm.iter()) {
        acc += m;
        *p = acc;
    }
    let total = prefix[n - 1];
    let mut out = Vec::with_capacity(n.saturating_sub(1));
    // Row i is the site after residue i+1: b ordinal i+1, y ordinal L-1-i (ordinals assigned
    // by the caller). prefix[i] = sum residues 0..=i.
    for &pfx in &prefix[..n - 1] {
        let b_neutral = pfx;
        let y_neutral = total - pfx + H2O;
        let mut row = Vec::with_capacity(ION_TYPES.len());
        for &(is_b, z) in ION_TYPES.iter() {
            let neutral = if is_b { b_neutral } else { y_neutral };
            row.push((neutral + z as f64 * PROTON) / z as f64);
        }
        out.push(row);
    }
    out
}
