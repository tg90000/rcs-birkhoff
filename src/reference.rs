//! Best known values from the QOBLIB `solutions/` folder, used only for reporting (never by the solver). Only exact
//! decompositions with whole-number weights count; the folder also holds approximate solutions from submissions that
//! allowed a small error.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// id -> (best known k among exact integer solutions, proven optimal, file holding a best exact solution)
pub fn best_known(solutions_dir: &Path) -> HashMap<String, (usize, bool, PathBuf)> {
    let mut best: HashMap<String, (usize, bool, PathBuf)> = HashMap::new();
    let Ok(entries) = std::fs::read_dir(solutions_dir) else { return best };
    for entry in entries.flatten() {
        let Ok(text) = std::fs::read_to_string(entry.path()) else { continue };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else { continue };
        let Some(map) = value.as_object() else { continue };
        for v in map.values() {
            let Some(k) = exact_k(v) else { continue };
            let id = v["id"].as_str().unwrap().to_string();
            let opt = v.get("optimal").and_then(|x| x.as_bool()).unwrap_or(false);
            let e = best.entry(id).or_insert((k, opt, entry.path()));
            if k < e.0 {
                *e = (k, opt, entry.path());
            } else if k == e.0 {
                e.1 |= opt;
            }
        }
    }
    best
}

/// Number of permutations if the entry is an exact integer decomposition of its matrix.
fn exact_k(v: &serde_json::Value) -> Option<usize> {
    let d: Vec<i64> =
        v.get("scaled_doubly_stochastic_matrix")?.as_array()?.iter().map(|x| x.as_i64()).collect::<Option<_>>()?;
    let n = (d.len() as f64).sqrt().round() as usize;
    let w: Vec<f64> = v.get("weights")?.as_array()?.iter().map(|x| x.as_f64()).collect::<Option<_>>()?;
    if w.iter().any(|x| x.fract() != 0.0) {
        return None;
    }
    let p: Vec<usize> =
        v.get("permutations")?.as_array()?.iter().map(|x| x.as_u64().map(|y| y as usize)).collect::<Option<_>>()?;
    if p.len() != w.len() * n {
        return None;
    }
    let mut acc = vec![0i64; n * n];
    for (l, &wl) in w.iter().enumerate() {
        for i in 0..n {
            let c = p[l * n + i].checked_sub(1)?;
            if c >= n {
                return None;
            }
            acc[i * n + c] += wl as i64;
        }
    }
    (acc == d).then(|| w.iter().filter(|&&x| x > 0.0).count())
}
