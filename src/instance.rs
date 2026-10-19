//! Test instances of QOBLIB problem 03 (minimum Birkhoff decomposition).
//!
//! An instance is a doubly stochastic n x n matrix A whose entries are multiples of 1/s. We store the scaled matrix
//! D = sA: its entries are nonnegative integers, and all its row and column sums are equal to s (the `scale`).
//! The scaling only lets us compute in exact integer arithmetic; it does not change the problem.
//!
//! The reader takes only `id`, `n`, `scale` and the matrix. The instance files also store the generating permutations
//! and their weights. These are not fields of `Instance`, so they are skipped when reading, and the solver never sees
//! them.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// Share of repeated values from which RCS treats a matrix as sparse (Sect. 2.2 of the article). Every threshold
/// between 41 % and 51 % gives the same classification of the test instances.
pub const SPARSE_THRESHOLD: f64 = 0.45;

/// The scaled matrix D = sA of one test instance, stored row by row.
#[derive(Clone, Debug)]
pub struct Instance {
    pub id: String,
    pub n: usize,
    /// The scale s: every row and every column of `d` sums to s.
    pub scale: i64,
    /// Entry (i, j) is `d[i * n + j]`. The QOBLIB checker uses the same layout: a permutation maps row i to
    /// column perm[i].
    pub d: Vec<i64>,
}

#[derive(Deserialize)]
struct RawInstance {
    id: String,
    n: usize,
    scale: i64,
    scaled_doubly_stochastic_matrix: Vec<i64>,
}

impl Instance {
    /// Entry D_ij.
    pub fn at(&self, i: usize, j: usize) -> i64 {
        self.d[i * self.n + j]
    }

    /// Checks that all entries are nonnegative and that every line (row or column) sums to the scale.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.n;
        if self.d.len() != n * n {
            return Err(format!("{}: matrix has {} entries, expected {}", self.id, self.d.len(), n * n));
        }
        if self.d.iter().any(|&v| v < 0) {
            return Err(format!("{}: negative entry", self.id));
        }
        for i in 0..n {
            let row: i64 = (0..n).map(|j| self.at(i, j)).sum();
            let col: i64 = (0..n).map(|j| self.at(j, i)).sum();
            if row != self.scale || col != self.scale {
                return Err(format!("{}: line {} sums to {}/{} instead of {}", self.id, i, row, col, self.scale));
            }
        }
        Ok(())
    }

    /// Number of nonzero entries.
    pub fn nnz(&self) -> usize {
        self.d.iter().filter(|&&v| v > 0).count()
    }

    /// Share of the nonzero entries whose value occurs at least twice in the matrix. It is high when a few
    /// permutations generated the matrix, because their weights reappear as entries, and low when many did.
    pub fn repeat_fraction(&self) -> f64 {
        let mut freq = std::collections::HashMap::new();
        for &v in self.d.iter().filter(|&&v| v > 0) {
            *freq.entry(v).or_insert(0usize) += 1;
        }
        let nnz = self.nnz().max(1);
        freq.values().filter(|&&f| f >= 2).sum::<usize>() as f64 / nnz as f64
    }

    /// Does RCS treat the matrix as sparse? Yes if at least 45 % of its nonzero entries have a repeated value.
    pub fn treated_as_sparse(&self) -> bool {
        self.repeat_fraction() >= SPARSE_THRESHOLD
    }

    /// The degree deg(D): the largest number of nonzero entries in a line. Every decomposition has at least this
    /// many terms, also with real weights (Lemma 1).
    pub fn degree(&self) -> usize {
        let n = self.n;
        (0..n)
            .map(|i| {
                let r = (0..n).filter(|&j| self.at(i, j) > 0).count();
                let c = (0..n).filter(|&j| self.at(j, i) > 0).count();
                r.max(c)
            })
            .max()
            .unwrap_or(0)
    }
}

/// Reads all instances of one QOBLIB `qbench_*.json` file, ordered by their key ("001", "002", ...).
pub fn read_instances(path: &Path) -> Result<Vec<Instance>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for value in raw.into_values() {
        if !value.is_object() || value.get("id").is_none() {
            continue; // for example "_license"
        }
        let r: RawInstance = serde_json::from_value(value).map_err(|e| format!("{}: {e}", path.display()))?;
        let inst = Instance { id: r.id, n: r.n, scale: r.scale, d: r.scaled_doubly_stochastic_matrix };
        inst.validate()?;
        out.push(inst);
    }
    Ok(out)
}

/// All instances of all `qbench_*.json` files in a directory, sorted by size, then sparse before dense, then index.
pub fn read_instance_dir(dir: &Path) -> Result<Vec<Instance>, String> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name().and_then(|f| f.to_str()).is_some_and(|f| f.starts_with("qbench_") && f.ends_with(".json"))
        })
        .collect();
    files.sort();
    let mut all = Vec::new();
    for f in files {
        all.extend(read_instances(&f)?);
    }
    all.sort_by_key(|i| sort_key(&i.id));
    Ok(all)
}

/// (n, number of generating permutations, index), read from an id such as "B16_256_3".
pub fn sort_key(id: &str) -> (usize, usize, usize) {
    let parts: Vec<usize> = id.trim_start_matches('B').split('_').filter_map(|p| p.parse().ok()).collect();
    (parts.first().copied().unwrap_or(0), parts.get(1).copied().unwrap_or(0), parts.get(2).copied().unwrap_or(0))
}
