//! A decomposition D = sum of c_l P_l with integer weights c_l (precision 1/s), its exact check, and its output in the
//! QOBLIB solution format.

use crate::instance::Instance;
use std::collections::HashMap;

/// A permutation as a map from row to column (0-based): the term covers the positions (i, perm[i]).
pub type Perm = Vec<u32>;

/// The terms (c_l, pi_l) of a decomposition, as two lists of equal length.
#[derive(Clone, Debug, Default)]
pub struct Decomposition {
    pub perms: Vec<Perm>,
    pub weights: Vec<i64>,
}

impl Decomposition {
    /// The length k: the number of terms.
    pub fn len(&self) -> usize {
        self.perms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.perms.is_empty()
    }

    pub fn push(&mut self, perm: Perm, weight: i64) {
        self.perms.push(perm);
        self.weights.push(weight);
    }

    /// Merges terms with the same permutation and drops zero weights. The length never grows.
    pub fn normalize(&mut self) {
        let mut index: HashMap<Perm, usize> = HashMap::new();
        let mut perms = Vec::new();
        let mut weights = Vec::new();
        for (p, &w) in self.perms.iter().zip(&self.weights) {
            if w == 0 {
                continue;
            }
            match index.get(p) {
                Some(&i) => weights[i] += w,
                None => {
                    index.insert(p.clone(), perms.len());
                    perms.push(p.clone());
                    weights.push(w);
                }
            }
        }
        self.perms = perms;
        self.weights = weights;
    }

    /// Exact check, as done by the QOBLIB checker with tolerance 0: valid permutations, positive integer weights
    /// that sum to the scale, and the sum of the terms equals D in every position.
    pub fn verify(&self, inst: &Instance) -> Result<(), String> {
        let n = inst.n;
        let total: i64 = self.weights.iter().sum();
        if total != inst.scale {
            return Err(format!("weights sum to {total}, scale is {}", inst.scale));
        }
        let mut acc = vec![0i64; n * n];
        let mut seen = vec![false; n];
        for (p, &w) in self.perms.iter().zip(&self.weights) {
            if w <= 0 {
                return Err(format!("non-positive weight {w}"));
            }
            if p.len() != n {
                return Err("permutation of wrong length".into());
            }
            seen.iter_mut().for_each(|s| *s = false);
            for (r, &c) in p.iter().enumerate() {
                let c = c as usize;
                if c >= n || seen[c] {
                    return Err(format!("invalid permutation {p:?}"));
                }
                seen[c] = true;
                acc[r * n + c] += w;
            }
        }
        if let Some(i) = (0..n * n).find(|&i| acc[i] != inst.d[i]) {
            return Err(format!("entry ({}, {}) is {} instead of {}", i / n, i % n, acc[i], inst.d[i]));
        }
        Ok(())
    }

    /// The solution object of a QOBLIB solution file (permutations 1-based, matrix as in the instance file).
    pub fn to_qoblib_json(&self, inst: &Instance, optimal: bool) -> serde_json::Value {
        let perms: Vec<u32> = self.perms.iter().flat_map(|p| p.iter().map(|&c| c + 1)).collect();
        serde_json::json!({
            "id": inst.id,
            "n": inst.n,
            "scale": inst.scale,
            "k": self.len(),
            "optimal": optimal,
            "scaled_doubly_stochastic_matrix": inst.d,
            "weights": self.weights,
            "permutations": perms,
        })
    }
}
