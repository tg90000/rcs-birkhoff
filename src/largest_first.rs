//! Recover, alternative second step: largest weight first (Sect. 3.2 of the article).
//!
//! The weights are placed one at a time, largest first, by one assignment problem per weight. The permutation of a
//! weight w is a cheapest perfect matching over the positions that can still take it. After w goes to position (i, j),
//! the rest D_ij - w must be filled by weights that are still free in row i and in column j. The cost of a position is
//! the number of terms that this rest needs: 0 (w fills the position), 1 (one free weight), 2 (two free weights), 3 (a
//! free weight plus two unplaced ones) or 4 (anything else). Right placements leave rests that a few free weights fill
//! exactly, wrong ones rarely do. This can fail, so the result is accepted only if the final residual is zero.

use crate::decomposition::Decomposition;
use crate::instance::Instance;
use crate::matching::{Matcher, FORBIDDEN};
use std::collections::{HashMap, HashSet};

/// Largest weight first with the given weights. Err holds the number of nonzero residual entries at the end, or minus
/// the number of placed weights if an assignment problem had no solution.
pub fn largest_weight_first(inst: &Instance, weights: &[i64]) -> Result<Decomposition, i64> {
    let n = inst.n;
    let k = weights.len();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| weights[b].cmp(&weights[a]));
    let mut r = inst.d.clone();
    // avail_row[i][l]: weight l is not yet placed in row i (the same for columns)
    let mut avail_row = vec![vec![true; k]; n];
    let mut avail_col = vec![vec![true; k]; n];
    let mut by_value: HashMap<i64, Vec<usize>> = HashMap::new();
    for (l, &w) in weights.iter().enumerate() {
        by_value.entry(w).or_default().push(l);
    }
    let mut placed = vec![false; k];
    let mut dec = Decomposition::default();
    let mut cost = vec![0i64; n * n];
    let mut matcher = Matcher::new(n);
    for &l in &order {
        let w = weights[l];
        placed[l] = true;
        let unplaced: Vec<usize> = (0..k).filter(|&m| !placed[m]).collect();
        let w_min = unplaced.iter().map(|&m| weights[m]).min().unwrap_or(i64::MAX);
        // sums of two different unplaced weights (over all lines)
        let mut pair_sums: HashSet<i64> = HashSet::new();
        for (a, &x) in unplaced.iter().enumerate() {
            for &y in &unplaced[a + 1..] {
                pair_sums.insert(weights[x] + weights[y]);
            }
        }
        for (i, row_avail) in avail_row.iter().enumerate() {
            for (j, col_avail) in avail_col.iter().enumerate() {
                let e = i * n + j;
                let rem = r[e] - w;
                cost[e] = if rem < 0 || (rem > 0 && rem < w_min) {
                    FORBIDDEN
                } else if rem == 0 {
                    0
                } else {
                    let avail = |m: usize| !placed[m] && row_avail[m] && col_avail[m];
                    let is_avail_value = |x: i64, except: usize| {
                        by_value.get(&x).is_some_and(|ms| ms.iter().any(|&m| m != except && avail(m)))
                    };
                    if is_avail_value(rem, usize::MAX) {
                        1
                    } else if unplaced
                        .iter()
                        .any(|&a| avail(a) && rem - weights[a] > 0 && is_avail_value(rem - weights[a], a))
                    {
                        2
                    } else if unplaced.iter().any(|&a| avail(a) && pair_sums.contains(&(rem - weights[a]))) {
                        3
                    } else {
                        4
                    }
                };
            }
        }
        let Some((perm, _)) = matcher.min_cost(&cost, n) else {
            return Err(-(dec.len() as i64));
        };
        for (i, &j) in perm.iter().enumerate() {
            r[i * n + j as usize] -= w;
            avail_row[i][l] = false;
            avail_col[j as usize][l] = false;
        }
        dec.push(perm, w);
    }
    if r.iter().all(|&v| v == 0) {
        dec.normalize();
        Ok(dec)
    } else {
        Err(r.iter().filter(|&&v| v != 0).count() as i64)
    }
}
