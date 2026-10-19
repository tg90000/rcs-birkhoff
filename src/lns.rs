//! Large neighborhood search on a complete decomposition (Sect. 3.4 of the article, paragraph "Improvement").
//!
//! We remove a group of terms, namely a random term with a small weight together with the terms that share the most
//! positions with the group. The sum of these terms is again a nonnegative matrix with equal line sums, with few
//! nonzero entries. We decompose it again with the randomized zeroing rule (three tries) and keep the result if it has
//! fewer terms, or the same number (so the search keeps moving). The decomposition stays exact in every step.

use crate::birkhoff_algorithm::{decompose_randomized, Rule};
use crate::decomposition::{Decomposition, Perm};
use crate::instance::Instance;
use rand::Rng;
use std::time::Instant;

/// Counters of one call of [`improve`].
pub struct LnsStats {
    pub iterations: u64,
    pub improvements: u64,
}

/// The sum of the terms (perms, weights) as an n x n matrix, row by row.
fn sum_of(n: usize, perms: &[Perm], weights: &[i64]) -> Vec<i64> {
    let mut d = vec![0i64; n * n];
    for (p, &w) in perms.iter().zip(weights) {
        for (i, &j) in p.iter().enumerate() {
            d[i * n + j as usize] += w;
        }
    }
    d
}

/// Improves `dec` in place until the deadline or until `running()` turns false, and calls `on_improve` after every
/// strict improvement.
pub fn improve(
    inst: &Instance,
    dec: &mut Decomposition,
    deadline: Instant,
    running: &dyn Fn() -> bool,
    rng: &mut impl Rng,
    on_improve: &mut dyn FnMut(&Decomposition),
) -> LnsStats {
    let n = inst.n;
    let mut stats = LnsStats { iterations: 0, improvements: 0 };
    while Instant::now() < deadline && dec.len() > 1 && running() {
        stats.iterations += 1;
        let k = dec.len();
        let group_size = rng.gen_range(3..=(k.min(24)));
        // the first term of the group: the smallest weight of three random terms
        let mut first = rng.gen_range(0..k);
        for _ in 0..2 {
            let other = rng.gen_range(0..k);
            if dec.weights[other] < dec.weights[first] {
                first = other;
            }
        }
        // grow the group by the overlap with the positions covered so far
        let mut chosen = vec![first];
        let mut covered = vec![false; n * n];
        let mark = |p: &Perm, covered: &mut Vec<bool>| {
            for (i, &j) in p.iter().enumerate() {
                covered[i * n + j as usize] = true;
            }
        };
        mark(&dec.perms[first], &mut covered);
        while chosen.len() < group_size {
            let mut best: Option<(usize, usize)> = None;
            for l in 0..k {
                if chosen.contains(&l) {
                    continue;
                }
                let overlap = dec.perms[l].iter().enumerate().filter(|&(i, &j)| covered[i * n + j as usize]).count();
                let score = overlap * 16 + rng.gen_range(0..16);
                if best.is_none_or(|(_, best_score)| score > best_score) {
                    best = Some((l, score));
                }
            }
            let (l, _) = best.unwrap();
            chosen.push(l);
            mark(&dec.perms[l], &mut covered);
        }
        let group_perms: Vec<Perm> = chosen.iter().map(|&l| dec.perms[l].clone()).collect();
        let group_weights: Vec<i64> = chosen.iter().map(|&l| dec.weights[l]).collect();
        let group_sum = sum_of(n, &group_perms, &group_weights);
        let mut repair: Option<Decomposition> = None;
        for _ in 0..3 {
            let candidate = decompose_randomized(&group_sum, n, Rule::Zeroing, rng, 500);
            if repair.as_ref().is_none_or(|best| candidate.len() < best.len()) {
                repair = Some(candidate);
            }
        }
        let repair = repair.unwrap();
        if repair.len() <= chosen.len() {
            let improved = repair.len() < chosen.len();
            let mut updated = Decomposition::default();
            for l in 0..k {
                if !chosen.contains(&l) {
                    updated.push(dec.perms[l].clone(), dec.weights[l]);
                }
            }
            for (p, w) in repair.perms.into_iter().zip(repair.weights) {
                updated.push(p, w);
            }
            updated.normalize();
            if updated.len() <= k {
                let strictly_shorter = updated.len() < k || improved;
                *dec = updated;
                if strictly_shorter {
                    stats.improvements += 1;
                    on_improve(dec);
                }
            }
        }
    }
    stats
}
