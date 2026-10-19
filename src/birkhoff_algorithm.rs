//! The Birkhoff algorithm with a choice rule (Sect. 3.4 of the article).
//!
//! In each step, the algorithm chooses a permutation whose positions all have positive entries in the residual R and
//! subtracts its permutation matrix with the largest weight that keeps R nonnegative, the smallest entry of R on the
//! permutation. Such a step is a zeroing step: it makes at least one entry zero, so there are at most nnz(D) steps.
//! The bottleneck version takes a permutation with the largest possible weight, the bottleneck value theta of R. The
//! rules below choose among these permutations. All costs are exact integers.

use crate::decomposition::{Decomposition, Perm};
use crate::instance::Instance;
use crate::matching::{Matcher, FORBIDDEN};
use crate::threshold::ThresholdSets;
use rand::{Rng, SeedableRng};

/// How the Birkhoff algorithm chooses the permutation of the next step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// A bottleneck permutation, without a tie-break (Dufossé and Uçar).
    Bottleneck,
    /// The zeroing rule: among the permutations with weight theta, one that covers the largest number of positions
    /// with entry theta, so that as many entries as possible become zero.
    Zeroing,
    /// The logarithmic tie-break rule: among the permutations chosen by the zeroing rule, one that maximizes the sum
    /// of log2(R_e / (R_e - theta)) over its positions e with R_e > theta, so that it removes as many binary digits of
    /// the remaining entries as possible.
    LogTieBreak,
    /// Birkhoff+ (Valls, Iosifidis and Tassiulas): an assignment with cost -1 + beta / R_e on the positive entries,
    /// beta = (smallest positive entry) / 2, scaled to integers by 2^20. Used only for comparisons.
    BirkhoffPlus,
}

/// Work space of the Birkhoff algorithm for matrices of one size. It is reused, for example, by the greedy
/// completions of the beam search.
pub struct BirkhoffAlgorithm {
    n: usize,
    matcher: Matcher,
    thresholds: ThresholdSets,
    r: Vec<i64>,
    cost: Vec<i64>,
}

impl BirkhoffAlgorithm {
    pub fn new(n: usize) -> Self {
        BirkhoffAlgorithm {
            n,
            matcher: Matcher::new(n),
            thresholds: ThresholdSets::new(n),
            r: vec![0; n * n],
            cost: vec![0; n * n],
        }
    }

    /// The decomposition of `d` (row by row, n x n, equal line sums) that the algorithm builds with `rule`. With
    /// `rng` and `noise > 0`, a random number in [0, noise) is added to the cost of every position (randomized rule).
    pub fn decompose(
        &mut self,
        d: &[i64],
        rule: Rule,
        rng: Option<&mut dyn rand::RngCore>,
        noise: i64,
    ) -> Decomposition {
        let mut dec = Decomposition::default();
        self.steps(d, rule, rng, noise, &mut |p, w| dec.push(p.clone(), w));
        // every step zeroes a position of its permutation, so no permutation repeats, and this only sorts the terms
        dec.normalize();
        dec
    }

    /// The length of the decomposition that `decompose` would build (deterministic rule, nothing is stored).
    pub fn length(&mut self, d: &[i64], rule: Rule) -> usize {
        let mut k = 0;
        self.steps(d, rule, None, 0, &mut |_, _| k += 1);
        k
    }

    /// Runs the algorithm on `d` and calls `step` with the permutation and the weight of every step.
    fn steps(
        &mut self,
        d: &[i64],
        rule: Rule,
        mut rng: Option<&mut dyn rand::RngCore>,
        noise: i64,
        step: &mut dyn FnMut(&Perm, i64),
    ) {
        let n = self.n;
        let (matcher, thresholds, r, cost) = (&mut self.matcher, &mut self.thresholds, &mut self.r, &mut self.cost);
        r.copy_from_slice(d);
        let mut prev: Option<Perm> = None;
        let mut prev_theta: Option<i64> = None;
        let mut nonzero = r.iter().filter(|&&v| v > 0).count();
        let zeroing = matches!(rule, Rule::Zeroing | Rule::LogTieBreak);
        if zeroing {
            thresholds.reset_keys(r);
        }
        while nonzero > 0 {
            let perm = if zeroing {
                // Bottleneck value theta, then a min-cost perfect matching inside the threshold set {R >= theta}.
                // Costs (all nonnegative): 0 for a position that becomes zero (R_e = theta); otherwise 1000 for the
                // zeroing rule, or 10^6 - 1000 log2(R_e / (R_e - theta)) for the logarithmic tie-break rule (zeros
                // first, then the most binary digits removed); plus the random noise of the randomized rules.
                let (bottleneck, theta, pos) =
                    thresholds.bottleneck(r, prev.as_deref(), prev_theta).expect("the residual has a perfect matching");
                for e in thresholds.threshold_set(pos) {
                    let v = r[e];
                    let jitter = match rng.as_deref_mut() {
                        Some(g) if noise > 0 => g.gen_range(0..noise),
                        _ => 0,
                    };
                    cost[e] = jitter
                        + if v == theta {
                            0
                        } else if rule == Rule::LogTieBreak {
                            1_000_000 - (1000.0 * (v as f64 / (v - theta) as f64).log2()).round() as i64
                        } else {
                            1000
                        };
                }
                prev_theta = Some(theta);
                thresholds.assign(pos, cost).unwrap_or(bottleneck)
            } else {
                let (bottleneck, _) =
                    matcher.bottleneck(r, n, prev.as_deref()).expect("the residual has a perfect matching");
                match rule {
                    Rule::BirkhoffPlus => {
                        let min_pos = r.iter().copied().filter(|&v| v > 0).min().unwrap() as f64;
                        let beta = 0.5 * min_pos;
                        for (c, &v) in cost.iter_mut().zip(r.iter()) {
                            *c = if v > 0 {
                                ((-1.0 + beta / v as f64) * (1u64 << 20) as f64).round() as i64
                            } else {
                                FORBIDDEN
                            };
                        }
                        matcher.min_cost(cost, n).expect("the residual has a perfect matching").0
                    }
                    _ => bottleneck,
                }
            };
            let w = perm.iter().enumerate().map(|(i, &j)| r[i * n + j as usize]).min().unwrap();
            debug_assert!(w > 0);
            for (i, &j) in perm.iter().enumerate() {
                let e = i * n + j as usize;
                r[e] -= w;
                if r[e] == 0 {
                    nonzero -= 1;
                }
            }
            if zeroing {
                thresholds.update_keys(&perm, r, w);
            }
            step(&perm, w);
            prev = Some(perm);
        }
    }
}

/// The decomposition of `d` (row by row, n x n, equal line sums) by the Birkhoff algorithm with `rule`.
pub fn decompose_matrix(
    d: &[i64],
    n: usize,
    rule: Rule,
    rng: Option<&mut dyn rand::RngCore>,
    noise: i64,
) -> Decomposition {
    BirkhoffAlgorithm::new(n).decompose(d, rule, rng, noise)
}

/// The decomposition of an instance by the Birkhoff algorithm with `rule`.
pub fn decompose(inst: &Instance, rule: Rule) -> Decomposition {
    decompose_matrix(&inst.d, inst.n, rule, None, 0)
}

/// The Birkhoff algorithm with a randomized rule (`Rule::Zeroing` or `Rule::LogTieBreak`): a random number in
/// [0, noise) is added to the cost of every position. One zeroed position is worth 1000 in the zeroing rule, so
/// noise 500 chooses at random among nearly equal permutations. One binary digit is worth 1000 in the logarithmic
/// tie-break rule, so noise 300 changes choices that differ by less than 0.3 binary digits.
pub fn decompose_randomized(d: &[i64], n: usize, rule: Rule, rng: &mut impl Rng, noise: i64) -> Decomposition {
    let mut g = rand::rngs::SmallRng::seed_from_u64(rng.gen());
    decompose_matrix(d, n, rule, Some(&mut g), noise)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The zeroing rule with the dense matching code (dense bottleneck matching and Hungarian method).
    fn dense_zeroing(d: &[i64], n: usize) -> Decomposition {
        let mut m = Matcher::new(n);
        let mut r = d.to_vec();
        let mut cost = vec![0i64; n * n];
        let mut dec = Decomposition::default();
        let mut prev: Option<Perm> = None;
        while r.iter().any(|&v| v > 0) {
            let (bottleneck, theta) = m.bottleneck(&r, n, prev.as_deref()).unwrap();
            for (c, &v) in cost.iter_mut().zip(r.iter()) {
                *c = if v < theta {
                    FORBIDDEN
                } else if v == theta {
                    -1000
                } else {
                    0
                };
            }
            let perm = m.min_cost(&cost, n).map(|(p, _)| p).unwrap_or(bottleneck);
            let w = perm.iter().enumerate().map(|(i, &j)| r[i * n + j as usize]).min().unwrap();
            for (i, &j) in perm.iter().enumerate() {
                r[i * n + j as usize] -= w;
            }
            dec.push(perm.clone(), w);
            prev = Some(perm);
        }
        dec.normalize();
        dec
    }

    /// The threshold sets give the same decompositions as the dense matching code, and the logarithmic tie-break rule
    /// gives exact decompositions.
    #[test]
    fn threshold_sets_match_dense_matching() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
        for n in [4usize, 6, 9, 13, 20] {
            for trial in 0..8 {
                let k0 = if trial % 2 == 0 { n } else { n * n };
                let mut d = vec![0i64; n * n];
                for _ in 0..k0 {
                    let mut p: Vec<usize> = (0..n).collect();
                    for i in (1..n).rev() {
                        p.swap(i, rng.gen_range(0..=i));
                    }
                    let w = rng.gen_range(1..10_000);
                    for (i, &j) in p.iter().enumerate() {
                        d[i * n + j] += w;
                    }
                }
                let fast = decompose_matrix(&d, n, Rule::Zeroing, None, 0);
                let reference = dense_zeroing(&d, n);
                assert_eq!(fast.perms, reference.perms, "n = {n}, trial {trial}");
                assert_eq!(fast.weights, reference.weights);
                let log = decompose_matrix(&d, n, Rule::LogTieBreak, None, 0);
                let mut acc = vec![0i64; n * n];
                for (p, &w) in log.perms.iter().zip(&log.weights) {
                    for (i, &j) in p.iter().enumerate() {
                        acc[i * n + j as usize] += w;
                    }
                }
                assert_eq!(acc, d);
            }
        }
    }
}
