//! Certify: lower bounds on k*(D) and the bound chain (Sect. 3.3 of the article).
//!
//! * Degree: every term covers exactly one position of each line, so k* >= deg(D) (Lemma 1).
//! * Two-line bound: take two lines with the multisets a and b of their nonzero entries, with p and q elements. Every
//!   term links one position of each line. Linked positions form groups in which the entries of the two lines have
//!   the same sum, so the groups are blocks, and a group with m positions needs at least m - 1 links. So
//!   k* >= p + q - gamma*(a, b), where the block number gamma* is the largest number of blocks into which the entries
//!   can be split (Proposition 3). Equal values can always form blocks of their own (Lemma 4), so
//!   gamma*(a, b) = M_2 + gamma*(a', b') with the unmatched values a' and b'; `blocks` bounds gamma*(a', b').
//!   Both bounds hold for real weights.
//! * SAT certificates: if the model "k terms" is unsatisfiable, then k* > k (`sat`). They hold only for weights with
//!   precision 1/s.

use crate::instance::Instance;
use std::time::{Duration, Instant};

/// The nonzero entries of a line: rows 0..n, then columns n..2n.
fn line(inst: &Instance, index: usize) -> Vec<i64> {
    let n = inst.n;
    if index < n {
        (0..n).map(|j| inst.at(index, j)).filter(|&v| v > 0).collect()
    } else {
        (0..n).map(|i| inst.at(i, index - n)).filter(|&v| v > 0).collect()
    }
}

/// Two lines after equal values have been paired (Lemma 4).
struct Pairing {
    /// M_2, the number of pairs of equal values.
    pairs: usize,
    /// The unmatched values of the two lines, sorted.
    unmatched_a: Vec<i64>,
    unmatched_b: Vec<i64>,
    /// The quick upper bound on gamma*.
    quick_gamma: usize,
}

impl Pairing {
    /// Pairs the equal values of two sorted lines and computes the quick upper bound on gamma*: a block of unmatched
    /// values has at least three values, a block of three is {x | y, z} with x = y + z, where y and z are unmatched
    /// values of the other line, and all other blocks have at least four values.
    fn new(a: &[i64], b: &[i64]) -> Self {
        let (mut i, mut j, mut pairs) = (0, 0, 0);
        let (mut unmatched_a, mut unmatched_b) = (Vec::new(), Vec::new());
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => {
                    unmatched_a.push(a[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    unmatched_b.push(b[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    pairs += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        unmatched_a.extend_from_slice(&a[i..]);
        unmatched_b.extend_from_slice(&b[j..]);
        let triples = unmatched_a.iter().filter(|&&x| is_pair_sum(&unmatched_b, x)).count()
            + unmatched_b.iter().filter(|&&y| is_pair_sum(&unmatched_a, y)).count();
        let unmatched = unmatched_a.len() + unmatched_b.len();
        let small_blocks = triples.min(unmatched / 3);
        let quick_gamma = pairs + small_blocks + (unmatched - 3 * small_blocks) / 4;
        Pairing { pairs, unmatched_a, unmatched_b, quick_gamma }
    }

    fn unmatched(&self) -> usize {
        self.unmatched_a.len() + self.unmatched_b.len()
    }

    /// The unmatched values with signs: + for the first line, - for the second, so a block is a zero-sum set.
    fn signed(&self) -> Vec<i64> {
        self.unmatched_a.iter().copied().chain(self.unmatched_b.iter().map(|&y| -y)).collect()
    }
}

/// Is x the sum of two different elements of the sorted slice v? Two pointers, O(len).
fn is_pair_sum(v: &[i64], x: i64) -> bool {
    if v.len() < 2 {
        return false;
    }
    let (mut i, mut j) = (0, v.len() - 1);
    while i < j {
        let sum = v[i] + v[j];
        if sum == x {
            return true;
        }
        if sum < x {
            i += 1;
        } else {
            j -= 1;
        }
    }
    false
}

/// Number of unmatched values up to which the dynamic program over subsets computes gamma* (2^m bytes + 2^m i64).
pub const EXACT_UNMATCHED: usize = 22;

/// The quick upper bound on gamma* for two lines (values in any order).
pub fn quick_block_bound(a: &[i64], b: &[i64]) -> usize {
    let (mut a, mut b) = (a.to_vec(), b.to_vec());
    a.sort_unstable();
    b.sort_unstable();
    Pairing::new(&a, &b).quick_gamma
}

/// The larger of the degree and the two-line bound over all pairs of lines, within a time limit. Pass 1 computes the
/// quick bound for every pair (longest lines first). Pass 2 visits the pairs that can still raise the bound, the most
/// promising first (largest p + q - M_2), with the stronger methods: the exact block number for up to 22 unmatched
/// values; otherwise the packing bound with T = 4, 5, 6 and, for up to 44 unmatched values, the exact block number by
/// meet in the middle. `keep_going(bound)` is called whenever the bound improves and from time to time; false stops
/// the scan.
pub fn two_line_bound(inst: &Instance, time_limit_secs: f64, keep_going: &mut dyn FnMut(usize) -> bool) -> usize {
    let start = Instant::now();
    let out_of_time = || start.elapsed().as_secs_f64() > time_limit_secs;
    let n = inst.n;
    let lines: Vec<Vec<i64>> = (0..2 * n)
        .map(|i| {
            let mut values = line(inst, i);
            values.sort_unstable();
            values
        })
        .collect();
    let mut bound = lines.iter().map(|l| l.len()).max().unwrap_or(0);
    if !keep_going(bound) {
        return bound;
    }
    let mut order: Vec<usize> = (0..2 * n).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(lines[i].len()));
    // pass 1: the quick bound; keep the pairs that the stronger methods could still improve, with the best bound they
    // could give (p + q - M_2 - 1 if unmatched values are left)
    let mut candidates: Vec<(usize, usize, usize)> = Vec::new();
    let mut visited = 0u64;
    'outer: for (x, &i) in order.iter().enumerate() {
        for &j in &order[x + 1..] {
            let (a, b) = (&lines[i], &lines[j]);
            if a.len() + b.len() <= bound {
                continue;
            }
            let pairing = Pairing::new(a, b);
            let quick = a.len() + b.len() - pairing.quick_gamma;
            let best_possible = a.len() + b.len() - pairing.pairs - (pairing.unmatched() > 0) as usize;
            if best_possible > quick {
                candidates.push((best_possible, i, j));
            }
            visited += 1;
            if quick > bound {
                bound = quick;
                if !keep_going(bound) {
                    return bound;
                }
            }
            if out_of_time() || (visited % 64 == 0 && !keep_going(bound)) {
                break 'outer;
            }
        }
    }
    // pass 2: stronger upper bounds on gamma*(a', b') for the most promising pairs (`blocks`): the dynamic program for
    // m <= 22; otherwise first the packing bound with T = 4, 5, 6 (milliseconds), and for m <= 44 the meet-in-the-
    // middle method (seconds) only if the target is still open
    candidates.sort_unstable_by(|x, y| y.cmp(x));
    let deadline = start + Duration::from_secs_f64(time_limit_secs);
    let (mut dp, mut sums) = (Vec::new(), Vec::new());
    for (best_possible, i, j) in candidates {
        if best_possible <= bound || out_of_time() {
            break;
        }
        let (a, b) = (&lines[i], &lines[j]);
        let pairing = Pairing::new(a, b);
        let base = a.len() + b.len() - pairing.pairs;
        if pairing.unmatched() <= EXACT_UNMATCHED {
            let gamma = crate::blocks::exact_block_number(&pairing.signed(), &mut dp, &mut sums);
            bound = bound.max(base - gamma);
        } else {
            // the packing bound, with larger blocks while the target is not reached
            for max_block in 4..=6 {
                let packing = crate::blocks::packing_bound(
                    &pairing.unmatched_a,
                    &pairing.unmatched_b,
                    max_block,
                    200_000,
                    deadline,
                );
                if let Some(gamma) = packing.filter(|&g| base > g) {
                    bound = bound.max(base - gamma);
                }
                if !keep_going(bound) || out_of_time() {
                    return bound;
                }
            }
            if pairing.unmatched() <= 44 {
                if let Some(gamma) = crate::blocks::exact_block_number_mitm(&pairing.signed(), 2_000_000, deadline) {
                    bound = bound.max(base - gamma);
                }
            }
        }
        if !keep_going(bound) {
            break;
        }
    }
    bound
}

/// One stage of the bound chain: its name, the bound after it, and the elapsed time (s) at its end.
#[derive(Clone, Debug)]
pub struct Stage {
    pub name: &'static str,
    pub value: usize,
    pub seconds: f64,
}

/// Result of the bound chain on its own (`bound_chain`).
#[derive(Clone, Debug)]
pub struct ChainResult {
    /// Proven lower bound on k*.
    pub lower_bound: usize,
    /// Stage that produced it.
    pub method: &'static str,
    /// Upper bound, used only to stop the chain early (the Birkhoff algorithm with the zeroing rule, or a SAT model
    /// found by the chain).
    pub upper_bound: usize,
    pub stages: Vec<Stage>,
}

/// Callbacks that connect the bound chain to its caller (the `bound` binary or the Certify thread of a run).
pub struct ChainHooks<'a> {
    /// Current upper bound (length of the best known decomposition); the chain stops when its bound reaches it.
    pub upper: &'a dyn Fn() -> usize,
    /// A new proven bound and the stage that proved it; returning false stops the chain.
    pub raise: &'a mut dyn FnMut(usize, &'static str) -> bool,
    /// A decomposition found by the SAT stage (exact; the caller may use it as a solution).
    pub solution: &'a mut dyn FnMut(&crate::decomposition::Decomposition),
    /// Shared stop flag of the run, checked by the SAT solver.
    pub stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Threads for the cases of the SAT certificates (n <= 7).
    pub case_threads: usize,
}

/// The bound chain: 1. the degree; 2. the two-line bound (`two_line_bound`, at most half of the remaining time, and at
/// most 2 s for matrices treated as dense with n >= 24); 3. SAT certificates: for k = L, L + 1, ... below the upper
/// bound, an unsatisfiable model "k terms" proves k* >= k + 1. The SAT stage runs for matrices treated as sparse with
/// n <= 32 and treated as dense with n <= 9; for n <= 7 the question is split into cases (`sat::decide_by_cases`).
/// Returns the bound, the stage that proved it, and the trace of all stages.
pub fn run_bound_chain(
    inst: &Instance,
    deadline: Instant,
    hooks: &mut ChainHooks,
) -> (usize, &'static str, Vec<Stage>) {
    let start = Instant::now();
    let mut stages = Vec::new();
    let mut bound = inst.degree();
    let mut method = "degree";
    stages.push(Stage { name: "degree", value: bound, seconds: start.elapsed().as_secs_f64() });
    if !(hooks.raise)(bound, method) {
        return (bound, method, stages);
    }
    // On matrices treated as dense, the stronger methods of pass 2 never closed the gap for n >= 24, and the SAT stage
    // never raised a bound for n >= 10 in our tests. So pass 2 gets at most 2 s there, and the SAT stage is skipped.
    let sparse = inst.treated_as_sparse();
    if bound < (hooks.upper)() {
        let mut two_line_time = deadline.saturating_duration_since(Instant::now()).as_secs_f64() * 0.5;
        if !sparse && inst.n >= 24 {
            two_line_time = two_line_time.min(2.0);
        }
        let upper = hooks.upper;
        let raise = &mut hooks.raise;
        let two_line = two_line_bound(inst, two_line_time, &mut |b| raise(b, "two-line") && b < upper());
        if two_line > bound {
            bound = two_line;
            method = "two-line";
        }
        stages.push(Stage { name: "two-line", value: bound, seconds: start.elapsed().as_secs_f64() });
    }
    if bound < (hooks.upper)() && (inst.n <= 9 || (sparse && inst.n <= 32)) {
        let mut k = bound;
        while k < (hooks.upper)() && Instant::now() < deadline {
            // n <= 7: cases over the admissible compositions of one line (much faster on unsatisfiable models);
            // larger n: the model without cases
            let answer = if inst.n <= 7 {
                crate::sat::decide_by_cases(
                    &inst.d,
                    inst.n,
                    k,
                    2.0,
                    200_000,
                    hooks.case_threads,
                    deadline,
                    hooks.stop.clone(),
                )
            } else {
                crate::sat::decompose_exactly_until(&inst.d, inst.n, k, deadline, hooks.stop.clone())
            };
            match answer {
                crate::sat::SatResult::Unsat => {
                    bound = k + 1;
                    method = "sat";
                    k += 1;
                    if !(hooks.raise)(bound, method) {
                        break;
                    }
                }
                crate::sat::SatResult::Sat(d) => {
                    debug_assert!(d.verify(inst).is_ok());
                    (hooks.solution)(&d);
                    break;
                }
                crate::sat::SatResult::Unknown => break,
            }
        }
        stages.push(Stage { name: "sat", value: bound, seconds: start.elapsed().as_secs_f64() });
    }
    (bound, method, stages)
}

/// The bound chain on its own, with its own budget (binary `bound`). It stops early when the bound meets a quick
/// decomposition (the Birkhoff algorithm with the zeroing rule; for matrices treated as sparse also the frequency rule
/// with fixing by unique subset sums), or a SAT model found on the way.
pub fn bound_chain(inst: &Instance, budget_secs: f64) -> ChainResult {
    let deadline = Instant::now() + Duration::from_secs_f64(budget_secs);
    let sparse = inst.treated_as_sparse();
    let mut quick = crate::birkhoff_algorithm::decompose(inst, crate::birkhoff_algorithm::Rule::Zeroing).len();
    if sparse {
        if let Ok(w) = crate::frequency::frequency_rule(inst) {
            if let crate::fixing::FixingResult::Found(d) = crate::fixing::Fixing::new(inst, &w).solve() {
                quick = quick.min(d.len());
            }
        }
    }
    let upper_bound = std::cell::Cell::new(quick);
    let upper = || upper_bound.get();
    let mut raise = |_: usize, _: &'static str| true;
    let mut solution = |d: &crate::decomposition::Decomposition| upper_bound.set(upper_bound.get().min(d.len()));
    let mut hooks =
        ChainHooks { upper: &upper, raise: &mut raise, solution: &mut solution, stop: None, case_threads: 1 };
    let (lower_bound, method, stages) = run_bound_chain(inst, deadline, &mut hooks);
    ChainResult { lower_bound, method, upper_bound: upper_bound.get(), stages }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_lines_allow_singleton_blocks() {
        // a = b = {1, 2, 3}: three blocks of size 2, bound p + q - 3 = 3
        assert_eq!(quick_block_bound(&[1, 2, 3], &[3, 2, 1]), 3);
    }

    #[test]
    fn exact_block_number_counts_zero_sum_groups() {
        let (mut dp, mut sums) = (Vec::new(), Vec::new());
        // {5 | 2, 3} and {7 | 7}: two blocks
        assert_eq!(crate::blocks::exact_block_number(&[5, 7, -2, -3, -7], &mut dp, &mut sums), 2);
        // {4, 6 | 3, 7}: no smaller balanced block, one block
        assert_eq!(crate::blocks::exact_block_number(&[4, 6, -3, -7], &mut dp, &mut sums), 1);
    }

    #[test]
    fn disjoint_values_need_larger_blocks() {
        // no equal values, no pair sums: blocks have at least four values, so gamma* <= 1 for 2 + 2 values
        assert_eq!(quick_block_bound(&[10, 20], &[11, 19]), 1);
    }
}
