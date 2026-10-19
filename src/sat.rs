//! SAT model (CaDiCaL) for the question "does D have a decomposition with precision 1/s and k terms?" (Sect. 3.3 of
//! the article, Propositions 7 and 8; encodings in Sect. 4).
//!
//! Variables: cover[l][e] (y_{l,e} in the article) = "term l covers position e", and the weight c_l of term l as a
//! binary number. Constraints: each term covers exactly one position of every line; every weight is at least one; a
//! term that covers position e has weight at most D_e; the weights of the terms that cover a position add up to its
//! entry (adder trees on bits(D_e) bits without overflow); the weights are sorted. Then the weights also sum to s. A
//! model is a decomposition with at most k distinct permutations. A weight of two or more can be split over a repeated
//! permutation, so for k <= s an unsatisfiable model proves that no decomposition with k or fewer terms exists. The
//! proof holds for weights of precision 1/s, not for arbitrary real weights.
//!
//! `decide_by_cases` splits the question into cases by how many terms cover each position of one line, in the style of
//! cube-and-conquer. This proves unsatisfiability much faster on small dense matrices. `reoptimize_tail` decomposes the
//! tail of a decomposition (the terms with the smallest weights) again with fewer terms.

use crate::decomposition::Decomposition;
use cadical::{Callbacks, Solver};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Answer of the SAT model.
pub enum SatResult {
    /// a decomposition with k terms
    Sat(Decomposition),
    /// no decomposition with k terms and precision 1/s
    Unsat,
    /// the time limit or the stop flag ended the solver
    Unknown,
}

/// Termination callback: an absolute deadline and an optional shared stop flag (set when the run is over).
pub struct Limit {
    deadline: Instant,
    stop: Option<Arc<AtomicBool>>,
}

impl Callbacks for Limit {
    fn terminate(&mut self) -> bool {
        Instant::now() >= self.deadline || self.stop.as_ref().is_some_and(|s| s.load(Ordering::Relaxed))
    }
}

/// A CNF formula under construction, with a literal that is fixed to true.
struct Cnf {
    solver: Solver<Limit>,
    next: i32,
    /// A literal fixed to true; its negation is false.
    true_lit: i32,
}

impl Cnf {
    fn with_limit(deadline: Instant, stop: Option<Arc<AtomicBool>>) -> Self {
        let mut solver: Solver<Limit> = Solver::new();
        solver.set_callbacks(Some(Limit { deadline, stop }));
        let mut cnf = Cnf { solver, next: 1, true_lit: 0 };
        cnf.true_lit = cnf.var();
        cnf.solver.add_clause([cnf.true_lit]);
        cnf
    }
    fn var(&mut self) -> i32 {
        let v = self.next;
        self.next += 1;
        v
    }
    fn clause(&mut self, lits: &[i32]) {
        self.solver.add_clause(lits.iter().copied());
    }
    fn and(&mut self, a: i32, b: i32) -> i32 {
        if a == self.true_lit {
            return b;
        }
        if b == self.true_lit {
            return a;
        }
        if a == -self.true_lit || b == -self.true_lit {
            return -self.true_lit;
        }
        let z = self.var();
        self.clause(&[-z, a]);
        self.clause(&[-z, b]);
        self.clause(&[z, -a, -b]);
        z
    }
    fn xor(&mut self, a: i32, b: i32) -> i32 {
        let z = self.var();
        self.clause(&[-z, a, b]);
        self.clause(&[-z, -a, -b]);
        self.clause(&[z, -a, b]);
        self.clause(&[z, a, -b]);
        z
    }
    /// (sum, carry) of a + b + c.
    fn full_adder(&mut self, a: i32, b: i32, c: i32) -> (i32, i32) {
        let ab = self.xor(a, b);
        let sum = self.xor(ab, c);
        let carry = self.var();
        // carry = majority(a, b, c)
        self.clause(&[-a, -b, carry]);
        self.clause(&[-a, -c, carry]);
        self.clause(&[-b, -c, carry]);
        self.clause(&[a, b, -carry]);
        self.clause(&[a, c, -carry]);
        self.clause(&[b, c, -carry]);
        (sum, carry)
    }
    /// x + y (LSB first) truncated to `width` bits with the overflow forced to zero.
    fn add_bounded(&mut self, x: &[i32], y: &[i32], width: usize) -> Vec<i32> {
        let zero = -self.true_lit;
        let mut out = Vec::with_capacity(width);
        let mut carry = zero;
        for bit in 0..width.max(x.len()).max(y.len()) {
            let a = *x.get(bit).unwrap_or(&zero);
            let b = *y.get(bit).unwrap_or(&zero);
            let (sum, carry_out) =
                if a == zero && b == zero && carry == zero { (zero, zero) } else { self.full_adder(a, b, carry) };
            if bit < width {
                out.push(sum);
            } else {
                self.clause(&[-sum]);
            }
            carry = carry_out;
        }
        if carry != zero {
            self.clause(&[-carry]);
        }
        out
    }
    /// The sum of the binary numbers `terms` (LSB first) as a tree of adders, with `width` bits and no overflow.
    fn sum_bounded(&mut self, mut terms: Vec<Vec<i32>>, width: usize) -> Vec<i32> {
        if terms.is_empty() {
            return vec![-self.true_lit; width];
        }
        while terms.len() > 1 {
            let mut next = Vec::with_capacity(terms.len() / 2 + 1);
            let mut it = terms.into_iter();
            while let Some(a) = it.next() {
                match it.next() {
                    Some(b) => next.push(self.add_bounded(&a, &b, width)),
                    None => next.push(a),
                }
            }
            terms = next;
        }
        terms.pop().unwrap()
    }
    /// Fixes the binary number `bits` (LSB first) to `value`.
    fn fix(&mut self, bits: &[i32], value: i64, width: usize) {
        for b in 0..width.max(bits.len()) {
            let one = (value >> b) & 1 == 1;
            let lit = *bits.get(b).unwrap_or(&-self.true_lit);
            self.clause(&[if one { lit } else { -lit }]);
        }
    }
    /// Exactly one of `lits` (sequential counter for at-most-one).
    fn exactly_one(&mut self, lits: &[i32]) {
        self.clause(lits);
        self.at_most_one(lits);
    }
    /// At most one of `lits` (pairwise for <= 4 literals, else sequential counter).
    fn at_most_one(&mut self, lits: &[i32]) {
        if lits.len() <= 1 {
            return;
        }
        if lits.len() <= 4 {
            for a in 0..lits.len() {
                for b in a + 1..lits.len() {
                    self.clause(&[-lits[a], -lits[b]]);
                }
            }
            return;
        }
        let mut prev = lits[0];
        for &lit in &lits[1..] {
            let seen = self.var();
            self.clause(&[-prev, seen]);
            self.clause(&[-lit, seen]);
            self.clause(&[-prev, -lit]);
            prev = seen;
        }
    }
    /// guard => value(bits) <= v (bits LSB first). For every 0-bit b of v: not (bits[b] and all higher 1-bits of v set
    /// in `bits`); a larger value differs from v first at such a position (higher 0-bits of v are covered by their own
    /// clause).
    fn le_const(&mut self, guard: i32, bits: &[i32], v: i64) {
        if v >= (1i64 << bits.len().min(62)) - 1 {
            return;
        }
        for b in 0..bits.len() {
            if (v >> b) & 1 == 1 {
                continue;
            }
            let mut clause = vec![-guard, -bits[b]];
            clause.extend((b + 1..bits.len()).filter(|&h| (v >> h) & 1 == 1).map(|h| -bits[h]));
            self.clause(&clause);
        }
    }
    /// a <= b for B-bit numbers (LSB first): b = a + d with d >= 0 and no overflow.
    fn le(&mut self, a: &[i32], b: &[i32]) {
        let difference: Vec<i32> = (0..a.len()).map(|_| self.var()).collect();
        let sum = self.add_bounded(a, &difference, b.len());
        for (x, y) in sum.iter().zip(b) {
            self.clause(&[-x, *y]);
            self.clause(&[*x, -y]);
        }
    }
}

fn bits_of(v: i64) -> usize {
    (64 - (v.max(1) as u64).leading_zeros()) as usize
}

/// Decides whether the matrix `d` (row by row, n x n, equal line sums) has a decomposition with exactly k terms (equal
/// permutations allowed), within `timeout` seconds.
pub fn decompose_exactly(d: &[i64], n: usize, k: usize, timeout: f32) -> SatResult {
    decompose_exactly_until(d, n, k, Instant::now() + Duration::from_secs_f32(timeout.max(0.0)), None)
}

/// Like `decompose_exactly`, with an absolute deadline and an optional stop flag.
pub fn decompose_exactly_until(
    d: &[i64],
    n: usize,
    k: usize,
    deadline: Instant,
    stop: Option<Arc<AtomicBool>>,
) -> SatResult {
    match Model::build(d, n, k, None, deadline, stop) {
        Some(mut m) => m.solve(),
        None => SatResult::Unsat,
    }
}

/// The model for k terms. Without a case: the terms are sorted by weight (c_1 <= ... <= c_k). With a case (the
/// positions of one line and the number of terms that cover each of them): the terms are grouped by their position in
/// that line, fixed there, and sorted by weight within each group instead of sorting all weights. Every decomposition
/// with k terms has such a labelling in the case it belongs to, so the cases of all admissible compositions together
/// are equivalent to the model without cases (Proposition 8).
struct Model {
    cnf: Cnf,
    cover: Vec<Vec<i32>>,
    weight_bits: Vec<Vec<i32>>,
    positions: Vec<usize>,
    n: usize,
}

impl Model {
    fn build(
        d: &[i64],
        n: usize,
        k: usize,
        case: Option<(&[usize], &[usize])>,
        deadline: Instant,
        stop: Option<Arc<AtomicBool>>,
    ) -> Option<Model> {
        let s: i64 = d[..n].iter().sum();
        if (k as i64) > s || k == 0 {
            return None;
        }
        let weight_width = bits_of(s);
        let mut cnf = Cnf::with_limit(deadline, stop);
        let positions: Vec<usize> = (0..n * n).filter(|&e| d[e] > 0).collect();
        let mut position_index = vec![usize::MAX; n * n];
        for (index, &e) in positions.iter().enumerate() {
            position_index[e] = index;
        }
        let cover: Vec<Vec<i32>> = (0..k).map(|_| positions.iter().map(|_| cnf.var()).collect()).collect();
        let weight_bits: Vec<Vec<i32>> = (0..k).map(|_| (0..weight_width).map(|_| cnf.var()).collect()).collect();
        for l in 0..k {
            for i in 0..n {
                let row: Vec<i32> =
                    (0..n).filter(|&j| d[i * n + j] > 0).map(|j| cover[l][position_index[i * n + j]]).collect();
                if row.is_empty() {
                    return None;
                }
                cnf.exactly_one(&row);
                let col: Vec<i32> =
                    (0..n).filter(|&r| d[r * n + i] > 0).map(|r| cover[l][position_index[r * n + i]]).collect();
                cnf.exactly_one(&col);
            }
            cnf.clause(&weight_bits[l]); // c_l >= 1
        }
        for (index, &e) in positions.iter().enumerate() {
            let v = d[e];
            let w = bits_of(v);
            let mut terms = Vec::with_capacity(k);
            for l in 0..k {
                // a term that covers e weighs at most D_e (comparator; it also clears the bits above bits(D_e))
                cnf.le_const(cover[l][index], &weight_bits[l], v);
                let masked: Vec<i32> = (0..w).map(|b| cnf.and(weight_bits[l][b], cover[l][index])).collect();
                terms.push(masked);
            }
            let total = cnf.sum_bounded(terms, w);
            cnf.fix(&total, v, w);
        }
        let total = cnf.sum_bounded(weight_bits.clone(), weight_width);
        cnf.fix(&total, s, weight_width);
        match case {
            None => {
                for l in 0..k.saturating_sub(1) {
                    let (a, b) = (weight_bits[l].clone(), weight_bits[l + 1].clone());
                    cnf.le(&a, &b);
                }
            }
            Some((line, counts)) => {
                let mut l = 0;
                for (&e, &m) in line.iter().zip(counts) {
                    let group: Vec<usize> = (l..l + m).collect();
                    for &t in &group {
                        cnf.clause(&[cover[t][position_index[e]]]);
                    }
                    for w in group.windows(2) {
                        let (a, b) = (weight_bits[w[0]].clone(), weight_bits[w[1]].clone());
                        cnf.le(&a, &b);
                    }
                    // the weights of the group add up to the entry (implied; it helps propagation)
                    let entry_width = bits_of(d[e]);
                    let parts: Vec<Vec<i32>> =
                        group.iter().map(|&t| weight_bits[t][..entry_width.min(weight_width)].to_vec()).collect();
                    let sum = cnf.sum_bounded(parts, entry_width);
                    cnf.fix(&sum, d[e], entry_width);
                    l += m;
                }
                debug_assert_eq!(l, k);
            }
        }
        Some(Model { cnf, cover, weight_bits, positions, n })
    }

    fn solve(&mut self) -> SatResult {
        match self.cnf.solver.solve() {
            Some(true) => {
                let mut dec = Decomposition::default();
                for l in 0..self.cover.len() {
                    let weight: i64 = (0..self.weight_bits[l].len())
                        .filter(|&b| self.cnf.solver.value(self.weight_bits[l][b]) == Some(true))
                        .map(|b| 1i64 << b)
                        .sum();
                    let mut perm = vec![0u32; self.n];
                    for (index, &e) in self.positions.iter().enumerate() {
                        if self.cnf.solver.value(self.cover[l][index]) == Some(true) {
                            perm[e / self.n] = (e % self.n) as u32;
                        }
                    }
                    dec.push(perm, weight);
                }
                dec.normalize();
                SatResult::Sat(dec)
            }
            Some(false) => SatResult::Unsat,
            None => SatResult::Unknown,
        }
    }
}

/// The admissible compositions of k over the positions of a line: min_counts[e] <= r_e <= D_e (every weight is at
/// least one) and the r_e sum to k. None if there are more than `limit`.
fn admissible_compositions(min_counts: &[usize], caps: &[i64], k: usize, limit: usize) -> Option<Vec<Vec<usize>>> {
    struct Compositions<'a> {
        min_counts: &'a [usize],
        caps: &'a [i64],
        out: Vec<Vec<usize>>,
        limit: usize,
    }
    fn extend(state: &mut Compositions, i: usize, left: usize, current: &mut Vec<usize>) -> bool {
        if i == state.caps.len() {
            if left == 0 {
                state.out.push(current.clone());
            }
            return state.out.len() <= state.limit;
        }
        let min_rest: usize = state.min_counts[i + 1..].iter().sum();
        let max_rest: i64 = state.caps[i + 1..].iter().sum();
        for m in state.min_counts[i]..=left.min(state.caps[i] as usize) {
            let after = left - m;
            if after < min_rest || (after as i64) > max_rest {
                continue;
            }
            current.push(m);
            let ok = extend(state, i + 1, after, current);
            current.pop();
            if !ok {
                return false;
            }
        }
        true
    }
    let mut state = Compositions { min_counts, caps, out: Vec::new(), limit };
    extend(&mut state, 0, k, &mut Vec::new()).then_some(state.out)
}

/// Decides "does D have a decomposition with precision 1/s and k terms?" by cases, in the style of cube-and-conquer
/// (Proposition 8). A case fixes how many terms cover each position of one line, the line (row or column) with the
/// fewest admissible compositions. A term that covers position e weighs at most U_e, the largest value of
/// min_i D_{i,pi(i)} over the permutations pi through e, so at least ceil(D_e / U_e) terms cover e, and the cases below
/// these numbers are skipped. The cases are solved on `threads` threads, first with `per_case` seconds each, then the
/// timed-out ones with four times longer limits until the deadline. Unsat only if every case is unsatisfiable (a proof
/// that k* > k); Sat as soon as one case has a model; Unknown otherwise.
#[allow(clippy::too_many_arguments)]
pub fn decide_by_cases(
    d: &[i64],
    n: usize,
    k: usize,
    per_case: f64,
    max_cases: usize,
    threads: usize,
    deadline: Instant,
    stop: Option<Arc<AtomicBool>>,
) -> SatResult {
    let min_counts = min_cover_counts(d, n);
    let lines: Vec<Vec<usize>> = (0..n)
        .map(|i| (0..n).map(|j| i * n + j).filter(|&e| d[e] > 0).collect())
        .chain((0..n).map(|j| (0..n).map(|i| i * n + j).filter(|&e| d[e] > 0).collect()))
        .collect();
    let mut best: Option<(Vec<usize>, Vec<Vec<usize>>)> = None;
    for line in lines {
        let caps: Vec<i64> = line.iter().map(|&e| d[e]).collect();
        let line_min_counts: Vec<usize> = line.iter().map(|&e| min_counts[e]).collect();
        let limit = best.as_ref().map_or(max_cases, |b| b.1.len().saturating_sub(1).min(max_cases));
        if let Some(cases) = admissible_compositions(&line_min_counts, &caps, k, limit) {
            if best.as_ref().is_none_or(|b| cases.len() < b.1.len()) {
                best = Some((line, cases));
            }
        }
    }
    let Some((line, mut open)) = best else { return SatResult::Unknown };
    // one local flag ends all case models: set when a case is satisfiable, or copied from the stop flag of the run
    let halt = Arc::new(AtomicBool::new(false));
    let stopped = || stop.as_ref().is_some_and(|f| f.load(Ordering::Relaxed));
    let found: std::sync::Mutex<Option<Decomposition>> = std::sync::Mutex::new(None);
    let mut budget = per_case;
    while !open.is_empty() {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let timed_out = std::sync::Mutex::new(Vec::new());
        let threads_left = std::sync::atomic::AtomicUsize::new(threads.max(1));
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while threads_left.load(Ordering::Relaxed) > 0 {
                    if stopped() || Instant::now() >= deadline {
                        halt.store(true, Ordering::Relaxed);
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            });
            for _ in 0..threads.max(1) {
                scope.spawn(|| {
                    loop {
                        let case = next.fetch_add(1, Ordering::Relaxed);
                        if case >= open.len() || halt.load(Ordering::Relaxed) {
                            break;
                        }
                        let case_deadline = deadline.min(Instant::now() + Duration::from_secs_f64(budget));
                        match Model::build(d, n, k, Some((&line, &open[case])), case_deadline, Some(halt.clone())) {
                            None => {}
                            Some(mut m) => match m.solve() {
                                SatResult::Unsat => {}
                                SatResult::Sat(dec) => {
                                    *found.lock().unwrap() = Some(dec);
                                    halt.store(true, Ordering::Relaxed);
                                }
                                SatResult::Unknown => {
                                    if !halt.load(Ordering::Relaxed) {
                                        timed_out.lock().unwrap().push(case);
                                    }
                                }
                            },
                        }
                    }
                    threads_left.fetch_sub(1, Ordering::Relaxed);
                });
            }
        });
        if let Some(dec) = found.lock().unwrap().take() {
            return SatResult::Sat(dec);
        }
        if halt.load(Ordering::Relaxed) || stopped() || Instant::now() >= deadline {
            return SatResult::Unknown;
        }
        let mut retry: Vec<usize> = timed_out.into_inner().unwrap();
        retry.sort_unstable();
        open = retry.into_iter().map(|case| open[case].clone()).collect();
        budget *= 4.0;
    }
    SatResult::Unsat
}

/// For every position e = (i, j) with D_e > 0: the least number of terms that cover e in any decomposition,
/// ceil(D_e / U_e), where U_e = min(D_e, bottleneck value of D without row i and column j) bounds the weight of every
/// term that covers e. This also holds for real weights.
fn min_cover_counts(d: &[i64], n: usize) -> Vec<usize> {
    let mut min_counts = vec![0usize; n * n];
    let m = n - 1;
    let mut minor = vec![0i64; m * m];
    let mut matcher = crate::matching::Matcher::new(m.max(1));
    for e in (0..n * n).filter(|&e| d[e] > 0) {
        let (ei, ej) = (e / n, e % n);
        let max_weight = if m == 0 {
            d[e]
        } else {
            let mut index = 0;
            for i in (0..n).filter(|&i| i != ei) {
                for j in (0..n).filter(|&j| j != ej) {
                    minor[index] = d[i * n + j];
                    index += 1;
                }
            }
            matcher.bottleneck(&minor, m, None).map_or(d[e], |(_, theta)| theta.min(d[e]))
        };
        min_counts[e] = ((d[e] + max_weight - 1) / max_weight) as usize;
    }
    min_counts
}

/// Largest tail model (tail terms x positions of their sum) that the SAT re-optimization of the tail builds (about
/// 0.85 GB).
const MAX_TAIL_TERM_POSITIONS: usize = 50_000;

/// The SAT re-optimization of the tail. The tail is the largest set of terms with the smallest weights whose sum has no
/// entry above `max_entry`. We ask the SAT model for a decomposition of this sum with fewer terms, as long as the
/// answers come before the deadline. The other terms stay unchanged, so the result is exact. Returns true if `dec`
/// became shorter.
pub fn reoptimize_tail(
    dec: &mut Decomposition,
    n: usize,
    max_entry: i64,
    deadline: Instant,
    stop: Option<&Arc<AtomicBool>>,
) -> bool {
    let mut order: Vec<usize> = (0..dec.len()).collect();
    order.sort_by(|&a, &b| dec.weights[b].cmp(&dec.weights[a]));
    // the tail: the longest suffix (in weight order) whose sum has entries <= max_entry
    let mut r = vec![0i64; n * n];
    let mut start = order.len();
    while start > 0 {
        let l = order[start - 1];
        let mut trial = r.clone();
        for (i, &j) in dec.perms[l].iter().enumerate() {
            trial[i * n + j as usize] += dec.weights[l];
        }
        if trial.iter().any(|&v| v > max_entry) {
            break;
        }
        r = trial;
        start -= 1;
    }
    let tail_terms = order.len() - start;
    if tail_terms < 3 {
        return false;
    }
    // memory guard: the model has about (terms x positions) cover variables plus their adders, so very large tails are
    // skipped (at n = 64 and cap 63, a tail of 75 terms over 2851 positions would need 3.6 GB)
    let positions = r.iter().filter(|&&v| v > 0).count();
    if tail_terms * positions > MAX_TAIL_TERM_POSITIONS {
        return false;
    }
    let mut best: Option<Decomposition> = None;
    let mut k = tail_terms - 1;
    while k >= 1 {
        if deadline.saturating_duration_since(Instant::now()).as_secs_f32() < 0.05 {
            break;
        }
        match decompose_exactly_until(&r, n, k, deadline, stop.cloned()) {
            SatResult::Sat(d) => {
                k = d.len().saturating_sub(1).min(k - 1);
                best = Some(d);
            }
            _ => break,
        }
    }
    let Some(tail) = best else { return false };
    let mut out = Decomposition::default();
    for &l in &order[..start] {
        out.push(dec.perms[l].clone(), dec.weights[l]);
    }
    for (p, w) in tail.perms.into_iter().zip(tail.weights) {
        out.push(p, w);
    }
    out.normalize();
    if out.len() < dec.len() {
        *dec = out;
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};

    fn random_matrix(n: usize, terms: usize, seed: u64) -> Vec<i64> {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut d = vec![0i64; n * n];
        for _ in 0..terms {
            let mut p: Vec<usize> = (0..n).collect();
            p.shuffle(&mut rng);
            let w = rng.gen_range(1..40);
            for (i, &j) in p.iter().enumerate() {
                d[i * n + j] += w;
            }
        }
        d
    }

    #[test]
    fn cases_agree_with_the_model_without_cases() {
        let far = || Instant::now() + Duration::from_secs(60);
        let mut checked = 0;
        for seed in 0..12u64 {
            let n = 3 + (seed as usize % 2);
            let d = random_matrix(n, 3 + seed as usize % 5, seed);
            for k in 1..=7 {
                let mono = decompose_exactly_until(&d, n, k, far(), None);
                let case = decide_by_cases(&d, n, k, 60.0, 100_000, 2, far(), None);
                let tag = |r: &SatResult| match r {
                    SatResult::Sat(_) => 1,
                    SatResult::Unsat => 0,
                    SatResult::Unknown => 2,
                };
                assert_eq!(tag(&mono), tag(&case), "n = {n}, k = {k}, seed = {seed}");
                if let SatResult::Sat(dec) = &case {
                    let mut acc = vec![0i64; n * n];
                    for (p, &w) in dec.perms.iter().zip(&dec.weights) {
                        for (i, &j) in p.iter().enumerate() {
                            acc[i * n + j as usize] += w;
                        }
                    }
                    assert_eq!(acc, d);
                }
                checked += 1;
            }
        }
        assert!(checked >= 80);
    }
}
