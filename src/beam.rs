//! Search: beam search over zeroing steps (Sect. 3.4 of the article, Fig. 4).
//!
//! A zeroing step subtracts w P from the residual R, where all positions of the permutation matrix P have positive
//! entries in R and w is the smallest of them, so at least one entry becomes zero. A decomposition with a zeroing
//! order is a sequence of zeroing steps (Proposition 9), so it is a path in the search tree below.
//!
//! The root is D, and a node at depth t is a residual after t zeroing steps. Its successors:
//! * `Successors::AllPermutations` (n <= 7): the zeroing steps with all permutations in the nonzero pattern of the
//!   residual, taken from the list of permutations in the nonzero pattern of D. No zeroing step is missed.
//! * `Successors::PerPosition` (larger n): one successor for each position e with 0 < R_e <= theta. It uses a
//!   permutation that covers e, covers only positions with entries of at least R_e, and covers the largest number of
//!   positions with entry R_e.
//!
//! Successors with the same residual are merged. A successor is pruned if its depth plus the degree of its residual
//! (Lemma 1) is at least the length of the best known decomposition. The others are scored by their depth plus the
//! length of their greedy completion, the decomposition of the residual by the Birkhoff algorithm, and the `width`
//! successors with the lowest scores form the next level. Successors and completions are computed in parallel (rayon).

use crate::birkhoff_algorithm::{BirkhoffAlgorithm, Rule};
use crate::decomposition::{Decomposition, Perm};
use crate::threshold::ThresholdSets;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// All permutations in the nonzero pattern of `d` (row i goes to column perm[i]); None if there are more than `limit`.
pub fn pattern_permutations(d: &[i64], n: usize, limit: usize) -> Option<Vec<Perm>> {
    fn extend(
        d: &[i64],
        n: usize,
        i: usize,
        used: &mut [bool],
        current: &mut Perm,
        out: &mut Vec<Perm>,
        limit: usize,
    ) -> bool {
        if i == n {
            out.push(current.clone());
            return out.len() <= limit;
        }
        for j in 0..n {
            if !used[j] && d[i * n + j] > 0 {
                used[j] = true;
                current.push(j as u32);
                let ok = extend(d, n, i + 1, used, current, out, limit);
                current.pop();
                used[j] = false;
                if !ok {
                    return false;
                }
            }
        }
        true
    }
    let mut out = Vec::new();
    extend(d, n, 0, &mut vec![false; n], &mut Vec::with_capacity(n), &mut out, limit).then_some(out)
}

/// How the successors of a node are formed.
#[derive(Clone)]
pub enum Successors<'a> {
    /// The zeroing steps with every permutation of the list that fits the residual.
    AllPermutations(&'a [Perm]),
    /// One successor per position e with 0 < R_e <= theta; at most `max_successors` per node (most positions zeroed
    /// first, then the larger weight).
    PerPosition { max_successors: usize },
}

/// The degree of the residual `r`: the largest number of nonzero entries in a line.
fn degree_of(r: &[i64], n: usize) -> usize {
    let mut best = 0;
    for i in 0..n {
        best = best.max(r[i * n..(i + 1) * n].iter().filter(|&&v| v > 0).count());
        best = best.max((0..n).filter(|&k| r[k * n + i] > 0).count());
    }
    best
}

/// The successors of one node as (permutation index, weight). The index points into the list of
/// `Successors::AllPermutations` or into `perms`.
struct NodeSuccessors {
    perms: Vec<Perm>,
    list: Vec<(u32, i64)>,
}

impl NodeSuccessors {
    fn get<'a>(&'a self, successors: &'a Successors, index: usize) -> (&'a Perm, i64) {
        let (p, w) = self.list[index];
        match successors {
            Successors::AllPermutations(all) => (&all[p as usize], w),
            Successors::PerPosition { .. } => (&self.perms[p as usize], w),
        }
    }
}

fn node_successors(
    r: &[i64],
    n: usize,
    successors: &Successors,
    thresholds: &mut ThresholdSets,
    cost: &mut [i64],
) -> NodeSuccessors {
    match successors {
        Successors::AllPermutations(all) => NodeSuccessors {
            perms: Vec::new(),
            list: all
                .iter()
                .enumerate()
                .filter_map(|(index, p)| {
                    let w = p.iter().enumerate().map(|(i, &j)| r[i * n + j as usize]).min().unwrap();
                    (w > 0).then_some((index as u32, w))
                })
                .collect(),
        },
        Successors::PerPosition { max_successors } => {
            // one augmenting path from the bottleneck matching when the entry of the position is unique, an
            // assignment problem on the threshold set otherwise (`ThresholdSets::successor_through`)
            thresholds.reset_keys(r);
            let Some((bottleneck, theta, _)) = thresholds.bottleneck(r, None, None) else {
                return NodeSuccessors { perms: vec![], list: vec![] };
            };
            let mut out: Vec<(Perm, i64)> = Vec::new();
            let mut seen: std::collections::HashSet<Perm> = std::collections::HashSet::new();
            for e in 0..n * n {
                let lambda = r[e];
                if lambda == 0 || lambda > theta {
                    continue;
                }
                let (ei, ej) = (e / n, e % n);
                if let Some(p) = thresholds.successor_through(r, ei, ej, &bottleneck, cost) {
                    debug_assert!(p[ei] as usize == ej);
                    debug_assert_eq!(p.iter().enumerate().map(|(i, &j)| r[i * n + j as usize]).min(), Some(lambda));
                    if seen.insert(p.clone()) {
                        out.push((p, lambda));
                    }
                }
            }
            if out.is_empty() {
                out.push((bottleneck, theta));
            }
            if out.len() > *max_successors {
                let zeroed =
                    |(p, w): &(Perm, i64)| p.iter().enumerate().filter(|(i, &j)| r[i * n + j as usize] == *w).count();
                out.sort_by_cached_key(|step| (std::cmp::Reverse(zeroed(step)), std::cmp::Reverse(step.1)));
                out.truncate(*max_successors);
            }
            let list = out.iter().enumerate().map(|(index, (_, w))| (index as u32, *w)).collect();
            NodeSuccessors { perms: out.into_iter().map(|(p, _)| p).collect(), list }
        }
    }
}

/// A node of the search tree: the residual and the zeroing steps on the path to it.
struct Node {
    r: Vec<i64>,
    steps: Vec<(Perm, i64)>,
}

/// Greedy completion over an explicit permutation list, without matching problems: each step takes the permutation
/// of the list with the largest smallest entry theta on the residual and, among those, the most positions with entry
/// theta. This is the zeroing rule, exact when the list holds every permutation of the nonzero pattern. Returns the
/// steps as (permutation index, weight).
fn pool_completion(r: &mut [i64], n: usize, positions: &[u16]) -> Vec<(u32, i64)> {
    let mut steps = Vec::new();
    let mut nonzero = r.iter().filter(|&&v| v > 0).count();
    while nonzero > 0 {
        // (theta, number of zeroed positions, index of the permutation)
        let mut best = (0i64, 0usize, u32::MAX);
        for (index, perm_positions) in positions.chunks_exact(n).enumerate() {
            let theta = perm_positions.iter().map(|&e| r[e as usize]).min().unwrap();
            if theta > 0 && theta >= best.0 {
                let zeroed = perm_positions.iter().filter(|&&e| r[e as usize] == theta).count();
                if theta > best.0 || zeroed > best.1 {
                    best = (theta, zeroed, index as u32);
                }
            }
        }
        let (theta, _, index) = best;
        debug_assert!(index != u32::MAX, "the residual has no permutation in the list");
        if index == u32::MAX {
            return Vec::new();
        }
        for &e in &positions[index as usize * n..(index as usize + 1) * n] {
            r[e as usize] -= theta;
            if r[e as usize] == 0 {
                nonzero -= 1;
            }
        }
        steps.push((index, theta));
    }
    steps
}

/// Lists of up to this many permutations use `pool_completion`, which is faster there than the matching problems.
const POOL_COMPLETION_MAX: usize = 120;

/// Upper bound on the successors formed per level, so memory and time per level stay bounded at any width.
pub const MAX_SUCCESSORS: usize = 1_500_000;

/// The residual after the zeroing step (p, w).
fn apply(r: &[i64], n: usize, p: &Perm, w: i64) -> Vec<i64> {
    let mut out = r.to_vec();
    for (i, &j) in p.iter().enumerate() {
        out[i * n + j as usize] -= w;
    }
    out
}

/// 64-bit hash value of a residual. A collision only drops a successor, which is harmless.
fn residual_hash(r: &[i64]) -> u64 {
    r.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &v| (h.rotate_left(5) ^ v as u64).wrapping_mul(0x5172_7f4b_1a2d_3c8f))
}

/// Settings of one beam search.
pub struct Beam<'a> {
    pub successors: Successors<'a>,
    /// The width W: the number of nodes kept per level.
    pub width: usize,
    /// Seed of the random tie-break between equal scores.
    pub seed: u64,
    /// Length of the best known decomposition; only shorter ones are reported.
    pub best_length: usize,
    /// Rule of the greedy completions with `Successors::PerPosition` (the list completion of small n always uses the
    /// zeroing rule).
    pub completion_rule: Rule,
}

/// Runs the beam search until the tree is exhausted or `running()` turns false. `on_improve` receives every
/// decomposition that is shorter than the best one so far.
///
/// The successors of a level are never stored all at once. A first parallel pass computes, for every pair of a node
/// and a zeroing step, the hash value and the degree of the residual, merges equal residuals and prunes. A second pass
/// rebuilds the remaining residuals and computes the lengths of their greedy completions. Only the `width` best
/// successors are kept.
pub fn beam_search(
    d: &[i64],
    n: usize,
    beam: &Beam,
    running: &(dyn Fn() -> bool + Sync),
    on_improve: &mut dyn FnMut(&Decomposition),
) -> Option<Decomposition> {
    let (successors, width) = (&beam.successors, beam.width);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(beam.seed);
    let mut best: Option<Decomposition> = None;
    let mut limit = beam.best_length;
    let pool_positions: Option<Vec<u16>> = match successors {
        Successors::AllPermutations(all) if all.len() <= POOL_COMPLETION_MAX => {
            Some(all.iter().flat_map(|p| p.iter().enumerate().map(|(i, &j)| (i * n + j as usize) as u16)).collect())
        }
        _ => None,
    };
    // greedy completion: (number of steps, the decomposition of the residual if it would beat `limit`)
    let completion =
        |completer: &mut BirkhoffAlgorithm, r: &[i64], depth: usize, limit: usize| -> (usize, Option<Decomposition>) {
            match (&pool_positions, successors) {
                (Some(positions), Successors::AllPermutations(all)) => {
                    let mut residual = r.to_vec();
                    let steps = pool_completion(&mut residual, n, positions);
                    let tail = (depth + steps.len() < limit).then(|| {
                        let mut tail = Decomposition::default();
                        for (index, w) in &steps {
                            tail.push(all[*index as usize].clone(), *w);
                        }
                        tail
                    });
                    (steps.len(), tail)
                }
                _ => {
                    // count first; the decomposition is built only when it beats the limit
                    let len = completer.length(r, beam.completion_rule);
                    (len, (depth + len < limit).then(|| completer.decompose(r, beam.completion_rule, None, 0)))
                }
            }
        };
    let mut level = vec![Node { r: d.to_vec(), steps: Vec::new() }];
    while !level.is_empty() && running() {
        let expanded: Vec<NodeSuccessors> = level
            .par_iter()
            .map_init(
                || (ThresholdSets::new(n), vec![0i64; n * n]),
                |(thresholds, cost), node| {
                    if running() {
                        node_successors(&node.r, n, successors, thresholds, cost)
                    } else {
                        NodeSuccessors { perms: vec![], list: vec![] }
                    }
                },
            )
            .collect();
        // all successors of the level as (node, successor of the node)
        let mut candidates: Vec<(u32, u32)> = Vec::new();
        'collect: for (node, node_successors) in expanded.iter().enumerate() {
            for successor in 0..node_successors.list.len() {
                if candidates.len() >= MAX_SUCCESSORS {
                    break 'collect;
                }
                candidates.push((node as u32, successor as u32));
            }
        }
        let depth = level[0].steps.len() + 1;
        // pass 1: hash value and degree, merge equal residuals (parallel sort by hash, first occurrence kept), prune
        let mut keys: Vec<(u64, u32)> = candidates
            .par_iter()
            .enumerate()
            .filter_map(|(index, &(node, successor))| {
                let (p, w) = expanded[node as usize].get(successors, successor as usize);
                let r = apply(&level[node as usize].r, n, p, w);
                let keep = depth + degree_of(&r, n) < limit || r.iter().all(|&v| v == 0);
                keep.then(|| (residual_hash(&r), index as u32))
            })
            .collect();
        keys.par_sort_unstable();
        keys.dedup_by_key(|key| key.0);
        let alive: Vec<(u32, u32)> = keys.iter().map(|&(_, index)| candidates[index as usize]).collect();
        // pass 2: greedy completions
        let scored: Vec<(usize, usize, Option<Decomposition>)> = alive
            .par_iter()
            .map_init(
                || BirkhoffAlgorithm::new(n),
                |completer, &(node, successor)| {
                    let (p, w) = expanded[node as usize].get(successors, successor as usize);
                    let r = apply(&level[node as usize].r, n, p, w);
                    let nnz = r.iter().filter(|&&v| v > 0).count();
                    if nnz == 0 {
                        return (depth, 0, Some(Decomposition::default()));
                    }
                    if !running() {
                        return (usize::MAX, nnz, None);
                    }
                    let (len, tail) = completion(completer, &r, depth, limit);
                    (depth + len, nnz, tail)
                },
            )
            .collect();
        let mut order: Vec<(usize, usize, u32, usize)> = Vec::with_capacity(alive.len());
        for (index, (score, nnz, tail)) in scored.into_iter().enumerate() {
            if score == usize::MAX {
                continue;
            }
            // the path to the successor followed by its greedy completion is a full decomposition
            if let Some(tail) = tail.filter(|_| score < limit) {
                let (node, successor) = alive[index];
                let mut dec = Decomposition::default();
                let (p, w) = expanded[node as usize].get(successors, successor as usize);
                for (q, v) in &level[node as usize].steps {
                    dec.push(q.clone(), *v);
                }
                dec.push(p.clone(), w);
                for (p, &w) in tail.perms.iter().zip(&tail.weights) {
                    dec.push(p.clone(), w);
                }
                dec.normalize();
                if dec.len() < limit {
                    limit = dec.len();
                    on_improve(&dec);
                    best = Some(dec);
                }
            }
            if nnz > 0 {
                order.push((score, nnz, rng.gen(), index));
            }
        }
        // the best `width` successors (with slack for those pruned below), sorted by score
        let keep = width.saturating_mul(2).max(16);
        if order.len() > keep {
            order.select_nth_unstable(keep);
            order.truncate(keep);
        }
        order.sort_unstable();
        let mut next = Vec::with_capacity(width.min(order.len()));
        for (_, _, _, index) in order {
            if next.len() >= width {
                break;
            }
            let (node, successor) = alive[index];
            let (p, w) = expanded[node as usize].get(successors, successor as usize);
            let r = apply(&level[node as usize].r, n, p, w);
            // a successor can still beat the best decomposition only if its depth plus its degree is smaller
            if depth + degree_of(&r, n) >= limit {
                continue;
            }
            let mut steps = level[node as usize].steps.clone();
            steps.push((p.clone(), w));
            next.push(Node { r, steps });
        }
        level = next;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::Instance;
    use rand::seq::SliceRandom;

    fn random_instance(n: usize, k: usize, seed: u64) -> Instance {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut d = vec![0i64; n * n];
        let mut scale = 0;
        for _ in 0..k {
            let mut p: Vec<usize> = (0..n).collect();
            p.shuffle(&mut rng);
            let w = rng.gen_range(1..1000);
            scale += w;
            for (i, &j) in p.iter().enumerate() {
                d[i * n + j] += w;
            }
        }
        Instance { id: format!("T{n}_{k}_{seed}"), n, scale, d }
    }

    #[test]
    fn full_pattern_has_all_permutations() {
        let mut d = vec![1i64; 25];
        d[0] = 0; // one zero position removes the 4! permutations through it
        assert_eq!(pattern_permutations(&[1i64; 25], 5, 1000).unwrap().len(), 120);
        assert_eq!(pattern_permutations(&d, 5, 1000).unwrap().len(), 96);
        assert!(pattern_permutations(&d, 5, 95).is_none());
    }

    #[test]
    fn beam_search_returns_exact_decompositions() {
        for seed in 0..6 {
            let n = 4 + (seed as usize % 3);
            let inst = random_instance(n, n * n, seed);
            let all = pattern_permutations(&inst.d, n, 5040).unwrap();
            for successors in [
                Successors::AllPermutations(&all),
                Successors::PerPosition { max_successors: usize::MAX },
                Successors::PerPosition { max_successors: 3 },
            ] {
                let beam =
                    Beam { successors, width: 20, seed, best_length: usize::MAX, completion_rule: Rule::Zeroing };
                let mut seen = Vec::new();
                let best = beam_search(&inst.d, n, &beam, &|| true, &mut |d| seen.push(d.len()));
                let best = best.expect("some decomposition");
                best.verify(&inst).expect("exact");
                assert!(seen.windows(2).all(|w| w[1] < w[0]), "improvements must be strict");
                assert_eq!(*seen.last().unwrap(), best.len());
            }
        }
    }

    #[test]
    fn lower_bound_never_exceeds_optimum() {
        for seed in 0..20 {
            let n = 3 + (seed as usize % 2);
            let inst = random_instance(n, 2 + seed as usize % 7, 100 + seed);
            let lb = crate::bounds::two_line_bound(&inst, 1.0, &mut |_| true);
            let mut k = 1;
            let opt = loop {
                if let crate::sat::SatResult::Sat(d) = crate::sat::decompose_exactly(&inst.d, n, k, 30.0) {
                    d.verify(&inst).unwrap();
                    break d.len();
                }
                k += 1;
            };
            assert!(lb <= opt, "{}: lower bound {lb} > optimum {opt}", inst.id);
        }
    }
}
