//! Recover, second step: fixing by unique subset sums (Sect. 3.2 of the article, Theorem 2).
//!
//! Given the weights c_1 <= ... <= c_k, let F_i and G_j be the index sets of the weights that are not yet placed in
//! row i and in column j. Equal weights count as different indices. A position is open until we fix its set of
//! weights. We visit the open positions in increasing order of their entries. For a position (i, j), we list the
//! subsets of F_i ∩ G_j with sum D_ij and stop after two. If there is exactly one, we fix the position to it and remove
//! its indices from F_i and G_j. If there is none, the weights do not fit D. If there are two, or if the enumeration
//! exceeds its budget, the position stays open. We repeat these passes until no position changes, and a depth-first
//! search decides the positions that are still open. If the weights are correct, every fixed position is correct.

use crate::decomposition::Decomposition;
use crate::instance::Instance;

#[derive(Clone)]
struct State {
    /// version counters of the row and column sets, and the versions seen at the last (ambiguous) look at a position
    row_ver: Vec<u32>,
    col_ver: Vec<u32>,
    seen: Vec<(u32, u32, u64)>,
    /// F_i and G_j: the weights not yet placed per row and per column, as bit sets over the weight indices
    row: Vec<Vec<u64>>,
    col: Vec<Vec<u64>>,
    /// open positions (indices into `positions`)
    open: Vec<usize>,
    /// fixed subset (weight indices) per position
    choice: Vec<Vec<u32>>,
}

/// Fixing by unique subset sums for one instance and one list of weights.
pub struct Fixing<'a> {
    inst: &'a Instance,
    /// the weights, ascending
    weights: Vec<i64>,
    /// the positions with D_e > 0, in increasing order of their entries
    positions: Vec<usize>,
    /// 64-bit words per bit set
    words: usize,
    node_limit: u64,
    nodes: u64,
}

/// Result of fixing by unique subset sums.
pub enum FixingResult {
    /// a decomposition with the given weights
    Found(Decomposition),
    /// the weights do not fit D
    Inconsistent,
    /// the search reached its budget
    Limit,
}

impl<'a> Fixing<'a> {
    pub fn new(inst: &'a Instance, weights: &[i64]) -> Self {
        let mut sorted = weights.to_vec();
        sorted.sort_unstable();
        let n = inst.n;
        let mut positions: Vec<usize> = (0..n * n).filter(|&e| inst.d[e] > 0).collect();
        positions.sort_by_key(|&e| inst.d[e]);
        let words = sorted.len().div_ceil(64);
        Fixing { inst, weights: sorted, positions, words, node_limit: 4_096, nodes: 0 }
    }

    /// The subsets of `indices` (ascending weight indices) with sum `target`, at most `limit` of them; None if the
    /// enumeration exceeds its budget.
    fn subsets(&mut self, indices: &[u32], target: i64, limit: usize) -> Option<Vec<Vec<u32>>> {
        // suffix sums for pruning
        let mut suffix = vec![0i64; indices.len() + 1];
        for t in (0..indices.len()).rev() {
            suffix[t] = suffix[t + 1] + self.weights[indices[t] as usize];
        }
        let mut out = Vec::new();
        let mut current = Vec::new();
        self.nodes = 0; // budget per enumeration
        let ok = self.enumerate_subsets(indices, &suffix, 0, target, &mut current, &mut out, limit);
        ok.then_some(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn enumerate_subsets(
        &mut self,
        indices: &[u32],
        suffix: &[i64],
        start: usize,
        remaining: i64,
        current: &mut Vec<u32>,
        out: &mut Vec<Vec<u32>>,
        limit: usize,
    ) -> bool {
        self.nodes += 1;
        if self.nodes > self.node_limit {
            return false;
        }
        if remaining == 0 {
            out.push(current.clone());
            return true;
        }
        if suffix[start] < remaining {
            return true;
        }
        for t in start..indices.len() {
            let w = self.weights[indices[t] as usize];
            if w > remaining {
                break;
            }
            current.push(indices[t]);
            let ok = self.enumerate_subsets(indices, suffix, t + 1, remaining - w, current, out, limit);
            current.pop();
            if !ok {
                return false;
            }
            if out.len() >= limit {
                return true;
            }
        }
        true
    }

    /// The indices in F_i ∩ G_j of position e = (i, j) whose weight is at most D_e.
    fn candidates(&self, state: &State, e: usize) -> Vec<u32> {
        let n = self.inst.n;
        let v = self.inst.d[e];
        let (r, c) = (e / n, e % n);
        let mut out = Vec::new();
        for word in 0..self.words {
            let mut bits = state.row[r][word] & state.col[c][word];
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let l = word * 64 + b;
                if self.weights[l] > v {
                    return out; // the weights are ascending
                }
                out.push(l as u32);
            }
        }
        out
    }

    fn fix(&self, state: &mut State, index: usize, subset: Vec<u32>) {
        let n = self.inst.n;
        let e = self.positions[index];
        state.row_ver[e / n] += 1;
        state.col_ver[e % n] += 1;
        for &l in &subset {
            let (word, b) = (l as usize / 64, l as usize % 64);
            state.row[e / n][word] &= !(1u64 << b);
            state.col[e % n][word] &= !(1u64 << b);
        }
        state.choice[index] = subset;
    }

    /// Passes over the open positions until no position changes; Err if the weights do not fit D.
    fn propagate(&mut self, state: &mut State) -> Result<bool, ()> {
        let n = self.inst.n;
        let mut budget: u64 = 4_096;
        loop {
            let mut progress = false;
            let mut still_open = Vec::with_capacity(state.open.len());
            let open = std::mem::take(&mut state.open);
            self.node_limit = budget;
            for &index in &open {
                let e = self.positions[index];
                // unchanged sets and no larger budget than last time: still ambiguous
                let key = (state.row_ver[e / n], state.col_ver[e % n], budget);
                if state.seen[index] == key {
                    still_open.push(index);
                    continue;
                }
                state.seen[index] = key;
                let candidates = self.candidates(state, e);
                // too many partial sums to list now: the position waits until its sets shrink
                let Some(subsets) = self.subsets(&candidates, self.inst.d[e], 2) else {
                    still_open.push(index);
                    continue;
                };
                match subsets.len() {
                    0 => return Err(()),
                    1 => {
                        self.fix(state, index, subsets.into_iter().next().unwrap());
                        progress = true;
                    }
                    _ => still_open.push(index),
                }
            }
            state.open = still_open;
            if state.open.is_empty() {
                return Ok(true);
            }
            if !progress {
                // a larger enumeration budget before the depth-first search takes over
                if budget >= 1 << 20 {
                    return Ok(true);
                }
                budget *= 8;
            }
        }
    }

    /// Ok(Some): solved; Ok(None): the weights do not fit this branch; Err: the budget is reached.
    fn search(&mut self, mut state: State) -> Result<Option<State>, ()> {
        match self.propagate(&mut state) {
            Err(()) => return Ok(None),
            Ok(false) => return Err(()),
            Ok(true) => {}
        }
        if state.open.is_empty() {
            let full = state.row.iter().chain(&state.col).all(|b| b.iter().all(|&w| w == 0));
            return Ok(full.then_some(state));
        }
        // branch on the open position with the fewest subsets
        self.node_limit = 1 << 20;
        let mut best: Option<(usize, Vec<Vec<u32>>)> = None;
        for p in 0..state.open.len() {
            let e = self.positions[state.open[p]];
            let candidates = self.candidates(&state, e);
            let Some(subsets) = self.subsets(&candidates, self.inst.d[e], 16) else { continue };
            if best.as_ref().is_none_or(|b| subsets.len() < b.1.len()) {
                best = Some((p, subsets));
            }
        }
        // every open position has too many subsets to list
        let Some((p, subsets)) = best else { return Err(()) };
        let index = state.open[p];
        for subset in subsets {
            let mut child = state.clone();
            child.open.remove(p);
            self.fix(&mut child, index, subset);
            if let Some(solved) = self.search(child)? {
                return Ok(Some(solved));
            }
        }
        Ok(None)
    }

    /// Fixes the positions and decides the open ones by depth-first search.
    pub fn solve(&mut self) -> FixingResult {
        let n = self.inst.n;
        let k = self.weights.len();
        let mut full = vec![0u64; self.words];
        for l in 0..k {
            full[l / 64] |= 1 << (l % 64);
        }
        let state = State {
            row_ver: vec![0; n],
            col_ver: vec![0; n],
            seen: vec![(u32::MAX, u32::MAX, 0); self.positions.len()],
            row: vec![full.clone(); n],
            col: vec![full; n],
            open: (0..self.positions.len()).collect(),
            choice: vec![Vec::new(); self.positions.len()],
        };
        match self.search(state) {
            Err(()) => FixingResult::Limit,
            Ok(None) => FixingResult::Inconsistent,
            Ok(Some(solved)) => {
                let mut perms = vec![vec![u32::MAX; n]; k];
                for (index, subset) in solved.choice.iter().enumerate() {
                    let e = self.positions[index];
                    for &l in subset {
                        perms[l as usize][e / n] = (e % n) as u32;
                    }
                }
                let mut dec = Decomposition::default();
                for (l, p) in perms.into_iter().enumerate() {
                    dec.push(p, self.weights[l]);
                }
                dec.normalize();
                FixingResult::Found(dec)
            }
        }
    }
}
