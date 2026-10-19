//! Matching code of the Birkhoff algorithm on threshold sets (Sect. 4 of the article, paragraph "Matching").
//!
//! A step of the Birkhoff algorithm needs the bottleneck value theta, the largest smallest entry of R over all
//! permutations, and then an assignment problem on the threshold set {R >= theta}, the set of positions with entry at
//! least theta. Here:
//! * the positions with positive entries are kept as keys (value << 20 | position), sorted by value. After a step, only
//!   n positions change, and we merge them into the sorted list. So the threshold set of every threshold is a suffix of
//!   the keys;
//! * theta never increases during a run of the Birkhoff algorithm, so the search for theta starts at the previous
//!   value and moves down in doubling steps. Every test is a perfect matching problem on a suffix, solved by the
//!   algorithm of Hopcroft and Karp on a compressed adjacency list;
//! * the assignment problem is solved by successive shortest paths with a binary heap, on the threshold set only.
//!   Among columns at the same distance, the one with the smallest index is taken, and a column is relaxed only on a
//!   strictly smaller distance. This is the tie-break of the Hungarian method, so the zeroing rule chooses the same
//!   permutations as with the dense code in `matching`.

use crate::decomposition::Perm;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const NONE: u32 = u32::MAX;
const POS_BITS: u32 = 20;
const POS_MASK: u64 = (1 << POS_BITS) - 1;

#[inline]
fn key_value(k: u64) -> i64 {
    (k >> POS_BITS) as i64
}

#[inline]
fn key_position(k: u64) -> usize {
    (k & POS_MASK) as usize
}

/// Work space for repeated steps of the Birkhoff algorithm on n x n matrices (n^2 < 2^20, entries < 2^44).
pub struct ThresholdSets {
    n: usize,
    /// The positive entries as keys (value << 20 | position), sorted.
    keys: Vec<u64>,
    /// Second key buffer for the radix sort and the merge.
    buffer: Vec<u64>,
    digit_count: Vec<u32>,
    removed: Vec<u64>,
    inserted: Vec<u64>,
    // compressed adjacency list of a threshold set: the columns of row i are adj[start[i]..start[i + 1]]
    start: Vec<u32>,
    adj: Vec<u32>,
    row_fill: Vec<u32>,
    // Hopcroft and Karp: the matched column of every row and the matched row of every column
    mate_l: Vec<u32>,
    mate_r: Vec<u32>,
    dist: Vec<u32>,
    queue: Vec<u32>,
    iter: Vec<u32>,
    stack: Vec<u32>,
    best: Perm,
    // successive shortest paths: potentials of rows (u) and columns (v), distances and predecessors of the columns,
    // and round stamps instead of clearing the arrays for every row
    u: Vec<i64>,
    v: Vec<i64>,
    col_dist: Vec<i64>,
    way: Vec<u32>,
    stamp: Vec<u32>,
    visited: Vec<u32>,
    round: u32,
    visited_cols: Vec<u32>,
    heap: BinaryHeap<(Reverse<i64>, Reverse<u32>)>,
}

impl ThresholdSets {
    pub fn new(n: usize) -> Self {
        ThresholdSets {
            n,
            keys: Vec::with_capacity(n * n),
            buffer: Vec::with_capacity(n * n),
            digit_count: vec![0; 256],
            start: vec![0; n + 1],
            adj: Vec::with_capacity(n * n),
            row_fill: vec![0; n + 1],
            mate_l: vec![NONE; n],
            mate_r: vec![NONE; n],
            dist: vec![0; n],
            queue: Vec::with_capacity(n),
            iter: vec![0; n],
            stack: Vec::with_capacity(n),
            best: vec![NONE; n],
            u: vec![0; n],
            v: vec![0; n],
            col_dist: vec![0; n],
            way: vec![NONE; n],
            stamp: vec![0; n],
            visited: vec![0; n],
            round: 0,
            visited_cols: Vec::with_capacity(n),
            heap: BinaryHeap::with_capacity(n),
            removed: Vec::with_capacity(n),
            inserted: Vec::with_capacity(n),
        }
    }

    /// Sorts the keys of the positive entries of `r` (radix sort on 24 value bits for large inputs).
    pub fn reset_keys(&mut self, r: &[i64]) {
        self.keys.clear();
        for (e, &x) in r.iter().enumerate() {
            if x > 0 {
                debug_assert!(x < 1 << 44 && e < 1 << POS_BITS);
                self.keys.push(((x as u64) << POS_BITS) | e as u64);
            }
        }
        if self.keys.len() < 1024 || self.keys.iter().any(|&k| key_value(k) >= 1 << 24) {
            self.keys.sort_unstable();
            return;
        }
        let m = self.keys.len();
        self.buffer.clear();
        self.buffer.resize(m, 0);
        for pass in 0..3 {
            let shift = POS_BITS + 8 * pass;
            self.digit_count.iter_mut().for_each(|c| *c = 0);
            for &k in &self.keys {
                self.digit_count[((k >> shift) & 255) as usize] += 1;
            }
            let mut sum = 0u32;
            for c in self.digit_count.iter_mut() {
                let x = *c;
                *c = sum;
                sum += x;
            }
            for &k in &self.keys {
                let b = ((k >> shift) & 255) as usize;
                self.buffer[self.digit_count[b] as usize] = k;
                self.digit_count[b] += 1;
            }
            std::mem::swap(&mut self.keys, &mut self.buffer);
        }
    }

    /// Updates the keys after a step: the n positions of `perm` went from `r_new + w` to `r_new`.
    pub fn update_keys(&mut self, perm: &[u32], r_new: &[i64], w: i64) {
        let n = self.n;
        self.removed.clear();
        self.inserted.clear();
        for (i, &j) in perm.iter().enumerate() {
            let e = i * n + j as usize;
            self.removed.push((((r_new[e] + w) as u64) << POS_BITS) | e as u64);
            if r_new[e] > 0 {
                self.inserted.push(((r_new[e] as u64) << POS_BITS) | e as u64);
            }
        }
        self.removed.sort_unstable();
        self.inserted.sort_unstable();
        self.buffer.clear();
        let (mut a, mut b) = (0, 0);
        for &k in &self.keys {
            while b < self.inserted.len() && self.inserted[b] < k {
                self.buffer.push(self.inserted[b]);
                b += 1;
            }
            if a < self.removed.len() && self.removed[a] == k {
                a += 1;
                continue;
            }
            self.buffer.push(k);
        }
        self.buffer.extend_from_slice(&self.inserted[b..]);
        debug_assert_eq!(a, self.removed.len(), "a removed key was not present");
        std::mem::swap(&mut self.keys, &mut self.buffer);
    }

    /// First key position with value >= v.
    fn first_at_least(&self, v: i64) -> usize {
        self.keys.partition_point(|&k| key_value(k) < v)
    }

    /// Compressed adjacency list of the positions keys[pos..].
    fn build_from(&mut self, pos: usize) {
        let n = self.n;
        self.row_fill.iter_mut().for_each(|c| *c = 0);
        for &k in &self.keys[pos..] {
            self.row_fill[key_position(k) / n] += 1;
        }
        let mut sum = 0u32;
        for i in 0..n {
            self.start[i] = sum;
            sum += self.row_fill[i];
            self.row_fill[i] = self.start[i];
        }
        self.start[n] = sum;
        self.adj.clear();
        self.adj.resize(sum as usize, 0);
        for &k in &self.keys[pos..] {
            let e = key_position(k);
            let i = e / n;
            self.adj[self.row_fill[i] as usize] = (e % n) as u32;
            self.row_fill[i] += 1;
        }
    }

    fn set_mates(&mut self, perm: Option<&[u32]>) {
        self.mate_l.iter_mut().for_each(|m| *m = NONE);
        self.mate_r.iter_mut().for_each(|m| *m = NONE);
        if let Some(p) = perm {
            for (i, &j) in p.iter().enumerate() {
                if j != NONE && (j as usize) < self.n && self.mate_r[j as usize] == NONE {
                    self.mate_l[i] = j;
                    self.mate_r[j as usize] = i as u32;
                }
            }
        }
    }

    /// Maximum matching size inside {r >= t} on the current adjacency list, starting from the current mates.
    fn hopcroft_karp(&mut self, r: &[i64], t: i64) -> usize {
        let n = self.n;
        for i in 0..n {
            let j = self.mate_l[i];
            if j != NONE && r[i * n + j as usize] < t {
                self.mate_l[i] = NONE;
                self.mate_r[j as usize] = NONE;
            }
        }
        let mut size = self.mate_l.iter().filter(|&&m| m != NONE).count();
        loop {
            self.queue.clear();
            for i in 0..n {
                if self.mate_l[i] == NONE {
                    self.dist[i] = 0;
                    self.queue.push(i as u32);
                } else {
                    self.dist[i] = u32::MAX;
                }
            }
            let mut found = false;
            let mut head = 0;
            while head < self.queue.len() {
                let i = self.queue[head] as usize;
                head += 1;
                for e in self.start[i]..self.start[i + 1] {
                    let w = self.mate_r[self.adj[e as usize] as usize];
                    if w == NONE {
                        found = true;
                    } else if self.dist[w as usize] == u32::MAX {
                        self.dist[w as usize] = self.dist[i] + 1;
                        self.queue.push(w);
                    }
                }
            }
            if !found {
                return size;
            }
            for i in 0..n {
                self.iter[i] = self.start[i];
            }
            for i in 0..n {
                if self.mate_l[i] == NONE && self.augment(i) {
                    size += 1;
                }
            }
        }
    }

    fn augment(&mut self, root: usize) -> bool {
        self.stack.clear();
        self.stack.push(root as u32);
        while let Some(&top) = self.stack.last() {
            let i = top as usize;
            if self.iter[i] == self.start[i + 1] {
                self.dist[i] = u32::MAX;
                self.stack.pop();
                continue;
            }
            let j = self.adj[self.iter[i] as usize] as usize;
            self.iter[i] += 1;
            let w = self.mate_r[j];
            if w == NONE {
                let mut col = j as u32;
                while let Some(left) = self.stack.pop() {
                    let prev = self.mate_l[left as usize];
                    self.mate_l[left as usize] = col;
                    self.mate_r[col as usize] = left;
                    col = prev;
                }
                return true;
            }
            if self.dist[w as usize] == self.dist[i] + 1 {
                self.stack.push(w);
            }
        }
        false
    }

    /// Is there a perfect matching inside {r >= value(keys[pos])}? (`pos` must be the first key of its value.)
    fn probe(&mut self, r: &[i64], pos: usize) -> bool {
        self.build_from(pos);
        self.hopcroft_karp(r, key_value(self.keys[pos])) == self.n
    }

    /// Bottleneck perfect matching of `r` (the keys must be current). `warm` is a starting matching, and
    /// `previous_theta` an upper bound on theta (the theta of the previous step). Returns the permutation, theta and
    /// the key position of theta, or None if the positive entries have no perfect matching.
    pub fn bottleneck(
        &mut self,
        r: &[i64],
        warm: Option<&[u32]>,
        previous_theta: Option<i64>,
    ) -> Option<(Perm, i64, usize)> {
        let n = self.n;
        if self.keys.is_empty() {
            return None;
        }
        self.set_mates(warm);
        // upper bound on theta: the previous theta, and the largest entry of every row
        let mut upper = previous_theta.unwrap_or(i64::MAX);
        for i in 0..n {
            upper = upper.min(r[i * n..(i + 1) * n].iter().copied().max().unwrap());
        }
        let hi = self.keys.partition_point(|&k| key_value(k) <= upper);
        if hi == 0 {
            return None;
        }
        // gallop down from the upper bound until a feasible threshold is found
        let mut bad = self.first_at_least(key_value(self.keys[hi - 1]));
        let mut good;
        if self.probe(r, bad) {
            self.best.copy_from_slice(&self.mate_l);
            return Some((self.best.clone(), key_value(self.keys[bad]), bad));
        }
        let mut step = 1usize;
        loop {
            if bad == 0 {
                return None;
            }
            let p = self.first_at_least(key_value(self.keys[bad.saturating_sub(step)]));
            if self.probe(r, p) {
                good = p;
                self.best.copy_from_slice(&self.mate_l);
                break;
            }
            bad = p;
            step *= 2;
        }
        // binary search over the distinct values strictly between good and bad
        loop {
            let above = self.keys.partition_point(|&k| key_value(k) <= key_value(self.keys[good]));
            if above >= bad {
                break;
            }
            let mid = self.first_at_least(key_value(self.keys[(above + bad) / 2]));
            if self.probe(r, mid) {
                good = mid;
                self.best.copy_from_slice(&self.mate_l);
            } else {
                bad = mid;
            }
        }
        Some((self.best.clone(), key_value(self.keys[good]), good))
    }

    /// Positions of the threshold set {r >= value(keys[pos])}, for filling a cost array.
    pub fn threshold_set(&self, pos: usize) -> impl Iterator<Item = usize> + '_ {
        self.keys[pos..].iter().map(|&k| key_position(k))
    }

    /// Min-cost perfect matching inside the threshold set {r >= value(keys[pos])} with the nonnegative costs `cost`
    /// (read only on the threshold set). Successive shortest paths from an empty matching, rows in order, Dijkstra
    /// with a heap ordered by (distance, column).
    pub fn assign(&mut self, pos: usize, cost: &[i64]) -> Option<Perm> {
        self.build_from(pos);
        self.successive_shortest_paths(cost, None)
    }

    /// Number of positions with entry exactly `v`.
    pub fn count_equal(&self, v: i64) -> usize {
        self.keys.partition_point(|&k| key_value(k) <= v) - self.first_at_least(v)
    }

    /// The permutation of the beam-search successor that belongs to position (ei, ej): a perfect matching through (ei,
    /// ej) inside {r >= lambda}, lambda = r[ei][ej], that covers the largest number of positions with entry lambda.
    /// `base` is a perfect matching inside {r >= theta}, theta >= lambda (the bottleneck matching of the node). If
    /// lambda occurs at no other position, every such matching zeroes exactly one position, and `base` is rerouted
    /// through (ei, ej) along one augmenting path. Otherwise an assignment problem on the threshold set without the row
    /// and the column of (ei, ej) is solved; `cost` is a work array of n^2 entries.
    pub fn successor_through(
        &mut self,
        r: &[i64],
        ei: usize,
        ej: usize,
        base: &[u32],
        cost: &mut [i64],
    ) -> Option<Perm> {
        let n = self.n;
        let lambda = r[ei * n + ej];
        let pos = self.first_at_least(lambda);
        self.build_from(pos);
        if self.count_equal(lambda) > 1 {
            for e in self.keys[pos..].iter().map(|&k| key_position(k)) {
                cost[e] = if r[e] == lambda { 0 } else { 1000 };
            }
            let mut perm = self.successive_shortest_paths(cost, Some((ei, ej)))?;
            perm[ei] = ej as u32;
            return Some(perm);
        }
        self.set_mates(Some(base));
        let old_col = self.mate_l[ei];
        if old_col as usize == ej {
            return Some(base.to_vec());
        }
        let old_row = self.mate_r[ej];
        self.mate_l[ei] = ej as u32;
        self.mate_r[ej] = ei as u32;
        self.mate_l[old_row as usize] = NONE;
        self.mate_r[old_col as usize] = NONE;
        // one augmenting path from the freed row old_row to the freed column old_col, never through column ej (row ei
        // is matched to ej and is therefore never entered)
        self.dist.iter_mut().for_each(|x| *x = 0);
        self.stack.clear();
        self.stack.push(old_row);
        self.iter[old_row as usize] = self.start[old_row as usize];
        self.dist[old_row as usize] = 1;
        while let Some(&top) = self.stack.last() {
            let i = top as usize;
            if self.iter[i] == self.start[i + 1] {
                self.stack.pop();
                continue;
            }
            let j = self.adj[self.iter[i] as usize] as usize;
            self.iter[i] += 1;
            if j == ej {
                continue;
            }
            let w = self.mate_r[j];
            if w == NONE {
                let mut col = j as u32;
                while let Some(left) = self.stack.pop() {
                    let prev = self.mate_l[left as usize];
                    self.mate_l[left as usize] = col;
                    self.mate_r[col as usize] = left;
                    col = prev;
                }
                return Some(self.mate_l.clone());
            }
            if self.dist[w as usize] == 0 {
                self.dist[w as usize] = 1;
                self.iter[w as usize] = self.start[w as usize];
                self.stack.push(w);
            }
        }
        None
    }

    /// Successive shortest paths on the current adjacency list; `skip`: a row and a column left out (both stay
    /// unmatched).
    fn successive_shortest_paths(&mut self, cost: &[i64], skip: Option<(usize, usize)>) -> Option<Perm> {
        let n = self.n;
        let (skip_i, skip_j) = skip.unwrap_or((usize::MAX, usize::MAX));
        self.u.iter_mut().for_each(|x| *x = 0);
        self.v.iter_mut().for_each(|x| *x = 0);
        self.mate_l.iter_mut().for_each(|x| *x = NONE);
        self.mate_r.iter_mut().for_each(|x| *x = NONE);
        for i0 in 0..n {
            if i0 == skip_i {
                continue;
            }
            self.round = self.round.wrapping_add(1);
            if self.round == 0 {
                self.stamp.iter_mut().for_each(|x| *x = 0);
                self.visited.iter_mut().for_each(|x| *x = 0);
                self.round = 1;
            }
            let round = self.round;
            self.visited_cols.clear();
            self.heap.clear();
            for e in self.start[i0]..self.start[i0 + 1] {
                let j = self.adj[e as usize] as usize;
                if j == skip_j {
                    continue;
                }
                let c = cost[i0 * n + j] - self.u[i0] - self.v[j];
                debug_assert!(c >= 0);
                if self.stamp[j] != round || c < self.col_dist[j] {
                    self.col_dist[j] = c;
                    self.stamp[j] = round;
                    self.way[j] = NONE;
                    self.heap.push((Reverse(c), Reverse(j as u32)));
                }
            }
            let mut found: Option<(usize, i64)> = None;
            while let Some((Reverse(d), Reverse(j))) = self.heap.pop() {
                let j = j as usize;
                if self.visited[j] == round || self.col_dist[j] != d {
                    continue;
                }
                self.visited[j] = round;
                self.visited_cols.push(j as u32);
                let i = self.mate_r[j];
                if i == NONE {
                    found = Some((j, d));
                    break;
                }
                let i = i as usize;
                for e in self.start[i]..self.start[i + 1] {
                    let j2 = self.adj[e as usize] as usize;
                    if j2 == skip_j || self.visited[j2] == round {
                        continue;
                    }
                    let new_dist = d + cost[i * n + j2] - self.u[i] - self.v[j2];
                    if self.stamp[j2] != round || new_dist < self.col_dist[j2] {
                        self.col_dist[j2] = new_dist;
                        self.stamp[j2] = round;
                        self.way[j2] = j as u32;
                        self.heap.push((Reverse(new_dist), Reverse(j2 as u32)));
                    }
                }
            }
            let (free_col, path_length) = found?;
            for &j in &self.visited_cols {
                let j = j as usize;
                let delta = path_length - self.col_dist[j];
                self.v[j] -= delta;
                let i = self.mate_r[j];
                if i != NONE {
                    self.u[i as usize] += delta;
                }
            }
            self.u[i0] += path_length;
            let mut j = free_col;
            loop {
                let prev_col = self.way[j];
                let i = if prev_col == NONE { i0 } else { self.mate_r[prev_col as usize] as usize };
                self.mate_r[j] = i as u32;
                self.mate_l[i] = j as u32;
                if prev_col == NONE {
                    break;
                }
                j = prev_col as usize;
            }
        }
        Some(self.mate_l.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::{Matcher, FORBIDDEN};
    use rand::{Rng, SeedableRng};

    /// The threshold sets give the same bottleneck value and an assignment of the same cost as the dense matcher.
    #[test]
    fn agrees_with_dense() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(5);
        for n in [3usize, 5, 8, 12] {
            for _ in 0..30 {
                // a random matrix with equal line sums: a sum of random weighted permutations
                let mut r = vec![0i64; n * n];
                for _ in 0..n {
                    let mut p: Vec<usize> = (0..n).collect();
                    for i in (1..n).rev() {
                        p.swap(i, rng.gen_range(0..=i));
                    }
                    let w = rng.gen_range(1..50);
                    for (i, &j) in p.iter().enumerate() {
                        r[i * n + j] += w;
                    }
                }
                let mut dense = Matcher::new(n);
                let (_, t_dense) = dense.bottleneck(&r, n, None).unwrap();
                let mut sets = ThresholdSets::new(n);
                sets.reset_keys(&r);
                let (_, t, pos) = sets.bottleneck(&r, None, None).unwrap();
                assert_eq!(t, t_dense);
                let mut cost = vec![0i64; n * n];
                let mut dense_cost = vec![FORBIDDEN; n * n];
                for e in sets.threshold_set(pos).collect::<Vec<_>>() {
                    cost[e] = if r[e] == t { 0 } else { rng.gen_range(1..1000) };
                    dense_cost[e] = cost[e];
                }
                let p = sets.assign(pos, &cost).unwrap();
                let (_, c_dense) = dense.min_cost(&dense_cost, n).unwrap();
                let c: i64 = p.iter().enumerate().map(|(i, &j)| cost[i * n + j as usize]).sum();
                assert_eq!(c, c_dense);
            }
        }
    }
}
