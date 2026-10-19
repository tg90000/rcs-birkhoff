//! Dense matching code for n x n matrices. The buffers are reused, so repeated calls do not allocate:
//! * the algorithm of Hopcroft and Karp on a compressed adjacency list;
//! * bottleneck perfect matching (the largest possible smallest entry): binary search over the distinct values, each
//!   test started from the previous matching;
//! * the Hungarian method (shortest augmenting paths with potentials, O(n^3)) on exact integer costs.
//!
//! The Birkhoff algorithm with the zeroing rule uses the faster code in `threshold`. This code serves the other rules,
//! the case split of the SAT certificates and the tests.

use crate::decomposition::Perm;

const NONE: u32 = u32::MAX;

/// Cost that marks a forbidden position in [`Matcher::min_cost`]. Large enough to dominate any sum of real costs
/// (|cost| <= 2^40 per position, n <= 2^10), small enough that n * FORBIDDEN cannot overflow i64.
pub const FORBIDDEN: i64 = 1 << 50;

/// Work space of the matching code for n x n matrices.
#[derive(Default)]
pub struct Matcher {
    n: usize,
    // compressed adjacency list: the columns of row i are adj[start[i]..start[i + 1]]
    start: Vec<u32>,
    adj: Vec<u32>,
    // Hopcroft and Karp: the matched column of every row and the matched row of every column
    mate_l: Vec<u32>,
    mate_r: Vec<u32>,
    dist: Vec<u32>,
    queue: Vec<u32>,
    iter: Vec<u32>,
    stack: Vec<u32>,
    // the distinct positive entries, for the binary search of the bottleneck value
    values: Vec<i64>,
    // Hungarian method, 1-based: potentials of rows (u) and columns (v), the row matched to every column (p), and the
    // predecessors (way), smallest reduced costs (minv) and visited marks of the columns
    u: Vec<i64>,
    v: Vec<i64>,
    p: Vec<u32>,
    way: Vec<u32>,
    minv: Vec<i64>,
    used: Vec<bool>,
}

impl Matcher {
    pub fn new(n: usize) -> Self {
        let mut m = Matcher::default();
        m.resize(n);
        m
    }

    fn resize(&mut self, n: usize) {
        if self.n == n {
            return;
        }
        self.n = n;
        self.start = vec![0; n + 1];
        self.adj = Vec::with_capacity(n * n);
        self.mate_l = vec![NONE; n];
        self.mate_r = vec![NONE; n];
        self.dist = vec![0; n];
        self.queue = Vec::with_capacity(n);
        self.iter = vec![0; n];
        self.stack = Vec::with_capacity(n);
        self.u = vec![0; n + 1];
        self.v = vec![0; n + 1];
        self.p = vec![0; n + 1];
        self.way = vec![0; n + 1];
        self.minv = vec![0; n + 1];
        self.used = vec![false; n + 1];
    }

    /// Builds the adjacency of the positions with r >= t (t >= 1).
    fn build(&mut self, r: &[i64], t: i64) {
        let n = self.n;
        self.adj.clear();
        for i in 0..n {
            self.start[i] = self.adj.len() as u32;
            for (j, &x) in r[i * n..(i + 1) * n].iter().enumerate() {
                if x >= t {
                    self.adj.push(j as u32);
                }
            }
        }
        self.start[n] = self.adj.len() as u32;
    }

    /// Maximum matching on the current adjacency, starting from the current mates (edges not in the adjacency are
    /// dropped first). Returns its size.
    fn hopcroft_karp(&mut self, r: &[i64], t: i64) -> usize {
        let n = self.n;
        // keep only warm-start edges that are still allowed
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

    /// Depth-first search for an augmenting path from the free row `root` along the layers of the breadth-first search.
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
                // the path is on the stack: match along it
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

    /// Perfect matching inside {r >= t}, warm-started from the current mates.
    fn perfect(&mut self, r: &[i64], t: i64) -> bool {
        self.build(r, t);
        self.hopcroft_karp(r, t) == self.n
    }

    fn current_perm(&self) -> Perm {
        self.mate_l.clone()
    }

    /// Bottleneck perfect matching on the positive entries of the matrix `r` (row by row): a permutation that maximizes
    /// the smallest entry. `warm`, for example the permutation of the previous step, is the starting matching. Returns
    /// the permutation and its bottleneck value, or None if the positive pattern has no perfect matching.
    pub fn bottleneck(&mut self, r: &[i64], n: usize, warm: Option<&[u32]>) -> Option<(Perm, i64)> {
        self.resize(n);
        self.values.clear();
        self.values.extend(r.iter().copied().filter(|&v| v > 0));
        self.values.sort_unstable();
        self.values.dedup();
        if self.values.is_empty() {
            return None;
        }
        self.set_mates(warm);
        if !self.perfect(r, self.values[0]) {
            return None;
        }
        let mut best = self.current_perm();
        let (mut lo, mut hi) = (0usize, self.values.len() - 1);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            let t = self.values[mid];
            if self.perfect(r, t) {
                best = self.current_perm();
                lo = mid;
            } else {
                // restore a valid matching for the next (lower) probe
                self.set_mates(Some(&best));
                hi = mid - 1;
            }
        }
        let b = best.iter().enumerate().map(|(i, &j)| r[i * n + j as usize]).min().unwrap();
        Some((best, b))
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

    /// Cheapest perfect assignment for the integer cost matrix (row by row, n x n), by the Hungarian method. Positions
    /// with cost >= FORBIDDEN are not allowed; returns None if every perfect assignment uses one.
    pub fn min_cost(&mut self, cost: &[i64], n: usize) -> Option<(Perm, i64)> {
        self.resize(n);
        let inf = i64::MAX / 4;
        self.u.iter_mut().for_each(|x| *x = 0);
        self.v.iter_mut().for_each(|x| *x = 0);
        self.p.iter_mut().for_each(|x| *x = 0);
        for i in 1..=n {
            self.p[0] = i as u32;
            let mut j0 = 0usize;
            self.minv.iter_mut().for_each(|m| *m = inf);
            self.used.iter_mut().for_each(|x| *x = false);
            loop {
                self.used[j0] = true;
                let i0 = self.p[j0] as usize;
                let mut delta = inf;
                let mut j1 = 0usize;
                let row = &cost[(i0 - 1) * n..i0 * n];
                for j in 1..=n {
                    if !self.used[j] {
                        let cur = row[j - 1] - self.u[i0] - self.v[j];
                        if cur < self.minv[j] {
                            self.minv[j] = cur;
                            self.way[j] = j0 as u32;
                        }
                        if self.minv[j] < delta {
                            delta = self.minv[j];
                            j1 = j;
                        }
                    }
                }
                for j in 0..=n {
                    if self.used[j] {
                        self.u[self.p[j] as usize] += delta;
                        self.v[j] -= delta;
                    } else {
                        self.minv[j] -= delta;
                    }
                }
                j0 = j1;
                if self.p[j0] == 0 {
                    break;
                }
            }
            loop {
                let j1 = self.way[j0] as usize;
                self.p[j0] = self.p[j1];
                j0 = j1;
                if j0 == 0 {
                    break;
                }
            }
        }
        let mut perm = vec![0u32; n];
        for j in 1..=n {
            perm[self.p[j] as usize - 1] = (j - 1) as u32;
        }
        let mut total = 0i64;
        for (i, &j) in perm.iter().enumerate() {
            let c = cost[i * n + j as usize];
            if c >= FORBIDDEN {
                return None;
            }
            total += c;
        }
        Some((perm, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn permute(p: &mut Vec<usize>, k: usize, f: &mut impl FnMut(&[usize])) {
        if k == p.len() {
            f(p);
            return;
        }
        for i in k..p.len() {
            p.swap(k, i);
            permute(p, k + 1, f);
            p.swap(k, i);
        }
    }

    #[test]
    fn min_cost_matches_brute_force() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
        let mut m = Matcher::default();
        for n in 1..=6 {
            for _ in 0..30 {
                let cost: Vec<i64> =
                    (0..n * n).map(|_| if rng.gen_bool(0.2) { FORBIDDEN } else { rng.gen_range(-20..20) }).collect();
                let got = m.min_cost(&cost, n).map(|x| x.1);
                let mut perm: Vec<usize> = (0..n).collect();
                let mut brute: Option<i64> = None;
                permute(&mut perm, 0, &mut |p| {
                    if p.iter().enumerate().all(|(i, &j)| cost[i * n + j] < FORBIDDEN) {
                        let c: i64 = (0..n).map(|i| cost[i * n + p[i]]).sum();
                        brute = Some(brute.map_or(c, |b: i64| b.min(c)));
                    }
                });
                assert_eq!(got, brute);
            }
        }
    }

    #[test]
    fn bottleneck_matches_brute_force() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(2);
        let mut m = Matcher::default();
        for n in 1..=6 {
            for _ in 0..30 {
                let r: Vec<i64> = (0..n * n).map(|_| if rng.gen_bool(0.3) { 0 } else { rng.gen_range(1..9) }).collect();
                let got = m.bottleneck(&r, n, None).map(|x| x.1);
                let mut perm: Vec<usize> = (0..n).collect();
                let mut brute: Option<i64> = None;
                permute(&mut perm, 0, &mut |p| {
                    let b = (0..n).map(|i| r[i * n + p[i]]).min().unwrap();
                    if b > 0 {
                        brute = Some(brute.map_or(b, |x: i64| x.max(b)));
                    }
                });
                assert_eq!(got, brute);
            }
        }
    }
}
