//! Upper bounds on the block number gamma* of the unmatched values of two lines (Sect. 3.3 of the article,
//! Propositions 5 and 6).
//!
//! The unmatched values are the values of the two lines that remain after equal values have been paired. We give the
//! values of the first line the sign + and those of the second line the sign -, so a block is a set with sum zero.
//! The two-line bound in `bounds` is k* >= p + q - M_2 - gamma*, so every method here only has to give a value that
//! is at least gamma* (the first two give gamma* exactly).
//!
//! * `exact_block_number` (m <= 22 values): gamma* is the largest number of zero-sum prefixes over all orderings,
//!   computed by a dynamic program over subsets in O(2^m m).
//! * `exact_block_number_mitm` (m <= 44): all zero-sum subsets by a meet-in-the-middle method, then the recursion
//!   g(R) = max(1, max{1 + g(R minus Z) : Z zero-sum, Z contains the element of R with the smallest index, Z != R}).
//! * `packing_bound` (any m): every block has at least three values. Blocks with at most T values are among the listed
//!   small zero-sum sets, and all other blocks have at least T + 1 values. So gamma* <= m / (T + 1) plus the best
//!   packing of small blocks with weight 1 - |Z| / (T + 1), and every feasible solution of the dual linear program
//!   bounds this packing. The dual is computed in floating point, rounded up, and checked against every block in
//!   integer arithmetic before the floor is taken, so rounding can only make the bound weaker, never wrong.

use std::collections::HashMap;
use std::time::Instant;

/// The exact block number gamma* by the dynamic program over zero-sum prefixes (Proposition 5, part 1). `v` holds the
/// signed unmatched values (total 0, m <= 26); the buffers are reused.
pub fn exact_block_number(v: &[i64], dp: &mut Vec<u8>, sums: &mut Vec<i64>) -> usize {
    let m = v.len();
    if m == 0 {
        return 0;
    }
    let size = 1usize << m;
    dp.clear();
    dp.resize(size, 0);
    sums.clear();
    sums.resize(size, 0);
    for mask in 1..size {
        let low = mask.trailing_zeros() as usize;
        sums[mask] = sums[mask & (mask - 1)] + v[low];
        let mut best = 0u8;
        let mut rest = mask;
        while rest != 0 {
            let i = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            best = best.max(dp[mask ^ (1 << i)]);
        }
        dp[mask] = best + (sums[mask] == 0) as u8;
    }
    dp[size - 1] as usize
}

/// The exact block number gamma* by listing all zero-sum subsets (meet in the middle) and the recursion over the
/// element with the smallest index (Proposition 5, part 2). None if there are more than `max_subsets` zero-sum
/// subsets, if m > 44, or if the deadline passes.
pub fn exact_block_number_mitm(v: &[i64], max_subsets: usize, deadline: Instant) -> Option<usize> {
    let m = v.len();
    if m == 0 {
        return Some(0);
    }
    if m > 44 {
        return None;
    }
    let split = m / 2;
    let half = |vals: &[i64]| -> Vec<(i64, u32)> {
        let mut out = vec![(0i64, 0u32); 1 << vals.len()];
        for mask in 1..out.len() {
            let low = mask.trailing_zeros() as usize;
            out[mask] = (out[mask & (mask - 1)].0 + vals[low], mask as u32);
        }
        out.sort_unstable();
        out
    };
    let left = half(&v[..split]);
    let right = half(&v[split..]);
    if Instant::now() > deadline {
        return None;
    }
    // zero-sum subsets: left sum s with right sum -s (two pointers: left ascending, right descending)
    let mut zero: Vec<u64> = Vec::new();
    let mut j = right.len();
    let mut i = 0;
    while i < left.len() {
        let s = left[i].0;
        let mut i_end = i;
        while i_end < left.len() && left[i_end].0 == s {
            i_end += 1;
        }
        while j > 0 && right[j - 1].0 > -s {
            j -= 1;
        }
        let mut j_start = j;
        while j_start > 0 && right[j_start - 1].0 == -s {
            j_start -= 1;
        }
        for l in &left[i..i_end] {
            for r in &right[j_start..j] {
                let mask = l.1 as u64 | ((r.1 as u64) << split);
                if mask != 0 {
                    zero.push(mask);
                    if zero.len() > max_subsets {
                        return None;
                    }
                }
            }
        }
        i = i_end;
    }
    drop(left);
    drop(right);
    let mut by_low: Vec<Vec<u64>> = vec![Vec::new(); m];
    for &z in &zero {
        by_low[z.trailing_zeros() as usize].push(z);
    }
    for blocks in &mut by_low {
        // small blocks first: they leave the most room for further blocks
        blocks.sort_unstable_by_key(|z| z.count_ones());
    }
    /// The largest number of blocks into which the set `remaining` (a mask) splits; every block contains the element
    /// of `remaining` with the smallest index.
    fn max_blocks(
        remaining: u64,
        by_low: &[Vec<u64>],
        memo: &mut HashMap<u64, u8>,
        steps: &mut u64,
        deadline: Instant,
    ) -> Option<u8> {
        if remaining == 0 {
            return Some(0);
        }
        if let Some(&known) = memo.get(&remaining) {
            return Some(known);
        }
        *steps += 1;
        if *steps % 4096 == 0 && Instant::now() > deadline {
            return None;
        }
        let low = remaining.trailing_zeros() as usize;
        let mut best = 1u8;
        for &block in &by_low[low] {
            if block & !remaining != 0 || block == remaining {
                continue;
            }
            let rest = remaining ^ block;
            // at most |rest| / 3 further blocks
            if ((rest.count_ones() / 3) as u8) < best {
                continue;
            }
            best = best.max(1 + max_blocks(rest, by_low, memo, steps, deadline)?);
        }
        memo.insert(remaining, best);
        Some(best)
    }
    let all = (1u64 << m) - 1;
    let mut memo: HashMap<u64, u8> = HashMap::new();
    let mut steps = 0u64;
    max_blocks(all, &by_low, &mut memo, &mut steps, deadline).map(|x| x as usize)
}

/// Subsets of `vals` with 1..=max_size elements: per size, (sum, mask) sorted by sum. Size t + 1 extends every size-t
/// subset by one element above its highest one. None if there are more than `cap` subsets in total.
fn subset_sums(vals: &[i64], max_size: usize, cap: usize) -> Option<Vec<Vec<(i64, u128)>>> {
    let mut out: Vec<Vec<(i64, u128)>> = vec![Vec::new(); max_size + 1];
    out[0].push((0, 0));
    let mut total = 0usize;
    for size in 0..max_size {
        let mut next = Vec::new();
        for &(sum, mask) in &out[size] {
            let start = if mask == 0 { 0 } else { 128 - mask.leading_zeros() as usize };
            for (i, &v) in vals.iter().enumerate().skip(start) {
                next.push((sum + v, mask | (1u128 << i)));
            }
        }
        total += next.len();
        if total > cap {
            return None;
        }
        out[size + 1] = next;
    }
    out[0].clear();
    for l in &mut out {
        l.sort_unstable_by_key(|x| x.0);
    }
    Some(out)
}

/// Dense primal simplex (Bland's rule) for max c^T y, A y <= 1, y >= 0 with A a 0/1 matrix given by its column
/// supports over `rows` rows. Returns the duals of the row constraints (floating point), or None on the iteration cap.
fn packing_lp_duals(rows: usize, cols: &[Vec<usize>], c: &[f64], deadline: Instant) -> Option<Vec<f64>> {
    let columns = cols.len() + rows; // structural and slack columns
    let width = columns + 1; // and the right-hand side
    let mut tableau = vec![0.0f64; rows * width];
    for (j, col) in cols.iter().enumerate() {
        for &i in col {
            tableau[i * width + j] = 1.0;
        }
    }
    for i in 0..rows {
        tableau[i * width + cols.len() + i] = 1.0;
        tableau[i * width + columns] = 1.0;
    }
    let mut objective = vec![0.0f64; width];
    for (j, &c_j) in c.iter().enumerate() {
        objective[j] = -c_j;
    }
    let mut basis: Vec<usize> = (0..rows).map(|i| cols.len() + i).collect();
    let eps = 1e-9;
    for iter in 0..200_000usize {
        if iter % 64 == 0 && Instant::now() > deadline {
            return None;
        }
        let Some(enter) = (0..columns).find(|&j| objective[j] < -eps) else { break };
        let mut leave: Option<usize> = None;
        let mut best = f64::INFINITY;
        for i in 0..rows {
            let a = tableau[i * width + enter];
            if a > eps {
                let ratio = tableau[i * width + columns] / a;
                if ratio < best - eps || (ratio <= best + eps && leave.is_none_or(|l| basis[i] < basis[l])) {
                    best = ratio;
                    leave = Some(i);
                }
            }
        }
        let r = leave?; // the problem is bounded, since 0 <= y <= 1
        let pivot = tableau[r * width + enter];
        for x in &mut tableau[r * width..(r + 1) * width] {
            *x /= pivot;
        }
        let (pivot_start, pivot_end) = (r * width, (r + 1) * width);
        let pivot_row: Vec<f64> = tableau[pivot_start..pivot_end].to_vec();
        for i in 0..rows {
            if i == r {
                continue;
            }
            let f = tableau[i * width + enter];
            if f.abs() > 0.0 {
                for (x, p) in tableau[i * width..(i + 1) * width].iter_mut().zip(&pivot_row) {
                    *x -= f * p;
                }
            }
        }
        let f = objective[enter];
        for (x, p) in objective.iter_mut().zip(&pivot_row) {
            *x -= f * p;
        }
        basis[r] = enter;
        if iter == 199_999 {
            return None;
        }
    }
    Some((0..rows).map(|i| objective[cols.len() + i]).collect())
}

/// The packing bound with blocks of three to `t` values (Proposition 6): an upper bound on gamma*(a | b), where `a`
/// are the unmatched values of the first line and `b` those of the second (both positive). None if the enumeration
/// exceeds `max_blocks`, the subset tables exceed their cap, or the deadline passes.
pub fn packing_bound(a: &[i64], b: &[i64], t: usize, max_blocks: usize, deadline: Instant) -> Option<usize> {
    let m = a.len() + b.len();
    if m == 0 {
        return Some(0);
    }
    if a.len() > 128 || b.len() > 128 || t < 3 {
        return None;
    }
    let sums_a = subset_sums(a, t - 1, 6_000_000)?;
    let sums_b = subset_sums(b, t - 1, 6_000_000)?;
    // blocks {A | B}: |A| + |B| <= t, sum A = sum B (two pointers per size pair)
    let mut blocks: Vec<(u128, u128, usize)> = Vec::new();
    for (size_a, list_a) in sums_a.iter().enumerate().take(t).skip(1) {
        for (size_b, list_b) in sums_b.iter().enumerate().take(t - size_a + 1).skip(1) {
            let (mut i, mut j) = (0, 0);
            while i < list_a.len() && j < list_b.len() {
                match list_a[i].0.cmp(&list_b[j].0) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        let sum = list_a[i].0;
                        let i_end = list_a[i..].iter().position(|x| x.0 != sum).map_or(list_a.len(), |p| i + p);
                        let j_end = list_b[j..].iter().position(|x| x.0 != sum).map_or(list_b.len(), |p| j + p);
                        for &(_, mask_a) in &list_a[i..i_end] {
                            for &(_, mask_b) in &list_b[j..j_end] {
                                blocks.push((mask_a, mask_b, size_a + size_b));
                                if blocks.len() > max_blocks {
                                    return None;
                                }
                            }
                        }
                        i = i_end;
                        j = j_end;
                    }
                }
            }
        }
    }
    if Instant::now() > deadline {
        return None;
    }
    let max_block_plus_one = (t + 1) as i64;
    const DUAL_SCALE: i64 = 1000;
    if blocks.is_empty() {
        // every block has >= t + 1 elements
        return Some((m as i64 / max_block_plus_one) as usize);
    }
    // LP rows: elements that occur in some block (others have dual 0)
    let element = |side: usize, bit: usize| if side == 0 { bit } else { a.len() + bit };
    let mut row_of = vec![usize::MAX; m];
    let mut cols: Vec<Vec<usize>> = Vec::with_capacity(blocks.len());
    let mut rows = 0;
    for &(mask_a, mask_b, _) in &blocks {
        let mut col = Vec::new();
        for (side, mask) in [(0, mask_a), (1, mask_b)] {
            let mut rest = mask;
            while rest != 0 {
                let bit = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                let e = element(side, bit);
                if row_of[e] == usize::MAX {
                    row_of[e] = rows;
                    rows += 1;
                }
                col.push(row_of[e]);
            }
        }
        cols.push(col);
    }
    // objective scaled by t + 1: c_Z = t + 1 - |Z| >= 1
    let c: Vec<f64> = blocks.iter().map(|block| (t + 1 - block.2) as f64).collect();
    let lp_dual = packing_lp_duals(rows, &cols, &c, deadline)?;
    // integer duals in units of 1 / (DUAL_SCALE (t + 1)), rounded up, then repaired block by block
    let mut dual: Vec<i64> =
        lp_dual.iter().map(|&x| (x.max(0.0) * DUAL_SCALE as f64 - 1e-7).ceil().max(0.0) as i64).collect();
    for (col, block) in cols.iter().zip(&blocks) {
        let need = (t + 1 - block.2) as i64 * DUAL_SCALE;
        let have: i64 = col.iter().map(|&r| dual[r]).sum();
        if have < need {
            dual[col[0]] += need - have;
        }
    }
    debug_assert!(cols
        .iter()
        .zip(&blocks)
        .all(|(col, block)| col.iter().map(|&r| dual[r]).sum::<i64>() >= (t + 1 - block.2) as i64 * DUAL_SCALE));
    // gamma* <= m / (t + 1) + (sum of the integer duals) / (DUAL_SCALE (t + 1))
    let total: i64 = dual.iter().sum();
    Some(((m as i64 * DUAL_SCALE + total) / (DUAL_SCALE * max_block_plus_one)) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    /// Random unmatched values built from shared parts, so that zero-sum blocks exist.
    fn random_unmatched(rng: &mut impl Rng, parts: usize) -> (Vec<i64>, Vec<i64>) {
        let w: Vec<i64> = (0..parts).map(|_| rng.gen_range(1..60)).collect();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let mut i = 0;
        while i < parts {
            let take = rng.gen_range(1..=3).min(parts - i);
            a.push(w[i..i + take].iter().sum());
            i += take;
        }
        let mut i = 0;
        while i < parts {
            let take = rng.gen_range(1..=3).min(parts - i);
            b.push(w[i..i + take].iter().sum());
            i += take;
        }
        // remove equal values of the two sides (they would have been paired)
        let mut a2 = Vec::new();
        for x in a {
            if let Some(p) = b.iter().position(|&y| y == x) {
                b.remove(p);
            } else {
                a2.push(x);
            }
        }
        (a2, b)
    }

    #[test]
    fn enumeration_and_packing_agree_with_dp() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
        let (mut dp, mut sums) = (Vec::new(), Vec::new());
        let far = Instant::now() + std::time::Duration::from_secs(60);
        let mut tested = 0;
        for _ in 0..400 {
            let parts = rng.gen_range(4..16);
            let (a, b) = random_unmatched(&mut rng, parts);
            let m = a.len() + b.len();
            if m == 0 || m > 20 {
                continue;
            }
            let v: Vec<i64> = a.iter().copied().chain(b.iter().map(|&y| -y)).collect();
            let exact = exact_block_number(&v, &mut dp, &mut sums);
            assert_eq!(
                exact_block_number_mitm(&v, 1 << 22, far),
                Some(exact),
                "enumeration differs from DP on {a:?} | {b:?}"
            );
            for t in 3..=6 {
                if let Some(p) = packing_bound(&a, &b, t, 1 << 20, far) {
                    assert!(p >= exact, "packing bound {p} < gamma* {exact} (t = {t}) on {a:?} | {b:?}");
                }
            }
            tested += 1;
        }
        assert!(tested > 100);
    }
}
