//! Recover, first step: finding the weights of a sparse matrix (Sect. 3.2 of the article).
//!
//! In every decomposition, each row splits the list of weights into groups whose sums are the entries of that row
//! (Lemma 1). So the weights sum to s, and every nonzero entry is a sum of weights. If the matrix is built from n
//! random permutations, a given one of them is the only one through its position in a given row with probability
//! (1 - 1/n)^(n-1), about 1/e. Its weight then appears as an entry about n/e times, while a sum of two weights appears
//! only where the two permutations meet. So the weights are the most frequent values. `fixing` and `largest_first`
//! place the weights found here on the positions of D.

use crate::decomposition::Decomposition;
use crate::instance::Instance;
use std::collections::HashMap;

/// The frequency rule: the weights of a sparse matrix (sorted ascending), or the reason why they could not be made
/// consistent.
///
/// 1. We sort the distinct values of the entries by frequency, most frequent first, and take the shortest prefix whose
///    sum is exactly s, but only if the prefix ends where the frequency drops. A value that occurs about twice as
///    often as a typical weight counts twice, because two weights can be equal.
/// 2. If this fails, a scan in the spirit of sparsest fit goes through the values in increasing order and takes a
///    value as a new weight unless it is the sum of at most three known weights (`sparsest_fit`).
pub fn frequency_rule(inst: &Instance) -> Result<Vec<i64>, (usize, i64)> {
    let n = inst.n;
    let mut freq: HashMap<i64, usize> = HashMap::new();
    for &v in inst.d.iter().filter(|&&v| v > 0) {
        *freq.entry(v).or_insert(0) += 1;
    }
    let mut by_freq: Vec<(usize, i64)> = freq.iter().map(|(&v, &f)| (f, v)).collect();
    by_freq.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    // typical frequency of a weight: the median frequency of the plain prefix that first reaches s
    let mut sum = 0i64;
    let mut prefix = 0usize;
    while prefix < by_freq.len() && sum < inst.scale {
        sum += by_freq[prefix].1;
        prefix += 1;
    }
    let mut prefix_frequencies: Vec<usize> = by_freq[..prefix].iter().map(|x| x.0).collect();
    prefix_frequencies.sort_unstable();
    let typical = prefix_frequencies.get(prefix_frequencies.len() / 2).copied().unwrap_or(1);
    // a value used by m permutations of equal weight occurs about m times as often (meaningful for typical >= 8)
    let multiplicity = |f: usize| {
        if typical >= 8 {
            ((f as f64 / typical as f64).round() as usize).max(1)
        } else {
            1
        }
    };
    let mut sum = 0i64;
    let mut weights: Vec<i64> = Vec::new();
    for (i, &(f, v)) in by_freq.iter().enumerate() {
        for _ in 0..multiplicity(f) {
            weights.push(v);
            sum += v;
        }
        if sum == inst.scale && f >= 2 && by_freq.get(i + 1).is_none_or(|&(g, _)| g < f || g <= 1) {
            weights.sort_unstable();
            return Ok(weights);
        }
        if sum > inst.scale {
            break;
        }
    }
    sparsest_fit(inst, &freq, (n / 8).max(3), 3)
}

/// Scan in the spirit of the sparsest-fit algorithm of Jagabathula and Shah: values in increasing order; a value
/// becomes a new weight unless it is the sum of at most `max_sum_terms` weights found so far, or if it occurs at least
/// `frequent` times. The identity sum = s repairs one missing or one extra weight.
fn sparsest_fit(
    inst: &Instance,
    freq: &HashMap<i64, usize>,
    frequent: usize,
    max_sum_terms: usize,
) -> Result<Vec<i64>, (usize, i64)> {
    let mut values: Vec<i64> = freq.keys().copied().collect();
    values.sort_unstable();
    let mut weights: Vec<i64> = Vec::new();
    // sums of at most `max_sum_terms` distinct weights
    let mut sums: Vec<std::collections::HashSet<i64>> = vec![Default::default(); max_sum_terms + 1];
    sums[0].insert(0);
    for &v in &values {
        let representable = (1..=max_sum_terms).any(|t| sums[t].contains(&v));
        if !representable || freq[&v] >= frequent {
            for t in (1..=max_sum_terms).rev() {
                let extended: Vec<i64> = sums[t - 1].iter().map(|x| x + v).collect();
                sums[t].extend(extended);
            }
            weights.push(v);
        }
        if weights.iter().sum::<i64>() > inst.scale {
            break;
        }
    }
    let sum: i64 = weights.iter().sum();
    if sum == inst.scale {
        return Ok(weights);
    }
    if sum < inst.scale {
        weights.push(inst.scale - sum);
        weights.sort_unstable();
        return Ok(weights);
    }
    let excess = sum - inst.scale;
    if let Some(pos) = weights.iter().position(|&w| w == excess) {
        let mut rest = weights.clone();
        rest.remove(pos);
        return Ok(rest);
    }
    Err((weights.len(), sum))
}

/// The enumeration of weight sets. As long as the best decomposition has more than n terms, a depth-first search tries
/// other weight sets in order of frequency: subsets of the distinct values, plus at most one weight that appears at no
/// position alone and follows from sum = s. Each set is tested by largest weight first, and every shorter valid
/// decomposition goes to `on_found`. The search stops after `max_tests` tests, at the deadline, or when `running()`
/// turns false (the run is over).
pub fn enumerate_weight_sets(
    inst: &Instance,
    max_tests: usize,
    deadline: std::time::Instant,
    running: &dyn Fn() -> bool,
    on_found: &mut dyn FnMut(&Decomposition),
) -> Option<Decomposition> {
    let mut freq: HashMap<i64, usize> = HashMap::new();
    for &v in inst.d.iter().filter(|&&v| v > 0) {
        *freq.entry(v).or_insert(0) += 1;
    }
    let mut values: Vec<(usize, i64)> = freq.iter().map(|(&v, &f)| (f, v)).collect();
    values.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    let values: Vec<i64> = values.into_iter().map(|(_, v)| v).collect();
    struct WeightSetSearch<'a, 'b> {
        on_found: &'b mut dyn FnMut(&Decomposition),
        inst: &'a Instance,
        values: &'a [i64],
        tests: usize,
        max_tests: usize,
        deadline: std::time::Instant,
        running: &'b dyn Fn() -> bool,
        best: Option<Decomposition>,
    }
    fn try_weights(search: &mut WeightSetSearch, weights: &[i64]) {
        search.tests += 1;
        let mut weights = weights.to_vec();
        weights.sort_unstable();
        if let Ok(dec) = crate::largest_first::largest_weight_first(search.inst, &weights) {
            if search.best.as_ref().is_none_or(|b| dec.len() < b.len()) {
                (search.on_found)(&dec);
                search.best = Some(dec);
            }
        }
    }
    /// Extends the chosen weights by the values from `index` on, or closes their sum with one more weight.
    fn extend(search: &mut WeightSetSearch, index: usize, chosen: &mut Vec<i64>, sum: i64) {
        if search.tests >= search.max_tests || std::time::Instant::now() >= search.deadline || !(search.running)() {
            return;
        }
        let s = search.inst.scale;
        let limit = search.best.as_ref().map_or(usize::MAX, |b| b.len());
        if chosen.len() >= limit {
            return;
        }
        if sum == s {
            try_weights(search, chosen);
            return;
        }
        // one more weight that closes the sum
        if chosen.len() + 1 < limit {
            chosen.push(s - sum);
            try_weights(search, chosen);
            chosen.pop();
        }
        if index == search.values.len() {
            return;
        }
        let v = search.values[index];
        if sum + v <= s {
            chosen.push(v);
            extend(search, index + 1, chosen, sum + v);
            chosen.pop();
        }
        extend(search, index + 1, chosen, sum);
    }
    let mut search =
        WeightSetSearch { on_found, inst, values: &values, tests: 0, max_tests, deadline, running, best: None };
    extend(&mut search, 0, &mut Vec::new(), 0);
    search.best
}
