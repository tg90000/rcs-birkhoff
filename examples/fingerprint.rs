//! Regression check for refactoring: prints a deterministic fingerprint of the main algorithms on every test instance
//! (both rules of the Birkhoff algorithm, the frequency rule with fixing by unique subset sums and largest weight
//! first, pieces of the two-line bound, and a small beam search). Compare the output before and after a change; it must
//! be identical when the results must not change.
//!
//!     cargo run --release --example fingerprint > fingerprint.txt

use rcs_birkhoff::beam::{beam_search, pattern_permutations, Beam, Successors};
use rcs_birkhoff::birkhoff_algorithm::{decompose, Rule};
use rcs_birkhoff::blocks::{exact_block_number, packing_bound};
use rcs_birkhoff::bounds::quick_block_bound;
use rcs_birkhoff::decomposition::Decomposition;
use rcs_birkhoff::fixing::{Fixing, FixingResult};
use rcs_birkhoff::frequency::frequency_rule;
use rcs_birkhoff::instance::read_instance_dir;
use rcs_birkhoff::largest_first::largest_weight_first;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::{Duration, Instant};

/// The length of a decomposition and a hash of its terms.
fn digest(dec: &Decomposition) -> String {
    let mut hasher = DefaultHasher::new();
    dec.perms.hash(&mut hasher);
    dec.weights.hash(&mut hasher);
    format!("{}:{:016x}", dec.len(), hasher.finish())
}

fn main() {
    let instances = read_instance_dir(Path::new("data/qoblib/03-birkhoff/instances")).unwrap();
    let far = Instant::now() + Duration::from_secs(3600);
    let (mut dp, mut sums) = (Vec::new(), Vec::new());
    for inst in &instances {
        let n = inst.n;
        let mut out = vec![inst.id.clone()];
        out.push(digest(&decompose(inst, Rule::Zeroing)));
        out.push(digest(&decompose(inst, Rule::LogTieBreak)));
        if inst.treated_as_sparse() {
            match frequency_rule(inst) {
                Ok(w) => {
                    out.push(format!("w{:?}", w.len()));
                    match Fixing::new(inst, &w).solve() {
                        FixingResult::Found(d) => out.push(format!("fix {}", digest(&d))),
                        FixingResult::Inconsistent => out.push("fix inconsistent".into()),
                        FixingResult::Limit => out.push("fix limit".into()),
                    }
                    match largest_weight_first(inst, &w) {
                        Ok(d) => out.push(format!("lwf {}", digest(&d))),
                        Err(e) => out.push(format!("lwf err {e}")),
                    }
                }
                Err(e) => out.push(format!("w err {e:?}")),
            }
        }
        // two-line pieces on the first line pairs (deterministic, no time limit)
        let line = |x: usize| -> Vec<i64> {
            let mut v: Vec<i64> = if x < n {
                (0..n).map(|j| inst.d[x * n + j]).collect()
            } else {
                (0..n).map(|i| inst.d[i * n + x - n]).collect()
            };
            v.retain(|&a| a > 0);
            v.sort_unstable();
            v
        };
        let mut pieces = Vec::new();
        for (a, b) in [(0, 1), (0, n), (1, n + 1), (n - 1, 2 * n - 1)] {
            if a == b {
                continue;
            }
            let (line_a, line_b) = (line(a), line(b));
            let quick = quick_block_bound(&line_a, &line_b);
            let (mut unmatched_a, mut unmatched_b) = (line_a.clone(), line_b.clone());
            for x in line_a.iter() {
                if let Some(p) = unmatched_b.iter().position(|y| y == x) {
                    unmatched_b.remove(p);
                    unmatched_a.remove(unmatched_a.iter().position(|y| y == x).unwrap());
                }
            }
            let m = unmatched_a.len() + unmatched_b.len();
            let exact = if m <= 22 {
                let signed: Vec<i64> = unmatched_a.iter().copied().chain(unmatched_b.iter().map(|&y| -y)).collect();
                exact_block_number(&signed, &mut dp, &mut sums) as i64
            } else {
                -1
            };
            let packing = packing_bound(&unmatched_a, &unmatched_b, 4, 200_000, far).map_or(-1, |g| g as i64);
            pieces.push(format!("{quick}/{exact}/{packing}"));
        }
        out.push(pieces.join(","));
        // a small fixed beam on dense instances up to n = 16
        if !inst.treated_as_sparse() && n <= 16 {
            let pool = if n <= 6 { pattern_permutations(&inst.d, n, 5040) } else { None };
            let successors = match &pool {
                Some(p) => Successors::AllPermutations(p),
                None => Successors::PerPosition { max_successors: usize::MAX },
            };
            let width = if n <= 6 { 20 } else { 2 };
            let beam = Beam { successors, width, seed: 7, best_length: usize::MAX, completion_rule: Rule::Zeroing };
            let mut seen = Vec::new();
            let best = beam_search(&inst.d, n, &beam, &|| true, &mut |d| seen.push(d.len()));
            out.push(format!("beam {:?} {}", seen, best.map_or("-".into(), |d| digest(&d))));
        }
        println!("{}", out.join(" | "));
    }
}
