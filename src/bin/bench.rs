//! Runs single methods on the test instances and compares them with the best known exact values. This is for method
//! studies; a full run of RCS is `solve`.
//!
//!     bench [--methods zeroing,logrule,bottleneck,birkhoffplus,largest-first,weight-sets,beam1000_30,beamp30_30,
//!            tail15,lns10] [--only B10_100,B16_256] [--max-n 64] [--threads 8] [--out results.csv]
//!
//! Method keys: `beam{width}_{seconds}` uses all permutations of the nonzero pattern when there are at most 5040 and
//! one successor per position otherwise, `beamp{width}_{seconds}[_{successors per node}]` always one successor per
//! position, `tail{cap}` the SAT re-optimization of the tail, and `lns{seconds}` the large neighborhood search.

use rand::SeedableRng;
use rayon::prelude::*;
use rcs_birkhoff::beam::{beam_search, pattern_permutations, Beam, Successors};
use rcs_birkhoff::birkhoff_algorithm::{decompose, Rule};
use rcs_birkhoff::cli::{is_selected, Args};
use rcs_birkhoff::decomposition::Decomposition;
use rcs_birkhoff::frequency::{enumerate_weight_sets, frequency_rule};
use rcs_birkhoff::instance::{read_instance_dir, Instance};
use rcs_birkhoff::largest_first::largest_weight_first;
use rcs_birkhoff::reference::best_known;
use std::path::Path;
use std::time::{Duration, Instant};

/// The numbers after the prefix of a method key, separated by "_" ("beam1000_30" gives [1000, 30]).
fn parameters(key: &str, prefix: &str) -> Vec<f64> {
    key[prefix.len()..].split('_').filter(|s| !s.is_empty()).map(|s| s.parse().expect("a number")).collect()
}

/// Runs one method on one instance and returns the decomposition and a note for the result table.
fn run(inst: &Instance, method: &str) -> (Decomposition, String) {
    let zeroing = || decompose(inst, Rule::Zeroing);
    match method {
        "zeroing" => (zeroing(), String::new()),
        "logrule" => (decompose(inst, Rule::LogTieBreak), String::new()),
        "bottleneck" => (decompose(inst, Rule::Bottleneck), String::new()),
        "birkhoffplus" => (decompose(inst, Rule::BirkhoffPlus), String::new()),
        "largest-first" => match frequency_rule(inst).map(|w| largest_weight_first(inst, &w)) {
            Ok(Ok(dec)) => (dec, String::new()),
            _ => (zeroing(), "largest weight first failed".into()),
        },
        "weight-sets" => {
            let deadline = Instant::now() + Duration::from_secs(10);
            match enumerate_weight_sets(inst, usize::MAX, deadline, &|| true, &mut |_| {}) {
                Some(dec) => (dec, String::new()),
                None => (zeroing(), "enumeration of weight sets failed".into()),
            }
        }
        key if key.starts_with("beam") => {
            let per_position = key.starts_with("beamp");
            let values = parameters(key, if per_position { "beamp" } else { "beam" });
            let width = values[0] as usize;
            let seconds = values.get(1).copied().unwrap_or(30.0);
            let max_successors = values.get(2).map_or(usize::MAX, |&v| v as usize);
            let start = Instant::now();
            let deadline = start + Duration::from_secs_f64(seconds);
            let all = if per_position { None } else { pattern_permutations(&inst.d, inst.n, 5040) };
            let successors = match &all {
                Some(perms) => Successors::AllPermutations(perms),
                None => Successors::PerPosition { max_successors },
            };
            let first = zeroing();
            let beam = Beam { successors, width, seed: 1, best_length: first.len(), completion_rule: Rule::Zeroing };
            let mut log = Vec::new();
            let best = beam_search(&inst.d, inst.n, &beam, &|| Instant::now() < deadline, &mut |dec| {
                log.push(format!("{}@{:.2}", dec.len(), start.elapsed().as_secs_f64()))
            });
            (best.unwrap_or(first), log.join(" "))
        }
        key if key.starts_with("tail") => {
            let cap = parameters(key, "tail").first().copied().unwrap_or(15.0) as i64;
            let mut dec = zeroing();
            let before = dec.len();
            let deadline = Instant::now() + Duration::from_secs(20);
            rcs_birkhoff::sat::reoptimize_tail(&mut dec, inst.n, cap, deadline, None);
            let note = format!("{before}->{}", dec.len());
            (dec, note)
        }
        key if key.starts_with("lns") => {
            let seconds = parameters(key, "lns").first().copied().unwrap_or(10.0);
            let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
            let mut dec = zeroing();
            let deadline = Instant::now() + Duration::from_secs_f64(seconds);
            let stats = rcs_birkhoff::lns::improve(inst, &mut dec, deadline, &|| true, &mut rng, &mut |_| {});
            (dec, format!("iterations {} improvements {}", stats.iterations, stats.improvements))
        }
        other => panic!("unknown method {other}"),
    }
}

fn main() {
    let args = Args::from_env();
    let instance_dir = args.string_or("--instances", "data/qoblib/03-birkhoff/instances");
    let solution_dir = args.string_or("--solutions", "data/qoblib/03-birkhoff/solutions");
    let methods = args.list("--methods");
    let methods =
        if methods.is_empty() { vec!["zeroing".into(), "bottleneck".into(), "birkhoffplus".into()] } else { methods };
    let filters = args.list("--only");
    let max_n: usize = args.parse_or("--max-n", 1000);
    let threads: usize = args.parse_or("--threads", 8);

    let best = best_known(Path::new(&solution_dir));
    let instances: Vec<Instance> = read_instance_dir(Path::new(&instance_dir))
        .expect("readable instances")
        .into_iter()
        .filter(|inst| inst.n <= max_n && is_selected(&inst.id, &filters))
        .collect();
    let jobs: Vec<(&Instance, &String)> =
        instances.iter().flat_map(|inst| methods.iter().map(move |method| (inst, method))).collect();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let rows: Vec<String> = pool.install(|| {
        jobs.par_iter()
            .map(|&(inst, method)| {
                let start = Instant::now();
                let (dec, note) = run(inst, method);
                let seconds = start.elapsed().as_secs_f64();
                let valid = dec.verify(inst).is_ok();
                let (best_k, best_optimal) =
                    best.get(&inst.id).map_or((String::new(), false), |(k, optimal, _)| (k.to_string(), *optimal));
                format!(
                    "{},{},{},{method},{},{best_k},{best_optimal},{seconds:.4},{valid},{note}",
                    inst.id,
                    inst.n,
                    inst.degree(),
                    dec.len()
                )
            })
            .collect()
    });
    let mut text = String::from("instance,n,degree,method,k,best_known,bk_optimal,seconds,valid,note\n");
    for row in &rows {
        text.push_str(row);
        text.push('\n');
    }
    match args.get("--out") {
        Some(path) => std::fs::write(path, text).unwrap(),
        None => print!("{text}"),
    }
}
