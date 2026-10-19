//! Merges runs into one result set (results/final): for every instance the best decomposition over all runs (fewest
//! terms; on a tie the run that proved optimality itself, then the shorter running time) with its running time and
//! files, and the largest lower bound over the runs and the bound files, with the stage that proved it. Instances in
//! `--timed-out` (no proof within the last time limit) are marked.
//!
//!     merge --runs results/runs/round1_t60,results/runs/round2_t600 [--bounds FILE]
//!           [--timed-out results/runs/round2_t600/unfinished.csv] [--out results/final]
//!
//! The results of the article use these two runs and no bound files.

use rcs_birkhoff::cli::Args;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

/// A CSV row as a map from column name to value.
type Record = HashMap<String, String>;

/// The best run of one instance: its summary row and the run directory.
struct BestRun {
    record: Record,
    dir: PathBuf,
}

fn read_csv(path: &Path) -> Vec<Record> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    let mut lines = text.lines();
    let header: Vec<String> = lines.next().unwrap_or("").split(',').map(String::from).collect();
    lines.map(|line| header.iter().cloned().zip(line.split(',').map(String::from)).collect()).collect()
}

/// Order of two runs of the same instance: fewer terms first, then a run that proved optimality itself, then the
/// shorter running time.
fn rank(record: &Record) -> (usize, bool, u64) {
    let proven = record.get("proven_optimal").is_some_and(|p| p == "true");
    let seconds: f64 = record["seconds"].parse().unwrap();
    (record["k"].parse().unwrap(), !proven, (seconds * 1e4) as u64)
}

fn main() {
    let args = Args::from_env();
    let runs: Vec<PathBuf> = args.list("--runs").into_iter().map(PathBuf::from).collect();
    let bound_files: Vec<PathBuf> = args.list("--bounds").into_iter().map(PathBuf::from).collect();
    let timed_out: HashSet<String> = args
        .get("--timed-out")
        .map(|path| read_csv(Path::new(&path)).into_iter().map(|r| r["instance"].clone()).collect())
        .unwrap_or_default();
    let out = PathBuf::from(args.string_or("--out", "results/final"));
    std::fs::create_dir_all(&out).unwrap();

    let mut best: BTreeMap<String, BestRun> = BTreeMap::new();
    let mut lower_bounds: HashMap<String, (usize, String)> = HashMap::new();
    let mut raise = |id: &str, bound: usize, method: &str| {
        let entry = lower_bounds.entry(id.to_string()).or_insert((0, String::new()));
        // a larger bound wins; at the same value, a named stage replaces "solve" (summaries without a stage column)
        if bound > entry.0 || (bound == entry.0 && entry.1 == "solve" && method != "solve") {
            *entry = (bound, method.to_string());
        }
    };
    for dir in &runs {
        for record in read_csv(&dir.join("summary.csv")) {
            let id = record["instance"].clone();
            let method = record.get("lower_bound_method").map_or("solve", |s| s.as_str());
            raise(&id, record["lower_bound"].parse().unwrap(), method);
            if best.get(&id).is_none_or(|b| rank(&record) < rank(&b.record)) {
                best.insert(id, BestRun { record, dir: dir.clone() });
            }
        }
    }
    for file in &bound_files {
        for record in read_csv(file) {
            raise(&record["instance"], record["lower_bound"].parse().unwrap(), &record["method"]);
        }
    }

    let mut summary = String::from(
        "instance,n,k,lower_bound,lower_bound_method,proven_optimal,time_to_solution,seconds,method,timed_out_600s\n",
    );
    let mut unfinished = String::from("instance,n,k,lower_bound\n");
    for (id, run) in &best {
        let record = &run.record;
        let k: usize = record["k"].parse().unwrap();
        let (lower_bound, method) = lower_bounds[id].clone();
        assert!(lower_bound <= k, "{id}: lower bound {lower_bound} exceeds the verified length k = {k}");
        let is_timed_out = timed_out.contains(id);
        summary.push_str(&format!(
            "{id},{},{k},{lower_bound},{method},{},{},{},{},{is_timed_out}\n",
            record["n"],
            k <= lower_bound,
            record["time_to_solution"],
            record["seconds"],
            record["method"]
        ));
        if is_timed_out {
            unfinished.push_str(&format!("{id},{},{k},{lower_bound}\n", record["n"]));
        }
        let (source, target) = (run.dir.join(id), out.join(id));
        std::fs::create_dir_all(&target).unwrap();
        for name in [format!("{id}_solution.json"), format!("{id}_history.json")] {
            std::fs::copy(source.join(&name), target.join(&name))
                .unwrap_or_else(|e| panic!("{}: {e}", source.join(&name).display()));
        }
    }
    std::fs::write(out.join("summary.csv"), summary).unwrap();
    std::fs::write(out.join("timed_out_600s.csv"), unfinished).unwrap();
    println!(
        "merged {} runs and {} bound files into {} ({} instances)",
        runs.len(),
        bound_files.len(),
        out.display(),
        best.len()
    );
}
