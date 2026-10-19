//! Runs RCS on the test instances. One run per instance contains both chains (`solver`): the bound chain raises a lower
//! bound while the solution chain searches; the run stops when the two meet (proven optimal) or at the time limit.
//!
//!     solve [--instances FILE_OR_DIR] [--only ID,FAMILY] [--time 60] [--jobs 1] [--threads 8] [--out results/run]
//!
//! `--jobs` instances run at the same time, each with `--threads` threads. Output, written as soon as it exists:
//! `<out>/<id>/<id>_solution.json` (QOBLIB format) and `<out>/<id>/<id>_history.json` after every improvement, one row
//! per finished instance in `<out>/summary.csv`, and in `<out>/unfinished.csv` the instances that used the whole budget
//! without a proof. Running again with the same `--out` continues: instances already in summary.csv are skipped.

use rcs_birkhoff::cli::{is_selected, Args};
use rcs_birkhoff::instance::{read_instance_dir, read_instances};
use rcs_birkhoff::solver::{solve, write_checkpoint, Config};
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// The mimalloc allocator: about 10 % faster than the system allocator in the beam search, with the same results.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Opens a CSV file for appending and writes `header` first if the file does not exist yet.
fn append_csv(path: &Path, header: &str) -> Mutex<File> {
    if !path.exists() {
        std::fs::write(path, format!("{header}\n")).unwrap();
    }
    Mutex::new(std::fs::OpenOptions::new().append(true).open(path).unwrap())
}

fn main() {
    let args = Args::from_env();
    let instance_path = args.string_or("--instances", "data/qoblib/03-birkhoff/instances");
    let filters = args.list("--only");
    let seconds: f64 = args.parse_or("--time", 60.0);
    let jobs: usize = args.parse_or("--jobs", 1);
    let threads: usize = args.parse_or("--threads", 8);
    let out = Path::new(&args.string_or("--out", "results/run")).to_path_buf();

    let instance_path = Path::new(&instance_path);
    let instances =
        if instance_path.is_dir() { read_instance_dir(instance_path) } else { read_instances(instance_path) }
            .expect("readable instances");
    let selected: Vec<_> = instances.into_iter().filter(|inst| is_selected(&inst.id, &filters)).collect();

    std::fs::create_dir_all(&out).unwrap();
    let summary_path = out.join("summary.csv");
    let finished: HashSet<String> = std::fs::read_to_string(&summary_path)
        .map(|text| text.lines().skip(1).filter_map(|line| line.split(',').next().map(String::from)).collect())
        .unwrap_or_default();
    let summary = append_csv(
        &summary_path,
        "instance,n,k,lower_bound,lower_bound_method,proven_optimal,time_to_solution,seconds,method",
    );
    let unfinished = append_csv(&out.join("unfinished.csv"), "instance,n,k,lower_bound,seconds");
    let todo: Vec<_> = selected.iter().filter(|inst| !finished.contains(&inst.id)).collect();
    eprintln!("{} instances to solve ({} already done)", todo.len(), selected.len() - todo.len());

    // every job takes the next instance from a shared counter, so no job is idle while instances remain
    let next = AtomicUsize::new(0);
    let job = || {
        while let Some(inst) = todo.get(next.fetch_add(1, Ordering::Relaxed)) {
            let dir = out.join(&inst.id);
            let config =
                Config { budget: Duration::from_secs_f64(seconds), threads, seed: 0, checkpoint: Some(dir.clone()) };
            let result = solve(inst, &config);
            result.best.verify(inst).expect("the best decomposition is exact");
            let (k, lower_bound) = (result.best.len(), result.lower_bound);
            let proven = k <= lower_bound;
            write_checkpoint(&dir, inst, &result.best, &result.history, proven);
            let method = result.history.last().map_or("", |h| h.method.as_str());
            eprintln!("{:<14} k={k:<4} lb={lower_bound:<4} {:7.2}s ({method})", inst.id, result.seconds);
            if !proven {
                let mut file = unfinished.lock().unwrap();
                writeln!(file, "{},{},{k},{lower_bound},{:.4}", inst.id, inst.n, result.seconds).unwrap();
                file.flush().unwrap();
            }
            let mut file = summary.lock().unwrap();
            writeln!(
                file,
                "{},{},{k},{lower_bound},{},{proven},{:.4},{:.4},{method}",
                inst.id, inst.n, result.lower_bound_method, result.time_to_solution, result.seconds
            )
            .unwrap();
            file.flush().unwrap();
        }
    };
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1) {
            scope.spawn(job);
        }
    });
}
