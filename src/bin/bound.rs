//! Runs the bound chain alone, with its own time budget per instance (outside any solve run). Rows are appended to
//! the output file as instances finish; running again with the same output skips the instances already there.
//!
//!     bound [--only ID,FAMILY] [--time 60] [--jobs 1] [--out results/bounds/chain.csv]
//!
//! Columns: lower_bound, method (the stage that proved it), seconds, quick upper bound (only used to stop early), and
//! the trace of all stages as `name:value@seconds;...`.

use rayon::prelude::*;
use rcs_birkhoff::bounds::bound_chain;
use rcs_birkhoff::cli::{is_selected, Args};
use rcs_birkhoff::instance::read_instance_dir;
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

fn main() {
    let args = Args::from_env();
    let filters = args.list("--only");
    let seconds: f64 = args.parse_or("--time", 60.0);
    let jobs: usize = args.parse_or("--jobs", 1);
    let out = args.string_or("--out", "results/bounds/chain.csv");

    let finished: HashSet<String> = std::fs::read_to_string(&out)
        .map(|text| text.lines().skip(1).filter_map(|line| line.split(',').next().map(String::from)).collect())
        .unwrap_or_default();
    if finished.is_empty() {
        std::fs::write(&out, "instance,n,lower_bound,method,seconds,quick_upper_bound,stages\n").unwrap();
    }
    let file = Mutex::new(std::fs::OpenOptions::new().append(true).open(&out).unwrap());
    let instances = read_instance_dir(Path::new("data/qoblib/03-birkhoff/instances")).expect("readable instances");
    let todo: Vec<_> =
        instances.iter().filter(|inst| is_selected(&inst.id, &filters) && !finished.contains(&inst.id)).collect();
    eprintln!("{} instances to bound ({} already done)", todo.len(), finished.len());

    rayon::ThreadPoolBuilder::new().num_threads(jobs).build().unwrap().install(|| {
        todo.par_iter().for_each(|inst| {
            let start = Instant::now();
            let result = bound_chain(inst, seconds);
            let trace: Vec<String> =
                result.stages.iter().map(|s| format!("{}:{}@{:.3}", s.name, s.value, s.seconds)).collect();
            let row = format!(
                "{},{},{},{},{:.3},{},{}\n",
                inst.id,
                inst.n,
                result.lower_bound,
                result.method,
                start.elapsed().as_secs_f64(),
                result.upper_bound,
                trace.join(";")
            );
            let mut file = file.lock().unwrap();
            file.write_all(row.as_bytes()).unwrap();
            file.flush().unwrap();
            eprint!("{row}");
        });
    });
}
