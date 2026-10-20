//! Writes the QOBLIB submission folder from a result set, in the format of the existing problem-03 submissions:
//! `<out>/<date>_<method>_<submitter>/README.md` and, per instance, `<id>_solution.json`, `<id>_summary.csv`,
//! `<id>_objective_time_series.json` and `README.md`.
//!
//!     package [--run results/final] [--bounds FILE] [--out qoblib_submission] [--date YYYY-MM-DD]
//!             [--threads 6]
//!
//! The optimality bound is the bound of the run. Only if a bound file is given with `--bounds`, the larger of the two
//! is used.

use rcs_birkhoff::cli::Args;
use rcs_birkhoff::instance::{read_instance_dir, Instance};
use std::collections::HashMap;
use std::path::Path;

const SUBMITTER: &str = "András Czégel";
const AFFILIATION: &str = "Bentax Plus";
/// Last part of the folder name, as in the other submissions (`<date>_<method>_<last name>`).
const FOLDER_NAME: &str = "Czegel";
const METHOD: &str = "RecoverCertifySearch";
const REFERENCE: &str = "https://arxiv.org/abs/2610.25837 (article), https://github.com/tg90000/rcs-birkhoff (code)";
/// AI disclosure of the QOBLIB AI policy (none, assisted or generated), with the tools and the parts they produced.
const AI_DISCLOSURE: &str = "assisted — AI assisted in coding and in formulating the documentation. Claude Code \
    (Anthropic) with the model Claude Opus 5.5 assisted in writing the solver code, the scripts and the documentation, \
    and Claude Fable 5.1 assisted in challenging ideas and in verifying them. The submitter directed the work and \
    checked the code and the results.";
const HARDWARE: &str = "Apple MacBook Pro, M4 Pro (8 performance + 4 efficiency cores), 24 GB RAM";

/// Columns of the QOBLIB submission template.
const HEADER: &str = "Problem,Submitter,Affiliation,Date,Reference,Best Objective Value,Optimality Bound,\
    Modeling Approach,# Decision Variables,# Binary Variables,# Integer Variables,# Continuous Variables,\
    # Non-Zero Coefficients,Coefficients Type,Coefficients Range,Workflow,Algorithm Type,Paradigm,# Runs,\
    # Feasible Runs,# Successful Runs,Success Threshold,Hardware Specifications,Total Runtime,Time to Solution,\
    CPU Runtime,GPU Runtime,QPU Runtime,Other HW Runtime,Remarks";

const MODELING: &str = "Exact decomposition, weights with the precision of the data. Solution chain: weights from the \
    frequency rule, placed by fixing by unique subset sums (sparse); Birkhoff algorithm with the zeroing rule, beam \
    search over zeroing steps, SAT re-optimization of the tail, large neighborhood search (dense). Bound chain in the \
    same run: degree, two-line bound, SAT certificates";

/// Quotes a CSV field if it contains a comma or a double quote.
fn quote(field: &str) -> String {
    if field.contains(',') || field.contains('"') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// The article's name of the method that found the best decomposition, from the label in the result files.
fn method_name(label: &str) -> &str {
    match label {
        "greedy-zeroing" => "Birkhoff algorithm with the zeroing rule",
        "greedy-digits" => "Birkhoff algorithm with the logarithmic tie-break rule",
        "frequency-weights+sweep" => "frequency rule and fixing by unique subset sums",
        "frequency-weights+peel" => "frequency rule and largest weight first",
        "weight-search+peel" => "enumeration of weight sets and largest weight first",
        "zbeam" => "beam search over zeroing steps",
        "sat-endgame" => "SAT re-optimization of the tail",
        "lns" => "large neighborhood search",
        "greedy-randomized" => "randomized restarts",
        "sat-exact" => "SAT model",
        other => other,
    }
}

/// Today's date (UTC) as YYYY-MM-DD, by the civil-from-days algorithm of H. Hinnant.
fn today() -> String {
    let seconds = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let days = (seconds / 86_400) as i64 + 719_468;
    let era = days.div_euclid(146_097);
    let day_of_era = days - era * 146_097;
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 { month_index + 3 } else { month_index - 9 };
    let year = year_of_era + era * 400 + (month <= 2) as i64;
    format!("{year:04}-{month:02}-{day:02}")
}

/// One instance of the result set.
struct InstanceResult<'a> {
    inst: &'a Instance,
    k: usize,
    lower_bound: usize,
    method: &'a str,
    time_to_solution: f64,
    runtime: f64,
}

/// The fields of the summary row, in the order of `HEADER`.
fn summary_fields(result: &InstanceResult, date: &str, threads: usize) -> Vec<String> {
    let inst = result.inst;
    let nonzeros = inst.nnz();
    vec![
        inst.id.clone(),
        SUBMITTER.into(),
        AFFILIATION.into(),
        date.into(),
        REFERENCE.into(),
        result.k.to_string(),
        result.lower_bound.to_string(),
        MODELING.into(),
        (nonzeros + 1).to_string(),
        nonzeros.to_string(),
        "1".into(),
        "0".into(),
        (2 * inst.n).to_string(),
        "Binary".into(),
        "1".into(),
        format!(
            "Parallel run ({threads} threads): solution chain and bound chain in one run, stopping when they meet; \
             final solution found by: {}",
            method_name(result.method)
        ),
        "Stochastic".into(),
        "Classical".into(),
        "1".into(),
        "1".into(),
        "1".into(),
        "0".into(),
        HARDWARE.into(),
        format!("{:.3}", result.runtime),
        format!("{:.3}", result.time_to_solution),
        format!("{:.3}", result.runtime * threads as f64),
        "0".into(),
        "0".into(),
        "0".into(),
        format!(
            "Exact decomposition (no tolerance), with weights of the precision of the data (integers at the given \
             scale). Proofs of optimality by the degree or two-line bound hold for real weights too; proofs by SAT \
             certificates (small dense instances) hold for weights of this precision. Runs: 60 s per instance; \
             instances without a proof of optimality were run again with 600 s; the shorter decomposition is reported \
             with the runtime of its run. Every optimality bound comes from the reported runs themselves. CPU runtime \
             is an upper estimate (wall time x {threads} threads). The planted decomposition in the instance file is \
             never read."
        ),
    ]
}

/// Writes the four files of one instance.
fn write_instance(dir: &Path, run_dir: &Path, result: &InstanceResult, fields: &[String]) {
    let id = &result.inst.id;
    std::fs::create_dir_all(dir).unwrap();
    // the solution file in the layout of the instance files, with the proof of optimality
    let text = std::fs::read_to_string(run_dir.join(format!("{id}_solution.json"))).unwrap();
    let solution: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mut entry = solution.as_object().unwrap().values().next().unwrap().as_object().unwrap().clone();
    entry.insert("optimal".into(), serde_json::Value::Bool(result.lower_bound >= result.k));
    let mut file = serde_json::Map::new();
    file.insert("001".into(), serde_json::Value::Object(entry));
    std::fs::write(dir.join(format!("{id}_solution.json")), serde_json::to_string_pretty(&file).unwrap()).unwrap();
    // the objective time series: one run, one entry per improvement
    let text = std::fs::read_to_string(run_dir.join(format!("{id}_history.json"))).unwrap();
    let history: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
    let series: Vec<serde_json::Value> = history
        .iter()
        .map(|h| serde_json::json!({"Time": h["Time"], "Incumbent": h["Incumbent"], "Error": 0.0}))
        .collect();
    let series_text = serde_json::to_string_pretty(&vec![series]).unwrap();
    std::fs::write(dir.join(format!("{id}_objective_time_series.json")), series_text).unwrap();
    // the summary in the QOBLIB template, and the same as a Markdown table
    let row: Vec<String> = fields.iter().map(|f| quote(f)).collect();
    std::fs::write(dir.join(format!("{id}_summary.csv")), format!("{HEADER}\n{}\n", row.join(","))).unwrap();
    let mut readme = format!(
        "# Submission for {id}\n\nThis directory contains the submission for the problem **{id}**.\n\n\
         | Field | Value 1 |\n| --- | --- |\n"
    );
    for (name, value) in HEADER.split(',').zip(fields) {
        readme.push_str(&format!("| {name} | {} |\n", value.replace('|', "/")));
    }
    std::fs::write(dir.join("README.md"), readme).unwrap();
}

/// The README of the submission folder: runs, method, AI disclosure and the results table.
fn submission_readme(date: &str, threads: usize, count: usize, table: &str) -> String {
    format!(
        "# {METHOD} — Minimum Birkhoff Decomposition (QOBLIB 03)\n\
        \n\
        Submitter: {SUBMITTER} ({AFFILIATION}). Date: {date}. Hardware: {HARDWARE}.\n\
        \n\
        ## Runs\n\
        \n\
        Every instance gets one run of 60 s wall clock ({threads} threads). Instances without a proof of optimality \
        get a second run of 600 s. The shorter decomposition is reported, with the runtime of its run, and every \
        optimality bound comes from these runs.\n\
        \n\
        All decompositions are **exact**: the weights have the precision of the data (whole numbers at the \
        instance's scale), no tolerance, checked with the official checker. The solver reads only the matrix; the \
        planted decomposition stored in the instance files is never used. \"Proven optimal\" means optimal among \
        decompositions with weights of this precision; the proofs by the degree or two-line bound hold for real \
        weights too, the SAT proofs (small dense instances) do not.\n\
        \n\
        ## Method\n\
        \n\
        Two chains run side by side in every run and stop when they meet; then the result is proven optimal.\n\
        \n\
        1. *Solution chain.* Sparse instances: the weights are read from how often each value occurs (frequency \
        rule), then placed position by position by fixing by unique subset sums, which provably makes no wrong \
        choice (largest weight first and an enumeration of weight sets run in parallel). Dense instances: the \
        Birkhoff algorithm with the zeroing rule, which makes as many entries zero as possible, a beam search over \
        zeroing steps, SAT re-optimization of the small-weight tail, and large neighborhood search. Every \
        decomposition is checked before it is accepted.\n\
        2. *Bound chain.* The degree (most nonzero entries in a line); the two-line bound (how many equal-sum blocks \
        two lines can be split into, counted exactly or bounded by a packing bound that is checked in integer \
        arithmetic); SAT certificates that fewer terms are impossible with weights of this precision (split into \
        cases for n <= 7).\n\
        \n\
        Definitions and proofs are in the accompanying article. Source code: Rust. Reference: {REFERENCE}.\n\
        \n\
        ## AI disclosure\n\
        \n\
        {AI_DISCLOSURE}\n\
        \n\
        ## Results ({count} instances)\n\
        \n\
        {table}"
    )
}

fn main() {
    let args = Args::from_env();
    let run_dir = Path::new(&args.string_or("--run", "results/final")).to_path_buf();
    let out = Path::new(&args.string_or("--out", "qoblib_submission")).to_path_buf();
    let date = args.get("--date").unwrap_or_else(today);
    let threads: usize = args.parse_or("--threads", 6);
    // optional bound file of the `bound` binary: instance -> lower bound
    let extra_bounds: HashMap<String, usize> = args
        .get("--bounds")
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|text| {
            let rows = text.lines().skip(1).map(|line| line.split(',').collect::<Vec<_>>());
            rows.map(|f| (f[0].to_string(), f[2].parse().unwrap())).collect()
        })
        .unwrap_or_default();
    let folder = out.join(format!("{}_{METHOD}_{FOLDER_NAME}", date.replace('-', "")));
    std::fs::create_dir_all(&folder).unwrap();

    let instances = read_instance_dir(Path::new("data/qoblib/03-birkhoff/instances")).expect("readable instances");
    let summary = std::fs::read_to_string(run_dir.join("summary.csv")).unwrap();
    let mut lines = summary.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let column = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let rows: HashMap<String, Vec<&str>> = lines
        .map(|line| line.split(',').collect::<Vec<_>>())
        .map(|row| (row[column("instance")].to_string(), row))
        .collect();

    let mut table = String::from(
        "| Instance | n | k (this work) | optimality bound | proven optimal | time to solution [s] | \
         total runtime [s] |\n|---|---|---|---|---|---|---|\n",
    );
    let mut count = 0;
    for inst in &instances {
        let Some(row) = rows.get(&inst.id) else { continue };
        let run_bound: usize = row[column("lower_bound")].parse().unwrap();
        let result = InstanceResult {
            inst,
            k: row[column("k")].parse().unwrap(),
            lower_bound: run_bound.max(extra_bounds.get(&inst.id).copied().unwrap_or(0)),
            method: row[column("method")],
            time_to_solution: row[column("time_to_solution")].parse().unwrap(),
            runtime: row[column("seconds")].parse().unwrap(),
        };
        let fields = summary_fields(&result, &date, threads);
        write_instance(&folder.join(&inst.id), &run_dir.join(&inst.id), &result, &fields);
        table.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.3} | {:.1} |\n",
            inst.id,
            inst.n,
            result.k,
            result.lower_bound,
            if result.lower_bound >= result.k { "yes" } else { "" },
            result.time_to_solution,
            result.runtime
        ));
        count += 1;
    }
    std::fs::write(folder.join("README.md"), submission_readme(&date, threads, count, &table)).unwrap();
    println!("wrote {count} instances to {}", folder.display());
}
