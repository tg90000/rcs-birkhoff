//! One run of RCS (Sect. 3.1 and 3.5 of the article, Algorithm 1, Table 1).
//!
//! A run has a time limit and consists of two chains that work in parallel:
//! * the solution chain: threads that offer decompositions to a shared best decomposition U. Every offer is checked in
//!   integer arithmetic before it is accepted, and every improvement is stored with its time;
//! * the bound chain: one thread (Certify) that raises a shared lower bound L (`bounds::run_bound_chain`).
//!
//! The run stops as soon as U has length L, because U is then optimal. Otherwise it stops at the time limit.
//!
//! The threads of a run (Table 1 of the article):
//! * Certify (all matrices). On a matrix treated as dense, its thread joins the solution chain when the chain ends;
//! * Recover (matrices treated as sparse): the frequency rule with fixing by unique subset sums, then the enumeration
//!   of weight sets; a second thread places the same weights largest first;
//! * SAT model (n <= 4): the first k = L, L + 1, ... with a satisfiable model gives a decomposition;
//! * beam search (matrices treated as dense): beam search over zeroing steps with growing width, on its own thread
//!   pool;
//! * SAT re-optimization of the tail (matrices treated as dense), repeated whenever the best decomposition changes;
//! * large neighborhood search (the remaining threads), with a randomized restart in every fourth round.
//!
//! The method labels in the time series ("greedy-zeroing", "zbeam", ...) are data keys of the result files and stay as
//! they are; `method_name` in `src/bin/package.rs` maps them to the names of the article.

use crate::beam::{beam_search, pattern_permutations, Beam, Successors, MAX_SUCCESSORS};
use crate::birkhoff_algorithm::{decompose, decompose_randomized, Rule};
use crate::decomposition::Decomposition;
use crate::frequency::{enumerate_weight_sets, frequency_rule};
use crate::instance::Instance;
use crate::largest_first::largest_weight_first;
use crate::sat::{decompose_exactly_until, reoptimize_tail, SatResult};
use rand::SeedableRng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// One improvement of the best decomposition: the time since the start, the new length, and the method that found it.
#[derive(Clone, Debug)]
pub struct Improvement {
    pub seconds: f64,
    pub k: usize,
    pub method: String,
}

/// Settings of a run.
#[derive(Clone, Debug)]
pub struct Config {
    /// time limit
    pub budget: Duration,
    pub threads: usize,
    pub seed: u64,
    /// If set, the best decomposition and its time series are written to this directory at every improvement, via a
    /// temporary file and a rename, so an interrupted run loses nothing.
    pub checkpoint: Option<std::path::PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Config { budget: Duration::from_secs(60), threads: 8, seed: 0, checkpoint: None }
    }
}

/// Writes `<dir>/<id>_solution.json` (format of the QOBLIB checker) and `<dir>/<id>_history.json`, each via a
/// temporary file and a rename.
pub fn write_checkpoint(
    dir: &std::path::Path,
    inst: &Instance,
    best: &Decomposition,
    history: &[Improvement],
    optimal: bool,
) {
    let write = |name: String, text: String| {
        let tmp = dir.join(format!(".{name}.tmp"));
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::rename(&tmp, dir.join(name));
        }
    };
    let _ = std::fs::create_dir_all(dir);
    let mut file = serde_json::Map::new();
    file.insert("1".into(), best.to_qoblib_json(inst, optimal));
    write(format!("{}_solution.json", inst.id), serde_json::to_string_pretty(&file).unwrap());
    let entries: Vec<_> = history
        .iter()
        .map(|h| serde_json::json!({"Time": h.seconds, "Incumbent": h.k, "Error": 0.0, "Method": h.method}))
        .collect();
    write(format!("{}_history.json", inst.id), serde_json::to_string_pretty(&entries).unwrap());
}

/// Result of a run.
pub struct Solution {
    /// Stage of the bound chain that gave the final lower bound ("degree", "two-line" or "sat").
    pub lower_bound_method: String,
    pub best: Decomposition,
    pub history: Vec<Improvement>,
    pub lower_bound: usize,
    pub seconds: f64,
    /// seconds until the final best decomposition was found
    pub time_to_solution: f64,
}

/// State shared by all threads of a run.
struct Shared<'a> {
    inst: &'a Instance,
    checkpoint: Option<std::path::PathBuf>,
    start: Instant,
    deadline: Instant,
    state: Mutex<State>,
    done: Arc<AtomicBool>,
}

struct State {
    best: Decomposition,
    history: Vec<Improvement>,
    lower_bound: usize,
    lower_bound_method: &'static str,
}

impl Shared<'_> {
    /// Accepts `dec` as the new best decomposition if it is exact and shorter.
    fn offer(&self, dec: &Decomposition, method: &str) {
        if dec.verify(self.inst).is_err() {
            debug_assert!(false, "{method} produced an invalid decomposition");
            return;
        }
        let mut state = self.state.lock().unwrap();
        if dec.len() < state.best.len() {
            state.history.push(Improvement {
                seconds: self.start.elapsed().as_secs_f64(),
                k: dec.len(),
                method: method.into(),
            });
            state.best = dec.clone();
            if state.best.len() <= state.lower_bound {
                self.done.store(true, Ordering::Relaxed);
            }
            if let Some(dir) = &self.checkpoint {
                write_checkpoint(dir, self.inst, &state.best, &state.history, state.best.len() <= state.lower_bound);
            }
        }
    }

    /// Raises the shared lower bound L; the run is over when L reaches the length of the best decomposition.
    fn raise_lower_bound(&self, lb: usize, method: &'static str) {
        let mut state = self.state.lock().unwrap();
        if lb > state.lower_bound {
            state.lower_bound = lb;
            state.lower_bound_method = method;
        }
        if state.best.len() <= state.lower_bound {
            self.done.store(true, Ordering::Relaxed);
            if let Some(dir) = &self.checkpoint {
                write_checkpoint(dir, self.inst, &state.best, &state.history, true);
            }
        }
    }

    fn best(&self) -> Decomposition {
        self.state.lock().unwrap().best.clone()
    }

    /// Length of the best decomposition.
    fn k(&self) -> usize {
        self.state.lock().unwrap().best.len()
    }

    fn running(&self) -> bool {
        !self.done.load(Ordering::Relaxed) && Instant::now() < self.deadline
    }

    fn remaining(&self) -> f32 {
        self.deadline.saturating_duration_since(Instant::now()).as_secs_f32()
    }
}

/// One run of RCS on an instance (Algorithm 1 of the article).
pub fn solve(inst: &Instance, config: &Config) -> Solution {
    let start = Instant::now();
    let deadline = start + config.budget;
    let n = inst.n;
    // the first decomposition of every run: the Birkhoff algorithm with the zeroing rule
    let first = decompose(inst, Rule::Zeroing);
    let history =
        vec![Improvement { seconds: start.elapsed().as_secs_f64(), k: first.len(), method: "greedy-zeroing".into() }];
    let run = Shared {
        inst,
        checkpoint: config.checkpoint.clone(),
        start,
        deadline,
        state: Mutex::new(State { best: first, history, lower_bound: inst.degree(), lower_bound_method: "degree" }),
        done: Arc::new(AtomicBool::new(false)),
    };
    if run.k() <= run.state.lock().unwrap().lower_bound {
        run.done.store(true, Ordering::Relaxed);
    }
    let sparse = inst.treated_as_sparse();
    let run_seed = config.seed;
    // dense n <= 7: the beam search over all permutations finishes its first passes within seconds, so the cases of
    // the SAT certificates take over threads (3 for n <= 5, 2 for n = 6, 7), and the beam search keeps the rest
    let case_threads = if sparse || n > 7 {
        1
    } else if n <= 5 {
        3
    } else {
        2
    };
    let shared = &run;
    let mut threads: Vec<Box<dyn FnOnce() + Send + '_>> = Vec::new();
    // Certify: the bound chain, inside the run
    threads.push(Box::new(move || {
        let upper = || shared.k();
        let mut raise = |lb: usize, method: &'static str| {
            shared.raise_lower_bound(lb, method);
            shared.running()
        };
        let mut solution = |d: &Decomposition| shared.offer(d, "sat");
        let mut hooks = crate::bounds::ChainHooks {
            upper: &upper,
            raise: &mut raise,
            solution: &mut solution,
            stop: Some(shared.done.clone()),
            case_threads,
        };
        crate::bounds::run_bound_chain(shared.inst, shared.deadline, &mut hooks);
        // matrices treated as dense: the chain ends early (no SAT stage for n >= 10), and the thread joins the
        // solution chain
        if !sparse && n >= 24 {
            restart_thread(shared, seed_of(run_seed, 101));
        } else if !sparse && n >= 10 {
            lns_thread(shared, seed_of(run_seed, 102));
        }
    }));
    if sparse {
        threads.push(Box::new(move || recover_thread(shared)));
        // largest weight first on the same weights, next to fixing by unique subset sums (faster, but it can fail, so
        // its result is checked like every other offer)
        threads.push(Box::new(move || {
            if let Ok(w) = frequency_rule(shared.inst) {
                if let Ok(d) = largest_weight_first(shared.inst, &w) {
                    shared.offer(&d, "frequency-weights+peel");
                }
            }
        }));
    }
    if n <= 4 {
        threads.push(Box::new(move || sat_model_thread(shared)));
    }
    if !sparse {
        let beam_threads = config.threads.saturating_sub(3 + case_threads - 1).max(1);
        let seed = config.seed;
        threads.push(Box::new(move || beam_thread(shared, beam_threads, seed)));
        threads.push(Box::new(move || tail_thread(shared)));
    }
    // treated as dense: Certify + beam search + tail + one large neighborhood search; treated as sparse: all
    // remaining threads run the large neighborhood search
    let used = if sparse { threads.len() } else { config.threads.saturating_sub(1).max(threads.len()) };
    let lns_threads = config.threads.saturating_sub(used).max(1);
    for index in 0..lns_threads {
        let seed = seed_of(config.seed, index as u64 + 1);
        threads.push(Box::new(move || lns_thread(shared, seed)));
    }
    std::thread::scope(|scope| {
        for thread in threads {
            scope.spawn(thread);
        }
    });
    let state = run.state.into_inner().unwrap();
    let time_to_solution = state.history.last().map_or(0.0, |h| h.seconds);
    Solution {
        best: state.best,
        history: state.history,
        lower_bound: state.lower_bound,
        lower_bound_method: state.lower_bound_method.to_string(),
        seconds: start.elapsed().as_secs_f64(),
        time_to_solution,
    }
}

/// Seed of a thread, derived from the seed of the run.
fn seed_of(seed: u64, thread: u64) -> u64 {
    seed.wrapping_mul(0x9e37_79b9).wrapping_add(thread)
}

/// Rule and noise of the randomized restarts: the logarithmic tie-break rule for matrices treated as dense with
/// n >= 12, where it gives shorter decompositions, and the zeroing rule otherwise.
fn restart_rule(inst: &Instance) -> (Rule, i64) {
    if inst.n >= 12 && !inst.treated_as_sparse() {
        (Rule::LogTieBreak, 300)
    } else {
        (Rule::Zeroing, 500)
    }
}

/// Randomized restarts with the logarithmic tie-break rule (treated as dense, n >= 24, where a restart takes
/// milliseconds to a second and the first level of the beam search takes minutes).
fn restart_thread(shared: &Shared, seed: u64) {
    let inst = shared.inst;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    shared.offer(&decompose(inst, Rule::LogTieBreak), "greedy-digits");
    while shared.running() {
        let d = decompose_randomized(&inst.d, inst.n, Rule::LogTieBreak, &mut rng, 300);
        shared.offer(&d, "greedy-randomized");
    }
}

/// Recover: the frequency rule with fixing by unique subset sums, then the enumeration of weight sets while the best
/// decomposition has more than n terms.
fn recover_thread(shared: &Shared) {
    let inst = shared.inst;
    if let Ok(w) = frequency_rule(inst) {
        if let crate::fixing::FixingResult::Found(d) = crate::fixing::Fixing::new(inst, &w).solve() {
            shared.offer(&d, "frequency-weights+sweep");
        }
    }
    if shared.running() && shared.k() > inst.n {
        let deadline = shared.deadline.min(Instant::now() + Duration::from_secs_f32(shared.remaining() * 0.4));
        enumerate_weight_sets(inst, usize::MAX, deadline, &|| shared.running(), &mut |d| {
            shared.offer(d, "weight-search+peel")
        });
    }
}

/// Small matrices (n <= 4): the SAT model for k = L, L + 1, ... below the length of the best decomposition. The first
/// satisfiable k gives a decomposition. (The proofs of optimality come from the Certify thread.)
fn sat_model_thread(shared: &Shared) {
    let inst = shared.inst;
    let mut k = shared.state.lock().unwrap().lower_bound;
    while shared.running() && k < shared.k() {
        match decompose_exactly_until(&inst.d, inst.n, k, shared.deadline, Some(shared.done.clone())) {
            SatResult::Sat(d) => {
                shared.offer(&d, "sat-exact");
                return;
            }
            SatResult::Unsat => k += 1,
            SatResult::Unknown => return,
        }
    }
}

/// Beam search over zeroing steps with growing width. When the tree is exhausted, the search starts again from D with
/// three times the width and a new seed.
fn beam_thread(shared: &Shared, threads: usize, seed: u64) {
    let inst = shared.inst;
    let n = inst.n;
    let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(threads).build() else { return };
    let perms = if n <= 7 { pattern_permutations(&inst.d, n, 5040) } else { None };
    // initial width: 200 000 divided by the number of permutations for n <= 7, then 30, 10, 3 and 1
    let (successors, mut width) = match &perms {
        Some(p) => (Successors::AllPermutations(p), (200_000 / p.len().max(1)).max(2)),
        None => (
            Successors::PerPosition { max_successors: usize::MAX },
            if n <= 12 {
                30
            } else if n <= 16 {
                10
            } else if n <= 32 {
                3
            } else {
                1
            },
        ),
    };
    // greedy completions with the zeroing rule: the logarithmic tie-break rule gave the same total length in our tests
    let completion_rule = Rule::Zeroing;
    let running = || shared.running();
    let mut round = 0u64;
    while shared.running() {
        pool.install(|| {
            let beam = Beam {
                successors: successors.clone(),
                width,
                seed: seed ^ round,
                best_length: shared.k(),
                completion_rule,
            };
            beam_search(&inst.d, n, &beam, &running, &mut |d| shared.offer(d, "zbeam"));
        });
        // three times wider each round, up to the limit of successors per level (then only the seed changes)
        width = width.saturating_mul(3).min(MAX_SUCCESSORS);
        round += 1;
    }
}

/// SAT re-optimization of the tail with the caps 15, 31 and 63, repeated whenever the best decomposition changed.
fn tail_thread(shared: &Shared) {
    let mut tried = usize::MAX;
    while shared.running() {
        let k = shared.k();
        if k == tried {
            std::thread::sleep(Duration::from_millis(20));
            continue;
        }
        tried = k;
        for cap in [15i64, 31, 63] {
            if !shared.running() {
                return;
            }
            let mut current = shared.best();
            let deadline =
                shared.deadline.min(Instant::now() + Duration::from_secs_f32((shared.remaining() * 0.3).min(10.0)));
            if reoptimize_tail(&mut current, shared.inst.n, cap, deadline, Some(&shared.done)) {
                shared.offer(&current, "sat-endgame");
            }
        }
    }
}

/// Large neighborhood search on the best decomposition, with a randomized restart in every fourth round.
fn lns_thread(shared: &Shared, seed: u64) {
    let inst = shared.inst;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let mut round = 0u64;
    while shared.running() {
        round += 1;
        if round % 4 == 1 {
            let (rule, noise) = restart_rule(inst);
            let d = decompose_randomized(&inst.d, inst.n, rule, &mut rng, noise);
            shared.offer(&d, "greedy-randomized");
        }
        let mut current = shared.best();
        let slice = shared.deadline.min(Instant::now() + Duration::from_millis(500));
        crate::lns::improve(inst, &mut current, slice, &|| shared.running(), &mut rng, &mut |d| shared.offer(d, "lns"));
    }
}
