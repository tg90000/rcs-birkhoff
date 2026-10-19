//! Compares the results with all QOBLIB problem-03 submissions. It draws the plots of
//! https://zib-aopt.github.io/QOBLIB/problem/03/ (running time and time to solution against the number of solved
//! instances, performance profile, time to solution against the size) with one line per submission plus this work,
//! and writes a table per family.
//!
//!     report [--ours results/final] [--bounds FILE] [--out results/report]
//!
//! It also writes the table per instance (lower bound and best decomposition side by side, with the methods behind
//! them, the gap, the times and the best previously known exact value) as `instances.{csv,md,tex,json}`, and the plots
//! `bounds.svg` and `tts_ours.svg`.
//!
//! An instance counts as solved if the objective is at most the best known value. There are two reference values: the
//! best previously known exact value from the QOBLIB `solutions/` folder, and the best value including this work.
//! Submissions with approximate decompositions (an allowed error > 0) are marked "(approx.)".

use plotters::coord::CoordTranslate;
use plotters::prelude::*;
use rcs_birkhoff::cli::Args;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// The result of one submission on one instance.
#[derive(Clone, Debug)]
struct Run {
    instance: String,
    n: usize,
    objective: f64,
    runtime: f64,
    time_to_solution: f64,
    proven_optimal: bool,
}

/// One submission: its label and its results.
#[derive(Clone, Debug)]
struct Method {
    name: String,
    approximate: bool,
    runs: Vec<Run>,
}

const THIS_WORK: &str = "This work";
/// Red for this work, as in the article; the other submissions use the tab10 palette.
const OURS: RGBColor = RGBColor(0xd6, 0x27, 0x28);
const PALETTE: [RGBColor; 10] = [
    RGBColor(0x1f, 0x77, 0xb4),
    RGBColor(0xff, 0x7f, 0x0e),
    RGBColor(0x2c, 0xa0, 0x2c),
    RGBColor(0x94, 0x67, 0xbd),
    RGBColor(0x8c, 0x56, 0x4b),
    RGBColor(0xe3, 0x77, 0xc2),
    RGBColor(0x7f, 0x7f, 0x7f),
    RGBColor(0xbc, 0xbd, 0x22),
    RGBColor(0x17, 0xbe, 0xcf),
    RGBColor(0x39, 0x3b, 0x79),
];
const BLUE: RGBColor = RGBColor(0x1f, 0x77, 0xb4);
const GREY: RGBColor = RGBColor(0x7f, 0x7f, 0x7f);

/// Times below this value are drawn at this value on the logarithmic axes.
const MIN_TIME: f64 = 1e-4;

/// Splits a CSV line at the commas outside double quotes.
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    for ch in line.chars() {
        match ch {
            '"' => quoted = !quoted,
            ',' if !quoted => fields.push(std::mem::take(&mut field)),
            c => field.push(c),
        }
    }
    fields.push(field);
    fields
}

/// The size n of an instance id such as "B16_256_3".
fn size_of(id: &str) -> usize {
    id.trim_start_matches('B').split('_').next().and_then(|x| x.parse().ok()).unwrap_or(0)
}

/// The family of an instance id: "B16_256_3" gives "B16_256".
fn family(id: &str) -> String {
    let parts: Vec<&str> = id.split('_').collect();
    format!("{}_{}", parts[0], parts[1])
}

fn number(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

/// The submissions in the QOBLIB folder `submissions/<folder>/<instance>/<instance>_summary.csv`.
fn load_submissions(dir: &Path) -> Vec<Method> {
    let mut methods = Vec::new();
    let mut folders: Vec<PathBuf> =
        std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    folders.sort();
    for folder in folders {
        let name = folder.file_name().unwrap().to_string_lossy().to_string();
        let mut runs = Vec::new();
        let mut approximate = false;
        let Ok(entries) = std::fs::read_dir(&folder) else { continue };
        for instance_dir in entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
            let id = instance_dir.file_name().unwrap().to_string_lossy().to_string();
            let Ok(text) = std::fs::read_to_string(instance_dir.join(format!("{id}_summary.csv"))) else { continue };
            let mut lines = text.lines();
            let header = parse_csv_line(lines.next().unwrap_or(""));
            let Some(row) = lines.next().map(parse_csv_line) else { continue };
            let column = |key: &str| {
                header.iter().position(|h| h.trim() == key).and_then(|i| row.get(i)).map(|s| s.trim().to_string())
            };
            let Some(objective) = column("Best Objective Value").and_then(|s| number(&s)) else { continue };
            let runtime = column("Total Runtime").and_then(|s| number(&s)).unwrap_or(f64::NAN);
            let time_to_solution = column("Time to Solution").and_then(|s| number(&s)).unwrap_or(runtime);
            let threshold = column("Success Threshold").and_then(|s| number(&s)).unwrap_or(0.0);
            let bound = column("Optimality Bound").and_then(|s| number(&s));
            let paradigm = column("Paradigm").unwrap_or_default();
            // a positive threshold allows an error in the decomposition, except for the MIP submission, where it is
            // the MIP gap; quantum submissions give approximate decompositions
            if (threshold > 0.0 && !name.contains("MIP")) || paradigm.to_lowercase().contains("quantum") {
                approximate = true;
            }
            runs.push(Run {
                n: size_of(&id),
                proven_optimal: bound.is_some_and(|b| (b - objective).abs() < 1e-9),
                instance: id,
                objective,
                runtime,
                time_to_solution,
            });
        }
        if !runs.is_empty() {
            let short = name.split_once('_').map(|x| x.1).unwrap_or(&name).to_string();
            let year = &name[..4];
            methods.push(Method { name: format!("{short} ({year})"), approximate, runs });
        }
    }
    methods
}

/// Output of the `bound` binary: instance -> (lower bound, stage, seconds, trace of the stages).
fn load_bounds(path: &Path) -> HashMap<String, (usize, String, f64, String)> {
    let Ok(text) = std::fs::read_to_string(path) else { return HashMap::new() };
    text.lines()
        .skip(1)
        .map(parse_csv_line)
        .map(|f| {
            let seconds = f[4].parse().unwrap_or(f64::NAN);
            (f[0].clone(), (f[2].parse().unwrap(), f[3].clone(), seconds, f.get(6).cloned().unwrap_or_default()))
        })
        .collect()
}

/// The results of this work as a submission.
fn load_ours(dir: &Path) -> Method {
    let text = std::fs::read_to_string(dir.join("summary.csv")).expect("summary.csv of the results");
    let mut lines = text.lines();
    let header = parse_csv_line(lines.next().unwrap());
    let column = |key: &str| header.iter().position(|h| h == key).unwrap();
    let runs = lines
        .map(parse_csv_line)
        .map(|r| Run {
            instance: r[column("instance")].clone(),
            n: r[column("n")].parse().unwrap(),
            objective: r[column("k")].parse().unwrap(),
            runtime: r[column("seconds")].parse().unwrap(),
            time_to_solution: r[column("time_to_solution")].parse().unwrap(),
            proven_optimal: r[column("k")] == r[column("lower_bound")],
        })
        .collect();
    Method { name: THIS_WORK.into(), approximate: false, runs }
}

/// One row of the table per instance: the lower bound next to the best decomposition.
struct Row {
    instance: String,
    class: &'static str,
    n: usize,
    lower_bound: usize,
    bound_method: String,
    bound_seconds: f64,
    bound_stages: String,
    k: usize,
    timed_out: bool,
    solution_method: String,
    time_to_solution: f64,
    runtime: f64,
    previous_best: Option<usize>,
}

fn instance_rows(
    dir: &Path,
    bounds: &HashMap<String, (usize, String, f64, String)>,
    previous_best: &HashMap<String, f64>,
) -> Vec<Row> {
    let text = std::fs::read_to_string(dir.join("summary.csv")).expect("summary.csv of the results");
    let mut lines = text.lines();
    let header = parse_csv_line(lines.next().unwrap());
    let column = |key: &str| header.iter().position(|h| h == key);
    let mut rows: Vec<Row> = lines
        .map(parse_csv_line)
        .map(|r| {
            let field = |key: &str| r[column(key).unwrap()].clone();
            let id = field("instance");
            let n: usize = field("n").parse().unwrap();
            let generators: usize = id.split('_').nth(1).unwrap().parse().unwrap();
            let run_bound: usize = field("lower_bound").parse().unwrap();
            let run_method = column("lower_bound_method").map_or("solve".to_string(), |c| r[c].clone());
            let (lower_bound, bound_method, bound_seconds, bound_stages) = match bounds.get(&id) {
                Some((b, method, t, stages)) if *b > run_bound => (*b, method.clone(), *t, stages.clone()),
                Some((b, _, t, stages)) if *b == run_bound => (run_bound, run_method, *t, stages.clone()),
                _ => (run_bound, run_method, f64::NAN, String::new()),
            };
            // merged results mark the instances without a proof in the last round; a single run marks every
            // instance without a proof
            let timed_out = match column("timed_out_600s") {
                Some(c) => r[c] == "true",
                None => column("proven_optimal").is_some_and(|c| r[c] != "true"),
            };
            Row {
                class: if generators == n * n && n > 1 { "dense" } else { "sparse" },
                n,
                lower_bound,
                bound_method,
                bound_seconds,
                bound_stages,
                timed_out,
                k: field("k").parse().unwrap(),
                solution_method: field("method"),
                time_to_solution: field("time_to_solution").parse().unwrap(),
                runtime: field("seconds").parse().unwrap(),
                previous_best: previous_best.get(&id).map(|&v| v as usize),
                instance: id,
            }
        })
        .collect();
    rows.sort_by_key(|r| {
        let index = r.instance.split('_').nth(2).and_then(|x| x.parse::<usize>().ok()).unwrap_or(0);
        (r.class == "dense", r.n, index)
    });
    rows
}

/// Writes the table per instance as CSV, Markdown, LaTeX (rows for the article) and JSON (for the visualization).
fn write_instance_tables(out: &Path, rows: &[Row], ours_dir: &Path, previous_files: &HashMap<String, PathBuf>) {
    let optional = |v: Option<usize>| v.map_or(String::new(), |x| x.to_string());
    let comparison = |r: &Row| match r.previous_best {
        None => "new".to_string(),
        Some(p) if r.k < p => format!("better ({})", r.k as i64 - p as i64),
        Some(p) if r.k == p => "equal".to_string(),
        Some(p) => format!("worse (+{})", r.k - p),
    };
    let mut csv = String::from(
        "instance,regime,n,lower_bound,lower_bound_method,bound_seconds,k,solution_method,gap,proven_optimal,\
         time_to_solution,runtime,previous_best_exact,vs_previous\n",
    );
    let mut markdown = String::from(
        "| instance | n | lower bound | k (ours) | gap | proven | previous best (exact) | vs previous | time to \
         solution [s] | bound method | solution method |\n|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    let mut latex = String::new();
    let mut json = Vec::new();
    let mut last_family = String::new();
    for r in rows {
        let gap = r.k - r.lower_bound.min(r.k);
        let proven = r.k <= r.lower_bound;
        csv.push_str(&format!(
            "{},{},{},{},{},{:.3},{},{},{},{},{:.4},{:.4},{},{}\n",
            r.instance,
            r.class,
            r.n,
            r.lower_bound,
            r.bound_method,
            r.bound_seconds,
            r.k,
            r.solution_method,
            gap,
            proven,
            r.time_to_solution,
            r.runtime,
            optional(r.previous_best),
            comparison(r)
        ));
        markdown.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {:.3} | {} | {} |\n",
            r.instance,
            r.n,
            r.lower_bound,
            r.k,
            gap,
            if proven { "yes" } else { "" },
            optional(r.previous_best),
            comparison(r),
            r.time_to_solution,
            r.bound_method,
            r.solution_method
        ));
        let current_family = family(&r.instance);
        if current_family != last_family && !last_family.is_empty() {
            latex.push_str("\\midrule\n");
        }
        last_family = current_family;
        latex.push_str(&format!(
            "\\texttt{{{}}} & {} & {} & {} & {} & {:.2} & {:.2} \\\\\n",
            r.instance.replace('_', "\\_"),
            r.lower_bound,
            r.k,
            gap,
            if proven { "$\\checkmark$" } else { "" },
            r.time_to_solution,
            r.runtime
        ));
        json.push(serde_json::json!({
            "instance": r.instance,
            "regime": r.class,
            "n": r.n,
            "lower_bound": r.lower_bound,
            "lower_bound_method": r.bound_method,
            "k": r.k,
            "solution_method": r.solution_method,
            "gap": gap,
            "proven_optimal": proven,
            "time_to_solution": r.time_to_solution,
            "runtime": r.runtime,
            "previous_best_exact": r.previous_best,
            "bound_stages": r.bound_stages,
            "bound_seconds": r.bound_seconds,
            "timed_out": r.timed_out,
            // paths relative to the folder of the visualization pages, which link to the decomposition viewer
            "solution_file": format!("../{}/{}/{}_solution.json", ours_dir.display(), r.instance, r.instance),
            "previous_file": previous_files.get(&r.instance).map(|p| format!("../{}", p.display())),
        }));
    }
    std::fs::write(out.join("instances.csv"), csv).unwrap();
    std::fs::write(out.join("instances.md"), markdown).unwrap();
    std::fs::write(out.join("instances.tex"), latex).unwrap();
    std::fs::write(out.join("instances.json"), serde_json::to_string_pretty(&json).unwrap()).unwrap();
}

/// Draws the legend of a chart in the common style.
fn draw_legend<'a, DB: DrawingBackend + 'a, CT: CoordTranslate>(
    chart: &mut ChartContext<'a, DB, CT>,
    position: SeriesLabelPosition,
    font_size: u32,
) {
    chart
        .configure_series_labels()
        .position(position)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", font_size))
        .draw()
        .unwrap();
}

/// Time to the best decomposition per instance against n (both logarithmic), sparse and dense, with the median per n.
fn tts_plot(path: &Path, rows: &[Row]) {
    let root = SVGBackend::new(path, (1000, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Time to the best decomposition", ("sans-serif", 30))
        .margin(14)
        .x_label_area_size(60)
        .y_label_area_size(90)
        .build_cartesian_2d((2.5f64..120f64).log_scale(), (MIN_TIME..1e3f64).log_scale())
        .unwrap();
    chart
        .configure_mesh()
        .x_desc("matrix size n (log)")
        .y_desc("seconds (log)")
        .x_labels(6)
        .x_label_formatter(&|x| format!("{x:.0}"))
        .y_label_formatter(&|y| {
            if *y >= 1.0 {
                format!("{y:.0}")
            } else {
                format!("{y:.4}").trim_end_matches('0').to_string()
            }
        })
        .label_style(("sans-serif", 20))
        .axis_desc_style(("sans-serif", 22))
        .draw()
        .unwrap();
    for (class, color) in [("sparse", BLUE), ("dense", OURS)] {
        let points: Vec<&Row> = rows.iter().filter(|r| r.class == class).collect();
        chart
            .draw_series(
                points
                    .iter()
                    .map(|r| Circle::new((r.n as f64, r.time_to_solution.max(MIN_TIME)), 3, color.mix(0.45).filled())),
            )
            .unwrap();
        let mut by_size: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        for r in &points {
            by_size.entry(r.n).or_default().push(r.time_to_solution.max(MIN_TIME));
        }
        let medians: Vec<(f64, f64)> = by_size.into_iter().map(|(n, times)| (n as f64, median(times))).collect();
        chart
            .draw_series(LineSeries::new(medians, color.stroke_width(3)))
            .unwrap()
            .label(format!("{class} (median per n)"))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(3)));
    }
    draw_legend(&mut chart, SeriesLabelPosition::UpperLeft, 20);
    root.present().unwrap();
}

/// Lower bound and best decomposition per instance, divided by n; with `with_previous` also the best previously known
/// exact value.
fn bounds_plot(path: &Path, rows: &[Row], with_previous: bool) {
    let root = SVGBackend::new(path, (1400, 620)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let y_max = rows
        .iter()
        .map(|r| r.k.max(if with_previous { r.previous_best.unwrap_or(0) } else { 0 }) as f64 / r.n as f64)
        .fold(1.0, f64::max)
        * 1.08;
    let mut chart = ChartBuilder::on(&root)
        .caption("Lower bound and best decomposition per instance (divided by n)", ("sans-serif", 30))
        .margin(14)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(0f64..rows.len() as f64, 0f64..y_max)
        .unwrap();
    chart
        .configure_mesh()
        .x_desc("instances (sparse, then dense; by n)")
        .y_desc("number of permutations / n")
        .label_style(("sans-serif", 20))
        .axis_desc_style(("sans-serif", 22))
        .x_label_formatter(&|x| format!("{x:.0}"))
        .disable_x_mesh()
        .light_line_style(WHITE.mix(0.0))
        .draw()
        .unwrap();
    let x = |i: usize| i as f64 + 0.5;
    let per_n = |value: usize, r: &Row| value as f64 / r.n as f64;
    let gaps: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            PathElement::new(
                vec![(x(i), per_n(r.lower_bound, r)), (x(i), per_n(r.k, r))],
                OURS.mix(0.35).stroke_width(1),
            )
        })
        .collect();
    chart.draw_series(gaps).unwrap();
    if with_previous {
        chart
            .draw_series(
                rows.iter().enumerate().filter_map(|(i, r)| {
                    r.previous_best.map(|p| Cross::new((x(i), per_n(p, r)), 3, GREY.stroke_width(1)))
                }),
            )
            .unwrap()
            .label("previous best exact k")
            .legend(move |(x, y)| Cross::new((x + 10, y), 3, GREY.stroke_width(1)));
    }
    chart
        .draw_series(
            rows.iter()
                .enumerate()
                .map(|(i, r)| TriangleMarker::new((x(i), per_n(r.lower_bound, r)), 4, BLUE.filled())),
        )
        .unwrap()
        .label("lower bound L")
        .legend(move |(x, y)| TriangleMarker::new((x + 10, y), 4, BLUE.filled()));
    chart
        .draw_series(rows.iter().enumerate().map(|(i, r)| Circle::new((x(i), per_n(r.k, r)), 3, OURS.filled())))
        .unwrap()
        .label("best decomposition k")
        .legend(move |(x, y)| Circle::new((x + 10, y), 3, OURS.filled()));
    draw_legend(&mut chart, SeriesLabelPosition::UpperLeft, 20);
    root.present().unwrap();
}

fn label(method: &Method) -> String {
    if method.approximate {
        format!("{} (approx.)", method.name)
    } else {
        method.name.clone()
    }
}

fn color_of(index: usize, method: &Method) -> RGBColor {
    if method.name == THIS_WORK {
        OURS
    } else {
        PALETTE[index % PALETTE.len()]
    }
}

fn line_width(method: &Method) -> u32 {
    if method.name == THIS_WORK {
        3
    } else {
        2
    }
}

/// Running time (or time to solution) against the number of instances solved, one line per method.
fn cactus(path: &Path, methods: &[Method], best: &HashMap<String, f64>, use_tts: bool, title: &str) {
    let root = SVGBackend::new(path, (1000, 640)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let total = best.len() as f64;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(14)
        .x_label_area_size(46)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..total, (MIN_TIME..1e5f64).log_scale())
        .unwrap();
    chart
        .configure_mesh()
        .x_desc("instances solved (objective <= best known) →")
        .y_desc(if use_tts { "time to solution (s, log)" } else { "runtime (s, log)" })
        .x_label_formatter(&|x| format!("{x:.0}"))
        .draw()
        .unwrap();
    for (i, method) in methods.iter().enumerate() {
        let mut times: Vec<f64> = method
            .runs
            .iter()
            .filter(|r| best.get(&r.instance).is_some_and(|&b| r.objective <= b + 1e-9))
            .map(|r| (if use_tts { r.time_to_solution } else { r.runtime }).max(MIN_TIME))
            .filter(|t| t.is_finite())
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let points: Vec<(f64, f64)> = times.iter().enumerate().map(|(x, &t)| (x as f64 + 1.0, t)).collect();
        let color = color_of(i, method);
        let style = ShapeStyle::from(&color).stroke_width(line_width(method));
        if method.approximate {
            chart.draw_series(DashedLineSeries::new(points.clone(), 6, 4, style)).unwrap();
        } else {
            chart.draw_series(LineSeries::new(points.clone(), style)).unwrap();
        }
        chart
            .draw_series(points.last().map(|&p| Circle::new(p, 4, color.filled())))
            .unwrap()
            .label(format!("{} — {} solved", label(method), times.len()))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(3)));
    }
    draw_legend(&mut chart, SeriesLabelPosition::UpperLeft, 13);
    root.present().unwrap();
}

/// Performance profile: the share of instances within a given gap from the best known value, one line per method.
fn profile(path: &Path, methods: &[Method], best: &HashMap<String, f64>, title: &str) {
    let root = SVGBackend::new(path, (1000, 640)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let total = best.len() as f64;
    // gaps on a logarithmic axis from 0.1 % to 1000 %; a gap of zero is drawn at the left edge
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(14)
        .x_label_area_size(46)
        .y_label_area_size(60)
        .build_cartesian_2d((0.1f64..1000f64).log_scale(), 0f64..100f64)
        .unwrap();
    chart
        .configure_mesh()
        .x_desc("optimality gap from best known (%, log; leftmost = exactly best known) →")
        .y_desc("instances within gap (% of all 375)")
        .y_label_formatter(&|y| format!("{y:.0}"))
        .draw()
        .unwrap();
    for (i, method) in methods.iter().enumerate() {
        let gaps: Vec<f64> = method
            .runs
            .iter()
            .filter_map(|r| best.get(&r.instance).map(|&b| 100.0 * (r.objective - b).max(0.0) / b))
            .collect();
        let points: Vec<(f64, f64)> = (0..=60)
            .map(|t| 0.1 * 10f64.powf(t as f64 / 15.0))
            .map(|x| (x, 100.0 * gaps.iter().filter(|&&g| g <= x || g < 1e-9).count() as f64 / total))
            .collect();
        let color = color_of(i, method);
        let style = ShapeStyle::from(&color).stroke_width(line_width(method));
        if method.approximate {
            chart.draw_series(DashedLineSeries::new(points, 6, 4, style)).unwrap()
        } else {
            chart.draw_series(LineSeries::new(points, style)).unwrap()
        }
        .label(label(method))
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(3)));
    }
    draw_legend(&mut chart, SeriesLabelPosition::LowerRight, 13);
    root.present().unwrap();
}

/// Time to solution against the size n, every instance as a point and the median per n as a line, per method.
fn scaling(path: &Path, methods: &[Method], title: &str) {
    let root = SVGBackend::new(path, (1000, 640)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(14)
        .x_label_area_size(46)
        .y_label_area_size(70)
        .build_cartesian_2d((2.5f64..120f64).log_scale(), (MIN_TIME..1e5f64).log_scale())
        .unwrap();
    chart.configure_mesh().x_desc("matrix size n (log) →").y_desc("time to solution (s, log)").draw().unwrap();
    for (i, method) in methods.iter().enumerate() {
        let color = color_of(i, method);
        let mut by_size: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        for r in method.runs.iter().filter(|r| r.time_to_solution.is_finite()) {
            by_size.entry(r.n).or_default().push(r.time_to_solution.max(MIN_TIME));
        }
        let medians: Vec<(f64, f64)> = by_size.into_iter().map(|(n, times)| (n as f64, median(times))).collect();
        chart
            .draw_series(
                method
                    .runs
                    .iter()
                    .filter(|r| r.time_to_solution.is_finite())
                    .map(|r| Circle::new((r.n as f64, r.time_to_solution.max(MIN_TIME)), 2, color.mix(0.35).filled())),
            )
            .unwrap();
        chart
            .draw_series(LineSeries::new(medians, ShapeStyle::from(&color).stroke_width(line_width(method))))
            .unwrap()
            .label(format!("{} (median per n)", label(method)))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(3)));
    }
    draw_legend(&mut chart, SeriesLabelPosition::UpperLeft, 13);
    root.present().unwrap();
}

/// The table per family: the mean length of every method, the previous best known mean and the comparison.
fn family_table(methods: &[Method], ours: &Method, previous_best: &HashMap<String, f64>) -> String {
    let mut families: BTreeMap<(usize, usize), Vec<String>> = BTreeMap::new();
    for r in &ours.runs {
        let generators: usize = family(&r.instance).split('_').nth(1).unwrap().parse().unwrap();
        families.entry((r.n, generators)).or_default().push(r.instance.clone());
    }
    let mut csv = String::from(
        "family,instances,previous_best_known_mean,ours_mean,ours_better,ours_equal,ours_worse,ours_proven_optimal",
    );
    for method in methods {
        csv.push_str(&format!(",{}", label(method).replace(',', ";")));
    }
    csv.push('\n');
    let ours_by_id: HashMap<&str, &Run> = ours.runs.iter().map(|r| (r.instance.as_str(), r)).collect();
    let mean = |values: Vec<f64>| {
        if values.is_empty() {
            String::new()
        } else {
            format!("{:.1}", values.iter().sum::<f64>() / values.len() as f64)
        }
    };
    for ((n, generators), ids) in &families {
        let ours_k = |id: &String| ours_by_id[id.as_str()].objective;
        let count = |compare: fn(f64, f64) -> bool| {
            ids.iter().filter(|id| previous_best.get(*id).is_some_and(|&b| compare(ours_k(id), b))).count()
        };
        csv.push_str(&format!(
            "B{n}_{generators},{},{},{},{},{},{},{}",
            ids.len(),
            mean(ids.iter().filter_map(|id| previous_best.get(id).copied()).collect()),
            mean(ids.iter().map(ours_k).collect()),
            count(|a, b| a < b),
            count(|a, b| a == b),
            count(|a, b| a > b),
            ids.iter().filter(|id| ours_by_id[id.as_str()].proven_optimal).count()
        ));
        for method in methods {
            let values = method.runs.iter().filter(|r| ids.contains(&r.instance)).map(|r| r.objective).collect();
            csv.push_str(&format!(",{}", mean(values)));
        }
        csv.push('\n');
    }
    csv
}

fn main() {
    let args = Args::from_env();
    let ours_dir = PathBuf::from(args.string_or("--ours", "results/final"));
    // only an explicitly given bound file can raise the bounds of the runs
    let bounds = args.get("--bounds").map_or_else(HashMap::new, |path| load_bounds(Path::new(&path)));
    let out = PathBuf::from(args.string_or("--out", "results/report"));
    let qoblib = Path::new("data/qoblib/03-birkhoff");
    std::fs::create_dir_all(&out).unwrap();

    let mut methods = load_submissions(&qoblib.join("submissions"));
    let ours = load_ours(&ours_dir);
    let best_known = rcs_birkhoff::reference::best_known(&qoblib.join("solutions"));
    let previous_best: HashMap<String, f64> =
        best_known.iter().map(|(id, (k, _, _))| (id.clone(), *k as f64)).collect();
    let previous_files: HashMap<String, PathBuf> =
        best_known.into_iter().map(|(id, (_, _, file))| (id, file)).collect();
    // reference values for all 375 instances: the best including this work, and the previous best (this work where
    // no exact value was known)
    let mut new_best: HashMap<String, f64> = HashMap::new();
    let mut previous_or_ours = previous_best.clone();
    for r in &ours.runs {
        let previous = previous_best.get(&r.instance).copied().unwrap_or(f64::INFINITY);
        new_best.insert(r.instance.clone(), previous.min(r.objective));
        previous_or_ours.entry(r.instance.clone()).or_insert(r.objective);
    }
    methods.push(ours.clone());

    cactus(
        &out.join("cactus_runtime.svg"),
        &methods,
        &new_best,
        false,
        "Runtime to reach the best known objective (incl. this work)",
    );
    cactus(
        &out.join("cactus_tts.svg"),
        &methods,
        &new_best,
        true,
        "Time to solution to reach the best known objective (incl. this work)",
    );
    cactus(
        &out.join("cactus_runtime_prev.svg"),
        &methods,
        &previous_or_ours,
        false,
        "Runtime to reach the previous best known (exact) objective",
    );
    profile(
        &out.join("profile.svg"),
        &methods,
        &new_best,
        "Solution quality (performance profile vs. best known incl. this work)",
    );
    profile(
        &out.join("profile_prev.svg"),
        &methods,
        &previous_or_ours,
        "Solution quality (performance profile vs. previous exact best known)",
    );
    scaling(&out.join("scaling.svg"), &methods, "Time-to-solution scaling with instance size");
    std::fs::write(out.join("families.csv"), family_table(&methods, &ours, &previous_best)).unwrap();

    let rows = instance_rows(&ours_dir, &bounds, &previous_best);
    write_instance_tables(&out, &rows, &ours_dir, &previous_files);
    bounds_plot(&out.join("bounds.svg"), &rows, true);
    bounds_plot(&out.join("bounds_ours.svg"), &rows, false);
    tts_plot(&out.join("tts_ours.svg"), &rows);
    println!("wrote {}", out.display());
}
