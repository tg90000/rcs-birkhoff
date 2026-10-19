//! Makes synthetic instances like the QOBLIB ones: k random permutations (k = n for "sparse", k = n^2 for "dense"),
//! with the gaps between k - 1 sorted random integers in [0, s] as weights. The generating permutations and weights are
//! stored in the same fields as in QOBLIB, for checks only; the solver never reads them.
//!
//!     generate --n 200 --kind sparse --count 3 --scale 10000000 --seed 1 --out data/synthetic/qbench_200_sparse.json

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rcs_birkhoff::cli::Args;

fn main() {
    let args = Args::from_env();
    let n: usize = args.get("--n").expect("--n is required").parse().unwrap();
    let kind = args.string_or("--kind", "sparse");
    let count: usize = args.parse_or("--count", 3);
    let scale: i64 = args.parse_or("--scale", 10_000_000);
    let seed: u64 = args.parse_or("--seed", 1);
    let out = args.get("--out").expect("--out is required");

    let k = if kind == "dense" { n * n } else { n };
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed ^ ((n as u64) << 20) ^ (k as u64));
    let mut file = serde_json::Map::new();
    for index in 1..=count {
        let mut cuts: Vec<i64> = (0..k - 1).map(|_| rng.gen_range(0..=scale)).collect();
        cuts.push(0);
        cuts.push(scale);
        cuts.sort_unstable();
        let weights: Vec<i64> = cuts.windows(2).map(|pair| pair[1] - pair[0]).collect();
        let mut d = vec![0i64; n * n];
        let mut permutations: Vec<u32> = Vec::with_capacity(k * n);
        let mut perm: Vec<u32> = (0..n as u32).collect();
        for &w in &weights {
            perm.shuffle(&mut rng);
            for (i, &j) in perm.iter().enumerate() {
                d[i * n + j as usize] += w;
            }
            permutations.extend(perm.iter().map(|&j| j + 1));
        }
        file.insert(
            index.to_string(),
            serde_json::json!({
                "id": format!("B{n}_{k}_{index}"),
                "n": n,
                "scale": scale,
                "scaled_doubly_stochastic_matrix": d,
                "weights": weights,
                "permutations": permutations,
            }),
        );
    }
    if let Some(dir) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::write(&out, serde_json::to_string(&serde_json::Value::Object(file)).unwrap()).unwrap();
}
