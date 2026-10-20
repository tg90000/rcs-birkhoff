# RCS: exact minimum Birkhoff decompositions

## Problem

A doubly stochastic matrix is a nonnegative matrix whose rows and columns all sum to one. By the theorem of Birkhoff
and von Neumann, it is a convex combination of permutation matrices. The minimum Birkhoff decomposition problem asks
for such a combination with as few permutation matrices as possible. It is NP-hard.

We consider matrices whose entries have a fixed number of decimal digits, as in problem 03 of the
[Quantum Optimization Benchmarking Library](https://github.com/ZIB-AOPT/QOBLIB) (QOBLIB), and look for exact
decompositions whose weights have the same precision. All computations use exact integer arithmetic.

## Algorithm

RCS (Recover, Certify, Search) runs two chains in parallel.

- The solution chain finds decompositions. For matrices built from few permutation matrices, it reads the weights from
  the frequencies of the entries and places them on the matrix positions (Recover). For matrices built from many
  permutation matrices, it runs a beam search over steps of the Birkhoff algorithm that each make at least one entry
  zero (Search).
- The bound chain proves lower bounds on the length: the largest number of nonzero entries in a row or column, a
  bound from pairs of rows or columns, and certificates of a SAT solver (Certify).

When the two chains meet, the decomposition is optimal. The [paper](https://arxiv.org/abs/2610.25837) gives the
details and the proofs.

The code is in Rust. To build it and to run it on one instance of QOBLIB:

```sh
git clone --filter=blob:none --sparse https://github.com/ZIB-AOPT/QOBLIB.git data/qoblib
git -C data/qoblib sparse-checkout set 03-birkhoff
cargo build --release
./target/release/solve --only B16_256_1 --time 30 --out results/try
```

`results/` holds the runs of the paper. `scripts/run_rounds.sh` repeats them, and `scripts/check_solutions.sh` checks
every decomposition with the checker of QOBLIB.

## Citation

A. Czégel, Exact minimum Birkhoff decompositions of doubly stochastic matrices with finite precision,
[arXiv:2610.25837](https://arxiv.org/abs/2610.25837), 2026.

```bibtex
@misc{czegel2026rcs,
  author        = {Cz{\'e}gel, Andr{\'a}s},
  title         = {Exact minimum {Birkhoff} decompositions of doubly stochastic matrices with finite precision},
  year          = {2026},
  eprint        = {2610.25837},
  archivePrefix = {arXiv},
  primaryClass  = {math.OC},
  url           = {https://arxiv.org/abs/2610.25837}
}
```

## License

MIT
