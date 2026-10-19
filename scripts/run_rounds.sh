#!/bin/sh
# Runs RCS on all test instances in two rounds, as in the paper:
#   round 1: every instance with a time limit of 60 s,
#   round 2: every instance without a proof of optimality with a time limit of 600 s.
# Results are written as they appear. If the solver stops, it is started again and continues with the instances that
# are not finished yet. At the end, both rounds are merged into results/final.
#
# Usage: scripts/run_rounds.sh

cd "$(dirname "$0")/.." || exit 1

solve="./target/release/solve --jobs 2 --threads 6"

# keep the computer awake on macOS
keep_awake=""
if command -v caffeinate > /dev/null; then
  keep_awake="caffeinate -ims"
fi

# run <output folder> <time limit> [solver options]
run() {
  out=$1
  time_limit=$2
  shift 2
  mkdir -p "$out"
  until $keep_awake $solve --time "$time_limit" --out "$out" "$@" >> "$out/run.log" 2>&1; do
    echo "$(date) solver stopped (status $?), continuing" >> "$out/run.log"
    sleep 5
  done
}

run results/runs/round1_t60 60

open_instances=$(tail -n +2 results/runs/round1_t60/unfinished.csv | cut -d, -f1 | paste -sd, -)
if [ -n "$open_instances" ]; then
  run results/runs/round2_t600 600 --only "$open_instances"
fi

./target/release/merge \
  --runs results/runs/round1_t60,results/runs/round2_t600 \
  --timed-out results/runs/round2_t600/unfinished.csv \
  --out results/final
echo "$(date) both rounds finished, results in results/final"
