#!/bin/sh
# Checks every decomposition in a result folder with the solution checker of QOBLIB.
#
# Usage: scripts/check_solutions.sh [results/final]

cd "$(dirname "$0")/.." || exit 1

results=${1:-results/final}
qoblib=data/qoblib/03-birkhoff

cargo build --release --quiet --manifest-path "$qoblib/check/Cargo.toml" --target-dir target/check || exit 1
checker=target/check/release/check_birkhoff

checked=0
invalid=0
for solution in "$results"/B*/B*_solution.json; do
  id=$(basename "$(dirname "$solution")")   # for example B16_256_3
  n=$(echo "$id" | cut -d_ -f1 | tr -d B)
  generators=$(echo "$id" | cut -d_ -f2)
  # dense instances are built from n^2 permutation matrices, sparse ones from n
  if [ "$generators" -eq $((n * n)) ]; then
    class=dense
  else
    class=sparse
  fi
  instances=$(printf "%s/instances/qbench_%02d_%s.json" "$qoblib" "$n" "$class")
  if ! "$checker" "$instances" "$solution" 2>&1 | tail -1 | grep -q "^VALID"; then
    echo "invalid: $id"
    invalid=$((invalid + 1))
  fi
  checked=$((checked + 1))
done

echo "$checked decompositions checked, $invalid invalid"
[ "$invalid" -eq 0 ]
