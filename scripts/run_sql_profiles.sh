#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_sql_profiles.sh [output_dir]
# Default output dir:
#   perf-artifacts/profiles

OUTPUT_DIR="${1:-perf-artifacts/profiles}"
mkdir -p "${OUTPUT_DIR}"

DISTINCT_FILTER="sql_distinct/miniql_low_card/20000"
JOIN_LEFT_FILTER="sql_join/miniql_left/20000"

echo "sql profile run started at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${OUTPUT_DIR}/run.log"
echo "output_dir=${OUTPUT_DIR}" | tee -a "${OUTPUT_DIR}/run.log"
echo "rustc=$(rustc -V)" | tee -a "${OUTPUT_DIR}/run.log"
echo "cargo=$(cargo -V)" | tee -a "${OUTPUT_DIR}/run.log"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "commit=$(git rev-parse HEAD)" | tee -a "${OUTPUT_DIR}/run.log"
fi

echo "profiling=${DISTINCT_FILTER}" | tee -a "${OUTPUT_DIR}/run.log"
cargo bench --bench sql "${DISTINCT_FILTER}" -- --profile-time 10 --noplot 2>&1 \
  | tee "${OUTPUT_DIR}/sql_distinct_profile.txt"
cargo flamegraph --bench sql --output "${OUTPUT_DIR}/sql_distinct_flamegraph.svg" -- "${DISTINCT_FILTER}" 2>&1 \
  | tee "${OUTPUT_DIR}/sql_distinct_flamegraph.txt"

echo "profiling=${JOIN_LEFT_FILTER}" | tee -a "${OUTPUT_DIR}/run.log"
cargo bench --bench sql "${JOIN_LEFT_FILTER}" -- --profile-time 10 --noplot 2>&1 \
  | tee "${OUTPUT_DIR}/sql_join_left_profile.txt"
cargo flamegraph --bench sql --output "${OUTPUT_DIR}/sql_join_left_flamegraph.svg" -- "${JOIN_LEFT_FILTER}" 2>&1 \
  | tee "${OUTPUT_DIR}/sql_join_left_flamegraph.txt"

echo "sql profile run finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUTPUT_DIR}/run.log"
