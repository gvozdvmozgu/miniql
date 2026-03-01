#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_perf_benches.sh [output_dir]
# Default output dir:
#   perf-artifacts/bench-results

OUTPUT_DIR="${1:-perf-artifacts/bench-results}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/alloc"

BENCHES=(sql join read_table)

echo "perf bench run started at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${OUTPUT_DIR}/run.log"
echo "output_dir=${OUTPUT_DIR}" | tee -a "${OUTPUT_DIR}/run.log"
echo "rustc=$(rustc -V)" | tee -a "${OUTPUT_DIR}/run.log"
echo "cargo=$(cargo -V)" | tee -a "${OUTPUT_DIR}/run.log"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "commit=$(git rev-parse HEAD)" | tee -a "${OUTPUT_DIR}/run.log"
fi

if [ -d target/criterion ]; then
  rm -rf target/criterion
fi

for bench in "${BENCHES[@]}"; do
  output_file="${OUTPUT_DIR}/${bench}-bench.txt"
  alloc_file="${OUTPUT_DIR}/alloc/${bench}.tsv"
  echo "running bench=${bench}" | tee -a "${OUTPUT_DIR}/run.log"
  rm -f "${alloc_file}"
  MINIQL_ALLOC_LOG="${alloc_file}" cargo bench --bench "${bench}" -- --output-format bencher 2>&1 \
    | tee "${output_file}"
done

python3 scripts/summarize_perf_metrics.py \
  --criterion-root target/criterion \
  --alloc-log "${OUTPUT_DIR}/alloc/sql.tsv" \
  --alloc-log "${OUTPUT_DIR}/alloc/join.tsv" \
  --alloc-log "${OUTPUT_DIR}/alloc/read_table.tsv" \
  --markdown-out "${OUTPUT_DIR}/perf_summary.md" \
  --json-out "${OUTPUT_DIR}/perf_summary.json"

echo "perf bench run finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUTPUT_DIR}/run.log"
