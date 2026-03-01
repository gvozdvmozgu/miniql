#!/usr/bin/env python3
"""Summarize benchmark p50/p95 latency and allocation counts per benchmark group."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    w = idx - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def bench_id_from_sample_path(criterion_root: Path, sample_path: Path) -> str:
    rel = sample_path.relative_to(criterion_root)
    parts = list(rel.parts)
    # .../<bench-id>/new/sample.json
    if len(parts) < 3 or parts[-2] != "new" or parts[-1] != "sample.json":
        raise ValueError(f"unexpected sample path shape: {sample_path}")
    return "/".join(parts[:-2])


def group_from_bench_id(bench_id: str) -> str:
    return bench_id.split("/", 1)[0]


def load_latency_by_group(criterion_root: Path) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = defaultdict(list)
    for sample_path in criterion_root.glob("**/new/sample.json"):
        try:
            bench_id = bench_id_from_sample_path(criterion_root, sample_path)
            with sample_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            iters = data.get("iters")
            times = data.get("times")
            if not isinstance(iters, list) or not isinstance(times, list):
                continue
            if len(iters) != len(times):
                continue
            group = group_from_bench_id(bench_id)
            for n_iter, total_ns in zip(iters, times):
                if not n_iter:
                    continue
                out[group].append(float(total_ns) / float(n_iter))
        except Exception:
            continue
    return out


def parse_alloc_line(line: str) -> Tuple[str, int, int, int, int, int, int] | None:
    parts = line.strip().split("\t")
    if len(parts) != 7:
        return None
    try:
        return (
            parts[0],
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
            int(parts[4]),
            int(parts[5]),
            int(parts[6]),
        )
    except ValueError:
        return None


def load_alloc_by_group(log_paths: Iterable[Path]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {
            "allocations": [],
            "deallocations": [],
            "reallocations": [],
            "bytes_allocated": [],
            "bytes_deallocated": [],
            "bytes_reallocated": [],
        }
    )
    for path in log_paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            parsed = parse_alloc_line(raw_line)
            if parsed is None:
                continue
            bench_id, allocs, deallocs, reallocs, bytes_alloc, bytes_dealloc, bytes_realloc = parsed
            group = group_from_bench_id(bench_id)
            row = out[group]
            row["allocations"].append(float(allocs))
            row["deallocations"].append(float(deallocs))
            row["reallocations"].append(float(reallocs))
            row["bytes_allocated"].append(float(bytes_alloc))
            row["bytes_deallocated"].append(float(bytes_dealloc))
            row["bytes_reallocated"].append(float(bytes_realloc))
    return out


def format_ns(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.1f}"


def render_markdown(
    lat_by_group: Dict[str, List[float]],
    alloc_by_group: Dict[str, Dict[str, List[float]]],
) -> str:
    groups = sorted(set(lat_by_group.keys()) | set(alloc_by_group.keys()))
    lines: List[str] = []
    lines.append("Perf metrics by benchmark group")
    lines.append("")
    lines.append("| Group | p50 ns/iter | p95 ns/iter | alloc p50 | alloc p95 | bytes p50 | bytes p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for group in groups:
        latencies = sorted(lat_by_group.get(group, []))
        p50 = percentile(latencies, 0.50)
        p95 = percentile(latencies, 0.95)
        alloc = alloc_by_group.get(group, {})
        alloc_p50 = percentile(sorted(alloc.get("allocations", [])), 0.50)
        alloc_p95 = percentile(sorted(alloc.get("allocations", [])), 0.95)
        bytes_p50 = percentile(sorted(alloc.get("bytes_allocated", [])), 0.50)
        bytes_p95 = percentile(sorted(alloc.get("bytes_allocated", [])), 0.95)
        lines.append(
            f"| `{group}` | {format_ns(p50)} | {format_ns(p95)} | "
            f"{format_ns(alloc_p50)} | {format_ns(alloc_p95)} | "
            f"{format_ns(bytes_p50)} | {format_ns(bytes_p95)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criterion-root", type=Path, required=True)
    parser.add_argument("--alloc-log", type=Path, action="append", default=[])
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    lat_by_group = load_latency_by_group(args.criterion_root)
    alloc_by_group = load_alloc_by_group(args.alloc_log)

    markdown = render_markdown(lat_by_group, alloc_by_group)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(markdown, encoding="utf-8")

    json_payload = {}
    for group in sorted(set(lat_by_group.keys()) | set(alloc_by_group.keys())):
        latencies = sorted(lat_by_group.get(group, []))
        json_payload[group] = {
            "p50_ns_per_iter": percentile(latencies, 0.50),
            "p95_ns_per_iter": percentile(latencies, 0.95),
            "alloc_p50": percentile(sorted(alloc_by_group.get(group, {}).get("allocations", [])), 0.50),
            "alloc_p95": percentile(sorted(alloc_by_group.get(group, {}).get("allocations", [])), 0.95),
            "bytes_alloc_p50": percentile(
                sorted(alloc_by_group.get(group, {}).get("bytes_allocated", [])), 0.50
            ),
            "bytes_alloc_p95": percentile(
                sorted(alloc_by_group.get(group, {}).get("bytes_allocated", [])), 0.95
            ),
        }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(json_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
