#!/usr/bin/env python3
"""Check SQL benchmark performance budgets from bencher-format output."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


BENCH_LINE_RE = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+bench:\s+([0-9][0-9_]*)\s+ns/iter")


@dataclass(frozen=True)
class BenchKey:
    group: str
    variant: str
    row_count: str


def parse_engine_variant(raw: str) -> Tuple[str, str]:
    if raw == "miniql":
        return ("miniql", "default")
    if raw == "rusqlite":
        return ("rusqlite", "default")
    if raw.startswith("miniql_"):
        return ("miniql", raw[len("miniql_") :])
    if raw.startswith("rusqlite_"):
        return ("rusqlite", raw[len("rusqlite_") :])
    raise ValueError(f"unsupported benchmark engine label '{raw}'")


def parse_bencher_lines(lines: Iterable[str]) -> Dict[BenchKey, Dict[str, int]]:
    parsed: Dict[BenchKey, Dict[str, int]] = defaultdict(dict)
    for line in lines:
        match = BENCH_LINE_RE.match(line.strip())
        if not match:
            continue

        full_name, ns_raw = match.groups()
        parts = full_name.split("/")
        if len(parts) != 3:
            raise ValueError(f"unsupported benchmark name shape: '{full_name}'")

        group, engine_variant, row_count = parts
        engine, variant = parse_engine_variant(engine_variant)
        ns = int(ns_raw.replace("_", ""))
        key = BenchKey(group=group, variant=variant, row_count=row_count)
        parsed[key][engine] = ns

    if not parsed:
        raise ValueError("no bencher benchmark lines found in input")
    return parsed


def load_group_budgets(path: Path) -> Dict[str, float]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    groups = data.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("budget config must contain a non-empty [groups] table")

    out: Dict[str, float] = {}
    for group, value in groups.items():
        if not isinstance(group, str):
            raise ValueError("group keys in [groups] must be strings")
        if not isinstance(value, (int, float)):
            raise ValueError(f"budget for group '{group}' must be numeric")
        if value <= 0:
            raise ValueError(f"budget for group '{group}' must be > 0")
        out[group] = float(value)
    return out


def build_group_ratios(
    parsed: Dict[BenchKey, Dict[str, int]],
) -> Dict[str, List[Tuple[BenchKey, float, int, int]]]:
    group_ratios: Dict[str, List[Tuple[BenchKey, float, int, int]]] = defaultdict(list)
    for key, engines in parsed.items():
        miniql = engines.get("miniql")
        rusqlite = engines.get("rusqlite")
        if miniql is None or rusqlite is None:
            raise ValueError(
                f"missing pair for {key.group}/{key.variant}/{key.row_count}: "
                f"miniql={miniql is not None}, rusqlite={rusqlite is not None}"
            )
        if rusqlite <= 0:
            raise ValueError(
                f"invalid rusqlite timing for {key.group}/{key.variant}/{key.row_count}: {rusqlite}"
            )
        ratio = miniql / rusqlite
        group_ratios[key.group].append((key, ratio, miniql, rusqlite))
    return group_ratios


def format_ratio(v: float) -> str:
    if math.isfinite(v):
        return f"{v:.3f}x"
    return "n/a"


def render_summary(
    budgets: Dict[str, float],
    group_ratios: Dict[str, List[Tuple[BenchKey, float, int, int]]],
) -> Tuple[str, bool]:
    lines: List[str] = []
    lines.append("SQL perf budget check (miniql / rusqlite)")
    lines.append("")
    lines.append("| Group | Budget | Observed max | Worst case | Status |")
    lines.append("|---|---:|---:|---|---|")

    ok = True
    for group in sorted(budgets.keys()):
        entries = group_ratios.get(group, [])
        if not entries:
            ok = False
            lines.append(f"| `{group}` | {format_ratio(budgets[group])} | n/a | n/a | FAIL (missing data) |")
            continue

        worst = max(entries, key=lambda item: item[1])
        key, observed, miniql_ns, rusqlite_ns = worst
        status = "PASS"
        if observed > budgets[group]:
            status = "FAIL"
            ok = False

        worst_case = (
            f"`{key.variant}/{key.row_count}` "
            f"(miniql={miniql_ns}ns, rusqlite={rusqlite_ns}ns)"
        )
        lines.append(
            f"| `{group}` | {format_ratio(budgets[group])} | {format_ratio(observed)} | {worst_case} | {status} |"
        )

    missing_budgets = sorted(set(group_ratios.keys()) - set(budgets.keys()))
    if missing_budgets:
        ok = False
        lines.append("")
        lines.append("Missing budget definitions for groups:")
        for group in missing_budgets:
            lines.append(f"- `{group}`")

    return ("\n".join(lines), ok)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Bencher output file path")
    parser.add_argument(
        "--budgets",
        required=True,
        type=Path,
        help="TOML file with [groups] ratio thresholds",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional Markdown summary output path (e.g., GITHUB_STEP_SUMMARY)",
    )
    args = parser.parse_args()

    try:
        raw = args.input.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            # PowerShell Tee-Object often writes UTF-16LE with BOM.
            text = raw.decode("utf-16")
        parsed = parse_bencher_lines(text.splitlines())
        budgets = load_group_budgets(args.budgets)
        group_ratios = build_group_ratios(parsed)
        summary, ok = render_summary(budgets, group_ratios)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"SQL perf budget check failed: {exc}", file=sys.stderr)
        return 2

    print(summary)
    if args.summary_file is not None:
        args.summary_file.write_text(summary + "\n", encoding="utf-8")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
