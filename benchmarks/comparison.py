import argparse
import json
from pathlib import Path


def _pct_delta(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return float("inf") if candidate != 0 else 0.0
    return ((candidate - baseline) / baseline) * 100.0


def _format_float(x: object, decimals: int = 4) -> str:
    if isinstance(x, (int, float)):
        return f"{x:.{decimals}f}"
    return str(x)


def load_json(path: Path) -> dict:
    """Load a JSON file from the specified path."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize(report: dict) -> dict:
    """Extract key metrics from a benchmark report dictionary."""
    return {
        "label": report.get("label"),
        "kind": report.get("kind"),
        "api": report.get("api"),
        "n_jobs": report.get("n_jobs"),
        "dataset": report.get("dataset"),
        "n_items": report.get("n_items"),
        "avg_time_sec": report.get("avg_time_sec"),
        "throughput_items_per_sec_avg": report.get("throughput_items_per_sec_avg"),
        "latency_ms_per_item_avg": report.get("latency_ms_per_item_avg"),
        "peak_memory_mb_tracemalloc": report.get("peak_memory_mb_tracemalloc"),
        "n_empty_outputs_last_run": report.get("n_empty_outputs_last_run"),
    }


def compare(baseline: dict, candidate: dict) -> str:
    """Compare two benchmark reports and return a formatted markdown string."""
    b = summarize(baseline)
    c = summarize(candidate)

    keys = [
        "avg_time_sec",
        "throughput_items_per_sec_avg",
        "latency_ms_per_item_avg",
        "peak_memory_mb_tracemalloc",
        "n_empty_outputs_last_run",
    ]

    lines = []
    lines.append("## Benchmark comparison\n")
    lines.append(f"- baseline: `{b.get('label')}`\n")
    lines.append(f"- candidate: `{c.get('label')}`\n")
    lines.append("\n")
    lines.append("### Context\n")
    for k in ("kind", "api", "n_jobs", "dataset", "n_items"):
        lines.append(f"- {k}: baseline={b.get(k)!r}, candidate={c.get(k)!r}\n")
    lines.append("\n")
    lines.append("### Metrics\n")
    lines.append("| metric | baseline | candidate | delta |\n")
    lines.append("|---|---:|---:|---:|\n")
    for k in keys:
        bv = b.get(k)
        cv = c.get(k)
        if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
            delta = _pct_delta(float(bv), float(cv))
            lines.append(f"| {k} | {_format_float(bv)} | {_format_float(cv)} | {delta:+.2f}% |\n")
        else:
            lines.append(f"| {k} | {bv!r} | {cv!r} | n/a |\n")
    return "".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark comparison tool."""
    parser = argparse.ArgumentParser(description="Compare two benchmark JSON reports.")
    parser.add_argument("--baseline", required=True, help="Path to baseline JSON report.")
    parser.add_argument("--candidate", required=True, help="Path to candidate JSON report.")
    parser.add_argument("--out", default="", help="Optional output markdown path.")
    return parser.parse_args()


def main() -> None:
    """Main entry point to load reports, compare them, and output results."""
    args = parse_args()
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    md = compare(load_json(baseline_path), load_json(candidate_path))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
    else:
        print(md)


if __name__ == "__main__":
    main()
