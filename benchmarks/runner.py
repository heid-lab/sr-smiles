import argparse
import cProfile
import json
import logging
import pstats
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from sr_smiles.io.logger import logger as sr_logger  # noqa: E402
    from sr_smiles.reaction_balancing import is_balanced  # noqa: E402
    from sr_smiles.transforms.rxn_to_sr import (  # noqa: E402
        RxnToSr,
        build_sr_smiles,
        extract_atom_and_bond_changes,
        get_chirality_aligned_smiles_and_mols,
        rxn_to_sr,
    )
    from sr_smiles.transforms.sr_to_rxn import SrToRxn, sr_to_rxn  # noqa: E402
except ImportError as e:
    print(f"Import error: {e}")
    print("Run this from repo root with dependencies installed, e.g. `poetry install --with dev`.")
    sys.exit(1)

REPORT_DIR = PROJECT_ROOT / "benchmarks" / "reports"
REACTION_COLUMNS = ("rxn_smiles", "reaction", "rxn", "AAM")
SR_COLUMNS = ("sr_smiles", "sr", "sr_smi")


def _extract_reactions_from_csv(csv_path: Path) -> list[str]:
    """Load reaction strings from known columns in a CSV file."""
    df = pd.read_csv(csv_path)
    col = next((c for c in REACTION_COLUMNS if c in df.columns), None)
    if col is None:
        return []
    return [s for s in df[col].dropna().astype(str).tolist() if ">>" in s]


def _extract_srs_from_csv(csv_path: Path) -> list[str]:
    """Load sr-SMILES strings from known columns in a CSV file."""
    df = pd.read_csv(csv_path)
    col = next((c for c in SR_COLUMNS if c in df.columns), None)
    if col is None:
        return []
    return [s for s in df[col].dropna().astype(str).tolist() if s]


def load_test_inputs(kind: str, dataset: str, include_all_test_data: bool) -> list[str]:
    """Load only real test inputs from tests/data."""
    data_dir = PROJECT_ROOT / "tests" / "data"

    if dataset == "sr_test_cases":
        csv_path = data_dir / "sr_test_cases.csv"
    else:
        csv_path = Path(dataset)
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / dataset
    if kind == "rxn":
        inputs = _extract_reactions_from_csv(csv_path)
    else:
        inputs = _extract_srs_from_csv(csv_path)

    if include_all_test_data:
        for extra_csv in sorted(data_dir.rglob("*.csv")):
            if kind == "rxn":
                inputs.extend(_extract_reactions_from_csv(extra_csv))
            else:
                inputs.extend(_extract_srs_from_csv(extra_csv))

    # Keep order stable but remove duplicates.
    return list(dict.fromkeys(inputs))


def _build_transform(
    kind: str,
    api: str,
    keep_atom_mapping: bool,
    remove_hydrogens: bool,
    balance_rxn: bool,
    add_atom_mapping: bool,
    n_jobs: int,
) -> Callable[[str], str]:
    """Return transform function for selected API."""
    if kind == "rxn":
        if api == "function":
            return lambda r: rxn_to_sr(  # noqa: E731
                r,
                keep_atom_mapping=keep_atom_mapping,
                remove_hydrogens=remove_hydrogens,
                balance_rxn=balance_rxn,
            )

        transformer = RxnToSr(
            keep_atom_mapping=keep_atom_mapping,
            remove_hydrogens=remove_hydrogens,
            balance_rxn=balance_rxn,
            n_jobs=n_jobs,
        )
        return transformer

    # kind == "sr"
    if api == "function":
        return lambda s: sr_to_rxn(s, add_atom_mapping=add_atom_mapping)  # noqa: E731

    transformer = SrToRxn(add_atom_mapping=add_atom_mapping, n_jobs=n_jobs)
    return transformer


def _run_once(transform: Callable[[str], str], reactions: list[str]) -> list[str]:
    return [transform(r) for r in reactions]


def run_diagnostics(args: argparse.Namespace) -> dict:
    """Run timing, memory, and optional profiler diagnostics."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_test_inputs(
        kind=args.kind, dataset=args.dataset, include_all_test_data=args.include_all_test_data
    )

    if not inputs:
        raise ValueError("No reactions found in selected test data source.")

    if args.max_samples and args.max_samples > 0:
        inputs = inputs[: args.max_samples]

    if args.only_balanced and args.kind == "rxn":
        inputs = [r for r in inputs if is_balanced(r)]

    transform = _build_transform(
        kind=args.kind,
        api=args.api,
        keep_atom_mapping=args.keep_atom_mapping,
        remove_hydrogens=args.remove_hydrogens,
        balance_rxn=args.balance_rxn,
        add_atom_mapping=args.add_atom_mapping,
        n_jobs=args.n_jobs,
    )

    if args.quiet_failures:
        sr_logger.setLevel(logging.ERROR)

    print(f"Loaded {len(inputs)} inputs from test data (kind={args.kind}).")
    print(f"API={args.api}, n_jobs={args.n_jobs}, repeats={args.repeats}, warmup={args.warmup}")

    for _ in range(args.warmup):
        _run_once(transform, inputs)

    run_seconds: list[float] = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        output = _run_once(transform, inputs)
        run_seconds.append(time.perf_counter() - t0)

    tracemalloc.start()
    _ = _run_once(transform, inputs)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_items = len(inputs)
    avg_sec = sum(run_seconds) / len(run_seconds)
    min_sec = min(run_seconds)
    max_sec = max(run_seconds)

    results = {
        "label": args.label,
        "kind": args.kind,
        "dataset": args.dataset,
        "include_all_test_data": args.include_all_test_data,
        "api": args.api,
        "n_jobs": args.n_jobs,
        "balance_rxn": args.balance_rxn,
        "add_atom_mapping": args.add_atom_mapping,
        "n_items": total_items,
        "warmup_runs": args.warmup,
        "only_balanced": args.only_balanced,
        "timing_sec_per_run": run_seconds,
        "avg_time_sec": avg_sec,
        "min_time_sec": min_sec,
        "max_time_sec": max_sec,
        "throughput_items_per_sec_avg": total_items / avg_sec,
        "latency_ms_per_item_avg": (avg_sec / total_items) * 1000,
        "peak_memory_mb_tracemalloc": peak_bytes / (1024 * 1024),
        "n_empty_outputs_last_run": sum(item == "" for item in output),
    }

    profile_base = REPORT_DIR / args.label

    if args.profile_cpu:
        profile = cProfile.Profile()
        profile.enable()
        _run_once(transform, inputs[: args.profile_samples])
        profile.disable()
        stats_path = profile_base.with_suffix(".cprofile.txt")
        with stats_path.open("w", encoding="utf-8") as f:
            stats = pstats.Stats(profile, stream=f).sort_stats("cumulative")
            stats.print_stats(args.profile_top)
        print(f"Saved CPU profile: {stats_path}")

    if args.profile_lines:
        try:
            import line_profiler
        except ImportError:
            print("line_profiler not installed; skipping line-level profile.")
        else:
            lp = line_profiler.LineProfiler()
            if args.kind == "rxn":
                lp.add_function(rxn_to_sr)
                lp.add_function(get_chirality_aligned_smiles_and_mols)
                lp.add_function(extract_atom_and_bond_changes)
                lp.add_function(build_sr_smiles)
            else:
                lp.add_function(sr_to_rxn)
            lp.runcall(lambda: _run_once(transform, inputs[: args.profile_samples]))
            line_path = profile_base.with_suffix(".line_profiler.txt")
            with line_path.open("w", encoding="utf-8") as f:
                lp.print_stats(stream=f)
            print(f"Saved line profile: {line_path}")

    json_path = profile_base.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved benchmark metrics: {json_path}")
    print(f"Throughput(avg): {results['throughput_items_per_sec_avg']:.2f} items/s")
    print(f"Peak memory (tracemalloc): {results['peak_memory_mb_tracemalloc']:.2f} MB")
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for benchmark execution."""
    parser = argparse.ArgumentParser(description="Benchmark rxn_to_sr with real test datasets.")
    parser.add_argument("--label", default="rxn_to_sr_baseline", help="Output report prefix.")
    parser.add_argument("--kind", choices=("rxn", "sr"), default="rxn", help="Input kind to benchmark.")
    parser.add_argument("--dataset", default="sr_test_cases", help="Dataset name or CSV path.")
    parser.add_argument(
        "--include-all-test-data",
        action="store_true",
        help="Include all CSVs under tests/data in addition to --dataset.",
    )
    parser.add_argument("--api", choices=("function", "class"), default="function")
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Process count for wrapper APIs (<=0 uses cpu_count)."
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of reactions (0 = all).")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--only-balanced", action="store_true", help="Filter input to already balanced reactions."
    )
    parser.add_argument("--keep-atom-mapping", action="store_true")
    parser.add_argument("--remove-hydrogens", action="store_true")
    parser.add_argument("--balance-rxn", action="store_true")
    parser.add_argument("--add-atom-mapping", action="store_true")
    parser.add_argument(
        "--quiet-failures",
        action="store_true",
        help="Suppress warning logs from failed conversions while benchmarking.",
    )
    parser.add_argument("--profile-cpu", action="store_true")
    parser.add_argument("--profile-lines", action="store_true")
    parser.add_argument("--profile-samples", type=int, default=500)
    parser.add_argument("--profile-top", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run_diagnostics(parse_args())
