## Summary

This PR adds **Windows-safe multiprocessing** support to the high-level wrapper APIs:

- `RxnToSr(..., n_jobs=...)`
- `SrToRxn(..., n_jobs=...)`

It also extends the benchmark tooling to measure both directions (`rxn` and `sr` inputs), include `n_jobs` in reports, and adds a small script to compare two benchmark JSON outputs.

## What changed

- **Multiprocessing wrappers**
  - Added `n_jobs` to `RxnToSr` and `SrToRxn` (default `1`).
  - `n_jobs <= 0` uses `os.cpu_count()` processes.
  - Preserves input order; returns the same container type (`list`, `Series`, `DataFrame`) as before.
  - Implementation uses `ProcessPoolExecutor` with **spawn-safe, top-level worker functions**.

- **Benchmarks**
  - `benchmarks/runner.py` now supports:
    - `--kind rxn|sr`
    - `--n-jobs <int>` (passed to wrapper APIs)
    - `--add-atom-mapping` (for `sr_to_rxn`)
  - Reports now include: `kind`, `n_jobs`, `n_items`, and `throughput_items_per_sec_avg`.

- **Comparison utility**
  - Added `benchmarks/comparison.py` to generate a small markdown comparison between two JSON reports.

## Benchmark results (local)

The multiprocessing speedup only shows up once the workload is large enough (process startup/IPC overhead dominates on small datasets).

### rxn_to_sr (`RxnToSr`) on repeated SN2 workload

- Workload: 11,220 reactions (561 SN2 reactions repeated 20x)
- Results (wall time):
  - `n_jobs=1`: **36.455 s** (307.8 rxn/s)
  - `n_jobs=4`: **10.653 s** (1053.2 rxn/s)
  - `n_jobs=8`: **7.292 s** (1538.7 rxn/s)

### sr_to_rxn (`SrToRxn`) on repeated sr_test_cases workload

- Workload: 9,300 sr-SMILES (31 cases repeated 300x)
- Results (wall time):
  - `n_jobs=1`: **16.556 s** (561.7 sr/s)
  - `n_jobs=4`: **6.065 s** (1533.5 sr/s)
  - `n_jobs=8`: **4.284 s** (2170.7 sr/s)

## How to run

### Benchmarks

```bash
python benchmarks/runner.py --label rxn_serial --kind rxn --dataset tests/data/sn2/test.csv --api class --n-jobs 1 --repeats 3
python benchmarks/runner.py --label rxn_mp --kind rxn --dataset tests/data/sn2/test.csv --api class --n-jobs 8 --repeats 3

python benchmarks/runner.py --label sr_serial --kind sr --dataset sr_test_cases --api class --n-jobs 1 --repeats 3
python benchmarks/runner.py --label sr_mp --kind sr --dataset sr_test_cases --api class --n-jobs 8 --repeats 3
```

### Compare two runs

```bash
python benchmarks/comparison.py --baseline benchmarks/reports/rxn_serial.json --candidate benchmarks/reports/rxn_mp.json
```

## Test plan

- `python -m pytest`
- Smoke-run CLI examples if desired:
  - `rxn2sr ...`
  - `sr2rxn ...`
