# Pegging for quadratic resource allocation with nested constraints

This directory contains the Python implementation of **Peg** and the computational results for *A fast pegging algorithm for the quadratic resource allocation problem with nested constraints*. Peg combines top-down splitting with a variable-fixing solver for each quadratic relaxation. At a split, it uses the parent allocation to fix some variables at their bounds before solving the child relaxation.

## Algorithms and attribution

The **Dec**, **Seq**, and **Inf** baseline implementations originate from the [QRAP_NC repository](https://github.com/mhhschootuiterkamp/QRAP_NC) accompanying M. H. H. Schoot Uiterkamp, J. L. Hurink, and M. E. T. Gerards, [“A fast algorithm for quadratic resource allocation problems with nested constraints”](https://doi.org/10.1016/j.cor.2021.105451), *Computers & Operations Research* 135 (2021), 105451. The manuscript cites this source as `\cite{schootuiterkamp2021fast}`. **Our main implementation is Peg**. Cold uses the same code with parent-based fixing switched off.

| Paper name | Name in result files | Method and local entry point |
| --- | --- | --- |
| Seq | `Seq` | Sequential algorithm of Schoot Uiterkamp et al., implemented in `QRAP_NC_seq`. |
| Inf | `Inf` | Top-down algorithm of van der Klauw et al. with breakpoint search, implemented in `QRAP_NC_infeasible`. |
| Peg | `Inf+Peg` | Proposed method, implemented in `QRAP_NC_infeasible_pegging` with `warm_start=True` (default). |
| Cold | `Peg-cold` | The same function with `warm_start=False`, used to measure the benefit of parent-based fixing. |

## Scripts and requirements

Run the examples below from this `codes/` directory. The experiment and plotting scripts require Python, NumPy, and Matplotlib. The recorded experiments used Python 3.10.9 and NumPy 1.23.5 on ARM64 macOS 14.4, with seed `20261111`.

| Script | Purpose |
| --- | --- |
| `Algorithms_QRAP_NC_final.py` | Solver functions for Seq, Inf, and Peg. |
| `qrap_scalability.py` | Generates paired instances, rotates solver order, times one call per method, checks the solutions, and writes run records and summaries. |
| `plot_qrap_results.py` | Reads the saved paired results, regenerates the three-panel figure. |

## Rebuild the paper figure from the saved results

```sh
python plot_qrap_results.py
```

This reads `results/` and writes `runtime-by-configuration.pdf` in `codes/`.

## Run the benchmark

The solver file's `--scalability` entry point still references the omitted Dec function. Use the driver directly for the four methods in the paper. The following shell function registers the available solvers. The driver adds Cold by calling Peg with parent-based fixing disabled.

```sh
run_qrap() {
    python - "$@" <<'PY'
import sys
import Algorithms_QRAP_NC_final as qrap
from qrap_scalability import main

main({
    "Seq": qrap.QRAP_NC_seq,
    "Inf": qrap.QRAP_NC_infeasible,
    "Inf+Peg": qrap.QRAP_NC_infeasible_pegging,
}, sys.argv[1:])
PY
}
```

For a quick check, run two base instances at each of two sizes under all three bound settings:

```sh
run_qrap --bounds paired --algorithms Seq Inf Inf+Peg Peg-cold \
    --sizes 10 20 --instances 2 --seed 20261111 --output runs/example
```

To use the paper's settings, omit `--sizes` to select all 19 sizes from 10 to 10,000,000. This gives 190 base instances and 570 paired variants. Use a fresh output directory for a new run. The archived `results/protocol.json` records the solver and driver hashes for the published timings.

```sh
run_qrap --bounds paired --algorithms Seq Inf Inf+Peg Peg-cold \
    --instances 10 --seed 20261111 --output runs/paper
```

## Instance records and result paths

The supplied experiment is in `results/`. A new paired run creates the same layout under the directory passed to `--output`. In the paths below, `<case>` is `two-sided`, `upper-only`, or `lower-only`.

| Path relative to the results directory | Contents |
| --- | --- |
| `protocol.json` | Seed, algorithms, bound settings, runtime versions, and source hashes. |
| `<case>/instances/n00000010-i000.json` | Example record for 10 variables and base instance 0. Each record contains solver times in seconds, objective values, available work counters, feasibility and agreement errors, and the validation flag `checked`. |
| `<case>/summary.csv` | One row per size and algorithm, with mean and median times, coefficient of variation, and available counters. |
| `<case>/power_fits.json`, `<case>/runtime.png` | Power-law fits and a timing plot for that bound setting. |
| `paired_summary.csv` | Mean times and variation for the same checked base instances across all three settings. |

The JSON files store run records. To recover the numerical input arrays, pass the record's size, seed, and instance index to the generator, then apply its bound setting:

```python
import json
from pathlib import Path
from qrap_scalability import paper_instance, bound_variant

record = json.loads(Path("results/upper-only/instances/n00000010-i000.json").read_text())
base = paper_instance(record["n"], record["seed"], record["instance"])
a, lower, upper, lower_nested, upper_nested = bound_variant(base, record["bounds"])
```
