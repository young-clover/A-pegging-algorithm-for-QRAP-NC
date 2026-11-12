"""Paired scalability experiment for the continuous QRAP with nested bounds."""
import argparse
import csv
import hashlib
import json
from itertools import combinations
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PAPER_SIZES = tuple(n for k in range(1, 8) for n in (10**k, 2*10**k, 5*10**k)
                    if n <= 10_000_000)


def paper_instance(n, seed, instance):
    """Section 5.1: two feasible random paths; the final resource is their mean."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, instance]))
    a = rng.uniform(0., 1., n)
    while np.any(a == 0.):
        zeros = a == 0.
        a[zeros] = rng.random(np.count_nonzero(zeros))
    lo = rng.uniform(.1, .5, n)
    hi = rng.uniform(.5, .9, n)
    p = rng.uniform(lo, hi).cumsum()
    q = rng.uniform(lo, hi).cumsum()
    L = np.minimum(p, q)
    U = np.maximum(p, q)
    L[-1] = U[-1] = (p[-1] + q[-1]) / 2
    return a, lo, hi, L, U


BOUND_CASES = ("two-sided", "upper-only", "lower-only")


def bound_variant(data, case):
    """Remove a proper-prefix bound by replacing it with the implied box sum."""
    a, lo, hi, L, U = data
    if case == "two-sided":
        return data
    R = U[-1]
    if case == "upper-only":
        L = np.cumsum(lo)
        L[-1] = R
    elif case == "lower-only":
        U = np.cumsum(hi)
        U[-1] = R
    else:
        raise ValueError(f"Unknown bound case: {case}")
    return a, lo, hi, L, U


def validate(x, data):
    a, lo, hi, L, U = data
    if x.shape != a.shape or not np.all(np.isfinite(x)):
        raise ArithmeticError("Invalid solution shape or nonfinite allocation")
    sums = np.cumsum(x)
    return dict(objective=float(np.sum((x/a)*x)/2),
                box_violation=float(max(0., (lo-x).max(), (x-hi).max())),
                prefix_violation=float(max(0., (L-sums).max(), (sums-U).max())),
                equality_error=float(abs(sums[-1]-U[-1])))


def run_instance(solvers, n, seed, index, ablation=False, case="two-sided", base=None):
    data = bound_variant(paper_instance(n, seed, index) if base is None else base, case)
    # Preserve the original call signature for callers using the ablation flag.
    if ablation and "Peg-cold" not in solvers:
        solvers = dict(solvers, **{"Peg-cold": solvers["Inf+Peg"]})
    order = list(solvers)
    offset = index % len(order)
    order = order[offset:] + order[:offset]
    outputs, runs = {}, {}
    for name in order:
        solver = solvers[name]
        # Memoryviews preserve Python scalar arithmetic and avoid copying INF slices.
        # SEQ normalizes four input bounds in place; isolate these before timing.
        args = [memoryview(v.copy() if name == "Seq" and j else v)
                for j, v in enumerate(data)]
        stats = {}
        kw = {} if name == "Dec" else {"stats": stats}
        if name == "Peg-cold":
            kw["warm_start"] = False
        t0 = time.perf_counter()
        result = solver(*args, **kw)
        elapsed = time.perf_counter()-t0
        x = np.asarray(result, dtype=float)
        del result, args
        runs[name] = dict(seconds=elapsed, **stats, **validate(x, data))
        outputs[name] = x
    dx = max((float(np.max(np.abs(x-y))) for x,y in combinations(outputs.values(), 2)), default=None)
    objs = [r["objective"] for r in runs.values()]
    relative_gap = (max(objs)-min(objs))/max(1., max(map(abs,objs))) if dx is not None else None
    primal = max(max(r["box_violation"],r["prefix_violation"],r["equality_error"])
                 for r in runs.values())
    # Absolute QRAP stopping errors can accumulate across disjoint INF leaves.
    agreement = dx is None or (dx <= 3e-5 and relative_gap <= 1e-6)
    ok = agreement and primal <= max(2e-5, 1e-10*abs(data[-1][-1]))
    return dict(n=n, instance=index, seed=seed, bounds=case, order=order, results=runs,
                max_coordinate_diff=dx, relative_objective_gap=relative_gap,
                max_primal_violation=primal, checked=bool(ok))


def summarize(rows, out):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["n"], []).append(row)
    records = []
    for n, group in sorted(grouped.items()):
        for name in group[0]["results"]:
            times = [r["results"][name]["seconds"] for r in group]
            mean = statistics.mean(times)
            record = dict(n=n, algorithm=name, instances=len(group), mean_s=mean,
                          cov=statistics.pstdev(times)/mean, median_s=statistics.median(times),
                          ratio_inf_to_method=statistics.mean(r["results"]["Inf"]["seconds"]
                                                               for r in group)/mean if "Inf" in group[0]["results"] else None,
                          verified=sum(r["checked"] for r in group))
            for key in ("breakpoint_searches","nodes","screened","prefixed","pegging_visits"):
                counts=[r["results"][name].get(key) for r in group]
                if any(c is not None for c in counts):
                    values=[c or 0 for c in counts]
                    record[key]=statistics.mean(values)
                    record[key+"_cov"]=statistics.pstdev(values)/record[key] if record[key] else 0.
            records.append(record)
    columns = list(dict.fromkeys(k for rec in records for k in rec))
    with (out/"summary.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    fits = {}
    for name in ("Seq","Dec","Inf","Inf+Peg","Peg-cold"):
        values = sorted((r["n"],r["mean_s"]) for r in records if r["algorithm"]==name)
        if not values:
            continue
        sizes,times=np.array(values).T
        line,=ax.loglog(sizes,times,marker="o",label=name)
        if len(sizes)>1:
            p,b=np.polyfit(np.log(sizes),np.log(times),1)
            fits[name]={"c1":float(np.exp(b)), "c2":float(p)}
            ax.loglog(sizes,np.exp(b)*sizes**p,linestyle=":",color=line.get_color(),alpha=.6)
    ax.set(xlabel="Number of variables", ylabel="Mean solver time (seconds)")
    ax.grid(True,which="both",alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"runtime.png",dpi=160)
    plt.close(fig)
    (out/"power_fits.json").write_text(json.dumps(fits,indent=2))


def paired_summary(rows_by_case, output):
    """Compare exactly matching checked instances, grouped by size and solver."""
    pairs = {}
    for case, rows in rows_by_case.items():
        for row in rows:
            pairs.setdefault((row["n"], row["seed"], row["instance"]), {})[case] = row
    groups = {}
    for (n, _, _), group in pairs.items():
        if all(case in group and group[case]["checked"] for case in BOUND_CASES):
            groups.setdefault(n, []).append(group)
    records = []
    for n, groups_at_n in sorted(groups.items()):
        for name in groups_at_n[0]["two-sided"]["results"]:
            record = dict(n=n, algorithm=name, pairs=len(groups_at_n))
            for case in BOUND_CASES:
                times = [g[case]["results"][name]["seconds"] for g in groups_at_n]
                mean = statistics.mean(times)
                record[f"{case}_mean_s"] = mean
                record[f"{case}_cov"] = statistics.pstdev(times) / mean
                if case != "two-sided":
                    record[f"two_over_{case}"] = record["two-sided_mean_s"] / mean
            records.append(record)
    if records:
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def main(solvers, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=PAPER_SIZES)
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bounds", choices=(*BOUND_CASES, "paired"), default="two-sided",
                        help="paired compares all three variants of each base instance")
    available = {**solvers, "Peg-cold": solvers["Inf+Peg"]}
    names = {name.lower(): name for name in available}
    parser.add_argument("--algorithms", nargs="+", choices=tuple(available),
                        type=lambda name: names.get(name.lower(), name),
                        help="run exactly these algorithms, in any bounds mode")
    parser.add_argument("--ablation", action="store_true",
                        help="include Peg-cold (automatic for one-sided or paired runs)")
    parser.add_argument("--include-dec", action="store_true",
                        help="include Dec (automatic for one-sided or paired runs)")
    opts = parser.parse_args(argv)
    if opts.instances <= 0 or opts.seed < 0 or any(n < 2 for n in opts.sizes):
        parser.error("require positive instances, n >= 2, and a nonnegative seed")

    extended = opts.bounds != "two-sided"
    defaults = [name for name in available
                if (name != "Dec" or extended or opts.include_dec)
                and (name != "Peg-cold" or extended or opts.ablation)]
    if opts.algorithms:
        if len(set(opts.algorithms)) != len(opts.algorithms):
            parser.error("--algorithms contains duplicate names")
        if opts.ablation and "Peg-cold" not in opts.algorithms:
            parser.error("--ablation requires Peg-cold in --algorithms")
        if opts.include_dec and "Dec" not in opts.algorithms:
            parser.error("--include-dec requires Dec in --algorithms")
    requested = opts.algorithms or defaults
    solvers = {name: fn for name, fn in available.items() if name in requested}
    cases = BOUND_CASES if opts.bounds == "paired" else (opts.bounds,)
    default_dir = "qrap_scalability_results" if not extended else f"qrap_{opts.bounds}_results"
    if opts.algorithms and set(requested) != set(defaults):
        default_dir += "_" + "_".join(solvers)
    out = opts.output or Path(default_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(available["Seq"].__code__.co_filename)
    protocol = dict(seed=opts.seed, bounds=opts.bounds, ablation="Peg-cold" in solvers,
                    algorithms=list(solvers),
                    python=sys.version, numpy=np.__version__, platform=platform.platform(),
                    solver_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    meta = out / "protocol.json"
    if meta.exists():
        saved = json.loads(meta.read_text())
        # The preceding driver timed the same calls and stored the same records
        # for its default multi-algorithm runs. Preserve those checkpoints.
        prior_driver = "2c39209902027503fd87f7412dce01c2420c4d0b7b94eb836b013048029a2f35"
        if saved != protocol and saved != dict(protocol, driver_sha256=prior_driver):
            parser.error("the saved protocol differs; choose a new output directory for this selection")
    meta.write_text(json.dumps(protocol, indent=2))
    dirs = {case: out / case if len(cases) > 1 else out for case in cases}
    for folder in dirs.values():
        (folder / "instances").mkdir(parents=True, exist_ok=True)
    for n in sorted(set(opts.sizes)):
        for idx in range(opts.instances):
            offset = idx % len(cases)
            base = None
            for case in cases[offset:] + cases[:offset]:
                path = dirs[case] / "instances" / f"n{n:08d}-i{idx:03d}.json"
                if path.exists() and json.loads(path.read_text())["checked"]:
                    continue
                if base is None:
                    base = paper_instance(n, opts.seed, idx)
                row = run_instance(solvers, n, opts.seed, idx, case=case, base=base)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(row, indent=1, allow_nan=False))
                tmp.replace(path)
                print(case, n, idx + 1,
                      {k: round(v["seconds"], 4) for k, v in row["results"].items()},
                      "PASS" if row["checked"] else "CHECK", flush=True)
                if not row["checked"]:
                    raise RuntimeError(f"Validation failed; inspect {path}. The next run retries it.")
            del base
        rows = {}
        for case, folder in dirs.items():
            rows[case] = [json.loads(p.read_text()) for p in sorted((folder / "instances").glob("*.json"))]
            summarize([row for row in rows[case] if row["checked"]], folder)
        if len(cases) > 1:
            paired_summary(rows, out / "paired_summary.csv")
