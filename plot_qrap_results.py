"""Regenerate the manuscript's original three-panel figure from paired QRAP runs.

From codes/: python plot_qrap_results.py
To refresh both LaTeX table bodies as well, add --tables-in ../manuscript/manuscript.tex.
The table option preserves the captions, column order, and formatting of the manuscript.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np

CASES = ("two-sided", "upper-only", "lower-only")
DISPLAY_NAMES = {"Seq": "Seq", "Inf": "Inf", "Inf+Peg": "Peg", "Peg-cold": "Cold"}
STYLES = {
    "Seq": ("o", "-", "#222222"),
    "Inf": ("^", "--", "#536d80"),
    "Inf+Peg": ("s", "-.", "#00838b"),
    "Peg-cold": ("D", ":", "#af480f"),
}


def load_results(folder):
    protocol = json.loads((folder / "protocol.json").read_text())
    if protocol["bounds"] != "paired" or set(protocol["algorithms"]) != set(STYLES):
        raise ValueError("Expected the paired four-method experiment.")
    records = defaultdict(list)
    triples = []
    for case in CASES:
        keys = set()
        for path in sorted((folder / case / "instances").glob("*.json")):
            run = json.loads(path.read_text())
            key = (run["n"], run["instance"])
            if (not run["checked"] or run["bounds"] != case or run["seed"] != protocol["seed"]
                    or key in keys or set(run["results"]) != set(STYLES)):
                raise ValueError(f"Invalid or duplicate record: {path}")
            keys.add(key)
            for method in STYLES:
                records[case, key[0], method].append(run["results"][method]["seconds"])
        triples.append(keys)
    if not triples[0] or not triples[0] == triples[1] == triples[2]:
        raise ValueError("The three settings must contain identical base instances.")
    stats = {key: (np.mean(t), np.std(t, ddof=0) / np.mean(t)) for key, t in records.items()}
    sizes = sorted({n for n, _ in triples[0]})
    with (folder / "paired_summary.csv").open(newline="") as stream:
        expected = {(n, m) for n in sizes for m in STYLES}
        for row in csv.DictReader(stream):
            n, method = int(row["n"]), row["algorithm"]
            expected.remove((n, method))
            for case in CASES:
                key = case, n, method
                if (int(row["pairs"]) != len(records[key]) or
                        not np.allclose(stats[key], [float(row[f"{case}_mean_s"]),
                                                     float(row[f"{case}_cov"])],
                                        rtol=1e-10, atol=0)):
                    raise ValueError(f"Paired summary disagrees with raw runs: {key}")
        if expected:
            raise ValueError("Incomplete paired summary.")
    return sizes, records, stats


def draw(sizes, records, stats, filename):
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.02, 3.02), sharex=True, sharey=True)
    fig.subplots_adjust(left=.068, right=.995, top=.936, bottom=.419, wspace=.13)
    ticks = 10.0 ** np.arange(np.floor(np.log10(sizes[0])), np.ceil(np.log10(sizes[-1])) + 1)
    for ax, case, panel in zip(axes, CASES, "abc"):
        ax.set(xscale="log", yscale="log", xlabel="Number of variables")
        ax.set_title(f"({panel}) {case.capitalize()}", fontsize=9, pad=3)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_locator(FixedLocator(10.0 ** np.arange(-5, 3)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", color=".82", linewidth=.35)
        ax.tick_params(width=.45, length=2, labelsize=7, pad=2)
        handles = []
        for method, (marker, style, color) in STYLES.items():
            face = color if method in ("Seq", "Inf") else "none"
            obs = [records[case, n, method] for n in sizes]
            ax.scatter(np.repeat(sizes, [len(x) for x in obs]), np.concatenate(obs),
                       s=4.5, marker=marker, edgecolors=color, facecolors=face,
                       linewidths=.32, alpha=.32, zorder=2)
            means = np.array([stats[case, n, method][0] for n in sizes])
            exponent, intercept = np.polyfit(np.log(sizes), np.log(means), 1)
            coef = np.exp(intercept)
            ax.plot(sizes, coef * np.asarray(sizes, dtype=float)**exponent,
                    linestyle=style, color=color, linewidth=.8, zorder=3)
            ax.plot(sizes, means, linestyle="none", marker=marker, markersize=3.0,
                    markerfacecolor=face, markeredgewidth=.65, color=color, zorder=4)
            sig, power = f"{coef:.2e}".split("e")
            label = rf"{DISPLAY_NAMES[method]}: ${sig}\cdot10^{{{int(power)}}}\cdot n^{{{exponent:.3f}}}$"
            handles.append(Line2D([], [], color=color, linestyle=style, marker=marker,
                                  markerfacecolor=face, markersize=3.1, linewidth=.8, label=label))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.39),
                  borderaxespad=0, frameon=False, fontsize=7.4, labelspacing=.20,
                  handlelength=1.9, handletextpad=.44)
    axes[0].set_ylabel("Execution time (s)", labelpad=2)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)
    plt.close(fig)


def scientific(value, best):
    mantissa, exponent = f"{value:.2e}".split("e")
    tex = rf"{mantissa}\!\cdot\!10^{{{int(exponent)}}}"
    return "$" + (r"\mathbf{" + tex + "}" if best else tex) + "$"





def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=here / "results")
    parser.add_argument("--output", type=Path, default=here.parent / "codes")

    args = parser.parse_args()
    sizes, records, stats = load_results(args.results)
    draw(sizes, records, stats, args.output / "runtime-by-configuration.pdf")
    


if __name__ == "__main__":
    main()
