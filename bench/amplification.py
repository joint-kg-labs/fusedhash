#!/usr/bin/env python3
"""Rewrite amplification of the synthetic workloads:

  A = sum_i |full_i| / |result|

where |full_i| is the number of facts accumulated after iteration i, i.e., the facts that the eager merge
of the sort-based design rewrites at iteration i. It is a property of the workload (the same for every
synchronous semi-naive evaluation); here it is read from the per-iteration log of SortMerge
(--verbose 1, single plan, one run).

  python3 bench/amplification.py [--only tc sg] [--datasets Grid50 ...]   -> results/rewrite_amplification.csv

The values of the paper are in results/a100/rewrite_amplification.csv.
"""
import argparse, csv, os, re, subprocess
import harness as H

WORKLOADS = {   # the synthetic workloads of the paper that SortMerge completes with the single plan
    "tc": ["G20K", "G10K", "Tree11", "G5K", "Tree13", "Tree15", "Tree17",
           "Grid50", "Grid100", "Grid150", "Grid200", "Grid250"],
    "sg": ["Tree6", "Tree7", "Tree8", "Tree9", "Tree10", "G5K",
           "Grid50", "Grid100", "Grid150", "Grid200", "Grid250"],
}
FULL = re.compile(r"\|full\|=(\d+)")


def amplification(prog, name, timeout):
    cmd = [os.path.join(H.BIN, "sortmerge_tcsg"), "--prog", prog, "--input", H.graph_bin(name),
           "--neq", "0", "--reps", "1", "--verbose", "1"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    sizes = [int(x) for x in FULL.findall(p.stderr)]
    if p.returncode != 0 or not sizes:
        raise RuntimeError(f"{prog} {name}: exit code {p.returncode}\n{p.stderr[-2000:]}")
    return len(sizes), sum(sizes) / sizes[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(WORKLOADS))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    out = os.path.join(H.ROOT, "results", "rewrite_amplification.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["program", "dataset", "iterations", "rewrite_amplification"])
        for prog in a.only:
            for name in WORKLOADS[prog]:
                if a.datasets and name not in a.datasets:
                    continue
                it, amp = amplification(prog, name, a.timeout)
                w.writerow([prog, name, it, f"{amp:.2f}"])
                f.flush()
                print(f"{prog} {name:8s} {it:5d} iterations  A = {amp:.2f}", flush=True)


if __name__ == "__main__":
    main()
