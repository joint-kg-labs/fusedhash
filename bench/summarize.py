#!/usr/bin/env python3
"""Summary of the results: the main numbers of the paper, computed from the CSV files of the experiments.

  python3 bench/summarize.py                     the results of the paper (results/a100)
  python3 bench/summarize.py --results results   the results of a new run

FusedHash times come from fusedhash_final.csv (the configuration of the paper); the times of the other
systems from real.csv, synthetic.csv and large_tc.csv. Speedups are other system / FusedHash; "gmean"
is the geometric mean over the workloads of a group that both systems complete.
"""
import argparse, csv, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def rows(d, name):
    path = os.path.join(d, name)
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def gmean(v):
    return math.exp(sum(map(math.log, v)) / len(v))


def line(label, v):
    if not v:
        print(f"  {label:34s} -")
        return
    print(f"  {label:34s} gmean {gmean(v):7.1f}x   min {min(v):7.2f}x   max {max(v):8.1f}x   ({len(v)} workloads)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "a100"))
    d = ap.parse_args().results

    final = {}                                   # (suite, program, dataset) -> FusedHash time (first plan row)
    for r in rows(d, "fusedhash_final.csv"):
        t = num(r["fusedhash_s"])
        if t:
            final.setdefault((r["suite"], r["program"], r["dataset"]), (t, r))
    groups = [("real", "real.csv"), ("synthetic", "synthetic.csv"), ("large", "large_tc.csv")]

    print("TC and SG: speedup of FusedHash")
    for other, col in (("FVlog", "fvlog_s"), ("Souffle", "souffle_s"), ("SortMerge", "sortmerge_s")):
        for suite, fname in groups[:2]:
            for prog in ("tc", "sg"):
                v = []
                for r in rows(d, fname):
                    fh, o = final.get((suite, prog, r["dataset"])), num(r[col])
                    if r["program"] == prog and fh and o:
                        v.append(o / fh[0])
                line(f"vs {other:9s} {suite:9s} {prog.upper()}", v)
    failed = []
    for suite, fname in groups:
        for r in rows(d, fname):
            if num(r["fvlog_s"]) is None and (suite, r["program"], r["dataset"]) in final:
                failed.append(f"{r['program'].upper()} {r['dataset']} ({r['fvlog_s']})")
    print(f"  FVlog fails, FusedHash completes: {len(failed)}: " + ", ".join(failed))

    gd = {(r["program"], r["dataset"]): r for r in rows(d, "gdlog.csv")} if os.path.exists(os.path.join(d, "gdlog.csv")) else {}
    if gd:
        print("TC and SG: speedup of FusedHash over GDlog (its own timers)")
        suite_of = {}
        for suite, fname in groups:
            for r in rows(d, fname):
                suite_of[(r["program"], r["dataset"])] = "real" if suite == "real" else "synthetic"
        for suite in ("real", "synthetic"):
            for prog in ("tc", "sg"):
                v = []
                for (p, ds), r in gd.items():
                    s_ = suite_of.get((p, ds))
                    fh = final.get(("real" if s_ == "real" else ("large" if ds in ("G60K", "G80K") else "synthetic"), p, ds))
                    if p == prog and s_ == suite and fh and num(r["gdlog_s"]):
                        v.append(num(r["gdlog_s"]) / fh[0])
                line(f"vs GDlog     {suite:9s} {prog.upper()}", v)
        gfail = [f"{p.upper()} {ds} ({r['gdlog_s']})" for (p, ds), r in gd.items() if num(r["gdlog_s"]) is None
                 and any((s_, p, ds) in final for s_ in ("real", "synthetic", "large"))]
        print(f"  GDlog fails, FusedHash completes: {len(gfail)}: " + ", ".join(gfail))
        bad = [f"{p} {ds}" for (p, ds), r in gd.items() if r["agree"] == "0"]
        print(f"  GDlog results different from the reference: {len(bad)}" + (": " + ", ".join(bad) if bad else ""))

    big = final.get(("large", "tc", "G80K"))
    if big:
        t, r = big
        print(f"  TC on G80K: {int(r['result']):,} facts in {t:.2f} s with P = {r['P']}, "
              f"peak {num(r['peak_used_gib']):.1f} GiB of GPU memory")

    print("Monotonic aggregation: speedup of FusedHash over SortMerge")
    agg = rows(d, "aggregation.csv") + rows(d, "large_agg.csv")
    for prog in ("cc", "sssp", "ctrl"):
        for fam in ("RMAT", "Grid", "Sym", "SW", "Own", "PSC"):
            v = [num(r["sortmerge_s"]) / num(r["fusedhash_s"]) for r in agg
                 if r["program"] == prog and r["dataset"].startswith(fam)
                 and num(r["sortmerge_s"]) and num(r["fusedhash_s"])]
            if v:
                line(f"{prog.upper():4s} {fam}", v)

    print("CSPA (appendix): speedup of FusedHash")
    for r in rows(d, "cspa.csv"):
        fh = num(r["fusedhash_s"])
        g = f"   vs GDlog {num(r['gdlog_s']) / fh:.1f}x" if num(r.get("gdlog_s")) else ""
        print(f"  {r['dataset']:12s} FusedHash {fh:.3f} s   vs FVlog {num(r['fvlog_s']) / fh:.1f}x{g}   "
              f"vs Souffle {num(r['souffle_s']) / fh:.0f}x")

    if os.path.exists(os.path.join(d, "csda.csv")):
        print("CSDA (appendix): speedup of FusedHash")
        for r in rows(d, "csda.csv"):
            fh = num(r["fusedhash_s"])
            print(f"  {r['dataset']:12s} FusedHash {fh:.3f} s   " + "   ".join(
                f"vs {n} {num(r[c]) / fh:.1f}x" for n, c in (("SortMerge", "sortmerge_s"), ("FVlog", "fvlog_s"),
                                                            ("GDlog", "gdlog_s"), ("Souffle", "souffle_s")) if num(r[c]))
                  + f"   ({r['iters']} iterations, agree {r['agree']})")

    if os.path.exists(os.path.join(d, "chain_gpu.csv")):
        print("Chains: speedup of FusedHash (looping kernel) over GDlog")
        for r in rows(d, "chain_gpu.csv"):
            fh, g = num(r["fusedhash_s"]), num(r["gdlog_s"])
            if fh and g:
                print(f"  {r['dataset']:12s} FusedHash {fh:.4f} s   GDlog {g:.3f} s   {g / fh:.0f}x")


if __name__ == "__main__":
    main()
