#!/usr/bin/env python3
"""GDlog on the chains of the paper (TC on chains of 1,000, 3,000 and 10,000 vertices: one iteration per
vertex, few facts per iteration), run alone: the result is compared with results/chain.csv (measured on the
same machine), and the times of FusedHash (final configuration, results/fusedhash_final.csv, suite chain), of
FusedHash without the looping kernel, of FVlog and of Souffle with one thread there are copied beside.

  python3 bench/chain_gpu.py [--resume]   -> results/chain_gpu.csv

gdlog_s: its fixpoint; gdlog_total_s: fixpoint + loading (as in gdlog.py).
"""
import argparse, csv, os
import harness as H

NAMES = ["chain-1000", "chain-3000", "chain-10000"]
FIELDS = ["dataset", "result", "iters", "gdlog_result", "gdlog_s", "gdlog_total_s", "gdlog_wall_s",
          "fusedhash_s", "noloop_s", "fvlog_s", "souffle1_s", "agree", "notes"]


def reference():
    ref = {}
    path = os.path.join(H.RESULTS, "chain.csv")
    if os.path.exists(path):
        ref = {r["dataset"]: dict(r) for r in csv.DictReader(open(path))}
    final = os.path.join(H.RESULTS, "fusedhash_final.csv")
    if os.path.exists(final):
        for r in csv.DictReader(open(final)):
            if r["suite"] == "chain" and r["dataset"] in ref:
                ref[r["dataset"]]["fusedhash_s"] = r["fusedhash_s"]
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    ref = reference()
    out = H.Csv("chain_gpu.csv", FIELDS, a.resume)
    for name in NAMES:
        if any(d == name for d, *_ in out.done):
            continue
        r = ref.get(name, {})
        want = r.get("result", "")
        gd = H.gdlog("TC", H.graph_facts(name))
        g = gd.data.get("result") if gd.ok else None
        agree = "" if not (g is not None and want) else int(str(g) == str(want))
        load = gd.data.get("load") if gd.ok else None
        g_total = gd.cell("total") if gd.ok and load is not None else ("" if gd.ok else gd.cell())
        print(f"{name:12s} expected {want}  gdlog {gd.cell()} (total {g_total}, result {g})  "
              f"fusedhash {r.get('fusedhash_s', '')}  fvlog {r.get('fvlog_s', '')}  {'MISMATCH' if agree == 0 else ''}",
              flush=True)
        out.row(dataset=name, result=want, iters=r.get("iters", ""), gdlog_result="" if g is None else g,
                gdlog_s=gd.cell(), gdlog_total_s=g_total, gdlog_wall_s=gd.cell("wall", "{:.2f}"),
                fusedhash_s=r.get("fusedhash_s", ""), noloop_s=r.get("noloop_s", ""), fvlog_s=r.get("fvlog_s", ""),
                souffle1_s=r.get("souffle1_s", ""), agree=agree, notes=gd.message if gd.kind == "error" else "")


if __name__ == "__main__":
    main()
