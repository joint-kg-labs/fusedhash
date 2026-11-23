#!/usr/bin/env python3
"""Appendix: context-sensitive dataflow analysis (CSDA, Graspan) on httpd, linux and postgresql, the
inputs of the FlowLog artifact, with FusedHash (its final configuration, as bench/fusedhash_final.py),
SortMerge, FVlog and GDlog (our CSDA drivers, fvlog/csda_test.cu and gdlog/csda.cu, written as their
TC drivers) and Souffle (one thread per core), all on the VM.

  NullNode(x,y) :- NullEdge(x,y).
  NullNode(x,y) :- NullNode(x,w), Edge(w,y).

It is TC with the recursion seeded by another relation: our engines run TC on Edge with --base NullEdge
(the seed facts), decomposable by x like TC, so the plans of with_plans apply.

  python3 data/prepare.py --only csda      (the inputs, once: downloaded from HuggingFace if missing)
  python3 bench/csda.py [--resume]         -> results/csda.csv
"""
import argparse, os
import harness as H
from fusedhash_final import CFG

NAMES = ["httpd", "linux", "postgresql"]
FIELDS = ["dataset", "edges", "null_edges", "result", "iters", "fusedhash_s", "fusedhash_plan", "fusedhash_set",
          "sortmerge_s", "sortmerge_plan", "fvlog_s", "gdlog_s", "gdlog_wall_s", "souffle_s", "souffle_rss_gib",
          "agree", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--datasets", nargs="*")
    a = ap.parse_args()
    out = H.Csv("csda.csv", FIELDS, a.resume)
    for n in NAMES:
        if (a.datasets and n not in a.datasets) or any(d == n for d, *_ in out.done):
            continue
        b = os.path.join(H.DATA, "csda_bin", n)
        edge, null = os.path.join(b, "edge.bin"), os.path.join(b, "nulledge.bin")
        if not os.path.exists(null):
            print(f"{n}: {null} missing (python3 data/prepare.py --only csda)", flush=True)
            continue
        csv_dir = os.path.join(H.DATA, "csda", n, f"csda-{n}")
        fh, fhP = H.with_plans("fusedhash_tcsg", True, prog="tc", input=edge, base=null, neq=0, **CFG)
        sm, smP = H.with_plans("sortmerge_tcsg", True, prog="tc", input=edge, base=null)
        facts = os.path.join(H.DATA, "csda_facts", n)
        pair = (os.path.join(facts, "edge.facts"), os.path.join(facts, "null_edge.facts"))
        fv = H.fvlog("CSDA_TEST", pair)
        gd = H.gdlog("CSDA", pair)
        sf = H.souffle("csda", csv_dir)
        ok = H.agree(fh, sm, fv, gd, sf)
        res = next((o.data["result"] for o in (fh, sm, fv, gd, sf) if o.ok and o.data.get("result") is not None), "")
        print(f"{n:10s} {res!s:>14}  fusedhash {fh.cell()} (P{fhP})  sortmerge {sm.cell()} (P{smP})  "
              f"fvlog {fv.cell()}  gdlog {gd.cell()}  souffle {sf.cell('time', '{:.2f}')}  "
              f"{'' if ok else 'MISMATCH'}", flush=True)
        out.row(dataset=n, edges=os.path.getsize(edge) // 8, null_edges=os.path.getsize(null) // 8, result=res,
                iters=fh.data.get("iters", ""), fusedhash_s=fh.cell(), fusedhash_plan=fhP,
                fusedhash_set=fh.data.get("set_mode", ""), sortmerge_s=sm.cell(), sortmerge_plan=smP,
                fvlog_s=fv.cell(), gdlog_s=gd.cell(), gdlog_wall_s=gd.cell("wall", "{:.2f}"),
                souffle_s=sf.cell("time", "{:.2f}"),
                souffle_rss_gib=f"{sf.data['rss_gib']:.1f}" if sf.data.get("rss_gib") else "", agree=int(ok),
                notes="; ".join(f"{k}: {o.message}" for k, o in (("fusedhash", fh), ("sortmerge", sm), ("fvlog", fv),
                                                                 ("gdlog", gd), ("souffle", sf)) if o.kind == "error"))


if __name__ == "__main__":
    main()
