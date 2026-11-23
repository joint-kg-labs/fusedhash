#!/usr/bin/env python3
"""The largest inputs, on the A100 (80 GiB of GPU memory) with 216 GiB of host memory: the same programs, systems and rules of measure as bench/synthetic.py and
bench/aggregation.py.

  TC  G60K, G80K (G_n with p = 0.001: 3.6 and 6.4 billion facts)   FusedHash, SortMerge, FVlog, Souffle
  CC, SSSP  RMAT-25, RMAT-26 (32 M and 64 M vertices, 10 edges each), Grid4000 (16 M vertices)

  python3 data/prepare.py --only large                  (the inputs, once)
  python3 bench/large.py [--only tc cc sssp ctrl] [--datasets ...] [--resume]
      -> results/large_tc.csv, results/large_agg.csv
"""
import argparse, json, os
import harness as H

TC = ["G60K", "G80K"]
AGG = {"cc": ["RMAT-25", "RMAT-26", "Grid4000"], "sssp": ["RMAT-25", "RMAT-26", "Grid4000"], "ctrl": []}
TC_FIELDS = ["program", "dataset", "nodes", "edges", "result", "fusedhash_s", "fusedhash_plan", "fusedhash_set",
             "sortmerge_s", "sortmerge_plan", "fvlog_s", "fvlog_result", "souffle_s", "souffle_rss_gib", "agree", "notes"]
AGG_FIELDS = ["program", "dataset", "rows", "result", "checksum", "iters_hash", "iters_sort", "map_keys",
              "fusedhash_s", "sortmerge_s", "agree", "notes"]


def errors(*named):
    return "; ".join(f"{n}: {o.message}" for n, o in named if o.kind == "error")


def tc(a):
    out = H.Csv("large_tc.csv", TC_FIELDS, a.resume)
    for name in TC:
        if (a.datasets and name not in a.datasets) or ("tc", name) in out.done:
            continue
        nodes, edges = H.graph_shape(name)
        fh, fhP = H.with_plans("fusedhash_tcsg", True, prog="tc", input=H.graph_bin(name), neq=0)
        sm, smP = H.with_plans("sortmerge_tcsg", True, prog="tc", input=H.graph_bin(name), neq=0)
        fv = H.fvlog("TC_TEST", H.graph_facts(name))
        sf = H.souffle("tc_fvlog", H.graph_facts_dir(name))
        ok = H.agree(fh, sm, fv, sf)
        res = next((o.data["result"] for o in (fh, sm, fv, sf) if o.ok and o.data.get("result") is not None), "")
        print(f"tc {name:11s} {res!s:>14}  fusedhash {fh.cell()} (P{fhP})  sortmerge {sm.cell()} (P{smP})  "
              f"fvlog {fv.cell()}  souffle {sf.cell()}  {'' if ok else 'MISMATCH'}", flush=True)
        out.row(program="tc", dataset=name, nodes=nodes, edges=edges, result=res,
                fusedhash_s=fh.cell(), fusedhash_plan=fhP, fusedhash_set=fh.data.get("set_mode", ""),
                sortmerge_s=sm.cell(), sortmerge_plan=smP, fvlog_s=fv.cell(),
                fvlog_result=fv.data.get("result", "") if fv.ok else "", souffle_s=sf.cell("time", "{:.2f}"),
                souffle_rss_gib=f"{sf.data['rss_gib']:.1f}" if sf.data.get("rss_gib") else "", agree=int(ok),
                notes=errors(("fusedhash", fh), ("sortmerge", sm), ("fvlog", fv), ("souffle", sf)))


def agg(a, progs):
    out = H.Csv("large_agg.csv", AGG_FIELDS, a.resume)
    for prog in progs:
        for name in AGG[prog]:
            if (a.datasets and name not in a.datasets) or (prog, name) in out.done:
                continue
            path = os.path.join(H.DATA, "agg", f"{name}.bin")
            meta = json.load(open(os.path.join(H.DATA, "agg", f"{name}.json")))
            args = dict(prog=prog, input=path, source=meta["source"])
            fh = H.engine("fusedhash_agg", **args)
            sm = H.engine("sortmerge_agg", **args)
            same = not (fh.ok and sm.ok) or (fh.data["result"] == sm.data["result"] and
                                              (prog == "ctrl" or fh.data["checksum"] == sm.data["checksum"]))
            ref = fh if fh.ok else sm
            print(f"{prog:4s} {name:15s} {ref.data.get('result', '')!s:>12}  hash {fh.cell()} "
                  f"({fh.data.get('iters', '')} it)  sort {sm.cell()} ({sm.data.get('iters', '')} it)  "
                  f"{'' if same else 'MISMATCH'}", flush=True)
            out.row(program=prog, dataset=name, rows=meta["rows"], result=ref.data.get("result", ""),
                    checksum=ref.data.get("checksum", ""), iters_hash=fh.data.get("iters", ""),
                    iters_sort=sm.data.get("iters", ""), map_keys=ref.data.get("map_keys", ""),
                    fusedhash_s=fh.cell(), sortmerge_s=sm.cell(), agree=int(same),
                    notes=errors(("fusedhash", fh), ("sortmerge", sm)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["tc", "cc", "sssp", "ctrl"])
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    progs = [p for p in ("cc", "sssp", "ctrl") if p in a.only]
    if progs:                                   # first the aggregations (GPU only, minutes), then TC (Souffle up to 900 s)
        agg(a, progs)
    if "tc" in a.only:
        tc(a)


if __name__ == "__main__":
    main()
