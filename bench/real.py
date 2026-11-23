#!/usr/bin/env python3
"""Table of the real graphs (FVlog datasets): TC and SG with FusedHash, SortMerge, the real FVlog
and Souffle, all on the same VM.

  python3 bench/real.py [--only tc sg] [--datasets fe_sphere ...] [--resume]   -> results/real.csv

Formulations. TC: Reach(x,y) <- Edge(x,y). Reach(x,z) <- Reach(x,y), Edge(y,z).
SG: the formulation of the FVlog code (test/src/sg_test.cu), with x != y in the base rule only:
SG(x,y) <- Edge(p,x), Edge(p,y), x != y.  SG(x,y) <- Edge(a,x), SG(a,b), Edge(b,y).
(The FVlog paper writes x != y in both rules; the counts differ by the pairs x = y at most.)
"""
import argparse
import harness as H

TC = ["fe_sphere", "fe_body", "fe_ocean", "Gnutella31", "usroads", "vsp_finan", "com-dblp", "CA-HepTh"]
SG = ["CA-HepTh", "fe_ocean", "fe_sphere", "fe_body", "vsp_finan", "Gnutella31", "usroads", "com-dblp"]
FIELDS = ["program", "dataset", "nodes", "edges", "result", "fusedhash_s", "fusedhash_plan", "fusedhash_set",
          "sortmerge_s", "sortmerge_plan", "fvlog_s", "fvlog_result", "souffle_s", "souffle_rss_gib", "agree", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["tc", "sg"])
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = H.Csv("real.csv", FIELDS, a.resume)
    for prog in a.only:
        for name in (TC if prog == "tc" else SG):
            if (a.datasets and name not in a.datasets) or (prog, name) in out.done:
                continue
            nodes, edges = H.graph_shape(name)
            fh, fhP = H.with_plans("fusedhash_tcsg", prog == "tc", prog=prog, input=H.graph_bin(name), neq=0)
            sm, smP = H.with_plans("sortmerge_tcsg", prog == "tc", prog=prog, input=H.graph_bin(name), neq=0)
            fv = H.fvlog("TC_TEST" if prog == "tc" else "SG_TEST", H.graph_facts(name))
            sf = H.souffle("tc_fvlog" if prog == "tc" else "sg_bigdatalog", H.graph_facts_dir(name))
            ok = H.agree(fh, sm, fv, sf)
            notes = "; ".join(f"{n}: {o.message}" for n, o in (("fusedhash", fh), ("sortmerge", sm), ("fvlog", fv),
                                                                ("souffle", sf)) if o.kind == "error")
            res = next((o.data["result"] for o in (fh, sm, fv, sf) if o.ok and o.data.get("result") is not None), "")
            print(f"{prog} {name:11s} {res!s:>14}  fusedhash {fh.cell()} (P{fhP})  sortmerge {sm.cell()} (P{smP})  "
                  f"fvlog {fv.cell()}  souffle {sf.cell()}  {'' if ok else 'MISMATCH'}", flush=True)
            out.row(program=prog, dataset=name, nodes=nodes, edges=edges, result=res,
                    fusedhash_s=fh.cell(), fusedhash_plan=fhP, fusedhash_set=fh.data.get("set_mode", ""),
                    sortmerge_s=sm.cell(), sortmerge_plan=smP, fvlog_s=fv.cell(),
                    fvlog_result=fv.data.get("result", "") if fv.ok else "", souffle_s=sf.cell("time", "{:.2f}"),
                    souffle_rss_gib=f"{sf.data['rss_gib']:.1f}" if sf.data.get("rss_gib") else "",
                    agree=int(ok), notes=notes)


if __name__ == "__main__":
    main()
