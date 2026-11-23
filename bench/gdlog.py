#!/usr/bin/env python3
"""GDlog (github.com/harp-lab/gdlog; GPUlog, "Optimizing Datalog for the GPU", ASPLOS 2025) on every TC
and SG input of the paper, run alone so that the other systems need not be measured again: its result
is compared with the result of the same program and dataset in results/real.csv, synthetic.csv and
large_tc.csv (measured on the same VM), and the times of FVlog and Souffle there, and of FusedHash in its
final configuration (results/fusedhash_final.csv, the one of the paper), are copied beside it.

  python3 bench/gdlog.py [--only tc sg] [--datasets ...] [--resume]   -> results/gdlog.csv

GDlog's printed timers leave out the loading of the relations (copy to the GPU, sort, indexes), which
our engines include: gdlog_s is its own time (the fixpoint; SG: base rule + fixpoint), gdlog_load_s the
loading (printed by the drivers patched by gdlog/patch_gdlog.py), gdlog_total_s their sum, the time to
compare with fusedhash_s; gdlog_wall_s is the wall time of its process (reading the text file and the
initialization of CUDA included).  agree is empty when no other system has a result to compare with.
CSPA is in bench/cspa.py, with every system.
"""
import argparse, csv, os
import harness as H
import real, synthetic, large

FIELDS = ["program", "dataset", "result", "ref_result", "gdlog_s", "gdlog_load_s", "gdlog_total_s",
          "gdlog_wall_s", "fusedhash_s", "fvlog_s", "souffle_s", "agree", "notes"]


def reference():
    """(program, dataset) -> the row of the results already measured (result and times), with the time
    of FusedHash in its final configuration when results/fusedhash_final.csv has it."""
    ref = {}
    for name in ("real.csv", "synthetic.csv", "large_tc.csv"):
        path = os.path.join(H.RESULTS, name)
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                ref[(r["program"], r["dataset"])] = dict(r)
    final = os.path.join(H.RESULTS, "fusedhash_final.csv")
    if os.path.exists(final):
        for r in csv.DictReader(open(final)):
            if r["suite"] in ("real", "synthetic", "large") and (r["program"], r["dataset"]) in ref:
                ref[(r["program"], r["dataset"])]["fusedhash_s"] = r["fusedhash_s"]
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["tc", "sg"])
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    ref = reference()
    out = H.Csv("gdlog.csv", FIELDS, a.resume)
    lists = {"tc": real.TC + synthetic.TC + large.TC, "sg": real.SG + synthetic.SG}
    for prog in a.only:
        for name in dict.fromkeys(lists[prog]):
            if (a.datasets and name not in a.datasets) or (prog, name) in out.done:
                continue
            if not os.path.exists(H.graph_facts(name)):
                print(f"{prog} {name}: {H.graph_facts(name)} missing (data/prepare.py)", flush=True)
                continue
            gd = H.gdlog("TC" if prog == "tc" else "SG", H.graph_facts(name))
            r = ref.get((prog, name), {})
            want = r.get("result", "")
            got = gd.data.get("result") if gd.ok else None
            agree = "" if not (gd.ok and want) else int(str(got) == str(want))
            load = gd.data.get("load") if gd.ok else None
            total = gd.cell("total") if gd.ok and load is not None else ("" if gd.ok else gd.cell())
            print(f"{prog} {name:11s} gdlog {gd.cell()} + load {'' if load is None else f'{load:.4f}'} = {total} "
                  f"(wall {gd.cell('wall', '{:.2f}')})  result {got}  expected {want or '?'}  "
                  f"fusedhash {r.get('fusedhash_s', '')}  fvlog {r.get('fvlog_s', '')}"
                  f"  {'MISMATCH' if agree == 0 else ''}", flush=True)
            out.row(program=prog, dataset=name, result=got if got is not None else "", ref_result=want,
                    gdlog_s=gd.cell(), gdlog_load_s="" if load is None else f"{load:.4f}", gdlog_total_s=total,
                    gdlog_wall_s=gd.cell("wall", "{:.2f}"),
                    fusedhash_s=r.get("fusedhash_s", ""), fvlog_s=r.get("fvlog_s", ""),
                    souffle_s=r.get("souffle_s", ""), agree=agree,
                    notes=gd.message if gd.kind == "error" else "")


if __name__ == "__main__":
    main()
