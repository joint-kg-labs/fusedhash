#!/usr/bin/env python3
"""Peak GPU memory of FusedHash and SortMerge on every input of the experiments (the high watermark
of the memory pool in the run: `peak_used_gib` the memory allocated, `peak_reserved_gib` the memory the
pool took from the device). One run per case (the warm-up included), with the engines of bin_mem/,
built with the peak in their RESULT line: make -C engines BIN=../bin_mem.

  python3 bench/memory.py [--only tc sg cc sssp ctrl cspa] [--resume]   -> results/memory.csv

Out of memory: the run stopped at the 80 GB of the A100 (peak "OOM").
"""
import argparse, json, os
import harness as H
import real, synthetic, aggregation, large, cspa

H.BIN = os.path.join(H.ROOT, "bin_mem")
FIELDS = ["program", "dataset", "result", "fusedhash_plan", "fusedhash_peak_used_gib", "fusedhash_peak_reserved_gib",
          "sortmerge_plan", "sortmerge_peak_used_gib", "sortmerge_peak_reserved_gib", "notes"]


def peak(o, key):
    if o.kind == "oom":
        return "OOM"
    return f"{o.data[key]:.2f}" if o.ok and key in o.data else o.kind


def cases(only):
    for prog in ("tc", "sg"):
        if prog in only:
            names = (real.TC + synthetic.TC + large.TC) if prog == "tc" else (real.SG + synthetic.SG)
            for name in dict.fromkeys(names):
                yield prog, name
    for prog in ("cc", "sssp", "ctrl"):
        if prog in only:
            for name in dict.fromkeys(aggregation.SUITE[prog] + large.AGG.get(prog, [])):
                yield prog, name
    if "cspa" in only:
        for name in cspa.NAMES:
            yield "cspa", name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["tc", "sg", "cc", "sssp", "ctrl", "cspa"])
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = H.Csv("memory.csv", FIELDS, a.resume)
    for prog, name in cases(a.only):
        if (prog, name) in out.done:
            continue
        if prog in ("tc", "sg"):
            if not os.path.exists(H.graph_bin(name)):
                continue
            fh, fhP = H.with_plans("fusedhash_tcsg", prog == "tc", prog=prog, input=H.graph_bin(name), neq=0, reps=1)
            sm, smP = H.with_plans("sortmerge_tcsg", prog == "tc", prog=prog, input=H.graph_bin(name), neq=0, reps=1)
        elif prog == "cspa":
            fh, fhP = H.engine("fusedhash_cspa", input=os.path.join(H.DATA, "cspa_facts", name), reps=1), 1
            sm, smP = H.Outcome("missing", message="no SortMerge for CSPA"), ""
        else:
            path = os.path.join(H.DATA, "agg", f"{name}.bin")
            if not os.path.exists(path):
                continue
            meta = json.load(open(os.path.join(H.DATA, "agg", f"{name}.json")))
            args = dict(prog=prog, input=path, source=meta["source"], reps=1)
            fh, fhP = H.engine("fusedhash_agg", **args), 1
            sm, smP = H.engine("sortmerge_agg", **args), 1
        res = next((o.data.get("result") for o in (fh, sm) if o.ok), "")
        print(f"{prog:4s} {name:15s} {res!s:>14}  fusedhash {peak(fh, 'peak_used_gib'):>7} GiB (P{fhP})  "
              f"sortmerge {peak(sm, 'peak_used_gib'):>7} GiB (P{smP})", flush=True)
        out.row(program=prog, dataset=name, result=res, fusedhash_plan=fhP,
                fusedhash_peak_used_gib=peak(fh, "peak_used_gib"), fusedhash_peak_reserved_gib=peak(fh, "peak_reserved_gib"),
                sortmerge_plan=smP, sortmerge_peak_used_gib=peak(sm, "peak_used_gib"),
                sortmerge_peak_reserved_gib=peak(sm, "peak_reserved_gib"),
                notes="; ".join(f"{n}: {o.message}" for n, o in (("fusedhash", fh), ("sortmerge", sm)) if o.kind == "error"))


if __name__ == "__main__":
    main()
