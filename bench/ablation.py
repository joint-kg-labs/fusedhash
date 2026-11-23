#!/usr/bin/env python3
"""Ablation studies of FusedHash on the GPU VM, one component switched at a time, on every TC
and SG workload of real.py and synthetic.py:

  base      hash table only (--bitmap off), one atomicAdd per new fact
  bitmap    adaptive choice between hash table and bitmap
  warp      bitmap + warp-aggregated append (one atomicAdd per group of active threads)
  split     bitmap + the other work distribution (one thread per candidate for TC, per tuple of
            Delta for SG)
  loop      bitmap + looping kernel when Delta is small (--loop)
  words     bitmap + word-level join when Delta is dense (--words)
  all       bitmap + loop + words: the default configuration of the engine
The variants before "loop" switch off the looping kernel and the word-level join, so that each one
measures a single optimization.

  python3 bench/ablation.py [--only tc sg] [--variants base bitmap loop words all] [--resume]
      -> results/ablation.csv
"""
import argparse
import harness as H
from real import TC as RTC, SG as RSG
from synthetic import TC as STC, SG as SSG

OFF = dict(loop=0, words=0)
VARIANTS = {"base": dict(bitmap="off", **OFF), "bitmap": dict(bitmap="auto", **OFF),
            "warp": dict(bitmap="auto", append="warp", **OFF), "split": dict(bitmap="auto", split="other", **OFF),
            "loop": dict(bitmap="auto", loop=262144, words=0), "words": dict(bitmap="auto", loop=0, words=1),
            "all": dict(bitmap="auto", loop=262144, words=1)}
FIELDS = ["key", "variant", "suite", "program", "dataset", "result", "plan", "time_s", "set_mode", "set_gib",
          "loop_iters", "word_iters"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["tc", "sg"])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = H.Csv("ablation.csv", FIELDS, a.resume)
    work = [("real", p, n) for p, names in (("tc", RTC), ("sg", RSG)) for n in names] + \
           [("synthetic", p, n) for p, names in (("tc", STC), ("sg", SSG)) for n in names]
    for suite, prog, name in work:
        if prog not in a.only or (a.datasets and name not in a.datasets):
            continue
        for v in a.variants:
            key = f"{suite}/{prog}/{name}"
            if (key, v) in out.done:
                continue
            opts = dict(VARIANTS[v])
            if opts.get("split") == "other":
                opts["split"] = "output" if prog == "tc" else "delta"
            o, P = H.with_plans("fusedhash_tcsg", prog == "tc", prog=prog, input=H.graph_bin(name), neq=0, **opts)
            print(f"{suite:9s} {prog} {name:11s} {v:7s} {o.cell()} (P{P}) {o.data.get('set_mode', '')}", flush=True)
            out.row(key=key, variant=v, suite=suite, program=prog, dataset=name, result=o.data.get("result", ""),
                    plan=P, time_s=o.cell(), set_mode=o.data.get("set_mode", ""),
                    set_gib=f"{o.data['set_bytes'] / 2**30:.3f}" if o.ok else "",
                    loop_iters=o.data.get("loop_iters", ""), word_iters=o.data.get("word_iters", ""))


if __name__ == "__main__":
    main()
