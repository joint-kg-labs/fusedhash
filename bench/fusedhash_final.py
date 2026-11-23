#!/usr/bin/env python3
"""FusedHash on the A100 with its configuration for this GPU, chosen with bench/ablation.py,
bench/configs.py and bench/loop_threshold.py: one thread per candidate for TC and SG (--split output),
the word-level join (--words 1) and the looping kernel only while an iteration has at most 16384
candidates (--loop 16384; the default of the engine, 262144, slows down the grids and the trees on the
A100, while 16384 keeps the whole gain on the chains). This is the configuration of FusedHash in all the
tables and figures of the paper. The other systems do not change: their times stay
those of real.py, synthetic.py, large.py and secondary.py.

Every TC and SG input of those experiments, best of two runs, the plans of with_plans (the first
number of partitions that fits); then the partitions of the largest TC inputs (P = 1..16, as
partitions_large.py). Rows: suite, program, dataset, P (the plan, or the forced number of partitions).

  python3 bench/fusedhash_final.py [--resume]   -> results/fusedhash_final.csv
"""
import argparse
import harness as H
import real, synthetic, large, partitions_large
from secondary import UNIFORM, FORESTS

CFG = dict(split="output", words=1, loop=16384)
FIELDS = ["key", "dataset", "suite", "program", "P", "result", "fusedhash_s", "fusedhash_set", "largest_part",
          "peak_used_gib", "peak_reserved_gib", "iters", "loop_iters", "word_iters", "notes"]


def cases():
    for suite, mod in (("real", real), ("synthetic", synthetic)):
        for prog in ("tc", "sg"):
            for name in (mod.TC if prog == "tc" else mod.SG):
                yield suite, prog, name
    for name in large.TC:
        yield "large", "tc", name
    for name in UNIFORM:
        yield "table5", "tc", name
    for name in FORESTS:
        yield "table5", "sg", name
    for name in ("chain-1000", "chain-3000", "chain-10000"):
        yield "chain", "tc", name


def record(out, key, suite, prog, name, o, P):
    d = o.data
    print(f"{suite:9s} {prog} {name:15s} P={P!s:<3} {d.get('result', '')!s:>12}  {o.cell()}  "
          f"peak {d.get('peak_used_gib', o.kind)}", flush=True)
    out.row(key=key, dataset=name, suite=suite, program=prog, P=P, result=d.get("result", ""), fusedhash_s=o.cell(),
            fusedhash_set=d.get("set_mode", ""), largest_part=d.get("largest_part", ""),
            peak_used_gib=f"{d['peak_used_gib']:.2f}" if o.ok else ("OOM" if o.kind == "oom" else o.kind),
            peak_reserved_gib=f"{d['peak_reserved_gib']:.2f}" if o.ok else "", iters=d.get("iters", ""),
            loop_iters=d.get("loop_iters", ""), word_iters=d.get("word_iters", ""),
            notes=o.message if o.kind == "error" else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = H.Csv("fusedhash_final.csv", FIELDS, a.resume)
    for suite, prog, name in cases():
        key = f"{suite}/{prog}"
        if (key, name) in out.done:
            continue
        o, P = H.with_plans("fusedhash_tcsg", prog == "tc", prog=prog, input=H.graph_bin(name), neq=0, **CFG)
        record(out, key, suite, prog, name, o, P)
    for name in partitions_large.NAMES:
        for P in (1, 2, 4, 8, 16):
            key = f"partitions/P{P}"
            if (key, name) in out.done:
                continue
            o = H.engine("fusedhash_tcsg", prog="tc", input=H.graph_bin(name), plan=P, **CFG)
            record(out, key, "partitions", "tc", name, o, P)


if __name__ == "__main__":
    main()
