#!/usr/bin/env python3
"""Configurations of FusedHash on the A100, on every TC and SG input of real.py, synthetic.py and
large.py (TC). The ablation (bench/ablation.py) shows that two defaults of the engine are not the best
on the A100: for TC one thread per candidate (--split output) is faster than one per tuple of Delta,
and the looping kernel with its default threshold (one block of 1024 threads up to 262,144 candidates)
slows down the grids and the trees, where a CUDA launch costs a few microseconds. All with the word-level join (--words 1):

  tc  output+loop   one thread per candidate, looping kernel (--loop 262144)
  tc  output        one thread per candidate, no looping kernel
  tc  delta         one thread per tuple of Delta (the default of the engine for TC), no looping kernel
  sg  output        one thread per candidate (the default), no looping kernel

  python3 bench/configs.py [--only tc sg] [--datasets ...] [--resume]   -> results/configs.csv
"""
import argparse
import harness as H
import real, synthetic, large

CONFIGS = {"tc": {"output+loop": dict(split="output", loop=262144, words=1),
                  "output": dict(split="output", loop=0, words=1),
                  "delta": dict(split="delta", loop=0, words=1)},
           "sg": {"output": dict(split="output", loop=0, words=1)}}
FIELDS = ["key", "dataset", "program", "config", "result", "fusedhash_s", "fusedhash_plan", "fusedhash_set",
          "loop_iters", "word_iters", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["tc", "sg"])
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = H.Csv("configs.csv", FIELDS, a.resume)
    for prog in a.only:
        names = (real.TC + synthetic.TC + large.TC) if prog == "tc" else (real.SG + synthetic.SG)
        for name in dict.fromkeys(names):
            for cfg, opts in CONFIGS[prog].items():
                key = f"{prog}/{cfg}"
                if (a.datasets and name not in a.datasets) or (key, name) in out.done:
                    continue
                fh, P = H.with_plans("fusedhash_tcsg", prog == "tc", prog=prog, input=H.graph_bin(name), neq=0, **opts)
                print(f"{prog} {name:11s} {cfg:12s} {fh.data.get('result', '')!s:>14}  {fh.cell()} (P{P})", flush=True)
                out.row(key=key, dataset=name, program=prog, config=cfg, result=fh.data.get("result", ""),
                        fusedhash_s=fh.cell(), fusedhash_plan=P, fusedhash_set=fh.data.get("set_mode", ""),
                        loop_iters=fh.data.get("loop_iters", ""), word_iters=fh.data.get("word_iters", ""),
                        notes=fh.message if fh.kind == "error" else "")


if __name__ == "__main__":
    main()
