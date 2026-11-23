#!/usr/bin/env python3
"""The threshold of the looping kernel of FusedHash on the A100 (--loop: the largest number of
candidates of an iteration that one block of 1024 threads runs; default 262144): on the chains
(thousands of iterations with a tiny Delta, where it pays off) and on the grids and trees (where with
a high threshold it slows down). All with one thread per candidate and the word-level join.

  python3 bench/loop_threshold.py [--resume]   -> results/loop_threshold.csv
"""
import argparse
import harness as H

THRESHOLDS = [0, 1024, 4096, 16384, 65536, 262144]
INPUTS = {"tc": ["chain-1000", "chain-3000", "chain-10000", "Grid100", "Grid150", "Grid200", "Grid250",
                 "Tree11", "Tree13", "Tree15", "Tree17"],
          "sg": ["Grid100", "Grid150", "Grid200", "Grid250", "Tree7", "Tree8", "Tree9", "Tree10"]}
FIELDS = ["key", "dataset", "program", "loop", "result", "fusedhash_s", "iters", "loop_iters", "word_iters"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = H.Csv("loop_threshold.csv", FIELDS, a.resume)
    for prog, names in INPUTS.items():
        for name in names:
            for thr in THRESHOLDS:
                key = f"{prog}/{thr}"
                if (key, name) in out.done:
                    continue
                o = H.engine("fusedhash_tcsg", prog=prog, input=H.graph_bin(name), neq=0, split="output", words=1, loop=thr)
                d = o.data
                print(f"{prog} {name:12s} loop={thr:<7d} {d.get('result', '')!s:>12}  {o.cell()}  "
                      f"(it {d.get('iters', '')}, loop it {d.get('loop_iters', '')})", flush=True)
                out.row(key=key, dataset=name, program=prog, loop=thr, result=d.get("result", ""), fusedhash_s=o.cell(),
                        iters=d.get("iters", ""), loop_iters=d.get("loop_iters", ""), word_iters=d.get("word_iters", ""))


if __name__ == "__main__":
    main()
