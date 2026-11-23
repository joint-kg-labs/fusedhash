#!/usr/bin/env python3
"""The smaller experiments of the paper, on the GPU VM:

  table5      Souffle with 1 thread and one per core, FVlog and FusedHash on uniform random graphs (TC) and
              random forests (SG), as in the ablation study of the paper
  partitions  FusedHash on uniform random graphs with P = 1, 2, 4, 8, 16 partitions: time, facts of
              the largest partition and peak of GPU memory (figure of the partitioned plan)
  chain       deep recursion: chains of 1,000, 3,000 and 10,000 vertices, FusedHash without and with
              the looping kernel, FVlog, Souffle-1 (table of the chains)

  python3 bench/secondary.py [--only table5 partitions chain]   -> results/<experiment>.csv
"""
import argparse
import harness as H

UNIFORM = ["uniform-3000", "uniform-5000", "uniform-12000", "uniform-20000"]
FORESTS = ["forest-6-3-20", "forest-10-2-16", "forest-12-2-8"]


def table5():
    out = H.Csv("table5.csv", ["program", "dataset", "result", "souffle1_s", "soufflen_s", "fvlog_s", "fusedhash_s",
                               "agree"])
    for prog, names in (("tc", UNIFORM), ("sg", FORESTS)):
        for name in names:
            dl = "tc_fvlog" if prog == "tc" else "sg_bigdatalog"
            s1 = H.souffle(dl, H.graph_facts_dir(name), threads=1)
            s64 = H.souffle(dl, H.graph_facts_dir(name))
            fv = H.fvlog("TC_TEST" if prog == "tc" else "SG_TEST", H.graph_facts(name))
            fh, _ = H.with_plans("fusedhash_tcsg", prog == "tc", prog=prog, input=H.graph_bin(name))
            ok = H.agree(s1, s64, fv, fh)
            print(f"{prog} {name:15s} souffle-1 {s1.cell()} souffle-n {s64.cell()} fvlog {fv.cell()} "
                  f"fusedhash {fh.cell()} {'' if ok else 'MISMATCH'}", flush=True)
            out.row(program=prog, dataset=name, result=fh.data.get("result", ""), souffle1_s=s1.cell(),
                    soufflen_s=s64.cell(), fvlog_s=fv.cell(), fusedhash_s=fh.cell(), agree=int(ok))


PARTITION_SIZES = [5000, 12000, 20000, 30000, 40000, 46000]


def partitions():
    """One row per (graph, P): the peak is the high watermark of the memory pool of the engine
    (used = allocated, reserved = taken from the device) in the best of two runs; the result stays
    in GPU memory one partition at a time (each partition frees its memory when it ends)."""
    out = H.Csv("partitions.csv", ["dataset", "P", "facts", "largest_part", "peak_used_gib", "peak_reserved_gib",
                                   "time_s"])
    for n in PARTITION_SIZES:
        name = f"uniform-{n}"
        for P in [1, 2, 4, 8, 16]:
            o = H.engine("fusedhash_tcsg", prog="tc", input=H.graph_bin(name), plan=P)
            d = o.data
            print(f"{name:14s} P={P:<2d} {d.get('result', ''):>14} largest {d.get('largest_part', ''):>14} "
                  f"peak {d.get('peak_used_gib', 0):6.2f} GiB  {o.cell()}", flush=True)
            out.row(dataset=name, P=P, facts=d.get("result", ""), largest_part=d.get("largest_part", ""),
                    peak_used_gib=f"{d['peak_used_gib']:.2f}" if o.ok else "",
                    peak_reserved_gib=f"{d['peak_reserved_gib']:.2f}" if o.ok else "", time_s=o.cell())


def chain():
    out = H.Csv("chain.csv", ["dataset", "result", "iters", "noloop_s", "noloop_ms_per_iter", "loop_s",
                              "loop_ms_per_iter", "fvlog_s", "souffle1_s", "agree"])
    for name in ["chain-1000", "chain-3000", "chain-10000"]:
        off = H.engine("fusedhash_tcsg", prog="tc", input=H.graph_bin(name), loop=0)
        on = H.engine("fusedhash_tcsg", prog="tc", input=H.graph_bin(name))
        fv = H.fvlog("TC_TEST", H.graph_facts(name))
        s1 = H.souffle("tc_fvlog", H.graph_facts_dir(name), threads=1)
        it = (on.data if on.ok else off.data).get("iters", 0)
        ms = lambda o: f"{o.data['time'] / it * 1e3:.4f}" if o.ok and it else ""
        ok = H.agree(off, on, fv, s1)
        print(f"{name:12s} no loop {off.cell()} ({ms(off)} ms/it)  loop {on.cell()} ({ms(on)} ms/it)  "
              f"fvlog {fv.cell()}  souffle-1 {s1.cell()} {'' if ok else 'MISMATCH'}", flush=True)
        out.row(dataset=name, result=(on.data if on.ok else off.data).get("result", ""), iters=it,
                noloop_s=off.cell(), noloop_ms_per_iter=ms(off), loop_s=on.cell(), loop_ms_per_iter=ms(on),
                fvlog_s=fv.cell(), souffle1_s=s1.cell(), agree=int(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["table5", "partitions", "chain"])
    a = ap.parse_args()
    for exp in a.only:
        {"table5": table5, "partitions": partitions, "chain": chain}[exp]()


if __name__ == "__main__":
    main()
