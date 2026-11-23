#!/usr/bin/env python3
"""Peak GPU memory of FusedHash against the number of partitions of the homomorphic decomposition, on
the largest TC inputs (as the partitions of bench/secondary.py on the uniform graphs): G80K, G60K and
G40K (6.4, 3.6 and 1.6 billion facts), com-dblp and fe_ocean. One row per (graph, P), with P = 1, 2, 4,
8, 16: the result, the facts of the largest partition, the high watermark of the memory pool (used =
allocated, reserved = taken from the device) and the time, best of two runs; OOM when the plan does not
fit the 80 GB of the A100. The result stays in GPU memory one partition at a time.

  python3 bench/partitions_large.py [--datasets ...] [--resume]   -> results/partitions_large.csv
"""
import argparse
import harness as H

NAMES = ["G80K", "G60K", "G40K", "com-dblp", "fe_ocean"]
FIELDS = ["dataset", "P", "facts", "largest_part", "peak_used_gib", "peak_reserved_gib", "time_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = H.Csv("partitions_large.csv", FIELDS, a.resume)
    for name in NAMES:
        for P in (1, 2, 4, 8, 16):
            if (a.datasets and name not in a.datasets) or (name, str(P)) in out.done:
                continue
            o = H.engine("fusedhash_tcsg", prog="tc", input=H.graph_bin(name), plan=P)
            d = o.data
            print(f"{name:10s} P={P:<2d} {d.get('result', ''):>14} largest {d.get('largest_part', ''):>14} "
                  f"peak {d['peak_used_gib'] if o.ok else o.kind:>6} GiB  {o.cell()}", flush=True)
            out.row(dataset=name, P=P, facts=d.get("result", ""), largest_part=d.get("largest_part", ""),
                    peak_used_gib=f"{d['peak_used_gib']:.2f}" if o.ok else ("OOM" if o.kind == "oom" else o.kind),
                    peak_reserved_gib=f"{d['peak_reserved_gib']:.2f}" if o.ok else "", time_s=o.cell())


if __name__ == "__main__":
    main()
