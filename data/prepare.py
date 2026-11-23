#!/usr/bin/env python3
"""Writes the inputs of every experiment in the formats of the three systems.

  python3 data/prepare.py [--only real,syn,agg,table5,cspa,csda,tests] [--force]
  python3 data/prepare.py --only large      the larger inputs of bench/large.py (A100 only)

Outputs (under data/):
  bin/<name>.bin           int32 rows (x, y): input of our CUDA engines (TC, SG)
  facts/<name>/edge.facts  tab-separated rows (x, y): input of Souffle and of FVlog
  agg/<name>.bin           int32 rows (u, v, w) for CC/SSSP, (y, z, w) for company control
  agg/<name>.json          {"source": s, "rows": n}; the aggregate inputs are generated here
                           (R-MAT, grids, small-world, Own-n) with the fixed seeds of
                           data/datasets.py, so they are always the same
  cspa/<name>/...          Assign.csv / Dereference.csv (already in the bundle, read by Souffle)
  cspa_facts/<name>/       assign.facts / dereference.facts, tab-separated (read by FVlog), and
                           assign.bin / dereference.bin (read by fusedhash_cspa)
  csda_bin/<name>/         edge.bin / nulledge.bin (read by our engines, TC with --base)
  csda_facts/<name>/       edge.facts / null_edge.facts, tab-separated (read by FVlog and GDlog);
                           Souffle reads the CSV files in csda/<name>/csda-<name>/

Real and synthetic graphs come from data/npy, written by data/download.py (real graphs, CSPA, PSC)
and by the generators of datasets.py (synthetic families, with fixed seeds); the aggregate inputs are
generated from the generators in datasets.py.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets  # noqa: E402

REAL = ["fe_sphere", "fe_body", "fe_ocean", "Gnutella31", "usroads", "vsp_finan", "com-dblp", "CA-HepTh"]
SYN = ["Grid50", "Grid100", "Grid150", "Grid200", "Grid250", "Tree6", "Tree7", "Tree8", "Tree9", "Tree10",
       "Tree11", "Tree13", "Tree15", "Tree17", "G5K", "G10K", "G20K", "G40K"]
AGG = ["RMAT-20", "RMAT-22", "RMAT-24", "Grid250", "Grid500", "Grid1000", "Grid2000",
       "Sym-usroads", "Sym-fe_ocean", "Sym-road_usa", "Sym-europe_osm",
       "SW-1e-4", "SW-1e-3", "SW-1e-2", "SW-1e-1",
       "Own-1M", "Own-2M", "Own-4M", "Own-8M", "Own-16M", "Own-32M", "Own-64M", "PSC"]
# Table 5 of the paper: uniform random graphs (n, 4n) for TC, random forests for SG
TABLE5_TC = [(3000, 12000), (5000, 20000), (12000, 48000), (20000, 80000), (30000, 120000)]
TABLE5_SG = [(6, 3, 20), (10, 2, 16), (12, 2, 8)]
PARTITION_ONLY = [(40000, 160000), (46000, 184000)]    # partitioned plan and peak memory only
# bench/large.py: the largest inputs, with the same generators and seeds;
# G60K and G80K are generated here (gnp, as G5K..G40K), not read from data/npy
LARGE_SYN = ["G60K", "G80K"]
LARGE_AGG = ["RMAT-25", "RMAT-26", "Grid4000"]


def write_bin(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.ascontiguousarray(rows, dtype=np.uint32).tofile(path)


def write_facts(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, np.asarray(rows, dtype=np.int64), fmt="%d", delimiter="\t")


def graph(name, E, force):
    b = os.path.join(HERE, "bin", f"{name}.bin")
    f = os.path.join(HERE, "facts", name, "edge.facts")
    if force or not os.path.exists(b):
        write_bin(b, E)
    if force or not os.path.exists(f):
        write_facts(f, E)
    print(f"  {name:14s} {len(E):>12,} edges, {int(E.max()) + 1:>10,} vertices")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="real,syn,agg,table5,cspa,csda,tests")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    parts = set(a.only.split(","))
    if "real" in parts:
        print("real graphs")
        for n in REAL:
            graph(n, datasets.real(n), a.force)
    if "syn" in parts:
        print("synthetic graphs (BigDatalog families)")
        for n in SYN:
            graph(n, datasets.synthetic(n), a.force)
    if "table5" in parts:
        print("uniform random graphs and random forests")
        for n, m in TABLE5_TC:
            graph(f"uniform-{n}", datasets.uniform_graph(n, m), a.force)
        for lv, br, rt in TABLE5_SG:
            Par = datasets.random_forest(lv, br, rt)        # rows (child, parent)
            graph(f"forest-{lv}-{br}-{rt}", Par[:, ::-1].copy(), a.force)   # edge (parent, child)
        for n, m in PARTITION_ONLY:
            graph(f"uniform-{n}", datasets.uniform_graph(n, m), a.force)
        for n in (1000, 3000, 10000):
            graph(f"chain-{n}", datasets.chain(n), a.force)
    if "large" in parts:
        print("larger inputs of bench/large.py (generated)")
        for n in LARGE_SYN:
            graph(n, datasets.make_synthetic(n), a.force)
    if "agg" in parts or "large" in parts:
        print("aggregate inputs (generated)")
        for n in (AGG if "agg" in parts else []) + (LARGE_AGG if "large" in parts else []):
            b = os.path.join(HERE, "agg", f"{n}.bin")
            j = os.path.join(HERE, "agg", f"{n}.json")
            if not a.force and os.path.exists(b) and os.path.exists(j):
                continue
            if n == "PSC" and not os.path.exists(os.path.join(HERE, "npy", "psc_own.npy")):
                print("  PSC            skipped (optional): run data/psc.py first")
                continue
            d = datasets.aggregate(n)
            if "Own" in d:
                rows = d["Own"]
                meta = {"rows": int(len(rows)), "source": 0}
            else:
                rows = np.concatenate([d["E"].astype(np.int64), d["W"].astype(np.int64)[:, None]], 1)
                meta = {"rows": int(len(rows)), "source": int(d["source"])}
            write_bin(b, rows)
            json.dump(meta, open(j, "w"))
            print(f"  {n:14s} {len(rows):>12,} rows")
    if "cspa" in parts:
        print("CSPA (Graspan inputs of FlowLog): tab-separated facts for FVlog")
        for n in ("httpd", "linux", "postgresql"):
            asg, der = datasets.cspa(n)
            d = os.path.join(HERE, "cspa_facts", n)
            if a.force or not os.path.exists(os.path.join(d, "dereference.facts")):
                write_facts(os.path.join(d, "assign.facts"), asg)
                write_facts(os.path.join(d, "dereference.facts"), der)
                write_bin(os.path.join(d, "assign.bin"), asg)
                write_bin(os.path.join(d, "dereference.bin"), der)
            print(f"  {n:14s} assign {len(asg):>10,}  dereference {len(der):>10,}")
    if "csda" in parts:
        print("CSDA (Graspan inputs of FlowLog): binary files for our engines, facts for FVlog and GDlog;"
              " Souffle reads the CSV files")
        for n in ("httpd", "linux", "postgresql"):
            d = os.path.join(HERE, "csda_bin", n)
            f = os.path.join(HERE, "csda_facts", n)
            if a.force or not os.path.exists(os.path.join(d, "nulledge.bin")) \
                    or not os.path.exists(os.path.join(f, "null_edge.facts")):
                e, ne = datasets.csda(n)
                write_bin(os.path.join(d, "edge.bin"), e)
                write_bin(os.path.join(d, "nulledge.bin"), ne)
                write_facts(os.path.join(f, "edge.facts"), e)
                write_facts(os.path.join(f, "null_edge.facts"), ne)
            rows = lambda f: os.path.getsize(os.path.join(d, f)) // 8
            print(f"  {n:14s} edge {rows('edge.bin'):>12,}  null edge {rows('nulledge.bin'):>10,}")
    if "tests" in parts:
        print("small inputs of the correctness tests")
        for n, E in {"t-uniform": datasets.uniform_graph(300, 900, 1), "t-grid": datasets.grid(12),
                     "t-tree": datasets.tree(6), "t-chain": datasets.chain(60),
                     "t-cycle": np.stack([np.arange(50), (np.arange(50) + 1) % 50], 1)}.items():
            graph(n, E, a.force)
            graph(n + "-wide", E + 70000, a.force)           # more than 65,535 constants: 64-bit keys
        for n, (lv, br, rt) in {"t-forest": (4, 2, 3)}.items():
            graph(n, datasets.random_forest(lv, br, rt)[:, ::-1].copy(), a.force)


if __name__ == "__main__":
    main()
