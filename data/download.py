#!/usr/bin/env python3
"""Downloads and converts the inputs that data/prepare.py does not generate.

  python3 data/download.py                      real graphs, synthetic families, CSPA and CSDA
  python3 data/download.py --only real cspa     some of them (real, synthetic, cspa, csda)

Outputs (under data/):
  raw/                       the downloaded archives (kept, so that the conversion can be rerun)
  npy/<name>.npy             real graphs: int32 rows (x, y) with dense ids 0..n-1
  npy/syn/<name>.npy         synthetic BigDatalog families (generated, fixed seeds)
  cspa/<name>/cspa-<name>/   Assign.csv and Dereference.csv of CSPA
  csda/<name>/csda-<name>/   Edge.csv and NullEdge.csv of CSDA

Real graphs: the TC and SG datasets of FVlog and GDlog, from SuiteSparse and SNAP, and two large road
networks used for CC and SSSP. Every stored entry of a file is a directed edge Edge(i,j): for the
symmetric SuiteSparse matrices the .mtx file holds only the lower triangle, so |Edge| = nnz/2, which
gives the sizes reported by FVlog (vsp_finan 552,020, fe_ocean 409,593, ...); SNAP files are read row
by row. Ids are remapped to 0..n-1 in increasing order.

CSPA: the httpd, linux and postgresql inputs (derived from Graspan) as distributed with the FlowLog
benchmark (https://pages.cs.wisc.edu/~m0riarty/dataset/csv/, also on Hugging Face as
NemoYuu/flowlog_benchmark).

PSC (company control on real data) is optional and is prepared by data/psc.py.
"""
import argparse, gzip, os, sys, tarfile, urllib.request, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
NPY = os.path.join(HERE, "npy")
sys.path.insert(0, HERE)
import datasets  # noqa: E402
from prepare import SYN  # noqa: E402

SS = "https://suitesparse-collection-website.herokuapp.com/MM/"
REAL = {   # name -> (url, expected number of edges or None)
    "vsp_finan": (SS + "DIMACS10/vsp_finan512_scagr7-2c_rlfddd.tar.gz", 552_020),
    "fe_ocean": (SS + "DIMACS10/fe_ocean.tar.gz", 409_593),
    "fe_body": (SS + "DIMACS10/fe_body.tar.gz", 163_734),
    "fe_sphere": (SS + "DIMACS10/fe_sphere.tar.gz", 49_152),
    "usroads": (SS + "Gleich/usroads.tar.gz", 165_435),
    "Gnutella31": ("https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz", 147_892),
    "CA-HepTh": ("https://snap.stanford.edu/data/ca-HepTh.txt.gz", 51_971),
    "com-dblp": ("https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz", 1_049_866),
    # large road networks (DIMACS10), used for CC and SSSP: deep recursion
    "road_usa": (SS + "DIMACS10/road_usa.tar.gz", None),
    "europe_osm": (SS + "DIMACS10/europe_osm.tar.gz", None),
}
CSPA_URL = "https://pages.cs.wisc.edu/~m0riarty/dataset/csv/cspa-{}.zip"
# CSDA: the same Graspan inputs in the FlowLog benchmark on Hugging Face (dataset NemoYuu/flowlog_benchmark)
CSDA_URL = "https://huggingface.co/datasets/NemoYuu/flowlog_benchmark/resolve/main/dataset/csv/csda-{}.zip"


def fetch(url, sub=""):
    d = os.path.join(RAW, sub)
    path = os.path.join(d, os.path.basename(url))
    if not os.path.exists(path):
        os.makedirs(d, exist_ok=True)
        print(f"  downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def read_mtx(path):
    with tarfile.open(path) as tf:
        m = next(x for x in tf.getmembers() if x.name.endswith(".mtx") and "/" in x.name
                 and os.path.basename(x.name)[:-4] == os.path.basename(os.path.dirname(x.name)))
        raw = tf.extractfile(m).read()
    header = raw[:raw.index(b"\n")].decode().lower()
    ncol = 2 if "pattern" in header else 3                  # pattern matrices: (i, j) only
    # skip the comments and the line of the dimensions, then numpy's C parser (loadtxt is too slow
    # for tens of millions of rows)
    pos = 0
    while raw.startswith(b"%", pos):
        pos = raw.index(b"\n", pos) + 1
    pos = raw.index(b"\n", pos) + 1
    vals = np.fromstring(raw[pos:].decode(), dtype=np.float64 if ncol == 3 else np.int64, sep=" ")
    return vals.reshape(-1, ncol)[:, :2].astype(np.int64) - 1


def read_snap(path):
    with gzip.open(path, "rt") as f:
        return np.loadtxt(f, dtype=np.int64, comments="#", usecols=(0, 1))


def real():
    os.makedirs(NPY, exist_ok=True)
    for name, (url, expected) in REAL.items():
        out = os.path.join(NPY, f"{name}.npy")
        if os.path.exists(out):
            continue
        path = fetch(url)
        E = read_mtx(path) if path.endswith(".tar.gz") else read_snap(path)
        ids, E = np.unique(E, return_inverse=True)          # dense ids 0..n-1
        E = E.reshape(-1, 2).astype(np.int32)
        if expected is not None and len(E) != expected:
            sys.exit(f"{name}: {len(E):,} edges, expected {expected:,}")
        np.save(out, E)
        print(f"  {name:11s} {len(E):>12,} edges {len(ids):>10,} vertices")


def synthetic():
    d = os.path.join(NPY, "syn")
    os.makedirs(d, exist_ok=True)
    for name in SYN:
        out = os.path.join(d, f"{name}.npy")
        if not os.path.exists(out):
            E = datasets.make_synthetic(name).astype(np.int32)
            np.save(out, E)
            print(f"  {name:8s} {int(E.max()) + 1:>12,} vertices {len(E):>12,} edges")


def cspa():
    for name in ("httpd", "linux", "postgresql"):
        d = os.path.join(HERE, "cspa", name)
        if os.path.exists(os.path.join(d, f"cspa-{name}", "Dereference.csv")):
            continue
        path = fetch(CSPA_URL.format(name), "cspa")
        with zipfile.ZipFile(path) as z:
            z.extractall(d)
        asg, der = datasets.cspa(name)
        print(f"  {name:11s} assign {len(asg):>10,}  dereference {len(der):>10,}")


def csda():
    for name in ("httpd", "linux", "postgresql"):
        d = os.path.join(HERE, "csda", name)
        if os.path.exists(os.path.join(d, f"csda-{name}", "NullEdge.csv")):
            continue
        path = fetch(CSDA_URL.format(name), "csda")
        with zipfile.ZipFile(path) as z:
            z.extractall(d)
        e, ne = datasets.csda(name)
        print(f"  {name:11s} edge {len(e):>12,}  null edge {len(ne):>10,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["real", "synthetic", "cspa", "csda"])
    a = ap.parse_args()
    for step in a.only:
        print(step)
        {"real": real, "synthetic": synthetic, "cspa": cspa, "csda": csda}[step]()


if __name__ == "__main__":
    main()
