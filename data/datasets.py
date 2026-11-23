"""Inputs of all the experiments: synthetic generators and loaders of the datasets on disk.

Generators (deterministic, dense ids 0..n-1, edges as (m x 2) int64 arrays):
  grid(k), tree(h), gnp(n)          BigDatalog families (Shkapsky et al., SIGMOD 2016)
  rmat(scale), weights(E)           Graph500 R-MAT graphs and integer weights for SSSP
  ownership(n)                      synthetic shareholdings for company control
  uniform_graph(n, m), chain(n), random_graph, chain_graph, random_forest   other inputs

Loaders (paths relative to data/):
  real(name)        npy/<name>.npy           FVlog graphs and road networks (data/download.py)
  synthetic(name)   npy/syn/<name>.npy       Grid<k>, Tree<h>, G<n>K (data/download.py --synthetic)
  aggregate(name)   inputs of CC/SSSP/company control, generated (data/prepare.py writes data/agg)
  cspa(name)        assign/dereference facts in cspa/<name>/cspa-<name>/ (data/download.py)
  csda(name)        Edge/NullEdge facts in csda/<name>/csda-<name>/ (data/download.py)
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "npy")
Q = 0.6             # fraction of internal nodes per level in the trees (see tree)


# ------------------------------------------------------------ BigDatalog families and other generators

def grid(k):
    n = k + 1
    idx = np.arange(n * n, dtype=np.int64).reshape(n, n)
    right = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1)
    down = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], 1)
    return np.concatenate([right, down])


def tree(h, seed=0, q=Q):
    rng = np.random.default_rng(seed)
    edges, frontier, next_id = [], np.array([0], dtype=np.int64), 1
    for _ in range(h):
        # exact fraction of internal nodes per level (low variance across seeds and heights)
        n_int = max(1, int(round(q * len(frontier))))
        parents = np.sort(rng.choice(frontier, size=n_int, replace=False))
        kids = rng.integers(2, 7, size=len(parents))           # 2..6 children
        child = np.arange(next_id, next_id + kids.sum(), dtype=np.int64)
        next_id += kids.sum()
        edges.append(np.stack([np.repeat(parents, kids), child], 1))
        frontier = child
    return np.concatenate(edges)


def gnp(n, p=0.001, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.binomial(n * (n - 1), p)                           # number of edges
    keys = np.unique(rng.integers(0, n * (n - 1), size=int(m * 1.02) + 16))
    keys = rng.permutation(keys)[:m]
    u = keys // (n - 1); v = keys % (n - 1)
    v = v + (v >= u)                                           # skip the self-loop
    return np.stack([u, v], 1).astype(np.int64)


def rmat(scale, edge_factor=10, seed=0, a=0.57, b=0.19, c=0.19):
    """R-MAT with the Graph500 parameters: 2^scale vertices, edge_factor * 2^scale edges
    (duplicates and self-loops removed), permuted ids."""
    rng = np.random.default_rng(seed)
    n, m = 1 << scale, edge_factor << scale
    u = np.zeros(m, dtype=np.int64); v = np.zeros(m, dtype=np.int64)
    for bit in range(scale):
        r = rng.random(m)
        down = r >= a + b                                      # quadrants c, d: bit of u
        right = ((r >= a) & (r < a + b)) | (r >= a + b + c)    # quadrants b, d: bit of v
        u |= down.astype(np.int64) << bit
        v |= right.astype(np.int64) << bit
    perm = rng.permutation(n)
    e = np.unique(np.stack([perm[u], perm[v]], 1), axis=0)
    return e[e[:, 0] != e[:, 1]]


def weights(E, seed=0, wmax=100):
    """Uniform integer weights in [1, wmax] for SSSP (deterministic given the graph)."""
    return np.random.default_rng(seed).integers(1, wmax + 1, size=len(E)).astype(np.int32)


def ownership(n, seed=0):
    """Synthetic ownership graph: rows (owner, owned company, share in millionths).
    Each company has 1 + Zipf owners (at most 20), chosen with a preference for low ids
    (holdings: id = n * u^2); the shares are Dirichlet(0.7) over a total uniform in
    [0.6, 1], so there is often a majority owner and sometimes joint control.
    There are cycles (including cross-shareholdings)."""
    rng = np.random.default_rng(seed)
    k = np.minimum(rng.zipf(2.0, size=n), 20)
    z = np.repeat(np.arange(n, dtype=np.int64), k)
    y = (n * rng.random(len(z)) ** 2).astype(np.int64)
    keep = y != z
    y, z = y[keep], z[keep]
    g = rng.gamma(0.7, size=len(z))
    tot = np.bincount(z, weights=g, minlength=n)
    share = g / tot[z] * rng.uniform(0.6, 1.0, size=n)[z]
    w = np.floor(share * 1e6).astype(np.int64)
    key = y * n + z                                            # sum repeated owners
    uk, inv = np.unique(key, return_inverse=True)
    ws = np.bincount(inv, weights=w).astype(np.int64)
    out = np.stack([uk // n, uk % n, ws], 1)
    return out[out[:, 2] > 0]


def make_synthetic(name):
    """Grid<k>, Tree<h>, G<n> or G<n>K."""
    fam, arg = name.rstrip("0123456789K"), name[len(name.rstrip("0123456789K")):]
    if fam == "Grid":
        return grid(int(arg))
    if fam == "Tree":
        return tree(int(arg))
    if fam == "G":
        return gnp(int(arg[:-1]) * 1000 if arg.endswith("K") else int(arg))
    raise ValueError(name)


def uniform_graph(n, m, seed=0):
    """m distinct edges without self-loops, uniform over n nodes."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, n, size=int(m * 1.3)); y = rng.integers(0, n, size=int(m * 1.3))
    e = np.unique(np.stack([x, y], 1)[x != y], axis=0)
    return e[rng.permutation(len(e))[:m]]


def chain(n):
    x = np.arange(n - 1); return np.stack([x, x + 1], 1)


def random_graph(n: int, m: int, seed: int = 0) -> np.ndarray:
    """E(x,y): m uniform edges over n nodes, without self-loops or duplicates."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, n, size=int(m * 1.2), dtype=np.int64)
    y = rng.integers(0, n, size=int(m * 1.2), dtype=np.int64)
    e = np.unique(np.stack([x, y], 1)[x != y], axis=0)
    return e[:m]


def chain_graph(n: int) -> np.ndarray:
    """Chain 0->1->...->n-1: quadratic TC, n-1 iterations (worst case for depth)."""
    x = np.arange(n - 1, dtype=np.int64)
    return np.stack([x, x + 1], 1)


def random_forest(levels: int, branching: int, roots: int, seed: int = 0) -> np.ndarray:
    """Par(child, parent): `roots` trees; each level has branching times the nodes of the
    previous one, and each node picks a random parent in the previous level. Nodes at the same
    level but in different trees are NOT of the same generation."""
    rng = np.random.default_rng(seed)
    pairs = []
    next_id = roots
    prev = np.arange(roots, dtype=np.int64)
    for _ in range(levels):
        k = len(prev) * branching
        cur = np.arange(next_id, next_id + k, dtype=np.int64)
        next_id += k
        parents = prev[rng.integers(0, len(prev), size=k)]
        pairs.append(np.stack([cur, parents], 1))
        prev = cur
    return np.concatenate(pairs)


# ------------------------------------------------------------ loaders

def real(name):
    path = os.path.join(DATA, f"{name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}: run python3 data/download.py first")
    return np.load(path).astype(np.int64)


def synthetic(name):
    path = os.path.join(DATA, "syn", f"{name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}: run python3 data/download.py --synthetic first")
    return np.load(path).astype(np.int64)


def aggregate(name):
    """Inputs of the aggregates: {"E", "W", "source"} for CC/SSSP, {"Own"} for company control.
      RMAT-<s>      R-MAT, 2^s vertices, 10 * 2^s edges; source = vertex of maximum degree
      Grid<k>       BigDatalog grid; source 0
      Sym-<g>       real graph npy/<g>.npy made undirected; source = vertex of maximum degree
      SW-<p>        Grid_2000 plus a fraction p of random edges (small world); source 0
      Own-<n>[M]    synthetic shareholdings over n (millions of) companies
      PSC           real shareholdings between UK companies (data/psc.py)"""
    cache = os.path.join(ROOT, "agg_cache", f"{name}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    if name == "PSC":                                  # UK register (data/psc.py)
        return {"Own": np.load(os.path.join(DATA, "psc_own.npy"))}
    fam, arg = name.split("-", 1) if "-" in name else ("Grid", name[4:])
    if fam == "Own":
        d = {"Own": ownership(int(arg[:-1]) << 20 if arg.endswith("M") else int(arg))}
    else:
        if fam == "RMAT":
            E = rmat(int(arg))
        elif fam == "Sym":
            E0 = real(arg)
            E = np.unique(np.concatenate([E0, E0[:, ::-1]]), axis=0)
            E = E[E[:, 0] != E[:, 1]]
        elif fam == "SW":
            G = grid(2000)
            n = int(G.max()) + 1
            rng = np.random.default_rng(1)
            R = rng.integers(0, n, size=(int(float(arg) * len(G)), 2))
            E = np.unique(np.concatenate([G, R[R[:, 0] != R[:, 1]]]), axis=0)
        else:
            E = grid(int(arg))
        src = int(np.argmax(np.bincount(E[:, 0]))) if fam in ("RMAT", "Sym") else 0
        d = {"E": E.astype(np.int32), "W": weights(E), "source": np.array(src)}
    return d                                           # prepare.py writes data/agg/<name>.bin


def cspa(name, root=None):
    """(assign, dereference) facts of CSPA (httpd, linux, postgresql)."""
    d = os.path.join(root or os.path.join(ROOT, "cspa"), name, f"cspa-{name}")
    a = np.loadtxt(os.path.join(d, "Assign.csv"), delimiter=",", dtype=np.int64)
    r = np.loadtxt(os.path.join(d, "Dereference.csv"), delimiter=",", dtype=np.int64)
    return a, r


def csda(name, root=None):
    """(edge, null_edge) facts of CSDA (httpd, linux, postgresql)."""
    d = os.path.join(root or os.path.join(ROOT, "csda"), name, f"csda-{name}")
    e = np.loadtxt(os.path.join(d, "Edge.csv"), delimiter=",", dtype=np.int64, ndmin=2)
    n = np.loadtxt(os.path.join(d, "NullEdge.csv"), delimiter=",", dtype=np.int64, ndmin=2)
    return e, n
