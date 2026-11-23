"""CPU oracles (Python sets, numpy) for the correctness tests."""
from __future__ import annotations
import heapq
from collections import defaultdict
import numpy as np


def fingerprint(pairs) -> int:
    """The fingerprint of a set of facts (x, y), as engines/common.cuh computes it with
    --fingerprint=1: the sum mod 2^64 of splitmix64(x << 32 | y)."""
    a = np.array(sorted(pairs), dtype=np.uint64).reshape(-1, 2)
    with np.errstate(over="ignore"):
        z = (a[:, 0] << np.uint64(32)) | a[:, 1]
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return int(z.sum(dtype=np.uint64))


def tc_reference(E: np.ndarray) -> set:
    from collections import defaultdict
    adj = defaultdict(list)
    for x, y in E.tolist():
        adj[x].append(y)
    T = set()
    for s in list(adj):
        seen = set(); stack = list(adj[s])
        while stack:
            v = stack.pop()
            if v in seen: continue
            seen.add(v); stack.extend(adj.get(v, ()))
        T.update((s, v) for v in seen)
    return T


def csda_reference(E: np.ndarray, N: np.ndarray) -> set:
    """CSDA: NullNode(x,y) :- NullEdge(x,y).  NullNode(x,y) :- NullNode(x,w), Edge(w,y)."""
    from collections import defaultdict
    adj = defaultdict(list)
    for w, y in E.tolist():
        adj[w].append(y)
    out = set()
    starts = defaultdict(list)
    for x, w in N.tolist():
        starts[x].append(w)
    for x, ws in starts.items():
        seen = set(); stack = list(ws)
        while stack:
            v = stack.pop()
            if v in seen: continue
            seen.add(v); stack.extend(adj.get(v, ()))
        out.update((x, v) for v in seen)
    return out


def sg_reference(Par: np.ndarray, neq: bool = False) -> set:
    """SG over Par(child, parent); x != y in the base rule and, if neq, also in the recursive one."""
    from collections import defaultdict
    children = defaultdict(list)     # parent -> [child]
    parent = {}
    for c, p in Par.tolist():
        children[p].append(c); parent[c] = p
    SG = set()
    for p, cs in children.items():
        for a in cs:
            for b in cs:
                if a != b: SG.add((a, b))
    delta = set(SG)
    while delta:
        new = set()
        for a, b in delta:
            for x in children.get(a, ()):
                for y in children.get(b, ()):
                    if (x, y) not in SG and (not neq or x != y): new.add((x, y))
        SG |= new; delta = new
    return SG


def as_set(res) -> set:
    x, y = res
    return set(zip(x.tolist(), y.tolist()))



INF = 0xFFFFFFFF
THRESHOLD = 500_000


def _summary(lab):
    """(vertices reached, sum of their labels, fingerprint of the facts (vertex, label))."""
    r = lab != INF
    v = np.nonzero(r)[0]
    return int(r.sum()), int(lab[r].astype(np.int64).sum()), fingerprint(zip(v.tolist(), lab[r].tolist()))


def cc_ref(E):
    K = int(E.max()) + 1
    lab = np.full(K, INF, dtype=np.int64)
    src = np.unique(E[:, 0]); lab[src] = src
    while True:
        new = lab.copy()
        np.minimum.at(new, E[:, 1], lab[E[:, 0]])
        if (new == lab).all():
            return _summary(lab)
        lab = new


def sssp_ref(E, W, source=0):
    K = int(E.max()) + 1
    adj = defaultdict(list)
    for (u, v), w in zip(E.tolist(), W.tolist()):
        adj[u].append((v, w))
    dist = np.full(K, INF, dtype=np.int64); dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w; heapq.heappush(pq, (d + w, v))
    return _summary(dist)


def control_ref(Own):
    """(|Control|, fingerprint of Control), with Control(x,x) for every company 0..K-1."""
    K = int(Own[:, :2].max()) + 1
    out = defaultdict(list)
    for y, z, w in Own.tolist():
        out[y].append((z, w))
    facts = []
    for x in range(K):
        ctrl, sums, stack = {x}, defaultdict(int), [x]
        while stack:
            y = stack.pop()
            for z, w in out[y]:
                if z == x:
                    continue
                old = sums[z]; sums[z] = old + w
                if old <= THRESHOLD < old + w:
                    ctrl.add(z); stack.append(z)
        facts.extend((x, z) for z in ctrl)
    return len(facts), fingerprint(facts)




def cspa_ref(assign, deref):
    """{VF, VA, MA: |relation|, fp_VF, fp_VA, fp_MA: its fingerprint}, with the CSPA rules of
    GDlog (souffle/cspa.dl)."""
    A = set(map(tuple, assign.tolist()))
    D = defaultdict(set)
    for y, x in deref.tolist():
        D[y].add(x)
    A2 = defaultdict(set)
    for x, z in A:
        A2[z].add(x)
    ids = set(assign.ravel().tolist())
    VF = A | {(i, i) for i in ids}
    MA = {(i, i) for i in ids}
    VA = set()
    while True:
        n = (len(VF), len(VA), len(MA))
        vf1, ma1 = defaultdict(set), defaultdict(set)
        for x, y in VF:
            vf1[x].add(y)
        for z, w in MA:
            ma1[z].add(w)
        VF |= {(x, y) for (x, z) in VF for y in vf1.get(z, ())} | \
              {(x, y) for (z, y) in MA for x in A2.get(z, ())}
        VA |= {(x, y) for z in list(vf1) for x in vf1[z] for y in vf1[z]} | \
              {(x, y) for z in list(vf1) for x in vf1[z] for w in ma1.get(z, ()) for y in vf1.get(w, ())}
        MA |= {(x, w) for (y, z) in VA for x in D.get(y, ()) for w in D.get(z, ())}
        if (len(VF), len(VA), len(MA)) == n:
            return {"VF": len(VF), "VA": len(VA), "MA": len(MA),
                    "fp_VF": fingerprint(VF), "fp_VA": fingerprint(VA), "fp_MA": fingerprint(MA)}
