#!/usr/bin/env python3
"""Correctness tests of the CUDA engines against CPU oracles, on small inputs with real recursion:
TC and SG with every variant of FusedHash (hash table, bitmap, forced bitmap; atomic and
warp-aggregated append; both work distributions), the looping kernel and the word-level join
(forced with minimal thresholds, switched off, and alternating with the tuple-level join in the
same computation), 32-bit and 64-bit keys (ids shifted beyond 65,535), partitioned plans, SortMerge, and the aggregate programs (CC, SSSP, company control)
with both engines, CSPA, and CSDA (TC seeded by another relation, --base). Each result of our engines is
checked as its number of facts and the
fingerprint of the set of its facts (--fingerprint=1, engines/common.cuh; reference.fingerprint), so
that a missing, extra or wrong fact is caught. If FVlog and Souffle are built, their counts are
checked too, and those of GDlog.

  python3 tests/run_tests.py            (exit code 1 if any test fails)
"""
import itertools, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, HERE)
import datasets as D            # noqa: E402
import harness as H             # noqa: E402
import reference as R           # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" (expected {want})"), flush=True)
    if not ok:
        failures.append(name)


def graphs():
    g = {"uniform": D.uniform_graph(300, 900, 1), "grid": D.grid(12), "tree": D.tree(6), "chain": D.chain(60),
         "cycle": np.stack([np.arange(50), (np.arange(50) + 1) % 50], 1),
         "forest": D.random_forest(4, 2, 3)[:, ::-1].copy()}
    return g


def write(tmp, name, rows, cols):
    p = os.path.join(tmp, f"{name}.bin")
    np.ascontiguousarray(rows, dtype=np.uint32).reshape(-1, cols).tofile(p)
    return p


def run(exe, **kw):
    """(facts, fingerprint of the facts) of a run of our engines, or the error."""
    o = H.engine(exe, reps=1, fingerprint=1, **kw)
    return (o.data.get("result"), o.data.get("fingerprint")) if o.ok else f"{o.kind}: {o.message}"


def counted(facts):
    """(facts, fingerprint) of a set of facts of an oracle."""
    return len(facts), R.fingerprint(facts)


def test_tcsg(tmp):
    print("TC / SG")
    for (gname, E), wide in itertools.product(graphs().items(), (0, 70000)):
        G = E + wide
        path = write(tmp, f"{gname}-{wide}", G, 2)
        tc_ref = counted(R.tc_reference(G))
        sg_ref = {neq: counted(R.sg_reference(G[:, ::-1].copy(), neq)) for neq in (0, 1)}
        tag = f"{gname}{' (64-bit keys)' if wide else ''}"
        for bitmap, append, split in itertools.product(("off", "auto", "force"), ("atomic", "warp"),
                                                       ("delta", "output")):
            opts = dict(bitmap=bitmap, append=append, split=split)
            check(f"fusedhash tc {tag} {opts}", run("fusedhash_tcsg", prog="tc", input=path, **opts), tc_ref)
            for neq in (0, 1):
                check(f"fusedhash sg neq={neq} {tag} {opts}",
                      run("fusedhash_tcsg", prog="sg", input=path, neq=neq, **opts), sg_ref[neq])
        check(f"fusedhash tc {tag} plan 3", run("fusedhash_tcsg", prog="tc", input=path, plan=3), tc_ref)
        check(f"fusedhash tc {tag} chunk 256", run("fusedhash_tcsg", prog="tc", input=path, chunk=256), tc_ref)
        check(f"fusedhash sg {tag} chunk 256", run("fusedhash_tcsg", prog="sg", input=path, chunk=256), sg_ref[0])
        check(f"sortmerge tc {tag}", run("sortmerge_tcsg", prog="tc", input=path), tc_ref)
        check(f"sortmerge tc {tag} plan 3", run("sortmerge_tcsg", prog="tc", input=path, plan=3), tc_ref)
        for neq in (0, 1):
            check(f"sortmerge sg neq={neq} {tag}", run("sortmerge_tcsg", prog="sg", input=path, neq=neq), sg_ref[neq])


def test_loop_words(tmp):
    """words=1e-9 makes every bitmap iteration word-level; loop=0 / words=0 switch them off; with
    bitmap auto the word-level join starts only after the switch from the hash table; loop-maxit 7
    forces several launches of the looping kernel."""
    print("TC / SG: looping kernel and word-level join")
    gs = dict(graphs(), dense=D.uniform_graph(1000, 2500, 7))
    combos = [dict(bitmap="force", words=1e-9, loop=0), dict(bitmap="force", words=1e-9, loop=262144),
              dict(bitmap="auto", words=1e-9, loop=262144), dict(bitmap="force", words=0, loop=262144),
              dict(bitmap="off", words=0, loop=262144, loop_maxit=7), dict(bitmap="force", words=1e-9, loop=64)]
    for (gname, E), wide in itertools.product(gs.items(), (0, 70000)):
        G = E + wide
        path = write(tmp, f"lw-{gname}-{wide}", G, 2)
        tc_ref = counted(R.tc_reference(G))
        sg_ref = {neq: counted(R.sg_reference(G[:, ::-1].copy(), neq)) for neq in (0, 1)}
        tag = f"{gname}{' (64-bit keys)' if wide else ''}"
        for opts in combos:
            for append in ("atomic", "warp"):
                o = dict(opts, append=append)
                check(f"fusedhash tc {tag} {o}", run("fusedhash_tcsg", prog="tc", input=path, **o), tc_ref)
                check(f"fusedhash tc {tag} plan 3 {o}", run("fusedhash_tcsg", prog="tc", input=path, plan=3, **o),
                      tc_ref)
                for neq in (0, 1):
                    check(f"fusedhash sg neq={neq} {tag} {o}",
                          run("fusedhash_tcsg", prog="sg", input=path, neq=neq, **o), sg_ref[neq])


def test_agg(tmp):
    print("CC / SSSP / company control")
    inputs = {"rmat": D.rmat(9), "grid": D.grid(8), "uniform": D.uniform_graph(400, 2000, 3)}
    for name, E in inputs.items():
        E = np.unique(np.concatenate([E, E[:, ::-1]]), axis=0)
        E = E[E[:, 0] != E[:, 1]]
        W = D.weights(E)
        path = write(tmp, f"agg-{name}", np.concatenate([E, W[:, None]], 1), 3)
        cc = R.cc_ref(E)
        ss = R.sssp_ref(E, W, 0)
        for exe in ("fusedhash_agg", "sortmerge_agg"):
            splits = ("delta", "output") if exe == "fusedhash_agg" else ("delta",)
            for split in splits:
                o = H.engine(exe, prog="cc", input=path, split=split, reps=1, fingerprint=1)
                check(f"{exe} cc {name} {split}", (o.data.get("result"), o.data.get("checksum"),
                                                   o.data.get("fingerprint")) if o.ok else o.kind, cc)
                o = H.engine(exe, prog="sssp", input=path, source=0, split=split, reps=1, fingerprint=1)
                check(f"{exe} sssp {name} {split}", (o.data.get("result"), o.data.get("checksum"),
                                                     o.data.get("fingerprint")) if o.ok else o.kind, ss)
    for n in (500, 3000):
        Own = D.ownership(n, seed=n)
        path = write(tmp, f"own-{n}", Own, 3)
        want = R.control_ref(Own)
        for exe in ("fusedhash_agg", "sortmerge_agg"):
            splits = ("delta", "output") if exe == "fusedhash_agg" else ("delta",)
            for split in splits:
                check(f"{exe} ctrl own-{n} {split}", run(exe, prog="ctrl", input=path, split=split), want)


def test_cspa(tmp):
    print("CSPA (induced subgraphs of httpd)")
    try:
        asg, der = D.cspa("httpd")
    except OSError:
        print("  skipped: data/cspa/httpd missing")
        return
    for t in (3000, 10000):
        A = asg[(asg < t).all(1)]
        Rr = der[(der < t).all(1)]
        _, inv = np.unique(np.concatenate([A.ravel(), Rr.ravel()]), return_inverse=True)
        A2 = inv[:A.size].reshape(A.shape)
        R2 = inv[A.size:].reshape(Rr.shape)
        d = os.path.join(tmp, f"cspa-{t}")
        os.makedirs(d, exist_ok=True)
        write(d, "assign", A2, 2)
        write(d, "dereference", R2, 2)
        want = R.cspa_ref(A2, R2)
        for chunk in (1 << 27, 4096):
            o = H.engine("fusedhash_cspa", input=d, chunk=chunk, reps=1, fingerprint=1)
            got = {k: o.data.get(k) for k in ("VF", "VA", "MA", "fp_VF", "fp_VA", "fp_MA")} if o.ok else o.kind
            check(f"fusedhash cspa t={t} chunk={chunk}", got, want)


def test_csda(tmp):
    """TC seeded by another relation (--base): random NullEdge facts on the test graphs, with ids
    beyond those of the graph too, every set mode of FusedHash, the looping kernel and the word-level
    join forced, a partitioned plan, and SortMerge."""
    print("CSDA (TC with --base)")
    rng = np.random.default_rng(5)
    for (gname, E), wide in itertools.product(graphs().items(), (0, 70000)):
        G = E + wide
        n = int(G.max()) + 1
        N = np.unique(np.stack([rng.integers(0, n + 40, 60), rng.integers(wide, n, 60)], 1), axis=0)
        gp, bp = write(tmp, f"csda-{gname}-{wide}", G, 2), write(tmp, f"csda-base-{gname}-{wide}", N, 2)
        want = counted(R.csda_reference(G, N))
        tag = f"{gname}{' (64-bit keys)' if wide else ''}"
        for opts in (dict(bitmap="off"), dict(bitmap="auto"), dict(bitmap="force"),
                     dict(bitmap="force", words=1e-9, loop=64), dict(split="output", words=1, loop=16384)):
            check(f"fusedhash csda {tag} {opts}", run("fusedhash_tcsg", prog="tc", input=gp, base=bp, **opts), want)
        check(f"fusedhash csda {tag} plan 3", run("fusedhash_tcsg", prog="tc", input=gp, base=bp, plan=3), want)
        check(f"sortmerge csda {tag}", run("sortmerge_tcsg", prog="tc", input=gp, base=bp), want)
        check(f"sortmerge csda {tag} plan 3", run("sortmerge_tcsg", prog="tc", input=gp, base=bp, plan=3), want)
        if not wide:                                       # FVlog and GDlog read tab-separated facts
            ef, nf = os.path.join(tmp, f"csda-{gname}.facts"), os.path.join(tmp, f"csda-null-{gname}.facts")
            np.savetxt(ef, G, fmt="%d", delimiter="\t")
            np.savetxt(nf, N, fmt="%d", delimiter="\t")
            for name, o in (("fvlog", H.fvlog("CSDA_TEST", (ef, nf), reps=1)), ("gdlog", H.gdlog("CSDA", (ef, nf), reps=1))):
                if o.kind != "missing":
                    check(f"{name} csda {gname}", o.data.get("result") if o.ok else o.kind, want[0])


def test_baselines():
    print("FVlog / GDlog / Souffle on the test graphs (if built)")
    for gname in ("t-uniform", "t-grid", "t-tree", "t-chain", "t-cycle"):
        if not os.path.exists(H.graph_bin(gname)):
            continue
        E = np.fromfile(H.graph_bin(gname), dtype=np.uint32).reshape(-1, 2).astype(np.int64)
        tc_ref = len(R.tc_reference(E))
        sg_ref = len(R.sg_reference(E[:, ::-1].copy(), False))
        for test, prog, want in (("TC_TEST", "tc_fvlog", tc_ref), ("SG_TEST", "sg_bigdatalog", sg_ref)):
            fv = H.fvlog(test, H.graph_facts(gname), reps=1)
            if fv.kind != "missing":
                check(f"fvlog {test} {gname}", fv.data.get("result") if fv.ok else fv.kind, want)
            gd = H.gdlog(test.split("_")[0], H.graph_facts(gname), reps=1)
            if gd.kind != "missing":
                check(f"gdlog {test.split('_')[0]} {gname}", gd.data.get("result") if gd.ok else gd.kind, want)
            sf = H.souffle(prog, H.graph_facts_dir(gname), reps=1)
            if sf.kind != "missing":
                check(f"souffle {prog} {gname}", sf.data.get("result") if sf.ok else sf.kind, want)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_tcsg(tmp)
        test_loop_words(tmp)
        test_agg(tmp)
        test_cspa(tmp)
        test_csda(tmp)
    test_baselines()
    print(f"\n{len(failures)} failure(s)")
    for f in failures:
        print("  ", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
