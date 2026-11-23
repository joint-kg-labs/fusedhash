"""Measurement rules shared by all the experiments on the GPU VM, applied in the same way to
every system:

  * one run = one process, with a timeout (default 900 s); a crash or an out-of-memory error
    never affects the following runs; runs are executed one at a time (never two GPU jobs);
  * our engines (FusedHash, SortMerge): warm-up on a small graph, then the best of two runs in
    the same process (the time covers index construction and fixpoint, with the input already in
    host memory; the result is counted and stays in GPU memory);
  * FVlog: its own timer around ram.execute(), as printed by its test drivers; best of two
    processes (FVlog has no warm-up option);
  * GDlog: its own timers, as printed by its test drivers (TC: the fixpoint, after the edges are
    loaded and indexed; SG: base rule + fixpoint; CSPA: initial and analysis SCC); best of two
    processes; for TC, SG and CSDA also the timer of the loading of the relations (copy to the GPU,
    sort, indexes), which its drivers measure and gdlog/patch_gdlog.py prints ("LOAD_TIME"): with it
    the time covers what ours covers (indexes + fixpoint, the input in host memory); the wall time of
    the process is kept too;
  * Souffle: compiled with OpenMP, one thread per core (24 on the A100 VM; 1 for the single-thread
    comparison), best of
    two runs; wall time of the process and peak RSS (/usr/bin/time -v); the result is counted
    with .printsize;
  * plans (decomposable programs only): single, then 4, 8, 16 partitions if memory runs out;
  * out of memory = exit code 3 or "OOM" printed by our engines, "out of memory" / bad_alloc in
    the messages of FVlog; the harness records OOM, timeout or crash, never a partial result;
  * every result is compared with the others (same number of facts), and a mismatch is recorded.
"""
from __future__ import annotations
import json, os, re, subprocess, time
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "bin")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FVLOG = os.path.join(ROOT, "tools", "vflog", "build", "test")
GDLOG = {"TC": os.path.join(ROOT, "tools", "gdlog", "build", "TC"),            # branch main
         "SG": os.path.join(ROOT, "tools", "gdlog-hash_diff", "build", "SG"),  # branch hash_diff
         "CSPA": os.path.join(ROOT, "tools", "gdlog-hash_diff", "build", "CSPA"),
         "CSDA": os.path.join(ROOT, "tools", "gdlog", "build", "CSDA")}       # our driver, gdlog/csda.cu
SOUFFLE = os.path.join(ROOT, "souffle", "bin")
PLANS = [1, 4, 8, 16]
TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", "900"))
THREADS = int(os.environ.get("SOUFFLE_THREADS", str(os.cpu_count())))   # one per core (no SMT on the VM)
OOM = ("out of memory", "outofmemory", "bad_alloc", "cudaerrormemoryallocation", "std::bad_alloc")


@dataclass
class Outcome:
    kind: str                                   # ok | oom | timeout | error | missing
    data: dict = field(default_factory=dict)
    message: str = ""

    def cell(self, key="time", fmt="{:.4f}"):
        if self.kind == "ok":
            return fmt.format(self.data[key])
        return {"oom": "OOM", "timeout": "timeout", "missing": "n/a"}.get(self.kind, "error")

    @property
    def ok(self):
        return self.kind == "ok"


def _run(cmd, timeout=TIMEOUT, cwd=None):
    t = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t
    return r, time.perf_counter() - t


def _is_oom(r):
    txt = (r.stdout + r.stderr).lower()
    return r.returncode == 3 or r.stdout.strip().endswith("OOM") or any(p in txt for p in OOM)


# ------------------------------------------------------------ our engines

def engine(exe, **args):
    """Runs bin/<exe> --k v ...; returns the parsed RESULT line."""
    path = os.path.join(BIN, exe)
    if not os.path.exists(path):
        return Outcome("missing", message=f"{path} not built")
    cmd = [path] + [f"--{k.replace('_', '-')}={v}" for k, v in args.items()]
    r, wall = _run(cmd)
    if r is None:
        return Outcome("timeout", message=f"> {TIMEOUT} s")
    if _is_oom(r):
        return Outcome("oom")
    res = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    if r.returncode or not res:
        tail = (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1]
        return Outcome("error", message=tail[:300])
    d = json.loads(res[-1][7:])
    d["wall"] = wall
    return Outcome("ok", d)


def with_plans(exe, decomposable, **args):
    """Tries the plans in order until memory suffices. Returns (outcome, plan)."""
    out = None
    for P in (PLANS if decomposable else [1]):
        out = engine(exe, plan=P, **args)
        if out.kind != "oom":
            return out, P
    return out, (PLANS[-1] if decomposable else 1)


# ------------------------------------------------------------ FVlog

def fvlog(test, path, reps=2):
    """TC_TEST <file> 0 0, SG_TEST <file> 0, CSPA_TEST <dir>, CSDA_TEST <edge.facts> <null_edge.facts> (our
    driver fvlog/csda_test.cu; path is then the pair). Best of `reps` processes."""
    exe = os.path.join(FVLOG, test)
    if not os.path.exists(exe):
        return Outcome("missing", message=f"{exe} not built")
    args = {"TC_TEST": [path, "0", "0"], "SG_TEST": [path, "0"], "CSPA_TEST": [path], "CSDA_TEST": list(path)}[test]
    best = None
    for _ in range(reps):
        r, wall = _run([exe] + args, cwd=os.path.dirname(exe))
        if r is None:
            return Outcome("timeout", message=f"> {TIMEOUT} s")
        if _is_oom(r):
            return Outcome("oom")
        m = re.findall(r"(?:Elapsed|Total) time:\s*([0-9.eE+-]+)\s*s", r.stdout)   # TC/SG: Elapsed, CSPA: Total
        if r.returncode or not m:
            tail = ((r.stderr or r.stdout).strip().splitlines() or [f"exit {r.returncode}"])[-1]
            return Outcome("error", message=tail[:300])
        sizes = {}
        for rel, full in re.findall(r"(\w+) Relation has FULL (\d+)", r.stdout):
            sizes[rel] = int(full)                  # last print of each relation = final size
        final = re.findall(r"FINAL_SIZE (\d+)", r.stdout)
        d = {"time": float(m[-1]), "wall": wall, "sizes": sizes,
             "result": int(final[-1]) if final else (max(sizes.values()) if sizes else None)}
        if best is None or d["time"] < best["time"]:
            best = d
    return Outcome("ok", best)


# ------------------------------------------------------------ GDlog

def gdlog(prog, path, reps=2):
    """GDlog (github.com/harp-lab/gdlog), as its run scripts call it: TC <edge.facts> 0 (with its merge
    buffer, the setting of its Table I), SG <edge.facts>, CSPA <dir with assign.facts and
    dereference.facts>, CSDA (edge.facts, null_edge.facts) (our driver gdlog/csda.cu: the TC of tc.cu
    on the transposed relations). Time = its own timers (TC: "TC time"; SG: "sg init time" + "sg time"; CSPA:
    "init scc time" + "analysis scc time"); load = "LOAD_TIME" (TC, SG, CSDA, when the drivers are
    patched; None otherwise); result = "Path counts", "sg counts", or the FINAL_SIZE lines that
    gdlog/patch_gdlog.py adds to CSPA. Best of `reps` processes (by time + load)."""
    exe = GDLOG[prog]
    if not os.path.exists(exe):
        return Outcome("missing", message=f"{exe} not built")
    args = {"TC": [path, "0"], "SG": [path], "CSPA": [path], "CSDA": list(path) + ["0"]}[prog]
    timers = {"TC": ["TC time"], "SG": ["sg init time", "sg time"],
              "CSPA": ["init scc time", "analysis scc time"], "CSDA": ["TC time"]}[prog]
    best = None
    for _ in range(reps):
        r, wall = _run([exe] + args, cwd=os.path.dirname(exe))
        if r is None:
            return Outcome("timeout", message=f"> {TIMEOUT} s")
        if _is_oom(r):
            return Outcome("oom")
        found = [re.findall(rf"^{re.escape(t)}:\s*([0-9.eE+-]+)", r.stdout, re.M) for t in timers]
        if r.returncode or not all(found):
            tail = ((r.stderr or r.stdout).strip().splitlines() or [f"exit {r.returncode}"])[-1]
            return Outcome("error", message=tail[:300])
        d = {"time": sum(float(f[-1]) for f in found), "wall": wall}
        load = re.findall(r"^LOAD_TIME\s+([0-9.eE+-]+)", r.stdout, re.M)
        d["load"] = float(load[-1]) if load else None
        d["total"] = d["time"] + (d["load"] or 0.0)
        if prog == "CSPA":
            d["sizes"] = {k: int(v) for k, v in re.findall(r"FINAL_SIZE (\w+) (\d+)", r.stdout)}
            d["result"] = d["sizes"].get("VF")
        else:
            m = re.findall(r"(?:Path counts|sg counts)\s+(\d+)", r.stdout)
            d["result"] = int(m[-1]) if m else None
        if best is None or d["total"] < best["total"]:
            best = d
    return Outcome("ok", best)


# ------------------------------------------------------------ Souffle

def souffle(prog, fact_dir, threads=THREADS, reps=2, timeout=TIMEOUT):
    exe = os.path.join(SOUFFLE, prog)
    if not os.path.exists(exe):
        return Outcome("missing", message=f"{exe} not built")
    best = None
    for _ in range(reps):
        cmd = ["/usr/bin/time", "-v", exe, "-F", fact_dir, "-j", str(threads)]
        r, wall = _run(cmd, timeout=timeout)
        if r is None:
            subprocess.run(["pkill", "-f", exe])
            return Outcome("timeout", message=f"> {timeout} s")
        m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", r.stderr)
        rss = int(m.group(1)) / 2**20 if m else None
        if r.returncode or not r.stdout.strip():
            first = ([l for l in r.stderr.splitlines() if l.strip() and "Command" not in l] or [""])[0]
            kind = "oom" if "bad_alloc" in r.stderr or "Killed" in r.stderr else "error"
            return Outcome(kind, {"rss_gib": rss}, first[:300])
        sizes = {k: int(v) for k, v in re.findall(r"^(\w+)\s+(\d+)\s*$", r.stdout, re.M)}
        d = {"time": wall, "rss_gib": rss, "sizes": sizes, "result": sum(sizes.values())}
        if best is None or d["time"] < best["time"]:
            best = d
        if wall > timeout / 2:                      # a second run would not fit the budget
            break
    return Outcome("ok", best)


# ------------------------------------------------------------ helpers

def graph_bin(name):
    return os.path.join(DATA, "bin", f"{name}.bin")


def graph_facts_dir(name):
    return os.path.join(DATA, "facts", name)


def graph_facts(name):
    return os.path.join(graph_facts_dir(name), "edge.facts")


def graph_shape(name):
    """(vertices, edges) of a prepared graph, from its binary file."""
    import numpy as np
    E = np.fromfile(graph_bin(name), dtype=np.uint32).reshape(-1, 2)
    return int(E.max()) + 1, len(E)


def agree(*outs):
    """True if all the successful outcomes report the same result."""
    vals = {o.data.get("result") for o in outs if o.ok and o.data.get("result") is not None}
    return len(vals) <= 1


class Csv:
    """Appends rows to results/<name>.csv (header on creation); flushes after every row."""

    def __init__(self, name, fields, resume=False):
        import csv
        os.makedirs(RESULTS, exist_ok=True)
        self.path = os.path.join(RESULTS, name)
        self.done = set()
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0   # empty: stopped before its header
        if resume and exists:
            self.done = {tuple(r[f] for f in fields[:2]) for r in csv.DictReader(open(self.path))}
        self.f = open(self.path, "a" if (resume and exists) else "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore")
        if not (resume and exists):
            self.w.writeheader()

    def row(self, **r):
        self.w.writerow(r)
        self.f.flush()
