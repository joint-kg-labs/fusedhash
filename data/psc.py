#!/usr/bin/env python3
"""Real ownership graph for company control, from the UK Companies House register of persons with
significant control (PSC snapshot, JSON lines). Optional: without it, company control runs on the
synthetic Own-n graphs only.

  curl -o data/raw/psc/psc-snapshot.zip \
    http://download.companieshouse.gov.uk/persons-with-significant-control-snapshot-<date>.zip
  python3 data/psc.py [data/raw/psc/psc-snapshot.zip]     -> data/npy/psc_own.npy

The paper uses the snapshot of 2026-09-28 (2.1 GB); Companies House publishes a new snapshot every
day and keeps only the recent ones, so later snapshots give slightly different counts.

Only the records of kind "corporate-entity-person-with-significant-control" are read, i.e.,
companies that control companies. The records of natural persons are discarded without being
decoded, and no personal data is stored.

  * edge: owner company (identification.registration_number) -> owned company (company_number);
    ceased relationships (ceased_on) are excluded;
  * share: the register reports bands only; we use the midpoint of the band of the shares
    (25-50% -> 37.5%, 50-75% -> 62.5%, 75-100% -> 87.5%), else of the voting rights;
  * ids: UK registration numbers normalized to 8 characters; foreign owners are distinguished by
    their country of registration.
Output rows: (owner, owned company, share in millionths), several records of a pair summed.
"""
import json, os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "raw", "psc", "psc-snapshot.zip")
OUT = os.path.join(HERE, "npy", "psc_own.npy")
BAND = re.compile(r"(ownership-of-shares|voting-rights)-(\d+)-to-(\d+)-percent")
UK = ("england", "wales", "scotland", "northern ireland", "united kingdom", "companies house", "uk")


def share(natures):
    best = {"ownership-of-shares": 0.0, "voting-rights": 0.0}
    for n in natures:
        m = BAND.match(n)
        if m:
            lo, hi = int(m.group(2)), int(m.group(3))
            best[m.group(1)] = max(best[m.group(1)], (lo + hi) / 200)
    return best["ownership-of-shares"] or best["voting-rights"]


def uk_number(s):
    s = re.sub(r"[^0-9A-Za-z]", "", s).upper()
    if not s:
        return None
    if s.isdigit():
        return s.zfill(8)
    m = re.match(r"^([A-Z]{2})(\d+)$", s)
    return m.group(1) + m.group(2).zfill(6) if m else s


def owner_key(ident):
    num = ident.get("registration_number") or ""
    if not num.strip():
        return None
    place = " ".join(str(ident.get(k, "")) for k in ("country_registered", "place_registered")).lower()
    if any(u in place for u in UK):
        n = uk_number(num)
        return "GB:" + n if n else None
    return "X:" + re.sub(r"\W", "", place)[:40] + ":" + re.sub(r"\W", "", num).upper()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ZIP
    p = subprocess.Popen(["unzip", "-p", path], stdout=subprocess.PIPE)
    ids, rows = {}, []
    nid = lambda k: ids.setdefault(k, len(ids))
    seen = kept = 0
    for line in p.stdout:
        seen += 1
        if b'"corporate-entity-person-with-significant-control"' not in line:
            continue                                       # natural persons and other kinds: skipped
        r = json.loads(line)
        d = r.get("data", {})
        if d.get("kind") != "corporate-entity-person-with-significant-control" or d.get("ceased_on"):
            continue
        w = share(d.get("natures_of_control", []))
        owner = owner_key(d.get("identification", {}))
        owned = uk_number(r.get("company_number", ""))
        if not w or owner is None or owned is None or owner == "GB:" + owned:
            continue
        rows.append((nid(owner), nid("GB:" + owned), int(w * 1e6)))
        kept += 1
        if kept % 200_000 == 0:
            print(f"  {seen:,} lines read, {kept:,} shareholdings", flush=True)
    p.wait()
    a = np.array(rows, dtype=np.int64)
    key = a[:, 0] * len(ids) + a[:, 1]                     # several records of the same pair: summed
    uk_, inv = np.unique(key, return_inverse=True)
    w = np.minimum(np.bincount(inv, weights=a[:, 2]), 1e6).astype(np.int64)
    out = np.stack([uk_ // len(ids), uk_ % len(ids), w], 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, out)
    print(f"lines {seen:,}; companies {len(ids):,}; shareholdings {len(out):,}; "
          f"foreign owners {sum(k.startswith('X:') for k in ids):,} -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
