#!/usr/bin/env python3
"""Appendix: context-sensitive points-to analysis (CSPA, rules of GDlog) on httpd, linux and
postgresql, with FVlog (its CSPA test), GDlog (its CSPA test, branch hash_diff), Souffle (one thread per
core) and FusedHash, all on the VM. GDlog's time is that of its own timers (gdlog_wall_s: its process).

  python3 bench/cspa.py        -> results/cspa.csv
"""
import os
import harness as H

NAMES = ["httpd", "linux", "postgresql"]


def main():
    out = H.Csv("cspa.csv", ["dataset", "VF", "VA", "MA", "fusedhash_s", "fvlog_s", "gdlog_s", "gdlog_wall_s",
                             "souffle_s", "souffle_rss_gib", "agree"])
    for n in NAMES:
        csv_dir = os.path.join(H.DATA, "cspa", n, f"cspa-{n}")
        facts_dir = os.path.join(H.DATA, "cspa_facts", n)
        fh = H.engine("fusedhash_cspa", input=facts_dir)
        fv = H.fvlog("CSPA_TEST", facts_dir)
        gd = H.gdlog("CSPA", facts_dir)
        sf = H.souffle("cspa", csv_dir)
        s = sf.data.get("sizes", {}) if sf.ok else {}
        v = fv.data.get("sizes", {}) if fv.ok else {}
        triples = []
        if sf.ok:
            triples.append((s.get("ValueFlow"), s.get("ValueAlias"), s.get("MemoryAlias")))
        if fv.ok:
            triples.append((v.get("value_flow"), v.get("value_alias"), v.get("memory_alias")))
        if gd.ok:
            g = gd.data.get("sizes", {})
            triples.append((g.get("VF"), g.get("VA"), g.get("MA")))
        if fh.ok:
            triples.append((fh.data.get("VF"), fh.data.get("VA"), fh.data.get("MA")))
        ok = len(set(triples)) <= 1
        vf, va, ma = triples[0] if triples else ("", "", "")
        print(f"{n:10s} VF={vf} VA={va} MA={ma}  fusedhash {fh.cell()}  fvlog {fv.cell()}  gdlog {gd.cell()}  "
              f"souffle {sf.cell()} "
              f"{'' if ok else 'MISMATCH ' + str(triples)}", flush=True)
        out.row(dataset=n, VF=vf, VA=va, MA=ma, fusedhash_s=fh.cell(), fvlog_s=fv.cell(), gdlog_s=gd.cell(),
                gdlog_wall_s=gd.cell("wall", "{:.2f}"),
                souffle_s=sf.cell("time", "{:.2f}"),
                souffle_rss_gib=f"{sf.data['rss_gib']:.1f}" if sf.data.get("rss_gib") else "", agree=int(ok))


if __name__ == "__main__":
    main()
