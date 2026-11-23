#!/usr/bin/env bash
# Compiles the Souffle programs into native OpenMP executables (run with -j <threads>).
#   souffle/build_programs.sh            -> souffle/bin/<prog>
# Programs: tc_fvlog (Reach of FVlog/GDlog), sg_bigdatalog (x != y in the base rule only, as in
# the SG test of the FVlog code), sg_fvlog (x != y in both rules, as written in the FVlog paper),
# cspa (rules of GDlog), csda (the program of the FlowLog artifact).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
SOUFFLE="${SOUFFLE:-$root/tools/souffle/bin/souffle}"
[ -x "$SOUFFLE" ] || SOUFFLE="$(command -v souffle)"
mkdir -p "$here/bin"
for prog in ${PROGS:-tc_fvlog sg_bigdatalog sg_fvlog cspa csda}; do
    "$SOUFFLE" -j "$(nproc)" -o "$here/bin/$prog" "$here/$prog.dl"
    echo "ok: souffle/bin/$prog"
done
