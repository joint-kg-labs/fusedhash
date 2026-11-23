#!/usr/bin/env python3
"""Adds to the CSPA test driver of GDlog (test/cspa.cu, branch hash_diff) the lines that print the final
sizes of ValueFlow, ValueAlias and MemoryAlias, so that the harness can check that GDlog derives the
same facts as the other systems (its TC and SG drivers already print the size of the result). Adds to
the TC (branch main) and SG (branch hash_diff) drivers a line that prints the first timer of
analysis_bench, which they measure but do not print: the loading of the relations (copy to the GPU, sort
and indexes), as "LOAD_TIME <t>". Our engines' times include the construction of their indexes, so the
harness reports GDlog's loading, its fixpoint and their sum. Only the drivers are touched, never the
engine; and adds the CSDA driver (gdlog/csda.cu, target CSDA on the branch main, built like TC, which
prints LOAD_TIME itself). The patch is idempotent.

  python3 gdlog/patch_gdlog.py tools/gdlog tools/gdlog-hash_diff
"""
import os, sys

hd = sys.argv[2] if len(sys.argv) > 2 else "tools/gdlog-hash_diff"
p = os.path.join(hd, "test", "cspa.cu")
s = open(p).read()
anchor = 'std::cout << "analysis scc time: " << timer.get_spent_time() << std::endl;'
if "FINAL_SIZE" in s:
    print("cspa.cu: already patched")
elif anchor not in s:
    sys.exit("cspa.cu: the line that prints the analysis time was not found; the driver changed")
else:
    add = "".join(f'\n    std::cout << "FINAL_SIZE {rel} " << {var}->full->tuple_counts << std::endl;'
                  for rel, var in (("VF", "value_flow_2__1_2"), ("VA", "value_alias_2__1_2"),
                                   ("MA", "memory_alias_2__1_2")))
    open(p, "w").write(s.replace(anchor, anchor + add, 1))
    print("cspa.cu: 3 lines added")

def print_load_time(path):
    """After the first timer.stop_timer() of analysis_bench (the loading of the relations), print it."""
    s = open(path).read()
    if "LOAD_TIME" in s:
        print(f"{os.path.basename(path)}: LOAD_TIME already printed")
        return
    start = s.find("void analysis_bench(")          # the definition (not a declaration, not analysis_bench2)
    start = start if start >= 0 else s.find("analysis_bench(")
    stop = s.find("timer.stop_timer();", start)
    if start < 0 or stop < 0 or s.find("timer.start_timer();", start) > stop:
        sys.exit(f"{path}: the first timer of analysis_bench was not found; the driver changed")
    at = stop + len("timer.stop_timer();")
    line = '\n    std::cout << "LOAD_TIME " << timer.get_spent_time() << std::endl;'
    open(path, "w").write(s[:at] + line + s[at:])
    print(f"{os.path.basename(path)}: LOAD_TIME added")


print_load_time(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "tools/gdlog", "test", "tc.cu"))
print_load_time(os.path.join(hd, "test", "sg.cu"))

# the CSDA driver on the branch main, built like TC
import shutil
main_dir = sys.argv[1] if len(sys.argv) > 1 else "tools/gdlog"
here = os.path.dirname(os.path.abspath(__file__))
shutil.copyfile(os.path.join(here, "csda.cu"), os.path.join(main_dir, "test", "csda.cu"))
cm = os.path.join(main_dir, "CMakeLists.txt")
c = open(cm).read()
if "add_executable(CSDA" in c:
    print("CMakeLists.txt: CSDA already there")
else:
    c += """
# CSDA (added by gdlog/patch_gdlog.py)
add_executable(CSDA "${PROJECT_SOURCE_DIR}/test/csda.cu")
target_link_libraries(CSDA gpu_ra)
target_compile_features(CSDA PUBLIC cxx_std_20)
set_target_properties(CSDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
"""
    open(cm, "w").write(c)
    print("CMakeLists.txt: CSDA added")
