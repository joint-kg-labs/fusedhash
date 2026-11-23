#!/usr/bin/env python3
"""Adds one line to the test drivers of FVlog (tc_test.cu, sg_test.cu) that prints the final size
of the result, so that the harness can check that FVlog derives the same facts as the other
systems, and adds the CSDA driver (fvlog/csda_test.cu, target CSDA_TEST in test/CMakeLists.txt).
Only the drivers are touched, never the engine; the patch is idempotent.

For each call  <ram>.rels["<rel>"]->print_stats();  it appends
  std::cout << "FINAL_SIZE " << <ram>.rels["<rel>"]->get_versioned_size(FULL) << std::endl;
"""
import re, sys, os

src = sys.argv[1] if len(sys.argv) > 1 else "tools/vflog/test/src"
pat = re.compile(r'(\w+)\.rels\["(\w+)"\]->print_stats\(\);')
for name in ("tc_test.cu", "sg_test.cu"):
    p = os.path.join(src, name)
    s = open(p).read()
    if "FINAL_SIZE" in s:
        print(f"{name}: already patched")
        continue
    s, n = pat.subn(lambda m: m.group(0) + f'\n    std::cout << "FINAL_SIZE " << {m.group(1)}.rels["{m.group(2)}"]'
                    f'->get_versioned_size(FULL) << std::endl;', s)
    open(p, "w").write(s)
    print(f"{name}: {n} line(s) added")

# the CSDA driver: copied next to the others, built like TC_TEST
here = os.path.dirname(os.path.abspath(__file__))
import shutil
shutil.copyfile(os.path.join(here, "csda_test.cu"), os.path.join(src, "csda_test.cu"))
cm = os.path.join(os.path.dirname(src), "CMakeLists.txt")
c = open(cm).read()
if "CSDA_TEST" in c:
    print("CMakeLists.txt: CSDA_TEST already there")
else:
    c += """
# CSDA (added by fvlog/patch_fvlog.py)
add_executable(CSDA_TEST src/csda_test.cu)
target_include_directories(CSDA_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(CSDA_TEST PRIVATE gpu_ra)
target_compile_features(CSDA_TEST PRIVATE cxx_std_17)
set_target_properties(CSDA_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(CSDA_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
"""
    open(cm, "w").write(c)
    print("CMakeLists.txt: CSDA_TEST added")
