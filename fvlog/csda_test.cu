// CSDA for FVlog (fvlog/csda_test.cu, copied into test/src/ by fvlog/patch_fvlog.py and
// built as CSDA_TEST). The program of the FlowLog artifact (csda.dl):
//   NullNode(x,y) :- NullEdge(x,y).   NullNode(x,y) :- NullNode(x,w), Edge(w,y).
// Written as tc_mir of test/src/tc_test.cu, with the base rule reading null_edge instead of edge; the
// timer is the same (around ram_machine.execute()), and the final size is printed as the patched TC
// and SG drivers print it (FINAL_SIZE).
//   CSDA_TEST <edge.facts> <null_edge.facts>      (tab-separated, as the other drivers read them)
#include "mir.cuh"
#include "ram_instruction.cuh"
#include "utils.cuh"
#include "vflog.cuh"

#include <iostream>
#include <rmm/mr/device/cuda_memory_resource.hpp>

void csda_mir(char *edge_path, char *null_path) {
    using namespace vflog::mir;
    auto prog = program({
        declare("edge", {"a", "b"}, edge_path),
        declare("null_edge", {"a", "b"}, null_path),
        declare("null_node", {"a", "b"}),
        scc(
            {
                rule(single_head("null_node", {var("a"), var("b")}),
                     {body("null_edge", {var("a"), var("b")})}),
            },
            false),
        scc(
            {
                rule(single_head("null_node", {var("a"), var("c")}),
                     {body("null_node", {var("a"), var("b")}),
                      body("edge", {var("b"), var("c")})}),
            },
            true),
    });

    auto compiler = RAMGenPass(prog);
    compiler.compile();
    auto ram = compiler.ram_instructions;
    std::cout << "Compiled RAM program" << std::endl;

    vflog::ram::RelationalAlgebraMachine ram_machine;
    ram_machine.add_program(ram);

    std::cout << "Start executing" << std::endl;
    KernelTimer timer;
    timer.start_timer();
    ram_machine.execute();
    timer.stop_timer();
    auto elapsed = timer.get_spent_time();
    std::cout << "Elapsed time: " << elapsed << "s" << std::endl;
    ram_machine.rels["null_node"]->print_stats();
    std::cout << "FINAL_SIZE " << ram_machine.rels["null_node"]->get_versioned_size(FULL) << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <edge.facts> <null_edge.facts>" << std::endl;
        return 1;
    }
    rmm::mr::cuda_memory_resource cuda_mr{};                 // memory_system_flag 0 of the TC driver
    rmm::mr::set_current_device_resource(&cuda_mr);
    csda_mir(argv[1], argv[2]);
    return 0;
}
