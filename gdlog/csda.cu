#include <chrono>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <vector>

#include "../include/exception.cuh"
#include "../include/lie.cuh"
#include "../include/timer.cuh"
#include "../include/print.cuh"

//////////////////////////////////////////////////////

long int get_row_size(const char *data_path) {
    std::ifstream f;
    f.open(data_path);
    char c;
    long i = 0;
    while (f.get(c))
        if (c == '\n')
            ++i;
    f.close();
    return i;
}

enum ColumnT{ U64, U32};

column_type *get_relation_from_file(const char *file_path, int total_rows,
                                    int total_columns, char separator,
                                    ColumnT ct) {
    column_type *data =
        (column_type *)malloc(total_rows * total_columns * sizeof(column_type));
    FILE *data_file = fopen(file_path, "r");
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_columns; j++) {
            if (j != (total_columns - 1)) {
                if (ct == U64){
                fscanf(data_file, "%lld%c", &data[(i * total_columns) + j],
                       &separator);
                } else {
                    fscanf(data_file, "%ld%c", &data[(i * total_columns) + j],
                       &separator);
                }
            } else {
                if (ct == U64) {
                    fscanf(data_file, "%lld", &data[(i * total_columns) + j]);
                } else {
                    fscanf(data_file, "%ld", &data[(i * total_columns) + j]);
                }
            }
        }
    }
    return data;
}

//////////////////////////////////////////////////////////////////

__device__ void reorder_path(tuple_type inner, tuple_type outer,
                             tuple_type newt) {
    newt[0] = inner[1];
    newt[1] = outer[1];
};
__device__ void reorder_path1(tuple_type inner, tuple_type outer,
                             tuple_type newt) {
    newt[0] = outer[1];
    newt[1] = inner[1];
};
__device__ tuple_generator_hook reorder_path_device = reorder_path;
__device__ tuple_generator_hook reorder_path1_device = reorder_path1;

__device__ void cp_1(tuple_type src, tuple_type dest) {
    dest[0] = src[1];
    dest[1] = src[0];
}
__device__ tuple_copy_hook cp_1_device = cp_1;

// CSDA for GDlog (gdlog/csda.cu, copied into test/ by gdlog/patch_gdlog.py):
//   NullNode(x,y) :- NullEdge(x,y).   NullNode(x,y) :- NullNode(x,w), Edge(w,y).
// GDlog's TC (test/tc.cu) is right-recursive, path(x,z) :- edge(x,y), path(y,z), with path seeded by the
// graph G and the join index edge_2__2_1 holding G reversed. CSDA is left-recursive; transposed it is
// that same TC: R(y,x) = NullNode(x,y) is seeded by NullEdge reversed and extended by
// R(y,x) :- Edge^T(y,w), R(w,x), i.e. G = Edge reversed. So path is loaded with NullEdge reversed and
// edge_2__2_1 with Edge as it is (G reversed); the code of the fixpoint is that of analysis_bench in
// tc.cu, unchanged, and |path| = |NullNode|.
void csda_bench(const char *edge_path, const char *null_path, int block_size, int grid_size,
                bool fully_disable_buffer = false) {
    KernelTimer timer;
    tuple_size_t edge_counts = get_row_size(edge_path);
    tuple_size_t null_counts = get_row_size(null_path);
    std::cout << "Input edge rows: " << edge_counts << ", null edge rows: " << null_counts << std::endl;
    column_type *raw_edge = get_relation_from_file(edge_path, edge_counts, 2, '\t', U32);
    column_type *raw_null = get_relation_from_file(null_path, null_counts, 2, '\t', U32);
    column_type *raw_null_rev = (column_type *)malloc(null_counts * 2 * sizeof(column_type));
    for (tuple_size_t i = 0; i < null_counts; i++) {
        raw_null_rev[i * 2 + 1] = raw_null[i * 2];
        raw_null_rev[i * 2] = raw_null[i * 2 + 1];
    }

    timer.start_timer();
    Relation *edge_2__2_1 = new Relation();
    edge_2__2_1->fully_disable_merge_buffer_flag = fully_disable_buffer;
    Relation *path_2__1_2 = new Relation();
    path_2__1_2->fully_disable_merge_buffer_flag = fully_disable_buffer;
    path_2__1_2->index_flag = false;
    load_relation(path_2__1_2, "path_2__1_2", 2, raw_null_rev, null_counts, 1, 0, grid_size, block_size);
    load_relation(edge_2__2_1, "edge_2__2_1", 2, raw_edge, edge_counts, 1, 0, grid_size, block_size);
    timer.stop_timer();
    std::cout << "LOAD_TIME " << timer.get_spent_time() << std::endl;   // as patch_gdlog.py adds to tc.cu

    timer.start_timer();
    LIE tc_scc(grid_size, block_size);
    tc_scc.reload_full_flag = false;
    tc_scc.add_relations(edge_2__2_1, true);
    tc_scc.add_relations(path_2__1_2, false);
    float join_detail[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    tuple_generator_hook reorder_path_host;
    cudaMemcpyFromSymbol(&reorder_path_host, reorder_path_device, sizeof(tuple_generator_hook));
    tc_scc.add_ra(RelationalJoin(edge_2__2_1, FULL, path_2__1_2, DELTA, path_2__1_2, reorder_path_host,
                                 nullptr, LEFT, grid_size, block_size, join_detail));
    tc_scc.fixpoint_loop();
    timer.stop_timer();
    std::cout << "Path counts " << path_2__1_2->full->tuple_counts << std::endl;
    std::cout << "TC time: " << timer.get_spent_time() << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "usage: CSDA <edge.facts> <null_edge.facts> [1: no merge buffer]" << std::endl;
        return 1;
    }
    int device_id, number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);
    int block_size = 512, grid_size = 32 * number_of_sm;      // as in tc.cu
    std::locale loc("");
    csda_bench(argv[1], argv[2], block_size, grid_size, argc > 3 && strcmp(argv[3], "1") == 0);
    return 0;
}
