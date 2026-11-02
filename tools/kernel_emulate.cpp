// Run the CUDA WFA kernel on the host, one "thread" at a time, so kernel
// changes and the Rust CPU aligner can be cross-checked without a GPU.
//
// Build (new kernel):  g++ -O2 -std=c++17 -I. -DKERNEL_FILE='"cuda/wfa_kernel.cu"' -o emu tools/kernel_emulate.cpp
// Run:                 ./emu <seed> <ntrials> <gap_open> <gap_extend> <mismatch> <max_score> <max_len> <fill_byte>
// Output (TSV):        query ref gap_open gap_extend mismatch max_score max_k score cigar
//
// `fill_byte` is written over the whole workspace before every pair, so a
// kernel that relies on lazy row initialisation is exercised against stale
// data. Feed the output to `cargo test` via BLEP_KERNEL_EMULATION_TSV to
// compare against `cpu_align`, or diff the outputs of two kernel versions.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

struct Dim3 { unsigned x, y, z; };
static Dim3 blockIdx = {0, 0, 0};
static Dim3 blockDim = {1, 1, 1};
static Dim3 threadIdx = {0, 0, 0};
#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__
#include KERNEL_FILE

static std::string cigar_string(const char* ops, int len) {
    std::string out;
    const char sym[4] = {'=', 'X', 'I', 'D'};
    int i = 0;
    while (i < len) {
        int j = i;
        while (j < len && ops[j] == ops[i]) j++;
        out += std::to_string(j - i);
        out += sym[(int)ops[i] & 3];
        i = j;
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 9) {
        fprintf(stderr, "usage: %s seed ntrials gap_open gap_extend mismatch max_score max_len fill_byte\n", argv[0]);
        return 2;
    }
    unsigned seed = (unsigned)atoi(argv[1]);
    int ntrials = atoi(argv[2]);
    int go = atoi(argv[3]), ge = atoi(argv[4]), mm = atoi(argv[5]);
    int max_score = atoi(argv[6]);
    int max_len = atoi(argv[7]);
    int fill = atoi(argv[8]);

    std::mt19937 rng(seed);
    const char* bases = "ACGT";
    auto rnd = [&](int n) { return (int)(rng() % (unsigned)n); };

    int max_k = max_len;
    int num_diags = 2 * max_k + 1;
    long ws_len = 4L * (max_score + 1) * num_diags;
    std::vector<int> workspace(ws_len);
    int max_cigar_len = 2 * max_len;
    std::vector<char> out_cigar(max_cigar_len);

    for (int t = 0; t < ntrials; t++) {
        int m = 1 + rnd(max_len);
        std::string ref;
        for (int i = 0; i < m; i++) ref += bases[rnd(4)];
        std::string q;
        if (rnd(10) == 0) {
            int n = 1 + rnd(max_len);
            for (int i = 0; i < n; i++) q += bases[rnd(4)];
        } else {
            q = ref;
            int edits = rnd(5);
            for (int e = 0; e < edits && !q.empty(); e++) {
                int kind = rnd(3);
                int pos = rnd((int)q.size());
                if (kind == 0) {
                    q[pos] = bases[rnd(4)];
                } else if (kind == 1) {
                    int run = 1 + rnd(3);
                    for (int r = 0; r < run && (int)q.size() < max_len; r++) q.insert(q.begin() + pos, bases[rnd(4)]);
                } else {
                    int run = 1 + rnd(3);
                    for (int r = 0; r < run && q.size() > 1 && pos < (int)q.size(); r++) q.erase(q.begin() + pos);
                }
            }
        }
        if (q.empty()) q = "A";

        int qlen = (int)q.size(), qoff = 0, rlen = m, roff = 0, pq = 0, pr = 0;
        int out_score = -99, out_len = -99;
        memset(workspace.data(), fill, ws_len * sizeof(int));
        std::fill(out_cigar.begin(), out_cigar.end(), 0);
        blockIdx.x = 0; threadIdx.x = 0; blockDim.x = 1;
        wfa_align_kernel(q.c_str(), &qlen, &qoff, ref.c_str(), &rlen, &roff, &pq, &pr, 1,
                         go, ge, mm, max_k, num_diags, max_score,
                         workspace.data(), &out_score, out_cigar.data(), &out_len, max_cigar_len);
        printf("%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n", q.c_str(), ref.c_str(), go, ge, mm, max_score, max_k,
               out_score, cigar_string(out_cigar.data(), std::max(out_len, 0)).c_str());
    }
    return 0;
}
