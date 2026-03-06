// WFA (Wavefront Alignment) CUDA kernel for GPU-accelerated sequence alignment.
//
// Each GPU thread handles one (query, reference) alignment pair.
//
// Memory layout:
//   - Wavefronts stored as flat arrays: wf[score][k + max_k]
//   - Three components per score: M (match/mismatch), I (insertion), D (deletion)
//   - Backtrack stored per (score, k): packed operation + source info
//
// Convention:
//   diagonal k = query_pos - ref_pos
//   offset = ref_pos (furthest reference position reached on diagonal k)
//   query_pos = offset + k

#define EMPTY (-2)

// CIGAR operation codes
#define OP_MATCH     0
#define OP_MISMATCH  1
#define OP_INSERTION 2
#define OP_DELETION  3

// Backtrack source component
#define COMP_M 0
#define COMP_I 1
#define COMP_D 2

// Pack backtrack entry: op(2 bits) | src_comp(2 bits) | prev_score(12 bits) | prev_k_offset(16 bits)
__device__ __forceinline__ int pack_bt(int op, int src_comp, int prev_score, int prev_k, int max_k) {
    return (op & 0x3)
         | ((src_comp & 0x3) << 2)
         | ((prev_score & 0xFFF) << 4)
         | (((prev_k + max_k) & 0xFFFF) << 16);
}

__device__ __forceinline__ void unpack_bt(int packed, int* op, int* src_comp, int* prev_score, int* prev_k, int max_k) {
    *op = packed & 0x3;
    *src_comp = (packed >> 2) & 0x3;
    *prev_score = (packed >> 4) & 0xFFF;
    *prev_k = ((packed >> 16) & 0xFFFF) - max_k;
}

// WFA alignment kernel.
//
// The host allocates a workspace buffer per thread of size:
//   3 * (max_score + 1) * num_diags ints  (for M, I, D wavefronts)
//   + (max_score + 1) * num_diags ints    (for backtrack)
// where num_diags = 2 * max_k + 1, max_k = max(max_read_len, max_ref_len)
//
// workspace layout per thread (all int32):
//   [0 .. S*D)                         = wf_m[score][diag]
//   [S*D .. 2*S*D)                     = wf_i[score][diag]
//   [2*S*D .. 3*S*D)                   = wf_d[score][diag]
//   [3*S*D .. 4*S*D)                   = bt_m[score][diag]  (backtrack for M component)
// where S = max_score + 1, D = num_diags
extern "C" __global__ void wfa_align_kernel(
    const char* __restrict__ queries,
    const int*  __restrict__ query_lengths,
    const int*  __restrict__ query_offsets,
    const char* __restrict__ refs,
    const int*  __restrict__ ref_lengths,
    const int*  __restrict__ ref_offsets,
    const int*  __restrict__ pair_query_idx,
    const int*  __restrict__ pair_ref_idx,
    int   num_pairs,
    int   gap_open,
    int   gap_extend,
    int   mismatch_penalty,
    int   max_k,           // max(max_read_len, max_ref_len)
    int   num_diags,       // 2 * max_k + 1
    int   max_score,       // score budget
    int*  __restrict__ workspace,       // Global workspace, size: num_pairs * 4 * (max_score+1) * num_diags
    int*  __restrict__ out_scores,
    char* __restrict__ out_cigars,
    int*  __restrict__ out_cigar_lengths,
    int   max_cigar_len
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;

    // Get sequences for this pair
    int qi = pair_query_idx[tid];
    int ri = pair_ref_idx[tid];
    const char* query = queries + query_offsets[qi];
    const char* ref_seq = refs + ref_offsets[ri];
    int n = query_lengths[qi];
    int m = ref_lengths[ri];
    int target_k = n - m;

    // Per-thread workspace pointers
    long S = (long)(max_score + 1);
    long D = (long)num_diags;
    long per_thread = 4L * S * D;
    int* my_ws = workspace + (long)tid * per_thread;

    int* wf_m = my_ws;
    int* wf_i = my_ws + S * D;
    int* wf_d = my_ws + 2L * S * D;
    int* bt_m = my_ws + 3L * S * D;

    // Initialize all wavefronts to EMPTY
    for (long i = 0; i < 4L * S * D; i++) {
        my_ws[i] = EMPTY;
    }

    // Output buffer
    char* my_cigar = out_cigars + (long)tid * max_cigar_len;

    // Macro for wavefront access
    #define WF_IDX(s, k) ((long)(s) * D + ((k) + max_k))
    #define WF_M(s, k) wf_m[WF_IDX(s, k)]
    #define WF_I(s, k) wf_i[WF_IDX(s, k)]
    #define WF_D(s, k) wf_d[WF_IDX(s, k)]
    #define BT_M(s, k) bt_m[WF_IDX(s, k)]

    // Initialize score 0
    WF_M(0, 0) = 0;

    // Extend score 0
    {
        int offset = WF_M(0, 0);
        int k = 0;
        while (offset < m && (offset + k) >= 0 && (offset + k) < n) {
            if (query[offset + k] == ref_seq[offset]) {
                offset++;
            } else {
                break;
            }
        }
        WF_M(0, 0) = offset;
    }

    // Check perfect match
    if (WF_M(0, target_k) >= m && target_k == 0) {
        // Perfect match
        int len = n < max_cigar_len ? n : max_cigar_len;
        for (int i = 0; i < len; i++) {
            my_cigar[i] = OP_MATCH;
        }
        out_scores[tid] = 0;
        out_cigar_lengths[tid] = len;
        return;
    }

    int final_score = -1;

    for (int s = 1; s <= max_score; s++) {
        // --- Insertions (gap in reference, query advances) ---
        // k increases by 1, offset stays
        int ins_s = s - gap_open - gap_extend;
        if (ins_s >= 0) {
            for (int ki = 0; ki < (int)D; ki++) {
                int prev_off = wf_m[ins_s * D + ki];
                if (prev_off == EMPTY) continue;
                int k = ki - max_k;
                int new_k = k + 1;
                int new_ki = new_k + max_k;
                if (new_ki >= 0 && new_ki < (int)D) {
                    if (prev_off > wf_i[s * D + new_ki]) {
                        wf_i[s * D + new_ki] = prev_off;
                        // Store backtrack: came from M at (ins_s, k)
                        // We'll store in bt_m when merged
                    }
                }
            }
        }

        // Insertion extensions (from I at s - gap_extend)
        int ins_ext_s = s - gap_extend;
        if (ins_ext_s >= 0) {
            for (int ki = 0; ki < (int)D; ki++) {
                int prev_off = wf_i[ins_ext_s * D + ki];
                if (prev_off == EMPTY) continue;
                int k = ki - max_k;
                int new_k = k + 1;
                int new_ki = new_k + max_k;
                if (new_ki >= 0 && new_ki < (int)D) {
                    if (prev_off > wf_i[s * D + new_ki]) {
                        wf_i[s * D + new_ki] = prev_off;
                    }
                }
            }
        }

        // --- Deletions (gap in query, reference advances) ---
        // k decreases by 1, offset increases by 1
        int del_s = s - gap_open - gap_extend;
        if (del_s >= 0) {
            for (int ki = 0; ki < (int)D; ki++) {
                int prev_off = wf_m[del_s * D + ki];
                if (prev_off == EMPTY) continue;
                int k = ki - max_k;
                int new_k = k - 1;
                int new_ki = new_k + max_k;
                int new_off = prev_off + 1;
                if (new_ki >= 0 && new_ki < (int)D) {
                    if (new_off > wf_d[s * D + new_ki]) {
                        wf_d[s * D + new_ki] = new_off;
                    }
                }
            }
        }

        // Deletion extensions (from D at s - gap_extend)
        int del_ext_s = s - gap_extend;
        if (del_ext_s >= 0) {
            for (int ki = 0; ki < (int)D; ki++) {
                int prev_off = wf_d[del_ext_s * D + ki];
                if (prev_off == EMPTY) continue;
                int k = ki - max_k;
                int new_k = k - 1;
                int new_ki = new_k + max_k;
                int new_off = prev_off + 1;
                if (new_ki >= 0 && new_ki < (int)D) {
                    if (new_off > wf_d[s * D + new_ki]) {
                        wf_d[s * D + new_ki] = new_off;
                    }
                }
            }
        }

        // --- Mismatches (from M at s - mismatch) ---
        int sub_s = s - mismatch_penalty;
        if (sub_s >= 0) {
            for (int ki = 0; ki < (int)D; ki++) {
                int prev_off = wf_m[sub_s * D + ki];
                if (prev_off == EMPTY) continue;
                int k = ki - max_k;
                int ref_pos = prev_off;
                int q_pos = prev_off + k;
                if (ref_pos < m && q_pos >= 0 && q_pos < n) {
                    int new_off = prev_off + 1;
                    if (new_off > wf_m[s * D + ki]) {
                        wf_m[s * D + ki] = new_off;
                        bt_m[s * D + ki] = pack_bt(OP_MISMATCH, COMP_M, sub_s, k, max_k);
                    }
                }
            }
        }

        // --- Merge I and D into M ---
        for (int ki = 0; ki < (int)D; ki++) {
            int i_off = wf_i[s * D + ki];
            if (i_off != EMPTY && i_off > wf_m[s * D + ki]) {
                wf_m[s * D + ki] = i_off;
                // Find the source of this insertion for backtracking
                // We need to figure out where this I came from
                // Check ins_open first, then ins_ext
                int k = ki - max_k;
                int prev_k = k - 1;
                int prev_ki = prev_k + max_k;
                if (ins_s >= 0 && prev_ki >= 0 && prev_ki < (int)D && wf_m[ins_s * D + prev_ki] == i_off) {
                    bt_m[s * D + ki] = pack_bt(OP_INSERTION, COMP_M, ins_s, prev_k, max_k);
                } else if (ins_ext_s >= 0 && prev_ki >= 0 && prev_ki < (int)D && wf_i[ins_ext_s * D + prev_ki] == i_off) {
                    bt_m[s * D + ki] = pack_bt(OP_INSERTION, COMP_I, ins_ext_s, prev_k, max_k);
                }
            }
            int d_off = wf_d[s * D + ki];
            if (d_off != EMPTY && d_off > wf_m[s * D + ki]) {
                wf_m[s * D + ki] = d_off;
                int k = ki - max_k;
                int prev_k = k + 1;
                int prev_ki = prev_k + max_k;
                if (del_s >= 0 && prev_ki >= 0 && prev_ki < (int)D && wf_m[del_s * D + prev_ki] != EMPTY && wf_m[del_s * D + prev_ki] + 1 == d_off) {
                    bt_m[s * D + ki] = pack_bt(OP_DELETION, COMP_M, del_s, prev_k, max_k);
                } else if (del_ext_s >= 0 && prev_ki >= 0 && prev_ki < (int)D && wf_d[del_ext_s * D + prev_ki] != EMPTY && wf_d[del_ext_s * D + prev_ki] + 1 == d_off) {
                    bt_m[s * D + ki] = pack_bt(OP_DELETION, COMP_D, del_ext_s, prev_k, max_k);
                }
            }
        }

        // --- Extend M wavefront ---
        for (int ki = 0; ki < (int)D; ki++) {
            int offset = wf_m[s * D + ki];
            if (offset == EMPTY) continue;
            int k = ki - max_k;
            while (offset < m && (offset + k) >= 0 && (offset + k) < n) {
                if (query[offset + k] == ref_seq[offset]) {
                    offset++;
                } else {
                    break;
                }
            }
            wf_m[s * D + ki] = offset;
        }

        // Check if target reached
        int target_ki = target_k + max_k;
        if (target_ki >= 0 && target_ki < (int)D) {
            if (wf_m[s * D + target_ki] >= m) {
                final_score = s;
                break;
            }
        }
    }

    if (final_score < 0) {
        // Score budget exceeded - signal CPU fallback
        out_scores[tid] = -1;
        out_cigar_lengths[tid] = 0;
        return;
    }

    // --- Backtrack to reconstruct CIGAR ---
    // Build operations in reverse, then reverse them
    char ops_buf[4096]; // Local buffer for operations (reverse order)
    int ops_count = 0;

    int cur_s = final_score;
    int cur_k = target_k;
    int cur_off = wf_m[cur_s * D + (cur_k + max_k)];

    while (cur_s > 0 && ops_count < 4095) {
        int packed = bt_m[cur_s * D + (cur_k + max_k)];
        if (packed == EMPTY) {
            // No backtrack - fill remaining with matches
            while (cur_off > 0 && ops_count < 4095) {
                ops_buf[ops_count++] = OP_MATCH;
                cur_off--;
            }
            break;
        }

        int op, src_comp, prev_s, prev_k;
        unpack_bt(packed, &op, &src_comp, &prev_s, &prev_k, max_k);

        // Get previous offset
        int prev_off;
        if (src_comp == COMP_M) {
            prev_off = wf_m[prev_s * D + (prev_k + max_k)];
        } else if (src_comp == COMP_I) {
            prev_off = wf_i[prev_s * D + (prev_k + max_k)];
        } else {
            prev_off = wf_d[prev_s * D + (prev_k + max_k)];
        }
        if (prev_off == EMPTY) prev_off = 0;

        // Offset after the operation (before extend)
        int off_after_op;
        if (op == OP_MISMATCH) {
            off_after_op = prev_off + 1;
        } else if (op == OP_INSERTION) {
            off_after_op = prev_off; // ref doesn't advance
        } else if (op == OP_DELETION) {
            off_after_op = prev_off + 1; // ref advances
        } else {
            off_after_op = prev_off;
        }

        // Matches from extend
        int num_matches = cur_off - off_after_op;
        for (int i = 0; i < num_matches && ops_count < 4095; i++) {
            ops_buf[ops_count++] = OP_MATCH;
        }

        // The operation itself
        if (ops_count < 4095) {
            ops_buf[ops_count++] = (char)op;
        }

        cur_s = prev_s;
        cur_k = prev_k;
        cur_off = prev_off;
    }

    // Handle remaining matches at score 0
    if (cur_s == 0) {
        while (cur_off > 0 && ops_count < 4095) {
            ops_buf[ops_count++] = OP_MATCH;
            cur_off--;
        }
    }

    // Reverse into output buffer
    int out_len = ops_count < max_cigar_len ? ops_count : max_cigar_len;
    for (int i = 0; i < out_len; i++) {
        my_cigar[i] = ops_buf[ops_count - 1 - i];
    }

    out_scores[tid] = final_score;
    out_cigar_lengths[tid] = out_len;

    #undef WF_IDX
    #undef WF_M
    #undef WF_I
    #undef WF_D
    #undef BT_M
}
