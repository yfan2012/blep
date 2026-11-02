# blep

**GPU-accelerated sequence alignment using the Wavefront Alignment (WFA) algorithm on CUDA.**

blep aligns batches of query sequences against reference sequences entirely on the GPU, producing CIGAR strings in TSV output. It is designed for high-throughput bioinformatics workflows where thousands to millions of pairwise alignments need to be computed quickly.

The code in this repo was written by an AI assistant (claude opus) under guidance and supervision. Code has been tested using the included toy data, was well as on real scale data (not available). 

## Features

- **CUDA-accelerated WFA alignment** — each query×reference pair is aligned by a dedicated GPU thread, enabling massive parallelism
- **Multi-GPU support** — batches are distributed across all available GPUs (or an explicit `--devices` list), each driven by its own worker thread for near-linear scaling
- **Strand-aware alignment** — every read is automatically aligned in both forward and reverse complement orientations; only the best-scoring alignment is reported with a strand indicator (`+`/`-`)
- **Read deduplication** — identical read sequences within a batch are aligned once and the result is fanned back out to every read, so PCR-duplicate-heavy amplicon data aligns many times faster with byte-identical output. `--collapse` optionally emits one row per (sequence, header-tag values) group with an `n_reads` count instead
- **Affine gap penalties** — configurable mismatch, gap-open, and gap-extension costs
- **FASTA & FASTQ input** — auto-detected format, with transparent gzip/bzip2/xz decompression via [niffler](https://crates.io/crates/niffler)
- **K-mer pre-filter** — optional k-mer overlap check (k ≤ 32, 2-bit packed) skips unlikely alignments before they reach the GPU, saving compute (applied independently to both orientations)
- **Fully pipelined, multi-stage architecture** — disk reading, batch preparation (deduplication, reverse-complement, packing, k-mer filtering), GPU alignment, CPU fallback + output formatting, and TSV writing each run on their own thread(s) connected by bounded queues, so a GPU never blocks on CPU work or I/O. Preparation and post-processing are parallelized across CPU cores (tunable via `--prep-threads` / `--post-threads`) to keep every GPU fed. A per-stage timing summary (at `-v`) reports whether the GPUs are starving, blocked, or fully busy.
- **Automatic GPU memory management** — queries available VRAM and splits work into sub-batches to avoid out-of-memory errors
- **CPU fallback** — pairs that exceed the GPU score budget are transparently re-aligned on the CPU
- **Header tag extraction** — arbitrary tags (e.g. `RG:Z:`) can be pulled from read headers and included as extra output columns
- **Extended CIGAR output** — uses `=` / `X` / `I` / `D` notation distinguishing matches from mismatches

## Algorithm Background

blep implements the **Wavefront Alignment (WFA)** algorithm ([Marco-Sola et al., 2021](https://doi.org/10.1093/bioinformatics/btaa777)). WFA is an exact, gap-affine sequence alignment algorithm that operates in score-space rather than the traditional quadratic coordinate-space of Needleman–Wunsch or Smith–Waterman.

### How WFA works

1. **Wavefront expansion** — Instead of filling an *n × m* dynamic-programming matrix, WFA maintains a set of *wavefronts* indexed by alignment score. Each wavefront records, for every diagonal *k = i − j*, the furthest reference offset reached with that score.

2. **Three components** — To support affine gap penalties, three wavefront arrays are maintained per score:
   - **M** (match/mismatch) — the main wavefront
   - **I** (insertion) — tracks open/extending gaps in the reference
   - **D** (deletion) — tracks open/extending gaps in the query

3. **Extend phase** — After computing a new score's wavefront, exact-match characters are consumed greedily along each diagonal at no additional cost. This makes WFA extremely fast for similar sequences, as long runs of matches are handled in O(1) per diagonal.

4. **Termination** — The algorithm terminates as soon as the wavefront on the target diagonal (*k = n − m*) reaches the end of the reference. The final score is the optimal alignment cost.

5. **Backtracking** — Operation history is recorded during expansion and used to reconstruct the CIGAR string once the target is reached.

The key advantage of WFA is that its time complexity is **O(ns)** where *s* is the optimal alignment score, rather than O(nm). For highly similar sequences this is dramatically faster than classical DP approaches.

## Installation

### Prerequisites

- **Rust** (edition 2024 / nightly toolchain) — install via [rustup](https://rustup.rs/)
- **CUDA Toolkit** — `nvcc` must be on your `PATH` (tested with CUDA 12.x)
- **NVIDIA GPU** — Ampere or newer recommended (the kernel is compiled with `-arch=sm_80`)

### Build

```bash
cd blep
cargo build --release
```

The build script ([`build.rs`](build.rs)) invokes `nvcc` to compile the CUDA kernel ([`cuda/wfa_kernel.cu`](cuda/wfa_kernel.cu)) to PTX at build time. The resulting PTX is embedded into the binary, so no external kernel files are needed at runtime.

## Usage

```bash
blep -r reads.fastq.gz -R references.fasta -o results.tsv -v
```

### Required arguments

| Flag | Long | Description |
|------|------|-------------|
| `-r` | `--reads` | Path to query sequences (FASTA/FASTQ, optionally gzipped) |
| `-R` | `--references` | Path to reference sequences (FASTA/FASTQ, optionally gzipped) |

### Optional arguments

| Flag | Long | Default | Description |
|------|------|---------|-------------|
| `-o` | `--output` | stdout | Output TSV file path |
| `-g` | `--gap-open` | `2` | Gap open penalty |
| `-e` | `--gap-extend` | `1` | Gap extension penalty |
| `-m` | `--mismatch` | `4` | Mismatch penalty |
| `-b` | `--batch-size` | `10000` | Number of reads per batch. Reads are deduplicated within a batch, so larger batches collapse more duplicates |
| | `--max-read-length` | `500` | Maximum read length (for GPU memory pre-allocation). Longer reads are not aligned: they get cigar `*` and are counted in a warning |
| | `--max-ref-length` | `500` | Maximum reference length (for GPU memory pre-allocation). A longer reference is an error at startup |
| | `--max-score` | `256` | Score budget on GPU; pairs exceeding this fall back to CPU |
| | `--devices` | all GPUs | Comma-separated CUDA device indices to use (e.g. `0,1,2`); defaults to every available device |
| | `--prep-threads` | `0` (auto) | Number of CPU threads that prepare batches (reverse-complement, packing, k-mer filter) before the GPUs. This is the stage most likely to starve the GPUs. Auto ≈ hardware threads minus GPU workers |
| | `--post-threads` | `0` (auto) | Number of CPU threads that post-process GPU results (CPU fallback + output formatting) off the GPU worker threads. Auto ≈ number of GPUs |
| `-k` | `--kmer-length` | `0` (off) | K-mer length (1–32) for the pre-alignment filter |
| `-t` | `--kmer-threshold` | `0.5` | Minimum fraction of shared k-mers to proceed with alignment |
| | `--header-tag` | — | Tag prefix to extract from read headers (repeatable) |
| | `--skip-fail` | off | Suppress output lines for pairs that fail the k-mer filter |
| | `--no-dedup` | off | Disable within-batch read deduplication (for A/B checks only; deduplication is lossless) |
| | `--collapse` | off | Emit one row per (sequence, header-tag values) group with a trailing `n_reads` column, instead of one row per read |
| `-v` | `--verbose` | — | Increase verbosity (`-v` info, `-vv` debug, `-vvv` trace) |

### Output format

Tab-separated values with columns:

```
read_name    reference_name    cigar    strand    [tag_values...]    [n_reads]
```

- **cigar** — Extended CIGAR string (`=` match, `X` mismatch, `I` insertion, `D` deletion), or `FAILED` if the k-mer pre-filter rejected the pair in both orientations, or `*` if alignment could not be computed (including reads longer than `--max-read-length`).
- **strand** — `+` if the best alignment is on the forward strand, `-` if the reverse complement produced a better score, or `.` if the pair was filtered/failed.
- **n_reads** — only with `--collapse`: the number of reads (same sequence, same tag values) the row stands for. It is always the last column, so positional parsers of the earlier columns keep working.

By default there is one row per read × reference, and all rows for a read are contiguous.

### Examples

Basic alignment:

```bash
blep -r reads.fasta -R refs.fasta -o out.tsv
```

With k-mer pre-filter (k=15, require 30% overlap):

```bash
blep -r reads.fastq.gz -R refs.fasta -k 15 -t 0.3 -o out.tsv -v
```

Custom penalties and larger score budget:
```bash
blep -r reads.fq -R refs.fa -g 4 -e 2 -m 6 --max-score 512 -o out.tsv
```

Extract a read-group tag from headers:

```bash
blep -r reads.fastq -R refs.fasta --header-tag "RG:Z:" -o out.tsv
```

Collapsed output (one row per unique sequence per cell barcode, with read counts):

```bash
blep -r reads.fastq -R refs.fasta --header-tag "RG:Z:" --collapse -o out.tsv
```

## Technical Details

### Architecture

```
┌───────────┐   raw batches   ┌───────────────────────────────────┐
│  Reader   │────────────────▶│  Prep pool  (P threads)           │
│  thread   │  (bounded queue) │ dedup · rev-comp · pack · k-mer   │
└───────────┘                  └──────────────────┬────────────────┘
                                                   │ prepared batches (bounded queue)
                         ┌─────────────────────────┼─────────────────────────┐
                         ▼                         ▼                         ▼
               ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
               │ GPU Worker 0 │          │ GPU Worker 1 │   ...    │ GPU Worker N │
               │ upload→WFA→  │          │ upload→WFA→  │          │ upload→WFA→  │
               │ download     │          │ download     │          │ download     │
               └──────┬───────┘          └──────┬───────┘          └──────┬───────┘
                      └─────────────────────────┼─────────────────────────┘
                                                │ GPU results (bounded queue)
                                ┌───────────────┴────────────────┐
                                ▼                                ▼
                       ┌──────────────┐                 ┌──────────────┐
                       │  Post pool   │      ...        │  Post pool   │
                       │ CPU fallback │                 │ + row format │
                       └──────┬───────┘                 └──────┬───────┘
                              └───────────────┬────────────────┘
                                              │ formatted rows (bounded queue)
                                       ┌──────▼───────┐
                                       │ Writer thread│
                                       │ (TSV output) │
                                       └──────────────┘
```

The pipeline is **fully decoupled and data-parallel**. Each stage runs on its own thread(s) and is connected to the next by a bounded `mpsc::sync_channel`; the pooled stages (prep, GPU workers, post) share their channel's receiver behind a `Mutex`, giving an MPMC queue built on std's MPSC. The design deliberately keeps CPU work off the GPU threads:

- **Reader** (1 thread) only reads raw batches from disk.
- **Prep pool** (`--prep-threads`, auto ≈ cores − GPUs) deduplicates, reverse-complements, packs, and k-mer-filters batches in parallel. This is the heaviest CPU stage and the one most likely to starve the GPUs, hence a pool.
- **GPU workers** (one per device) do *only* GPU work — upload, kernel launch, synchronize, download — then hand the raw results downstream and immediately pull the next batch. Each owns its own CUDA context, kernel module, and reference copy.
- **Post pool** (`--post-threads`) runs the CPU fallback aligner (for pairs over the score budget), resolves each unique sequence once, and fans the result out to its reads as output rows, so a GPU is never blocked re-aligning or formatting.
- **Writer** (1 thread) serializes rows to the output TSV, flushing periodically.

Because the GPU workers never touch disk, the CPU fallback, formatting, or the writer lock, they stay busy as long as the prep pool can keep the prepared-batch queue full. Run with `-v` to get a per-stage timing summary that reports what fraction of GPU time was spent aligning vs. starved (waiting for input) vs. blocked (waiting on downstream), plus a hint on which knob to turn.

**Output ordering:** each batch's rows are written together, so all rows for a given read stay contiguous. However, because batches flow through parallel prep, GPU, and post stages, the relative order of batches in the output is not guaranteed to match the input order. Sort downstream if a stable order is required.

### Reverse complement alignment

Every read is aligned against each reference in **both orientations** (forward and reverse complement). The best-scoring alignment is reported along with a strand indicator (`+` or `-`). This is always-on and requires no additional flags.

When the k-mer pre-filter is enabled, it is applied independently to each orientation:
- If only the forward orientation passes → only forward is aligned, reported as `+`
- If only the reverse complement passes → only RC is aligned, reported as `-`
- If both pass → both are aligned on GPU, the lower score wins
- If neither passes → the pair is reported as `FAILED` with strand `.`

This approach ensures that reverse-strand reads are correctly identified without requiring external pre-processing, while the k-mer filter still provides speedup by skipping unlikely orientation×reference combinations.

### Read deduplication

Alignment depends only on a read's sequence, never on its name, quality string or header tags. Within each batch (`--batch-size` reads), blep groups reads by exact (uppercased) sequence, aligns each distinct sequence once against every reference in both orientations, and fans the result back out to every read that carried it. This is lossless: the default output is identical to a run with `--no-dedup`, just faster. PCR amplicon data is highly redundant (the wild-type allele and the common edits of each amplicon recur thousands of times), so the prep, GPU and CPU-fallback stages all shrink by the duplication factor.

Deduplication is per batch, so larger `--batch-size` values collapse more duplicates. The per-batch log line (`-v`) reports the ratio, e.g. `aligning batch of 10000 reads (2314 unique, 4.3x)`, and the final summary reports the overall ratio.

With `--collapse`, blep writes one row per (sequence, header-tag values) group instead of one row per read. The group key includes the tag values, so the same amplicon sequence seen in two cells (`--header-tag RG:Z:`) stays two rows. `read_name` is the first read of the group and a trailing `n_reads` column gives the group size. Groups are formed within a batch, so a (sequence, tags) combination that spans batches appears on several rows; downstream code should sum `n_reads` rather than count rows. The missionbio `count_cigars_per_cell.py` does this automatically when it sees an `n_reads` header column; `extract_unaligned.py` needs per-read output.

### Input normalization and length limits

- Reads and references are uppercased on load, so alignment is case-insensitive and `acgt` deduplicates with `ACGT`.
- Reads longer than `--max-read-length` are not aligned (the GPU workspace is sized from that limit). They are reported with cigar `*` and strand `.` against every reference, and a warning at the end gives their count and the longest read seen, so the limit can be raised and the run repeated.
- A reference longer than `--max-ref-length` is an error at startup.
- `--kmer-length` must be between 1 and 32. K-mers are 2-bit packed; windows containing a non-ACGT base (e.g. `N`) are ignored, and a read with no valid k-mers passes the filter.

### CUDA kernel

The WFA kernel in [`cuda/wfa_kernel.cu`](cuda/wfa_kernel.cu) assigns **one thread per alignment pair**. Each thread operates on its own slice of a pre-allocated global-memory workspace:

| Region | Size (per thread) | Purpose |
|--------|-------------------|---------|
| `wf_m` | `(max_score+1) × num_diags` ints | M-wavefront (match/mismatch) |
| `wf_i` | `(max_score+1) × num_diags` ints | I-wavefront (insertions) |
| `wf_d` | `(max_score+1) × num_diags` ints | D-wavefront (deletions) |
| `bt_m` | `(max_score+1) × num_diags` ints | Backtrack entries (packed) |

where `num_diags = 2 × max_k + 1` and `max_k = max(max_read_length, max_ref_length)`.

Backtrack entries are bit-packed into a single 32-bit integer: 2 bits for the operation, 2 bits for the source component, 12 bits for the previous score, and 16 bits for the previous diagonal index.

The workspace is allocated once per GPU worker (sized for the largest sub-batch that fits in 75% of free VRAM) and reused across sub-batches. The kernel initialises wavefront rows lazily: row *s* is reset to EMPTY at the start of score iteration *s*, so a pair that aligns with score *S* touches *S*+1 rows rather than the whole `max_score+1`-row workspace, and stale contents from earlier launches are never read. Loops over a row are restricted to the band of diagonals reachable at that score (|*k*| ≤ *s* / `gap_extend`, since every k-changing transition costs at least `gap_extend`); cells outside the band are EMPTY by construction, so results are identical to scanning every diagonal.

The kernel is compiled to PTX by `nvcc` at build time (via [`build.rs`](build.rs)) and embedded into the Rust binary using `include_str!`. At runtime it is loaded through [cudarc](https://crates.io/crates/cudarc).

### GPU memory management

Before launching kernels, each GPU worker queries its own device's available VRAM with `cudarc::driver::result::mem_get_info()` and computes the per-pair memory footprint. It uses 75% of free memory and automatically splits large batches into sub-batches that fit within the budget; the wavefront workspace for a full sub-batch is allocated once up front and only the small per-sub-batch output buffers are allocated per launch. Because sizing is per-device, GPUs with differing amounts of free memory are each used to their own capacity.

### CPU fallback

If a pair's alignment score exceeds `--max-score` on the GPU (the kernel returns score = −1), blep transparently re-aligns that pair on the CPU using a pure-Rust WFA implementation (`cpu_align` in [`src/main.rs`](src/main.rs)). This runs on the post-processing pool, not the GPU worker threads, so a batch full of over-budget pairs does not stall a GPU.

The CPU aligner is a direct port of the kernel (same wavefront layout, lazily allocated rows, banded loops, same backtrack rules), so it produces the same CIGARs; it gives up at score 4096, in which case the pair is reported as `*`. It returns the alignment score, which is used when choosing between the forward and reverse-complement orientations, so a pair whose orientations both exceeded the GPU budget is still assigned the better strand.

### Testing

`cargo test` runs CPU-only unit tests (deduplication and fan-out, collapsed output, k-mer packing, the CPU aligner); no GPU is needed. `./test.sh` runs the release binary end to end on the toy data in [`test_data/`](test_data/) and checks that deduplicated, `--no-dedup` and `--collapse` runs agree; it needs a CUDA GPU.

[`tools/kernel_emulate.cpp`](tools/kernel_emulate.cpp) compiles the CUDA kernel as ordinary host C++ and runs it on random read/reference pairs, so kernel changes can be checked for identical output against a previous version, and the Rust CPU aligner can be cross-checked against the kernel, without a GPU:

```bash
g++ -O2 -std=c++17 -I. -DKERNEL_FILE='"cuda/wfa_kernel.cu"' -o emu tools/kernel_emulate.cpp
./emu 1 4000 2 1 4 60 40 127 > emu.tsv          # seed trials gap_open gap_extend mismatch max_score max_len fill_byte
BLEP_KERNEL_EMULATION_TSV=$PWD/emu.tsv cargo test cpu_align_matches_kernel_emulation
```

### Dependencies

| Crate | Purpose |
|-------|---------|
| [clap](https://crates.io/crates/clap) | Command-line argument parsing |
| [bio](https://crates.io/crates/bio) | FASTA/FASTQ I/O |
| [niffler](https://crates.io/crates/niffler) | Transparent decompression (gzip, bzip2, xz) |
| [cudarc](https://crates.io/crates/cudarc) | Safe Rust bindings for CUDA driver API |
| [csv](https://crates.io/crates/csv) | TSV output writing |
| [anyhow](https://crates.io/crates/anyhow) | Error handling |
| [log](https://crates.io/crates/log) / [env_logger](https://crates.io/crates/env_logger) | Logging |

