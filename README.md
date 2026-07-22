# blep

**GPU-accelerated sequence alignment using the Wavefront Alignment (WFA) algorithm on CUDA.**

blep aligns batches of query sequences against reference sequences entirely on the GPU, producing CIGAR strings in TSV output. It is designed for high-throughput bioinformatics workflows where thousands to millions of pairwise alignments need to be computed quickly.

The code in this repo was written by an AI assistant (claude opus) under guidance and supervision. Code has been tested using the included toy data, was well as on real scale data (not available). 

## Features

- **CUDA-accelerated WFA alignment** — each query×reference pair is aligned by a dedicated GPU thread, enabling massive parallelism
- **Multi-GPU support** — batches are distributed across all available GPUs (or an explicit `--devices` list), each driven by its own worker thread for near-linear scaling
- **Strand-aware alignment** — every read is automatically aligned in both forward and reverse complement orientations; only the best-scoring alignment is reported with a strand indicator (`+`/`-`)
- **Affine gap penalties** — configurable mismatch, gap-open, and gap-extension costs
- **FASTA & FASTQ input** — auto-detected format, with transparent gzip/bzip2/xz decompression via [niffler](https://crates.io/crates/niffler)
- **K-mer pre-filter** — optional k-mer overlap check skips unlikely alignments before they reach the GPU, saving compute (applied independently to both orientations)
- **Double-buffered pipeline** — a background CPU thread reads and pre-filters the next batch while the GPU aligns the current one
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
| `-b` | `--batch-size` | `10000` | Number of reads per batch |
| | `--max-read-length` | `500` | Maximum read length (for GPU memory pre-allocation) |
| | `--max-ref-length` | `500` | Maximum reference length (for GPU memory pre-allocation) |
| | `--max-score` | `256` | Score budget on GPU; pairs exceeding this fall back to CPU |
| | `--devices` | all GPUs | Comma-separated CUDA device indices to use (e.g. `0,1,2`); defaults to every available device |
| `-k` | `--kmer-length` | `0` (off) | K-mer length for the pre-alignment filter |
| `-t` | `--kmer-threshold` | `0.5` | Minimum fraction of shared k-mers to proceed with alignment |
| | `--header-tag` | — | Tag prefix to extract from read headers (repeatable) |
| | `--skip-fail` | off | Suppress output lines for pairs that fail the k-mer filter |
| `-v` | `--verbose` | — | Increase verbosity (`-v` info, `-vv` debug, `-vvv` trace) |

### Output format

Tab-separated values with columns:

```
read_name    reference_name    cigar    strand    [tag_values...]
```

- **cigar** — Extended CIGAR string (`=` match, `X` mismatch, `I` insertion, `D` deletion), or `FAILED` if the k-mer pre-filter rejected the pair in both orientations, or `*` if alignment could not be computed.
- **strand** — `+` if the best alignment is on the forward strand, `-` if the reverse complement produced a better score, or `.` if the pair was filtered/failed.

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

## Technical Details

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  CPU Reader Thread (background)                                │
│  ┌───────────┐   ┌────────────┐   ┌──────────────┐            │
│  │ Read batch│──▶│ Generate   │──▶│ K-mer filter │──▶ Batch   │
│  │ from disk │   │ rev. comp. │   │ (fwd + RC)   │      │      │
│  └───────────┘   └────────────┘   └──────────────┘      │      │
└─────────────────────────────────────────────────────────┼──────┘
                                                          │ shared queue
                                        ┌─────────────────┼─────────────────┐
                                        ▼                 ▼                 ▼
                              ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                              │ GPU Worker 0 │  │ GPU Worker 1 │  │ GPU Worker N │
                              │ upload→WFA→  │  │ upload→WFA→  │  │ upload→WFA→  │
                              │ download→best│  │ download→best│  │ download→best│
                              └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                                     └─────────────────┼─────────────────┘
                                                       ▼
                                          ┌─────────────────────────┐
                                          │ TSV writer (mutex-guarded)│
                                          └─────────────────────────┘
```

The pipeline is **pipelined and data-parallel**: a single background CPU thread reads, reverse-complements, and k-mer-filters batches, then hands each prepared batch to whichever GPU worker is free. Each worker owns its own CUDA context, kernel module, and reference copy, and processes an entire batch independently. Communication uses a bounded `mpsc::sync_channel(num_devices * 2)` whose receiver is shared across workers behind a `Mutex` (an MPMC queue built on std's MPSC).

**Output ordering:** each batch's rows are written under a single lock acquisition, so all rows for a given read stay contiguous. However, because batches finish on whichever GPU is free first, the relative order of batches in the output is not guaranteed to match the input order when more than one device is used. Sort downstream if a stable order is required.

### Reverse complement alignment

Every read is aligned against each reference in **both orientations** (forward and reverse complement). The best-scoring alignment is reported along with a strand indicator (`+` or `-`). This is always-on and requires no additional flags.

When the k-mer pre-filter is enabled, it is applied independently to each orientation:
- If only the forward orientation passes → only forward is aligned, reported as `+`
- If only the reverse complement passes → only RC is aligned, reported as `-`
- If both pass → both are aligned on GPU, the lower score wins
- If neither passes → the pair is reported as `FAILED` with strand `.`

This approach ensures that reverse-strand reads are correctly identified without requiring external pre-processing, while the k-mer filter still provides speedup by skipping unlikely orientation×reference combinations.

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

The kernel is compiled to PTX by `nvcc` at build time (via [`build.rs`](build.rs)) and embedded into the Rust binary using `include_str!`. At runtime it is loaded through [cudarc](https://crates.io/crates/cudarc).

### GPU memory management

Before launching kernels, each GPU worker queries its own device's available VRAM with `cudarc::driver::result::mem_get_info()` and computes the per-pair memory footprint. It uses 75% of free memory and automatically splits large batches into sub-batches that fit within the budget. Because sizing is per-device, GPUs with differing amounts of free memory are each used to their own capacity.

### CPU fallback

If a pair's alignment score exceeds `--max-score` on the GPU (the kernel returns score = −1), blep transparently re-aligns that pair on the CPU using a pure-Rust WFA implementation in [`src/main.rs`](src/main.rs:554).

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

