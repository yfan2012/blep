use std::path::PathBuf;
use std::fs::File;
use std::io::{self, Read, BufReader};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::sync::mpsc;
use std::thread;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, Context, anyhow, ensure};
use clap::Parser;
use bio::io::{fasta, fastq};
use bio::io::fasta::FastaRead;
use bio::io::fastq::FastqRead;
use log::{info, debug, warn};
use csv::WriterBuilder;
use niffler::{get_reader, compression};
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

/// BLEP: GPU-accelerated sequence alignment using the WFA algorithm (CUDA)
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Path to the FASTA/FASTQ file containing reads to align (can be gzipped)
    #[clap(short = 'r', long)]
    reads: PathBuf,

    /// Path to the FASTA/FASTQ file containing reference sequences (can be gzipped)
    #[clap(short = 'R', long)]
    references: PathBuf,

    /// Path to the output TSV file (default: stdout)
    #[clap(short = 'o', long)]
    output: Option<PathBuf>,

    /// Gap open penalty
    #[clap(short = 'g', long, default_value = "2")]
    gap_open: i32,

    /// Gap extension penalty
    #[clap(short = 'e', long, default_value = "1")]
    gap_extend: i32,

    /// Mismatch penalty
    #[clap(short = 'm', long, default_value = "4")]
    mismatch: i32,

    /// Batch size for processing reads (number of reads to load at once).
    /// Reads are deduplicated within a batch, so larger batches collapse more
    /// duplicates.
    #[clap(short = 'b', long, default_value = "10000")]
    batch_size: usize,

    /// Maximum read length (for GPU memory pre-allocation). Longer reads are
    /// not aligned; they are reported with cigar `*` and counted in a warning.
    #[clap(long, default_value = "500")]
    max_read_length: usize,

    /// Maximum reference length (for GPU memory pre-allocation). A longer
    /// reference is an error at startup.
    #[clap(long, default_value = "500")]
    max_ref_length: usize,

    /// Maximum alignment score budget (alignments exceeding this fall back to CPU)
    #[clap(long, default_value = "256")]
    max_score: usize,

    /// Comma-separated list of CUDA device indices to use (e.g. "0,1,2").
    /// If omitted, all available CUDA devices are used. Each device runs in its
    /// own worker thread and processes batches in parallel.
    #[clap(long, value_delimiter = ',')]
    devices: Option<Vec<usize>>,

    /// Number of CPU threads that prepare batches (deduplication,
    /// reverse-complement, packing, and the k-mer pre-filter) before they reach
    /// the GPUs. This preparation is the stage most likely to starve the GPUs,
    /// so it runs as a parallel pool.
    /// 0 = auto (roughly all hardware threads not reserved for GPU workers).
    #[clap(long, default_value = "0")]
    prep_threads: usize,

    /// Number of CPU threads that post-process GPU results (CPU fallback
    /// alignment and output formatting) off the GPU worker threads, so a GPU is
    /// never blocked formatting or re-aligning. 0 = auto.
    #[clap(long, default_value = "0")]
    post_threads: usize,

    /// K-mer length for the pre-alignment filter (1-32). When set (> 0), unique
    /// k-mers of the read are checked against the reference before running WFA.
    /// K-mers containing a non-ACGT base are ignored.
    #[clap(short = 'k', long, default_value = "0")]
    kmer_length: usize,

    /// Minimum fraction (0.0–1.0) of unique read k-mers that must be present in
    /// the reference for alignment to proceed. Pairs below this threshold are
    /// reported as FAILED. Only used when --kmer-length > 0.
    #[clap(short = 't', long, default_value = "0.5")]
    kmer_threshold: f64,

    /// Tags to extract from FASTQ/FASTA read headers and include as extra output
    /// columns. Specify the full tag prefix (e.g. "RG:Z:"). The tag value is the
    /// whitespace-delimited token immediately following the prefix in the header
    /// description. May be specified multiple times.
    #[clap(long = "header-tag", num_args = 1)]
    header_tags: Vec<String>,

    /// Skip output for read/reference pairs where both orientations fail the
    /// k-mer filter. By default, failed pairs are emitted with "FAILED" in the
    /// cigar column.
    #[clap(long)]
    skip_fail: bool,

    /// Disable within-batch read deduplication. By default, reads with identical
    /// sequences in a batch are aligned once and the result is reused for every
    /// copy. Deduplication is lossless, so this flag exists only for A/B checks.
    #[clap(long)]
    no_dedup: bool,

    /// Collapse the output: emit one row per (read sequence, header-tag values)
    /// group instead of one row per read, with a trailing `n_reads` column
    /// giving the number of reads in the group. Consecutive reads that share a
    /// name (unmerged mates written by fastp) are collapsed as one unit and
    /// keep one shared `read_name`, so mates can still be paired downstream;
    /// `n_reads` then counts pairs. `read_name` is the first read of the group.
    /// Groups are formed within a batch.
    #[clap(long)]
    collapse: bool,

    /// Verbosity level
    #[clap(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

// CIGAR operation codes (must match kernel defines)
const OP_MATCH: u8 = 0;
const OP_MISMATCH: u8 = 1;
const OP_INSERTION: u8 = 2;
const OP_DELETION: u8 = 3;

/// Longest k-mer that fits the 2-bit packed `u64` representation.
const MAX_KMER_LEN: usize = 32;

/// Score cap for the CPU fallback aligner.
const CPU_MAX_SCORE: i32 = 4096;

/// Sentinel for an unset wavefront offset (matches the kernel's EMPTY).
const WF_EMPTY: i32 = -2;

/// A sequence record with ID, sequence, and optional header description
#[derive(Debug, Clone)]
struct SequenceRecord {
    id: String,
    desc: Option<String>,
    seq: Vec<u8>,
}

/// Enum representing the format of a sequence file
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequenceFormat {
    Fasta,
    Fastq,
    Unknown,
}

fn main() -> Result<()> {
    let args = Args::parse();
    setup_logging(args.verbose);

    info!("Starting BLEP (GPU-accelerated WFA alignment)");
    info!("Max read length: {}, Max ref length: {}, Max score: {}",
          args.max_read_length, args.max_ref_length, args.max_score);
    info!("Reverse complement alignment: enabled (always-on)");

    if args.kmer_length > MAX_KMER_LEN {
        return Err(anyhow!("--kmer-length must be <= {} (got {})", MAX_KMER_LEN, args.kmer_length));
    }
    let kmer_filter_enabled = args.kmer_length > 0;
    if kmer_filter_enabled {
        info!("K-mer pre-filter enabled: k={}, threshold={:.2}",
              args.kmer_length, args.kmer_threshold);
    }

    let dedup = !args.no_dedup;
    let collapse = args.collapse;
    if dedup {
        info!("Read deduplication: enabled (identical sequences within a batch are aligned once)");
    } else {
        info!("Read deduplication: disabled (--no-dedup)");
    }
    if collapse {
        info!("Output: collapsed to one row per (sequence, header tags) group with an n_reads column");
    }

    let header_tags = args.header_tags.clone();
    if !header_tags.is_empty() {
        info!("Header tags to extract: {:?}", header_tags);
    }

    // Determine which CUDA devices to use: an explicit --devices list, or every
    // available device when the flag is omitted.
    let device_ids: Vec<usize> = match &args.devices {
        Some(list) if !list.is_empty() => list.clone(),
        _ => {
            let count = CudaContext::device_count()
                .context("Failed to query CUDA device count")?;
            if count <= 0 {
                return Err(anyhow!("No CUDA devices available"));
            }
            (0..count as usize).collect()
        }
    };
    let num_devices = device_ids.len();
    info!("Using {} CUDA device(s): {:?}", num_devices, device_ids);

    // PTX kernel source (compiled at build time by build.rs). This is a
    // &'static str; each worker loads it into a module on its own device.
    let ptx_src: &'static str = include_str!(env!("WFA_PTX_PATH"));

    // Read reference sequences (shared read-only across all GPU workers).
    info!("Reading reference sequences from {}", args.references.display());
    let references = read_sequences_fully(&args.references)?;
    if references.is_empty() {
        return Err(anyhow!("No reference sequences found in {}", args.references.display()));
    }
    info!("Loaded {} reference sequences", references.len());

    // The GPU workspace is sized from --max-ref-length; a longer reference can
    // never be aligned, so refuse it up front rather than silently failing.
    if let Some(longest) = references.iter().max_by_key(|r| r.seq.len()) {
        if longest.seq.len() > args.max_ref_length {
            return Err(anyhow!(
                "Reference '{}' is {} bp, longer than --max-ref-length {}. Raise --max-ref-length.",
                longest.id, longest.seq.len(), args.max_ref_length));
        }
    }

    // Inverted k-mer index over the references (used by the prep threads' pre-filter)
    let kmer_index = KmerIndex::build(&references, if kmer_filter_enabled { args.kmer_length } else { 0 });
    if kmer_filter_enabled {
        info!("K-mer index: {} distinct {}-mers across {} references",
              kmer_index.len(), args.kmer_length, references.len());
    }

    // Pack reference sequences once on the host; each worker uploads its own
    // copy to its device. Shared across worker threads via Arc.
    let (ref_data, ref_lengths, ref_offsets) = pack_sequences(&references);
    let ref_data = Arc::new(ref_data);
    let ref_lengths = Arc::new(ref_lengths.iter().map(|&x| x as i32).collect::<Vec<i32>>());
    let ref_offsets = Arc::new(ref_offsets.iter().map(|&x| x as i32).collect::<Vec<i32>>());

    // Reference records shared read-only with the prep pool (k-mer filter) and
    // the post pool (output formatting + CPU fallback).
    let references_arc = Arc::new(references);
    let kmer_index_arc = Arc::new(kmer_index);

    // Create the output writer and emit the header row up front. The writer is
    // then moved into a single dedicated writer thread (below).
    let mut writer = create_writer(&args.output)?;
    {
        let mut header_fields: Vec<&str> = vec!["read_name", "reference_name", "cigar", "strand"];
        for tag in &header_tags {
            header_fields.push(tag.as_str());
        }
        if collapse {
            header_fields.push("n_reads");
        }
        writer.write_record(&header_fields)?;
    }

    // Pre-compute GPU parameters (identical across devices)
    let max_k = std::cmp::max(args.max_read_length, args.max_ref_length);
    let num_diags = 2 * max_k + 1;
    let max_score = args.max_score;
    let max_cigar_len = args.max_read_length + args.max_ref_length;

    // Per-pair GPU memory footprint. Each worker combines this with its own
    // device's free memory to size sub-batches.
    let ws_per_pair = 4 * (max_score + 1) * num_diags; // ints for workspace
    let ws_bytes_per_pair = ws_per_pair * 4; // 4 bytes per i32
    let cigar_bytes_per_pair = max_cigar_len; // 1 byte per i8
    let output_bytes_per_pair = 4 + cigar_bytes_per_pair + 4; // score(i32) + cigar(i8s) + cigar_len(i32)
    let total_bytes_per_pair = ws_bytes_per_pair + output_bytes_per_pair;

    // Copy the scalar penalties/flags out of `args` so each worker closure can
    // capture them by value (they are `Copy`).
    let gap_open = args.gap_open;
    let gap_extend = args.gap_extend;
    let mismatch = args.mismatch;
    let skip_fail = args.skip_fail;
    let batch_size = args.batch_size;
    let kmer_length = args.kmer_length;
    let kmer_threshold = args.kmer_threshold;
    let max_read_length = args.max_read_length;

    // Decide how many CPU threads to devote to each CPU-bound stage. Preparation
    // (dedup, k-mer filter, reverse-complement, packing) is the heavy stage that
    // tends to starve the GPUs, so it gets most of the cores; post-processing is
    // usually light (only pairs over the score budget hit the CPU aligner).
    let hw_par = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let prep_threads = if args.prep_threads > 0 {
        args.prep_threads
    } else {
        std::cmp::max(2, hw_par.saturating_sub(num_devices + 1))
    };
    let post_threads = if args.post_threads > 0 {
        args.post_threads
    } else {
        std::cmp::max(2, num_devices)
    };

    // ================================================================
    // Pipeline: reader -> prep pool -> GPU workers -> post pool -> writer
    //
    //   reader (1)      : reads raw batches from disk
    //   prep pool (P)   : dedup, reverse-complement, pack, k-mer pre-filter
    //   GPU workers (N) : upload -> WFA kernel -> download  (GPU work only)
    //   post pool (Q)   : CPU fallback + output row formatting (fan-out to reads)
    //   writer (1)      : serializes rows to the output TSV
    //
    // Each stage is decoupled by a bounded channel so a GPU never has to read
    // from disk, run the CPU fallback, format rows, or wait on the writer.
    // ================================================================

    info!("Starting alignment of reads from {}", args.reads.display());
    info!("Pipeline threads: 1 reader, {} prep, {} GPU worker(s), {} post, 1 writer (hw parallelism: {})",
          prep_threads, num_devices, post_threads, hw_par);

    // Stage 1 -> 2: raw batches from the reader to the prep pool.
    let (raw_tx, raw_rx) = mpsc::sync_channel::<Vec<SequenceRecord>>(prep_threads * 2);
    let raw_rx = Arc::new(Mutex::new(raw_rx));

    // Stage 2 -> 3: prepared batches from the prep pool to the GPU workers.
    let (prep_tx, prep_rx) = mpsc::sync_channel::<PreparedBatch>(num_devices * 2);
    let prep_rx = Arc::new(Mutex::new(prep_rx));

    // Stage 3 -> 4: GPU results from the workers to the post-processing pool.
    let (post_tx, post_rx) = mpsc::sync_channel::<GpuBatchResult>(num_devices * 2);
    let post_rx = Arc::new(Mutex::new(post_rx));

    // Stage 4 -> 5: formatted rows from the post pool to the single writer.
    let (rows_tx, rows_rx) = mpsc::sync_channel::<Vec<Vec<String>>>(post_threads * 2);

    let pipeline_start = Instant::now();

    // --- Stage 1: reader thread ---
    let reads_path = args.reads.clone();
    let reader_handle = thread::spawn(move || -> Result<StageStats> {
        let mut reader = create_sequence_reader(&reads_path)?;
        let mut stats = StageStats::default();
        loop {
            let t0 = Instant::now();
            let batch = reader.read_batch(batch_size)?;
            stats.busy += t0.elapsed();
            if batch.is_empty() {
                break;
            }
            stats.batches += 1;
            let t1 = Instant::now();
            if raw_tx.send(batch).is_err() {
                // Downstream is gone; nothing left to consume raw batches.
                break;
            }
            stats.send_wait += t1.elapsed();
        }
        Ok(stats)
    });

    // --- Stage 2: prep pool ---
    let mut prep_handles = Vec::with_capacity(prep_threads);
    for _ in 0..prep_threads {
        let raw_rx = Arc::clone(&raw_rx);
        let prep_tx = prep_tx.clone();
        let references_arc = Arc::clone(&references_arc);
        let kmer_index_arc = Arc::clone(&kmer_index_arc);
        let handle = thread::spawn(move || -> Result<StageStats> {
            let mut stats = StageStats::default();
            loop {
                let t0 = Instant::now();
                let batch = {
                    let guard = raw_rx.lock().expect("raw batch channel mutex poisoned");
                    guard.recv()
                };
                stats.recv_wait += t0.elapsed();
                let batch = match batch {
                    Ok(b) => b,
                    Err(_) => break, // reader finished and closed the channel
                };

                let t1 = Instant::now();
                let prepared = prepare_batch(
                    batch,
                    &references_arc,
                    &kmer_index_arc,
                    kmer_filter_enabled,
                    kmer_length,
                    kmer_threshold,
                    dedup,
                    max_read_length,
                );
                stats.busy += t1.elapsed();
                stats.batches += 1;

                let t2 = Instant::now();
                if prep_tx.send(prepared).is_err() {
                    break; // GPU workers all exited
                }
                stats.send_wait += t2.elapsed();
            }
            Ok(stats)
        });
        prep_handles.push(handle);
    }
    // Only the prep threads hold senders now; drop main's so the prepared
    // channel closes once every prep thread finishes.
    drop(prep_tx);

    // --- Stage 3: GPU worker threads (one per device) ---
    let mut worker_handles = Vec::with_capacity(num_devices);
    for &device_id in &device_ids {
        let prep_rx = Arc::clone(&prep_rx);
        let post_tx = post_tx.clone();
        let ref_data = Arc::clone(&ref_data);
        let ref_lengths = Arc::clone(&ref_lengths);
        let ref_offsets = Arc::clone(&ref_offsets);

        let handle = thread::spawn(move || -> Result<WorkerStats> {
            // Initialize this device's CUDA context (binds it to this thread).
            let ctx = CudaContext::new(device_id)
                .with_context(|| format!("Failed to initialize CUDA device {}", device_id))?;
            ctx.bind_to_thread()
                .with_context(|| format!("Failed to bind CUDA device {}", device_id))?;
            let stream = ctx.default_stream();
            info!("CUDA device {} initialized", device_id);

            // Load the WFA kernel into this device's context.
            let module = ctx.load_module(Ptx::from_src(ptx_src))
                .context("Failed to load WFA PTX module")?;
            let kernel = module.load_function("wfa_align_kernel")
                .context("Failed to load wfa_align_kernel function")?;

            // Upload reference sequences to this device (persistent across batches).
            let d_refs = stream.memcpy_stod(ref_data.as_slice())?;
            let d_ref_lengths = stream.memcpy_stod(ref_lengths.as_slice())?;
            let d_ref_offsets = stream.memcpy_stod(ref_offsets.as_slice())?;

            // Query this device's free memory and size GPU sub-batches at 75%.
            let (free_mem, total_mem) = cudarc::driver::result::mem_get_info()
                .context("Failed to query GPU memory")?;
            let usable_mem = (free_mem as f64 * 0.75) as usize;
            let max_gpu_pairs = if total_bytes_per_pair > 0 {
                std::cmp::max(1, usable_mem / total_bytes_per_pair)
            } else {
                batch_size
            };
            info!("Device {}: {:.0} MB free / {:.0} MB total, {:.1} KB/pair, max {} pairs per GPU sub-batch",
                  device_id, free_mem as f64 / 1048576.0, total_mem as f64 / 1048576.0,
                  total_bytes_per_pair as f64 / 1024.0, max_gpu_pairs);

            // One wavefront workspace per worker, sized for the largest
            // sub-batch. The kernel initialises rows lazily, so the buffer is
            // allocated once and reused across sub-batches without clearing.
            let mut d_workspace: CudaSlice<i32> = stream
                .alloc_zeros(max_gpu_pairs * ws_per_pair)
                .context("Failed to allocate GPU workspace")?;

            let mut stats = WorkerStats::default();

            loop {
                // Pull the next prepared batch from the shared queue. The lock is
                // held only across recv(); when the prep pool finishes and drops
                // its senders, recv() returns Err and this worker exits.
                let t0 = Instant::now();
                let prepared = {
                    let guard = prep_rx.lock().expect("prepared batch channel mutex poisoned");
                    guard.recv()
                };
                stats.recv_wait += t0.elapsed();
                let prepared = match prepared {
                    Ok(p) => p,
                    Err(_) => break,
                };

                stats.batches += 1;
                stats.reads += prepared.num_reads;
                stats.unique += prepared.num_unique;
                let num_gpu_pairs = prepared.pair_query_idx.len();

                info!("Device {}: aligning batch of {} reads ({} unique, {:.1}x), {} GPU pairs ({} both-filtered)",
                      device_id, prepared.num_reads, prepared.num_unique,
                      dedup_ratio(prepared.num_reads, prepared.num_unique),
                      num_gpu_pairs, prepared.kmer_fail_count);

                // GPU work only: upload, launch, synchronize, download.
                let t1 = Instant::now();
                let (scores, cigar_ops, cigar_lengths) = run_gpu_alignment(
                    &stream, &kernel, &prepared, &mut d_workspace,
                    &d_refs, &d_ref_lengths, &d_ref_offsets,
                    gap_open, gap_extend, mismatch,
                    max_k, num_diags, max_score, max_cigar_len, max_gpu_pairs,
                )?;
                stats.gpu_busy += t1.elapsed();

                // Hand the raw results to the post pool and immediately loop back
                // for the next batch; formatting and CPU fallback happen off this
                // thread so the GPU is not left idle.
                let t2 = Instant::now();
                let result = GpuBatchResult { prepared, scores, cigar_ops, cigar_lengths };
                if post_tx.send(result).is_err() {
                    break; // post pool all exited
                }
                stats.send_wait += t2.elapsed();
            }

            Ok(stats)
        });
        worker_handles.push(handle);
    }
    // Only the GPU workers hold post senders now.
    drop(post_tx);

    // --- Stage 4: post-processing pool ---
    let mut post_handles = Vec::with_capacity(post_threads);
    for _ in 0..post_threads {
        let post_rx = Arc::clone(&post_rx);
        let rows_tx = rows_tx.clone();
        let references_arc = Arc::clone(&references_arc);
        let header_tags = header_tags.clone();
        let handle = thread::spawn(move || -> Result<(StageStats, PostTotals)> {
            let mut stats = StageStats::default();
            let mut totals = PostTotals::default();
            loop {
                let t0 = Instant::now();
                let result = {
                    let guard = post_rx.lock().expect("post channel mutex poisoned");
                    guard.recv()
                };
                stats.recv_wait += t0.elapsed();
                let result = match result {
                    Ok(r) => r,
                    Err(_) => break,
                };

                let t1 = Instant::now();
                let outcome = format_batch_rows(
                    &result,
                    references_arc.as_slice(),
                    &header_tags,
                    gap_open, gap_extend, mismatch, max_k,
                    skip_fail, collapse,
                );
                stats.busy += t1.elapsed();
                stats.batches += 1;
                totals.absorb(&outcome);

                let t2 = Instant::now();
                if rows_tx.send(outcome.rows).is_err() {
                    break; // writer exited
                }
                stats.send_wait += t2.elapsed();
            }
            Ok((stats, totals))
        });
        post_handles.push(handle);
    }
    // Only the post threads hold row senders now.
    drop(rows_tx);

    // --- Stage 5: writer thread ---
    let writer_handle = thread::spawn(move || -> Result<StageStats> {
        let mut writer = writer;
        let mut stats = StageStats::default();
        // Flush periodically so output appears incrementally without paying a
        // syscall per batch. This thread is off the GPU critical path.
        let mut since_flush = 0u64;
        loop {
            let t0 = Instant::now();
            let rows = rows_rx.recv();
            stats.recv_wait += t0.elapsed();
            let rows = match rows {
                Ok(r) => r,
                Err(_) => break,
            };
            let t1 = Instant::now();
            for row in &rows {
                writer.write_record(row)?;
            }
            since_flush += 1;
            if since_flush >= 32 {
                writer.flush()?;
                since_flush = 0;
            }
            stats.busy += t1.elapsed();
            stats.batches += 1;
        }
        let t2 = Instant::now();
        writer.flush()?;
        stats.busy += t2.elapsed();
        Ok(stats)
    });

    // ---- Collect results and per-stage timing from every thread ----
    let mut gpu = WorkerStats::default();
    for (i, handle) in worker_handles.into_iter().enumerate() {
        let stats = handle.join()
            .map_err(|e| anyhow!("GPU worker thread {} panicked: {:?}", i, e))?
            .with_context(|| format!("GPU worker for device {} failed", device_ids[i]))?;
        gpu.batches += stats.batches;
        gpu.reads += stats.reads;
        gpu.unique += stats.unique;
        gpu.gpu_busy += stats.gpu_busy;
        gpu.recv_wait += stats.recv_wait;
        gpu.send_wait += stats.send_wait;
    }

    let mut prep = StageStats::default();
    for (i, handle) in prep_handles.into_iter().enumerate() {
        let s = handle.join()
            .map_err(|e| anyhow!("prep thread {} panicked: {:?}", i, e))?
            .context("prep thread failed")?;
        prep.merge(&s);
    }

    let mut post = StageStats::default();
    let mut post_totals = PostTotals::default();
    for (i, handle) in post_handles.into_iter().enumerate() {
        let (s, t) = handle.join()
            .map_err(|e| anyhow!("post thread {} panicked: {:?}", i, e))?
            .context("post thread failed")?;
        post.merge(&s);
        post_totals.merge(&t);
    }

    let reader_stats = reader_handle.join()
        .map_err(|e| anyhow!("reader thread panicked: {:?}", e))?
        .context("reader thread failed")?;

    let writer_stats = writer_handle.join()
        .map_err(|e| anyhow!("writer thread panicked: {:?}", e))?
        .context("writer thread failed")?;

    let wall = pipeline_start.elapsed();

    if post_totals.cpu_fallbacks > 0 {
        info!("CPU fallback used for {} pairs total", post_totals.cpu_fallbacks);
    }
    if post_totals.too_long_reads > 0 {
        warn!("{} reads were longer than --max-read-length {} (longest: {} bp) and were reported with cigar '*'. \
               Raise --max-read-length to align them.",
              post_totals.too_long_reads, max_read_length, post_totals.longest_read);
    }
    info!("Alignment completed. Processed {} reads ({} unique sequences aligned, {:.2}x deduplication) in {} batches across {} device(s)",
          gpu.reads, gpu.unique, dedup_ratio(gpu.reads, gpu.unique), gpu.batches, num_devices);

    log_pipeline_timing(
        wall, &reader_stats, &prep, &gpu, &post, &writer_stats,
        prep_threads, num_devices, post_threads,
    );

    Ok(())
}

/// Reads-per-unique-sequence ratio for log lines (1.0 when nothing was processed).
fn dedup_ratio(reads: usize, unique: usize) -> f64 {
    if unique == 0 { 1.0 } else { reads as f64 / unique as f64 }
}

/// A batch that has been read from disk, deduplicated and pre-filtered by a
/// prep thread, ready for GPU alignment.
///
/// Alignment work is organised around *unique sequences*: every read maps to
/// one unique sequence via `read_to_uidx`, and all per-pair arrays below are
/// indexed by `ui * num_refs + ri` where `ui` is a unique-sequence index.
struct PreparedBatch {
    /// The reads of this batch in input order (names, descriptions, sequences).
    batch: Vec<SequenceRecord>,
    /// For each unique sequence, the index into `batch` of its first read.
    unique_rep: Vec<usize>,
    /// For each read, the index of its unique sequence.
    read_to_uidx: Vec<usize>,
    /// Per unique sequence: longer than --max-read-length, so never aligned.
    too_long: Vec<bool>,
    /// Packed unique sequences (forward) for GPU upload.
    query_data: Vec<i8>,
    query_lengths: Vec<i32>,
    query_offsets: Vec<i32>,
    /// Packed unique sequences (reverse complement) for GPU upload.
    rc_query_data: Vec<i8>,
    rc_query_lengths: Vec<i32>,
    rc_query_offsets: Vec<i32>,
    /// Per (unique, ref) pair: whether that orientation is aligned on the GPU.
    fwd_kmer_pass: Vec<bool>,
    rc_kmer_pass: Vec<bool>,
    /// Number of (unique, ref) pairs rejected by the k-mer filter in both orientations.
    kmer_fail_count: usize,
    /// GPU pair list: unique-sequence index, reference index, orientation.
    pair_query_idx: Vec<i32>,
    pair_ref_idx: Vec<i32>,
    pair_is_rc: Vec<bool>,
    /// Per (unique, ref) pair: position in the GPU pair list for each orientation.
    fwd_gpu_idx: Vec<Option<usize>>,
    rc_gpu_idx: Vec<Option<usize>>,
    num_reads: usize,
    num_unique: usize,
    num_refs: usize,
}

/// GPU-computed alignment results for one batch, awaiting CPU-side formatting
/// (and, for pairs over the score budget, CPU fallback). Produced by a GPU
/// worker, consumed by a post-processing thread.
struct GpuBatchResult {
    prepared: PreparedBatch,
    scores: Vec<i32>,
    cigar_ops: Vec<Vec<i8>>,
    cigar_lengths: Vec<i32>,
}

/// Per-GPU-worker counters, aggregated after all GPU threads finish. The three
/// durations partition each worker's wall time: `gpu_busy` doing GPU work,
/// `recv_wait` blocked waiting for a batch (upstream starvation), and
/// `send_wait` blocked handing results downstream (downstream backpressure).
#[derive(Default, Clone)]
struct WorkerStats {
    batches: u64,
    reads: usize,
    unique: usize,
    gpu_busy: Duration,
    recv_wait: Duration,
    send_wait: Duration,
}

/// Wall-time breakdown accumulated by one CPU pipeline stage. `busy` is time
/// doing useful work; `recv_wait` is time blocked waiting for input (upstream
/// can't keep up); `send_wait` is time blocked handing output downstream
/// (downstream can't keep up).
#[derive(Default, Clone)]
struct StageStats {
    busy: Duration,
    recv_wait: Duration,
    send_wait: Duration,
    batches: u64,
}

impl StageStats {
    fn merge(&mut self, other: &StageStats) {
        self.busy += other.busy;
        self.recv_wait += other.recv_wait;
        self.send_wait += other.send_wait;
        self.batches += other.batches;
    }
}

/// Counters produced by the post-processing stage for the end-of-run summary.
#[derive(Default, Clone)]
struct PostTotals {
    cpu_fallbacks: usize,
    too_long_reads: usize,
    longest_read: usize,
}

impl PostTotals {
    fn absorb(&mut self, outcome: &PostBatchOutcome) {
        self.cpu_fallbacks += outcome.cpu_fallbacks;
        self.too_long_reads += outcome.too_long_reads;
        self.longest_read = self.longest_read.max(outcome.longest_read);
    }

    fn merge(&mut self, other: &PostTotals) {
        self.cpu_fallbacks += other.cpu_fallbacks;
        self.too_long_reads += other.too_long_reads;
        self.longest_read = self.longest_read.max(other.longest_read);
    }
}

/// Output rows for one batch plus the counters the summary needs.
struct PostBatchOutcome {
    rows: Vec<Vec<String>>,
    cpu_fallbacks: usize,
    too_long_reads: usize,
    longest_read: usize,
}

/// The resolved outcome of one (unique sequence, reference) pair.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PairCall {
    /// The k-mer filter rejected both orientations.
    Failed,
    /// The read exceeds --max-read-length and was not aligned.
    TooLong,
    /// Neither orientation could be aligned (GPU and CPU both gave up).
    Unaligned,
    /// Best orientation's alignment.
    Aligned { cigar: String, strand: &'static str },
}

/// Results from a GPU kernel launch
struct KernelResults {
    scores: Vec<i32>,
    cigar_ops: Vec<Vec<i8>>,
    cigar_lengths: Vec<i32>,
}

/// Log the per-stage timing breakdown and a one-line diagnosis of where the
/// GPUs spend their time. The `busy`/`recv-wait`/`send-wait` split for the GPU
/// stage answers "are the GPUs starving, blocked, or actually working?".
fn log_pipeline_timing(
    wall: Duration,
    reader: &StageStats,
    prep: &StageStats,
    gpu: &WorkerStats,
    post: &StageStats,
    writer: &StageStats,
    prep_threads: usize,
    num_devices: usize,
    post_threads: usize,
) {
    let s = |d: Duration| d.as_secs_f64();

    info!("=== Pipeline timing (wall {:.1}s | 1 reader, {} prep, {} GPU, {} post, 1 writer) ===",
          s(wall), prep_threads, num_devices, post_threads);
    info!("  reader : busy {:.1}s  send-wait {:.1}s  ({} batches)",
          s(reader.busy), s(reader.send_wait), reader.batches);
    info!("  prep   : busy {:.1}s  recv-wait {:.1}s  send-wait {:.1}s  (summed over {} threads)",
          s(prep.busy), s(prep.recv_wait), s(prep.send_wait), prep_threads);
    info!("  gpu    : busy {:.1}s  recv-wait {:.1}s  send-wait {:.1}s  (summed over {} threads)",
          s(gpu.gpu_busy), s(gpu.recv_wait), s(gpu.send_wait), num_devices);
    info!("  post   : busy {:.1}s  recv-wait {:.1}s  send-wait {:.1}s  (summed over {} threads)",
          s(post.busy), s(post.recv_wait), s(post.send_wait), post_threads);
    info!("  writer : busy {:.1}s  recv-wait {:.1}s",
          s(writer.busy), s(writer.recv_wait));

    let gpu_total = gpu.gpu_busy + gpu.recv_wait + gpu.send_wait;
    if gpu_total.as_secs_f64() > 0.0 {
        let busy = 100.0 * s(gpu.gpu_busy) / s(gpu_total);
        let starve = 100.0 * s(gpu.recv_wait) / s(gpu_total);
        let backpr = 100.0 * s(gpu.send_wait) / s(gpu_total);
        info!("  GPU time split: {:.0}% aligning, {:.0}% starved (waiting for input), {:.0}% blocked (waiting on post/writer)",
              busy, starve, backpr);
        if starve > 20.0 {
            warn!("GPUs starved {:.0}% of the time — the reader/prep stages can't keep up. \
                   Try raising --prep-threads (now {}) or --batch-size, or pre-decompressing the input.",
                  starve, prep_threads);
        } else if backpr > 20.0 {
            warn!("GPUs blocked {:.0}% of the time waiting on downstream — post/writer can't keep up. \
                   Try raising --post-threads (now {}).",
                  backpr, post_threads);
        }
    }
}

// ============================================================
// Batch preparation (runs on the prep pool)
// ============================================================

/// Map every read of a batch to a unique-sequence index.
///
/// Returns `(unique_rep, read_to_uidx)`: `unique_rep[ui]` is the index of the
/// first read carrying unique sequence `ui` (so unique sequences are numbered in
/// first-seen order), and `read_to_uidx[qi]` is the unique index of read `qi`.
/// With `dedup == false` both are the identity mapping.
fn dedup_reads(batch: &[SequenceRecord], dedup: bool) -> (Vec<usize>, Vec<usize>) {
    let n = batch.len();
    if !dedup {
        return ((0..n).collect(), (0..n).collect());
    }
    let mut seq_index: HashMap<&[u8], usize> = HashMap::with_capacity(n);
    let mut unique_rep: Vec<usize> = Vec::new();
    let mut read_to_uidx: Vec<usize> = Vec::with_capacity(n);
    for (qi, rec) in batch.iter().enumerate() {
        let ui = *seq_index.entry(rec.seq.as_slice()).or_insert_with(|| {
            unique_rep.push(qi);
            unique_rep.len() - 1
        });
        read_to_uidx.push(ui);
    }
    (unique_rep, read_to_uidx)
}

/// Turn a raw batch of reads into a `PreparedBatch`: deduplicate the sequences,
/// generate reverse complements, pack both orientations for GPU upload, run the
/// k-mer pre-filter, and build the per-pair index lists. This is the CPU-heavy
/// work that runs in parallel across the prep pool so it does not starve the
/// GPUs.
fn prepare_batch(
    batch: Vec<SequenceRecord>,
    references: &[SequenceRecord],
    kmer_index: &KmerIndex,
    kmer_filter_enabled: bool,
    kmer_length: usize,
    kmer_threshold: f64,
    dedup: bool,
    max_read_length: usize,
) -> PreparedBatch {
    let num_reads = batch.len();
    let num_refs = references.len();

    // Deduplicate: every read maps to a unique sequence; all alignment work
    // below is per unique sequence.
    let (unique_rep, read_to_uidx) = dedup_reads(&batch, dedup);
    let num_unique = unique_rep.len();
    let num_pairs = num_unique * num_refs;

    let unique_seqs: Vec<&[u8]> = unique_rep.iter().map(|&qi| batch[qi].seq.as_slice()).collect();
    let too_long: Vec<bool> = unique_seqs.iter().map(|s| s.len() > max_read_length).collect();

    // Generate reverse complement sequences for each unique sequence
    let rc_seqs: Vec<Vec<u8>> = unique_seqs.iter().map(|s| reverse_complement(s)).collect();
    let rc_slices: Vec<&[u8]> = rc_seqs.iter().map(|s| s.as_slice()).collect();

    // Pack forward and reverse-complement unique sequences for GPU upload,
    // converting lengths/offsets to i32 here so the GPU worker uploads directly.
    let (query_data, query_lengths_us, query_offsets_us) = pack_sequences_raw(&unique_seqs);
    let (rc_query_data, rc_query_lengths_us, rc_query_offsets_us) = pack_sequences_raw(&rc_slices);
    let query_lengths: Vec<i32> = query_lengths_us.iter().map(|&x| x as i32).collect();
    let query_offsets: Vec<i32> = query_offsets_us.iter().map(|&x| x as i32).collect();
    let rc_query_lengths: Vec<i32> = rc_query_lengths_us.iter().map(|&x| x as i32).collect();
    let rc_query_offsets: Vec<i32> = rc_query_offsets_us.iter().map(|&x| x as i32).collect();

    // Per-pair orientation eligibility: start from "align both orientations",
    // then drop over-length sequences and k-mer filter failures.
    let mut fwd_kmer_pass = vec![true; num_pairs];
    let mut rc_kmer_pass = vec![true; num_pairs];
    let mut kmer_fail_count = 0usize;
    // Scratch: shared k-mer count per reference for the sequence being filtered.
    let mut shared = vec![0u32; num_refs];

    for ui in 0..num_unique {
        if too_long[ui] {
            for ri in 0..num_refs {
                let global_idx = ui * num_refs + ri;
                fwd_kmer_pass[global_idx] = false;
                rc_kmer_pass[global_idx] = false;
            }
            continue;
        }
        if !kmer_filter_enabled {
            continue;
        }

        // One index lookup per read k-mer yields the shared count for every
        // reference at once; the pass test per reference is then a division.
        let fwd_kmers = collect_kmers_packed(unique_seqs[ui], kmer_length);
        kmer_index.count_shared(&fwd_kmers, &mut shared);
        for ri in 0..num_refs {
            if !kmer_fraction_passes(shared[ri] as usize, fwd_kmers.len(), kmer_threshold) {
                fwd_kmer_pass[ui * num_refs + ri] = false;
            }
        }

        let rc_kmers = collect_kmers_packed(&rc_seqs[ui], kmer_length);
        kmer_index.count_shared(&rc_kmers, &mut shared);
        for ri in 0..num_refs {
            let global_idx = ui * num_refs + ri;
            if !kmer_fraction_passes(shared[ri] as usize, rc_kmers.len(), kmer_threshold) {
                rc_kmer_pass[global_idx] = false;
            }
            // Count pairs where BOTH orientations fail
            if !fwd_kmer_pass[global_idx] && !rc_kmer_pass[global_idx] {
                kmer_fail_count += 1;
            }
        }
    }

    // Build pair indices for passing pairs (both orientations)
    let mut pair_query_idx = Vec::with_capacity(num_pairs * 2);
    let mut pair_ref_idx = Vec::with_capacity(num_pairs * 2);
    let mut pair_is_rc = Vec::with_capacity(num_pairs * 2);

    // Forward pairs
    let mut fwd_gpu_idx: Vec<Option<usize>> = vec![None; num_pairs];
    for ui in 0..num_unique {
        for ri in 0..num_refs {
            let global_idx = ui * num_refs + ri;
            if fwd_kmer_pass[global_idx] {
                fwd_gpu_idx[global_idx] = Some(pair_query_idx.len());
                pair_query_idx.push(ui as i32);
                pair_ref_idx.push(ri as i32);
                pair_is_rc.push(false);
            }
        }
    }

    // RC pairs
    let mut rc_gpu_idx: Vec<Option<usize>> = vec![None; num_pairs];
    for ui in 0..num_unique {
        for ri in 0..num_refs {
            let global_idx = ui * num_refs + ri;
            if rc_kmer_pass[global_idx] {
                rc_gpu_idx[global_idx] = Some(pair_query_idx.len());
                pair_query_idx.push(ui as i32);
                pair_ref_idx.push(ri as i32);
                pair_is_rc.push(true);
            }
        }
    }

    PreparedBatch {
        batch,
        unique_rep,
        read_to_uidx,
        too_long,
        query_data,
        query_lengths,
        query_offsets,
        rc_query_data,
        rc_query_lengths,
        rc_query_offsets,
        fwd_kmer_pass,
        rc_kmer_pass,
        kmer_fail_count,
        pair_query_idx,
        pair_ref_idx,
        pair_is_rc,
        fwd_gpu_idx,
        rc_gpu_idx,
        num_reads,
        num_unique,
        num_refs,
    }
}

// ============================================================
// GPU alignment (runs on the GPU worker threads)
// ============================================================

/// Run GPU alignment for one prepared batch on the given device stream/kernel.
/// Returns the per-GPU-pair scores, CIGAR ops, and CIGAR lengths (indexed by the
/// pair's position in `prepared.pair_*`). This performs only GPU work (upload,
/// kernel launch, synchronize, download); output formatting and CPU fallback are
/// done later on a post-processing thread.
///
/// This is invoked from each GPU worker thread; `stream`, `kernel`,
/// `d_workspace` and the `d_ref*` slices all belong to that worker's device
/// context.
fn run_gpu_alignment(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernel: &cudarc::driver::CudaFunction,
    prepared: &PreparedBatch,
    d_workspace: &mut CudaSlice<i32>,
    d_refs: &CudaSlice<i8>,
    d_ref_lengths: &CudaSlice<i32>,
    d_ref_offsets: &CudaSlice<i32>,
    gap_open: i32,
    gap_extend: i32,
    mismatch: i32,
    max_k: usize,
    num_diags: usize,
    max_score: usize,
    max_cigar_len: usize,
    max_gpu_pairs: usize,
) -> Result<(Vec<i32>, Vec<Vec<i8>>, Vec<i32>)> {
    let num_gpu_pairs = prepared.pair_query_idx.len();

    // Upload forward query data to GPU (lengths/offsets already i32)
    let d_fwd_queries = stream.memcpy_stod(&prepared.query_data)?;
    let d_fwd_query_lengths = stream.memcpy_stod(&prepared.query_lengths)?;
    let d_fwd_query_offsets = stream.memcpy_stod(&prepared.query_offsets)?;

    // Upload RC query data to GPU
    let d_rc_queries = stream.memcpy_stod(&prepared.rc_query_data)?;
    let d_rc_query_lengths = stream.memcpy_stod(&prepared.rc_query_lengths)?;
    let d_rc_query_offsets = stream.memcpy_stod(&prepared.rc_query_offsets)?;

    // Collect GPU results
    let mut all_scores = vec![0i32; num_gpu_pairs];
    let mut all_cigar_ops: Vec<Vec<i8>> = vec![Vec::new(); num_gpu_pairs];
    let mut all_cigar_lengths = vec![0i32; num_gpu_pairs];

    // Split GPU pairs into forward and RC sub-groups for separate kernel launches
    let mut fwd_pair_indices: Vec<usize> = Vec::new();
    let mut rc_pair_indices: Vec<usize> = Vec::new();
    for (i, &is_rc) in prepared.pair_is_rc.iter().enumerate() {
        if is_rc {
            rc_pair_indices.push(i);
        } else {
            fwd_pair_indices.push(i);
        }
    }

    // --- Launch forward orientation alignments ---
    if !fwd_pair_indices.is_empty() {
        let fwd_query_idx: Vec<i32> = fwd_pair_indices.iter()
            .map(|&i| prepared.pair_query_idx[i])
            .collect();
        let fwd_ref_idx: Vec<i32> = fwd_pair_indices.iter()
            .map(|&i| prepared.pair_ref_idx[i])
            .collect();

        let results = launch_alignment_kernel(
            stream, kernel, d_workspace,
            &d_fwd_queries, &d_fwd_query_lengths, &d_fwd_query_offsets,
            d_refs, d_ref_lengths, d_ref_offsets,
            &fwd_query_idx, &fwd_ref_idx,
            gap_open, gap_extend, mismatch,
            max_k, num_diags, max_score, max_cigar_len, max_gpu_pairs,
        )?;

        for (local_idx, &global_gpu_idx) in fwd_pair_indices.iter().enumerate() {
            all_scores[global_gpu_idx] = results.scores[local_idx];
            all_cigar_lengths[global_gpu_idx] = results.cigar_lengths[local_idx];
            all_cigar_ops[global_gpu_idx] = results.cigar_ops[local_idx].clone();
        }
    }

    // --- Launch reverse complement orientation alignments ---
    if !rc_pair_indices.is_empty() {
        let rc_query_idx: Vec<i32> = rc_pair_indices.iter()
            .map(|&i| prepared.pair_query_idx[i])
            .collect();
        let rc_ref_idx: Vec<i32> = rc_pair_indices.iter()
            .map(|&i| prepared.pair_ref_idx[i])
            .collect();

        let results = launch_alignment_kernel(
            stream, kernel, d_workspace,
            &d_rc_queries, &d_rc_query_lengths, &d_rc_query_offsets,
            d_refs, d_ref_lengths, d_ref_offsets,
            &rc_query_idx, &rc_ref_idx,
            gap_open, gap_extend, mismatch,
            max_k, num_diags, max_score, max_cigar_len, max_gpu_pairs,
        )?;

        for (local_idx, &global_gpu_idx) in rc_pair_indices.iter().enumerate() {
            all_scores[global_gpu_idx] = results.scores[local_idx];
            all_cigar_lengths[global_gpu_idx] = results.cigar_lengths[local_idx];
            all_cigar_ops[global_gpu_idx] = results.cigar_ops[local_idx].clone();
        }
    }

    Ok((all_scores, all_cigar_ops, all_cigar_lengths))
}

// ============================================================
// Output formatting + CPU fallback (runs on the post pool)
// ============================================================

/// Resolve one orientation of a (unique sequence, reference) pair: take the GPU
/// result if the pair was aligned within the score budget, otherwise re-align on
/// the CPU. `query` is only invoked when the CPU fallback is needed (so the
/// reverse complement is not recomputed for pairs the GPU finished).
/// Returns `(score, cigar)` or `None` if the orientation was filtered out or
/// could not be aligned at all.
fn resolve_orientation<'a, F>(
    passed: bool,
    gpu_idx: Option<usize>,
    result: &GpuBatchResult,
    query: F,
    reference: &[u8],
    gap_open: i32,
    gap_extend: i32,
    mismatch: i32,
    max_k: usize,
    cpu_fallbacks: &mut usize,
    label: &str,
    read_id: &str,
    ref_id: &str,
) -> Option<(i32, String)>
where
    F: FnOnce() -> Cow<'a, [u8]>,
{
    if !passed {
        return None;
    }
    let gpu_idx = gpu_idx.expect("pair that passed the filter must have a GPU index");
    let score = result.scores[gpu_idx];
    let cigar_len = result.cigar_lengths[gpu_idx] as usize;
    if score >= 0 && cigar_len > 0 {
        let ops: Vec<u8> = result.cigar_ops[gpu_idx].iter().map(|&x| x as u8).collect();
        return Some((score, ops_to_cigar(&ops)));
    }
    *cpu_fallbacks += 1;
    debug!("CPU fallback ({}) for read {} vs ref {}", label, read_id, ref_id);
    let q = query();
    cpu_align(&q, reference, gap_open, gap_extend, mismatch, max_k)
}

/// Append one output row for a resolved pair. `n_reads` is `Some` only in
/// collapsed mode and is always the last column, so positional parsers of the
/// leading columns keep working in both modes.
fn push_row(
    rows: &mut Vec<Vec<String>>,
    call: &PairCall,
    read_id: &str,
    ref_id: &str,
    tag_values: &[String],
    n_reads: Option<usize>,
    skip_fail: bool,
) {
    let (cigar, strand): (&str, &str) = match call {
        PairCall::Failed => {
            if skip_fail {
                return;
            }
            ("FAILED", ".")
        }
        PairCall::TooLong | PairCall::Unaligned => ("*", "."),
        PairCall::Aligned { cigar, strand } => (cigar.as_str(), strand),
    };
    let mut row = Vec::with_capacity(5 + tag_values.len());
    row.push(read_id.to_string());
    row.push(ref_id.to_string());
    row.push(cigar.to_string());
    row.push(strand.to_string());
    row.extend(tag_values.iter().cloned());
    if let Some(n) = n_reads {
        row.push(n.to_string());
    }
    rows.push(row);
}

/// Collapsed output mode: a set of *templates* in one batch that are
/// indistinguishable. A template is a run of consecutive reads sharing a name
/// (a single merged read, or the two mates of an unmerged pair); its identity
/// is the ordered list of its members' (unique sequence, tag values).
struct ReadGroup {
    /// Index of the first read of the first template; its name labels every row.
    first_read: usize,
    /// (unique sequence index, tag values) for each member of the template.
    members: Vec<(usize, Vec<String>)>,
    /// How many templates collapsed into this group (the `n_reads` column).
    n_templates: usize,
}

/// Build the TSV output rows for one batch from its GPU results.
///
/// Step 1 resolves every (unique sequence, reference) pair once: pick the best
/// orientation, transparently re-aligning on the CPU any orientation the GPU
/// could not finish within the score budget. Step 2 fans the resolved calls out
/// to the reads: one row per read x reference in input order (rows for a read
/// stay contiguous), or, with `collapse`, one row per group x reference with a
/// trailing `n_reads` column, where a group is a set of identical templates
/// (a merged read, or an unmerged pair sharing one name) from the same cell.
///
/// `max_k` is the kernel's diagonal half-range (`max(max_read_length,
/// max_ref_length)`), passed to the CPU fallback so it explores exactly the
/// same wavefront as the GPU would.
///
/// Runs on a post-processing thread, off the GPU path.
fn format_batch_rows(
    result: &GpuBatchResult,
    references: &[SequenceRecord],
    header_tags: &[String],
    gap_open: i32,
    gap_extend: i32,
    mismatch: i32,
    max_k: usize,
    skip_fail: bool,
    collapse: bool,
) -> PostBatchOutcome {
    let prepared = &result.prepared;
    let num_refs = prepared.num_refs;
    let mut cpu_fallbacks = 0usize;

    // ---- Step 1: resolve each (unique sequence, reference) pair once ----
    let mut calls: Vec<PairCall> = Vec::with_capacity(prepared.num_unique * num_refs);
    for ui in 0..prepared.num_unique {
        let rep = &prepared.batch[prepared.unique_rep[ui]];
        let seq = rep.seq.as_slice();
        for ri in 0..num_refs {
            let global_idx = ui * num_refs + ri;

            if prepared.too_long[ui] {
                calls.push(PairCall::TooLong);
                continue;
            }

            let fwd_passed = prepared.fwd_kmer_pass[global_idx];
            let rc_passed = prepared.rc_kmer_pass[global_idx];

            // If neither orientation passed the k-mer filter, report FAILED
            if !fwd_passed && !rc_passed {
                calls.push(PairCall::Failed);
                continue;
            }

            let fwd_result = resolve_orientation(
                fwd_passed, prepared.fwd_gpu_idx[global_idx], result,
                || Cow::Borrowed(seq), &references[ri].seq,
                gap_open, gap_extend, mismatch, max_k, &mut cpu_fallbacks,
                "fwd", &rep.id, &references[ri].id,
            );
            let rc_result = resolve_orientation(
                rc_passed, prepared.rc_gpu_idx[global_idx], result,
                || Cow::Owned(reverse_complement(seq)), &references[ri].seq,
                gap_open, gap_extend, mismatch, max_k, &mut cpu_fallbacks,
                "rc", &rep.id, &references[ri].id,
            );

            // Pick the best orientation (lower score = better alignment)
            calls.push(match (fwd_result, rc_result) {
                (None, None) => PairCall::Unaligned,
                (Some((_, cigar)), None) => PairCall::Aligned { cigar, strand: "+" },
                (None, Some((_, cigar))) => PairCall::Aligned { cigar, strand: "-" },
                (Some((fwd_score, fwd_cigar)), Some((rc_score, rc_cigar))) => {
                    if fwd_score <= rc_score {
                        PairCall::Aligned { cigar: fwd_cigar, strand: "+" }
                    } else {
                        PairCall::Aligned { cigar: rc_cigar, strand: "-" }
                    }
                }
            });
        }
    }

    // ---- Step 2: fan out to reads ----
    let tag_values_for = |qi: usize| -> Vec<String> {
        header_tags.iter()
            .map(|tag| extract_tag_value(prepared.batch[qi].desc.as_deref(), tag))
            .collect()
    };

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut too_long_reads = 0usize;
    let mut longest_read = 0usize;

    if !collapse {
        for qi in 0..prepared.num_reads {
            let ui = prepared.read_to_uidx[qi];
            if prepared.too_long[ui] {
                too_long_reads += 1;
                longest_read = longest_read.max(prepared.batch[qi].seq.len());
            }
            let tag_values = tag_values_for(qi);
            for ri in 0..num_refs {
                push_row(
                    &mut rows, &calls[ui * num_refs + ri],
                    &prepared.batch[qi].id, &references[ri].id,
                    &tag_values, None, skip_fail,
                );
            }
        }
    } else {
        // Collapsed mode. Consecutive reads sharing a name (unmerged mates, as
        // fastp writes them) form one template. Templates with identical
        // members, i.e. the same ordered (unique sequence, tag values) list,
        // collapse into one group in first-seen order. Every member still gets
        // its own rows, all under the group's first read name, so downstream
        // code can pair mates up again; `n_reads` counts templates.
        let mut group_of: HashMap<Vec<(usize, Vec<String>)>, usize> = HashMap::new();
        let mut groups: Vec<ReadGroup> = Vec::new();
        let mut start = 0;
        while start < prepared.num_reads {
            let mut end = start + 1;
            while end < prepared.num_reads && prepared.batch[end].id == prepared.batch[start].id {
                end += 1;
            }
            let mut members = Vec::with_capacity(end - start);
            for qi in start..end {
                let ui = prepared.read_to_uidx[qi];
                if prepared.too_long[ui] {
                    too_long_reads += 1;
                    longest_read = longest_read.max(prepared.batch[qi].seq.len());
                }
                members.push((ui, tag_values_for(qi)));
            }
            match group_of.entry(members) {
                Entry::Occupied(e) => groups[*e.get()].n_templates += 1,
                Entry::Vacant(e) => {
                    let members = e.key().clone();
                    e.insert(groups.len());
                    groups.push(ReadGroup { first_read: start, members, n_templates: 1 });
                }
            }
            start = end;
        }
        for g in &groups {
            let name = &prepared.batch[g.first_read].id;
            for (ui, tag_values) in &g.members {
                for ri in 0..num_refs {
                    push_row(
                        &mut rows, &calls[ui * num_refs + ri],
                        name, &references[ri].id,
                        tag_values, Some(g.n_templates), skip_fail,
                    );
                }
            }
        }
    }

    PostBatchOutcome { rows, cpu_fallbacks, too_long_reads, longest_read }
}

/// Launch the WFA alignment kernel for a set of pairs, handling sub-batching for
/// GPU memory. `d_workspace` is the worker's persistent wavefront workspace; it
/// must hold at least `max_gpu_pairs` pairs' worth of ints.
fn launch_alignment_kernel(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernel: &cudarc::driver::CudaFunction,
    d_workspace: &mut CudaSlice<i32>,
    d_queries: &CudaSlice<i8>,
    d_query_lengths: &CudaSlice<i32>,
    d_query_offsets: &CudaSlice<i32>,
    d_refs: &CudaSlice<i8>,
    d_ref_lengths: &CudaSlice<i32>,
    d_ref_offsets: &CudaSlice<i32>,
    pair_query_idx: &[i32],
    pair_ref_idx: &[i32],
    gap_open: i32,
    gap_extend: i32,
    mismatch: i32,
    max_k: usize,
    num_diags: usize,
    max_score: usize,
    max_cigar_len: usize,
    max_gpu_pairs: usize,
) -> Result<KernelResults> {
    let num_pairs = pair_query_idx.len();
    let ws_per_pair = 4 * (max_score + 1) * num_diags;

    let mut all_scores = vec![0i32; num_pairs];
    let mut all_cigar_ops: Vec<Vec<i8>> = vec![Vec::new(); num_pairs];
    let mut all_cigar_lengths = vec![0i32; num_pairs];

    let mut pair_offset = 0;
    let mut sub_batch_idx = 0;
    while pair_offset < num_pairs {
        let sub_batch_size = std::cmp::min(max_gpu_pairs, num_pairs - pair_offset);
        sub_batch_idx += 1;

        let sub_pair_query_idx = &pair_query_idx[pair_offset..pair_offset + sub_batch_size];
        let sub_pair_ref_idx = &pair_ref_idx[pair_offset..pair_offset + sub_batch_size];
        let d_pair_query_idx = stream.memcpy_stod(sub_pair_query_idx)?;
        let d_pair_ref_idx = stream.memcpy_stod(sub_pair_ref_idx)?;

        let total_ws = sub_batch_size * ws_per_pair;
        ensure!(total_ws <= d_workspace.len(),
                "GPU workspace too small: {} ints needed for {} pairs, {} allocated",
                total_ws, sub_batch_size, d_workspace.len());
        let ws_mb = (total_ws * 4) / (1024 * 1024);
        debug!("GPU sub-batch {}: {} pairs, workspace {} MB", sub_batch_idx, sub_batch_size, ws_mb);

        let mut d_out_scores: CudaSlice<i32> = stream.alloc_zeros(sub_batch_size)?;
        let total_cigar_bytes = sub_batch_size * max_cigar_len;
        let mut d_out_cigars: CudaSlice<i8> = stream.alloc_zeros(total_cigar_bytes)?;
        let mut d_out_cigar_lengths: CudaSlice<i32> = stream.alloc_zeros(sub_batch_size)?;

        let block_size = 128u32;
        let grid_size = ((sub_batch_size as u32) + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let arg_num_pairs = sub_batch_size as i32;
        let arg_gap_open = gap_open;
        let arg_gap_extend = gap_extend;
        let arg_mismatch = mismatch;
        let arg_max_k = max_k as i32;
        let arg_num_diags = num_diags as i32;
        let arg_max_score = max_score as i32;
        let arg_max_cigar_len = max_cigar_len as i32;

        let mut builder = stream.launch_builder(kernel);
        builder
            .arg(d_queries)
            .arg(d_query_lengths)
            .arg(d_query_offsets)
            .arg(d_refs)
            .arg(d_ref_lengths)
            .arg(d_ref_offsets)
            .arg(&d_pair_query_idx)
            .arg(&d_pair_ref_idx)
            .arg(&arg_num_pairs)
            .arg(&arg_gap_open)
            .arg(&arg_gap_extend)
            .arg(&arg_mismatch)
            .arg(&arg_max_k)
            .arg(&arg_num_diags)
            .arg(&arg_max_score)
            .arg(&mut *d_workspace)
            .arg(&mut d_out_scores)
            .arg(&mut d_out_cigars)
            .arg(&mut d_out_cigar_lengths)
            .arg(&arg_max_cigar_len);

        unsafe { builder.launch(cfg) }
            .context("Failed to launch WFA kernel")?;

        stream.synchronize().context("CUDA synchronize failed")?;

        let sub_scores = stream.memcpy_dtov(&d_out_scores)?;
        let sub_cigar_ops_raw: Vec<i8> = stream.memcpy_dtov(&d_out_cigars)?;
        let sub_cigar_lengths = stream.memcpy_dtov(&d_out_cigar_lengths)?;

        for i in 0..sub_batch_size {
            let idx = pair_offset + i;
            all_scores[idx] = sub_scores[i];
            all_cigar_lengths[idx] = sub_cigar_lengths[i];
            let cigar_len = sub_cigar_lengths[i] as usize;
            let start = i * max_cigar_len;
            all_cigar_ops[idx] = sub_cigar_ops_raw[start..start + cigar_len].to_vec();
        }

        pair_offset += sub_batch_size;
    }

    if sub_batch_idx > 1 {
        debug!("Processed in {} GPU sub-batches", sub_batch_idx);
    }

    Ok(KernelResults {
        scores: all_scores,
        cigar_ops: all_cigar_ops,
        cigar_lengths: all_cigar_lengths,
    })
}

// ============================================================
// Reverse complement
// ============================================================

/// Compute the reverse complement of a DNA sequence. Sequences are uppercased
/// on load, but lowercase is tolerated here for callers that bypass the reader.
fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter().rev().map(|&b| match b {
        b'A' | b'a' => b'T',
        b'T' | b't' => b'A',
        b'C' | b'c' => b'G',
        b'G' | b'g' => b'C',
        b'N' | b'n' => b'N',
        _ => b'N',
    }).collect()
}

// ============================================================
// CIGAR conversion
// ============================================================

fn ops_to_cigar(ops: &[u8]) -> String {
    if ops.is_empty() {
        return String::new();
    }
    let mut cigar = String::new();
    let mut current_op = ops[0];
    let mut count = 1u32;
    for &op in &ops[1..] {
        if op == current_op {
            count += 1;
        } else {
            cigar.push_str(&count.to_string());
            cigar.push(op_to_char(current_op));
            current_op = op;
            count = 1;
        }
    }
    cigar.push_str(&count.to_string());
    cigar.push(op_to_char(current_op));
    cigar
}

fn op_to_char(op: u8) -> char {
    match op {
        OP_MATCH => '=',
        OP_MISMATCH => 'X',
        OP_INSERTION => 'I',
        OP_DELETION => 'D',
        _ => '?',
    }
}

// ============================================================
// CPU fallback WFA alignment
// ============================================================

// Backtrack source component (matches the kernel's COMP_* defines)
const COMP_M: u8 = 0;
const COMP_I: u8 = 1;
const COMP_D: u8 = 2;

/// Marker for a backtrack cell that was never written.
const BT_NONE: u8 = u8::MAX;

/// Backtrack record for one (score, diagonal) cell of the M wavefront.
#[derive(Clone, Copy)]
struct BtEntry {
    prev_s: i32,
    prev_k: i32,
    op: u8,
    src: u8,
}

const BT_UNSET: BtEntry = BtEntry { prev_s: 0, prev_k: 0, op: BT_NONE, src: 0 };

/// Flat, lazily grown wavefront storage for the CPU aligner. Row `s` of each
/// component holds `num_diags` cells indexed by `ki = k + max_k`, mirroring
/// the kernel's layout so both aligners produce identical CIGARs.
struct CpuWavefronts {
    num_diags: usize,
    max_k: i32,
    m: Vec<i32>,
    i: Vec<i32>,
    d: Vec<i32>,
    bt: Vec<BtEntry>,
}

impl CpuWavefronts {
    fn new(max_k: i32) -> Self {
        CpuWavefronts {
            num_diags: (2 * max_k + 1) as usize,
            max_k,
            m: Vec::new(),
            i: Vec::new(),
            d: Vec::new(),
            bt: Vec::new(),
        }
    }

    /// Make sure rows `0..=s` exist (new rows start EMPTY / unset).
    fn ensure_row(&mut self, s: i32) {
        let need = (s as usize + 1) * self.num_diags;
        if self.m.len() < need {
            self.m.resize(need, WF_EMPTY);
            self.i.resize(need, WF_EMPTY);
            self.d.resize(need, WF_EMPTY);
            self.bt.resize(need, BT_UNSET);
        }
    }

    #[inline]
    fn idx(&self, s: i32, ki: i32) -> usize {
        s as usize * self.num_diags + ki as usize
    }

    /// Inclusive `ki` range of diagonals that can be populated at score `s`:
    /// every k-changing transition costs at least `gap_extend`.
    fn band(&self, s: i32, gap_extend: i32) -> (i32, i32) {
        let half = if gap_extend <= 0 { self.max_k } else { (s / gap_extend).min(self.max_k) };
        (self.max_k - half, self.max_k + half)
    }
}

/// Gap-affine WFA alignment on the CPU, used for pairs whose score exceeds the
/// GPU budget. A direct port of the kernel (same wavefront layout, same
/// backtrack rules), with lazily allocated rows and banded loops.
/// `workspace_max_k` is the kernel's `max_k` (`max(max_read_length,
/// max_ref_length)`); using the same diagonal range makes the CPU result
/// identical to what the GPU would compute. Pass 0 to size from the pair alone.
/// Returns `(score, cigar)`, or `None` if no alignment is found within
/// `CPU_MAX_SCORE`.
fn cpu_align(
    query: &[u8],
    reference: &[u8],
    gap_open: i32,
    gap_extend: i32,
    mismatch: i32,
    workspace_max_k: usize,
) -> Option<(i32, String)> {
    let n = query.len() as i32;
    let m = reference.len() as i32;
    let max_k = std::cmp::max(std::cmp::max(n, m), std::cmp::max(workspace_max_k as i32, 1));
    let target_k = n - m;

    let mut wf = CpuWavefronts::new(max_k);
    let nd = wf.num_diags as i32;

    // Greedy match extension along diagonal k from reference offset `off`.
    let extend = |mut off: i32, k: i32| -> i32 {
        while off < m && off + k >= 0 && off + k < n
            && query[(off + k) as usize] == reference[off as usize]
        {
            off += 1;
        }
        off
    };

    // Score 0
    wf.ensure_row(0);
    let j0 = wf.idx(0, max_k);
    wf.m[j0] = extend(0, 0);
    if target_k == 0 && wf.m[j0] >= m {
        return Some((0, format!("{}=", n)));
    }

    let mut final_score = -1i32;

    for s in 1..=CPU_MAX_SCORE {
        wf.ensure_row(s);
        let (lo_s, hi_s) = wf.band(s, gap_extend);

        // --- Insertions (gap in reference, query advances): k + 1, offset stays ---
        let ins_s = s - gap_open - gap_extend;
        if ins_s >= 0 {
            let (lo, hi) = wf.band(ins_s, gap_extend);
            for ki in lo..=hi {
                let prev_off = wf.m[wf.idx(ins_s, ki)];
                if prev_off == WF_EMPTY { continue; }
                let new_ki = ki + 1;
                if new_ki >= 0 && new_ki < nd {
                    let j = wf.idx(s, new_ki);
                    if prev_off > wf.i[j] { wf.i[j] = prev_off; }
                }
            }
        }
        let ins_ext_s = s - gap_extend;
        if ins_ext_s >= 0 {
            let (lo, hi) = wf.band(ins_ext_s, gap_extend);
            for ki in lo..=hi {
                let prev_off = wf.i[wf.idx(ins_ext_s, ki)];
                if prev_off == WF_EMPTY { continue; }
                let new_ki = ki + 1;
                if new_ki >= 0 && new_ki < nd {
                    let j = wf.idx(s, new_ki);
                    if prev_off > wf.i[j] { wf.i[j] = prev_off; }
                }
            }
        }

        // --- Deletions (gap in query, reference advances): k - 1, offset + 1 ---
        let del_s = s - gap_open - gap_extend;
        if del_s >= 0 {
            let (lo, hi) = wf.band(del_s, gap_extend);
            for ki in lo..=hi {
                let prev_off = wf.m[wf.idx(del_s, ki)];
                if prev_off == WF_EMPTY { continue; }
                let new_ki = ki - 1;
                let new_off = prev_off + 1;
                if new_ki >= 0 && new_ki < nd {
                    let j = wf.idx(s, new_ki);
                    if new_off > wf.d[j] { wf.d[j] = new_off; }
                }
            }
        }
        let del_ext_s = s - gap_extend;
        if del_ext_s >= 0 {
            let (lo, hi) = wf.band(del_ext_s, gap_extend);
            for ki in lo..=hi {
                let prev_off = wf.d[wf.idx(del_ext_s, ki)];
                if prev_off == WF_EMPTY { continue; }
                let new_ki = ki - 1;
                let new_off = prev_off + 1;
                if new_ki >= 0 && new_ki < nd {
                    let j = wf.idx(s, new_ki);
                    if new_off > wf.d[j] { wf.d[j] = new_off; }
                }
            }
        }

        // --- Mismatches (from M at s - mismatch) ---
        let sub_s = s - mismatch;
        if sub_s >= 0 {
            let (lo, hi) = wf.band(sub_s, gap_extend);
            for ki in lo..=hi {
                let prev_off = wf.m[wf.idx(sub_s, ki)];
                if prev_off == WF_EMPTY { continue; }
                let k = ki - max_k;
                let q_pos = prev_off + k;
                if prev_off < m && q_pos >= 0 && q_pos < n {
                    let new_off = prev_off + 1;
                    let j = wf.idx(s, ki);
                    if new_off > wf.m[j] {
                        wf.m[j] = new_off;
                        wf.bt[j] = BtEntry { prev_s: sub_s, prev_k: k, op: OP_MISMATCH, src: COMP_M };
                    }
                }
            }
        }

        // --- Merge I and D into M ---
        for ki in lo_s..=hi_s {
            let j = wf.idx(s, ki);
            let i_off = wf.i[j];
            if i_off != WF_EMPTY && i_off > wf.m[j] {
                wf.m[j] = i_off;
                let k = ki - max_k;
                let prev_k = k - 1;
                let prev_ki = ki - 1;
                if prev_ki >= 0 && prev_ki < nd {
                    if ins_s >= 0 && wf.m[wf.idx(ins_s, prev_ki)] == i_off {
                        wf.bt[j] = BtEntry { prev_s: ins_s, prev_k, op: OP_INSERTION, src: COMP_M };
                    } else if ins_ext_s >= 0 && wf.i[wf.idx(ins_ext_s, prev_ki)] == i_off {
                        wf.bt[j] = BtEntry { prev_s: ins_ext_s, prev_k, op: OP_INSERTION, src: COMP_I };
                    }
                }
            }
            let d_off = wf.d[j];
            if d_off != WF_EMPTY && d_off > wf.m[j] {
                wf.m[j] = d_off;
                let k = ki - max_k;
                let prev_k = k + 1;
                let prev_ki = ki + 1;
                if prev_ki >= 0 && prev_ki < nd {
                    let from_m = del_s >= 0 && {
                        let v = wf.m[wf.idx(del_s, prev_ki)];
                        v != WF_EMPTY && v + 1 == d_off
                    };
                    if from_m {
                        wf.bt[j] = BtEntry { prev_s: del_s, prev_k, op: OP_DELETION, src: COMP_M };
                    } else if del_ext_s >= 0 {
                        let v = wf.d[wf.idx(del_ext_s, prev_ki)];
                        if v != WF_EMPTY && v + 1 == d_off {
                            wf.bt[j] = BtEntry { prev_s: del_ext_s, prev_k, op: OP_DELETION, src: COMP_D };
                        }
                    }
                }
            }
        }

        // --- Extend M wavefront ---
        for ki in lo_s..=hi_s {
            let j = wf.idx(s, ki);
            let off = wf.m[j];
            if off == WF_EMPTY { continue; }
            wf.m[j] = extend(off, ki - max_k);
        }

        // Check if target reached
        let target_ki = target_k + max_k;
        if target_ki >= 0 && target_ki < nd && wf.m[wf.idx(s, target_ki)] >= m {
            final_score = s;
            break;
        }
    }

    if final_score < 0 {
        return None;
    }

    // --- Backtrack to reconstruct CIGAR (same rules as the kernel) ---
    let mut ops: Vec<u8> = Vec::with_capacity((n + m) as usize);
    let mut cur_s = final_score;
    let mut cur_k = target_k;
    let mut cur_off = wf.m[wf.idx(cur_s, cur_k + max_k)];

    while cur_s > 0 {
        let bt = wf.bt[wf.idx(cur_s, cur_k + max_k)];
        if bt.op == BT_NONE {
            // No backtrack - fill remaining with matches
            while cur_off > 0 { ops.push(OP_MATCH); cur_off -= 1; }
            break;
        }
        let pj = wf.idx(bt.prev_s, bt.prev_k + max_k);
        let mut prev_off = match bt.src {
            COMP_M => wf.m[pj],
            COMP_I => wf.i[pj],
            _ => wf.d[pj],
        };
        if prev_off == WF_EMPTY { prev_off = 0; }
        // Offset after the operation (before extend)
        let off_after_op = match bt.op {
            OP_MISMATCH | OP_DELETION => prev_off + 1,
            _ => prev_off,
        };
        // Matches from extend, then the operation itself
        for _ in 0..(cur_off - off_after_op).max(0) { ops.push(OP_MATCH); }
        ops.push(bt.op);
        cur_s = bt.prev_s;
        cur_k = bt.prev_k;
        cur_off = prev_off;
    }
    // Remaining matches at score 0
    if cur_s == 0 {
        while cur_off > 0 { ops.push(OP_MATCH); cur_off -= 1; }
    }
    ops.reverse();
    Some((final_score, ops_to_cigar(&ops)))
}

// ============================================================
// Header tag extraction
// ============================================================

fn extract_tag_value(desc: Option<&str>, tag_prefix: &str) -> String {
    match desc {
        Some(d) => {
            for token in d.split_whitespace() {
                if let Some(value) = token.strip_prefix(tag_prefix) {
                    return value.to_string();
                }
            }
            String::new()
        }
        None => String::new(),
    }
}

// ============================================================
// K-mer pre-filter
// ============================================================

/// 2-bit encode a nucleotide; `None` for anything other than A/C/G/T.
#[inline]
fn encode_base(b: u8) -> Option<u64> {
    match b {
        b'A' => Some(0),
        b'C' => Some(1),
        b'G' => Some(2),
        b'T' => Some(3),
        _ => None,
    }
}

/// Collect the distinct k-mers of `seq` as 2-bit packed `u64`s (k <= 32).
/// Windows containing a non-ACGT base (e.g. `N`) are skipped. Sequences are
/// expected to be uppercase (the readers uppercase on load).
fn collect_kmers_packed(seq: &[u8], k: usize) -> HashSet<u64> {
    let mut set = HashSet::new();
    if k == 0 || k > MAX_KMER_LEN || seq.len() < k {
        return set;
    }
    let mask: u64 = if k == MAX_KMER_LEN { u64::MAX } else { (1u64 << (2 * k)) - 1 };
    let mut kmer: u64 = 0;
    let mut valid = 0usize;
    for &b in seq {
        match encode_base(b) {
            Some(code) => {
                kmer = ((kmer << 2) | code) & mask;
                valid += 1;
                if valid >= k {
                    set.insert(kmer);
                }
            }
            None => {
                kmer = 0;
                valid = 0;
            }
        }
    }
    set
}

/// Inverted k-mer index over the references: for every distinct packed k-mer
/// that occurs in any reference, the sorted list of references containing it.
/// The pre-filter counts shared k-mers for all references with one lookup per
/// read k-mer, so its cost per read does not grow with the number of
/// references. Built once at startup and shared read-only by the prep pool.
struct KmerIndex {
    num_refs: usize,
    refs_by_kmer: HashMap<u64, Vec<u32>>,
}

impl KmerIndex {
    /// Build the index for `k`-mers (k = 0 yields an empty index, used when the
    /// filter is disabled).
    fn build(references: &[SequenceRecord], k: usize) -> Self {
        let mut refs_by_kmer: HashMap<u64, Vec<u32>> = HashMap::new();
        if k > 0 {
            for (ri, r) in references.iter().enumerate() {
                // collect_kmers_packed yields each k-mer once per reference and
                // references are visited in order, so every list is sorted and
                // has no duplicates.
                for kmer in collect_kmers_packed(&r.seq, k) {
                    refs_by_kmer.entry(kmer).or_default().push(ri as u32);
                }
            }
        }
        KmerIndex { num_refs: references.len(), refs_by_kmer }
    }

    /// Number of distinct k-mers in the index.
    fn len(&self) -> usize {
        self.refs_by_kmer.len()
    }

    /// For each reference, count how many of `read_kmers` it contains. `shared`
    /// is a caller-owned scratch slice of length `num_refs`; it is overwritten.
    /// This equals, for every reference, the number of distinct read k-mers found
    /// in that reference's k-mer set.
    fn count_shared(&self, read_kmers: &HashSet<u64>, shared: &mut [u32]) {
        debug_assert_eq!(shared.len(), self.num_refs);
        shared.fill(0);
        for kmer in read_kmers {
            if let Some(refs) = self.refs_by_kmer.get(kmer) {
                for &ri in refs {
                    shared[ri as usize] += 1;
                }
            }
        }
    }
}

/// True when at least `threshold` of a read's `total` distinct k-mers (`hits` of
/// them) occur in the reference. A read with no valid k-mers (shorter than k, or
/// all ambiguous) passes.
fn kmer_fraction_passes(hits: usize, total: usize, threshold: f64) -> bool {
    if total == 0 {
        return true;
    }
    (hits as f64 / total as f64) >= threshold
}

// ============================================================
// Sequence packing for GPU transfer
// ============================================================

fn pack_sequences(records: &[SequenceRecord]) -> (Vec<i8>, Vec<usize>, Vec<usize>) {
    let seqs: Vec<&[u8]> = records.iter().map(|r| r.seq.as_slice()).collect();
    pack_sequences_raw(&seqs)
}

/// Pack raw byte sequences (e.g., unique reads or their reverse complements)
/// into one buffer plus per-sequence lengths and offsets for GPU transfer.
fn pack_sequences_raw(seqs: &[&[u8]]) -> (Vec<i8>, Vec<usize>, Vec<usize>) {
    let total: usize = seqs.iter().map(|s| s.len()).sum();
    let mut data = Vec::with_capacity(total);
    let mut lengths = Vec::with_capacity(seqs.len());
    let mut offsets = Vec::with_capacity(seqs.len());
    for seq in seqs {
        offsets.push(data.len());
        lengths.push(seq.len());
        data.extend(seq.iter().map(|&b| b as i8));
    }
    (data, lengths, offsets)
}

// ============================================================
// File I/O
// ============================================================

fn detect_format(path: &PathBuf) -> Result<SequenceFormat> {
    let (mut reader, _) = get_reader(Box::new(File::open(path)?))?;
    let mut buffer = [0; 1];
    if reader.read(&mut buffer)? == 0 {
        return Ok(SequenceFormat::Unknown);
    }
    match buffer[0] {
        b'>' => Ok(SequenceFormat::Fasta),
        b'@' => Ok(SequenceFormat::Fastq),
        _ => {
            warn!("Could not detect file format. Assuming FASTA.");
            Ok(SequenceFormat::Fasta)
        }
    }
}

trait BatchReader {
    fn read_batch(&mut self, batch_size: usize) -> Result<Vec<SequenceRecord>>;
}

/// Shared batching logic. Pull records until `batch_size` is reached, then
/// keep going while the next record has the same name as the last one, so
/// consecutive same-name records (unmerged mates as fastp writes them) never
/// straddle a batch boundary. A record read past the boundary is parked in
/// `pending` for the next call. Sequences are uppercased so alignment is
/// case-insensitive and identical sequences deduplicate regardless of case.
fn fill_batch<F>(
    pending: &mut Option<SequenceRecord>,
    batch_size: usize,
    mut next: F,
) -> Result<Vec<SequenceRecord>>
where
    F: FnMut() -> Result<Option<SequenceRecord>>,
{
    let mut batch = Vec::with_capacity(batch_size + 1);
    if let Some(rec) = pending.take() {
        batch.push(rec);
    }
    loop {
        let rec = match next()? {
            Some(r) => r,
            None => break,
        };
        if batch.len() >= batch_size && batch.last().map_or(false, |last| last.id != rec.id) {
            *pending = Some(rec);
            break;
        }
        batch.push(rec);
    }
    Ok(batch)
}

struct FastaBatchReader {
    reader: fasta::Reader<BufReader<Box<dyn Read>>>,
    pending: Option<SequenceRecord>,
}

impl FastaBatchReader {
    fn new(reader: Box<dyn Read>) -> Self {
        Self { reader: fasta::Reader::from_bufread(BufReader::new(reader)), pending: None }
    }
}

impl BatchReader for FastaBatchReader {
    fn read_batch(&mut self, batch_size: usize) -> Result<Vec<SequenceRecord>> {
        let mut record = fasta::Record::new();
        let reader = &mut self.reader;
        fill_batch(&mut self.pending, batch_size, || {
            reader.read(&mut record)?;
            if record.is_empty() {
                return Ok(None);
            }
            Ok(Some(SequenceRecord {
                id: record.id().to_string(),
                desc: record.desc().map(|d| d.to_string()),
                seq: record.seq().to_ascii_uppercase(),
            }))
        })
    }
}

struct FastqBatchReader {
    reader: fastq::Reader<BufReader<Box<dyn Read>>>,
    pending: Option<SequenceRecord>,
}

impl FastqBatchReader {
    fn new(reader: Box<dyn Read>) -> Self {
        Self { reader: fastq::Reader::from_bufread(BufReader::new(reader)), pending: None }
    }
}

impl BatchReader for FastqBatchReader {
    fn read_batch(&mut self, batch_size: usize) -> Result<Vec<SequenceRecord>> {
        let mut record = fastq::Record::new();
        let reader = &mut self.reader;
        fill_batch(&mut self.pending, batch_size, || {
            reader.read(&mut record)?;
            if record.is_empty() {
                return Ok(None);
            }
            Ok(Some(SequenceRecord {
                id: record.id().to_string(),
                desc: record.desc().map(|d| d.to_string()),
                seq: record.seq().to_ascii_uppercase(),
            }))
        })
    }
}

fn create_sequence_reader(path: &PathBuf) -> Result<Box<dyn BatchReader>> {
    let format = detect_format(path)?;
    let (reader, comp) = get_reader(Box::new(File::open(path)?))?;
    if comp != compression::Format::No {
        info!("Detected compressed file: {:?}", comp);
    }
    match format {
        SequenceFormat::Fasta => {
            info!("Detected FASTA format for {}", path.display());
            Ok(Box::new(FastaBatchReader::new(reader)))
        },
        SequenceFormat::Fastq => {
            info!("Detected FASTQ format for {}", path.display());
            Ok(Box::new(FastqBatchReader::new(reader)))
        },
        SequenceFormat::Unknown => Err(anyhow!("Unknown sequence file format")),
    }
}

fn read_sequences_fully(path: &PathBuf) -> Result<Vec<SequenceRecord>> {
    let mut reader = create_sequence_reader(path)?;
    let mut seqs = Vec::new();
    loop {
        let batch = reader.read_batch(1000)?;
        if batch.is_empty() { break; }
        seqs.extend(batch);
    }
    Ok(seqs)
}

fn create_writer(output_path: &Option<PathBuf>) -> Result<csv::Writer<Box<dyn io::Write + Send>>> {
    let w: Box<dyn io::Write + Send> = match output_path {
        Some(p) => Box::new(File::create(p)?),
        None => Box::new(io::stdout()),
    };
    Ok(WriterBuilder::new().delimiter(b'\t').from_writer(w))
}

fn setup_logging(verbosity: u8) {
    let level = match verbosity {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    env_logger::Builder::new().filter_level(level).init();
}

// ============================================================
// Tests (CPU only; no GPU required)
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: &str, desc: Option<&str>, seq: &str) -> SequenceRecord {
        SequenceRecord {
            id: id.to_string(),
            desc: desc.map(|d| d.to_string()),
            seq: seq.as_bytes().to_vec(),
        }
    }

    /// Fabricate GPU results for a prepared batch: forward pairs align
    /// perfectly (score 0, all matches), RC pairs are all mismatches (score
    /// `rc_score`). Pairs listed in `gpu_failed` get score -1 (CPU fallback).
    fn fake_gpu_result(prepared: PreparedBatch, rc_score: i32, gpu_failed: &[usize]) -> GpuBatchResult {
        let n = prepared.pair_query_idx.len();
        let mut scores = vec![0i32; n];
        let mut cigar_ops: Vec<Vec<i8>> = vec![Vec::new(); n];
        let mut cigar_lengths = vec![0i32; n];
        for i in 0..n {
            let ui = prepared.pair_query_idx[i] as usize;
            let len = prepared.query_lengths[ui] as usize;
            if gpu_failed.contains(&i) {
                scores[i] = -1;
                continue;
            }
            if prepared.pair_is_rc[i] {
                scores[i] = rc_score;
                cigar_ops[i] = vec![OP_MISMATCH as i8; len];
            } else {
                scores[i] = 0;
                cigar_ops[i] = vec![OP_MATCH as i8; len];
            }
            cigar_lengths[i] = len as i32;
        }
        GpuBatchResult { prepared, scores, cigar_ops, cigar_lengths }
    }

    #[test]
    fn dedup_maps_reads_to_first_seen_unique() {
        let batch = vec![
            rec("r0", None, "ACGT"), rec("r1", None, "TTTT"), rec("r2", None, "ACGT"),
            rec("r3", None, "GGCC"), rec("r4", None, "TTTT"),
        ];
        let (rep, map) = dedup_reads(&batch, true);
        assert_eq!(rep, vec![0, 1, 3]);
        assert_eq!(map, vec![0, 1, 0, 2, 1]);

        let (rep, map) = dedup_reads(&batch, false);
        assert_eq!(rep, vec![0, 1, 2, 3, 4]);
        assert_eq!(map, rep);
    }

    #[test]
    fn prepare_batch_aligns_unique_sequences_only() {
        let refs = vec![rec("refA", None, "ACGTACGTAC"), rec("refB", None, "TTTTGGGGCC")];
        let batch = vec![
            rec("r0", None, "ACGTACGTAC"), rec("r1", None, "ACGTACGTAC"), rec("r2", None, "TTTTGGGGCC"),
        ];
        let p = prepare_batch(batch, &refs, &KmerIndex::build(&refs, 0), false, 0, 0.5, true, 500);
        assert_eq!(p.num_reads, 3);
        assert_eq!(p.num_unique, 2);
        assert_eq!(p.read_to_uidx, vec![0, 0, 1]);
        assert_eq!(p.query_lengths, vec![10, 10]);
        assert_eq!(p.rc_query_lengths, vec![10, 10]);
        // 2 unique x 2 refs x 2 orientations, all referencing unique indices
        assert_eq!(p.pair_query_idx.len(), 8);
        assert!(p.pair_query_idx.iter().all(|&q| (q as usize) < 2));
        assert_eq!(p.too_long, vec![false, false]);
        assert_eq!(p.kmer_fail_count, 0);
    }

    #[test]
    fn too_long_reads_get_no_pairs() {
        let refs = vec![rec("refA", None, "ACGTACGTAC")];
        let batch = vec![rec("short", None, "ACGTAC"), rec("long", None, "ACGTACGTACGTACGT")];
        let p = prepare_batch(batch, &refs, &KmerIndex::build(&refs, 0), false, 0, 0.5, true, 10);
        assert_eq!(p.too_long, vec![false, true]);
        assert_eq!(p.pair_query_idx.len(), 2); // the short read only, both orientations
        assert!(p.pair_query_idx.iter().all(|&q| q == 0));
        assert_eq!(p.fwd_gpu_idx[1], None);
        assert_eq!(p.rc_gpu_idx[1], None);
    }

    #[test]
    fn kmer_filter_rejects_unrelated_pairs() {
        let refs = vec![rec("refA", None, "ACGTACGTACGTACGT")];
        let index = KmerIndex::build(&refs, 4);
        let batch = vec![
            rec("same", None, "ACGTACGTACGTACGT"),
            rec("polyT", None, "TTTTTTTTTTTTTTTT"),
        ];
        let p = prepare_batch(batch, &refs, &index, true, 4, 0.5, true, 500);
        assert!(p.fwd_kmer_pass[0]);
        assert!(!p.fwd_kmer_pass[1]);
        assert!(!p.rc_kmer_pass[1]); // RC of polyT is polyA: still no shared 4-mers
        assert_eq!(p.kmer_fail_count, 1);
    }

    #[test]
    fn packed_kmers_skip_ambiguous_bases() {
        let set = collect_kmers_packed(b"ACGTN", 2);
        assert_eq!(set.len(), 3);
        assert!(set.contains(&0b0001)); // AC
        assert!(set.contains(&0b0110)); // CG
        assert!(set.contains(&0b1011)); // GT
        assert!(collect_kmers_packed(b"NNNN", 2).is_empty());
        assert!(collect_kmers_packed(b"ACG", 4).is_empty()); // shorter than k
        assert_eq!(collect_kmers_packed(b"ACGTACGT", 4).len(), 4); // ACGT CGTA GTAC TACG
        // A window straddling an N is dropped, but k-mers after it resume.
        assert_eq!(collect_kmers_packed(b"ACGTNACGT", 4).len(), 1);
        // k = 32 exercises the full-width mask.
        let s33 = b"ACGTACGTACGTACGTACGTACGTACGTACGTA";
        assert_eq!(s33.len(), 33);
        assert_eq!(collect_kmers_packed(s33, 32).len(), 2);
    }

    #[test]
    fn kmer_fraction_threshold() {
        assert!(kmer_fraction_passes(4, 4, 0.99));
        assert!(kmer_fraction_passes(1, 4, 0.25));
        assert!(!kmer_fraction_passes(1, 4, 0.5));
        assert!(kmer_fraction_passes(0, 0, 1.0)); // no k-mers: pass
    }

    #[test]
    fn kmer_index_lists_are_sorted_and_count_shared_kmers() {
        let refs = vec![
            rec("r0", None, "ACGTACGT"),   // 4-mers: ACGT CGTA GTAC TACG
            rec("r1", None, "ACGTTTTT"),   // 4-mers: ACGT CGTT GTTT TTTT
            rec("r2", None, "GGGG"),       // 4-mers: GGGG
        ];
        let index = KmerIndex::build(&refs, 4);
        assert_eq!(index.len(), 8);
        let acgt = collect_kmers_packed(b"ACGT", 4).into_iter().next().unwrap();
        assert_eq!(index.refs_by_kmer[&acgt], vec![0, 1]);

        let mut shared = vec![0u32; 3];
        index.count_shared(&collect_kmers_packed(b"ACGTTTTT", 4), &mut shared);
        assert_eq!(shared, vec![1, 4, 0]); // r0 shares ACGT only; r1 all four; r2 none
        index.count_shared(&collect_kmers_packed(b"NNNN", 4), &mut shared);
        assert_eq!(shared, vec![0, 0, 0]);
    }

    /// Tiny deterministic generator so the test needs no extra dependency.
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn kmer_index_matches_per_reference_sets() {
        // Random references; reads are mutated copies of references (so they
        // share many k-mers) with occasional ambiguous bases. The index must give
        // exactly the counts obtained by probing each reference's own k-mer set.
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let bases = b"ACGT";
        let refs: Vec<SequenceRecord> = (0..40)
            .map(|i| {
                let seq: Vec<u8> = (0..(30 + i * 3)).map(|_| bases[(rng.next() % 4) as usize]).collect();
                rec(&format!("r{}", i), None, std::str::from_utf8(&seq).unwrap())
            })
            .collect();
        for k in [3usize, 8, 15] {
            let index = KmerIndex::build(&refs, k);
            let sets: Vec<HashSet<u64>> = refs.iter().map(|r| collect_kmers_packed(&r.seq, k)).collect();
            let mut shared = vec![0u32; refs.len()];
            for _ in 0..300 {
                let src = &refs[(rng.next() % refs.len() as u64) as usize].seq;
                let read: Vec<u8> = src.iter().map(|&b| match rng.next() % 12 {
                    0 => bases[(rng.next() % 4) as usize],
                    1 => b'N',
                    _ => b,
                }).collect();
                let kmers = collect_kmers_packed(&read, k);
                index.count_shared(&kmers, &mut shared);
                for (ri, set) in sets.iter().enumerate() {
                    let expected = kmers.iter().filter(|km| set.contains(km)).count();
                    assert_eq!(shared[ri] as usize, expected, "k={} ref={}", k, ri);
                }
            }
        }
    }

    #[test]
    fn cpu_align_cases() {
        let (go, ge, mm) = (2, 1, 4);
        assert_eq!(cpu_align(b"ACGTACGT", b"ACGTACGT", go, ge, mm, 0), Some((0, "8=".to_string())));
        assert_eq!(cpu_align(b"ACGAACGT", b"ACGTACGT", go, ge, mm, 0), Some((4, "3=1X4=".to_string())));
        assert_eq!(cpu_align(b"ACGTTACG", b"ACGTACG", go, ge, mm, 0), Some((3, "4=1I3=".to_string())));
        assert_eq!(cpu_align(b"ACGTACG", b"ACGTTACG", go, ge, mm, 0), Some((3, "4=1D3=".to_string())));
        // Multi-base deletion: open + 2 extensions = 2 + 3*1 = 5
        let reference = b"ACGTTGCAAGTCCATGGACT";
        let mut query = reference[..8].to_vec();
        query.extend_from_slice(&reference[11..]);
        assert_eq!(cpu_align(&query, reference, go, ge, mm, 0), Some((5, "8=3D9=".to_string())));
        // Multi-base insertion, pe_23amp penalties (-g 5 -e 1 -m 3): 5 + 3*1 = 8
        assert_eq!(cpu_align(reference, &query, 5, 1, 3, 0), Some((8, "8=3I9=".to_string())));
    }

    #[test]
    fn cpu_align_scores_are_comparable_across_orientations() {
        // Divergent pair: still aligns (within CPU_MAX_SCORE) with a high score.
        let (score, cigar) = cpu_align(b"TTTTGGGGCC", b"ACGTACGTAC", 2, 1, 4, 0).unwrap();
        assert!(score > 0);
        assert!(!cigar.is_empty());
    }

    #[test]
    fn format_rows_fan_out_matches_per_read_output() {
        let refs = vec![rec("refA", None, "ACGTACGTAC"), rec("refB", None, "TTTTGGGGCC")];
        let batch = vec![
            rec("r0", Some("RG:Z:c1"), "ACGTACGTAC"),
            rec("r1", Some("RG:Z:c1"), "ACGTACGTAC"),
            rec("r2", Some("RG:Z:c2"), "ACGTACGTAC"),
            rec("r3", Some("RG:Z:c1"), "TTTTGGGGCC"),
        ];
        let tags = vec!["RG:Z:".to_string()];

        let with = fake_gpu_result(prepare_batch(batch.clone(), &refs, &KmerIndex::build(&refs, 0), false, 0, 0.5, true, 500), 7, &[]);
        let without = fake_gpu_result(prepare_batch(batch, &refs, &KmerIndex::build(&refs, 0), false, 0, 0.5, false, 500), 7, &[]);
        let a = format_batch_rows(&with, &refs, &tags, 2, 1, 4, 0, false, false);
        let b = format_batch_rows(&without, &refs, &tags, 2, 1, 4, 0, false, false);

        assert_eq!(a.rows.len(), 8); // 4 reads x 2 refs
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.rows[0], vec!["r0", "refA", "10=", "+", "c1"]);
        assert_eq!(a.rows[1], vec!["r0", "refB", "10=", "+", "c1"]);
        assert_eq!(a.rows[4], vec!["r2", "refA", "10=", "+", "c2"]);
        assert_eq!(a.rows[6], vec!["r3", "refA", "10=", "+", "c1"]);
        assert_eq!(a.cpu_fallbacks, 0);
        assert_eq!(a.too_long_reads, 0);
    }

    #[test]
    fn collapse_groups_by_sequence_and_tags() {
        let refs = vec![rec("refA", None, "ACGTACGTAC"), rec("refB", None, "TTTTGGGGCC")];
        let batch = vec![
            rec("r0", Some("RG:Z:c1"), "ACGTACGTAC"),
            rec("r1", Some("RG:Z:c1"), "ACGTACGTAC"),
            rec("r2", Some("RG:Z:c2"), "ACGTACGTAC"),
            rec("r3", Some("RG:Z:c1"), "TTTTGGGGCC"),
        ];
        let tags = vec!["RG:Z:".to_string()];
        let res = fake_gpu_result(prepare_batch(batch, &refs, &KmerIndex::build(&refs, 0), false, 0, 0.5, true, 500), 7, &[]);
        let out = format_batch_rows(&res, &refs, &tags, 2, 1, 4, 0, false, true);

        assert_eq!(out.rows.len(), 6); // 3 groups x 2 refs
        assert_eq!(out.rows[0], vec!["r0", "refA", "10=", "+", "c1", "2"]);
        assert_eq!(out.rows[1], vec!["r0", "refB", "10=", "+", "c1", "2"]);
        assert_eq!(out.rows[2], vec!["r2", "refA", "10=", "+", "c2", "1"]);
        assert_eq!(out.rows[4], vec!["r3", "refA", "10=", "+", "c1", "1"]);
        let sum_ref_a: usize = out.rows.iter().filter(|r| r[1] == "refA").map(|r| r[5].parse::<usize>().unwrap()).sum();
        assert_eq!(sum_ref_a, 4);
    }

    #[test]
    fn collapse_keeps_unmerged_mates_together() {
        let refs = vec![rec("refA", None, "ACGTACGTAC")];
        // Two identical unmerged pairs (mates share a name, written consecutively)
        // from one cell, then a merged read whose sequence equals the first mate.
        let batch = vec![
            rec("p1", Some("RG:Z:c1"), "ACGTACGTAC"), rec("p1", Some("RG:Z:c1"), "TTTTGGGGCC"),
            rec("p2", Some("RG:Z:c1"), "ACGTACGTAC"), rec("p2", Some("RG:Z:c1"), "TTTTGGGGCC"),
            rec("m1", Some("RG:Z:c1"), "ACGTACGTAC"),
            // same pair but a different cell: separate group
            rec("p3", Some("RG:Z:c2"), "ACGTACGTAC"), rec("p3", Some("RG:Z:c2"), "TTTTGGGGCC"),
        ];
        let tags = vec!["RG:Z:".to_string()];
        let res = fake_gpu_result(prepare_batch(batch, &refs, &KmerIndex::build(&refs, 0), false, 0, 0.5, true, 500), 7, &[]);
        let out = format_batch_rows(&res, &refs, &tags, 2, 1, 4, 0, false, true);
        assert_eq!(out.rows, vec![
            vec!["p1", "refA", "10=", "+", "c1", "2"],
            vec!["p1", "refA", "10=", "+", "c1", "2"],
            vec!["m1", "refA", "10=", "+", "c1", "1"],
            vec!["p3", "refA", "10=", "+", "c2", "1"],
            vec!["p3", "refA", "10=", "+", "c2", "1"],
        ]);
        // n_reads summed over member rows equals the number of input records
        let total: usize = out.rows.iter().map(|r| r[5].parse::<usize>().unwrap()).sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn reader_keeps_same_name_records_in_one_batch() {
        let fq = "@a RG:Z:c1\nACGT\n+\nIIII\n@a RG:Z:c1\nGGGG\n+\nIIII\n@b\ntttt\n+\nIIII\n@c\nCCCC\n+\nIIII\n@c\nAAAA\n+\nIIII\n";
        let ids = |b: &[SequenceRecord]| b.iter().map(|x| x.id.clone()).collect::<Vec<_>>();
        let mut r = FastqBatchReader::new(Box::new(std::io::Cursor::new(fq.as_bytes().to_vec())));
        let b1 = r.read_batch(1).unwrap();
        assert_eq!(ids(&b1), vec!["a", "a"]);
        let b2 = r.read_batch(1).unwrap();
        assert_eq!(ids(&b2), vec!["b"]);
        assert_eq!(b2[0].seq, b"TTTT".to_vec()); // uppercased on load
        let b3 = r.read_batch(1).unwrap();
        assert_eq!(ids(&b3), vec!["c", "c"]);
        assert!(r.read_batch(1).unwrap().is_empty());

        // A batch size larger than the file returns everything at once.
        let mut r = FastqBatchReader::new(Box::new(std::io::Cursor::new(fq.as_bytes().to_vec())));
        assert_eq!(r.read_batch(100).unwrap().len(), 5);
        assert!(r.read_batch(100).unwrap().is_empty());

        let fa = ">x d1\nACGT\n>x d2\nGGGG\n>y\nTT\n";
        let mut r = FastaBatchReader::new(Box::new(std::io::Cursor::new(fa.as_bytes().to_vec())));
        assert_eq!(ids(&r.read_batch(1).unwrap()), vec!["x", "x"]);
        assert_eq!(ids(&r.read_batch(1).unwrap()), vec!["y"]);
    }

    #[test]
    fn cpu_fallback_uses_real_scores_to_pick_strand() {
        // One unique sequence vs one reference; the forward GPU result "fails"
        // (score -1) so the CPU re-aligns it. The RC result is a fabricated
        // all-mismatch alignment with score 7.
        let refs = vec![rec("refA", None, "ACGTACGTAC")];
        let tags: Vec<String> = Vec::new();

        // Case 1: the sequence equals the reference, CPU forward score 0 beats RC 7.
        let p = prepare_batch(vec![rec("r0", None, "ACGTACGTAC")], &refs, &KmerIndex::build(&refs, 0), false, 0, 0.5, true, 500);
        let fwd_idx = p.fwd_gpu_idx[0].unwrap();
        let res = fake_gpu_result(p, 7, &[fwd_idx]);
        let out = format_batch_rows(&res, &refs, &tags, 2, 1, 4, 0, false, false);
        assert_eq!(out.rows, vec![vec!["r0", "refA", "10=", "+"]]);
        assert_eq!(out.cpu_fallbacks, 1);

        // Case 2: the sequence is unrelated, CPU forward score is high, RC 7 wins.
        let p = prepare_batch(vec![rec("r0", None, "TTTTGGGGCC")], &refs, &KmerIndex::build(&refs, 0), false, 0, 0.5, true, 500);
        let fwd_idx = p.fwd_gpu_idx[0].unwrap();
        let res = fake_gpu_result(p, 7, &[fwd_idx]);
        let out = format_batch_rows(&res, &refs, &tags, 2, 1, 4, 0, false, false);
        assert_eq!(out.rows, vec![vec!["r0", "refA", "10X", "-"]]);
        assert_eq!(out.cpu_fallbacks, 1);
    }

    #[test]
    fn too_long_and_failed_rows() {
        let refs = vec![rec("refA", None, "ACGTACGTAC")];
        let index = KmerIndex::build(&refs, 4);
        let batch = vec![
            rec("long", None, "ACGTACGTACGTACGT ACGT".replace(' ', "").as_str()),
            rec("polyT", None, "TTTTTTTTTT"),
        ];
        let p = prepare_batch(batch, &refs, &index, true, 4, 0.5, true, 12);
        let res = fake_gpu_result(p, 7, &[]);
        let tags: Vec<String> = Vec::new();

        let out = format_batch_rows(&res, &refs, &tags, 2, 1, 4, 0, false, false);
        assert_eq!(out.rows, vec![
            vec!["long", "refA", "*", "."],
            vec!["polyT", "refA", "FAILED", "."],
        ]);
        assert_eq!(out.too_long_reads, 1);
        assert_eq!(out.longest_read, 20);

        let out = format_batch_rows(&res, &refs, &tags, 2, 1, 4, 0, true, false);
        assert_eq!(out.rows, vec![vec!["long", "refA", "*", "."]]); // --skip-fail drops FAILED only
    }

    #[test]
    fn helpers() {
        assert_eq!(ops_to_cigar(&[0, 0, 1, 0, 2, 2, 3]), "2=1X1=2I1D");
        assert_eq!(ops_to_cigar(&[]), "");
        assert_eq!(reverse_complement(b"ACGTN"), b"NACGT".to_vec());
        assert_eq!(reverse_complement(b"acgt"), b"ACGT".to_vec());
        assert_eq!(extract_tag_value(Some("RG:Z:CELL1 merged_150_120"), "RG:Z:"), "CELL1");
        assert_eq!(extract_tag_value(Some("RG:Z:CELL1 merged_150_120"), "merged"), "_150_120");
        assert_eq!(extract_tag_value(None, "RG:Z:"), "");
        assert_eq!(dedup_ratio(10, 4), 2.5);
        assert_eq!(dedup_ratio(0, 0), 1.0);
    }

    /// Cross-check the CPU aligner against the CUDA kernel compiled as host
    /// C++ (`tools/kernel_emulate.cpp`). Set `BLEP_KERNEL_EMULATION_TSV` to the
    /// harness output to enable; without it this test is a no-op.
    #[test]
    fn cpu_align_matches_kernel_emulation() {
        let Ok(path) = std::env::var("BLEP_KERNEL_EMULATION_TSV") else { return; };
        let text = std::fs::read_to_string(&path).expect("read kernel emulation TSV");
        let mut checked = 0usize;
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 9 { continue; }
            let (q, r) = (f[0].as_bytes(), f[1].as_bytes());
            let go: i32 = f[2].parse().unwrap();
            let ge: i32 = f[3].parse().unwrap();
            let mm: i32 = f[4].parse().unwrap();
            let max_score: i32 = f[5].parse().unwrap();
            let max_k: usize = f[6].parse().unwrap();
            let score: i32 = f[7].parse().unwrap();
            let cigar = f[8];
            let got = cpu_align(q, r, go, ge, mm, max_k);
            if score >= 0 {
                assert_eq!(got, Some((score, cigar.to_string())), "pair {} vs {}", f[0], f[1]);
            } else {
                // The kernel gave up at max_score; the CPU (cap 4096) must not find a cheaper alignment.
                assert!(got.as_ref().map_or(true, |(s, _)| *s > max_score),
                        "pair {} vs {}: kernel exceeded {} but CPU found {:?}", f[0], f[1], max_score, got);
            }
            checked += 1;
        }
        assert!(checked > 0, "emulation TSV had no pairs");
        eprintln!("cross-checked {} pairs against the kernel emulation", checked);
    }
}
