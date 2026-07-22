use std::path::PathBuf;
use std::fs::File;
use std::io::{self, Read, BufReader};
use std::collections::HashSet;
use std::sync::mpsc;
use std::thread;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, Context, anyhow};
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

    /// Batch size for processing reads (number of reads to load at once)
    #[clap(short = 'b', long, default_value = "10000")]
    batch_size: usize,

    /// Maximum read length (for GPU memory pre-allocation)
    #[clap(long, default_value = "500")]
    max_read_length: usize,

    /// Maximum reference length (for GPU memory pre-allocation)
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

    /// Number of CPU threads that prepare batches (reverse-complement, packing,
    /// and the k-mer pre-filter) before they reach the GPUs. This preparation is
    /// the stage most likely to starve the GPUs, so it runs as a parallel pool.
    /// 0 = auto (roughly all hardware threads not reserved for GPU workers).
    #[clap(long, default_value = "0")]
    prep_threads: usize,

    /// Number of CPU threads that post-process GPU results (CPU fallback
    /// alignment and output formatting) off the GPU worker threads, so a GPU is
    /// never blocked formatting or re-aligning. 0 = auto.
    #[clap(long, default_value = "0")]
    post_threads: usize,

    /// K-mer length for the pre-alignment filter. When set (> 0), unique k-mers
    /// of the read are checked against the reference before running WFA.
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

    /// Verbosity level
    #[clap(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

// CIGAR operation codes (must match kernel defines)
const OP_MATCH: u8 = 0;
const OP_MISMATCH: u8 = 1;
const OP_INSERTION: u8 = 2;
const OP_DELETION: u8 = 3;

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

    let kmer_filter_enabled = args.kmer_length > 0;
    if kmer_filter_enabled {
        info!("K-mer pre-filter enabled: k={}, threshold={:.2}",
              args.kmer_length, args.kmer_threshold);
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
    info!("Loaded {} reference sequences", references.len());

    // Build owned k-mer sets for each reference (used by the prep threads' pre-filter)
    let ref_kmer_sets_owned: Vec<HashSet<Vec<u8>>> = if kmer_filter_enabled {
        references.iter().map(|r| collect_kmers_owned(&r.seq, args.kmer_length)).collect()
    } else {
        Vec::new()
    };

    // Pack reference sequences once on the host; each worker uploads its own
    // copy to its device. Shared across worker threads via Arc.
    let (ref_data, ref_lengths, ref_offsets) = pack_sequences(&references);
    let ref_data = Arc::new(ref_data);
    let ref_lengths = Arc::new(ref_lengths.iter().map(|&x| x as i32).collect::<Vec<i32>>());
    let ref_offsets = Arc::new(ref_offsets.iter().map(|&x| x as i32).collect::<Vec<i32>>());

    // Reference records shared read-only with the prep pool (k-mer filter) and
    // the post pool (output formatting + CPU fallback).
    let references_arc = Arc::new(references);
    let ref_kmer_sets_arc = Arc::new(ref_kmer_sets_owned);

    // Create the output writer and emit the header row up front. The writer is
    // then moved into a single dedicated writer thread (below).
    let mut writer = create_writer(&args.output)?;
    {
        let mut header_fields: Vec<&str> = vec!["read_name", "reference_name", "cigar", "strand"];
        for tag in &header_tags {
            header_fields.push(tag.as_str());
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

    // Decide how many CPU threads to devote to each CPU-bound stage. Preparation
    // (k-mer filter, reverse-complement, packing) is the heavy stage that tends
    // to starve the GPUs, so it gets most of the cores; post-processing is
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
    //   prep pool (P)   : reverse-complement, pack, k-mer pre-filter
    //   GPU workers (N) : upload -> WFA kernel -> download  (GPU work only)
    //   post pool (Q)   : CPU fallback + output row formatting
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
        let ref_kmer_sets_arc = Arc::clone(&ref_kmer_sets_arc);
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
                    &ref_kmer_sets_arc,
                    kmer_filter_enabled,
                    kmer_length,
                    kmer_threshold,
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
                let num_gpu_pairs = prepared.pair_query_idx.len();

                info!("Device {}: aligning batch of {} reads, {} GPU pairs ({} both-filtered)",
                      device_id, prepared.num_reads, num_gpu_pairs, prepared.kmer_fail_count);

                // GPU work only: upload, launch, synchronize, download.
                let t1 = Instant::now();
                let (scores, cigar_ops, cigar_lengths) = run_gpu_alignment(
                    &stream, &kernel, &prepared,
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
        let handle = thread::spawn(move || -> Result<(StageStats, usize)> {
            let mut stats = StageStats::default();
            let mut cpu_fallbacks = 0usize;
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
                let (rows, fb) = format_batch_rows(
                    &result,
                    references_arc.as_slice(),
                    &header_tags,
                    gap_open, gap_extend, mismatch,
                    max_score, skip_fail,
                );
                stats.busy += t1.elapsed();
                stats.batches += 1;
                cpu_fallbacks += fb;

                let t2 = Instant::now();
                if rows_tx.send(rows).is_err() {
                    break; // writer exited
                }
                stats.send_wait += t2.elapsed();
            }
            Ok((stats, cpu_fallbacks))
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
    let mut total_cpu_fallbacks = 0usize;
    for (i, handle) in post_handles.into_iter().enumerate() {
        let (s, fb) = handle.join()
            .map_err(|e| anyhow!("post thread {} panicked: {:?}", i, e))?
            .context("post thread failed")?;
        post.merge(&s);
        total_cpu_fallbacks += fb;
    }

    let reader_stats = reader_handle.join()
        .map_err(|e| anyhow!("reader thread panicked: {:?}", e))?
        .context("reader thread failed")?;

    let writer_stats = writer_handle.join()
        .map_err(|e| anyhow!("writer thread panicked: {:?}", e))?
        .context("writer thread failed")?;

    let wall = pipeline_start.elapsed();

    if total_cpu_fallbacks > 0 {
        info!("CPU fallback used for {} pairs total", total_cpu_fallbacks);
    }
    info!("Alignment completed. Processed {} reads in {} batches across {} device(s)",
          gpu.reads, gpu.batches, num_devices);

    log_pipeline_timing(
        wall, &reader_stats, &prep, &gpu, &post, &writer_stats,
        prep_threads, num_devices, post_threads,
    );

    Ok(())
}

/// A batch that has been read from disk and pre-filtered by a prep thread,
/// ready for GPU alignment.
struct PreparedBatch {
    batch: Vec<SequenceRecord>,
    query_data: Vec<i8>,
    query_lengths: Vec<i32>,
    query_offsets: Vec<i32>,
    rc_query_data: Vec<i8>,
    rc_query_lengths: Vec<i32>,
    rc_query_offsets: Vec<i32>,
    fwd_kmer_pass: Vec<bool>,
    rc_kmer_pass: Vec<bool>,
    kmer_fail_count: usize,
    pair_query_idx: Vec<i32>,
    pair_ref_idx: Vec<i32>,
    pair_is_rc: Vec<bool>,
    fwd_gpu_idx: Vec<Option<usize>>,
    rc_gpu_idx: Vec<Option<usize>>,
    num_reads: usize,
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

/// Turn a raw batch of reads into a `PreparedBatch`: generate reverse
/// complements, pack both orientations for GPU upload, run the k-mer pre-filter,
/// and build the per-pair index lists. This is the CPU-heavy work that runs in
/// parallel across the prep pool so it does not starve the GPUs.
fn prepare_batch(
    batch: Vec<SequenceRecord>,
    references: &[SequenceRecord],
    ref_kmer_sets: &[HashSet<Vec<u8>>],
    kmer_filter_enabled: bool,
    kmer_length: usize,
    kmer_threshold: f64,
) -> PreparedBatch {
    let num_reads = batch.len();
    let num_refs = references.len();
    let num_pairs = num_reads * num_refs;

    // Generate reverse complement sequences for each read
    let rc_seqs: Vec<Vec<u8>> = batch.iter()
        .map(|rec| reverse_complement(&rec.seq))
        .collect();

    // Pack forward and reverse-complement query sequences for GPU upload,
    // converting lengths/offsets to i32 here so the GPU worker uploads directly.
    let (query_data, query_lengths_us, query_offsets_us) = pack_sequences(&batch);
    let (rc_query_data, rc_query_lengths_us, rc_query_offsets_us) = pack_sequences_raw(&rc_seqs);
    let query_lengths: Vec<i32> = query_lengths_us.iter().map(|&x| x as i32).collect();
    let query_offsets: Vec<i32> = query_offsets_us.iter().map(|&x| x as i32).collect();
    let rc_query_lengths: Vec<i32> = rc_query_lengths_us.iter().map(|&x| x as i32).collect();
    let rc_query_offsets: Vec<i32> = rc_query_offsets_us.iter().map(|&x| x as i32).collect();

    // Run k-mer pre-filter for both orientations
    let mut fwd_kmer_pass = vec![true; num_pairs];
    let mut rc_kmer_pass = vec![true; num_pairs];
    let mut kmer_fail_count = 0usize;

    if kmer_filter_enabled {
        for qi in 0..num_reads {
            let fwd_kmers = collect_kmers_owned(&batch[qi].seq, kmer_length);
            let rc_kmers = collect_kmers_owned(&rc_seqs[qi], kmer_length);

            for ri in 0..num_refs {
                let global_idx = qi * num_refs + ri;

                // Check forward orientation
                if !fwd_kmers.is_empty() {
                    let hits = fwd_kmers.iter()
                        .filter(|km| ref_kmer_sets[ri].contains(km.as_slice()))
                        .count();
                    let fraction = hits as f64 / fwd_kmers.len() as f64;
                    if fraction < kmer_threshold {
                        fwd_kmer_pass[global_idx] = false;
                    }
                }

                // Check reverse complement orientation
                if !rc_kmers.is_empty() {
                    let hits = rc_kmers.iter()
                        .filter(|km| ref_kmer_sets[ri].contains(km.as_slice()))
                        .count();
                    let fraction = hits as f64 / rc_kmers.len() as f64;
                    if fraction < kmer_threshold {
                        rc_kmer_pass[global_idx] = false;
                    }
                }

                // Count pairs where BOTH orientations fail
                if !fwd_kmer_pass[global_idx] && !rc_kmer_pass[global_idx] {
                    kmer_fail_count += 1;
                }
            }
        }
    }

    // Build pair indices for passing pairs (both orientations)
    let mut pair_query_idx = Vec::with_capacity(num_pairs * 2);
    let mut pair_ref_idx = Vec::with_capacity(num_pairs * 2);
    let mut pair_is_rc = Vec::with_capacity(num_pairs * 2);

    // Forward pairs
    let mut fwd_gpu_idx: Vec<Option<usize>> = vec![None; num_pairs];
    for qi in 0..num_reads {
        for ri in 0..num_refs {
            let global_idx = qi * num_refs + ri;
            if fwd_kmer_pass[global_idx] {
                fwd_gpu_idx[global_idx] = Some(pair_query_idx.len());
                pair_query_idx.push(qi as i32);
                pair_ref_idx.push(ri as i32);
                pair_is_rc.push(false);
            }
        }
    }

    // RC pairs
    let mut rc_gpu_idx: Vec<Option<usize>> = vec![None; num_pairs];
    for qi in 0..num_reads {
        for ri in 0..num_refs {
            let global_idx = qi * num_refs + ri;
            if rc_kmer_pass[global_idx] {
                rc_gpu_idx[global_idx] = Some(pair_query_idx.len());
                pair_query_idx.push(qi as i32);
                pair_ref_idx.push(ri as i32);
                pair_is_rc.push(true);
            }
        }
    }

    PreparedBatch {
        batch,
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
/// This is invoked from each GPU worker thread; `stream`, `kernel`, and the
/// `d_ref*` slices all belong to that worker's device context.
fn run_gpu_alignment(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernel: &cudarc::driver::CudaFunction,
    prepared: &PreparedBatch,
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
            stream, kernel,
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
            stream, kernel,
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

/// Build the TSV output rows for one batch from its GPU results. For each
/// read×reference pair, pick the best orientation; pairs the GPU could not
/// resolve within the score budget are transparently re-aligned on the CPU here.
/// Returns the rows (one per pair, unless filtered/skipped) and the number of
/// CPU fallbacks performed. Runs on a post-processing thread, off the GPU path.
fn format_batch_rows(
    result: &GpuBatchResult,
    references: &[SequenceRecord],
    header_tags: &[String],
    gap_open: i32,
    gap_extend: i32,
    mismatch: i32,
    max_score: usize,
    skip_fail: bool,
) -> (Vec<Vec<String>>, usize) {
    let prepared = &result.prepared;
    let all_scores = &result.scores;
    let all_cigar_ops = &result.cigar_ops;
    let all_cigar_lengths = &result.cigar_lengths;
    let num_refs = prepared.num_refs;

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut cpu_fallback_count = 0usize;
    for qi in 0..prepared.num_reads {
        let tag_values: Vec<String> = if !header_tags.is_empty() {
            header_tags.iter().map(|tag| {
                extract_tag_value(prepared.batch[qi].desc.as_deref(), tag)
            }).collect()
        } else {
            Vec::new()
        };

        for ri in 0..num_refs {
            let global_idx = qi * num_refs + ri;

            let fwd_passed = prepared.fwd_kmer_pass[global_idx];
            let rc_passed = prepared.rc_kmer_pass[global_idx];

            // If neither orientation passed the k-mer filter, report FAILED
            // (or skip entirely if --skip-fail is set)
            if !fwd_passed && !rc_passed {
                if !skip_fail {
                    let mut row = vec![
                        prepared.batch[qi].id.clone(),
                        references[ri].id.clone(),
                        "FAILED".to_string(),
                        ".".to_string(),
                    ];
                    for val in &tag_values {
                        row.push(val.clone());
                    }
                    rows.push(row);
                }
                continue;
            }

            // Get forward result (if it was aligned)
            let fwd_result = if fwd_passed {
                let gpu_idx = prepared.fwd_gpu_idx[global_idx]
                    .expect("forward-passed pair must have a GPU index");
                let score = all_scores[gpu_idx];
                let cigar_len = all_cigar_lengths[gpu_idx] as usize;
                if score >= 0 && cigar_len > 0 {
                    let ops: Vec<u8> = all_cigar_ops[gpu_idx]
                        .iter()
                        .map(|&x| x as u8)
                        .collect();
                    Some((score, ops_to_cigar(&ops)))
                } else {
                    cpu_fallback_count += 1;
                    debug!("CPU fallback (fwd) for read {} vs ref {}", prepared.batch[qi].id, references[ri].id);
                    let cigar = cpu_align(
                        &prepared.batch[qi].seq,
                        &references[ri].seq,
                        gap_open,
                        gap_extend,
                        mismatch,
                    );
                    if cigar == "*" {
                        None
                    } else {
                        Some((max_score as i32 + 1, cigar))
                    }
                }
            } else {
                None
            };

            // Get RC result (if it was aligned)
            let rc_result = if rc_passed {
                let gpu_idx = prepared.rc_gpu_idx[global_idx]
                    .expect("rc-passed pair must have a GPU index");
                let score = all_scores[gpu_idx];
                let cigar_len = all_cigar_lengths[gpu_idx] as usize;
                if score >= 0 && cigar_len > 0 {
                    let ops: Vec<u8> = all_cigar_ops[gpu_idx]
                        .iter()
                        .map(|&x| x as u8)
                        .collect();
                    Some((score, ops_to_cigar(&ops)))
                } else {
                    cpu_fallback_count += 1;
                    debug!("CPU fallback (rc) for read {} vs ref {}", prepared.batch[qi].id, references[ri].id);
                    let rc_seq = reverse_complement(&prepared.batch[qi].seq);
                    let cigar = cpu_align(
                        &rc_seq,
                        &references[ri].seq,
                        gap_open,
                        gap_extend,
                        mismatch,
                    );
                    if cigar == "*" {
                        None
                    } else {
                        Some((max_score as i32 + 1, cigar))
                    }
                }
            } else {
                None
            };

            // Pick the best orientation (lower score = better alignment)
            let (cigar, strand) = match (fwd_result, rc_result) {
                (None, None) => ("*".to_string(), "."),
                (Some((_score, cigar)), None) => (cigar, "+"),
                (None, Some((_score, cigar))) => (cigar, "-"),
                (Some((fwd_score, fwd_cigar)), Some((rc_score, rc_cigar))) => {
                    if fwd_score <= rc_score {
                        (fwd_cigar, "+")
                    } else {
                        (rc_cigar, "-")
                    }
                }
            };

            let mut row = vec![
                prepared.batch[qi].id.clone(),
                references[ri].id.clone(),
                cigar,
                strand.to_string(),
            ];
            for val in &tag_values {
                row.push(val.clone());
            }
            rows.push(row);
        }
    }

    (rows, cpu_fallback_count)
}

/// Launch the WFA alignment kernel for a set of pairs, handling sub-batching for GPU memory.
fn launch_alignment_kernel(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernel: &cudarc::driver::CudaFunction,
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
        let ws_mb = (total_ws * 4) / (1024 * 1024);
        debug!("GPU sub-batch {}: {} pairs, workspace {} MB", sub_batch_idx, sub_batch_size, ws_mb);

        let mut d_workspace: CudaSlice<i32> = stream.alloc_zeros(total_ws)?;
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
            .arg(&mut d_workspace)
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

/// Compute the reverse complement of a DNA sequence.
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

fn cpu_align(query: &[u8], reference: &[u8], gap_open: i32, gap_extend: i32, mismatch: i32) -> String {
    use std::collections::HashMap;

    let n = query.len() as i32;
    let m = reference.len() as i32;
    let target_k = n - m;

    let mut wf_m: HashMap<(i32, i32), i32> = HashMap::new();
    let mut wf_i: HashMap<(i32, i32), i32> = HashMap::new();
    let mut wf_d: HashMap<(i32, i32), i32> = HashMap::new();
    let mut bt: HashMap<(i32, i32), (i32, i32, u8, u8)> = HashMap::new();

    wf_m.insert((0, 0), 0);

    // Extend score 0
    if let Some(off) = wf_m.get_mut(&(0, 0)) {
        while *off < m && (*off) < n {
            if query[(*off) as usize] == reference[(*off) as usize] {
                *off += 1;
            } else {
                break;
            }
        }
    }

    if wf_m.get(&(0, target_k)).copied().unwrap_or(-1) >= m {
        return format!("{}=", n);
    }

    let mut final_score = -1i32;

    for s in 1..=4096i32 {
        // Insertions
        let ins_s = s - gap_open - gap_extend;
        if ins_s >= 0 {
            let entries: Vec<_> = wf_m.iter()
                .filter(|&(&(sc, _), _)| sc == ins_s)
                .map(|(&(_, k), &off)| (k, off)).collect();
            for (k, off) in entries {
                let nk = k + 1;
                if off > wf_i.get(&(s, nk)).copied().unwrap_or(-1) {
                    wf_i.insert((s, nk), off);
                }
            }
        }
        let ins_ext_s = s - gap_extend;
        if ins_ext_s >= 0 {
            let entries: Vec<_> = wf_i.iter()
                .filter(|&(&(sc, _), _)| sc == ins_ext_s)
                .map(|(&(_, k), &off)| (k, off)).collect();
            for (k, off) in entries {
                let nk = k + 1;
                if off > wf_i.get(&(s, nk)).copied().unwrap_or(-1) {
                    wf_i.insert((s, nk), off);
                }
            }
        }

        // Deletions
        let del_s = s - gap_open - gap_extend;
        if del_s >= 0 {
            let entries: Vec<_> = wf_m.iter()
                .filter(|&(&(sc, _), _)| sc == del_s)
                .map(|(&(_, k), &off)| (k, off)).collect();
            for (k, off) in entries {
                let nk = k - 1;
                let noff = off + 1;
                if noff > wf_d.get(&(s, nk)).copied().unwrap_or(-1) {
                    wf_d.insert((s, nk), noff);
                }
            }
        }
        let del_ext_s = s - gap_extend;
        if del_ext_s >= 0 {
            let entries: Vec<_> = wf_d.iter()
                .filter(|&(&(sc, _), _)| sc == del_ext_s)
                .map(|(&(_, k), &off)| (k, off)).collect();
            for (k, off) in entries {
                let nk = k - 1;
                let noff = off + 1;
                if noff > wf_d.get(&(s, nk)).copied().unwrap_or(-1) {
                    wf_d.insert((s, nk), noff);
                }
            }
        }

        // Mismatches
        let sub_s = s - mismatch;
        if sub_s >= 0 {
            let entries: Vec<_> = wf_m.iter()
                .filter(|&(&(sc, _), _)| sc == sub_s)
                .map(|(&(_, k), &off)| (k, off)).collect();
            for (k, off) in entries {
                if off < m && (off + k) >= 0 && (off + k) < n {
                    let noff = off + 1;
                    if noff > wf_m.get(&(s, k)).copied().unwrap_or(-1) {
                        wf_m.insert((s, k), noff);
                        bt.insert((s, k), (sub_s, k, OP_MISMATCH, 0));
                    }
                }
            }
        }

        // Merge I -> M
        let i_entries: Vec<_> = wf_i.iter()
            .filter(|&(&(sc, _), _)| sc == s)
            .map(|(&(_, k), &off)| (k, off)).collect();
        for (k, off) in i_entries {
            if off > wf_m.get(&(s, k)).copied().unwrap_or(-1) {
                wf_m.insert((s, k), off);
                let prev_k = k - 1;
                let ins_s = s - gap_open - gap_extend;
                let ins_ext_s = s - gap_extend;
                if ins_s >= 0 && wf_m.get(&(ins_s, prev_k)).copied().unwrap_or(-1) == off {
                    bt.insert((s, k), (ins_s, prev_k, OP_INSERTION, 0));
                } else if ins_ext_s >= 0 && wf_i.get(&(ins_ext_s, prev_k)).copied().unwrap_or(-1) == off {
                    bt.insert((s, k), (ins_ext_s, prev_k, OP_INSERTION, 1));
                } else {
                    bt.insert((s, k), (s, k, OP_INSERTION, 0));
                }
            }
        }

        // Merge D -> M
        let d_entries: Vec<_> = wf_d.iter()
            .filter(|&(&(sc, _), _)| sc == s)
            .map(|(&(_, k), &off)| (k, off)).collect();
        for (k, off) in d_entries {
            if off > wf_m.get(&(s, k)).copied().unwrap_or(-1) {
                wf_m.insert((s, k), off);
                let prev_k = k + 1;
                let del_s = s - gap_open - gap_extend;
                let del_ext_s = s - gap_extend;
                if del_s >= 0 && wf_m.get(&(del_s, prev_k)).copied().map(|o| o + 1).unwrap_or(-1) == off {
                    bt.insert((s, k), (del_s, prev_k, OP_DELETION, 0));
                } else if del_ext_s >= 0 && wf_d.get(&(del_ext_s, prev_k)).copied().map(|o| o + 1).unwrap_or(-1) == off {
                    bt.insert((s, k), (del_ext_s, prev_k, OP_DELETION, 2));
                } else {
                    bt.insert((s, k), (s, k, OP_DELETION, 0));
                }
            }
        }

        // Extend M
        let m_keys: Vec<_> = wf_m.iter()
            .filter(|&(&(sc, _), _)| sc == s)
            .map(|(&(_, k), _)| k).collect();
        for k in m_keys {
            if let Some(off) = wf_m.get_mut(&(s, k)) {
                while *off < m && (*off + k) >= 0 && (*off + k) < n {
                    if query[(*off + k) as usize] == reference[*off as usize] {
                        *off += 1;
                    } else {
                        break;
                    }
                }
            }
        }

        // Check target
        if let Some(&off) = wf_m.get(&(s, target_k)) {
            if off >= m {
                final_score = s;
                break;
            }
        }
    }

    if final_score < 0 {
        return String::from("*");
    }

    // Backtrack
    let mut ops = Vec::new();
    let mut cur_s = final_score;
    let mut cur_k = target_k;
    let mut cur_off = wf_m.get(&(cur_s, cur_k)).copied().unwrap_or(m);

    while cur_s > 0 {
        if let Some(&(prev_s, prev_k, op, src)) = bt.get(&(cur_s, cur_k)) {
            let prev_off = match src {
                0 => wf_m.get(&(prev_s, prev_k)).copied().unwrap_or(0),
                1 => wf_i.get(&(prev_s, prev_k)).copied().unwrap_or(0),
                2 => wf_d.get(&(prev_s, prev_k)).copied().unwrap_or(0),
                _ => 0,
            };
            let off_after = match op {
                OP_MISMATCH => prev_off + 1,
                OP_INSERTION => prev_off,
                OP_DELETION => prev_off + 1,
                _ => prev_off,
            };
            for _ in 0..(cur_off - off_after) { ops.push(OP_MATCH); }
            ops.push(op);
            cur_s = prev_s;
            cur_k = prev_k;
            cur_off = prev_off;
        } else {
            while cur_off > 0 { ops.push(OP_MATCH); cur_off -= 1; }
            break;
        }
    }
    while cur_off > 0 { ops.push(OP_MATCH); cur_off -= 1; }
    ops.reverse();
    ops_to_cigar(&ops)
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

/// Collect all unique k-mers from a sequence as owned `Vec<u8>`.
fn collect_kmers_owned(seq: &[u8], k: usize) -> HashSet<Vec<u8>> {
    let mut set = HashSet::new();
    if seq.len() >= k {
        for window in seq.windows(k) {
            set.insert(window.to_vec());
        }
    }
    set
}

// ============================================================
// Sequence packing for GPU transfer
// ============================================================

fn pack_sequences(records: &[SequenceRecord]) -> (Vec<i8>, Vec<usize>, Vec<usize>) {
    let mut data = Vec::new();
    let mut lengths = Vec::new();
    let mut offsets = Vec::new();
    for rec in records {
        offsets.push(data.len());
        lengths.push(rec.seq.len());
        for &b in &rec.seq {
            data.push(b as i8);
        }
    }
    (data, lengths, offsets)
}

/// Pack raw byte sequences (e.g., reverse complements) for GPU transfer.
fn pack_sequences_raw(seqs: &[Vec<u8>]) -> (Vec<i8>, Vec<usize>, Vec<usize>) {
    let mut data = Vec::new();
    let mut lengths = Vec::new();
    let mut offsets = Vec::new();
    for seq in seqs {
        offsets.push(data.len());
        lengths.push(seq.len());
        for &b in seq {
            data.push(b as i8);
        }
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

struct FastaBatchReader {
    reader: fasta::Reader<BufReader<Box<dyn Read>>>,
}

impl FastaBatchReader {
    fn new(reader: Box<dyn Read>) -> Self {
        Self { reader: fasta::Reader::from_bufread(BufReader::new(reader)) }
    }
}

impl BatchReader for FastaBatchReader {
    fn read_batch(&mut self, batch_size: usize) -> Result<Vec<SequenceRecord>> {
        let mut batch = Vec::with_capacity(batch_size);
        let mut record = fasta::Record::new();
        for _ in 0..batch_size {
            self.reader.read(&mut record)?;
            if record.is_empty() { break; }
            batch.push(SequenceRecord {
                id: record.id().to_string(),
                desc: record.desc().map(|d| d.to_string()),
                seq: record.seq().to_vec(),
            });
        }
        Ok(batch)
    }
}

struct FastqBatchReader {
    reader: fastq::Reader<BufReader<Box<dyn Read>>>,
}

impl FastqBatchReader {
    fn new(reader: Box<dyn Read>) -> Self {
        Self { reader: fastq::Reader::from_bufread(BufReader::new(reader)) }
    }
}

impl BatchReader for FastqBatchReader {
    fn read_batch(&mut self, batch_size: usize) -> Result<Vec<SequenceRecord>> {
        let mut batch = Vec::with_capacity(batch_size);
        let mut record = fastq::Record::new();
        for _ in 0..batch_size {
            self.reader.read(&mut record)?;
            if record.is_empty() { break; }
            batch.push(SequenceRecord {
                id: record.id().to_string(),
                desc: record.desc().map(|d| d.to_string()),
                seq: record.seq().to_vec(),
            });
        }
        Ok(batch)
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
