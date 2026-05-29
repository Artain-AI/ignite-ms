//! IgniteMS Benchmark — measures raw embedding throughput.
//!
//! Reads plain text input (one text per line, or JSONL with "text" field),
//! feeds to the engine, measures sustained msg/s. No aggregation, no PCA.
//!
//! Usage:
//!     ignite-ms-bench --input texts.jsonl --model-dir /opt/ignite-ms/model
//!     ignite-ms-bench --input texts.txt --format plain --model-dir ./model

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use ignite_ms::{Config, Engine, Message};

/// IgniteMS Benchmark — raw embedding throughput measurement.
#[derive(Parser)]
#[command(name = "ignite-ms-bench", version)]
struct Cli {
    /// Input file(s): JSONL with "text" field, or plain text (one per line)
    #[arg(long, num_args = 1..)]
    input: Vec<String>,

    /// Input format: "jsonl" (default, reads "text" field) or "plain" (raw lines)
    #[arg(long, default_value = "jsonl")]
    format: String,

    /// Model directory (must contain model.onnx/model.engine + tokenizer.json)
    #[arg(long, default_value = "/opt/ignite-ms/model")]
    model_dir: PathBuf,

    /// Model name to provision under --model-dir, e.g. intfloat/e5-small-v2
    #[arg(long)]
    model: Option<String>,

    /// GPU IDs (comma-separated, or "all")
    #[arg(long, default_value = "all")]
    gpus: String,

    /// Max characters per message
    #[arg(long, default_value = "512")]
    truncation: usize,

    /// Limit number of messages (0 = unlimited)
    #[arg(long, default_value = "0")]
    max_messages: u64,

    /// Tokenizer workers
    #[arg(long, default_value = "8")]
    tokenize_workers: usize,

    /// Warm-up messages (not counted in throughput)
    #[arg(long, default_value = "10000")]
    warmup: u64,

    /// Enable latency profiling (records per-batch timing, reports percentiles)
    #[arg(long, default_value = "false")]
    latency: bool,

    /// Enable INT8 quantization (FP16+INT8 mixed precision). Default is FP16-only.
    #[arg(long, default_value = "false")]
    int8: bool,

    /// Count-only mode: do not materialize embedding payloads into Rust-owned Vec<f32>.
    /// Still runs full inference and waits for output readiness; useful to isolate
    /// Rust-side output handling overhead from GPU/TRT time.
    #[arg(long, default_value = "false")]
    count_only: bool,

    /// Optional comma-separated bucket boundaries override, e.g. "24,32,64,128,256"
    #[arg(long)]
    buckets: Option<String>,

    /// Optional comma-separated batch size override matching --buckets length.
    #[arg(long)]
    batch_sizes: Option<String>,

    /// Batch timeout in ms. Fires partial batches if bucket doesn't fill within this time.
    /// 0 = disabled. Default: 5ms.
    #[arg(long, default_value = "5")]
    batch_timeout_ms: u64,

    /// Embedding dedup cache capacity. Identical normalized texts reuse cached embeddings.
    /// 0 = disabled. Recommended: 10000000 for datasets with repetitive text.
    #[arg(long, default_value = "0")]
    dedup_capacity: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let t_start = Instant::now();

    println!("IgniteMS Benchmark v{}", env!("CARGO_PKG_VERSION"));
    println!("================================================================");

    // Resolve input files
    let files = resolve_inputs(&cli.input)?;
    println!("  input:       {} files", files.len());
    println!("  format:      {}", cli.format);
    println!("  model:       {}", cli.model_dir.display());
    println!("  truncation:  {} chars", cli.truncation);
    println!(
        "  max_msgs:    {}",
        if cli.max_messages == 0 {
            "unlimited".into()
        } else {
            format!("{}", cli.max_messages)
        }
    );
    println!("  warmup:      {} msgs", cli.warmup);
    println!("  count_only:  {}", cli.count_only);
    if let Some(ref buckets) = cli.buckets {
        println!("  buckets:     {}", buckets);
    }
    if let Some(ref batch_sizes) = cli.batch_sizes {
        println!("  batch_sizes: {}", batch_sizes);
    }
    println!();

    let bucket_override = cli.buckets.as_deref().map(parse_usize_csv).transpose()?;
    let batch_size_override = cli
        .batch_sizes
        .as_deref()
        .map(parse_usize_csv)
        .transpose()?;
    if let (Some(ref buckets), Some(ref batch_sizes)) = (&bucket_override, &batch_size_override) {
        if buckets.len() != batch_sizes.len() {
            return Err(format!(
                "--buckets/--batch-sizes length mismatch: {} vs {}",
                buckets.len(),
                batch_sizes.len()
            )
            .into());
        }
    }

    // Initialize engine
    let t_init = Instant::now();
    let mut config = Config {
        model: cli.model.clone(),
        model_dir: cli.model_dir.clone(),
        gpus: parse_gpus(&cli.gpus),
        truncation: cli.truncation,
        tokenize_workers: cli.tokenize_workers,
        int8: cli.int8,
        batch_timeout_ms: cli.batch_timeout_ms,
        dedup_capacity: cli.dedup_capacity,
        ..Default::default()
    };
    if let Some(buckets) = bucket_override {
        config.buckets = buckets;
    }
    if let Some(batch_sizes) = batch_size_override {
        config.batch_sizes = batch_sizes;
    }
    let mut engine = Engine::new(config)?;
    println!(
        "[init] {:.2}s (vocab_cache={} entries)",
        t_init.elapsed().as_secs_f64(),
        engine.vocab_cache_size()
    );
    println!();

    // Read files → messages
    let max_msgs = if cli.max_messages == 0 {
        u64::MAX
    } else {
        cli.max_messages
    };
    let format = cli.format.clone();
    let n_readers = files.len().min(32).max(4);

    let (msg_tx, msg_rx) = crossbeam::channel::bounded::<Message>(8192);
    let msg_count = Arc::new(AtomicU64::new(0));

    {
        let (ftx, frx) = crossbeam::channel::bounded::<PathBuf>(files.len());
        for f in &files {
            let _ = ftx.send(f.clone());
        }
        drop(ftx);

        for _ in 0..n_readers {
            let frx = frx.clone();
            let tx = msg_tx.clone();
            let gc = Arc::clone(&msg_count);
            let max = max_msgs;
            let fmt = format.clone();

            std::thread::spawn(move || {
                let mut idx: u64 = 0;
                while let Ok(path) = frx.recv() {
                    let f = match std::fs::File::open(&path) {
                        Ok(f) => f,
                        Err(_) => continue,
                    };
                    let reader: Box<dyn BufRead + Send> =
                        if path.extension().map(|e| e == "zst").unwrap_or(false) {
                            match zstd::stream::Decoder::new(f) {
                                Ok(d) => Box::new(BufReader::with_capacity(4 * 1024 * 1024, d)),
                                Err(_) => continue,
                            }
                        } else {
                            Box::new(BufReader::with_capacity(4 * 1024 * 1024, f))
                        };

                    for line in reader.lines() {
                        let line = match line {
                            Ok(l) => l,
                            Err(_) => continue,
                        };
                        if gc.load(Ordering::Relaxed) >= max {
                            return;
                        }

                        let text = match fmt.as_str() {
                            "plain" => {
                                let trimmed = line.trim().to_string();
                                if trimmed.is_empty() {
                                    continue;
                                }
                                trimmed
                            }
                            _ => {
                                // JSONL: extract "text" field
                                match serde_json::from_str::<serde_json::Value>(&line) {
                                    Ok(v) => match v.get("text").and_then(|t| t.as_str()) {
                                        Some(t) if !t.trim().is_empty() => t.trim().to_string(),
                                        _ => continue,
                                    },
                                    Err(_) => continue,
                                }
                            }
                        };

                        idx += 1;
                        let msg = Message {
                            id: idx.to_string(),
                            text,
                        };
                        gc.fetch_add(1, Ordering::Relaxed);
                        if tx.send(msg).is_err() {
                            return;
                        }
                    }
                }
            });
        }
        drop(frx);
    }
    drop(msg_tx);
    println!("[bench] readers: {} threads", n_readers);

    // Feed to engine
    let (rx, handle) = engine.embed(msg_rx.into_iter())?;

    // Consume batches (discard embeddings — just measure throughput)
    let warmup = cli.warmup;
    let record_latency = cli.latency;
    let mut total: u64 = 0;
    let mut t_bench_start: Option<Instant> = None;
    let mut post_warmup: u64 = 0;
    let mut last_batch_time: Option<Instant> = None;
    // (batch_size, latency_ms)
    let mut latencies: Vec<(u32, f64)> = if record_latency {
        Vec::with_capacity(16384)
    } else {
        Vec::new()
    };

    for batch in rx {
        total += batch.n_rows as u64;
        if total >= warmup && t_bench_start.is_none() {
            t_bench_start = Some(Instant::now());
            last_batch_time = Some(Instant::now());
            println!("[bench] warm-up done ({} msgs), measuring...", total);
        }
        if t_bench_start.is_some() {
            post_warmup += batch.n_rows as u64;
            if record_latency {
                if let Some(prev) = last_batch_time {
                    let now = Instant::now();
                    let elapsed_ms = (now - prev).as_secs_f64() * 1000.0;
                    latencies.push((batch.n_rows as u32, elapsed_ms));
                    last_batch_time = Some(now);
                }
            }
        }
    }

    let stats = handle.finish();
    let bench_elapsed = t_bench_start
        .map(|t| t.elapsed().as_secs_f64())
        .unwrap_or(1.0);
    let bench_throughput = post_warmup as f64 / bench_elapsed;

    println!();
    println!("================================================================");
    println!("  RESULTS");
    println!("================================================================");
    println!("  total_messages:    {}", total);
    println!("  warmup_discarded:  {}", warmup.min(total));
    println!("  measured_messages: {}", post_warmup);
    println!("  measured_time:     {:.2}s", bench_elapsed);
    println!("  throughput:        {:.0} msg/s", bench_throughput);
    println!("  engine_reported:   {:.0} msg/s", stats.throughput());
    println!(
        "  cache_hit_rate:    {:.1}%",
        stats.cache_hit_rate() * 100.0
    );
    println!(
        "  total_wall_clock:  {:.2}s",
        t_start.elapsed().as_secs_f64()
    );

    if record_latency && !latencies.is_empty() {
        let mut lat_ms: Vec<f64> = latencies.iter().map(|(_, ms)| *ms).collect();
        lat_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = lat_ms.len();
        let p50 = lat_ms[n / 2];
        let p95 = lat_ms[(n as f64 * 0.95) as usize];
        let p99 = lat_ms[(n as f64 * 0.99) as usize];
        let avg_batch: f64 = latencies.iter().map(|(bs, _)| *bs as f64).sum::<f64>() / n as f64;

        println!("  --- latency ---");
        println!("  latency_samples:   {}", n);
        println!("  avg_batch_size:    {:.1}", avg_batch);
        println!("  latency_p50:       {:.2} ms", p50);
        println!("  latency_p95:       {:.2} ms", p95);
        println!("  latency_p99:       {:.2} ms", p99);
        println!("  latency_min:       {:.2} ms", lat_ms[0]);
        println!("  latency_max:       {:.2} ms", lat_ms[n - 1]);

        // Machine-readable output
        println!("LATENCY_P50={:.4}", p50);
        println!("LATENCY_P95={:.4}", p95);
        println!("LATENCY_P99={:.4}", p99);
        println!("LATENCY_SAMPLES={}", n);
        println!("LATENCY_AVG_BATCH_SIZE={:.1}", avg_batch);
    }

    println!("================================================================");

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn resolve_inputs(inputs: &[String]) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files: Vec<PathBuf> = Vec::new();
    for input in inputs {
        let path = PathBuf::from(input);
        if path.is_file() {
            files.push(path.canonicalize().unwrap_or(path));
        } else if path.is_dir() {
            collect_recursive(&path, &mut files);
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn collect_recursive(dir: &PathBuf, files: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            collect_recursive(&path, files);
        } else if path.is_file() {
            files.push(path.canonicalize().unwrap_or(path));
        }
    }
}

fn parse_gpus(spec: &str) -> Option<Vec<u32>> {
    if spec == "all" {
        return None;
    }
    Some(
        spec.split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect(),
    )
}

fn parse_usize_csv(raw: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let values = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err("empty comma-separated list".into());
    }
    Ok(values)
}
