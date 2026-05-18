use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use ignite_ms::{Config, Engine};
use indicatif::{ProgressBar, ProgressStyle};

use crate::input::{self, InputFormat};
use crate::output;
use crate::reorder::ReorderBuffer;

#[derive(Args)]
pub struct EmbedArgs {
    /// Model name (e.g. "intfloat/e5-small-v2") or path to local model directory
    #[arg(long, short = 'm')]
    model: String,

    /// Input file(s) — JSONL or plain text. Supports .zst and .gz compression.
    #[arg(long, short = 'i', num_args = 1..)]
    input: Vec<String>,

    /// Output file path. Format auto-detected: .npy (default) or .parquet
    #[arg(long, short = 'o', default_value = "embeddings.npy")]
    output: String,

    /// Input format: "jsonl" (default) or "plain" (one text per line)
    #[arg(long, default_value = "jsonl", value_parser = parse_format)]
    format: String,

    /// GPU IDs: "all" (default) or comma-separated (e.g. "0,1,2")
    #[arg(long, default_value = "all")]
    gpus: String,

    /// Model/engine cache directory
    #[arg(long, default_value = "/opt/ignite-ms/cache")]
    cache_dir: String,

    /// Max characters per text (longer texts truncated)
    #[arg(long, default_value = "512")]
    truncation: usize,

    /// Number of tokenizer worker threads
    #[arg(long, default_value = "8")]
    tokenize_workers: usize,

    /// Text prefix (e.g. "passage: " for e5 models). Omit to use model default.
    #[arg(long)]
    prefix: Option<String>,

    /// Enable INT8 mixed-precision inference
    #[arg(long)]
    int8: bool,

    /// S3 prefix for engine cache (optional, avoids local recompilation)
    #[arg(long)]
    engine_cache: Option<String>,

    /// Embedding dedup cache capacity (0 = disabled)
    #[arg(long, default_value = "0")]
    dedup_capacity: usize,

    /// Batch timeout in ms (0 = disabled)
    #[arg(long, default_value = "5")]
    batch_timeout_ms: u64,

    /// Optional comma-separated bucket boundaries override, e.g. "32,64,128,256"
    #[arg(long)]
    buckets: Option<String>,

    /// Optional comma-separated batch size override matching --buckets length.
    #[arg(long)]
    batch_sizes: Option<String>,

    /// Suppress progress output
    #[arg(long)]
    quiet: bool,
}

pub fn run(args: EmbedArgs) -> Result<(), Box<dyn std::error::Error>> {
    let t_start = Instant::now();

    eprintln!("ignite-ms embed v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("================================================================");

    let files = input::resolve_inputs(&args.input)?;
    if files.is_empty() {
        return Err("no input files found".into());
    }
    eprintln!("  model:       {}", args.model);
    eprintln!("  input:       {} file(s)", files.len());
    eprintln!("  output:      {}", args.output);
    eprintln!("  truncation:  {} chars", args.truncation);
    if let Some(ref buckets) = args.buckets {
        eprintln!("  buckets:     {}", buckets);
    }
    if let Some(ref batch_sizes) = args.batch_sizes {
        eprintln!("  batch_sizes: {}", batch_sizes);
    }

    let model_dir = PathBuf::from(&args.cache_dir);
    let model_name = if args.model.contains('/') || std::path::Path::new(&args.model).exists() {
        Some(args.model.clone())
    } else {
        Some(format!("intfloat/{}", args.model))
    };

    let bucket_override = args.buckets.as_deref().map(parse_usize_csv).transpose()?;
    let batch_size_override = args
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

    let mut config = Config {
        model: model_name,
        model_dir,
        engine_cache: args.engine_cache.clone(),
        gpus: parse_gpus(&args.gpus).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?,
        truncation: args.truncation,
        tokenize_workers: args.tokenize_workers,
        int8: args.int8,
        batch_timeout_ms: args.batch_timeout_ms,
        dedup_capacity: args.dedup_capacity,
        prefix: args.prefix.unwrap_or_default(),
        telemetry: true,
        ..Default::default()
    };
    if let Some(buckets) = bucket_override {
        config.buckets = buckets;
    }
    if let Some(batch_sizes) = batch_size_override {
        config.batch_sizes = batch_sizes;
    }

    let t_init = Instant::now();
    let mut engine = Engine::new(config)?;
    let hidden_dim = engine.hidden_dim();
    eprintln!(
        "  init:        {:.1}s ({}D, vocab_cache={})",
        t_init.elapsed().as_secs_f64(),
        hidden_dim,
        engine.vocab_cache_size(),
    );
    eprintln!();

    let format = match args.format.as_str() {
        "plain" => InputFormat::Plain,
        _ => InputFormat::Jsonl,
    };

    let (msg_iter, user_ids) = input::read_sequential(&files, format)?;

    let (rx, handle) = engine.embed(msg_iter)?;

    let mut writer = output::create_writer(&args.output, hidden_dim)?;
    writer.begin(hidden_dim)?;

    let pb = if !args.quiet {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        Some(pb)
    } else {
        None
    };

    let total_input = user_ids.len() as u64;
    let mut reorder = ReorderBuffer::new(hidden_dim);
    let mut n_written: u64 = 0;
    let mut n_inference_errors: u64 = 0;
    let t_embed = Instant::now();

    for batch in rx {
        if batch.inference_failed {
            n_inference_errors += batch.ids.len() as u64;
            for id in &batch.ids {
                let seq: u64 = id.parse().unwrap_or(0);
                reorder.mark_skipped(seq);
            }
        } else if batch.n_rows == 0 {
            for id in &batch.ids {
                let seq: u64 = id.parse().unwrap_or(0);
                reorder.mark_skipped(seq);
            }
        }
        for i in 0..batch.n_rows {
            let seq: u64 = batch.ids[i].parse().unwrap_or(0);
            let embedding = batch.embedding(i);
            reorder.insert(seq, embedding);
        }

        while let Some((seq, emb)) = reorder.pop_next() {
            let user_id = user_ids.get(seq as usize).map(|s| s.as_str());
            writer.write_row(seq, user_id, &emb)?;
            n_written += 1;
        }

        if let Some(ref pb) = pb {
            let elapsed = t_embed.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                let rate = n_written as f64 / elapsed;
                pb.set_message(format!(
                    "{} embedded | {:.0} msg/s | {:.1}s",
                    n_written, rate, elapsed
                ));
                pb.tick();
            }
        }
    }

    // Mark gaps as skipped (engine dropped these during normalization)
    // so the reorder buffer can advance past them.
    for seq in reorder.next_seq()..total_input {
        if !reorder.has_pending(seq) {
            reorder.mark_skipped(seq);
        }
    }

    // Flush remaining reorder entries
    while let Some((seq, emb)) = reorder.pop_next() {
        let user_id = user_ids.get(seq as usize).map(|s| s.as_str());
        writer.write_row(seq, user_id, &emb)?;
        n_written += 1;
    }

    writer.finish(n_written)?;

    if let Some(ref pb) = pb {
        pb.finish_and_clear();
    }

    let stats = handle.finish();
    let elapsed = t_start.elapsed().as_secs_f64();
    let embed_elapsed = t_embed.elapsed().as_secs_f64();

    eprintln!();
    eprintln!("================================================================");
    eprintln!("  Embedded:    {} messages", n_written);
    eprintln!("  Skipped:     {} (short/empty)", stats.messages_skipped);
    if n_inference_errors > 0 {
        eprintln!(
            "  ERRORS:      {} messages lost to inference failures",
            n_inference_errors
        );
    }
    eprintln!(
        "  Throughput:  {:.0} msg/s",
        n_written as f64 / embed_elapsed
    );
    eprintln!("  Output:      {}", args.output);
    eprintln!("  Total time:  {:.1}s", elapsed);
    eprintln!("================================================================");

    if n_inference_errors > 0 {
        return Err(format!(
            "{} messages lost to inference failures (output is partial)",
            n_inference_errors
        )
        .into());
    }

    Ok(())
}

fn parse_format(s: &str) -> Result<String, String> {
    match s {
        "jsonl" | "plain" => Ok(s.to_string()),
        _ => Err(format!(
            "invalid format {:?} (expected \"jsonl\" or \"plain\")",
            s
        )),
    }
}

fn parse_gpus(spec: &str) -> Result<Option<Vec<u32>>, String> {
    if spec == "all" {
        return Ok(None);
    }
    let mut ids = Vec::new();
    for part in spec.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let id: u32 = trimmed
            .parse()
            .map_err(|_| format!("invalid GPU id: {:?}", trimmed))?;
        ids.push(id);
    }
    if ids.is_empty() {
        return Err("--gpus requires at least one GPU id or \"all\"".into());
    }
    Ok(Some(ids))
}

fn parse_usize_csv(raw: &str) -> Result<Vec<usize>, String> {
    let values = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err("empty comma-separated list".into());
    }
    Ok(values)
}
