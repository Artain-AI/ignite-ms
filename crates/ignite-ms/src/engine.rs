//! Core engine — the public interface of ignite-ms.
//!
//! Owns: model, tokenizer, vocab cache, GPU sessions.
//! Does: normalize → tokenize → batch → GPU inference → D2H → deliver batches.
//! Does NOT: read files, parse JSON, know about sources, aggregate.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crossbeam::channel::{bounded, Receiver, Sender};
use tokenizers::Tokenizer;

use crate::bucket::{Batch, BucketSpec, Bucketizer};
use crate::cache;
use crate::error::Error;
use crate::normalize::{Normalizer, NormalizerConfig};
use crate::tokenize::{CachedTokenizer, TokenizedMessage};

#[cfg(feature = "native-trt")]
use crate::inference::TrtSession;

// ─── Public Types ────────────────────────────────────────────────────────────

/// Configuration for the embedding engine.
pub struct Config {
    /// Model name (e.g. "intfloat/multilingual-e5-small").
    /// If set, ignite-ms downloads tokenizer + config from HuggingFace, downloads raw ONNX
    /// from Xenova's HF repo, and appends mean-pooling + L2-norm automatically.
    /// All files are cached under model_dir.
    pub model: Option<String>,
    /// S3 prefix for pre-compiled TRT engine cache (e.g. "s3://bucket/engines/").
    /// Engines are keyed by GPU architecture + bucket config. Optional — without this,
    /// engines are compiled locally on every fresh instance (3-5 min cold start).
    pub engine_cache: Option<String>,
    /// Local cache directory for model files and compiled engines.
    /// If `model` is set, files are downloaded here automatically.
    /// If `model` is None, must contain tokenizer.json + model.onnx already.
    pub model_dir: PathBuf,
    /// GPU device IDs to use. None = auto-detect all available GPUs.
    pub gpus: Option<Vec<u32>>,
    /// Sequence length buckets for batching (e.g. [32, 64, 128, 256]).
    pub buckets: Vec<usize>,
    /// Max batch size per bucket (matched positionally with `buckets`).
    pub batch_sizes: Vec<usize>,
    /// Number of tokenizer worker threads.
    pub tokenize_workers: usize,
    /// Max characters per input text (truncated before tokenization).
    pub truncation: usize,
    /// Lowercase text during normalization.
    pub lowercase: bool,
    /// Minimum text length after normalization (messages shorter than this are skipped).
    pub min_chars: usize,
    /// Text prefix prepended after normalization (e.g. "passage: " for e5 models).
    pub prefix: String,
    /// Enable INT8 quantization (FP16+INT8 mixed precision). ~2x throughput on compute-bound models.
    /// Default: true. TRT dynamically selects FP16 or INT8 per layer for optimal speed/accuracy.
    pub int8: bool,
    /// Emit structured JSON telemetry to stderr on finish.
    pub telemetry: bool,
    /// Batch timeout in milliseconds. If a bucket hasn't filled a full batch within this time,
    /// fire a partial batch. Improves batch fill when text lengths are unevenly distributed.
    /// 0 = disabled (only fire on full batches). Default: 5ms.
    pub batch_timeout_ms: u64,
    /// Embedding dedup cache capacity. After normalization, identical texts reuse cached
    /// embeddings instead of re-tokenizing and re-running GPU inference. This is transparent
    /// to the client — every message ID still gets a result.
    /// 0 = disabled. Recommended: 10_000_000 for large runs with repetitive data.
    pub dedup_capacity: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: None,
            engine_cache: None,
            model_dir: PathBuf::from("/opt/ignite-ms/cache"),
            gpus: None,
            buckets: vec![24, 32, 40, 48, 56, 64, 80, 96, 128, 256],
            batch_sizes: vec![4096, 4096, 3072, 2048, 2048, 2048, 1536, 1536, 1024, 512],
            tokenize_workers: 8,
            truncation: 256,
            lowercase: true,
            min_chars: 10,
            prefix: "passage: ".to_string(),
            int8: false,
            telemetry: true,
            batch_timeout_ms: 5,
            dedup_capacity: 0,
        }
    }
}

/// A completed batch of embeddings.
pub struct EmbeddingBatch {
    pub ids: Vec<String>,
    pub embeddings: Vec<f32>,
    pub n_rows: usize,
    pub hidden_dim: usize,
    /// True if this batch has n_rows == 0 due to an inference error (not input filtering).
    pub inference_failed: bool,
}

impl EmbeddingBatch {
    #[inline]
    pub fn embedding(&self, i: usize) -> &[f32] {
        let start = i * self.hidden_dim;
        &self.embeddings[start..start + self.hidden_dim]
    }
}

/// A text message to embed.
pub struct Message {
    pub id: String,
    pub text: String,
}

/// Statistics from an embed run.
#[derive(Debug, Clone)]
pub struct EmbedStats {
    pub messages_processed: u64,
    pub messages_skipped: u64,
    pub batches_computed: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub runtime_cache_hits: u64,
    pub runtime_cache_size: u64,
    pub dedup_hits: u64,
    pub dedup_size: u64,
    pub elapsed_secs: f64,
    pub per_gpu_batches: Vec<u64>,
    pub per_bucket_batches: Vec<u64>,
    pub per_bucket_messages: Vec<u64>,
    pub inference_latency_ms: LatencyStats,
    pub avg_batch_fill: f64,
    pub n_gpus: usize,
    pub n_buckets: usize,
}

/// Latency percentile stats (milliseconds).
#[derive(Debug, Clone, Default)]
pub struct LatencyStats {
    pub min: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub mean: f64,
    pub count: u64,
}

impl EmbedStats {
    pub fn throughput(&self) -> f64 {
        self.messages_processed as f64 / self.elapsed_secs.max(1e-9)
    }
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
    pub fn to_json(&self) -> String {
        let per_gpu: Vec<String> = self.per_gpu_batches.iter().map(|b| b.to_string()).collect();
        let per_bucket_b: Vec<String> = self
            .per_bucket_batches
            .iter()
            .map(|b| b.to_string())
            .collect();
        let per_bucket_m: Vec<String> = self
            .per_bucket_messages
            .iter()
            .map(|b| b.to_string())
            .collect();
        format!(
            concat!(
                "{{",
                "\"messages_processed\":{},",
                "\"messages_skipped\":{},",
                "\"batches_computed\":{},",
                "\"elapsed_secs\":{:.3},",
                "\"throughput\":{:.0},",
                "\"cache_hit_rate\":{:.4},",
                "\"cache_hits\":{},",
                "\"cache_misses\":{},",
                "\"runtime_cache_hits\":{},",
                "\"runtime_cache_size\":{},",
                "\"dedup_hits\":{},",
                "\"dedup_size\":{},",
                "\"n_gpus\":{},",
                "\"n_buckets\":{},",
                "\"per_gpu_batches\":[{}],",
                "\"per_bucket_batches\":[{}],",
                "\"per_bucket_messages\":[{}],",
                "\"avg_batch_fill\":{:.3},",
                "\"inference_latency_ms\":{{\"min\":{:.2},\"p50\":{:.2},\"p95\":{:.2},\"p99\":{:.2},\"max\":{:.2},\"mean\":{:.2},\"count\":{}}}",
                "}}"
            ),
            self.messages_processed,
            self.messages_skipped,
            self.batches_computed,
            self.elapsed_secs,
            self.throughput(),
            self.cache_hit_rate(),
            self.cache_hits,
            self.cache_misses,
            self.runtime_cache_hits,
            self.runtime_cache_size,
            self.dedup_hits,
            self.dedup_size,
            self.n_gpus,
            self.n_buckets,
            per_gpu.join(","),
            per_bucket_b.join(","),
            per_bucket_m.join(","),
            self.avg_batch_fill,
            self.inference_latency_ms.min,
            self.inference_latency_ms.p50,
            self.inference_latency_ms.p95,
            self.inference_latency_ms.p99,
            self.inference_latency_ms.max,
            self.inference_latency_ms.mean,
            self.inference_latency_ms.count,
        )
    }
}

// ─── Embedding dedup cache ───────────────────────────────────────────────────
//
// Fixed-size direct-mapped cache (like CPU L1/L2). Pre-allocated contiguous memory,
// no locks, no heap allocations at runtime. Collisions overwrite (lossy).
// slot = hash % capacity. Each slot stores the key (AtomicU64) + embedding data.
//
// Intentionally approximate: a concurrent overwrite between key-check and data-read
// can return stale data from a different entry. This is acceptable — same tradeoff
// as CPU cache lines. Worst case: one embedding gets a neighbor's vector (both valid
// embeddings, just from a different text). No UB, no crash, negligible impact at scale.

struct EmbedCacheInner {
    keys: Vec<AtomicU64>,
    data: Vec<AtomicU32>,
    capacity: usize,
    hidden_dim: usize,
}

impl EmbedCacheInner {
    fn new(capacity: usize, hidden_dim: usize) -> Self {
        Self {
            keys: (0..capacity).map(|_| AtomicU64::new(0)).collect(),
            data: (0..capacity * hidden_dim)
                .map(|_| AtomicU32::new(0))
                .collect(),
            capacity,
            hidden_dim,
        }
    }

    #[inline]
    fn get(&self, hash: u64, out: &mut [f32]) -> bool {
        let slot = (hash % self.capacity as u64) as usize;
        let stored = self.keys[slot].load(Ordering::Acquire);
        if stored == hash && hash != 0 {
            let offset = slot * self.hidden_dim;
            for i in 0..self.hidden_dim {
                out[i] = f32::from_bits(self.data[offset + i].load(Ordering::Relaxed));
            }
            true
        } else {
            false
        }
    }

    #[inline]
    fn insert(&self, hash: u64, embedding: &[f32]) {
        if hash == 0 {
            return;
        }
        let slot = (hash % self.capacity as u64) as usize;
        let offset = slot * self.hidden_dim;
        debug_assert!(offset + self.hidden_dim <= self.data.len());
        for i in 0..self.hidden_dim {
            self.data[offset + i].store(embedding[i].to_bits(), Ordering::Relaxed);
        }
        self.keys[slot].store(hash, Ordering::Release);
    }

    fn occupancy(&self) -> usize {
        self.keys
            .iter()
            .filter(|k| k.load(Ordering::Relaxed) != 0)
            .count()
    }
}

unsafe impl Sync for EmbedCacheInner {}
unsafe impl Send for EmbedCacheInner {}

type EmbedCache = Arc<EmbedCacheInner>;

#[inline]
fn fnv1a(text: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn detect_gpu_count() -> u32 {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index", "--format=csv,noheader"])
        .output();
    match output {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            s.lines().filter(|l| !l.trim().is_empty()).count() as u32
        }
        _ => 1,
    }
}

fn resolve_gpus(configured: &Option<Vec<u32>>) -> Vec<u32> {
    match configured {
        Some(gpus) => gpus.clone(),
        None => {
            let n = detect_gpu_count();
            (0..n).collect()
        }
    }
}

// ─── Internal telemetry ─────────────────────────────────────────────────────

struct GpuWorkerStats {
    batches: u64,
    total_rows: u64,
    total_capacity: u64,
    latencies_us: Vec<u64>,
    per_bucket_batches: Vec<u64>,
    per_bucket_messages: Vec<u64>,
}

fn compute_latency_stats(mut latencies_us: Vec<u64>) -> LatencyStats {
    if latencies_us.is_empty() {
        return LatencyStats::default();
    }
    latencies_us.sort_unstable();
    let n = latencies_us.len();
    let sum: u64 = latencies_us.iter().sum();
    LatencyStats {
        min: latencies_us[0] as f64 / 1000.0,
        p50: latencies_us[n / 2] as f64 / 1000.0,
        p95: latencies_us[(n as f64 * 0.95) as usize] as f64 / 1000.0,
        p99: latencies_us[(n as f64 * 0.99).min((n - 1) as f64) as usize] as f64 / 1000.0,
        max: latencies_us[n - 1] as f64 / 1000.0,
        mean: (sum as f64 / n as f64) / 1000.0,
        count: n as u64,
    }
}

// ─── Engine ──────────────────────────────────────────────────────────────────

pub struct Engine {
    tokenizer: Arc<Tokenizer>,
    vocab_cache: cache::SharedCache,
    normalizer: Normalizer,
    config: Config,
    hidden_dim: usize,
    #[cfg(feature = "native-trt")]
    sessions: Vec<Vec<Option<TrtSession>>>, // [bucket_idx][gpu_idx]
}

impl Config {
    fn validate(&self) -> Result<(), Error> {
        if self.buckets.is_empty() {
            return Err(Error::Config("buckets cannot be empty".into()));
        }
        if self.batch_sizes.len() != self.buckets.len() {
            return Err(Error::Config(format!(
                "batch_sizes length ({}) must match buckets length ({})",
                self.batch_sizes.len(),
                self.buckets.len()
            )));
        }
        for (i, &bs) in self.batch_sizes.iter().enumerate() {
            if bs == 0 {
                return Err(Error::Config(format!("batch_sizes[{}] cannot be 0", i)));
            }
        }
        for i in 1..self.buckets.len() {
            if self.buckets[i] <= self.buckets[i - 1] {
                return Err(Error::Config(format!(
                    "buckets must be strictly increasing, but buckets[{}]={} <= buckets[{}]={}",
                    i,
                    self.buckets[i],
                    i - 1,
                    self.buckets[i - 1]
                )));
            }
        }
        if self.truncation == 0 {
            return Err(Error::Config("truncation must be > 0".into()));
        }
        if self.tokenize_workers == 0 {
            return Err(Error::Config("tokenize_workers must be > 0".into()));
        }
        if let Some(ref gpus) = self.gpus {
            if gpus.is_empty() {
                return Err(Error::Config("gpus list cannot be empty".into()));
            }
        }
        Ok(())
    }
}

impl Engine {
    /// Initialize: load tokenizer, vocab cache, find/compile TRT engine.
    /// If `config.model` is set, downloads and exports model from HuggingFace on first use.
    pub fn new(mut config: Config) -> Result<Self, Error> {
        config.validate()?;

        // Provision model if model name is specified
        if let Some(model) = &config.model {
            let resolved_dir = crate::provision::ensure_model(model, &config.model_dir)?;
            config.model_dir = resolved_dir;
        }

        let model_dir = &config.model_dir;

        // Detect hidden dimension from model config or ONNX
        let hidden_dim = detect_hidden_dim(model_dir)?;

        // Auto-scale batch sizes based on model size to avoid GPU OOM.
        // Default batch_sizes are tuned for hidden_dim <= 768 (e5-small/base on 80GB A100).
        // Scale down proportionally for larger models.
        if hidden_dim > 768 {
            let scale = 768.0 / hidden_dim as f64;
            let orig = config.batch_sizes.clone();
            config.batch_sizes = config
                .batch_sizes
                .iter()
                .map(|&bs| ((bs as f64 * scale).ceil() as usize).max(64))
                .collect();
            eprintln!(
                "[ignite-ms] auto-scaled batch_sizes for hidden_dim={}: {:?} -> {:?}",
                hidden_dim, orig, config.batch_sizes
            );
        }

        // Tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::Model(format!(
                "tokenizer.json not found in {}",
                model_dir.display()
            )));
        }
        let mut tok = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("{}", e)))?;
        let max_len = *config.buckets.last().unwrap_or(&256);
        tok.with_truncation(Some(tokenizers::TruncationParams {
            max_length: max_len,
            ..Default::default()
        }))
        .map_err(|e| Error::Tokenizer(format!("{}", e)))?;
        tok.with_padding(None);
        let tokenizer = Arc::new(tok);

        // Vocab cache
        let cache_path = model_dir.join("vocab_cache.bin");
        let vocab_cache = if cache_path.exists() {
            cache::load(&cache_path).map_err(|e| Error::Model(format!("vocab cache: {}", e)))?
        } else {
            Arc::new(HashMap::new())
        };

        // Normalizer
        let normalizer = Normalizer::new(NormalizerConfig {
            lowercase: config.lowercase,
            max_chars: config.truncation,
            min_chars: config.min_chars,
            prefix: config.prefix.clone(),
        });

        // Find or compile per-bucket TRT engines (tight shape bounds for optimal kernels)
        #[cfg(feature = "native-trt")]
        let engine_paths: Vec<PathBuf> = {
            let onnx = model_dir.join("model.onnx");
            let gpu_arch = crate::provision::detect_gpu_arch();
            let model_name = config.model.as_deref().unwrap_or("").to_string();

            if onnx.exists() {
                let t_engines = Instant::now();

                // First pass: try local cache and S3 cache, collect indices needing compilation
                let mut paths: Vec<Option<PathBuf>> = Vec::with_capacity(config.buckets.len());
                let mut to_compile: Vec<(usize, PathBuf, usize, usize)> = Vec::new();

                for bi in 0..config.buckets.len() {
                    let bs = config.batch_sizes[bi];
                    let sl = config.buckets[bi];
                    let suffix = if config.int8 { "_int8" } else { "" };
                    let bucket_engine =
                        model_dir.join(format!("model_b{}_s{}{}.engine", bs, sl, suffix));

                    if bucket_engine.exists() {
                        paths.push(Some(bucket_engine));
                    } else if let (Some(ref cache), Some(ref arch)) =
                        (&config.engine_cache, &gpu_arch)
                    {
                        if let Some(p) = crate::provision::fetch_cached_engine(
                            cache,
                            &model_name,
                            arch,
                            bs,
                            sl,
                            config.int8,
                            &bucket_engine,
                        ) {
                            paths.push(Some(p));
                        } else {
                            paths.push(None);
                            to_compile.push((bi, bucket_engine, bs, sl));
                        }
                    } else {
                        paths.push(None);
                        to_compile.push((bi, bucket_engine, bs, sl));
                    }
                }

                // Compile all missing engines in parallel
                if !to_compile.is_empty() {
                    eprintln!(
                        "[ignite-ms] compiling {} engines in parallel...",
                        to_compile.len()
                    );
                    let mut compile_handles: Vec<
                        thread::JoinHandle<Result<(usize, PathBuf), String>>,
                    > = Vec::new();

                    for (bi, bucket_engine, bs, sl) in to_compile {
                        let onnx_c = onnx.clone();
                        let int8 = config.int8;
                        let engine_cache = config.engine_cache.clone();
                        let arch = gpu_arch.clone();
                        let model_name_c = model_name.clone();

                        compile_handles.push(thread::spawn(move || {
                            eprintln!(
                                "[ignite-ms]   compiling engine: batch={} seq={} int8={}",
                                bs, sl, int8
                            );
                            let result = if int8 {
                                crate::inference::compile_engine_for_bucket_int8(
                                    &onnx_c,
                                    &bucket_engine,
                                    bs,
                                    sl,
                                )
                            } else {
                                crate::inference::compile_engine_for_bucket(
                                    &onnx_c,
                                    &bucket_engine,
                                    bs,
                                    sl,
                                )
                            };

                            if let Err(ref e) = result {
                                return Err(format!("engine b{}_s{}: {}", bs, sl, e));
                            }

                            // Upload to S3 cache
                            if let (Some(ref cache), Some(ref arch)) = (&engine_cache, &arch) {
                                crate::provision::upload_engine(
                                    cache,
                                    &model_name_c,
                                    arch,
                                    bs,
                                    sl,
                                    int8,
                                    &bucket_engine,
                                );
                            }

                            Ok((bi, bucket_engine))
                        }));
                    }

                    for h in compile_handles {
                        let (bi, path) = h
                            .join()
                            .map_err(|_| Error::Inference("engine compile thread panicked".into()))?
                            .map_err(|e| Error::Inference(e))?;
                        paths[bi] = Some(path);
                    }

                    eprintln!(
                        "[ignite-ms] all engines ready ({:.1}s)",
                        t_engines.elapsed().as_secs_f64()
                    );
                }

                paths.into_iter().map(|p| p.unwrap()).collect()
            } else {
                Vec::new()
            }
        };
        // Create TRT sessions in parallel (during init, not during embed)
        #[cfg(feature = "native-trt")]
        let sessions = {
            let gpus: Vec<u32> = resolve_gpus(&config.gpus);
            let n_gpus = gpus.len();
            let n_buckets = config.buckets.len();

            if !engine_paths.is_empty() {
                let t0 = Instant::now();
                let mut handles: Vec<
                    thread::JoinHandle<(usize, usize, Result<TrtSession, String>)>,
                > = Vec::new();
                for bi in 0..n_buckets {
                    let bs = config.batch_sizes[bi];
                    let sl = config.buckets[bi];
                    for (gi, &gpu_id) in gpus.iter().enumerate() {
                        let path = engine_paths[bi].clone();
                        handles.push(thread::spawn(move || {
                            (bi, gi, TrtSession::new(&path, gpu_id, bs, sl, hidden_dim))
                        }));
                    }
                }
                let mut sessions: Vec<Vec<Option<TrtSession>>> = (0..n_buckets)
                    .map(|_| (0..n_gpus).map(|_| None).collect())
                    .collect();
                let mut failed = 0usize;
                for h in handles {
                    match h.join() {
                        Ok((bi, gi, Ok(s))) => {
                            sessions[bi][gi] = Some(s);
                        }
                        Ok((_, _, Err(e))) => {
                            eprintln!("[ignite-ms] WARNING: TRT session creation failed: {}", e);
                            failed += 1;
                        }
                        Err(_) => {
                            failed += 1;
                        }
                    }
                }
                let total: usize = sessions
                    .iter()
                    .flat_map(|v| v.iter())
                    .filter(|s| s.is_some())
                    .count();
                if total == 0 {
                    return Err(Error::Inference(
                        "no TRT sessions created — check GPU availability".into(),
                    ));
                }
                if failed > 0 {
                    eprintln!(
                        "[ignite-ms] WARNING: {} session(s) failed, {} succeeded",
                        failed, total
                    );
                }
                eprintln!(
                    "[ignite-ms] {} TRT sessions created in {:.2}s",
                    total,
                    t0.elapsed().as_secs_f64()
                );
                sessions
            } else {
                (0..n_buckets)
                    .map(|_| (0..n_gpus).map(|_| None).collect())
                    .collect()
            }
        };

        Ok(Engine {
            tokenizer,
            vocab_cache,
            normalizer,
            config,
            hidden_dim,
            #[cfg(feature = "native-trt")]
            sessions,
        })
    }

    /// Embed messages. Returns a receiver for embedding batches and a handle for stats.
    ///
    /// Usage:
    /// ```ignore
    /// let (rx, handle) = engine.embed(messages)?;
    /// for batch in rx {
    ///     for i in 0..batch.n_rows {
    ///         process(batch.ids[i], batch.embedding(i));
    ///     }
    /// }
    /// let stats = handle.finish();
    /// ```
    pub fn embed<I>(
        &mut self,
        messages: I,
    ) -> Result<(Receiver<EmbeddingBatch>, EmbedHandle), Error>
    where
        I: IntoIterator<Item = Message> + Send + 'static,
    {
        let n_buckets = self.config.buckets.len();
        let max_len = *self.config.buckets.last().unwrap_or(&256);
        let bucket_lens = self.config.buckets.clone();
        let hidden_dim = self.hidden_dim;

        // Output channel
        let (out_tx, out_rx): (Sender<EmbeddingBatch>, Receiver<EmbeddingBatch>) = bounded(32);

        // Embedding dedup cache
        let dedup: Option<EmbedCache> = if self.config.dedup_capacity > 0 {
            Some(Arc::new(EmbedCacheInner::new(
                self.config.dedup_capacity,
                hidden_dim,
            )))
        } else {
            None
        };

        // Counters
        let processed = Arc::new(AtomicU64::new(0));
        let skipped = Arc::new(AtomicU64::new(0));
        let batches_done = Arc::new(AtomicU64::new(0));
        let dedup_hit_counter = Arc::new(AtomicU64::new(0));

        // Live telemetry counters (shared with monitor thread)
        let live_cache_hits = Arc::new(AtomicU64::new(0));
        let live_cache_misses = Arc::new(AtomicU64::new(0));
        let live_bucket_msgs: Arc<Vec<AtomicU64>> =
            Arc::new((0..n_buckets).map(|_| AtomicU64::new(0)).collect());
        let live_total_rows = Arc::new(AtomicU64::new(0));
        let live_total_capacity = Arc::new(AtomicU64::new(0));

        // Per-bucket channels (tokenizer → GPU workers)
        // Tuple: (tokenized, msg_id, dedup_hash)
        let mut bucket_txs: Vec<Sender<(TokenizedMessage, String, u64)>> = Vec::new();
        let mut bucket_rxs: Vec<Receiver<(TokenizedMessage, String, u64)>> = Vec::new();
        for _ in 0..n_buckets {
            let (tx, rx) = bounded(8192);
            bucket_txs.push(tx);
            bucket_rxs.push(rx);
        }

        // Input channel (feeder → tokenizers)
        let (input_tx, input_rx): (Sender<Message>, Receiver<Message>) = bounded(4096);

        // Feeder thread
        thread::spawn(move || {
            for msg in messages {
                if input_tx.send(msg).is_err() {
                    break;
                }
            }
        });

        // Tokenizer workers
        let mut tok_handles = Vec::new();
        for _ in 0..self.config.tokenize_workers {
            let rx = input_rx.clone();
            let txs = bucket_txs.clone();
            let tok = Arc::clone(&self.tokenizer);
            let vc = Arc::clone(&self.vocab_cache);
            let bl = bucket_lens.clone();
            let ml = max_len;
            let proc = Arc::clone(&processed);
            let skip = Arc::clone(&skipped);
            let norm = self.normalizer.clone();
            let lch = Arc::clone(&live_cache_hits);
            let lcm = Arc::clone(&live_cache_misses);
            let lbm = Arc::clone(&live_bucket_msgs);
            let dedup_w = dedup.clone();
            let dedup_hits_w = Arc::clone(&dedup_hit_counter);
            let out_tx_w = out_tx.clone();
            let hd = hidden_dim;

            tok_handles.push(thread::spawn(move || {
                let mut cached = CachedTokenizer::new(tok, ml, vc);
                let mut last_reported_hits = 0u64;
                let mut last_reported_misses = 0u64;
                let mut flush_ctr = 0u32;
                let mut dedup_buf = vec![0.0f32; hd];
                while let Ok(msg) = rx.recv() {
                    let text = match norm.normalize(&msg.text) {
                        Some(t) => t,
                        None => {
                            skip.fetch_add(1, Ordering::Relaxed);
                            let _ = out_tx_w.send(EmbeddingBatch {
                                ids: vec![msg.id],
                                embeddings: Vec::new(),
                                n_rows: 0,
                                hidden_dim: hd,
                                inference_failed: false,
                            });
                            continue;
                        }
                    };

                    let text_hash = fnv1a(&text);

                    // Dedup cache check — skip tokenization + GPU entirely on hit
                    if let Some(ref dc) = dedup_w {
                        if dc.get(text_hash, &mut dedup_buf) {
                            proc.fetch_add(1, Ordering::Relaxed);
                            dedup_hits_w.fetch_add(1, Ordering::Relaxed);
                            let _ = out_tx_w.send(EmbeddingBatch {
                                ids: vec![msg.id],
                                embeddings: dedup_buf.clone(),
                                n_rows: 1,
                                hidden_dim: hd,
                                inference_failed: false,
                            });
                            continue;
                        }
                    }

                    let tokenized = cached.tokenize_one(&text);
                    proc.fetch_add(1, Ordering::Relaxed);
                    let bi = assign_bucket(&bl, tokenized.len);
                    lbm[bi].fetch_add(1, Ordering::Relaxed);
                    if txs[bi].send((tokenized, msg.id, text_hash)).is_err() {
                        break;
                    }

                    flush_ctr += 1;
                    if flush_ctr >= 4096 {
                        let h = cached.hits;
                        let m = cached.misses;
                        lch.fetch_add(h - last_reported_hits, Ordering::Relaxed);
                        lcm.fetch_add(m - last_reported_misses, Ordering::Relaxed);
                        last_reported_hits = h;
                        last_reported_misses = m;
                        flush_ctr = 0;
                    }
                }
                lch.fetch_add(cached.hits - last_reported_hits, Ordering::Relaxed);
                lcm.fetch_add(cached.misses - last_reported_misses, Ordering::Relaxed);
                (
                    cached.hits,
                    cached.misses,
                    cached.runtime_hits,
                    cached.runtime_cache_size() as u64,
                )
            }));
        }
        drop(input_rx);
        drop(bucket_txs);

        // GPU workers — use pre-created sessions from Engine::new()
        #[cfg(feature = "native-trt")]
        let mut all_sessions = std::mem::take(&mut self.sessions);
        let gpus: Vec<u32> = resolve_gpus(&self.config.gpus);
        let n_gpus = gpus.len();
        let mut gpu_handles: Vec<thread::JoinHandle<GpuWorkerStats>> = Vec::new();
        let mut batcher_handles: Vec<thread::JoinHandle<()>> = Vec::new();

        if n_gpus <= 1 {
            // ─── Single-GPU path: per-bucket routing (proven at 259K msg/s) ─────
            for (bi, bucket_rx) in bucket_rxs.into_iter().enumerate() {
                let batch_size = self.config.batch_sizes[bi];
                let bucket_len = self.config.buckets[bi];

                #[cfg(feature = "native-trt")]
                let session: Option<TrtSession> =
                    if bi < all_sessions.len() && !all_sessions[bi].is_empty() {
                        all_sessions[bi][0].take()
                    } else {
                        None
                    };

                let tx = out_tx.clone();
                let batch_ctr = Arc::clone(&batches_done);
                let ltr = Arc::clone(&live_total_rows);
                let ltc = Arc::clone(&live_total_capacity);
                let dedup_g = dedup.clone();

                gpu_handles.push(thread::spawn(move || {
                    #[cfg(feature = "native-trt")]
                    let mut session = session;
                    let mut bucketizer = Bucketizer::new(vec![BucketSpec {
                        len: bucket_len,
                        batch_size,
                    }]);
                    let mut ids_buffer: Vec<String> = Vec::with_capacity(batch_size);
                    let mut hashes_buffer: Vec<u64> = Vec::with_capacity(batch_size);
                    let mut output_buf: Vec<f32> = vec![0.0; batch_size * hidden_dim];
                    let ttids_buf: Vec<i64> = vec![0i64; batch_size * bucket_len];
                    let mut stats = GpuWorkerStats {
                        batches: 0,
                        total_rows: 0,
                        total_capacity: 0,
                        latencies_us: Vec::new(),
                        per_bucket_batches: vec![0u64; n_buckets],
                        per_bucket_messages: vec![0u64; n_buckets],
                    };

                    while let Ok((tokenized, id, hash)) = bucket_rx.recv() {
                        ids_buffer.push(id);
                        hashes_buffer.push(hash);
                        if let Some(batch) = bucketizer.push(tokenized) {
                            let n_real = batch.n_real;
                            let batch_ids: Vec<String> = ids_buffer.drain(..n_real).collect();
                            let batch_hashes: Vec<u64> = hashes_buffer.drain(..n_real).collect();

                            let t_infer = Instant::now();
                            let mut infer_ok = false;
                            #[cfg(feature = "native-trt")]
                            if let Some(ref mut sess) = session {
                                if sess
                                    .infer(
                                        &batch.input_ids,
                                        &batch.attention_mask,
                                        Some(&ttids_buf[..batch.input_ids.len()]),
                                    )
                                    .is_ok()
                                {
                                    if sess.get_output(&mut output_buf).is_ok() {
                                        infer_ok = true;
                                    }
                                }
                            }
                            #[cfg(not(feature = "native-trt"))]
                            {
                                infer_ok = false;
                            }
                            stats
                                .latencies_us
                                .push(t_infer.elapsed().as_micros() as u64);
                            stats.batches += 1;
                            stats.total_rows += n_real as u64;
                            stats.total_capacity += batch_size as u64;
                            stats.per_bucket_batches[bi] += 1;
                            stats.per_bucket_messages[bi] += n_real as u64;
                            ltr.fetch_add(n_real as u64, Ordering::Relaxed);
                            ltc.fetch_add(batch_size as u64, Ordering::Relaxed);

                            if infer_ok {
                                if let Some(ref dc) = dedup_g {
                                    for (i, &h) in batch_hashes.iter().enumerate() {
                                        let start = i * hidden_dim;
                                        dc.insert(h, &output_buf[start..start + hidden_dim]);
                                    }
                                }
                                let emb_len = n_real * hidden_dim;
                                let _ = tx.send(EmbeddingBatch {
                                    ids: batch_ids,
                                    embeddings: output_buf[..emb_len].to_vec(),
                                    n_rows: n_real,
                                    hidden_dim,
                                    inference_failed: false,
                                });
                            } else {
                                let _ = tx.send(EmbeddingBatch {
                                    ids: batch_ids,
                                    embeddings: Vec::new(),
                                    n_rows: 0,
                                    hidden_dim,
                                    inference_failed: true,
                                });
                            }
                            batch_ctr.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    for batch in bucketizer.flush() {
                        let n_real = batch.n_real;
                        let batch_ids: Vec<String> = ids_buffer.drain(..n_real).collect();
                        let batch_hashes: Vec<u64> = hashes_buffer.drain(..n_real).collect();

                        let t_infer = Instant::now();
                        let mut infer_ok = false;
                        #[cfg(feature = "native-trt")]
                        if let Some(ref mut sess) = session {
                            if sess
                                .infer(
                                    &batch.input_ids,
                                    &batch.attention_mask,
                                    Some(&ttids_buf[..batch.input_ids.len()]),
                                )
                                .is_ok()
                            {
                                if sess.get_output(&mut output_buf).is_ok() {
                                    infer_ok = true;
                                }
                            }
                        }
                        #[cfg(not(feature = "native-trt"))]
                        {
                            infer_ok = false;
                        }
                        stats
                            .latencies_us
                            .push(t_infer.elapsed().as_micros() as u64);
                        stats.batches += 1;
                        stats.total_rows += n_real as u64;
                        stats.total_capacity += batch_size as u64;
                        stats.per_bucket_batches[bi] += 1;
                        stats.per_bucket_messages[bi] += n_real as u64;
                        ltr.fetch_add(n_real as u64, Ordering::Relaxed);
                        ltc.fetch_add(batch_size as u64, Ordering::Relaxed);

                        if infer_ok {
                            if let Some(ref dc) = dedup_g {
                                for (i, &h) in batch_hashes.iter().enumerate() {
                                    let start = i * hidden_dim;
                                    dc.insert(h, &output_buf[start..start + hidden_dim]);
                                }
                            }
                            let emb_len = n_real * hidden_dim;
                            let _ = tx.send(EmbeddingBatch {
                                ids: batch_ids,
                                embeddings: output_buf[..emb_len].to_vec(),
                                n_rows: n_real,
                                hidden_dim,
                                inference_failed: false,
                            });
                        } else {
                            let _ = tx.send(EmbeddingBatch {
                                ids: batch_ids,
                                embeddings: Vec::new(),
                                n_rows: 0,
                                hidden_dim,
                                inference_failed: true,
                            });
                        }
                        batch_ctr.fetch_add(1, Ordering::Relaxed);
                    }
                    stats
                }));
            }
        } else {
            // ─── Multi-GPU path: shared batch queue for optimal load balancing ──
            // Shared ready-batch channel: batcher threads → GPU workers
            let (batch_tx, batch_rx): (
                Sender<(Batch, Vec<String>, Vec<u64>)>,
                Receiver<(Batch, Vec<String>, Vec<u64>)>,
            ) = bounded(64);

            // Batcher threads: one per bucket, accumulates from bucket_rx → ready batches
            let batch_timeout = if self.config.batch_timeout_ms > 0 {
                Some(std::time::Duration::from_millis(
                    self.config.batch_timeout_ms,
                ))
            } else {
                None
            };
            for (bi, bucket_rx) in bucket_rxs.into_iter().enumerate() {
                let batch_size = self.config.batch_sizes[bi];
                let bucket_len = self.config.buckets[bi];
                let btx = batch_tx.clone();
                let timeout = batch_timeout;

                batcher_handles.push(thread::spawn(move || {
                    let mut bucketizer = Bucketizer::new(vec![BucketSpec {
                        len: bucket_len,
                        batch_size,
                    }]);
                    let mut ids_buffer: Vec<String> = Vec::with_capacity(batch_size);
                    let mut hashes_buffer: Vec<u64> = Vec::with_capacity(batch_size);

                    if let Some(dur) = timeout {
                        loop {
                            match bucket_rx.recv_timeout(dur) {
                                Ok((tokenized, id, hash)) => {
                                    ids_buffer.push(id);
                                    hashes_buffer.push(hash);
                                    if let Some(mut batch) = bucketizer.push(tokenized) {
                                        batch.bucket_idx = bi;
                                        let batch_ids: Vec<String> =
                                            ids_buffer.drain(..batch.n_real).collect();
                                        let batch_hashes: Vec<u64> =
                                            hashes_buffer.drain(..batch.n_real).collect();
                                        if btx.send((batch, batch_ids, batch_hashes)).is_err() {
                                            return;
                                        }
                                    }
                                }
                                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                                    for mut batch in bucketizer.flush() {
                                        batch.bucket_idx = bi;
                                        let batch_ids: Vec<String> =
                                            ids_buffer.drain(..batch.n_real).collect();
                                        let batch_hashes: Vec<u64> =
                                            hashes_buffer.drain(..batch.n_real).collect();
                                        if btx.send((batch, batch_ids, batch_hashes)).is_err() {
                                            return;
                                        }
                                    }
                                }
                                Err(crossbeam::channel::RecvTimeoutError::Disconnected) => break,
                            }
                        }
                    } else {
                        while let Ok((tokenized, id, hash)) = bucket_rx.recv() {
                            ids_buffer.push(id);
                            hashes_buffer.push(hash);
                            if let Some(mut batch) = bucketizer.push(tokenized) {
                                batch.bucket_idx = bi;
                                let batch_ids: Vec<String> =
                                    ids_buffer.drain(..batch.n_real).collect();
                                let batch_hashes: Vec<u64> =
                                    hashes_buffer.drain(..batch.n_real).collect();
                                if btx.send((batch, batch_ids, batch_hashes)).is_err() {
                                    return;
                                }
                            }
                        }
                    }
                    for mut batch in bucketizer.flush() {
                        batch.bucket_idx = bi;
                        let batch_ids: Vec<String> = ids_buffer.drain(..batch.n_real).collect();
                        let batch_hashes: Vec<u64> = hashes_buffer.drain(..batch.n_real).collect();
                        if btx.send((batch, batch_ids, batch_hashes)).is_err() {
                            break;
                        }
                    }
                }));
            }
            drop(batch_tx);

            // GPU workers: one per GPU, each owns all bucket sessions for that GPU
            #[cfg(feature = "native-trt")]
            {
                let batch_sizes: Vec<(usize, usize)> = self
                    .config
                    .buckets
                    .iter()
                    .zip(self.config.batch_sizes.iter())
                    .map(|(&bl, &bs)| (bl, bs))
                    .collect();
                // Reorganize sessions from [bucket][gpu] to [gpu][bucket]
                let mut per_gpu_sessions: Vec<Vec<Option<TrtSession>>> = (0..n_gpus)
                    .map(|_| (0..n_buckets).map(|_| None).collect())
                    .collect();
                for bi in 0..n_buckets {
                    for gi in 0..n_gpus {
                        per_gpu_sessions[gi][bi] = all_sessions[bi][gi].take();
                    }
                }

                // Only spawn workers for GPUs that have all bucket sessions
                let per_gpu_sessions: Vec<_> = per_gpu_sessions
                    .into_iter()
                    .filter(|sessions| sessions.iter().all(|s| s.is_some()))
                    .collect();
                if per_gpu_sessions.is_empty() {
                    return Err(Error::Inference(
                        "no GPUs have complete session sets — cannot run inference".into(),
                    ));
                }

                for gpu_sessions in per_gpu_sessions {
                    let rx = batch_rx.clone();
                    let tx = out_tx.clone();
                    let batch_ctr = Arc::clone(&batches_done);
                    let bs_info = batch_sizes.clone();
                    let ltr = Arc::clone(&live_total_rows);
                    let ltc = Arc::clone(&live_total_capacity);
                    let dedup_g = dedup.clone();

                    gpu_handles.push(thread::spawn(move || {
                        let mut sessions = gpu_sessions;
                        let max_bs = bs_info.iter().map(|(_, bs)| *bs).max().unwrap_or(512);
                        let max_bl = bs_info.iter().map(|(bl, _)| *bl).max().unwrap_or(256);
                        let mut output_buf: Vec<f32> = vec![0.0; max_bs * hidden_dim];
                        let ttids_buf: Vec<i64> = vec![0i64; max_bs * max_bl];
                        let mut stats = GpuWorkerStats {
                            batches: 0,
                            total_rows: 0,
                            total_capacity: 0,
                            latencies_us: Vec::new(),
                            per_bucket_batches: vec![0u64; n_buckets],
                            per_bucket_messages: vec![0u64; n_buckets],
                        };

                        while let Ok((batch, batch_ids, batch_hashes)) = rx.recv() {
                            let bi = batch.bucket_idx;
                            let n_real = batch.n_real;
                            let capacity =
                                bs_info.get(bi).map(|(_, bs)| *bs).unwrap_or(n_real) as u64;

                            let t_infer = Instant::now();
                            let mut infer_ok = false;
                            if let Some(Some(ref mut sess)) = sessions.get_mut(bi) {
                                if sess
                                    .infer(
                                        &batch.input_ids,
                                        &batch.attention_mask,
                                        Some(&ttids_buf[..batch.input_ids.len()]),
                                    )
                                    .is_ok()
                                {
                                    if sess.get_output(&mut output_buf).is_ok() {
                                        infer_ok = true;
                                    }
                                }
                            }
                            stats
                                .latencies_us
                                .push(t_infer.elapsed().as_micros() as u64);
                            stats.batches += 1;
                            stats.total_rows += n_real as u64;
                            stats.total_capacity += capacity;
                            stats.per_bucket_batches[bi] += 1;
                            stats.per_bucket_messages[bi] += n_real as u64;
                            ltr.fetch_add(n_real as u64, Ordering::Relaxed);
                            ltc.fetch_add(capacity, Ordering::Relaxed);

                            if infer_ok {
                                if let Some(ref dc) = dedup_g {
                                    for (i, &h) in batch_hashes.iter().enumerate() {
                                        let start = i * hidden_dim;
                                        dc.insert(h, &output_buf[start..start + hidden_dim]);
                                    }
                                }
                                let emb_len = n_real * hidden_dim;
                                let _ = tx.send(EmbeddingBatch {
                                    ids: batch_ids,
                                    embeddings: output_buf[..emb_len].to_vec(),
                                    n_rows: n_real,
                                    hidden_dim,
                                    inference_failed: false,
                                });
                            } else {
                                let _ = tx.send(EmbeddingBatch {
                                    ids: batch_ids,
                                    embeddings: Vec::new(),
                                    n_rows: 0,
                                    hidden_dim,
                                    inference_failed: true,
                                });
                            }
                            batch_ctr.fetch_add(1, Ordering::Relaxed);
                        }
                        stats
                    }));
                }
            }

            #[cfg(not(feature = "native-trt"))]
            compile_error!(
                "ignite-ms requires feature \"native-trt\" — GPU inference is not optional"
            );

            drop(batch_rx);
        }
        drop(out_tx);

        // Monitor thread — prints progress every 2 seconds (only when telemetry enabled)
        let mon_telemetry = self.config.telemetry;
        let mon_processed = Arc::clone(&processed);
        let mon_skipped = Arc::clone(&skipped);
        let mon_batches = Arc::clone(&batches_done);
        let mon_cache_hits = Arc::clone(&live_cache_hits);
        let mon_cache_misses = Arc::clone(&live_cache_misses);
        let mon_bucket_msgs = Arc::clone(&live_bucket_msgs);
        let mon_total_rows = Arc::clone(&live_total_rows);
        let mon_total_capacity = Arc::clone(&live_total_capacity);
        let mon_dedup_hits = Arc::clone(&dedup_hit_counter);
        let mon_n_buckets = n_buckets;
        let mon_bucket_lens = bucket_lens.clone();
        let mon_shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mon_sd = Arc::clone(&mon_shutdown);
        let mon_t0 = Instant::now();
        let monitor = thread::spawn(move || {
            if !mon_telemetry {
                return;
            }
            let mut last_proc = 0u64;
            let mut tick = 0u64;
            loop {
                thread::sleep(std::time::Duration::from_secs(2));
                if mon_sd.load(Ordering::Relaxed) {
                    break;
                }
                let proc = mon_processed.load(Ordering::Relaxed);
                let skip = mon_skipped.load(Ordering::Relaxed);
                let batches = mon_batches.load(Ordering::Relaxed);
                let elapsed = mon_t0.elapsed().as_secs_f64();
                let rate = (proc - last_proc) as f64 / 2.0;
                let avg = proc as f64 / elapsed.max(1e-9);
                eprintln!(
                    "[t+{:.0}s] proc={} skip={} batches={} | rate={:.0} avg={:.0} msg/s",
                    elapsed, proc, skip, batches, rate, avg
                );

                // Extended telemetry every 10 seconds
                tick += 1;
                if tick % 5 == 0 {
                    let hits = mon_cache_hits.load(Ordering::Relaxed);
                    let misses = mon_cache_misses.load(Ordering::Relaxed);
                    let total_cache = hits + misses;
                    let hit_rate = if total_cache > 0 {
                        hits as f64 / total_cache as f64
                    } else {
                        0.0
                    };

                    let rows = mon_total_rows.load(Ordering::Relaxed);
                    let cap = mon_total_capacity.load(Ordering::Relaxed);
                    let batch_fill = if cap > 0 {
                        rows as f64 / cap as f64
                    } else {
                        0.0
                    };

                    let dedup_h = mon_dedup_hits.load(Ordering::Relaxed);
                    let dedup_rate = if proc > 0 {
                        dedup_h as f64 / proc as f64
                    } else {
                        0.0
                    };

                    let bucket_dist: Vec<String> = (0..mon_n_buckets)
                        .map(|i| {
                            let count = mon_bucket_msgs[i].load(Ordering::Relaxed);
                            let pct = if proc > 0 {
                                count as f64 / proc as f64 * 100.0
                            } else {
                                0.0
                            };
                            format!("b{}={:.1}%", mon_bucket_lens[i], pct)
                        })
                        .collect();

                    if dedup_h > 0 {
                        eprintln!(
                            "[ignite-ms:live] cache_hit={:.3} batch_fill={:.3} dedup={:.3} buckets=[{}]",
                            hit_rate, batch_fill, dedup_rate, bucket_dist.join(" ")
                        );
                    } else {
                        eprintln!(
                            "[ignite-ms:live] cache_hit={:.3} batch_fill={:.3} buckets=[{}]",
                            hit_rate,
                            batch_fill,
                            bucket_dist.join(" ")
                        );
                    }
                }

                last_proc = proc;
                if proc > 0 && rate == 0.0 {
                    break;
                } // done
            }
        });

        let telemetry = self.config.telemetry;
        let handle = EmbedHandle {
            tok_handles,
            gpu_handles,
            batcher_handles,
            monitor,
            mon_shutdown,
            processed,
            skipped,
            batches_done,
            dedup_hit_counter,
            dedup_cache: dedup,
            t0: Instant::now(),
            n_gpus,
            n_buckets,
            telemetry,
        };

        Ok((out_rx, handle))
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
    pub fn vocab_cache_size(&self) -> usize {
        self.vocab_cache.len()
    }
}

// ─── EmbedHandle ─────────────────────────────────────────────────────────────

/// Handle to a running embed operation. Call `finish()` after consuming all batches.
pub struct EmbedHandle {
    tok_handles: Vec<thread::JoinHandle<(u64, u64, u64, u64)>>,
    gpu_handles: Vec<thread::JoinHandle<GpuWorkerStats>>,
    batcher_handles: Vec<thread::JoinHandle<()>>,
    monitor: thread::JoinHandle<()>,
    mon_shutdown: Arc<std::sync::atomic::AtomicBool>,
    processed: Arc<AtomicU64>,
    skipped: Arc<AtomicU64>,
    batches_done: Arc<AtomicU64>,
    dedup_hit_counter: Arc<AtomicU64>,
    dedup_cache: Option<EmbedCache>,
    t0: Instant,
    n_gpus: usize,
    n_buckets: usize,
    telemetry: bool,
}

impl EmbedHandle {
    /// Wait for pipeline to finish and return stats.
    /// Call this AFTER draining the Receiver.
    pub fn finish(self) -> EmbedStats {
        self.mon_shutdown.store(true, Ordering::Relaxed);
        let _ = self.monitor.join();

        let mut total_hits = 0u64;
        let mut total_misses = 0u64;
        let mut total_runtime_hits = 0u64;
        let mut total_runtime_size = 0u64;
        for h in self.tok_handles {
            if let Ok((h, m, rh, rs)) = h.join() {
                total_hits += h;
                total_misses += m;
                total_runtime_hits += rh;
                total_runtime_size += rs;
            }
        }

        for h in self.batcher_handles {
            let _ = h.join();
        }

        let mut per_gpu_batches = Vec::with_capacity(self.n_gpus);
        let mut per_bucket_batches = vec![0u64; self.n_buckets];
        let mut per_bucket_messages = vec![0u64; self.n_buckets];
        let mut all_latencies: Vec<u64> = Vec::new();
        let mut total_rows = 0u64;
        let mut total_capacity = 0u64;

        for h in self.gpu_handles {
            if let Ok(s) = h.join() {
                per_gpu_batches.push(s.batches);
                total_rows += s.total_rows;
                total_capacity += s.total_capacity;
                all_latencies.extend(s.latencies_us);
                for (i, &b) in s.per_bucket_batches.iter().enumerate() {
                    if i < self.n_buckets {
                        per_bucket_batches[i] += b;
                    }
                }
                for (i, &m) in s.per_bucket_messages.iter().enumerate() {
                    if i < self.n_buckets {
                        per_bucket_messages[i] += m;
                    }
                }
            }
        }

        let avg_batch_fill = if total_capacity > 0 {
            total_rows as f64 / total_capacity as f64
        } else {
            0.0
        };

        let latency_stats = compute_latency_stats(all_latencies);

        let dedup_hits = self.dedup_hit_counter.load(Ordering::Relaxed);
        let dedup_size = self
            .dedup_cache
            .as_ref()
            .map(|dc| dc.occupancy() as u64)
            .unwrap_or(0);

        let stats = EmbedStats {
            messages_processed: self.processed.load(Ordering::Relaxed),
            messages_skipped: self.skipped.load(Ordering::Relaxed),
            batches_computed: self.batches_done.load(Ordering::Relaxed),
            cache_hits: total_hits,
            cache_misses: total_misses,
            runtime_cache_hits: total_runtime_hits,
            runtime_cache_size: total_runtime_size,
            dedup_hits,
            dedup_size,
            elapsed_secs: self.t0.elapsed().as_secs_f64(),
            per_gpu_batches,
            per_bucket_batches,
            per_bucket_messages,
            inference_latency_ms: latency_stats,
            avg_batch_fill,
            n_gpus: self.n_gpus,
            n_buckets: self.n_buckets,
        };

        if self.telemetry {
            eprintln!("[ignite-ms:telemetry] {}", stats.to_json());
        }

        stats
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn assign_bucket(bucket_lens: &[usize], msg_len: usize) -> usize {
    for (i, &blen) in bucket_lens.iter().enumerate() {
        if msg_len <= blen {
            return i;
        }
    }
    bucket_lens.len() - 1
}

/// Detect hidden dimension from config.json (HuggingFace format) or fall back to 384.
fn detect_hidden_dim(model_dir: &std::path::Path) -> Result<usize, Error> {
    let config_path = model_dir.join("config.json");
    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Model(format!("reading config.json: {}", e)))?;
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
            // Try hidden_size (BERT/E5), then d_model (T5), then default
            if let Some(dim) = v.get("hidden_size").and_then(|d| d.as_u64()) {
                eprintln!("[ignite-ms] detected hidden_dim={} from config.json", dim);
                return Ok(dim as usize);
            }
            if let Some(dim) = v.get("d_model").and_then(|d| d.as_u64()) {
                eprintln!(
                    "[ignite-ms] detected hidden_dim={} from config.json (d_model)",
                    dim
                );
                return Ok(dim as usize);
            }
        }
    }
    eprintln!("[ignite-ms] WARNING: config.json not found, assuming hidden_dim=384");
    Ok(384)
}
