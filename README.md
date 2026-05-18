# IgniteMS

253,000 msg/s on 8x A100. Up to 3x faster than Hugging Face TEI on same hardware.

*357,893 msg/s sustained in production with workload-specific tuning.*

IgniteMS is a batch text embedding engine. Rust, native TensorRT, no Python at runtime. You give it text, it gives you embeddings.

Use it for workloads where millions of texts need embeddings quickly: vector DB reindexing, search rebuilds after model swaps, corpus-scale processing.

## Numbers

p4d.24xlarge (8x A100 40GB), 1M MSMARCO passages, e5-small-v2, FP16:

| GPUs | IgniteMS | TEI | Speedup |
|-----:|---------:|----:|--------:|
| 1 | 50,127 msg/s | 16,648 | 3.0x |
| 8 | 253,578 msg/s | 96,492 | 2.6x |

For reference: SentenceTransformers on the same single GPU does ~2,500 msg/s (20x slower).

### Tested models

| Model | Dimensions | 1 GPU msg/s | 8 GPU msg/s | Scaling |
|-------|---:|---:|---:|---:|
| `intfloat/e5-small-v2` | 384 | 50,127 | 253,578 | 5.1x |
| `intfloat/multilingual-e5-small` | 384 | 41,842 | 253,950 | 6.1x |
| `intfloat/multilingual-e5-base` | 768 | 17,319 | 122,169 | 7.1x |
| `intfloat/multilingual-e5-large` | 1024 | 5,610 | 41,668 | 7.4x |

Scaling is 1-GPU to 8-GPU on the same machine. Larger models scale closer to 8x because inference dominates the pipeline. Smaller models are faster per-GPU but CPU tokenization becomes the bottleneck at high GPU counts. Your scaling will depend on text length distribution, model size, and hardware.

Works with Hugging Face encoder models that export to ONNX and compile to TensorRT. Models are downloaded and compiled on first run.

### Production run

Real production pipeline, not a controlled benchmark:

| Metric | Value | Note |
|--------|------:|------|
| Messages embedded | 685,520,494 | |
| Sustained throughput | 357,893 msg/s | average across full run |
| Peak throughput | 506,589 msg/s | short text, GPUs saturated |
| Low throughput | 196,676 msg/s | dense/long text files, reader-bound |
| Wall clock | 1,915s (31.9 min) | |
| Hardware | 1x p4d.24xlarge | 8x A100 40GB, spot |

Full pipeline: read zstd-compressed social media events (Reddit, Hacker News), extract and normalize text, tokenize, infer on 8 GPUs, write aggregated parquet output. Not a GPU microbenchmark.

For cost context: at ~$12.68/hr p4d spot pricing, this production run cost about $0.01 per 1M messages embedded. On the same 68-token/message dataset, OpenAI `text-embedding-3-small` would be about $1.36 per 1M messages at current API pricing.

## Why it's fast

No single trick. Just removing waste everywhere:

- **TensorRT** compiles kernels specific to the GPU architecture and batch shape. Not generic ONNX or PyTorch.
- **Bucketed batching** groups texts by token length, reducing padding waste.
- **CPU-side pipeline** keeps tokenization, batching, and GPU dispatch moving together.
- **Rust end-to-end.** No GIL, no Python request path, no HTTP serialization at runtime.
- **Multi-GPU in one process.** One process can drive multiple GPUs with lock-free work distribution. No one-container-per-GPU serving stack.
- **Engine caching.** TRT engines compile once and are reused until the model, runtime, or profile changes.

## Quickstart

Docker (just needs Docker + NVIDIA runtime):

```bash
python3 quickstart.py
```

Native (needs Rust, CUDA 12+, TensorRT 10+):

```bash
python3 quickstart.py --native
```

Downloads a public dataset, embeds it, writes output. First run takes ~5 minutes for TensorRT engine compilation. After that, engines are cached and startup is instant.

## Docker

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/data" \
  -v ignite-ms-cache:/cache \
  ghcr.io/artain-ai/ignite-ms:latest \
  embed \
  --model intfloat/e5-small-v2 \
  --input /data/input.jsonl \
  --output /data/embeddings.npy \
  --cache-dir /cache \
  --gpus all
```

Image has the production CLI (`ignite-ms`), benchmark CLI (`ignite-ms-bench`), and all dependencies for model prep.

## Benchmark

Reproduce the numbers:

```bash
python3 benchmark.py                                          # Docker, defaults
python3 benchmark.py --mode native --model e5-small-v2        # native
python3 benchmark.py --gpu-counts 1,8 --skip-tei              # IgniteMS only
```

Downloads data, prepares models, runs both IgniteMS and TEI, reports results. See [BENCHMARKING.md](BENCHMARKING.md) for full results, methodology, and caveats.

## Input / Output

Input: JSONL (`{"text": "..."}`) or plain text, one per line. Handles `.zst` and `.gz` compression.

Output: `.npy` (NumPy array) or `.parquet` (with IDs). Row order preserved.

```bash
ignite-ms embed \
  --model intfloat/e5-small-v2 \
  --input corpus.jsonl.zst \
  --output embeddings.npy \
  --gpus all
```

## Layout

```
crates/ignite-ms/          core engine
crates/ignite-ms-embed/    production CLI (ignite-ms)
crates/ignite-ms-bench/    benchmark CLI (ignite-ms-bench)
native/                    TensorRT C++ bridge
examples/                  library usage
benchmark.py               IgniteMS vs TEI benchmark
quickstart.py              one-command demo
```

## Building from source

```bash
cargo build --release -p ignite-ms-embed
cargo build --release -p ignite-ms-bench
```

Needs CUDA 12+ and TensorRT 10+ headers on the host.

## Requirements

Docker mode: NVIDIA GPU, Docker, NVIDIA container runtime.

Native mode: NVIDIA GPU, CUDA 12+, TensorRT 10+, Rust 1.85+, Python 3.10+.

## Security

Report vulnerabilities privately. See [SECURITY.md](SECURITY.md).

## Contributing

Contributions require CLA. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0.

Artain may offer future versions under different terms. Versions released under Apache 2.0 stay Apache 2.0.
