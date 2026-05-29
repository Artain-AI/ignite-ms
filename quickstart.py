#!/usr/bin/env python3
"""
ignite-ms quickstart — download a public dataset and embed it.

Usage:
    python quickstart.py                         # Docker, MS MARCO 8.8M passages
    python quickstart.py --dataset wikipedia     # Wikipedia paragraphs
    python quickstart.py --dataset reddit        # Reddit comments subset
    python quickstart.py --native                # Build from source instead of Docker

Requirements (Docker mode — default):
    - GPU instance with nvidia-container-runtime
    - Docker installed
    - ~5GB free disk space (dataset + output)
    - Internet access (downloads dataset from HuggingFace on first run)

Requirements (--native mode):
    - GPU instance with CUDA 12.x + TensorRT 10.x
    - Rust toolchain (installed automatically if missing)
    - ~5GB free disk space (dataset + output)
    - Internet access

This script will:
    1. Download a public text dataset from HuggingFace
    2. Convert it to JSONL format
    3. Run ignite-ms (Docker by default) to produce embeddings
    4. Report throughput and output location

Docker first run prepares the model cache and compiles TensorRT engines for
your GPU. Later runs reuse the persistent cache volume and start faster.

NOTE: This downloads real datasets from the internet. The largest download
is ~1.5GB (Reddit). Total disk usage including embeddings will be 2-5GB
depending on dataset choice.
"""

import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request

DATA_DIR = "./ignite-ms-data"
DOCKER_IMAGE = "ghcr.io/artain-ai/ignite-ms:latest"
DOCKER_CACHE_VOLUME = "ignite-ms-cache"
DOCKER_CACHE_DIR = "/cache"
NATIVE_CACHE_DIR = os.path.join(DATA_DIR, "cache")

DATASETS = {
    "msmarco": {
        "name": "MS MARCO Passages",
        "description": "8.8M search passages, short text (~60 words avg)",
        "source": "Tevatron/msmarco-passage-corpus",
        "download_gb": 1.1,
        "entries": "8.8M",
        "text_field": "text",
        "id_field": "docid",
    },
    "wikipedia": {
        "name": "Wikipedia (BeIR/NQ corpus)",
        "description": "2.7M Wikipedia passages, medium length",
        "source": "BeIR-NQ",
        "download_gb": 0.8,
        "entries": "2.7M",
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip",
        "text_field": "text",
        "id_field": "_id",
    },
    "reddit": {
        "name": "Reddit 2020 Comments (subset)",
        "description": "5M Reddit posts, short noisy real-world text",
        "source": "sentence-transformers/reddit-title-body",
        "download_gb": 1.5,
        "entries": "5M",
        "max_entries": 5_000_000,
        "text_field": "body",
        "id_field": None,
    },
}


def print_banner():
    print()
    print("=" * 70)
    print("  ignite-ms quickstart")
    print("=" * 70)
    print()


def print_warning(dataset_info):
    print("  This script will download data from the internet.")
    print()
    print(f"  Dataset:    {dataset_info['name']}")
    print(f"  Entries:    {dataset_info['entries']}")
    print(f"  Download:   ~{dataset_info['download_gb']} GB")
    print(f"  Disk total: ~{dataset_info['download_gb'] * 2:.1f} GB (dataset + embeddings)")
    print()


def run(cmd, *, check=False, capture=False, timeout=None, cwd=None):
    kwargs = {"timeout": timeout}
    if cwd is not None:
        kwargs["cwd"] = cwd
    if capture:
        kwargs.update({"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True})
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        raise RuntimeError("command failed: " + " ".join(cmd))
    return result


def check_docker():
    try:
        result = run(["docker", "info"], capture=True, timeout=10)
        if result.returncode != 0:
            print("ERROR: Docker is not running. Start Docker and try again.")
            sys.exit(1)
    except FileNotFoundError:
        print("ERROR: Docker not found. Install Docker and try again.")
        sys.exit(1)

    result = run([
        "docker", "run", "--rm", "--gpus", "all",
        "nvidia/cuda:12.4.1-base-ubuntu22.04",
        "nvidia-smi",
    ], capture=True, timeout=60)
    if result.returncode != 0:
        print("ERROR: GPU access via Docker failed.")
        print("       Ensure nvidia-container-runtime is installed.")
        sys.exit(1)

    print(f"  Pulling/checking image: {DOCKER_IMAGE}")
    result = run(["docker", "pull", DOCKER_IMAGE], capture=True)
    if result.returncode != 0:
        print("ERROR: failed to pull ignite-ms Docker image.")
        print((result.stderr or result.stdout or "").strip())
        sys.exit(1)

    result = run([
        "docker", "run", "--rm", "--entrypoint", "sh", DOCKER_IMAGE,
        "-lc", "command -v ignite-ms",
    ], capture=True)
    if result.returncode != 0:
        print("ERROR: Docker image does not contain ignite-ms.")
        sys.exit(1)


def check_native_deps():
    """Check that Rust, CUDA, and TRT are available for native build."""
    if not shutil.which("cargo"):
        print("  Rust not found. Installing via rustup...")
        result = run(
            ["sh", "-c", "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"],
            capture=True,
        )
        if result.returncode != 0:
            print("ERROR: Failed to install Rust.")
            sys.exit(1)
        os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]
        print("  Rust installed.")

    result = run(["nvidia-smi"], capture=True)
    if result.returncode != 0:
        print("ERROR: nvidia-smi failed. CUDA drivers not available.")
        sys.exit(1)

    trt_header = "/usr/include/x86_64-linux-gnu/NvInfer.h"
    if not os.path.exists(trt_header) and not os.path.exists("/usr/include/NvInfer.h"):
        print("WARNING: TensorRT headers not found in standard paths.")
        print("         Build may fail. Ensure TensorRT 10.x is installed.")
        print()


def build_native():
    """Build ignite-ms-embed from source."""
    print("  Building ignite-ms from source (release)...")
    print("  This may take 2-5 minutes on first build.")
    print()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    result = run(
        ["cargo", "build", "--release", "-p", "ignite-ms-embed"],
        cwd=repo_root,
    )
    if result.returncode != 0:
        print("ERROR: Build failed. Check CUDA/TRT installation.")
        sys.exit(1)

    binary = os.path.join(repo_root, "target", "release", "ignite-ms")
    if not os.path.exists(binary):
        print(f"ERROR: Binary not found at {binary}")
        sys.exit(1)

    print(f"  Build complete: {binary}")
    print()
    return binary


def download_msmarco(output_jsonl):
    print("  [1/3] Downloading MS MARCO passages from HuggingFace...")
    print("        Source: Tevatron/msmarco-passage-corpus")
    print()

    try:
        from datasets import load_dataset
    except ImportError:
        print("  Installing 'datasets' library...")
        run([
            sys.executable, "-m", "pip", "install", "-q",
            "--root-user-action=ignore", "--disable-pip-version-check", "datasets",
        ], check=True)
        from datasets import load_dataset

    ds = load_dataset("Tevatron/msmarco-passage-corpus", split="train")

    print(f"  [2/3] Converting {len(ds):,} passages to JSONL...")
    ds = ds.rename_column("docid", "id")
    ds = ds.select_columns(["id", "text"])
    ds.to_json(output_jsonl, num_proc=os.cpu_count())

    print(f"        Done: {output_jsonl}")
    return len(ds)


def download_wikipedia(output_jsonl):
    print("  [1/3] Downloading Wikipedia (BeIR/NQ) corpus...")
    print("        Source: public.ukp.informatik.tu-darmstadt.de")
    print()

    zip_path = os.path.join(DATA_DIR, "nq.zip")
    url = DATASETS["wikipedia"]["url"]

    tmp_path = zip_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp_path, "wb") as f:
            shutil.copyfileobj(response, f)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"ERROR: failed to download Wikipedia dataset: {e}")
        sys.exit(1)
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print("ERROR: downloaded Wikipedia dataset is empty.")
        sys.exit(1)
    os.replace(tmp_path, zip_path)

    print("  [2/3] Extracting and converting to JSONL...")
    import zipfile

    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        corpus_file = [n for n in zf.namelist() if "corpus.jsonl" in n]
        if not corpus_file:
            print("ERROR: corpus.jsonl not found in zip")
            sys.exit(1)

        with zf.open(corpus_file[0]) as src, open(output_jsonl, "w") as dst:
            for line in src:
                row = json.loads(line)
                text = row.get("text", "").strip()
                if not text:
                    continue
                title = row.get("title", "")
                if title:
                    text = f"{title}. {text}"
                json.dump({"id": row.get("_id", ""), "text": text}, dst)
                dst.write("\n")
                count += 1
                if count % 500_000 == 0:
                    print(f"        {count:,} passages")

    os.remove(zip_path)
    print(f"        Done: {count:,} passages → {output_jsonl}")
    return count


def download_reddit(output_jsonl):
    print("  [1/3] Downloading Reddit 2020 comments from HuggingFace...")
    print("        Source: sentence-transformers/reddit-title-body")
    print(f"        Limiting to {DATASETS['reddit']['max_entries']:,} entries")
    print()

    try:
        from datasets import load_dataset
    except ImportError:
        print("  Installing 'datasets' library...")
        run([
            sys.executable, "-m", "pip", "install", "-q",
            "--root-user-action=ignore", "--disable-pip-version-check", "datasets",
        ], check=True)
        from datasets import load_dataset

    max_entries = DATASETS["reddit"]["max_entries"]
    ds = load_dataset(
        "sentence-transformers/reddit-title-body",
        split="train",
        streaming=True,
    )

    print(f"  [2/3] Converting to JSONL (max {max_entries:,} entries)...")
    count = 0
    with open(output_jsonl, "w", buffering=8 * 1024 * 1024) as f:
        for row in ds:
            body = row.get("body", "").strip()
            if not body or len(body) < 10:
                continue
            f.write(json.dumps({"id": str(count), "text": body}))
            f.write("\n")
            count += 1
            if count % 1_000_000 == 0:
                print(f"        {count:,} / {max_entries:,}")
            if count >= max_entries:
                break

    print(f"        Done: {count:,} comments → {output_jsonl}")
    return count


def run_embed(input_jsonl, output_path, model="intfloat/e5-small-v2"):
    abs_input_dir = os.path.abspath(os.path.dirname(input_jsonl) or ".")
    abs_output_dir = os.path.abspath(os.path.dirname(output_path) or ".")
    os.makedirs(abs_output_dir, exist_ok=True)
    input_name = os.path.basename(input_jsonl)
    output_name = os.path.basename(output_path)

    print(f"  [3/3] Embedding with ignite-ms (Docker)...")
    print(f"        Model:  {model}")
    print(f"        Input:  {input_jsonl}")
    print(f"        Output: {output_path}")
    print(f"        Cache:  Docker volume {DOCKER_CACHE_VOLUME}:{DOCKER_CACHE_DIR}")
    print()
    print("        NOTE: First run downloads model + compiles TensorRT engines.")
    print("              Later runs reuse the Docker cache volume.")
    print()

    t0 = time.time()

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{abs_input_dir}:/input:ro",
        "-v", f"{abs_output_dir}:/output",
        "-v", f"{DOCKER_CACHE_VOLUME}:{DOCKER_CACHE_DIR}",
        DOCKER_IMAGE,
        "embed",
        "--model", model,
        "--input", f"/input/{input_name}",
        "--output", f"/output/{output_name}",
        "--cache-dir", DOCKER_CACHE_DIR,
        "--gpus", "all",
        "--truncation", "256",
        "--batch-timeout-ms", "10",
    ]
    result = run(cmd)

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  ERROR: ignite-ms exited with code {result.returncode}")
        sys.exit(1)

    print()
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Output: {output_path}")

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Size:   {size_mb:.1f} MB")


def run_embed_native(binary, input_jsonl, output_path, model="intfloat/e5-small-v2"):
    print(f"  [3/3] Embedding with ignite-ms (native)...")
    print(f"        Model:  {model}")
    print(f"        Input:  {input_jsonl}")
    print(f"        Output: {output_path}")
    print(f"        Cache:  {NATIVE_CACHE_DIR}")
    print()
    print("        NOTE: First run downloads model + compiles TensorRT engines.")
    print("              Subsequent runs use cached engines.")
    print()

    t0 = time.time()

    os.makedirs(NATIVE_CACHE_DIR, exist_ok=True)
    cmd = [
        binary,
        "embed",
        "--model", model,
        "--input", os.path.abspath(input_jsonl),
        "--output", os.path.abspath(output_path),
        "--cache-dir", os.path.abspath(NATIVE_CACHE_DIR),
        "--gpus", "all",
        "--truncation", "256",
        "--batch-timeout-ms", "10",
    ]
    result = run(cmd)

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  ERROR: ignite-ms exited with code {result.returncode}")
        sys.exit(1)

    print()
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Output: {output_path}")

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Size:   {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="ignite-ms quickstart — download data + embed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets:
  msmarco     MS MARCO passages, 8.8M short texts (~1.1GB download)
  wikipedia   Wikipedia paragraphs, 2.7M medium texts (~764MB download)
  reddit      Reddit 2020 comments, 5M posts (~1.5GB download)
""",
    )
    parser.add_argument(
        "--dataset",
        choices=["msmarco", "wikipedia", "reddit"],
        default="msmarco",
        help="Dataset to download and embed (default: msmarco)",
    )
    parser.add_argument(
        "--model",
        default="intfloat/e5-small-v2",
        help="Embedding model (default: intfloat/e5-small-v2)",
    )
    parser.add_argument(
        "--output",
        help="Output path. Extension selects format: .npy or .parquet.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if JSONL already exists",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Build from source and run natively (no Docker). Requires Rust + CUDA + TRT.",
    )
    parser.add_argument(
        "--no-docker-check",
        action="store_true",
        help="Skip Docker/GPU availability check",
    )

    args = parser.parse_args()
    dataset_info = DATASETS[args.dataset]

    print_banner()
    print_warning(dataset_info)

    os.makedirs(DATA_DIR, exist_ok=True)

    output_jsonl = os.path.join(DATA_DIR, f"{args.dataset}.jsonl")
    output_path = args.output or os.path.join(DATA_DIR, f"{args.dataset}_embeddings.npy")

    native_binary = None
    if args.native:
        print("  Mode: native (build from source)")
        print()
        check_native_deps()
        native_binary = build_native()
    elif not args.no_docker_check:
        print("  Mode: Docker")
        print("  Checking Docker + GPU access...")
        check_docker()
        print("  OK")
        print()

    if args.skip_download and os.path.exists(output_jsonl):
        print(f"  Skipping download, using existing: {output_jsonl}")
        print()
    else:
        if args.dataset == "msmarco":
            download_msmarco(output_jsonl)
        elif args.dataset == "wikipedia":
            download_wikipedia(output_jsonl)
        elif args.dataset == "reddit":
            download_reddit(output_jsonl)
        print()

    if native_binary:
        run_embed_native(native_binary, output_jsonl, output_path, model=args.model)
    else:
        run_embed(output_jsonl, output_path, model=args.model)

    print()
    print("=" * 70)
    print("  Quickstart complete!")
    print()
    print("  To load embeddings in Python:")
    if output_path.endswith(".parquet"):
        print("    import pyarrow.parquet as pq")
        print(f"    table = pq.read_table('{output_path}')")
        print("    print(table.num_rows)")
    else:
        print("    import numpy as np")
        print(f"    embeddings = np.load('{output_path}')")
        print("    print(embeddings.shape)  # (N, 384)")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
