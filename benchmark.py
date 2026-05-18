#!/usr/bin/env python3
"""
ignite-ms benchmark: compare ignite-ms against Hugging Face TEI.

Default mode is Docker. Native mode builds the local Rust binary and requires
CUDA + TensorRT development headers on the host.

Examples:
    python benchmark.py
    python benchmark.py --mode native
    python benchmark.py --input ./data/input.jsonl
    python benchmark.py --model intfloat/e5-base-v2
    python benchmark.py --gpu-counts 1,8
    python benchmark.py --skip-tei
"""

import argparse
import asyncio
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import struct
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DATA_DIR = Path("./ignite-ms-data")
DEFAULT_TEXTS = 1_000_000
DEFAULT_MODEL = "e5-small"
DEFAULT_GPU_COUNTS = "auto"
DOCKER_IMAGE = "ghcr.io/artain-ai/ignite-ms:latest"
TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:cuda-1.9"
TEI_BATCH = 32
TEI_CONC_PER_WORKER = 8
TEI_PORT_BASE = 8080
TEI_REQUEST_RETRIES = 8
DOCKER_TRT_IMAGE_FAMILY = "nvcr.io/nvidia/tensorrt:24.10-py3"
E5_SMALL_BUCKETS = "32,64,128,256"
E5_SMALL_BATCH_SIZES = "4096,2048,2048,512"
DEFAULT_BUCKETS = "24,32,40,48,56,64,80,96,128,256"
DEFAULT_BATCH_SIZES = "4096,4096,3072,2048,2048,2048,1536,1536,1024,512"
MIN_ENGINE_BYTES = 10 * 1024 * 1024

MODEL_SPECS = {
    "e5-small": {
        "name": "e5-small",
        "model_id": "intfloat/multilingual-e5-small",
        "hf_repo": "intfloat/multilingual-e5-small",
        "hf_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "onnx_repo": "Xenova/multilingual-e5-small",
        "onnx_revision": "761b726dd34fb83930e26aab4e9ac3899aa1fa78",
        "onnx_path": "onnx/model.onnx",
        "hidden_dim": 384,
        "buckets": E5_SMALL_BUCKETS,
        "batch_sizes": E5_SMALL_BATCH_SIZES,
    },
    "e5-small-v2": {
        "name": "e5-small-v2",
        "model_id": "intfloat/e5-small-v2",
        "hf_repo": "intfloat/e5-small-v2",
        "hf_revision": "ffb93f3bd4047442299a41ebb6fa998a38507c52",
        "onnx_repo": "Xenova/e5-small-v2",
        "onnx_revision": "02af79985278377e65c724a76275707cb0333c70",
        "onnx_path": "onnx/model.onnx",
        "hidden_dim": 384,
        "buckets": E5_SMALL_BUCKETS,
        "batch_sizes": E5_SMALL_BATCH_SIZES,
    },
    "e5-base": {
        "name": "e5-base",
        "model_id": "intfloat/multilingual-e5-base",
        "hf_repo": "intfloat/multilingual-e5-base",
        "hf_revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
        "onnx_repo": "Xenova/multilingual-e5-base",
        "onnx_revision": "1ec9243030a27d1a115d5c340572074c125b58b2",
        "onnx_path": "onnx/model.onnx",
        "hidden_dim": 768,
        "buckets": DEFAULT_BUCKETS,
        "batch_sizes": DEFAULT_BATCH_SIZES,
    },
    "e5-large": {
        "name": "e5-large",
        "model_id": "intfloat/multilingual-e5-large",
        "hf_repo": "intfloat/multilingual-e5-large",
        "hf_revision": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
        "onnx_repo": "Xenova/multilingual-e5-large",
        "onnx_revision": "00fc3aeb3dbb95842de2ac1961d33c6319acf57b",
        "onnx_path": "onnx/model.onnx",
        "hidden_dim": 1024,
        "buckets": DEFAULT_BUCKETS,
        "batch_sizes": DEFAULT_BATCH_SIZES,
    },
}
MODEL_ALIASES = {
    "e5": "e5-small",
    "small": "e5-small",
    "multilingual-e5-small": "e5-small",
    "intfloat/multilingual-e5-small": "e5-small",
    "intfloat/e5-small-v2": "e5-small-v2",
    "intfloat/multilingual-e5-base": "e5-base",
    "intfloat/multilingual-e5-large": "e5-large",
}

VERBOSE = False
QUIET = False


def log(msg="", *, force=False):
    if force or not QUIET:
        print(msg, flush=True)


def run(cmd, *, cwd=None, check=False, capture=False, env=None):
    if VERBOSE:
        log("  $ " + " ".join(str(c) for c in cmd), force=True)
    kwargs = {
        "cwd": cwd,
        "env": env,
        "text": True,
    }
    if capture:
        kwargs.update({"stdout": subprocess.PIPE, "stderr": subprocess.PIPE})
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def captured_stdout(cmd):
    try:
        result = run(cmd, capture=True)
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    return out or None


def detect_native_trt_version():
    out = captured_stdout([
        sys.executable,
        "-c",
        "import tensorrt as trt; print(trt.__version__)",
    ])
    if out:
        return out.splitlines()[-1].strip()

    out = captured_stdout(["trtexec", "--version"])
    if out:
        for line in out.splitlines():
            if "TensorRT" in line or "Version" in line:
                return line.strip()
        return out.splitlines()[0].strip()

    for package in ["libnvinfer10", "libnvinfer-dev", "tensorrt"]:
        out = captured_stdout(["dpkg-query", "-W", "-f=${Version}", package])
        if out:
            return out.strip()
    return None


def detect_docker_trt_version(image):
    out = captured_stdout([
        "docker", "run", "--rm", "--entrypoint", "python3", image,
        "-c", "import tensorrt as trt; print(trt.__version__)",
    ])
    if out:
        return out.splitlines()[-1].strip()

    out = captured_stdout([
        "docker", "run", "--rm", "--entrypoint", "dpkg-query", image,
        "-W", "-f=${Version}", "libnvinfer10",
    ])
    return out.strip() if out else None


def docker_image_digest(image):
    out = captured_stdout(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image])
    if not out:
        return None
    try:
        digests = json.loads(out)
    except json.JSONDecodeError:
        return out
    return digests[0] if digests else None


def ensure_python_package(import_name, package_name):
    if importlib.util.find_spec(import_name) is not None:
        return
    log(f"  Installing Python package: {package_name}")
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--root-user-action=ignore",
        "--disable-pip-version-check",
        package_name,
    ], check=True)


def ensure_python_package_version(import_name, package_name, expected_version):
    installed = None
    if importlib.util.find_spec(import_name) is not None:
        dist_name = package_name.split("==", 1)[0]
        try:
            installed = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
    if installed == expected_version:
        return
    log(f"  Installing Python package: {package_name}")
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--root-user-action=ignore",
        "--disable-pip-version-check",
        package_name,
    ], check=True)


def ensure_python_deps(skip_tei, native=False):
    ensure_python_package("datasets", "datasets")
    if native:
        ensure_python_package_version("onnx", "onnx==1.16.1", "1.16.1")
        ensure_python_package_version("numpy", "numpy==1.26.4", "1.26.4")
        ensure_python_package_version("tokenizers", "tokenizers==0.20.3", "0.20.3")
    if not skip_tei:
        ensure_python_package("aiohttp", "aiohttp")


def default_data_dir():
    if Path("/mnt/nvme").is_dir():
        return Path("/mnt/nvme/benchmark")
    return Path("./ignite-ms-data")


def default_cache_root():
    if Path("/mnt/nvme").is_dir():
        return Path("/mnt/nvme/ignite-ms-bench-cache")
    return Path("./ignite-ms-bench-cache")


def default_input_path(texts):
    if texts == DEFAULT_TEXTS:
        return DATA_DIR / "msmarco_medium_1m_passage.jsonl"
    return DATA_DIR / f"msmarco_medium_{texts}_passage.jsonl"


def resolve_model_spec(model):
    key = MODEL_ALIASES.get(model, model)
    if key in MODEL_SPECS:
        return MODEL_SPECS[key]
    supported = ", ".join(sorted(MODEL_SPECS))
    raise SystemExit(
        f"ERROR: unsupported model for native auto-provisioning: {model}. "
        f"Use one of: {supported}, or provide a complete --model-dir."
    )


def nonempty_file(path, min_bytes=1):
    return path.is_file() and path.stat().st_size >= min_bytes


def hf_url(repo, revision, file_path):
    return f"https://huggingface.co/{repo}/resolve/{revision}/{file_path}"


def download_file(url, dest):
    if nonempty_file(dest):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    log(f"  Downloading {dest.name}...")
    if shutil.which("curl"):
        result = run(["curl", "-fL", "--retry", "3", "-o", str(tmp), url], capture=True)
        if result.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"ERROR: download failed: {url}\n{result.stderr.strip()}")
    else:
        try:
            with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as f:
                shutil.copyfileobj(response, f)
        except (urllib.error.URLError, TimeoutError) as e:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"ERROR: download failed: {url}\n{e}") from e
    if not nonempty_file(tmp):
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"ERROR: downloaded empty file: {url}")
    tmp.replace(dest)


def download_hf_file(repo, revision, file_path, dest):
    download_file(hf_url(repo, revision, file_path), dest)


def bucket_args_for_spec(spec):
    return ["--buckets", spec["buckets"], "--batch-sizes", spec["batch_sizes"]]


def expected_engine_files(model_dir, spec):
    buckets = parse_csv_ints(spec["buckets"], "model buckets")
    batch_sizes = parse_csv_ints(spec["batch_sizes"], "model batch sizes")
    if len(buckets) != len(batch_sizes):
        raise SystemExit("ERROR: model bucket and batch-size counts differ")
    return [model_dir / f"model_b{bs}_s{bucket}.engine" for bucket, bs in zip(buckets, batch_sizes)]


def missing_engine_files(model_dir, spec):
    return [p for p in expected_engine_files(model_dir, spec) if not nonempty_file(p, MIN_ENGINE_BYTES)]


def vocab_cache_count(path):
    if not path.is_file():
        return None
    with path.open("rb") as f:
        if f.read(8) != b"IMSVCACH":
            return None
        version = struct.unpack("<I", f.read(4))[0]
        if version != 1:
            return None
        return struct.unpack("<Q", f.read(8))[0]


def file_fingerprint(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def vocab_cache_meta_path(vocab_cache):
    return vocab_cache.with_suffix(vocab_cache.suffix + ".meta.json")


def vocab_cache_current(vocab_cache, input_path, tokenizer_path, sample_n):
    meta_path = vocab_cache_meta_path(vocab_cache)
    if not nonempty_file(vocab_cache) or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected = {
        "builder": "corpus-v1",
        "input": file_fingerprint(input_path),
        "tokenizer": file_fingerprint(tokenizer_path),
        "sample_n": sample_n,
    }
    return meta == expected


def tokenizer_vocab_cache_current(vocab_cache, tokenizer_path):
    meta_path = vocab_cache_meta_path(vocab_cache)
    if not nonempty_file(vocab_cache) or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected = {
        "builder": "tokenizer-v1",
        "tokenizer": file_fingerprint(tokenizer_path),
    }
    return meta == expected


def write_tokenizer_vocab_cache_meta(vocab_cache, tokenizer_path):
    meta = {
        "builder": "tokenizer-v1",
        "tokenizer": file_fingerprint(tokenizer_path),
    }
    vocab_cache_meta_path(vocab_cache).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_corpus_vocab_cache(input_path, tokenizer_path, vocab_cache, sample_n=2_000_000):
    from tokenizers import Tokenizer

    log("  Building corpus vocab cache...")
    words = Counter()
    count = 0
    with input_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"ERROR: invalid JSONL at {input_path}:{line_no}: {e}") from e
            text = obj.get("text")
            if not isinstance(text, str):
                continue
            for word in text.lower().split():
                if len(word) <= 45 and any(c.isalpha() for c in word):
                    words[word] += 1
            count += 1
            if count >= sample_n:
                break

    tok = Tokenizer.from_file(str(tokenizer_path))
    cache = {}
    word_prefix = chr(9601)
    for word, _ in words.most_common(1_500_000):
        if not word or len(word) > 45 or word.isdigit():
            continue
        key = word_prefix + word
        if key in cache:
            continue
        enc = tok.encode(key, add_special_tokens=False)
        if enc.ids:
            cache[key] = enc.ids

    tmp = vocab_cache.with_suffix(vocab_cache.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(b"IMSVCACH")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", len(cache)))
        for key, ids in cache.items():
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<I", len(key_bytes)))
            f.write(key_bytes)
            f.write(struct.pack("<I", len(ids)))
            for token_id in ids:
                f.write(struct.pack("<I", token_id))
    tmp.replace(vocab_cache)

    meta = {
        "builder": "corpus-v1",
        "input": file_fingerprint(input_path),
        "tokenizer": file_fingerprint(tokenizer_path),
        "sample_n": sample_n,
    }
    vocab_cache_meta_path(vocab_cache).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"  Vocab cache:   {len(cache)} entries", force=True)


def repo_script(name):
    return Path(__file__).resolve().parent / "crates" / "ignite-ms" / "scripts" / name


def model_meta_path(model_dir):
    return model_dir / ".ignite-ms-model.json"


def model_meta(spec):
    return {
        "model_id": spec["model_id"],
        "hf_repo": spec["hf_repo"],
        "hf_revision": spec["hf_revision"],
        "onnx_repo": spec["onnx_repo"],
        "onnx_revision": spec["onnx_revision"],
        "onnx_path": spec["onnx_path"],
        "hidden_dim": spec["hidden_dim"],
    }


def model_cache_current(model_dir, spec):
    path = model_meta_path(model_dir)
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")) == model_meta(spec)
    except json.JSONDecodeError:
        return False


def clear_model_artifacts(model_dir):
    for name in [
        "tokenizer.json",
        "config.json",
        "model.onnx",
        "model.onnx_data",
        "model_raw.onnx",
        "vocab_cache.bin",
        "vocab_cache.bin.meta.json",
        ".ignite-ms-model.json",
    ]:
        (model_dir / name).unlink(missing_ok=True)
    for path in model_dir.glob("model_b*_s*.engine"):
        path.unlink(missing_ok=True)


def engine_meta_path(model_dir):
    return model_dir / ".ignite-ms-engines.json"


def engine_meta(spec, runtime_info):
    return {
        "model_id": spec["model_id"],
        "buckets": spec["buckets"],
        "batch_sizes": spec["batch_sizes"],
        "runtime_mode": runtime_info.get("mode"),
        "tensorrt_version": runtime_info.get("tensorrt_version"),
        "docker_image_digest": runtime_info.get("docker_image_digest"),
    }


def engine_cache_current(model_dir, spec, runtime_info):
    path = engine_meta_path(model_dir)
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")) == engine_meta(spec, runtime_info)
    except json.JSONDecodeError:
        return False


def clear_engine_artifacts(model_dir):
    for path in model_dir.glob("model_b*_s*.engine"):
        path.unlink(missing_ok=True)
    engine_meta_path(model_dir).unlink(missing_ok=True)


def write_engine_meta(model_dir, spec, runtime_info):
    engine_meta_path(model_dir).write_text(
        json.dumps(engine_meta(spec, runtime_info), indent=2),
        encoding="utf-8",
    )


def prepare_docker_model_dir(host_model_dir, spec, cache_root):
    docker_model_dir = cache_root / "docker-models" / spec["name"]
    docker_model_dir.mkdir(parents=True, exist_ok=True)

    if not model_cache_current(docker_model_dir, spec):
        if any(docker_model_dir.iterdir()):
            log("  Docker model cache does not match requested model; replacing cached artifacts.")
        clear_model_artifacts(docker_model_dir)

    for name in [
        "tokenizer.json",
        "config.json",
        "model.onnx",
        "model.onnx_data",
        "vocab_cache.bin",
        "vocab_cache.bin.meta.json",
        ".ignite-ms-model.json",
    ]:
        src = host_model_dir / name
        if src.exists():
            shutil.copy2(src, docker_model_dir / name)

    missing = [
        name for name in ["tokenizer.json", "config.json", "model.onnx", "vocab_cache.bin"]
        if not nonempty_file(docker_model_dir / name)
    ]
    if missing:
        raise SystemExit(f"ERROR: Docker model cache missing required files: {', '.join(missing)}")

    log(f"  Docker model: {docker_model_dir}", force=True)
    return docker_model_dir


def prepare_docker_engines(input_path, model_dir, spec, gpu_id, docker_image, runtime_info):
    if missing_engine_files(model_dir, spec) == [] and not engine_cache_current(model_dir, spec, runtime_info):
        log("  Docker TensorRT engine metadata changed; rebuilding engines.")
        clear_engine_artifacts(model_dir)

    missing = missing_engine_files(model_dir, spec)
    if not missing:
        log("  Docker TensorRT engines cached.", force=True)
        write_engine_meta(model_dir, spec, runtime_info)
        return

    log("  Docker TensorRT engines missing; compiling once before benchmark...")
    for path in missing:
        log(f"    missing: {path.name}")

    cmd = [
        "docker", "run", "--rm",
        "--gpus", docker_gpus_arg([gpu_id]),
        "-v", f"{input_path.resolve().parent}:/input:ro",
        "-v", f"{model_dir.resolve()}:/model",
        "--entrypoint", "ignite-ms-bench",
        docker_image,
        "--input", f"/input/{input_path.resolve().name}",
        "--format", "jsonl",
        "--model-dir", "/model",
        "--gpus", str(gpu_id),
        "--truncation", "512",
        "--max-messages", "1",
        "--warmup", "0",
    ]
    cmd.extend(bucket_args_for_spec(spec))
    rc, _, lines = stream_process(cmd, interesting=("compil", "engine", "init", "ERROR", "WARNING"))
    if rc != 0:
        raise SystemExit("ERROR: Docker TensorRT engine build failed\n" + tail(lines))

    missing = missing_engine_files(model_dir, spec)
    if missing:
        names = ", ".join(p.name for p in missing)
        raise SystemExit(f"ERROR: Docker engine build did not create expected files: {names}")
    write_engine_meta(model_dir, spec, runtime_info)


def prepare_native_model(binary, model, model_dir_arg, cache_root, input_path, gpu_id, runtime_info=None):
    runtime_info = runtime_info or {"mode": "native"}
    spec = resolve_model_spec(model)
    model_dir = model_dir_arg or cache_root / "models" / spec["name"]
    model_dir.mkdir(parents=True, exist_ok=True)

    log(f"  Native model:  {spec['model_id']}", force=True)
    log(f"  Model cache:   {model_dir}", force=True)

    if not model_cache_current(model_dir, spec):
        if any(model_dir.iterdir()):
            log("  Model cache does not match requested model; replacing cached artifacts.")
        clear_model_artifacts(model_dir)

    download_hf_file(spec["hf_repo"], spec["hf_revision"], "tokenizer.json", model_dir / "tokenizer.json")
    download_hf_file(spec["hf_repo"], spec["hf_revision"], "config.json", model_dir / "config.json")

    pooled_onnx = model_dir / "model.onnx"
    if not nonempty_file(pooled_onnx):
        raw_onnx = model_dir / "model_raw.onnx"
        download_hf_file(spec["onnx_repo"], spec["onnx_revision"], spec["onnx_path"], raw_onnx)
        log("  Exporting pooled ONNX model...")
        run([
            sys.executable,
            str(repo_script("export_pooling.py")),
            "--input",
            str(raw_onnx),
            "--output",
            str(pooled_onnx),
            "--hidden-dim",
            str(spec["hidden_dim"]),
        ], check=True)
    if not nonempty_file(pooled_onnx):
        raise SystemExit(f"ERROR: model.onnx was not created: {pooled_onnx}")

    vocab_cache = model_dir / "vocab_cache.bin"
    tokenizer_path = model_dir / "tokenizer.json"
    if vocab_cache_current(vocab_cache, input_path, tokenizer_path, 2_000_000):
        log(f"  Vocab cache:   cached ({vocab_cache_count(vocab_cache)} entries)", force=True)
    else:
        existing_count = vocab_cache_count(vocab_cache)
        if existing_count is not None:
            log(f"  Rebuilding vocab cache from benchmark corpus (old cache had {existing_count} entries).")
        build_corpus_vocab_cache(input_path, tokenizer_path, vocab_cache, 2_000_000)
    if not nonempty_file(vocab_cache):
        raise SystemExit(f"ERROR: vocab_cache.bin was not created: {vocab_cache}")

    if missing_engine_files(model_dir, spec) == [] and not engine_cache_current(model_dir, spec, runtime_info):
        log("  TensorRT engine metadata changed; rebuilding engines.")
        clear_engine_artifacts(model_dir)

    missing = missing_engine_files(model_dir, spec)
    if missing:
        log("  TensorRT engines missing; compiling once before benchmark...")
        for path in missing:
            log(f"    missing: {path.name}")
        cmd = [
            str(binary),
            "--input", str(input_path.resolve()),
            "--format", "jsonl",
            "--model-dir", str(model_dir),
            "--gpus", str(gpu_id),
            "--truncation", "512",
            "--max-messages", "1",
            "--warmup", "0",
        ]
        cmd.extend(bucket_args_for_spec(spec))
        rc, _, lines = stream_process(cmd, interesting=("compil", "engine", "init", "ERROR", "WARNING"))
        if rc != 0:
            raise SystemExit("ERROR: TensorRT engine build failed\n" + tail(lines))
        missing = missing_engine_files(model_dir, spec)
        if missing:
            names = ", ".join(p.name for p in missing)
            raise SystemExit(f"ERROR: TensorRT engine build did not create expected files: {names}")
    else:
        log("  TensorRT engines cached.", force=True)

    write_engine_meta(model_dir, spec, runtime_info)
    model_meta_path(model_dir).write_text(json.dumps(model_meta(spec), indent=2), encoding="utf-8")
    return spec, model_dir


def detect_gpus():
    try:
        result = run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture=True,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    ids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                ids.append(int(line))
            except ValueError:
                pass
    return ids


def parse_csv_ints(value, name):
    try:
        vals = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as e:
        raise SystemExit(f"ERROR: invalid {name}: {value}") from e
    if not vals:
        raise SystemExit(f"ERROR: {name} cannot be empty")
    return vals


def resolve_gpu_counts(spec, available_gpus):
    n = len(available_gpus)
    if n == 0:
        raise SystemExit("ERROR: no NVIDIA GPUs detected with nvidia-smi")
    if spec != "auto":
        counts = parse_csv_ints(spec, "--gpu-counts")
    elif n >= 8:
        counts = [1, 8]
    elif n == 1:
        counts = [1]
    else:
        counts = [1, n]
        log(f"  WARNING: auto GPU counts use [1, 8] on recommended 8-GPU hosts; using [1, {n}] here.")

    for count in counts:
        if count < 1:
            raise SystemExit("ERROR: GPU counts must be positive")
        if count > n:
            raise SystemExit(f"ERROR: requested {count} GPUs but only {n} detected")
    return counts


def gpu_ids_for_count(count, available_gpus):
    return available_gpus[:count]


def docker_gpus_arg(gpu_ids):
    gpu_spec = ",".join(str(g) for g in gpu_ids)
    if len(gpu_ids) == 1:
        return f"device={gpu_spec}"
    return f'"device={gpu_spec}"'


def ensure_docker_ready():
    if not shutil.which("docker"):
        try_install_docker()
    if not shutil.which("docker"):
        raise SystemExit("ERROR: Docker is not installed. Install Docker or run with --mode native.")

    result = run(["docker", "info"], capture=True)
    if result.returncode != 0:
        raise SystemExit("ERROR: Docker is not running or the current user cannot access it.")

    image = "nvidia/cuda:12.4.1-base-ubuntu22.04"
    result = run(["docker", "run", "--rm", "--gpus", "all", image, "nvidia-smi"], capture=True)
    if result.returncode != 0:
        raise SystemExit(
            "ERROR: Docker GPU access failed. Install/configure nvidia-container-toolkit, "
            "then verify `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`."
        )


def try_install_docker():
    runner = []
    if os.geteuid() != 0:
        if not shutil.which("sudo"):
            return
        runner = ["sudo"]

    if shutil.which("apt-get"):
        log("  Docker not found. Installing Docker with apt-get...")
        run(runner + ["apt-get", "update"], check=False)
        run(runner + ["apt-get", "install", "-y", "docker.io"], check=False)
    elif shutil.which("dnf"):
        log("  Docker not found. Installing Docker with dnf...")
        run(runner + ["dnf", "install", "-y", "docker"], check=False)
    elif shutil.which("yum"):
        log("  Docker not found. Installing Docker with yum...")
        run(runner + ["yum", "install", "-y", "docker"], check=False)

    if shutil.which("systemctl"):
        run(runner + ["systemctl", "enable", "--now", "docker"], check=False)


def pull_image(image):
    result = run(["docker", "image", "inspect", image], capture=True)
    if result.returncode == 0:
        return
    log(f"  Pulling {image}...")
    run(["docker", "pull", image], check=True)


def ensure_docker_bench_binary(image):
    result = run(
        ["docker", "run", "--rm", "--entrypoint", "sh", image, "-lc", "command -v ignite-ms-bench"],
        capture=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "ERROR: Docker image does not contain ignite-ms-bench. "
            "Rebuild/publish the Docker image from the updated Dockerfile, or use --mode native. "
            "The old image only supports `ignite-ms embed`, which writes embeddings and is not a raw benchmark."
        )


def ensure_rust():
    if shutil.which("cargo"):
        return
    log("  Rust not found. Installing rustup toolchain...")
    cmd = "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    result = subprocess.run(["sh", "-c", cmd])
    if result.returncode != 0:
        raise SystemExit("ERROR: failed to install Rust")
    cargo_bin = Path.home() / ".cargo" / "bin"
    os.environ["PATH"] = f"{cargo_bin}:{os.environ['PATH']}"


def check_native_deps():
    ensure_rust()
    missing = []
    for path in ["/usr/include/x86_64-linux-gnu/NvInfer.h", "/usr/include/NvInfer.h"]:
        if Path(path).exists():
            break
    else:
        missing.append("TensorRT headers")
    if not shutil.which("nvidia-smi"):
        missing.append("nvidia-smi")
    if missing:
        raise SystemExit(
            "ERROR: native mode is missing " + ", ".join(missing) +
            ". Use --mode docker or install CUDA 12.x + TensorRT 10.x development packages."
        )


def build_native_binary():
    check_native_deps()
    repo_root = Path(__file__).resolve().parent
    log("  Building ignite-ms native benchmark binary...")
    env = os.environ.copy()
    env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " -A warnings").strip()
    run(["cargo", "build", "--release", "-p", "ignite-ms-bench"], cwd=repo_root, check=True, env=env)
    bench = repo_root / "target" / "release" / "ignite-ms-bench"
    if not bench.exists():
        raise SystemExit(f"ERROR: native benchmark binary not found: {bench}")
    return bench


def prepare_default_input(path, max_texts):
    if path.exists():
        existing = count_lines(path)
        if existing >= max_texts:
            log(f"  Dataset cached: {path} ({existing:,} texts)")
            return existing
        log(f"  Dataset exists but has only {existing:,} texts; regenerating.")
        path.unlink()

    ensure_python_package("datasets", "datasets")
    from datasets import load_dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    log("  Generating default benchmark input from MSMARCO v2.1...")
    log("  Slice: 1M passage-prefixed texts, original length 50-256 chars.")

    ds = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)
    count = 0
    with tmp.open("w", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
        for item in ds:
            passages = item.get("passages", {})
            texts = passages.get("passage_text", []) if isinstance(passages, dict) else []
            for text in texts:
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if 50 <= len(text) <= 256:
                    f.write(json.dumps({"id": str(count), "text": "passage: " + text}))
                    f.write("\n")
                    count += 1
                    if count % 100_000 == 0:
                        log(f"    {count:,} texts")
                    if count >= max_texts:
                        break
            if count >= max_texts:
                break

    if count == 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit("ERROR: default dataset generation produced no texts")
    tmp.replace(path)
    log(f"  Dataset ready: {path} ({count:,} texts)")
    return count


def limit_input_file(input_path, max_texts):
    total = validate_input_jsonl(input_path)
    if total <= max_texts:
        return input_path, total

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stat = input_path.stat()
    key = f"{input_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{max_texts}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    limited = DATA_DIR / f"{input_path.stem}_first_{max_texts}_{digest}{input_path.suffix}"
    if limited.exists() and count_lines(limited) == max_texts:
        return limited, max_texts

    tmp = limited.with_suffix(limited.suffix + ".tmp")
    written = 0
    with input_path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(line)
            written += 1
            if written >= max_texts:
                break
    tmp.replace(limited)
    return limited, written


def validate_input_jsonl(path, max_texts=None):
    if not path.exists():
        raise SystemExit(f"ERROR: input file not found: {path}")
    count = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if max_texts is not None and count >= max_texts:
                break
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"ERROR: invalid JSONL at {path}:{line_no}: {e}") from e
            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(f"ERROR: missing non-empty text field at {path}:{line_no}")
            count += 1
    if count == 0:
        raise SystemExit(f"ERROR: input has no usable rows: {path}")
    return count


def count_lines(path):
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def parse_ignite_throughput(stderr_lines):
    for line in stderr_lines:
        if "Throughput:" not in line:
            continue
        parts = line.replace(",", "").split()
        for i, part in enumerate(parts):
            if part == "Throughput:" and i + 1 < len(parts):
                try:
                    return float(parts[i + 1])
                except ValueError:
                    pass
    return None


def parse_bench_throughput(lines):
    for line in lines:
        stripped = line.strip().replace(",", "")
        if stripped.startswith("throughput:"):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass
    return None


def stream_process(cmd, interesting=()):
    if VERBOSE:
        log("  $ " + " ".join(str(c) for c in cmd), force=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        if VERBOSE or (not QUIET and any(k in line for k in interesting)):
            log("    " + line.rstrip())
    proc.wait()
    return proc.returncode, [], lines


def run_ignite_docker(input_path, model_dir, spec, gpu_ids, docker_image, n_texts):
    input_path = input_path.resolve()
    model_dir = model_dir.resolve()
    gpu_spec = ",".join(str(g) for g in gpu_ids)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", docker_gpus_arg(gpu_ids),
        "-v", f"{input_path.parent}:/input:ro",
        "-v", f"{model_dir}:/model",
        "--entrypoint", "ignite-ms-bench",
        docker_image,
        "--input", f"/input/{input_path.name}",
        "--format", "jsonl",
        "--model-dir", "/model",
        "--gpus", gpu_spec,
        "--truncation", "512",
        "--max-messages", str(n_texts),
        "--warmup", "10000",
        "--latency",
    ]
    cmd.extend(bucket_args_for_spec(spec))

    t0 = time.time()
    rc, _, lines = stream_process(
        cmd,
        interesting=("download", "compil", "engine", "ready", "init", "warm-up", "RESULTS", "throughput", "latency", "ERROR", "WARNING"),
    )
    elapsed = time.time() - t0
    if rc != 0:
        return {"ok": False, "error": tail(lines), "elapsed": elapsed}
    throughput = parse_bench_throughput(lines)
    if throughput is None:
        throughput = validate_input_jsonl(input_path) / elapsed if elapsed > 0 else 0
    return {"ok": True, "throughput": throughput, "elapsed": elapsed, "errors": 0}


def run_ignite_native(binary, input_path, model_dir, spec, gpu_ids, n_texts):
    input_path = input_path.resolve()
    gpu_spec = ",".join(str(g) for g in gpu_ids)
    cmd = [
        str(binary),
        "--input", str(input_path),
        "--format", "jsonl",
        "--model-dir", str(model_dir),
        "--gpus", gpu_spec,
        "--truncation", "512",
        "--max-messages", str(n_texts),
        "--warmup", "10000",
        "--latency",
    ]
    cmd.extend(bucket_args_for_spec(spec))

    t0 = time.time()
    rc, _, lines = stream_process(
        cmd,
        interesting=("init", "warm-up", "RESULTS", "throughput", "latency", "ERROR", "WARNING"),
    )
    elapsed = time.time() - t0
    if rc != 0:
        return {"ok": False, "error": tail(lines), "elapsed": elapsed}
    throughput = parse_bench_throughput(lines)
    if throughput is None:
        throughput = validate_input_jsonl(input_path) / elapsed if elapsed > 0 else 0
    return {"ok": True, "throughput": throughput, "elapsed": elapsed, "errors": 0}


def tail(lines, n=20):
    return "".join(lines[-n:]).strip()


def stop_tei_workers():
    if not shutil.which("docker"):
        return
    result = run(
        ["docker", "ps", "-aq", "--filter", "name=tei-bench-"],
        capture=True,
    )
    ids = [x for x in result.stdout.splitlines() if x.strip()] if result.returncode == 0 else []
    if ids:
        run(["docker", "rm", "-f", *ids], capture=True)


def start_tei_workers(model, gpu_ids, tei_image):
    stop_tei_workers()
    for idx, gpu_id in enumerate(gpu_ids):
        port = TEI_PORT_BASE + idx
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", f"tei-bench-{idx}",
            "--gpus", f"device={gpu_id}",
            "-p", f"{port}:80",
            "-v", "tei-bench-cache:/data",
            tei_image,
            "--model-id", model,
            "--pooling", "mean",
            "--auto-truncate",
            "--max-client-batch-size", "64",
            "--max-batch-tokens", "65536",
            "--max-concurrent-requests", "256",
            "--dtype", "float16",
        ]
        result = run(cmd, capture=True)
        if result.returncode != 0:
            stop_tei_workers()
            return False, f"failed to start TEI worker {idx}: {result.stderr.strip()}"

    deadline = time.time() + 300
    last_notice = 0
    while time.time() < deadline:
        ready = True
        for idx in range(len(gpu_ids)):
            if not tei_health_ok(TEI_PORT_BASE + idx):
                ready = False
                break
        if ready:
            log("    TEI health OK; verifying inference...")
            if tei_inference_ready(TEI_PORT_BASE):
                time.sleep(5)
                return True, None
            return False, "TEI health passed but /embed inference did not become ready"
        elapsed = int(300 - (deadline - time.time()))
        if elapsed - last_notice >= 30:
            last_notice = elapsed
            log(f"    TEI starting... {elapsed}s")
        time.sleep(1)

    logs = []
    for idx in range(len(gpu_ids)):
        result = run(["docker", "logs", f"tei-bench-{idx}"], capture=True)
        logs.append(result.stderr or result.stdout)
    stop_tei_workers()
    return False, "TEI failed to start\n" + "\n".join(logs)[-4000:]


def tei_health_ok(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def tei_inference_ready(port):
    payload = json.dumps({"inputs": ["warmup test"]}).encode()
    for _ in range(30):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status == 200:
                    resp.read()
                    return True
        except Exception:
            time.sleep(2)
    return False


async def tei_client(endpoints, input_path, n_texts, batch_size, concurrency, warmup_batches):
    import aiohttp

    texts = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            if len(texts) >= n_texts:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                texts.append(text)

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def post_embed(endpoint, batch):
            last_error = None
            for attempt in range(TEI_REQUEST_RETRIES):
                try:
                    async with session.post(endpoint + "/embed", json={"inputs": batch}) as resp:
                        if resp.status == 200:
                            await resp.read()
                            return True, None
                        body = (await resp.text())[:400]
                        last_error = f"status={resp.status} body={body}"
                        if resp.status not in (429, 503, 504):
                            return False, last_error
                except Exception as e:
                    last_error = repr(e)
                delay = min(0.25 * (2 ** attempt), 8.0) + random.random() * 0.1
                await asyncio.sleep(delay)
            return False, last_error

        for idx, batch in enumerate(batches[:warmup_batches]):
            endpoint = endpoints[idx % len(endpoints)]
            ok, err = await post_embed(endpoint, batch)
            if not ok:
                return {"ok": False, "error": f"TEI warmup failed: {err}", "errors": 1}

        measure_batches = batches[warmup_batches:]
        completed = 0
        errors = 0
        first_error = None
        sem = asyncio.Semaphore(concurrency)

        async def send(idx, batch):
            nonlocal completed, errors, first_error
            endpoint = endpoints[idx % len(endpoints)]
            async with sem:
                ok, err = await post_embed(endpoint, batch)
                if ok:
                    completed += len(batch)
                else:
                    errors += 1
                    if first_error is None:
                        first_error = err

        t0 = time.time()
        await asyncio.gather(*(send(i, b) for i, b in enumerate(measure_batches)))
        elapsed = time.time() - t0

    throughput = completed / elapsed if elapsed > 0 else 0
    return {
        "ok": errors == 0 and completed > 0,
        "throughput": throughput,
        "elapsed": elapsed,
        "measured_texts": completed,
        "errors": errors,
        "error": first_error,
    }


def run_tei(input_path, model, gpu_ids, tei_image, n_texts, batch_size, concurrency_per_worker):
    ok, error = start_tei_workers(model, gpu_ids, tei_image)
    if not ok:
        return {"ok": False, "error": error, "errors": 1}
    try:
        endpoints = [f"http://127.0.0.1:{TEI_PORT_BASE + i}" for i in range(len(gpu_ids))]
        return asyncio.run(
            tei_client(
                endpoints=endpoints,
                input_path=input_path,
                n_texts=n_texts,
                batch_size=batch_size,
                concurrency=concurrency_per_worker * len(gpu_ids),
                warmup_batches=50,
            )
        )
    finally:
        stop_tei_workers()


def print_results(results, model, input_path, tei_batch, tei_concurrency_per_worker):
    log("", force=True)
    log("  ignite-ms vs TEI", force=True)
    log(f"  Model: {model}", force=True)
    log(f"  Input: {input_path}", force=True)
    log(f"  TEI: batch={tei_batch}, concurrency_per_worker={tei_concurrency_per_worker}", force=True)
    log("", force=True)
    log(f"  {'GPUs':<6} {'ignite-ms':<16} {'TEI':<16} {'Speedup':<10} {'Notes'}", force=True)
    log(f"  {'-' * 6} {'-' * 16} {'-' * 16} {'-' * 10} {'-' * 20}", force=True)
    for row in results:
        ims = fmt_rate(row.get("ignite_ms"))
        tei = fmt_rate(row.get("tei"))
        speedup = "-"
        if row.get("ignite_ms") and row.get("tei"):
            speedup = f"{row['ignite_ms'] / row['tei']:.2f}x"
        notes = row.get("notes", "")
        log(f"  {row['gpus']:<6} {ims:<16} {tei:<16} {speedup:<10} {notes}", force=True)
    log("", force=True)


def fmt_rate(value):
    if not value:
        return "-"
    return f"{value:,.0f}/s"


def save_results(results, args, input_path, n_texts, model_dir=None, model_id=None, runtime_info=None, spec=None):
    out = Path(args.output)
    data = {
        "model": args.model,
        "model_id": model_id or args.model,
        "model_dir": str(model_dir) if model_dir else (str(args.model_dir) if args.model_dir else None),
        "runtime": runtime_info or {},
        "buckets": spec["buckets"] if spec else None,
        "batch_sizes": spec["batch_sizes"] if spec else None,
        "mode": args.mode,
        "input": str(input_path),
        "n_texts": n_texts,
        "gpu_counts": [r["gpus"] for r in results],
        "docker_image": args.ignite_image if args.mode == "docker" else None,
        "tei_image": None if args.skip_tei else args.tei_image,
        "tei_batch": args.tei_batch,
        "tei_concurrency_per_worker": args.tei_concurrency_per_worker,
        "results": results,
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["gpus", "system", "throughput_msg_s", "measured_seconds", "errors", "notes"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({
                "gpus": row["gpus"],
                "system": "ignite-ms",
                "throughput_msg_s": row.get("ignite_ms") or "",
                "measured_seconds": row.get("ignite_elapsed") or "",
                "errors": row.get("ignite_errors") or 0,
                "notes": row.get("ignite_note") or "",
            })
            if not args.skip_tei:
                writer.writerow({
                    "gpus": row["gpus"],
                    "system": "tei",
                    "throughput_msg_s": row.get("tei") or "",
                    "measured_seconds": row.get("tei_elapsed") or "",
                    "errors": row.get("tei_errors") or 0,
                    "notes": row.get("tei_note") or "",
                })
    log(f"  Results JSON: {out}", force=True)
    log(f"  Results CSV:  {csv_path}", force=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ignite-ms and TEI on a public MSMARCO benchmark slice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["docker", "native"], default="docker")
    parser.add_argument("--native", action="store_true", help="Deprecated alias for --mode native.")
    parser.add_argument("--input", type=Path, help="Input JSONL with a text field. If omitted, MSMARCO v2.1 1M passage slice is generated.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-dir", type=Path, help="Native mode model directory with tokenizer/model.onnx/TRT engines. Overrides --model provisioning path for ignite-ms.")
    parser.add_argument("--cache-root", type=Path, help="Native mode cache root for downloaded models and TensorRT engines.")
    parser.add_argument("--data-dir", type=Path, help="Directory for generated default input and temporary limited custom inputs.")
    parser.add_argument("--texts", type=int, default=DEFAULT_TEXTS, help="Rows to use/generate.")
    parser.add_argument("--gpu-counts", default=DEFAULT_GPU_COUNTS, help='Comma-separated counts, or "auto". Auto is 1,8 on 8-GPU hosts.')
    parser.add_argument("--skip-tei", action="store_true")
    parser.add_argument("--tei-batch", type=int, default=TEI_BATCH)
    parser.add_argument("--tei-concurrency-per-worker", type=int, default=TEI_CONC_PER_WORKER)
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--ignite-image", default=DOCKER_IMAGE)
    parser.add_argument("--tei-image", default=TEI_IMAGE)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    global VERBOSE, QUIET, DATA_DIR
    VERBOSE = args.verbose
    QUIET = args.quiet
    if args.native:
        args.mode = "native"
    if args.tei_batch < 1:
        raise SystemExit("ERROR: --tei-batch must be positive")
    if args.tei_concurrency_per_worker < 1:
        raise SystemExit("ERROR: --tei-concurrency-per-worker must be positive")
    DATA_DIR = args.data_dir or default_data_dir()
    cache_root = args.cache_root or default_cache_root()

    log("", force=True)
    log("=" * 72, force=True)
    log("  ignite-ms benchmark", force=True)
    log("=" * 72, force=True)
    log("", force=True)

    ensure_python_deps(args.skip_tei, native=args.mode == "native")
    available_gpus = detect_gpus()
    gpu_counts = resolve_gpu_counts(args.gpu_counts, available_gpus)
    log(f"  GPUs detected: {available_gpus}", force=True)
    log(f"  GPU counts:    {gpu_counts}", force=True)
    log(f"  Mode:          {args.mode}", force=True)
    log(f"  Model:         {args.model}", force=True)

    if args.mode == "docker":
        ensure_docker_ready()
        pull_image(args.ignite_image)
        ensure_docker_bench_binary(args.ignite_image)
        runtime_info = {
            "mode": "docker",
            "tensorrt_version": detect_docker_trt_version(args.ignite_image),
            "docker_image": args.ignite_image,
            "docker_image_digest": docker_image_digest(args.ignite_image),
            "docker_trt_image_family": DOCKER_TRT_IMAGE_FAMILY,
        }
        native_binary = None
    else:
        runtime_info = {
            "mode": "native",
            "tensorrt_version": detect_native_trt_version(),
            "docker_trt_image_family": DOCKER_TRT_IMAGE_FAMILY,
        }
        native_binary = build_native_binary()

    log(f"  TensorRT:      {runtime_info.get('tensorrt_version') or 'unknown'}", force=True)
    if args.mode == "docker" and runtime_info.get("docker_image_digest"):
        log(f"  Docker image:  {runtime_info['docker_image_digest']}", force=True)
    if args.mode == "native" and not runtime_info.get("tensorrt_version"):
        log("  WARNING: could not detect native TensorRT version.", force=True)
    if args.mode == "native":
        log(f"  Native TRT may differ from Docker benchmark image ({DOCKER_TRT_IMAGE_FAMILY}); native and Docker engine caches are separate.", force=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.input:
        input_path, n_texts = limit_input_file(args.input, args.texts)
    else:
        input_path = default_input_path(args.texts)
        n_texts = prepare_default_input(input_path, args.texts)
    log(f"  Input:         {input_path} ({n_texts:,} texts)", force=True)

    bench_spec = resolve_model_spec(args.model)
    native_spec = None
    prepared_model_dir = None
    tei_model = bench_spec["model_id"]
    if args.mode == "native":
        native_spec, prepared_model_dir = prepare_native_model(
            native_binary,
            args.model,
            args.model_dir,
            cache_root,
            input_path,
            available_gpus[0],
            runtime_info,
        )
        tei_model = native_spec["model_id"]
    elif args.mode == "docker":
        host_bench_binary = Path(__file__).resolve().parent / "target" / "release" / "ignite-ms-bench"
        if not host_bench_binary.exists():
            log("  Building host benchmark binary for Docker model preparation...")
            host_bench_binary = build_native_binary()
        _, host_model_dir = prepare_native_model(
            host_bench_binary,
            args.model,
            args.model_dir,
            cache_root,
            input_path,
            available_gpus[0],
            {"mode": "native", "tensorrt_version": detect_native_trt_version()},
        )
        prepared_model_dir = prepare_docker_model_dir(host_model_dir, bench_spec, cache_root)
        prepare_docker_engines(
            input_path,
            prepared_model_dir,
            bench_spec,
            available_gpus[0],
            args.ignite_image,
            runtime_info,
        )
    log("", force=True)

    if not args.skip_tei:
        # Clean up stale TEI containers from interrupted previous runs before
        # measuring ignite-ms. Do not run Docker GPU checks or image pulls here.
        stop_tei_workers()
        time.sleep(5)

    results = []
    log("  === ignite-ms phase ===", force=True)
    for count in gpu_counts:
        gpu_ids = gpu_ids_for_count(count, available_gpus)
        log(f"  --- {count} GPU{'s' if count != 1 else ''} ({','.join(map(str, gpu_ids))}) ---", force=True)
        output_path = DATA_DIR / f"ignite_bench_{count}gpu.npy"
        output_path.unlink(missing_ok=True)

        log("    ignite-ms: running...", force=True)
        if args.mode == "docker":
            ignite = run_ignite_docker(
                input_path,
                prepared_model_dir,
                bench_spec,
                gpu_ids,
                args.ignite_image,
                n_texts,
            )
        else:
            ignite = run_ignite_native(
                native_binary,
                input_path,
                prepared_model_dir,
                native_spec,
                gpu_ids,
                n_texts,
            )
        row = {
            "gpus": count,
            "devices": gpu_ids,
            "ignite_ms": ignite.get("throughput") if ignite.get("ok") else None,
            "ignite_elapsed": ignite.get("elapsed"),
            "ignite_errors": ignite.get("errors", 0),
            "ignite_note": "" if ignite.get("ok") else ignite.get("error", "failed")[:500],
        }
        if row["ignite_ms"]:
            log(f"    ignite-ms: {row['ignite_ms']:,.0f} msg/s", force=True)
        else:
            log("    ignite-ms: FAILED", force=True)
            if row["ignite_note"]:
                log("      " + row["ignite_note"], force=True)
        results.append(row)
        log("", force=True)

    if not args.skip_tei:
        log("  === TEI phase ===", force=True)
        ensure_docker_ready()
        pull_image(args.tei_image)
        stop_tei_workers()
        for row in results:
            count = row["gpus"]
            gpu_ids = row["devices"]
            log(f"  --- {count} GPU{'s' if count != 1 else ''} ({','.join(map(str, gpu_ids))}) ---", force=True)
            stop_tei_workers()
            time.sleep(3)
            log("    TEI: running...", force=True)
            tei = run_tei(
                input_path,
                tei_model,
                gpu_ids,
                args.tei_image,
                n_texts,
                args.tei_batch,
                args.tei_concurrency_per_worker,
            )
            row.update({
                "tei": tei.get("throughput") if tei.get("ok") else None,
                "tei_elapsed": tei.get("elapsed"),
                "tei_errors": tei.get("errors", 0),
                "tei_note": "" if tei.get("ok") else (tei.get("error") or "failed")[:500],
            })
            if row["tei"]:
                log(f"    TEI:       {row['tei']:,.0f} msg/s", force=True)
            else:
                log("    TEI:       FAILED", force=True)
                if row["tei_note"]:
                    log("      " + row["tei_note"], force=True)
            log("", force=True)

        stop_tei_workers()

    print_results(results, tei_model, input_path, args.tei_batch, args.tei_concurrency_per_worker)
    save_results(results, args, input_path, n_texts, prepared_model_dir, tei_model, runtime_info, bench_spec)

    if any(r.get("ignite_ms") is None for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
