//! Model provisioning — fully automated model setup on first use.
//!
//! Given a model name (e.g. "intfloat/multilingual-e5-small"), handles:
//!   1. Install Python dependencies (onnx, numpy, tokenizers)
//!   2. Download tokenizer.json + config.json + raw ONNX in parallel
//!   3. Append mean-pooling + L2-norm to ONNX graph (Python subprocess)
//!   4. Build vocab cache from tokenizer (parallel with ONNX export)
//!   5. Cache everything locally
//!
//! Engine cache (optional, S3):
//!   Downloads/uploads pre-compiled TRT engines keyed by GPU arch + bucket config.
//!   Skips the 3-5 min TRT compilation on subsequent instances.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Instant;

use crate::error::Error;

/// Known model mappings: model name → (HF repo for tokenizer/config, ONNX source repo, hidden_dim)
/// When we move to our own HF repos, we just update the ONNX source.
struct ModelInfo {
    hf_repo: &'static str,
    hf_revision: &'static str,
    onnx_repo: &'static str,
    onnx_revision: &'static str,
    onnx_path: &'static str,
    hidden_dim: usize,
}

fn resolve_model_info(model: &str) -> Option<ModelInfo> {
    match model {
        "intfloat/multilingual-e5-small" => Some(ModelInfo {
            hf_repo: "intfloat/multilingual-e5-small",
            hf_revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3",
            onnx_repo: "Xenova/multilingual-e5-small",
            onnx_revision: "761b726dd34fb83930e26aab4e9ac3899aa1fa78",
            onnx_path: "onnx/model.onnx",
            hidden_dim: 384,
        }),
        "intfloat/multilingual-e5-base" => Some(ModelInfo {
            hf_repo: "intfloat/multilingual-e5-base",
            hf_revision: "d128750597153bb5987e10b1c3493a34e5a4502a",
            onnx_repo: "Xenova/multilingual-e5-base",
            onnx_revision: "1ec9243030a27d1a115d5c340572074c125b58b2",
            onnx_path: "onnx/model.onnx",
            hidden_dim: 768,
        }),
        "intfloat/multilingual-e5-large" => Some(ModelInfo {
            hf_repo: "intfloat/multilingual-e5-large",
            hf_revision: "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
            onnx_repo: "Xenova/multilingual-e5-large",
            onnx_revision: "00fc3aeb3dbb95842de2ac1961d33c6319acf57b",
            onnx_path: "onnx/model.onnx",
            hidden_dim: 1024,
        }),
        "intfloat/e5-small-v2" => Some(ModelInfo {
            hf_repo: "intfloat/e5-small-v2",
            hf_revision: "ffb93f3bd4047442299a41ebb6fa998a38507c52",
            onnx_repo: "Xenova/e5-small-v2",
            onnx_revision: "02af79985278377e65c724a76275707cb0333c70",
            onnx_path: "onnx/model.onnx",
            hidden_dim: 384,
        }),
        "intfloat/e5-base-v2" => Some(ModelInfo {
            hf_repo: "intfloat/e5-base-v2",
            hf_revision: "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
            onnx_repo: "Xenova/e5-base-v2",
            onnx_revision: "21f8d0e36fdfe76e6a023802dfb293fc6d750ad1",
            onnx_path: "onnx/model.onnx",
            hidden_dim: 768,
        }),
        "intfloat/e5-large-v2" => Some(ModelInfo {
            hf_repo: "intfloat/e5-large-v2",
            hf_revision: "f169b11e22de13617baa190a028a32f3493550b6",
            onnx_repo: "Xenova/e5-large-v2",
            onnx_revision: "840fd2207f68e253697ed85392a482ff7657ad11",
            onnx_path: "onnx/model.onnx",
            hidden_dim: 1024,
        }),
        _ => None,
    }
}

fn is_nonempty_file(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false)
}

fn find_python() -> Result<String, Error> {
    // Respect explicit override
    if let Ok(py) = std::env::var("PYTHON") {
        return Ok(py);
    }
    for candidate in [
        "python3",
        "python3.12",
        "python3.11",
        "python3.10",
        "python",
    ] {
        let ok = Command::new(candidate)
            .args(["--version"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            // Verify pip is available
            let has_pip = Command::new(candidate)
                .args(["-m", "pip", "--version"])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            if has_pip {
                return Ok(candidate.to_string());
            }
        }
    }
    Err(Error::Model(
        "python3 with pip not found. Install Python 3.10+ with pip, or set PYTHON env var.".into(),
    ))
}

/// Ensure model files exist locally. Downloads and exports if missing.
pub fn ensure_model(model: &str, cache_dir: &Path) -> Result<PathBuf, Error> {
    let model_dir = cache_dir.join(model.replace('/', "--"));
    std::fs::create_dir_all(&model_dir)?;

    // Already provisioned?
    if is_nonempty_file(&model_dir.join("model.onnx"))
        && is_nonempty_file(&model_dir.join("tokenizer.json"))
        && is_nonempty_file(&model_dir.join("config.json"))
        && is_nonempty_file(&model_dir.join("vocab_cache.bin"))
    {
        eprintln!(
            "[provision] model '{}' cached at {}",
            model,
            model_dir.display()
        );
        return Ok(model_dir);
    }

    let info = resolve_model_info(model).ok_or_else(|| {
        Error::Model(format!(
            "unknown model '{}'. Supported: intfloat/multilingual-e5-small, intfloat/multilingual-e5-base, \
             intfloat/multilingual-e5-large, intfloat/e5-small-v2, intfloat/e5-base-v2, intfloat/e5-large-v2",
            model
        ))
    })?;

    let t_start = Instant::now();
    eprintln!("[provision] setting up model '{}'...", model);

    let python = find_python()?;
    eprintln!("[provision]   using python: {}", python);

    // Install Python dependencies (idempotent, ~2s if already present)
    ensure_python_deps(&python)?;
    eprintln!(
        "[provision]   python deps ready ({:.1}s)",
        t_start.elapsed().as_secs_f64()
    );

    // Download all files in parallel
    let t_dl = Instant::now();
    let tok_dest = model_dir.join("tokenizer.json");
    let cfg_dest = model_dir.join("config.json");
    let raw_onnx = model_dir.join("model_raw.onnx");

    let hf_repo = info.hf_repo.to_string();
    let hf_revision = info.hf_revision.to_string();
    let onnx_repo = info.onnx_repo.to_string();
    let onnx_revision = info.onnx_revision.to_string();
    let onnx_path = info.onnx_path.to_string();

    let tok_dest_c = tok_dest.clone();
    let hf_repo_c = hf_repo.clone();
    let hf_revision_c = hf_revision.clone();
    let h_tok = thread::spawn(move || {
        download_hf_file(&hf_repo_c, &hf_revision_c, "tokenizer.json", &tok_dest_c)
    });

    let cfg_dest_c = cfg_dest.clone();
    let hf_repo_c2 = hf_repo.clone();
    let hf_revision_c2 = hf_revision.clone();
    let h_cfg = thread::spawn(move || {
        download_hf_file(&hf_repo_c2, &hf_revision_c2, "config.json", &cfg_dest_c)
    });

    let raw_onnx_c = raw_onnx.clone();
    let h_onnx = thread::spawn(move || {
        download_hf_file(&onnx_repo, &onnx_revision, &onnx_path, &raw_onnx_c)
    });

    h_tok
        .join()
        .map_err(|_| Error::Model("tokenizer download thread panicked".into()))??;
    h_cfg
        .join()
        .map_err(|_| Error::Model("config download thread panicked".into()))??;
    h_onnx
        .join()
        .map_err(|_| Error::Model("ONNX download thread panicked".into()))??;
    eprintln!(
        "[provision]   downloads complete ({:.1}s)",
        t_dl.elapsed().as_secs_f64()
    );

    // Run ONNX export and vocab cache build in parallel
    let t_proc = Instant::now();
    let final_onnx = model_dir.join("model.onnx");
    let vocab_out = model_dir.join("vocab_cache.bin");

    let raw_onnx_c2 = raw_onnx.clone();
    let final_onnx_c = final_onnx.clone();
    let hidden_dim = info.hidden_dim;
    let python_c = python.clone();
    let h_export = thread::spawn(move || {
        export_with_pooling(&python_c, &raw_onnx_c2, &final_onnx_c, hidden_dim)
    });

    let tok_for_cache = tok_dest.clone();
    let python_c2 = python.clone();
    let h_vocab = thread::spawn(move || build_vocab_cache(&python_c2, &tok_for_cache, &vocab_out));

    h_export
        .join()
        .map_err(|_| Error::Model("ONNX export thread panicked".into()))??;
    h_vocab
        .join()
        .map_err(|_| Error::Model("vocab cache thread panicked".into()))??;
    eprintln!(
        "[provision]   export + vocab cache ({:.1}s)",
        t_proc.elapsed().as_secs_f64()
    );

    // Clean up raw ONNX
    let _ = std::fs::remove_file(&raw_onnx);

    eprintln!(
        "[provision] model '{}' ready ({:.1}s total)",
        model,
        t_start.elapsed().as_secs_f64()
    );
    Ok(model_dir)
}

/// Install Python packages needed for provisioning.
fn ensure_python_deps(python: &str) -> Result<(), Error> {
    eprintln!("[provision]   ensuring python deps (onnx, numpy, tokenizers)...");

    let output = Command::new(python)
        .args([
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "onnx==1.16.1",
            "numpy==1.26.4",
            "tokenizers==0.20.3",
        ])
        .output()
        .map_err(|e| Error::Model(format!("failed to run pip ({}): {}", python, e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Model(format!(
            "pip install failed: {}",
            stderr.trim()
        )));
    }

    Ok(())
}

/// Download a single file from HuggingFace Hub.
/// Uses temp file + atomic rename to prevent partial/corrupt cache entries.
fn download_hf_file(repo: &str, revision: &str, file_path: &str, dest: &Path) -> Result<(), Error> {
    if dest.exists() {
        let size = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
        if size > 0 {
            return Ok(());
        }
        let _ = std::fs::remove_file(dest);
    }
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        repo, revision, file_path
    );
    eprintln!("[provision]   downloading {}...", url);

    let tmp_dest = dest.with_extension("tmp");
    let output = Command::new("curl")
        .args([
            "-fSL",
            "--retry",
            "3",
            "-o",
            &tmp_dest.to_string_lossy(),
            &url,
        ])
        .output()
        .map_err(|e| Error::Model(format!("failed to run curl: {}", e)))?;

    if !output.status.success() {
        let _ = std::fs::remove_file(&tmp_dest);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Model(format!(
            "download failed: {} — {}",
            url,
            stderr.trim()
        )));
    }

    let size = std::fs::metadata(&tmp_dest).map(|m| m.len()).unwrap_or(0);
    if size == 0 {
        let _ = std::fs::remove_file(&tmp_dest);
        return Err(Error::Model(format!(
            "download produced empty file: {}",
            url
        )));
    }

    std::fs::rename(&tmp_dest, dest)?;
    eprintln!(
        "[provision]   saved {} ({:.1} MB)",
        dest.display(),
        size as f64 / 1e6
    );
    Ok(())
}

/// Run the Python export script to append mean-pooling + L2-norm to the ONNX model.
fn export_with_pooling(
    python: &str,
    raw_onnx: &Path,
    output: &Path,
    hidden_dim: usize,
) -> Result<(), Error> {
    eprintln!("[provision]   appending mean-pool + L2-norm to ONNX...");

    let script = include_str!("../scripts/export_pooling.py");

    let script_path = output.with_file_name(".export_pooling.py");
    std::fs::write(&script_path, script)?;

    let result = Command::new(python)
        .arg(&script_path)
        .arg("--input")
        .arg(raw_onnx)
        .arg("--output")
        .arg(output)
        .arg("--hidden-dim")
        .arg(hidden_dim.to_string())
        .output()
        .map_err(|e| Error::Model(format!("failed to run {} for ONNX export: {}", python, e)))?;

    let _ = std::fs::remove_file(&script_path);

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(Error::Model(format!(
            "ONNX export failed: {}",
            stderr.trim()
        )));
    }

    if !is_nonempty_file(output) {
        return Err(Error::Model(
            "ONNX export produced no non-empty output file".to_string(),
        ));
    }

    let size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "[provision]   model.onnx ready ({:.1} MB)",
        size as f64 / 1e6
    );
    Ok(())
}

/// Run the Python script to build vocab_cache.bin from tokenizer.json.
fn build_vocab_cache(python: &str, tokenizer_path: &Path, output: &Path) -> Result<(), Error> {
    if output.exists() {
        if is_nonempty_file(output) {
            return Ok(());
        }
        let _ = std::fs::remove_file(output);
    }
    eprintln!("[provision]   building vocab cache...");

    let script = include_str!("../scripts/build_vocab_cache.py");
    let script_path = output.with_file_name(".build_vocab_cache.py");
    std::fs::write(&script_path, script)?;

    let result = Command::new(python)
        .arg(&script_path)
        .arg("--tokenizer")
        .arg(tokenizer_path)
        .arg("--output")
        .arg(output)
        .output()
        .map_err(|e| Error::Model(format!("failed to run {} for vocab cache: {}", python, e)))?;

    let _ = std::fs::remove_file(&script_path);

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(Error::Model(format!(
            "vocab cache build failed: {}",
            stderr.trim()
        )));
    }

    if !is_nonempty_file(output) {
        return Err(Error::Model(
            "vocab cache build produced no non-empty output file".to_string(),
        ));
    }

    let size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "[provision]   vocab_cache.bin ready ({:.1} MB)",
        size as f64 / 1e6
    );
    Ok(())
}

// ─── Engine Cache (S3, optional) ────────────────────────────────────────────

/// Build the engine cache key for a specific configuration.
fn engine_cache_key(gpu_arch: &str, batch_size: usize, seq_len: usize, int8: bool) -> String {
    let suffix = if int8 { "_int8" } else { "" };
    format!(
        "{}/model_b{}_s{}{}.engine",
        gpu_arch, batch_size, seq_len, suffix
    )
}

/// Detect GPU architecture string (e.g. "sm_80" for A100).
pub fn detect_gpu_arch() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let cap = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if cap.is_empty() {
        return None;
    }
    Some(format!("sm_{}", cap.replace('.', "")))
}

/// Try to download a pre-compiled engine from S3.
pub fn fetch_cached_engine(
    engine_cache: &str,
    model: &str,
    gpu_arch: &str,
    batch_size: usize,
    seq_len: usize,
    int8: bool,
    local_path: &Path,
) -> Option<PathBuf> {
    if local_path.exists() {
        return Some(local_path.to_path_buf());
    }

    let key = engine_cache_key(gpu_arch, batch_size, seq_len, int8);
    let model_slug = model.replace('/', "--");
    let s3_path = format!(
        "{}/{}/{}",
        engine_cache.trim_end_matches('/'),
        model_slug,
        key
    );

    let output = Command::new("aws")
        .args(["s3", "cp", &s3_path, &local_path.to_string_lossy()])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if output.status.success() && local_path.exists() {
        eprintln!("[provision] engine cache hit: {}", s3_path);
        Some(local_path.to_path_buf())
    } else {
        None
    }
}

/// Upload a compiled engine to S3.
pub fn upload_engine(
    engine_cache: &str,
    model: &str,
    gpu_arch: &str,
    batch_size: usize,
    seq_len: usize,
    int8: bool,
    local_path: &Path,
) {
    let key = engine_cache_key(gpu_arch, batch_size, seq_len, int8);
    let model_slug = model.replace('/', "--");
    let s3_path = format!(
        "{}/{}/{}",
        engine_cache.trim_end_matches('/'),
        model_slug,
        key
    );

    let output = Command::new("aws")
        .args(["s3", "cp", &local_path.to_string_lossy(), &s3_path])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            eprintln!("[provision] engine uploaded: {}", s3_path);
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            eprintln!(
                "[provision] WARNING: engine upload failed: {}",
                stderr.trim()
            );
        }
        Err(e) => {
            eprintln!("[provision] WARNING: engine upload failed: {}", e);
        }
    }
}
