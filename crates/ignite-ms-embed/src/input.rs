use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use ignite_ms::Message;

#[derive(Clone, Copy)]
pub enum InputFormat {
    Jsonl,
    Plain,
}

pub fn resolve_inputs(inputs: &[String]) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files: Vec<PathBuf> = Vec::new();
    for input in inputs {
        let path = PathBuf::from(input);
        if path.is_file() {
            files.push(path.canonicalize().unwrap_or(path));
        } else if path.is_dir() {
            collect_recursive(&path, &mut files);
        } else {
            return Err(format!("not found: {}", input).into());
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn collect_recursive(dir: &Path, files: &mut Vec<PathBuf>) {
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

/// Reads all input files sequentially, assigning monotonic sequence IDs.
/// Returns the message iterator and a Vec of user-provided IDs (for Parquet output).
/// Pre-filters texts shorter than `min_chars` so the engine never skips messages
/// (which would stall the reorder buffer).
pub fn read_sequential(
    files: &[PathBuf],
    format: InputFormat,
) -> Result<(Vec<Message>, Vec<String>), Box<dyn std::error::Error>> {
    let mut messages = Vec::new();
    let mut user_ids = Vec::new();
    let mut seq: u64 = 0;
    let mut skipped: u64 = 0;
    let mut malformed: u64 = 0;

    for path in files {
        let reader = open_reader(path)?;
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => {
                    malformed += 1;
                    continue;
                }
            };

            let (text, user_id) = match format {
                InputFormat::Plain => {
                    let trimmed = line.trim().to_string();
                    if trimmed.is_empty() {
                        continue;
                    }
                    (trimmed, seq.to_string())
                }
                InputFormat::Jsonl => match parse_jsonl_line(&line) {
                    Some((text, id)) => (text, id),
                    None => {
                        if !line.trim().is_empty() {
                            malformed += 1;
                        }
                        continue;
                    }
                },
            };

            if text.len() < 10 {
                skipped += 1;
                continue;
            }

            messages.push(Message {
                id: seq.to_string(),
                text,
            });
            user_ids.push(user_id);
            seq += 1;
        }
    }

    if malformed > 0 {
        eprintln!("  WARNING:     {} malformed lines skipped", malformed);
    }
    eprintln!(
        "  messages:    {} read from input ({} filtered short)",
        messages.len(),
        skipped
    );
    Ok((messages, user_ids))
}

fn parse_jsonl_line(line: &str) -> Option<(String, String)> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let text = v.get("text").and_then(|t| t.as_str())?;
    let text = text.trim().to_string();
    if text.is_empty() {
        return None;
    }
    let id = v
        .get("id")
        .and_then(|i| i.as_str())
        .unwrap_or("")
        .to_string();
    Some((text, id))
}

fn open_reader(path: &Path) -> Result<Box<dyn BufRead>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "zst" | "zstd" => {
            let decoder = zstd::stream::Decoder::new(file)?;
            Ok(Box::new(BufReader::with_capacity(4 * 1024 * 1024, decoder)))
        }
        "gz" => {
            let decoder = flate2::read::GzDecoder::new(file);
            Ok(Box::new(BufReader::with_capacity(4 * 1024 * 1024, decoder)))
        }
        _ => Ok(Box::new(BufReader::with_capacity(4 * 1024 * 1024, file))),
    }
}
