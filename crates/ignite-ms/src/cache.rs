//! Vocabulary cache — pre-tokenized word→token_ids HashMap.
//!
//! Loaded once at startup, shared across all tokenizer workers via Arc.
//! Provides nanosecond lookups for 98.4% of words (vs 2-5ms per encode call).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

const CACHE_MAGIC: &[u8; 8] = b"IMSVCACH";
const CACHE_VERSION: u32 = 1;

/// A shared, read-only vocabulary cache.
pub type SharedCache = Arc<HashMap<String, Vec<u32>>>;

/// Load a vocabulary cache from the binary format.
/// Returns an Arc-wrapped HashMap for zero-cost sharing across threads.
pub fn load(path: &Path) -> Result<SharedCache, String> {
    let raw = load_raw(path)?;
    Ok(Arc::new(raw))
}

/// Load the raw HashMap (not wrapped in Arc).
pub fn load_raw(path: &Path) -> Result<HashMap<String, Vec<u32>>, String> {
    let f = File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let mut r = BufReader::with_capacity(1024 * 1024, f);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)
        .map_err(|e| format!("read magic: {}", e))?;
    if &magic != CACHE_MAGIC {
        return Err("invalid cache file magic bytes".to_string());
    }

    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)
        .map_err(|e| format!("read version: {}", e))?;
    let version = u32::from_le_bytes(buf4);
    if version != CACHE_VERSION {
        return Err(format!(
            "unsupported cache version: {} (expected {})",
            version, CACHE_VERSION
        ));
    }

    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8)
        .map_err(|e| format!("read count: {}", e))?;
    let num_entries = u64::from_le_bytes(buf8) as usize;

    let mut cache = HashMap::with_capacity(num_entries);
    for _ in 0..num_entries {
        r.read_exact(&mut buf4)
            .map_err(|e| format!("read key_len: {}", e))?;
        let key_len = u32::from_le_bytes(buf4) as usize;

        let mut key_bytes = vec![0u8; key_len];
        r.read_exact(&mut key_bytes)
            .map_err(|e| format!("read key: {}", e))?;
        let key = String::from_utf8(key_bytes).map_err(|e| format!("utf8 key: {}", e))?;

        r.read_exact(&mut buf4)
            .map_err(|e| format!("read num_ids: {}", e))?;
        let num_ids = u32::from_le_bytes(buf4) as usize;

        let mut ids = Vec::with_capacity(num_ids);
        for _ in 0..num_ids {
            r.read_exact(&mut buf4)
                .map_err(|e| format!("read id: {}", e))?;
            ids.push(u32::from_le_bytes(buf4));
        }

        cache.insert(key, ids);
    }

    Ok(cache)
}

/// Save a vocabulary cache to the binary format.
pub fn save(cache: &HashMap<String, Vec<u32>>, path: &Path) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::with_capacity(1024 * 1024, f);

    w.write_all(CACHE_MAGIC)?;
    w.write_all(&CACHE_VERSION.to_le_bytes())?;
    w.write_all(&(cache.len() as u64).to_le_bytes())?;

    for (key, ids) in cache {
        let key_bytes = key.as_bytes();
        w.write_all(&(key_bytes.len() as u32).to_le_bytes())?;
        w.write_all(key_bytes)?;
        w.write_all(&(ids.len() as u32).to_le_bytes())?;
        for &id in ids {
            w.write_all(&id.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}
