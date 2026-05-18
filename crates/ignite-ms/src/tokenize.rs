//! Vocabulary-cached tokenizer — per-worker instances with shared static cache
//! and per-worker runtime cache for dynamic word learning.
//!
//! Each worker splits text on whitespace, looks up each word in:
//!   1. Shared static cache (pre-built from tokenizer vocab or corpus)
//!   2. Per-worker runtime cache (populated on first miss, stays for the run)
//! Falls through to the real tokenizer only on double-miss.

use std::collections::HashMap;
use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::cache;

const RUNTIME_CACHE_MAX: usize = 500_000;

/// A tokenized message: token IDs + real length.
#[derive(Clone)]
pub struct TokenizedMessage {
    pub ids: Vec<u32>,
    pub len: usize,
}

/// Per-worker cached tokenizer. Owns a reference to the shared static cache
/// and maintains a local runtime cache for words seen during this run.
pub struct CachedTokenizer {
    tokenizer: Arc<Tokenizer>,
    static_cache: cache::SharedCache,
    runtime_cache: HashMap<String, Vec<u32>>,
    cls_id: u32,
    sep_id: u32,
    max_len: usize,
    key_buf: String,
    pub hits: u64,
    pub misses: u64,
    pub runtime_hits: u64,
}

impl CachedTokenizer {
    pub fn new(tokenizer: Arc<Tokenizer>, max_len: usize, cache: cache::SharedCache) -> Self {
        let cls_id = tokenizer.token_to_id("<s>").unwrap_or(0);
        let sep_id = tokenizer.token_to_id("</s>").unwrap_or(2);
        Self {
            tokenizer,
            static_cache: cache,
            runtime_cache: HashMap::with_capacity(32_768),
            cls_id,
            sep_id,
            max_len,
            key_buf: String::with_capacity(64),
            hits: 0,
            misses: 0,
            runtime_hits: 0,
        }
    }

    pub fn tokenize_one(&mut self, text: &str) -> TokenizedMessage {
        let mut ids: Vec<u32> = Vec::with_capacity(self.max_len);
        ids.push(self.cls_id);

        let content_limit = self.max_len - 1;

        for word in text.split_whitespace() {
            if ids.len() >= content_limit {
                break;
            }

            let word_ids = self.lookup_word(word);
            let remaining = content_limit - ids.len();
            let take = word_ids.len().min(remaining);
            ids.extend_from_slice(&word_ids[..take]);
        }

        ids.push(self.sep_id);
        let len = ids.len();
        TokenizedMessage { ids, len }
    }

    fn lookup_word(&mut self, word: &str) -> Vec<u32> {
        self.key_buf.clear();
        self.key_buf.push('\u{2581}');
        self.key_buf.push_str(word);

        // 1. Static cache (pre-built, shared across workers)
        if let Some(ids) = self.static_cache.get(&self.key_buf) {
            self.hits += 1;
            return ids.clone();
        }

        // 2. Runtime cache (per-worker, populated on first miss)
        if let Some(ids) = self.runtime_cache.get(&self.key_buf) {
            self.runtime_hits += 1;
            self.hits += 1;
            return ids.clone();
        }

        // 3. Full BPE encode
        self.misses += 1;
        let result = match self.tokenizer.encode(self.key_buf.as_str(), false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(_) => vec![3],
        };

        // Store in runtime cache (skip very long words — unlikely to repeat)
        if self.runtime_cache.len() < RUNTIME_CACHE_MAX && word.len() <= 45 {
            self.runtime_cache
                .insert(self.key_buf.clone(), result.clone());
        }

        result
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn runtime_cache_size(&self) -> usize {
        self.runtime_cache.len()
    }
}
