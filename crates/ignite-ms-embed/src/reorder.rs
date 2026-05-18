use std::collections::{BTreeMap, HashSet};

/// Reorder buffer that accumulates out-of-order embeddings and yields them sequentially.
/// Bucketing causes batches to return in arbitrary order; this ensures output row N = input row N.
/// Skipped sequence numbers (short/empty texts the engine drops) must be registered so the
/// buffer can advance past them.
pub struct ReorderBuffer {
    next_seq: u64,
    pending: BTreeMap<u64, Vec<f32>>,
    skipped: HashSet<u64>,
    hidden_dim: usize,
}

impl ReorderBuffer {
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            next_seq: 0,
            pending: BTreeMap::new(),
            skipped: HashSet::new(),
            hidden_dim,
        }
    }

    pub fn insert(&mut self, seq: u64, embedding: &[f32]) {
        debug_assert_eq!(embedding.len(), self.hidden_dim);
        self.pending.insert(seq, embedding.to_vec());
    }

    pub fn mark_skipped(&mut self, seq: u64) {
        self.skipped.insert(seq);
    }

    /// Pop the next in-order embedding if available, skipping over known-skipped entries.
    pub fn pop_next(&mut self) -> Option<(u64, Vec<f32>)> {
        // Advance past any skipped entries
        while self.skipped.contains(&self.next_seq) {
            self.skipped.remove(&self.next_seq);
            self.next_seq += 1;
        }
        if let Some(emb) = self.pending.remove(&self.next_seq) {
            let seq = self.next_seq;
            self.next_seq += 1;
            Some((seq, emb))
        } else {
            None
        }
    }

    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    pub fn has_pending(&self, seq: u64) -> bool {
        self.pending.contains_key(&seq)
    }
}
