//! Length-bucketed batcher — accumulates tokenized messages into
//! fixed-shape batches optimized for GPU inference.

use crate::tokenize::TokenizedMessage;

/// A ready-to-execute batch of tokenized sequences.
/// Buffers are flat contiguous [batch_size × bucket_len] in row-major order.
pub struct Batch {
    pub bucket_idx: usize,
    pub bucket_len: usize,
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub n_real: usize,
}

/// Specification for a single bucket.
pub struct BucketSpec {
    pub len: usize,
    pub batch_size: usize,
}

/// Accumulates tokenized messages and emits fixed-shape batches.
pub struct Bucketizer {
    buckets: Vec<BucketState>,
}

struct BucketState {
    spec: BucketSpec,
    pending: Vec<TokenizedMessage>,
}

impl Bucketizer {
    pub fn new(specs: Vec<BucketSpec>) -> Self {
        let mut sorted = specs;
        sorted.sort_by_key(|b| b.len);
        let buckets = sorted
            .into_iter()
            .map(|spec| BucketState {
                pending: Vec::with_capacity(spec.batch_size),
                spec,
            })
            .collect();
        Self { buckets }
    }

    /// Push one message. Returns a Batch if the assigned bucket filled up.
    pub fn push(&mut self, msg: TokenizedMessage) -> Option<Batch> {
        let bucket_idx = self.assign_bucket(msg.len);
        let bucket = &mut self.buckets[bucket_idx];
        let max = bucket.spec.len;

        let truncated = if msg.len > max {
            TokenizedMessage {
                ids: msg.ids[..max].to_vec(),
                len: max,
            }
        } else {
            msg
        };

        bucket.pending.push(truncated);
        if bucket.pending.len() >= bucket.spec.batch_size {
            Some(Self::build_batch(bucket_idx, bucket))
        } else {
            None
        }
    }

    /// Flush all partial buckets as final batches.
    pub fn flush(&mut self) -> Vec<Batch> {
        let mut out = Vec::new();
        for (i, b) in self.buckets.iter_mut().enumerate() {
            if !b.pending.is_empty() {
                out.push(Self::build_batch(i, b));
            }
        }
        out
    }

    fn assign_bucket(&self, msg_len: usize) -> usize {
        for (i, b) in self.buckets.iter().enumerate() {
            if msg_len <= b.spec.len {
                return i;
            }
        }
        self.buckets.len() - 1
    }

    fn build_batch(bucket_idx: usize, bucket: &mut BucketState) -> Batch {
        let bs = bucket.spec.batch_size;
        let bl = bucket.spec.len;
        let pending = std::mem::replace(&mut bucket.pending, Vec::with_capacity(bs));
        let n_real = pending.len();

        let mut input_ids = vec![0i64; bs * bl];
        let mut attention_mask = vec![0i64; bs * bl];

        for (row, msg) in pending.into_iter().enumerate() {
            let offset = row * bl;
            let n = msg.ids.len().min(bl);
            for j in 0..n {
                input_ids[offset + j] = msg.ids[j] as i64;
                attention_mask[offset + j] = 1;
            }
        }

        Batch {
            bucket_idx,
            bucket_len: bl,
            input_ids,
            attention_mask,
            n_real,
        }
    }
}
