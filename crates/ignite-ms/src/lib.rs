//! # IgniteMS
//!
//! High-throughput text embedding engine. 400K+ msg/s on 8× A100 GPUs.
//!
//! ## Architecture
//! - Client feeds `(id, text)` messages
//! - Engine: normalize → tokenize → batch → GPU inference → D2H
//! - Client receives `EmbeddingBatch` via channel
//!
//! The library knows nothing about data sources, file formats, or what
//! the client does with embeddings. It just turns text into vectors, fast.
//!
//! ## Usage
//! ```ignore
//! use ignite_ms::{Engine, Config, Message};
//!
//! let engine = Engine::new(Config::default())?;
//! let messages = vec![
//!     Message { id: "user_1".into(), text: "hello world".into() },
//!     Message { id: "user_2".into(), text: "rust is fast".into() },
//! ];
//!
//! let (rx, handle) = engine.embed(messages)?;
//! for batch in rx {
//!     for i in 0..batch.n_rows {
//!         println!("{}: dim0={:.4}", batch.ids[i], batch.embedding(i)[0]);
//!     }
//! }
//! let stats = handle.finish();
//! println!("{:.0} msg/s", stats.throughput());
//! ```

pub mod bucket;
pub mod cache;
pub mod normalize;
pub mod provision;
pub mod tokenize;

#[cfg(feature = "native-trt")]
pub mod inference;

mod engine;
mod error;

pub use engine::{Config, EmbedHandle, EmbedStats, EmbeddingBatch, Engine, LatencyStats, Message};
pub use error::Error;
