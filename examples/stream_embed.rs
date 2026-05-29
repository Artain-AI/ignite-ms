//! Minimal library usage: read JSONL from stdin, emit embeddings to stdout.
//!
//! ```bash
//! cat data.jsonl | cargo run --release --example stream_embed
//! ```

use std::io::{self, BufRead};

use ignite_ms::{Config, Engine, Message};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        model: Some("intfloat/e5-small-v2".into()),
        ..Default::default()
    };

    let mut engine = Engine::new(config)?;
    let hidden_dim = engine.hidden_dim();

    let messages: Vec<Message> = io::stdin()
        .lock()
        .lines()
        .enumerate()
        .filter_map(|(i, line)| {
            let line = line.ok()?;
            let v: serde_json::Value = serde_json::from_str(&line).ok()?;
            let text = v["text"].as_str()?.to_string();
            Some(Message {
                id: i.to_string(),
                text,
            })
        })
        .collect();

    eprintln!(
        "Loaded {} messages, embedding ({}D)...",
        messages.len(),
        hidden_dim
    );

    let (rx, handle) = engine.embed(messages)?;

    for batch in rx {
        for i in 0..batch.n_rows {
            let emb = batch.embedding(i);
            println!(
                "{}\t{:.6},{:.6},{:.6},...",
                batch.ids[i], emb[0], emb[1], emb[2]
            );
        }
    }

    let stats = handle.finish();
    eprintln!("Done: {} msg/s", stats.throughput() as u64);
    Ok(())
}
