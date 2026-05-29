//! Error types for ignite-ms.

use std::fmt;

#[derive(Debug)]
pub enum Error {
    Config(String),
    Model(String),
    Tokenizer(String),
    Inference(String),
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(msg) => write!(f, "config: {}", msg),
            Error::Model(msg) => write!(f, "model: {}", msg),
            Error::Tokenizer(msg) => write!(f, "tokenizer: {}", msg),
            Error::Inference(msg) => write!(f, "inference: {}", msg),
            Error::Io(e) => write!(f, "io: {}", e),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
