pub mod npy;
pub mod parquet_writer;

use std::path::Path;

pub trait OutputWriter: Send {
    fn begin(&mut self, hidden_dim: usize) -> std::io::Result<()>;
    fn write_row(
        &mut self,
        seq: u64,
        user_id: Option<&str>,
        embedding: &[f32],
    ) -> std::io::Result<()>;
    fn finish(&mut self, n_rows: u64) -> std::io::Result<()>;
}

pub fn create_writer(
    output: &str,
    hidden_dim: usize,
) -> Result<Box<dyn OutputWriter>, Box<dyn std::error::Error>> {
    let ext = Path::new(output)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("npy");

    match ext {
        "parquet" => Ok(Box::new(parquet_writer::ParquetOutputWriter::new(
            output, hidden_dim,
        )?)),
        _ => Ok(Box::new(npy::NpyWriter::new(output)?)),
    }
}
