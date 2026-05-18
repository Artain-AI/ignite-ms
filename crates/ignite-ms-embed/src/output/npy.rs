use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

use super::OutputWriter;

const HEADER_LEN: usize = 128;

/// Streaming NPY writer. Writes embeddings as they arrive, patches the shape header at the end.
pub struct NpyWriter {
    file: File,
    hidden_dim: usize,
    rows_written: u64,
}

impl NpyWriter {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            file,
            hidden_dim: 0,
            rows_written: 0,
        })
    }

    fn write_header(&mut self, n_rows: u64) -> std::io::Result<()> {
        self.file.seek(SeekFrom::Start(0))?;

        // NPY v1.0 magic
        let magic: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];
        self.file.write_all(&magic)?;
        // Version 1.0
        self.file.write_all(&[1u8, 0u8])?;

        // Header dict
        let dict = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
            n_rows, self.hidden_dim
        );
        // Pad to HEADER_LEN (including the 2-byte length field already written)
        let pad_len = HEADER_LEN - 10 - dict.len() - 1; // 10 = magic(6) + version(2) + header_len(2)
        let header_data_len = (dict.len() + pad_len + 1) as u16; // +1 for newline

        self.file.write_all(&header_data_len.to_le_bytes())?;
        self.file.write_all(dict.as_bytes())?;
        for _ in 0..pad_len {
            self.file.write_all(b" ")?;
        }
        self.file.write_all(b"\n")?;

        Ok(())
    }
}

impl OutputWriter for NpyWriter {
    fn begin(&mut self, hidden_dim: usize) -> std::io::Result<()> {
        self.hidden_dim = hidden_dim;
        // Write placeholder header (will be patched in finish)
        self.write_header(0)?;
        Ok(())
    }

    fn write_row(
        &mut self,
        _seq: u64,
        _user_id: Option<&str>,
        embedding: &[f32],
    ) -> std::io::Result<()> {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                embedding.as_ptr() as *const u8,
                embedding.len() * std::mem::size_of::<f32>(),
            )
        };
        self.file.write_all(bytes)?;
        self.rows_written += 1;
        Ok(())
    }

    fn finish(&mut self, n_rows: u64) -> std::io::Result<()> {
        self.file.flush()?;
        self.write_header(n_rows)?;
        self.file.flush()?;
        Ok(())
    }
}
