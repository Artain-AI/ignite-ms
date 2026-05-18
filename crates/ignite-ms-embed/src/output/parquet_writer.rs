use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use super::OutputWriter;

const ROW_GROUP_SIZE: usize = 50_000;

pub struct ParquetOutputWriter {
    writer: Option<ArrowWriter<File>>,
    schema: Arc<Schema>,
    hidden_dim: usize,
    ids: Vec<String>,
    embeddings: Vec<f32>,
}

impl ParquetOutputWriter {
    pub fn new(path: &str, hidden_dim: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let list_field = Arc::new(Field::new("item", DataType::Float32, true));
        let fsl_type = DataType::FixedSizeList(list_field, hidden_dim as i32);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("embedding", fsl_type, false),
        ]));

        let file = File::create(Path::new(path))?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build();

        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        Ok(Self {
            writer: Some(writer),
            schema,
            hidden_dim,
            ids: Vec::with_capacity(ROW_GROUP_SIZE),
            embeddings: Vec::with_capacity(ROW_GROUP_SIZE * hidden_dim),
        })
    }

    fn flush_batch(&mut self) -> std::io::Result<()> {
        if self.ids.is_empty() {
            return Ok(());
        }

        let n = self.ids.len();

        let mut id_builder = StringBuilder::with_capacity(n, n * 20);
        for id in &self.ids {
            id_builder.append_value(id);
        }
        let id_array: ArrayRef = Arc::new(id_builder.finish());

        let values = Float32Array::from(self.embeddings.clone());
        let list_field = Arc::new(Field::new("item", DataType::Float32, true));
        let emb_array: ArrayRef = Arc::new(FixedSizeListArray::new(
            list_field,
            self.hidden_dim as i32,
            Arc::new(values),
            None,
        ));

        let batch = RecordBatch::try_new(self.schema.clone(), vec![id_array, emb_array])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        self.writer
            .as_mut()
            .unwrap()
            .write(&batch)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        self.ids.clear();
        self.embeddings.clear();

        Ok(())
    }
}

impl OutputWriter for ParquetOutputWriter {
    fn begin(&mut self, _hidden_dim: usize) -> std::io::Result<()> {
        Ok(())
    }

    fn write_row(
        &mut self,
        seq: u64,
        user_id: Option<&str>,
        embedding: &[f32],
    ) -> std::io::Result<()> {
        let id = match user_id {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => seq.to_string(),
        };
        self.ids.push(id);
        self.embeddings.extend_from_slice(embedding);

        if self.ids.len() >= ROW_GROUP_SIZE {
            self.flush_batch()?;
        }
        Ok(())
    }

    fn finish(&mut self, _n_rows: u64) -> std::io::Result<()> {
        self.flush_batch()?;
        self.writer
            .take()
            .unwrap()
            .close()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(())
    }
}
