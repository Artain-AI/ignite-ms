//! Native TensorRT inference + compilation via C FFI.
//!
//! Bypasses ORT: compiles ONNX→engine, loads serialized TRT engine,
//! pre-allocates GPU buffers, runs inference with cudaMemcpy + enqueueV3.

use std::ffi::CString;
use std::path::Path;

/// Opaque C handle.
#[repr(C)]
pub struct TrtRunnerHandle {
    _private: [u8; 0],
}

extern "C" {
    fn trt_compile(
        onnx_path: *const std::ffi::c_char,
        engine_path: *const std::ffi::c_char,
        min_batch: i32,
        opt_batch: i32,
        max_batch: i32,
        min_seq: i32,
        opt_seq: i32,
        max_seq: i32,
        fp16: i32,
    ) -> i32;

    fn trt_create(
        engine_path: *const std::ffi::c_char,
        gpu_id: i32,
        batch_size: i32,
        seq_len: i32,
        hidden_dim: i32,
    ) -> *mut TrtRunnerHandle;

    fn trt_infer(
        runner: *mut TrtRunnerHandle,
        input_ids: *const i64,
        attention_mask: *const i64,
        token_type_ids: *const i64,
    ) -> i32;

    fn trt_get_output(runner: *mut TrtRunnerHandle, h_output: *mut f32) -> i32;

    fn trt_destroy(runner: *mut TrtRunnerHandle);
}

/// Compile an ONNX model to a TRT engine file.
/// This is a one-time operation (~2-5 min) cached to disk.
pub fn compile_engine(
    onnx_path: &Path,
    engine_path: &Path,
    max_batch: usize,
    max_seq: usize,
) -> Result<(), String> {
    let onnx_str = onnx_path
        .to_str()
        .ok_or_else(|| "ONNX path not valid UTF-8".to_string())?;
    let engine_str = engine_path
        .to_str()
        .ok_or_else(|| "engine path not valid UTF-8".to_string())?;

    let c_onnx = CString::new(onnx_str).map_err(|e| format!("CString: {}", e))?;
    let c_engine = CString::new(engine_str).map_err(|e| format!("CString: {}", e))?;

    eprintln!(
        "[ignite-ms] compiling TRT engine: {} -> {}",
        onnx_str, engine_str
    );
    eprintln!(
        "[ignite-ms]   max_batch={} max_seq={} fp16=true",
        max_batch, max_seq
    );

    let rc = unsafe {
        trt_compile(
            c_onnx.as_ptr(),
            c_engine.as_ptr(),
            1,                      // min_batch
            (max_batch / 2) as i32, // opt_batch
            max_batch as i32,       // max_batch
            32,                     // min_seq
            128,                    // opt_seq
            max_seq as i32,         // max_seq
            1,                      // fp16
        )
    };

    if rc != 0 {
        Err("trt_compile failed".to_string())
    } else {
        eprintln!("[ignite-ms] engine compilation complete");
        Ok(())
    }
}

/// Compile a per-bucket engine with tight shape bounds for optimal kernel selection.
/// Uses the exact (batch_size, seq_len) as both opt and max shapes.
pub fn compile_engine_for_bucket(
    onnx_path: &Path,
    engine_path: &Path,
    batch_size: usize,
    seq_len: usize,
) -> Result<(), String> {
    let onnx_str = onnx_path
        .to_str()
        .ok_or_else(|| "ONNX path not valid UTF-8".to_string())?;
    let engine_str = engine_path
        .to_str()
        .ok_or_else(|| "engine path not valid UTF-8".to_string())?;

    let c_onnx = CString::new(onnx_str).map_err(|e| format!("CString: {}", e))?;
    let c_engine = CString::new(engine_str).map_err(|e| format!("CString: {}", e))?;

    let rc = unsafe {
        trt_compile(
            c_onnx.as_ptr(),
            c_engine.as_ptr(),
            1,                 // min_batch (allow partial batches on flush)
            batch_size as i32, // opt_batch = exact target
            batch_size as i32, // max_batch = exact target
            seq_len as i32,    // min_seq = exact
            seq_len as i32,    // opt_seq = exact
            seq_len as i32,    // max_seq = exact
            1,                 // fp16
        )
    };

    if rc != 0 {
        Err(format!(
            "trt_compile failed for bs={} seq={}",
            batch_size, seq_len
        ))
    } else {
        Ok(())
    }
}

/// Compile a per-bucket engine with INT8 quantization (FP16 + INT8 mixed precision).
/// TRT selects the best precision per layer for optimal throughput.
/// Calibration is done automatically with synthetic data and cached to disk.
pub fn compile_engine_for_bucket_int8(
    onnx_path: &Path,
    engine_path: &Path,
    batch_size: usize,
    seq_len: usize,
) -> Result<(), String> {
    let onnx_str = onnx_path
        .to_str()
        .ok_or_else(|| "ONNX path not valid UTF-8".to_string())?;
    let engine_str = engine_path
        .to_str()
        .ok_or_else(|| "engine path not valid UTF-8".to_string())?;

    let c_onnx = CString::new(onnx_str).map_err(|e| format!("CString: {}", e))?;
    let c_engine = CString::new(engine_str).map_err(|e| format!("CString: {}", e))?;

    eprintln!(
        "[ignite-ms] compiling INT8 per-bucket engine: batch={} seq={}",
        batch_size, seq_len
    );

    let rc = unsafe {
        trt_compile(
            c_onnx.as_ptr(),
            c_engine.as_ptr(),
            1,                 // min_batch
            batch_size as i32, // opt_batch = exact target
            batch_size as i32, // max_batch = exact target
            seq_len as i32,    // min_seq = exact
            seq_len as i32,    // opt_seq = exact
            seq_len as i32,    // max_seq = exact
            2,                 // fp16=2 means FP16 + INT8
        )
    };

    if rc != 0 {
        Err(format!(
            "trt_compile INT8 failed for bs={} seq={}",
            batch_size, seq_len
        ))
    } else {
        Ok(())
    }
}

/// A native TRT inference session for a specific (gpu, batch_size, seq_len).
pub struct TrtSession {
    ptr: *mut TrtRunnerHandle,
    pub batch_size: usize,
    pub seq_len: usize,
    pub hidden_dim: usize,
}

unsafe impl Send for TrtSession {}

impl TrtSession {
    /// Load engine and create session with pre-allocated GPU buffers.
    pub fn new(
        engine_path: &Path,
        gpu_id: u32,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<Self, String> {
        let path_str = engine_path
            .to_str()
            .ok_or_else(|| "engine path not valid UTF-8".to_string())?;
        let c_path = CString::new(path_str).map_err(|e| format!("CString: {}", e))?;

        let ptr = unsafe {
            trt_create(
                c_path.as_ptr(),
                gpu_id as i32,
                batch_size as i32,
                seq_len as i32,
                hidden_dim as i32,
            )
        };
        if ptr.is_null() {
            return Err(format!(
                "trt_create failed: engine={} gpu={}",
                path_str, gpu_id
            ));
        }

        Ok(Self {
            ptr,
            batch_size,
            seq_len,
            hidden_dim,
        })
    }

    /// Run inference. Input arrays must be [batch_size × seq_len] contiguous i64.
    pub fn infer(
        &mut self,
        input_ids: &[i64],
        attention_mask: &[i64],
        token_type_ids: Option<&[i64]>,
    ) -> Result<(), String> {
        let expected = self.batch_size * self.seq_len;
        if input_ids.len() != expected || attention_mask.len() != expected {
            return Err(format!(
                "size mismatch: expected {} got {}/{}",
                expected,
                input_ids.len(),
                attention_mask.len()
            ));
        }

        let ttids_ptr = match token_type_ids {
            Some(t) => t.as_ptr(),
            None => std::ptr::null(),
        };

        let rc = unsafe {
            trt_infer(
                self.ptr,
                input_ids.as_ptr(),
                attention_mask.as_ptr(),
                ttids_ptr,
            )
        };
        if rc != 0 {
            Err("trt_infer failed".to_string())
        } else {
            Ok(())
        }
    }

    /// Copy the output embeddings from GPU to a pre-allocated host buffer.
    /// Buffer must be at least batch_size * hidden_dim floats.
    /// Call this after infer() to get the embedding results.
    pub fn get_output(&self, output: &mut [f32]) -> Result<(), String> {
        let expected = self.batch_size * self.hidden_dim;
        if output.len() < expected {
            return Err(format!(
                "output buffer too small: {} < {}",
                output.len(),
                expected
            ));
        }
        let rc = unsafe { trt_get_output(self.ptr, output.as_mut_ptr()) };
        if rc != 0 {
            Err("trt_get_output failed (cudaMemcpy error)".to_string())
        } else {
            Ok(())
        }
    }
}

impl Drop for TrtSession {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { trt_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
