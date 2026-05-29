// trt_runner.cpp — Native TensorRT inference + engine compilation.
//
// Provides a C API for Rust FFI:
//   trt_compile() — compile ONNX model to serialized TRT engine
//   trt_create()  — load engine, create context, pre-allocate GPU buffers
//   trt_infer()   — copy inputs to GPU, run inference, synchronize
//   trt_destroy() — cleanup
//
// Built automatically by ignite-ms's build.rs via the `cc` crate.
// Requires: TensorRT headers (NvInfer.h), CUDA toolkit, nvonnxparser.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

// TRT logger (required by the runtime)
class SimpleLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            fprintf(stderr, "[TRT] %s\n", msg);
        }
    }
};

static SimpleLogger gLogger;

// INT8 Entropy Calibrator — uses real tokenized text for calibration.
// Reads pre-generated calibration data from a binary file (calibration_data.bin).
// File format: [u32 n_samples][u32 seq_len] then n_samples * seq_len int64 input_ids
//              followed by n_samples * seq_len int64 attention_mask.
// TRT profiles layer activations to determine optimal quantization ranges.
// After first calibration, results are cached to a file for instant reload.
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(int batch_size, int seq_len, int n_batches,
                          const std::string& cache_path,
                          const std::string& calib_data_path)
        : batch_size_(batch_size), seq_len_(seq_len),
          n_batches_(n_batches), current_batch_(0),
          cache_path_(cache_path) {
        size_t input_bytes = (size_t)batch_size * seq_len * sizeof(int64_t);
        cudaMalloc(&d_input_ids_, input_bytes);
        cudaMalloc(&d_attention_mask_, input_bytes);
        cudaMalloc(&d_token_type_ids_, input_bytes);

        // Try to load real calibration data
        std::ifstream df(calib_data_path, std::ios::binary);
        if (df.good()) {
            uint32_t n_samples = 0, file_seq_len = 0;
            df.read(reinterpret_cast<char*>(&n_samples), sizeof(uint32_t));
            df.read(reinterpret_cast<char*>(&file_seq_len), sizeof(uint32_t));
            fprintf(stderr, "[calibrator] Loading real data: %u samples, seq_len=%u (engine seq=%d)\n",
                    n_samples, file_seq_len, seq_len);

            // Read all input_ids and attention_masks
            size_t total_tokens = (size_t)n_samples * file_seq_len;
            h_input_ids_.resize(total_tokens);
            h_attention_mask_.resize(total_tokens);
            df.read(reinterpret_cast<char*>(h_input_ids_.data()), total_tokens * sizeof(int64_t));
            df.read(reinterpret_cast<char*>(h_attention_mask_.data()), total_tokens * sizeof(int64_t));

            calib_n_samples_ = n_samples;
            calib_seq_len_ = file_seq_len;
            has_real_data_ = true;
            // Adjust n_batches based on available data
            int max_batches = n_samples / batch_size;
            if (max_batches < n_batches_) n_batches_ = std::max(max_batches, 1);
            fprintf(stderr, "[calibrator] Using %d calibration batches of %d samples\n",
                    n_batches_, batch_size);
        } else {
            fprintf(stderr, "[calibrator] WARNING: %s not found, using synthetic data\n",
                    calib_data_path.c_str());
            has_real_data_ = false;
            // Fallback: synthetic data (will produce suboptimal quantization)
            std::vector<int64_t> ids(batch_size * seq_len);
            std::vector<int64_t> mask(batch_size * seq_len, 1);
            for (int i = 0; i < batch_size * seq_len; i++) {
                ids[i] = (i % 999) + 1;
            }
            cudaMemcpy(d_input_ids_, ids.data(), input_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_attention_mask_, mask.data(), input_bytes, cudaMemcpyHostToDevice);
        }
        // Zero out token_type_ids (most models don't use them meaningfully)
        cudaMemset(d_token_type_ids_, 0, input_bytes);
    }

    ~Int8EntropyCalibrator() {
        cudaFree(d_input_ids_);
        cudaFree(d_attention_mask_);
        cudaFree(d_token_type_ids_);
    }

    int32_t getBatchSize() const noexcept override { return batch_size_; }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        if (current_batch_ >= n_batches_) return false;

        if (has_real_data_) {
            // Copy real data for this batch
            size_t batch_elems = (size_t)batch_size_ * seq_len_;
            size_t offset = (size_t)current_batch_ * batch_size_ * calib_seq_len_;
            // Prepare batch: truncate/pad each sample from calib_seq_len_ to seq_len_
            std::vector<int64_t> batch_ids(batch_elems, 0);
            std::vector<int64_t> batch_mask(batch_elems, 0);
            int copy_len = std::min(seq_len_, (int)calib_seq_len_);
            for (int s = 0; s < batch_size_; s++) {
                size_t src_off = offset + (size_t)s * calib_seq_len_;
                size_t dst_off = (size_t)s * seq_len_;
                if (src_off + copy_len <= h_input_ids_.size()) {
                    memcpy(&batch_ids[dst_off], &h_input_ids_[src_off], copy_len * sizeof(int64_t));
                    memcpy(&batch_mask[dst_off], &h_attention_mask_[src_off], copy_len * sizeof(int64_t));
                }
            }
            size_t bytes = batch_elems * sizeof(int64_t);
            cudaMemcpy(d_input_ids_, batch_ids.data(), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_attention_mask_, batch_mask.data(), bytes, cudaMemcpyHostToDevice);
        }

        for (int i = 0; i < nbBindings; i++) {
            if (strcmp(names[i], "input_ids") == 0) {
                bindings[i] = d_input_ids_;
            } else if (strcmp(names[i], "attention_mask") == 0) {
                bindings[i] = d_attention_mask_;
            } else if (strcmp(names[i], "token_type_ids") == 0) {
                bindings[i] = d_token_type_ids_;
            } else {
                bindings[i] = d_attention_mask_;
            }
        }
        current_batch_++;
        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept override {
        cache_data_.clear();
        std::ifstream f(cache_path_, std::ios::binary);
        if (f.good()) {
            f.seekg(0, std::ios::end);
            length = f.tellg();
            f.seekg(0, std::ios::beg);
            cache_data_.resize(length);
            f.read(cache_data_.data(), length);
            fprintf(stderr, "[calibrator] INT8 calibration cache loaded (%zu bytes)\n", length);
            return cache_data_.data();
        }
        length = 0;
        return nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        std::ofstream f(cache_path_, std::ios::binary);
        f.write(static_cast<const char*>(cache), length);
        fprintf(stderr, "[calibrator] INT8 calibration cache saved (%zu bytes)\n", length);
    }

private:
    int batch_size_, seq_len_, n_batches_, current_batch_;
    int calib_n_samples_ = 0;
    int calib_seq_len_ = 0;
    bool has_real_data_ = false;
    std::string cache_path_;
    void* d_input_ids_ = nullptr;
    void* d_attention_mask_ = nullptr;
    void* d_token_type_ids_ = nullptr;
    std::vector<int64_t> h_input_ids_;
    std::vector<int64_t> h_attention_mask_;
    std::vector<char> cache_data_;
};

struct TrtRunner {
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t compute_stream;
    cudaStream_t copy_stream;
    cudaEvent_t compute_done;

    // Pre-allocated buffers
    void* d_input_ids;
    void* d_attention_mask;
    void* d_token_type_ids;
    void* d_output;          // single device output buffer
    float* h_output;         // pinned host output buffer for async D2H

    int batch_size;
    int seq_len;
    int hidden_dim;
    int gpu_id;
    int has_token_type_ids;  // 1 if model uses token_type_ids, 0 otherwise
};

extern "C" {

// Compile an ONNX model to a serialized TRT engine file.
// Uses FP16 (and optionally INT8), dynamic batch/seq shapes with the given min/opt/max.
// Returns 0 on success, -1 on failure.
int trt_compile(
    const char* onnx_path,
    const char* engine_path,
    int min_batch, int opt_batch, int max_batch,
    int min_seq, int opt_seq, int max_seq,
    int fp16
) {
    // INT8 mode: if fp16 == 2, enable both FP16 + INT8
    int use_int8 = (fp16 == 2) ? 1 : 0;
    int use_fp16 = (fp16 >= 1) ? 1 : 0;

    fprintf(stderr, "[trt_compile] ONNX: %s\n", onnx_path);
    fprintf(stderr, "[trt_compile] Output: %s\n", engine_path);
    fprintf(stderr, "[trt_compile] Shapes: batch=[%d,%d,%d] seq=[%d,%d,%d] fp16=%d int8=%d\n",
            min_batch, opt_batch, max_batch, min_seq, opt_seq, max_seq, use_fp16, use_int8);

    // Create builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    if (!builder) {
        fprintf(stderr, "[trt_compile] failed to create builder\n");
        return -1;
    }

    // Create network (explicit batch)
    const uint32_t flags = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flags);
    if (!network) {
        fprintf(stderr, "[trt_compile] failed to create network\n");
        delete builder;
        return -1;
    }

    // Parse ONNX
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser) {
        fprintf(stderr, "[trt_compile] failed to create ONNX parser\n");
        delete network;
        delete builder;
        return -1;
    }

    fprintf(stderr, "[trt_compile] parsing ONNX model...\n");
    if (!parser->parseFromFile(onnx_path,
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        fprintf(stderr, "[trt_compile] ONNX parse failed:\n");
        for (int i = 0; i < parser->getNbErrors(); i++) {
            fprintf(stderr, "  %s\n", parser->getError(i)->desc());
        }
        delete parser;
        delete network;
        delete builder;
        return -1;
    }
    fprintf(stderr, "[trt_compile] ONNX parsed: %d inputs, %d outputs\n",
            network->getNbInputs(), network->getNbOutputs());

    // Log I/O names
    for (int i = 0; i < network->getNbInputs(); i++) {
        auto* inp = network->getInput(i);
        fprintf(stderr, "[trt_compile]   input[%d]: %s\n", i, inp->getName());
    }
    for (int i = 0; i < network->getNbOutputs(); i++) {
        auto* out = network->getOutput(i);
        fprintf(stderr, "[trt_compile]   output[%d]: %s\n", i, out->getName());
    }

    // Build config
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, (size_t)32 << 30);  // 32 GB (A100 has 40GB)
    if (use_fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        fprintf(stderr, "[trt_compile] FP16 enabled\n");
    }

    // INT8 quantization (with calibration)
    Int8EntropyCalibrator* calibrator = nullptr;
    if (use_int8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        // Calibration cache stored alongside the engine
        std::string cache_path = std::string(engine_path) + ".int8cache";
        // Look for calibration data in the same directory as the ONNX model
        std::string onnx_dir = std::string(onnx_path);
        size_t last_slash = onnx_dir.rfind('/');
        std::string calib_data_path = (last_slash != std::string::npos)
            ? onnx_dir.substr(0, last_slash) + "/calibration_data.bin"
            : "calibration_data.bin";
        // TRT uses kOPT shapes for calibration — calibrator buffers must match
        calibrator = new Int8EntropyCalibrator(
            opt_batch,     // must match optimization profile's opt_batch
            opt_seq,       // must match optimization profile's opt_seq
            10,            // number of calibration batches
            cache_path,
            calib_data_path
        );
        config->setInt8Calibrator(calibrator);
        fprintf(stderr, "[trt_compile] INT8 enabled (calibration batch=%d seq=%d, cache: %s)\n",
                opt_batch, opt_seq, cache_path.c_str());
    }

    // Optimization profile (dynamic shapes)
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    // Only set profiles for inputs that actually exist in the model
    for (int i = 0; i < network->getNbInputs(); i++) {
        const char* name = network->getInput(i)->getName();
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims2(min_batch, min_seq));
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims2(opt_batch, opt_seq));
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims2(max_batch, max_seq));
    }
    config->addOptimizationProfile(profile);

    // Build engine
    fprintf(stderr, "[trt_compile] building engine (this takes 2-5 minutes)...\n");
    nvinfer1::IHostMemory* engine_data = builder->buildSerializedNetwork(*network, *config);
    if (!engine_data) {
        fprintf(stderr, "[trt_compile] engine build FAILED\n");
        delete config;
        delete parser;
        delete network;
        delete builder;
        return -1;
    }

    // Write to file
    std::ofstream out(engine_path, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "[trt_compile] cannot open output file: %s\n", engine_path);
        delete engine_data;
        delete config;
        delete parser;
        delete network;
        delete builder;
        return -1;
    }
    out.write(reinterpret_cast<const char*>(engine_data->data()), engine_data->size());
    out.close();

    fprintf(stderr, "[trt_compile] engine saved: %s (%.1f MB)\n",
            engine_path, (double)engine_data->size() / 1e6);

    delete engine_data;
    if (calibrator) delete calibrator;
    delete config;
    delete parser;
    delete network;
    delete builder;
    return 0;
}

// Create a TRT runner from a serialized engine file.
// Returns NULL on failure.
TrtRunner* trt_create(const char* engine_path, int gpu_id, int batch_size, int seq_len, int hidden_dim) {
    cudaSetDevice(gpu_id);

    // Read engine file
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        fprintf(stderr, "[trt_runner] cannot open engine: %s\n", engine_path);
        return nullptr;
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    if (!file.read(data.data(), size)) {
        fprintf(stderr, "[trt_runner] cannot read engine: %s\n", engine_path);
        return nullptr;
    }
    file.close();

    // Create runtime and deserialize engine
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        fprintf(stderr, "[trt_runner] failed to create runtime\n");
        return nullptr;
    }

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(data.data(), size);
    if (!engine) {
        fprintf(stderr, "[trt_runner] failed to deserialize engine from %s\n", engine_path);
        delete runtime;
        return nullptr;
    }

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        fprintf(stderr, "[trt_runner] failed to create execution context\n");
        delete engine;
        delete runtime;
        return nullptr;
    }

    // Create CUDA streams + event
    cudaStream_t compute_stream, copy_stream;
    cudaEvent_t compute_done;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&copy_stream);
    cudaEventCreate(&compute_done);

    // Pre-allocate device memory
    size_t input_bytes = (size_t)batch_size * seq_len * sizeof(int64_t);
    size_t output_bytes = (size_t)batch_size * hidden_dim * sizeof(float);

    void* d_input_ids = nullptr;
    void* d_attention_mask = nullptr;
    void* d_token_type_ids = nullptr;
    void* d_output = nullptr;
    float* h_output = nullptr;

    cudaMalloc(&d_input_ids, input_bytes);
    cudaMalloc(&d_attention_mask, input_bytes);
    cudaMalloc(&d_output, output_bytes);
    // Pinned host memory for async D2H copy
    cudaMallocHost(&h_output, output_bytes);

    // Check if engine has token_type_ids input
    int has_token_type_ids = 0;
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* tname = engine->getIOTensorName(i);
        if (strcmp(tname, "token_type_ids") == 0) {
            has_token_type_ids = 1;
            break;
        }
    }
    if (has_token_type_ids) {
        cudaMalloc(&d_token_type_ids, input_bytes);
    }

    // Set tensor addresses on the context
    context->setTensorAddress("input_ids", d_input_ids);
    context->setTensorAddress("attention_mask", d_attention_mask);
    if (has_token_type_ids) {
        context->setTensorAddress("token_type_ids", d_token_type_ids);
    }
    context->setTensorAddress("sentence_embedding", d_output);

    // Set input shapes (dynamic shapes need explicit setting)
    nvinfer1::Dims dims;
    dims.nbDims = 2;
    dims.d[0] = batch_size;
    dims.d[1] = seq_len;
    context->setInputShape("input_ids", dims);
    context->setInputShape("attention_mask", dims);
    if (has_token_type_ids) {
        context->setInputShape("token_type_ids", dims);
    }

    TrtRunner* runner = new TrtRunner();
    runner->runtime = runtime;
    runner->engine = engine;
    runner->context = context;
    runner->compute_stream = compute_stream;
    runner->copy_stream = copy_stream;
    runner->compute_done = compute_done;
    runner->d_input_ids = d_input_ids;
    runner->d_attention_mask = d_attention_mask;
    runner->d_token_type_ids = d_token_type_ids;
    runner->d_output = d_output;
    runner->h_output = h_output;
    runner->batch_size = batch_size;
    runner->seq_len = seq_len;
    runner->hidden_dim = hidden_dim;
    runner->gpu_id = gpu_id;
    runner->has_token_type_ids = has_token_type_ids;

    return runner;
}

// Run inference + async D2H copy.
// After this returns, output is being copied to pinned host memory.
// Call trt_get_output() to wait for copy and read results.
// Returns 0 on success, -1 on failure.
int trt_infer(TrtRunner* runner, const int64_t* h_input_ids, const int64_t* h_attention_mask, const int64_t* h_token_type_ids) {
    if (!runner) return -1;
    cudaSetDevice(runner->gpu_id);

    size_t input_bytes = (size_t)runner->batch_size * runner->seq_len * sizeof(int64_t);
    size_t output_bytes = (size_t)runner->batch_size * runner->hidden_dim * sizeof(float);

    // Copy inputs to device
    cudaMemcpyAsync(runner->d_input_ids, h_input_ids, input_bytes, cudaMemcpyHostToDevice, runner->compute_stream);
    cudaMemcpyAsync(runner->d_attention_mask, h_attention_mask, input_bytes, cudaMemcpyHostToDevice, runner->compute_stream);
    if (runner->has_token_type_ids) {
        if (h_token_type_ids) {
            cudaMemcpyAsync(runner->d_token_type_ids, h_token_type_ids, input_bytes, cudaMemcpyHostToDevice, runner->compute_stream);
        } else {
            cudaMemsetAsync(runner->d_token_type_ids, 0, input_bytes, runner->compute_stream);
        }
    }

    // Run inference
    bool ok = runner->context->enqueueV3(runner->compute_stream);
    if (!ok) {
        fprintf(stderr, "[trt_runner] enqueueV3 failed\n");
        return -1;
    }

    // Signal compute done, then async copy output to pinned host memory
    cudaEventRecord(runner->compute_done, runner->compute_stream);
    cudaStreamWaitEvent(runner->copy_stream, runner->compute_done, 0);
    cudaMemcpyAsync(runner->h_output, runner->d_output, output_bytes, cudaMemcpyDeviceToHost, runner->copy_stream);

    return 0;
}

// Wait for output copy and return embeddings.
// h_output must be pre-allocated: batch_size * hidden_dim * sizeof(float).
// Returns 0 on success, -1 on failure.
int trt_get_output(TrtRunner* runner, float* h_output) {
    if (!runner || !h_output) return -1;
    cudaSetDevice(runner->gpu_id);
    size_t output_bytes = (size_t)runner->batch_size * runner->hidden_dim * sizeof(float);

    // Wait for async D2H copy to complete
    cudaStreamSynchronize(runner->copy_stream);

    // Copy from pinned host buffer to caller's buffer
    memcpy(h_output, runner->h_output, output_bytes);
    return 0;
}

// Cleanup
void trt_destroy(TrtRunner* runner) {
    if (!runner) return;
    cudaFree(runner->d_input_ids);
    cudaFree(runner->d_attention_mask);
    if (runner->d_token_type_ids) cudaFree(runner->d_token_type_ids);
    cudaFree(runner->d_output);
    cudaFreeHost(runner->h_output);
    cudaEventDestroy(runner->compute_done);
    cudaStreamDestroy(runner->compute_stream);
    cudaStreamDestroy(runner->copy_stream);
    delete runner->context;
    delete runner->engine;
    delete runner->runtime;
    delete runner;
}

} // extern "C"
