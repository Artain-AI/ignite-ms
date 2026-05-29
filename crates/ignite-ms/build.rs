fn main() {
    #[cfg(feature = "native-trt")]
    {
        let cuda_include = std::env::var("CUDA_INCLUDE_PATH")
            .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
        let cuda_lib =
            std::env::var("CUDA_LIB_PATH").unwrap_or_else(|_| "/usr/local/cuda/lib64".to_string());
        let trt_include =
            std::env::var("TRT_INCLUDE_PATH").unwrap_or_else(|_| "/usr/include".to_string());
        let trt_lib = std::env::var("TRT_LIB_PATH").unwrap_or_else(|_| "/usr/lib64".to_string());

        cc::Build::new()
            .cpp(true)
            .file("../../native/trt_runner.cpp")
            .include(&cuda_include)
            .include(&trt_include)
            .opt_level(2)
            .flag("-std=c++17")
            .flag("-Wno-deprecated-declarations")
            .compile("trt_runner");

        println!("cargo:rustc-link-search=native={}", cuda_lib);
        println!("cargo:rustc-link-search=native={}", trt_lib);
        // Also search pip-installed CUDA runtime (common on AL2023/dnf setups)
        println!("cargo:rustc-link-search=native=/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib");
        println!("cargo:rustc-link-lib=dylib=nvinfer");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=nvonnxparser");
        println!("cargo:rustc-link-lib=dylib=stdc++");

        println!("cargo:rerun-if-changed=../../native/trt_runner.cpp");
        println!("cargo:rerun-if-env-changed=CUDA_INCLUDE_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_LIB_PATH");
        println!("cargo:rerun-if-env-changed=TRT_INCLUDE_PATH");
        println!("cargo:rerun-if-env-changed=TRT_LIB_PATH");
    }
}
