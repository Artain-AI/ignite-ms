# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Build the Rust binary
# ═══════════════════════════════════════════════════════════════════════
FROM nvcr.io/nvidia/tensorrt:24.10-py3 AS builder

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.85.0
ENV PATH="/root/.cargo/bin:${PATH}"

ENV CUDA_INCLUDE_PATH=/usr/local/cuda/include
ENV CUDA_LIB_PATH=/usr/local/cuda/lib64
ENV TRT_INCLUDE_PATH=/usr/include/x86_64-linux-gnu
ENV TRT_LIB_PATH=/usr/lib/x86_64-linux-gnu

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY native/ native/

RUN cargo build --release -p ignite-ms-embed -p ignite-ms-bench

# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Runtime image
# ═══════════════════════════════════════════════════════════════════════
FROM nvcr.io/nvidia/tensorrt:24.10-py3

RUN pip install --quiet --no-cache-dir onnx==1.16.1 numpy==1.26.4 tokenizers==0.20.3

COPY --from=builder /build/target/release/ignite-ms /usr/local/bin/ignite-ms
COPY --from=builder /build/target/release/ignite-ms-bench /usr/local/bin/ignite-ms-bench
COPY crates/ignite-ms/scripts/ /opt/ignite-ms/scripts/

ENV IGNITE_MS_CACHE=/opt/ignite-ms/cache
RUN mkdir -p /opt/ignite-ms/cache

ENTRYPOINT ["ignite-ms"]
CMD ["--help"]
