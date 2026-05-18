# Contributing to IgniteMS

We welcome contributions to IgniteMS! This document explains how to get involved.

## Licensing

IgniteMS is licensed under Apache 2.0. Artain may offer future versions of IgniteMS or
related commercial editions under different terms. Versions released under Apache 2.0
remain available under Apache 2.0.

## Contributor License Agreement (CLA)

Before we can accept your first contribution, you must sign our Contributor License
Agreement. This grants Artain the necessary rights to distribute your contribution
under the project's license and any future license terms.

**How to sign:**

When you open your first pull request, confirm the CLA checkbox in the pull request
template or post this exact comment:

> I have read the CLA Document (version 1.0) and I agree to its terms.

Maintainers will not merge external contributions until the CLA acceptance is recorded
on the pull request.

The full CLA text is available in [CLA.md](CLA.md).

**Why a CLA?**

We use a CLA (not just a DCO) because it grants explicit relicensing rights. This allows
Artain to offer future versions under different terms if needed, while ensuring that all
Apache 2.0 releases remain permanently available under Apache 2.0. You retain copyright
of your contributions.

**Corporate contributors:**

If you're contributing on behalf of your employer, the CLA covers this (Section 8).
For large organizations preferring a formal Corporate CLA, contact legal@artain.com.

## How to Contribute

### Reporting Issues

- Use GitHub Issues for bugs and feature requests
- Include reproduction steps, environment details (GPU, TRT version, OS), and expected vs actual behavior

### Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run `cargo fmt --all` and `cargo clippy --workspace -- -D warnings`
4. If you added functionality, add or update tests
5. Open a PR with a clear description of what and why

### Development Setup

Requirements:
- Rust 1.85+
- NVIDIA GPU with CUDA 12+ and TensorRT 10+
- Python 3.10+ (for model provisioning)

```bash
cargo build --release -p ignite-ms-embed
```

GitHub-hosted CI checks formatting, Python syntax, Dockerfile syntax, and leak
patterns. Full Rust builds and clippy require CUDA/TensorRT and should be run on
a GPU build machine before release.

### Code Style

- `cargo fmt` — non-negotiable
- No warnings under `clippy -D warnings`
- Minimal comments — only when the "why" is non-obvious
- No unnecessary abstractions

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0, subject to the terms of the CLA.
