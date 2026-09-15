# Installation

## 1. Create a new rust app or library

```bash
cargo new myapp
cd myapp
```

## 2. Add the correct hanzo version

### Standard

```bash
cargo add --git https://github.com/hanzoai/ml.git hanzo-ml
```

### CUDA

First, make sure that Cuda is correctly installed.
- `nvcc --version` should print information about your Cuda compiler driver.
- `nvidia-smi --query-gpu=compute_cap --format=csv` should print your GPUs compute capability, e.g. something
like:

```bash
compute_cap
8.9
```

You can also compile the Cuda kernels for a specific compute cap using the 
`CUDA_COMPUTE_CAP=<compute cap>` environment variable.

If any of the above commands errors out, please make sure to update your Cuda version.

Add the `hanzo-ml` crate with the cuda feature:

```bash
cargo add --git https://github.com/hanzoai/ml.git hanzo-ml --features "cuda"
```

### cuTile

The opt-in `cutile` feature enables CUDA and re-exports the cuTile version used by Hanzo:

```bash
cargo add --git https://github.com/huggingface/hanzo.git hanzo-ml --features "cutile"
```

It requires Rust 1.89 or newer, CUDA 13.2 or newer, NVIDIA driver r580 or newer, clang and libclang
at build time, and the CUDA `tileiras` compiler at runtime. Set `CUDA_TOOLKIT_PATH` to the toolkit
root when it is not installed in a standard location. See [Writing cuTile kernels](cutile.md) for
the API and CUDA architecture requirements.

### MKL

You can also see the `mkl` feature which can get faster inference on CPU.

Add the `hanzo-ml` crate with the mkl feature:

```bash
cargo add --git https://github.com/hanzoai/ml.git hanzo-ml --features "mkl"
```

### Metal

Metal is exclusive to MacOS.

Add the `hanzo-ml` crate with the metal feature:

```bash
cargo add --git https://github.com/hanzoai/ml.git hanzo-ml --features "metal"
```

## 3. Building

Run `cargo build` to make sure everything can be correctly built.

```bash
cargo build
```
