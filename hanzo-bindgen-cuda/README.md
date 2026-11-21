# hanzo-bindgen-cuda

Compile CUDA kernels from a `build.rs`, either to PTX embedded as string constants or to a
static library called through FFI. `nvcc` runs once per kernel file, in parallel, and only
when the source changed. The library name is `bindgen_cuda`.

`CUDA_COMPUTE_CAP` picks the architecture (else `nvidia-smi` is asked), `NVCC` the compiler
and `NVCC_CCBIN` its host compiler.


## PTX inclusion
Let's say you have a file

`src/cuda.cu`
```cuda
__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}
```

Add it as a build dependency:

```bash
cargo add --build hanzo-bindgen-cuda
```

And then create this `build.rs`

```no_run
fn main() {
    let builder = bindgen_cuda::Builder::default();
    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/lib.rs");
}
```

This will create a src file containing the following code:

```ignore
pub const CUDA: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx"));
```

You can then use the PTX directly in your rust code with a library like [cudarc](https://github.com/coreylowman/cudarc/).

## Raw cuda calls
Alternatively you can build a static library that you can link against in build.rs in order to call cuda directly with the c code.

`src/cuda.cu`

```cuda
__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int run() {
    cuda_hello<<<1,1>>>(); 
    return 0;
}
```


Then write the `build.rs`:

```no_run
fn main() {
    let builder = bindgen_cuda::Builder::default();
    builder.build_lib("libcuda.a");
    println!("cargo:rustc-link-lib=cuda");
}
```

Which you can then interface through FFI in `src/lib.rs`:


```no_run
extern "C" {
    fn cuda_hello();
}
fn main(){
    unsafe{ cuda_hello();}
}
```

