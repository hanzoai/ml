# hanzo
[![discord server](https://dcbadge.vercel.app/api/server/hugging-face-879548962464493619)](https://discord.gg/hugging-face-879548962464493619)
[![Latest version](https://img.shields.io/crates/v/hanzo-ml-core.svg)](https://crates.io/crates/hanzo-ml-core)
[![Documentation](https://docs.rs/hanzo-ml-core/badge.svg)](https://docs.rs/hanzo-ml-core)
[![License](https://img.shields.io/github/license/base-org/node?color=blue)](https://github.com/huggingface/hanzo/blob/main/LICENSE-MIT)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://github.com/huggingface/hanzo/blob/main/LICENSE-APACHE)

ML is a minimalist ML framework for Rust with a focus on performance (including GPU support) 
and ease of use. Try our online demos: 
[whisper](https://huggingface.co/spaces/lmz/hanzo-ml-whisper),
[LLaMA2](https://huggingface.co/spaces/lmz/hanzo-ml-llama2),
[T5](https://huggingface.co/spaces/radames/ML-T5-Generation-Wasm),
[yolo](https://huggingface.co/spaces/lmz/hanzo-ml-yolo),
[Segment
Anything](https://huggingface.co/spaces/radames/hanzo-ml-segment-anything-wasm).

## Get started

Make sure that you have [`hanzo-ml-core`](https://github.com/huggingface/hanzo/tree/main/hanzo-ml-core) correctly installed as described in [**Installation**](https://huggingface.github.io/hanzo/guide/installation.html).

Let's see how to run a simple matrix multiplication.
Write the following to your `myapp/src/main.rs` file:
```rust
use hanzo_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let c = a.matmul(&b)?;
    println!("{c}");
    Ok(())
}
```

`cargo run` should display a tensor of shape `Tensor[[2, 4], f32]`.


Having installed `hanzo` with Cuda support, simply define the `device` to be on GPU:

```diff
- let device = Device::Cpu;
+ let device = Device::new_cuda(0)?;
```

For more advanced examples, please have a look at the following section.

## Check out our examples

These online demos run entirely in your browser:
- [yolo](https://huggingface.co/spaces/lmz/hanzo-ml-yolo): pose estimation and
  object recognition.
- [whisper](https://huggingface.co/spaces/lmz/hanzo-ml-whisper): speech recognition.
- [LLaMA2](https://huggingface.co/spaces/lmz/hanzo-ml-llama2): text generation.
- [T5](https://huggingface.co/spaces/radames/ML-T5-Generation-Wasm): text generation.
- [Phi-1.5, and Phi-2](https://huggingface.co/spaces/radames/ML-Phi-1.5-Wasm): text generation.
- [Segment Anything Model](https://huggingface.co/spaces/radames/hanzo-ml-segment-anything-wasm): Image segmentation.
- [BLIP](https://huggingface.co/spaces/radames/ML-BLIP-Image-Captioning): image captioning.

We also provide some command line based examples using state of the art models:

- [LLaMA v1, v2, and v3](./hanzo-ml-examples/examples/llama/): general LLM, includes
  the SOLAR-10.7B variant.
- [Falcon](./hanzo-ml-examples/examples/falcon/): general LLM.
- [Codegeex4](./hanzo-ml-examples/examples/codegeex4-9b/): Code completion, code interpreter, web search, function calling, repository-level
- [GLM4](./hanzo-ml-examples/examples/glm4/): Open Multilingual Multimodal Chat LMs by THUDM
- [Gemma v1 and v2](./hanzo-ml-examples/examples/gemma/): 2b and 7b+/9b general LLMs from Google Deepmind.
- [RecurrentGemma](./hanzo-ml-examples/examples/recurrent-gemma/): 2b and 7b
  Griffin based models from Google that mix attention with a RNN like state.
- [Phi-1, Phi-1.5, Phi-2, and Phi-3](./hanzo-ml-examples/examples/phi/): 1.3b,
  2.7b, and 3.8b general LLMs with performance on par with 7b models.
- [StableLM-3B-4E1T](./hanzo-ml-examples/examples/stable-lm/): a 3b general LLM
  pre-trained on 1T tokens of English and code datasets. Also supports
  StableLM-2, a 1.6b LLM trained on 2T tokens, as well as the code variants.
- [Mamba](./hanzo-ml-examples/examples/mamba/): an inference only
  implementation of the Mamba state space model.
- [Mistral7b-v0.1](./hanzo-ml-examples/examples/mistral/): a 7b general LLM with
  better performance than all publicly available 13b models as of 2023-09-28.
- [Mixtral8x7b-v0.1](./hanzo-ml-examples/examples/mixtral/): a sparse mixture of
  experts 8x7b general LLM with better performance than a Llama 2 70B model with
  much faster inference.
- [StarCoder](./hanzo-ml-examples/examples/bigcode/) and
  [StarCoder2](./hanzo-ml-examples/examples/starcoder2/): LLM specialized to code generation.
- [Qwen1.5](./hanzo-ml-examples/examples/qwen/): Bilingual (English/Chinese) LLMs.
- [RWKV v5 and v6](./hanzo-ml-examples/examples/rwkv/): An RNN with transformer level LLM
  performance.
- [Replit-code-v1.5](./hanzo-ml-examples/examples/replit-code/): a 3.3b LLM specialized for code completion.
- [Yi-6B / Yi-34B](./hanzo-ml-examples/examples/yi/): two bilingual
  (English/Chinese) general LLMs with 6b and 34b parameters.
- [Quantized LLaMA](./hanzo-ml-examples/examples/quantized/): quantized version of
  the LLaMA model using the same quantization techniques as
  [llama.cpp](https://github.com/ggerganov/llama.cpp).

<img src="https://github.com/huggingface/hanzo/raw/main/hanzo-ml-examples/examples/quantized/assets/aoc.gif" width="600">
  
- [Stable Diffusion](./hanzo-ml-examples/examples/stable-diffusion/): text to
  image generative model, support for the 1.5, 2.1, SDXL 1.0 and Turbo versions.

<img src="https://github.com/huggingface/hanzo/raw/main/hanzo-ml-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg" width="200">

- [Wuerstchen](./hanzo-ml-examples/examples/wuerstchen/): another text to
  image generative model.

<img src="https://github.com/huggingface/hanzo/raw/main/hanzo-ml-examples/examples/wuerstchen/assets/cat.jpg" width="200">

- [yolo-v3](./hanzo-ml-examples/examples/yolo-v3/) and
  [yolo-v8](./hanzo-ml-examples/examples/yolo-v8/): object detection and pose
  estimation models.

<img src="https://github.com/huggingface/hanzo/raw/main/hanzo-ml-examples/examples/yolo-v8/assets/bike.od.jpg" width="200"><img src="https://github.com/huggingface/hanzo/raw/main/hanzo-ml-examples/examples/yolo-v8/assets/bike.pose.jpg" width="200">
- [segment-anything](./hanzo-ml-examples/examples/segment-anything/): image
  segmentation model with prompt.

<img src="https://github.com/huggingface/hanzo/raw/main/hanzo-ml-examples/examples/segment-anything/assets/sam_merged.jpg" width="200">

- [SegFormer](./hanzo-ml-examples/examples/segformer/): transformer based semantic segmentation model.
- [Whisper](./hanzo-ml-examples/examples/whisper/): speech recognition model.
- [EnCodec](./hanzo-ml-examples/examples/encodec/): high-quality audio compression
  model using residual vector quantization.
- [MetaVoice](./hanzo-ml-examples/examples/metavoice/): foundational model for
  text-to-speech.
- [Parler-TTS](./hanzo-ml-examples/examples/parler-tts/): large text-to-speech
  model.
- [T5](./hanzo-ml-examples/examples/t5), [Bert](./hanzo-ml-examples/examples/bert/),
  [JinaBert](./hanzo-ml-examples/examples/jina-bert/) : useful for sentence embeddings.
- [DINOv2](./hanzo-ml-examples/examples/dinov2/): computer vision model trained
  using self-supervision (can be used for imagenet classification, depth
  evaluation, segmentation).
- [VGG](./hanzo-ml-examples/examples/vgg/),
  [RepVGG](./hanzo-ml-examples/examples/repvgg): computer vision models.
- [BLIP](./hanzo-ml-examples/examples/blip/): image to text model, can be used to
  generate captions for an image.
- [CLIP](./hanzo-ml-examples/examples/clip/): multi-model vision and language
  model.
- [TrOCR](./hanzo-ml-examples/examples/trocr/): a transformer OCR model, with
  dedicated submodels for hand-writing and printed recognition.
- [Marian-MT](./hanzo-ml-examples/examples/marian-mt/): neural machine translation
  model, generates the translated text from the input text.
- [Moondream](./hanzo-ml-examples/examples/moondream/): tiny computer-vision model 
  that can answer real-world questions about images.

Run them using commands like:
```
cargo run --example quantized --release
```

In order to use **CUDA** add `--features cuda` to the example command line. If
you have cuDNN installed, use `--features cudnn` for even more speedups.

There are also some wasm examples for whisper and
[llama2.c](https://github.com/karpathy/llama2.c). You can either build them with
`trunk` or try them online:
[whisper](https://huggingface.co/spaces/lmz/hanzo-ml-whisper),
[llama2](https://huggingface.co/spaces/lmz/hanzo-ml-llama2),
[T5](https://huggingface.co/spaces/radames/ML-T5-Generation-Wasm),
[Phi-1.5, and Phi-2](https://huggingface.co/spaces/radames/ML-Phi-1.5-Wasm),
[Segment Anything Model](https://huggingface.co/spaces/radames/hanzo-ml-segment-anything-wasm).

For LLaMA2, run the following command to retrieve the weight files and start a
test server:
```bash
cd hanzo-ml-wasm-examples/llama2-c
wget https://huggingface.co/spaces/lmz/hanzo-ml-llama2/resolve/main/model.bin
wget https://huggingface.co/spaces/lmz/hanzo-ml-llama2/resolve/main/tokenizer.json
trunk serve --release --port 8081
```
And then head over to
[http://localhost:8081/](http://localhost:8081/).

<!--- ANCHOR: useful_libraries --->

## Useful External Resources
- [`hanzo-ml-tutorial`](https://github.com/ToluClassics/hanzo-ml-tutorial): A
  very detailed tutorial showing how to convert a PyTorch model to ML.
- [`hanzo-ml-lora`](https://github.com/EricLBuehler/hanzo-ml-lora): Efficient and
  ergonomic LoRA implementation for ML. `hanzo-ml-lora` has      
  out-of-the-box LoRA support for many models from ML, which can be found
  [here](https://github.com/EricLBuehler/hanzo-ml-lora/tree/master/hanzo-ml-lora-transformers/examples).
- [`optimisers`](https://github.com/KGrewal1/optimisers): A collection of optimisers
  including SGD with momentum, AdaGrad, AdaDelta, AdaMax, NAdam, RAdam, and RMSprop.
- [`hanzo-ml-vllm`](https://github.com/EricLBuehler/hanzo-ml-vllm): Efficient platform for inference and
  serving local LLMs including an OpenAI compatible API server.
- [`hanzo-ml-ext`](https://github.com/mokeyish/hanzo-ml-ext): An extension library to ML that provides PyTorch functions not currently available in ML.
- [`hanzo-ml-coursera-ml`](https://github.com/vishpat/hanzo-ml-coursera-ml): Implementation of ML algorithms from Coursera's [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) course.
- [`kalosm`](https://github.com/floneum/floneum/tree/master/interfaces/kalosm): A multi-modal meta-framework in Rust for interfacing with local pre-trained models with support for controlled generation, custom samplers, in-memory vector databases, audio transcription, and more.
- [`hanzo-ml-sampling`](https://github.com/EricLBuehler/hanzo-ml-sampling): Sampling techniques for ML.
- [`gpt-from-scratch-rs`](https://github.com/jeroenvlek/gpt-from-scratch-rs): A port of Andrej Karpathy's _Let's build GPT_ tutorial on YouTube showcasing the ML API on a toy problem.
- [`hanzo-ml-einops`](https://github.com/tomsanbear/hanzo-ml-einops): A pure rust implementation of the python [einops](https://github.com/arogozhnikov/einops) library.
- [`atoma-infer`](https://github.com/atoma-network/atoma-infer): A Rust library for fast inference at scale, leveraging FlashAttention2 for efficient attention computation, PagedAttention for efficient KV-cache memory management, and multi-GPU support. It is OpenAI api compatible.
- [`llms-from-scratch-rs`](https://github.com/nerdai/llms-from-scratch-rs): A comprehensive Rust translation of the code from Sebastian Raschka's Build an LLM from Scratch book.

If you have an addition to this list, please submit a pull request.

<!--- ANCHOR_END: useful_libraries --->

<!--- ANCHOR: features --->

## Features

- Simple syntax, looks and feels like PyTorch.
    - Model training.
    - Embed user-defined ops/kernels, such as [flash-attention v2](https://github.com/huggingface/hanzo/blob/89ba005962495f2bfbda286e185e9c3c7f5300a3/hanzo-ml-flash-attn/src/lib.rs#L152).
- Backends.
    - Optimized CPU backend with optional MKL support for x86 and Accelerate for macs.
    - CUDA backend for efficiently running on GPUs, multiple GPU distribution via NCCL.
    - WASM support, run your models in a browser.
- Included models.
    - Language Models.
        - LLaMA v1, v2, and v3 with variants such as SOLAR-10.7B.
        - Falcon.
        - StarCoder, StarCoder2.
        - Phi 1, 1.5, 2, and 3.
        - Mamba, Minimal Mamba
        - Gemma v1 2b and 7b+, v2 2b and 9b.
        - Mistral 7b v0.1.
        - Mixtral 8x7b v0.1.
        - StableLM-3B-4E1T, StableLM-2-1.6B, Stable-Code-3B.
        - Replit-code-v1.5-3B.
        - Bert.
        - Yi-6B and Yi-34B.
        - Qwen1.5, Qwen1.5 MoE.
        - RWKV v5 and v6.
    - Quantized LLMs.
        - Llama 7b, 13b, 70b, as well as the chat and code variants.
        - Mistral 7b, and 7b instruct.
        - Mixtral 8x7b.
        - Zephyr 7b a and b (Mistral-7b based).
        - OpenChat 3.5 (Mistral-7b based).
    - Text to text.
        - T5 and its variants: FlanT5, UL2, MADLAD400 (translation), CoEdit (Grammar correction).
        - Marian MT (Machine Translation).
    - Text to image.
        - Stable Diffusion v1.5, v2.1, XL v1.0.
        - Wurstchen v2.
    - Image to text.
        - BLIP.
        - TrOCR.
    - Audio.
        - Whisper, multi-lingual speech-to-text.
        - EnCodec, audio compression model.
        - MetaVoice-1B, text-to-speech model.
        - Parler-TTS, text-to-speech model.
    - Computer Vision Models.
        - DINOv2, ConvMixer, EfficientNet, ResNet, ViT, VGG, RepVGG, ConvNeXT,
          ConvNeXTv2, MobileOne, EfficientVit (MSRA), MobileNetv4, Hiera, FastViT.
        - yolo-v3, yolo-v8.
        - Segment-Anything Model (SAM).
        - SegFormer.
- File formats: load models from safetensors, npz, ggml, or PyTorch files.
- Serverless (on CPU), small and fast deployments.
- Quantization support using the llama.cpp quantized types.

<!--- ANCHOR_END: features --->

## How to use

<!--- ANCHOR: cheatsheet --->
Cheatsheet:

|            | Using PyTorch                            | Using ML                                                     |
|------------|------------------------------------------|------------------------------------------------------------------|
| Creation   | `torch.Tensor([[1, 2], [3, 4]])`         | `Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?`           |
| Creation   | `torch.zeros((2, 2))`                    | `Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?`               |
| Indexing   | `tensor[:, :4]`                          | `tensor.i((.., ..4))?`                                           |
| Operations | `tensor.view((2, 2))`                    | `tensor.reshape((2, 2))?`                                        |
| Operations | `a.matmul(b)`                            | `a.matmul(&b)?`                                                  |
| Arithmetic | `a + b`                                  | `&a + &b`                                                        |
| Device     | `tensor.to(device="cuda")`               | `tensor.to_device(&Device::new_cuda(0)?)?`                            |
| Dtype      | `tensor.to(dtype=torch.float16)`         | `tensor.to_dtype(&DType::F16)?`                                  |
| Saving     | `torch.save({"A": A}, "model.bin")`      | `hanzo::safetensors::save(&HashMap::from([("A", A)]), "model.safetensors")?` |
| Loading    | `weights = torch.load("model.bin")`      | `hanzo::safetensors::load("model.safetensors", &device)`        |

<!--- ANCHOR_END: cheatsheet --->


## Structure

- [hanzo-ml-core](./hanzo-ml-core): Core ops, devices, and `Tensor` struct definition
- [hanzo-ml-nn](./hanzo-ml-nn/): Tools to build real models
- [hanzo-ml-examples](./hanzo-ml-examples/): Examples of using the library in realistic settings
- [hanzo-ml-kernels](./hanzo-ml-kernels/): CUDA custom kernels
- [hanzo-ml-datasets](./hanzo-ml-datasets/): Datasets and data loaders.
- [hanzo-ml-transformers](./hanzo-ml-transformers): transformers-related utilities.
- [hanzo-ml-flash-attn](./hanzo-ml-flash-attn): Flash attention v2 layer.
- [hanzo-ml-onnx](./hanzo-ml-onnx/): ONNX model evaluation.

## FAQ

### Why should I use ML?

<!--- ANCHOR: goals --->

ML's core goal is to *make serverless inference possible*. Full machine learning frameworks like PyTorch
are very large, which makes creating instances on a cluster slow. ML allows deployment of lightweight
binaries.

Secondly, ML lets you *remove Python* from production workloads. Python overhead can seriously hurt performance,
and the [GIL](https://www.backblaze.com/blog/the-python-gil-past-present-and-future/) is a notorious source of headaches.

Finally, Rust is cool! A lot of the HF ecosystem already has Rust crates, like [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers).

<!--- ANCHOR_END: goals --->

### Other ML frameworks

- [dfdx](https://github.com/coreylowman/dfdx) is a formidable crate, with shapes being included
  in types. This prevents a lot of headaches by getting the compiler to complain about shape mismatches right off the bat.
  However, we found that some features still require nightly, and writing code can be a bit daunting for non rust experts.

  We're leveraging and contributing to other core crates for the runtime so hopefully both crates can benefit from each
  other.

- [burn](https://github.com/burn-rs/burn) is a general crate that can leverage multiple backends so you can choose the best
  engine for your workload.

- [tch-rs](https://github.com/LaurentMazare/tch-rs.git) Bindings to the torch library in Rust. Extremely versatile, but they 
  bring in the entire torch library into the runtime. The main contributor of `tch-rs` is also involved in the development
  of `hanzo`.

### Common Errors

#### Missing symbols when compiling with the mkl feature.

If you get some missing symbols when compiling binaries/tests using the mkl
or accelerate features, e.g. for mkl you get:
```
  = note: /usr/bin/ld: (....o): in function `blas::sgemm':
          .../blas-0.22.0/src/lib.rs:1944: undefined reference to `sgemm_' collect2: error: ld returned 1 exit status

  = note: some `extern` functions couldn't be found; some native libraries may need to be installed or have their path specified
  = note: use the `-l` flag to specify native libraries to link
  = note: use the `cargo:rustc-link-lib` directive to specify the native libraries to link with Cargo
```
or for accelerate:
```
Undefined symbols for architecture arm64:
            "_dgemm_", referenced from:
                hanzo_core::accelerate::dgemm::h1b71a038552bcabe in libhanzo_core...
            "_sgemm_", referenced from:
                hanzo_core::accelerate::sgemm::h2cf21c592cba3c47 in libhanzo_core...
          ld: symbol(s) not found for architecture arm64
```

This is likely due to a missing linker flag that was needed to enable the mkl library. You
can try adding the following for mkl at the top of your binary:
```rust
extern crate intel_mkl_src;
```
or for accelerate:
```rust
extern crate accelerate_src;
```

#### Cannot run the LLaMA examples: access to source requires login credentials

```
Error: request error: https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/tokenizer.json: status code 401
```

This is likely because you're not permissioned for the LLaMA-v2 model. To fix
this, you have to register on the huggingface-hub, accept the [LLaMA-v2 model
conditions](https://huggingface.co/meta-llama/Llama-2-7b-hf), and set up your
authentication token. See issue
[#350](https://github.com/huggingface/hanzo/issues/350) for more details.

#### Missing cute/cutlass headers when compiling flash-attn

```
  In file included from kernels/flash_fwd_launch_template.h:11:0,
                   from kernels/flash_fwd_hdim224_fp16_sm80.cu:5:
  kernels/flash_fwd_kernel.h:8:10: fatal error: cute/algorithm/copy.hpp: No such file or directory
   #include <cute/algorithm/copy.hpp>
            ^~~~~~~~~~~~~~~~~~~~~~~~~
  compilation terminated.
  Error: nvcc error while compiling:
```
[cutlass](https://github.com/NVIDIA/cutlass) is provided as a git submodule so you may want to run the following command to check it in properly.
```bash
git submodule update --init
```

#### Compiling with flash-attention fails

```
/usr/include/c++/11/bits/std_function.h:530:146: error: parameter packs not expanded with ‘...’:
```

This is a bug in gcc-11 triggered by the Cuda compiler. To fix this, install a different, supported gcc version - for example gcc-10, and specify the path to the compiler in the NVCC_CCBIN environment variable.
```
env NVCC_CCBIN=/usr/lib/gcc/x86_64-linux-gnu/10 cargo ...
```

#### Linking error on windows when running rustdoc or mdbook tests

```
Couldn't compile the test.
---- .\hanzo-ml-book\src\inference\hub.md - Using_the_hub::Using_in_a_real_model_ (line 50) stdout ----
error: linking with `link.exe` failed: exit code: 1181
//very long chain of linking
 = note: LINK : fatal error LNK1181: cannot open input file 'windows.0.48.5.lib'
```

Make sure you link all native libraries that might be located outside a project target, e.g., to run mdbook tests, you should run:

```
mdbook test hanzo-ml-book -L .\target\debug\deps\ `
-L native=$env:USERPROFILE\.cargo\registry\src\index.crates.io-6f17d22bba15001f\windows_x86_64_msvc-0.42.2\lib `
-L native=$env:USERPROFILE\.cargo\registry\src\index.crates.io-6f17d22bba15001f\windows_x86_64_msvc-0.48.5\lib
```

#### Extremely slow model load time with WSL

This may be caused by the models being loaded from `/mnt/c`, more details on
[stackoverflow](https://stackoverflow.com/questions/68972448/why-is-wsl-extremely-slow-when-compared-with-native-windows-npm-yarn-processing).

#### Tracking down errors

You can set `RUST_BACKTRACE=1` to be provided with backtraces when a hanzo
error is generated.

#### CudaRC error

If you encounter an error like this one `called `Result::unwrap()` on an `Err` value: LoadLibraryExW { source: Os { code: 126, kind: Uncategorized, message: "The specified module could not be found." } }` on windows. To fix copy and rename these 3 files (make sure they are in path). The paths depend on your cuda version.
`c:\Windows\System32\nvcuda.dll` -> `cuda.dll`
`c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cublas64_12.dll` -> `cublas.dll`
`c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\curand64_10.dll` -> `curand.dll`
