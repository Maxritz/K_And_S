## Actual findings from cloning tiny-vllm (2026-10-25)

> These are the real findings from inspecting the cloned repo. The port is not yet attempted — these
> are the prerequisites and challenges identified before starting.

### What tiny-vllm actually is

- **llama.cpp-style C++/CUDA inference engine**, but written as a **course** — the repo is a
  teaching resource where the author implements a minimal vLLM-like inference engine step by step.
- **Model**: Llama 3.2 1B Instruct, loaded from Safetensors (`model.safetensors` file, not downloaded
  with the repo — you need to obtain it separately).
- **Architecture**: LLaMA-style with GQA (Grouped Query Attention), 16 layers, 2048 embedding dim,
  8192 hidden dim, 512 KV dim, 64 head dim, 32 Q heads, 8 K heads, 8 V heads, 128256 vocab size.
- **BFloat16**: uses `__nv_bfloat16` throughout (CUDA/HIP BF16 type). All weights and computation
  are in BF16.
- **KV cache**: 2GB preallocated, paged attention with 16-token blocks.
- **Batching**: static + continuous batching (queue-based).
- **Compute**: cuBLAS `cublasGemmEx` for all matmuls (Q, K, V, O, gate, up, down, embed proj).
  Custom CUDA kernels for: embedding gather, RMSNorm, RoPE, softmax, attention score computation,
  paged attention.

### File structure

```
tiny-vllm/
├── CMakeLists.txt      # CMake build, CUDA or HIP, needs CUDA toolkit or ROCm
├── README.md           # Course material + intro
├── build.sh            # Build script
├── run.sh              # Run script (runs ./build/tiny-vllm)
├── check.sh            # Check script
├── full_test.sh        # Full test
├── ncu.sh              # NVIDIA Compute Profiler
├── nsys.sh             # NSys profiler
├── reference.txt       # Reference output for verification
├── python/             # Python helpers (likely tokenization, model download, etc.)
├── src/
│   ├── main.cpp        # Main inference loop, weight loading, prefill, decode, batching
│   ├── kernels.cu      # CUDA kernels: embedding, RMSNorm, RoPE, softmax, attention, paged attn
│   └── kernels.cuh     # Kernel declarations
├── include/
│   └── json.hpp        # nlohmann/json single-header (25830 lines!)
└── tests/
    └── test_softmax.cu # Softmax test kernel
```

### Key findings for CPU porting

#### 1. BFloat16 is everywhere

- The entire codebase uses `__nv_bfloat16` (CUDA/HIP BF16 type). This is a CUDA/HIP-specific type.
- For CPU porting, you need a BF16 implementation on CPU. Options:
  - **Intel AMX / AVX-512 BF16**: Intel 4th Gen Xeon+ and some consumer CPUs have BF16 support via AVX-512_BF16. But consumer Intel (10th-14th Gen) doesn't have BF16.
  - **ARM NEON**: ARMv8.2+ has BF16 instructions. If targeting ARM CPU, BF16 is native.
  - **Software BF16**: you can implement BF16 as a struct with conversion to/from FP32. Slower, but works on any CPU.
  - **Use FP32 instead**: simplest — load BF16 weights, convert to FP32 on load, run FP32 inference. Loses the memory/energy advantage of BF16 but is simplest and most portable.

#### 2. cuBLAS is used for all matmuls

- `cublasGemmEx` with `CUDA_R_16BF` (BF16) and `CUBLAS_COMPUTE_32F` (compute in FP32).
- For CPU porting, you need a CPU BLAS library that supports BF16 (or use FP32):
  - **Intel MKL**: supports BF16 via AVX-512_BF16 (4th Gen Xeon+). On consumer Intel (10th-14th Gen), MKL falls back to FP32 or uses AVX2 for FP32.
  - **OpenBLAS**: may support BF16 on some archs. Check availability.
  - **Eigen**: supports BF16 via `Eigen::bfloat16` (since Eigen 3.4). Can use for BF16 matmul on CPU, but slower than MKL/OpenBLAS.
  - **OneDNN (MKL-DNN)**: Intel's deep learning library, supports BF16 on supported CPUs. More complex to integrate.
  - **FP32 fallback**: use any BLAS library (MKL, OpenBLAS, Eigen) with FP32. Simplest if you convert weights to FP32 on load.

#### 3. Custom CUDA kernels need CPU equivalents

- **embeddingGatherKernel**: gather embedding rows by token ID. On CPU: just a loop or a gather operation. Trivial to implement in C++ (no GPU needed — it's a memory lookup).
- **rmsNormKernel**: RMSNorm with tree reduction. On CPU: compute RMS over the embedding dim, then scale. Can use SIMD (AVX2/AVX-512) for the dot product, or just a loop. Trivial.
- **RoPE kernel**: rotary position embedding. On CPU: compute sin/cos tables (or precompute), apply rotation to Q and K. Some math, but straightforward.
- **Softmax kernel**: online softmax (likely). On CPU: implement softmax with max subtraction for numerical stability. Straightforward.
- **Attention kernel**: likely computes Q*K^T / sqrt(d) + causal mask, then softmax, then * V. On CPU: matrix multiply + softmax + matmul. Use BLAS for the matmuls, implement softmax in C++.
- **Paged attention kernel**: PagedAttention (vLLM's key innovation) — handles non-contiguous KV cache blocks. On CPU: you can avoid PagedAttention entirely — just use a contiguous KV cache buffer (no paging needed on CPU since CPU memory isn't fragmented like GPU VRAM). This is a significant simplification for CPU.

#### 4. PagedAttention is a GPU memory optimization — not needed on CPU

- PagedAttention exists because GPU VRAM is fragmented and limited. On CPU with plenty of RAM,
  you don't need paging — you can just allocate a contiguous KV cache that grows with sequence length.
- **Porting simplification**: drop PagedAttention for CPU. Use a simple contiguous KV cache. This
  removes a significant chunk of complexity (paged attention kernel, block table management, etc.).

#### 5. The model is hardcoded for Llama 3.2 1B

- All dimensions are `constexpr` hardcoded in main.cpp and kernels.cu:
  - `N_LAYERS = 16`
  - `EMBEDDING_LENGTH = 2048`
  - `HIDDEN_DIM = 8192`
  - `KV_DIM = 512`
  - `HEAD_DIM = 64`
  - `NUM_Q_HEADS = 32`, `NUM_K_HEADS = 8`, `NUM_V_HEADS = 8`
  - `VOCAB_SIZE = 128256`
  - `MAX_SEQ_LEN = 2048`
- For CPU porting, this is fine — you can keep the same hardcoded dimensions. The model is small
  (1B params), so it fits in CPU RAM easily (even BF16: 1B * 2 bytes = 2GB; FP32: 4GB).

#### 6. The `model.safetensors` file is NOT in the repo

- The CMakeLists.txt and main.cpp expect a `model.safetensors` file in the working directory.
- This file is not included in the repo (it's the actual model weights — too large for git).
- You need to obtain the Llama 3.2 1B Instruct safetensors file from HuggingFace.
- **Porting prerequisite**: download `meta-llama/Llama-3.2-1B-Instruct` safetensors from HuggingFace.
  - URL: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main
  - The safetensors file is `model.safetensors` (likely a few GB for 1B params in BF16).

#### 7. Tokenizer

- The README mentions tokenization as a section. Need to check the `python/` directory for tokenizer
  code. Likely uses HuggingFace tokenizers or sentencepiece.
- For CPU C++ port: need a C++ tokenizer. If the tokenizer is LLaMA-style (BPE with sentencepiece),
  use sentencepiece C++ API. If it's tiktoken (GPT-4 style), need a C++ tiktoken implementation.
- **Need to inspect**: what tokenizer does tiny-vllm actually use? Check `python/` dir.

#### 8. HIP port already exists in the codebase!

- The CMakeLists.txt has `USE_HIP` option — you can build with HIP for AMD GPUs.
- `cuda_to_hip.h` header — likely a CUDA-to-HIP portability layer.
- This is relevant: if tiny-vllm already has a HIP port, then the kernels are written in a
  CUDA/HIP-compatible way. For CPU porting, you'd still need to convert the CUDA kernels to CPU
  code, but the HIP port shows the kernels can be written in a portable way.
- **Interesting angle**: if the HIP port works, you could potentially run tiny-vllm on an AMD GPU
  (like RX 7900 XT, gfx1100) via HIP. But the user asked for CPU, not AMD GPU.

#### 9. Dependencies

- **CUDA toolkit** (or ROCm for HIP): required for the original build. CUDA::cublas, CUDA::cudart.
- **nlohmann/json**: single-header, included in `include/json.hpp` (25830 lines). No external dependency.
- **Standard C++17**: the codebase is C++17.
- For CPU porting: replace CUDA toolkit with a CPU BLAS library (MKL, OpenBLAS, Eigen). Keep nlohmann/json. Add a BF16 implementation if keeping BF16.

### Specific challenges / errors we'll likely hit

1. **No `model.safetensors`**: the repo doesn't include the model. Need to download from HuggingFace.
   - **Error**: "Can't open model.safetensors file" — exactly what main.cpp prints if the file is missing.
   - **Fix**: download `meta-llama/Llama-3.2-1B-Instruct` safetensors. Need HuggingFace account? Check if Llama 3.2 1B requires accepting terms.

2. **BF16 on CPU**: `__nv_bfloat16` is CUDA/HIP-specific. No standard CPU equivalent.
   - **Error**: `__nv_bfloat16` not available on CPU. Compilation error.
   - **Fix**: replace `__nv_bfloat16` with a CPU BF16 implementation (struct + conversion operators), or convert to FP32 on load.

3. **cuBLAS not available on CPU**: `cublasGemmEx` is CUDA-only.
   - **Error**: CUDA::cublas not found when building without CUDA.
   - **Fix**: replace with CPU BLAS (MKL `cblas_`, OpenBLAS `openblas_`, or Eigen matmul).

4. **CUDA kernels not runnable on CPU**: `embeddingGatherKernel`, `rmsNormKernel`, etc. are `__global__` CUDA kernels.
   - **Error**: `__global__` functions don't compile for CPU.
   - **Fix**: rewrite each kernel as a CPU function. Most are straightforward (embedding gather = lookup loop, RMSNorm = dot product + scale, etc.).

5. **PagedAttention on CPU**: complex kernel + block table management.
   - **Simplification**: drop PagedAttention for CPU. Use contiguous KV cache.

6. **Tokenizer**: need to identify which tokenizer tiny-vllm uses, then find a C++ implementation.
   - **Error**: if using sentencepiece C++ API, need to build sentencepiece on Windows with MSVC.
   - **Fix**: build sentencepiece with CMake/MSVC, or use a prebuilt binary, or use HuggingFace tokenizers C++ API, or implement a simple BPE tokenizer if the model uses a simple scheme.

7. **Continuous batching**: the code has a queue-based continuous batching system.
   - **CPU simplification**: for CPU-only single-request inference, you can drop continuous batching. Just process one prompt at a time. Much simpler.

8. **Windows-specific**: the codebase is developed on Linux. Need to check for Linux-specific code.
   - **Potential issues**: file paths (Linux `/` vs Windows `\`), thread APIs (pthread vs Windows threads), library linking (Linux .so vs Windows .dll/.lib).
   - **Fix**: use C++17 `std::filesystem` for paths (cross-platform). Use OpenMP for threading (cross-platform with MSVC). Link BLAS as Windows DLL or static lib.

### Porting strategy for CPU (concrete plan)

**Phase 0: Baseline**
- Get the original tiny-vllm running on GPU (if CUDA available) to get reference output.
- Or use the `reference.txt` file in the repo for expected output.
- Identify the tokenizer and model format precisely.

**Phase 1: Weight loading on CPU**
- Replace CUDA `cudaMalloc` + `cudaMemcpy` with CPU `malloc` / `new` / `std::vector`.
- Read safetensors file on CPU (parse header, read weights into CPU memory).
- Store weights as BF16 array on CPU (or convert to FP32 on load).

**Phase 2: Replace CUDA kernels with CPU implementations**
- `embeddingGather` → CPU loop: for each token, copy embedding row from weights to input buffer.
- `rmsNorm` → CPU: compute RMS over embedding dim, scale and multiply by norm weights.
- `RoPE` → CPU: apply rotation to Q and K using precomputed sin/cos tables.
- `softmax` → CPU: max-subtract softmax for numerical stability.
- `attention` → CPU: Q*K^T (BLAS), apply causal mask, softmax, * V (BLAS).
- Drop PagedAttention → contiguous KV cache.

**Phase 3: Replace cuBLAS with CPU BLAS**
- `cublasGemmEx(CUDA_R_16BF, CUBLAS_COMPUTE_32F)` → CPU BLAS matmul with BF16 or FP32.
- If using Eigen: `Eigen::bfloat16` matmul, or convert to FP32 and use `Eigen::MatrixXf`.
- If using MKL: `cblas_sgemm` for FP32, or BF16 variants if available.

**Phase 4: Tokenizer**
- Identify tokenizer (check `python/` dir).
- Implement or link C++ tokenizer (sentencepiece, tiktoken-cpp, HuggingFace tokenizers).
- Verify tokenizer output matches Python reference.

**Phase 5: Generation loop**
- Implement prefill (process prompt tokens, fill KV cache).
- Implement decode (generate one token at a time, append to KV cache).
- Implement sampling (greedy, top-k, top-p, temperature).
- Verify output matches `reference.txt` or GPU reference.

**Phase 6: Simplify**
- Drop continuous batching (single-request CPU inference).
- Drop PagedAttention (contiguous KV cache).
- Consider FP32 instead of BF16 if BF16 CPU support is limited.

### What makes this port hard vs easy

**Easy**:
- The model is small (1B params) — fits in CPU RAM easily.
- The architecture is standard LLaMA with GQA — well-documented, many CPU implementations exist.
- No custom exotic ops — just embedding, RMSNorm, RoPE, attention, SiLU, MLP, softmax. All standard.
- PagedAttention can be dropped for CPU (contiguous KV cache).
- Continuous batching can be dropped for single-request CPU inference.
- The kernels are mostly simple (embedding gather, RMSNorm, softmax) — easy to reimplement on CPU.

**Hard**:
- BF16 support on CPU is limited (no BF16 on consumer Intel 10th-14th Gen, no `__nv_bfloat16` on CPU).
- cuBLAS replacement needs a CPU BLAS library with BF16 support (or FP32 fallback).
- Tokenizer needs to match Python reference exactly.
- The codebase is a course — it's intentionally incomplete in places (TODOs everywhere). This makes it
  realistic for learning, but means you need to fill in gaps.
- `reference.txt` may or may not be complete — need to verify against actual GPU output if possible.

### Alternative: use llama.cpp as the inference engine

- **llama.cpp** already implements LLaMA inference on CPU with GGUF format.
- If you convert the Llama 3.2 1B safetensors to GGUF, you can run it with llama.cpp on CPU.
- llama.cpp supports BF16, FP16, INT8, INT4 quantization.
- llama.cpp has a C API (libllama) that you can call from your own C++ code.
- This avoids reimplementing the entire inference engine — you just call llama.cpp's API.
- **Trade-off**: you don't learn how the inference engine works internally (the point of tiny-vllm
  as a course). But you get a working CPU inference engine quickly.

### Alternative: ONNX Runtime on CPU

- Export the model from PyTorch to ONNX (if you can load it in PyTorch first).
- Run ONNX Runtime on CPU in C++.
- ONNX Runtime has optimized CPU kernels (MKL, oneDNN).
- **Trade-off**: you need to get the model into PyTorch first to export to ONNX. If the repo doesn't
  include a PyTorch loading script, you need to write one. Then ONNX export may or may not work
  (dynamic shapes, custom ops, etc.).

---

## Actual next steps

1. **Download `model.safetensors`** from HuggingFace (`meta-llama/Llama-3.2-1B-Instruct`).
   - Check if it requires accepting terms on HuggingFace.
   - The file is ~2GB for BF16 1B model. Need to obtain it.

2. **Inspect `python/` directory** to understand the tokenizer and any Python helpers.
   - `ls tiny-vllm/python/` — see what's there.
   - Check if there's a tokenizer file or Python tokenization script.

3. **Try building the original tiny-vllm** with CUDA (if CUDA is available in this environment).
   - `cd tiny-vllm && bash build.sh` — see if it builds.
   - If CUDA is not available, this confirms we need the CPU port.

4. **Check `reference.txt`** for expected output.
   - `cat tiny-vllm/reference.txt` — see what the expected output looks like.

5. **Decide on approach**:
   - **Full reimplementation**: rewrite kernels in C++ CPU, use CPU BLAS, implement tokenizer.
   - **llama.cpp**: convert safetensors to GGUF, use llama.cpp C API.
   - **ONNX Runtime**: export to ONNX from PyTorch, run with ONNX Runtime CPU.

6. **Start the C++ CPU port** with the chosen approach.

---

## CPU port feasibility assessment

| Aspect | CPU port feasibility | Notes |
|---|---|---|
| Weight loading (safetensors) | Easy | Parse safetensors header + read weights. C++ can do this. Need to obtain the file first. |
| BF16 on CPU | Medium | No native BF16 on consumer Intel. Options: software BF16 (slower), FP32 fallback (simplest), or use ARM CPU with NEON BF16. |
| cuBLAS replacement | Medium | Need CPU BLAS with BF16 or FP32 fallback. MKL/OpenBLAS/Eigen available. |
| Custom CUDA kernels → CPU | Easy-Medium | Most kernels are simple (embedding, RMSNorm, softmax). Attention is the most complex but still doable. |
| PagedAttention | Easy to drop | Not needed on CPU. Use contiguous KV cache. |
| Continuous batching | Easy to drop | Single-request CPU inference is simpler. |
| Tokenizer | Medium | Need to identify tokenizer and find C++ implementation. Sentencepiece C++ is the most likely candidate. |
| Windows build | Medium | Need to handle Windows paths, library linking, threading. C++17 `std::filesystem` + OpenMP helps. |
| Model size (1B) | Easy | Fits in CPU RAM easily. BF16: 2GB. FP32: 4GB. |
| Reference output | Unknown | `reference.txt` exists but need to verify it's complete and matches the model. |

**Overall**: CPU port is feasible. The main challenges are BF16 on CPU and tokenizer matching. The model is small enough that FP32 CPU inference is realistic (if slower than GPU). The easiest path to a working CPU inference is probably llama.cpp (convert to GGUF, use existing CPU kernels), but that defeats the learning purpose of tiny-vllm as a course. A from-scratch CPU reimplementation is more educational but more work.

