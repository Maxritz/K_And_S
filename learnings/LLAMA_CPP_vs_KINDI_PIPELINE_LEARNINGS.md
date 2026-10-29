# llama.cpp vs Kindi Kepler — Pipeline Architecture Comparison
> Generic reference flow. Source: kands analyze_full_pipeline + analyze_core_engine on both trees.
> Evidence: PASS = matched patterns, NOT FOUND = no markers in source.

## 1. Dispatch Strategy

| Engine | Dispatch model | Key difference |
|---|---|---|
| **Kindi (Kepler)** | Python host (engine.py/tp.py) launches CUDA kernels via CuPy `<<<>>>` at per-token or batched-token granularity. `wrappers()` generates `_T` variants at runtime via NVRTC. | Host-side JIT recompilation per variant; `blockIdx.y = token` for batched prompts (chunk.py/chunk1.py). |
| **llama.cpp CUDA** | C++ host (`ggml-cuda.cu`) has a **dispatch table** (`kernel_dispatch`) mapping op types to kernel launchers. One `.cu` file per op (284 files). | Static dispatch table at compile time; no runtime JIT. Per-op `.cu` files. |
| **llama.cpp Vulkan** | C++ host (`ggml-vulkan.cpp`) dispatches via `vkCmdDispatch` with compute pipelines loaded from `.comp` shaders. | Shader pipeline state objects; no PTX/HIP. 196 `.comp` files + feature-tests/. |
| **llama.cpp CPU** | C host (`ggml-cpu.c`) dispatches via function pointer table (`ggml_cpu_pool`) with arch-specific SIMD (AVX2/AVX-512/NEON/SVE). | 31 source files; OpenMP parallelism; no GPU kernel dispatch. |

## 2. Kernel Count & Granularity

| Metric | Kindi | llama.cpp CUDA | llama.cpp Vulkan | llama.cpp CPU |
|---|---|---|---|---|
| Files scanned | 30 | 284 | 196 | 31 |
| Kernel entry points | 77 | 166 | (in .comp shaders) | (inline SIMD) |
| Kernel launches from host | 16 | 188 | 54 | 0 (host calls) |

**Key difference:** llama.cpp explodes one logical op into many template-instantiated files (.cu per op for CUDA, .comp per op+config for Vulkan). Kindi keeps kernels in 4 .cu files with runtime parameterization via NVRTC.

## 3. Pipeline Stages Present

| Stage | Kindi | llama.cpp CUDA | llama.cpp Vulkan | Notes |
|---|---|---|---|---|
| engine_init | PASS (22) | PASS (1) | PASS (7) | PASS (1) | llama.cpp Vulkan has richer init (VkInstance, device discovery); CUDA/CPU minimal |
| model_load | PASS (7) | PASS (2) | PASS (1) | PASS (7) | Both use GGUF; Kindi has JIT, llama.cpp static mmap |
| weight_repack | PASS (54) | PASS (127) | PASS (103) | PASS (154) | llama.cpp has extensive dequant paths; CPU has per-arch quants.c |
| prompt | PASS (109) | PASS (3) | **NOT FOUND** | PASS (4) | Vulkan: prompt handled by same decode dispatch, no separate marker |
| decode_loop | PASS (91) | PASS (55) | PASS (145) | PASS (47) | Vulkan dispatches more broadly (graph-based) |
| attention | PASS (77) | PASS (137) | PASS (56) | PASS (36) | llama.cpp CUDA: flash-attn variants (fattn-*); Vulkan: flash_attn_decode phases |
| kv_cache | PASS (13) | PASS (12) | PASS (9) | PASS (2) | Kindi uses q8_0 cache, llama uses ggml tensor layout |
| gemm | PASS (144) | PASS (234) | PASS (310) | PASS (360) | Vulkan: subgroup ops; CPU: AVX2/AVX-512 SIMD loops |
| normalization | PASS (88) | PASS (120) | PASS (97) | PASS (73) | All RMS norm; CPU uses AVX reduction |
| ffn | PASS (47) | PASS (316) | PASS (141) | PASS (98) | Kindi: fused SwiGLU; llama.cpp: separate op files; CPU: SIMD-optimized |
| gdn_ssm | PASS (109) | PASS (83) | PASS (37) | PASS (11) | Kindi: GDN (gated delta network); llama.cpp: gated_delta_net |
| speculative | PASS (50) | PASS (4) | **NOT FOUND** | PASS (1) | Vulkan: not implemented; CPU/CUDA: minimal |
| tensor_parallel | PASS (59) | PASS (104) | PASS (132) | PASS (33) | llama.cpp: NCCL-style allreduce; CPU: OpenMP-based |
| output | PASS (46) | PASS (165) | PASS (56) | PASS (41) | Output projection + tied head |
| sampling | PASS (88) | PASS (255) | PASS (52) | PASS (7) | llama.cpp CUDA: extensive top-k/top-p; CPU: basic |

## 4. Constructs Used

| Construct | Kindi | llama.cpp CUDA | llama.cpp Vulkan | Porting relevance |
|---|---|---|---|---|
| `kernel_entry` | `__global__` | `__global__` | shader entry points | Same semantic |
| `shfl_sync` | `__shfl_xor_sync` | `__shfl_xor_sync` | — | Vulkan uses subgroup ops instead |
| `mma.sync` | Present (fp16) | Present | — | Vulkan: no tensor cores → scalar FMA or DP4a |
| `dp4a` | Present (sm_50+ path) | Present | — | Vulkan: `spv_dp4` via GL_KHR_integer_dot_product |
| `atomic` | Present | Present | Present | Cross-ecosystem: `atomicAdd` / `OpAtomicAdd` |
| `local_memory` | `__shared__` | `__shared__` | `shared` / `local_accessor` | Spelling differs |
| `cp.async` | Present | Present | — | Vulkan: manual global→shared store + barrier |
| `cuda_arch_macro` | `__CUDA_ARCH__` | `__CUDA_ARCH__` | — | Vulkan: `GL_*` / `VK_*` defines |
| `sync` | `__syncthreads` | `__syncthreads` | `barrier` / `group_barrier` | Portable pattern |
| `ballot_sync` | — | Present | — | CUDA-only, not on Kepler |
| `cooperative_groups` | — | Present | — | CUDA-specific |
| `dynamic_parallelism` | — | Present | — | CUDA-only: `<<<>>>` device-side |
| `syncwarp` | — | Present | — | CUDA-only: no-op on AMD |
| `wg_barrier` | — | — | Present | Vulkan: `groupBarrier()` |
| `global_id` | — | — | Present | Vulkan: `gl_GlobalInvocationID` |
| `subgroup_ops` | — | — | Present | Vulkan: `subgroupShuffle`, `subgroupAdd` |

**CPU-only constructs:** llama.cpp CPU backend uses inline SIMD intrinsics (`_mm256_*`, `__m256i`, `__m512i`, `neon`, `sve`) instead of GPU kernels. No `kernel_entry`, `local_memory`, `shfl_sync`, or `mma.sync` — all compute is via CPU SIMD loops with OpenMP parallelism.

## 5. Porting: Kindi Kepler → Other Backends

### To CUDA→Vulkan (SPIR-V)
- `mma.sync` → scalar FMA chains or `spv_dp4` (GL_KHR_integer_dot_product)
- `__shfl_xor_sync` → `subgroupShuffle` (SPIR-V subgroup ops)
- `__ldg` → plain load (no read-only data cache in SPIR-V)
- `KLDG` macro (sm_30 ldg fallback) → plain deref
- `dp4a` → no direct equivalent; use `OpDot` with int vectors (SPIR-V 1.5+ dot product)
- `<<<>>>` launch → `vkCmdDispatch`
- ReBAR / pinned host → Vulkan host-visible memory or staging buffers

### To CUDA→HIP (RDNA)
- `__global__` → `extern "C" __global__`
- `__shfl_xor_sync` → `__shfl_xor` (mask dropped, HIP handles)
- `__ldg` → `__ldg` (supported) or plain load
- `dp4a` → `amd_mixed_dot` builtin (encodes on gfx1201)
- `KLDG` macro → conditional on `__HIP__
- mma → wmma (rocWMMA) or scalar reconstruction

## 6. What llama.cpp Vulkan Does That Kindi Doesn't

1. **Subgroup operations for GEMM** — uses `subgroupAdd`, `subgroupShuffle` for reductions instead of warp shuffles.
2. **Shader storage buffer objects (SSBO)** — all tensors are SSBOs in Vulkan, vs Kindi's global memory.
3. **Pipeline layout state objects** — Vulkan pre-compiles pipeline state; Kindi uses runtime kernel selection.
4. **Descriptor sets** — Vulkan binds resources via descriptor sets; Kindi uses raw kernel arguments.
5. **No speculative decode** — Vulkan backend skips speculative verification (NOT FOUND in scan).

## 7. What Kindi Does That llama.cpp Doesn't

1. **5-trit ternary weights** — `k_g5n` layout with 5-trit lane packing (no llama.cpp equivalent).
2. **GDN (Gated Delta Network)** — recurrent convolution + delta state update (`k_gdn_pre`, `k_gdn_delta`).
3. **Hadamard folding** — `k_rmsfold` / `k_smfold` fuses Hadamard transform into RMS norm.
4. **Two-card tensor parallel with host-memory swap** — TPEngine uses pinned host memory for partial sums.
5. **Runtime JIT compilation** — Kindi compiles kernels at runtime via NVRTC/CuPy; llama.cpp links at build time.
6. **Speculative verify+commit** — `spec.cu` with wb=0 verify pass + commit-first-n-kernels.
7. **fp64 norm sums** — fp64 accumulator for exact bit-identical RMS norm.

## 8. Pipeline Flow: Kindi vs llama.cpp

```
KINDI (Kepler, single GPU or 2-card TPEngine):
  CPU/Python host                    GPU
  ├─ wrapper() generates _T variants (NVRTC JIT)
  ├─ k_xrms  (fp64 RMS norm, Hadamard fold)
  ├─ k_gdn_pre / k_gdn_delta (if GDN layer)
  ├─ mm_core {g4_unpack | g5n_unpack} + fp32 GEMM (no dp4a on sm_30)
  ├─ k_attn_flash16 / k_attn_sq8 (q8_0 scores)
  ├─ k_store_q8 (KV cache as q8_0)
  ├─ spec.cu verify (wb=0) → commit if speculative
  └─ next token

LLAMA.CPP CUDA:
  C++ host                          GPU
  ├─ dispatch_table[OP_TYPE] → launcher fn
  ├─ per-op .cu files (166 kernel entries)
  ├─ norm.cu (RMS norm)
  ├─ mmq.cu / mmf.cu (GEMM, with MMVQ for vectors)
  ├─ fattn-* (flash attention, template-instantiated)
  ├─ rope.cu (RoPE)
  ├─ convert.cu (dequantize weights)
  └─ top-k.cu / topk-moe.cu (sampling)

LLAMA.CPP VULKAN:
  C++ host                         GPU (SPIR-V)
  ├─ vkCmdBindPipeline (compute)
  ├─ dispatch via VkDescriptorSet
  ├─ .comp shaders (196 files)
  ├─ subgroup ops for reductions
  ├─ SSBO for all tensors
  └─ no speculative, no flash-attn variants

LLAMA.CPP CPU:
  C host                           CPU (SIMD)
  ├─ ggml_cpu_pool[OP_TYPE] → fn ptr
  ├─ per-op SIMD (AVX2/AVX-512/NEON/SVE)
  ├─ ggml-cpu.c dispatch (31 files)
  ├─ #pragma omp parallel for
  ├─ inline _mm256_* intrinsics
  └─ no GPU kernel dispatch at all
```

## 9. Dispatch Pattern Comparison

| Pattern | Kindi | llama.cpp CUDA | llama.cpp Vulkan | llama.cpp CPU |
|---|---|---|---|---|
| **Launch style** | `<<<<>>>` via CuPy | `<<<<>>>` / `cuLaunchKernel` | `vkCmdDispatch` | `ggml_cpu_pool` fn ptr |
| **Batched** | BLOCK (148 hits) | BLOCK (1080) | BLOCK (571) | BLOCK (370) |
| **Per-token** | 9 hits | 3 hits | 0 | 0 |
| **Persistent** | 2 hits | 1 hit | 0 | 1 hit |
| **Cooperative** | 0 | 10 hits | 93 hits | 0 |
| **Dynamic parallelism** | 0 | 175 hits | 0 | 0 |
| **Streamed** | 29 | 628 | 33 | 20 |
| **Host-mapped** | 23 | 26 | 18 | 0 |
| **Zero-copy** | 8 | 2 | 2 | 9 |

### Key Insights:

1. **llama.cpp CUDA uses dynamic parallelism** (175 hits) — device-side `<<<>>>` launches. Kindi does NOT use this; all launches are host-side. This is a porting constraint: dynamic parallelism requires CUDA and cannot be directly translated to HIP or Vulkan.

2. **llama.cpp Vulkan uses cooperative groups** (93 hits) — grid-wide sync via Vulkan barriers. Kindi has zero. Vulkan requires explicit `vkCmdPipelineBarrier` instead.

3. **llama.cpp CUDA has per-op files** (284 files) while Kindi has 4 `.cu` files. llama.cpp's approach is more parallelizable at compile time but harder to port (166 kernel entry points vs 77).

4. **llama.cpp CPU has zero GPU dispatch patterns** — everything is SIMD CPU compute. Uses custom allocator (28 hits) and OpenMP parallelism, not CUDA/Vulkan. The CPU backend shares pipeline stage markers with GPU backends because `ggml-cpu.c` implements the same `ggml_*` API surface.

## 10. Functional Pipeline Mapping — Stage to Device Function

This maps each pipeline stage to the actual device functions used across backends.
"Functions of devices" — each stage is a function, implemented differently per backend.

### Dispatch Model Comparison

| Stage | Kindi (Kepler) | llama.cpp CUDA | llama.cpp Vulkan | llama.cpp CPU |
|---|---|---|---|---|
| **Graph** | Imperative per-stage call from Python host | `ggml_cgraph` DAG, dispatched via `ggml_cuda_compute_forward()` switch | `ggml_cgraph` DAG, dispatched via `ggml_vk_compute_forward()` switch | `ggml_cgraph` DAG, dispatched via `ggml_cpu_pool` fn ptr table |
| **Dispatch granularity** | Per-token or batched (T tokens) | Per-graph-node (each GGML op) | Per-graph-node (each GGML op) | Per-graph-node (each GGML op) |
| **Dispatch location** | Python host (engine.py) | C++ host (ggml-cuda.cu:2262) | C++ host (ggml-vulkan.cpp:12941) | C host (ggml-cpu.c) |

### Stage-to-Function Mapping Table

| Pipeline Stage | GGML Op | Kindi CUDA Kernel | llama.cpp CUDA File | llama.cpp Vulkan Shader | llama.cpp CPU Function | Device Function |
|---|---|---|---|---|---|---|
| **Normalization** | `GGML_OP_NORM` / `RMS_NORM` | `k_xrms` / `k_xrmsN`, `k_rmsfold` | `norm.cu` (77KB) | `norm.comp` | `ggml_cpu_norm` | Reduce sum of squares, compute scale, apply |
| **Weight unpack** | `GGML_OP_MUL_MAT` (dequant) | `g5n_unpack` / `g4_unpack` in `mm_core` | `convert.cu` (34KB) | `dequant_*.comp` (many) | `ggml_cpu_get_rows` | Decode packed weights to fp32 |
| **GEMM/MatMul** | `GGML_OP_MUL_MAT` | `k_mm_g5n` / `k_mm_g4` (`mm_core`) | `mmq.cu` (64KB), `mmf.cu`, `mmvq.cu` | `mul_mmq.comp`, `mul_mm.comp` | `ggml_cpu_mul_mat` | Matrix multiply, K-loop over tiles |
| **Attention** | `GGML_OP_SOFT_MAX`, `GGML_OP_FLASH_ATTN_EXT` | `k_attn_flash16`, `k_attn_sq8` | `fattn.cu` (31KB), `fattn-tile-*.cu` | `flash_attn*.comp` (6 shaders) | `ggml_cpu_soft_max` | QKV matmul + softmax (flash = online) |
| **KV Cache Store** | (internal) | `k_store_q8` (q8_0 cache) | `cpy.cu` (27KB) | `copy.comp` | `ggml_cpu_cpy` | Copy/append KV to cache buffer |
| **RoPE** | `GGML_OP_ROPE` | `k_rope` | `rope.cu` (47KB) | Implicit in shader | `ggml_cpu_rope` | Apply rotary embedding |
| **Layer Add** | `GGML_OP_ADD` | `k_add` | `add-id.cu` | `add.comp`, `add1.comp` | `ggml_cpu_add` | Element-wise add |
| **Activation** | `GGML_OP_SILU` etc | (fused in GEMM) | `unary.cu` (29KB) | `unary.comp`, `silu_back.comp` | `ggml_cpu_unary` | SiLU, GELU, etc. |
| **FFN Gate** | `GGML_OP_MLP`, `GGML_OP_GLU` | `k_gdn_gates` (fused) | `gated_delta_net.cu` | `geglu*.comp` | `ggml_cpu_gated_delta_net` | Gated linear unit computation |
| **Sampling** | `GGML_OP_TOP_K`, `GGML_OP_TOP_P` | (CPU-side in engine.py) | `top-k.cu` (13KB), `topk-moe.cu` | `topk_*.comp` (4 shaders) | `ggml_cpu_top_k` | Select next token from logits |
| **Speculative** | (custom in spec.cu) | `k_sgdn_pre_T` (wb=0 verify) | No direct equivalent | NOT FOUND | No equivalent | Verify draft tokens |
| **Tensor Parallel** | (custom in tp.py) | `k_xrms` swaps partials via pinned host | `allreduce.cu` (46KB) | `add.comp` (partial sum) | `ggml_cpu_all_reduce` | Cross-GPU/CUDA synchronization |
| **GDN/SSM** | (custom, Kindi-specific) | `k_gdn_pre`, `k_gdn_deltan` | `gated_delta_net.cu` (17KB), `ssm-*.cu` | `gated_delta_net.comp`, `ssm_*.comp` (2 shaders) | `ggml_cpu_gated_delta_net` | Gated delta network recurrent step |

## 11. Methodology Flow — Prefill / Decode / Transformer (Logic Comparison)

A generic LLM inference flow as engine-agnostic logic, with llama.cpp logic
and Kindi logic side by side. The same conceptual steps are executed by all
engines; the dispatch strategy and kernel implementation differ.

### Phase A — Initialization & Model Load

```
[GENERIC LOGIC]
  1. Detect backend device(s) — GPU count, memory, ReBAR
  2. Allocate context (KV cache, weight mmap, scratch buffers)
  3. Load model weights from GGUF (mmap / streaming)
  4. Select weight format (q4_0, q8_0, ternary, fp16, etc.)

| Step | Llama Logic | Kindi Logic |
|---|---|---|
| 1 | ggml_backend_registry_init: probe Vulkan/CUDA/CPU/Metal backends | install.sh: PROFILE=kepler if ccn<50, model=2b, ctx 24576/65536 |
| 2 | ggml_backend_cpu/GPU: alloc context via ggml_init_params | Engine.__init__ / TPEngine.__init__: allocate U2, XP, KV buffers |
| 3 | gguf_init_from_reader: mmap GGUF at data_offset | layout.py: repack GGUF weights to k_g5n/g5/g4 internal tile format |
| 4 | Quant type from GGUF tensor.type (q8_0, q4_0, etc.) | weight_format = k_g4 (2-bit) or k_g5 (5-trit ternary) |
```

### Phase B — Prefill (Prompt Processing)

```
[GENERIC LOGIC]
  1. Embed input tokens (token IDs -> fp32/fp16 vectors)
  2. For each transformer layer:
     a. RMS_norm + Hadamard fold
     b. Attention: QKV GEMM + RoPE + softmax + mask + context-V GEMM
     c. Residual add
     d. FFN: RMS_norm + gate/up GEMM + SiLU + down GEMM
     e. Residual add
  3. Output projection (tied head)
  4. Store K/V in cache for decode phase

| Step | Llama Logic (llama.cpp) | Kindi Logic (Kepler) |
|---|---|---|
| 1 | token embedding: cpy.cu (copy) or get_rows.cu | k_l2w fused with embedding lookup |
| 2a | norm.cu: k_norm / k_rms_norm (separate kernel per op) | k_xrms / k_rmsfold (fp64 sum, Hadamard fused) |
| 2b | fattn.cu: flash attention with Q4_K dequant in-tile | chunk.cu: k_attn_flash16 / k_attn_sq8 (q8_0 scores) |
| 2c | add-id.cu: residual via k_add_id | Residual fused into k_xrms / k_smfold |
| 2d | gated_delta_net.cu (gdn) + mmf.cu + mmq.cu (matmul) | chunk.cu: k_gdn_conv / k_gdn_delta + k_mm_g5n (5-trit unpack) |
| 2e | add-id.cu / add.cu: second residual | Same fused path |
| 3 | out-prod.cu: lm_head matmul | k_l2w (logits = weight-to-act) |
| 4 | rope.cu + cpy.cu: KV cache store | k_store_q8: q8_0 cache (int8 + fp16 scale) |
| Notes | Graph-based: ops added to ggml_cgraph, dispatched after build | Stage-based: chunk.py builds plan per token batch, JIT via NVRTC |
```

### Phase C — Decode (Token-by-Token Generation)

```
[GENERIC LOGIC]
  1. Embed previous token
  2. For each transformer layer:
     a. RMS_norm
     b. Attention over KV cache (Q only, K/V from cache)
     c. Residual add
     d. FFN
     e. Residual add
  3. Output projection
  4. Sample next token (top-k / top-p / temperature)

| Step | Llama Logic (llama.cpp) | Kindi Logic (Kepler) |
|---|---|---|
| 1 | Same embedding lookup | k_rope (fused with embed) |
| 2a | norm.cu: RMS normalize (separate op) | k_xrms (fp64 accumulator) |
| 2b | fattn.cu: single-query flash attn against KV cache | k_attn_flash16: q8_0 KV scores, masked attention |
| 2c | add.cu: residual add | k_l2w (fused) + k_rmsfold |
| 2d | mmf.cu (gate/up) + mmq.cu (down) | k_mm_g4 or k_mm_g5n (mm_core) |
| 2e | add.cu | Same fused path |
| 3 | out-prod.cu: lm_head | k_l2w |
| 4 | top-k.cu + topk-moe.cu: sampling on GPU | Engine.py: CPU-side sampling (argmax/topk in Python) |
| Notes | Batched decode via GQA (grouped query attention) | wrappers(): generates _T variants for token-batched decode |
|       | Dynamic parallelism: device-side kernel launches | No dynamic parallelism; all launches host-side |
```

### Phase D — Transformer Block (Per-Layer)

```
[GENERIC LOGIC — one transformer layer iteration]
  INPUT:  X[t] = embedding or previous layer output
  STEP 1: X_norm = RMS_NORM(X) + Hadamard_fold
  STEP 2: Q = X_norm * Wq;   K = X_norm * Wk;   V = X_norm * Wv
  STEP 3: Q' = RoPE(Q);   K'.append_to_cache(K);   V'.append_to_cache(V)
  STEP 4: Attention_out = FlashAttention(Q', K_cache, V_cache)
  STEP 5: X_attn = X + Attention_out
  STEP 6: X_ffn_norm = RMS_NORM(X_attn)
  STEP 7: Gate = SiLU(X_ffn_norm * W_gate);   Up = X_ffn_norm * W_up
  STEP 8: Hidden = Gate * Up
  STEP 9: FFN_out = Hidden * W_down
  STEP 10: Output = X_attn + FFN_out

| Step | Llama Logic | Kindi Logic |
|---|---|---|
| 1 | norm.cu (separate k_rms_norm) | k_xrms + k_rmsfold (fused) |
| 2-3 | mmq.cu / mmf.cu (GEMM) + rope.cu | k_mm_* (mm_core with unpack) + k_rope |
| 4 | fattn.cu (flash_attn, q4_k in-tile dequant) | k_attn_flash16 / k_attn_sq8 |
| 5 | add.cu (k_add) | Residual fused into k_l2w |
| 6 | norm.cu | k_xrms |
| 7-8 | mmf.cu + gated_delta_net.cu or geglu | k_gdn_gates / k_smfold |
| 9 | mmq.cu | k_mm_* |
| 10 | add.cu | Residual fused |
| Notes | Graph node per GGML op; tensor parallel via allreduce.cu | TPEngine: two-card row-split, k_xrms swaps partials via pinned host mem |
```

### Phase E — Speculative Decode

```
[GENERIC LOGIC]
  1. Draft model proposes n tokens
  2. Verify each proposed token against target model (no state write)
  3. Accept/reject: commit accepted tokens, redo rejected
  4. Continue standard decode

| Step | Llama Logic | Kindi Logic |
|---|---|---|
| 1 | No built-in speculative in llama.cpp | N/A (Kindi has custom speculative) |
| 2 | No equivalent | spec.cu: k_sgdn_pre_T with wb=0 (verify pass, no write-back) |
| 3 | No equivalent | Commit kernels: advance state by first n accepted tokens |
| 4 | Standard decode | Standard decode path |
| Notes | — | Two-card TPEngine: partial sums swapped via pinned host mem in k_xrms |
```

## 12. Key Methodological Differences

1. **Dispatch granularity**: llama.cpp dispatches per-graph-node (each GGML op becomes a kernel launch); Kindi dispatches per-pipeline-stage (entire decode step in one kernel set).

2. **Fusion strategy**: llama.cpp fuses at the kernel-template level (dequant-in-tile, flash-attn-with-softmax); Kindi fuses at the arithmetic level (Hadamard fold into RMS norm, QKV into gate).

3. **Graph vs imperative**: llama.cpp builds a `ggml_cgraph` DAG then dispatches; Kindi is imperative Python host driving CUDA kernel launches directly.

4. **KV cache format**: llama.cpp uses GGML tensor layout (q8_0 with metadata in tensor struct); Kindi uses explicit q8_0 store kernel (`k_store_q8` with int8 + fp16 scale per 32-value block).

5. **Two-card strategy**: llama.cpp uses NCCL-style allreduce; Kindi uses TPEngine with pinned host memory swap in `k_xrms` for partial sums.

6. **Template instantiation**: llama.cpp has template-instantiated `.cu` files (e.g. fattn-tile for every tile shape); Kindi parameterizes at runtime via NVRTC.

## 13. Methodology Flow Map — Device Functions per Backend

This section maps each pipeline concept ("device function") to its concrete implementation
across all four backends. The device function is the "what" — the conceptual step in the
LLM inference pipeline. The implementation is the "how" — the specific function, kernel,
or construct each backend uses.

### Device Functions: Prefill (Prompt Processing)

Generic prefill flow: embed tokens -> for each layer: norm+fold, QKV GEMM, RoPE, attention, residual, FFN, residual -> output norm -> head logits -> store KV cache.

| # | Device Function | Llama Logic (llama.cpp CUDA) | Llama Logic (llama.cpp Vulkan) | Llama Logic (llama.cpp CPU) | Kindi Logic (Kepler) | Porting Note |
|---|---|---|---|---|---|---|
| 1 | Embed | `ggml_cuda_op_cpy` / `ggml_cuda_get_rows` -> `cpy.cu` / `get_rows.cu` | `copy.comp` / `get_rows.comp` | `ggml_cpu_cpy` | `k_deq_emb3` / `k_deq_emb4` (Hadamard fold in) | CPU: fp32 lookup table; Kindi: int8+fp16 with Hadamard signs |
| 2 | RMS Norm | `norm.cu`: `k_norm<__RMS_NORM>` | `norm.comp` (shared memory reduction) | `ggml_compute_forward_rms_norm` (AVX2/AVX-512) | `k_rmsfold` + `k_xrms` (fp64 sum + fold) | CUDA/Vulkan: separate norm kernel; Kindi: fused into residual path |
| 3 | Hadamard fold | (none — llama.cpp doesn't fold) | (none) | (none) | `k_fold_fwht` / `k_rmsfold` / `k_smfold` (FWHT in-place) | Unique to Kindi: folded weights into Hadamard domain |
| 4 | QKV GEMM | `mmf.cu`: `mmf_q` fused gate+up; `mmq.cu`: `mul_mat_q4_K_q8_1` | `mul_mm.comp` with subgroups | `ggml_cpu_mul_mat` (AVX2/AVX-512) | `k_mm_g4` / `k_mm_g5n` (mm_core with unpack) | llama: dequant-in-tile; Kindi: lane unpack |
| 5 | K-norm | `rope.cu`: `k_rope_1d` | `norm.comp` passes | `ggml_cpu_get_rows` | `k_rmsw` + `k_rope2` | CUDA/Vulkan separate; Kindi fuses rmsw+rope |
| 6 | Q-norm | `norm.cu` | `norm.comp` | `ggml_cpu_norm` | `k_rmsw_s` | Same |
| 7 | RoPE | `rope.cu`: `rope_cos_sin` + `rope_interp1d` | Implicit in `mul_mmq.comp` | `ggml_cpu_rope` | `k_rope2` | Vulkan: no separate RoPE shader (fused in matmul) |
| 8 | Attention (flash) | `fattn.cu`: `fattn_v2` (online softmax) | `flash_attn*.comp` (6 shaders, subgroup reduce) | `ggml_cpu_soft_max` (not really fused) | `k_attn_flash16_T` / `k_attn_sq8_T` (q8_0 scores) | Vulkan: subgroup ops; CPU: no flash-attn |
| 9 | Residual add | `add-id.cu`: `k_add_id` | `add_id.comp` | `ggml_cpu_add` | Fused into `k_rmsfold` / `k_l2w` | Kindi: residual absorbed into next norm |
| 10 | FFN gate+up | `mmf.cu`: `mmf_g` (gated linear) | `mul_mm.comp` + `unary.comp` | `ggml_cpu_mul_mat` + `ggml_cpu_unary` | `k_smfold` (gate * up, Hadamard fold) | Kindi: gate+up+fold in one kernel |
| 11 | FFN down | `mmq.cu`: `mul_mat_q4_K` | `mul_mmq.comp` | `ggml_cpu_mul_mat` | `k_mm_g4` / `k_mm_g5n` | Same matmul, different unpack path |
| 12 | Output norm | `norm.cu` | `norm.comp` | `ggml_cpu_rms_norm` | `k_xrms` (final fold) | |
| 13 | Head projection | `out-prod.cu`: `k_out_prod` | `out_prod.comp` | `ggml_cpu_out_prod` | `k_l2w` (weight-to-activation) | Kindi: log_softmax not needed, argmax direct |
| 14 | KV store | `cpy.cu`: `cpy_cache` | `copy.comp` | Memory copy | `k_store_q8` + `k_q8row` | Kindi: q8_0 cache write with int8+fp16 scale block |

### Device Functions: Decode (Token-by-Token)

Generic decode flow: embed 1 token -> for each layer: norm+fold, QKV GEMM (Q only), RoPE, attention over KV cache, residual, FFN, residual -> output norm -> head logits -> sample.

| # | Device Function | Llama Logic (CUDA) | Llama Logic (Vulkan) | Llama Logic (CPU) | Kindi Logic (Kepler) | Porting Note |
|---|---|---|---|---|---|---|
| 1 | Embed | `cpy.cu` | `copy.comp` | `ggml_cpu_cpy` | `k_deq_emb3` | Same |
| 2 | RMS Norm | `norm.cu` | `norm.comp` | `ggml_compute_forward_rms_norm` | `k_xrms` / `k_rmsfold` | Kindi: fp64 sum accumulator |
| 3 | QKV GEMM (Q,K,V) | `mmq.cu` batch=1 | `mul_mmq.comp` batch=1 | `ggml_cpu_mul_mat` | `k_mm_g4` / `k_mm_g5n` (mm_core) | CUDA: per-op files; Kindi: 4 .cu with NVRTC |
| 4 | K/V norm | `norm.cu` | `norm.comp` | inline | `k_rmsw` | Same |
| 5 | RoPE | `rope.cu` | implicit | `ggml_cpu_rope` | `k_rope2` | |
| 6 | KV store | (already cached) | (already cached) | (already cached) | `k_store_q8` / `k_q8row` | Only Kindi writes new KV |
| 7 | Attention | `fattn.cu` flash decode | `flash_attn_decode_phase_1.comp` | `ggml_cpu_soft_max` | `k_attn_flash16` / `k_attn_sq8` | Vulkan: 6-phase flash decode; CPU: no fused |
| 8 | Softmax | Online in fattn | `flash_attn_decode_phase_1.comp` | `ggml_compute_forward_soft_max` | `k_softmax_rows` | |
| 9 | Context-V | `fattn.cu` | `flash_attn_decode_phase_1.comp` | `ggml_cpu_mul_mat` | `k_attn_vq8t` / `k_attn_vq8w` | |
| 10 | Q fold | `rope.cu` / `add.cu` | `add.comp` | `ggml_cpu_add` | `k_fold_fwht` | Kindi: Hadamard fold of attention output |
| 11 | Attn output | `mmq.cu` | `mul_mmq.comp` | `ggml_cpu_mul_mat` | `k_mm_g4` / `k_mm_g5n` | |
| 12 | FFN (fused) | `mmf.cu` + `mmq.cu` | `mul_mm.comp` + `unary.comp` | `ggml_cpu_mul_mat` + `ggml_cpu_unary` | `k_smfold` + `k_mm_g4` | Kindi: gate+up+fold+down as one logical block |
| 13 | Final norm | `norm.cu` | `norm.comp` | `ggml_compute_forward_rms_norm` | `k_xrms` (final fold) | |
| 14 | Head | `out-prod.cu` + `top-k.cu` | `out_prod.comp` + `topk_*.comp` | `ggml_cpu_argmax` | `k_l2w` + `k_argmax` | llama: GPU sampling; Kindi: CPU sampling in Python |
| 15 | Sampling | `top-k.cu`, `topk-moe.cu` | `topk_*.comp` | `ggml_cpu_argmax` (greedy) | Engine.py: argmax/topk in Python | llama: GPU sampling kernels; Kindi: host-side |

### Device Functions: Speculative Decode (Kindi-specific)

Kindi has a custom speculative decode path (spec.cu / spec.py). llama.cpp CUDA and CPU have
no speculative decode; llama.cpp Vulkan has NOT FOUND speculative decode markers.

| # | Device Function | Kndi Logic (Kepler) | Porting Note |
|---|---|---|---|
| 1 | Draft propose | NgramDraft (Python host) or Draft2B (2B pack) | Host-side n-gram matching |
| 2 | Verify (wb=0) | `k_sgdn_pre_T` with wb=0: verify pass, no GDN state write | Spec verify uses same kernels with write-back flag=0 |
| 3 | Verify attention | `k_attn_sq8m` / `k_softmax_m` / `k_attn_vq8w_m` | Up to QB=4 queries per launch |
| 4 | Accept/reject | Python host: `spec_accept()` rejection sampling | CPU-side decision |
| 5 | Commit state | `k_sconv_C` + `k_sgdn_deltan_C`: advance GDN state by n accepted tokens | State advance kernel |
| 6 | Restore (reject) | `_Copy`: device-to-device copy via CUDA graph | Graph with single copy nodes |
| 7 | Draft model graph | `k_deq_emb4_p`: chained greedy steps, token from device register | Uses tokp (device token pointer) |
| 8 | Sampling (spec) | `k_band` + `k_topk`: temperature/top-k on device | GPU-accelerated |

### Device Functions: Tensor Parallel (2-Card Coordination)

| # | Device Function | Kndi Logic (2-Card TPEngine) | Llama Logic | Porting Note |
|---|---|---|---|---|
| 1 | Weight split | `Shard._slices`: row groups split by NK, NV, heads, ffn_blocks | `allreduce.cu` (NCCL-style) | llama: allreduce; Kindi: row-parallel partition |
| 2 | GEMV split | `k_g4` / `k_g5n` with column-phase split (gridDim.y) | `mmq.cu` with NCCL gather | Kindi: parts left in `part` buffer |
| 3 | Partial exchange | `k_xrms`: swaps partials via mapped pinned host memory | `allreduce.cu`: device-to-device copy | Kindi: host-memory swap, llama: P2P |
| 4 | Sync coordination | `_exchange`: event-recorded memcpies, ring buffer of 8 events | NCCL allreduce barrier | Kindi: host-ring; llama: NCCL |
| 5 | Plan replay | `cuLaunchKernel` via ctypes, segment-by-segment | Same `cuLaunchKernel` but C++ native | Kindi: ctypes overhead 1301/us/token; llama: native |
| 6 | CUDA graph | `spec.Graph`: cuGraphInstantion per plan | (not used) | Kindi: per-token graph instantiation |
| 7 | KV cache | q8_0 on both cards, positions in sync | Same GGML tensor layout | Both use q8_0 cache |

### Device Functions: Memory Management

| Pattern | Kndi Logic (Kepler) | Llama Logic (CUDA) | Llama Logic (Vulkan) | Llama Logic (CPU) |
|---|---|---|---|---|
| **Weight storage** | g5_blob: 5-trit packed uint32, cached to disk (g5_*.bin) | GGUF mmap + dequant on load | GGUF streamed to SSBO | GGUF mmap in host memory |
| **Weight unpack** | `k_g4` / `k_g5n`: lane-based unpack (byte position major for 2-bit, 5-trit packs for 5-trit) | `convert.cu`: inline dequant per op | `dequant_*.comp`: shader dequant | inline SIMD unpack |
| **KV cache alloc** | `alloc_ctx`: q8_0 int8 + fp16 scale, per-position | `ggml_cuda_op_mul_mat`: tensor views | `copy.comp`: buffer views | `ggml_cpu_cpy`: memcpy |
| **KV cache store** | `k_store_q8`: int8 values + fp16 scale per 32-value block | `cpy.cu` / `cpy_cache` | `copy.comp` | memcpy |
| **Scratch buffers** | Engine.py: pre-allocated `part`, `xp`, `xn`, `qkv`, etc. | `common.cuh`: per-op workspace | Per-shader shared memory | `ggml_cpu_state`: workspace |
| **Host-pinned** | `cp.cuda.PinnedMemory`: xbuf for tp exchange | (minimal) | (none — Vulkan buffers) | (none — malloc) |

### Device Functions: Construct Usage Summary

| Construct | Kndi | llama CUDA | llama Vulkan | llama CPU | Notes |
|---|---|---|---|---|---|
| `__shfl_xor_sync` | yes (in norm, fwht) | yes | `subgroupShuffle` | N/A | Vulkan uses subgroup, CUDA uses `__shfl_sync` |
| `__syncthreads` | yes | yes | `barrier()` | N/A | Portable |
| `__shared__` | yes | yes | `shared` | N/A | Spelling |
| `__ldg` | KLDG macro (sm_35+) | yes | N/A | N/A | Kindi has fallback for sm_30 |
| `dp4a` | no (sm_30) | yes (int8 GEMM) | `GL_EXT_integer_dot_product` | N/A | Kindi: 5-trit unpack instead |
| `mma.sync` | no (sm_30) | yes (fp16 TFLOPs) | `GL_KHR_cooperative_matrix` (broken on AMD) | N/A | llama CUDA: tensor cores; Vulkan: avoid cooperative matrix on AMD |
| `dynamic_parallelism` | no | yes (175 hits) | no | no | CUDA-only; Kindi: host-side launch |
| `cooperative_groups` | no | yes | barrier via pipeline | no | Vulkan uses pipeline barriers |
| Inline SIMD | no | no | no | yes (AVX2/512/NEON/SVE) | CPU backend only |

## 14. Methodology Flow — End-to-End Pipeline Maps

The complete methodology flow shown as the engine's path from user input to token output.

### LLM Engine Agnostic Pipeline Flow

```
[GENERIC LOGIC — any LLM engine]
INPUT: prompt tokens
  |
  v
[Phase A: Init]
  detect_device() -> allocate() -> load_weights() -> select_format()
  |
  v
[Phase B: Prefill]                       <-- prompt tokens in a batch
  FOR token t IN prompt:                 <-- sequential or parallel
    embed(t)
    FOR layer il IN layers:
      norm_and_fold(residual)
      IF gdn_layer(il): gdn_compute(il)
      ELSE: attn_compute(il)
      ffn_compute(il)
  output_norm() -> head_logits() -> kv_cache_store()
  |
  v
[Phase C: Decode]                        <-- auto-regressive, 1 token at a time
  FOR step IN generate(n):
    embed(prev_token)
    FOR layer il IN layers:
      norm_and_fold(residual)
      gdn_or_attn(il)
      ffn(il)
    output_norm() -> head_logits()
    IF speculative_enabled:
      verify_tokens() -> accept_or_reject() -> commit_or_restore()
    sample(next_token, method=top_k/top_p/greedy)
  |
  v
[Phase D: Output]                       <-- streaming token out
  return/emit token
```

### Llama Logic — Full Pipeline Flow

```
[LLAMA LOGIC — llama.cpp CUDA/Vulkan/CPU]
INPUT: prompt tokens
  |
  v
[Phase A: Init]
  llama_init_backend() -> ggml_backend_registry_init()
  llamalloc: ggml_init() -> ggml_init_params
  GGUF mmap: gguf_init_from_reader() with quant_type from tensor.type
  device selection: --n-gpu-layers, --type q4_0/q8_0/etc
  |
  v
[Phase B: Prefill]
  ggml_eval()                          <-- llama.cpp entry
    |-> build_eval_graph()             <-- builds ggml_cgraph DAG
    |-> graph_add_tensor_ops()         <-- each op = graph node
    |-> ggml_backend_sched_eval()      <-- dispatches all nodes
        |
        | CUDA: ggml_cuda_compute_forward() [line 2262]
        |   case GGML_OP_RMS_NORM: norm.cu -> k_norm
        |   case GGML_OP_MUL_MAT:    mmq.cu -> mul_mat_q4_K_q8_1
        |   case GGML_OP_ADD:       add-id.cu -> k_add_id
        |   case GGML_OP_SILU:      unary.cu -> k_silu
        |   case GGML_OP_SOFT_MAX:  fattn.cu -> fattn_v2 (flash)
        |
        | Vulkan: ggml_vk_compute_forward() [line 12941]
        |   case GGML_OP_MUL_MAT:    mul_mmq.comp (subgroup ops)
        |   case GGML_OP_RMS_NORM:  norm.comp (shared mem reduce)
        |   case GGML_OP_SOFT_MAX:  flash_attn_decode comp shaders
        |
        | CPU: ggml_compute_forward() [ggml-cpu.c]
        |   case GGML_OP_RMS_NORM:  ggml_compute_forward_rms_norm (AVX2)
        |   case GGML_OP_MUL_MAT:   ggml_cpu_mul_mat (AVX-512)
        |   case GGML_OP_ADD:       ggml_cpu_add (vectorized)
        |-> kv_cache_save_past()                 <-- cpy.cu / copy.comp
  |
  v
[Phase C: Decode]
  llama_decode()                       <-- single-token decode loop
    |-> ggml_eval()                     <-- same path as prefill
    |-> llama_sample_top_p_top_k()      <-- on GPU via topk-moe.cu
    |   OR sampling on CPU (ggml_cpu)
  |
  v
[Phase D: Output]
  emit token to stream
```

### Kindi Logic — Full Pipeline Flow (Kepler, sm_30-sm_37)

```
[KINDI LOGIC — kindi.cu / chunk.cu / chunk1.cu / spec.cu]
INPUT: prompt tokens
  |
  v
[Phase A: Init]
  install.sh PROFILE=kepler                  <-- sm_30-sm_37, model=2b, ctx 24576/65536
  Engine.__init__()
    |-> Kernels()                          <-- cp.RawModule(code=kindi.cu) -> NVRTC JIT
    |-> layout.repack()                    <-- GGUF -> k_g5n/g5/g4 internal tile format
    |-> chunk1.reserve()                 <-- allocate chunk buffers
    |-> ctx fit: budget = free - 96MB; fit = budget // per_pos
    |-> K/V cache: q8_0 (int8 + fp16 scale per 32-value block)
    |-> RoPE tables: cos_t, sin_t (precomputed fp32)
  |
  v
[Phase B: Prompt (Prefill)]
  engine.feed_batch(toks, pos)  [engine.py:312]
    |-> chunk1.feed_batch()     [chunk1.py:194]
      |-> feed_chunk()          [chunk1.py:104]    <-- KINDI_CHUNK=256 tokens per batch
        |-> k_deq_emb3          <-- embed T tokens, Hadamard sign fold in
        |-> For layer il IN layers:
        |   IF gdn_layer(il):
        |     |-> _T(k_rmsfold_T)  <-- RMS + Hadamard fold, pending part merge
        |     |-> _gemm(attn_qkv)  <-- k_mm_g5n: 5-trit unpack + GEMM
        |     |-> _gemm(attn_gate)
        |     |-> _T(k_gdn_gates_T) <-- gdn gate computation
        |     |-> k_gdn_conv_T    <-- gdn conv step (1D conv over state)
        |     |-> _T(k_l2w_T)     <-- linear to 2*NK space
        |     |-> _T(k_gdn_deltan_T) <-- gdn delta update
        |     |-> _T(k_smfold_T)   <-- sm fold (Hadamard)
        |     |-> _gemm(ssm_out)   <-- residual back
        |   ELSE (attention layer):
        |     |-> _T(k_rmsfold_T)
        |     |-> _gemm(attn_q/k/v)
        |     |-> _T(k_rmsw_T)     <-- K row-wise norm
        |     |-> _T(k_rope2_T)    <-- RoPE
        |     |-> _T(k_store_q8_T) <-- store q8_0 KV cache
        |     |-> _T(k_q8row_T)    <-- quantize Q row to q8_0
        |     |-> P.cut()          <-- segment boundary
        |     |=> run_core()       <-- chunk attention (flash or exact)
        |     |-> _fold_fwht_T    <-- fold attention output
        |     |-> _gemm(attn_output)
        |   IF NOT last_layer:
        |     |-> _gemm(ffn_gate/up)  <-- gate + up vectors
        |     |-> _T(k_smfold_T)      <-- SiLU fold (Hadamard)
        |     |-> _gemm(ffn_down)
        |          |--- two-card: _exchange() swaps partials via pinned host
      |--- fallback: feed one-by-one via feed() -> body() [engine.py:293]
        |-> body(tok, pos):
        |   embed(tok) -> k_deq_emb3
        |   for il in layers:
        |     gdn(il, h) / attn(il, h, pos, kvi) + ffn(il, h)
        |   (same kernels, T=1, no _T wrapper)
    |-> remaining tokens go through chunk.py feed()
  |
  v
[Phase C: Decode (one token)]
  engine.step(tok, pos)  [engine.py:317]
    |-> body(tok, pos)  [engine.py:293]
      |-> embed(tok)                <-- k_deq_emb3
      |-> For layer il IN layers:
      |   IF gdn_layer(il):
      |     |-> rmsfold()           <-- k_rmsfold (fp64) + k_fold_fwht
      |     |-> gemv(attn_qkv, gate) <-- k_g4 / k_g5n
      |     |-> k_gdn_gates          <-- gdn gate
      |     |-> k_gdn_conv           <-- gdn conv
      |     |-> k_l2w                <-- linear
      |     |-> k_gdn_delta          <-- delta update
      |     |-> rms_rows             <-- k_rmsw
      |     |-> k_smfold             <-- Hadamard fold
      |     |-> gemv(ssm_out)        <-- residual
      |   ELSE (attention layer):
      |     |-> rmsfold()            <-- k_rmsfold
      |     |-> gemv(Q, K, V)        <-- k_g4 / k_g5n (mm_core)
      |     |-> k_rmsw (K norm)      <-- k_rmsw
      |     |-> k_rope               <-- RoPE
      |     |-> Kc/Vc[pos] = store   <-- fp16 K/V cache (not q8 in single-card path)
      |     |-> k_attn_s4w           <-- attention scores (q8_0 Q)
      |     |-> k_softmax_rows       <-- softmax
      |     |-> k_attn_v4p           <-- value projection (partial sum)
      |     |-> k_attn_comb           <-- combine partials
      |     |-> k_fold_fwht           <-- Hadamard fold
      |     |-> gemv(attn_output)    <-- residual
      |   IF NOT last:
      |     |-> ffn(il, h)           <-- RMS + gate/up/silu + down
      |-> return h
    |-> rmsfold(h, "output_norm.weight")  <-- final norm + fold
    |-> head()                             <-- k_l2w + k_argmax (greedy)
    |-> return token
  |
  v
[Phase D: Speculative Decode (if KINDI_TC=1)]
  spec.py: tc_step(tok, pos)
    |-> If draft tokens available:
    |   |-> verify([tok] + drafts, pos)  <-- wb=0 (no state write)
    |   |   -> k_sgdn_pre_T, k_attn_sq8m, k_softmax_m, k_attn_vq8w_m, k_attn_combn_m
    |   |-> accept/reject host-side
    |   |-> commit(n accepted)           <-- k_sconv_C, k_sgdn_deltan_C
    |   |-> restore (if rejected)         <-- device-to-device copy graph
    |-> Else: standard decode above
  |-> CUDA graph replay: cuGraphLaunch (one node per kernel, deps in plan order)
  |
  v
[Phase E: Output]
  token ID returned to caller
```

### Two-Card TPEngine Pipeline Flow (Kindi Logic, spec.py)

For the 27B model on two Quadro K1200s:

```
[TPENGINE — kindi.tp]
INPUT: prompt tokens
  |
  v
[Phase A: Init — Two Cards]
  TPEngine.__init__()
    |-> Shard(card 0): row-split K/V heads, GDN groups, FFN blocks
    |-> Shard(card 1): same, complementary rows
    |-> _host_buffers: PinnedMemory for xbuf (UVA, both cards see)
    |-> _embedding: token_embd in host-pinned (k_deq_emb3 reads on both cards)
    |-> build_plans() [tp.py:465]:
      |   body_plan: k_deq_emb4 -> layer loop -> plan_body / plan_feed
      |   plan_head: mixin -> gemv(output.weight) -> k_argmax
      |-> CUDA graphs NOT used on Kepler (sm_30); per-segment replay via cuLaunchKernel
    |-> _chunk.build(): builds per-token plans for chunk processing
    |-> _chunk.reserve(): allocates chunk_T buffers per card
  |
  v
[Phase B/C: Unified Run — tp.py:_run()]
  _run(tok, pos, head=True/False)
    |-> For each segment i in plan.segs:
    |   |-> card 0: cuCtxSetCurrent(card0.ctxh)
    |   |-> card 1: cuCtxSetCurrent(card1.ctxh)
    |   |-> launch all segment kernels on card 0
    |   |-> launch all segment kernels on card 1
    |   |-> IF segment i in P0.xch:
    |   |   |-> _exchange(): memcpPeecAsync(card1.Y -> card0.YP, card0.Y -> card1.YP)
    |   |   |-> event-synchronized (ring buffer of 8 events)
    |   |-> IF segment i in P0.cores:
    |   |   |-> run_core(): chunked attention (on both cards independently)
    |   |-> Every 8 segments: cuEventRecord + cuEventSynchronize (16 segments back)
    |
  Key kernel: k_xrms (tp.py:564)
    - Reads card 0's partial (Y) + card 1's partial (YP) from pinned host memory
    - Adds them: sum = Y + YP
    - Performs RMS norm + Hadamard fold on the sum
    - Writes folded residual to next stage
    - Both cards compute same result from same inputs (bit-identical)
  |
  v
[Phase D: Speculative (if KINDI_TC=1 + sm_80+)]
  Same as single-card spec.py, but:
    - Tensor-core plans use CUDA graphs (Graph class, cuGraph APIs)
    - Weight repack to tc_layout (k_tc_pack)
    - All-reduce replaced by k_xrms host-memory swap
```
