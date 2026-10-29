# GPU Inference Pipeline Methodology — Backend-Agnostic

> Distilled from llama.cpp (ggml-org) @ commit `338fce6`, verified against the actual
> tree by two independent audits (2026-10-30). Every pattern below cites the real
> mechanism; where a claim could not be verified it is labeled UNVALIDATED, never stated.
> This file REPLACES the invented API names that appeared in earlier learnings
> (`kernel_dispatch`, `ggml_cpu_pool`, `ggml_cpu_mul_mat` — none exist; see §9).

## 0. The universal anatomy

Every mature LLM backend — CUDA/HIP, Vulkan, CPU — is the same five stages. Only the
vocabulary changes:

| Stage | CUDA/HIP | Vulkan | CPU |
|---|---|---|---|
| 1. Graph walk | backend `graph_compute` → evaluate-and-capture | `ggml_backend_vk_graph_compute` walks nodes | `ggml_graph_compute` thread-pool walk |
| 2. Op select | `switch (dst->op)` in compute_forward | `switch (node->op)` in build_graph | `switch (tensor->op)` in compute_forward |
| 3. Kernel select | per-op wrapper + arch/shape dispatch | pipeline map keyed by `{src0_type, src1_type, f16acc}` | `type_traits_cpu[type]` table lookup |
| 4. Record/launch | `<<<>>>` on one of 8 streams, or captured into cudaGraph | `vkCmdDispatch` into primary OneTimeSubmit cmd buffer | direct function call on pool thread |
| 5. Submit/sync | stream order + cudaGraph replay | amortized `vkQueueSubmit` + timeline semaphores | thread-pool barrier per node |

**Lesson:** a port is not "translate kernels". It is "re-implement stages 3–5 for the
target's submission model". Stages 1–2 are backend-independent graph mechanics.

## 1. Shape-driven kernel selection: decode vs prefill

All three backends split GEMM into two regimes on the SAME op, chosen by shape —
never by an explicit engine flag:

- **Decode (GEMV)**: batch of queries == 1 (or ≤ 8 columns). Vector-dot kernels:
  `mul_mat_vec*.comp` per quant type (Vulkan); `mmvq.cu` vec-dot selection (CUDA);
  `vec_dot` from type-traits (CPU). Weights stream once per token → memory-bound.
- **Prefill (GEMM)**: T > threshold. Tile kernels + optional split-K:
  `mul_mm.comp`/coopmat + `mul_mat_split_k_reduce.comp` (Vulkan, split-K guessed
  from shape); `mmq.cu`/`mmf.cu` (CUDA); row-block loops (CPU).

**Threshold evidence:** Vulkan picks GEMV when `ne[1] == 1` or `≤ mul_mat_vec_max_cols
(=8)` single-batch. CUDA FA picks VEC vs TILE vs MMA by `Q->ne[1]` + arch + K/V dtype.

**Porting rule (kindi lesson confirmed):** a batched-T path must cover EVERY stage, not
just the body GEMV. The kindi port's 1.35–1.65× prefill ceiling existed because
attention stayed per-token while the body went T-batched.

## 2. Capability-gated pipeline maps (query, don't assume)

The Vulkan backend is the cleanest example of the capability-ladder pattern:

1. **Query extensions + features** at device init: `VK_KHR_cooperative_matrix`,
   `VK_NV_cooperative_matrix2`, chained feature structs via `getProperties2()`.
2. **AND with the returned feature bit** — extension present ≠ feature enabled.
3. **Shape probe**: enumerate cooperative-matrix M/N/K shapes; disable the path if no
   f16 shape with f32/f16 accumulator exists.
4. **Key the pipeline map by capability + types**, not by a runtime branch: the map is
   populated from *different shader sets* per device class (coopmat2 block / coopmat1
   block / scalar block), then looked up per op.
5. **Fallback is a real compiled path**, not a stub: no-coopmat devices get the scalar
   subgroup `mul_mm.comp`/`mul_mmq.comp` pipelines with optional integer-dot variants.

**Porting rule:** feature-present-but-not-accelerated ⇒ do not select the path
(kindi: `4x8BitPacked*Accelerated=0` ⇒ OpSDotKHR scalarizes ⇒ float FMA wins;
Meteor Lake CPU: AVX-VNNI present but no AVX-512 ⇒ INT8 loses to F32). Same veto,
three backends.

## 3. Quant dispatch: one op, three dispatch idioms

- **CPU**: a static per-type traits table (`type_traits_cpu[GGML_TYPE_COUNT]`) gives
  `from_float` / `vec_dot` / `nrows`. mul_mat is ONE generic function parameterized by
  the table. Adding a quant = adding table entries + one vec_dot kernel.
- **CUDA**: a compile-time `switch` over types in `mmq.cu`/`mmvq.cu` (legacy Q1_0–Q8_0,
  K-quants Q2_K–Q6_K, I-quants IQ1_S … IQ4_NL — **I-quants ARE supported on CUDA,
  Vulkan, and Metal; "GPU can't do I-quants" is FALSE**).
- **Vulkan**: one compiled pipeline per (type × tile-size × aligned) combination,
  registered into the map at device init; GEMV variants per type additionally.

**Porting rule:** match the target's extension idiom. C targets want the traits-table
shape (kindi's Python plan-builder is the same idea at NVRTC granularity); shader
targets want the pre-compiled pipeline-map shape.

## 4. Submission & sync models (where ports bleed)

- **CUDA/HIP**: up to 8 non-blocking streams per device. Decode replay uses **stream
  capture → cudaGraphInstantiate → cudaGraphLaunch**, with a 2-stable-calls warmup rule
  and `cudaGraphExecUpdate` for changed nodes. Ops that can't be captured check
  `cudaStreamCaptureStatus` themselves.
- **Vulkan**: 1 compute + 1 transfer queue. Submission is **amortized by FLOPs** — the
  graph loop flushes roughly every ~200 GFLOP and adapts after each submit. Command
  buffers are primary, `eOneTimeSubmit`, recycled from a pool; timeline semaphores
  chain them; a fence only on explicit sync. Descriptor sets are reused across graphs
  when bindings match.
- **CPU**: no submission at all; the "barrier" is the thread-pool sync per node.

**Porting rule (kindi measured):** per-dispatch fence = 17,852 syncs for 40 tokens →
3.3 tok/s. Batching to one submit per token → 5.5 tok/s. Barrier cost scales with work
in flight, not with barrier existence: fuse between big kernels, not around small ones.

## 5. Memory patterns

- Weights live in device-local buffers; allocation prefers ReBAR
  (device-local|host-visible) then falls back to plain device-local; UMA devices
  (Meteor Lake iGPU) prefer host-visible.
- H2D goes through a lazily-grown staging buffer (`vkCmdCopyBuffer` + deferred memcpy);
  host-memory import (`VK_EXT_external_memory_host`) when usable.
- Scratch is **preallocated and persistent** (`prealloc_x/y/split_k`) — never
  per-dispatch allocation (kindi: `allocation: 0 allocs` in all healthy traces).
- **The backend has NO KV-cache object.** K/V arrive as ordinary graph tensors;
  rope is a graph op applied before the cache write; quantized KV gets dequantized to
  an f16 scratch once per pass when beneficial. KV dtype lives at the model layer.

**Porting rule:** if your port re-copies weights or re-stages per pass, fix that before
touching shaders (kindi 27B: 5.9 GB of staging copies = 18% of wall).

## 6. Flash-attention selection matrix

CUDA picks among VEC / TILE / MMA_F16 by a scoring function over: arch generation,
`Q->ne[1]`, head dims (template-instance files are pre-compiled per dkq×dv pair), K/V
quant type, sparse-gather support. TILE/MMA always convert K/V to F16 first.
Vulkan adds device-specific paths: Intel Xe dual-phase decode (phase1 QK → phase2 PV)
gated on an `xe_fa_opt` check, plus split-K over KV when occupancy is low, combined by
a reduce shader.

**Porting rule:** one FA kernel is never enough; ship a selection function and let
shape+capability pick. But gate every path on measurement (kindi: its "flash16" naming
in older learnings was wrong — the real kernels are `k_att_flash16_T`/`k_att_flash16k_T`
over an fp16 cache, and the q8-score family is `k_attn_sq8*`).

## 7. Multi-GPU / tensor parallel

- Peer copy = `cudaMemcpyPeerAsync` + event record/wait (HIP: the vendor shim header
  maps the whole API mechanically — `cudaMalloc→hipMalloc`, `cudaGraph*→hipGraph*`).
- AllReduce: NCCL when available, else an internal staged-through-pinned-host-DRAM
  path, else butterfly — selected via a function-pointer chain and env override.
- No split-buffer code in the CUDA backend at this commit; multi-GPU splitting is the
  scheduler's job (sub-graphs per backend).
- kindi's TP does the same thing differently: partials exchanged INSIDE the RMS-norm
  kernel through mapped pinned host memory + epoch/flag spinning — no memcpyPeer, no
  `_exchange()` function (earlier learnings invented both).

## 8. What "pipeline" means for a port (the methodology)

1. **Walk the real graph** — dump `n_nodes`, op histogram, tensor shapes for one token
   (llama.cpp: `--dump` or a trace; kindi: `--trace` per-component table).
2. **Classify every op** into decode-GEMV / prefill-GEMM / elementwise / norm / FA /
   sampling. The kernel-selection rules of §1–§3 follow from this table.
3. **Find the submission boundary** — count submits per token. >1 per op = port smell.
4. **Find the memory boundary** — staging bytes per pass vs weight bytes. Ratio >1 = bug.
5. **Re-derive the capability gates** on YOUR device (kindi's §2 dump is the template:
   extension, feature, shape-probe, measured-acceleration bit).
6. **Only then port kernels**, in the order the op histogram ranks them.

## 9. Corrections ledger (replaces invented names)

Earlier learnings stated mechanisms that do not exist. The real ones:

| Invented (WRONG) | Real mechanism (verified) |
|---|---|
| `kernel_dispatch` table in CUDA | `ggml_cuda_compute_forward()` switch, ggml-cuda.cu:2262 |
| `ggml_cpu_pool` fn-ptr table | `type_traits_cpu[]` table, ggml-cpu.c:215; generic forward in ggml-cpu.c:1745 |
| `ggml_cpu_mul_mat/rope/soft_max/...` | `ggml_compute_forward_*` family, ggml-cpu.c:1255+, parameterized by the traits table |
| "284 .cu files" | 191 (71 top-level + 120 template-instances) |
| "196 .comp files" | 155 (+7 feature tests) |
| "dynamic parallelism, 175 hits" | 0 `cudaLaunchDevice` — all launches host-side |
| "no speculative decode in llama.cpp" | examples/speculative exists; driven via `params.speculative.*`, no `llama_speculative_*` C API |
| "Vulkan: no RoPE shader / no flash-attn" | rope_norm/rope_neox/multi/vision + 8 flash_attn* shaders |
| "CUDA/Vulkan don't support I-quants" | IQ1_S…IQ4_NL present in mmq/mmvq and Vulkan vec pipelines |
| kindi `k_attn_flash16` | `k_att_flash16_T`/`k_att_flash16k_T` (fp16 cache); q8 scores = `k_attn_sq8*` |
| kindi `_exchange()`/memcpyPeerAsync | exchange inside `k_xrms` via mapped pinned host memory |
| kindi "4 .cu files / 77 kernels" | 7 .cu files / 109 `__global__` kernels |

## Evidence labels

| Claim cluster | Status |
|---|---|
| §0–§7 mechanisms + file:line cites | PASS — verified in tree @338fce6 by two audits |
| Vulkan flush ~200 GFLOP, 8 streams, capture-warmup-2 rule | PASS — exact code reads |
| "No VkPipelineCache object; caching = pipeline structs + embedded SPIR-V" | PASS |
| "1301 launches/token, ~16 µs via ctypes" (kindi) | UNVALIDATED — found in spec.py docstring by one audit, not found by the other; do not cite as measured |
| Any tok/s figure | Must come from portingwork/PERFORMANCE.md traces only |
