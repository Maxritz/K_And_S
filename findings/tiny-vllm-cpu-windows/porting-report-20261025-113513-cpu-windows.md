# Autonomous Porting Report

**Date/time:** 2026-10-25 11:35 UTC
**Repo:** `/workspace/chats/07ce5bca12d680f57df70775/tiny-vllm`
**Source arch:** `sm_80` — Ampere (RTX 30 series)
**Target arch:** `gfx1201` — RDNA4 (RX 9070 / 9070 XT / AI PRO R9700)
**Target backend:** `hip`
**Target language:** `cxx`
**Project type:** `llm`
**Target OS:** `windows`
**Mode:** `strict`
**Performance target:** N/A tok/s

---

## Phase 0 — Due Diligence

> _"Every port starts with a yes/no decision."_

### 1. Source Inventory
- **Repo:** `tiny-vllm` (cloned at `/workspace/chats/07ce5bca12d680f57df70775/tiny-vllm`)
- **Kernels found:** 3 GPU kernels
- **Source language:** CUDA C++ (.cu files)
- **CUDA constructs used in source:** __syncthreads
- **Manual constructs (wmma/mma/nvrtc/cudaGraph in code):** 0

### 2. Target Feasibility
- **Source arch supports:** tensor_cores, wmma, dot4, dot8, bf16_native
- **Source code uses:** __syncthreads

### 3. Should We Port? (Value Check)
| Question | Answer |
|---|---|
| Does the port add value? | Yes — enables cross-platform GPU support |
| Can target achieve reasonable perf? | Yes — simple compute-bound kernel |
| Does target support required features? | Yes |

### 4. Effort Estimate: LOW (1-2 hours)
- 3 small kernels (~10 lines each)
- 0 manual-only constructs
- Straightforward memory operations

---

## Phase 0 — Auto-Detected Parameters

### GPU Toolkits Detected (from kands://tk-tools)

| Env var | Value |
|---|---|
| ROCM_PATH | ❌ not set |
| CUDA_PATH | ❌ not set |
| VULKAN_SDK | ❌ not set |
| ONEAPI_ROOT | ❌ not set |
| OPENVINO_ROOT | ❌ not set |

### Compilers Detected (runtime)

- `cl.exe`: ❌ not in PATH
- `gcc`: ✅ `/usr/bin/gcc`
- `g++`: ✅ `/usr/bin/g++`
- `clang++`: ❌ not in PATH
- `perl.exe`: ❌ not in PATH

### Model Folders Detected

- No standard model directories found


---

## Phase 2 — Kernel Discovery

**GPU kernels found:** 3
**CPU kernels found:** 0

### GPU Kernel Files
- `src/kernels.cu` (gpu_cuda) — CUDA/HIP kernel source
- `src/kernels.cuh` (gpu_cuda) — CUDA/HIP kernel header
- `tests/test_softmax.cu` (gpu_cuda) — CUDA/HIP kernel source

---

## Phase 3 — Architecture Comparison

Comparing `sm_80` → `gfx1201`

| Feature | Source (sm_80) | Target (gfx1201) | Gap |
|---|---|---|---|
| wave_size | 32 | 32 | SAME |
| lds_kib | 164 | 128 | DIFF |
| tensor_cores | True | False | DIFF |
| wmma | True | True | SAME |
| dot4 | True | True | SAME |
| dot8 | True | True | SAME |
| bf16_native | True | True | SAME |
| fp8_native | False | False | SAME |

### Porting Implications

- ⚠️ Source has tensor cores, target does NOT → MMA kernels need dot4/F32 fallback

### Target arch known issues
- builds validated
- no 4-bit matrix-core kernel ships for this target: the tuned BLAS solution set has none, and its generator cannot express a 4-bit datatype at all, so a 4-bit path has to be hand-written. The vendor's kernel-template library does expose packed-int4 at K=16 and K=32
- matrix-core datatypes are same-type only here (fp16, bf16, int8, fp8 in all four mixes, packed int4): there is no narrow-weight/wide-activation pair, so choosing a 4-bit weight operand forces a 4-bit activation operand, which is a numerical change to whatever the activations feed
- 4-bit float exists in the generator but its opcode is gated to a later target family and is absent from this instruction set
- measured on this target: int4 16x16x16 = 161 TOPS, int4 16x16x32 = 336 TOPS, fp16 16x16x16 = 179 TFLOPS, and both int4 forms issue about 0.22 matrix ops per CU per cycle rather than one, so any figure derived from one-per-cycle is roughly 4.5x too high


---

## Phase 4 — Toolchain Lookup

**Target vendor:** AMD
**Target OS:** windows

### Compilers and SDKs

| Tool | Path |
|---|---|
| hipcc | `${ROCM_PATH}/bin/hipcc` |
| clang | `${ROCM_PATH}/llvm/bin/clang++` |
| llvm_objdump | `${ROCM_PATH}/llvm/bin/llvm-objdump` |
| llvm_dis | `${ROCM_PATH}/llvm/bin/llvm-dis` |
| rocgdb | `${ROCM_PATH}/bin/rocgdb` |

**Libraries:** hip, rocblas, rocblaslt, rocwmma, hsa-runtime
**Supported arches:** gfx908, gfx1030, gfx1031, gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1200, gfx1201

### Windows notes

Windows ROCm: requires ROCm 6.0+; HIP runtime ships as DLL; may need HSA_OVERRIDE_GFX=gfx1201 for RDNA4


---

## Phase 5 — Performance Estimates

**Date/time:** 2026-10-25 11:35 UTC
**Current time (for search filtering):** 2026-10-25 11:35 UTC
### RDNA4 (RX 9070 / 9070 XT / AI PRO R9700)

| Property | Value |
|---|---|
| name | RDNA4 (RX 9070 XT class) |
| vendor | AMD |
| compute_units | 72 |
| base_clock_mhz | 1700 |
| boost_clock_mhz | 2520 |
| fp32_peak_gflops | 17.4 |
| fp16_peak_gflops | 34.9 |
| bf16_peak_gflops | 34.9 |
| tensor_cores | False |
| memory_bandwidth_gb_s | 1152 |
| l2_cache_mb | 64 |
| lds_kib | 128 |
| known_bottlenecks | ['No WMMA — same F32 fallback as gfx1030 for attention', '128KiB LDS — can use larger shared memory tiles', '17% higher memory bandwidth vs gfx1100'] |

### Workload Estimates

| fp32_gemm_4096 | 16.5 GFLOPS (95% efficiency) |
| fp16_gemm_4096 | 33.1 GFLOPS (95% efficiency) |
| bf16_gemv | bf16-native: 32.8 GFLOPS |
| flash_attn_4096 | 4.8 ms/token (F32 fallback) |
| dot4_i8_dequant | 1030 GB/s effective (90% of peak) |


---

## Phase 6 — Reference Research

**Search time:** 2026-10-25 11:35 UTC

### Recommended searches

### GitHub repos to check

### ArXiv keywords to search


> Run `search_arxiv(query="...")` or `search_github(query="...")` with these terms to find
> real-world benchmarks and reference implementations.

---

## Phase 7 — Porting Strategy (per kernel)

### CUDA → HIP

**Complexity:** low
**Mode:** `strict`

**Auto-translatable:** 2 constructs
- hipify-perl
- hipify-python

**Manual transforms needed:** 23 constructs
- `cudaMalloc` → `hipMalloc`
- `cudaFree` → `hipFree`
- `cudaMemcpy` → `hipMemcpy`
- `cudaMemcpyHostToDevice` → `hipMemcpyHostToDevice`
- `cudaMemcpyDeviceToHost` → `hipMemcpyDeviceToHost`
- `cudaMemcpyDeviceToDevice` → `hipMemcpyDeviceToDevice`
- `cudaStreamCreate` → `hipStreamCreate`
- `cudaStreamSynchronize` → `hipStreamSynchronize`
- `cudaEventRecord` → `hipEventRecord`
- `cudaEventSynchronize` → `hipEventSynchronize`
... and 13 more

**Manual-only (no auto translation):** 8 constructs
- `wmma` → `rocWMMA (gfx1100+, gfx1200/gfx1201) or F32 fallback` — RDNA4 (gfx1200/gfx1201) supports rocWMMA. Use __builtin_amdgcn_wmma_*_gfx12 intrinsics. RDNA2 (gfx1030) has NO WMMA — use F32 fallback
- `mma` → `rocBLASLt or dot4 instructions` — Tensor cores → dot4 or rocBLASLt
- `tensor_core_mma` → `rocWMMA or rocBLASLt MMA API` — Requires ROCm 5.7+
- `nvrtc` → `hiprtc or runtime compilation via HIPRTC` — Different API but similar concept
- `cuMemPool` → `hipMemPool (if available) or hipMalloc managed` — Memory pooling support varies by ROCm version
- `cudaGraph` → `hipGraph (experimental)` — HIP graphs are experimental; may need fallback
- `cuStreamWorkDescrib` → `hipStreamWorkStealing` — Different semantics
- `header_order` → `Include runtime FIRST` — MUST #include hip_runtime.h BEFORE custom headers using __device__/__forceinline__/__shared__

### Strategy Decision (for `strict` mode)

| Kernel category | Strategy | Reason |
|---|---|---|
| Memory (cudaMalloc → hipMalloc) | Copy | Direct replacement |
| Kernel launch (<<<>>> → hipLaunchKernelGGL) | Copy | hipify-perl auto-translates |
| WMMA/MMA | Adapt | Source: ✅ WMMA → Target: ✅ WMMA |
| Tensor cores/MMA | Adapt | Source: ✅ TC → Target: ❌ TC |
| C++17 host code | Adapt/Rewrite | C++ RAII for buffers. Use std::vector + hip::array or D3D12 smart pointers. |


---

## Phase 8 — Action Plan

### Strict mode (recommended for production):

1. **Build setup:** Read `kands://tk-tools` for compiler paths and flags
2. **Shader compile:** `hipcc --offload-arch=gfx1201 -O3` (hipcc)
3. **Host compile:** `g++ -std=c++17 -O2 host.cpp -o host` (gcc/g++/clang)
4. **CMake:** `cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=Release`
5. **Backend interop:** Kernels in HIP C++ (.cu/.hip → .hsaco). Host uses hip runtime API. C interop via hip_runtime_api.h.
6. **STRACE instrumentation:** Add breadcrumb tracing to ALL kernel dispatch files BEFORE testing (base rule). Set TRACE_BREADCRUMB=1 or pass --breadcrumb
7. **Parity testing:** Run CPU oracle + GPU output comparison (see `kands://diagnostics`)
8. **PERF_TRACING:** Run with TRACE_PERF=1 or --perf-tracing to capture timing. Iterate using sherlock-it loop
9. **TDR safety:** Use fence polling with <=1500ms timeout on Windows (not WaitForSingleObject(INFINITE))
10. **Export:** Run `export_contribution(engine_name='...', arch='gfx1201')` to share results
11. **Verify:** `ctest -R _parity` && run host binary with `--selftest` flag

### Loose mode (get it running first):

1. **Get it compiling:** Copy + auto-translate with `hipify-perl`
2. **Fallback everywhere:** Use FP32 where BF16/WMMA/Tensor cores are missing
3. **Quick test:** Verify output is in the right ballpark (tolerance 1e-2)
4. **Iterate:** Run perf tracing, note bottlenecks, run `suggest_optimizations`
5. **Optimize:** One kernel at a time, track with `record_performance`
6. **Share:** Export your learnings with `export_contribution`

---

## Next Steps

Run these in sequence:
```bash
# 1. Look up your toolchain
query_toolchain(gpu_vendor="AMD", os="windows", host_compiler="auto")

# 2. Check kernel compatibility
check_arch_compatibility(kernel_list=["flash_attn", "router_topk", "kv_q4"], arch="gfx1201")

# 3. Compare archs
compare_archs(arch_a="sm_80", arch_b="gfx1201")

# 4. Search for references
search_references(query="gfx1201 hip performance benchmark")

# 5. After you optimize something:
record_performance(arch="gfx1201", workload="flash_attn", tok_per_s=12.5)

# 6. Check if you're close to your target
compare_to_target(arch="gfx1201", workload="flash_attn", target_tok_per_s=20)

# 7. Export your learnings
export_contribution(engine_name="my-port", arch="gfx1201", workload="flash_attn")
```


## 🏁 MCP Self-Check: A (115/100)
| Basic | Score | Status |
|---|---|---|
| 1. Analysis | 23/33 | ✅ |
| 2. Tracing Mandate | 43/33 | ✅ |
| 3. Porting Process | 49/34 | ✅ |
| **Total** | **115/100** | **Grade: A** |

**Checklist:**
  ✅ Auto-detected source architecture
  ✅ Suggested target architecture
  ✅ Detected available toolchains
  ✅ STRACE instrumentation referenced in report
  ✅ CPU oracle pattern included for numerical verification
  ✅ TDR safety patterns included
  ✅ STRACE+PERF_TRACING mandated as required Phase 8 step
  ✅ Phase 7 porting strategy generated
  ✅ Phase 8 action plan generated
  ✅ Backend-specific patterns for 'hip'
  ✅ Language-specific guidance for 'cxx'
  ✅ Memory management mapping included
  ✅ Synchronization pattern mapping included
  ✅ Build commands for target language

---
*All Self-Check Gate rules passed (or explicitly bypassed with audit record).*

---
**Report saved to:** `/workspace/chats/07ce5bca12d680f57df70775/tiny-vllm/findings/porting-report-cxx-hip-20261025-113513.md`
