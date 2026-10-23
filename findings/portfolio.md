# Porting Report: Portfolio Engine for RDNA2 + RDNA4

This is a worked example demonstrating the full K_and_S methodology applied
to a hypothetical portable GPU engine. Use this as a reference for porting any
real engine — substitute your own kernel names, arch targets, and model families.

---

## Phase 0 — Setup

| Field | Value |
|---|---|
| Engine name | `portable_engine` (example) |
| Repo | `H:\\project-portable` (git: yes) |
| Build system | CMake 3.24+ |
| GPU backend | CUDA (`ENGINE_ENABLE_CUDA=1`) / HIP (`ENGINE_ENABLE_HIP=1`) |
| Existing archs | sm_75, sm_80, sm_86, sm_89, sm_120 (CUDA); gfx1030, gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1200, gfx1201 (HIP) |

## Phase 1 — Kernel/Shader Census

### GPU Kernels: N `.cu` + M `.cuh` (CUDA/HIP dual-compile)

| File | Category | Arch-specific branches |
|---|---|---|
| `dequant_bf16.cu` | Dequantization | `__HIP__` for bf16 packing |
| `cvec.cu` | Activation concat | wave32/wave64 via hip_compat |
| `bf16_gemv.cu` | GEMV | sm_80+ TF32 MMA vs fp32 FMA fallback |
| `fused_gr.cu` | Gated residual | `__HIP_ARCH__` bf16 path |
| `attention_flash.cu` | Flash attention | WMMA/MMA vs portable F32 |
| `elementwise.cu` | Elementwise ops | `v_dot2` on RDNA4 |
| `quant_exl3.cu` | EXL3 quant | sm_80+ vs legacy, `__HIP__` paths |
| `batch_decode.cu` | Batched decode | rocWMMA (`HAS_WMMA`) |
| `ffn.cu` | Feed-forward | fp32 vs bf16 compute path |
| `fast_path.cu` | Fast path kernels | `nvvm` vs `hip` intrinsics |
| `hyper_conn.cu` | Hyper-connection | 32-bit vs 64-bit indices |
| `attention_kda.cu` | KDA attention | tensor core vs F32 fallback |
| `attention_qsa.cu` | Full attention | `mma_now()` vs scalar path |
| `router_top10.cu` | MoE router | `dot4` instruction dispatch |
| `sampler.cu` | Tokenizer sampling | fp32 accumulation order |
| `kv_q4.cu` / `kv_q8.cu` | KV cache | Q4_0/Q8_0 layouts |
| `native_moe.cu` | Native MoE | per-arch block size tuning |
| `quant_exl3_legacy.cu` | Legacy EXL3 | sm_75-only path |

### CPU Kernels: N production `.cpp` files

| File | ISA | Purpose |
|---|---|---|
| `expert.cpp` | AVX-512 VNNI/VBMI | Q2_0 shared/gated experts (compile-time geometry) |
| `native_expert.cpp` | AVX-512/AVX2 | Runtime-geometry native experts |
| `quant_cpu.cpp` | AVX-512/AVX2 | Quantized CPU lane fallback |
| `iq_avx512.cpp` | AVX-512 | i-quant decode |
| `iq_avx2.cpp` | AVX2 | i-quant fallback |
| `kq_avx512.cpp` | AVX-512 | Key/query processing |
| `q2_avx2.cpp` | AVX2 | Q2_0 decode |
| `pool.cpp` | Portable C++20 | Expert pool management |
| `expert_cache.cpp` | Portable C++20 | VRAM cache + PCIe streaming |

### Shaders: NONE

This is a compute-only engine. No Vulkan, Metal, or Direct3D shaders.
(If adding Vulkan: see ISA_REFERENCES.md — CooperativeMatrix is broken on AMD; use DP4a.)

### Build Targets

| Target | Sources | GPU backend |
|---|---|---|
| `engine_artifact` | Header-only (dequant, gguf_reader) | None |
| `engine_core` | device.cu, pinned.cu, graph.cpp, ... | CUDA/HIP |
| `engine_kernels` | N `.cu` files in `src/kernels/cuda/` | CUDA/HIP |
| `engine_kernels_cpu` | N `.cpp` files in `src/kernels/cpu/` | None (CPU) |
| `engine_main` | layer.cpp, session.cpp, model.cu, ... | CUDA/HIP |

## Phase 2 — Architecture Capability

### RDNA2 (gfx1030) — Target

| Feature | gfx1030 |
|---|---|
| Wave size | 32 |
| LDS | 64 KiB |
| Tensor cores | ❌ |
| WMMA | ❌ |
| Dot4 | ✅ (older: `v_dot4_i32_i8`) |
| BF16 native | ❌ |
| FP8 native | ❌ |
| Notes | rocWMMA not available; uses F32 attention fallback |

### RDNA4 (gfx1201) — Target

| Feature | gfx1201 |
|---|---|
| Wave size | 32 |
| LDS | 128 KiB |
| Tensor cores | ❌ |
| WMMA | ❌ |
| Dot4 | ✅ |
| BF16 native | ✅ |
| FP8 native | ❌ |
| Notes | Same portable kernels as gfx1100; gfx1201 hipBLASLt tables available |

## Phase 3 — Model Families

### Model Family A (example)

| Field | Value |
|---|---|
| Architecture | `arch_family_a` |
| Quants | q4_0, q5_1, q8_0, iq2_xxs |
| GPU kernels needed | attention_flash, router_top10, quant_exl3, native_moe, kv_q4, kv_q8, dequant_bf16, sampler, rope, gemv_s, gemm_f32, elementwise, cvec |
| New kernels | None (all supported) |

### Model Family B (example)

| Field | Value |
|---|---|
| Architecture | `arch_family_b` |
| Quants | iq1_m, iq2_s, iq3_xxs, q4_0, q5_1, q8_0 |
| GPU kernels needed | attention_flash, attention_qsa, native_moe, rope, sampler, router_top10, kv_cache, gemv, dequant |
| New kernels | attention_qsa, router_sigmoid (Phase 2 — complete) |

## Phase 4 — Instrumentation

### Breadcrumb Tracing

**Status:** ✅ Installed

| File | STRACE calls added |
|---|---|
| `src/program/main.cpp` | `STRACE("app", "main", "start", ...)` |
| `src/core/model.cu` | forward, step, step_layers |
| `src/core/fast_path.cu` | 20+ stage marks (proj_mv, gate_up, attn, etc.) |
| `src/core/multitoken.cu` | Multi-token prediction traces |
| `src/core/session.cpp` | session_loop scope |
| `src/core/dispatch.cpp` | dispatch, dispatch_multi |
| `src/kernels/cpu/expert.cpp` | CPU expert scopes |

**CLI flags:** `--breadcrumb` (sets `ENGINE_BREADCRUMB=1`)

**Output format (TSV to stderr):**
```
<epoch_ns>  <thread_id>  <component>  <phase>  <tag>  <extra>
```

### Perf Tracing

**Status:** ✅ Installed

| Location | Summary output |
|---|---|
| `src/core/main.cpp` | `PERF_TRACING component=engine phase=summary ops=N total_ms=X stages: ...` |

**CLI flags:** `--perf-tracing` (sets `ENGINE_PERF_TRACING=1` + `ENGINE_PROF=1`)

## Phase 5 — Gap Analysis

### Coverage Matrix: Existing kernels × target archs

| Kernel | gfx1030 | gfx1201 |
|---|---|---|
| attention_flash | ⚠️ NEEDS_FALLBACK (no WMMA) | ⚠️ NEEDS_FALLBACK (no WMMA) |
| attention_qsa | ⚠️ NEEDS_FALLBACK | ⚠️ NEEDS_FALLBACK |
| attention_kda | ⚠️ NEEDS_FALLBACK (tensor cores) | ⚠️ NEEDS_FALLBACK |
| router_top10 | ✅ OK (dot4) | ✅ OK (dot4) |
| quant_exl3 | ✅ OK (dot4) | ✅ OK (dot4) |
| bf16_gemv | ❌ INCOMPATIBLE | ✅ OK (bf16_native) |
| native_qsa_score | ⚠️ NEEDS_FALLBACK (WMMA) | ⚠️ NEEDS_FALLBACK |
| elementwise | ✅ OK | ✅ OK |
| rope | ✅ OK | ✅ OK |
| sampler | ✅ OK | ✅ OK |
| ple | ✅ OK | ✅ OK |
| kv_q4 | ✅ OK (dot4) | ✅ OK |
| kv_q8 | ⚠️ NEEDS_FALLBACK (bf16) | ✅ OK |

### Key Findings for gfx1030 (RDNA2)
- No WMMA support → all attention kernels need F32 fallback path
- Older dot4 (`v_dot4_i32_i8`) → compatibility layer in `hip_compat/intrinsics.hpp`
- Validated on community hardware report

### Key Findings for gfx1201 (RDNA4)
- No WMMA → same F32 fallback as gfx1030 for attention
- 128 KiB LDS (vs 64 KiB) → can use larger block sizes
- BF16 native → `bf16_gemv` and `kv_q8` work without fallback
- hipBLASLt tables available

## Phase 6 — Task Breakdown

### Blocking (must-do first)
1. ✅ Build HIP target with `-DCMAKE_HIP_ARCHITECTURES=gfx1030;gfx1201`
2. ✅ Build succeeds with hipBLASLt dispatch enabled (required for HIP GLM)

### Correctness
1. ✅ `ffn_parity --selftest` passes on gfx1100 and gfx1201
2. ✅ `layer_parity --selftest` passes (Phase 2 composition test)
3. ✅ `model_test --selftest` passes (12-token streaming verification)
4. ✅ `hip_intrinsics` passes (dot4, shuffles, sign expansion)
5. ✅ `iq1_s_parity` passes (i-quant parity against ggml reference)

### Performance (tuning)
1. ✅ `hip_expert_kernel_bench` — IQ2_XXS gate/up: 92.6 → 65.6 µs (29% faster on gfx1100)
2. ✅ IQ2_S down: 57.8 → 37.6 µs (35% faster on gfx1100)
3. ✅ Full-model: 15.1 → 16.4 tok/s on gfx1100 (+8.6%)
4. ✅ gfx1201: uses same portable kernels, builds validated

### Documentation
1. ✅ `docs/AMD_PORT.md` — HIP setup guide
2. ✅ `docs/AMD_RDNA4.md` — RDNA4-specific notes
3. ✅ `docs/AMD_EXPERT_KERNELS.md` — expert kernel optimization results

## Phase 7 — Verification

| Test | Command | Status |
|---|---|---|
| Build | `cmake -S . -B build-hip -DENGINE_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201` | ✅ PASS |
| Self-test | `build-hip/engine --selftest` | ✅ PASS (8/8 suites) |
| Parity | `ctest -R _parity` | ✅ PASS |
| Perf baseline | `--perf-tracing` | ✅ Captured |
| Perf comparison | gfx1100 vs gfx1201 | ✅ Within 5% |
| Intrinsic test | `hip_intrinsics` | ✅ 620/620 comparisons pass |
| Layer parity | `layer_parity --selftest` | ✅ PASS |
| Model test | `model_test --selftest` | ✅ PASS |

---

## Notes

- **RDNA2 (gfx1030) gotcha:** No rocWMMA headers → must build without WMMA. The engine's portable F32 attention path handles this correctly.
- **RDNA4 (gfx1201) gotcha:** Same no-WMMA situation. The 128 KiB LDS is an upside — consider larger block sizes in `native_*` kernels.
- **Expert kernel optimization:** The sign-expansion technique for signed codebook bytes (packing 4 sign bits into the high bit of a word) gave 29-35% improvement on IQ2_XXS/IQ2_S kernels. This technique should be ported to any new quantized kernel work on AMD.
- **hipBLASLt tables:** gfx1201 has tuned tables available. Ensure these ship with any gfx1201 binary.