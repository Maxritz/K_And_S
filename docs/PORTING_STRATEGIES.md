# Porting Strategies — Copy, Adapt, or Rewrite?

> **Mode**: `{{MODE}}` (strict | loose)
> **Engine**: {{ENGINE}}
> **Source arch**: {{SOURCE_ARCH}} → **Target arch**: {{TARGET_ARCH}}
> **Language**: {{SOURCE_LANG}} → {{TARGET_LANG}}

---

## Three Approaches to Porting

When porting a kernel or shader from one GPU architecture to another (or from one
backend to another), there are three strategies. The choice between them is a
trade-off between **speed**, **correctness**, and **performance** — and which one
you should pick depends on the mode.

| Strategy | Speed | Correctness | Performance | When to use |
|---|---|---|---|---|
| **Copy** (literal port) | ⚡ Fastest | ✅ High (if done right) | ⚠️ May miss arch optimizations | Same backend (CUDA→CUDA), new arch, same ISA family |
| **Adapt** (method-level port) | ⚡⚡ Medium | ✅ High (with parity tests) | ✅ Good | Different backend (CUDA→HIP), similar arch features |
| **Rewrite** (from-scratch methodology) | ⚡⚡⚡ Slow | ⚠️ Must verify | ✅✅ Best possible | Different paradigm (WMMA→F32), or no existing reference |

---

## Detailed Strategy Matrix

### 1. Copy — "Port the code, keep the method"

**What it means:**
- Translate syntax from source to target (e.g., CUDA intrinsics to HIP intrinsics)
- Keep the same algorithm, same memory access pattern, same instruction sequence
- Only change the API calls (`cudaMalloc` → `hipMalloc`, `__syncthreads()` → `__syncthreads()`)

**Example:** Porting a CUDA CUDA kernel to HIP:
```cpp
// CUDA:
__global__ void vec_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// HIP (literal copy):
__global__ void vec_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

**When to use:**
- Same backend family (CUDA sm_80 → CUDA sm_89)
- Same ISA family (HIP gfx1100 → HIP gfx1201)
- Only minor feature differences (one has WMMA, other doesn't — fallback is acceptable)
- You have a verified reference and just need it on the new arch

**K_and_S mode guidance:**
- **Strict**: Only acceptable for same-backend, same-ISA-family ports. Must verify all intrinsics available.
- **Loose**: Good for "getting it running" quickly. Mark with `TODO(kands:strict-review)`.

**Verification required:**
- [ ] All intrinsics exist on target arch (check ISA references)
- [ ] Memory model is compatible (LDS size, shared mem banks)
- [ ] Wave size assumptions hold (32 vs 64)
- [ ] Parity test passes against reference

---

### 2. Adapt — "Port the method, rewrite the code"

**What it means:**
- Keep the same algorithm/methodology
- Change the implementation approach to fit the target arch's features
- E.g., CUDA WMMA → HIP rocWMMA; or tensor cores → dot4 instructions

**Example:** Porting NVIDIA tensor core MMA to AMD DP4/WMMA:
```cpp
// CUDA (sm_80+):
__global__ void mma_kernel(half* a, half* b, float* c) {
    // Uses mma.m8n8k4.row.col.f16.f32 instruction
    __asm__ volatile("mma.m8n8k4..." : ...);
}

// HIP (gfx1100+, via rocWMMA or dot4):
__global__ void mma_kernel(half* a, half* b, float* c) {
    // Uses __builtin_amdgcn_wave_dot_f32 or rocWMMA intrinsics
    // Same algorithm, different instruction sequence
}
```

**When to use:**
- Different backend (CUDA → HIP, CUDA → oneAPI SYCL)
- Arch features differ significantly (tensor cores vs dot4 vs DPAS)
- You understand the algorithm but need a different instruction sequence

**K_and_S mode guidance:**
- **Strict**: Verify the adapted code produces bit-identical results
- **Loose**: Use F32 fallback as interim, optimize later

**Verification required:**
- [ ] Algorithm produces same numerical results (parity test)
- [ ] All target arch features used correctly (wave size, LDS alignment)
- [ ] Performance is acceptable (measure tok/s, compare to reference)

---

### 3. Rewrite — "Use a different method entirely"

**What it means:**
- The target arch has fundamentally different performance characteristics
- The original method is wrong for the new arch
- You need to rethink the algorithm, memory layout, or tiling strategy

**Example:** Porting a CUDA WMMA-based attention kernel to gfx1030 (no WMMA):
```cpp
// Original (CUDA sm_80, uses tensor cores):
// - Tile: 16x16x4 FP16 via mma.m16n8k8
// - Relies on tensor core throughput
// - Uses LDmatrix for prefetch

// Rewrite for gfx1030 (RDNA2, no WMMA):
// - Tile: 32x32x1 via v_dot2 + fma chains
// - Uses v_dot4_i32_i8 for integer ops (different instruction set)
// - Relies on scalar ALU + shared memory instead of tensor cores
// - Different tiling strategy (larger blocks, fewer warps)
```

**When to use:**
- No feature parity (tensor cores → no tensor cores)
- The source algorithm is fundamentally incompatible (e.g., CUDA graph → Vulkan compute)
- The target arch has a superior method you should use (e.g., AMD's packed sign expansion vs scalar ops)

**K_and_S mode guidance:**
- **Strict**: Must re-verify against the full parity harness — this is effectively a new kernel
- **Loose**: Get a working F32 fallback first, then optimize

**Verification required:**
- [ ] Full parity test (not just "looks right")
- [ ] Performance matches or exceeds target expectations
- [ ] All edge cases (null experts, boundary tiles, etc.) covered

---

## How to Choose (Decision Flow)

```
Question 1: Is the target the same backend (CUDA→CUDA, HIP→HIP)?
├─ YES → Same backend family. Can you use Copy?
│  ├─ All intrinsics available on target arch? → YES: Copy
│  └─ NO → Adapt (find equivalent instructions)
└─ NO → Different backend. Can you use Adapt?
   ├─ Algorithm makes sense on target (attention, MoE, KV cache)? → YES: Adapt
   └─ Algorithm is backend-specific (CUDA graphs → Vulkan)? → NO: Rewrite
```

**Quick decision table:**

| Source → Target | Strategy | Rationale |
|---|---|---|
| CUDA sm_80 → CUDA sm_89 | Copy | Same backend, same ISA, minor feature changes |
| CUDA sm_80 → HIP gfx1100 | Adapt | Different backend, similar arch features (both have dot4, WMMA) |
| CUDA sm_80 → HIP gfx1030 | Adapt+Rewrite | No WMMA on RDNA2 — must rewrite attention path, can adapt router/MoE |
| CUDA sm_80 → HIP gfx908 (CDNA) | Rewrite | Wave64 vs wave32, different scheduling |
| CUDA sm_80 → oneAPI Arc | Rewrite | DPAS instead of MMA, different memory model |
| Python (PyTorch) → C++ (CUDA) | Rewrite | Different abstraction level |
| C++ (CUDA) → Rust (CUDA FFI) | Adapt | Same kernel, different host-side memory management |

---

## Strict vs Loose Mode Guidance

### Strict Mode

In strict mode, you must follow ALL of these:

1. **No placeholder/stub code** — every kernel must produce real output. If a feature is
   missing, implement a correct (if slower) fallback path — never a stub.
2. **Every kernel** gets an audit checklist (see `docs/DOX_STYLE.md`)
3. **Copy strategy** allowed only if intrinsics are verified via ISA references (`kands://isa-references`)
4. **Adapt strategy** requires bit-exact parity with reference
5. **Rewrite strategy** requires new parity test harness
6. **Perf baseline** must be captured before changes
7. **All arch branches** (`#if __HIP__`, `#ifdef __CUDA_ARCH__`) must be resolved
8. **FMA contraction** must be controlled explicitly (`--fmad=false` or `-ffp-contract=off`)

### Loose Mode

In loose mode, you can:

1. **Copy strategy** with minimal changes — "make it compile and run"
2. **Use F32 fallback** for missing arch features (WMMA → F32 GEMM)
3. **Mark uncertain paths** with `TODO(kands:strict-review)`
4. **Best-effort parity** — get it working, optimize later
5. **Skip perf baseline** until after it runs
6. **Assume intrinsics are portable** — check later in strict pass

**Note:** Even in loose mode, never use stub/placeholder code. If a kernel can't compile
because a feature is missing, use the correct F32 fallback path.

---

## Security & Verification Requirements

Regardless of mode, these rules always apply:

### Autonomous Port Security
- `autonomous_port(github_url=...)` is **sandboxed by default** (file count limit, timeout, no script execution)
- For untrusted repos, additionally use **container isolation** + **dependency vulnerability scanning**
- Read `kands://diagnostics` → "Secure Autonomous Operations" section for risk levels

### Numerical Equivalence (CPU-as-Oracle)
Every ported kernel MUST be verified against the CPU oracle:

1. **Before porting** — capture CPU reference output (seed=42, deterministic)
2. **After porting** — run identical inputs through GPU path
3. **Compare** using `verify_numerical_equivalence(arch=..., workload=..., source_values=[...], target_values=[...])`
4. **Common divergence causes:**
   - FMA contraction → use `--fmad=false` (CUDA), `-ffp-contract=off` (GCC/Clang)
   - Reduction order → fix block size or use Kahan summation
   - Transcendental functions → use libm explicitly
   - WMMA/DP4a edge cases → verify signed/unsigned operand handling

### No Placeholder/Stubs Policy
- **Never** add stub, demo, or "fake it works" code
- If a kernel can't run on the target arch, use a **correct F32 fallback**
- If a feature is missing, **document it** and implement a real (if slower) path
- Every kernel must produce **real, verifiable output**

---

## Worked Example Table (Generic)

> Replace with your engine's actual kernel names. This shows the decision matrix.

| Kernel | Source arch | Target arch | Strategy | Mode |
|---|---|---|---|---|
| `flash_attn` | sm_80 (WMMA) | gfx1201 (no WMMA) | Adapt (F32 fallback) | strict: parity test required |
| `router_top10` | sm_80 (dot4) | gfx1030 (dot4, older) | Copy (verify v_dot4_i32_i8) | strict: ISA check |
| `exl3` | sm_80 (mma) | gfx1100 (dot4) | Adapt (dot4 instead of mma) | strict: parity test |
| `bf16_gemv` | sm_80 (tf32) | gfx1030 (no bf16) | Rewrite (FP32 path) | loose: F32 fallback first |
| GLM-5.3 `glm_dsa` | New (Phase 2) | gfx1201 | Rewrite | strict: new parity harness |
| DS4 `ds4_kernels` | gfx1100 | gfx1030 | Adapt | strict: verify dot4 version |

**Use `kands://findings/maya` for the complete kernel-by-kernel breakdown.**
