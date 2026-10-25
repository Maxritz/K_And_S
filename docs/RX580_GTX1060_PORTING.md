# RX 580 / GTX 1060 — DirectX / Vulkan / OpenCL Reality Check

> These are ~2016-2017 era mid-range cards. Neither has ROCm/HIP as a primary driver path.
> Neither has machine-readable ISA. Both rely on DirectX / Vulkan / OpenCL for cross-vendor compute.

---

## RX 580 (Polaris, gfx803, GCN 4th-gen)

| Property | Value |
|---|---|
| Architecture | Polaris (GCN 4th-gen) |
| ISA | GCN 4 |
| gfx ID | gfx803 (fiji) |
| Wave size | 64 |
| LDS per CU | 64 KiB |
| Matrix cores | No (no WMMA, no mfma) |
| Dot product | v_dot4_i32_i8 (4-way int8) + v_fma_f32 |
| Open kernel driver | AMDGPU (in-tree Linux kernel, always supported) |
| ROCm/HIP support | ROCm supports gfx803 but it's **legacy** — ROCm 5.x was the last good era for Polaris. ROCm 6+ has limited gfx803 support. ROCm 7/ROCm 10 targeting gfx803 is not the main use case. |
| Machine-readable ISA | **No** — GCN 4 ISA is PDF-only, not GPUOpen XML |
| IsaDecoder API | **No** — GPUOpen IsaDecoder (isa_spec_manager) only parses RDNA 1–4 + CDNA 1–5 XML, not GCN |

### What RX 580 actually runs

- **DirectX 12** — Yes, fully supported. Polaris supports DX12 Feature Level 12_0 (or 12_1 depending on drivers). DXC can compile HLSL → DXIL for Polaris. This is the primary compute path on Windows.
- **Vulkan** — Yes, fully supported. Mesa radv driver (open-source) and AMDVLK both support Polaris. Vulkan 1.2+ on Polaris. CooperativeMatrix is **broken on AMD drivers** (radv/AMDVLK) — must use DP4a (GL_KHR_shader_integer_dot_product) for matmul.
- **OpenCL** — Yes, ROCm's OpenCL stack (ocl-runtime) historically supported gfx803. OpenCL 2.0 on Polaris. This is the older path — ROCm OpenCL is being deprecated in favor of HIP, but on gfx803 it's still the most mature option.
- **HIP/ROCm** — Supported but legacy. ROCm can target gfx803 but you're on older ROCm versions. The HIP runtime works, the hipcc compiler targets gfx803 via `--offload-arch=gfx803`. The issue is ROCm's ongoing shift — gfx803 is not a priority arch for ROCm 6/7/10.
- **CUDA** — No. NVIDIA only.

### ISA reality

RX 580 instruction encoding is **not** in GPUOpen's machine-readable ISA XML set (that's RDNA 1–4 + CDNA 1–5 only). To verify what your HIP kernel compiled to:

```bash
# Disassemble HIP kernel for gfx803 (fiji)
llvm-objdump -mcpu=fiji -d kernel.o

# Or with roc-objdump (ROCm toolchain)
roc-objdump -disassemble kernel.o
```

Then you read the GCN 4 ISA PDF manually. There is no `isa_spec_manager` XML for gfx803. You are on your own with the PDF.

### v_dot4 on RX 580

- `v_dot4_i32_i8` — available on gfm803 (fiji). 4-way signed int8 dot product into 32-bit.
- `v_dot4_i32_iu8` — unsigned variant also available.
- Neither is a "tensor core" — it's a scalar/vector dot-product instruction, not a matrix multiply.
- For matmul on RX 580: use DP4a-style (4x int8 dot products per instruction) — same pattern as Vulkan DP4a fallback.

### OpenCL on RX 580

ROCm's OpenCL runtime (ocl-runtime) supports gfx803. OpenCL 2.0 features: SVM, pipes, atomics. The OpenCL kernel language is different from HIP — you write `__kernel void` functions with `__global`/`__local` address spaces, not `__global__ __device__` CUDA-style. Porting CUDA → OpenCL on RX 580 means:

- `cudaMalloc` → `clCreateBuffer`
- `cudaMemcpy` → `clEnqueueCopyBuffer`
- `cudaLaunchKernel` → `clEnqueueNDRangeKernel`
- `__global__` → `__kernel`
- `__shared__` → `__local`
- `__syncthreads()` → `barrier(CLK_LOCAL_MEM_FENCE)`
- `__shfl_sync` → there is no warp shuffle in OpenCL 2.0 standard — you need sub-group operations which Polaris may or may not support. On Polaris, use work-group shared memory + manual lane communication.

OpenCL is the most mature compute path on RX 580 for Linux. But it's also the most different from CUDA syntactically.

---

## GTX 1060 (Pascal, GM206, sm_61)

| Property | Value |
|---|---|
| Architecture | Pascal |
| ISA | Pascal (SM 6.1) |
| Compute capability | sm_61 / compute_61 |
| Chip | GM206 |
| Tensor cores | No (sm_70+ only) |
| Matrix instructions | No mma.sync (sm_70+), no WMMA |
| Warp shuffle | `__shfl_sync`, `__ballot_sync` available |
| Open kernel driver | **No** — NVIDIA open kernel modules require Turing (sm_7.5+). GTX 1060 uses proprietary NVIDIA driver on Linux. |
| Machine-readable ISA | **No** — NVIDIA ISA docs are PDF-only, never machine-readable |
| IsaDecoder API | **No** — no NVIDIA ISA decoder exists (open source or otherwise) |

### What GTX 1060 actually runs

- **DirectX 12** — Yes, fully supported. Pascal supports DX12 Feature Level 12_0. DXC → DXIL for Pascal. This is the primary compute path on Windows.
- **Vulkan** — Yes, fully supported. GTX 1060 supports Vulkan 1.1/1.2 (driver-dependent). CooperativeMatrix is **supported on NVIDIA Vulkan drivers** (unlike AMD) — you can use `GL_NV_cooperative_matrix` on GTX 1060 if the driver is new enough. But GTX 1060 has no tensor cores, so CooperativeMatrix is only useful if you're doing FP16 matmul via the fp16 cooperative matrix path (not the int8/bf16 path that requires tensor cores).
- **OpenCL** — Yes, NVIDIA's OpenCL stack supports sm_61. OpenCL 1.2 (basic) + some 2.0 features. NVIDIA's OpenCL is less commonly used than CUDA for GTX 1060 but it's available.
- **CUDA** — Yes, fully supported. This is the primary compute path on Linux/Windows. GTX 1060 works with CUDA toolkit, nvcc targeting sm_61.
- **HIP/ROCm** — No. AMD only.

### ISA reality

NVIDIA **never** published machine-readable ISA for any architecture (not even for Turing+ where they opened the kernel modules). GTX 1060 ISA is in NVIDIA's Pascal PTX/SASS reference PDF docs. To verify what your CUDA kernel compiled to:

```bash
# Disassemble CUDA kernel for sm_61
cuobjdump --dump-sass kernel.cubin
# or
nvdisasm kernel.cubin
```

Then you read the SASS output manually and cross-reference against the Pascal PTX/SASS PDF. There is no open ISA decoder for NVIDIA. Period.

The closest thing to an "open GPU ISA decoder" for NVIDIA is:
- Reading SASS output manually
- Using `cuobjdump`/`nvdisasm` (proprietary tools included with CUDA toolkit)
- NVIDIA's open-gpu-kernel-modules expose kernel-space GPU interfaces but **not** an ISA decoder

### What GTX 1060 CUDA kernels look like

Since GTX 1060 is sm_61 (Pascal), CUDA kernels targeting it have:
- No `mma.sync` (that's sm_70 Tesla/Volta+)
- No `__shfl_sync` with mask? Actually `__shfl_sync` was introduced in sm_70... on sm_61 you use `__shfl` (no sync variant)
- `__ballot_sync` — same, sm_70+ feature. On sm_61 use `__ballot` (no sync)
- `__activemask` — sm_70+ only
- `__threadfence` / `__threadfence_block` — available
- `__ldg` — available
- `__float2half_rn` / `__half2float` — available (FP16)
- `__float2bfloat16_rn` — not available (BF16 is sm_80+ Ampere). On sm_61 you have to implement BF16 manually (extract exponent + mantissa, repack).
- `__dp4a` — available (4-way int8 dot product on sm_61). Same as AMD's v_dot4.
- `__hadd2`, `__hsub2`, `__mul2`, `__hadd2` — half2 operations available on sm_61.

So a GTX 1060 CUDA kernel has: basic CUDA runtime API, `__shfl` (no sync), `__ballot` (no sync), `__ldg`, `__float2half_rn`, `__dp4a`, half2 ops. No tensor cores, no mma.sync, no BF16 native, no `__activemask`, no `__shfl_sync`.

When porting from GTX 1060 CUDA → AMD HIP target:
- `__shfl_sync` → `__shfl` (drop sync variant, drop mask arg)
- `__ballot_sync` → `__ballot` (drop sync variant)
- `__float2bfloat16_rn` → manual BF16 implementation (GTX 1060 doesn't have native BF16; neither does gfx1030 — gfx1100+ has native BF16)
- `__dp4a` → `amd_mixed_dot` / `v_dot4_i32_i8` / `__builtin_amdgcn_udot4` (HIP equivalent)
- No mma.sync → no matrix core path on either side (GTX 1060 has none; if targeting gfx1030 that's also none; if targeting gfx1100+ you get WMMA but the source has no mma.sync so there's nothing to port)

---

## DirectX / Vulkan / OpenCL on both cards

### RX 580

| Backend | Support | Notes |
|---|---|---|
| **DirectX 12** | ✓ Full | DX12 FL 12_0/12_1. DXC compiles HLSL → DXIL. This is the primary Windows compute path. |
| **Vulkan** | ✓ Full | Mesa radv + AMDVLK. Vulkan 1.2+. **CooperativeMatrix BROKEN on AMD drivers** — use DP4a (GL_KHR_shader_integer_dot_product). |
| **OpenCL** | ✓ Mature | ROCm ocl-runtime supports gfx803. OpenCL 2.0. This is the most mature Linux compute path for Polaris. |
| **HIP/ROCm** | ✓ Legacy | Supported but ROCm 6/7/10 deprioritize gfx803. Use ROCm 5.x-era toolchain for best gfx803 support. |
| **CUDA** | ✗ | NVIDIA only. |

### GTX 1060

| Backend | Support | Notes |
|---|---|---|
| **DirectX 12** | ✓ Full | DX12 FL 12_0. DXC → DXIL. Primary Windows compute path. |
| **Vulkan** | ✓ Full | Vulkan 1.1/1.2. **CooperativeMatrix SUPPORTED on NVIDIA Vulkan** (GL_NV_cooperative_matrix) — but no tensor cores, so only FP16 matmul path is usable, not int8/bf16. |
| **OpenCL** | ✓ Available | NVIDIA OpenCL stack supports sm_61. OpenCL 1.2 + partial 2.0. Less commonly used than CUDA. |
| **CUDA** | ✓ Full | Primary Linux/Windows compute path. CUDA toolkit + nvcc targeting sm_61. |
| **HIP/ROCm** | ✗ | AMD only. |

---

## Cross-vendor: when RX 580 and GTX 1060 meet

### Common ground

Both cards:
- Have **no tensor cores** (GTX 1060 has no tensor cores; RX 580 has no matrix cores at all)
- Have **int8 dot product** — GTX 1060 has `__dp4a`, RX 580 has `v_dot4_i32_i8`. Both do 4-way int8 dot per instruction.
- Support **Vulkan** with DP4a fallback for matmul (RX 580 has no CooperativeMatrix on AMD; GTX 1060 has CooperativeMatrix but no tensor cores so it's FP16-only there)
- Support **DirectX 12** with DXC → DXIL
- Support **OpenCL** (RX 580 via ROCm OpenCL, GTX 1060 via NVIDIA OpenCL)
- Are **not** primary ROCm/HIP targets (RX 580 is legacy gfx803; GTX 1060 is NVIDIA, can't run HIP at all)
- Have **no machine-readable ISA** (RX 580: GCN 4 PDF; GTX 1060: Pascal PDF; neither has an open ISA decoder)

### The DX12/Vulkan/OpenCL path is the common one

If you have a CUDA kernel from GTX 1060 and want it to run on RX 580:

1. **CUDA → HLSL → DXIL → run on RX 580 via DirectX 12** — this is the most direct path. Write the kernel in HLSL (DirectX shader language), compile with DXC to DXIL, run on RX 580 via D3D12 compute queue. The kernel semantics translate well because DX12 compute shaders are similar to CUDA/HIP in structure (thread groups, shared memory, barriers).

2. **CUDA → HLSL → SPIR-V → run on RX 580 via Vulkan** — cross-compile HLSL to SPIR-V via DXC (`dxc -spirv`), then run on RX 580 via Vulkan. Or write directly in GLSL (.comp) and compile with glslc. On RX 580 use DP4a for matmul (no CooperativeMatrix on AMD).

3. **CUDA → OpenCL → run on RX 580 via ROCm OpenCL** — translate CUDA kernels to OpenCL C. This is the most mature Linux path for RX 580 because ROCm OpenCL has the best gfx803 support. But OpenCL syntax is different from CUDA — you need to rewrite the kernel, not just swap APIs.

4. **CUDA → HIP → run on RX 580 via HIP/ROCm** — translate CUDA kernels to HIP C++. This works on RX 580 but you're on legacy ROCm for gfx803. HIP targets gfx803 via `--offload-arch=gfx803`. This is the path K_and_S primarily documents, but for RX 580 specifically, the HIP/ROCm story is weaker than DirectX/Vulkan/OpenCL.

### GTX 1060 as source → RX 580 as target

If the source is a GTX 1060 CUDA kernel (sm_61) and the target is RX 580 (gfx803):

- Source has: `__shfl`, `__ballot`, `__ldg`, `__float2half_rn`, `__dp4a`, half2 ops, basic CUDA runtime. No mma.sync, no BF16, no tensor cores.
- Target (RX 580) has: `v_dot4_i32_i8` (equivalent to `__dp4a`), `v_fma_f32`, wave64, 64 KiB LDS. No matrix cores, no WMMA, no mfma. GCN 4 ISA is PDF-only (no machine-readable ISA, no IsaDecoder XML).
- Port path: CUDA → HIP (hipcc targeting gfx803) using `--offload-arch=gfx803`. hipify-perl can do the bulk of the translation. Manual fixes: `__shfl_sync` → `__shfl` (no mask), `__ballot_sync` → `__ballot`, `__dp4a` → `v_dot4_i32_i8` / `amd_mixed_dot`, `__float2bfloat16_rn` → manual BF16 (both GTX 1060 and gfx803 lack native BF16).
- Verification: `llvm-objdump -mcpu=fiji -d kernel.o` + GCN 4 ISA PDF (no machine-readable ISA, no IsaDecoder).

### RX 580 as source → GTX 1060 as target

Less common (usually you go from NVIDIA source to AMD target) but possible:
- Source is OpenCL / Vulkan / DirectX on RX 580
- Target is CUDA on GTX 1060
- Port path: OpenCL C → CUDA C++ (manual rewrite, similar structure), or Vulkan GLSL → CUDA (cross-compile path), or DirectX HLSL → CUDA (manual rewrite)
- The DX12/Vulkan side is similar on both cards — DXIL and SPIR-V work on both. The main difference is CUDA vs HIP/OpenCL on the host API side.

---

## Summary table: RX 580 vs GTX 1060 compute paths

| | RX 580 (Polaris gfx803) | GTX 1060 (Pascal sm_61) |
|---|---|---|
| **Primary compute (Linux)** | HIP/ROCm (legacy) or OpenCL (mature) | CUDA (primary) or OpenCL (available) |
| **Primary compute (Windows)** | DirectX 12 (DXC → DXIL) or Vulkan | DirectX 12 (DXC → DXIL) or CUDA |
| **Vulkan** | ✓ Vulkan 1.2 (radv/AMDVLK). CoopMat BROKEN — use DP4a. | ✓ Vulkan 1.1/1.2. CoopMat SUPPORTED (FP16 only, no tensor cores). |
| **DirectX 12** | ✓ DX12 FL 12_0/12_1. DXC → DXIL. | ✓ DX12 FL 12_0. DXC → DXIL. |
| **OpenCL** | ✓ ROCm OpenCL 2.0 (most mature for Polaris) | ✓ NVIDIA OpenCL 1.2+partial 2.0 |
| **HIP/ROCm** | ✓ Legacy (ROCm 5.x era best for gfx803) | ✗ AMD only |
| **CUDA** | ✗ NVIDIA only | ✓ Full (sm_61) |
| **Tensor cores / matrix cores** | ✗ No | ✗ No |
| **int8 dot product** | ✓ v_dot4_i32_i8 (GCN 4) | ✓ __dp4a (Pascal sm_61) |
| **BF16 native** | ✗ No | ✗ No (sm_80+ Ampere) |
| **FP16 native** | ✓ Yes (v_cvt_f16_f32 etc.) | ✓ Yes (__float2half_rn etc.) |
| **Machine-readable ISA** | ✗ GCN 4 is PDF-only, not in GPUOpen XML | ✗ NVIDIA ISA is PDF-only, never machine-readable |
| **ISA decoder (open source)** | ✗ GPUOpen IsaDecoder doesn't parse GCN | ✗ No NVIDIA ISA decoder exists |
| **Disassembly** | `llvm-objdump -mcpu=fiji -d kernel.o` + GCN 4 PDF | `cuobjdump --dump-sass kernel.cubin` + Pascal PTX/SASS PDF |
| **ROCm/HIP is primary path?** | No — DirectX/Vulkan/OpenCL are more mature for gfx803 | N/A — GTX 1060 is NVIDIA, can't run HIP |

---

## Key insight

RX 580 and GTX 1060 are **not** cards where the ROCm/HIP or CUDA toolchain is the primary story:

- **RX 580**: the mature compute paths are DirectX 12, Vulkan, and OpenCL (ROCm OpenCL). HIP/ROCm works but gfx803 is legacy in ROCm 6+. And there's no machine-readable ISA — you're on GCN 4 PDF + objdump.
- **GTX 1060**: the mature compute path is CUDA (Linux/Windows) or DirectX 12/Vulkan (Windows). OpenCL is available but less used. And there's no machine-readable ISA — you're on Pascal PDF + cuobjdump.

If you're porting **between** these cards (or from either to a newer AMD/NVIDIA card), the DirectX 12 and Vulkan paths are the most cross-vendor-compatible because:
- DXIL works on both cards via DXC
- SPIR-V works on both via glslc/dxc -spirv
- DP4a works on both (RX 580 as the only matmul option; GTX 1060 as a fallback alongside CooperativeMatrix FP16)

The HIP/CUDA paths are vendor-specific — HIP on RX 580 (legacy), CUDA on GTX 1060 (primary). For cross-vendor, DX12 and Vulkan are the common ground.
