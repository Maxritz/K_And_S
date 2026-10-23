# Machine-Readable GPU ISA References

> Use these to verify instruction availability, decode kernels, and compare generated code
> across architectures during porting.

---

## AMD — Machine Readable ISA

**URL:** https://gpuopen.com/machine-readable-isa/

**What it provides:** XML files describing AMD's GPU ISA for every supported architecture.
Each XML defines instructions, encodings, operands, data formats, and human-readable
description strings. Also includes the **IsaDecoder C++ API** for parsing/decoding.

**Supported architectures:**
| Arch | Product family |
|---|---|
| CDNA 5 | AMD Instinct MI450 |
| CDNA 4 | AMD Instinct MI350 |
| CDNA 3 | AMD Instinct MI300 |
| CDNA 2 | AMD Instinct MI200 |
| CDNA 1 | AMD Instinct MI100 |
| **RDNA 4** | RX 9000 series (gfx120x) |
| **RDNA 3.5** | Ryzen AI Max, Strix Halo (gfx115x) |
| **RDNA 3** | RX 7000 series (gfx110x) |
| **RDNA 2** | RX 6000 series (gfx103x) |
| RDNA 1 | RX 5000 series (gfx101x) |

**Download:** https://gpuopen.com/download/machine-readable-isa/latest/ (ZIP bundle)

**IsaDecoder API:** https://github.com/GPUOpen-Tools/isa_spec_manager

**Use for porting:** Decode compiled `.isa` / `.vega` binaries. Check whether your kernel's
generated instructions are available on the target arch (e.g., `v_dot4_i32_i8` exists on RDNA2
but not the newer packed variants on RDNA4).

---

## Intel — Portable ISA (PISA) + IGA

**PISA spec:** https://intel.github.io/pisa/

**What it provides:** A portable, stable ISA specification for Intel GPUs, spanning from
Broadwell (GEN8) through Arc (GEN12+). Defines the virtual ISA that the Intel Graphics
Compiler (IGC) emits, plus a target directive system (`.target 100` etc.) for version
selection.

**PISA generations:**
| `.target` | Product | Notes |
|---|---|---|
| 8 | Broadwell (GEN8) | |
| 9 | Skylake/Kaby Lake (GEN9/GEN9.5) | |
| 11 | Ice Lake (GEN11) | |
| 12 | Tiger Lake (GEN12) | |
| 12.1 | Tiger Lake (GEN12) | |
| 12.2 | Xe HP | |
| 12.3 | Xe HPC | |
| 12.5 | Xe HPG (Arc) | DPAS matrix instruction |

**IGA (Intel Graphics Disassembler):** Part of the
[Intel Graphics Compiler](https://github.com/intel/intel-graphics-compiler/tree/master/visa/iga)

```bash
# Clone and build IGA
git clone https://github.com/intel/intel-graphics-compiler.git
cd intel-graphics-compiler/visa/iga
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
# Disassemble:
./iga64 -d -p <device_id> kernel.isa.bin
```

**PRM (Programmer's Reference Manual):**
- Intel Arc A-Series / Data Center Flex Series: https://kiwitree.net/~lina/intel-gfx-docs/prm/acm/intel-gfx-prm-osrc-acm-vol03-gpu-overview.pdf

**Use for porting:** When porting a CUDA/HIP kernel to Intel Arc, use PISA/IGA to check
whether the matrix multiply instruction (DPAS) you need is available on the target generation,
and whether workgroup sizes / subgroup sizes match (Intel Arc uses EU subgroups of 8 threads
in SIMD-8/SIMD-16 modes, vs AMD's wave32).

---

## NVIDIA — PTX ISA

**PTX ISA docs:** https://docs.nvidia.com/cuda/parallel-thread-execution/

**PDF downloads (per CUDA version):**
- PTX ISA 9.4 (latest, CUDA 13.x): https://docs.nvidia.com/cuda/pdf/ptx_isa_9.4.pdf
- PTX ISA 8.8 (CUDA 12.6): https://docs.nvidia.com/cuda/pdf/ptx_isa_8.8.pdf
- Archive: https://developer.nvidia.com/cuda-toolkit-archive

**What it provides:** Virtual ISA specification defining the PTX assembly language and
the mapping from PTX instructions (`mma`, `ldmatrix`, `cp.async`, `tensor` core ops)
to native NVIDIA GPU instructions. PTX is the stable, cross-generation interface
that `nvcc` emits and the NVIDIA driver JIT-translates to native SASS.

**Key PTX version → SM mapping:**
| PTX ISA | CUDA toolkit | SM support |
|---|---|---|
| PTX 8.x | CUDA 12.x | sm_70 – sm_90 |
| PTX 7.x | CUDA 11.x | sm_70 – sm_89 |
| PTX 6.x | CUDA 11.x | sm_70 – sm_86 |
| PTX 9.x | CUDA 13.x | sm_80 – sm_100+ |

**Open source components:** https://developer.download.nvidia.com/compute/cuda/opensource/

**Use for porting:** Compare PTX instruction usage between source and target. PTX ISA docs
flag which instructions require tensor cores (e.g., `mma.m8n8k16`), which are
SM-80+ only (`ldmatrix`), and which are universally portable (`ld.global`, `shl`, `add`).

---

---

## Vulkan — GLSL / SPIR-V

**Vulkan spec:** https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html

**Relevant extensions for GPU compute:**
| Extension | Description | Status on AMD |
|---|---|---|
| `GL_KHR_cooperative_matrix` | Vulkan CooperativeMatrix — matrix multiply/load/store | ❌ **Broken on AMD drivers** (radv and AMDVLK) |
| `GL_KHR_shader_integer_dot_product` | DP4a / DP4-style integer dot products | ✅ Works |
| `GL_EXT_shader_explicit_degree` | Explicit compute/bounds control for loop unrolling | ✅ |
| `GL_ARB_gpu_shader_half_float` / `GL_EXT_shader_explicit_arithmetic` | BF16 FP controls | ✅ on RDNA3/RDNA4 |

**PISA (Intel) Vulkan backend:** https://intel.github.io/pisa/ — the PISA spec also defines the Vulkan SPIR-V emission layer. Use `IGA` to disassemble and verify.

**Use for porting:** When targeting Vulkan compute on AMD, **do not** use `GL_KHR_cooperative_matrix`. Instead, fall back to `GL_KHR_shader_integer_dot_product` (DP4a) for matmul. The `spv_dp4` / `spv_dp4a` instructions are the portable AMD Vulkan matmul path. NVIDIA Vulkan drivers support CooperativeMatrix natively (SM 7.5+ via driver version 535.x+).

**Verification:**
- AMD: `radv`'s `spirv` validator will warn if you use CooperativeMatrix. Build with `-DVK_USE_PLATFORM_WIN32_KHR` on Windows or `-DVK_USE_PLATFORM_XLIB_KHR` on Linux.
- NVIDIA: Use `nvdisasm` on the SPIR-V → SASS path, or `vulkaninfo --show-formats` to check supported formats.

---

## Quick Reference Card

| Vendor | Machine-Readable ISA | Disassembler | Arch range |
|---|---|---|---|
| AMD | https://gpuopen.com/machine-readable-isa/ (XML) | IsaDecoder API | RDNA 1–4, CDNA 1–5 |
| Intel | https://intel.github.io/pisa/ (spec) | IGA (`intel-graphics-compiler/visa/iga`) | GEN8 – Arc (GEN12+) |
| NVIDIA | https://docs.nvidia.com/cuda/pdf/ptx_isa_*.pdf | `cuobjdump --disassemble` | sm_70 – sm_100+ |

## When Porting, Consult ISA For:

1. **Dot-product instructions**: Does `v_dot4_i32_i8` exist on this AMD arch? Does NVIDIA's
   SM have `dp4a`? Does Intel Arc have DPAS?
2. **Tensor/Math core ops**: Which arches support `mma`/`wmma`/`tcgen`/DPAS?
3. **Shared memory instructions**: `ldmatrix` (NVIDIA SM-80+), `smem` loads (AMD),
   `shared` access (Intel) — what's the bandwidth and addressing model?
4. **Warp/wave instructions**: `shfl` (NVIDIA), `v_swappc_b64` (AMD), `send` (Intel)
5. **Barriers**: `bar` (NVIDIA PTX), `__syncthreads` → native barrier (AMD),
   `nbarrier` (Intel)

## Workflow Integration

```
kands-mcp> read kands://isa-references
→ get this file

kands-mcp> query_kernel_features(repo="/path/to/engine", pattern="mma|dot4|wmma|dpas")
→ find which kernels use math instructions

kands-mcp> compare_archs(arch_a="gfx1100", arch_b="gfx1030")
→ see that WMMA/tensor_cores differ

kands-mcp> check_arch_compatibility(kernel_list=["native_flash_attn"], arch="gfx1030")
→ "⚠️ NEEDS_FALLBACK: needs WMMA" → consult ISA docs for F32 fallback path
```
