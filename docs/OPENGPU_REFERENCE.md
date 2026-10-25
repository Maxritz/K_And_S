# OpenGPU / GPUOpen Reference

## NVIDIA Open GPU Kernel Modules

**Repository:** https://github.com/NVIDIA/open-gpu-kernel-modules  
**Version:** R515 driver series (615.78.08)  
**License:** GPL/MIT dual license  
**Supported GPUs:** Turing (RTX 20xx, GTX 16xx) and later  
**Supported Kernels:** Linux kernel 4.15+  
**Architectures:** x86_64, aarch64

### What it is

This is the open-source release of NVIDIA's Linux GPU kernel modules. It provides the same userspace ABI as the proprietary driver — CUDA, OpenGL, Vulkan all work the same way on top of it. The difference is purely in the kernel-space code.

### Relationship to CUDA/HIP porting

This is **not directly relevant** to CUDA→HIP kernel porting (that's userspace). The CUDA toolkit does not change. However, it is relevant when:

- You're building custom Linux GPU drivers
- You need to debug UVM (unified virtual memory) issues at the kernel level
- You're doing GPU passthrough / VFIO work
- You're comparing NVIDIA vs AMD kernel driver architectures

### How to use

```bash
# Build
make modules -j$(nproc)

# Install (uninstall proprietary first)
make modules_install -j$(nproc)

# Requires GSP firmware from matching driver release
sh ./NVIDIA-Linux-x86_64-615.78.08.run --no-kernel-modules
```

### Key components

| Module | Purpose |
|---|---|
| `nvidia.ko` | Core GPU management, memory, contexts |
| `nvidia-drm.ko` | Direct Rendering Manager (X/Wayland) |
| `nvidia-modeset.ko` | Display mode setting |
| `nvidia-uvm.ko` | Unified Virtual Memory |

### Documentation

See the NVIDIA GPU driver README: https://us.download.nvidia.com/XFree86/Linux-x86_64/615.78.08/README/kernel_open.html

### Supported GPU mapping (Turing and later)

| Card | Architecture | PCI ID | Supported |
|---|---|---|---|
| GTX 1650 | Turing (TU117) | - | ✓ |
| GTX 1660 / Super / Ti | Turing (TU116/TU106) | - | ✓ |
| RTX 2060 / Super / Ti | Turing (TU106/TU104) | 1E89/1E81/1E82 | ✓ |
| RTX 2070 / Super / 2080 / Super / 2080 Ti | Turing (TU106/TU104/TU102) | 1E84/1E87/1E8B/1E8F/1E04/1E07/1E02 | ✓ |
| RTX 30xx (Ampere) | Ampere (GA102/GA104/GA106/GA107) | - | ✓ |
| RTX 40xx (Ada) | Ada (AD102/AD103/AD104) | - | ✓ |
| RTX 50xx (Blackwell) | Blackwell (BK102/BK104) | - | ✓ |
| **GTX 1060** | **Pascal (GM206) — NOT supported by open kernel modules** | - | ✗ |
| **GTX 1070 / 1080 / 1080 Ti** | **Pascal — NOT supported** | - | ✗ |

> **GTX 1060 note:** The open kernel modules require Turing or later (SM 7.5+). GTX 1060 is Pascal (SM 6.1, compute_61, GM206) and uses the proprietary driver. For Pascal targets you still get CUDA on top of the proprietary NVIDIA driver — the open modules are a kernel-space change only. If you want to port from a GTX 1060 source, the CUDA toolkit works the same way; only the kernel module differs.

---

## NVIDIA Open GPU Documentation

**Repository:** https://github.com/nvidia/open-gpu-doc

Chip/hardware interface specifications: register layouts, power management, VRAM controller, GSP firmware interface. This is the documentation side of the open kernel modules — the ISA docs themselves are not included here.

### Relevance to porting

Same as the kernel modules: not relevant for userspace CUDA/HIP porting, but useful for anyone working at the kernel/driver level or comparing NVIDIA vs AMD GPU architecture at the hardware interface level.

### vs AMD equivalent

AMD's equivalent is the in-tree AMDGPU kernel driver + command processor firmware in Mesa. AMD publishes machine-readable ISA XML separately via GPUOpen.

---

## GPUOpen IsaDecoder API

**Repository:** https://github.com/GPUOpen-Tools/isa_spec_manager  
**Vendor:** AMD (GPUOpen)  
**ISA Source:** https://gpuopen.com/machine-readable-isa/  
**Supported ISA:** RDNA 1–4, CDNA 1–5 (**NOT GCN / Polaris / gfx803**)

### What it is

A C++ library + headers for parsing AMD GPU ISA XML files and decode instruction encodings. This is what lets you programmatically understand what an AMDGCN assembly blob actually says.

### Why it matters for porting

When you port a CUDA kernel to HIP, you care about what instruction the compiler *actually* emitted, not just what you wrote in source. The IsaDecoder lets you:

- Confirm `v_dot4_i32_i8` vs `v_dot4_i32_iu8` in the final binary
- Verify the compiler didn't emit an unsupported encoding on gfx1031/gfx1201
- Understand register usage and operand layout from the disassembly
- Automate ISA availability checks across target architectures

### How to use

1. Download ISA XML for your target arch from https://gpuopen.com/machine-readable-isa/
2. Build the IsaDecoder library
3. Parse the XML and use the decoder to interpret disassembled instructions

```cpp
// Example: decode an instruction from disassembly
#include "isa_spec_manager/isa_spec_manager.h"

auto spec = IsaSpecManager::load("gfx1201.xml");
auto instr = spec->decode("v_dot4_i32_i8", ...);
// instr now has fields, operands, encoding known
```

### ISA XML coverage

| Architecture | Chip | Card examples | ISA XML | IsaDecoder parses it? |
|---|---|---|---|---|
| gfx803 | Polaris (GCN 4th-gen) | RX 580, RX 570, RX 480, RX 470, RX 460 | **None** (GCN 4 ISA PDF only) | ✗ **No** — IsaDecoder only parses RDNA/CDNA XML, not GCN |
| gfx900 / gfx901 | RDNA1 (gfx900/TN/5700, gfx901/5600) | RX 5700 / 5700 XT / 5600 XT | rdna1.xml | ✓ Yes |
| gfx902 / gfx904 / gfx906 / gfx909 | RDNA2 (6000/6800/6900/6700) | RX 6000 / 6800 / 6800 XT / 6900 XT / 6700 XT | rdna2.xml | ✓ Yes |
| gfx1030 / gfx1031 / gfx1034 | RDNA2.5/RDNA3.5/RDNA4 (6600/7800/7600) | RX 6600 / 7800 / 7600 | rdna2.5+rdna3.xml | ✓ Yes |
| gfx1100 / gfx1101 / gfx1102 / gfx1103 | RDNA3 (7600XT/7700/7900/7900 XT) | RX 7600 XT / 7700 XT / 7900 / 7900 XT | rdna3.xml | ✓ Yes |
| gfx1150 / gfx1151 / gfx1153 | RDNA3 (7800/7900) | RX 7800 / 7900 | rdna3.xml | ✓ Yes |
| gfx1200 / gfx1201 | RDNA4 (9000/9000 XT) | RX 9000 / 9000 XT | rdna4.xml | ✓ Yes |
| gfx1250 | CDNA4 (MI350) | MI350 | cdna4.xml | ✓ Yes |

### RX 580 — Polaris (gfx803) is NOT in GPUOpen ISA XML

The **RX 580** is Polaris (gfx803, GCN 4th-gen), not RDNA. It was released in 2017, predates the RDNA architecture family entirely. Its ISA is GCN 4th-gen, which is covered by the **AMD GCN 4 ISA PDF manual** — not by the GPUOpen RDNA/CDNA XML files.

| Card | Architecture | ISA | gfx ID | IsaDecoder XML? |
|---|---|---|---|---|
| RX 580 | Polaris (GCN 4th-gen) | GCN 4 ISA | gfx803 | ✗ **No** — use GCN 4 PDF manual + llvm-objdump -mcpu=fiji |

For Polaris ISA references, you need the older GCN ISA documentation from AMD (not the GPUOpen machine-readable XML). The IsaDecoder API only parses RDNA/CDNA XML — it does not support GCN 4th-gen. If you need to verify instruction encoding on RX 580, you'll need:

- AMD GCN 4 ISA manual (PDF)
- `llvm-objdump -mcpu=fiji` to disassemble HIP kernels for gfx803
- Manual cross-reference against the GCN 4 ISA PDF

This is different from the RDNA path: on RDNA you get machine-readable ISA XML. On Polaris (gfx803) you're stuck with PDFs.

### GTX 1060 — Pascal, not Turing

| Card | Architecture | Compute | SM | Chip | Year | Open kernel modules? | ISA XML? |
|---|---|---|---|---|---|---|---|
| GTX 1060 | Pascal | sm_61 / compute_61 | 6.1 | GM206 | 2016 | ✗ Proprietary only | ✗ NVIDIA ISA is PDF-only |

- **Not supported by NVIDIA open kernel modules** (requires Turing+ SM 7.5+)
- CUDA still works on proprietary driver — CUDA toolkit, nvcc, PTX all work as normal
- ISA reference is NVIDIA's Pascal PTX/SASS PDF docs (not machine-readable, never machine-readable)
- If porting from GTX 1060 source, you're targeting sm_61 — no tensor cores, no mma.sync, no WMMA
- When porting from a GTX 1060 (Pascal) source, the CUDA source uses sm_61 features: no tensor core instructions, no mma.sync (that's sm_70+), basic CUDA runtime API. The port to HIP is simpler than from a newer CUDA source because there are fewer GPU-specific intrinsics to translate. But the HIP target arch still needs to support whatever the source used — if the source used `__shfl_sync` or `__ballot_sync`, those translate to HIP's `__shfl` / `__ballot` (drop the mask argument). On sm_61 you use `__shfl` (no sync) and `__ballot` (no sync) — the `*_sync` variants are sm_70+ only.

### vs NVIDIA

NVIDIA does **not** publish machine-readable ISA. Their ISA docs are PDFs. The closest NVIDIA equivalent is:
- `cuobjdump --dump-sass` to get the assembly
- Manual reading of the PTX/SASS reference docs
- NVIDIA's open-gpu-kernel-modules expose some GPU interface but **not** a full ISA decoder

This asymmetry is important for porting: on AMD you can programmatically verify what your kernel compiled to (if your target arch is RDNA/CDNA, not GCN). On NVIDIA you have to eyeball it. For both GTX 1060 (Pascal) and RX 580 (Polaris), you're reading PDFs — neither has machine-readable ISA.

---

## Cross-Reference: NVIDIA vs AMD Open GPU Projects

| | NVIDIA | AMD |
|---|---|---|
| **Kernel driver** | open-gpu-kernel-modules (R515+, Turing+, Linux only) | AMDGPU (in-tree Linux kernel, all arches) |
| **Kernel docs** | open-gpu-doc (register layouts, PM, GSP) — Turing+ only | AMDGPU kernel driver + DC (display core) in Mesa |
| **ISA reference (RDNA/CDNA)** | PDF/PCIe only (no machine-readable) | GPUOpen ISA XML (machine-readable) |
| **ISA reference (GCN/Polaris)** | PDF/PCIe only | AMD GCN ISA PDF (not in GPUOpen XML) |
| **ISA reference (Pascal)** | PDF/PCIe only — never machine-readable | N/A (NVIDIA card) |
| **ISA decoder** | No open ISA decoder (any arch) | GPUOpen IsaDecoder API (RDNA/CDNA only, not GCN/Polaris) |
| **Userspace driver ABI** | Same as proprietary (CUDA/OpenGL/Vulkan work unchanged) | Same as proprietary (ROCm/HIP works unchanged) |
| **GSP firmware** | Required (included in matching driver release) — Turing+ only | Firmware in Mesa / linux-firmware |
| **GTX 1060 (Pascal)** | Proprietary driver only. CUDA works. No open kernel modules. | N/A |
| **RX 580 (Polaris)** | N/A | AMDGPU in-tree. HIP/ROCm legacy (ROCm 5.x era best). OpenCL mature. DirectX/Vulkan fully supported. |

---

## When to reference these in porting work

- **NVIDIA open-gpu-kernel-modules**: reference when explaining Linux driver internals, UVM debugging, or when comparing NVIDIA's open driver vs AMD's in-tree AMDGPU driver. **Not relevant for GTX 1060** (Pascal, not Turing+). **Relevant for RX 580** — but RX 580 uses AMDGPU (in-tree, open already), not NVIDIA.
- **GPUOpen IsaDecoder + ISA XML**: reference whenever you need to verify instruction encoding on a target AMD arch — this is the authoritative source for "does `v_dot4_i32_i8` exist on gfx1031?" **Only works for RDNA/CDNA arches, not Polaris (gfx803/RX 580).** RX 580 needs GCN 4 PDF manual + objdump.
- **NVIDIA open-gpu-doc**: reference when discussing NVIDIA GPU hardware interfaces at the kernel/driver level. **Not relevant for GTX 1060** (Pascal not covered).
- **IsaDecoder C++ API**: reference when writing tooling that needs to parse AMDGCN assembly (kernel audits, perf analysis, automated ISA checks). **Does not parse GCN ISA.** RX 580 (gfx803) is out of scope.
- **GCN 4 ISA PDF**: reference for RX 580 (Polaris gfx803) instruction verification — this is the ISA reference for RX 580, not GPUOpen XML.
- **Pascal PTX/SASS PDF**: reference for GTX 1060 (sm_61) instruction verification — this is the ISA reference for GTX 1060, not any open-source decoder.

---

## Common consumer GPU cards: ISA reference quick lookup

| Card | Vendor | Architecture | ISA family | gfx/SM ID | Open kernel modules? | ISA XML available? | IsaDecoder parses it? |
|---|---|---|---|---|---|---|---|
| **RX 580** | AMD | Polaris (GCN 4) | GCN 4 | gfx803 | AMDGPU (in-tree, always) | ✗ (GCN 4 PDF only) | ✗ **No** (GCN not in IsaDecoder) |
| RX 5700 / 5700 XT | AMD | RDNA 1 | RDNA 1 | gfx900/gfx901 | AMDGPU (in-tree) | ✓ (rdna1.xml) | ✓ Yes |
| RX 6600 / 6600 XT | AMD | RDNA 2.5 | RDNA 2.5 | gfx1030 | AMDGPU (in-tree) | ✓ (rdna2.5.xml) | ✓ Yes |
| RX 6800 / 6800 XT / 6900 XT | AMD | RDNA 2 | RDNA 2 | gfx902/gfx904/gfx906/gfx909 | AMDGPU (in-tree) | ✓ (rdna2.xml) | ✓ Yes |
| RX 7600 / 7600 XT | AMD | RDNA 3 | RDNA 3 | gfx1100/gfx1034 | AMDGPU (in-tree) | ✓ (rdna3.xml) | ✓ Yes |
| RX 7700 XT / 7800 XT / 7900 / 7900 XT | AMD | RDNA 3 | RDNA 3 | gfx1101/gfx1150/gfx1102/gfx1151/gfx1103/gfx1153 | AMDGPU (in-tree) | ✓ (rdna3.xml) | ✓ Yes |
| RX 9000 / 9000 XT | AMD | RDNA 4 | RDNA 4 | gfx1200/gfx1201 | AMDGPU (in-tree) | ✓ (rdna4.xml) | ✓ Yes |
| MI100 | AMD | CDNA 1 | CDNA 1 | gfx908/gfx90a | AMDGPU (in-tree) | ✓ (cdna1.xml) | ✓ Yes |
| MI200 / MI250 / MI250X | AMD | CDNA 2 | CDNA 2 | gfx940/gfx941/gfx942 | AMDGPU (in-tree) | ✓ (cdna2.xml) | ✓ Yes |
| MI350 | AMD | CDNA 4 | CDNA 4 | gfx1250 | AMDGPU (in-tree) | ✓ (cdna4.xml) | ✓ Yes |
| **GTX 1060 / 1070 / 1080 / 1080 Ti** | NVIDIA | Pascal | Pascal (SM 6.1) | sm_61 | ✗ (proprietary only) | ✗ (Pascal PDF only) | ✗ No (no NVIDIA ISA decoder) |
| GTX 1650 / 1660 / 1660 Ti / Super | NVIDIA | Turing | Turing (SM 7.5) | sm_75 | ✓ (open-gpu-kernel-modules) | ✗ (PDF only) | ✗ No (no NVIDIA ISA decoder) |
| RTX 2060 / 2070 / 2080 / 2080 Ti / Super | NVIDIA | Turing | Turing (SM 7.5) | sm_75 | ✓ (open-gpu-kernel-modules) | ✗ (PDF only) | ✗ No (no NVIDIA ISA decoder) |
| RTX 30xx (Ampere) | NVIDIA | Ampere | Ampere (SM 8.0/8.6/8.9) | sm_80/sm_86/sm_89 | ✓ (open-gpu-kernel-modules) | ✗ (PDF only) | ✗ No (no NVIDIA ISA decoder) |
| RTX 40xx (Ada) | NVIDIA | Ada Lovelace | Ada (SM 8.9) | sm_89 | ✓ (open-gpu-kernel-modules) | ✗ (PDF only) | ✗ No (no NVIDIA ISA decoder) |
| RTX 50xx (Blackwell) | NVIDIA | Blackwell | Blackwell (SM 12.x) | sm_120 | ✓ (open-gpu-kernel-modules) | ✗ (PDF only) | ✗ No (no NVIDIA ISA decoder) |

> **Key takeaway:** GPUOpen IsaDecoder (isa_spec_manager) **only** parses RDNA 1–4 and CDNA 1–5 XML. It does **not** parse GCN (Polaris RX 580 etc.). And there is **no** open-source ISA decoder for any NVIDIA architecture (Pascal, Turing, Ampere, Ada, Blackwell) — NVIDIA ISA is always PDF-only. For GTX 1060 and RX 580, you read PDFs + disassembly output. For RDNA/CDNA cards (RX 5700+), you get machine-readable XML + IsaDecoder.

---

## RX 580 and GTX 1060: cross-vendor reality check

See [`RX580_GTX1060_PORTING.md`](./RX580_GTX1060_PORTING.md) for the full breakdown.

### Short version

- **RX 580**: mature compute paths are DirectX 12, Vulkan, OpenCL (ROCm OpenCL). HIP/ROCm works but gfx803 is legacy in ROCm 6+. No machine-readable ISA — GCN 4 PDF + `llvm-objdump -mcpu=fiji`.
- **GTX 1060**: mature compute path is CUDA (Linux/Windows) + DirectX 12/Vulkan (Windows). OpenCL available but less used. No machine-readable ISA — Pascal PDF + `cuobjdump --dump-sass`.
- **Neither has tensor cores**. Both have int8 dot product (`__dp4a` on GTX 1060, `v_dot4_i32_i8` on RX 580).
- **Neither has machine-readable ISA**. RX 580: GCN 4 PDF. GTX 1060: Pascal PDF. Neither has an open ISA decoder.
- **Neither is a primary HIP/ROCm target**: RX 580 is legacy gfx803 in ROCm; GTX 1060 can't run HIP at all.
- **Cross-vendor path**: DirectX 12 (DXIL via DXC) and Vulkan (SPIR-V via glslc/dxc -spirv) work on both. DP4a works on both for int8 matmul.

### Long version

See [`RX580_GTX1060_PORTING.md`](./RX580_GTX1060_PORTING.md) for:
- RX 580: DirectX 12, Vulkan (CoopMat broken — DP4a), OpenCL (ROCm, mature for gfx803), HIP/ROCm (legacy), ISA reality (GCN 4 PDF, no IsaDecoder for GCN)
- GTX 1060: DirectX 12, Vulkan (CoopMat supported but FP16-only, no tensor cores), OpenCL (NVIDIA, available), CUDA (primary), ISA reality (Pascal PDF, no NVIDIA ISA decoder ever)
- Cross-vendor paths: CUDA → HLSL → DXIL → RX 580 via D3D12; CUDA → HLSL → SPIR-V → RX 580 via Vulkan; CUDA → OpenCL → RX 580 via ROCm OpenCL; CUDA → HIP → RX 580 via HIP/ROCm (legacy)
- RX 580 as source → GTX 1060 as target: OpenCL/Vulkan/DX12 → CUDA path
- Feature comparison: what each card has and doesn't have (no tensor cores, no BF16 native, int8 dot available on both, FP16 native on both, no machine-readable ISA on either)
