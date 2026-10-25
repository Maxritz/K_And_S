# Intel CPU Gen + Intel iGPU / AMD APU Porting Helpers

> CPU host + iGPU is a different kind of port: the host CPU matters for the compiler toolchain,
> the iGPU matters for the shader compiler + ISA. This doc covers the CPU generations that matter
> (Zen 2/Zen 3 AMD, Intel 10th-14th Gen) and the iGPU paths that run on them (Intel Xe-LP GEN12,
> Intel Arc, Meteor Lake, Battlemage), plus the AMD APU side (RDNA iGPUs, Strix Point/Radeon 8060S).

---

## CPU Host: what matters for porting

The CPU host determines:
- **Compiler availability**: MSVC version, GCC/Clang version, ROCm's hipcc (which uses Clang/LLVM under the hood)
- **CXX standard support**: `/std:c++20` or `-std=c++20` — needed for CUDA/HIP/SYCL host code
- **Build system**: CMake generator (Visual Studio, Ninja, Unix Makefiles) depends on host compiler
- **ISA extensions**: AVX2, AVX-512, AMX — for CPU kernel paths (not GPU, but relevant if the port has a CPU fallback)

---

## AMD Zen 2 / Zen 3 host CPUs

| Generation | Codename | Examples | znver | GCC/Clang support | Notes for porting |
|---|---|---|---|---|---|
| Zen 2 | Matisse | Ryzen 3000 series (5600X, 5800X, 5900X, 5950X) | znver2 | GCC 10+, Clang 10+ | AMX not available (AMX is Zen 4+). AVX2 + FMA3. Good hipcc host (Clang 10+ under ROCm). 64-bit only. No AVX-512 (AMD dropped AVX-512 on Zen 2/Zen 3; it came back on Zen 4). |
| Zen 3 | Vermeer/Matisse refresh | Ryzen 5000 series (5600, 5800X3D, 5900X, 5950X) | znver3 | GCC 11+, Clang 12+ | Same as Zen 2 — no AVX-512, no AMX. Better IPC than Zen 2. Still a solid hipcc host. 64-bit only. |

### Zen 2 / Zen 3 as HIP host

- Clang++ version: ROCm 5.x used Clang 10/11; ROCm 6.x+ uses Clang 14+. For Zen 2/Zen 3, ROCm 6.x hipcc works fine — the CPU ISA is not the limiting factor.
- GCC version: GCC 11+ covers Zen 3 optimizations. hipcc on Linux uses the system Clang, not GCC.
- No AVX-512: if the CPU kernel path relies on AVX-512, it won't run on Zen 2/Zen 3. Use AVX2 + FMA3 fallback.
- AMX (AMD Matrix eXtensions) is Zen 4+ only — not available on Zen 2/Zen 3. If a CPU GEMM uses AMX, it won't work on Zen 2/Zen 3.

### Zen 2 / Zen 3 as CUDA host

- CUDA on Linux: nvcc uses the system GCC (or Clang via `-ccbin`). Zen 2/Zen 3 with GCC 11+ works fine for CUDA host compilation.
- CUDA on Windows: needs MSVC — AMD CPUs don't matter for Windows CUDA (you need a Windows AMD GPU + MSVC, not a specific CPU).

---

## Intel CPU generations 10th-14th Gen (host + iGPU context)

| Gen | Codename | Architecture | iGPU | Examples | Compiler context | Porting notes |
|---|---|---|---|---|---|---|
| 10th | Comet Lake / Lake Comet | 14nm | Intel HD Graphics 630 (Gen12 UHD) | i9-10900K, i7-10700K, i5-10400 | MSVC 2019+ (v142) or GCC 10+ | iGPU is Gen12 LP (UHD 630) — weak, only suitable for light Vulkan compute, no DPAS. Use as host for CUDA builds (Windows) or OpenVINO CPU inference. Not a GPU compute target. |
| 11th | Rocket Lake | 14nm | Intel UHD Graphics 11th Gen (Gen12) | i9-11900K, i7-11700K, i5-11600K | MSVC 2019+ or GCC 11+ | Same Gen12 iGPU as 10th Gen (UHD). Still weak for compute. Best use: host for CUDA/oneAPI builds. Not a GPU compute target. |
| 12th | Alder Lake | 10nm + 14nm (hybrid) | Intel UHD Graphics (Gen12) + Arc graphics (some configs) | i9-12900K, i7-12700K, i5-12600K | MSVC 2022 (v143) or GCC 12+ | Hybrid CPU (P-cores + E-cores) — CMake/Ninja handles this fine. iGPU is still Gen12 unless you have an Arc-capable SKU. Host for CUDA/oneAPI builds. AVX-512 dropped on Alder Lake (Intel disabled it on hybrid arch). |
| 13th | Raptor Lake | 10nm + 14nm (hybrid) | Intel UHD (Gen12) + Arc graphics (some) | i9-13900K, i7-13700K, i5-13600K | MSVC 2022 (v143) or GCC 12+ | Same as 12th Gen — hybrid, AVX-512 disabled. iGPU is Gen12. Host for CUDA/oneAPI builds. AVX-512 not available. |
| 14th | Raptor Lake Refresh | 10nm + 14nm (hybrid) | Intel UHD (Gen12) + Arc graphics (some) | i9-14900K, i7-14700K, i5-14600K | MSVC 2022 (v143) or GCC 13+ | Same as 12th/13th Gen. Hybrid. AVX-512 disabled. iGPU is Gen12. Host for CUDA/oneAPI builds. Note: 13th/14th Gen i9 instability issues (Vmin shift) — be aware if running sustained GPU compute workloads that stress the CPU power delivery. |

### Intel 10th-14th Gen iGPU reality

- **10th/11th/12th/13th/14th Gen Intel CPUs have Gen12 iGPU (UHD Graphics)** unless you have an Arc-capable SKU (those are discrete Arc cards, not iGPU).
- Gen12 iGPU (UHD 630 / 11th Gen UHD) is **GEN12_LP** in ARCH_MAP terms — weak for compute:
  - No DPAS matrix instruction (that's GEN12_HPC / Arc only)
  - No WMMA equivalent
  - BF16 not native (Gen12 LP has limited BF16 support)
  - EU subgroups of 8 threads (vs AMD wave32)
  - Only suitable for light Vulkan compute, OpenVINO CPU inference, or very small SYCL kernels
  - **Not** a real GPU compute target for LLM kernels
- If you need real Intel iGPU compute, look at:
  - **Meteor Lake (Intel Core Ultra 100 series)** — GEN12_LP with better iGPU (8 CU, Xe-LP, DPAS supported, BF16 native) — this is the first Intel iGPU that's worth considering for compute
  - **Battlemage (Intel Arc B-series)** — discrete Arc GPU, GEN12_HPC-class, DPAS + WMMA-like via SYCL
  - **Arc A-series (Alchemist)** — discrete, GEN12_LP-class but with DPAS, BF16 native, 32 CU — much better than iGPU

### Intel 10th-14th Gen as CUDA host

- Windows: MSVC 2019 (v142) for CUDA 11.x, MSVC 2022 (v143) for CUDA 12.x. Intel CPUs work fine as CUDA hosts on Windows — the CPU vendor doesn't matter, just the MSVC version.
- Linux: GCC 10+ (10th Gen), GCC 11+ (11th Gen), GCC 12+ (12th-14th Gen). CUDA host compilation works on Intel CPUs same as on AMD.
- AVX-512: 10th/11th Gen Intel has AVX-512. 12th Gen and later (hybrid) **disables AVX-512** — if your CPU kernel path uses AVX-512, it won't run on 12th/13th/14th Gen Intel.

### Intel 10th-14th Gen as oneAPI/SYCL host

- oneAPI DPC++ (icpx) works on Intel 10th-14th Gen Intel CPUs as host.
- For iGPU SYCL: only Meteor Lake iGPU (Intel Core Ultra) has a real iGPU compute path via SYCL/oneAPI. 10th-14th Gen iGPU (Gen12 LP UHD) is not a practical SYCL compute target.
- For discrete Arc SYCL: the CPU is the host, the Arc card (Alchemist/Battlemage) is the GPU. CPU gen matters for the host compiler (GCC/Clang/MSVC), not for the GPU compute.

---

## Intel iGPU compute targets (when the iGPU is the GPU)

### Meteor Lake (Intel Core Ultra 100 series — 135H, 135U, 155H, etc.)

| Property | Value |
|---|---|
| ARCH_MAP id | `meteor_lake` |
| iGPU arch | Xe-LP (GEN12_LP variant, 8 CU) |
| EUs | 128 (8 CU × 16 EUs) |
| Wave size | 16 |
| LDS | 128 KiB |
| BF16 native | Yes |
| DPAS | Yes (matrix instruction via oneAPI DPC++/SYCL) |
| WMMA | No (no WMMA equivalent — DPAS is the matrix path) |
| FP8 native | No |
| Dot4 | Yes |
| ISA docs | https://intel.github.io/pisa/ (XED) |
| ISA disassembler | IGA (https://github.com/intel/intel-graphics-compiler/tree/master/visa/iga) |
| Vulkan CM | Requires Mesa 24.3+ for CooperativeMatrix |
| Memory | Shared DDR5 system memory (no dedicated VRAM) |
| Bandwidth | ~72 GB/s (DDR5-5600 dual-channel) |
| Power | 28W max package, sustained factor ~0.7 |
| Perf | FP32: ~5.6 GFLOPS, BF16 via DPAS: ~42.8 GFLOPS (but memory-bound at 72 GB/s) |

### Meteor Lake as a porting target

- **SYCL/oneAPI DPC++** is the primary compute path. Use `dpcpp` compiler from oneAPI.
- DPAS is the matrix instruction — not WMMA, not tensor cores. DPAS does 16x16 matrix multiply-add in hardware.
- BF16 native — can do BF16 math without emulation.
- 72 GB/s shared memory bandwidth is the bottleneck — this is why Meteor Lake iGPU is memory-bound for LLM workloads. Discrete GPUs have 500-2000 GB/s.
- Vulkan path: possible but requires Mesa 24.3+ for CooperativeMatrix. DP4a fallback works on Meteor Lake Vulkan too.
- Best use case: small LLM inference (7B-13B models quantized to Q4_0 or lower) where memory bandwidth is the limit, not compute. Not for large models or high-throughput serving.

### Meteor Lake known issues

- Integrated GPU shares system memory — performance depends on DDR5 speed and dual-channel config
- No dedicated VRAM — cache misses are expensive
- Power/thermal constraints on laptop APUs — 28W envelope throttles under sustained load
- Vulkan CooperativeMatrix requires Mesa 24.3+ — check `kands://isa-references`

### Battlemage (Intel Arc B-series — B580, B570, etc.)

| Property | Value |
|---|---|
| ARCH_MAP id | `gen12_hpc` (same bucket as Arc A-series for now, but Battlemage is GEN12_HPC-class) |
| Compute units | 24 |
| EUs | 384 (24 CU × 16 EUs) |
| Wave size | 8 |
| LDS | 64 KiB (per CU) |
| BF16 native | Yes |
| DPAS | Yes (matrix instruction via oneAPI DPC++/SYCL) |
| WMMA | No (DPAS is the matrix path) |
| FP8 native | Check ISA docs — Battlemage may have FP8 (GEN12_HPC roadmaps mention it) |
| Dot4 | Yes |
| ISA docs | https://intel.github.io/pisa/ |
| ISA disassembler | IGA |
| Memory | Dedicated GDDR6 (512 GB/s for B580) |
| Perf | FP32: ~14.2 GFLOPS, BF16 via DPAS: ~109.2 GFLOPS |

### Battlemage as a porting target

- SYCL/oneAPI DPC++ is the primary compute path (same as Alchemist Arc).
- DPAS is the matrix instruction.
- 512 GB/s dedicated GDDR6 — much better than Meteor Lake iGPU (72 GB/s) but worse than discrete AMD (960 GB/s) or NVIDIA (1008 GB/s).
- Supports Vulkan + SPIR-V, DirectX 12 (via Vulkan interop or native D3D12 on Windows).
- oneAPI supported_archs includes `arc_battlemage` — via `icpx` compiler.
- OpenVINO supported for Battlemage (post-training quantization + model optimizer).

---

## AMD APU iGPU (RDNA iGPUs)

### Strix Point / Strix Halo (Radeon 8060S, Radeon 8060, etc.)

| Property | Value |
|---|---|
| Architecture | RDNA 3 (gfx1100-class iGPU) / RDNA 3.5 (Strix Halo, gfx1150-class) |
| APU examples | Ryzen AI 9 365 (Strix Point, gfx1100 iGPU), upcoming Strix Halo (Radeon 8060S, gfx1150-class) |
| iGPU CUs | 12 (Strix Point) / 40 (Strix Halo — rumored) |
| Wave size | 32 |
| LDS | 128 KiB |
| BF16 native | Yes (RDNA 3) |
| WMMA | gfx1100: yes (RDNA 3 WMMA), gfx1150-class: yes (RDNA 3.5 WMMA) |
| Dot4 | Yes |
| FP8 native | No (RDNA 3), Check for RDNA 3.5 |
| ISA docs | https://gpuopen.com/machine-readable-isa/ |
| ISA disassembler | IsaDecoder API (GPUOpen) |
| Memory | Shared DDR5/LPDDR5X system memory |
| Perf | Strix Point: ~5 TFLOPS FP32, Strix Halo: ~30-50 TFLOPS FP32 (rumored) |

### Strix Point APU as a porting target

- HIP/ROCm supports gfx1100 iGPU (same as discrete RDNA 3) — use `hipcc --offload-arch=gfx1100` for Strix Point iGPU.
- WMMA available on Strix Point (gfx1100-class) — can use WMMA fp16 path for attention.
- Shares system memory — bandwidth is ~72-128 GB/s depending on DDR5/LPDDR5X config. Bottleneck for large models.
- Vulkan path: RDNA 3 Vulkan supports DP4a; CooperativeMatrix is broken on AMD drivers (radv/AMDVLK) — use DP4a.
- Use case: laptop LLM inference (7B-13B models) where power/thermal matters more than throughput. Not for server-class workloads.

---

## CPU + iGPU porting decision matrix

| Scenario | CPU | iGPU | Compute path | Notes |
|---|---|---|---|---|
| CUDA source → Linux AMD iGPU (Strix Point, gfx1100) | Zen 2/3/4 (any) | RDNA 3 iGPU | HIP/ROCm (hipcc --offload-arch=gfx1100) | WMMA available on gfx1100 iGPU. Shared memory bandwidth is the bottleneck. |
| CUDA source → Linux Intel iGPU (Meteor Lake) | Intel 13th/14th Gen (or any) | Xe-LP iGPU (8 CU) | oneAPI DPC++/SYCL (dpcpp) | DPAS for matrix ops. 72 GB/s shared memory. BF16 native. Not for large models. |
| CUDA source → Windows AMD iGPU (Strix Point) | Intel 10th-14th Gen or AMD | RDNA 3 iGPU | DirectX 12 (DXC → DXIL) or Vulkan | DX12 is the Windows path for AMD iGPU. Vulkan via radv/AMDVLK. DP4a for matmul (CoopMat broken on AMD). |
| CUDA source → Windows Intel iGPU (Meteor Lake) | Intel 10th-14th Gen | Xe-LP iGPU | DirectX 12 (DXC → DXIL) or Vulkan or SYCL on Windows | DX12 on Intel iGPU via D3D12. Vulkan via Intel ICD. SYCL on Windows with oneAPI. DPAS via SYCL. |
| CUDA source → discrete AMD GPU (RX 7900 XTX, gfx1100) | Any CPU (Zen 2/3/4, Intel 10th-14th) | Discrete RDNA 3 | HIP/ROCm (hipcc --offload-arch=gfx1100) | CPU gen doesn't matter for the GPU path. Zen 2/3 is fine as hipcc host. |
| CUDA source → discrete NVIDIA GPU (RTX 3090, sm_80) | Any CPU | Discrete NVIDIA | CUDA (nvcc --arch=sm_80) | CPU gen doesn't matter. MSVC on Windows, GCC/Clang on Linux. |
| CUDA source → discrete Intel Arc (A770, Alchemist) | Any CPU | Arc A-series (GEN12_LP, 32 CU) | oneAPI DPC++/SYCL (icpx) or Vulkan or DirectX 12 | DPAS for matrix ops. BF16 native. 560 GB/s GDDR6. Vulkan CM supported (unlike AMD). Windows: DX12 via Vulkan interop. |
| CUDA source → Arc B580 (Battlemage) | Any CPU | Arc B-series (GEN12_HPC, 24 CU) | oneAPI DPC++/SYCL (icpx) | Newer than Alchemist. DPAS. BF16. 512 GB/s GDDR6. Check FP8 support. |
| CPU kernel (no GPU) → AMD Zen 2/Zen 3 host | Zen 2/Zen 3 | N/A (CPU only) | AVX2 + FMA3 (no AVX-512, no AMX) | If CPU kernel uses AVX-512 or AMX, rewrite for AVX2/FMA3. Use `-march=znver2` or `-march=znver3` for GCC. |
| CPU kernel (no GPU) → Intel 12th-14th Gen host | Intel 12th-14th Gen | N/A (CPU only) | AVX2 (AVX-512 disabled on hybrid) | If CPU kernel uses AVX-512, rewrite for AVX2. Add `-march=alderlake` or `-march=raptorlake` for GCC. P-cores + E-cores: use taskset or CMake to pin to P-cores for performance. |

---

## Toolchain specifics per CPU + iGPU combo

### AMD Zen 2/Zen 3 + HIP/ROCm (discrete or APU)

- **ROCm version**: ROCm 6.x works on Zen 2/Zen 3. ROCm 5.x also works (older LLVM). ROCm 7.x/ROCm 10.x works (Clang 14+ under hipcc).
- **hipcc host**: uses Clang++ from ROCm's LLVM. Zen 2/Zen 3 are fine — the CPU ISA is not the bottleneck for hipcc host code.
- **Compiler flag**: `hipcc -O3 -std=c++20 --offload-arch=gfx1100` for RDNA 3 target.
- **GCC version**: GCC 11+ recommended for Zen 3. GCC 10 works for Zen 2.
- **No AVX-512**: AMD dropped AVX-512 on Zen 2/Zen 3. If CPU kernel path uses AVX-512, use AVX2 fallback.

### Intel 10th-14th Gen + oneAPI DPC++ (discrete Arc or Meteor Lake iGPU)

- **oneAPI version**: oneAPI 2023+ for GEN12 targets. oneAPI 2024+ recommended for Battlemage.
- **icpx compiler**: `${ONEAPI_ROOT}/compiler/latest/linux/bin/icpx` (Linux) or `${ONEAPI_ROOT}\bin\icpx.exe` (Windows).
- **SYCL device filter**: set `SYCL_DEVICE_FILTER=level_zero` or `SYCL_DEVICE_FILTER=opencl` for Arc; for Meteor Lake iGPU use `opencl` or `level_zero` depending on driver.
- **Compiler flag**: `icpx -O3 -std=c++20 -fintelfpga` — no, that's FPGA. For GPU: `icpx -O3 -std=c++20 -fsycl -fsycl-targets=spir64_gen` for SPIR-V target, or `-fsycl-targets=sycl_xtarget` for SYCL.
- **MSVC on Windows**: oneAPI on Windows uses MSVC 2019+ (v142) or 2022 (v143) as host compiler. Intel 10th Gen → MSVC 2019. 12th-14th Gen → MSVC 2022 (v143).

### Intel 10th-14th Gen + CUDA (Windows, discrete NVIDIA GPU)

- **CUDA version**: CUDA 12.x for MSVC 2022 (v143). CUDA 11.5-11.8 for MSVC 2019 (v142).
- **MSVC version**: 10th/11th Gen → MSVC 2019 (v142). 12th-14th Gen → MSVC 2022 (v143).
- **nvcc host**: nvcc wraps cl.exe on Windows. Intel CPU is fine — CPU vendor doesn't matter for CUDA host compilation.
- **Note**: AVX-512 is available on 10th/11th Gen Intel but disabled on 12th-14th Gen (hybrid). CUDA host code doesn't typically use AVX-512 directly, but if your CPU kernel path does, it won't work on 12th-14th Gen.

### Intel 10th-14th Gen + OpenVINO (CPU inference, no GPU)

- **OpenVINO version**: 2023.0+ for Intel 10th-14th Gen.
- **Supported**: OpenVINO CPU inference on Intel 10th-14th Gen works with AVX2 (10th-11th Gen also has AVX-512 option, but 12th+ doesn't).
- **Use case**: CPU-only inference when no GPU is available. Not a GPU port — this is the fallback path.
- **OpenVINO GPU**: requires discrete Arc or Meteor Lake iGPU. 10th-14th Gen iGPU (Gen12 LP UHD) is not a practical OpenVINO GPU target.

---

## ISA references per CPU + iGPU path

| Target | ISA reference | Disassembler | Notes |
|---|---|---|---|
| AMD RDNA 3 (gfx1100, Strix Point APU) | https://gpuopen.com/machine-readable-isa/ (RDNA3 XML) | IsaDecoder API (GPUOpen) | Machine-readable XML — can programmatically verify instructions |
| AMD RDNA 3.5 (gfx1150, Strix Halo APU) | https://gpuopen.com/machine-readable-isa/ (RDNA3 XML) | IsaDecoder API (GPUOpen) | Same XML as gfx1100 — RDNA 3.5 is RDNA3-class |
| Intel Xe-LP (Meteor Lake iGPU, GEN12_LP) | https://intel.github.io/pisa/ (XED/PISA specs) | IGA (intel-graphics-compiler/visa/iga) | Machine-readable PISA specs — can programmatically check instruction availability |
| Intel GEN12_HPC (Arc A-series Alchemist, Arc B-series Battlemage) | https://intel.github.io/pisa/ (XED/PISA specs) | IGA | Same as Meteor Lake — GEN12_HPC uses the same PISA ISA as Xe-LP for the most part, with DPAS added |
| NVIDIA (GTX 1060, RTX 3090, etc.) | NVIDIA PTX/SASS PDF docs (not machine-readable) | cuobjdump / nvdisasm (proprietary, CUDA toolkit) | No open ISA decoder for NVIDIA — ever |
| AMD GCN (RX 580, gfx803) | AMD GCN 4 ISA PDF (not machine-readable, not in GPUOpen XML) | llvm-objdump -mcpu=fiji | GPUOpen IsaDecoder doesn't parse GCN — use PDF + objdump |

---

## When to use what

- **Zen 2/Zen 3 host + discrete AMD GPU (RX 7900 XTX)**: use HIP/ROCm. CPU gen doesn't affect the GPU path. Zen 2/3 is fine as hipcc host.
- **Zen 2/Zen 3 host + AMD APU (Strix Point, gfx1100 iGPU)**: use HIP/ROCm targeting gfx1100. WMMA available on iGPU. Shared memory bandwidth is the bottleneck.
- **Intel 10th-14th Gen host + discrete Intel Arc (A770/B580)**: use oneAPI DPC++/SYCL (icpx). DPAS for matrix ops. CPU gen matters for host compiler (GCC/Clang/MSVC version), not for the GPU path.
- **Intel 13th/14th Gen host + Meteor Lake iGPU**: use oneAPI DPC++/SYCL targeting the iGPU via OpenCL/Level Zero. DPAS for matrix ops. 72 GB/s shared memory — memory-bound. Best for small LLM inference.
- **Intel 10th-14th Gen host + CUDA (Windows, discrete NVIDIA GPU)**: use CUDA (nvcc). MSVC 2019 for 10th/11th Gen, MSVC 2022 for 12th-14th Gen. CPU gen matters only for the MSVC version.
- **CPU-only (no GPU)**: Zen 2/Zen 3 → AVX2 + FMA3 (no AVX-512, no AMX). Intel 12th-14th Gen → AVX2 (AVX-512 disabled). Intel 10th/11th Gen → AVX2 or AVX-512 available. Use the appropriate `-march` flag.

---

## Quick reference: what each generation can actually run

| CPU Gen | AVX-512? | AMX? | Good CUDA host? | Good oneAPI host? | Good HIP host? | iGPU compute viable? |
|---|---|---|---|---|---|---|
| AMD Zen 2 (Ryzen 3000) | No (AMD dropped AVX-512) | No (AMX is Zen 4+) | Yes (GCC 10+ / Clang 10+) | Yes (oneAPI on Zen 2 works) | Yes (ROCm hipcc uses Clang, CPU ISA fine) | N/A (no AMD iGPU on Zen 2 desktop) |
| AMD Zen 3 (Ryzen 5000) | No (dropped) | No (AMX is Zen 4+) | Yes (GCC 11+ / Clang 12+) | Yes | Yes | N/A (no AMD iGPU on Zen 3 desktop) |
| Intel 10th Gen (Comet Lake) | Yes | No (AMX is 12th Gen+) | Yes (MSVC 2019 / GCC 10+) | Yes | N/A (no AMD GPU) | No (Gen12 LP UHD is weak) |
| Intel 11th Gen (Rocket Lake) | Yes | No | Yes (MSVC 2019 / GCC 11+) | Yes | N/A | No (Gen12 UHD is weak) |
| Intel 12th Gen (Alder Lake) | No (disabled on hybrid) | No | Yes (MSVC 2022 / GCC 12+) | Yes | N/A | No (Gen12 UHD) / Yes if Arc discrete |
| Intel 13th Gen (Raptor Lake) | No (disabled on hybrid) | No | Yes (MSVC 2022 / GCC 12+) | Yes | N/A | No (Gen12 UHD) / Yes if Arc discrete or Meteor Lake iGPU |
| Intel 14th Gen (Raptor Lake Refresh) | No (disabled on hybrid) | No | Yes (MSVC 2022 / GCC 13+) | Yes | N/A | No (Gen12 UHD) / Yes if Arc discrete or Meteor Lake iGPU |
| Intel Core Ultra (Meteor Lake, 135H/155H) | No (hybrid) | No (AMX is Xeon only on these) | Yes (MSVC 2022 / GCC 13+) | Yes — iGPU SYCL works via OpenCL/L0 | N/A | **Yes** — Meteor Lake iGPU (8 CU Xe-LP) is the first Intel iGPU worth computing on |
