# Porting Report Template — Fill Me For Each New Engine

> Copy this to `findings/<ENGINE_NAME>.md` and fill in each section.

---

## 0. Engine Metadata

| Field | Value |
|---|---|
| Engine name | `[e.g., Project Maya / Strata]` |
| Repo URL | `[e.g., https://github.com/Niko1221/Strata]` |
| Primary GPU backends | `[CUDA | HIP | Vulkan | Metal | OpenCL]` |
| Current GPU targets | `[sm_75, gfx1100, ...]` |
| Build system | `[CMake / Bazel / Meson / Makefile / custom]` |

## 1. Kernel & Shader Census

### 1.1 GPU kernels (`.cu` / `.comp` / `.metal`)

| File | Backend | Purpose | Arch-specific branches |
|---|---|---|---|
| (fill from discover.py) | | | |

### 1.2 CPU kernels (`.cpp` x86 / `.cpp` ARM)

| File | ISA target | Purpose |
|---|---|---|

### 1.3 Shaders / graphics pipeline (if any)

| File | Type | Stage | Purpose |
|---|---|---|---|
| _(none expected for compute engine)_ | | | |

### 1.4 Build targets

| Target | Type | Sources | GPU backend |
|---|---|---|---|
| | | | |

## 2. Architecture Capability Mapping

### 2.1 Existing arch support (from build config)

| Arch ID | Vendor | Wave | LDS KiB | Tensor Cores | Dot4 | WMMA | Notes |
|---|---|---|---|---|---|---|---|

### 2.2 Target archs for this port

| Arch ID | Name | Status | Needs new code? |
|---|---|---|---|

### 2.3 Architecture-specific code locations

| File:Line | Guard | Arch(s) | What it does |
|---|---|---|---|

## 3. Model Family Mapping

### 3.1 Supported models in this engine

| Family | Engine binary | Architecture | Quant formats | Source |
|---|---|---|---|---|

### 3.2 Kernel dispatch points per model

For each model family, trace the hot path:

| Stage | Kernel(s) | File:Line | Notes |
|---|---|---|---|
| Attention | | | |
| Routing | | | |
| MoE gate/up | | | |
| MoE down | | | |
| KV cache | | | |
| Embedding | | | |
| Norm | | | |
| Sampler | | | |
| Vision | | | |
| Shared expert | | | |

### 3.3 New kernels needed

| Kernel name | Purpose | Model | Difficulty |
|---|---|---|---|

## 4. Porting Gap Analysis

### 4.1 Coverage matrix: kernels × target archs

| Kernel | Target arch 1 (sm_XX / gfxXXX) | Target arch 2 | Target arch 3 |
|---|---|---|---|
| native_flash_attn | OK / MISS / TUNE / NEW | | |
| (...) | | | |

### 4.2 Memory requirements

| Model | Quant | Weights GB | KV/token KB | Experts | Min VRAM+RAM |
|---|---|---|---|---|---|

### 4.3 Performance path critical kernels

List the 5-10 kernels that matter most for tok/s:

1.
2.
3.

## 5. Porting Task Breakdown

### Must-have (blocking)

| # | Task | Kernel(s) | Est. days |
|---|---|---|---|

### Correctness (parity tests)

| # | Task | Test | Est. days |
|---|---|---|---|

### Performance (tuning)

| # | Task | Target tok/s | Est. days |
|---|---|---|---|

### Documentation

| # | Task |
|---|

## 6. Verification Checklist

| Test | Command | Expected | Status |
|---|---|---|---|
| Build | | | |
| Self-test | | | |
| Parity | | | |
| Intrinsics | | | |
| Performance | | | |
