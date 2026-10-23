# Doxygen-Style Kernel Audit Template (C++)

> **Purpose**: Audit any kernel/shader file for portability across GPU architectures.
> **Mode**: `{{MODE}}` (strict | loose)
> **Language**: C++

---

## How to Use

Place the following Doxygen-style block at the top of each kernel file you are auditing:

```cpp
/// {{FILE_NAME}} — {{KERNEL_PURPOSE}}
///
/// Target backend: {{BACKEND}} (CUDA / HIP / Vulkan / Metal / oneAPI)
/// Target architectures: {{ARCHS}}
///
/// ## Portability Audit (K_and_S {{MODE_UPPER}} mode)
///
/// {% if MODE == "strict" %}
/// | Check | Status | Notes |
/// |---|---|---|
/// | No arch-specific guards (`__CUDA_ARCH__`/`__HIP__`) | ❓ UNCHECKED | |
/// | No wave64 dependencies (gfx908) | ❓ UNCHECKED | |
/// | No WMMA instructions (unavailable on gfx1030, gfx1200, gfx1201) | ❓ UNCHECKED | |
/// | No bf16 native ops on non-bf16 arches | ❓ UNCHECKED | |
/// | No FP8 ops on non-FP8 arches | ❓ UNCHECKED | |
/// | Tensor core usage (sm_80+ only) | ❓ UNCHECKED | |
/// | Dot4 usage (present on all AMD RDNA/CDNA) | ❓ UNCHECKED | |
/// | 64 KiB LDS assumption (not 128 KiB on gfx12xx) | ❓ UNCHECKED | |
/// | Shared memory bank conflict patterns | ❓ UNCHECKED | |
/// {% else %}
/// | Check | Status | Notes |
/// |---|---|---|
/// | Uses `#if defined(STRATA_USE_HIP)` for backend split | ❓ UNCHECKED | |
/// | Has F32 fallback for GPU-only instructions | ❓ UNCHECKED | |
/// | No hard dependency on unsupported arch features | ❓ UNCHECKED | |
/// {% endif %}
///
/// ## K_and_S tracing markers
///
/// ```cpp
/// STRACE("{{COMPONENT}}", "{{PHASE}}", "{{TAG}}", "{{EXTRA}}");
/// STRACE_SCOPE("{{COMPONENT}}", "{{PHASE}}", "{{TAG}}", "{{EXTRA}}");  // RAII begin/end
/// ```
```

---

## Audit Checklist (copy-paste into each file)

### Strict Mode (exhaustive)

- [ ] **Architecture branches**: Find all `#if __CUDA_ARCH__`, `#ifdef __HIP__`, `#if defined(gfx1100)` etc.
- [ ] **Tensor core / MMA**: Check for `mma`, `wmma`, `tcgen` instructions — ensure F32 fallback exists
- [ ] **Dot4 / integer ops**: Check for `v_dot4_i32_i8` (RDNA2) vs newer variants (RDNA4)
- [ ] **BF16 / FP8 native**: Check `__hmul`, `__float2bfloat16` — verify fallback on non-native bf16 arches
- [ ] **WMMA usage**: Check rocWMMA / cuBLASLt — must be guarded by `STRATA_HAS_WMMA`
- [ ] **LDS size**: Verify ≤ 64 KiB on RDNA2, ≤ 128 KiB on RDNA4 (128 KiB on CDNA)
- [ ] **Wave size**: Verify no assumptions about 32 vs 64 (gfx908 wave64)
- [ ] **Shared memory bank config**: Check `__shared__` layouts for bank conflicts
- [ ] **64-bit addressing**: Check pointer size assumptions (important for large weight files)
- [ ] **FMA contraction**: Check if `--fmad=false` / `-ffp-contract=off` needed for bit-exact parity
- [ ] **Rounding mode**: Check `__fmul_rn`, `__fadd_rn` usage
- [ ] **Kernel launch config**: Verify grid/block dims work on target arch's max threads/block

### Loose Mode (best-effort)

- [ ] **Backend dispatch**: Verify `#if STRATA_USE_HIP` vs CUDA path exists
- [ ] **Fallback path**: Verify F32 fallback exists for any GPU-only instruction
- [ ] **Feature guard**: Any arch-specific feature is behind a compile-time guard
- [ ] **TODO markers**: Add `TODO(kands:strict-review)` for uncertain paths

---

## C++ Doxygen Block Template

```cpp
/// @file {{filename}}
/// @brief {{brief_description}}
///
/// @par Portability (K_and_S)
/// Backend: {{backend}} | Arch: {{archs}} | Mode: {{mode}}
/// Audit: TODO(kands:audit-{{mode}})
///
/// @par Breadcrumb tracing
/// ```cpp
/// STRACE("{{component}}", "{{phase}}", "{{tag}}", "extra=%s", "data");
/// STRACE_SCOPE("{{component}}", "{{phase}}", "{{tag}}", "extra=%s", "data");
/// ```
```

---

## Rust (rustdoc) Template

```rust
//! {{module_name}} — {{description}}
//!
//! # Portability (K_and_S, mode: {{mode}})
//!
//! Backend: `{{backend}}` | Arches: {{archs}}
//!
//! ## Audit Checklist
//! - [ ] Arch-specific `#[cfg(target_feature = "...")]` branches
//! - [ ] GPU instruction intrinsics (check via `spirv-tools` or `rust-vk-parse`)
//! - [ ] TODO(kands:strict-review) for unaudited paths
//!
//! ## Breadcrumb tracing
//! ```rust
//! strata_trace!("{{component}}", "{{phase}}", "{{tag}}", "extra={}", data);
//! ```
```

---

## Python Docstring Template

```python
"""
{{module_name}} — {{description}}.

Portability (K_and_S, mode: {{mode}}):
  Backend: {{backend}} | Arches: {{archs}}

Audit TODO(kands:audit-{{mode}}):
  - [ ] Arch-specific imports / ctypes dispatch
  - [ ] Check kernel.py / kernels/ subdirectory

Breadcrumb tracing:
  STRACE("{{component}}", "{{phase}}", "{{tag}}", "extra")
"""
```
