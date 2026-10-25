# Porting Methodology: From Unknown Engine → Known Architecture + Model

> **Audience:** porting engineer, graphics/compute engineer, performance optimizer.
> **Goal:** produce a complete inventory of kernels/shaders, map them to target GPU
> architectures and model families, instrument for debugging, and establish
> performance baselines — all with reusable, checklisted steps.

---

## Phase 0 — Setup

| Step | Command | Output |
|---|---|---|
| 0.1 Clone repo | `git clone <repo>` | Working tree (or copy existing source) |
| 0.2 Identify build system | `ls CMakeLists.txt setup.py setup.sh *.vcxproj meson.build` | Build tool name |
| 0.3 Identify GPU backend | grep for `cuda`, `hip`, `vulkan`, `metal`, `opencl` | Backend name |
| 0.4 Scan environment | `read kands://status` for toolkits + env vars | Available: ROCm/CUDA/Vulkan/DirectX/oneAPI paths |
| 0.5 Diagnose the target backend | `discover_kernels(repo=..., target_backend=..., target_arch=..., host_os=...)` | Requirement rows, Pareto vital few, PCIe ReBAR gate |
| 0.6 Set goal + guardrails | `set_goal(goal="...", target_arch="...", ...)` | kands://goal state |
| 0.7 Establish build | Build at least one target (e.g. `--help`) | Build OK / Fail |

### Phase 0 Gate: Backend Readiness

Step 0.5 answers *what the target needs before the first kernel is edited*. Every
requirement row is probed and ends in exactly one label: **PASS**, **FAIL**,
**NOT RUN** or **UNVALIDATED**. Rows are weighted `impact (1-5) x likelihood
(0-1)`; the Pareto vital few (80% of the total score) is the resolution order.
A non-PASS row is not a detail to remember later: it either selects its declared
fallback or stops the port until it is resolved or explicitly waived.

PCIe ReBAR is part of the gate for every backend that can use a large BAR
(`hip`, `cuda`, `vulkan`, `directx12`, `sycl`). A non-PASS ReBAR selects the
staging + chunked-copy fallback and that choice is recorded in the plan; it is
never assumed and never described as zero-copy without evidence. Metal has no
discrete BAR window, which is recorded as not applicable.

`autonomous_port` runs the same diagnosis in its Phase 0.

### Phase 0 Rule: No Placeholder Stub Code

> **Never** add placeholder, stub, demo, or "TODO fake" code to make things "work."
> This includes: stubbed success paths, hard-coded results, fake implementations,
> commented-out code that should be real, "works for this example only" solutions.
>
> If a kernel cannot be ported to the target arch, implement it with a **correct
> F32 fallback** — never a stub. If a feature is missing, document it and
> implement a real (if slower) path. Every kernel must produce real output.

## Phase 1 — Kernel/Shader Census

**Goal:** enumerate every kernel and shader in the source tree.

```bash
# Run through the MCP server:
python -m server
# Then call: discover_kernels(repo="/path/to/engine")
```

### Step 1.1 — GPU Kernel Sources
- `.cu` / `.cuh` / `.hip` → CUDA/HIP kernels
- `.comp` → Vulkan compute shaders
- `.metal` → Metal shaders
- `.hlsl` → Direct3D shaders
- `.cl` → OpenCL kernels
- `.spv` → SPIR-V binaries

### Step 1.2 — Source-to-Target Mapping
Read `CMakeLists.txt` / `meson.build` to map each source into its build target.

### Step 1.3 — Architecture Branches
Find `#if`/`#ifdef` guards for GPU architectures:
- `__CUDA_ARCH__`, `sm_XX`, `SM_XX` → CUDA arch
- `__HIP__`, `gfx1XX`, `wave32`, `wave64` → HIP/AMD arch
- `VULKAN_`, `MTL_DEVICE`, `VK_` → Vulkan/Metal

## Phase 2 — Architecture Capability Mapping

Read `ARCH_MAP.json` for the canonical GPU feature matrix, or build your own:

| Feature | CUDA sm_75 | CUDA sm_89 | AMD gfx1030 | AMD gfx1201 |
|---|---|---|---|---|
| Wave size | 32 | 128 | 32 | 32 |
| Tensor cores | ✅ | ✅ | ❌ | ❌ |
| WMMA | ✅ | ✅ | ❌ | ❌ |
| Dot4 | ✅ | ✅ | ✅ (older) | ✅ |
| BF16 native | ❌ | ✅ | ❌ | ✅ |
| FP8 native | ❌ | ❌ | ❌ | ❌ |

Use `compare_archs(arch_a, arch_b)` to diff two architectures.

## Phase 3 — Model Family Mapping

Read `MODEL_MAP.json` for the model-to-kernel matrix.  Each entry lists:

| Field | Example |
|---|---|
| `engine` | `engine_alpha` |
| `kernels_needed` | `{attention: [...], router: [...], moe_gate_up: [...]}` |
| `kernels_new` | Kernels needing new implementation |
| `dispatch_points` | Where model-specific logic branches |

Use `check_arch_compatibility(kernel_list, arch)` to validate per-kernel.

## Phase 4 — Instrumentation (Breadcrumb + Perf Tracing)

> **New in v0.1.0:** Add breadcrumb tracing BEFORE porting to catch regressions,
> and perf-tracing AFTER porting to measure improvements.

### Step 4.1 — Breadcrumb Tracing

Add a lightweight tracing header (example from Project Maya):

```cpp
// include/strata/core/breadcrumb.hpp
#define STRACE(component, phase, tag, ...) \
    do { \
        if (::strata::trace::breadcrumb_enabled()) { \
            char _extra[256]; \
            std::snprintf(_extra, sizeof(_extra), __VA_ARGS__); \
            ::strata::trace::emit(component, phase, tag, _extra); \
        } \
    } while (0)
```

**Output to stderr (TSV):**
```
<epoch_ns>  <thread_id>  <component>  <phase>  <tag>  <extra>
```

**Zero-cost when disabled:** inline runtime check on `STRATA_BREADCRUMB` env var.

### Step 4.2 — Perf Tracing

Add a summary report at process exit:

```cpp
// At main() exit:
if (std::getenv("STRATA_PERF_TRACING") != nullptr) {
    std::fprintf(stderr, "PERF_TRACING component=<engine> phase=summary ops=%lld gpu_ms=%.2f\n",
                 tokens, gpu_ms);
}
```

**CLI flags:**
- `--breadcrumb` → sets `STRATA_BREADCRUMB=1`
- `--perf-tracing` → sets `STRATA_PERF_TRACING=1` + auto-enables `STRATA_GLM_PROF` / `DS4_PROF`

### Step 4.3 — Where to Instrument

Instrument every kernel dispatch and model stage:

| Stage | Component | Key traces |
|---|---|---|
| Model init | `app` | `main:start` |
| Layer loop | `glm` / `ds4` | `forward`, `step`, `layer`, `run_layer` |
| Attention | `glm` | `qsa_decode_attn`, `qsa_prompt_attn` |
| Router | `moe` | `dispatch`, `dispatch_multi` |
| Expert pool | `pool` | `admit`, `begin`, `end` |
| Prefill | `prefill` | All `PfPhase` markers |
| Vision | `ds4` | `VisionTower::encode` |
| Speculation | `app` | `session_loop` |

### Step 4.4 — Using Instrumentation for Porting

1. **Before porting:** Run with `--perf-tracing` on the source arch to capture a baseline.
2. **After porting:** Run with `--perf-tracing` on the target arch and diff the stage-by-stage breakdown.
3. **During debugging:** Run with `--breadcrumb` to trace the exact code path taken per token.

The perf summary format `PERF_TRACING component=X phase=summary ops=N gpu_ms=X.XX` is
designed for diffing — pipe both runs into a script and compare stage timings.

## Phase 5 — Porting Gap Analysis

### Step 5.1 — Coverage Matrix

| Kernel | Arch 1 | Arch 2 | Arch 3 |
|---|---|---|---|
| native_flash_attn | OK | MISS | NEW |

### Step 5.2 — Performance Path

Identify the 5-10 hottest kernels (from perf tracing) and map each to the target arch's feature set.

### Step 5.3 — Memory Requirements

Document per-model+arch memory: weights, activations, KV cache, expert cache.

## Phase 6 — Porting Task Breakdown

Order by: **Blocking → Correctness → Tuning → Documentation**

## Phase 7 — Verification

| Test | Command | Pass Criteria |
|---|---|---|
| Build | `cmake --build .` | Compiles for target arch |
| Perf baseline | `--perf-tracing` | Matches Phase 4 baseline ±10% |
| Perf comparison | `--perf-tracing --benchmark` | Stage timings within 2× of reference |
| Correctness | `ctest -R parity` | All parity tests pass |
| Breadcrumb trace | `--breadcrumb` | No unexpected component/phase entries |

## Phase 8 — Share (Contribute Back)

**Goal:** turn your work into a reusable contribution for the K_and_S bundle.

| Step | Command | Output |
|---|---|---|
| 8.1 Record measurements | `record_performance(arch, workload, tok_per_s)` | perf_history.json updated |
| 8.2 Export findings | `export_contribution(engine_name, arch, workload)` | findings/<name>.md created |
| 8.3 Sanitize | Auto-applied (no emails, paths, git hashes) | Clean markdown file |
| 8.4 Prepare PR | `prepare_pull_request()` | git commands + PR description |
| 8.5 Submit | `git add, commit, push, gh pr create` | Pull request opened |

> **Remember:** Never add placeholder/stub code. Every kernel must produce real,
> verifiable output. Use the CPU-as-oracle strategy (see `kands://diagnostics`)
> for verification. Record your perf data — it helps future porters.

---

## Optimizer's Quick Reference

When optimizing on a NEW architecture:

1. Run `--perf-tracing` to get the stage breakdown
2. Find the slow stage (e.g., `gpu_ms between_layers=45%`)
3. Use `query_kernel_features` to find arch-specific branches for that kernel
4. Use `compare_archs` to see what features the target is missing
5. Read `kands://models` to find alternative kernels for the same stage
6. Instrument with `--breadcrumb` for fine-grained path tracing during tuning

---

## Reuse Template

To port a new engine, run `autonomous_port(repo_path="/path/to/engine", source_arch="sm_80", target_arch="gfx1201")`
for a fully automated report, or copy `findings/portfolio.md` as a template, follow the phases, and commit
your findings back to `findings/<engine_name>.md`.
