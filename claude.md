# claude.md — Claude-Specific Project Configuration

This file contains project-specific configuration and guidelines for Claude code assistance.

---

## Project Overview

This is a GPU kernel porting codebase targeting AMD RDNA and CDNA architectures from CUDA/NVIDIA sources. The project uses the K_and_S (Kernel & Shader Porting) skill bundle for AI agent guidance.

---

## Agent Configuration

- **Primary agent mode**: `strict` — exhaustive porting with numerical verification
- **Target backends**: hip (AMD ROCm), cuda (NVIDIA), vulkan, directx, sycl
- **Target architectures**: gfx1030, gfx1031, gfx1100, gfx1150, gfx1201 (and cdna2, cdna3)
- **Current focus**: Autonomous port path selection without user prompting

---

## Key Directories

- `mcp/server.py` — Main porting orchestration and autonomous path selection
- `mcp/api_map.py` — API/feature reference for pre-porting analysis
- `mcp/porting_map.py` — Function-to-function porting map builder
- `ARCH_MAP.json` — Architecture capability registry (vendor + generation features)
- `MODEL_MAP.json` — Model family → kernel/dispatch matrix mapping
- `TARGET_LANGUAGES.json` — Target language metadata for GPU porting
- `docs/` — Doxygen-style audit templates and documentation
- `findings/` — Porting contribution reports and portfolio

---

## Porting Workflow Phases (0-8)

**Phase 0**: Establish baseline — detect toolchain, capture perf tracing
**Phase 1**: Census — discover all kernel files in repo
**Phase 2**: Compare — compare source vs target architecture features
**Phase 3**: Query — query kernel features and arch capabilities
**Phase 4**: Detect — find installed compilers on the machine
**Phase 5**: Estimate — estimate performance on target arch
**Phase 6**: Target — gap analysis against target tok/s
**Phase 7**: Auto-plan — autonomous port path selection
**Phase 8**: Verify — numerical equivalence and export contributions

---

## Prohibited Anti-Patterns

- ❌ Never assume `hipify-perl handles everything`
- ❌ Never skip parity testing with CPU oracle
- ❌ Never deploy experimental code untested
- ❌ Never hardcode toolchain paths
- ❌ Never ignore the current date (search tools filter by date)
- ❌ Never modify files out of scope

---

## Required Guardrails

- ✅ Always call `reject_common_pattern()` before implementing "standard" techniques
- ✅ Always verify via `kands://isa-references` for instruction availability
- ✅ Always capture `--perf-tracing` baseline before changes
- ✅ Always run `verify_numerical_equivalence()` after each port
- ✅ Always use `detect_toolchain()` to find actually-installed compilers
- ✅ Document assumptions as `TODO(kands:strict-review)` for follow-up

---

## Architecture Notes

- **gfx1030**: RDNA 1.0, WMMA unavailable, F32 fallback path
- **gfx1031**: RDNA 3.5, no WMMA, bf16_native=True, dot4=True (added to ARCH_MAP.json)
- **gfx1100**: RDNA 2, fully validated with WMMA attention path
- **gfx1150**: RDNA 3, same portable kernels as gfx1100
- **gfx1201**: RDNA 4, validated with same portable kernels as gfx1100 but different LDS layout
- **cdna2**: AMD CDNA 2 (MI200), wmma=True, bf16_native=True, dot4=True, tensor_cores=True
- **cdna3**: AMD CDNA 3 (MI300), adds fp8_native=True on top of cdna2 features

---

## Language Backend Mapping

- `hip` → HIP C++ (.cu → .hsaco), ROCm runtime
- `cuda` → CUDA C++ (.cu → .cubin), CUDA runtime
- `vulkan` → GLSL/SPIR-V (.comp → .spv), Vulkan device/queues
- `directx` → HLSL (.hlsl → .dxil), DirectX 12 device/command lists
- `sycl` → SYCL parallel_for, DPC++ queue/event model

---

## Contribution Export

After porting work is complete, export findings with:
```
export_contribution(engine_name="{{ENGINE}}", arch="{{TARGET_ARCH}}")
```

This generates a report in `findings/portfolio.md` with:
- Architecture chosen and rationale
- Kernels ported and fidelity grades
- Feature gaps and workarounds
- Performance estimates and verification results