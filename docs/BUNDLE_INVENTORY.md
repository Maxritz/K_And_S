# K_and_S Bundle — Complete File Inventory

This document tracks all files in the K_and_S bundle for audit and contribution purposes.

## Data Files (JSON)

| File | Size | Contents |
|---|---|---|
| `ARCH_MAP.json` | 18 arches | GPU architecture → kernel features (NVIDIA/AMD/Intel + meteor_lake) with Vulkan DP4a notes |
| `MODEL_MAP.json` | 5 families | Model family → kernel requirements + dispatch points |
| `TK_TOOLS.json` | 7 toolkits | GPU (ROCm 10.1/CUDA/Vulkan/DirectX/OpenVINO/oneAPI) + host (MSVC/MinGW/MSYS2/GCC/Clang/Strawberry Perl) |
| `CROSS_PORT.json` | — | C++↔Python↔Rust↔PTX/SPIR-V type/memory/ABI mappings |
| `MODEL_FORMATS.json` | — | GGUF/Safetensors/ONNX/HDF5 format matrices |
| `PORTING_PATTERNS.json` | — | CUDA↔HIP↔SYCL↔Vulkan↔DirectX + sync_patterns (TDR/fence) |
| `PERF_BENCHMARKS.json` | — | Theoretical perf estimates + roofline models (7 arches) |
| `AI_FRAMEWORKS.json` | 12 frameworks | TileLang, HipKittens, ThunderKittens, Triton-Windows, ExecuTorch, TorchInductor |

## Documentation (Markdown)

| File | Purpose |
|---|---|
| `PORTING_METHOD.md` | Phase 0–8 methodology (incl. Phase 8: Share/Contribute) |
| `PORTING_STRATEGIES.md` | Copy vs Adapt vs Rewrite + strict/loose modes |
| `ISA_REFERENCES.md` | AMD XML, Intel PISA/IGA, NVIDIA PTX + Vulkan notes |
| `docs/DIAGNOSTICS.md` | Ponytail-diag, sherlock-it, CPU-as-oracle, perf optimization loop |
| `docs/DOX_STYLE.md` | Doxygen/rustdoc/Python audit templates |
| `docs/GOAL_TEMPLATE.md` | Goal + guardrails template |
| `CONTRIBUTING.md` | How to contribute findings back |
| `TEMPLATE.md` | Blank porting report template |
| `README.md` | Quick start + usage guide |

## MCP Server

| Path | Purpose |
|---|---|
| `mcp/server.py` | Main MCP server (26 tools, 20 resources, MCP 2.x, runtime toolchain detection) |
| `mcp/reference_search.py` | ArXiv + GitHub search utilities |
| `mcp/pyproject.toml` | `pip install -e .` packaging |
| `mcp/claude_desktop_config.json` | Ready-to-use MCP client config |
| `mcp/state/goal.json` | Goal + guardrail state (local) |
| `mcp/state/perf_history.json` | Performance measurement history (local) |
| `mcp/state/analysis/` | Cached analysis results (markdown) |

## Key Numbers

- **18** GPU architectures (NVIDIA/AMD/Intel + meteor_lake)
- **5** model families (DeepSeek, GLM, Qwen, LLaMA, etc.)
- **7** GPU toolkits + **6** host toolchains (incl. Strawberry Perl)
- **5** model file formats (GGUF, Safetensors, ONNX, HDF5, MLC)
- **26** MCP tools (incl. `detect_toolchain`, `search_online`, `reject_common_pattern`, `radical_experiment_plan`)
- **20** MCP resources (incl. `kands://radical-experiments`)
- **Phase 0-8** methodology (incl. Phase 8: Share/Contribute)
- **3** diagnostic methods (Ponytail-diag, sherlock-it, CPU-as-oracle)

## Design Principles

1. Engine-agnostic — works with any GPU engine
2. Date-aware — current UTC date exposed for search filtering
3. Auto-sanitized — emails, paths, git hashes stripped from recordings
4. Cached analysis — re-run only when source data changes
5. No placeholders — every kernel must produce real output
6. CPU-as-oracle — verification strategy for correctness (`verify_numerical_equivalence`)
7. Flexible goals — users can override guardrails (logged for audit)
8. Vulkan-aware — AMD CooperativeMatrix broken → use DP4a
9. Runtime detection — `detect_toolchain()` probes filesystem/PATH/env vars dynamically
10. No hardcoded paths — toolchain paths resolved at runtime, not in config files

## Data Storage Design

- **JSON files** — configuration, static reference data (archs, patterns, benchmarks)
- **perf_history.json** — append-only performance measurements (candidate for LMDB migration)
- **LMDB option** — master DB (accumulated porting knowledge) + per-project sub-DBs
  - Master: patterns, findings, optimization results
  - Sub-DBs: project-specific config, cache, notes
  - Use when perf_history grows beyond ~10K records or real-time queries needed