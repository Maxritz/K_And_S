# K_and_S — Kernel & Shader Porting Skill Bundle

A reusable methodology (as an MCP server) for mapping any GPU engine's kernels and shaders
to target GPU architectures and model families, with breadcrumb/perf-tracing instrumentation
built into every phase. The name "K_and_S" is mnemonic: **K**ernels **a**nd **S**haders.

---

## What This Bundle Provides

| Artifact | Purpose |
|---|---|
| `PORTING_METHOD.md` | Phase 0–8 methodology (setup → census → arch map → models → instrument → gaps → tasks → verify → share) |
| `ARCH_MAP.json` | GPU architecture → kernel features matrix (17 entries: NVIDIA/AMD/Intel) |
| `MODEL_MAP.json` | Model family → kernel requirements + dispatch points |
| `TK_TOOLS.json` | GPU toolkits (ROCm, CUDA, Vulkan, DirectX, oneAPI, OpenVINO) + host toolchains (MSVC, MinGW, MSYS2, GCC, Clang) |
| `CROSS_PORT.json` | C++ ↔ Python ↔ Rust ↔ PTX/SPIR-V porting matrix |
| `AI_FRAMEWORKS.json` | PyTorch, TensorFlow, JAX, llama.cpp + LLM engine APIs (OpenAI, vLLM, TGI, Triton, MCP) |
| `MODEL_FORMATS.json` | GGUF, Safetensors, ONNX, HDF5 — quant support + GPU porting implications |
| `PORTING_PATTERNS.json` | CUDA↔HIP↔SYCL↔Vulkan↔DirectX transformation patterns |
| `PERF_BENCHMARKS.json` | Theoretical performance + web search queries by arch |
| `ISA_REFERENCES.md` | Machine-readable ISA docs for AMD, Intel, NVIDIA + Vulkan DP4a vs CoopMatrix |
| `PORTING_STRATEGIES.md` | Copy vs Adapt vs Rewrite decision guide + strict/loose modes |
| `DIAGNOSTICS.md` | Ponytail-diag + sherlock-it methods, CPU-as-oracle, iterative perf loop |
| `DOX_STYLE.md` | Doxygen/rustdoc/Python docstring audit templates |
| `GOAL_TEMPLATE.md` | Goal + guardrails template for autonomous porting |
| `CONTRIBUTING.md` | How to contribute findings back to the bundle |
| `TEMPLATE.md` | Blank porting report template |
| `mcp/server.py` | MCP server (16 tools + 16 resources, MCP 2.x) |
| `mcp/reference_search.py` | ArXiv + GitHub reference search utilities |
| `findings/portfolio.md` | Worked example (engine-agnostic) |

---

## Quick Start

### As an MCP Server (recommended)

```bash
# 1. Install MCP
pip install mcp

# 2. Add to your MCP client config:
{
  "mcpServers": {
    "kands": {
      "command": "python",
      "args": ["-m", "mcp_kands.server"],
      "env": { "PYTHONPATH": "/path/to/K_and_S/mcp" }
    }
  }
}
```

### 3. In your AI assistant, ask:

```
# Browse all GPU architectures
> Read kands://architectures

# Set your ultimate goal + guardrails (prevents drift)
> Call set_goal with goal="Port llama.cpp to gfx1201 at 25+ tok/s", target_arch="gfx1201", target_backend="hip", target_tok_per_s=25

# Auto-clone + analyze a repo (fully autonomous)
> Call autonomous_port with github_url="https://github.com/ggerganov/llama.cpp", source_arch="sm_80", target_arch="gfx1201"

# Check a proposed action against your guardrails
> Call check_guardrail with action="modify src/kernels/attention.cu", reason="adding DP4a fallback"

# Search for optimization references
> Call search_references with query="AMD RDNA4 flash attention DP4a performance"

# After optimizing, record your result
> Call record_performance with arch="gfx1201", workload="flash_attn", tok_per_s=18.5

# Export findings to share back
> Call export_contribution with engine_name="llama-port", arch="gfx1201"
```

### As a CLI tool

```python
PYTHONPATH=/path/to/K_and_S/mcp python -c "
import asyncio, sys
sys.path.insert(0, '/path/to/K_and_S/mcp')
from server import _dispatch_tool
result = asyncio.run(_dispatch_tool('autonomous_port', {
    'repo_path': '/path/to/engine',
    'source_arch': 'sm_80',
    'target_arch': 'gfx1201',
    'target_backend': 'hip',
    'mode': 'strict'
}))
print(result[0].text)
"
```

---

## MCP Resources

| URI | Description |
|---|---|
| `kands://architectures` | All GPU arches with capabilities (wave size, tensor cores, dot4, WMMA, bf16, FP8) — includes Vulkan DP4a notes |
| `kands://models` | Model families → kernel requirements (DeepSeek, GLM, Qwen, LLaMA) |
| `kands://porting-steps` | Phase 0-8 full methodology |
| `kands://porting-template` | Blank porting report template |
| `kands://tracing-steps` | Breadcrumb + perf-tracing instrumentation guide |
| `kands://isa-references` | AMD XML, Intel PISA/IGA, NVIDIA PTX + Vulkan CooperativeMatrix/DP4a |
| `kands://tk-tools` | GPU + host toolchain matrix (ROCm/CUDA/Vulkan/DirectX/oneAPI/OpenVINO) |
| `kands://cpu-os` | Host toolchain matrix (MSVC/MinGW/MSYS2/GCC/Clang) — paths like `C:\msys64` |
| `kands://cross-port` | C++ ↔ Python ↔ Rust ↔ PTX/SPIR-V type/memory/ABI mappings |
| `kands://ai-frameworks` | PyTorch/TF/JAX/llama.cpp + LLM engine APIs |
| `kands://model-formats` | GGUF, Safetensors, ONNX, HDF5 — quant support matrix |
| `kands://porting-patterns` | CUDA↔HIP↔SYCL↔Vulkan↔DirectX transformation patterns |
| `kands://porting-strategies` | Copy vs Adapt vs Rewrite guide + strict/loose modes |
| `kands://diagnostics` | Ponytail-diag + sherlock-it methods, CPU-as-oracle, perf optimization loop |
| `kands://dox-style` | Doxygen/rustdoc/Python audit templates |
| `kands://status` | Live date/time, resource counts, data freshness |
| `kands://perf-history` | Local performance measurements (learned over time) |
| `kands://perf-benchmarks` | Theoretical perf estimates + web search queries |
| `kands://setup-guide` | Interactive target selection (backend, OS, arch, host compiler) |
| `kands://goal` | Current goal + guardrails state + override log |
| `kands://findings/{name}` | Completed porting analyses |

## MCP Tools

| Tool | Purpose |
|---|---|
| `discover_kernels` | Scan any repo for `.cu`/`.comp`/`.metal`/`.cl`/`.cpp` kernel sources |
| `check_arch_compatibility` | Check if kernel list is compatible with a target arch |
| `query_kernel_features` | Find arch-specific `#if` branches + feature flags in source |
| `compare_archs` | Diff two arches' feature matrices |
| `generate_porting_report` | Walk Phase 0-8 for a target engine+arch+model |
| `generate_tracing_plan` | Produce STRACE() placement plan for any engine |
| `diff_perf_traces` | Compare before/after `--perf-tracing` output |
| `query_toolchain` | Look up compiler paths, nvcc flags, OS-specific toolchain details |
| `estimate_performance` | Estimate theoretical + real-world GPU perf for a workload |
| `compare_to_target` | Compare your target tok/s vs estimated achievable |
| `record_performance` | Record measured performance into local history (auto-sanitized) |
| `suggest_optimizations` | Read perf history, suggest next optimization steps |
| `export_contribution` | Export findings as reusable git contribution template |
| `set_goal` | Define ultimate goal + guardrails (prevents agent drift) |
| `update_goal` | Override/update goal or guardrails (flexible, logged) |
| `check_guardrail` | Verify action is within defined guardrails |
| `prepare_pull_request` | Generate git commands + PR description for contributions |
| `autonomous_port` | **Full auto:** clone → discover → analyze → research → plan (one call) |
| `search_arxiv` | Scan ArXiv for GPU ISO/kernel/porting papers |
| `search_github` | Search GitHub for kernel/porting code references |
| `search_references` | Combined ArXiv + GitHub search |

---

## The Core Question This Bundle Answers

> "To port engine X to architecture Y for model family Z, which kernels and shaders
> must compile, which must be re-implemented, and which must be performance-tuned?"

The bundle answers this with a deterministic, auditable checklist — so a future port
to a new GPU or model family is a checklist, not a research project.

---

## Key Features

- **Engine-agnostic:** Works with any GPU engine (llama.cpp, Maya/Strata, your own)
- **Vulkan-aware:** Documents that CooperativeMatrix is broken on AMD → use DP4a
- **Date-aware:** Every search shows current UTC date/time; perf history is timestamped
- **Auto-sanitized:** Personal data (emails, paths, git hashes) stripped from all recordings
- **Cached analysis:** Results saved to markdown; re-run only when source data changes
- **CPU-as-oracle:** Verification strategy using CPU reference implementations as golden output
- **Iterative perf loop:** Declare bottleneck → trace (sherlock-it) → dissect → hypothesize → experiment → verify
- **Community contribution:** One-click export of findings as shareable templates

---

## Vulkan on AMD: The Gotchas

| Issue | Solution |
|---|---|
| `GL_KHR_cooperative_matrix` is broken on AMD Vulkan drivers | Use `GL_KHR_shader_integer_dot_product` (DP4a / `spv_dp4`) for matmul |
| No rocWMMA on gfx1030/gfx1200/gfx1201 | Use F32 fallback for attention |
| Wave64 on CDNA (gfx908) | Use logical-32 half-wavefront compat layer |
| 64 KiB LDS on RDNA2 | Optimize shared memory tile sizes |

---

## Contributing

See `CONTRIBUTING.md` for full guidelines, or simply:

```bash
# After using the bundle:
# 1. record_performance(arch="gfx1201", workload="flash_attn", tok_per_s=22.0)
# 2. export_contribution(engine_name="my-engine", arch="gfx1201")
# 3. prepare_pull_request()  # gets the exact git commands
```

---

*The bundle repo on GitHub (github.com/MaxRitz/K_and_S) is open for contributions. Help make GPU porting a shared skill, not a secret.*
