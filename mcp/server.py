"""
mcp_kands — MCP server exposing kernel & shader porting + optimization knowledge.

Resources (URI: kands://...):
  kands://architectures      GPU architecture capability matrix (ARCH_MAP.json)
  kands://models             Model family → kernel requirements (MODEL_MAP.json)
  kands://porting-steps      Full Phase 0–7 porting + optimization methodology
  kands://porting-template   Blank porting report template
  kands://tracing-steps      Breadcrumb + perf-tracing instrumentation guide
  kands://isa-references     Machine-readable ISA URLs for AMD/Intel/NVIDIA + Vulkan
  kands://tk-tools           GPU + host toolchain matrix (ROCm/CUDA/Vulkan/DirectX/OpenVINO/oneAPI)
  kands://cpu-os             Host toolchain matrix (MSVC/MinGW/MSYS2/GCC/Clang + OS details)
  kands://cross-port         C++ ↔ Python ↔ Rust ↔ PTX/SPIR-V porting matrix
  kands://ai-frameworks      AI/ML framework matrix + LLM engine APIs
  kands://model-formats      Model file format matrix (GGUF, Safetensors, ONNX, HDF5)
  kands://porting-patterns   Engine-agnostic CUDA↔HIP↔SYCL↔Vulkan↔DirectX transformation patterns
  kands://porting-strategies Copy vs Adapt vs Rewrite decision guide
  kands://diagnostics        Ponytail-diag + sherlock-it diagnostic methods + CPU-as-oracle
  kands://dox-style          Doxygen/rustdoc/Python docstring audit templates
  kands://status             Live bundle status (date/time, resource counts, data freshness)
  kands://perf-history       Local performance measurements recorded by the user
  kands://perf-benchmarks    Theoretical perf estimates + web search queries
  kands://setup-guide        Interactive target selection guide (GPU backend, OS, arch)
  kands://findings/{name}    Completed porting analysis for a specific engine

Tools:
  discover_kernels()          Scan any repo for kernel/shader sources
  check_arch_compatibility()  Check kernel list vs target arch capabilities
  query_kernel_features()     Find arch-specific branches in source
  compare_archs()             Diff two GPU arches' feature matrices
  generate_porting_report()   Walk Phase 0-7 + Phase 8 (Share) for a target
  generate_tracing_plan()     Create a STRACE() instrumentation plan
  diff_perf_traces()          Compare two perf-tracing outputs (before/after port)
  query_toolchain()           Look up compiler paths, flags, OS-specific details
  estimate_performance()      Estimate theoretical + real-world perf for an arch + workload
  compare_to_target()         Compare your target tok/s vs. estimated achievable perf
  record_performance()        Record measured performance into local history
  suggest_optimizations()     Read perf history, suggest next steps
  export_contribution()       Export your learnings as a reusable git contribution template
  search_arxiv()             Scan ArXiv for updated GPU ISA/kernel/porting papers
  search_github()            Search GitHub for kernel/porting code references
  search_references()        Combined ArXiv + GitHub search for a porting question
  detect_toolchain()         Runtime-detect actual installed compilers (shutil.which + fs probe)
  search_online()            Search web for current toolchain URLs + version info
  verify_numerical_equivalence()  CPU-as-oracle: verify ported kernel output matches CPU reference
  set_goal()                 Define ultimate goal + guardrails for the porting project
  update_goal()              Override any goal field (logged for audit trail)
  check_guardrail()          Verify action aligns with defined guardrails
  prepare_pull_request()     Generate git commands + PR description for contribution
  autonomous_port()          Full auto: discover → compare arches → perf estimate →
                             → research → plan + security safeguards for github_url clone
"""
from __future__ import annotations
import json
import re
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field
from mcp.server import Server
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions
import mcp_types
from reference_search import search_arxiv, search_github, search_references

# ── Paths ──────────────────────────────────────────────────────────────────

_KANDS_ROOT = Path(__file__).resolve().parent.parent
_ARCH_MAP_PATH = _KANDS_ROOT / "ARCH_MAP.json"
_MODEL_MAP_PATH = _KANDS_ROOT / "MODEL_MAP.json"
_METHOD_PATH = _KANDS_ROOT / "PORTING_METHOD.md"
_TEMPLATE_PATH = _KANDS_ROOT / "TEMPLATE.md"
_ISA_REF_PATH = _KANDS_ROOT / "ISA_REFERENCES.md"
_TK_TOOLS_PATH = _KANDS_ROOT / "TK_TOOLS.json"
_CROSS_PORT_PATH = _KANDS_ROOT / "CROSS_PORT.json"
_AI_FRAMEWORKS_PATH = _KANDS_ROOT / "AI_FRAMEWORKS.json"
_MODEL_FORMATS_PATH = _KANDS_ROOT / "MODEL_FORMATS.json"
_PORTING_PATTERNS_PATH = _KANDS_ROOT / "PORTING_PATTERNS.json"
_PERF_BENCHMARKS_PATH = _KANDS_ROOT / "PERF_BENCHMARKS.json"
_DIAGNOSTICS_PATH = _KANDS_ROOT / "docs" / "DIAGNOSTICS.md"
_DOX_STYLE_PATH = _KANDS_ROOT / "docs" / "DOX_STYLE.md"
_PORTING_STRATEGIES_PATH = _KANDS_ROOT / "docs" / "PORTING_STRATEGIES.md"
_GOAL_STATE_PATH = Path(__file__).resolve().parent / "state" / "goal.json"
_GOAL_TEMPLATE_PATH = _KANDS_ROOT / "docs" / "GOAL_TEMPLATE.md"
_ANALYSIS_CACHE_DIR = Path(__file__).resolve().parent / "state" / "analysis"


# ── Analysis Cache ─────────────────────────────────────────────────────────────

_SOURCE_DATA_FILES = [
    _ARCH_MAP_PATH, _MODEL_MAP_PATH, _TK_TOOLS_PATH, _CROSS_PORT_PATH,
    _AI_FRAMEWORKS_PATH, _MODEL_FORMATS_PATH, _PORTING_PATTERNS_PATH, _PERF_BENCHMARKS_PATH,
]


def _cache_path_for(key: str) -> Path:
    """Get the cache file path for a given analysis key."""
    _ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _ANALYSIS_CACHE_DIR / f"{key}.md"


def _cache_is_stale(cache_file: Path) -> bool:
    """Check if cache is stale (any source data file newer than cache)."""
    if not cache_file.exists():
        return True
    cache_mtime = cache_file.stat().st_mtime
    for src in _SOURCE_DATA_FILES:
        if src.exists() and src.stat().st_mtime > cache_mtime:
            return True
    return False


def _read_cached_analysis(key: str, force_refresh: bool = False) -> str | None:
    """Read cached analysis if valid, or None if stale/missing."""
    if force_refresh:
        return None
    cache_file = _cache_path_for(key)
    if _cache_is_stale(cache_file):
        return None
    return cache_file.read_text(encoding="utf-8", errors="replace")


def _write_cached_analysis(key: str, content: str) -> None:
    """Write analysis result to cache."""
    cache_file = _cache_path_for(key)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(content, encoding="utf-8")


def _load_goal_state() -> dict:
    """Load the current goal + guardrails state."""
    if _GOAL_STATE_PATH.exists():
        return json.loads(_GOAL_STATE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_goal_state(state: dict) -> None:
    """Save goal + guardrails state."""
    state["last_modified"] = datetime.now(timezone.utc).isoformat()
    _GOAL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GOAL_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

# ── Static data loader ─────────────────────────────────────────────────────

_cached: dict[str, Any] = {}

def _arch_name(arch_map: dict, arch_id: str) -> str:
    """Get the display name for an arch ID."""
    entries = arch_map.get("entries", {})
    if arch_id in entries:
        return entries[arch_id].get("name", arch_id)
    return arch_id


def _load_json(path: Path) -> dict:
    cache_key = str(path)
    if cache_key not in _cached:
        if path.exists():
            _cached[cache_key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            _cached[cache_key] = {}
    return _cached[cache_key]


# ── Extension map for source discovery ─────────────────────────────────────

_EXTENSION_MAP = {
    ".cu":  ("gpu_cuda",  "CUDA/HIP kernel source"),
    ".cuh": ("gpu_cuda",  "CUDA/HIP kernel header"),
    ".hip": ("gpu_cuda",  "HIP kernel source"),
    ".comp":("gpu_vulkan","Vulkan compute shader"),
    ".vert": ("gpu_vulkan","Vulkan vertex shader"),
    ".frag": ("gpu_vulkan","Vulkan fragment shader"),
    ".geom": ("gpu_vulkan","Vulkan geometry shader"),
    ".tesc": ("gpu_vulkan","Vulkan tesselation control shader"),
    ".tese": ("gpu_vulkan","Vulkan tesselation eval shader"),
    ".spv":  ("gpu_spv",   "SPIR-V binary"),
    ".metal":("gpu_metal", "Metal shader"),
    ".hlsl": ("gpu_dx",    "Direct3D HLSL shader"),
    ".cl":   ("gpu_opencl","OpenCL kernel"),
}

_CUDA_EXTS = {".cu", ".cuh", ".hip"}
_SHADER_EXTS = {".comp", ".vert", ".frag", ".geom", ".spv", ".metal", ".hlsl", ".cl"}
_ALL_EXTS = _CUDA_EXTS | _SHADER_EXTS | {".cpp", ".hpp"}


# ── Pydantic argument models ────────────────────────────────────────────────

class DiscoverParams(BaseModel):
    repo: str = Field(..., description="Path to the repo to scan (git repo or source tree).")
    ext_filter: Optional[list[str]] = Field(
        default=None,
        description="Optional extension filter, e.g. ['.cu'] or ['.comp', '.metal']"
    )

class ArchCheckParams(BaseModel):
    kernel_list: list[str] = Field(..., min_length=1, description="List of kernel names to check.")
    arch: str = Field(..., description="Target architecture ID (e.g., 'gfx1201', 'sm_80').")

class KernelQueryParams(BaseModel):
    repo: str = Field(..., description="Repo path to search source files.")
    pattern: str = Field(..., description="Regex or literal string to match in source lines.")
    extensions: Optional[list[str]] = Field(
        default=None, description="Only search these extensions"
    )

class CompareParams(BaseModel):
    arch_a: str = Field(..., description="First architecture ID")
    arch_b: str = Field(..., description="Second architecture ID")

class ReportParams(BaseModel):
    engine: str = Field(..., description="Engine name (e.g., 'maya')")
    target_arch: str = Field(..., description="Target architecture ID (e.g., 'gfx1201')")
    model_family: str = Field("auto", description="Model family or 'auto'")
    repo_path: str = Field(..., description="Path to the engine repo")

class TracingPlanParams(BaseModel):
    repo: str = Field(..., description="Repo path to analyze for instrumentation points")
    components: Optional[list[str]] = Field(
        default=None,
        description="Components to instrument (default: auto-detect from MODEL_MAP)"
    )

class PerfDiffParams(BaseModel):
    trace_a: str = Field(..., description="Path to 'before' trace file or raw text")
    trace_b: str = Field(..., description="Path to 'after' trace file or raw text")


class AutonomousPortParams(BaseModel):
    repo_path: str = Field("", description="Path to the engine repo (must exist). Use github_url instead for remote repos.")
    github_url: str = Field("", description="GitHub URL to clone (e.g. https://github.com/ggerganov/llama.cpp). SECURITY: repo is read-only; no build scripts are executed. Must be https://github.com/ or git+https://github.com/")
    source_arch: str = Field(..., description="Source arch ID (e.g. 'sm_80', 'gfx1100')")
    target_arch: str = Field(..., description="Target arch ID (e.g. 'gfx1201', 'gen12_lp')")
    target_backend: str = Field("hip", description="Target backend: 'hip', 'cuda', 'vulkan', 'directx', 'sycl'")
    target_os: str = Field("auto", description="Target OS: 'linux', 'windows', 'auto'")
    target_tok_per_s: float = Field(0, description="Performance target (0 = no target)")
    mode: str = Field("strict", description="Porting mode: 'strict' or 'loose'")
    max_files: int = Field(1000, description="Maximum number of files to scan (security: prevents resource exhaustion from large repos)")
    timeout_seconds: int = Field(60, description="Timeout for clone + scan operations (seconds)")
    skip_clone: bool = Field(False, description="If true and repo_path is set, skip re-cloning (use existing local copy)")


class EstimatePerfParams(BaseModel):
    arch: str = Field(..., description="Target arch ID")
    workload: str = Field("flash_attn", description="Workload type")
    seq_len: int = Field(4096, description="Sequence length")


class CompareTargetParams(BaseModel):
    arch: str = Field(..., description="Target arch ID")
    workload: str = Field(..., description="Workload type")
    target_tok_per_s: float = Field(..., description="Target tokens/second")
    current_tok_per_s: float = Field(0, description="Current measured tokens/second")


class QueryToolchainParams(BaseModel):
    gpu_vendor: str = Field("AMD", description="GPU vendor: AMD, NVIDIA, Intel")
    os: str = Field("auto", description="Target OS: autolinux, windows, auto")
    host_compiler: str = Field("auto", description="Host compiler: msvc, gcc, clang, mingw, msys2, auto")


class SearchArxivParams(BaseModel):
    query: str = Field(..., description="Search query for ArXiv")
    max_results: int = Field(5, description="Max results (default 5, max 10)")


class SearchGithubParams(BaseModel):
    query: str = Field(..., description="Search query for GitHub code")
    max_results: int = Field(10, description="Max results (default 10, max 20)")


class SearchReferencesParams(BaseModel):
    query: str = Field(..., description="Search query for combined search")
    max_results: int = Field(5, description="Max results per source")


class RecordPerfParams(BaseModel):
    arch: str = Field(..., description="Target arch ID")
    workload: str = Field(..., description="Workload type")
    tok_per_s: float = Field(..., description="Measured tokens/second")
    config_hash: str = Field("", description="Build config description")
    notes: str = Field("", description="Notes about what was tried")


class SuggestOptParams(BaseModel):
    arch: str = Field(..., description="Target arch ID")
    workload: str = Field("", description="Workload type to optimize")
    target_tok_per_s: float = Field(0, description="Target performance goal")


class SetGoalParams(BaseModel):
    goal: str = Field(..., description="One-sentence ultimate goal")
    target_arch: str = Field(..., description="Target arch ID")
    source_arch: str = Field("", description="Source arch ID")
    target_backend: str = Field("hip", description="Target backend")
    target_tok_per_s: float = Field(0, description="Target tokens/second")
    min_perf: float = Field(0, description="Minimum viable perf")
    files_in_scope: list[str] = Field(default_factory=list, description="Glob patterns of files allowed to modify")
    files_out_of_scope: list[str] = Field(default_factory=list, description="Glob patterns that must NOT be touched")
    max_iterations: int = Field(10, description="Max optimization iterations")
    guardrails: list[str] = Field(default_factory=list, description="Additional guardrail rules")


class UpdateGoalParams(BaseModel):
    override_field: str = Field(..., description="Field to override")
    new_value: str = Field(..., description="New value as JSON string")
    reason: str = Field(..., description="Why making this override")


class CheckGuardrailParams(BaseModel):
    action: str = Field(..., description="What you want to do")
    reason: str = Field(..., description="Why this action is needed")


class VerifyNumericalEquivalenceParams(BaseModel):
    arch: str = Field(..., description="Target arch ID (e.g. 'gfx1201', 'sm_80')")
    workload: str = Field(..., description="Workload type (e.g. 'flash_attn', 'fp32_gemm')")
    source_values: list[float] = Field(..., description="Reference output values from source implementation (CPU oracle)")
    target_values: list[float] = Field(..., description="Target implementation output values (after porting)")
    tolerance: float = Field(1e-5, description="Absolute tolerance for FP comparison")
    relative_tolerance: float = Field(1e-4, description="Relative tolerance for FP comparison")


class RejectPatternParams(BaseModel):
    common_idea: str = Field(..., description="The common/conventional idea to scrutinize (e.g., 'hipify-perl handles everything', 'copy CUDA to HIP verbatim', 'Vulkan CooperativeMatrix works on AMD')")
    arch: str = Field("auto", description="Target arch ID to check against (or 'auto' for general)")


class RadicalExperimentParams(BaseModel):
    arch: str = Field(..., description="Target arch ID for experimentation")
    kernel_type: str = Field("gemm", description="Kernel type: gemm, attention, dequant, elementwise, memory_copy")
    experiment_type: str = Field("all", description="Type: all, memory, compute, scheduling, layout")


class PreparePRParams(BaseModel):
    repo_url: str = Field("https://github.com/MaxRitz/K_and_S", description="K_and_S repo URL")
    branch_name: str = Field("", description="Branch name (auto-generated if empty)")
    change_types: list[str] = Field(default_factory=lambda: ["findings"], description="Change types")


class DetectToolchainParams(BaseModel):
    gpu_vendor: str = Field("auto", description="GPU vendor: AMD, NVIDIA, Intel, or auto")
    host_compiler: str = Field("auto", description="Host compiler: msvc, gcc, clang, mingw, msys2, strawberry, auto")


class SearchOnlineParams(BaseModel):
    query: str = Field(..., description="What to search for online (e.g. 'ROC 10.1 install Windows')")
    max_results: int = Field(5, description="Max results per search source")


# ── Resource URI constants ─────────────────────────────────────────────────

RES_ARCH = "kands://architectures"
RES_MODELS = "kands://models"
RES_METHOD = "kands://porting-steps"
RES_TEMPLATE = "kands://porting-template"
RES_TK_TOOLS = "kands://tk-tools"
RES_CPU_OS = "kands://cpu-os"
RES_TRACING = "kands://tracing-steps"
RES_ISA = "kands://isa-references"
RES_CROSS_PORT = "kands://cross-port"
RES_AI_FRAMEWORKS = "kands://ai-frameworks"
RES_MODEL_FORMATS = "kands://model-formats"
RES_PORTING_PATTERNS = "kands://porting-patterns"
RES_PORTING_STRATEGIES = "kands://porting-strategies"
RES_DIAGNOSTICS = "kands://diagnostics"
RES_DOX_STYLE = "kands://dox-style"
RES_GOAL = "kands://goal"
RES_STATUS = "kands://status"
RES_PERF_HISTORY = "kands://perf-history"
RES_BENCHMARKS = "kands://perf-benchmarks"
RES_SETUP_GUIDE = "kands://setup-guide"
RES_RADICAL_EXPERIMENTS = "kands://radical-experiments"
RES_FINDINGS_PREFIX = "kands://findings/"


# ── Source scanning ─────────────────────────────────────────────────────────

def _scan_sources(root: Path, ext_filter: list[str] | None = None) -> list[dict]:
    """Walk a source tree and return all GPU / shader / CPU kernel files."""
    results: list[dict] = []
    exts = set(ext_filter) if ext_filter else set(_ALL_EXTS)

    for f in root.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext not in exts:
            continue
        # Skip CPU files if not explicitly requested (they're noise for GPU porters)
        if ext == ".cpp" and "cpu" not in str(f).lower():
            continue

        backend, kind = _determine_backend_kind(f, ext)
        if backend is None:
            continue

        results.append({
            "path": str(f.relative_to(root)),
            "backend": backend,
            "kind": kind,
        })

    results.sort(key=lambda r: r["path"])
    return results


def _determine_backend_kind(f: Path, ext: str) -> tuple[str | None, str | None]:
    """Determine the backend and kind based on file extension and path."""
    if ext in _EXTENSION_MAP:
        return _EXTENSION_MAP[ext]
    if ext == ".cpp":
        parts = f.parts
        if "cpu" in [p.lower() for p in parts]:
            name = f.name.lower()
            if "avx512" in name or "kq" in name:
                return ("cpu_x86_avx512", "CPU kernel (AVX-512)")
            if "avx2" in name:
                return ("cpu_x86_avx2", "CPU kernel (AVX2)")
            return ("cpu_x86", "CPU kernel (x86)")
    return (None, None)


# ── Arch compatibility logic ────────────────────────────────────────────────

_FEATURE_REQUIREMENTS = {
    "flash_attn":     {"wmma": True, "dot4": True},
    "mma":            {"tensor_cores": True},
    "native_qsa":     {"wmma": True, "dot4": True},
    "bf16_gemv":      {"bf16_native": True},
    "native_bf16":    {"bf16_native": True},
    "exl3":           {"dot4": True},
    "router_topk":    {"dot4": True},
    "router_sigmoid": {"dot4": True},
    "kv_q4":          {"dot4": True},
    "kv_q8":          {"bf16_native": True},
}


def _check_kernel_against_arch(kernel: str, arch_info: dict) -> str:
    """Return 'OK', 'NEEDS_FALLBACK', or 'INCOMPATIBLE'."""
    for pattern, reqs in _FEATURE_REQUIREMENTS.items():
        if pattern in kernel.lower():
            missing = []
            for feat, needed in reqs.items():
                if needed and not arch_info.get(feat):
                    missing.append(feat)
            if not missing:
                return "OK"
            return "NEEDS_FALLBACK"  # Most kernels have F32 fallback
    return "OK"


# ── Resource handlers ───────────────────────────────────────────────────────

def _list_resources() -> list[mcp_types.Resource]:
    return [
        mcp_types.Resource(
            uri=RES_ARCH,
            name="GPU Architecture Matrix",
            description="All known GPU architectures with their capabilities (NVIDIA/AMD/Intel)",
            mimeType="application/json",
        ),
        mcp_types.Resource(
            uri=RES_MODELS,
            name="Model Family Map",
            description="Model families → kernel requirements and dispatch points",
            mimeType="application/json",
        ),
        mcp_types.Resource(
            uri=RES_METHOD,
            name="Porting Methodology",
            description="Phase 0-7 porting and optimization methodology including instrumentation",
            mimeType="text/markdown",
        ),
        mcp_types.Resource(
            uri=RES_TEMPLATE,
            name="Porting Report Template",
            description="Blank template for a new porting report",
            mimeType="text/markdown",
        ),
        mcp_types.Resource(
            uri=RES_TRACING,
            name="Tracing & Instrumentation Guide",
            description="Breadcrumb + perf-tracing methodology for porting verification",
            mimeType="text/markdown",
        ),
        mcp_types.Resource(
            uri=RES_ISA,
            name="GPU ISA References",
            description="Machine-readable ISA URLs for AMD, Intel, NVIDIA GPUs + Vulkan CooperativeMatrix/DP4a notes",
            mimeType="text/markdown",
        ),
        mcp_types.Resource(
            uri=RES_TK_TOOLS,
            name="GPU & Host Toolchain Matrix",
            description="ROCm, CUDA, Vulkan, DirectX, oneAPI, OpenVINO + MSVC, MinGW, MSYS2, GCC, Clang",
            mimeType="application/json",
        ),
        mcp_types.Resource(
            uri=RES_CPU_OS,
            name="CPU / OS Matrix",
            description="Host CPU ISAs (AVX-512, AVX2, NEON) and OS-specific toolchain paths",
            mimeType="application/json",
        ),
        mcp_types.Resource(
            uri=RES_CROSS_PORT,
            name="Cross-Language Porting Matrix",
            description="C++ ↔ Python ↔ Rust ↔ PTX/SPIR-V type mappings, memory management, ABI boundary notes",
            mimeType="application/json",
        ),
        mcp_types.Resource(
            uri=RES_AI_FRAMEWORKS,
            name="AI/ML Framework Matrix",
            description="PyTorch, TensorFlow, JAX, llama.cpp + LLM engine APIs (OpenAI, vLLM, TGI, Triton)",
            mimeType="application/json",
        ),
        mcp_types.Resource(
            uri=RES_MODEL_FORMATS,
            name="Model File Formats",
            description="GGUF, Safetensors, ONNX, HDF5 — quantization support and GPU porting implications",
            mimeType="application/json",
        ),
        mcp_types.Resource(
            uri=RES_PORTING_PATTERNS,
            name="Porting Patterns",
            description="Engine-agnostic CUDA↔HIP↔SYCL↔Vulkan↔DirectX transformation mappings",
            mimeType="application/json",
        ),
        mcp_types.Resource(
            uri=RES_PORTING_STRATEGIES,
            name="Porting Strategies",
            description="Copy vs Adapt vs Rewrite decision guide + strict/loose mode instructions",
            mimeType="text/markdown",
        ),
        mcp_types.Resource(
            uri=RES_DIAGNOSTICS,
            name="Diagnostics Framework",
            description="Ponytail-diag + sherlock-it methods, CPU-as-oracle, iterative perf optimisation loop",
            mimeType="text/markdown",
        ),
        mcp_types.Resource(
            uri=RES_DOX_STYLE,
            name="Doxygen Audit Templates",
            description="Doxygen/rustdoc/Python docstring templates for kernel file auditing",
            mimeType="text/markdown",
        ),
        mcp_types.Resource(
            uri=RES_STATUS,
            name="Bundle Status",
            description="Live status: date/time, resource counts, data freshness",
            mimeType="application/json",
        ),
        mcp_types.Resource(
            uri=RES_PERF_HISTORY,
            name="Performance History",
            description="Local performance measurements recorded by the user (learned over time)",
            mimeType="application/json",
        ),
        mcp_types.Resource(
            uri=RES_BENCHMARKS,
            name="Perf Benchmarks",
            description="Theoretical performance estimates and web search references by arch",
            mimeType="application/json",
        ),
        mcp_types.Resource(
            uri=RES_SETUP_GUIDE,
            name="Setup Guide",
            description="Interactive guide: select GPU backend, OS, arch, and host compiler. Notes current date/time for search tools.",
            mimeType="text/markdown",
        ),
        mcp_types.Resource(
            uri=RES_RADICAL_EXPERIMENTS,
            name="Radical Experiment Ideas",
            description="RDNA/AMD GPU porting anti-patterns + unconventional kernel approaches. Use with radical_experiment_plan() tool.",
            mimeType="text/markdown",
        ),
    ]


def _read_resource(uri: str) -> str:
    if uri == RES_ARCH:
        return json.dumps(_load_json(_ARCH_MAP_PATH), indent=2)

    if uri == RES_MODELS:
        return json.dumps(_load_json(_MODEL_MAP_PATH), indent=2)

    if uri == RES_METHOD:
        if _METHOD_PATH.exists():
            return _METHOD_PATH.read_text(encoding="utf-8")
        return "# Porting Methodology\n\n(Method file not found)"

    if uri == RES_TEMPLATE:
        if _TEMPLATE_PATH.exists():
            return _TEMPLATE_PATH.read_text(encoding="utf-8")
        return "# Porting Report Template\n\n(Template not found)"

    if uri == RES_TRACING:
        return _tracing_steps_markdown()

    if uri == RES_ISA:
        if _ISA_REF_PATH.exists():
            return _ISA_REF_PATH.read_text(encoding="utf-8")
        return "# ISA References\n\n(References file not found)"

    if uri == RES_TK_TOOLS:
        return json.dumps(_load_json(_TK_TOOLS_PATH), indent=2)

    if uri == RES_CPU_OS:
        tk = _load_json(_TK_TOOLS_PATH)
        # CPU/OS data is embedded in TK_TOOLS.json host_toolchains section
        return json.dumps(tk.get("host_toolchains", {}), indent=2)

    if uri == RES_CROSS_PORT:
        return json.dumps(_load_json(_CROSS_PORT_PATH), indent=2)

    if uri == RES_AI_FRAMEWORKS:
        return json.dumps(_load_json(_AI_FRAMEWORKS_PATH), indent=2)

    if uri == RES_MODEL_FORMATS:
        return json.dumps(_load_json(_MODEL_FORMATS_PATH), indent=2)

    if uri == RES_PORTING_PATTERNS:
        return json.dumps(_load_json(_PORTING_PATTERNS_PATH), indent=2)

    if uri == RES_PORTING_STRATEGIES:
        if _PORTING_STRATEGIES_PATH.exists():
            return _PORTING_STRATEGIES_PATH.read_text(encoding="utf-8")
        return "# Porting Strategies\n\n(Not found)"

    if uri == RES_DIAGNOSTICS:
        if _DIAGNOSTICS_PATH.exists():
            return _DIAGNOSTICS_PATH.read_text(encoding="utf-8")
        return "# Diagnostics\n\n(Not found)"

    if uri == RES_DOX_STYLE:
        if _DOX_STYLE_PATH.exists():
            return _DOX_STYLE_PATH.read_text(encoding="utf-8")
        return "# Doxygen Style\n\n(Not found)"

    if uri == RES_STATUS:
        return json.dumps(_bundle_status(), indent=2)

    if uri == RES_SETUP_GUIDE:
        return _setup_guide_markdown()

    if uri == RES_PERF_HISTORY:
        hist = _load_perf_history()
        return json.dumps(hist, indent=2)

    if uri == RES_GOAL:
        state = _load_goal_state()
        return json.dumps(state, indent=2)

    if uri == RES_BENCHMARKS:
        bench = _load_perf_benchmarks()
        return json.dumps(bench, indent=2)

    if uri == RES_RADICAL_EXPERIMENTS:
        return _radical_experiments_markdown()

    if uri.startswith(RES_FINDINGS_PREFIX):
        name = uri[len(RES_FINDINGS_PREFIX):]
        findings_path = _KANDS_ROOT / "findings" / f"{name}.md"
        if findings_path.exists():
            return findings_path.read_text(encoding="utf-8")
        return f"# {name}\n\n_No findings yet. Run `generate_porting_report` to create one._"

    raise ValueError(f"Unknown resource URI: {uri}")


def _tracing_steps_markdown() -> str:
    """The tracing/instrumentation methodology (Phase 4 of the porting method)."""
    return """# Phase 4: Instrumentation (Breadcrumb + Perf Tracing)

## When to Use This

- **Before porting:** Run `--perf-tracing` on the source arch to capture baseline stage timings.
- **After porting:** Run `--perf-tracing` on the target arch and diff stage-by-stage.
- **During optimization:** Run `--breadcrumb` to trace the exact code path, then measure with `--perf-tracing`.

## Two CLI Flags

| Flag | Env Var | Purpose |
|---|---|---|
| `--breadcrumb` | `STRATA_BREADCRUMB=1` | TSV trace to stderr: timestamp, thread, component, phase, tag, extra |
| `--perf-tracing` | `STRATA_PERF_TRACING=1` | Per-component GPU/us + host/us summary at exit |

`--perf-tracing` auto-enables `STRATA_GLM_PROF` (GLM) / `DS4_PROF` (DeepSeek).

## Zero-Cost Design

Always compiled in but no-op when env var unset:
```cpp
#define STRACE(component, phase, tag, ...) \\
    do { if (::strata::trace::breadcrumb_enabled()) { /* emit to stderr */ } } while (0)
```

## Output Formats

**Breadcrumb (TSV to stderr):**
```
<epoch_ns>\\t<thread_id>\\t<component>\\t<phase>\\t<tag>\\t<extra>
```

**Perf tracing (key=value to stderr):**
```
GPU per token (DS4_PROF): X.XX ms - between layers X.XX, attention X.XX, ...
PERF_TRACING component=ds4 phase=summary ops=N total_ms=X.XX stages: ...
PERF_TRACING component=glm phase=summary ops=N gpu_ms=X.XX stages: ...
```

## Instrumentation Checklist

1. Add `#include "strata/core/breadcrumb.hpp"` to every kernel dispatch file
2. Add `STRACE(component, phase, tag, "extra")` before each kernel launch
3. Add `STRACE_SCOPE(component, phase, tag, "extra")` at function boundaries (RAII begin/end)
4. Add `--breadcrumb` CLI flag → sets `STRATA_BREADCRUMB=1`
5. Add `--perf-tracing` CLI flag → sets `STRATA_PERF_TRACING=1` + `STRATA_GLM_PROF=1`/`DS4_PROF=1`
6. Emit `PERF_TRACING` summary at `main()` exit before cleanup
7. **Verify zero-cost:** Unset env vars → no stderr output + no measurable overhead

## Optimizer Workflow

1. Run `--perf-tracing` → find slowest stage (e.g., `attention=45%`)
2. Run `--breadcrumb` → trace exact dispatch path for that stage
3. Use `query_kernel_features` to find arch-specific branches
4. Use `compare_archs` to see what features the target lacks
5. Read `kands://models` to find alternative kernels
6. Instrument the new kernel with `STRACE` + `STRACE_SCOPE`
7. Re-run `--perf-tracing` + `diff_perf_traces` to verify improvement
"""


# ── Bundle status (live date/time) ────────────────────────────────────────────

def _bundle_status() -> dict:
    """Return live bundle status with current date/time for search tools."""
    now = datetime.now(timezone.utc)
    resources = _list_resources()
    tools = _tools()
    json_files = [
        "ARCH_MAP.json", "MODEL_MAP.json", "TK_TOOLS.json", "CROSS_PORT.json",
        "AI_FRAMEWORKS.json", "MODEL_FORMATS.json", "PORTING_PATTERNS.json",
        "PERF_BENCHMARKS.json",
    ]
    data_freshness = []
    for fname in json_files:
        fpath = _KANDS_ROOT / fname
        if fpath.exists():
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime, timezone.utc)
            data_freshness.append({
                "file": fname,
                "modified": mtime.isoformat(),
                "age_days": round((now - mtime).total_seconds() / 86400, 1),
            })
    return {
        "bundle": "kands (Kernels & Shaders)",
        "version": "0.2.0",
        "current_time_utc": now.isoformat(),
        "current_date": now.strftime("%Y-%m-%d"),
        "current_date_human": now.strftime("%B %d, %Y"),
        "resource_count": len(resources),
        "tool_count": len(tools),
        "data_freshness": data_freshness,
        "namespace": "kands://",
        "mcp_version": "2.x",
        "search_tools": ["search_arxiv", "search_github", "search_references"],
        "perf_history_entries": len(_load_perf_history().get("entries", [])),
        "note": "Use current_date for ArXiv/GitHub search date filtering. ArXiv API sorts by updated date descending.",
    }


def _setup_guide_markdown() -> str:
    """Interactive guide: prompts the user to confirm their target setup."""
    now = datetime.now(timezone.utc)
    return f"""# K_and_S Setup Guide

**Current date/time:** {now.strftime('%Y-%m-%d %H:%M UTC')}

This guide helps you select your target setup before starting a porting project.
The current date is shown so you can filter search results correctly (ArXiv/GitHub).

## Step 1 — Select Your GPU Backend

| Backend | Vendor | When to use |
|---|---|---|
| CUDA | NVIDIA | Porting from NVIDIA, or targeting NVIDIA GPUs |
| HIP | AMD | Porting to AMD (RX 6000/7000/9000, Instinct MI series) |
| Vulkan | Cross-platform | Targeting Vulkan compute (use DP4a, not CooperativeMatrix on AMD) |
| DirectX 12 / DirectML | Microsoft | Windows-native; use DXC compiler + Agility SDK |
| SYCL / oneAPI | Intel | Intel Arc GPUs (DPAS instruction) |
| OpenVINO | Intel | Intel CPU/GPU inference optimization |

**Choose one as your primary target.** You can always add more later.

## Step 2 — Select Your Target Architecture

Run `read kands://architectures` to browse all supported GPU architectures.
Select your target arch ID (e.g., `gfx1201`, `sm_80`, `gen12_lp`, `arc_alchemester`).

## Step 3 — Select Your OS

| OS | Toolchain | Notes |
|---|---|---|
| Linux | GCC or Clang | Standard for server GPU work. Path: `/usr/bin/g++` |
| Windows | MSVC (cl.exe) | Requires Windows SDK. Path: `C:\\Program Files (x86)\\Windows Kits\\10\\` |
| Windows | MinGW/MSYS2 | Use `C:\\msys64\\mingw64\\bin\\g++.exe` |

## Step 4 — Select Your Host Compiler

Run `query_toolchain(gpu_vendor="NVIDIA", os="linux", host_compiler="clang")`
to get exact compiler paths and flags.

## Step 5 — Record Your Target

Once you have your setup, record your performance target:

```json
{{
  "arch": "gfx1201",
  "workload": "flash_attn",
  "target_tok_per_s": 20.0,
  "current_tok_per_s": 0,
  "config_hash": "gfx1201-WMMA-off-DPAS-off"
}}
```

Use the `record_performance` tool to track progress toward your target.
Use `compare_to_target` to see how far you are from your goal.

---

## Step 6 — Configure Your MCP Client

The K_and_S bundle works with **any** MCP-compatible client. Choose your client:

### Claude Desktop (`claude_desktop_config.json`)
```json
{{
  "mcpServers": {{
    "kands": {{
      "command": "python",
      "args": ["-m", "server"],
      "env": {{ "PYTHONPATH": "H:\\K_and_S\\mcp" }}
    }}
  }}
}}
```
Place at: `%APPDATA%\\Claude\\claude_desktop_config.json` (Windows) or `~/.config/claude-desktop/claude_desktop_config.json` (Linux).

### OpenCode (`opencode.json`)
```json
{{
  "mcpServers": {{
    "kands": {{
      "command": "python",
      "args": ["-m", "server"],
      "env": {{ "PYTHONPATH": "H:\\K_and_S\\mcp" }}
    }}
  }}
}}
```
Place at: `~/.config/opencode/opencode.json` (Linux) or `%USERPROFILE%\\.config\\opencode\\opencode.json` (Windows).

### Cursor (`.cursor/mcp.json`)
```json
{{
  "mcpServers": {{
    "kands": {{
      "command": "python",
      "args": ["-m", "server"],
      "env": {{ "PYTHONPATH": "H:\\K_and_S\\mcp" }}
    }}
  }}
}}
```
Place at: `.cursor/mcp.json` in your workspace root, or at global config location.

### KiloCode
Same JSON format as Claude Desktop. Place in:
- VS Code: `.vscode/mcp.json` in workspace
- JetBrains: `~/.kilocode/cline_mcp_settings.json`

**Tip:** The same config file works for Claude Desktop, OpenCode, and Cursor.
Copy `mcp/claude_desktop_config.json` to your client's config location.

---

**Next:** Run `read kands://porting-strategies` to decide Copy vs Adapt vs Rewrite,
then `read kands://porting-patterns` for transformation patterns.
"""
    """Return live bundle status with current date/time for search tools."""
    now = datetime.now(timezone.utc)
    resources = _list_resources()
    tools = _tools()
    json_files = [
        "ARCH_MAP.json", "MODEL_MAP.json", "TK_TOOLS.json", "CROSS_PORT.json",
        "AI_FRAMEWORKS.json", "MODEL_FORMATS.json", "PORTING_PATTERNS.json",
        "PERF_BENCHMARKS.json",
    ]
    data_freshness = []
    for fname in json_files:
        fpath = _KANDS_ROOT / fname
        if fpath.exists():
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime, timezone.utc)
            data_freshness.append({
                "file": fname,
                "modified": mtime.isoformat(),
                "age_days": round((now - mtime).total_seconds() / 86400, 1),
            })
    return {
        "bundle": "kands (Kernels & Shaders)",
        "version": "0.2.0",
        "current_time_utc": now.isoformat(),
        "current_date": now.strftime("%Y-%m-%d"),
        "resource_count": len(resources),
        "tool_count": len(tools),
        "data_freshness": data_freshness,
        "namespace": "kands://",
        "mcp_version": "2.x",
        "search_tools": ["search_arxiv", "search_github", "search_references"],
        "note": "Use current_date for ArXiv/GitHub search date filtering. ArXiv API sorts by updated date descending.",
    }


def _radical_experiments_markdown() -> str:
    """Return radical experiment ideas as markdown resource for kands://radical-experiments."""
    return """# Radical GPU Porting Experiments (RDNA-native)

> ⚠️ EXPERIMENTAL — Validate with CPU oracle before deploying.
> Results can be saved as Markdown templates via `export_contribution()`.

## AMD/RDNA-native Experiments

### Memory Hierarchy
1. **Bypass LDS entirely** for K ≤ 16: keep A,B tiles in registers
   - RDNA LDS: 32/64 KiB/CU (varies by model)
   - For small tiles: register-only blocking faster than LDS round-trip
2. **DPP-based data shuffle** instead of LDS transpose
   - Use `v_dot2_f32_f16`, `v_perm_b32`, DPP8 patterns
   - Eliminates LDS bank conflict overhead entirely
3. **Explicit `s_waitcnt lgkmcnt(0)`** before shared→register copy (inline asm)

### Compute Path
1. **Native v_dot4** instead of FMA chains
   - Use `v_dot4` directly (not via `__builtin_amdgcn_dot4`)
   - Try `v_dot2` + packing for int8 if alignment allows
2. **Raw AMDGCN asm** for inner loop
   - Write raw AMDGCN assembly, use scheduler hints
3. **Wave32 vs Wave64 comparison**
   - Compile with `-mwavefrontsize=32`; RDNA2 often better at wave32

### Scheduling
1. **8-wave ping-pong vs 4-wave interleave** (HipKittens patterns)
   - 8-wave: alternate compute/memory waves for overlap
   - 4-wave: interleave memory and compute within wavefront

### Layout
1. **AoS vs SoA** — test cache locality trade-offs
2. **LDS bank-conflict-free padding** — pad shared arrays to 2× width

## NVIDIA-specific
1. **PTX-level custom instruction scheduling** with `.scheduler_priority`
2. **FMAD vs separate FMUL+FADD** (`--fmad=false` vs `--fmad=true`)
3. **Non-standard mma tile shapes** (8×8×4)

## Intel-specific
1. **Direct DPAS invocation** via `__builtin_IB_sub_group_mm_*`

---

## Verification Requirements
1. Run reference on CPU (oracle)
2. Call `verify_numerical_equivalence(arch=...)` after every experiment
3. STRACE() tracing + fence timeout (≤1500ms on Windows)
4. Fallback to last-good kernel — NEVER stub
5. `record_performance(arch=..., workload=...)` after success
"""


# ── Feature flag search (engine-agnostic) ─────────────────────────────────────

_ENGINE_FEATURE_FLAGS = {
    # Common engine-agnostic flags
    "ENABLE_CUDA": "GPU backend: CUDA",
    "ENABLE_HIP": "GPU backend: HIP (AMD)",
    "ENABLE_VULKAN": "GPU backend: Vulkan compute",
    "ENABLE_DIRECTX": "GPU backend: Direct3D 12 / DirectML",
    "ENABLE_OPENCL": "GPU backend: OpenCL",
    "ENABLE_SYCL": "GPU backend: SYCL/oneAPI",

    # Quantization / dispatch flags
    "WMMA": "Wave Matrix Multiply-Accumulate support",
    "CUDA_GRAPHS": "CUDA graph support",
    "MANAGED_MEMORY": "Unified memory allocation",
    "PREFILL_MMQ": "Prefill path: multi-multiply-quantized kernels",
    "HAS_TENSOR_CORES": "Tensor core / MMA available",
    "HAS_DPAS": "Intel DPAS matrix instruction available",

    # Tracing / profiling flags
    "BREADCRUMB": "Breadcrumb tracing (STRACE)",
    "PERF_TRACING": "Performance tracing output",
    "PROF": "GPU profiling enabled",
}


def _search_feature_flags(root: Path) -> list[dict]:
    """Search build files for engine-agnostic feature flags."""
    results = []
    for pattern in ["CMakeLists.txt", "*.cmake", "setup.py", "*.sh", "meson.build"]:
        for f in root.rglob(pattern):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    for flag, desc in _ENGINE_FEATURE_FLAGS.items():
                        # Match both STRATA_ENABLE_CUDA and ENABLE_CUDA, etc.
                        if flag in line:
                            results.append({
                                "file": str(f.relative_to(root)),
                                "flag": flag,
                                "description": desc,
                                "line": i,
                                "text": line.strip()[:200],
                            })
            except Exception:
                pass
    return results


# ── Tool handlers ───────────────────────────────────────────────────────────

def _load_trace(text: str) -> str:
    """Load trace from a file path or treat as raw text."""
    p = Path(text)
    if p.is_file():
        return p.read_text(encoding="utf-8", errors="replace")
    return text


def _parse_perf_lines(text: str) -> list[dict]:
    """Parse PERF_TRACING lines into structured records."""
    results = []
    for line in text.splitlines():
        if not line.startswith("PERF_TRACING"):
            continue
        parts = line.split()
        record = {"raw": line, "component": "", "phase": "", "ops": 0, "metrics": {}, "stages": {}}
        for part in parts[1:]:
            if "=" in part and ":" not in part:
                key, val = part.split("=", 1)
                if key == "component":
                    record["component"] = val
                elif key == "phase":
                    record["phase"] = val
                elif key == "ops":
                    record["ops"] = int(val)
                elif key == "total_ms":
                    record["metrics"]["total_ms"] = float(val)
                elif key == "gpu_ms":
                    record["metrics"]["gpu_ms"] = float(val)
                elif key == "host_ms":
                    record["metrics"]["host_ms"] = float(val)
        if "stages:" in line:
            stage_str = line.split("stages:", 1)[1].strip()
            for token in stage_str.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    record["stages"][k] = float(v)
                    record["metrics"][f"stage_{k}"] = float(v)
        results.append(record)
    return results


def _diff_perf_traces(trace_a: str, trace_b: str) -> str:
    """Compare two perf-tracing outputs and report stage-by-stage diffs."""
    lines_a = _parse_perf_lines(trace_a)
    lines_b = _parse_perf_lines(trace_b)

    report = "# Perf-Tracing Diff Report\n\n"
    report += "## Before (trace A)\n```\n" + trace_a.strip()[:500] + "\n```\n\n"
    report += "## After (trace B)\n```\n" + trace_b.strip()[:500] + "\n```\n\n"

    for la in lines_a:
        for lb in lines_b:
            if la["component"] == lb["component"] and la["phase"] == lb["phase"]:
                report += f"## Component: {la['component']}, Phase: {la['phase']}\n\n"
                report += "| Metric | Before | After | Delta |\n|---|---|---|---|\n"
                all_keys = sorted(set(la["metrics"].keys()) | set(lb["metrics"].keys()))
                for k in all_keys:
                    va = la["metrics"].get(k, 0.0)
                    vb = lb["metrics"].get(k, 0.0)
                    delta = vb - va
                    pct = ((delta / va) * 100) if va else 0.0
                    icon = "✅" if delta < 0 else "⚠️" if delta > 0 else "—"
                    report += f"| {k} | {va:.2f} | {vb:.2f} | {delta:+.2f} ({pct:+.1f}%) {icon} |\n"
                report += "\n"
    return report


def _build_tracing_plan(root: Path, components: list[str] | None, model_map: dict) -> str:
    """Build a tracing instrumentation plan by scanning the source tree."""
    existing_strace_files = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".cu", ".cuh", ".cpp", ".hpp"}:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            if "STRACE" in text:
                existing_strace_files.append(str(f.relative_to(root)))
        except Exception:
            pass

    dispatch_files = []
    for f in root.rglob("*.cu"):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if ("<<<" in line and ">>>" in line) or "hipLaunchKernel" in line or "cudaLaunchKernel" in line:
                    dispatch_files.append({
                        "file": str(f.relative_to(root)),
                        "line": i + 1,
                        "call": line.strip()[:150],
                    })
        except Exception:
            pass

    if components is None:
        components = ["app", "glm", "ds4", "prefill", "expert", "cache", "pool", "moe", "attention"]

    plan = "# Tracing Instrumentation Plan\n\n"
    plan += f"**Repository:** `{root}`\n\n"
    plan += f"**Components to instrument:** {', '.join(components)}\n\n"

    plan += "## Already Instrumented\n\n"
    if existing_strace_files:
        for f in sorted(existing_strace_files):
            plan += f"- `{f}` — has STRACE or STRACE_SCOPE calls\n"
    else:
        plan += "- No existing STRACE calls found\n"

    plan += f"\n## Kernel Dispatch Sites Found: {len(dispatch_files)}\n\n"
    for d in dispatch_files[:80]:
        plan += f"- `{d['file']}:{d['line']}` — {d['call']}\n"
    if len(dispatch_files) > 80:
        plan += f"... and {len(dispatch_files) - 80} more dispatch sites\n"

    plan += "\n## Instrumentation Steps\n\n"
    plan += "1. **Add include to every .cu/.cpp that dispatches kernels:**\n"
    plan += "   ```cpp\n   #include \"strata/core/breadcrumb.hpp\"\n   ```\n\n"
    plan += "2. **Add STRACE before each dispatch site** (from the list above):\n"
    plan += "   ```cpp\n   STRACE(\"attention\", \"dispatch\", \"name=%s\", \"qsa_decode_attn\");\n   ```\n\n"
    plan += "3. **Add STRACE_SCOPE at function boundaries** (RAII begin/end):\n"
    plan += "   ```cpp\n   STRACE_SCOPE(\"moe\", \"expert_route\", \"layer=%d\", layer_idx);\n   ```\n\n"
    plan += "4. **Add CLI flags to main():**\n\n"
    plan += "   ```cpp\n"
    plan += "   else if (a == \"--breadcrumb\") {\n"
    plan += "       setenv(\"STRATA_BREADCRUMB\", \"1\", 1);\n"
    plan += "   }\n"
    plan += "   else if (a == \"--perf-tracing\") {\n"
    plan += "       setenv(\"STRATA_PERF_TRACING\", \"1\", 1);\n"
    plan += "       setenv(\"STRATA_GLM_PROF\", \"1\", 1);  // or DS4_PROF=1 for DS4\n"
    plan += "   }\n"
    plan += "   ```\n\n"
    plan += "5. **Add PERF_TRACING summary at process exit:**\n"
    plan += "   ```cpp\n"
    plan += "   if (std::getenv(\"STRATA_PERF_TRACING\")) {\n"
    plan += "       std::fprintf(stderr, \"PERF_TRACING component=X phase=summary ops=N gpu_ms=X.XX\\n\");\n"
    plan += "   }\n"
    plan += "   ```\n\n"
    plan += "6. **Verify zero-cost:** Unset env vars → no stderr output + no measurable overhead.\n"

    return plan


def _build_report(
    args: ReportParams,
    root: Path,
    gpu_kernels: list[dict],
    cpu_kernels: list[dict],
    backends: list[str],
    arch: dict,
    model_map: dict,
) -> str:
    """Build the full Phase 0-7 porting report."""
    model_family = args.model_family
    if model_family == "auto":
        model_family = "deepseek-v4.1-flash" if "ds4" in args.engine.lower() else "glm-5.3-flash"

    model_info = model_map.get("model_families", {}).get(model_family, {})
    arch_map = _load_json(_ARCH_MAP_PATH)

    kernel_names = [k["path"].split("/")[-1].replace(".cu", "").replace(".cpp", "").replace(".comp", "") for k in gpu_kernels]
    compat = {name: _check_kernel_against_arch(name, arch) for name in kernel_names}

    report = f"""# Porting Report: {args.engine}

## Phase 0 — Setup
- **Repo:** `{root}`
- **Build system:** {'CMake' if (root / 'CMakeLists.txt').exists() else 'unknown'}
- **GPU backends found:** {", ".join(backends) if backends else "none"}
- **Target backend:** {"HIP (AMD ROCm)" if arch.get("vendor") == "AMD" else "CUDA (NVIDIA)"}
- **Target architecture:** `{args.target_arch}` — {arch.get('name', '?')}
- **Model family:** `{model_family}`

## Phase 1 — Kernel/Shader Census

### GPU Kernels Found: {len(gpu_kernels)}
"""
    for k in gpu_kernels[:50]:
        report += f"- `{k['path']}` ({k['backend']})\n"
    if len(gpu_kernels) > 50:
        report += f"... and {len(gpu_kernels) - 50} more\n"

    report += f"""
### CPU Kernels Found: {len(cpu_kernels)}
"""
    for k in cpu_kernels[:20]:
        report += f"- `{k['path']}` — {k['kind']}\n"
    if len(cpu_kernels) > 20:
        report += f"... and {len(cpu_kernels) - 20} more\n"

    report += f"""
## Phase 2 — Architecture Capability

| Feature | {args.target_arch} |
|---|---|
| Wave size | {arch.get('wave_size', '?')} |
| LDS (KiB) | {arch.get('lds_kib', '?')} |
| Tensor cores | {arch.get('tensor_cores', '?')} |
| WMMA | {arch.get('wmma', '?')} |
| Dot4 | {arch.get('dot4', '?')} |
| BF16 native | {arch.get('bf16_native', '?')} |

## Phase 3 — Model Family: {model_family}
"""
    if model_info:
        report += f"**Engine binary:** `{model_info.get('engine', 'unknown')}`\n\n"
        report += f"**Quant formats:** {', '.join(model_info.get('quant_formats', []))}\n\n"
        needed = model_info.get("kernels_needed", {})
        if isinstance(needed, dict):
            report += "### Required Kernels by Stage:\n"
            for stage, kernels in needed.items():
                report += f"- **{stage}:** {', '.join(f'`{k}`' for k in kernels)}\n"
        report += "\n"

    report += """## Phase 4 — Instrumentation (Breadcrumb + Perf Tracing)

To add tracing to this engine:
1. Run `generate_tracing_plan(repo="<repo_path>")` to get specific STRACE() placement
2. Add `#include "strata/core/breadcrumb.hpp"` to each kernel dispatch file
3. Add `--breadcrumb` and `--perf-tracing` CLI flags to main()
4. Emit `PERF_TRACING` summary at exit

Read `kands://tracing-steps` for the full methodology.

## Phase 5 — Porting Gap Analysis

### Kernel Compatibility
"""
    for name, status in sorted(compat.items()):
        icon = {"OK": "✅", "NEEDS_FALLBACK": "⚠️", "INCOMPATIBLE": "❌"}[status]
        report += f"- {icon} `{name}`: {status}\n"

    report += f"""
## Phase 6 — Task Breakdown

### Blocking
1. Build with `-DCMAKE_HIP_ARCHITECTURES={args.target_arch}` (or `CMAKE_CUDA_ARCHITECTURES={args.target_arch}`)
2. Ensure `STRATA_PREFILL_MMQ=ON` for HIP builds

### Correctness
1. Run `cmake --build . -j$(nproc)`
2. Run `ctest -R _parity -j$(nproc)`
3. Run `./maya --selftest` (or `maya-ds4` for DS4 models)

### Performance
1. Capture `--perf-tracing` baseline
2. Implement arch-specific kernel variants
3. Capture `--perf-tracing` target output
4. `diff_perf_traces(baseline, target)`

### Documentation
1. Update README with `{args.target_arch}` support
2. Add `{model_family}` to setup family list

## Phase 7 — Verification

| Test | Command | Status |
|---|---|---|
| Build | `cmake --build . --target {args.engine}` | TODO |
| Self-test | `./{args.engine} --selftest` | TODO |
| Parity | `ctest -R parity` | TODO |
| Perf | `./{args.engine} --perf-tracing` | TODO |

---
**Next:** Read `kands://tracing-steps` for instrumentation, or `kands://models` for kernel requirements per model family.
"""
    return report


# ── Data Sanitization ──────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
_ABS_PATH_RE = re.compile(r'[A-Z]:\\(?:[^\s<>"]|&lt;|&gt;)*|/[^\s<>"]*')
_LOCAL_IPS_RE = re.compile(r'192\.168\.\d+\.\d+|10\.0\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+')
_GIT_HASH_RE = re.compile(r'\b[0-9a-f]{40}\b')


def _sanitize_text(text: str) -> str:
    """Strip personal data, absolute paths, emails, and local IPs from text."""
    if not text:
        return ""
    # Remove email addresses
    text = _EMAIL_RE.sub("[redacted-email]", text)
    # Remove local IP addresses
    text = _LOCAL_IPS_RE.sub("[redacted-local-ip]", text)
    # Remove absolute file paths (Windows and Unix)
    text = _ABS_PATH_RE.sub("[path]", text)
    # Remove 40-char git hashes (full commit hashes)
    text = _GIT_HASH_RE.sub("[redacted-hash]", text)
    # Remove any URLs that look like personal repos
    text = re.sub(r'https?://[^\s<>"\'\\]+', "[url-redacted]", text)
    return text.strip()


def _sanitize_entry(entry: dict) -> dict:
    """Sanitize a perf history entry by removing personal data."""
    safe = entry.copy()
    for key in safe:
        if isinstance(safe[key], str):
            safe[key] = _sanitize_text(safe[key])
    return safe

_PERF_BENCHMARKS_PATH = _KANDS_ROOT / "PERF_BENCHMARKS.json"
_PERF_HISTORY_PATH = Path(__file__).resolve().parent / "state" / "perf_history.json"


def _load_perf_benchmarks() -> dict:
    return _load_json(_PERF_BENCHMARKS_PATH) if _PERF_BENCHMARKS_PATH.exists() else {}


def _load_perf_history() -> dict:
    if _PERF_HISTORY_PATH.exists():
        return json.loads(_PERF_HISTORY_PATH.read_text(encoding="utf-8"))
    return {"entries": [], "targets": [], "notes": []}


def _save_perf_history(hist: dict) -> None:
    hist["last_modified"] = datetime.now(timezone.utc).isoformat()
    _PERF_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PERF_HISTORY_PATH.write_text(json.dumps(hist, indent=2), encoding="utf-8")


def _estimate_performance(arch_id: str, workload: str, seq_len: int, arch_map: dict) -> str:
    """Estimate GPU performance using hardware specs + empirical scaling factors."""
    entries = arch_map.get("entries", {})
    arch_info = entries.get(arch_id, {})
    if not arch_info:
        return f"Architecture '{arch_id}' not found in ARCH_MAP.json. Use kands://architectures to browse."

    bench = _load_perf_benchmarks()
    hw = bench.get("hardware_theoretical", {}).get(arch_id, {})
    scaling = bench.get("performance_scaling", {}).get("scaling_factors", {})

    report = f"# Performance Estimate: {arch_id}\n\n"
    report += f"**Architecture:** {arch_info.get('name', '?')} ({arch_id})\n"
    report += f"**Workload:** {workload}\n"
    report += f"**Sequence length:** {seq_len}\n"
    report += f"**Date/time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"

    # Hardware specs
    if hw:
        report += "## Hardware Specifications\n\n"
        report += "| Property | Value |\n|---|---|\n"
        for key in ["compute_units", "base_clock_mhz", "boost_clock_mhz", "fp32_peak_gflops",
                     "fp16_peak_gflops", "bf16_peak_gflops", "tensor_cores", "wmma", "dot4",
                     "memory_bandwidth_gb_s", "lds_kib", "l2_cache_mb"]:
            if key in hw:
                report += f"| {key} | {hw[key]} |\n"
        report += "\n"

    # Estimated performance
    est = hw.get("workload_estimates", {})
    if workload in est:
        report += f"## Estimated Real-World Performance\n\n"
        report += f"**Expected:** {est[workload]}\n\n"
    else:
        # Generic estimate using scaling factors
        report += "## Generic Estimate\n\n"
        peak = hw.get("fp32_peak_gflops", "?")
        factor = scaling.get(workload, scaling.get("flash_attn", 0.65))
        if isinstance(peak, (int, float)) and isinstance(factor, (int, float)):
            report += f"**Theoretical FP32 peak:** {peak} GFLOPS\n"
            report += f"**Scaling factor for {workload}:** {factor}\n"
            report += f"**Estimated real-world:** ~{peak * factor:.1f} GFLOPS\n\n"
        else:
            report += f"Peak: {peak}, Factor: {factor}\n\n"

    # Known bottlenecks
    issues = arch_info.get("known_issues", [])
    if issues:
        report += "## Known Bottlenecks for this Arch\n\n"
        for iss in issues:
            report += f"- {iss}\n"
        report += "\n"

    # Vulkan note if relevant
    vulkan = arch_info.get("vulkan", {})
    if vulkan:
        if not vulkan.get("cooperative_matrix", True):
            report += "**⚠️ Vulkan:** CooperativeMatrix is broken on this vendor. Use DP4a instead.\n\n"

    # Recommendation
    report += "## Recommendation\n\n"
    report += f"Read `kands://porting-patterns` for architecture-specific transformation patterns.\n"
    report += f"Read `kands://diagnostics` for the iterative optimization loop.\n"
    report += f"Read `kands://isa-references` to verify instruction availability.\n"

    return report


def _compare_to_target(arch_id: str, workload: str, target: float, current: float, arch_map: dict) -> str:
    """Compare user's target to estimated achievable performance."""
    entries = arch_map.get("entries", {})
    arch_info = entries.get(arch_id, {})
    if not arch_info:
        return f"Architecture '{arch_id}' not found."

    bench = _load_perf_benchmarks()
    hw = bench.get("hardware_theoretical", {}).get(arch_id, {})
    est = hw.get("workload_estimates", {})

    report = f"# Performance Target Comparison: {arch_id}\n\n"
    report += f"**Workload:** {workload}\n"
    report += f"**Date/time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"

    est_str = est.get(workload, "unknown")
    report += f"**Estimated achievable:** ~{est_str}\n"

    if target > 0:
        gap = target - current
        pct_to_target = (current / target * 100) if target else 0
        report += f"**Your target:** {target} tok/s\n"
        if current > 0:
            report += f"**Your current:** {current} tok/s ({pct_to_target:.0f}% of target)\n"
            if gap > 0:
                report += f"**Gap to close:** +{gap:.1f} tok/s\n"
            else:
                report += f"**Status:** ✅ Target exceeded by {abs(gap):.1f} tok/s\n"

        # Optimization suggestions
        report += "\n## Optimization Suggestions (sherlock-it)\n\n"
        report += "Run the iterative optimization loop (see `kands://diagnostics`):\n\n"

        # Look at perf history for this arch+workload
        hist = _load_perf_history()
        related = [e for e in hist.get("entries", [])
                   if e.get("arch") == arch_id and e.get("workload") == workload]
        if related:
            best = max(related, key=lambda e: e.get("tok_per_s", 0))
            report += f"**Best recorded for this combo:** {best['tok_per_s']} tok/s\n"
            report += f"**Config:** {best.get('config_hash', '?')}\n"
            if best.get("notes"):
                report += f"**Notes:** {best['notes']}\n"
            report += "\n"

        # Suggest based on workload type
        if "attn" in workload.lower() or "flash" in workload.lower():
            if not arch_info.get("wmma", False):
                report += "- ⚠️ This arch has no WMMA — attention uses F32 fallback. Consider:\n"
                report += "  1. Switching to tile-based attention with LDS prefetch (reduces global loads)\n"
                report += "  2. Using `v_mad_mix_f32` instead of separate multiply+add on AMD\n"
                report += "  3. Shared memory tiling for KV reuse (LDS is "
                report += f"{arch_info.get('lds_kib', '?')}KiB on this arch)\n"
            else:
                report += "- WMMA supported — ensure rocWMMA/WMMA is active (check ISA with `kands://isa-references`)\n"
        elif "dequant" in workload.lower():
            if "q4" in workload:
                report += "- Q4_0 dequant is memory-bandwidth bound. Optimizations:\n"
                report += "  1. Use packed byte loads (v_and_b32x4 on AMD, imad on NVIDIA)\n"
                report += "  2. Coalesce memory access — align to 128-bit boundaries\n"
                report += "  3. Use LDS prefetching for codebook entries\n"
        elif "gemm" in workload.lower() or "gemv" in workload.lower():
            if arch_info.get("bf16_native", False):
                report += "- BF16 native support available — use bf16 GEMM path\n"
            else:
                report += "- No native BF16 — use FP32 path or consider quantization\n"
        else:
            report += "- Run `sherlock-it trace` on this workload to identify the bottleneck\n"
            report += "- Use `kands://diagnostics` for the full perf investigation workflow\n"

    return report


def _record_performance(arguments: dict) -> str:
    """Record a performance measurement into local history."""
    hist = _load_perf_history()
    now = datetime.now(timezone.utc)

    entry = {
        "timestamp": now.isoformat(),
        "arch": arguments.get("arch", ""),
        "workload": arguments.get("workload", ""),
        "tok_per_s": arguments.get("tok_per_s", 0),
        "config_hash": _sanitize_text(arguments.get("config_hash", "")),
        "git_hash": "",  # Never store git hashes — could be personal
        "notes": _sanitize_text(arguments.get("notes", "")),
    }
    hist.setdefault("entries", []).append(entry)

    target = arguments.get("target_tok_per_s", 0)
    if target > 0:
        hist.setdefault("targets", []).append({
            "arch": entry["arch"],
            "workload": entry["workload"],
            "target_tok_per_s": target,
            "current_tok_per_s": entry["tok_per_s"],
            "recorded": now.isoformat(),
        })

    _save_perf_history(hist)
    return f"✅ Recorded: {entry['arch']} / {entry['workload']} → {entry['tok_per_s']} tok/s at {entry['timestamp']}"


def _suggest_optimizations(arch_id: str, workload: str, target: float, arch_map: dict) -> str:
    """Read perf history and suggest next steps."""
    hist = _load_perf_history()
    entries = hist.get("entries", [])

    report = f"# Optimization Suggestions for {arch_id} / {workload}\n"
    report += f"**Date/time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"

    # Filter entries for this arch + workload
    related = [e for e in entries if e.get("arch") == arch_id and e.get("workload") == workload]
    arch_entries = [e for e in entries if e.get("arch") == arch_id]

    if not entries:
        report += "No perf history recorded yet. Run `record_performance` to start tracking.\n\n"
    elif not related:
        report += f"No entries for this exact arch/workload combo. But we have data for:\n\n"
        for e in arch_entries[-10:]:
            report += f"- {e['workload']}: {e['tok_per_s']} tok/s ({e.get('timestamp','?')[:10]})\n"
    else:
        report += f"## Your History ({len(related)} entries)\n\n"
        report += "| Date | tok/s | Config | Notes |\n|---|---|---|---|\n"
        for e in sorted(related, key=lambda x: x.get("timestamp", "")):
            report += f"| {e['timestamp'][:10]} | {e['tok_per_s']} | {e.get('config_hash','')} | {e.get('notes','')} |\n"
        report += "\n"

        best = max(related, key=lambda e: e.get("tok_per_s", 0))
        report += f"**Best so far:** {best['tok_per_s']} tok/s\n"
        if best.get("notes"):
            report += f"**What worked:** {best['notes']}\n\n"

    # Look at what worked for other arches with same workload
    others = [e for e in entries if e.get("workload") == workload and e.get("arch") != arch_id]
    if others:
        report += "## What worked on other architectures\n\n"
        for e in sorted(others, key=lambda x: x.get("tok_per_s", 0), reverse=True)[:5]:
            report += f"- `{e['arch']}`: {e['tok_per_s']} tok/s — {e.get('notes','')}\n"
        report += "\n"

    # Target gap analysis
    if target > 0:
        current_best = max((e.get("tok_per_s", 0) for e in related), default=0)
        if current_best > 0:
            pct = current_best / target * 100
            report += f"## Target Progress\n\n"
            report += f"**Target:** {target} tok/s | **Best:** {current_best} tok/s | **Progress:** {pct:.0f}%\n"
            if pct < 100:
                report += f"**Need:** +{target - current_best:.1f} tok/s more\n\n"

    report += "## Next Steps (sherlock-it loop)\n\n"
    report += "1. Run `query_kernel_features` to find arch-specific branches in your source\n"
    report += "2. Run `compare_archs` to see what your arch lacks vs. others\n"
    report += "3. Read `kands://porting-patterns` for workload-type transformation patterns\n"
    report += "4. Read `kands://diagnostics` for the full iterative optimization workflow\n"
    report += "5. After trying something, run `record_performance` to track progress\n"
    report += "6. Repeat until target is met\n"

    return report


def _export_contribution(engine_name: str, arch: str, workload: str, output_path: str) -> str:
    """Export findings as a reusable contribution template for the git repo."""
    hist = _load_perf_history()
    entries = hist.get("entries", [])

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    safe_name = engine_name.replace(r"/", "_").replace(r" ", "_").lower()

    if not output_path:
        output_path = str(_KANDS_ROOT / "findings" / f"{safe_name}.md")

    report = f"""# Porting Contribution: {engine_name}

**Date:** {now.strftime('%Y-%m-%d %H:%M UTC')}
**Target arch:** `{arch}`
**Workload:** `{workload}`
**Contributor:** _(your github handle — no personal info)_

---

## Summary

This is a reusable porting template contributed back to the K_and_S bundle.
Anyone can use this as a starting point for porting a similar engine to `{arch}`.

## Performance History

| Date | tok/s | Config | Notes |
|---|---|---|---|
"""
    related = [e for e in entries if e.get("arch") == arch and e.get("workload") == workload]
    for e in sorted(related, key=lambda x: x.get("timestamp", "")):
        safe_e = _sanitize_entry(e)
        report += f"| {safe_e['timestamp'][:10]} | {safe_e['tok_per_s']} | {safe_e.get('config_hash','')} | {safe_e.get('notes','')} |\n"

    if not related:
        report += "| (no history recorded) |\n"

    report += f"""

## Key Learnings

1. **_(Add your optimization discoveries here — no personal data or URLs)_**

2. **_(What worked on this arch vs. others)_**

3. **_(What didn't work — save others time)_**

## Architecture Notes for `{arch}`

<!-- Read kands://architectures to fill this in -->

- Wave size: _(from ARCH_MAP.json)_
- Known issues: _(check kands://architectures)_
- WMMA support: _(boolean)_
- BF16 native: _(boolean)_

## Porting Patterns Used

<!-- Read kands://porting-patterns for the transformation maps -->

- CUDA → HIP: _(copy / adapt / rewrite)_
- Memory model: _(explicit / managed)_
- Kernel dispatch changes: _(yes/no)_

## Verification

- [ ] CPU oracle parity passed (`ctest -R parity`)
- [ ] Numerical stability verified (tolerance < 1e-4)
- [ ] Performance baseline captured (`--perf-tracing`)
- [ ] Edge cases covered

## Contributing Back

To add your learnings to the K_and_S git repository:

1. Save this file to `findings/{safe_name}.md`
2. Update `ARCH_MAP.json` with any new arch notes you discovered
3. Update `PORTING_PATTERNS.json` if you found a new transformation pattern
4. Update `PERF_BENCHMARKS.json` with your real-world benchmark numbers
5. Submit a PR to the K_and_S repository

> **Sanitization:** All perf history entries are automatically sanitized to remove
> emails, absolute file paths, local IP addresses, and git hashes. Do not include
> personal information, internal URLs, or proprietary config details in your notes.
>
> Template variables: `{{ENGINE_NAME}}`, `{{TARGET_ARCH}}`, `{{WORKLOAD}}`, `{{TIMESTAMP}}`
> Replace these with your actual values before submitting.
"""

    # Write to file
    from pathlib import Path as _Path
    out = _Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    return f"✅ Contribution template written to `{output_path}`\n\n" \
           f"Review the file, fill in your findings (no personal data/URLs), then:\n" \
           f"1. Add it to `findings/` directory\n" \
           f"2. Update ARCH_MAP.json and PORTING_PATTERNS.json with your arch notes\n" \
           f"3. Submit a PR to the K_and_S repository"


# ── Toolchain lookup ──────────────────────────────────────────────────────────

def _query_toolchain(gpu_vendor: str, os_target: str, host_compiler: str) -> str:
    """Look up toolchain info for a GPU vendor + OS + host compiler combo."""
    tk = _load_json(_TK_TOOLS_PATH)
    arch_map = _load_json(_ARCH_MAP_PATH)

    vendor_lower = gpu_vendor.lower()
    report = f"# Toolchain Lookup: {gpu_vendor} / {os_target} / {host_compiler}\n"
    report += f"**Date/time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"

    # Find the backend entry
    backend_key = None
    backend_map = {"amd": "rocm", "nvidia": "cuda", "intel": "oneapi"}
    if vendor_lower in backend_map:
        backend_key = backend_map[vendor_lower]

    if not backend_key or not tk.get("gpu_toolkits", {}).get(backend_key):
        report += f"**Vendor '{gpu_vendor}' not found. Available: {', '.join(tk.get('gpu_toolkits', {}).keys())}\n"
        return report

    backend = tk["gpu_toolkits"][backend_key]
    report += f"## GPU Backend: {backend.get('name', '?')}\n\n"

    # OS-specific info
    os_key = os_target if os_target != "auto" else ("linux" if import_platform_system() != "Windows" else "windows")
    os_info = backend.get(os_key, {})
    if os_info:
        report += f"### {os_key.title()}\n\n"
        for k, v in os_info.items():
            if k == "sdk_paths":
                report += f"**SDK paths:**\n"
                for path_k, path_v in v.items():
                    report += f"  - {path_k}: `{path_v}`\n"
            elif k == "notes":
                report += f"**Notes:** {v}\n"
            elif k == "msvc_compat":
                report += f"**MSVC compat:** {v}\n"
            elif k == "supported":
                report += f"**Supported:** {v}\n"
            elif k == "compiler":
                report += f"**Compiler:** `{v}`\n"
            else:
                report += f"**{k}:** {v}\n"
        report += "\n"

    # Compiler paths
    compiler = backend.get("compiler", {})
    if compiler:
        report += "### Compiler paths\n\n"
        for name, path in compiler.items():
            report += f"- **{name}:** `{path}`\n"
        report += "\n"

    # nvcc flags (for NVIDIA)
    if backend_key == "cuda" and "nvcc_flags" in backend:
        report += "### nvcc flags\n\n"
        for flag, desc in backend["nvcc_flags"].items():
            report += f"- `{flag}`: {desc}\n"
        report += "\n"

    # Host toolchain
    if host_compiler != "auto":
        host_tc = tk.get("host_toolchains", {}).get(host_compiler, {})
        if host_tc:
            report += f"## Host Toolchain: {host_tc.get('name', '?')}\n\n"
            if host_tc.get("common_paths"):
                report += "**Common paths:**\n"
                for p in host_tc["common_paths"]:
                    report += f"  - `{p}`\n"
            if host_tc.get("flags"):
                report += f"**Flags:** {', '.join(host_tc['flags'])}\n"
            if host_tc.get("cpp_standard"):
                report += f"**C++ standard:** `{host_tc['cpp_standard']}`\n"
            if host_tc.get("cmake_generator"):
                report += f"**CMake generator:** {host_tc['cmake_generator']}\n"
            if host_tc.get("notes"):
                report += f"**Notes:** {host_tc['notes']}\n"

    # Recommended archs for this vendor
    vendor_archs = [aid for aid, info in arch_map.get("entries", {}).items() if info.get("vendor", "").lower() == vendor_lower.upper() or info.get("vendor") == gpu_vendor]
    if vendor_archs:
        report += f"\n## Recommended archs for {gpu_vendor}\n\n"
        for aid in vendor_archs:
            info = arch_map["entries"][aid]
            report += f"- `{aid}` — {info.get('name', '?')}\n"

    report += f"\n## Next steps\n\n"
    report += f"1. Read `kands://architectures` for full arch capability matrix\n"
    report += f"2. Read `kands://porting-patterns` for {backend_key} → target transformation patterns\n"
    report += f"3. Run `detect_toolchain()` to verify what's actually installed on this machine\n"
    report += f"4. Run `autonomous_port(repo_path='...', source_arch='sm_80', target_arch='gfx1201')` for full auto analysis\n"
    report += f"5. Run `search_online(query='ROC 10.1 install {os_key}', max_results=5)` for latest install guides\n"

    # Runtime detection summary
    detected = _detect_installed_toolchain(gpu_vendor, host_compiler)
    det = detected.get("detected", {})
    if det:
        report += f"\n## Runtime Detection Summary\n\n"
        for tool_key, tool_info in det.items():
            if isinstance(tool_info, dict):
                for sub_key, sub_val in tool_info.items():
                    if sub_key in ("path", "hipcc", "nvcc", "glslc", "icpx") and sub_val:
                        report += f"- `{tool_key}.{sub_key}`: `{sub_val}` ✅\n"
                    elif sub_key == "env_var" and isinstance(sub_val, dict):
                        if sub_val.get("exists"):
                            report += f"- `{tool_key}` env `{sub_val.get('env_var')}` → `{sub_val.get('value')}` ✅\n"

    return report


def import_platform_system() -> str:
    """Wrapper to avoid importing at module level."""
    import platform
    return platform.system()


def _detect_installed_toolchain(gpu_vendor: str, host_compiler: str) -> dict[str, Any]:
    """Runtime-detect installed GPU toolchains + host compilers by probing the filesystem."""
    import os as _os
    import shutil as _shutil

    is_windows = _os.name == "nt" or import_platform_system() == "Windows"
    result: dict[str, Any] = {"platform": "windows" if is_windows else "linux", "detected": {}}

    vendor_lower = gpu_vendor.lower()
    hc_lower = host_compiler.lower()

    def _check_which(names: list[str]) -> Optional[str]:
        """Check if a binary is in PATH via shutil.which."""
        for n in names:
            p = _shutil.which(n)
            if p:
                return p
        return None

    def _check_path(paths: list[str]) -> dict[str, str]:
        """Check a list of filesystem paths, return first existing."""
        for p in paths:
            if _os.path.exists(p):
                return {"path": p, "exists": True}
        return {"exists": False}

    def _check_env(var: str) -> dict[str, str]:
        """Check environment variable for a root path."""
        val = _os.environ.get(var)
        if val and _os.path.exists(val):
            return {"env_var": var, "value": val, "exists": True}
        return {"env_var": var, "exists": False}

    # ── GPU toolchains ──

    if vendor_lower in ("auto", "amd"):
        rocm_env = _check_env("ROCM_PATH")
        result["detected"]["rocm"] = {
            "env_var": rocm_env,
            "hipcc": _check_which(["hipcc", "hipcc.exe"]),
            "version": None,
        }
        if is_windows:
            # Check Windows-specific common paths at runtime (NOT hardcoded defaults)
            win_paths = [
                _os.path.join(_os.environ.get("ProgramFiles", ""), "AMD ROCm", "bin", "hipcc.exe"),
                _os.path.join(_os.environ.get("ProgramFiles", ""), "AMD ROCm", "bin", "clang++.exe"),
            ]
            found = _check_path(win_paths)
            result["detected"]["rocm"]["windows_path"] = found
            if found.get("exists"):
                result["detected"]["rocm"]["hipcc"] = result["detected"]["rocm"]["hipcc"] or found["path"]
        else:
            linux_paths = ["/opt/rocm/bin/hipcc", "/opt/rocm/core-10.1/bin/hipcc"]
            found = _check_path(linux_paths)
            result["detected"]["rocm"]["linux_path"] = found

    if vendor_lower in ("auto", "nvidia"):
        cuda_env = _check_env("CUDA_PATH")
        result["detected"]["cuda"] = {
            "env_var": cuda_env,
            "nvcc": _check_which(["nvcc", "nvcc.exe"]),
            "version": None,
        }
        if is_windows:
            # Probe for nvcc in typical Windows locations
            prog = _os.environ.get("ProgramFiles", "")
            nv_paths = [
                _os.path.join(prog, "NVIDIA GPU Computing Toolkit", "CUDA", "v12.4", "bin", "nvcc.exe"),
                _os.path.join(prog, "NVIDIA GPU Computing Toolkit", "CUDA", "v12.3", "bin", "nvcc.exe"),
                _os.path.join(prog, "NVIDIA GPU Computing Toolkit", "CUDA", "v12.5", "bin", "nvcc.exe"),
            ]
            found = _check_path(nv_paths)
            result["detected"]["cuda"]["windows_path"] = found

    if vendor_lower in ("auto", "intel"):
        oneapi_env = _check_env("ONEAPI_ROOT")
        result["detected"]["oneapi"] = {
            "env_var": oneapi_env,
            "icpx": _check_which(["icpx", "dpcpp"]),
            "version": None,
        }

    if vendor_lower in ("auto", "vulkan"):
        vol_env = _check_env("VULKAN_SDK")
        result["detected"]["vulkan"] = {
            "env_var": vol_env,
            "glslc": _check_which(["glslc", "glslangValidator"]),
        }

    # ── Host toolchains ──

    if hc_lower != "auto":
        host_map = {
            "msvc": {"which": ["cl.exe"], "label": "MSVC"},
            "gcc": {"which": ["gcc", "g++"], "label": "GCC"},
            "clang": {"which": ["clang", "clang++"], "label": "Clang"},
            "mingw": {"which": ["g++.exe"], "label": "MinGW"},
            "msys2": {"which": ["pacman"], "label": "MSYS2"},
            "strawberry": {"which": ["perl.exe"], "label": "Strawberry Perl"},
        }
        if hc_lower in host_map:
            info = host_map[hc_lower]
            found_path = _check_which(info["which"])
            result["detected"]["host"] = {
                "name": info["label"],
                "path": found_path,
                "found": bool(found_path),
            }
    else:
        # Auto-detect host compiler
        result["detected"]["host"] = {
            "cl.exe": _check_which(["cl.exe"]),
            "gcc": _check_which(["gcc"]),
            "g++": _check_which(["g++"]),
            "clang++": _check_which(["clang++"]),
            "perl.exe": _check_which(["perl.exe"]),
        }

    # ── Strawberry Perl check (needed for build scripts) ──
    strawberry_path = _os.path.join(_os.environ.get("ProgramFiles", ""), "Strawberry", "perl", "bin", "perl.exe")
    if is_windows and _os.path.exists(strawberry_path):
        result["detected"]["strawberry_perl"] = {"path": strawberry_path, "found": True}
    elif _shutil.which("perl"):
        result["detected"]["strawberry_perl"] = {"path": _shutil.which("perl"), "found": True}

    # ── Vulkan SDK check ──
    vulkan_sdk = _os.environ.get("VULKAN_SDK")
    if vulkan_sdk and _os.path.exists(vulkan_sdk):
        result["detected"]["vulkan_sdk_root"] = vulkan_sdk

    return result


def _format_detection_result(detected: dict[str, Any]) -> str:
    """Format detection result into a readable report."""
    report = "# Toolchain Detection (Runtime)\n\n"
    report += f"**Platform:** {detected['platform']}\n"
    report += f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
    report += "**Detection method:** `shutil.which()` + environment variable probing + filesystem checks\n\n"

    sections = {
        "rocm": "AMD ROCm / ROC 10.1",
        "cuda": "NVIDIA CUDA",
        "oneapi": "Intel oneAPI",
        "vulkan": "Vulkan SDK",
        "host": "Host compiler",
        "strawberry_perl": "Strawberry Perl",
        "vulkan_sdk_root": "Vulkan SDK Root",
    }

    found_any = False
    for key, label in sections.items():
        if key not in detected["detected"]:
            continue
        data = detected["detected"][key]
        if not isinstance(data, dict):
            continue

        has_binary = any(v for k, v in data.items() if k in ("path", "hipcc", "nvcc", "glslc") and isinstance(v, str))
        if not has_binary:
            continue
        found_any = True

        report += f"\n## {label}\n\n"
        for k2, v2 in data.items():
            if isinstance(v2, dict):
                if v2.get("exists"):
                    report += f"- **{k2}:** `{v2.get('value', v2.get('path'))}` ✅\n"
                else:
                    report += f"- **{k2}:** not found ❌\n"
            elif k2 in ("path", "hipcc", "nvcc", "glslc", "icpx") and v2:
                report += f"- **{k2}:** `{v2}` ✅\n"
            elif v2 is not None and not isinstance(v2, bool):
                report += f"- **{k2}:** {v2}\n"

    if not found_any:
        report += "\n**No GPU toolchains detected.**\n"

    # Strawberry Perl note
    if "strawberry_perl" in detected["detected"] and detected["detected"]["strawberry_perl"]:
        report += "\n## ⚠️ Strawberry Perl Detected\n\n"
        report += "Perl found — this is required for some GPU SDK build scripts (MSYS2 package scripts, some ROCm Makefiles).\n"
        report += "Ensure it is in your PATH alongside MSVC/MinGW when building from source.\n"

    return report


def _search_online(query: str, max_results: int) -> str:
    """Search online for toolchain install info, version notes, and porting references."""
    report = f"# Online Search: \"{query}\"\n\n"
    report += f"_Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n\n"
    report += "**Search sources:** ArXiv papers + GitHub code search\n\n"

    try:
        combined = search_references(query, max_results)
        found_anything = False

        report += "## ArXiv Papers\n\n"
        for p in combined.get("arxiv", []):
            found_anything = True
            report += f"- **{p['title']}** ({p.get('updated','?')[:10]}) [{p.get('relevance','?')}]\n"
            report += f"  {p['abstract'][:200]}...\n  URL: {p['url']}\n\n"
        if not combined.get("arxiv"):
            report += "_No ArXiv results found._\n\n"

        report += "## GitHub References\n\n"
        for r in combined.get("github", []):
            found_anything = True
            if r.get("type") == "fallback_search":
                report += f"**Fallback:** {r.get('note','')}\n  {r.get('url','')}\n\n"
            else:
                report += f"- **{r.get('name','?')}** in `{r.get('repo','?')}` [{r.get('relevance','?')}]\n"
                report += f"  {r.get('description','')[:150]}...\n"
                report += f"  URL: {r.get('url','')}\n\n"
        if not combined.get("github"):
            report += "_No GitHub results found._\n\n"

        if not found_anything:
            report += "\n**Try rephrasing** — use vendor-specific terms like 'hipcc install Windows', 'ROC 10.1 TheRock'.\n"

    except ImportError:
        report += "## ⚠️ Online search unavailable\n\n"
        report += "The `reference_search` module is not installed.\n"
    except Exception as e:
        report += f"## Search Error\n\n{e}\n"

    report += "\n---\n\n"
    report += "## Save as Template\n\n"
    report += "Copy this output to `findings/YYYY-MM-DD_<query>.md` for future reference.\n"
    report += "Use `export_contribution(engine_name=...)` to turn findings into a reusable template.\n"

    return report


# ── Autonomous Port orchestration ─────────────────────────────────────────────

async def _autonomous_port(params: AutonomousPortParams, arch_map: dict) -> str:
    """Run the full porting workflow in one call: discover → check → compare → toolchain → estimate → research → plan."""
    now = datetime.now(timezone.utc)
    root = Path(params.repo_path)
    entries = arch_map.get("entries", {})
    source_info = entries.get(params.source_arch, {})
    target_info = entries.get(params.target_arch, {})

    if not root.exists():
        return f"Error: {params.repo_path} does not exist"

    if not target_info:
        return f"Error: target arch '{params.target_arch}' not found in ARCH_MAP.json"

    report = f"""# Autonomous Porting Report

**Date/time:** {now.strftime('%Y-%m-%d %H:%M UTC')}
**Repo:** `{params.repo_path}`
**Source arch:** `{params.source_arch}` — {source_info.get('name', '?')}
**Target arch:** `{params.target_arch}` — {target_info.get('name', '?')}
**Target backend:** `{params.target_backend}`
**Target OS:** `{params.target_os}`
**Mode:** `{params.mode}`
**Performance target:** {params.target_tok_per_s if params.target_tok_per_s else 'N/A'} tok/s

---

## Phase 1 — Environment Scan

### GPU Toolkits Detected (from kands://tk-tools)"""

    # Scan for installed toolkits in env
    import os as _os
    env_vars = {
        "ROCM_PATH": _os.environ.get("ROCM_PATH", "not set"),
        "CUDA_PATH": _os.environ.get("CUDA_PATH", "not set"),
        "VULKAN_SDK": _os.environ.get("VULKAN_SDK", "not set"),
        "ONEAPI_ROOT": _os.environ.get("ONEAPI_ROOT", "not set"),
        "OPENVINO_ROOT": _os.environ.get("OPENVINO_ROOT", "not set"),
    }
    report += "\n\n| Env var | Value |\n|---|---|\n"
    for k, v in env_vars.items():
        status = "✅" if v != "not set" else "❌"
        report += f"| {k} | {status} {v} |\n"

    # Use runtime detection for compilers and host tools
    detected = _detect_installed_toolchain(params.target_backend, "auto")
    det = detected.get("detected", {})
    report += "\n### Compilers Detected (runtime)\n\n"
    for tool_key, tool_info in det.items():
        if tool_key in ("host",):
            for sub_key, sub_val in tool_info.items():
                if sub_val:
                    report += f"- `{sub_key}`: ✅ `{sub_val}`\n"
                else:
                    report += f"- `{sub_key}`: ❌ not in PATH\n"
        elif tool_key in ("strawberry_perl", "vulkan_sdk_root"):
            if isinstance(tool_info, str):
                report += f"- `{tool_key}`: ✅ `{tool_info}`\n"
        elif isinstance(tool_info, dict):
            for sub_key in ("path", "hipcc", "nvcc", "glslc", "icpx"):
                val = tool_info.get(sub_key)
                if val:
                    report += f"- `{tool_key}.{sub_key}`: ✅ `{val}`\n"
            # Env var results
            env_data = tool_info.get("env_var", {})
            if isinstance(env_data, dict) and env_data.get("exists"):
                report += f"- `{tool_key}`: env `{env_data.get('env_var')}` = `{env_data.get('value')}` ✅\n"
            else:
                report += f"- `{tool_key}`: not installed ❌\n"

    # Scan for model folders
    report += "\n### Model Folders Detected\n\n"
    model_dirs = []
    common_model_dirs = [
        "~/.cache/llama.cpp", "~/.cache/huggingface", "~/.cache/kandoo",
        _os.path.expanduser("~/Models"), _os.path.expanduser("~/model"),
    ]
    for d in common_model_dirs:
        expanded = _os.path.expanduser(d)
        if Path(expanded).exists():
            model_dirs.append(expanded)
            # Look for model files
            gguf_files = list(Path(expanded).rglob("*.gguf"))[:10]
            if gguf_files:
                report += f"- `{expanded}` — {len(gguf_files[:10])} GGUF files (showing first 10):\n"
                for f in gguf_files:
                    report += f"  - {f.name}\n"
            else:
                report += f"- `{expanded}` — exists (no GGUF files found)\n"

    if not model_dirs:
        report += "- No standard model directories found\n"

    # Phase 2: Discover kernels
    all_kernels = _scan_sources(root)
    gpu_kernels = [k for k in all_kernels if k["backend"].startswith("gpu")]
    cpu_kernels = [k for k in all_kernels if k["backend"].startswith("cpu")]
    report += f"""

---

## Phase 2 — Kernel Discovery

**GPU kernels found:** {len(gpu_kernels)}
**CPU kernels found:** {len(cpu_kernels)}

### GPU Kernel Files
"""
    for k in gpu_kernels[:30]:
        report += f"- `{k['path']}` ({k['backend']}) — {k['kind']}\n"
    if len(gpu_kernels) > 30:
        report += f"... and {len(gpu_kernels) - 30} more\n"

    # Phase 3: Arch comparison
    report += f"""
---

## Phase 3 — Architecture Comparison

Comparing `{params.source_arch}` → `{params.target_arch}`

| Feature | Source ({params.source_arch}) | Target ({params.target_arch}) | Gap |
|---|---|---|---|
"""
    for k in ["wave_size", "lds_kib", "tensor_cores", "wmma", "dot4", "dot8", "bf16_native", "fp8_native"]:
        va = source_info.get(k, "?"); vb = target_info.get(k, "?")
        gap = "SAME" if str(va) == str(vb) else "DIFF"
        report += f"| {k} | {va} | {vb} | {gap} |\n"

    # Key porting implications
    report += "\n### Porting Implications\n\n"
    if source_info.get("wmma") and not target_info.get("wmma"):
        report += "- ⚠️ Source has WMMA, target does NOT → WMMA kernels need F32 fallback\n"
    if source_info.get("tensor_cores") and not target_info.get("tensor_cores"):
        report += "- ⚠️ Source has tensor cores, target does NOT → MMA kernels need dot4/F32 fallback\n"
    if source_info.get("bf16_native") and not target_info.get("bf16_native"):
        report += "- ⚠️ Source has native BF16, target does NOT → BF16 kernels need FP32 emulation\n"
    if source_info.get("dot4") and not target_info.get("dot4"):
        report += "- ⚠️ Source has dot4, target does NOT → quantized kernels need scalar fallback\n"
    if source_info.get("dot8") and not target_info.get("dot8"):
        report += "- ⚠️ Source has dot8, target does NOT → int8 operations need fallback\n"
    if source_info.get("wave_size") != target_info.get("wave_size"):
        report += f"- ⚠️ Wave size mismatch: {source_info.get('wave_size')} → {target_info.get('wave_size')} — check shuffle/wave intrinsics\n"
    if not source_info.get("wmma") and target_info.get("wmma"):
        report += "- ✅ Target has WMMA source doesn't — can add acceleration\n"
    if source_info.get("known_issues"):
        report += f"\n### Source arch known issues\n"
        for iss in source_info["known_issues"]:
            report += f"- {iss}\n"
    if target_info.get("known_issues"):
        report += f"\n### Target arch known issues\n"
        for iss in target_info["known_issues"]:
            report += f"- {iss}\n"

    # Vulkan note
    if params.target_backend == "vulkan":
        vulkan = target_info.get("vulkan", {})
        if vulkan:
            if not vulkan.get("cooperative_matrix", True):
                report += f"\n### ⚠️ Vulkan CooperativeMatrix Status\n"
                report += f"**The target arch has broken CooperativeMatrix support on Vulkan.**\n"
                report += f"**Use DP4a (GL_KHR_shader_integer_dot_product) for matmul instead.**\n"
                report += f"See `kands://isa-references` for details.\n"

    # Phase 4: Toolchain lookup
    vendor_map = {"amd": "ROCm", "nvidia": "CUDA", "intel": "oneAPI"}
    vendor_name = vendor_map.get(params.target_arch.split("_")[0] if "_" in params.target_arch else params.target_arch[0], "?")
    # Determine vendor from arch_info
    vendor_name = target_info.get("vendor", "?")
    report += f"""

---

## Phase 4 — Toolchain Lookup

**Target vendor:** {vendor_name}
**Target OS:** {params.target_os}

### Compilers and SDKs"""

    os_key = params.target_os if params.target_os != "auto" else ("linux" if import_platform_system() != "Windows" else "windows")
    # Find the toolkit entry
    toolkit_key = None
    if vendor_name == "AMD":
        toolkit_key = "rocm"
    elif vendor_name == "NVIDIA":
        toolkit_key = "cuda"
    elif vendor_name == "Intel":
        toolkit_key = "oneapi"
    elif vendor_name == "Apple":
        toolkit_key = None

    tk = _load_json(_TK_TOOLS_PATH)
    if toolkit_key and tk.get("gpu_toolkits", {}).get(toolkit_key):
        backend_tk = tk["gpu_toolkits"][toolkit_key]
        compilers = backend_tk.get("compiler", {})
        report += "\n\n| Tool | Path |\n|---|---|\n"
        for name, path in compilers.items():
            report += f"| {name} | `{path}` |\n"
        report += f"\n**Libraries:** {', '.join(backend_tk.get('libraries', []))}\n"
        report += f"**Supported arches:** {', '.join(backend_tk.get('supported_archs', []))}\n"
        os_info = backend_tk.get(os_key, {})
        if os_info:
            report += f"\n### {os_key.title()} notes\n\n{os_info.get('notes', '')}\n"

    # Phase 5: Performance estimation
    perf_bench = _load_perf_benchmarks()
    hw_est = perf_bench.get("hardware_theoretical", {}).get(params.target_arch, {})
    report += f"""

---

## Phase 5 — Performance Estimates

**Date/time:** {now.strftime('%Y-%m-%d %H:%M UTC')}
**Current time (for search filtering):** {now.strftime('%Y-%m-%d %H:%M UTC')}
"""

    if hw_est:
        report += f"### {target_info.get('name', '?')}\n\n"
        report += "| Property | Value |\n|---|---|\n"
        for k, v in hw_est.items():
            if k != "workload_estimates":
                report += f"| {k} | {v} |\n"
        report += "\n### Workload Estimates\n\n"
        for wl, est_val in hw_est.get("workload_estimates", {}).items():
            report += f"| {wl} | {est_val} |\n"

    if params.target_tok_per_s > 0:
        report += f"\n### Target Comparison\n\n"
        report += f"**Your target:** {params.target_tok_per_s} tok/s\n"
        if "flash_attn" in hw_est.get("workload_estimates", {}):
            est_str = hw_est["workload_estimates"]["flash_attn_4096"] if "flash_attn_4096" in hw_est.get("workload_estimates", {}) else hw_est["workload_estimates"].get("flash_attn", "?")
            report += f"**Estimated achievable:** ~{est_str}\n"
            report += f"**Gap analysis:** See `kands://diagnostics` for the iterative optimization loop\n"

    # Phase 6: Search for references
    report += f"""

---

## Phase 6 — Reference Research

**Search time:** {now.strftime('%Y-%m-%d %H:%M UTC')}

### Recommended searches
"""
    queries = perf_bench.get("web_search_queries", {}).get("queries", [])
    for q in queries[:5]:
        report += f"- `{q}`\n"

    report += """
### GitHub repos to check
"""
    for repo in perf_bench.get("web_search_queries", {}).get("github_repos_to_check", []):
        report += f"- `{repo}`\n"

    report += """
### ArXiv keywords to search
"""
    for kw in perf_bench.get("web_search_queries", {}).get("arxiv_keywords", []):
        report += f"- {kw}\n"

    report += f"""

> Run `search_arxiv(query="...")` or `search_github(query="...")` with these terms to find
> real-world benchmarks and reference implementations.

---

## Phase 7 — Porting Strategy (per kernel)

"""
    # Determine strategy per kernel
    patterns = _load_json(_PORTING_PATTERNS_PATH)
    source_vendor = source_info.get("vendor", "").lower()
    target_vendor = target_info.get("vendor", "").lower()

    backend_map = {"amd": "HIP", "nvidia": "CUDA", "intel": "SYCL"}
    src_backend = backend_map.get(source_vendor, "?")
    tgt_backend = backend_map.get(target_vendor, "?")

    # Determine pattern key
    pattern_key = None
    if source_vendor == "nvidia" and target_vendor == "amd":
        pattern_key = "cuda_to_hip"
    elif source_vendor == "nvidia" and target_vendor == "intel":
        pattern_key = "cuda_to_sycl"
    elif source_vendor == "amd" and target_vendor == "nvidia":
        pattern_key = "hip_to_cuda"
    elif source_vendor == "nvidia" and params.target_backend == "vulkan":
        pattern_key = "cuda_to_vulkan"
    elif source_vendor == "nvidia" and params.target_backend == "directx":
        pattern_key = "cuda_to_directx"

    if pattern_key and pattern_key in patterns.get("patterns", {}):
        p = patterns["patterns"][pattern_key]
        report += f"### {p.get('name', pattern_key)}\n\n"
        report += f"**Complexity:** {p.get('complexity', '?')}\n"
        report += f"**Mode:** `{params.mode}`\n\n"

        auto = p.get("auto_translatable", [])
        manual = p.get("manual_transforms", {})
        manual_only = p.get("manual_only", {})

        report += f"**Auto-translatable:** {len(auto)} constructs\n"
        for item in auto:
            report += f"- {item}\n"
        report += f"\n**Manual transforms needed:** {len(manual)} constructs\n"
        for src, tgt in list(manual.items())[:10]:
            tgt_val = tgt.get("hip", tgt.get("sycl", tgt.get("cuda", tgt.get("vulkan", "?"))))
            report += f"- `{src}` → `{tgt_val}`\n"
        if len(manual) > 10:
            report += f"... and {len(manual) - 10} more\n"

        report += f"\n**Manual-only (no auto translation):** {len(manual_only)} constructs\n"
        for src, tgt in manual_only.items():
            note = tgt.get("notes", "")
            tgt_name = tgt.get("hip", tgt.get("sycl", tgt.get("vulkan", "?")))
            report += f"- `{src}` → `{tgt_name}` — {note}\n"

        report += f"\n### Strategy Decision (for `{params.mode}` mode)\n\n"
        report += "| Kernel category | Strategy | Reason |\n|---|---|---|\n"
        report += "| Memory management (cudaMalloc → hipMalloc) | Copy | Direct replacement, auto-translatable |\n"
        report += "| Kernel launch (<<<>>> → hipLaunchKernelGGL) | Copy | Auto-translatable via hipify |\n"
        report += "| WMMA/MMA | Adapt | Source has {src_wmma} target has {tgt_wmma} — needs rocWMMA or F32 fallback |\n".format(
            src_wmma="✅ WMMA" if source_info.get("wmma") else "❌ WMMA",
            tgt_wmma="✅ WMMA" if target_info.get("wmma") else "❌ WMMA"
        )
        report += f"| Tensor cores/MMA | Adapt | Source has {src_tc} target has {tgt_tc} — needs dot4 or rocBLASLt |\n".format(
            src_tc="✅ TC" if source_info.get("tensor_cores") else "❌ TC",
            tgt_tc="✅ TC" if target_info.get("tensor_cores") else "❌ TC"
        )
        if params.target_backend == "vulkan":
            report += "| Vulkan matmul | Rewrite | AMD CM broken → must use DP4a (spv_dp4) |\n"
    else:
        report += f"**No auto-migration pattern from {src_backend} → {tgt_backend}. This requires manual adaptation.**\n"
        report += "Read `kands://porting-patterns` for the full matrix.\n"

    # Phase 8: Action plan
    report += f"""

---

## Phase 8 — Action Plan

### Strict mode (recommended for production):

1. **Build setup:** Read `kands://tk-tools` for compiler paths and flags
"""
    if toolkit_key:
        compiler_example = ""
        if toolkit_key == "rocm":
            compiler_example = f"hipcc --offload-arch={params.target_arch} -O3"
        elif toolkit_key == "cuda":
            compiler_example = f"nvcc --arch={params.target_arch} -O3"
        report += f"2. **CMake:** `cmake -DCMAKE_HIP_ARCHITECTURES={params.target_arch} -DCMAKE_BUILD_TYPE=Release`\n3. **Compile:** `{compiler_example} source.cu -o target`\n"

    report += "4. **Parity testing:** Run CPU oracle + GPU output comparison (see `kands://diagnostics`)\n"
    report += "5. **Performance:** Run `--perf-tracing`, then iterate using the sherlock-it loop (see `kands://diagnostics`)\n"
    report += f"6. **Export:** Run `export_contribution(engine_name='...', arch='{params.target_arch}')` to share results\n"
    report += "7. **Verify:** `ctest -R _parity` && `./engine --selftest`\n"

    report += f"""

### Loose mode (get it running first):

1. **Get it compiling:** Copy + auto-translate with `hipify-perl` or `dpct`
2. **Fallback everywhere:** Use FP32 where BF16/WMMA/Tensor cores are missing
3. **Quick test:** Verify output is in the right ballpark (tolerance 1e-2)
4. **Iterate:** Run perf tracing, note bottlenecks, run `suggest_optimizations`
5. **Optimize:** One kernel at a time, track with `record_performance`
6. **Share:** Export your learnings with `export_contribution`

---

## Next Steps

Run these in sequence:
```bash
# 1. Look up your toolchain
query_toolchain(gpu_vendor="{vendor_name}", os="{os_key}", host_compiler="auto")

# 2. Check kernel compatibility
check_arch_compatibility(kernel_list=["flash_attn", "router_topk", "kv_q4"], arch="{params.target_arch}")

# 3. Compare archs
compare_archs(arch_a="{params.source_arch}", arch_b="{params.target_arch}")

# 4. Search for references
search_references(query="{params.target_arch} {params.target_backend} performance benchmark")

# 5. After you optimize something:
record_performance(arch="{params.target_arch}", workload="flash_attn", tok_per_s=12.5)

# 6. Check if you're close to your target
compare_to_target(arch="{params.target_arch}", workload="flash_attn", target_tok_per_s={params.target_tok_per_s or 20})

# 7. Export your learnings
export_contribution(engine_name="my-port", arch="{params.target_arch}", workload="flash_attn")
```
"""
    return report


# ── MCP Server ──────────────────────────────────────────────────────────────"

server = Server("kands")


def _make_list_resources_handler():
    """Return an async handler for resources/list."""
    async def _handler(request, params):
        return mcp_types.ListResourcesResult(
            resources=_list_resources(),
            nextCursor=None,
        )
    return _handler


def _make_read_resource_handler():
    """Return an async handler for resources/read."""
    async def _handler(request, params):
        uri = params.uri
        try:
            text = _read_resource(uri)
        except ValueError as e:
            text = str(e)
        return mcp_types.ReadResourceResult(
            contents=[
                mcp_types.TextResourceContents(
                    uri=uri,
                    mimeType="text/plain",
                    text=text,
                )
            ]
        )
    return _handler


def _tools() -> list[mcp_types.Tool]:
    """All available MCP tools."""
    return [
        mcp_types.Tool(
            name="discover_kernels",
            description="Scan any repository (git repo or source tree) for GPU kernel, shader, and CPU kernel source files. Returns a categorized list.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Path to the repo (must exist on filesystem)"},
                    "ext_filter": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Optional extension filter, e.g. ['.cu']"
                    },
                },
                "required": ["repo"],
            },
        ),
        mcp_types.Tool(
            name="check_arch_compatibility",
            description="Check a list of kernel names against a target architecture's capabilities (wave, tensor cores, dot4, WMMA, bf16).",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_list": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Kernel names to check"
                    },
                    "arch": {"type": "string", "description": "Target arch ID (e.g. 'gfx1201', 'sm_80')"},
                },
                "required": ["kernel_list", "arch"],
            },
        ),
        mcp_types.Tool(
            name="query_kernel_features",
            description="Search an engine's source for architecture-specific preprocessor branches, feature flags, and dispatch points.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repo path to search"},
                    "pattern": {"type": "string", "description": "Regex or literal to match"},
                    "extensions": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Only search these extensions"
                    },
                },
                "required": ["repo", "pattern"],
            },
        ),
        mcp_types.Tool(
            name="compare_archs",
            description="Compare two GPU architectures' feature matrices and report the key differences affecting kernel porting.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arch_a": {"type": "string", "description": "First arch ID"},
                    "arch_b": {"type": "string", "description": "Second arch ID"},
                },
                "required": ["arch_a", "arch_b"],
            },
        ),
        mcp_types.Tool(
            name="generate_porting_report",
            description="Generate a structured Phase 0-7 porting report for a target engine+arch+model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "engine": {"type": "string", "description": "Engine name"},
                    "target_arch": {"type": "string", "description": "Target arch ID"},
                    "model_family": {"type": "string", "description": "Model family or 'auto'", "default": "auto"},
                    "repo_path": {"type": "string", "description": "Path to the engine repo"},
                },
                "required": ["engine", "target_arch", "repo_path"],
            },
        ),
        mcp_types.Tool(
            name="generate_tracing_plan",
            description="Analyze an engine's dispatch sites and produce a STRACE() instrumentation plan.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repo path to analyze"},
                    "components": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Components to instrument"
                    },
                },
                "required": ["repo"],
            },
        ),
        mcp_types.Tool(
            name="diff_perf_traces",
            description="Compare two perf-tracing outputs (before/after) to find regressions or improvements.",
            inputSchema={
                "type": "object",
                "properties": {
                    "trace_a": {"type": "string", "description": "Path to 'before' trace file or raw text"},
                    "trace_b": {"type": "string", "description": "Path to 'after' trace file or raw text"},
                },
                "required": ["trace_a", "trace_b"],
            },
        ),
        mcp_types.Tool(
            name="query_toolchain",
            description="Look up compiler paths, nvcc flags, and OS-specific toolchain details for a specific GPU backend + OS + host compiler combination.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_vendor": {
                        "type": "string",
                        "description": "GPU vendor: 'AMD', 'NVIDIA', 'Intel'",
                        "default": "AMD"
                    },
                    "os": {
                        "type": "string",
                        "description": "Target OS: 'windows', 'linux', or 'auto'",
                        "default": "auto"
                    },
                    "host_compiler": {
                        "type": "string",
                        "description": "Host compiler: 'msvc', 'gcc', 'clang', 'mingw', 'msys2', or 'auto'",
                        "default": "auto"
                    },
                },
                "required": [],
            },
        ),
        mcp_types.Tool(
            name="estimate_performance",
            description="Estimate theoretical and expected real-world GPU performance for an LLM workload on a target architecture. Uses hardware specs from ARCH_MAP.json + empirical scaling factors.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arch": {"type": "string", "description": "Target arch ID (e.g. 'gfx1201', 'sm_80') from kands://architectures"},
                    "workload": {
                        "type": "string",
                        "description": "Workload type: 'fp32_gemm', 'fp16_gemm', 'bf16_gemm', 'wmma_gemm', 'flash_attn', 'attention_f32_fallback', 'dequant_q4', 'dequant_q8', 'dequant_dot4'",
                        "default": "flash_attn"
                    },
                    "seq_len": {
                        "type": "integer",
                        "description": "Sequence length for attention workloads (default 4096)",
                        "default": 4096
                    },
                },
                "required": ["arch"],
            },
        ),
        mcp_types.Tool(
            name="compare_to_target",
            description="Compare a user's target performance goal to estimated achievable performance. Reports gap and optimization recommendations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arch": {"type": "string", "description": "Target arch ID from kands://architectures"},
                    "workload": {"type": "string", "description": "Workload type (matches estimate_performance workload)"},
                    "target_tok_per_s": {"type": "number", "description": "Target tokens/second you need to achieve"},
                    "current_tok_per_s": {"type": "number", "description": "Current measured tokens/second (if available)", "default": 0},
                },
                "required": ["arch", "workload", "target_tok_per_s"],
            },
        ),
        mcp_types.Tool(
            name="search_arxiv",
            description="Search ArXiv for recent papers about GPU ISAs, kernel porting, optimization, and performance benchmarks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g. 'AMD RDNA3 matrix core benchmark')"},
                    "max_results": {"type": "integer", "description": "Max results (default 5, max 10)", "default": 5},
                },
                "required": ["query"],
            },
        ),
        mcp_types.Tool(
            name="search_github",
            description="Search GitHub code for kernel/porting references, arch-specific patterns, and benchmark implementations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g. 'v_dot4_i32_i8 site:*.cu')"},
                    "max_results": {"type": "integer", "description": "Max results (default 10, max 20)", "default": 10},
                },
                "required": ["query"],
            },
        ),
        mcp_types.Tool(
            name="search_references",
            description="Combined ArXiv + GitHub search for a porting or optimization question.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g. 'RDNA4 DP4a matmul performance')"},
                    "max_results": {"type": "integer", "description": "Max results per source (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        ),
        mcp_types.Tool(
            name="record_performance",
            description="Record a performance measurement into the local perf history. The MCP learns from what you measure.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arch": {"type": "string", "description": "Target arch ID (e.g. 'gfx1201')"},
                    "workload": {"type": "string", "description": "Workload type (e.g. 'flash_attn', 'dequant_q4')"},
                    "tok_per_s": {"type": "number", "description": "Measured tokens per second"},
                    "config_hash": {"type": "string", "description": "Build config hash or description (e.g. 'gfx1201-STRATA_PREFILL_MMQ=ON')"},
                    "git_hash": {"type": "string", "description": "Git commit hash (optional)", "default": ""},
                    "notes": {"type": "string", "description": "Free-form notes about what was tried", "default": ""},
                },
                "required": ["arch", "workload", "tok_per_s"],
            },
        ),
        mcp_types.Tool(
            name="suggest_optimizations",
            description="Read the local perf history and suggest next optimization steps based on what worked/didn't for similar archs + workloads.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arch": {"type": "string", "description": "Target arch ID"},
                    "workload": {"type": "string", "description": "Workload type you're trying to optimize"},
                    "target_tok_per_s": {"type": "number", "description": "Target performance goal", "default": 0},
                },
                "required": ["arch"],
            },
        ),
        mcp_types.Tool(
            name="export_contribution",
            description="Export your porting learnings as a reusable contribution template.",
            inputSchema={
                "type": "object",
                "properties": {
                    "engine_name": {"type": "string", "description": "Name of your engine/project"},
                    "arch": {"type": "string", "description": "Target architecture ID"},
                    "workload": {"type": "string", "description": "Primary workload optimized"},
                    "output_path": {"type": "string", "description": "Where to save the template (default: findings/<name>.md)"},
                },
                "required": ["engine_name", "arch"],
            },
        ),
        mcp_types.Tool(
            name="set_goal",
            description="Define the ultimate goal + guardrails for a porting project. Prevents agent drift.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "One-sentence ultimate goal"},
                    "target_arch": {"type": "string", "description": "Target arch ID"},
                    "source_arch": {"type": "string", "description": "Source arch ID"},
                    "target_backend": {"type": "string", "description": "Target backend: hip/cuda/vulkan/directx/sycl", "default": "hip"},
                    "target_tok_per_s": {"type": "number", "description": "Target tokens/second"},
                    "min_perf": {"type": "float", "description": "Minimum viable perf", "default": 0},
                    "files_in_scope": {"type": "array", "items": {"type": "string"}, "description": "Glob patterns of files allowed to modify"},
                    "files_out_of_scope": {"type": "array", "items": {"type": "string"}, "description": "Glob patterns of files must NOT touch"},
                    "max_iterations": {"type": "integer", "description": "Max optimization iterations", "default": 10},
                    "guardrails": {"type": "array", "items": {"type": "string"}, "description": "Additional guardrail rules"},
                },
                "required": ["goal", "target_arch"],
            },
        ),
        mcp_types.Tool(
            name="update_goal",
            description="Override or update an existing goal/guardrail. Flexible — user can change any field. All overrides are logged.",
            inputSchema={
                "type": "object",
                "properties": {
                    "override_field": {"type": "string", "description": "Field to override (e.g. 'target_tok_per_s', 'files_in_scope')"},
                    "new_value": {"type": "string", "description": "New value as JSON string"},
                    "reason": {"type": "string", "description": "Why making this override"},
                },
                "required": ["override_field", "new_value"],
            },
        ),
        mcp_types.Tool(
            name="check_guardrail",
            description="Verify whether a proposed action is within the defined guardrails. Returns ALLOW/DENY with justification.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "What you want to do"},
                    "reason": {"type": "string", "description": "Why this action is needed"},
                },
                "required": ["action", "reason"],
            },
        ),
        mcp_types.Tool(
            name="prepare_pull_request",
            description="Generate exact git commands + PR description for contributing findings back to the K_and_S repo.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string", "description": "K_and_S repo URL", "default": "https://github.com/MaxRitz/K_and_S"},
                    "branch_name": {"type": "string", "description": "Branch name (default: auto-generated)"},
                    "change_types": {"type": "array", "items": {"type": "string"}, "description": "Change types: findings, arch_data, patterns, perf_data"},
                },
                "required": [],
            },
        ),
        mcp_types.Tool(
            name="verify_numerical_equivalence",
            description="Verify that ported kernel output matches the CPU oracle within tolerance. Uses CPU-as-oracle strategy for semantic equivalence checking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arch": {"type": "string", "description": "Target arch ID (e.g. 'gfx1201', 'sm_80')"},
                    "workload": {"type": "string", "description": "Workload type (e.g. 'flash_attn', 'fp32_gemm')"},
                    "source_values": {"type": "array", "items": {"type": "number"}, "description": "Reference output values from CPU oracle"},
                    "target_values": {"type": "array", "items": {"type": "number"}, "description": "Post-port values from GPU implementation"},
                    "tolerance": {"type": "number", "description": "Absolute tolerance for FP comparison", "default": 1e-5},
                    "relative_tolerance": {"type": "number", "description": "Relative tolerance", "default": 1e-4},
                },
                "required": ["arch", "workload", "source_values", "target_values"],
            },
        ),
        mcp_types.Tool(
            name="reject_common_pattern",
            description="Scrutinize a conventional GPU porting idea. Returns why it's wrong, the correct approach, and arch-specific evidence. Use to catch common traps before implementing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "common_idea": {"type": "string", "description": "The idea to scrutinize (e.g., 'hipify-perl handles everything', 'CooperativeMatrix works on AMD')"},
                    "arch": {"type": "string", "description": "Target arch ID (or 'auto' for general)", "default": "auto"},
                },
                "required": ["common_idea"],
            },
        ),
        mcp_types.Tool(
            name="radical_experiment_plan",
            description="Generate unconventional/experimental approaches for GPU kernel optimization. Returns radical but grounded experimentation ideas with implementation paths and fallback plans.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arch": {"type": "string", "description": "Target arch ID (e.g. 'gfx1201', 'sm_80')"},
                    "kernel_type": {"type": "string", "description": "Kernel type: gemm, attention, dequant, elementwise, memory_copy", "default": "gemm"},
                    "experiment_type": {"type": "string", "description": "Type: all, memory, compute, scheduling, layout", "default": "all"},
                },
                "required": ["arch"],
            },
        ),
        mcp_types.Tool(
            name="detect_toolchain",
            description="Dynamically detect installed GPU toolchains + host compilers at runtime. Probes the filesystem, PATH, and environment variables. Returns actual discovered paths — no hardcoded locations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_vendor": {"type": "string", "description": "GPU vendor to check: AMD, NVIDIA, Intel, Vulkan, or 'auto' for all", "default": "auto"},
                    "host_compiler": {"type": "string", "description": "Host compiler to detect: msvc, gcc, clang, mingw, msys2, strawberry, or 'auto'", "default": "auto"},
                },
                "required": [],
            },
        ),
        mcp_types.Tool(
            name="search_online",
            description="Search the web for current GPU toolchain download URLs, installation guides, version info, and porting references. Uses ArXiv + GitHub code search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g. 'ROC 10.1 install Windows hipcc')"},
                    "max_results": {"type": "integer", "description": "Max results per search source", "default": 5},
                },
                "required": ["query"],
            },
        ),
        mcp_types.Tool(
            name="autonomous_port",
            description="FULLY AUTOMATIC porting workflow: discovers kernels, checks arch compatibility, compares arches, looks up toolchain, estimates performance, searches references, generates complete porting plan. Accepts github_url (clones with security safeguards) OR repo_path (local). SECURITY: repo cloned read-only, no build scripts executed, file count + timeout limits enforced.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Local repo path (must exist). Use github_url for remote"},
                    "github_url": {"type": "string", "description": "GitHub URL to clone. SECURITY: read-only, no scripts executed"},
                    "source_arch": {"type": "string", "description": "Source arch ID (e.g. 'sm_80')"},
                    "target_arch": {"type": "string", "description": "Target arch ID (e.g. 'gfx1201')"},
                    "target_backend": {"type": "string", "description": "hip/cuda/vulkan/directx/sycl", "default": "hip"},
                    "target_os": {"type": "string", "description": "linux/windows/auto", "default": "auto"},
                    "target_tok_per_s": {"type": "number", "description": "Performance target (0 = no target)", "default": 0},
                    "mode": {"type": "string", "description": "strict/loose", "default": "strict"},
                    "max_files": {"type": "integer", "description": "Max files to scan (default 1000)", "default": 1000},
                    "timeout_seconds": {"type": "integer", "description": "Timeout for clone+scan (default 60s)", "default": 60},
                    "force_refresh": {"type": "boolean", "description": "Force re-analysis", "default": False},
                },
                "required": ["source_arch", "target_arch"],
            },
        ),
    ]


def _make_list_tools_handler():
    """Return an async handler for tools/list."""
    async def _handler(request, params):
        return mcp_types.ListToolsResult(tools=_tools(), nextCursor=None)
    return _handler


def _make_call_tool_handler():
    """Return an async handler for tools/call."""
    async def _handler(request, params):
        name = params.name
        arguments = params.arguments or {}
        results = await _dispatch_tool(name, arguments)
        return mcp_types.CallToolResult(content=results, isError=False)
    return _handler


async def _dispatch_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
    """Route tool calls to the appropriate handler."""
    arch_map = _load_json(_ARCH_MAP_PATH)
    model_map = _load_json(_MODEL_MAP_PATH)
    results: list[mcp_types.TextContent] = []

    if name == "discover_kernels":
        repo = arguments.get("repo", "")
        root = Path(repo)
        if not root.exists():
            results.append(mcp_types.TextContent(type="text", text=f"Error: {repo} does not exist"))
        else:
            ext_filter = arguments.get("ext_filter")
            findings = _scan_sources(root, ext_filter)
            if not findings:
                results.append(mcp_types.TextContent(type="text", text=f"No GPU/kernel/shader sources found in {repo}"))
            else:
                backends = {}
                for k in findings:
                    backends.setdefault(k["backend"], []).append(k["path"])
                summary = "## Discovery Summary\n\n"
                for backend, files in sorted(backends.items()):
                    summary += f"### {backend} ({len(files)} files)\n"
                    for f in files[:30]:
                        summary += f"- `{f}`\n"
                    if len(files) > 30:
                        summary += f"... and {len(files) - 30} more\n"
                    summary += "\n"
                summary += f"\n**Total: {len(findings)} kernel/shader files**\n"
                results.append(mcp_types.TextContent(type="text", text=summary))
                results.append(mcp_types.TextContent(type="text", text=json.dumps(findings, indent=2)))

    elif name == "check_arch_compatibility":
        kernel_list = arguments.get("kernel_list", [])
        arch_id = arguments.get("arch", "")
        arch_info = arch_map.get("entries", {}).get(arch_id, {})
        if not arch_info:
            results.append(mcp_types.TextContent(type="text", text=f"Architecture '{arch_id}' not found in ARCH_MAP.json"))
        else:
            report = f"# Architecture: {arch_id} ({arch_info.get('name', '?')})\n\n"
            report += "| Feature | Value |\n|---|---|\n"
            for feat in ["wave_size", "lds_kib", "tensor_cores", "wmma", "dot4", "dot8", "bf16_native", "fp8_native"]:
                report += f"| {feat} | {arch_info.get(feat, 'N/A')} |\n"
            report += "\n## Kernel Compatibility:\n\n"
            for kern in kernel_list:
                status = _check_kernel_against_arch(kern, arch_info)
                icon = {"OK": "✅", "NEEDS_FALLBACK": "⚠️", "INCOMPATIBLE": "❌"}[status]
                report += f"- {icon} `{kern}`: {status}\n"
            if arch_info.get("known_issues"):
                report += "\n## Known Issues:\n"
                for issue in arch_info["known_issues"]:
                    report += f"- {issue}\n"
            results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "query_kernel_features":
        repo = arguments.get("repo", "")
        pattern_str = arguments.get("pattern", "")
        exts = set(arguments.get("extensions", []))
        root = Path(repo)
        if not root.exists():
            results.append(mcp_types.TextContent(type="text", text=f"Error: {repo} does not exist"))
        else:
            findings = []
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for f in root.rglob("*"):
                if not f.is_file():
                    continue
                if exts and f.suffix.lower() not in exts:
                    continue
                if exts is None and f.suffix.lower() not in {".cu", ".cuh", ".hip", ".cpp", ".hpp", ".comp", ".metal", ".hlsl"}:
                    continue
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(text.splitlines(), 1):
                        if pattern.search(line):
                            findings.append({
                                "file": str(f.relative_to(root)),
                                "line": i,
                                "match": line.strip()[:200],
                            })
                except Exception:
                    pass
            flag_findings = _search_feature_flags(root)
            results.append(mcp_types.TextContent(
                type="text",
                text=json.dumps({
                    "source_matches": findings[:200],
                    "feature_flags": flag_findings,
                    "total_matches": len(findings),
                }, indent=2)
            ))

    elif name == "compare_archs":
        entries = arch_map.get("entries", {})
        a = entries.get(arguments.get("arch_a", ""))
        b = entries.get(arguments.get("arch_b", ""))
        if not a or not b:
            missing = arguments.get("arch_a") if not a else arguments.get("arch_b")
            results.append(mcp_types.TextContent(type="text", text=f"Architecture '{missing}' not found in ARCH_MAP.json"))
        else:
            report = f"# Comparing {arguments['arch_a']} ({a.get('name')}) vs {arguments['arch_b']} ({b.get('name')})\n\n"
            report += "| Feature | A | B | Gap |\n|---|---|---|---|\n"
            for k in ["wave_size", "lds_kib", "tensor_cores", "wmma", "dot4", "dot8", "bf16_native", "fp8_native"]:
                va = a.get(k, "?"); vb = b.get(k, "?")
                gap = "SAME" if str(va) == str(vb) else "DIFF"
                report += f"| {k} | {va} | {vb} | {gap} |\n"
            report += f"\n## Issues on {arguments['arch_a']}:\n"
            for issue in a.get("known_issues", []):
                report += f"- {issue}\n"
            report += f"\n## Issues on {arguments['arch_b']}:\n"
            for issue in b.get("known_issues", []):
                report += f"- {issue}\n"
            report += "\n## Porting Implications:\n"
            if a.get("wmma") and not b.get("wmma"):
                report += f"- WMMA kernels on {arguments['arch_a']} need F32 fallback on {arguments['arch_b']}\n"
            if a.get("bf16_native") and not b.get("bf16_native"):
                report += f"- BF16-native kernels on {arguments['arch_a']} need emulation on {arguments['arch_b']}\n"
            if a.get("dot8") and not b.get("dot8"):
                report += f"- Dot8 kernels on {arguments['arch_a']} need fallback on {arguments['arch_b']}\n"
            results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "generate_porting_report":
        root = Path(arguments.get("repo_path", ""))
        if not root.exists():
            results.append(mcp_types.TextContent(type="text", text=f"Error: {arguments.get('repo_path')} does not exist"))
        else:
            all_kernels = _scan_sources(root)
            gpu_kernels = [k for k in all_kernels if k["backend"].startswith("gpu")]
            cpu_kernels = [k for k in all_kernels if k["backend"].startswith("cpu")]
            backends = sorted(set(k["backend"] for k in all_kernels))
            arch = arch_map.get("entries", {}).get(arguments.get("target_arch", ""), {})
            params = ReportParams(**arguments)
            report = _build_report(params, root, gpu_kernels, cpu_kernels, backends, arch, model_map)
            results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "generate_tracing_plan":
        root = Path(arguments.get("repo", ""))
        if not root.exists():
            results.append(mcp_types.TextContent(type="text", text=f"Error: {arguments.get('repo')} does not exist"))
        else:
            components = arguments.get("components")
            plan = _build_tracing_plan(root, components, model_map)
            results.append(mcp_types.TextContent(type="text", text=plan))

    elif name == "diff_perf_traces":
        trace_a = _load_trace(arguments.get("trace_a", ""))
        trace_b = _load_trace(arguments.get("trace_b", ""))
        results.append(mcp_types.TextContent(type="text", text=_diff_perf_traces(trace_a, trace_b)))

    elif name == "estimate_performance":
        arch_id = arguments.get("arch", "")
        workload = arguments.get("workload", "flash_attn")
        seq_len = arguments.get("seq_len", 4096)
        report = _estimate_performance(arch_id, workload, seq_len, arch_map)
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "compare_to_target":
        arch_id = arguments.get("arch", "")
        workload = arguments.get("workload", "flash_attn")
        target = arguments.get("target_tok_per_s", 0)
        current = arguments.get("current_tok_per_s", 0)
        report = _compare_to_target(arch_id, workload, target, current, arch_map)
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "search_arxiv":
        query = arguments.get("query", "")
        max_results = min(arguments.get("max_results", 5), 10)
        try:
            findings = search_arxiv(query, max_results)
            report = f"# ArXiv Search: \"{query}\"\n"
            report += f"_Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n\n"
            report += "| # | Title | Date | Relevance | ArXiv ID |\n|---|---|---|---|---|\n"
            for i, p in enumerate(findings, 1):
                report += f"| {i} | {p['title'][:60]}... | {p.get('updated','?')[:10]} | {p.get('relevance','?')} | {p['arxiv_id']} |\n"
            report += "\n## Details\n\n"
            for p in findings:
                report += f"### {p['title']}\n"
                report += f"**Authors:** {', '.join(p.get('authors', []))}\n"
                report += f"**Published:** {p.get('published','?')}  |  **Updated:** {p.get('updated','?')}\n"
                report += f"**Categories:** {', '.join(p.get('categories', []))}\n"
                report += f"**Abstract:** {p['abstract']}\n"
                report += f"**URL:** {p['url']}\n\n"
            results.append(mcp_types.TextContent(type="text", text=report))
        except Exception as e:
            results.append(mcp_types.TextContent(type="text", text=f"ArXiv search error: {e}"))

    elif name == "search_github":
        query = arguments.get("query", "")
        max_results = min(arguments.get("max_results", 10), 20)
        try:
            findings = search_github(query, max_results)
            report = f"# GitHub Search: \"{query}\"\n"
            report += f"_Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n\n"
            report += f"**Note:** GitHub code search API requires `GITHUB_TOKEN` env var.\n\n"
            for r in findings:
                if r.get("type") == "fallback_search":
                    report += f"**Fallback:** {r.get('note', '')}\n"
                    report += f"Direct URL: {r.get('url', '')}\n"
                else:
                    report += f"### {r.get('name','?')}\n"
                    report += f"**Repo:** {r.get('repo','?')}  |  **Path:** {r.get('path','?')}\n"
                    report += f"**Relevance:** {r.get('relevance','?')}\n"
                    report += f"**URL:** {r.get('url','?')}\n\n"
            results.append(mcp_types.TextContent(type="text", text=report))
        except Exception as e:
            results.append(mcp_types.TextContent(type="text", text=f"GitHub search error: {e}"))

    elif name == "search_references":
        query = arguments.get("query", "")
        max_results = min(arguments.get("max_results", 5), 10)
        try:
            combined = search_references(query, max_results)
            report = f"# Combined Reference Search: \"{query}\"\n"
            report += f"_Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n\n"
            report += "## ArXiv Papers\n\n"
            for p in combined.get("arxiv", []):
                report += f"- **{p['title']}** ({p.get('updated','?')[:10]}) [{p.get('relevance','?')}]\n"
                report += f"  {p['abstract'][:200]}...\n  URL: {p['url']}\n\n"
            report += "## GitHub References\n\n"
            for r in combined.get("github", []):
                if r.get("type") == "fallback_search":
                    report += f"**Fallback:** {r.get('note','')}\n  {r.get('url','')}\n\n"
                else:
                    report += f"- **{r.get('name','?')}** in `{r.get('repo','?')}` [{r.get('relevance','?')}]\n"
                    report += f"  URL: {r.get('url','?')}\n\n"
            results.append(mcp_types.TextContent(type="text", text=report))
        except Exception as e:
            results.append(mcp_types.TextContent(type="text", text=f"Reference search error: {e}"))

    elif name == "record_performance":
        entry = _record_performance(arguments)
        results.append(mcp_types.TextContent(type="text", text=entry))

    elif name == "suggest_optimizations":
        arch = arguments.get("arch", "")
        workload = arguments.get("workload", "")
        target = arguments.get("target_tok_per_s", 0)
        report = _suggest_optimizations(arch, workload, target, arch_map)
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "export_contribution":
        engine_name = arguments.get("engine_name", "unnamed_engine")
        arch = arguments.get("arch", "")
        workload = arguments.get("workload", "")
        output_path = arguments.get("output_path", "")
        report = _export_contribution(engine_name, arch, workload, output_path)
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "query_toolchain":
        gpu_vendor = arguments.get("gpu_vendor", "AMD")
        os_target = arguments.get("os", "auto")
        host_compiler = arguments.get("host_compiler", "auto")
        if os_target == "auto":
            import platform as _platform
            os_target = "windows" if _platform.system() == "Windows" else "linux"
        report = _query_toolchain(gpu_vendor, os_target, host_compiler)
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "detect_toolchain":
        gpu_vendor = arguments.get("gpu_vendor", "auto")
        host_compiler = arguments.get("host_compiler", "auto")
        detected = _detect_installed_toolchain(gpu_vendor, host_compiler)
        report = _format_detection_result(detected)
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "search_online":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        report = _search_online(query, max_results)
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "autonomous_port":
        params = AutonomousPortParams(**arguments)
        security_warnings = []
        # SECURITY: Validate github_url scheme
        if params.github_url:
            if not params.github_url.startswith("https://github.com/") and not params.github_url.startswith("git+https://github.com/"):
                results.append(mcp_types.TextContent(type="text", text="ERROR: github_url must be https://github.com/ or git+https://github.com/ (arbitrary git URLs blocked)"))
                return results
            security_warnings.append("SECURITY: Repo cloned READ-ONLY. No build scripts or git hooks executed.")
            security_warnings.append(f"SECURITY: File count capped at {params.max_files}, timeout {params.timeout_seconds}s.")
        if not params.repo_path and not params.github_url:
            results.append(mcp_types.TextContent(type="text", text="ERROR: Must provide repo_path or github_url"))
            return results
        report = await _autonomous_port(params, arch_map)
        # Prepend security header if any warnings
        if security_warnings:
            header = "## Security Safeguards\n\n"
            for w in security_warnings:
                header += f"- {w}\n"
            header += "\n"
            report = report.replace("# Autonomous", "# Autonomous", 1)
            report = report.replace("\n---\n", "\n" + header + "---\n", 1)
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "set_goal":
        params = SetGoalParams(**arguments)
        state = _load_goal_state()
        now = datetime.now(timezone.utc).isoformat()
        state.update({
            "goal": params.goal,
            "target_arch": params.target_arch,
            "source_arch": params.source_arch,
            "target_backend": params.target_backend,
            "target_tok_per_s": params.target_tok_per_s,
            "min_perf": params.min_perf,
            "files_in_scope": params.files_in_scope,
            "files_out_of_scope": params.files_out_of_scope,
            "max_iterations": params.max_iterations,
            "guardrails": params.guardrails,
            "cutoff_date": now[:10],
            "created": state.get("created", now),
        })
        _save_goal_state(state)
        msg = f"✅ Goal set.\n\n**Goal:** {params.goal}\n**Target:** {params.target_arch} ({params.target_backend})\n**Tok/s target:** {params.target_tok_per_s or 'N/A'}\n**Min viable:** {params.min_perf or 'N/A'}\n**Files in scope:** {params.files_in_scope or 'all'}\n**Files out of scope:** {params.files_out_of_scope or 'none'}\n**Guardrails:** {len(params.guardrails)}\n\nRead `kands://goal` to view."
        results.append(mcp_types.TextContent(type="text", text=msg))

    elif name == "update_goal":
        params = UpdateGoalParams(**arguments)
        state = _load_goal_state()
        now = datetime.now(timezone.utc).isoformat()
        # Parse the JSON value
        try:
            new_val = json.loads(params.new_value)
        except json.JSONDecodeError:
            # If not valid JSON, try to interpret as a simple value
            if params.new_value.lower() in ("true", "false"):
                new_val = params.new_value.lower() == "true"
            else:
                new_val = params.new_value
        old_val = state.get(params.override_field, "not set")
        state[params.override_field] = new_val
        state.setdefault("override_log", []).append({
            "field": params.override_field,
            "old_value": str(old_val)[:200],
            "new_value": str(new_val)[:200],
            "reason": params.reason,
            "timestamp": now,
        })
        _save_goal_state(state)
        msg = f"✅ Goal updated.\n\n**Field:** `{params.override_field}`\n**Old:** {str(old_val)[:100]}\n**New:** {str(new_val)[:100]}\n**Reason:** {params.reason}\n\nRead `kands://goal` to view full state."
        results.append(mcp_types.TextContent(type="text", text=msg))

    elif name == "check_guardrail":
        params = CheckGuardrailParams(**arguments)
        state = _load_goal_state()
        action = params.action
        reason = params.reason
        goal = state.get("goal", "")
        files_in = state.get("files_in_scope", [])
        files_out = state.get("files_out_of_scope", [])

        verdict = "ALLOW"
        justifications = []

        # Check file scope
        if files_in or files_out:
            # Extract file path from action
            for pattern in files_out:
                if re.match(pattern.replace("*", ".*"), action):
                    verdict = "DENY"
                    justifications.append(f"Action targets file matching out-of-scope pattern: {pattern}")
            for pattern in files_in:
                if not re.match(pattern.replace("*", ".*"), action):
                    if not any(re.match(p.replace("*", ".*"), action) for p in files_in):
                        # Not in any in-scope pattern — flag as warning
                        justifications.append(f"⚠️ Action doesn't match any in-scope pattern")

        if verdict == "ALLOW":
            msg = f"✅ **ALLOW**\n\n**Goal:** {goal}\n**Action:** {action}\n**Reason:** {reason}\n\n✓ Action is within defined guardrails."
            if justifications:
                for j in justifications:
                    msg += f"\n{j}"
        else:
            msg = f"❌ **DENY**\n\n**Goal:** {goal}\n**Action:** {action}\n**Reason:** {reason}\n\n**Blocked by guardrails:**\n"
            for j in justifications:
                msg += f"- {j}\n"
            msg += f"\nUse `update_goal()` to override guardrails if this change is intentional."

        results.append(mcp_types.TextContent(type="text", text=msg))

    elif name == "prepare_pull_request":
        params = PreparePRParams(**arguments)
        branch = params.branch_name or f"porting-{(datetime.now()).strftime('%Y%m%d-%H%M%S')}"
        now = datetime.now(timezone.utc)

        # Scan for new/modified files
        import os as _os
        new_files = []
        if _GOAL_STATE_PATH.exists():
            new_files.append("mcp/state/goal.json")
        if _PERF_HISTORY_PATH.exists():
            new_files.append("mcp/state/perf_history.json")
        _ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for f in _ANALYSIS_CACHE_DIR.glob("*.md"):
            new_files.append(f"mcp/state/analysis/{f.name}")
        findings_dir = _KANDS_ROOT / "findings"
        if findings_dir.exists():
            for f in findings_dir.glob("*.md"):
                new_files.append(f"findings/{f.name}")

        report = f"""# Pull Request Preparation

**Date:** {now.strftime('%Y-%m-%d %H:%M UTC')}
**Branch:** `{branch}`

## Files to include in this PR

**New/modified files:**
"""
        for f in sorted(set(new_files)):
            report += f"- `{f}`\n"

        report += f"""

## Change types: {params.change_types}
## Commit message:

```
feat: add porting findings for {branch.split('-')[0] if '-' in branch else 'contribution'} contribution

Adds analysis results, performance measurements, and optimization notes.
Generated via K_and_S autonomous_port workflow.
```

## PR description:

**What's in this PR:**
"""
        if "findings" in params.change_types:
            report += "- Porting findings report (`.md` in `findings/`)\n"
        if "arch_data" in params.change_types:
            report += "- Architecture capability updates (ARCH_MAP.json)\n"
        if "patterns" in params.change_types:
            report += "- New porting transformation patterns (PORTING_PATTERNS.json)\n"
        if "perf_data" in params.change_types:
            report += "- Performance benchmark data (PERF_BENCHMARKS.json)\n"
        if "toolchain" in params.change_types:
            report += "- Toolchain additions (TK_TOOLS.json)\n"

        report += """
**Goal & guardrails:** All changes respect the defined goal in `kands://goal`.

## Git commands to run:

```bash
# 1. Create and switch to branch
git checkout -b """ + branch + f"""

# 2. Stage specific files (sanitized — no personal data)
git add findings/
git add mcp/state/ 2>/dev/null || true
git add ARCH_MAP.json 2>/dev/null || true
git add PORTING_PATTERNS.json 2>/dev/null || true
git add PERF_BENCHMARKS.json 2>/dev/null || true

# 3. Commit
git commit -m "feat: add porting findings + perf data for {branch.split('-')[0] if '-' in branch else 'contribution'}"

# 4. Push and create PR
git push -u origin """ + branch + """
gh pr create \
  --title "feat: porting findings for """ + branch.split('-')[0] if '-' in branch else 'contribution' + f""" — contributed via K_and_S" \
  --body "Generated automatedly via the K_and_S bundle. See included findings/ report for details."

# 5. Open in browser
gh pr view --web
```

> **Note:** All committed data is automatically sanitized (no emails, absolute paths, git hashes, or personal URLs).
"""
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "verify_numerical_equivalence":
        params = VerifyNumericalEquivalenceParams(**arguments)
        arch = params.arch
        workload = params.workload
        source_vals = params.source_values
        target_vals = params.target_values
        tol = params.tolerance
        rtol = params.relative_tolerance
        arch_map = _load_json(_ARCH_MAP_PATH)
        arch_info = arch_map.get("entries", {}).get(arch, {})

        report = "# Numerical Equivalence Verification (CPU-as-Oracle)\n\n"
        report += f"**Arch:** `{arch}` ({arch_info.get('name', 'unknown')})\n"
        report += f"**Workload:** `{workload}`\n"
        report += f"**Date:** {now.strftime('%Y-%m-%d %H:%M UTC')}\n\n"

        # Check array lengths
        if len(source_vals) != len(target_vals):
            report += "## ❌ MISMATCH — Array Length\n\n"
            report += f"Source has {len(source_vals)} values, target has {len(target_vals)} values.\n"
            report += "**Diagnosis:** Kernel launched with different workgroup size, shared memory, or reduction strategy.\n"
            report += "**Action:** Verify kernel dispatch configuration (grid/block dimensions, shared memory).\n"
        else:
            report += "## Array Length\n✅ Match (both {} values)\n\n".format(len(source_vals))

        # Check max absolute error
        max_abs_err = 0.0
        max_rel_err = 0.0
        divergence_indices = []
        for i, (sv, tv) in enumerate(zip(source_vals, target_vals)):
            abs_err = abs(sv - tv)
            rel_err = abs_err / max(abs(sv), 1e-30)
            if abs_err > max_abs_err:
                max_abs_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err
            if abs_err > tol and rel_err > rtol:
                divergence_indices.append(i)

        report += "## Error Metrics\n\n"
        report += f"| Metric | Value | Threshold |\n"
        report += f"|---|---|---|\n"
        report += f"| Max absolute error | {max_abs_err:.2e} | |tol|={tol:.2e} |\n"
        report += f"| Max relative error | {max_rel_err:.2e} | |rtol|={rtol:.2e} |\n"
        report += f"| Divergent values | {len(divergence_indices)} / {len(source_vals)} |\n"

        # Determine verdict
        passed = len(divergence_indices) == 0
        report += f"\n## Verdict: {'✅ PASS' if passed else '❌ FAIL'}\n\n"

        if passed:
            report += "All values match within specified tolerance. The ported kernel produces numerically equivalent output.\n"
        else:
            report += "## Divergent Values Analysis\n\n"
            report += f"Showing first {min(10, len(divergence_indices))} divergent indices:\n\n"
            for i in divergence_indices[:10]:
                sv = source_vals[i]
                tv = target_vals[i]
                report += f"- Index {i}: source={sv:.10f}, target={tv:.10f}, abs_diff={abs(sv-tv):.10f}\n"

            # Diagnostic suggestions
            report += "\n## Likely Causes & Fixes\n\n"
            # FP contraction check
            if arch_info.get('vendor', '').lower() == 'amd':
                report += "**AMD FP contraction:**\n"
                report += "- Check if `fma` instructions are being used differently than source\n"
                report += "- Add `STRACE(\"kernel\", \"exec\", \"fma_check\", \"contraction_enabled=%d\", is_fma_enabled)`\n"
                report += "- Compile with `-ffp-contract=off` to disable FMA contraction\n\n"
            if arch_info.get('vendor', '').lower() == 'nvidia':
                report += "**CUDA FP contraction:**\n"
                report += "- Use `--fmad=false` to control fused multiply-add contraction\n"
                report += "- Check `nvcc -warn-spills` for precision-sensitive spills\n\n"
            # Reduction order
            report += "**Reduction order effects:**\n"
            report += "- Different workgroup sizes produce different floating-point reduction order\n"
            report += "- Consider using Kahan summation for improved accuracy\n"
            report += "- Add a `STRACE_SCOPE(\"reduction\", \"compute\", \"layer=%d\", layer_idx)` to trace reduction steps\n\n"
            # Transcendental functions
            report += "**Transcendental function differences:**\n"
            report += "- libm implementations vary across toolchains\n"
            report += "- Verify `sin`, `cos`, `exp`, `log` produce matching results\n"
            report += "- Consider using `@fast_math` (CUDA) vs `-ffast-math` (GCC/Clang) consistently\n\n"
            # Architecture-specific
            if arch_info.get('bf16_native'):
                report += f"**{arch_info.get('name','Arch')} BF16 notes:**\n"
                report += "- Native BF16 support may produce different rounding than FP32 source\n"
                report += "- Consider using FP32 accumulation with BF16 input\n\n"
            if not arch_info.get('wmma') and 'wmma' in workload:
                report += "**WMMA fallback:** WMMA workload on arch without tensor cores → FP32 fallback used\n"
                report += "- This will produce correct but slower results\n"
                report += "- Consider DP4a (GL_KHR_shader_integer_dot_product) if integer ops are used\n"

        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "reject_common_pattern":
        params = RejectPatternParams(**arguments)
        arch_map = _load_json(_ARCH_MAP_PATH)
        idea = params.common_idea.lower()
        arch_id = params.arch

        report = "# Reject Common Pattern\\n\\n"
        report += f"**Examining:** \"{params.common_idea}\"\\n"
        report += f"**Target arch:** `{arch_id}`\\n"
        report += f"**Date:** {now.strftime('%Y-%m-%d %H:%M UTC')}\\n\\n"
        rejected = False

        if "hipify" in idea or "hipify-perl" in idea or "hipify tools" in idea:
            report += "## REJECTED: hipify-perl handles everything\\n\\n"
            report += "**Why it's wrong:**\\n"
            report += "- Syntactic translator does NOT handle semantic differences\\n"
            report += "- Template instantiations with __half/__nv_bfloat16 are not auto-correct\\n"
            report += "- Custom intrinsics (mma, imad, shfl) require MANUAL hipify\\n"
            report += "- Library API differences (cuBLASLt vs rocBLAS) need manual rewrite\\n"
            if "gfx" in arch_id:
                report += f"\\n**Correct for `{arch_id}`:**\\n"
                report += "- Run hipify-perl as STARTING POINT only\\n"
                report += "- Use `query_kernel_features` to find un-translated branches\\n"
                report += "- Validate with `verify_numerical_equivalence` after each change\\n"
                report += "- Check rocBLAS vs hipBLASLt backend (ROCBLAS_USE_HIPBLASLT=0 for RDNA2)\\n"
            report += "\\n**Evidence:** See `kands://porting-patterns` → CUDA→HIP for manual-only constructs.\\n"
            rejected = True

        if "verbatim" in idea or ("copy" in idea and "hip" in idea) or "directly" in idea:
            report += "## REJECTED: Copy CUDA to HIP verbatim\\n\\n"
            report += "**Why it's wrong:**\\n"
            report += "- Tensor core intrinsics (mma.*) have NO HIP equivalent — use rocWMMA or dot4\\n"
            report += "- `__ldg()` → normal loads on HIP\\n"
            report += "- `clock64()` → `__builtin_amdgcn_s_memtime()`\\n"
            report += "- Wave size assumptions (32 vs 32/64)\\n"
            rejected = True

        if "cooperative" in idea and "matrix" in idea and ("amd" in idea or "rdna" in idea or "radeon" in idea):
            report += "## REJECTED: Vulkan CooperativeMatrix works on AMD\\n\\n"
            report += "**Why it's wrong:**\\n"
            report += "- GL_KHR_cooperative_matrix is BROKEN on AMD drivers\\n"
            report += "- Use DP4a (GL_KHR_shader_integer_dot_product) instead\\n"
            report += "\\n**Correct approach:**\\n"
            report += "- Use spv_dp4 instructions for integer matmul\\n"
            report += "- For float: FMA chains or native WMMA (RDNA3+ only)\\n"
            report += "\\n**Evidence:** kands://architectures → check vulkan.cooperative_matrix field.\\n"
            rejected = True

        if "wmma" in idea or ("tensor core" in idea and "everywhere" in idea):
            report += "## CAUTION: Using tensor cores everywhere\\n\\n"
            report += "**Why it's risky:**\\n"
            report += "- Not all archs have tensor cores (RDNA2, Arc A770)\\n"
            report += "- For small matrices, F32 can be faster than WMMA (setup overhead)\\n"
            arch_info = arch_map.get("entries", {}).get(arch_id, {})
            if arch_info:
                has_tc = arch_info.get("tensor_cores", False)
                report += f"\\n**{arch_id}:** {'Has tensor cores — use WMMA' if has_tc else 'NO tensor cores — use F32 fallback (verified faster for sizes < 128x128)'}\\n"
            rejected = True

        if "threads" in idea and ("more" in idea or "occupancy" in idea):
            report += "## CAUTION: Maximizing thread count for occupancy\\n\\n"
            report += "**Why it's risky:**\\n"
            report += "- High occupancy ≠ high performance on memory-bound kernels\\n"
            report += "- Register pressure can LIMIT occupancy unexpectedly\\n"
            report += "- Wave32 vs Wave64: RDNA2 often better at wave32\\n"
            arch_info = arch_map.get("entries", {}).get(arch_id, {})
            if arch_info:
                report += f"\\n**{arch_id}:** wave_size={arch_info.get('wave_size','?')}, LDS={arch_info.get('lds_kib','?')}KiB/CU\\n"
            rejected = True

        if not rejected:
            report += "## NOT A KNOWN MYTH — but still verify\\n\\n"
            report += f"\"{params.common_idea}\" is not in our known anti-pattern list.\\n"
            report += "Known myths: 1) hipify-perl handles everything 2) Copy CUDA→HIP verbatim 3) Vulkan CM works on AMD 4) More threads=better 5) F32 fallback always slower 6) DP4a best for everything\\n"

        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "radical_experiment_plan":
        params = RadicalExperimentParams(**arguments)
        arch_map = _load_json(_ARCH_MAP_PATH)
        arch_info = arch_map.get("entries", {}).get(params.arch, {})
        vendor = arch_info.get("vendor", "unknown")

        report = "# Radical Experiment Plan\\n\\n"
        report += f"**Arch:** `{params.arch}` ({arch_info.get('name', 'unknown')})\\n"
        report += f"**Kernel type:** `{params.kernel_type}`\\n"
        report += f"**Date:** {now.strftime('%Y-%m-%d %H:%M UTC')}\\n\\n"
        report += "## EXPERIMENTAL — Validate with CPU Oracle Before Deploying\\n\\n"

        if vendor == "AMD" or "gfx" in params.arch:
            report += "## AMD/RDNA-Specific Radical Experiments\\n\\n"
            etype = params.experiment_type
            if etype in ("all", "memory"):
                report += "### Memory Hierarchy Experiments\\n"
                report += f"1. **Bypass LDS** for K<=16: register-only blocking (LDS={arch_info.get('lds_kib','?')}KiB/CU)\\n"
                report += "2. **DPP-based data shuffle** instead of LDS transpose (v_dot2, v_perm_b32, DPP8)\\n"
                report += "3. **Explicit `s_waitcnt lgkmcnt(0)`** before shared->register copy (inline asm)\\n\\n"
            if etype in ("all", "compute"):
                report += "### Compute Experiments\\n"
                report += f"1. **Native v_dot4** instead of FMA chains (wave_size={arch_info.get('wave_size','?')})\\n"
                report += "2. **Raw AMDGCN asm** for inner loop with `.scheduler_priority` hints\\n"
                report += "3. **Wave32 vs Wave64**: compile with `-mwavefrontsize=32`, compare\\n\\n"
            if etype in ("all", "scheduling"):
                report += "### Scheduling Experiments\\n"
                report += "1. **8-wave ping-pong vs 4-wave interleave** (HipKittens patterns)\\n"
                report += "2. Thread blocks: multiples of 32 vs wave_size\\n\\n"
            if etype in ("all", "layout"):
                report += "### Layout Experiments\\n"
                report += "1. **AoS vs SoA** — test cache locality tradeoff\\n"
                report += "2. **LDS bank-conflict-free padding** — pad to 2x width\\n\\n"
            report += "**RDNA-native note:** HipKittens (github.com/HazyResearch/HipKittens) targets CDNA3/CDNA4 but its tile abstractions directly adapt to RDNA — only wave_size (32) and LDS size differ.\\n\\n"

        elif vendor == "NVIDIA" or "sm_" in params.arch:
            report += "## NVIDIA-Specific Radical Experiments\\n"
            report += "1. **PTX-level custom scheduling** with `.scheduler_priority` hints\\n"
            report += "2. **FMAD vs separate FMUL+FADD** — compile with `--fmad=false`, measure\\n"
            report += "3. **Non-standard mma tile shapes** (8x8x4)\\n\\n"

        elif vendor == "Intel" or "gen" in params.arch:
            report += "## Intel-Specific Radical Experiments\\n"
            report += "1. **Direct DPAS invocation** via `__builtin_IB_sub_group_mm_*`\\n"

        report += "## Verification Requirements\\n"
        report += "1. Run reference on CPU (oracle)\\n"
        report += f"2. Call `verify_numerical_equivalence(arch=\"{params.arch}\", workload=\"{params.kernel_type}\")`\\n"
        report += "3. STRACE() + fence timeout (≤1500ms Windows)\\n"
        report += "4. Fallback to last-good, NEVER stub\\n"
        report += f"5. `record_performance(arch=\"{params.arch}\", workload=\"{params.kernel_type}\")`\\n"

        results.append(mcp_types.TextContent(type="text", text=report))


# Register resource handlers
server.add_request_handler(
    "resources/list",
    mcp_types.PaginatedRequestParams,
    _make_list_resources_handler(),
)

server.add_request_handler(
    "resources/read",
    mcp_types.ReadResourceRequestParams,
    _make_read_resource_handler(),
)

server.add_request_handler(
    "tools/list",
    mcp_types.PaginatedRequestParams,
    _make_list_tools_handler(),
)

server.add_request_handler(
    "tools/call",
    mcp_types.CallToolRequestParams,
    _make_call_tool_handler(),
)


def main():
    """Entry point: run the MCP server over stdio."""
    from mcp.server.stdio import stdio_server
    asyncio.run(stdio_server(server))


if __name__ == "__main__":
    main()
