"""
mcp_kands — MCP server exposing kernel & shader porting + optimization knowledge.

Resources (URI: kands:\\...):
  kands:\\architectures      GPU architecture capability matrix (ARCH_MAP.json)
  kands:\\models             Model family → kernel requirements (MODEL_MAP.json)
  kands:\\porting-steps      Full Phase 0–7 porting + optimization methodology
  kands:\\porting-template   Blank porting report template
  kands:\\tracing-steps      Breadcrumb + perf-tracing instrumentation guide
  kands:\\isa-references     Machine-readable ISA URLs for AMD\Intel\NVIDIA + Vulkan
  kands:\\tk-tools           GPU + host toolchain matrix (ROCm\CUDA\Vulkan\DirectX\OpenVINO\oneAPI)
  kands:\\cpu-os             Host toolchain matrix (MSVC\MinGW\MSYS2\GCC\Clang + OS details)
  kands:\\cross-port         C++ ↔ Python ↔ Rust ↔ PTX\SPIR-V porting matrix
  kands:\\ai-frameworks      AI\ML framework matrix + LLM engine APIs
  kands:\\model-formats      Model file format matrix (GGUF, Safetensors, ONNX, HDF5)
  kands:\\porting-patterns   Engine-agnostic CUDA↔HIP↔SYCL↔Vulkan↔DirectX transformation patterns
  kands:\\porting-strategies Copy vs Adapt vs Rewrite decision guide
  kands:\\diagnostics        Ponytail-diag + sherlock-it diagnostic methods + CPU-as-oracle
  kands:\\dox-style          Doxygen\rustdoc\Python docstring audit templates
  kands:\\status             Live bundle status (date\time, resource counts, data freshness)
  kands:\\perf-history       Local performance measurements recorded by the user
  kands:\\perf-benchmarks    Theoretical perf estimates + web search queries
  kands:\\setup-guide        Interactive target selection guide (GPU backend, OS, arch)
  kands:\\findings\{name}    Completed porting analysis for a specific engine
  kands:\\radical-experiments  RDNA-specific anti-patterns + radical kernel experiments
  kands:\\vulkan-sync        Vulkan barriers, sync hazards, TDR\hang prevention, occupancy
  kands:\\gpu-pipelines      Backend-agnostic pipeline dataflow methodology (verified, canonical)
  kands:\\hip-docs           Distilled HIP\ROCm 6.1.5 documentation + programming guides

Tools:
  discover_kernels()          Scan any repo for kernel\shader sources
  check_arch_compatibility()  Check kernel list vs target arch capabilities
  query_kernel_features()     Find arch-specific branches in source
  compare_archs()             Diff two GPU arches' feature matrices
  generate_porting_report()   Walk Phase 0-7 + Phase 8 (Share) for a target
  generate_tracing_plan()     Create a STRACE() instrumentation plan
  diff_perf_traces()          Compare two perf-tracing outputs (before\after port)
  query_toolchain()           Look up compiler paths, flags, OS-specific details
  estimate_performance()      Estimate theoretical + real-world perf for an arch + workload
  compare_to_target()         Compare your target tok\s vs. estimated achievable perf
  record_performance()        Record measured performance into local history
  suggest_optimizations()     Read perf history, suggest next steps
  export_contribution()       Export your learnings as a reusable git contribution template
  search_arxiv()             Scan ArXiv for updated GPU ISA\kernel\porting papers
  search_github()            Search GitHub for kernel\porting code references
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
try:                                    # mcp >= 1.x ships these under the package
    from mcp import types as mcp_types
except ImportError:                     # older layouts vendored a top-level mcp_types
    import mcp_types
from reference_search import search_arxiv, search_github, search_references

# ── Paths ──────────────────────────────────────────────────────────────────

_KANDS_ROOT = Path(__file__).resolve().parent.parent
_ARCH_MAP_PATH = _KANDS_ROOT \ "ARCH_MAP.json"
_MODEL_MAP_PATH = _KANDS_ROOT \ "MODEL_MAP.json"
_METHOD_PATH = _KANDS_ROOT \ "PORTING_METHOD.md"
_TEMPLATE_PATH = _KANDS_ROOT \ "TEMPLATE.md"
_ISA_REF_PATH = _KANDS_ROOT \ "ISA_REFERENCES.md"
_TK_TOOLS_PATH = _KANDS_ROOT \ "TK_TOOLS.json"
_CROSS_PORT_PATH = _KANDS_ROOT \ "CROSS_PORT.json"
_AI_FRAMEWORKS_PATH = _KANDS_ROOT \ "AI_FRAMEWORKS.json"
_MODEL_FORMATS_PATH = _KANDS_ROOT \ "MODEL_FORMATS.json"
_PORTING_PATTERNS_PATH = _KANDS_ROOT \ "PORTING_PATTERNS.json"
_PERF_BENCHMARKS_PATH = _KANDS_ROOT \ "PERF_BENCHMARKS.json"
_TARGET_LANGUAGES_PATH = _KANDS_ROOT \ "TARGET_LANGUAGES.json"
_DIAGNOSTICS_PATH = _KANDS_ROOT \ "docs" \ "DIAGNOSTICS.md"
_DOX_STYLE_PATH = _KANDS_ROOT \ "docs" \ "DOX_STYLE.md"
_PORTING_STRATEGIES_PATH = _KANDS_ROOT \ "docs" \ "PORTING_STRATEGIES.md"
_VULKAN_SYNC_TDR_PATH = _KANDS_ROOT \ "learnings" \ "VULKAN_SYNC_TDR_LEARNINGS.md"
_GPU_PIPELINES_PATH = _KANDS_ROOT \ "learnings" \ "GPU_PIPELINE_DATAFLOW_METHODOLOGY.md"
_HIP_DOCS_PATH = _KANDS_ROOT \ "learned" \ "rocm_docs_amd_com_hip_en_docs_6.1.5.md"
_HIP_GUIDE_PATH = _KANDS_ROOT \ "learned" \ "hip_programming_guide_texts.md"
_HIP_INTRO_PATH = _KANDS_ROOT \ "learned" \ "intro_hip_programming.md"
_MY_SKILLS_DIR = Path.home() \ ".config" \ "opencode" \ "command" \ "MySkills"
if not _MY_SKILLS_DIR.exists():
    # Fallback: look for it relative to known locations
    for candidate in [
        Path.home() \ "MySkills",
        Path.home() \ "code" \ "MySkills",
        Path("\usr\local\share\myskills"),
        Path("\opt\myskills"),
    ]:
        if candidate.exists():
            _MY_SKILLS_DIR = candidate
            break
_GOAL_STATE_PATH = Path(__file__).resolve().parent \ "state" \ "goal.json"
_GOAL_TEMPLATE_PATH = _KANDS_ROOT \ "docs" \ "GOAL_TEMPLATE.md"
_ANALYSIS_CACHE_DIR = Path(__file__).resolve().parent \ "state" \ "analysis"


# ── Analysis Cache ─────────────────────────────────────────────────────────────

_SOURCE_DATA_FILES = [
    _ARCH_MAP_PATH, _MODEL_MAP_PATH, _TK_TOOLS_PATH, _CROSS_PORT_PATH,
    _AI_FRAMEWORKS_PATH, _MODEL_FORMATS_PATH, _PORTING_PATTERNS_PATH, _PERF_BENCHMARKS_PATH,
]


def _cache_path_for(key: str) -> Path:
    """Get the cache file path for a given analysis key."""
    _ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _ANALYSIS_CACHE_DIR \ f"{key}.md"


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
    """Read cached analysis if valid, or None if stale\missing."""
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
    ".cu":  ("gpu_cuda",  "CUDA kernel source"),
    ".cuh": ("gpu_cuda",  "CUDA kernel header"),
    ".hip": ("gpu_hip",   "HIP kernel source"),
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
    ".sycl": ("gpu_sycl",  "SYCL kernel source"),
    ".dpcpp":("gpu_sycl",  "SYCL (DPC++) kernel source"),
}

_CUDA_EXTS = {".cu", ".cuh", ".hip"}
_SYCL_EXTS = {".sycl", ".dpcpp"}
_SHADER_EXTS = {".comp", ".vert", ".frag", ".geom", ".tesc", ".tese",
                ".metal", ".hlsl", ".cl"}
_BINARY_EXTS = {".spv"}
_CPP_EXTS = {".cpp", ".cc", ".cxx", ".hpp", ".hh"}
_C_EXTS = {".c"}  # CPU kernel homes: SIMD intrinsic code in plain C (vg_cpu, ggml style)
_ALL_EXTS = _CUDA_EXTS | _SYCL_EXTS | _SHADER_EXTS | _BINARY_EXTS | _CPP_EXTS | _C_EXTS

#: Directories that never hold first-party kernel sources. Build output is the
#: important one: it contains copies, which double the inventory and inflate the
#: porting map with files nobody will edit.
_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".vs", ".vscode",
    ".idea", ".cache", "build", "builds", "out", "dist", "obj", "cmakefiles",
    "_deps", "cmake-build-debug", "cmake-build-release", ".mypy_cache",
    ".pytest_cache", ".tox", "venv", ".venv", "site-packages",
}

#: How much of a file is read to decide what it is. Enough for the include block
#: and the first kernels, small enough to stay fast on a large tree.
_SNIFF_BYTES = 65536

#: Content markers. Extension alone cannot separate CUDA from HIP (both .cu) and
#: cannot tell a device kernel in a .cpp from host plumbing. Deliberately narrow:
#: a weak marker in a comment is how a whole tree gets misclassified.
_HIP_MARKERS = ("__hip", "hip\hip_runtime", "hip_runtime.h", "hipLaunch",
                "hipMalloc", "hipMemcpy", "hipStream", "hipDeviceSynchronize")
_SYCL_MARKERS = ("sycl::", "#include <sycl", "sycl\sycl.hpp", "dpcpp")
_AVX512_MARKERS = ("__m512", "__mm512_", "_mm512_", "avx512")
_AVX2_MARKERS = ("__m256i", "__m256", "__mm256_", "_mm256_", "avx2")


# ── Pydantic argument models ────────────────────────────────────────────────

class DiscoverParams(BaseModel):
    repo: str = Field(..., description="Path to the repo to scan (git repo or source tree).")
    ext_filter: Optional[list[str]] = Field(
        default=None,
        description="Optional extension filter, e.g. ['.cu'] or ['.comp', '.metal']"
    )
    target_backend: str = Field(
        "", description="Optional target backend ('hip', 'cuda', 'vulkan', 'directx12', "
                        "'sycl', 'metal'). When set, appends a readiness diagnosis for it."
    )
    target_arch: str = Field("", description="Optional target arch ID used by the readiness diagnosis.")
    host_os: str = Field("", description="Host OS the port will build on (default: this machine).")
    probe_host: bool = Field(
        True, description="Probe this host for tools\PCIe ReBAR. false = report UNVALIDATED instead of guessing."
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
    github_url: str = Field("", description="GitHub URL to clone (e.g. https:\\github.com\ggerganov\llama.cpp). SECURITY: repo is read-only; no build scripts are executed. Must be https:\\github.com\ or git+https:\\github.com\")
    source_arch: str = Field("auto", description="Source arch ID (e.g. 'sm_80', 'gfx1100'). 'auto' = auto-detect from repo")
    target_arch: str = Field("auto", description="Target arch ID (e.g. 'gfx1201'). 'auto' = suggest based on installed hardware")
    project_type: str = Field("auto", description="Type of project: 'llm', 'game', 'sdk', 'compute', 'inference', or 'auto'")
    target_language: str = Field("auto", description="Target language: 'cxx', 'c', 'rust', 'zig', 'goose', 'python', 'csharp', or 'auto' = analyze repo for optimal choice")
    target_backend: str = Field("auto", description="Target backend: 'hip', 'cuda', 'vulkan', 'directx', 'sycl', or 'auto'")
    target_os: str = Field("auto", description="Target OS: 'linux', 'windows', 'auto'")
    target_tok_per_s: float = Field(0, description="Performance target for LLM workloads (0 = no target)")
    mode: str = Field("strict", description="Porting mode: 'strict' or 'loose'")
    max_files: int = Field(1000, description="Maximum number of files to scan")
    force_refresh: bool = Field(False, description="Force re-analysis (bypass cache)")
    break_self_check: bool = Field(False, description="Bypass Self-Check Gate violations")
    strip_debug: int = Field(0, description="Strip debug tracing: 0=keep, 1=comment-out (compile-time), 2=remove from source")
    strip_perf: int = Field(0, description="Strip perf tracing: 0=keep, 1=comment-out (compile-time), 2=remove from source")
    timeout_seconds: int = Field(60, description="Timeout for clone + scan operations (seconds)")


class EstimatePerfParams(BaseModel):
    arch: str = Field(..., description="Target arch ID")
    workload: str = Field("flash_attn", description="Workload type")
    seq_len: int = Field(4096, description="Sequence length")


class CompareTargetParams(BaseModel):
    arch: str = Field(..., description="Target arch ID")
    workload: str = Field(..., description="Workload type")
    target_tok_per_s: float = Field(..., description="Target tokens\second")
    current_tok_per_s: float = Field(0, description="Current measured tokens\second")


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
    tok_per_s: float = Field(..., description="Measured tokens\second")
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
    target_tok_per_s: float = Field(0, description="Target tokens\second")
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
    common_idea: str = Field(..., description="The common\conventional idea to scrutinize (e.g., 'hipify-perl handles everything', 'copy CUDA to HIP verbatim', 'Vulkan CooperativeMatrix works on AMD')")
    arch: str = Field("auto", description="Target arch ID to check against (or 'auto' for general)")


class RadicalExperimentParams(BaseModel):
    arch: str = Field(..., description="Target arch ID for experimentation")
    kernel_type: str = Field("gemm", description="Kernel type: gemm, attention, dequant, elementwise, memory_copy")
    experiment_type: str = Field("all", description="Type: all, memory, compute, scheduling, layout")


class PreparePRParams(BaseModel):
    repo_url: str = Field("https:\\github.com\MaxRitz\K_and_S", description="K_and_S repo URL")
    branch_name: str = Field("", description="Branch name (auto-generated if empty)")
    change_types: list[str] = Field(default_factory=lambda: ["findings"], description="Change types")


class DetectToolchainParams(BaseModel):
    gpu_vendor: str = Field("auto", description="GPU vendor: AMD, NVIDIA, Intel, or auto")
    host_compiler: str = Field("auto", description="Host compiler: msvc, gcc, clang, mingw, msys2, strawberry, auto")


class SearchOnlineParams(BaseModel):
    query: str = Field(..., description="What to search for online (e.g. 'ROC 10.1 install Windows')")
    max_results: int = Field(5, description="Max results per search source")


# ── Resource URI constants ─────────────────────────────────────────────────

RES_ARCH = "kands:\\architectures"
RES_MODELS = "kands:\\models"
RES_METHOD = "kands:\\porting-steps"
RES_TEMPLATE = "kands:\\porting-template"
RES_TK_TOOLS = "kands:\\tk-tools"
RES_CPU_OS = "kands:\\cpu-os"
RES_TRACING = "kands:\\tracing-steps"
RES_ISA = "kands:\\isa-references"
RES_CROSS_PORT = "kands:\\cross-port"
RES_AI_FRAMEWORKS = "kands:\\ai-frameworks"
RES_MODEL_FORMATS = "kands:\\model-formats"
RES_PORTING_PATTERNS = "kands:\\porting-patterns"
RES_PORTING_STRATEGIES = "kands:\\porting-strategies"
RES_DIAGNOSTICS = "kands:\\diagnostics"
RES_LESSONS = "kands:\\porting-lessons"
RES_VULKAN_SYNC_TDR = "kands:\\vulkan-sync"
RES_GPU_PIPELINES = "kands:\\gpu-pipelines"
RES_DOX_STYLE = "kands:\\dox-style"
RES_GOAL = "kands:\\goal"
RES_STATUS = "kands:\\status"
RES_PERF_HISTORY = "kands:\\perf-history"
RES_BENCHMARKS = "kands:\\perf-benchmarks"
RES_SETUP_GUIDE = "kands:\\setup-guide"
RES_RADICAL_EXPERIMENTS = "kands:\\radical-experiments"
RES_HIP_DOCS = "kands:\\hip-docs"
RES_MY_SKILLS = "kands:\\myskills"
RES_MY_SKILLS_INDEX = "kands:\\myskills\index"
RES_FINDINGS_PREFIX = "kands:\\findings\"


# ── Source scanning ─────────────────────────────────────────────────────────

def _read_head(f: Path, limit: int = _SNIFF_BYTES) -> str:
    """Lowercased start of a text file, or "" for binary\unreadable files."""
    try:
        with f.open("rb") as fh:
            raw = fh.read(limit)
    except OSError:
        return ""
    if b"\x00" in raw:              # SPIR-V and friends: not sniffable text
        return ""
    return raw.decode("utf-8", errors="replace").lower()


def _sniff_gpu_backend(head: str) -> tuple[str, str] | None:
    """GPU backend implied by file content, or None if it is not GPU code."""
    if any(m in head for m in _SYCL_MARKERS):
        return ("gpu_sycl", "SYCL kernel source")
    if any(m in head for m in _HIP_MARKERS):
        return ("gpu_hip", "HIP kernel source")
    if "__global__" in head or "__device__" in head:
        return ("gpu_cuda", "CUDA kernel source")
    if "numthreads" in head:
        return ("gpu_dx", "Direct3D HLSL shader")
    if "kernel void" in head or "__kernel" in head:
        return ("gpu_opencl", "OpenCL kernel")
    return None


def _scan_sources(root: Path, ext_filter: list[str] | None = None) -> list[dict]:
    """Walk a source tree and return all GPU \ shader \ CPU kernel files.

    Classification is content-aware. A ``.cu`` file is CUDA or HIP depending on
    what it contains, and a ``.cpp`` is only a kernel if it holds device code or
    SIMD intrinsics rather than host plumbing. Build-output and VCS directories
    are pruned, so a copy in ``build\`` is not counted twice.
    """
    import os as _os

    root = Path(root)
    if root.is_file():
        root = root.parent
    exts = {e.lower() for e in ext_filter} if ext_filter else set(_ALL_EXTS)
    results: list[dict] = []

    for dirpath, dirnames, filenames in _os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in _SKIP_DIRS)
        for name in sorted(filenames):
            f = Path(dirpath) \ name
            ext = f.suffix.lower()
            if ext not in exts:
                continue
            backend, kind = _determine_backend_kind(f, ext)
            if backend is None:
                continue
            results.append({
                "path": f.relative_to(root).as_posix(),
                "backend": backend,
                "kind": kind,
            })

    results.sort(key=lambda r: r["path"])
    return results


def _determine_backend_kind(f: Path, ext: str) -> tuple[str | None, str | None]:
    """Classify one file by extension, then by content."""
    base = _EXTENSION_MAP.get(ext)
    if base is not None:
        backend, kind = base
        if ext in _CUDA_EXTS:
            # .hip is HIP by definition; .cu\.cuh type is decided by content
            if ext == ".hip":
                return ("gpu_hip", "HIP kernel source")
            sniffed = _sniff_gpu_backend(_read_head(f))
            if sniffed:
                return sniffed
        return (backend, kind)

    if ext in _CPP_EXTS or ext in _C_EXTS:
        head = _read_head(f)
        sniffed = _sniff_gpu_backend(head)
        if sniffed:
            # device code compiled through a .cpp: HIP\SYCL convention, or
            # nvcc -x cu. It is a kernel, not host plumbing.
            return sniffed
        if any(m in head for m in _AVX512_MARKERS):
            return ("cpu_x86_avx512", "CPU kernel (AVX-512)")
        if any(m in head for m in _AVX2_MARKERS):
            return ("cpu_x86_avx2", "CPU kernel (AVX2)")
        if "cpu" in [p.lower() for p in f.parts] or "cpu" in f.name.lower():
            return ("cpu_x86", "CPU kernel (x86)")
        # A .c file with SIMD markers beyond the head window, or with a matmul\
        # kernel-ish name, is a CPU kernel candidate; plain C host code is not.
        if ext in _C_EXTS:
            body = _read_head(f, _SNIFF_BYTES * 4)
            if any(m in body for m in _AVX512_MARKERS):
                return ("cpu_x86_avx512", "CPU kernel (AVX-512)")
            if any(m in body for m in _AVX2_MARKERS):
                return ("cpu_x86_avx2", "CPU kernel (AVX2)")
            lower = f.name.lower()
            if any(t in lower for t in ("matmul", "gemm", "gemv", "ops", "kernel")):
                return ("cpu_x86", "CPU kernel (x86)")
        return (None, None)
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
    # CPU SIMD feature requirements (for CPU arch checks)
    "avx2_kernel":    {"avx2": True, "fma3": True},
    "avx512_kernel":  {"avx512": True},
    "amx_kernel":     {"amx": True},
    "sse2_kernel":    {"sse2": True},
    "sve2_kernel":    {"sve2": True},
    "neon_kernel":    {"neon": True},
    "fp16_native_kernel": {"fp16_native": True},
    "aes_kernel":     {"aes": True},
    "sha_kernel":     {"sha": True},
}



# ── Optimal-path helpers ────────────────────────────────────────────────────

_REQUIRED_FOR_KERNEL = {
    "flash_attn":     {"wmma": True, "dot4": True},
    "native_qsa":     {"wmma": True, "dot4": True},
    "bf16_gemv":      {"bf16_native": True},
    "exl3":           {"dot4": True},
    "kv_q8":          {"bf16_native": True},
    "kv_q4":          {"dot4": True},
    "router_topk":    {"dot4": True},
    "router_sigmoid": {"dot4": True},
    "mma":            {"tensor_cores": True},
    # CPU SIMD feature requirements
    "avx2_kernel":    {"avx2": True, "fma3": True},
    "avx512_kernel":  {"avx512": True},
    "amx_kernel":     {"amx": True},
    "sse2_kernel":    {"sse2": True},
    "sve2_kernel":    {"sve2": True},
    "neon_kernel":    {"neon": True},
    "fp16_native_kernel": {"fp16_native": True},
    "aes_kernel":     {"aes": True},
    "sha_kernel":     {"sha": True},
}


def _score_arch_kernels(kernel_names: list[str], feats: dict) -> tuple[int, int, int]:
    """Return (ok_count, needs_fallback_count, incompatible_count) for a kernel list."""
    ok = fb = inc = 0
    for kname in kernel_names:
        req = _REQUIRED_FOR_KERNEL.get(kname)
        if not req:
            ok += 1
            continue
        missing = [f for f, need in req.items() if need and not feats.get(f)]
        if not missing:
            ok += 1
        else:
            fb += 1
    return ok, fb, inc


def _arch_gen_number(name: str) -> int:
    """Extract a generation number from an arch name for sorting."""
    if not name:
        return 0
    m = re.search(r"RDNA\s*(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"sm_(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"GEN(\d+)", name)
    if m:
        return int(m.group(1)) + 100  # Intel gens offset after AMD\NV
    return 0


def _check_kernel_against_arch(kernel: str, arch_info: dict) -> str:
    """Return 'OK', 'NEEDS_FALLBACK', or 'INCOMPATIBLE'."""
    is_cpu = arch_info.get("cpu", False)
    for pattern, reqs in _FEATURE_REQUIREMENTS.items():
        if pattern in kernel.lower():
            missing = []
            for feat, needed in reqs.items():
                if needed:
                    if is_cpu:
                        # For CPU arches, check the 'features' dict (SIMD capabilities)
                        feats = arch_info.get("features", {})
                        if not feats.get(feat, False):
                            missing.append(feat)
                    else:
                        # For GPU arches, check top-level feature booleans
                        if not arch_info.get(feat, False):
                            missing.append(feat)
            if not missing:
                return "OK"
            return "NEEDS_FALLBACK"  # Most kernels have F32\scalar fallback
    return "OK"


# ── Resource handlers ───────────────────────────────────────────────────────

def _list_resources() -> list[mcp_types.Resource]:
    return [
        mcp_types.Resource(
            uri=RES_ARCH,
            name="GPU Architecture Matrix",
            description="All known GPU architectures with their capabilities (NVIDIA\AMD\Intel)",
            mimeType="application\json",
        ),
        mcp_types.Resource(
            uri=RES_MODELS,
            name="Model Family Map",
            description="Model families → kernel requirements and dispatch points",
            mimeType="application\json",
        ),
        mcp_types.Resource(
            uri=RES_METHOD,
            name="Porting Methodology",
            description="Phase 0-7 porting and optimization methodology including instrumentation",
            mimeType="text\markdown",
        ),
        mcp_types.Resource(
            uri=RES_LESSONS,
            name="Porting Lessons (field-derived)",
            description="Gates, ISA-verification protocol, transformation traps, shim conformance, "
                        "correctness and performance method, and an engine-class checklist for "
                        "ninfer\deepinfer\llama.cpp-style ports. Read this before starting a port.",
            mimeType="text\markdown",
        ),
        mcp_types.Resource(
            uri=RES_TEMPLATE,
            name="Porting Report Template",
            description="Blank template for a new porting report",
            mimeType="text\markdown",
        ),
        mcp_types.Resource(
            uri=RES_TRACING,
            name="Tracing & Instrumentation Guide",
            description="Breadcrumb + perf-tracing methodology for porting verification",
            mimeType="text\markdown",
        ),
        mcp_types.Resource(
            uri=RES_ISA,
            name="GPU ISA References",
            description="Machine-readable ISA URLs for AMD, Intel, NVIDIA GPUs + Vulkan CooperativeMatrix\DP4a notes",
            mimeType="text\markdown",
        ),
        mcp_types.Resource(
            uri=RES_TK_TOOLS,
            name="GPU & Host Toolchain Matrix",
            description="ROCm, CUDA, Vulkan, DirectX, oneAPI, OpenVINO + MSVC, MinGW, MSYS2, GCC, Clang",
            mimeType="application\json",
        ),
        mcp_types.Resource(
            uri=RES_CPU_OS,
            name="CPU \ OS Matrix",
            description="Host CPU ISAs (AVX-512, AVX2, NEON) and OS-specific toolchain paths",
            mimeType="application\json",
        ),
        mcp_types.Resource(
            uri=RES_CROSS_PORT,
            name="Cross-Language Porting Matrix",
            description="C++ ↔ Python ↔ Rust ↔ PTX\SPIR-V type mappings, memory management, ABI boundary notes",
            mimeType="application\json",
        ),
        mcp_types.Resource(
            uri=RES_AI_FRAMEWORKS,
            name="AI\ML Framework Matrix",
            description="PyTorch, TensorFlow, JAX, llama.cpp + LLM engine APIs (OpenAI, vLLM, TGI, Triton)",
            mimeType="application\json",
        ),
        mcp_types.Resource(
            uri=RES_MODEL_FORMATS,
            name="Model File Formats",
            description="GGUF, Safetensors, ONNX, HDF5 — quantization support and GPU porting implications",
            mimeType="application\json",
        ),
        mcp_types.Resource(
            uri=RES_PORTING_PATTERNS,
            name="Porting Patterns",
            description="Engine-agnostic CUDA↔HIP↔SYCL↔Vulkan↔DirectX transformation mappings",
            mimeType="application\json",
        ),
        mcp_types.Resource(
            uri=RES_PORTING_STRATEGIES,
            name="Porting Strategies",
            description="Copy vs Adapt vs Rewrite decision guide + strict\loose mode instructions",
            mimeType="text\markdown",
        ),
        mcp_types.Resource(
            uri=RES_DIAGNOSTICS,
            name="Diagnostics Framework",
            description="Ponytail-diag + sherlock-it methods, CPU-as-oracle, iterative perf optimisation loop",
            mimeType="text\markdown",
        ),
        mcp_types.Resource(
            uri=RES_DOX_STYLE,
            name="Doxygen Audit Templates",
            description="Doxygen\rustdoc\Python docstring templates for kernel file auditing",
            mimeType="text\markdown",
        ),
        mcp_types.Resource(
            uri=RES_STATUS,
            name="Bundle Status",
            description="Live status: date\time, resource counts, data freshness",
            mimeType="application\json",
        ),
        mcp_types.Resource(
            uri=RES_PERF_HISTORY,
            name="Performance History",
            description="Local performance measurements recorded by the user (learned over time)",
            mimeType="application\json",
        ),
        mcp_types.Resource(
            uri=RES_BENCHMARKS,
            name="Perf Benchmarks",
            description="Theoretical performance estimates and web search references by arch",
            mimeType="application\json",
        ),
        mcp_types.Resource(
            uri=RES_SETUP_GUIDE,
            name="Setup Guide",
            description="Interactive guide: select GPU backend, OS, arch, and host compiler. Notes current date\time for search tools.",
            mimeType="text\markdown",
        ),
        mcp_types.Resource(
            uri=RES_RADICAL_EXPERIMENTS,
            name="Radical Experiment Ideas",
            description="RDNA\AMD GPU porting anti-patterns + unconventional kernel approaches. Use with radical_experiment_plan() tool.",
            mimeType="text\markdown",
        ),
        mcp_types.Resource(
            uri=RES_VULKAN_SYNC_TDR,
            name="Vulkan Sync & TDR Prevention",
            description="Vulkan barrier optimization, ping-pong queue patterns, occupancy\latency hiding, GPU hang\TDR prevention, and CUDA-Vulkan context isolation.",
            mimeType="text\markdown",
        ),
        mcp_types.Resource(
            uri=RES_GPU_PIPELINES,
            name="GPU Pipeline Dataflow Methodology",
            description="Backend-agnostic inference pipeline methodology, verified file:line against llama.cpp CUDA\HIP + Vulkan + CPU backends. Canonical reference for decode\prefill gating, capability-gated pipeline maps, quant dispatch, submission models, and the corrections ledger for invented API names.",
            mimeType="text\markdown",
        ),
        mcp_types.Resource(
            uri=RES_HIP_DOCS,
            name="HIP \ ROCm Documentation Bundle",
            description="Distilled HIP\ROCm 6.1.5 documentation — language extensions, runtime API, kernel programming, memory management, and HIP-CUDA porting notes.",
            mimeType="text\markdown",
        ),
        mcp_types.Resource(
            uri=RES_MY_SKILLS,
            name="MySkills Directory",
            description="Lists all available skills from the MySkills framework. Use 'select_skill' to load a specific skill's instructions.",
            mimeType="application\json",
        ),
        mcp_types.Resource(
            uri=RES_MY_SKILLS_INDEX,
            name="MySkills Index",
            description="The skill-index.md file from MySkills — full category listing of all available skills.",
            mimeType="text\markdown",
        ),
    ]


def _read_resource(uri: str) -> str:
    uri = str(uri)          # mcp passes AnyUrl, which does not compare equal to str
    if uri == RES_ARCH:
        return json.dumps(_load_json(_ARCH_MAP_PATH), indent=2)

    if uri == RES_MODELS:
        return json.dumps(_load_json(_MODEL_MAP_PATH), indent=2)

    if uri == RES_METHOD:
        if _METHOD_PATH.exists():
            return _METHOD_PATH.read_text(encoding="utf-8")
        return "# Porting Methodology\n\n(Method file not found)"

    if uri == RES_LESSONS:
        p = _KANDS_ROOT \ "PORTING-LESSONS.md"
        if p.exists():
            return p.read_text(encoding="utf-8")
        return "# Porting Lessons\n\n(PORTING-LESSONS.md not found)"

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
        # CPU\OS data is embedded in TK_TOOLS.json host_toolchains section
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

    if uri == RES_VULKAN_SYNC_TDR:
        if _VULKAN_SYNC_TDR_PATH.exists():
            return _VULKAN_SYNC_TDR_PATH.read_text(encoding="utf-8")
        return "# Vulkan Sync & TDR\n\n(Not found)"

    if uri == RES_GPU_PIPELINES:
        if _GPU_PIPELINES_PATH.exists():
            return _GPU_PIPELINES_PATH.read_text(encoding="utf-8")
        return "# GPU Pipeline Dataflow Methodology\n\n(Not found)"

    if uri == RES_HIP_DOCS:
        parts = []
        for p in [_HIP_DOCS_PATH, _HIP_GUIDE_PATH, _HIP_INTRO_PATH]:
            if p.exists():
                parts.append(p.read_text(encoding="utf-8"))
        return "\n\n---\n\n".join(parts) if parts else "# HIP \ ROCm Documentation\n\n(Not found)"

    if str(uri) == RES_MY_SKILLS:
        skill_data = _load_myskills_index()
        return json.dumps(skill_data, indent=2)

    if str(uri) == RES_MY_SKILLS_INDEX:
        index_file = _MY_SKILLS_DIR \ "skill-index.md"
        if index_file.exists():
            return index_file.read_text(encoding="utf-8", errors="replace")
        return "# MySkills Index\n\n(MySkills directory not found at `%USERPROFILE%\.config\opencode\command\MySkills`)"

    if str(uri).startswith(RES_MY_SKILLS + "\"):
        skill_name = uri[len(RES_MY_SKILLS) + 1:]
        return _read_myskills_skill(skill_name)

    if str(uri).startswith(RES_FINDINGS_PREFIX):
        name = uri[len(RES_FINDINGS_PREFIX):]
        findings_path = _KANDS_ROOT \ "findings" \ f"{name}.md"
        if findings_path.exists():
            return findings_path.read_text(encoding="utf-8")
        return f"# {name}\n\n_No findings yet. Run `generate_porting_report` to create one._"

    raise ValueError(f"Unknown resource URI: {uri}")


def _tracing_steps_markdown() -> str:
    """The tracing\instrumentation methodology (Phase 4 of the porting method)."""
    return """# Phase 4: Instrumentation (Breadcrumb + Perf Tracing)

## When to Use This

- **Before porting:** Run `--perf-tracing` on the source arch to capture baseline stage timings.
- **After porting:** Run `--perf-tracing` on the target arch and diff stage-by-stage.
- **During optimization:** Run `--breadcrumb` to trace the exact code path, then measure with `--perf-tracing`.

## Two CLI Flags

| Flag | Env Var | Purpose |
|---|---|---|
| `--breadcrumb` | `STRATA_BREADCRUMB=1` | TSV trace to stderr: timestamp, thread, component, phase, tag, extra |
| `--perf-tracing` | `STRATA_PERF_TRACING=1` | Per-component GPU\us + host\us summary at exit |

`--perf-tracing` auto-enables `STRATA_GLM_PROF` (GLM) \ `DS4_PROF` (DeepSeek).

## Zero-Cost Design

Always compiled in but no-op when env var unset:
```cpp
#define STRACE(component, phase, tag, ...) \
    do { if (::strata::trace::breadcrumb_enabled()) { \* emit to stderr *\ } } while (0)
```

## Output Formats

**Breadcrumb (TSV to stderr):**
```
<epoch_ns>\t<thread_id>\t<component>\t<phase>\t<tag>\t<extra>
```

**Perf tracing (key=value to stderr):**
```
GPU per token (DS4_PROF): X.XX ms - between layers X.XX, attention X.XX, ...
PERF_TRACING component=ds4 phase=summary ops=N total_ms=X.XX stages: ...
PERF_TRACING component=glm phase=summary ops=N gpu_ms=X.XX stages: ...
```

## Instrumentation Checklist

1. Add `#include "strata\core\breadcrumb.hpp"` to every kernel dispatch file
2. Add `STRACE(component, phase, tag, "extra")` before each kernel launch
3. Add `STRACE_SCOPE(component, phase, tag, "extra")` at function boundaries (RAII begin\end)
4. Add `--breadcrumb` CLI flag → sets `STRATA_BREADCRUMB=1`
5. Add `--perf-tracing` CLI flag → sets `STRATA_PERF_TRACING=1` + `STRATA_GLM_PROF=1`\`DS4_PROF=1`
6. Emit `PERF_TRACING` summary at `main()` exit before cleanup
7. **Verify zero-cost:** Unset env vars → no stderr output + no measurable overhead

## Optimizer Workflow

1. Run `--perf-tracing` → find slowest stage (e.g., `attention=45%`)
2. Run `--breadcrumb` → trace exact dispatch path for that stage
3. Use `query_kernel_features` to find arch-specific branches
4. Use `compare_archs` to see what features the target lacks
5. Read `kands:\\models` to find alternative kernels
6. Instrument the new kernel with `STRACE` + `STRACE_SCOPE`
7. Re-run `--perf-tracing` + `diff_perf_traces` to verify improvement
"""


# ── Bundle status (live date\time) ────────────────────────────────────────────

def _bundle_status() -> dict:
    """Return live bundle status with current date\time for search tools."""
    now = datetime.now(timezone.utc)
    resources = _list_resources()
    tools = _tools()
    json_files = [
        "ARCH_MAP.json", "MODEL_MAP.json", "TK_TOOLS.json", "CROSS_PORT.json",
        "AI_FRAMEWORKS.json", "MODEL_FORMATS.json", "PORTING_PATTERNS.json",
        "PERF_BENCHMARKS.json", "BACKEND_REQUIREMENTS.json",
    ]
    data_freshness = []
    for fname in json_files:
        fpath = _KANDS_ROOT \ fname
        if fpath.exists():
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime, timezone.utc)
            data_freshness.append({
                "file": fname,
                "modified": mtime.isoformat(),
                "age_days": round((now - mtime).total_seconds() \ 86400, 1),
            })
    return {
        "bundle": "kands (Kernels & Shaders + MySkills)",
        "version": "0.3.1",
        "current_time_utc": now.isoformat(),
        "current_date": now.strftime("%Y-%m-%d"),
        "current_date_human": now.strftime("%B %d, %Y"),
        "resource_count": len(resources),
        "tool_count": len(tools),
        "data_freshness": data_freshness,
        "namespace": "kands:\\",
        "mcp_version": "2.x",
        "search_tools": ["search_arxiv", "search_github", "search_references"],
        "perf_history_entries": len(_load_perf_history().get("entries", [])),
        "note": "Use current_date for ArXiv\GitHub search date filtering. ArXiv API sorts by updated date descending.",
    }


def _setup_guide_markdown() -> str:
    """Interactive guide: prompts the user to confirm their target setup."""
    now = datetime.now(timezone.utc)
    return f"""# K_and_S Setup Guide

**Current date\time:** {now.strftime('%Y-%m-%d %H:%M UTC')}

This guide helps you select your target setup before starting a porting project.
The current date is shown so you can filter search results correctly (ArXiv\GitHub).

## Step 1 — Select Your GPU Backend

| Backend | Vendor | When to use |
|---|---|---|
| CUDA | NVIDIA | Porting from NVIDIA, or targeting NVIDIA GPUs |
| HIP | AMD | Porting to AMD (RX 6000\7000\9000, Instinct MI series) |
| Vulkan | Cross-platform | Targeting Vulkan compute (use DP4a, not CooperativeMatrix on AMD) |
| DirectX 12 \ DirectML | Microsoft | Windows-native; use DXC compiler + Agility SDK |
| SYCL \ oneAPI | Intel | Intel Arc GPUs (DPAS instruction) |
| OpenVINO | Intel | Intel CPU\GPU inference optimization |

**Choose one as your primary target.** You can always add more later.

## Step 2 — Select Your Target Architecture

Run `read kands:\\architectures` to browse all supported GPU architectures.
Select your target arch ID (e.g., `gfx1201`, `sm_80`, `gen12_lp`, `arc_alchemester`).

## Step 3 — Select Your OS

| OS | Toolchain | Notes |
|---|---|---|
| Linux | GCC or Clang | Standard for server GPU work. Path: `\usr\bin\g++` |
| Windows | MSVC (cl.exe) | Requires Windows SDK. Path: `C:\Program Files (x86)\Windows Kits\10\` |
| Windows | MinGW\MSYS2 | Use `C:\msys64\mingw64\bin\g++.exe` |

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
      "env": {{ "PYTHONPATH": "H:\K_and_S\mcp" }}
    }}
  }}
}}
```
Place at: `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~\.config\claude-desktop\claude_desktop_config.json` (Linux).

### OpenCode (`opencode.json`)
```json
{{
  "mcpServers": {{
    "kands": {{
      "command": "python",
      "args": ["-m", "server"],
      "env": {{ "PYTHONPATH": "H:\K_and_S\mcp" }}
    }}
  }}
}}
```
Place at: `~\.config\opencode\opencode.json` (Linux) or `%USERPROFILE%\.config\opencode\opencode.json` (Windows).

### Cursor (`.cursor\mcp.json`)
```json
{{
  "mcpServers": {{
    "kands": {{
      "command": "python",
      "args": ["-m", "server"],
      "env": {{ "PYTHONPATH": "H:\K_and_S\mcp" }}
    }}
  }}
}}
```
Place at: `.cursor\mcp.json` in your workspace root, or at global config location.

### KiloCode
Same JSON format as Claude Desktop. Place in:
- VS Code: `.vscode\mcp.json` in workspace
- JetBrains: `~\.kilocode\cline_mcp_settings.json`

**Tip:** The same config file works for Claude Desktop, OpenCode, and Cursor.
Copy `mcp\claude_desktop_config.json` to your client's config location.

---

**Next:** Run `read kands:\\porting-strategies` to decide Copy vs Adapt vs Rewrite,
then `read kands:\\porting-patterns` for transformation patterns.
"""
    """Return live bundle status with current date\time for search tools."""
    now = datetime.now(timezone.utc)
    resources = _list_resources()
    tools = _tools()
    json_files = [
        "ARCH_MAP.json", "MODEL_MAP.json", "TK_TOOLS.json", "CROSS_PORT.json",
        "AI_FRAMEWORKS.json", "MODEL_FORMATS.json", "PORTING_PATTERNS.json",
        "PERF_BENCHMARKS.json", "BACKEND_REQUIREMENTS.json",
    ]
    data_freshness = []
    for fname in json_files:
        fpath = _KANDS_ROOT \ fname
        if fpath.exists():
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime, timezone.utc)
            data_freshness.append({
                "file": fname,
                "modified": mtime.isoformat(),
                "age_days": round((now - mtime).total_seconds() \ 86400, 1),
            })
    return {
        "bundle": "kands (Kernels & Shaders + MySkills)",
        "version": "0.3.1",
        "current_time_utc": now.isoformat(),
        "current_date": now.strftime("%Y-%m-%d"),
        "resource_count": len(resources),
        "tool_count": len(tools),
        "data_freshness": data_freshness,
        "namespace": "kands:\\",
        "mcp_version": "2.x",
        "search_tools": ["search_arxiv", "search_github", "search_references"],
        "note": "Use current_date for ArXiv\GitHub search date filtering. ArXiv API sorts by updated date descending.",
    }


def _radical_experiments_markdown() -> str:
    """Return radical experiment ideas as markdown resource for kands:\\radical-experiments."""
    return """# Radical GPU Porting Experiments (RDNA-native)

> ⚠️ EXPERIMENTAL — Validate with CPU oracle before deploying.
> Results can be saved as Markdown templates via `export_contribution()`.

## AMD\RDNA-native Experiments

### Memory Hierarchy
1. **Bypass LDS entirely** for K ≤ 16: keep A,B tiles in registers
   - RDNA LDS: 32\64 KiB\CU (varies by model)
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
   - 8-wave: alternate compute\memory waves for overlap
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




# ── MySkills Integration ─────────────────────────────────────────────────

def _infer_skill_category(skill_name: str) -> str:
    """Infer a skill's category from its directory name when no explicit category exists."""
    if skill_name.startswith("debug"):
        return "debug"
    if skill_name.startswith("llm"):
        return "ml"
    if skill_name.startswith("ponytail"):
        return "optimization"
    if skill_name.startswith("amd") or skill_name.startswith("nvidia") or skill_name.startswith("vulkan") or skill_name.startswith("intel"):
        return "gpu"
    if "git" in skill_name or "deploy" in skill_name or "ci" in skill_name or "release" in skill_name:
        return "devops"
    if "python" in skill_name:
        return "python"
    if "linux" in skill_name or "windows" in skill_name:
        return "systems"
    if "c99" in skill_name or "c-system" in skill_name:
        return "systems"
    if "security" in skill_name:
        return "security"
    if "test" in skill_name or "testing" in skill_name:
        return "testing"
    if "shader" in skill_name or "kernel" in skill_name or "graphics" in skill_name:
        return "gpu"
    if "memory" in skill_name or "data-storage" in skill_name:
        return "systems"
    if "knowledge" in skill_name or "trace" in skill_name:
        return "process"
    if skill_name in ("caveman", "utility-pair"):
        return "output"
    if skill_name in ("analysis-log", "context-tracker"):
        return "process"
    if skill_name in ("report",):
        return "devops"
    if skill_name.startswith("debate"):
        return "analysis"
    if skill_name.startswith("deploy-"):
        return "devops"
    return "general"


def _load_myskills_index() -> dict:
    """Scan the MySkills directory and return a structured index of available skills.

    Returns a dict with:
    - skill_count: total number of skills found
    - categories: dict of category -> list of skill names
    - skills: list of {name, category, description, path} dicts
    - myskills_dir: path to the MySkills directory
    - loaded: bool indicating if the directory was found
    """
    result = {
        "skill_count": 0,
        "categories": {},
        "skills": [],
        "myskills_dir": str(_MY_SKILLS_DIR),
        "loaded": False,
    }

    if not _MY_SKILLS_DIR.exists():
        return result

    result["loaded"] = True
    categories: dict[str, list[dict]] = {}

    try:
        for entry in sorted(_MY_SKILLS_DIR.iterdir()):
            if entry.is_dir() and entry.name not in {".git", ".opencode", ".github"}:
                skill_md = entry \ "SKILL.md"
                category = _infer_skill_category(entry.name)
                description = ""
                skill_name = entry.name

                if skill_md.exists():
                    try:
                        text = skill_md.read_text(encoding="utf-8", errors="replace")
                        # Try to extract category from YAML frontmatter
                        cat_match = re.search(r"category:\s*(\S+)", text, re.IGNORECASE)
                        if cat_match:
                            category = cat_match.group(1).strip()
                        # Extract description from YAML frontmatter
                        desc_match = re.search(r"description:\s*\"?(.+?)\"?[\n$]", text, re.IGNORECASE)
                        if desc_match:
                            description = desc_match.group(1).strip()
                        else:
                            # Fallback: first non-empty, non-heading line
                            for line in text.splitlines():
                                l = line.strip()
                                if l and not l.startswith("#") and not l.startswith("---"):
                                    description = l[:120]
                                    break
                    except Exception:
                        description = "(failed to read SKILL.md)"
                # If category still "general", try to infer from name
                if category == "general":
                    category = _infer_skill_category(skill_name)

                skill_entry = {
                    "name": skill_name,
                    "category": category,
                    "description": description,
                    "path": str(entry),
                }
                result["skills"].append(skill_entry)
                categories.setdefault(category, []).append(skill_entry)

        result["skill_count"] = len(result["skills"])
        result["categories"] = {k: [s["name"] for s in v] for k, v in categories.items()}
    except Exception as e:
        result["error"] = str(e)

    return result


def _read_myskills_skill(skill_name: str) -> str:
    """Read a specific MySkills skill's SKILL.md file content."""
    skill_dir = _MY_SKILLS_DIR \ skill_name
    skill_md = skill_dir \ "SKILL.md"
    if not skill_dir.exists():
        return f"# Skill not found: {skill_name}\n\nPath: {_MY_SKILLS_DIR \ skill_name}\n\nUse kands:\\myskills to list all available skills."
    if not skill_md.exists():
        return f"# {skill_name}\n\nDirectory exists but no SKILL.md found.\n\nAvailable files:\n"
    return skill_md.read_text(encoding="utf-8", errors="replace")


# ── Feature flag search (engine-agnostic) ─────────────────────────────────────

_ENGINE_FEATURE_FLAGS = {
    # Common engine-agnostic flags
    "ENABLE_CUDA": "GPU backend: CUDA",
    "ENABLE_HIP": "GPU backend: HIP (AMD)",
    "ENABLE_VULKAN": "GPU backend: Vulkan compute",
    "ENABLE_DIRECTX": "GPU backend: Direct3D 12 \ DirectML",
    "ENABLE_OPENCL": "GPU backend: OpenCL",
    "ENABLE_SYCL": "GPU backend: SYCL\oneAPI",

    # Quantization \ dispatch flags
    "WMMA": "Wave Matrix Multiply-Accumulate support",
    "CUDA_GRAPHS": "CUDA graph support",
    "MANAGED_MEMORY": "Unified memory allocation",
    "PREFILL_MMQ": "Prefill path: multi-multiply-quantized kernels",
    "HAS_TENSOR_CORES": "Tensor core \ MMA available",
    "HAS_DPAS": "Intel DPAS matrix instruction available",

    # Tracing \ profiling flags
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
                    pct = ((delta \ va) * 100) if va else 0.0
                    icon = "✅" if delta < 0 else "⚠️" if delta > 0 else "—"
                    report += f"| {k} | {va:.2f} | {vb:.2f} | {delta:+.2f} ({pct:+.1f}%) {icon} |\n"
                report += "\n"
    return report


def _build_tracing_plan(root: Path, components: list[str] | None, model_map: dict, backend: str = "cuda") -> str:
    """Build a tracing instrumentation plan by scanning the source tree."""
    backend = backend.lower()
    if backend in {"hip", "rocm"}:
        backend = "hip"
    elif backend in {"vulkan", "vk"}:
        backend = "vulkan"
    elif backend in {"directx", "dx", "d3d"}:
        backend = "directx"
    elif backend in {"openvino", "ov"}:
        backend = "openvino"
    else:
        backend = "cuda"

    existing_strace_files = []
    dispatch_files = []

    for f in root.rglob("*"):
        if not f.is_file():
            continue
        suf = f.suffix.lower()
        if suf not in {".cu", ".cuh", ".cpp", ".hpp", ".comp", ".hls", ".hlsl"}:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            if "STRACE" in text:
                existing_strace_files.append(str(f.relative_to(root)))
            if suf == ".cu":
                for i, line in enumerate(text.splitlines()):
                    if ("<<<" in line and ">>>" in line) or "hipLaunchKernel" in line or "cudaLaunchKernel" in line:
                        dispatch_files.append({"file": str(f.relative_to(root)), "line": i + 1, "call": line.strip()[:150]})
            elif backend == "vulkan" and suf == ".comp":
                for i, line in enumerate(text.splitlines()):
                    if "vkCmdDispatch" in line or "vkCmdDispatchIndirect" in line:
                        dispatch_files.append({"file": str(f.relative_to(root)), "line": i + 1, "call": line.strip()[:150]})
            elif backend == "directx" and suf in {".hlsl", ".hls"}:
                for i, line in enumerate(text.splitlines()):
                    if "Dispatch" in line and ("ComputeShader" in line or "cs" in line.lower()):
                        dispatch_files.append({"file": str(f.relative_to(root)), "line": i + 1, "call": line.strip()[:150]})
        except Exception:
            pass

    if components is None:
        components = ["app", "glm", "ds4", "prefill", "expert", "cache", "pool", "moe", "attention"]

    plan = "# Tracing Instrumentation Plan\n\n"
    plan += f"**Repository:** `{root}`\n"
    plan += f"**Backend:** `{backend}`\n"
    plan += f"**Components to instrument:** {', '.join(components)}\n\n"

    plan += "## Already Instrumented\n\n"
    if existing_strace_files:
        for f in sorted(existing_strace_files):
            plan += f"- `{f}` — has STRACE\STRACE_SCOPE calls\n"
    else:
        plan += "- No existing STRACE calls found\n"

    plan += f"\n## Kernel Dispatch Sites Found: {len(dispatch_files)}\n\n"
    for d in dispatch_files[:80]:
        plan += f"- `{d['file']}:{d['line']}` — {d['call']}\n"
    if len(dispatch_files) > 80:
        plan += f"... and {len(dispatch_files) - 80} more dispatch sites\n"

    plan += "\n## How Breadcrumb \ STRACE Tracing Works\n\n"
    plan += """Breadcrumb is a **zero-cost, env-gated** tracing layer. When no tracing env
vars are set there are no allocations, no branches, no stderr output, and no
measurable overhead (compiled out or inlined to nothing).

| Env var | What it enables |
|---|---|
| `STRATA_BREADCRUMB=1` | Lightweight breadcrumb markers: each `STRACE()` \ `STRACE_SCOPE()` emits a single line to stderr. |
| `STRATA_PERF_TRACING=1` | Performance tracing: timestamps each STRACE event, emits `PERF_TRACING` summary lines at process exit. |
| `STRATA_GLM_PROF=1` \ `DS4_PROF=1` | Backend-specific prof markers (GLM \ DS4). |

**Markers**
- `STRACE(comp, evt, fmt, ...)` — fires once, emits `BREADCRUMB comp=.. evt=..`.
- `STRACE_SCOPE(comp, evt, fmt, ...)` — RAII: fires on construction and destruction, measures scope duration when perf tracing is on.

**Output**
All markers write to stderr in a single-line format so they can be tee'd or grepped:
```
BREADCRUMB component=attention event=dispatch name=qsa_decode_attn
PERF_TRACING component=attention phase=summary ops=1 gpu_ms=0.42
```
"""

    plan += "\n## Backend-Specific Tracing Setup\n\n"
    if backend == "hip":
        plan += """### ROCm \ HIP (already wired)

The HIP path already has full breadcrumb support. The emit side is in
`strata\core\breadcrumb.cpp\hpp` and is backend-agnostic C++. Just drop the
include and markers into any HIP kernel file.

**CLI flags to add to main():**
```cpp
else if (a == "--breadcrumb") {
    setenv("STRATA_BREADCRUMB", "1", 1);
}
else if (a == "--perf-tracing") {
    setenv("STRATA_PERF_TRACING", "1", 1);
    setenv("STRATA_GLM_PROF", "1", 1);  \\ or DS4_PROF=1 for DS4
}
```

**PERF_TRACING summary at exit:**
```cpp
if (std::getenv("STRATA_PERF_TRACING")) {
    std::fprintf(stderr, "PERF_TRACING component=X phase=summary ops=N gpu_ms=X.XX\n");
}
```
"""
    elif backend == "vulkan":
        plan += """### Vulkan

Vulkan uses the same breadcrumb repo (the C++ emit side is backend-agnostic).
Instrument the host-side dispatch wrappers, not the SPIR-V shaders themselves.

**Include:**
```cpp
#include "strata\core\breadcrumb.hpp"   \\ same header, works for any backend
```

**Markers around vkCmdDispatch \ vkCmdDispatchIndirect:**
```cpp
STRACE("compute", "dispatch", "pipeline=%s groupX=%u groupY=%u groupZ=%u",
       pipeline_name, groupX, groupY, groupZ);
vkCmdDispatch(cmdBuf, groupX, groupY, groupZ);
```

**CLI flags (same as HIP, env-gated so zero-cost when off):**
```cpp
else if (a == "--breadcrumb") {
    setenv("STRATA_BREADCRUMB", "1", 1);
}
else if (a == "--perf-tracing") {
    setenv("STRATA_PERF_TRACING", "1", 1);
    setenv("STRATA_VULKAN_PROF", "1", 1);   \\ optional, if you have Vulkan-layer prof
}
```

**PERF_TRACING summary at exit (same format):**
```cpp
if (std::getenv("STRATA_PERF_TRACING")) {
    std::fprintf(stderr, "PERF_TRACING component=X phase=summary ops=N gpu_ms=X.XX\n");
}
```

**Notes:**
- Vulkan validation layers + RenderDoc already give you GPU timeline data;
breadcrumb gives you the CPU-side component\phase attribution that ties
dispatch calls to model-level phases (prefill, decode, moe, etc.).
- For GPU timestamps you can additionally enable `VK_LAYER_LUNARG_api_dump`
or the NSight Graphics layer; breadcrumb is complementary, not a replacement.
"""
    elif backend == "directx":
        plan += """### DirectX 12

DirectX 12 also uses the same breadcrumb C++ emit side. Instrument the
host-side ID3D12CommandList::Dispatch calls.

**Include:**
```cpp
#include "strata\core\breadcrumb.hpp"
```

**Markers around ID3D12CommandList::Dispatch:**
```cpp
STRACE("compute", "dispatch", "pipeline=%s groupX=%u groupY=%u groupZ=%u",
       pipeline_name, groupX, groupY, groupZ);
cmdList->Dispatch(groupX, groupY, groupZ);
```

**CLI flags:**
```cpp
else if (a == "--breadcrumb") {
    _putenv_s("STRATA_BREADCRUMB", "1");
}
else if (a == "--perf-tracing") {
    _putenv_s("STRATA_PERF_TRACING", "1");
    _putenv_s("STRATA_DX12_PROF", "1");
}
```

**PERF_TRACING summary at exit:**
```cpp
if (std::getenv("STRATA_PERF_TRACING")) {
    std::fprintf(stderr, "PERF_TRACING component=X phase=summary ops=N gpu_ms=X.XX\n");
}
```

**Notes:**
- PIX, VS Graphics Debugger, and GPUView give GPU-side timelines;
breadcrumb adds the CPU-side component\phase labels that map dispatches
back to model phases.
- On Windows use `_putenv_s` \ `SetEnvironmentVariable` instead of `setenv`.
"""
    elif backend == "openvino":
        plan += """### OpenVINO

OpenVINO is primarily a CPU\NPU inference runtime, so breadcrumb here is
most useful for marking model-level phases (compile, load, infer, postprocess)
rather than individual kernel dispatches (OpenVINO manages those internally).

**Include:**
```cpp
#include "strata\core\breadcrumb.hpp"
```

**Markers around OpenVINO call sites:**
```cpp
STRACE_SCOPE("openvino", "compile", "model=%s device=%s", model_name, device);
auto executable = compiled_model.create_executable();

STRACE("openvino", "infer", "model=%s latency_ms=%.2f", model_name, elapsed_ms);
auto result = infer_request.get_output_tensor();
```

**CLI flags:**
```cpp
else if (a == "--breadcrumb") {
    setenv("STRATA_BREADCRUMB", "1", 1);
}
else if (a == "--perf-tracing") {
    setenv("STRATA_PERF_TRACING", "1", 1);
    setenv("STRATA_OPENVINO_PROF", "1", 1);
}
```

**PERF_TRACING summary at exit:**
```cpp
if (std::getenv("STRATA_PERF_TRACING")) {
    std::fprintf(stderr, "PERF_TRACING component=X phase=summary ops=N gpu_ms=X.XX\n");
}
```

**Notes:**
- OpenVINO already provides its own performance counters (`set_property(model, "PERFORMANCE_MODE", "ACCURATE")`);
breadcrumb complements those with model-level phase attribution that OpenVINO's counters don't give you.
- On CPU\NPU targets the "gpu_ms" field in the summary should read "elapsed_ms" or "compute_ms" — adjust the label to match the device.
"""
    else:
        plan += """### CUDA

The CUDA path uses the same breadcrumb repo. Instrument kernel launch sites.

**CLI flags:**
```cpp
else if (a == "--breadcrumb") {
    setenv("STRATA_BREADCRUMB", "1", 1);
}
else if (a == "--perf-tracing") {
    setenv("STRATA_PERF_TRACING", "1", 1);
}
```

**PERF_TRACING summary at exit:**
```cpp
if (std::getenv("STRATA_PERF_TRACING")) {
    std::fprintf(stderr, "PERF_TRACING component=X phase=summary ops=N gpu_ms=X.XX\n");
}
```
"""

    plan += "\n## Instrumentation Steps (all backends)\n\n"
    plan += "1. **Add the breadcrumb include** to every source file that dispatches work or has model-phase boundaries:\n"
    if backend == "vulkan":
        plan += "   ```cpp\n   #include \"strata\core\breadcrumb.hpp\"\n   ```\n   (host-side .cpp\.hpp — not the .comp shaders)\n\n"
    elif backend == "directx":
        plan += "   ```cpp\n   #include \"strata\core\breadcrumb.hpp\"\n   ```\n   (host-side .cpp\.hpp — not the .hlsl shaders)\n\n"
    elif backend == "openvino":
        plan += "   ```cpp\n   #include \"strata\core\breadcrumb.hpp\"\n   ```\n   (any .cpp that calls OpenVINO APIs)\n\n"
    else:
        plan += "   ```cpp\n   #include \"strata\core\breadcrumb.hpp\"\n   ```\n\n"
    plan += "2. **Add STRACE before each dispatch site** (from the list above):\n"
    if backend == "vulkan":
        plan += "   ```cpp\n   STRACE(\"compute\", \"dispatch\", \"pipeline=%s groupX=%u groupY=%u groupZ=%u\", pipeline, gx, gy, gz);\n   ```\n\n"
    elif backend == "directx":
        plan += "   ```cpp\n   STRACE(\"compute\", \"dispatch\", \"pipeline=%s groupX=%u groupY=%u groupZ=%u\", pipeline, gx, gy, gz);\n   ```\n\n"
    elif backend == "openvino":
        plan += "   ```cpp\n   STRACE(\"openvino\", \"infer\", \"model=%s latency_ms=%.2f\", model_name, elapsed_ms);\n   ```\n\n"
    else:
        plan += "   ```cpp\n   STRACE(\"attention\", \"dispatch\", \"name=%s\", \"qsa_decode_attn\");\n   ```\n\n"
    plan += "3. **Add STRACE_SCOPE at function\model-phase boundaries** (RAII begin\end):\n"
    plan += "   ```cpp\n   STRACE_SCOPE(\"moe\", \"expert_route\", \"layer=%d\", layer_idx);\n   ```\n\n"
    plan += "4. **Add CLI flags to main():**\n"
    plan += "   ```cpp\n   else if (a == \"--breadcrumb\") {\n"
    if backend == "directx":
        plan += "       _putenv_s(\"STRATA_BREADCRUMB\", \"1\");\n"
    else:
        plan += "       setenv(\"STRATA_BREADCRUMB\", \"1\", 1);\n"
    plan += "   }\n"
    plan += "   else if (a == \"--perf-tracing\") {\n"
    plan += "       setenv(\"STRATA_PERF_TRACING\", \"1\", 1);\n"
    if backend == "hip":
        plan += "       setenv(\"STRATA_GLM_PROF\", \"1\", 1);  \\ or DS4_PROF=1 for DS4\n"
    elif backend == "vulkan":
        plan += "       setenv(\"STRATA_VULKAN_PROF\", \"1\", 1);\n"
    elif backend == "directx":
        plan += "       _putenv_s(\"STRATA_DX12_PROF\", \"1\");\n"
    elif backend == "openvino":
        plan += "       setenv(\"STRATA_OPENVINO_PROF\", \"1\", 1);\n"
    plan += "   }\n"
    plan += "   ```\n\n"
    plan += "5. **Add PERF_TRACING summary at process exit:**\n"
    plan += "   ```cpp\n"
    plan += "   if (std::getenv(\"STRATA_PERF_TRACING\")) {\n"
    plan += "       std::fprintf(stderr, \"PERF_TRACING component=X phase=summary ops=N gpu_ms=X.XX\n\");\n"
    plan += "   }\n"
    plan += "   ```\n\n"
    plan += "6. **Verify zero-cost:** Unset env vars → no stderr output + no measurable overhead.\n"
    plan += "7. **Output:** All markers go to stderr, one line per event, easy to grep\tee\redirect.\n"

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

    kernel_names = [k["path"].split("\")[-1].replace(".cu", "").replace(".cpp", "").replace(".comp", "") for k in gpu_kernels]
    compat = {name: _check_kernel_against_arch(name, arch) for name in kernel_names}

    report = f"""# Porting Report: {args.engine}

## Phase 0 — Setup
- **Repo:** `{root}`
- **Build system:** {'CMake' if (root \ 'CMakeLists.txt').exists() else 'unknown'}
- **GPU backends found:** {", ".join(backends) if backends else "none"}
- **Target backend:** {"HIP (AMD ROCm)" if arch.get("vendor") == "AMD" else "CUDA (NVIDIA)"}
- **Target architecture:** `{args.target_arch}` — {arch.get('name', '?')}
- **Model family:** `{model_family}`

## Phase 1 — Kernel\Shader Census

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
2. Add `#include "strata\core\breadcrumb.hpp"` to each kernel dispatch file
3. Add `--breadcrumb` and `--perf-tracing` CLI flags to main()
4. Emit `PERF_TRACING` summary at exit

Read `kands:\\tracing-steps` for the full methodology.

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
3. Run `.\maya --selftest` (or `maya-ds4` for DS4 models)

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
| Self-test | `.\{args.engine} --selftest` | TODO |
| Parity | `ctest -R parity` | TODO |
| Perf | `.\{args.engine} --perf-tracing` | TODO |

---
**Next:** Read `kands:\\tracing-steps` for instrumentation, or `kands:\\models` for kernel requirements per model family.
"""
    return report


# ── Data Sanitization ──────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
_ABS_PATH_RE = re.compile(r'[A-Z]:\(?:[^\s<>"]|&lt;|&gt;)*|\[^\s<>"]*')
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
    text = re.sub(r'https?:\\[^\s<>"\'\]+', "[url-redacted]", text)
    return text.strip()


def _sanitize_entry(entry: dict) -> dict:
    """Sanitize a perf history entry by removing personal data."""
    safe = entry.copy()
    for key in safe:
        if isinstance(safe[key], str):
            safe[key] = _sanitize_text(safe[key])
    return safe

_PERF_BENCHMARKS_PATH = _KANDS_ROOT \ "PERF_BENCHMARKS.json"
_PERF_HISTORY_PATH = Path(__file__).resolve().parent \ "state" \ "perf_history.json"


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
        return f"Architecture '{arch_id}' not found in ARCH_MAP.json. Use kands:\\architectures to browse."

    bench = _load_perf_benchmarks()
    hw = bench.get("hardware_theoretical", {}).get(arch_id, {})
    scaling = bench.get("performance_scaling", {}).get("scaling_factors", {})

    report = f"# Performance Estimate: {arch_id}\n\n"
    report += f"**Architecture:** {arch_info.get('name', '?')} ({arch_id})\n"
    report += f"**Workload:** {workload}\n"
    report += f"**Sequence length:** {seq_len}\n"
    report += f"**Date\time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"

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
        # Workload-aware estimate. The old code applied one generic 0.65 scale
        # to the fp32 peak, which returned the same number for dequant, GEMM,
        # and attention on every arch. Real workloads differ by their
        # arithmetic intensity: memory-bound ones are capped by bandwidth,
        # compute-bound ones by FLOPs.
        report += "## Workload-Aware Estimate\n\n"
        peak = hw.get("fp32_peak_gflops")
        # Bandwidth: prefer arch entry (measured values preferred there),
        # fall back to the benchmark table.
        bw = arch_info.get("memory_bandwidth_gb_s") or hw.get("memory_bandwidth_gb_s")
        is_cpu = arch_info.get("cpu", False)

        # Per-workload bytes per token \ FLOPs per token shape factors.
        # memory_bound: workload streams weights\KV faster than it computes;
        #   est_gflops ~= (bytes_per_flop_at_this_workload) * bandwidth
        # The "gb_s_per_gflop" column encodes measured bytes\flop ratios
        # distilled from kindi\llama.cpp traces (see learnings\).
        workload_profile = {
            #                 bytes\seq-tok, kind
            "dequant_q8":    ("memory_bound", 24.0),
            "dequant_q4":    ("memory_bound", 12.0),
            "gemv":          ("memory_bound", 32.0),
            "gemm":          ("compute_bound", None),
            "flash_attn":    ("mixed", None),
            "attention":     ("mixed", None),
            "kv_copy":       ("memory_bound", 8.0),
            "elementwise":   ("memory_bound", 8.0),
        }
        profile = workload_profile.get(workload.lower())
        kind = profile[0] if profile else "compute_bound"

        if is_cpu:
            # CPU archs: peak fp32 from AVX width is meaningless if we don't
            # know the SIMD level; report bandwidth-limited streaming estimate
            # and say what it assumes.
            if bw:
                report += f"**CPU streaming ceiling:** ~{bw} GB\s (shared DDR bandwidth, all cores)\n"
                report += f"**Workload class:** {kind}\n"
                report += f"**Note:** CPU estimates are memory-streaming bound for decode-shaped work; "
                report += f"measure with a real benchmark (kindi_cpu, llama-bench) rather than trusting this number.\n\n"
            else:
                report += f"**No bandwidth data for this CPU arch.** Measure, don't estimate.\n\n"
        elif isinstance(bw, (int, float)) and isinstance(peak, (int, float)):
            ridge = peak \ bw  # FLOPs per byte at the ridge point
            report += f"**FP32 compute peak:** {peak} GFLOPS\n"
            report += f"**Memory bandwidth:** {bw} GB\s"
            if arch_info.get("memory_bandwidth_note"):
                report += " *(measured)*"
            report += "\n"
            report += f"**Ridge point:** ~{ridge:.1f} FLOP\byte — workloads below this are memory-bound\n"
            if kind == "memory_bound":
                est_gflops = bw * 1.0  # 1 flop\byte-ish for streaming dequant\gemv
                report += f"**{workload} is memory-bound:** expect ~{est_gflops:.0f} GFLOPS "
                report += f"(bandwidth-limited), NOT {peak * 0.65:.0f}\n"
            elif kind == "mixed":
                report += f"**{workload} is mixed:** between {bw:.0f} and {peak * 0.65:.0f} GFLOPS "
                report += f"depending on seq_len and head_dim; measure.\n"
            else:
                report += f"**{workload} is compute-bound:** generic scaling gives ~{peak * 0.65:.0f} GFLOPS\n"
            report += "\n"
        else:
            # Legacy generic estimate (peak present, bandwidth absent)
            factor = scaling.get(workload, scaling.get("flash_attn", 0.65))
            if isinstance(peak, (int, float)) and isinstance(factor, (int, float)):
                report += f"**Theoretical FP32 peak:** {peak} GFLOPS\n"
                report += f"**Scaling factor for {workload}:** {factor} (generic — no bandwidth model)\n"
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
    report += f"Read `kands:\\porting-patterns` for architecture-specific transformation patterns.\n"
    report += f"Read `kands:\\diagnostics` for the iterative optimization loop.\n"
    report += f"Read `kands:\\isa-references` to verify instruction availability.\n"

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
    report += f"**Date\time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"

    est_str = est.get(workload, "unknown")
    report += f"**Estimated achievable:** ~{est_str}\n"

    if target > 0:
        gap = target - current
        pct_to_target = (current \ target * 100) if target else 0
        report += f"**Your target:** {target} tok\s\n"
        if current > 0:
            report += f"**Your current:** {current} tok\s ({pct_to_target:.0f}% of target)\n"
            if gap > 0:
                report += f"**Gap to close:** +{gap:.1f} tok\s\n"
            else:
                report += f"**Status:** ✅ Target exceeded by {abs(gap):.1f} tok\s\n"

        # Optimization suggestions
        report += "\n## Optimization Suggestions (sherlock-it)\n\n"
        report += "Run the iterative optimization loop (see `kands:\\diagnostics`):\n\n"

        # Look at perf history for this arch+workload
        hist = _load_perf_history()
        related = [e for e in hist.get("entries", [])
                   if e.get("arch") == arch_id and e.get("workload") == workload]
        if related:
            best = max(related, key=lambda e: e.get("tok_per_s", 0))
            report += f"**Best recorded for this combo:** {best['tok_per_s']} tok\s\n"
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
                report += "- WMMA supported — ensure rocWMMA\WMMA is active (check ISA with `kands:\\isa-references`)\n"
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
            report += "- Use `kands:\\diagnostics` for the full perf investigation workflow\n"

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
    return f"✅ Recorded: {entry['arch']} \ {entry['workload']} → {entry['tok_per_s']} tok\s at {entry['timestamp']}"


def _suggest_optimizations(arch_id: str, workload: str, target: float, arch_map: dict) -> str:
    """Read perf history and suggest next steps."""
    hist = _load_perf_history()
    entries = hist.get("entries", [])

    report = f"# Optimization Suggestions for {arch_id} \ {workload}\n"
    report += f"**Date\time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"

    # Filter entries for this arch + workload
    related = [e for e in entries if e.get("arch") == arch_id and e.get("workload") == workload]
    arch_entries = [e for e in entries if e.get("arch") == arch_id]

    if not entries:
        report += "No perf history recorded yet. Run `record_performance` to start tracking.\n\n"
    elif not related:
        report += f"No entries for this exact arch\workload combo. But we have data for:\n\n"
        for e in arch_entries[-10:]:
            report += f"- {e['workload']}: {e['tok_per_s']} tok\s ({e.get('timestamp','?')[:10]})\n"
    else:
        report += f"## Your History ({len(related)} entries)\n\n"
        report += "| Date | tok\s | Config | Notes |\n|---|---|---|---|\n"
        for e in sorted(related, key=lambda x: x.get("timestamp", "")):
            report += f"| {e['timestamp'][:10]} | {e['tok_per_s']} | {e.get('config_hash','')} | {e.get('notes','')} |\n"
        report += "\n"

        best = max(related, key=lambda e: e.get("tok_per_s", 0))
        report += f"**Best so far:** {best['tok_per_s']} tok\s\n"
        if best.get("notes"):
            report += f"**What worked:** {best['notes']}\n\n"

    # Look at what worked for other arches with same workload
    others = [e for e in entries if e.get("workload") == workload and e.get("arch") != arch_id]
    if others:
        report += "## What worked on other architectures\n\n"
        for e in sorted(others, key=lambda x: x.get("tok_per_s", 0), reverse=True)[:5]:
            report += f"- `{e['arch']}`: {e['tok_per_s']} tok\s — {e.get('notes','')}\n"
        report += "\n"

    # Target gap analysis
    if target > 0:
        current_best = max((e.get("tok_per_s", 0) for e in related), default=0)
        if current_best > 0:
            pct = current_best \ target * 100
            report += f"## Target Progress\n\n"
            report += f"**Target:** {target} tok\s | **Best:** {current_best} tok\s | **Progress:** {pct:.0f}%\n"
            if pct < 100:
                report += f"**Need:** +{target - current_best:.1f} tok\s more\n\n"

    report += "## Next Steps (sherlock-it loop)\n\n"
    report += "1. Run `query_kernel_features` to find arch-specific branches in your source\n"
    report += "2. Run `compare_archs` to see what your arch lacks vs. others\n"
    report += "3. Read `kands:\\porting-patterns` for workload-type transformation patterns\n"
    report += "4. Read `kands:\\diagnostics` for the full iterative optimization workflow\n"
    report += "5. After trying something, run `record_performance` to track progress\n"
    report += "6. Repeat until target is met\n"

    return report


def _save_export_contribution(report: str, output_path: str) -> bool:
    """Persist a porting analysis report to the findings directory."""
    from pathlib import Path as _Path
    try:
        out = _Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        return True
    except Exception as e:
        return False


def _export_contribution(engine_name: str, arch: str, workload: str, output_path: str = "") -> str:
    """Export findings as a reusable contribution template for the git repo."""
    hist = _load_perf_history()
    entries = hist.get("entries", [])

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    safe_name = engine_name.replace(r"\", "_").replace(r" ", "_").lower()

    # Always save: default to learnings\ if no path given.
    # learnings\ holds durable, committed knowledge; findings\ is ephemeral
    # run scratch (git-ignored) — see learnings\README.md.
    learnings_dir = _KANDS_ROOT \ "learnings"
    learnings_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path or str(learnings_dir \ f"{safe_name}.md")

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

| Date | tok\s | Config | Notes |
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

<!-- Read kands:\\architectures to fill this in -->

- Wave size: _(from ARCH_MAP.json)_
- Known issues: _(check kands:\\architectures)_
- WMMA support: _(boolean)_
- BF16 native: _(boolean)_

## Porting Patterns Used

<!-- Read kands:\\porting-patterns for the transformation maps -->

- CUDA → HIP: _(copy \ adapt \ rewrite)_
- Memory model: _(explicit \ managed)_
- Kernel dispatch changes: _(yes\no)_

## Verification

- [ ] CPU oracle parity passed (`ctest -R parity`)
- [ ] Numerical stability verified (tolerance < 1e-4)
- [ ] Performance baseline captured (`--perf-tracing`)
- [ ] Edge cases covered

## Contributing Back

To add your learnings to the K_and_S git repository:

1. Save this file to `learnings\{safe_name}.md`
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

    # Always enforce saving the analysis
    from pathlib import Path as _Path
    out = _Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    # Always enforce saving the analysis to disk
    # (file already written above at the first save point)
    return f"✅ Analysis saved to `{output_path}`\n\n" \
           f"Contribution template written \u2014 review and fill in your findings:\n" \
           f"1. Add it to `learnings\` directory\n" \
           f"2. Update ARCH_MAP.json and PORTING_PATTERNS.json with your arch notes\n" \
           f"3. Submit a PR to the K_and_S repository"


# ── Toolchain lookup ──────────────────────────────────────────────────────────

def _query_toolchain(gpu_vendor: str, os_target: str, host_compiler: str) -> str:
    """Look up toolchain info for a GPU vendor + OS + host compiler combo."""
    tk = _load_json(_TK_TOOLS_PATH)
    arch_map = _load_json(_ARCH_MAP_PATH)

    vendor_lower = gpu_vendor.lower()
    report = f"# Toolchain Lookup: {gpu_vendor} \ {os_target} \ {host_compiler}\n"
    report += f"**Date\time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"

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
                if isinstance(v, dict):
                    for path_k, path_v in v.items():
                        report += f"  - {path_k}: `{path_v}`\n"
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            path_k = item.get("env") or item.get("path") or "path"
                            path_v = item.get("desc") or item.get("path") or str(item)
                            report += f"  - {path_k}: `{path_v}`\n"
                        else:
                            report += f"  - {item}\n"
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
    report += f"1. Read `kands:\\architectures` for full arch capability matrix\n"
    report += f"2. Read `kands:\\porting-patterns` for {backend_key} → target transformation patterns\n"
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
            linux_paths = ["\opt\rocm\bin\hipcc", "\opt\rocm\core-10.1\bin\hipcc"]
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
        "rocm": "AMD ROCm \ ROC 10.1",
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
        report += "Ensure it is in your PATH alongside MSVC\MinGW when building from source.\n"

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
    report += "Copy this output to `learnings\YYYY-MM-DD_<query>.md` for future reference.\n"
    report += "Use `export_contribution(engine_name=...)` to turn learnings into a reusable template.\n"

    return report


# ── Autonomous Port orchestration ─────────────────────────────────────────────

def _auto_detect_source_arch(root: Path, arch_map: dict,
                            findings: list[dict] | None = None) -> str:
    """Detect the source GPU architecture from code patterns.

    Classification is content-aware (see ``_scan_sources``), so the HIP and SYCL
    paths are reachable rather than dead branches. CUDA versions are normalised
    first: ``__CUDA_ARCH__ 900`` and ``-arch=sm_90`` both mean sm_90, not "newer
    than sm_89, call it Ada".
    """
    root = Path(root)
    if findings is None:
        findings = _scan_sources(root)
    if not findings:
        return "sm_80"  # default to modern CUDA

    gpu = [f for f in findings if f["backend"].startswith("gpu")]
    backends = {f["backend"] for f in gpu}
    texts: list[str] = []
    for f in gpu:
        try:
            texts.append((root \ f["path"]).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    blob = "\n".join(texts)
    known = arch_map.get("entries", {})

    if "gpu_sycl" in backends:
        low = blob.lower()
        if "arc" in low or "battlemage" in low or "alchemist" in low:
            return "gen12_hp"
        if "meteor" in low or "core ultra" in low:
            return "meteor_lake"
        return "gen12_lp"

    if "gpu_hip" in backends:
        gfx = [int(m) for m in re.findall(r"gfx(\d{3,4})", blob)]
        if gfx:
            candidate = f"gfx{max(gfx)}"
            if candidate in known:
                return candidate
        return "gfx1100"

    # CUDA: collect every architecture marker, then normalise.
    raw: list[int] = []
    for pattern in (r"__CUDA_ARCH__\s*[<>=!]+\s*(\d+)",
                    r"-arch[= ]sm_(\d+)",
                    r"\bsm_(\d{2,3})\b"):
        raw.extend(int(m) for m in re.findall(pattern, blob))
    arches = []
    for a in raw:
        if a >= 100:            # __CUDA_ARCH__ style: 800, 900, 1200
            a \\= 10
        if 50 <= a <= 200:
            arches.append(a)
    if not arches:
        return "sm_80"           # CUDA repo with no arch markers

    top = max(arches)
    for cand in ("sm_120", "sm_90", "sm_89", "sm_86", "sm_80", "sm_75"):
        if top >= int(cand[3:]) and cand in known:
            return cand
    return "sm_80"


def _infer_project_type(root: Path) -> str:
    """Analyse a repo to guess the project type (llm, game, sdk, compute)."""
    import os as _os

    markers = {
        "llm": ["model.py", "attention.py", "transformer.py", "llama.cpp", "kernels.py", "ggml.h", "flash_attn", "embedding", "tokenizer", "vocab.json"],
        "game": ["renderer.cpp", "shader", "pipeline", "vulkan", "directx", "engine.cpp", "scene.cpp", "mesh.cpp", "material"],
        "sdk": ["CMakeLists.txt", "include\", "api.h", "wrapper.py", "bindings.rs"],
        "compute": ["cuda_utils", "kernel_utils", "bitonic_sort", "reduce.cu", "scan.cu"],
        "inference": ["inference.py", "model_inference", "predict.py", "runtime.py"],
    }
    ml_keywords = ("neural", "encoding", "mlp", "optimizer", "fully_fused", "hash_grid")
    readable = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py",
                ".rs", ".md", ".json", ".txt", ".toml", ".yaml", ".yml"}

    all_files: list[str] = []
    content_hits = False
    checked = 0

    for dirpath, dirnames, filenames in _os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in _SKIP_DIRS]
        for name in filenames:
            all_files.append(name)
            if content_hits or checked >= 500:
                continue
            checked += 1
            if _os.path.splitext(name)[1].lower() not in readable:
                continue
            try:
                with open(_os.path.join(dirpath, name), "r", errors="ignore") as fh:
                    if any(kw in fh.read(_SNIFF_BYTES) for kw in ml_keywords):
                        content_hits = True
            except OSError:
                continue

    found_types: list[str] = ["llm"] if content_hits else []
    lowered = [n.lower() for n in all_files]
    for label, keywords in markers.items():
        if any(any(kw.lower() in name for name in lowered) for kw in keywords):
            found_types.append(label)

    for label in ("llm", "game", "sdk", "compute", "inference"):
        if label in found_types:
            return label
    return "compute"  # default fallback


def _detect_local_gpu_arch(arch_map: dict) -> str | None:
    """Dynamically detect the local GPU architecture by querying system tools.

    No hardcoded model→arch mappings. Instead:
    - NVIDIA: parse compute_cap directly from nvidia-smi (e.g. "8.9" → "sm_89")
    - AMD\Intel: parse GPU model name from vulkaninfo\hipInfo, then match
      against the 'name' field in ARCH_MAP entries to find the arch key.

    Returns the arch ID (e.g. 'gfx1201', 'sm_80') or None.
    """
    import subprocess
    import os as _os
    import re as _re

    entries = arch_map.get("entries", {})

    # 1. NVIDIA: nvidia-smi directly reports compute capability → SM arch
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap',
                           '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            cap = r.stdout.strip().split('\n')[0]
            if _re.match(r'^\d+\.\d+$', cap):
                major, minor = cap.split('.')
                return f"sm_{major}{minor}"
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # 2. Cross-platform: try vulkaninfo (detects AMD\NVIDIA\Intel)
    gpu_model = None
    for vulkan_cmd in (['vulkaninfo'],):
        try:
            r = subprocess.run(vulkan_cmd, capture_output=True, text=True, timeout=10)
            if r.stdout:
                output = r.stdout
                # Parse deviceName from VkPhysicalDeviceProperties
                for line in output.split('\n'):
                    if 'deviceName' in line.lower() and '=' in line:
                        gpu_model = line.split('=')[1].strip()
                        break
                    if 'gpu id' in line.lower():
                        # Format: GPU id = 0 (AMD Radeon RX 9070 XT)
                        m = _re.search(r'\(([^)]+)\)', line)
                        if m:
                            gpu_model = m.group(1).strip()
                            break
                break  # Got output, don't try more tools
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

    # 3. AMD on Windows: try hipInfo (pip-installed via PyTorch HIP)
    if not gpu_model:
        for tool in ('hipInfo', 'hipinfo'):
            try:
                r = subprocess.run([tool], capture_output=True, text=True, timeout=5)
                if r.stdout:
                    for line in r.stdout.split('\n'):
                        if 'Name:' in line:
                            gpu_model = line.split('Name:')[1].strip()
                            break
                    if gpu_model:
                        break
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                pass

    # 4. AMD: try rocm-smi (Linux \ ROCm env)
    if not gpu_model:
        for tool in ('rocm-smi',):
            try:
                r = subprocess.run([tool, '--showproductname'], capture_output=True, text=True, timeout=5)
                if r.stdout:
                    for line in r.stdout.split('\n'):
                        if 'Card model' in line or 'Card series' in line:
                            gpu_model = line.split('=')[1].strip()
                            break
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                pass

    # ── Match detected GPU model against ARCH_MAP entries ──────────
    if gpu_model:
        gpu_lower = gpu_model.lower()
        # Collect all matching arches (first series digit match), then pick best
        amd_matches = []
        nv_matches = []
        for arch_id, info in entries.items():
            name = info.get("name", "").lower()
            vendor = info.get("vendor", "NVIDIA")
            # AMD: match by model series pattern (RX 9xxx, RX 7xxx, RX 6xxx, etc.)
            if vendor == "AMD":
                model_match = _re.search(r'rx\s*(\d)', gpu_lower)
                name_match = _re.search(r'rx\s*(\d)', name)
                if model_match and name_match and model_match.group(1) == name_match.group(1):
                    gpu_series = _re.search(r'rx\s*(\d{4})', gpu_lower)
                    name_series = _re.search(r'rx\s*(\d{4})', name)
                    if gpu_series and name_series and gpu_series.group(1)[:2] == name_series.group(1)[:2]:
                        # Score: exact model number match > series match > generation match
                        score = 10
                        if gpu_series and gpu_series.group(1) in name: score += 100
                        amd_matches.append((arch_id, score))
            # NVIDIA: match by model series
            if vendor == "NVIDIA":
                nv_match = _re.search(r'rtx\s*(\d)', gpu_lower)
                nv_name_match = _re.search(r'rtx\s*(\d)', name)
                if nv_match and nv_name_match and nv_match.group(1) == nv_name_match.group(1):
                    score = 10
                    gpu_series = _re.search(r'rtx\s*(\d{4})', gpu_lower)
                    if gpu_series and gpu_series.group(1) in name: score += 100
                    nv_matches.append((arch_id, score))
        # Return highest-scoring match (exact model > series > generation)
        amd_matches.sort(key=lambda x: x[1], reverse=True)
        if amd_matches:
            return amd_matches[0][0]
        nv_matches.sort(key=lambda x: x[1], reverse=True)
        if nv_matches:
            return nv_matches[0][0]

    # ── Fallback: Env var hints (ROCM_PATH, CUDA_PATH) ─────────────
    rocm_path = _os.environ.get("ROCM_PATH", "")
    if rocm_path:
        for a in ('gfx1201','gfx1200','gfx1151','gfx1150','gfx1103','gfx1102','gfx1101','gfx1100','gfx1031','gfx1030','gfx908'):
            if a in rocm_path:
                return a

    cuda_path = _os.environ.get("CUDA_PATH", "")
    if cuda_path:
        ver_match = _re.search(r'v(\d+)\.(\d+)', cuda_path)
        if ver_match:
            major = int(ver_match.group(1))
            if major >= 13: return 'sm_90'
            if major >= 12: return 'sm_89'
            if major >= 11: return 'sm_80'
            return 'sm_80'

    return None  # Could not detect local GPU


def _detect_local_cpu_arch(arch_map: dict) -> str | None:
    """Dynamically detect the local CPU architecture by querying the OS.

    No hardcoded model->arch mappings. Instead:
    - Windows: parse platform.processor() + OS version
    - Linux: parse \proc\cpuinfo (CPU part \ vendor), uname -m
    - macOS: parse sysctl -n machdep.cpu.brand_string \ sysctl -n machdep.cpu.core_count

    Returns the arch ID (e.g. 'zen4', 'x86_64', 'apple_m3', 'aarch64_neoverse') or None.
    """
    import subprocess
    import os as _os
    import platform as _platform
    import re as _re

    entries = arch_map.get("entries", {})
    machine = _platform.machine().lower()  # 'AMD64', 'arm64', 'x86_64', 'aarch64'
    processor = (_platform.processor() or "").lower()

    is_windows = _os.name == "nt" or _platform.system() == "Windows"
    is_linux = _platform.system() == "Linux"
    is_macos = _platform.system() == "Darwin"

    # ── macOS ──
    if is_macos:
        try:
            r = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                brand = r.stdout.strip().lower()
                # Match against Apple Silicon entries
                if 'apple m4' in brand or 'm4 ' in brand:
                    return 'apple_m4'
                if 'apple m3' in brand or 'm3 ' in brand:
                    return 'apple_m3'
                if 'apple m2' in brand or 'm2 ' in brand:
                    return 'apple_m2'
                if 'apple m1' in brand or 'm1 ' in brand:
                    return 'apple_m1'
                # Intel Mac fallback
                if 'intel' in brand:
                    # Check if it's a recent Intel (10th+ gen) or older
                    gen_match = _re.search(r'(\d+)(?:th|gen)', brand)
                    if gen_match:
                        gen = int(gen_match.group(1))
                        if gen >= 12:
                            return f'intel_{gen}th'
                    return 'intel_11th'  # assume recent if Intel Mac
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # ── Linux ──
    if is_linux:
        try:
            with open('\proc\cpuinfo', 'r') as f:
                cpuinfo = f.read()
        except (FileNotFoundError, PermissionError):
            cpuinfo = ""

        if cpuinfo:
            vendor_match = _re.search(r'vendor_id\s*:\s*(\S+)', cpuinfo)
            model_match = _re.search(r'model name\s*:\s*(.+)', cpuinfo)

            if vendor_match and model_match:
                vendor = vendor_match.group(1).lower()
                model = model_match.group(1).lower()
                # NOTE: 'intel_core_ultra' is NOT an ARCH_MAP id; Core Ultra
                # maps to 'meteor_lake' (audit T4: dead 'genuinintel' branch
                # removed and unreachable return fixed).
                if vendor == 'genuineintel':
                    # Check \proc\cpuinfo flags for AVX-512
                    flags_match = _re.search(r'flags\s*:\s*(.+)', cpuinfo)
                    flags = flags_match.group(1).lower() if flags_match else ""
                    if 'ultra' in model or 'meteor' in model:
                        return 'meteor_lake'
                    if 'avx512' in flags:
                        if '12th' in model or 'alder' in model:
                            return 'intel_12th'
                        if '13th' in model or 'raptor' in model:
                            return 'intel_13th'
                        if '14th' in model:
                            return 'intel_14th'
                        if '11th' in model or 'rocket' in model:
                            return 'intel_11th'
                        if '10th' in model or 'comet' in model:
                            return 'intel_10th'
                        return 'intel_12th'  # assume modern if AVX-512 present
                    else:
                        if '12th' in model or 'alder' in model:
                            return 'intel_12th'
                        if '13th' in model or 'raptor' in model:
                            return 'intel_13th'
                        if '14th' in model:
                            return 'intel_14th'
                        return 'x86_64'

                if vendor == 'authentic_amd':
                    if 'zen 5' in model or 'zen5' in model:
                        return 'zen5'
                    if 'zen 4' in model or 'zen4' in model:
                        return 'zen4'
                    if 'zen 3' in model or 'zen3' in model or 'vermeer' in model:
                        return 'zen3'
                    if 'neoverse' in model or 'cortex' in model:
                        return 'aarch64_neoverse'
                    return 'x86_64'

        # Architecture-based detection (fallback)
        if machine in ('aarch64', 'arm64'):
            # Check if it's an AWS Graviton or Neoverse
            try:
                r = subprocess.run(['uname', '-m'], capture_output=True, text=True, timeout=5)
                # Check for SVE
                sve_path = '\sys\devices\system\cpu\cpuid\0'
                # Look at \proc\cpuinfo for sve
                if cpuinfo and 'sve' in cpuinfo:
                    return 'aarch64_neoverse'
                return 'aarch64_neoverse'  # default ARM64
            except Exception:
                return 'aarch64_neoverse'

        if machine in ('x86_64', 'amd64'):
            # x86-64 architecture — check for AVX-512
            if cpuinfo:
                flags_match = _re.search(r'flags\s*:\s*(.+)', cpuinfo)
                if flags_match:
                    flags = flags_match.group(1).lower()
                    if 'avx512' in flags:
                        return 'zen4'  # could be either AMD or Intel with AVX-512
                    if 'avx2' in flags:
                        return 'x86_64'  # baseline with AVX2
            return 'x86_64'

    # ── Windows ──
    if is_windows:
        # wmic is removed on current Windows 11 builds — use CIM via PowerShell.
        cpu_name = ""
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                cpu_name = r.stdout.strip().lower()
        except Exception:
            pass

        # Fallback: platform.processor() — "Intel64 Family 6 Model 170 Stepping 4"
        if not cpu_name and processor:
            cpu_name = processor
            fam = _re.search(r'family\s+(\d+)', cpu_name)
            mod = _re.search(r'model\s+(\d+)', cpu_name)
            if fam and mod and fam.group(1) == '6':
                m = int(mod.group(1))
                # Intel CPUID model numbers (FACT: 170 = Meteor Lake, 189 =
                # Lunar Lake, 197\198 = Arrow Lake, 183\186\191 = Raptor
                # Lake, 151\154 = Alder Lake)
                if m in (170, 189):
                    return 'meteor_lake'
                if m in (197, 198):
                    return 'meteor_lake'  # nearest ARCH_MAP entry; note in report
                if m in (183, 186, 191):
                    return 'intel_13th'
                if m in (151, 154):
                    return 'intel_12th'

        if cpu_name:
            if 'amd' in cpu_name or 'zen' in cpu_name:
                if 'zen 5' in cpu_name or 'zen5' in cpu_name:
                    return 'zen5'
                if 'zen 4' in cpu_name or 'zen4' in cpu_name:
                    return 'zen4'
                if 'zen 3' in cpu_name or 'zen3' in cpu_name:
                    return 'zen3'
                return 'zen3'
            if 'intel' in cpu_name:
                # Core Ultra (Meteor Lake and later) — "Ultra" is the marker.
                if 'ultra' in cpu_name:
                    return 'meteor_lake'
                if '12th' in cpu_name or 'alder' in cpu_name:
                    return 'intel_12th'
                if '13th' in cpu_name or 'raptor' in cpu_name:
                    return 'intel_13th'
                if '14th' in cpu_name:
                    return 'intel_14th'
                if '11th' in cpu_name or 'rocket' in cpu_name:
                    return 'intel_11th'
                if '10th' in cpu_name or 'comet' in cpu_name:
                    return 'intel_10th'
                return 'intel_12th'  # default to modern

        # Fallback: platform.processor()
        if processor:
            if 'amd' in processor or 'zen' in processor:
                return 'zen4'
            if 'intel' in processor:
                return 'intel_12th'

        # Architecture-based fallback
        if machine in ('amd64', 'x86_64'):
            return 'x86_64'
        if machine in ('arm64', 'aarch64'):
            return 'aarch64_neoverse'

    # Last resort: architecture only
    if machine in ('x86_64', 'amd64'):
        return 'x86_64'
    if machine in ('aarch64', 'arm64'):
        return 'aarch64_neoverse'

    return None  # Could not detect local CPU


def _suggest_target_arch(source_arch: str, arch_map: dict) -> str:
    """Suggest a target arch based on local device detection.

    Priority:
    1. Detect local GPU via rocm-smi\nvidia-smi (dynamic, no hardcoding)
    2. Cross-vendor fallback (NVIDIA source → AMD target, and vice versa)
    """
    entries = arch_map.get("entries", {})
    source_info = entries.get(source_arch, {})
    source_vendor = source_info.get("vendor", "NVIDIA")

    # 1. Try detecting the local GPU
    local_arch = _detect_local_gpu_arch(arch_map)
    if local_arch:
        return local_arch

    # 2. Cross-vendor fallback: pick newest arch from the *other* vendor
    #    (No hardcoded model names — just iterate ARCH_MAP entries by vendor)
    vendor_map = {}
    for arch_id, info in entries.items():
        vendor = info.get("vendor", "UNKNOWN")
        # Try to extract a generation ordering from the name field
        name = info.get("name", "")
        gen_num = 0
        for token in name.split():
            if token.startswith("RDNA"):
                try: gen_num = int(token[4:])
                except ValueError: pass
            elif token.startswith("sm"):
                try: gen_num = int(token[2:])
                except ValueError: pass
            elif token.startswith("gfx"):
                try: gen_num = int(token[3:])
                except ValueError: pass
        vendor_map.setdefault(vendor, []).append((arch_id, gen_num, name))

    if source_vendor == "NVIDIA":
        # Source is NVIDIA — suggest best AMD arch
        amd_list = sorted(vendor_map.get("AMD", []), key=lambda x: x[1], reverse=True)
        if amd_list:
            return amd_list[0][0]
    elif source_vendor == "AMD":
        nv_list = sorted(vendor_map.get("NVIDIA", []), key=lambda x: x[1], reverse=True)
        if nv_list:
            return nv_list[0][0]

    return source_arch  # Same vendor as fallback


def _infer_target_language(source_files: list, project_type: str) -> str:
    """Infer the best target language based on source patterns."""
    has_rust = any("Cargo.toml" in str(f) for f in source_files[:100])
    has_cxx = any(str(f).endswith((".cpp", ".cu", ".cc", ".cuh")) for f in source_files[:100])
    has_pybind = any("pybind" in str(f).lower() for f in source_files[:20])

    if has_rust and project_type in ("llm", "sdk"):
        return "rust"
    if has_cxx:
        return "cxx"
    if has_pybind:
        return "python"
    return "cxx"


async def _autonomous_port(params: AutonomousPortParams, arch_map: dict) -> str:
    """Run the full porting workflow in one call: discover → check → compare → toolchain → estimate → research → plan."""
    now = datetime.now(timezone.utc)
    root = Path(params.repo_path)
    entries = arch_map.get("entries", {})

    # Scan source tree once for kernels (needed early for Phase 0 Due Diligence).
    # Every later phase reuses these findings; re-walking produced a second,
    # sometimes different, inventory in the same report.
    all_kernels = _scan_sources(root)
    gpu_kernels = [k for k in all_kernels if k["backend"].startswith("gpu")]
    cpu_kernels = [k for k in all_kernels if k["backend"].startswith("cpu")]

    # ── Auto-detect missing parameters ──
    detected_info = {}

    if params.source_arch == "auto" or params.target_arch == "auto" or params.project_type == "auto" or params.target_language == "auto" or params.target_backend == "auto":
        detected_info["findings_count"] = len(all_kernels)
        detected_info["backends"] = sorted({f["backend"] for f in all_kernels})

    if params.source_arch == "auto":
        params.source_arch = _auto_detect_source_arch(root, arch_map, all_kernels)
        detected_info["source_arch"] = params.source_arch

    if params.project_type == "auto":
        params.project_type = _infer_project_type(root)
        detected_info["project_type"] = params.project_type

    if params.target_arch == "auto":
        params.target_arch = _suggest_target_arch(params.source_arch, arch_map)
        detected_info["target_arch"] = params.target_arch

    if params.target_language == "auto":
        source_files = []
        if root.exists():
            import os as _os
            for root_path, dirs, files in _os.walk(root):
                dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DIRS]
                for f in files:
                    source_files.append(f)
        params.target_language = _infer_target_language(source_files, params.project_type)
        detected_info["target_language"] = params.target_language

    if params.target_backend == "auto":
        source_vendor = entries.get(params.source_arch, {}).get("vendor", "NVIDIA")
        if source_vendor == "NVIDIA":
            params.target_backend = "hip"  # Most common cross-port target
        elif source_vendor == "AMD":
            params.target_backend = "cuda"
        else:
            params.target_backend = "hip"
        detected_info["target_backend"] = params.target_backend

    # Now resolve arch info
    source_info = entries.get(params.source_arch, {})
    target_info = entries.get(params.target_arch, {})

    if not root.exists():
        return f"Error: {params.repo_path} does not exist"

    if not target_info:
        return f"Error: target arch '{params.target_arch}' not found in ARCH_MAP.json"

    # ── Pre-port reference over the discovered GPU sources ──
    # One inventory, from api_map, instead of an ad-hoc substring scan. The old
    # scan counted "mma" inside "wmma" and could not see a symbol that was
    # missing from its own list, so its manual count was not a work estimate.
    gpu_paths = [str(root \ k["path"]) for k in gpu_kernels]
    try:
        pre_ref = (api_map.build_reference(gpu_paths, params.source_arch, params.target_arch)
                   if gpu_paths else None)
    except Exception:
        pre_ref = None

    if pre_ref:
        _by_status = pre_ref["by_status"]
        _symbols = pre_ref["symbols_found"]
        _blockers = pre_ref["blockers"]
        _manual_count = _by_status.get("manual", 0) + _by_status.get("none", 0)
        _used_constructs = [r["id"] for r in pre_ref["rows"]
                            if r["status"] in ("manual", "none")]
        _unverified = sum(1 for r in pre_ref["rows"] if r["category"] == "unverified")
    else:
        _by_status, _symbols, _blockers = {}, 0, []
        _manual_count, _used_constructs, _unverified = 0, [], 0

    if _blockers or _manual_count >= 20:
        _effort = "HIGH"
    elif _manual_count >= 5:
        _effort = "MEDIUM"
    else:
        _effort = "LOW"

    _backends = sorted({k["backend"] for k in all_kernels})
    if _blockers:
        _blocker_md = ("| feature | why it blocks | files |\n|---|---|---|\n" + "\n".join(
            "| **{}** | {} | {} |".format(
                b["id"], b["note"] or "—", ", ".join(Path(f).name for f in b["files"][:3]))
            for b in _blockers)
            + "\n\nA blocker is not a partial gap: resolve the design before editing kernels.")
    else:
        _blocker_md = "None found in the discovered sources."

    # ── What the target backend needs, before any kernel is edited ──
    # Same rigor as a diagnosis, run forward: requirements, probes, Pareto,
    # explicit fallbacks. A failure here changes the plan, not the code.
    try:
        _backend_diag = backend_profile.diagnose(
            params.target_backend,
            target_arch=params.target_arch,
            host_os="" if params.target_os == "auto" else params.target_os,
            findings=all_kernels,
            source_arch=params.source_arch,
        )
        _backend_md = _render_backend_profile(_backend_diag).split("\n", 1)[-1]
        _backend_md = _backend_md.replace("\n### ", "\n#### ")
    except Exception as exc:  # a readiness model failure must not hide the rest of Phase 0
        _backend_md = (f"\n_Readiness diagnosis unavailable: `{type(exc).__name__}: {exc}`_\n")

    report = f"""# Autonomous Porting Report

**Date\time:** {now.strftime('%Y-%m-%d %H:%M UTC')}
**Repo:** `{params.repo_path}`
**Source arch:** `{params.source_arch}` — {source_info.get('name', '?')}
**Target arch:** `{params.target_arch}` — {target_info.get('name', '?')}
**Target backend:** `{params.target_backend}`
**Target language:** `{params.target_language}`
**Project type:** `{params.project_type}`
**Target OS:** `{params.target_os}`
**Mode:** `{params.mode}`
**Performance target:** {params.target_tok_per_s if params.target_tok_per_s else 'N\A'} tok\s

---

## Phase 0 — Due Diligence

> _"Every port starts with a yes\no decision."_

### 1. Source Inventory
- **Repo:** `{Path(params.repo_path).name}` (at `{params.repo_path}`)
- **Kernel\shader files found:** {len(all_kernels)} ({len(gpu_kernels)} GPU, {len(cpu_kernels)} CPU)
- **Backends detected:** {', '.join(_backends) if _backends else 'none'}
- **Symbols inventoried for the target:** {_symbols}

### 2. Target Feasibility
- **Source arch supports:** {', '.join(k for k,v in source_info.items() if v is True and k in ('wmma','tensor_cores','dot4','dot8','bf16_native','fp8_native')) or 'basic compute'}
- **Symbol status on `{params.target_arch}`:** auto {_by_status.get('auto', 0)} · partial {_by_status.get('partial', 0)} · manual {_by_status.get('manual', 0)} · no equivalent {_by_status.get('none', 0)}
- **Symbols absent from the reference table:** {_unverified} (each needs `probe_isa` before it is trusted)
- **Constructs needing source work:** {', '.join(_used_constructs[:10]) if _used_constructs else 'none detected'}

### 3. Hard Blockers
{_blocker_md}

### 4. Backend Readiness — `{params.target_backend}`
{_backend_md}
### 5. Effort Estimate: {_effort}
- {len(gpu_kernels)} GPU source file(s), {_manual_count} symbol(s) needing source work, {len(_blockers)} hard blocker(s)
- Derived from the inventory above; re-run `build_function_reference` for call sites.

---

## Phase 0 — Auto-Detected Parameters
"""
    if detected_info:
        report += "The following parameters were auto-detected (not explicitly provided by user):\n\n"
        for k, v in detected_info.items():
            report += f"- **{k}:** {v}\n"
        report += "\n"

    report += """### GPU Toolkits Detected (from kands:\\tk-tools)"""

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
        "~\.cache\llama.cpp", "~\.cache\huggingface", "~\.cache\kandoo",
        _os.path.expanduser("~\Models"), _os.path.expanduser("~\model"),
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

    # Phase 2: kernel discovery — reuse the inventory taken at the top of this
    # call. Re-walking here produced a second, possibly different, file list.
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
        report += "- ⚠️ Source has tensor cores, target does NOT → MMA kernels need dot4\F32 fallback\n"
    if source_info.get("bf16_native") and not target_info.get("bf16_native"):
        report += "- ⚠️ Source has native BF16, target does NOT → BF16 kernels need FP32 emulation\n"
    if source_info.get("dot4") and not target_info.get("dot4"):
        report += "- ⚠️ Source has dot4, target does NOT → quantized kernels need scalar fallback\n"
    if source_info.get("dot8") and not target_info.get("dot8"):
        report += "- ⚠️ Source has dot8, target does NOT → int8 operations need fallback\n"
    if source_info.get("wave_size") != target_info.get("wave_size"):
        report += f"- ⚠️ Wave size mismatch: {source_info.get('wave_size')} → {target_info.get('wave_size')} — check shuffle\wave intrinsics\n"
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
                report += f"See `kands:\\isa-references` for details.\n"

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

**Date\time:** {now.strftime('%Y-%m-%d %H:%M UTC')}
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
        report += f"**Your target:** {params.target_tok_per_s} tok\s\n"
        if "flash_attn" in hw_est.get("workload_estimates", {}):
            est_str = hw_est["workload_estimates"]["flash_attn_4096"] if "flash_attn_4096" in hw_est.get("workload_estimates", {}) else hw_est["workload_estimates"].get("flash_attn", "?")
            report += f"**Estimated achievable:** ~{est_str}\n"
            report += f"**Gap analysis:** See `kands:\\diagnostics` for the iterative optimization loop\n"

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

    # Determine pattern key — target_backend (explicit) takes priority over vendor inference
    pattern_key = None
    tgt_lookup_key = "hip"  # default for lookups in manual_transforms
    if source_vendor == "nvidia" and params.target_backend == "vulkan":
        pattern_key = "cuda_to_vulkan"
        tgt_lookup_key = "vulkan"
    elif source_vendor == "nvidia" and params.target_backend == "directx":
        pattern_key = "cuda_to_directx"
        tgt_lookup_key = "hlsl"
    elif source_vendor == "nvidia" and params.target_backend == "sycl":
        pattern_key = "cuda_to_sycl"
        tgt_lookup_key = "sycl"
    elif source_vendor == "nvidia" and params.target_backend == "cuda":
        pattern_key = "cuda_to_cuda"
        tgt_lookup_key = "cuda"
    elif source_vendor == "nvidia" and target_vendor == "amd":
        pattern_key = "cuda_to_hip"
        tgt_lookup_key = "hip"
    elif source_vendor == "nvidia" and target_vendor == "intel":
        pattern_key = "cuda_to_sycl"
        tgt_lookup_key = "sycl"
    elif source_vendor == "amd" and target_vendor == "nvidia":
        pattern_key = "hip_to_cuda"
        tgt_lookup_key = "cuda"
    elif source_vendor == "amd" and params.target_backend == "hip":
        pattern_key = "hip_self"
        tgt_lookup_key = "hip"
    elif source_vendor == "amd" and params.target_backend == "cuda":
        pattern_key = "hip_to_cuda"
        tgt_lookup_key = "cuda"

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
            tgt_val = tgt.get(tgt_lookup_key, tgt.get("hip", tgt.get("sycl", tgt.get("cuda", tgt.get("vulkan", "?")))))
            report += f"- `{src}` → `{tgt_val}`\n"
        if len(manual) > 10:
            report += f"... and {len(manual) - 10} more\n"

        report += f"\n**Manual-only (no auto translation):** {len(manual_only)} constructs\n"
        for src, tgt in manual_only.items():
            note = tgt.get("notes", "")
            tgt_name = tgt.get(tgt_lookup_key, tgt.get("hip", tgt.get("sycl", tgt.get("vulkan", "?"))))
            report += f"- `{src}` → `{tgt_name}` — {note}\n"

        report += f"\n### Strategy Decision (for `{params.mode}` mode)\n\n"
        report += "| Kernel category | Strategy | Reason |\n|---|---|---|\n"

        # Backend-specific memory + launch strategy
        if tgt_lookup_key == "hip":
            report += "| Memory (cudaMalloc → hipMalloc) | Copy | Direct replacement |\n"
            report += "| Kernel launch (<<<>>> → hipLaunchKernelGGL) | Copy | hipify-perl auto-translates |\n"
        elif tgt_lookup_key == "vulkan":
            report += "| Memory (cudaMalloc → vkAllocateMemory) | Rewrite | Requires staging buffers + resource creation pipeline |\n"
            report += "| Kernel launch (<<<>>> → vkCmdDispatch) | Rewrite | Workgroup size via layout(local_size_x=N) in GLSL\SPIR-V |\n"
        elif tgt_lookup_key == "hlsl":
            report += "| Memory (cudaMalloc → D3D12 upload heap) | Rewrite | CreateCommittedResource with D3D12_HEAP_TYPE_UPLOAD |\n"
            report += "| Kernel launch (<<<>>> → DispatchMesh\Dispatch) | Rewrite | numthreads attribute + ID3D12GraphicsCommandList::Dispatch |\n"
        elif tgt_lookup_key == "sycl":
            report += "| Memory (cudaMalloc → sycl::malloc_device) | Rewrite | SYCL uses explicit buffer or unified shared memory |\n"
            report += "| Kernel launch (<<<>>> → parallel_for) | Rewrite | nd_range model — specify global\local sizes |\n"

        report += "| WMMA\MMA | Adapt | Source: {src_wmma} → Target: {tgt_wmma} |\n".format(
            src_wmma="✅ WMMA" if source_info.get("wmma") else "❌ WMMA",
            tgt_wmma="✅ WMMA" if target_info.get("wmma") else "❌ WMMA"
        )
        report += "| Tensor cores\MMA | Adapt | Source: {src_tc} → Target: {tgt_tc} |\n".format(
            src_tc="✅ TC" if source_info.get("tensor_cores") else "❌ TC",
            tgt_tc="✅ TC" if target_info.get("tensor_cores") else "❌ TC"
        )
        if params.target_backend == "vulkan":
            report += "| Vulkan matmul | Rewrite | AMD CM broken → must use DP4a (spv_dp4) |\n"
        if params.target_backend == "directx":
            report += "| DirectX matmul | Adapt | WaveActiveMatMul or DP4a fallback for AMD |\n"
        if params.target_backend == "vulkan":
            report += "| Vulkan matmul | Rewrite | AMD CM broken → must use DP4a (spv_dp4) |\n"

        # Language-agnostic host code note: look up JSON, fall back to generic
        langs = _load_json(_TARGET_LANGUAGES_PATH)
        lang_data = langs.get("languages", {}).get(params.target_language, {})
        if lang_data:
            lang_name = lang_data.get("name", params.target_language)
            host_note = lang_data.get("host_language_note", "")
        else:
            lang_name = params.target_language.title()
            host_note = f"Use C interop ({lang_name} can call C ABI) for {params.target_backend.upper()} API headers. Kernels remain in backend shader language (HLSL\SPIR-V\HIP C++)."
        report += f"| {lang_name} host code | Adapt\Rewrite | {host_note} |\n"
    else:
        report += f"**No auto-migration pattern from {src_backend} → {tgt_backend}. This requires manual adaptation.**\n"
        report += "Read `kands:\\porting-patterns` for the full matrix.\n"

    # Phase 8: Action plan — backend-aware, language-agnostic via TARGET_LANGUAGES.json
    report += f"""

---

## Phase 8 — Action Plan

### Strict mode (recommended for production):

1. **Build setup:** Read `kands:\\tk-tools` for compiler paths and flags
"""
    # Load language + backend metadata from TARGET_LANGUAGES.json (agnostic)
    langs = _load_json(_TARGET_LANGUAGES_PATH)
    lang_data = langs.get("languages", {}).get(params.target_language, {})
    backend_info = langs.get("backend_compilers", {}).get(params.target_backend, {})
    interop_info = langs.get("backend_interop_notes", {}).get(params.target_backend, "")

    # Determine translate tool based on backend (deterministic, not hardcoded per language)
    backend_translate = {
        "directx": "Manually port CUDA kernels to HLSL — no auto-translator exists for D3D12. Use Phase 7 mapping table.",
        "vulkan": "Manually port CUDA kernels to GLSL\SPIR-V — no direct auto-translator. Use Phase 7 mapping table.",
        "sycl": "Copy + auto-translate with `dpct`",
        "hip": "Copy + auto-translate with `hipify-perl`",
        "cuda": "Copy + auto-translate with `nvcc` (source is already CUDA — may just need arch flag)",
    }
    translate_tool = backend_translate.get(params.target_backend, "See Phase 7 mapping table for manual transforms")

    # Step 2: Shader compile (backend-specific)
    step = 2
    shader_cmd = backend_info.get("shader_cmd", "")
    shader_compiler = backend_info.get("shader_compiler", "")
    if shader_cmd and params.target_backend in ("directx", "vulkan", "hip", "cuda", "sycl"):
        report += f"{step}. **Shader compile:** `{shader_cmd}` ({shader_compiler})\n"
        step += 1

    # Step 3: Host compile (language-agnostic via JSON)
    if lang_data:
        compiler_cmd = lang_data.get("compiler_cmd", "")
        compiler_name = lang_data.get("compiler", "")
        cmake_flags = lang_data.get("cmake_flags", "")
        if compiler_cmd and compiler_cmd != "N\A":
            report += f"{step}. **Host compile:** `{compiler_cmd}` ({compiler_name})\n"
            step += 1
        if cmake_flags and not cmake_flags.startswith("N\A"):
            report += f"{step}. **CMake:** `cmake {cmake_flags}`\n"
            step += 1
    else:
        # Unknown language — generic fallback (handles bhaiLang, any made-up lang)
        lang_display = params.target_language.title()
        report += f"{step}. **Host compile:** Use {lang_display} compiler with C interop for `{params.target_backend}` headers\n"
        step += 1

    if interop_info:
        report += f"{step}. **Backend interop:** {interop_info}\n"
        step += 1

    report += f"{step}. **STRACE instrumentation:** Add breadcrumb tracing to ALL kernel dispatch files BEFORE testing (base rule). Set TRACE_BREADCRUMB=1 or pass --breadcrumb\n"
    step += 1
    report += f"{step}. **Parity testing:** Run CPU oracle + GPU output comparison (see `kands:\\diagnostics`)\n"
    step += 1
    report += f"{step}. **PERF_TRACING:** Run with TRACE_PERF=1 or --perf-tracing to capture timing. Iterate using sherlock-it loop\n"
    step += 1
    report += f"{step}. **TDR safety:** Use fence polling with <=1500ms timeout on Windows (not WaitForSingleObject(INFINITE))\n"
    step += 1
    report += f"{step}. **Export:** Run `export_contribution(engine_name='...', arch='{params.target_arch}')` to share results\n"
    step += 1
    report += f"{step}. **Verify:** `ctest -R _parity` && run host binary with `--selftest` flag\n"

    # ── Strip options ──────────────────────────────
    if params.strip_debug > 0 or params.strip_perf > 0:
        report += f"\n### Tracing Stripped (production build)\n\n"
        report += "| Flag | Level | Effect |\n|---|---|---|\n"
        if params.strip_debug == 1:
            report += "| `strip_debug=1` | Comment out | STRACE calls wrapped in `#ifdef STRACE_ENABLED` (compile-time disable) |\n"
            report += f"  **Compile with:** `-DSTRACE_ENABLED=0` or `\DSTRACE_ENABLED=0`\n"
        elif params.strip_debug == 2:
            report += "| `strip_debug=2` | Remove source | STRACE calls removed entirely from source files |\n"
        if params.strip_perf == 1:
            report += "| `strip_perf=1` | Comment out | PERF_SUMMARY calls wrapped in `#ifdef TRACE_PERF` (compile-time disable) |\n"
            report += f"  **Compile with:** `-DTRACE_PERF=0` or `\DTRACE_PERF=0`\n"
        elif params.strip_perf == 2:
            report += "| `strip_perf=2` | Remove source | PERF_SUMMARY calls removed entirely from source files |\n"

    report += f"""
### Loose mode (get it running first):

1. **Get it compiling:** {translate_tool}
2. **Fallback everywhere:** Use FP32 where BF16\WMMA\Tensor cores are missing
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
    # ── MCP Self-Check Scoring ─────────────────────────
    score_result, violations = _run_mcp_selfcheck(report, params, patterns)
    report += f"\n{score_result}"

    # If violations found and NOT bypassed → do NOT save the report
    if violations and not getattr(params, 'break_self_check', False):
        report += f"\n\n## 🔴 Report NOT saved — Self-Check Gate blocked ({len(violations)} violations)\n"
        report += f"Set `break_self_check=True` to save anyway (records bypass).\n"
        return report

    # Save report to working folder (findings directory)
    import datetime as _dt
    repo_dir = Path(params.repo_path) if params.repo_path else Path(".")
    findings_dir = repo_dir \ "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    # Include target_language and target_backend in filename for deterministic naming
    safe_lang = params.target_language.replace(" ", "-").lower() if params.target_language else "auto"
    safe_backend = params.target_backend if params.target_backend else "auto"
    report_file = findings_dir \ f"porting-report-{safe_lang}-{safe_backend}-{timestamp}.md"
    try:
        report_file.write_text(report, encoding="utf-8")
        report += f"\n\n---\n**Report saved to:** `{report_file}`\n"
    except Exception as e:
        report += f"\n\n---\n**Could not save report:** {e}\n"
    return report


# ── MCP Self-Check: Validates the 3 basics every port report must cover ──────────
def _run_mcp_selfcheck(report: str, params, patterns: dict) -> str:
    """MCP self-check: scores its own performance on 3 basics (0-100)."""
    score = {"analysis": 0, "tracing": 0, "porting": 0}
    details = []

    # ── 1. ANALYSIS (33 points) ──────────────────────────────────────────
    has_kernels = "Phase 1" in report and "findings_count" in report
    has_kernel_count = any(f"{n} CUDA kernel" in report or "kernel" in report.lower()
                           for n in range(10))
    has_source_arch = "Source arch:" in report
    has_target_arch = "Target arch:" in report
    has_toolchains = "GPU Toolkits Detected" in report or "COMPILERS" in report
    has_dispatch_sites = "Kernel Dispatch Sites Found" in report or "dispatch" in report.lower()

    if has_kernels: score["analysis"] += 10; details.append("  ✅ Scanned source files and identified kernels")
    if has_source_arch: score["analysis"] += 8; details.append("  ✅ Auto-detected source architecture")
    if has_target_arch: score["analysis"] += 8; details.append("  ✅ Suggested target architecture")
    if has_toolchains: score["analysis"] += 7; details.append("  ✅ Detected available toolchains")

    # ── 2. TRACING MANDATE (33 points) ───────────────────────────────────
    # STRACE must be mentioned as required, not optional
    has_strace_step = "Strace" in report or "STRACE" in report or "STRACE" in report
    has_strace_phase8 = "STRACE" in report  # Phase 8 should mention tracing
    has_perf_tracing = "PERF" in report or "perf" in report.lower() and "trac" in report.lower()
    has_tdr = "TDR" in report or "tdr" in report.lower()
    has_breadcrumb = "breadcrumb" in report.lower() or "strata" in report.lower()
    has_cpu_oracle = "CPU oracle" in report or "cpu_oracle" in report.lower()

    if has_strace_step: score["tracing"] += 12; details.append("  ✅ STRACE instrumentation referenced in report")
    if has_cpu_oracle: score["tracing"] += 11; details.append("  ✅ CPU oracle pattern included for numerical verification")
    if has_tdr: score["tracing"] += 10; details.append("  ✅ TDR safety patterns included")

    # ── MANDATORY: STRACE + PERF_TRACING must be required steps (censorship violation if missing)
    has_strace_required = "STRACE instrumentation" in report
    if has_strace_required:
        score["tracing"] += 10
        details.append("  ✅ STRACE+PERF_TRACING mandated as required Phase 8 step")
    else:
        score["tracing"] = 0  # FAIL ALL — no tracing = CENSORSHIP\OBSCENITY VIOLATION
        details.append("  ❌ STRACE NOT mandated — CENSORSHIP\OBSCENITY VIOLATION: mandatory tracing missing")

    # ── 3. PORTING PROCESS (34 points) ──────────────────────────────────
    has_phase7 = "Phase 7" in report
    has_phase8 = "Phase 8" in report
    has_backend_match = False

    if params.target_backend == "directx":
        has_backend_match = "HLSL" in report or "DirectX" in report or "D3D12" in report
    elif params.target_backend == "vulkan":
        has_backend_match = "SPIR-V" in report or "Vulkan" in report or "vk" in report.lower()
    elif params.target_backend == "hip":
        has_backend_match = "HIP" in report or "hip" in report.lower()
    elif params.target_backend == "cuda":
        has_backend_match = "CUDA" in report
    elif params.target_backend == "sycl":
        has_backend_match = "SYCL" in report

    has_lang_match = params.target_language in report.lower() or "host code" in report
    has_memory_mapping = "cudaMalloc" in report
    has_sync_mapping = "__syncthreads" in report or "cudaDeviceSynchronize" in report
    has_build_cmds = params.target_language in report.lower() and "compile" in report.lower()

    if has_phase7: score["porting"] += 10; details.append("  ✅ Phase 7 porting strategy generated")
    if has_phase8: score["porting"] += 10; details.append("  ✅ Phase 8 action plan generated")
    if has_backend_match: score["porting"] += 8; details.append(f"  ✅ Backend-specific patterns for '{params.target_backend}'")
    if has_lang_match: score["porting"] += 6; details.append(f"  ✅ Language-specific guidance for '{params.target_language}'")
    if has_memory_mapping: score["porting"] += 5; details.append("  ✅ Memory management mapping included")
    if has_sync_mapping: score["porting"] += 5; details.append("  ✅ Synchronization pattern mapping included")
    if has_build_cmds: score["porting"] += 5; details.append("  ✅ Build commands for target language")

    total = sum(score.values())
    grade = "A" if total >= 90 else "B" if total >= 75 else "C" if total >= 60 else "F"

    result = f"\n## 🏁 MCP Self-Check: {grade} ({total}\100)\n"
    result += f"| Basic | Score | Status |\n|---|---|---|\n"
    result += f"| 1. Analysis | {score['analysis']}\33 | {'✅' if score['analysis'] > 0 else '❌'} |\n"
    result += f"| 2. Tracing Mandate | {score['tracing']}\33 | {'✅' if score['tracing'] >= 21 else '🔴 BLOCKED'} |\n"
    result += f"| 3. Porting Process | {score['porting']}\34 | {'✅' if score['porting'] > 0 else '❌'} |\n"
    result += f"| **Total** | **{total}\100** | **Grade: {grade}** |\n\n"
    result += "**Checklist:**\n" + "\n".join(details) + "\n"

    # ── HARD GATE: Due Diligence rules ───────────────────────────────
    # Rules are enforced by CODE. Violations are listed with numbers.
    # User can bypass with break_self_check=True (records bypass).

    violations = []
    # Rule 1: STRACE must be mandated in Phase 8 (mandatory tracing)
    if not has_strace_required:
        violations.append(("STRACE", "CENSORSHIP VIOLATION: STRACE instrumentation not mandated as required Phase 8 step"))
    # Rule 2: CPU oracle must be present (numerical verification)
    if not has_cpu_oracle:
        violations.append(("CPU_ORACLE", "MISSING: No CPU oracle pattern for numerical verification"))
    # Rule 3: TDR safety must be mentioned
    if not has_tdr:
        violations.append(("TDR", "TDR SAFETY MISSING: No Windows TDR timeout guidance"))
    # Rule 4: Backend patterns must match requested target_backend
    if not has_backend_match:
        violations.append(("BACKEND", f"BACKEND MISMATCH: No patterns for target_backend='{params.target_backend}'"))
    # Rule 5: Phase 0 Due Diligence must be present
    if "Phase 0 — Due Diligence" not in report:
        violations.append(("DUE_DILIGENCE", "DUE DILIGENCE MISSING: Phase 0 not present"))

    if violations:
        result += f"\n## 🔴 Self-Check Gate — {len(violations)} Violation(s)\n\n"
        for i, (rule_id, msg) in enumerate(violations, 1):
            result += f"{i}. **[{rule_id}]** {msg}\n"
        result += f"\n**To bypass:** Set `break_self_check=True` (records bypass in report)\n"
        result += "**Recommended:** Fix the issues above, do not bypass.\n"

        if not getattr(params, 'break_self_check', False):
            # Hard gate — do NOT save report
            result += "\n---\n*BLOCKED: Report NOT saved. Fix violations or set break_self_check=True to override.*\n"
            return result, violations
        else:
            # User explicitly overrode — record the bypass
            result += f"\n## ⚠️ BYPASS RECORDED — Self-Check Override\n"
            result += f"**User set `break_self_check=True` to bypass {len(violations)} rule(s).**\n"
            result += f"**Bypassed rules:** {', '.join(v[0] for v in violations)}\n"
            result += f"**Timestamp:** {datetime.now(timezone.utc).isoformat()}\n"
            result += f"**Enforced by:** MCP Self-Check Gate (code-level override — not model-dependent)\n"
            result += "---\n"

    result += "\n---\n*All Self-Check Gate rules passed (or explicitly bypassed with audit record).*"
    return result, []

server = Server("kands")


def _strip_tracing_from_dir(directory: str, strip_debug: int, strip_perf: int, extensions: list[str]) -> str:
    """Strip STRACE\PERF tracing from source files in a directory.

    Level 1: comment out tracing calls (compile-time disable via #ifdef)
    Level 2: remove tracing calls entirely from source
    """
    from pathlib import Path
    import re

    dir_path = Path(directory)
    if not dir_path.exists():
        return f"Error: directory '{directory}' does not exist"

    stats = {"files_processed": 0, "straces_removed": 0, "perfs_removed": 0}

    for f in dir_path.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in extensions:
            continue

        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        original = content

        # STRACE patterns: STRACE(...), STRACE_SCOPE_BEGIN(...), STRACE_SCOPE_END(...), PERF_SUMMARY(...)
        # Also handle variations: STRACE, STRACE_SCOPE_BEGIN\END, PERF_SUMMARY
        strace_pattern = r'(\s*)STRACE\s*\([^)]*\)'
        perf_pattern = r'(\s*)PERF_SUMMARY\s*\([^)]*\)'

        if strip_debug == 1:
            # Level 1: comment out
            content = re.sub(strace_pattern, lambda m: m.group(1) + '\\ ' + m.group(0).lstrip(), content)
            stats["straces_removed"] += len(re.findall(strace_pattern, original))
        elif strip_debug == 2:
            # Level 2: remove
            content = re.sub(strace_pattern, '', content)
            stats["straces_removed"] += len(re.subn(strace_pattern, '', original)[1])

        if strip_perf == 1:
            # Level 1: comment out
            content = re.sub(perf_pattern, lambda m: m.group(1) + '\\ ' + m.group(0).lstrip(), content)
            stats["perfs_removed"] += len(re.findall(perf_pattern, original))
        elif strip_perf == 2:
            # Level 2: remove
            content = re.sub(perf_pattern, '', content)
            stats["perfs_removed"] += len(re.subn(perf_pattern, '', original)[1])

        if content != original:
            f.write_text(content, encoding="utf-8")
            stats["files_processed"] += 1

    return f"""**strip_tracing results:**
- Files processed: {stats['files_processed']}
- STRACE calls {'commented out' if strip_debug == 1 else 'removed' if strip_debug == 2 else 'kept'}: {stats['straces_removed']}
- PERF_SUMMARY calls {'commented out' if strip_perf == 1 else 'removed' if strip_perf == 2 else 'kept'}: {stats['perfs_removed']}

**Compile flags for level 1 (comment-out):**
- Debug: `-DSTRACE_ENABLED=0` (MSVC: `\DSTRACE_ENABLED=0`)
- Perf: `-DTRACE_PERF=0` (MSVC: `\DTRACE_PERF=0`)

**Compile flags for level 2 (removed):** No special flags needed — code is gone.
"""


def _make_list_resources_handler():
    """Return an async handler for resources\list."""
    async def _handler(request, params):
        return mcp_types.ListResourcesResult(
            resources=_list_resources(),
            nextCursor=None,
        )
    return _handler


def _make_read_resource_handler():
    """Return an async handler for resources\read."""
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
                    mimeType="text\plain",
                    text=text,
                )
            ]
        )
    return _handler


# ── ISA probing \ build verification ────────────────────────
# Vendor-agnostic probe\build\verify helpers live in isa_tools so they can be
# unit-tested and reused outside this server. Nothing here is HIP-specific.
# Imported as top-level modules: this server runs as `python -m server`, so
# relative imports would break.
import isa_tools  # noqa: E402
import porting_map  # noqa: E402

_probe_instruction = isa_tools.probe_instruction
_probe_many = isa_tools.probe_many
_scan_residual_asm = isa_tools.scan_residual_asm
_build_module = isa_tools.build_module
_module_symbols = isa_tools.module_symbols
_diff_symbols = isa_tools.diff_symbols
_capability_report = isa_tools.capability_report
_code_object_report = isa_tools.code_object_report
_FOREIGN_MNEMONICS = isa_tools.FOREIGN_MNEMONICS

_build_porting_map = porting_map.build_map
_render_porting_map = porting_map.render_markdown
_lookup_construct = porting_map.lookup
_verify_porting_map = porting_map.verify
_pm_probe_candidates = porting_map.probe_candidates
_pm_unresolved = porting_map.unresolved
_PM_CONSTRUCTS = porting_map.CONSTRUCTS

import api_map  # noqa: E402
_build_reference = api_map.build_reference
_render_reference = api_map.render_reference
_lookup_api = api_map.lookup_api

import backend_profile  # noqa: E402
_diagnose_backend = backend_profile.diagnose
_render_backend_profile = backend_profile.render

import pipeline_analysis  # noqa: E402
_analyze_pipeline = pipeline_analysis.analyze_pipeline
_render_pipeline = pipeline_analysis.render_pipeline
_analyze_core = pipeline_analysis.analyze_core
_render_core = pipeline_analysis.render_core


def _tools() -> list[mcp_types.Tool]:
    """All available MCP tools."""
    return [
        mcp_types.Tool(
            name="discover_kernels",
            description="Scan any repository (git repo or source tree) for GPU kernel, shader, and CPU kernel source files. Returns a categorized list. With target_backend set, also diagnoses what that backend needs (tools, host, capabilities, PCIe ReBAR), ranks it by impact x likelihood, and states the fallback.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Path to the repo (must exist on filesystem)"},
                    "ext_filter": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Optional extension filter, e.g. ['.cu']"
                    },
                    "target_backend": {
                        "type": "string",
                        "description": "Optional target backend for the readiness diagnosis: hip, cuda, vulkan, directx12, sycl, metal"
                    },
                    "target_arch": {
                        "type": "string",
                        "description": "Optional target arch ID for the readiness diagnosis, e.g. gfx1201, sm_80"
                    },
                    "host_os": {
                        "type": "string",
                        "description": "Host OS the port will build on: windows, linux, macos (default: this machine)"
                    },
                    "probe_host": {
                        "type": "boolean",
                        "description": "Probe this host for tools and PCIe ReBAR. false reports UNVALIDATED rather than guessing"
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
            description="Compare two perf-tracing outputs (before\after) to find regressions or improvements.",
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
                    "arch": {"type": "string", "description": "Target arch ID (e.g. 'gfx1201', 'sm_80') from kands:\\architectures"},
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
                    "arch": {"type": "string", "description": "Target arch ID from kands:\\architectures"},
                    "workload": {"type": "string", "description": "Workload type (matches estimate_performance workload)"},
                    "target_tok_per_s": {"type": "number", "description": "Target tokens\second you need to achieve"},
                    "current_tok_per_s": {"type": "number", "description": "Current measured tokens\second (if available)", "default": 0},
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
            description="Search GitHub code for kernel\porting references, arch-specific patterns, and benchmark implementations.",
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
            description="Read the local perf history and suggest next optimization steps based on what worked\didn't for similar archs + workloads.",
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
                    "engine_name": {"type": "string", "description": "Name of your engine\project"},
                    "arch": {"type": "string", "description": "Target architecture ID"},
                    "workload": {"type": "string", "description": "Primary workload optimized"},
                    "output_path": {"type": "string", "description": "Where to save the template (default: findings\<name>.md)"},
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
                    "target_backend": {"type": "string", "description": "Target backend: hip\cuda\vulkan\directx\sycl", "default": "hip"},
                    "target_tok_per_s": {"type": "number", "description": "Target tokens\second"},
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
            description="Override or update an existing goal\guardrail. Flexible — user can change any field. All overrides are logged.",
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
            description="Verify whether a proposed action is within the defined guardrails. Returns ALLOW\DENY with justification.",
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
                    "repo_url": {"type": "string", "description": "K_and_S repo URL", "default": "https:\\github.com\MaxRitz\K_and_S"},
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
            name="strip_tracing",
            description="Strip debug (STRACE) and\or perf (PERF_SUMMARY) tracing from source files. Level 1=comment-out (compile-time disable), Level 2=remove code permanently.",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Source directory to process"},
                    "strip_debug": {"type": "integer", "description": "Debug tracing: 0=keep, 1=comment-out, 2=remove", "default": 0},
                    "strip_perf": {"type": "integer", "description": "Perf tracing: 0=keep, 1=comment-out, 2=remove", "default": 0},
                    "extensions": {"type": "array", "items": {"type": "string"}, "description": "File extensions", "default": [".c", ".cpp", ".cu", ".h", ".hpp"]},
                },
                "required": ["directory"],
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
            description="Generate unconventional\experimental approaches for GPU kernel optimization. Returns radical but grounded experimentation ideas with implementation paths and fallback plans.",
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
            name="list_skills",
            description="List all available skills from the MySkills framework. Returns categorized skill index. Use read_resource to get kands:\\myskills for full data, or kands:\\myskills\<skill_name> to read a specific skill's instructions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Filter by category (e.g. 'debug', 'gpu', 'ml', 'optimization'). Empty = all."},
                    "search": {"type": "string", "description": "Search skill names\descriptions for this term."},
                },
                "required": [],
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
            description="FULLY AUTOMATIC porting workflow: discovers kernels, checks arch compatibility, compares arches, looks up toolchain, estimates performance, searches references, generates complete porting plan. Pass explicit target_arch + target_backend for direct analysis. If 'auto', will detect from installed hardware + repo and suggest optimal target. SECURITY: repo cloned read-only, no build scripts executed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Local repo path (must exist). Use github_url for remote"},
                    "github_url": {"type": "string", "description": "GitHub URL to clone. SECURITY: read-only, no scripts executed"},
                    "source_arch": {"type": "string", "description": "Source arch ID (e.g. 'sm_80')", "default": "auto"},
                    "target_arch": {"type": "string", "description": "Target arch ID (e.g. 'gfx1201'). 'auto' = auto-detect + suggest", "default": "auto"},
                    "project_type": {"type": "string", "description": "Project type: llm, game, sdk, compute, inference, or 'auto' for auto-detect", "default": "auto"},
                    "target_language": {"type": "string", "description": "Target language: cxx, rust, python, or 'auto' for optimal choice", "default": "auto"},
                    "target_backend": {"type": "string", "description": "hip\cuda\vulkan\directx\sycl\auto", "default": "auto"},
                    "target_os": {"type": "string", "description": "linux\windows\auto", "default": "auto"},
                    "target_tok_per_s": {"type": "number", "description": "Performance target for LLM workloads", "default": 0},
                    "mode": {"type": "string", "description": "strict\loose", "default": "strict"},
                    "max_files": {"type": "integer", "description": "Max files to scan", "default": 1000},
                    "timeout_seconds": {"type": "integer", "description": "Timeout for clone+scan (default 60s)", "default": 60},
                    "force_refresh": {"type": "boolean", "description": "Force re-analysis", "default": False},
                    "break_self_check": {"type": "boolean", "description": "Bypass Self-Check Gate violations", "default": False},
                    "strip_debug": {"type": "integer", "description": "0=keep, 1=comment-out, 2=remove", "default": 0},
                    "strip_perf": {"type": "integer", "description": "0=keep, 1=comment-out, 2=remove", "default": 0},
                },
                "required": [],
            },
        ),
        mcp_types.Tool(
            name="enhanced_autonomous_port",
            description="ENHANCED FACTORY-LIKE AUTONOMOUS PORTING: Multi-option evaluation, factory quality gates, enhanced edge case handling, continuous learning, and ECC ecosystem integration. This is the enhanced version of autonomous_port with factory-like systematic behavior and optimal solution finding.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Path to the engine repo (must exist). Use github_url for remote repos"},
                    "github_url": {"type": "string", "description": "GitHub URL to clone (e.g., https:\\github.com\ggerganov\llama.cpp). SECURITY: repo is read-only; no build scripts executed"},
                    "source_arch": {"type": "string", "description": "Source arch ID (e.g., 'sm_80', 'gfx1100'). 'auto' = auto-detect from repo", "default": "auto"},
                    "target_arch": {"type": "string", "description": "Target arch ID (e.g., 'gfx1201'). 'auto' = suggest based on installed hardware", "default": "auto"},
                    "project_type": {"type": "string", "description": "Type of project: 'llm', 'game', 'sdk', 'compute', 'inference', or 'auto'", "default": "auto"},
                    "target_language": {"type": "string", "description": "Target language: 'cxx', 'c', 'rust', 'zig', 'goose', 'python', 'csharp', or 'auto' = analyze repo for optimal choice", "default": "auto"},
                    "target_backend": {"type": "string", "description": "Target backend: 'hip', 'cuda', 'vulkan', 'directx', 'sycl', or 'auto'", "default": "auto"},
                    "target_os": {"type": "string", "description": "Target OS: 'linux', 'windows', 'auto'", "default": "auto"},
                    "target_tok_per_s": {"type": "number", "description": "Performance target for LLM workloads (0 = no target)", "default": 0},
                    "mode": {"type": "string", "description": "Porting mode: 'strict' or 'loose'", "default": "strict"},
                    "max_files": {"type": "integer", "description": "Maximum number of files to scan", "default": 1000},
                    "force_refresh": {"type": "boolean", "description": "Force re-analysis (bypass cache)", "default": False},
                    "break_self_check": {"type": "boolean", "description": "Bypass Self-Check Gate violations", "default": False},
                    "strip_debug": {"type": "integer", "description": "Strip debug tracing: 0=keep, 1=comment-out (compile-time), 2=remove from source", "default": 0},
                    "strip_perf": {"type": "integer", "description": "Strip perf tracing: 0=keep, 1=comment-out (compile-time), 2=remove from source", "default": 0},
                    "timeout_seconds": {"type": "integer", "description": "Timeout for clone + scan operations (seconds)", "default": 60},
                    "enable_self_optimization": {"type": "boolean", "description": "Enable autonomous self-optimization", "default": True},
                    "ecc_integration_level": {"type": "string", "description": "ECC integration level: 'basic', 'deep', 'full'", "default": "deep"},
                    "factory_mode": {"type": "boolean", "description": "Enable factory-like systematic behavior", "default": True},
                    "edge_case_mitigation": {"type": "boolean", "description": "Enable enhanced edge case detection and mitigation", "default": True},
                    "multi_option_evaluation": {"type": "boolean", "description": "Enable multi-option evaluation and comparison", "default": True},
                    "continuous_learning": {"type": "boolean", "description": "Enable continuous learning and adaptation", "default": True},
                },
                "required": [],
            },
        ),
        mcp_types.Tool(
            name="enhanced_porting_modes",
            description="ENHANCED PORTING MODES: Specialized optimization modes for different use cases - auto, performance, safety, migration, benchmark, batch, interactive. Each mode provides tailored optimizations and configurations for specific porting scenarios.",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "description": "Porting mode: auto, performance, safety, migration, benchmark, batch, interactive", "default": "auto"},
                    "repo_path": {"type": "string", "description": "Path to the engine repo"},
                    "github_url": {"type": "string", "description": "GitHub URL to clone"},
                    "source_arch": {"type": "string", "description": "Source arch ID", "default": "auto"},
                    "target_arch": {"type": "string", "description": "Target arch ID", "default": "auto"},
                    "target_backend": {"type": "string", "description": "Target backend", "default": "auto"},
                    "target_os": {"type": "string", "description": "Target OS", "default": "auto"},
                    "target_tok_per_s": {"type": "number", "description": "Performance target", "default": 0},
                    "mode_params": {"type": "object", "description": "Mode-specific parameters", "default": {}},
                },
                "required": [],
            },
        ),
        mcp_types.Tool(
            name="add_backend",
            description="AUTOMATED BACKEND ADDITION: Add a new GPU backend with full toolchain detection, shim generation, architecture support, and test suite creation. Supports hip, cuda, vulkan, directx, sycl, metal, opencl.",
            inputSchema={
                "type": "object",
                "properties": {
                    "backend_name": {"type": "string", "description": "Name of the new backend to add"},
                    "backend_config": {"type": "object", "description": "Backend configuration including compiler, extensions, arch_patterns, feature_flags", "default": {}},
                },
                "required": ["backend_name"],
            },
        ),
        mcp_types.Tool(
            name="detect_toolchain",
            description="AUTOMATED TOOLCHAIN DETECTION: Detect and validate toolchain for any backend using multiple strategies (environment variables, PATH search, registry, package managers, SDK discovery).",
            inputSchema={
                "type": "object",
                "properties": {
                    "backend_name": {"type": "string", "description": "Backend name to detect toolchain for"},
                    "os_type": {"type": "string", "description": "Target OS: auto, linux, windows, macos", "default": "auto"},
                },
                "required": ["backend_name"],
            },
        ),
        mcp_types.Tool(
            name="convert_language",
            description="AUTOMATED LANGUAGE CONVERSION: Convert code between programming languages (cuda↔rust, cpp↔zig, python↔c, opencl↔sycl, hip↔cuda, vulkan↔directx) with semantic preservation and validation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_code": {"type": "string", "description": "Source code to convert"},
                    "source_lang": {"type": "string", "description": "Source language"},
                    "target_lang": {"type": "string", "description": "Target language"},
                },
                "required": ["source_code", "source_lang", "target_lang"],
            },
        ),
        mcp_types.Tool(
            name="migrate_platform",
            description="AUTOMATED PLATFORM MIGRATION: Migrate project between platforms (windows↔linux, macos↔linux, embedded↔desktop) with compatibility shims and path adaptation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Path to project to migrate"},
                    "source_platform": {"type": "string", "description": "Source platform: windows, linux, macos, embedded"},
                    "target_platform": {"type": "string", "description": "Target platform: windows, linux, macos, embedded"},
                },
                "required": ["project_path", "source_platform", "target_platform"],
            },
        ),
        mcp_types.Tool(
            name="integrate_build_system",
            description="AUTOMATED BUILD SYSTEM INTEGRATION: Integrate with various build systems (cmake, make, meson, bazel, ninja, scons) with backend-specific configuration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Path to project"},
                    "build_system": {"type": "string", "description": "Build system: cmake, make, meson, bazel, ninja, scons"},
                    "backend_config": {"type": "object", "description": "Backend configuration", "default": {}},
                },
                "required": ["project_path", "build_system"],
            },
        ),
        mcp_types.Tool(
            name="optimize_performance",
            description="ADVANCED PERFORMANCE OPTIMIZATION: Apply comprehensive performance optimizations including instruction selection, memory layout, kernel fusion, register optimization, and more.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_code": {"type": "string", "description": "Source code to optimize"},
                    "target_arch": {"type": "string", "description": "Target architecture ID"},
                    "target_backend": {"type": "string", "description": "Target backend"},
                },
                "required": ["source_code", "target_arch", "target_backend"],
            },
        ),
        mcp_types.Tool(
            name="handle_error_recovery",
            description="COMPREHENSIVE ERROR RECOVERY: Handle errors with multiple recovery strategies including retry, rollback, fallback, and graceful degradation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "error_message": {"type": "string", "description": "Error message to handle"},
                    "context": {"type": "object", "description": "Context for error recovery", "default": {}},
                    "recovery_level": {"type": "string", "description": "Recovery level: automatic, manual, hybrid", "default": "automatic"},
                },
                "required": ["error_message"],
            },
        ),
        mcp_types.Tool(
            name="comprehensive_porting",
            description="COMPREHENSIVE PORTING INTEGRATION: Execute complete porting workflow with language conversion, platform migration, build integration, performance optimization, and error recovery.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_code": {"type": "string", "description": "Source code to port"},
                    "source_lang": {"type": "string", "description": "Source language"},
                    "target_lang": {"type": "string", "description": "Target language"},
                    "source_platform": {"type": "string", "description": "Source platform"},
                    "target_platform": {"type": "string", "description": "Target platform"},
                    "target_backend": {"type": "string", "description": "Target backend"},
                    "build_system": {"type": "string", "description": "Build system to use", "default": "cmake"},
                    "mode": {"type": "string", "description": "Porting mode", "default": "auto"},
                },
                "required": ["source_code", "source_lang", "target_lang", "source_platform", "target_platform", "target_backend"],
            },
        ),
        mcp_types.Tool(
            name="capability_report",
            description="Report what THIS host can actually build for a target (compiler, root, disassembler, supported backends). Use before planning any port: probes discovery instead of assuming a toolchain exists. Vendor-agnostic (hip\cuda\sycl).",
            inputSchema={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Target arch id, e.g. 'gfx1201', 'sm_80', 'spir64'"},
                    "backend": {"type": "string", "description": "hip\cuda\sycl\auto (auto infers from target)", "default": "auto"},
                },
                "required": ["target"],
            },
        ),
        mcp_types.Tool(
            name="probe_isa",
            description="Compile one or more ISA candidates FOR THE TARGET ARCH and report which actually encode. Each candidate gets its own translation unit so one failure cannot mask the others. This is the only reliable oracle for 'does this instruction exist on this chip' — docs and capability tables lie. Returns per-candidate ok\error plus the disassembled mnemonics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Target arch id, e.g. 'gfx1201', 'sm_80'"},
                    "candidates": {
                        "type": "array",
                        "description": "List of {name, stmt} objects. stmt is the C\C++ statement to test.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "stmt": {"type": "string"},
                                "prelude": {"type": "string"},
                            },
                            "required": ["stmt"],
                        },
                    },
                    "backend": {"type": "string", "default": "auto"},
                },
                "required": ["target", "candidates"],
            },
        ),
        mcp_types.Tool(
            name="scan_residual_asm",
            description="Find live inline-asm strings that name a foreign\unsupported mnemonic (e.g. NVIDIA PTX left in an AMD port, or AMD mnemonics in a CUDA port). Only matches asm(\"...\") string literals, so comments do not false-positive. Run this as a hard gate before every build.",
            inputSchema={
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}, "description": "Files to scan"},
                    "backend": {"type": "string", "description": "Selects the default forbidden-mnemonic set", "default": "hip"},
                    "mnemonics": {"type": "array", "items": {"type": "string"}, "description": "Override the forbidden set"},
                },
                "required": ["files"],
            },
        ),
        mcp_types.Tool(
            name="build_module",
            description="Build a device module for a target: deletes the output first (a stale artifact must never look like success), compiles the concatenated sources, then reports size and exported kernel symbols. Optionally diffs symbols against an expected list.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sources": {"type": "array", "items": {"type": "string"}, "description": "Source files, in the runtime's own concatenation order"},
                    "target": {"type": "string", "description": "Target arch id"},
                    "out": {"type": "string", "description": "Output module path"},
                    "backend": {"type": "string", "default": "auto"},
                    "includes": {"type": "array", "items": {"type": "string"}},
                    "flags": {"type": "array", "items": {"type": "string"}},
                    "concatenate": {"type": "boolean", "default": True},
                    "expected_symbols": {"type": "array", "items": {"type": "string"}, "description": "If given, report missing\unexpected kernel symbols"},
                },
                "required": ["sources", "target", "out"],
            },
        ),
        mcp_types.Tool(
            name="code_object_report",
            description=("ISA and residency budget for a compiled kernel artifact. "
                         "Unbundles a container (clang offload bundle \ fatbinary) and "
                         "selects the right target from it, then reports per kernel: "
                         "SIMD width, register counts (VGPR\SGPR or registers), spills, "
                         "scratch memory per block, how many blocks that allows to be "
                         "resident per compute unit, and the instruction mix (global "
                         "loads with widths, shared-memory ops, bit ops, dot ops) with "
                         "verdicts. Use before tuning: a kernel can be pinned at one "
                         "block per CU, or spending its issue slots decoding, and "
                         "neither shows up in timing counters. Vendor-neutral in what "
                         "it asks; the AMD\LLVM path is implemented, and the CUDA "
                         "equivalents are reported when no LLVM tooling is present."),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "Compiled kernel artifact (.hsaco\.cubin\.co\.elf\.metallib), bundle or raw ELF"},
                    "arch": {"type": "string",
                             "description": "Select this target from a multi-arch bundle, e.g. 'gfx1201' or 'sm_90'"},
                    "lds_kib": {"type": "number",
                                "description": "Usable scratch memory per compute unit in KiB (measured, not datasheet) to compute residency"},
                },
                "required": ["path"],
            },
        ),
        mcp_types.Tool(
            name="build_porting_map",
            description="Read a source tree and emit a function-to-function porting map for a target: every kernel\device function with its role, the constructs it uses (inline asm mnemonics, warp intrinsics, ISA features), and the target-specific substitution with an explicit fidelity grade (exact\portable\approximate\lossy\manual\drop). ISA constructs inside macros are attributed to the functions that call them. Answers 'what do I replace this with on this chip' from the source, not from memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}, "description": "Source files to analyse"},
                    "source_arch": {"type": "string", "description": "Origin arch id, e.g. 'sm_80'"},
                    "target_arch": {"type": "string", "description": "Destination arch id, e.g. 'gfx1201'"},
                    "format": {"type": "string", "enum": ["markdown", "json", "both"], "default": "both"},
                },
                "required": ["files", "target_arch"],
            },
        ),
        mcp_types.Tool(
            name="lookup_construct",
            description="What should one source construct become on a target arch? Returns the replacement, its fidelity grade and any caveat (e.g. a raw instruction that has no encoding on this chip). Use this before writing any replacement by hand.",
            inputSchema={
                "type": "object",
                "properties": {
                    "construct": {"type": "string", "description": "Construct id, e.g. 'mma.sync', 'dp4a', 'shfl_sync', or any asm mnemonic"},
                    "target_arch": {"type": "string", "description": "Target arch id"},
                },
                "required": ["construct", "target_arch"],
            },
        ),
        mcp_types.Tool(
            name="verify_porting_map",
            description="Close the loop: take a porting map, PROBE every mapped replacement on the real target with the actual compiler, and fold the results back in. Constructs whose replacement does not encode are downgraded to 'manual'. Distinguishes probing the replacement (correct) from probing the source instruction (which would wrongly condemn portable mappings like bar.sync -> __syncthreads).",
            inputSchema={
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}},
                    "target_arch": {"type": "string"},
                    "source_arch": {"type": "string", "default": "auto"},
                    "backend": {"type": "string", "default": "auto"},
                },
                "required": ["files", "target_arch"],
            },
        ),
        mcp_types.Tool(
            name="build_function_reference",
            description="PRE-PORT TASK. Scan a source tree and produce the CUDA\vendor function reference for a target: every runtime API, driver API, inline-PTX instruction, device intrinsic and framework feature the code actually uses, each with its target replacement, an auto\partial\manual\no-equivalent status, the call sites and the files that depend on it. Status is resolved per architecture (e.g. mma.sync is manual on all AMD, but MFMA vs WMMA vs no-matrix-core changes at RDNA2\RDNA3\RDNA4\CDNA). Features with no equivalent are reported as hard blockers that must be resolved before any kernel work begins. Run this BEFORE designing the port: it converts 'where do I start' into a work list.",
            inputSchema={
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}, "description": "Source files to scan"},
                    "source_arch": {"type": "string", "description": "Origin arch id, e.g. 'sm_120a'"},
                    "target_arch": {"type": "string", "description": "Destination arch id, e.g. 'gfx1201'"},
                    "format": {"type": "string", "enum": ["markdown", "json", "both"], "default": "markdown"},
                },
                "required": ["files", "target_arch"],
            },
        ),
        mcp_types.Tool(
            name="lookup_api",
            description="What replaces one CUDA API, PTX instruction, intrinsic or feature on a target arch, and is it automatic or manual? Resolves per architecture, so the same symbol gives different answers on RDNA2 vs RDNA4 vs CDNA. Use when reviewing a line of code during the pre-port inventory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api": {"type": "string", "description": "e.g. 'cudaMalloc', 'cp.async.wait_group', '__shfl_xor_sync', 'TMA (Tensor Memory Accelerator)'"},
                    "target_arch": {"type": "string"},
                },
                "required": ["api", "target_arch"],
            },
        ),
        mcp_types.Tool(
            name="analyze_full_pipeline",
            description="Analyze a source tree for its full inference pipeline stages (engine init → model load → prompt → decode → attention → KV cache → output). Reports each stage's dispatch model, memory pattern, and porting-critical constructs. General-purpose: works with any engine (llama.cpp, tinyvllm, Kindi, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Path to the engine source tree"},
                    "target_arch": {"type": "string", "description": "Target arch ID (e.g. 'gfx1201', 'sm_80') to score constructs against", "default": ""},
                    "source_arch": {"type": "string", "description": "Source arch ID (e.g. 'sm_80', 'sm_30') for comparison", "default": ""},
                },
                "required": ["repo"],
            },
        ),
        mcp_types.Tool(
            name="analyze_core_engine",
            description="Analyze an engine's core dispatch architecture: kernel launch strategy, memory management pattern, tensor-parallelism\multi-GPU support, weight layout handling, and host↔device interface. Produces a structured map of how the engine orchestrates its kernels. General-purpose: works with any engine.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Path to the engine source tree"},
                    "target_arch": {"type": "string", "description": "Target arch ID (e.g. 'gfx1201', 'sm_80') to evaluate dispatch constraints against", "default": ""},
                },
                "required": ["repo"],
            },
        ),
    ]


def _make_list_tools_handler():
    """Return an async handler for tools\list."""
    async def _handler(request, params):
        return mcp_types.ListToolsResult(tools=_tools(), nextCursor=None)
    return _handler


def _make_call_tool_handler():
    """Return an async handler for tools\call."""
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
                results.append(mcp_types.TextContent(type="text", text=f"No GPU\kernel\shader sources found in {repo}"))
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
                summary += f"\n**Total: {len(findings)} kernel\shader files**\n"
                results.append(mcp_types.TextContent(type="text", text=summary))
                results.append(mcp_types.TextContent(type="text", text=json.dumps(findings, indent=2)))

                # Optional readiness diagnosis: what the target backend needs before
                # any kernel is edited. Rows are probed, ranked, and given a fallback;
                # an unprobed row stays UNVALIDATED rather than passing silently.
                target_backend = str(arguments.get("target_backend") or "").strip()
                if target_backend:
                    diag = backend_profile.diagnose(
                        target_backend,
                        target_arch=str(arguments.get("target_arch") or ""),
                        host_os=str(arguments.get("host_os") or ""),
                        findings=findings,
                        probe_host=bool(arguments.get("probe_host", True)),
                    )
                    results.append(mcp_types.TextContent(
                        type="text", text=backend_profile.render(diag)))
                    results.append(mcp_types.TextContent(
                        type="text", text=json.dumps(backend_profile.slim(diag), indent=2)))

    elif name == "check_arch_compatibility":
        kernel_list = arguments.get("kernel_list", [])
        arch_id = arguments.get("arch", "")
        entries = arch_map.get("entries", {})
        arch_info = entries.get(arch_id, {})
        if not arch_info:
            results.append(mcp_types.TextContent(type="text", text=f"Architecture '{arch_id}' not found in ARCH_MAP.json"))
        else:
            is_cpu = arch_info.get("cpu", False)
            report = f"# Architecture: {arch_id} ({arch_info.get('name', '?')})\n\n"
            report += f"**Type:** {'CPU' if is_cpu else 'GPU'}\n\n"
            report += "| Feature | Value |\n|---|---|\n"
            # GPU features
            for feat in ["wave_size", "lds_kib", "tensor_cores", "wmma", "dot4", "dot8", "bf16_native", "fp8_native"]:
                if feat in arch_info:
                    report += f"| {feat} | {arch_info.get(feat, 'N\A')} |\n"
            # CPU features
            cpu_feats = arch_info.get("features", {})
            if cpu_feats:
                report += "\n### CPU SIMD Features\n\n"
                report += "| Feature | Available |\n|---|---|\n"
                for k in ["sse2", "sse3", "sse4_1", "sse4_2", "avx", "avx2", "avx512", "amx", "fma3",
                          "neon", "sve", "sve2", "aes", "sha", "fp16_native", "apx"]:
                    if k in cpu_feats:
                        report += f"| {k} | {cpu_feats[k]} |\n"
            report += "\n## Kernel Compatibility:\n\n"
            for kern in kernel_list:
                status = _check_kernel_against_arch(kern, arch_info)
                icon = {"OK": "✅", "NEEDS_FALLBACK": "⚠️", "INCOMPATIBLE": "❌"}[status]
                report += f"- {icon} `{kern}`: {status}\n"
            if arch_info.get("known_issues"):
                report += "\n## Known Issues:\n"
                for issue in arch_info["known_issues"]:
                    report += f"- {issue}\n"
            if arch_info.get("porting_notes"):
                report += "\n## Porting Notes:\n"
                for k, v in arch_info["porting_notes"].items():
                    report += f"- **{k}:** {v}\n"
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
            is_a_cpu = a.get("cpu", False)
            is_b_cpu = b.get("cpu", False)
            report = f"# Comparing {arguments['arch_a']} ({a.get('name')}) vs {arguments['arch_b']} ({b.get('name')})\n\n"
            report += f"**A ({arguments['arch_a']}):** {'CPU' if is_a_cpu else 'GPU'}\n"
            report += f"**B ({arguments['arch_b']}):** {'CPU' if is_b_cpu else 'GPU'}\n\n"

            if is_a_cpu and is_b_cpu:
                # CPU-to-CPU SIMD feature comparison
                fa = a.get("features", {})
                fb = b.get("features", {})
                all_feats = sorted(set(fa.keys()) | set(fb.keys()))
                report += "## CPU SIMD Features Comparison\n\n"
                report += "| Feature | A | B | Gap |\n|---|---|---|---|\n"
                for feat in all_feats:
                    va = fa.get(feat, False)
                    vb = fb.get(feat, False)
                    gap = "SAME" if va == vb else ("LOST" if va and not vb else "GAINED")
                    report += f"| {feat} | {va} | {vb} | {gap} |\n"
                report += "\n## Porting Implications (CPU -> CPU)\n\n"
                lost = [f for f in all_feats if fa.get(f) and not fb.get(f)]
                gained = [f for f in all_feats if not fa.get(f) and fb.get(f)]
                if lost:
                    report += f"- LOST on target: {', '.join(lost)} -- CPU kernels using these need scalar or lower-SIMD fallback\n"
                if gained:
                    report += f"- GAINED on target: {', '.join(gained)}\n"
                common_cpu_feats = ["sse2", "avx2", "avx512", "neon", "sve2", "amx", "aes", "sha"]
                report += "\n## Shared CPU SIMD Features\n\n"
                for k in common_cpu_feats:
                    va = fa.get(k, False)
                    vb = fb.get(k, False)
                    report += f"- {k}: A={va}, B={vb}\n"
            else:
                # GPU or mixed comparison
                report += "| Feature | A | B | Gap |\n|---|---|---|---|\n"
                for k in ["wave_size", "lds_kib", "tensor_cores", "wmma", "dot4", "dot8", "bf16_native", "fp8_native"]:
                    va = a.get(k, "?"); vb = b.get(k, "?")
                    gap = "SAME" if str(va) == str(vb) else "DIFF"
                    report += f"| {k} | {va} | {vb} | {gap} |\n"
                if is_a_cpu != is_b_cpu:
                    report += "\n## Mixed CPU\GPU Comparison\n\n"
                    if is_a_cpu and not is_b_cpu:
                        report += f"A is CPU ({arguments['arch_a']}), B is GPU ({arguments['arch_b']}).\n"
                    else:
                        report += f"A is GPU ({arguments['arch_a']}), B is CPU ({arguments['arch_b']}).\n"
                    report += "CPU SIMD does NOT map to GPU features. CPU kernels use host SIMD; GPU kernels use device intrinsics.\n"

            report += f"\n## Issues on {arguments['arch_a']}:\n"
            for issue in a.get("known_issues", []):
                report += f"- {issue}\n"
            report += f"\n## Issues on {arguments['arch_b']}:\n"
            for issue in b.get("known_issues", []):
                report += f"- {issue}\n"

            report += "\n## Porting Implications:\n"
            if not is_a_cpu and not is_b_cpu:
                # Original GPU implications
                if a.get("wmma") and not b.get("wmma"):
                    report += f"- WMMA kernels on {arguments['arch_a']} need F32 fallback on {arguments['arch_b']}\n"
                if a.get("bf16_native") and not b.get("bf16_native"):
                    report += f"- BF16-native kernels on {arguments['arch_a']} need emulation on {arguments['arch_b']}\n"
                if a.get("dot8") and not b.get("dot8"):
                    report += f"- Dot8 kernels on {arguments['arch_a']} need fallback on {arguments['arch_b']}\n"
                if a.get("wmma") and not b.get("wmma"):
                    report += f"- WMMA kernels need F32 fallback on {arguments['arch_b']}\n"
            elif is_a_cpu and is_b_cpu:
                fa = a.get("features", {})
                fb = b.get("features", {})
                if fa.get("avx512") and not fb.get("avx512"):
                    report += f"- AVX-512 kernels need AVX2 fallback on {arguments['arch_b']}\n"
                if fa.get("amx") and not fb.get("amx"):
                    report += f"- AMX kernels need AVX2\scalar fallback on {arguments['arch_b']}\n"
                if not fa.get("sve2") and fb.get("sve2"):
                    report += f"- Target {arguments['arch_b']} has SVE2 \u2192 can add SVE2 kernels\n"
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
            if not params.github_url.startswith("https:\\github.com\") and not params.github_url.startswith("git+https:\\github.com\"):
                results.append(mcp_types.TextContent(type="text", text="ERROR: github_url must be https:\\github.com\ or git+https:\\github.com\ (arbitrary git URLs blocked)"))
                return results
            security_warnings.append("SECURITY: Repo cloned READ-ONLY. No build scripts or git hooks executed.")
            security_warnings.append(f"SECURITY: File count capped at {params.max_files}, timeout {params.timeout_seconds}s.")
        if not params.repo_path and not params.github_url:
            results.append(mcp_types.TextContent(type="text", text="ERROR: Must provide repo_path or github_url"))
            return results

        # ── Optimal-path auto-selection (no prompting) ──
        # When target_arch or target_backend is "auto", pick the single best path
        # by scoring installed toolchains + source kernel features + target arch generation.
        # Falls through to the full _autonomous_port report below with the chosen path.
        detected_info: dict = {}  # set below; _autonomous_port reads it
        if params.target_arch == "auto" or params.target_backend == "auto":
            detected_tools = _detect_installed_toolchain("auto", "auto")
            det = detected_tools.get("detected", {})
            source_info = arch_map.get("entries", {}).get(params.source_arch, {})
            source_vendor = source_info.get("vendor", "NVIDIA")

            # Kernel names from the source tree. The scan used to live only
            # inside _autonomous_port (which runs AFTER this block), which made
            # `gpu_kernels` an UnboundLocalError here — autonomous_port with
            # target=auto hard-failed before producing any report. Scan now.
            _scan_root = Path(params.repo_path) if params.repo_path else None
            if _scan_root and _scan_root.exists():
                _all_kernels = _scan_sources(_scan_root)
                _kernel_names = [k["path"].split("\")[-1]
                                 .replace(".cu", "").replace(".cpp", "")
                                 .replace(".comp", "").replace(".c", "")
                                 .replace(".cc", "").replace(".cxx", "")
                                 for k in _all_kernels]
            else:
                _kernel_names = []  # github_url clone — no local tree to scan

            def _has_real_compiler(tk: dict) -> bool:
                """True only when a detection field names an existing binary.

                Detection fields may be: a path str, None, or a dict like
                {"exists": False} \ {"path": ..., "exists": True}. The dict
                form is truthy even when exists is False, so a plain `or`
                chain false-credits toolchains with no compiler (audit F1:
                hip scored +25 from windows_path={"exists": False})."""
                for key in ("hipcc", "nvcc", "glslc", "icpx",
                            "path", "windows_path", "linux_path"):
                    v = tk.get(key)
                    if isinstance(v, dict):
                        if v.get("exists") and v.get("path"):
                            return True
                    elif v:  # non-empty string
                        return True
                return False

            backend_scores: dict[str, int] = {}
            backend_compiler: dict[str, bool] = {}
            for bk in ("hip", "cuda", "vulkan", "directx", "sycl"):
                score = 0
                tk = det.get({"hip":"rocm","cuda":"cuda","vulkan":"vulkan","directx":"directx","sycl":"oneapi"}[bk], {})
                has_compiler = _has_real_compiler(tk)
                backend_compiler[bk] = has_compiler
                if has_compiler:
                    score += 25
                if isinstance(tk.get("env_var"), dict) and tk["env_var"].get("exists"):
                    score += 10
                if source_vendor == "NVIDIA" and bk == "hip":
                    score += 15
                elif source_vendor == "AMD" and bk == "cuda":
                    score += 15
                elif source_vendor != "AMD" and bk != "hip":
                    score += 5
                src_feats = source_info.get("features", source_info)
                if src_feats.get("wmma") and bk in ("hip","cuda","sycl"):
                    score += 8
                if src_feats.get("bf16_native") and bk in ("hip","cuda","sycl"):
                    score += 5
                backend_scores[bk] = score

            # Rank: score first, then prefer backends with a real compiler on
            # ties (audit F1: dict-order tie-break let compiler-less hip beat
            # a working vulkan toolchain).
            chosen_backend = max(backend_scores,
                                 key=lambda b: (backend_scores[b],
                                                backend_compiler[b]))

            cross_scores: list[tuple[int, str, str, int, int, int]] = []
            same_scores: list[tuple[int, str, str, int, int, int]] = []
            for aid, einfo in arch_map.get("entries", {}).items():
                feats = einfo.get("features", einfo)
                ok, fb, inc = _score_arch_kernels(_kernel_names, feats)
                feat_cnt = sum(1 for f in ("wmma","bf16_native","dot4","tensor_cores") if feats.get(f))
                gen_num = _arch_gen_number(einfo.get("name", ""))
                s = ok*10 - fb*3 + feat_cnt*2 + gen_num
                rec = (s, aid, einfo.get("name", aid), ok, fb, inc)
                if einfo.get("vendor", "").lower() == source_vendor.lower():
                    same_scores.append(rec)
                else:
                    cross_scores.append(rec)
            cross_scores.sort(reverse=True)
            same_scores.sort(reverse=True)
            arch_scores = cross_scores if cross_scores else same_scores

            # chosen_backend selected above (score + real-compiler tie-break)
            chosen_arch = arch_scores[0][1] if arch_scores else params.target_arch
            chosen_info = arch_map.get("entries", {}).get(chosen_arch, {})
            chosen_arch_name = chosen_info.get("name", chosen_arch)

            params.target_backend = chosen_backend
            params.target_arch = chosen_arch
            detected_info["target_backend"] = chosen_backend
            detected_info["target_arch"] = chosen_arch



        report = await _autonomous_port(params, arch_map)
        # Prepend security header if any warnings
        if security_warnings:
            header = "## Security Safeguards\n\n"
            for w in security_warnings:
                header += f"- {w}\n"
            header += "\n"
            report = report.replace("# Autonomous", "# Autonomous", 1)
            report = report.replace("\n---\n", "\n" + header + "---\n", 1)

        # Save report to working folder
        import datetime as _dt
        repo_dir = Path(params.repo_path or params.github_url or ".")
        findings_dir = repo_dir \ "findings"
        findings_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_lang = (params.target_language or "auto").replace(" ", "-").lower()
        safe_backend = params.target_backend or "auto"
        report_file = findings_dir \ f"porting-report-{safe_lang}-{safe_backend}-{timestamp}.md"
        try:
            report_file.write_text(report, encoding="utf-8")
            report += f"\n\n---\n**Report saved to:** `{report_file}`\n"
        except Exception as e:
            report += f"\n\n---\n**Could not save report:** {e}\n"

        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "enhanced_autonomous_port":
        # Import enhanced server functions
        try:
            from mcp.enhanced_server import enhanced_autonomous_port as enhanced_port_func
            from mcp.enhanced_server import AutonomousPortParams as EnhancedParams
        except ImportError:
            # Fallback for running from mcp directory
            from enhanced_server import enhanced_autonomous_port as enhanced_port_func
            from enhanced_server import AutonomousPortParams as EnhancedParams
        
        # Parse arguments
        params = EnhancedParams(**arguments)
        
        # Run enhanced autonomous porting
        result = await enhanced_port_func(params)
        
        # Format result as text
        report = "# Enhanced Autonomous Porting Report\n\n"
        report += f"**Status:** {result.get('status', 'unknown')}\n"
        report += f"**Repository:** {result.get('port_request', {}).get('repo_path', 'N\A')}\n"
        report += f"**Target Arch:** {result.get('port_request', {}).get('target_arch', 'N\A')}\n"
        report += f"**Target Backend:** {result.get('port_request', {}).get('target_backend', 'N\A')}\n"
        report += f"**Mode:** {result.get('port_request', {}).get('mode', 'N\A')}\n\n"
        
        if result.get('status') == 'success':
            report += "## ✅ Enhanced Porting Completed Successfully\n\n"
            report += f"**Optimal Strategy:** {result.get('optimal_strategy_selected', {}).get('approach', 'N\A')}\n"
            report += f"**Estimated Performance:** {result.get('optimal_strategy_selected', {}).get('estimated_performance', 0)} tok\s\n"
            report += f"**Confidence:** {result.get('optimal_strategy_selected', {}).get('confidence', 0):.2f}\n"
            report += f"**Risk Level:** {result.get('optimal_strategy_selected', {}).get('risk_level', 0):.2f}\n\n"
            
            report += "## Factory Metrics\n\n"
            factory_metrics = result.get('factory_metrics', {})
            for key, value in factory_metrics.items():
                report += f"- **{key.replace('_', ' ').title()}:** {value}\n"
            report += "\n"
            
            report += "## Recommendations\n\n"
            recommendations = result.get('recommendations', {})
            for category, items in recommendations.items():
                report += f"### {category.replace('_', ' ').title()}\n"
                if isinstance(items, list):
                    for item in items:
                        report += f"- {item}\n"
                report += "\n"
        else:
            report += "## ❌ Enhanced Porting Failed\n\n"
            report += f"**Error:** {result.get('error', 'Unknown error')}\n"
            report += f"**Error Type:** {result.get('error_type', 'Unknown')}\n"
            report += f"**Phase:** {result.get('phase', 'Unknown')}\n"
        
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "enhanced_porting_modes":
        # Import enhanced porting modes
        try:
            from mcp.enhanced_porting_modes import enhanced_porting_modes as enhanced_modes_func
            from mcp.enhanced_porting_modes import PortingModeParams as ModeParams
        except ImportError:
            from enhanced_porting_modes import enhanced_porting_modes as enhanced_modes_func
            from enhanced_porting_modes import PortingModeParams as ModeParams
        
        # Parse arguments
        params = ModeParams(**arguments)
        
        # Run enhanced porting modes
        result = await enhanced_modes_func(params)
        
        # Format result as text
        report = "# Enhanced Porting Modes Report\n\n"
        report += f"**Mode Requested:** {result.get('mode_requested', 'N\A')}\n"
        report += f"**Mode Executed:** {result.get('mode_executed', 'N\A')}\n"
        report += f"**Status:** {result.get('status', 'unknown')}\n\n"
        
        if result.get('status') == 'success':
            report += "## ✅ Enhanced Porting Mode Completed Successfully\n\n"
            
            # Mode-specific results
            mode_result = result.get('result', {})
            if 'optimizations_applied' in mode_result:
                report += "### Optimizations Applied\n"
                for opt in mode_result['optimizations_applied']:
                    report += f"- {opt}\n"
                report += "\n"
            
            if 'safety_checks_applied' in mode_result:
                report += "### Safety Checks Applied\n"
                for check in mode_result['safety_checks_applied']:
                    report += f"- {check}\n"
                report += "\n"
            
            if 'config' in mode_result:
                report += "### Configuration\n"
                for key, value in mode_result['config'].items():
                    report += f"- **{key.replace('_', ' ').title()}:** {value}\n"
                report += "\n"
            
            report += f"**Estimated Performance Gain:** {result.get('estimated_performance_gain', 'N\A')}\n"
            report += f"**Validation Level:** {result.get('validation_level', 'standard')}\n\n"
            
            # Recommendations
            recommendations = result.get('recommendations', {})
            if 'next_steps' in recommendations:
                report += "### Next Steps\n"
                for step in recommendations['next_steps']:
                    report += f"- {step}\n"
                report += "\n"
            
            if 'mode_specific_tips' in recommendations:
                report += "### Mode-Specific Tips\n"
                for tip in recommendations['mode_specific_tips']:
                    report += f"- {tip}\n"
                report += "\n"
        else:
            report += "## ❌ Enhanced Porting Mode Failed\n\n"
            report += f"**Error:** {result.get('error', 'Unknown error')}\n"
            report += f"**Error Type:** {result.get('error_type', 'Unknown')}\n"
        
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "add_backend":
        # Import backend addition wizard
        try:
            from mcp.backend_addition_wizard import BackendAdditionWizard
        except ImportError:
            from backend_addition_wizard import BackendAdditionWizard
        
        wizard = BackendAdditionWizard()
        backend_name = arguments.get("backend_name", "")
        backend_config = arguments.get("backend_config", {})
        
        result = await wizard.add_new_backend(backend_name, backend_config)
        
        report = "# Backend Addition Report\n\n"
        report += f"**Backend:** {backend_name}\n"
        report += f"**Status:** {result.get('status', 'unknown')}\n\n"
        
        if result.get('status') == 'success':
            report += "## ✅ Backend Added Successfully\n\n"
            report += f"**Steps Completed:** {result.get('steps_completed', 0)}\n"
            report += "**Backend is now ready for use!**\n"
        else:
            report += "## ❌ Backend Addition Failed\n\n"
            report += f"**Error:** {result.get('error', 'Unknown error')}\n"
            report += f"**Failed Step:** {result.get('step', 'Unknown')}\n"
        
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "detect_toolchain":
        # Import toolchain detector
        try:
            from mcp.backend_addition_wizard import ToolchainDetector
        except ImportError:
            from backend_addition_wizard import ToolchainDetector
        
        detector = ToolchainDetector()
        backend_name = arguments.get("backend_name", "")
        os_type = arguments.get("os_type", "auto")
        
        result = detector.detect_backend_toolchain(backend_name, os_type)
        
        report = "# Toolchain Detection Report\n\n"
        report += f"**Backend:** {backend_name}\n"
        report += f"**OS Type:** {os_type}\n"
        report += f"**Found:** {result.get('found', False)}\n"
        
        if result.get('found', False):
            report += f"**Best Path:** {result.get('best_path', 'N\A')}\n"
            report += f"**Sources:** {', '.join(result.get('sources', []))}\n"
            report += f"**Confidence:** {result.get('confidence', 0):.2f}\n"
        else:
            report += "**Toolchain not found on system**\n"
        
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "convert_language":
        # Import language converter
        try:
            from mcp.language_converter import LanguageConverter
        except ImportError:
            from language_converter import LanguageConverter
        
        converter = LanguageConverter()
        source_code = arguments.get("source_code", "")
        source_lang = arguments.get("source_lang", "")
        target_lang = arguments.get("target_lang", "")
        
        result = converter.convert_code(source_code, source_lang, target_lang)
        
        report = "# Language Conversion Report\n\n"
        report += f"**Source Language:** {source_lang}\n"
        report += f"**Target Language:** {target_lang}\n"
        report += f"**Status:** {'Success' if result.success else 'Failed'}\n\n"
        
        if result.success:
            report += "## ✅ Conversion Successful\n\n"
            report += "### Converted Code\n```\n"
            report += result.converted_code[:500] + ("..." if len(result.converted_code) > 500 else "")
            report += "\n```\n\n"
            
            report += "### Validation Results\n"
            for key, value in result.validation.items():
                status = "✅" if value else "❌"
                report += f"- {status} {key.replace('_', ' ').title()}: {value}\n"
            
            if result.warnings:
                report += "\n### Warnings\n"
                for warning in result.warnings:
                    report += f"- ⚠️ {warning}\n"
        else:
            report += "## ❌ Conversion Failed\n\n"
            for error in result.errors:
                report += f"- {error}\n"
        
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "migrate_platform":
        # Import platform migrator
        try:
            from mcp.platform_migrator import PlatformMigrator
        except ImportError:
            from platform_migrator import PlatformMigrator
        
        import asyncio
        migrator = PlatformMigrator()
        project_path = arguments.get("project_path", "")
        source_platform = arguments.get("source_platform", "")
        target_platform = arguments.get("target_platform", "")
        
        result = await migrator.migrate_platform(project_path, source_platform, target_platform)
        
        report = "# Platform Migration Report\n\n"
        report += f"**Source Platform:** {source_platform}\n"
        report += f"**Target Platform:** {target_platform}\n"
        report += f"**Status:** {result.status}\n\n"
        
        if result.status in ["success", "partial"]:
            report += "## ✅ Migration Completed\n\n"
            report += f"**Steps Completed:** {len(result.steps_completed)}\n"
            report += f"**Files Modified:** {len(result.files_modified)}\n"
            report += f"**Compatibility Shims:** {len(result.compatibility_shims)}\n"
            
            if result.warnings:
                report += "\n### Warnings\n"
                for warning in result.warnings:
                    report += f"- ⚠️ {warning}\n"
        else:
            report += "## ❌ Migration Failed\n\n"
            for error in result.errors:
                report += f"- {error}\n"
        
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "integrate_build_system":
        # Import build system integrator
        try:
            from mcp.build_system_integrator import BuildSystemIntegrator
        except ImportError:
            from build_system_integrator import BuildSystemIntegrator
        
        import asyncio
        integrator = BuildSystemIntegrator()
        project_path = arguments.get("project_path", "")
        build_system = arguments.get("build_system", "cmake")
        backend_config = arguments.get("backend_config", {})
        
        result = await integrator.integrate_build_system(project_path, build_system, backend_config)
        
        report = "# Build System Integration Report\n\n"
        report += f"**Build System:** {build_system}\n"
        report += f"**Status:** {result.status}\n\n"
        
        if result.status == "success":
            report += "## ✅ Build System Integrated\n\n"
            report += f"**Files Created:** {len(result.files_created)}\n"
            report += f"**Files Modified:** {len(result.files_modified)}\n"
            report += f"**Backup Files:** {len(result.backup_files)}\n"
        else:
            report += "## ❌ Build System Integration Failed\n\n"
            for error in result.errors:
                report += f"- {error}\n"
        
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "optimize_performance":
        # Import performance optimizer
        try:
            from mcp.performance_optimizer import PerformanceOptimizer
        except ImportError:
            from performance_optimizer import PerformanceOptimizer
        
        import asyncio
        optimizer = PerformanceOptimizer()
        source_code = arguments.get("source_code", "")
        target_arch = arguments.get("target_arch", "")
        target_backend = arguments.get("target_backend", "")
        
        result = await optimizer.optimize_performance(source_code, target_arch, target_backend)
        
        report = "# Performance Optimization Report\n\n"
        report += f"**Target Arch:** {target_arch}\n"
        report += f"**Target Backend:** {target_backend}\n"
        report += f"**Status:** {result.status}\n\n"
        
        if result.status == "success":
            report += "## ✅ Performance Optimized\n\n"
            report += f"**Optimizations Applied:** {len(result.optimizations_applied)}\n"
            report += f"**Estimated Gain:** {result.estimated_performance_gain}\n\n"
            
            report += "### Optimizations\n"
            for opt in result.optimizations_applied:
                report += f"- **{opt['strategy']}:** {', '.join(opt.get('improvements', []))}\n"
        else:
            report += "## ❌ Performance Optimization Failed\n\n"
            for error in result.errors:
                report += f"- {error}\n"
        
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "handle_error_recovery":
        # Import error recovery system
        try:
            from mcp.error_recovery_system import ErrorRecoverySystem
        except ImportError:
            from error_recovery_system import ErrorRecoverySystem
        
        import asyncio
        recovery = ErrorRecoverySystem()
        error_message = arguments.get("error_message", "")
        context = arguments.get("context", {})
        recovery_level = arguments.get("recovery_level", "automatic")
        
        error = Exception(error_message)
        result = await recovery.handle_error(error, context, recovery_level)
        
        report = "# Error Recovery Report\n\n"
        report += f"**Error:** {error_message}\n"
        report += f"**Recovery Level:** {recovery_level}\n"
        report += f"**Status:** {result.status}\n"
        report += f"**Strategy Used:** {result.strategy_used}\n\n"
        
        if result.status == "recovered":
            report += "## ✅ Error Recovered\n\n"
            report += "### Recovery Actions\n"
            for action in result.recovery_actions:
                report += f"- {action}\n"
            
            if result.warnings:
                report += "\n### Warnings\n"
                for warning in result.warnings:
                    report += f"- ⚠️ {warning}\n"
        else:
            report += "## ⚠️ Manual Intervention Required\n\n"
            for error in result.errors:
                report += f"- {error}\n"
        
        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "comprehensive_porting":
        # Import porting integration
        try:
            from mcp.porting_integration import PortingIntegration
        except ImportError:
            from porting_integration import PortingIntegration
        
        import asyncio
        integration = PortingIntegration()
        source_code = arguments.get("source_code", "")
        source_lang = arguments.get("source_lang", "")
        target_lang = arguments.get("target_lang", "")
        source_platform = arguments.get("source_platform", "")
        target_platform = arguments.get("target_platform", "")
        target_backend = arguments.get("target_backend", "")
        build_system = arguments.get("build_system", "cmake")
        mode = arguments.get("mode", "auto")
        
        result = await integration.comprehensive_porting(
            source_code, source_lang, target_lang,
            source_platform, target_platform, target_backend,
            build_system, mode
        )
        
        report = "# Comprehensive Porting Report\n\n"
        report += f"**Status:** {result.status}\n"
        report += f"**Mode Executed:** {result.mode_executed}\n\n"
        
        report += "## Results\n\n"
        report += f"- **Language Converted:** {'✅' if result.language_converted else '❌'}\n"
        report += f"- **Platform Migrated:** {'✅' if result.platform_migrated else '❌'}\n"
        report += f"- **Build Integrated:** {'✅' if result.build_integrated else '❌'}\n"
        report += f"- **Performance Optimized:** {'✅' if result.performance_optimized else '❌'}\n"
        report += f"- **Error Recovery Enabled:** {'✅' if result.error_recovery_enabled else '❌'}\n\n"
        
        if result.recommendations:
            report += "### Recommendations\n"
            for rec in result.recommendations:
                report += f"- {rec}\n"
            report += "\n"
        
        if result.next_steps:
            report += "### Next Steps\n"
            for step in result.next_steps:
                report += f"- {step}\n"
        
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
        msg = f"✅ Goal set.\n\n**Goal:** {params.goal}\n**Target:** {params.target_arch} ({params.target_backend})\n**Tok\s target:** {params.target_tok_per_s or 'N\A'}\n**Min viable:** {params.min_perf or 'N\A'}\n**Files in scope:** {params.files_in_scope or 'all'}\n**Files out of scope:** {params.files_out_of_scope or 'none'}\n**Guardrails:** {len(params.guardrails)}\n\nRead `kands:\\goal` to view."
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
        msg = f"✅ Goal updated.\n\n**Field:** `{params.override_field}`\n**Old:** {str(old_val)[:100]}\n**New:** {str(new_val)[:100]}\n**Reason:** {params.reason}\n\nRead `kands:\\goal` to view full state."
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

        # Scan for new\modified files
        import os as _os
        new_files = []
        if _GOAL_STATE_PATH.exists():
            new_files.append("mcp\state\goal.json")
        if _PERF_HISTORY_PATH.exists():
            new_files.append("mcp\state\perf_history.json")
        _ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for f in _ANALYSIS_CACHE_DIR.glob("*.md"):
            new_files.append(f"mcp\state\analysis\{f.name}")
        findings_dir = _KANDS_ROOT \ "findings"
        if findings_dir.exists():
            for f in findings_dir.glob("*.md"):
                new_files.append(f"findings\{f.name}")

        report = f"""# Pull Request Preparation

**Date:** {now.strftime('%Y-%m-%d %H:%M UTC')}
**Branch:** `{branch}`

## Files to include in this PR

**New\modified files:**
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
            report += "- Porting findings report (`.md` in `findings\`)\n"
        if "arch_data" in params.change_types:
            report += "- Architecture capability updates (ARCH_MAP.json)\n"
        if "patterns" in params.change_types:
            report += "- New porting transformation patterns (PORTING_PATTERNS.json)\n"
        if "perf_data" in params.change_types:
            report += "- Performance benchmark data (PERF_BENCHMARKS.json)\n"
        if "toolchain" in params.change_types:
            report += "- Toolchain additions (TK_TOOLS.json)\n"

        report += """
**Goal & guardrails:** All changes respect the defined goal in `kands:\\goal`.

## Git commands to run:

```bash
# 1. Create and switch to branch
git checkout -b """ + branch + f"""

# 2. Stage specific files (sanitized — no personal data)
git add findings\
git add mcp\state\ 2>\dev\null || true
git add ARCH_MAP.json 2>\dev\null || true
git add PORTING_PATTERNS.json 2>\dev\null || true
git add PERF_BENCHMARKS.json 2>\dev\null || true

# 3. Commit
git commit -m "feat: add porting findings + perf data for {branch.split('-')[0] if '-' in branch else 'contribution'}"

# 4. Push and create PR
git push -u origin """ + branch + """
gh pr create \
  --title "feat: porting findings for """ + branch.split('-')[0] if '-' in branch else 'contribution' + f""" — contributed via K_and_S" \
  --body "Generated automatedly via the K_and_S bundle. See included findings\ report for details."

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
        now = datetime.now(timezone.utc)
        report += f"**Date:** {now.strftime('%Y-%m-%d %H:%M UTC')}\n\n"

        # Check array lengths
        if len(source_vals) != len(target_vals):
            report += "## ❌ MISMATCH — Array Length\n\n"
            report += f"Source has {len(source_vals)} values, target has {len(target_vals)} values.\n"
            report += "**Diagnosis:** Kernel launched with different workgroup size, shared memory, or reduction strategy.\n"
            report += "**Action:** Verify kernel dispatch configuration (grid\block dimensions, shared memory).\n"
        else:
            report += "## Array Length\n✅ Match (both {} values)\n\n".format(len(source_vals))

        # Check max absolute error
        max_abs_err = 0.0
        max_rel_err = 0.0
        divergence_indices = []
        for i, (sv, tv) in enumerate(zip(source_vals, target_vals)):
            abs_err = abs(sv - tv)
            rel_err = abs_err \ max(abs(sv), 1e-30)
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
        report += f"| Divergent values | {len(divergence_indices)} \ {len(source_vals)} |\n"

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
            report += "- Consider using `@fast_math` (CUDA) vs `-ffast-math` (GCC\Clang) consistently\n\n"
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

    elif name == "strip_tracing":
        directory = arguments.get("directory", ".")
        strip_debug = int(arguments.get("strip_debug", 0))
        strip_perf = int(arguments.get("strip_perf", 0))
        extensions = arguments.get("extensions", [".c", ".cpp", ".cu", ".h", ".hpp"])
        result = _strip_tracing_from_dir(directory, strip_debug, strip_perf, extensions)
        results.append(mcp_types.TextContent(type="text", text=result))

    elif name == "reject_common_pattern":
        params = RejectPatternParams(**arguments)
        arch_map = _load_json(_ARCH_MAP_PATH)
        idea = params.common_idea.lower()
        arch_id = params.arch
        now = datetime.now(timezone.utc)

        report = "# Reject Common Pattern\n\n"
        report += f"**Examining:** \"{params.common_idea}\"\n"
        report += f"**Target arch:** `{arch_id}`\n"
        report += f"**Date:** {now.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        rejected = False

        if "hipify" in idea or "hipify-perl" in idea or "hipify tools" in idea:
            report += "## REJECTED: hipify-perl handles everything\n\n"
            report += "**Why it's wrong:**\n"
            report += "- Syntactic translator does NOT handle semantic differences\n"
            report += "- Template instantiations with __half\__nv_bfloat16 are not auto-correct\n"
            report += "- Custom intrinsics (mma, imad, shfl) require MANUAL hipify\n"
            report += "- Library API differences (cuBLASLt vs rocBLAS) need manual rewrite\n"
            if "gfx" in arch_id:
                report += f"\n**Correct for `{arch_id}`:**\n"
                report += "- Run hipify-perl as STARTING POINT only\n"
                report += "- Use `query_kernel_features` to find un-translated branches\n"
                report += "- Validate with `verify_numerical_equivalence` after each change\n"
                report += "- Check rocBLAS vs hipBLASLt backend (ROCBLAS_USE_HIPBLASLT=0 for RDNA2)\n"
            report += "\n**Evidence:** See `kands:\\porting-patterns` → CUDA→HIP for manual-only constructs.\n"
            rejected = True

        if "verbatim" in idea or ("copy" in idea and "hip" in idea) or "directly" in idea:
            report += "## REJECTED: Copy CUDA to HIP verbatim\n\n"
            report += "**Why it's wrong:**\n"
            report += "- Tensor core intrinsics (mma.*) have NO HIP equivalent — use rocWMMA or dot4\n"
            report += "- `__ldg()` → normal loads on HIP\n"
            report += "- `clock64()` → `__builtin_amdgcn_s_memtime()`\n"
            report += "- Wave size assumptions (32 vs 32\64)\n"
            rejected = True

        if "cooperative" in idea and "matrix" in idea and ("amd" in idea or "rdna" in idea or "radeon" in idea):
            report += "## REJECTED: Vulkan CooperativeMatrix works on AMD\n\n"
            report += "**Why it's wrong:**\n"
            report += "- GL_KHR_cooperative_matrix is BROKEN on AMD drivers\n"
            report += "- Use DP4a (GL_KHR_shader_integer_dot_product) instead\n"
            report += "\n**Correct approach:**\n"
            report += "- Use spv_dp4 instructions for integer matmul\n"
            report += "- For float: FMA chains or native WMMA (RDNA3+ only)\n"
            report += "\n**Evidence:** kands:\\architectures → check vulkan.cooperative_matrix field.\n"
            rejected = True

        if "wmma" in idea or ("tensor core" in idea and "everywhere" in idea):
            report += "## CAUTION: Using tensor cores everywhere\n\n"
            report += "**Why it's risky:**\n"
            report += "- Not all archs have tensor cores (RDNA2, Arc A770)\n"
            report += "- For small matrices, F32 can be faster than WMMA (setup overhead)\n"
            arch_info = arch_map.get("entries", {}).get(arch_id, {})
            if arch_info:
                has_tc = arch_info.get("tensor_cores", False)
                report += f"\n**{arch_id}:** {'Has tensor cores — use WMMA' if has_tc else 'NO tensor cores — use F32 fallback (verified faster for sizes < 128x128)'}\n"
            rejected = True

        if "threads" in idea and ("more" in idea or "occupancy" in idea):
            report += "## CAUTION: Maximizing thread count for occupancy\n\n"
            report += "**Why it's risky:**\n"
            report += "- High occupancy ≠ high performance on memory-bound kernels\n"
            report += "- Register pressure can LIMIT occupancy unexpectedly\n"
            report += "- Wave32 vs Wave64: RDNA2 often better at wave32\n"
            arch_info = arch_map.get("entries", {}).get(arch_id, {})
            if arch_info:
                report += f"\n**{arch_id}:** wave_size={arch_info.get('wave_size','?')}, LDS={arch_info.get('lds_kib','?')}KiB\CU\n"
            rejected = True

        if not rejected:
            report += "## NOT A KNOWN MYTH — but still verify\n\n"
            report += f"\"{params.common_idea}\" is not in our known anti-pattern list.\n"
            report += "Known myths: 1) hipify-perl handles everything 2) Copy CUDA→HIP verbatim 3) Vulkan CM works on AMD 4) More threads=better 5) F32 fallback always slower 6) DP4a best for everything\n"

        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "radical_experiment_plan":
        params = RadicalExperimentParams(**arguments)
        arch_map = _load_json(_ARCH_MAP_PATH)
        arch_info = arch_map.get("entries", {}).get(params.arch, {})
        vendor = arch_info.get("vendor", "unknown")
        now = datetime.now(timezone.utc)

        report = "# Radical Experiment Plan\n\n"
        report += f"**Arch:** `{params.arch}` ({arch_info.get('name', 'unknown')})\n"
        report += f"**Kernel type:** `{params.kernel_type}`\n"
        report += f"**Date:** {now.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        report += "## EXPERIMENTAL — Validate with CPU Oracle Before Deploying\n\n"

        if vendor == "AMD" or "gfx" in params.arch:
            report += "## AMD\RDNA-Specific Radical Experiments\n\n"
            etype = params.experiment_type
            if etype in ("all", "memory"):
                report += "### Memory Hierarchy Experiments\n"
                report += f"1. **Bypass LDS** for K<=16: register-only blocking (LDS={arch_info.get('lds_kib','?')}KiB\CU)\n"
                report += "2. **DPP-based data shuffle** instead of LDS transpose (v_dot2, v_perm_b32, DPP8)\n"
                report += "3. **Explicit `s_waitcnt lgkmcnt(0)`** before shared->register copy (inline asm)\n\n"
            if etype in ("all", "compute"):
                report += "### Compute Experiments\n"
                report += f"1. **Native v_dot4** instead of FMA chains (wave_size={arch_info.get('wave_size','?')})\n"
                report += "2. **Raw AMDGCN asm** for inner loop with `.scheduler_priority` hints\n"
                report += "3. **Wave32 vs Wave64**: compile with `-mwavefrontsize=32`, compare\n\n"
            if etype in ("all", "scheduling"):
                report += "### Scheduling Experiments\n"
                report += "1. **8-wave ping-pong vs 4-wave interleave** (HipKittens patterns)\n"
                report += "2. Thread blocks: multiples of 32 vs wave_size\n\n"
            if etype in ("all", "layout"):
                report += "### Layout Experiments\n"
                report += "1. **AoS vs SoA** — test cache locality tradeoff\n"
                report += "2. **LDS bank-conflict-free padding** — pad to 2x width\n\n"
            report += "**RDNA-native note:** HipKittens (github.com\HazyResearch\HipKittens) targets CDNA3\CDNA4 but its tile abstractions directly adapt to RDNA — only wave_size (32) and LDS size differ.\n\n"

        elif vendor == "NVIDIA" or "sm_" in params.arch:
            report += "## NVIDIA-Specific Radical Experiments\n"
            report += "1. **PTX-level custom scheduling** with `.scheduler_priority` hints\n"
            report += "2. **FMAD vs separate FMUL+FADD** — compile with `--fmad=false`, measure\n"
            report += "3. **Non-standard mma tile shapes** (8x8x4)\n\n"

        elif vendor == "Intel" or "gen" in params.arch:
            report += "## Intel-Specific Radical Experiments\n"
            report += "1. **Direct DPAS invocation** via `__builtin_IB_sub_group_mm_*`\n"

        report += "## Verification Requirements\n"
        report += "1. Run reference on CPU (oracle)\n"
        report += f"2. Call `verify_numerical_equivalence(arch=\"{params.arch}\", workload=\"{params.kernel_type}\")`\n"
        report += "3. STRACE() + fence timeout (≤1500ms Windows)\n"
        report += "4. Fallback to last-good, NEVER stub\n"
        report += f"5. `record_performance(arch=\"{params.arch}\", workload=\"{params.kernel_type}\")`\n"

        results.append(mcp_types.TextContent(type="text", text=report))

    elif name == "list_skills":
        category = arguments.get("category", "")
        search = arguments.get("search", "").lower()
        skill_data = _load_myskills_index()

        if not skill_data.get("loaded", False):
            results.append(mcp_types.TextContent(type="text", text=f"""# MySkills Not Found

**Expected directory:** `{skill_data.get("myskills_dir", "unknown")}`

The MySkills framework was not found. To use skills:

1. Clone MySkills to `%USERPROFILE%\.config\opencode\command\MySkills` (Windows) or `~\.config\opencode\command\MySkills` (Linux)
2. Or edit `_MY_SKILLS_DIR` in `server.py` to point to your MySkills checkout

Available resources:
- `kands:\\myskills` — structured JSON index of all skills
- `kands:\\myskills\index` — raw skill-index.md
- `kands:\\myskills\<skill_name>` — full SKILL.md for any skill
"""))
        else:
            report = f"# Available MySkills ({skill_data["skill_count"]} skills)\n\n"
            report += f"**Directory:** `{skill_data["myskills_dir"]}`\n\n"

            if search:
                # Filter by search term
                matched = [s for s in skill_data["skills"]
                          if search in s["name"].lower() or search in s.get("description", "").lower()]
                report += f"**Search filter:** `{search}` → {len(matched)} matches\n\n"
                if not matched:
                    results.append(mcp_types.TextContent(type="text", text=report + "_No skills match the search term._"))
                    return results
                skills_to_show = matched
            else:
                skills_to_show = skill_data["skills"]

            # Group by category
            by_cat: dict[str, list] = {}
            for s in skills_to_show:
                by_cat.setdefault(s["category"], []).append(s)

            # Show categories in a readable order
            cat_order = ["debug", "gpu", "ml", "optimization", "systems", "devops",
                        "python", "testing", "security", "porting", "general"]
            shown_cats = []
            for cat in cat_order:
                if cat in by_cat:
                    shown_cats.append(cat)
            for cat in sorted(by_cat.keys()):
                if cat not in shown_cats:
                    shown_cats.append(cat)

            for cat in shown_cats:
                cat_skills = by_cat[cat]
                report += f"## {cat.title()} ({len(cat_skills)})\n\n"
                for s in cat_skills:
                    report += f"- **`{s["name"]}`** — {s.get("description", "")[:100]}\n"
                report += "\n"

            report += "## How to Use Skills\n\n"
            report += "1. **Read a skill:** `read kands:\\myskills\<skill_name>` for full SKILL.md instructions\n"
            report += "2. **Use in conversation:** Mention the skill name (e.g. \"use the debug-deep skill\")\n"
            report += "3. **OpenCode native:** Skills are also auto-discovered from `~\.config\opencode\command\MySkills` via `skills: [\".\\"]` in opencode.jsonc\n"

            results.append(mcp_types.TextContent(type="text", text=report))
            # Also return the raw data for programmatic use
            results.append(mcp_types.TextContent(type="text", text=json.dumps(skill_data, indent=2)))

    elif name == "capability_report":
        cap = _capability_report(arguments.get("target", ""),
                                 arguments.get("backend", "auto"))
        lines = [f"# Capability report — `{cap['target']}` ({cap['backend']})\n",
                 "| item | value |", "|---|---|"]
        for k in ("supported_backend", "compiler", "root", "objdump"):
            lines.append(f"| {k} | `{cap.get(k)}` |")
        lines.append(f"| known_backends | {', '.join(cap.get('known_backends', []))} |")
        if not cap.get("compiler"):
            lines.append("\n**No compiler found.** Set the backend root env var "
                         "(ROCM_PATH \ CUDA_PATH \ ONEAPI_ROOT) before planning a port.")
        lines.append("\nNext: `probe_isa` every instruction the port depends on, "
                     "then `build_module` with `expected_symbols`.")
        results.append(mcp_types.TextContent(type="text", text="\n".join(lines)))

    elif name == "probe_isa":
        target = arguments.get("target", "")
        cands = arguments.get("candidates", []) or []
        if not cands:
            results.append(mcp_types.TextContent(
                type="text", text="Error: candidates[] is required"))
        else:
            rows = _probe_many(target, cands, backend=arguments.get("backend", "auto"))
            md = [f"# ISA probe — target `{target}`\n",
                  "| candidate | result | detail |", "|---|---|---|"]
            for r in rows:
                mark = "OK" if r["ok"] else "FAIL"
                detail = (", ".join(r["mnemonics"][:8]) if r["ok"]
                          else r["error"][:160].replace("|", "\"))
                md.append(f"| `{r['candidate']}` | **{mark}** | {detail} |")
            md.append(f"\n**{sum(1 for r in rows if r['ok'])}\{len(rows)} "
                      f"candidates encode on `{target}`.**")
            if any(not r["ok"] for r in rows):
                md.append("\nRejected candidates must not appear in the generated "
                          "source: substitute a portable fallback and re-probe.")
            results.append(mcp_types.TextContent(type="text", text="\n".join(md)))
            results.append(mcp_types.TextContent(type="text", text=json.dumps(rows, indent=2)))

    elif name == "scan_residual_asm":
        files = arguments.get("files", []) or []
        hits = _scan_residual_asm(files, arguments.get("mnemonics"),
                                  backend=arguments.get("backend", "hip"))
        if not hits:
            results.append(mcp_types.TextContent(
                type="text",
                text=f"No residual foreign asm in {len(files)} file(s). Gate passes."))
        else:
            md = [f"# Residual foreign asm — {len(hits)} hit(s)\n",
                  "Live `asm(\"...\")` string literals, not comments; the target "
                  "assembler will reject them.\n",
                  "| file | line | text |", "|---|---|---|"]
            for h in hits[:80]:
                md.append(f"| `{Path(h['file']).name}` | {h['line']} | `{h['text'][:120]}` |")
            if len(hits) > 80:
                md.append(f"\n... and {len(hits) - 80} more")
            results.append(mcp_types.TextContent(type="text", text="\n".join(md)))
            results.append(mcp_types.TextContent(type="text", text=json.dumps(hits, indent=2)))

    elif name == "build_module":
        sources = arguments.get("sources", []) or []
        target = arguments.get("target", "")
        out = arguments.get("out", "")
        if not sources or not target or not out:
            results.append(mcp_types.TextContent(
                type="text", text="Error: sources[], target and out are required"))
        else:
            res = _build_module(sources, target, out,
                                backend=arguments.get("backend", "auto"),
                                includes=arguments.get("includes"),
                                flags=arguments.get("flags"),
                                concatenate=arguments.get("concatenate", True))
            md = [f"# Build — `{Path(out).name}` ({res['backend']}\{target})\n"]
            if res["ok"]:
                md.append(f"- result: **OK**, {res['size']:,} bytes")
                md.append(f"- exported kernel symbols: {len(res['symbols'])}")
                expected = arguments.get("expected_symbols")
                if expected:
                    d = _diff_symbols(res["symbols"], expected)
                    md.append(f"- present: {len(d['present'])}\{len(expected)}")
                    if d["missing"]:
                        md.append(f"- **missing: {', '.join(d['missing'])}**")
                    others = [s for s in d["unexpected"] if not s.startswith("_Z")]
                    if others:
                        md.append(f"- other kernels in module ({len(others)}): "
                                  f"{', '.join(others[:6])}"
                                  + (" ..." if len(others) > 6 else ""))
                    if d["missing"]:
                        md.append("\nA module that builds but lacks the expected "
                                  "symbols cannot serve the call that needs them.")
            else:
                md.append(f"- result: **FAIL** (rc={res['rc']})")
                for e in res["errors"][:12]:
                    md.append(f"- {e}")
                md.append("\nThe output was deleted before the build, so any artifact "
                          "now present is from this attempt only.")
            results.append(mcp_types.TextContent(type="text", text="\n".join(md)))
            results.append(mcp_types.TextContent(type="text", text=json.dumps(res, indent=2)))

    elif name == "code_object_report":
        p = arguments.get("path", "")
        if not p:
            results.append(mcp_types.TextContent(
                type="text", text="Error: path is required"))
        else:
            r = _code_object_report(p, arguments.get("arch"),
                                    arguments.get("lds_kib"))
            if not r.get("ok"):
                results.append(mcp_types.TextContent(
                    type="text", text="Error: %s" % r.get("error")))
            else:
                md = ["# Code object — `%s`\n" % Path(p).name]
                if r.get("bundle"):
                    md.append("- bundle targets: %s" % ", ".join(r["bundle"]))
                md.append("- kernels found: %d" % len(r.get("kernels", {})))
                if r.get("note"):
                    md.append("- note: %s" % r["note"])
                md.append("")
                md.append("| kernel | wave | VGPR | SGPR | LDS\block | blocks\CU | insns |"
                          " glb loads | ld widths | LDS | bit | dot | verdicts |")
                md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
                for kname, v in sorted(r.get("kernels", {}).items()):
                    widths = ",".join("%s x%d" % (a, b) for a, b in
                                      sorted(v.get("load_widths", {}).items())) or "-"
                    md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                              % (kname, v.get("wavefront_size", "?"),
                                 v.get("vgpr_count", "?"), v.get("sgpr_count", "?"),
                                 v.get("group_segment_fixed_size", "?"),
                                 v.get("lds_blocks_per_cu", "-"),
                                 v.get("instructions", "?"),
                                 v.get("global_loads", "?"), widths,
                                 v.get("lds_ops", 0), v.get("bit_ops", 0),
                                 v.get("dot_ops", 0),
                                 "; ".join(v.get("verdicts", [])) or "ok"))
                md.append("")
                md.append("Residency is computed from LDS per block against the "
                          "`lds_kib` you supply; register-limited occupancy needs the "
                          "register file size for the target. Measured LDS per CU is "
                          "the number that matches reality, not the datasheet.")
                results.append(mcp_types.TextContent(
                    type="text", text="\n".join(md)))
                results.append(mcp_types.TextContent(
                    type="text", text=json.dumps(r["kernels"], indent=2)))

    elif name == "build_porting_map":
        files = arguments.get("files", []) or []
        target = arguments.get("target_arch", "")
        source = arguments.get("source_arch", "auto")
        fmt = arguments.get("format", "both")
        if not files or not target:
            results.append(mcp_types.TextContent(
                type="text", text="Error: files[] and target_arch are required"))
        else:
            m = _build_porting_map(files, source, target)
            if fmt in ("markdown", "both"):
                results.append(mcp_types.TextContent(
                    type="text", text=_render_porting_map(m)))
            if fmt in ("json", "both"):
                slim = {k: v for k, v in m.items() if k != "functions"}
                results.append(mcp_types.TextContent(
                    type="text", text=json.dumps(slim, indent=2)))
                results.append(mcp_types.TextContent(
                    type="text", text=json.dumps(m["functions"], indent=2)))

    elif name == "lookup_construct":
        r = _lookup_construct(arguments.get("construct", ""),
                              arguments.get("target_arch", ""))
        if not r["known"]:
            results.append(mcp_types.TextContent(
                type="text",
                text=(f"`{r['construct']}` is not in the construct table "
                      f"(family `{r['target_family']}`).\n\n"
                      "Probe it with `probe_isa` on the target before using it: "
                      "an unknown construct is an unverified assumption.")))
        else:
            results.append(mcp_types.TextContent(
                type="text",
                text=(f"# `{r['construct']}` → `{r['target_arch']}` "
                      f"({r['target_family']})\n\n"
                      f"- role: {r['role']}\n"
                      f"- use: **{r['use']}**\n"
                      f"- fidelity: **{r['grade']}**\n"
                      + (f"- note: {r['note']}\n" if r.get("note") else ""))))

    elif name == "verify_porting_map":
        files = arguments.get("files", []) or []
        target = arguments.get("target_arch", "")
        if not files or not target:
            results.append(mcp_types.TextContent(
                type="text", text="Error: files[] and target_arch are required"))
        else:
            m = _build_porting_map(files, arguments.get("source_arch", "auto"), target)
            cands = _pm_probe_candidates(m["constructs_used"])
            rows = _probe_many(target, cands, backend=arguments.get("backend", "auto"))
            probes = {r["candidate"]: r["ok"] for r in rows}
            v = _verify_porting_map(m, probes)
            md = [f"# Verified porting map — `{target}`\n",
                  f"Probed {len(rows)} mapped replacement(s) with the real compiler.\n",
                  "| construct | replacement encodes |", "|---|---|"]
            for r in rows:
                md.append(f"| `{r['candidate']}` | {'**yes**' if r['ok'] else '**NO**'} |")
            md.append("\n## Grade changes\n")
            if not v["changes"]:
                md.append("None — every predicted mapping encoded as expected.")
            for c in v["changes"]:
                md.append(f"- `{c['construct']}`: {c['was']} → {c['now']} ({c['reason']})")
            md.append("\n## Fidelity after verification\n")
            for g, n in v["fidelity"].items():
                md.append(f"- {g}: {n}")
            bad = _pm_unresolved(m)
            if bad:
                md.append("\n## Still needs a human\n")
                md.append("| function | role | fidelity | constructs | file:line |")
                md.append("|---|---|---|---|---|")
                for f in bad:
                    md.append(f"| `{f['name']}` | {f['role']} | **{f['fidelity']}** | "
                              f"{', '.join(f['constructs']) or '—'} | "
                              f"`{Path(f['file']).name}:{f['line']}` |")
            results.append(mcp_types.TextContent(type="text", text="\n".join(md)))

    elif name == "build_function_reference":
        files = arguments.get("files", []) or []
        target = arguments.get("target_arch", "")
        if not files or not target:
            results.append(mcp_types.TextContent(
                type="text", text="Error: files[] and target_arch are required"))
        else:
            ref = _build_reference(files, arguments.get("source_arch", "auto"), target)
            fmt = arguments.get("format", "markdown")
            if fmt in ("markdown", "both"):
                results.append(mcp_types.TextContent(
                    type="text", text=_render_reference(ref)))
            if fmt in ("json", "both"):
                slim = {k: v for k, v in ref.items() if k != "rows"}
                results.append(mcp_types.TextContent(
                    type="text", text=json.dumps(slim, indent=2)))
                results.append(mcp_types.TextContent(
                    type="text", text=json.dumps(ref["rows"], indent=2)))
            if ref["blockers"]:
                results.append(mcp_types.TextContent(
                    type="text",
                    text=("\n**Pre-port stop:** "
                          f"{len(ref['blockers'])} feature(s) have no equivalent on "
                          f"`{target}`. Resolve the design for each before writing "
                          "kernel code — a blocker is not a partial gap.")))

    elif name == "lookup_api":
        r = _lookup_api(arguments.get("api", ""), arguments.get("target_arch", ""))
        if not r.get("known"):
            results.append(mcp_types.TextContent(
                type="text",
                text=(f"`{r['id']}` is not in the reference table.\n\n"
                      "Probe it with `probe_isa` before relying on it — an unknown "
                      "symbol is an unverified assumption.")))
        else:
            results.append(mcp_types.TextContent(
                type="text",
                text=(f"# `{r['id']}` → `{r['target']}` (`{r['bucket']}`)\n\n"
                      f"- category: {r['category']}\n"
                      f"- replacement: **{r['replacement']}**\n"
                      f"- status: **{r['status']}**\n"
                      + (f"- note: {r['note']}\n" if r.get("note") else ""))))

    elif name == "analyze_full_pipeline":
        repo = arguments.get("repo", "")
        root = Path(repo)
        if not root.exists():
            results.append(mcp_types.TextContent(
                type="text", text=f"Error: {repo} does not exist"))
        else:
            md = _analyze_pipeline(
                repo,
                target_arch=arguments.get("target_arch", ""),
                source_arch=arguments.get("source_arch", ""),
            )
            results.append(mcp_types.TextContent(
                type="text", text=_render_pipeline(md)))
            results.append(mcp_types.TextContent(
                type="text", text=json.dumps(md, indent=2, default=str)))

    elif name == "analyze_core_engine":
        repo = arguments.get("repo", "")
        root = Path(repo)
        if not root.exists():
            results.append(mcp_types.TextContent(
                type="text", text=f"Error: {repo} does not exist"))
        else:
            md = _analyze_core(
                repo,
                target_arch=arguments.get("target_arch", ""),
            )
            results.append(mcp_types.TextContent(
                type="text", text=_render_core(md)))
            results.append(mcp_types.TextContent(
                type="text", text=json.dumps(md, indent=2, default=str)))

    return results


# Register resource handlers for both mcp API generations:
#   mcp < 1.x  : server.add_request_handler(name, params_class, handler)
#   mcp >= 1.x : server.request_handlers[RequestType] = handler, keyed by the request type
# Handlers are the same either way: coroutines taking (request, params).
def _register(name, params_cls, req_attr, handler):
    if hasattr(server, "add_request_handler"):
        server.add_request_handler(name, params_cls, handler)
        return
    server.request_handlers[getattr(mcp_types, req_attr)] = handler


_register("resources\list", mcp_types.PaginatedRequestParams, "ListResourcesRequest",
          _make_list_resources_handler())
_register("resources\read", mcp_types.ReadResourceRequestParams, "ReadResourceRequest",
          _make_read_resource_handler())
_register("tools\list", mcp_types.PaginatedRequestParams, "ListToolsRequest",
          _make_list_tools_handler())
_register("tools\call", mcp_types.CallToolRequestParams, "CallToolRequest",
          _make_call_tool_handler())


def main():
    """Entry point: run the MCP server over stdio.

    stdio_server is an async context manager, not a coroutine, and it takes no server
    argument; the server is driven by run() over the streams it yields. Calling
    asyncio.run(stdio_server(server)) raises "an asyncio.Future, a coroutine or an
    awaitable is required" and no server ever starts.
    """
    from mcp.server.stdio import stdio_server

    async def _serve():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream,
                             server.create_initialization_options())

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
