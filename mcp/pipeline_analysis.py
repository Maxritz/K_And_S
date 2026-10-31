"""
pipeline_analysis — general-purpose inference pipeline + core engine analysis.

Two analyzers that work with ANY engine source tree, not tied to a specific
project. They scan for pipeline-stage markers, dispatch patterns, memory
management, and porting-critical constructs, then report a structured map
suitable for pre-port planning.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import porting_map  # noqa: E402

# ---------------------------------------------------------------------------
# Pipeline-stage signature patterns (generic by role)
# ---------------------------------------------------------------------------

_PIPELINE_STAGES: list[tuple[str, str, str]] = [
    ("engine_init",   "Engine\\GPU initialization",      r"cudaInit|cublasInit|hipInit|init_.*(engine|gpu|device|model)|Engine\s*\(|TPEngine\s*\(|ggml_init|ggml_backend_init"),
    ("model_load",    "Model loading from GGUF\\safetensors\\onnx", r"gguf_init|load_model|load_tensor|load_weight|llama_load|read_gguf|mmap|ggml_load_tensor"),
    ("weight_repack", "Weight format repack\\layout transform", r"repack|convert_weight|reformat|layout.*weight|quantize_row|to_g4|g5n|unpack_weight|layout\.py|ggml_transform_tensor|dequantize"),
    ("prompt",        "Prompt\\prefill processing",      r"prefill|prompt|chunk.*plan|k_chunk|k_attn.*flash|prompt_tokens|process_batch|feed_batch|process_batch"),
    ("decode_loop",   "Token-by-token decode dispatch", r"decode|next_token|k_xrms|k_gdn|evaluate|step_token|forward\b|eval\b|decode_prompt"),
    ("attention",     "Attention kernel",               r"k_attn|attn_flash|attention\s*\(|scaled_dot|q_attn|kv_cache.*update|flash.*attn"),
    ("kv_cache",      "KV cache store\\retrieve",        r"k_store_q8|kv_cache|store_kv|q8_.*scale|k_cache|cache.*append|cache_copy|ggml_kv_cache"),
    ("gemm",          "GEMM\\matmul core",               r"k_mm|mm_core|gemm|gemv|matmul|k_g4|k_g5|dequant|dequantize|ggml_gemm|ggml_mul"),
    ("normalization", "LayerNorm\\RMSNorm",              r"k_xrms|k_norm|layer_norm|rms_norm|k_l2w|normalize|folding|rms_norm|norm"),
    ("ffn",           "Feed-forward network",           r"ffn|swiglu|gate.*up|mlp|mlp_gate|up_proj|down_proj|moe|expert"),
    ("gdn_ssm",       "GDN\\SSM recurrent state",        r"k_gdn|gdn_pre|gdn_delta|ssm|delta.*state|conv.*state|recurrent|state.*write|hidden.*state"),
    ("speculative",   "Speculative decoding",           r"speculat|verify.*pass|k_sgdn|commit.*token|draft.*token|accept|medusa|draft"),
    ("tensor_parallel", "Tensor-parallel \ multi-GPU",  r"TPEngine|tensor_parallel|all_reduce|all_gather|xchg|partial.*sum|two.card|distribute|nccl|rpc|mpi"),
    ("output",        "Output projection \ logit head", r"k_l2w|logit|head.*output|tied.*head|output.*norm|lm_head|embed|result_logits"),
    ("sampling",      "Sampling \ token selection",     r"sample|argmax|topk|top_p|logits.*sample|temperature|greedy|sampling|sample_topp"),
]

# ---------------------------------------------------------------------------
# Dispatch-pattern signature patterns
# ---------------------------------------------------------------------------

_DISPATCH_PATTERNS: list[tuple[str, str, str]] = [
    ("per_token",   "Single-token-per-launch dispatch",   r"blockIdx\.x.*token|threadIdx\.x.*token|per\.token|single\.token|decode.*token|1\s*token"),
    ("batched",     "Batched token dispatch (blockIdx.y=token)", r"blockIdx\.y|batched.*launch|batch.*token|token\.batch|_T\b|wrapper.*token|n_tokens.*batch"),
    ("persistent",  "Persistent kernel (loop over tokens)", r"persistent|while.*true.*token|continue.*launch|while.*token|infinite.*loop|while.*grid"),
    ("cooperative", "Grid-wide sync \ cooperative groups", r"cooperative|grid\.sync|__syncthreads.*grid|nvptx.*sync|__syncthreads_count"),
    ("dynamic_par", "Dynamic parallelism (device-side launch)", r"<<<[^>]*>>>|kernel.*<<<|launch.*from.*device|\b<<<"),  # careful
    ("streamed",    "Streamed execution", r"stream|cudaStream|hipStream|async.*copy|ggml_backend_evt"),
]

# ---------------------------------------------------------------------------
# Memory-pattern signature patterns
# ---------------------------------------------------------------------------

_MEMORY_PATTERNS: list[tuple[str, str, str]] = [
    ("host_mapped", "Host-mapped \ pinned memory", r"host.*map|pinned|ReBAR|rebar|host_register|hostGetPointer|hipHostMalloc|cudaHostRegister|fine\.grained|ggml_mmap"),
    ("staging",     "Staging copy buffer",         r"staging|intermediate.*copy|memcpy.*async|transfer.*buffer|ring\.buffer|staging_copy"),
    ("managed",     "Unified\\Managed memory",      r"cudaMallocManaged|hipMallocManaged|managed.*memory|cuMemMap|um.*alloc"),
    ("custom_alloc", "Custom allocator",            r"alloc.*pool|gpu.*allocator|memory.*arena|buf.*alloc|custom.*allocat|ggml_init_params|salloc"),
    ("zero_copy",   "Zero-copy \ mmap weights",    r"mmap|zero\.copy|direct.*map|weight.*stream|model.*map|gguf_init_from|read_only"),
]

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

_KNOWN_EXTS = {".cu", ".cuh", ".cpp", ".hpp", ".cxx", ".cc", ".py",
               ".comp", ".glsl", ".hlsl", ".metal", ".cl", ".rs"}
_SKIP_DIRS = {"build", "dist", ".git", ".venv", "node_modules",
              "__pycache__", "third_party", ".deps", "CMakeFiles"}


def _scan_tree(repo: str) -> list[dict[str, Any]]:
    """Walk a source tree and return categorized files."""
    root = Path(repo)
    if not root.exists():
        return []
    files: list[dict[str, Any]] = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in _KNOWN_EXTS:
            continue
        if any(p in f.parts for p in _SKIP_DIRS):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        files.append({
            "path": str(f.relative_to(root).as_posix()),
            "ext": f.suffix.lower(),
            "size": f.stat().st_size,
            "text": text,
            "text_head": text[:8000],
        })
    return files


def _match_lines(text: str, pattern: str, limit: int = 20) -> list[dict[str, Any]]:
    """Find all lines matching a pattern in text."""
    out: list[dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(pattern, line, re.IGNORECASE):
            out.append({"line": i, "match": line.strip()[:200]})
            if len(out) >= limit:
                break
    return out


def analyze_pipeline(repo: str, target_arch: str = "", source_arch: str = "") -> dict[str, Any]:
    """Analyse the full inference pipeline of any engine source tree.

    Scans for pipeline-stage markers across all source files and collects
    dispatch + memory management patterns. Scores porting-critical constructs
    against target\\source arch family using the existing porting_map module.
    """
    files = _scan_tree(repo)
    stage_hits: dict[str, list[dict[str, Any]]] = {}

    for fi in files:
        for stage_id, desc, pattern in _PIPELINE_STAGES:
            hits = _match_lines(fi["text"], pattern, limit=30)
            if hits:
                stage_hits.setdefault(stage_id, []).append({
                    "file": fi["path"],
                    "matches": hits[:20],
                    "match_count": len(hits),
                })

    target_family = porting_map.target_family(target_arch) if target_arch else ""
    source_family = porting_map.target_family(source_arch) if source_arch else ""

    # Collect all constructs across all files
    all_constructs: set[str] = set()
    for fi in files:
        cs = porting_map.detect_constructs(fi["text"])
        all_constructs.update(cs)

    construct_table: dict[str, Any] = {}
    if target_arch and all_constructs:
        for c in sorted(all_constructs):
            sub = porting_map._substitution(c, target_family)
            construct_table[c] = {
                "role": porting_map.CONSTRUCTS.get(c, {}).get("role", "?"),
                "target_use": sub.get("use", ""),
                "grade": sub.get("grade", "manual"),
                "note": sub.get("note", ""),
            }

    pipeline_flow: list[dict[str, Any]] = []
    for stage_id, desc, pattern in _PIPELINE_STAGES:
        hits = stage_hits.get(stage_id, [])
        total = sum(h["match_count"] for h in hits)
        pipeline_flow.append({
            "stage": stage_id,
            "description": desc,
            "files_with_stage": [h["file"] for h in hits],
            "total_matches": total,
            "evidence": "PASS" if total > 0 else "NOT FOUND",
        })

    # Dispatch patterns
    dispatch_findings: dict[str, Any] = {}
    for fi in files:
        for dp_id, dp_desc, dp_pattern in _DISPATCH_PATTERNS:
            hits = _match_lines(fi["text"], dp_pattern, limit=15)
            if hits:
                d = dispatch_findings.setdefault(dp_id,
                    {"description": dp_desc, "files": [], "total": 0})
                d["files"].append({"file": fi["path"], "matches": hits})
                d["total"] += len(hits)

    # Memory patterns
    mem_findings: dict[str, Any] = {}
    for fi in files:
        for mp_id, mp_desc, mp_pattern in _MEMORY_PATTERNS:
            hits = _match_lines(fi["text"], mp_pattern, limit=15)
            if hits:
                m = mem_findings.setdefault(mp_id,
                    {"description": mp_desc, "files": [], "total": 0})
                m["files"].append({"file": fi["path"], "matches": hits})
                m["total"] += len(hits)

    return {
        "repo": repo,
        "target_arch": target_arch,
        "target_family": target_family,
        "source_arch": source_arch,
        "source_family": source_family,
        "files_scanned": len(files),
        "stages": pipeline_flow,
        "dispatch_patterns": dispatch_findings,
        "memory_patterns": mem_findings,
        "constructs_used": sorted(all_constructs),
        "construct_table": construct_table,
    }


# ---------------------------------------------------------------------------
# Core engine dispatch analysis
# ---------------------------------------------------------------------------

# Patterns for identifying dispatch \ kernel-launch orchestration
_CORE_DISPATCH_RE = r"(cuda|hip|HIP|cuMem|nvrtc|kernel)\s*(.+Launch|LaunchKernel|launchKernel)"
_CORE_KERNELS = [
    ("kernel_decl",     "Kernel entry point",       r"__global__"),
    ("kernel_launch",   "Kernel launch from host",  r"<<<|LaunchKernel|\.launch\("),
    ("kernel_dispatch", "Dispatch table \ function map", r"dispatch|kern_table|kernel_table|dispatch_table|fn_table"),
    ("weight_format",   "Weight format selector",   r"gguf|_format|_quant_type|GGML_.*_F32|GGML_.*_I8|GGML_.*_Q4|GGML_.*_Q8|GGML_.*_F16"),
    ("backend_select",  "Backend selection",        r"backend.*=|ggml_backend|hipSetBackend|cudaSetDevice|cudaGetDevice"),
    ("stream_sync",     "Stream \ sync management", r"stream|event|cudaStreamSync|hipStreamSync|__syncthreads|barrier"),
    ("memory_alloc",    "Device memory allocation", r"cudaMalloc|hipMalloc|ggml_backend_alloc|alloc.*buffer|device_alloc"),
    ("memory_copy",     "Host↔device copy",         r"cudaMemcpy|hipMemcpy|upload|download|copy.*to.*device|copy.*to.*host|ggml_backend_tensor_copy"),
]


def analyze_core(repo: str, target_arch: str = "") -> dict[str, Any]:
    """Analyse an engine's core dispatch architecture.

    Looks at how the engine launches kernels, manages memory, selects backends,
    handles weight formats, and orchestrates multi-GPU \ tensor-parallel ops.
    Returns a structured map of the engine's dispatch model.
    """
    files = _scan_tree(repo)
    target_family = porting_map.target_family(target_arch) if target_arch else ""

    dispatch_map: dict[str, Any] = {}
    all_constructs: set[str] = set()

    for fi in files:
        text = fi["text"]
        for kid, kdesc, kpat in _CORE_KERNELS:
            hits = _match_lines(text, kpat, limit=20)
            if hits:
                d = dispatch_map.setdefault(kid,
                    {"description": kdesc, "files": [], "total": 0})
                d["files"].append({"file": fi["path"], "matches": hits})
                d["total"] += len(hits)

        cs = porting_map.detect_constructs(text)
        all_constructs.update(cs)

    # Score constructs against target
    construct_table: dict[str, Any] = {}
    if target_arch and all_constructs:
        for c in sorted(all_constructs):
            sub = porting_map._substitution(c, target_family)
            construct_table[c] = {
                "role": porting_map.CONSTRUCTS.get(c, {}).get("role", "?"),
                "target_use": sub.get("use", ""),
                "grade": sub.get("grade", "manual"),
                "note": sub.get("note", ""),
            }

    # Dispatch pattern summary
    dispatch_summary: dict[str, Any] = {}
    if dispatch_map.get("kernel_launch", {}).get("total", 0) > 0:
        launch_files = [e["file"] for e in dispatch_map.get("kernel_launch", {}).get("files", [])]
        dispatch_summary["launch_style"] = "explicit kernel launch (<<< or LaunchKernel)"
        dispatch_summary["launch_files"] = launch_files
    if dispatch_map.get("kernel_dispatch", {}).get("total", 0) > 0:
        dispatch_summary["dispatch_style"] = "dispatch table \ function map"
    if dispatch_map.get("backend_select", {}).get("total", 0) > 0:
        dispatch_summary["backend_style"] = "runtime backend selection"

    return {
        "repo": repo,
        "target_arch": target_arch,
        "target_family": target_family,
        "files_scanned": len(files),
        "dispatch_map": dispatch_map,
        "dispatch_summary": dispatch_summary,
        "constructs_used": sorted(all_constructs),
        "construct_table": construct_table,
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_pipeline(md: dict[str, Any]) -> str:
    """Render analyze_pipeline() result as markdown."""
    L: list[str] = [f"# Full Pipeline Analysis — `{md['repo']}`\n"]
    if md.get("target_arch"):
        L.append(f"**Target:** `{md['target_arch']}` ({md.get('target_family','?')}) | ")
    if md.get("source_arch"):
        L.append(f"**Source:** `{md['source_arch']}` ({md.get('source_family','?')}) | ")
    L.append(f"**Files scanned:** {md['files_scanned']}\n")
    L.append("---\n")

    # Stages
    L.append("## Pipeline Stages\n\n")
    L.append("| # | Stage | Evidence | Files | Matches |\n")
    L.append("|---|---|---|---|---|\n")
    for i, s in enumerate(md["stages"], 1):
        files_str = ", ".join(s["files_with_stage"][:5])
        if len(s["files_with_stage"]) > 5:
            files_str += f" (+{len(s['files_with_stage'])-5} more)"
        L.append(f"| {i} | `{s['stage']}` — {s['description']} | {s['evidence']} | {files_str or '—'} | {s['total_matches']} |")

    # Dispatch patterns
    L.append("\n## Dispatch Patterns\n\n")
    if not md["dispatch_patterns"]:
        L.append("_No distinctive dispatch patterns detected._\n")
    for dp_id, dp in sorted(md["dispatch_patterns"].items()):
        L.append(f"### `{dp_id}` — {dp['description']}\n")
        L.append(f"**{dp['total']} hit(s)** in {len(dp['files'])} file(s):\n\n")
        for f in dp["files"]:
            L.append(f"- `{f['file']}` ({len(f['matches'])} match(es)):")
            for m in f["matches"][:5]:
                L.append(f"  - L{m['line']}: `{m['match']}`")
            L.append("")
        L.append("")

    # Memory patterns
    L.append("## Memory Management Patterns\n\n")
    if not md["memory_patterns"]:
        L.append("_No specific memory patterns detected._\n")
    for mp_id, mp in sorted(md["memory_patterns"].items()):
        L.append(f"### `{mp_id}` — {mp['description']}\n")
        L.append(f"**{mp['total']} hit(s)** in {len(mp['files'])} file(s):\n\n")
        for f in mp["files"]:
            L.append(f"- `{f['file']}` ({len(f['matches'])} match(es)):")
            for m in f["matches"][:3]:
                L.append(f"  - L{m['line']}: `{m['match']}`")
            L.append("")
        L.append("")

    # Constructs
    if md["constructs_used"] and md.get("construct_table"):
        L.append("## Porting-Critical Constructs\n\n")
        L.append("| construct | role | target use | grade | note |\n")
        L.append("|---|---|---|---|---|\n")
        for c, t in sorted(md["construct_table"].items()):
            L.append(f"| `{c}` | {t['role']} | {t['target_use']} | **{t['grade']}** | {t['note']} |")
        L.append("")

    return "\n".join(L)


def render_core(md: dict[str, Any]) -> str:
    """Render analyze_core() result as markdown."""
    L: list[str] = [f"# Core Engine Analysis — `{md['repo']}`\n"]
    if md.get("target_arch"):
        L.append(f"**Target:** `{md['target_arch']}` ({md.get('target_family','?')}) | ")
    L.append(f"**Files scanned:** {md['files_scanned']}\n")
    L.append("---\n")

    L.append("## Dispatch Summary\n\n")
    ds = md.get("dispatch_summary", {})
    for k, v in ds.items():
        if isinstance(v, list):
            L.append(f"- **{k}:** {', '.join(v[:5])}")
        else:
            L.append(f"- **{k}:** {v}")
    L.append("")

    L.append("\n## Dispatch Map\n\n")
    L.append("| Pattern | Description | Files | Hits |\n")
    L.append("|---|---|---|---|\n")
    for kid, kd in sorted(md["dispatch_map"].items()):
        files_str = ", ".join(f["file"] for f in kd["files"][:5])
        if len(kd["files"]) > 5:
            files_str += f" (+{len(kd['files'])-5} more)"
        L.append(f"| `{kid}` | {kd['description']} | {files_str or '—'} | {kd['total']} |")
    L.append("")

    if md["constructs_used"] and md.get("construct_table"):
        L.append("## Porting-Critical Constructs\n\n")
        L.append("| construct | role | target use | grade | note |\n")
        L.append("|---|---|---|---|---|\n")
        for c, t in sorted(md["construct_table"].items()):
            L.append(f"| `{c}` | {t['role']} | {t['target_use']} | **{t['grade']}** | {t['note']} |")
        L.append("")

    return "\n".join(L)