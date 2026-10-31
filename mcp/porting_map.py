"""
porting_map — build a function-to-function porting map from source, for a target.

The problem this solves
-----------------------
A porting effort needs to answer, per function, "what do I replace this with on
the target, and how much fidelity do I lose?" Answering that from memory is how
a port gets designed around an instruction the target cannot encode.

So the map is built in two layers:

1. **Construct table** (`CONSTRUCTS`) — reusable knowledge: for each source
   construct, what each target family substitutes, with an explicit fidelity
   grade. Data, not prose.
2. **Function extraction** — the actual source is parsed: every kernel and
   device function, the constructs it uses, its inferred role, and the worst
   fidelity grade among its constructs.

`verify_map()` then replaces assumption with measurement: every ISA construct in
the map is compiled for the real target via `isa_tools.probe_instruction`, and
the grade is upgraded or downgraded accordingly.

Fidelity grades, worst to best:
    manual      no known mapping; needs a human
    lossy       mapping changes numerics materially
    approximate mapping is not bit-identical but is structurally correct
    portable    semantics preserved by a target-native equivalent
    exact       identical instruction semantics available
    drop        construct is unnecessary on the target (e.g. a cache hint)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- fidelity

ORDER = ["manual", "lossy", "approximate", "portable", "exact", "drop"]


def worst(grades: list[str]) -> str:
    known = [g for g in grades if g in ORDER]
    if not known:
        return "manual"
    return min(known, key=ORDER.index)


# --------------------------------------------------------------------------- construct table
# `family` keys: amd_rdna (gfx10\\11\\12), amd_cdna (gfx90x\\94x), nv_sm80, nv_sm70,
# intel_xe. "any" applies to every target.
CONSTRUCTS: dict[str, dict[str, Any]] = {
    # ---- NVIDIA PTX matrix / async ----
    "mma.sync": {
        "kind": "isa", "family": "nvidia-ptx", "role": "matrix multiply-accumulate",
        "pattern": r"mma\.sync\.[a-z0-9.]+",
        "map": {
            "amd_rdna": {"use": "wmma (rocWMMA) if a tile matches, else scalar dot reconstruction",
                         "grade": "approximate",
                         "note": "gfx12 has no mma.sync; cross-lane K fragments cannot be reproduced 1:1"},
            "amd_cdna": {"use": "mfma / v_mfma_*", "grade": "portable"},
            "intel_xe": {"use": "joint_matrix / DPAS", "grade": "portable"},
        },
    },
    "cp.async": {
        "kind": "isa", "family": "nvidia-ptx", "role": "async global->shared copy",
        "pattern": r"cp\.async\.[a-z0-9_.]+",
        "map": {
            "amd_rdna": {"use": "plain global->lds store + __syncthreads()", "grade": "portable",
                         "note": "no async-copy ISA; prefetch manually or rely on occupancy"},
            "amd_cdna": {"use": "global_load_lds + s_waitcnt", "grade": "portable"},
            "intel_xe": {"use": "async_copy / prefetch", "grade": "portable"},
        },
    },
    "bar.sync": {
        "kind": "isa", "family": "nvidia-ptx", "role": "named barrier",
        "pattern": r"bar\.sync",
        "map": {"any": {"use": "__syncthreads() (or s_barrier)", "grade": "portable"}},
    },
    "ld.global.cv": {
        "kind": "isa", "family": "nvidia-ptx", "role": "uncached\\volatile global load",
        "pattern": r"ld\.global\.(?:cv|nc|ca|lu)",
        "map": {"any": {"use": "ordinary load (or __ldg for read-only)", "grade": "drop"}},
    },
    "cvta.to.shared": {
        "kind": "isa", "family": "nvidia-ptx", "role": "shared address-space conversion",
        "pattern": r"cvta\.to\.shared",
        "map": {"any": {"use": "flat pointer cast to uintptr_t", "grade": "portable"}},
    },
    "dp4a": {
        "kind": "isa", "family": "nvidia-ptx", "role": "4-way int8 dot product",
        "pattern": r"dp4a\.s32|__dp4a|amd_mixed_dot|v_dot4|v_dot8|udot4|sdot4",
        "map": {
            "amd_rdna": {"use": "amd_mixed_dot(char4,...) -> v_dot4_i32_iu8, or __builtin_amdgcn_udot4 -> v_dot4_u32_u8",
                         "grade": "exact",
                         "note": "measured on gfx1201: the builtins DO emit v_dot4 hardware. "
                                 "Only the raw `v_dot4_i32_i8` spelling fails (gfx9 encoding) — "
                                 "do not hand-write the mnemonic, use the builtin."},
            "amd_cdna": {"use": "v_dot4_i32_i8 / __builtin_amdgcn_sdot4", "grade": "exact"},
            "nv_sm80": {"use": "__dp4a", "grade": "exact"},
        },
    },
    # ---- warp intrinsics ----
    "shfl_sync": {
        "kind": "intrinsic", "family": "cuda", "role": "warp lane exchange",
        "pattern": r"__shfl(?:_xor|_up|_down)?_sync\s*\(",
        "map": {
            "amd_rdna": {"use": "__shfl* (3-arg, width=wave_size); mask is implicit",
                         "grade": "portable",
                         "note": "HIP *_sync builtins demand a 64-bit mask; drop to the non-sync form when all lanes are active"},
            "amd_cdna": {"use": "__shfl* or ds_bpermute", "grade": "portable"},
            "intel_xe": {"use": "sub_group shuffle", "grade": "portable"},
        },
    },
    "ballot_sync": {
        "kind": "intrinsic", "family": "cuda", "role": "warp vote",
        "pattern": r"__(?:ballot|any|all)_sync\s*\(",
        "map": {"amd_rdna": {"use": "__ballot / __any / __all", "grade": "portable"},
                "intel_xe": {"use": "sub_group any\\all", "grade": "portable"}},
    },
    "activemask": {
        "kind": "intrinsic", "family": "cuda", "role": "active lane mask",
        "pattern": r"__activemask\s*\(",
        "map": {"amd_rdna": {"use": "full mask constant (all lanes active by construction)",
                             "grade": "portable"}},
    },
    "syncwarp": {
        "kind": "intrinsic", "family": "cuda", "role": "warp barrier",
        "pattern": r"__syncwarp\s*\(",
        "map": {"amd_rdna": {"use": "no-op within a wave, or __builtin_amdgcn_wave_barrier",
                             "grade": "drop"}},
    },
    "ldg": {
        "kind": "intrinsic", "family": "cuda", "role": "read-only cached load",
        "pattern": r"__ldg\s*\(",
        "map": {"amd_rdna": {"use": "__ldg (supported) or plain load", "grade": "portable"}},
    },
    "atomic": {
        "kind": "intrinsic", "family": "cuda", "role": "atomic RMW",
        "pattern": r"\batomic(?:Add|CAS|Exch|Min|Max|And|Or|Xor|Inc|Dec)\s*\(",
        "map": {"amd_rdna": {"use": "atomicAdd etc. (HIP equivalents)", "grade": "portable"},
                "intel_xe": {"use": "atomic_ref", "grade": "portable"}},
    },
    "fence": {
        "kind": "intrinsic", "family": "cuda", "role": "memory fence",
        "pattern": r"__threadfence(?:_block|_system)?\s*\(",
        "map": {"amd_rdna": {"use": "__threadfence* (HIP) or s_waitcnt + fence",
                             "grade": "portable"}},
    },
    # ---- fp16 conversion ----
    "cvt_f16": {
        "kind": "isa", "family": "mixed", "role": "fp16<->fp32 conversion",
        "pattern": r"cvt\.(?:rn\.)?f16|__float2half|__half2float|__half2|v_cvt_",
        "map": {
            "amd_rdna": {"use": "v_cvt_f16_f32 / v_cvt_f32_f16 (both encode on gfx12); "
                                "v_cvt_pk_f16_f32 does NOT",
                         "grade": "portable"},
            "nv_sm80": {"use": "__float2half etc.", "grade": "exact"},
        },
    },
    # ---- launch / host side ----
    "launch_bounds": {
        "kind": "qualifier", "family": "cuda", "role": "occupancy hint",
        "pattern": r"__launch_bounds__\s*\(",
        "map": {"amd_rdna": {"use": "__launch_bounds__ (accepted) or __attribute__((amdgpu_flat_work_group_size))",
                             "grade": "portable",
                             "note": "strip before scanning signatures: the parenthesised hint corrupts parameter parsing"},
        },
    },
    "cuda_arch_macro": {
        "kind": "macro", "family": "cuda", "role": "arch-conditional compilation",
        "pattern": r"__CUDA_ARCH__|__HIP_DEVICE_COMPILE__|__HIP__",
        "map": {"any": {"use": "keep both branches; select with the target's own macro",
                        "grade": "portable"}},
    },
    "texture": {
        "kind": "feature", "family": "cuda", "role": "texture\\surface read",
        "pattern": r"\btexture\s*<|tex1Dfetch|tex2D\s*\(",
        "map": {"amd_rdna": {"use": "image object or plain global load", "grade": "lossy"},
                "intel_xe": {"use": "image accessor", "grade": "lossy"}},
    },
    "cooperative_groups": {
        "kind": "feature", "family": "cuda", "role": "grid-wide sync / groups",
        "pattern": r"cooperative_groups|grid\.sync\s*\(",
        "map": {"amd_rdna": {"use": "no grid sync; restructure into multiple launches",
                             "grade": "manual"}},
    },
    "dynamic_parallelism": {
        "kind": "feature", "family": "cuda", "role": "device-side kernel launch",
        "pattern": r"<<<[^>]*>>>",
        "map": {"amd_rdna": {"use": "restructure into a host loop or a persistent kernel",
                             "grade": "manual"}},
    },
    "sync": {
        "kind": "intrinsic", "family": "cuda", "role": "block barrier",
        "pattern": r"__syncthreads\s*\(",
        "map": {"any": {"use": "__syncthreads()", "grade": "exact"}},
    },
    # ---- cross-ecosystem: kernel entry / barrier / local memory / ids ----
    "kernel_entry": {
        "kind": "qualifier", "family": "cross", "role": "kernel entry point",
        "pattern": (r"__global__|\b__kernel\b|\bkernel\s+void\b|\bparallel_for\b|"
                    r"\[numthreads\s*\(|#\[spirv\(|#\[kernel\]"),
        "map": {
            "amd_rdna": {"use": "extern \"C\" __global__ void f(...)", "grade": "portable"},
            "nv_sm80": {"use": "__global__ void f(...)", "grade": "portable"},
            "intel_xe": {"use": "parallel_for<nd_range> or SYCL kernel", "grade": "portable"},
        },
    },
    "wg_barrier": {
        "kind": "intrinsic", "family": "cross", "role": "work-group barrier",
        "pattern": (r"\bbarrier\s*\(\s*CLK_|threadgroup_barrier|"
                    r"GroupMemoryBarrier(?:WithGroupSync)?|group_barrier|"
                    r"\bbarrier\s*\(\s*\)"),
        "map": {
            "amd_rdna": {"use": "__syncthreads()", "grade": "portable"},
            "nv_sm80": {"use": "__syncthreads()", "grade": "portable"},
            "intel_xe": {"use": "group_barrier(group) / nd_item::barrier", "grade": "portable"},
        },
        "notes": "a work-group is a thread-block: same semantics, different spelling",
    },
    "local_memory": {
        "kind": "qualifier", "family": "cross", "role": "work-group shared memory",
        "pattern": (r"\b__shared__\b|\b__local\b|\bthreadgroup\b|\bgroupshared\b|"
                    r"local_accessor|shared\s+\w+\s+\w+\s*\["),
        "map": {
            "amd_rdna": {"use": "__shared__ T name[N]  (LDS)", "grade": "portable"},
            "nv_sm80": {"use": "__shared__ T name[N]", "grade": "portable"},
            "intel_xe": {"use": "local_accessor<T> (group scope)", "grade": "portable"},
        },
    },
    "global_id": {
        "kind": "intrinsic", "family": "cross", "role": "thread index",
        "pattern": (r"\bget_global_id\s*\(|thread_position_in_grid|"
                    r"SV_DispatchThreadID|gl_GlobalInvocationID|"
                    r"get_global_linear_id"),
        "map": {
            "amd_rdna": {"use": "blockIdx.x*blockDim.x + threadIdx.x", "grade": "portable"},
            "nv_sm80": {"use": "blockIdx.x*blockDim.x + threadIdx.x", "grade": "portable"},
            "intel_xe": {"use": "item.get_global_id(0)", "grade": "portable"},
        },
    },
    "subgroup_ops": {
        "kind": "intrinsic", "family": "cross", "role": "subgroup\\wave exchange",
        "pattern": (r"sub_group_\w+|simd_(?:shuffle|sum|broadcast|group)|"
                    r"Wave(?:ReadLaneAt|ActiveCount|PrefixSum|Shuffle)|"
                    r"subgroup(?:Shuffle|Broadcast|Add|Ballot)"),
        "map": {
            "amd_rdna": {"use": "__shfl_* / DPP (wave32)", "grade": "portable"},
            "nv_sm80": {"use": "__shfl_sync / __reduce_* (warp)", "grade": "portable"},
            "intel_xe": {"use": "sub_group shuffle\\broadcast", "grade": "portable"},
        },
        "notes": "subgroup == warp\\wave; width differs by arch, so state it",
    },
}

#: Function-name fragments -> inferred role. Ordered; first hit wins.
ROLE_HINTS: list[tuple[str, str]] = [
    ("attn", "attention"), ("softmax", "attention"), ("rope", "positional"),
    ("gemv", "gemv"), ("gemm", "gemm"), ("mm_", "gemm"), ("g4", "gemv"),
    ("g5", "gemv"), ("gdn", "recurrent"), ("ssm", "recurrent"), ("conv", "recurrent"),
    ("rms", "normalisation"), ("norm", "normalisation"), ("l2w", "normalisation"),
    ("fold", "hadamard"), ("fwht", "hadamard"), ("hadamard", "hadamard"),
    ("deq", "dequantise"), ("q8", "quantise"), ("pack", "layout"),
    ("argmax", "sampling"), ("topk", "sampling"), ("softmax", "sampling"),
    ("expand", "elementwise"), ("band", "elementwise"), ("mask", "elementwise"),
    ("xchg", "communication"), ("sum", "reduction"), ("store", "layout"),
    ("copy", "layout"), ("tc_", "tensor-core"), ("wmma", "tensor-core"),
]


# --------------------------------------------------------------------------- source parsing
#
# Kernel entry points are spelled differently in every ecosystem, so function
# discovery is table-driven. A CUDA-only matcher silently returns zero
# functions for a SYCL, OpenCL, GLSL, HLSL or Metal tree — the map then looks
# empty and the port is planned from nothing.

LANGUAGES: dict[str, dict[str, Any]] = {
    "cuda": {
        "ext": (".cu", ".cuh", ".hip"),
        "markers": ((r"__global__", "global"), (r"__device__", "device"),
                    (r"__host__", "host")),
    },
    "opencl": {
        "ext": (".cl",),
        "markers": ((r"__kernel\b", "kernel"), (r"\bkernel\b", "kernel")),
    },
    "metal": {
        "ext": (".metal",),
        "markers": ((r"\bkernel\s+void\b", "kernel"),
                    (r"\bvertex\s+", "vertex"), (r"\bfragment\s+", "fragment")),
    },
    "sycl": {
        "ext": (".sycl", ".dp.cpp"),
        "markers": ((r"parallel_for\s*<", "kernel"), (r"parallel_for\s*\(", "kernel"),
                    (r"SYCL_EXTERNAL", "device")),
    },
    "glsl": {
        "ext": (".glsl", ".vert", ".frag", ".comp", ".geom", ".tesc", ".tese"),
        "markers": ((r"\bvoid\s+(main)\s*\(", "entry"),
                    (r"layout\s*\([^)]*\)\s*(?:in|out)?\s*\w+\s+(\w+)\s*\(", "shader")),
    },
    "hlsl": {
        "ext": (".hlsl", ".hlsli"),
        "markers": ((r"\[numthreads\s*\([^)]*\)\]\s*(?:void|float[234]?|half[234]?|int|uint)\s+(\w+)\s*\(", "compute"),
                    (r"\b(?:float|half|int|uint)[234]?\s+(\w+)\s*\(", "entry")),
    },
    "rust_gpu": {
        "ext": (".rs",),
        "markers": ((r"#\[spirv\(", "entry"), (r"#\[kernel\]", "kernel"),
                    (r"#\[no_mangle\]", "entry")),
    },
}


def language_for(path: str, src: str = "") -> str | None:
    """Infer source language from the file name, then from content."""
    name = Path(path).name.lower()
    if name.endswith(".dp.cpp"):
        return "sycl"
    suffix = Path(path).suffix.lower()
    for lang, spec in LANGUAGES.items():
        if suffix in spec["ext"]:
            return lang
    # ambiguous C++\\Rust containers: sniff for the framework's own signatures
    if src:
        if "#include <sycl" in src or "sycl::" in src:
            return "sycl"
        if "__global__" in src or "__device__" in src or "cuda" in src.lower()[:400]:
            return "cuda"
        if "#[spirv(" in src or "#[kernel]" in src:
            return "rust_gpu"
    return None


def _markers_for(path: str, src: str = "") -> tuple[str, list[tuple[str, str]]]:
    """(language, [(regex, kind)]) for a file."""
    lang = language_for(path, src)
    if lang is None:
        lang = "cuda"          # closest superset for bare kernel code
    return lang, list(LANGUAGES[lang]["markers"])


#: Output extension per (target backend, source language). ``None`` keeps the
#: source extension unchanged — the file is already in a language the target
#: backend consumes directly.
TARGET_EXTENSIONS: dict[str, dict[str, str | None]] = {
    "hip": {
        "cuda": ".hip",        # HIP source; .cu also works if the build sets -x hip
        "opencl": ".cpp",      # rewritten as HIP C++
        "metal": ".cpp",
        "sycl": ".cpp",
        "glsl": ".cpp",        # shader rewritten as a compute kernel
        "hlsl": ".cpp",
        "rust_gpu": ".rs",
    },
    "cuda": {
        "cuda": ".cu",
        "opencl": ".cu",
        "metal": ".cu",
        "sycl": ".cu",
        "glsl": ".cu",
        "hlsl": ".cu",
        "rust_gpu": ".rs",
    },
    "sycl": {
        "cuda": ".cpp",
        "opencl": ".cpp",
        "metal": ".cpp",
        "sycl": None,
        "glsl": ".cpp",
        "hlsl": ".cpp",
        "rust_gpu": ".rs",
    },
    "vulkan": {
        "glsl": None,          # already a Vulkan-consumable shader
        "hlsl": ".comp",       # cross-compile via dxc -> SPIR-V
        "cuda": ".comp",
        "opencl": ".comp",
        "metal": ".comp",
        "sycl": ".comp",
        "rust_gpu": ".comp",
    },
    "directx": {
        "hlsl": None,
        "glsl": ".hlsl",       # via spirv-cross / manual
        "cuda": ".hlsl",
        "opencl": ".hlsl",
        "metal": ".hlsl",
        "sycl": ".hlsl",
        "rust_gpu": ".hlsl",
    },
}


def output_extension(src_path: str, backend: str, src_text: str = "") -> str | None:
    """Output extension for a ported file, or None to keep the original.

    Returns the literal source suffix when the language is unchanged, so callers
    can distinguish "keep" from "unknown backend".
    """
    lang = language_for(src_path, src_text)
    if lang is None:
        return None
    table = TARGET_EXTENSIONS.get(backend)
    if table is None:
        return None
    out = table.get(lang)
    if out is None:
        return Path(src_path).suffix
    return out


#: The language each backend consumes natively (no rewrite needed).
NATIVE_LANG = {"hip": "cuda", "cuda": "cuda", "sycl": "sycl",
               "vulkan": "glsl", "directx": "hlsl"}


def plan_outputs(files: list[str], backend: str, out_dir: str = "") -> list[dict[str, Any]]:
    """Map every source file to its ported path (extension-to-extension plan).

    Preserves relative structure under ``out_dir``; without ``out_dir`` the new
    name sits beside the source so the plan can be inspected before writing.
    """
    plan: list[dict[str, Any]] = []
    native = NATIVE_LANG.get(backend)
    for f in files:
        p = Path(f)
        head = ""
        if p.is_file():
            try:
                with p.open("r", encoding="utf-8", errors="replace") as fh:
                    head = fh.read(4096)
            except Exception:
                head = ""
        lang = language_for(f, head)
        ext = output_extension(f, backend, head)
        if ext is None:
            ext = p.suffix
        dst = p.with_suffix(ext)
        if out_dir:
            dst = Path(out_dir) / dst.name
        needs_rewrite = (lang != native) or (ext.lower() != p.suffix.lower())
        plan.append({
            "source": str(p),
            "language": lang or "unknown",
            "target_ext": ext,
            "output": str(dst),
            "rewrite": bool(needs_rewrite),
        })
    return plan


_COMMENT_RE = re.compile(r"\\\\[^\n]*|\\\*.*?\*\\", re.S)


def strip_comments(src: str) -> str:
    """Blank out comments while preserving line structure and offsets."""
    def blank(m: re.Match) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))
    return _COMMENT_RE.sub(blank, src)


def strip_launch_bounds(src: str) -> str:
    """Blank __launch_bounds__(...) while preserving offsets and line numbers.

    This must happen before locating a parameter list: the hint's parentheses
    are the first '(' after the qualifier, so leaving it in makes every such
    function parse as '__launch_bounds__' and silently vanish from the map.
    """
    return re.sub(r"__launch_bounds__\s*\([^)]*\)",
                  lambda m: re.sub(r"[^\n]", " ", m.group(0)), src)


_MACRO_RE = re.compile(r"^[ \t]*#define[ \t]+(\w+)((?:[^\n]*\\\n)*[^\n]*)", re.M)


def macro_constructs(src: str) -> dict[str, list[str]]:
    """Macro name -> constructs its definition uses.

    ISA constructs frequently live in macros (TC_MMA, ATC_MMA_*) rather than in
    function bodies, so a body-only scan reports a clean bill of health on code
    that is entirely built from an unsupported instruction.
    """
    out: dict[str, list[str]] = {}
    for m in _MACRO_RE.finditer(src):
        cs = detect_constructs(m.group(2))
        if cs:
            out[m.group(1)] = cs
    return out


@dataclass
class FuncInfo:
    name: str
    kind: str                     # global | device | host
    file: str
    line: int
    signature: str
    role: str
    constructs: list[str] = field(default_factory=list)
    fidelity: str = "manual"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_KIND_RE = re.compile(r"__global__|__device__|__host__")


def _match_brace(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_functions(src: str, path: str) -> list[FuncInfo]:
    """Find every kernel\\device function with its body, signature and constructs.

    Language is inferred from the file name; see LANGUAGES. For marker forms
    with no name (SYCL lambdas, GLSL entry points) a stable synthetic name is
    derived so the entry still appears in the map.
    """
    clean = strip_launch_bounds(strip_comments(src))
    macros = macro_constructs(clean)
    lang, markers = _markers_for(path, src)
    out: list[FuncInfo] = []
    seen: set[tuple[int, str]] = set()
    for pat, kind in markers:
        for m in re.finditer(pat, clean):
            if m.start() in {s for s, _ in seen}:
                continue
            # some markers already include the '(' (e.g. "void main(")
            if clean[m.end() - 1] == "(":
                p_open = m.end() - 1
            else:
                p_open = clean.find("(", m.end())
            if p_open < 0:
                continue
            head = clean[m.end():p_open]
            ids = re.findall(r"[A-Za-z_]\w*", head)
            line = clean.count("\n", 0, m.start()) + 1
            if ids:
                name = ids[-1]
            else:
                name = f"{lang}_entry"
            p_close = clean.find(")", p_open)
            if p_close < 0:
                continue
            b_open = clean.find("{", p_close)
            if b_open < 0:
                continue
            semi = clean.find(";", p_close)
            if 0 <= semi < b_open:
                continue                      # declaration only
            b_close = _match_brace(clean, b_open)
            if b_close < 0:
                continue
            key = (m.start(), name)
            if key in seen:
                continue
            seen.add(key)
            body = clean[b_open:b_close + 1]
            sig = re.sub(r"\s+", " ", clean[m.start():b_open]).strip()
            constructs = detect_constructs(body)
            for mname, mcs in macros.items():
                if re.search(r"\b" + re.escape(mname) + r"\s*\(", body):
                    for c in mcs:
                        if c not in constructs:
                            constructs.append(c)
            role = classify_role(name, body)
            grade = worst([_grade_for(c, "amd_rdna") for c in constructs]) if constructs else "exact"
            info = FuncInfo(name=name, kind=kind, file=path, line=line,
                            signature=sig[:200], role=role,
                            constructs=sorted(constructs), fidelity=grade)
            if lang != "cuda":
                info.notes.append(f"language: {lang}")
            if "template" in clean[max(0, m.start() - 200):m.start()]:
                info.notes.append("templated: map instantiation-by-instantiation")
            out.append(info)
    return out


def detect_constructs(text: str) -> list[str]:
    """Construct ids present in a snippet."""
    found = []
    for cid, spec in CONSTRUCTS.items():
        if re.search(spec["pattern"], text):
            found.append(cid)
    return found


def classify_role(name: str, body: str) -> str:
    low = name.lower()
    for frag, role in ROLE_HINTS:
        if frag in low:
            return role
    for frag, role in ROLE_HINTS:
        if frag in body.lower():
            return role
    return "other"


def _grade_for(construct: str, family: str) -> str:
    spec = CONSTRUCTS.get(construct)
    if not spec:
        return "manual"
    m = spec["map"]
    if family in m:
        return m[family]["grade"]
    if "any" in m:
        return m["any"]["grade"]
    # closest available family for grading purposes
    for alt in ("amd_rdna", "amd_cdna", "nv_sm80"):
        if alt in m:
            return m[alt]["grade"]
    return "manual"


def _substitution(construct: str, family: str) -> dict[str, str]:
    spec = CONSTRUCTS.get(construct, {})
    m = spec.get("map", {})
    if family in m:
        return m[family]
    if "any" in m:
        return m["any"]
    for alt in ("amd_rdna", "amd_cdna", "nv_sm80"):
        if alt in m:
            d = dict(m[alt])
            d["note"] = (d.get("note", "") + " (grade borrowed from %s)" % alt).strip()
            return d
    return {"use": "no known mapping", "grade": "manual"}


# --------------------------------------------------------------------------- map building

def target_family(arch: str) -> str:
    a = arch.lower()
    if a.startswith("gfx1"):          # gfx10\\11\\12 = RDNA
        return "amd_rdna"
    if a.startswith("gfx9"):
        return "amd_cdna"
    if a.startswith(("sm_", "compute_")):
        try:
            n = int(re.sub(r"\D", "", a)[:3])
        except ValueError:
            n = 80
        return "nv_sm70" if n < 80 else "nv_sm80"
    if a.startswith(("spir", "intel", "gen")):
        return "intel_xe"
    return "amd_rdna"


def backend_for(family: str) -> str:
    """Target backend implied by an architecture family."""
    return {"amd_rdna": "hip", "amd_cdna": "hip",
            "nv_sm80": "cuda", "nv_sm70": "cuda",
            "intel_xe": "sycl"}.get(family, "hip")


def build_map(files: list[str], source_arch: str, target_arch: str,
              target_backend: str | None = None) -> dict[str, Any]:
    """Build the full function-to-function porting map."""
    fam = target_family(target_arch)
    backend = target_backend or backend_for(fam)
    funcs: list[FuncInfo] = []
    for f in files:
        p = Path(f)
        if not p.is_file():
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        funcs.extend(extract_functions(src, str(p)))

    used = sorted({c for fn in funcs for c in fn.constructs})
    table = {}
    for c in used:
        sub = _substitution(c, fam)
        table[c] = {
            "role": CONSTRUCTS.get(c, {}).get("role", "?"),
            "target_use": sub.get("use", ""),
            "grade": sub.get("grade", "manual"),
            "note": sub.get("note", ""),
        }

    by_role: dict[str, int] = {}
    by_grade: dict[str, int] = {}
    for fn in funcs:
        by_role[fn.role] = by_role.get(fn.role, 0) + 1
        by_grade[fn.fidelity] = by_grade.get(fn.fidelity, 0) + 1

    return {
        "source_arch": source_arch,
        "target_arch": target_arch,
        "target_family": fam,
        "target_backend": backend,
        "function_count": len(funcs),
        "constructs_used": used,
        "construct_table": table,
        "roles": dict(sorted(by_role.items(), key=lambda kv: -kv[1])),
        "fidelity": dict(sorted(by_grade.items(), key=lambda kv: ORDER.index(kv[0])
                                if kv[0] in ORDER else 99)),
        "file_plan": plan_outputs(files, backend),
        "functions": [fn.as_dict() for fn in funcs],
    }


def render_markdown(m: dict[str, Any]) -> str:
    """Human-readable map: construct table first, then per-function decisions."""
    L = [f"# Porting map — `{m['source_arch']}` → `{m['target_arch']}` "
         f"({m['target_family']})\n",
         f"{m['function_count']} functions analysed.\n",
         "## Constructs in use\n",
         "| construct | role | use on target | fidelity | note |",
         "|---|---|---|---|---|"]
    order = {g: i for i, g in enumerate(ORDER)}
    for c, t in sorted(m["construct_table"].items(),
                       key=lambda kv: order.get(kv[1]["grade"], 99)):
        L.append(f"| `{c}` | {t['role']} | {t['target_use']} | **{t['grade']}** | "
                 f"{t['note']} |")
    L.append("\n## Fidelity summary\n")
    for g, n in m["fidelity"].items():
        L.append(f"- {g}: {n} function(s)")
    L.append("\n## Functions needing attention\n")
    L.append("| function | role | worst fidelity | constructs | file:line |")
    L.append("|---|---|---|---|---|")
    bad = [f for f in m["functions"]
           if f["fidelity"] in ("manual", "lossy", "approximate")]
    bad.sort(key=lambda f: order.get(f["fidelity"], 99))
    for f in bad:
        L.append(f"| `{f['name']}` | {f['role']} | **{f['fidelity']}** | "
                 f"{', '.join(f['constructs']) or '—'} | "
                 f"`{Path(f['file']).name}:{f['line']}` |")
    exact = [f for f in m["functions"] if f["fidelity"] in ("exact", "portable", "drop")]
    L.append(f"\n{len(exact)} function(s) port mechanically "
             f"(exact\\portable\\drop) — no decision required.\n")

    plan = m.get("file_plan") or []
    if plan:
        L.append("\n## File plan (extension → extension)\n")
        L.append(f"Target backend: `{m.get('target_backend', '?')}`\n")
        L.append("| source | language | output | rewritten |")
        L.append("|---|---|---|---|")
        for e in plan[:40]:
            L.append(f"| `{Path(e['source']).name}` | {e['language']} | "
                     f"`{Path(e['output']).name}` | {'yes' if e['rewrite'] else 'copy'} |")
        if len(plan) > 40:
            L.append(f"\n... and {len(plan) - 40} more")
        n_rewrite = sum(1 for e in plan if e["rewrite"])
        L.append(f"\n{n_rewrite} of {len(plan)} file(s) change extension and need a rewrite; "
                 f"the rest are copied verbatim.")
    return "\n".join(L)


def lookup(construct: str, target_arch: str) -> dict[str, Any]:
    """What should `construct` become on `target_arch`?"""
    fam = target_family(target_arch)
    if construct not in CONSTRUCTS:
        return {"construct": construct, "known": False,
                "target_arch": target_arch, "target_family": fam,
                "use": "unknown construct — probe it on the target",
                "grade": "manual"}
    sub = _substitution(construct, fam)
    return {"construct": construct, "known": True,
            "role": CONSTRUCTS[construct].get("role", ""),
            "target_arch": target_arch, "target_family": fam,
            "use": sub.get("use", ""), "grade": sub.get("grade", "manual"),
            "note": sub.get("note", "")}


# --------------------------------------------------------------------------- verification

#: Canned statements that exercise each construct's *replacement*, ready for
#: isa_tools.probe_instruction. The probe template provides `int a, b, acc`
#: and `char4 ca, cb`.
REPLACEMENT_PROBES: dict[str, str] = {
    "dp4a": "acc = amd_mixed_dot(ca, cb, acc, false);",
    "sync": "__syncthreads();",
    "bar.sync": "__syncthreads();",
    "cp.async": "__syncthreads();",
    "ld.global.cv": "acc += a;",
    "cvta.to.shared": "acc += (int)((unsigned int)(uintptr_t)&a);",
    "shfl_sync": "acc = __shfl_xor(acc, 1, 32);",
    "ballot_sync": "acc += __ballot(acc != 0);",
    "activemask": "acc += (int)0xFFFFFFFFu;",
    "syncwarp": "acc += 0;",
    "fence": "__threadfence(); acc += 0;",
    "cvt_f16": 'unsigned short sh; asm("v_cvt_f16_f32 %0, %1;" : "=v"(sh) : "v"(acc)); acc += sh;',
    "ldg": "acc = __ldg(&a);",
    "atomic": "atomicAdd(&acc, 1);",
    "launch_bounds": "acc += 1;",
    "cuda_arch_macro": "acc += 1;",
    "mma.sync": "acc += (int)((signed char)(a & 0xFF)) * (int)((signed char)(b & 0xFF));",
}


def probe_candidates(constructs: list[str] | None = None) -> list[dict[str, str]]:
    """Build probe_isa candidates for the replacements of these constructs."""
    ids = constructs if constructs is not None else sorted(REPLACEMENT_PROBES)
    out = []
    for c in ids:
        stmt = REPLACEMENT_PROBES.get(c)
        if stmt:
            out.append({"name": c, "stmt": stmt})
    return out


def verify(m: dict[str, Any], replacement_probes: dict[str, bool]) -> dict[str, Any]:
    """Fold real probe results into the map.

    ``replacement_probes`` maps construct id -> whether that construct's
    **mapped replacement** encodes on the target.

    The distinction matters: `bar.sync` has no AMD encoding, yet its replacement
    `__syncthreads()` does, so the mapping is sound. Probing the source
    instruction instead would wrongly condemn a perfectly portable mapping.
    Use :func:`probe_candidates` to generate the statements to test.
    """
    changes = []
    for cid, ok in replacement_probes.items():
        if cid not in m["construct_table"]:
            continue
        t = m["construct_table"][cid]
        if not ok and t["grade"] in ("exact", "portable"):
            changes.append({"construct": cid, "was": t["grade"], "now": "manual",
                            "reason": "mapped replacement does not encode on target"})
            t["grade"] = "manual"
            t["note"] = (t["note"] + "; REPLACEMENT PROBE FAILED").strip("; ")
        elif ok and t["grade"] == "manual":
            changes.append({"construct": cid, "was": "manual", "now": "portable",
                            "reason": "mapped replacement encodes on target"})
            t["grade"] = "portable"
    # recompute per-function worst grade
    for fn in m["functions"]:
        if fn["constructs"]:
            fn["fidelity"] = worst([m["construct_table"].get(c, {}).get("grade", "manual")
                                    for c in fn["constructs"]])
    return {"changes": changes,
            "fidelity": dict(sorted(
                (g, sum(1 for f in m["functions"] if f["fidelity"] == g))
                for g in ORDER if any(f["fidelity"] == g for f in m["functions"])))}


def unresolved(m: dict[str, Any]) -> list[dict[str, Any]]:
    """Functions whose mapped fidelity still needs a human decision."""
    bad = [f for f in m["functions"] if f["fidelity"] in ("manual", "lossy", "approximate")]
    bad.sort(key=lambda f: ORDER.index(f["fidelity"]) if f["fidelity"] in ORDER else 99)
    return bad