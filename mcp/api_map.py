"""
api_map — pre-porting API/feature reference, derived from source.

Produces the document you want *before* writing code: every CUDA runtime call,
driver call, PTX instruction and device intrinsic the tree actually uses, what
replaces it on the target, whether that replacement is automatic or manual, and
which files depend on it.

Three things this encodes that a construct-only map misses:

1. **Status is per target, not global.** ``cudaGraphLaunch`` is automatic on a
   recent ROCm and absent on an old one. ``mma.sync`` is manual everywhere on
   AMD, but *how* differs by generation: MFMA is CDNA-only, WMMA covers RDNA3+,
   and RDNA2 has no matrix core at all.
2. **Some gaps are hard blockers, not partial gaps.** TMA tensor loads need a
   target feature that neither gfx1031 nor gfx1201 has. Saying "partial" there
   is how a plan stays alive that cannot ship.
3. **Provenance matters.** "Used in 3 files, 11 sites" is what turns a table
   into a work estimate.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- status vocabulary
STATUS = ("auto", "partial", "manual", "none")
STATUS_RANK = {s: i for i, s in enumerate(STATUS)}


def worst_status(values: list[str]) -> str:
    known = [v for v in values if v in STATUS_RANK]
    return max(known, key=STATUS_RANK.get) if known else "manual"


#: Finer architecture buckets than the porting-map families, because capability
#: changes *within* a vendor generation. Buckets are neutral strings; when
#: ARCH_MAP.json is available the bucket is derived from it rather than guessed
#: from the arch id, so a new arch only needs a data entry.
_ARCH_MAP_CACHE: dict[str, Any] = {}


def _arch_registry() -> dict[str, Any]:
    """Load ARCH_MAP.json once (the shared capability source of truth)."""
    if _ARCH_MAP_CACHE:
        return _ARCH_MAP_CACHE
    cands = [Path(__file__).resolve().parent.parent / "ARCH_MAP.json",
             Path("H:/K_and_S/ARCH_MAP.json")]
    for p in cands:
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                _ARCH_MAP_CACHE.update(data.get("entries", data))
                return _ARCH_MAP_CACHE
        except Exception:
            continue
    return {}


def arch_info(arch: str) -> dict[str, Any]:
    """Capability record for an arch id, or {} when unknown."""
    reg = _arch_registry()
    if arch in reg:
        return reg[arch]
    low = arch.lower()
    for k, v in reg.items():
        if k.lower() == low:
            return v
    return {}


def arch_bucket(arch: str) -> str:
    """Neutral bucket key for an arch.

    Derived from the registry when possible (vendor + generation), so this is
    not AMD-shaped: ``nvidia_sm80``, ``amd_rdna4``, ``intel_xe`` all resolve the
    same way. Falls back to prefix parsing only for unknown ids.
    """
    info = arch_info(arch)
    vendor = str(info.get("vendor", "")).lower()
    gen = str(info.get("generation") or info.get("family") or "").lower()
    if vendor and gen:
        return f"{vendor}_{gen}"

    a = arch.lower()
    if vendor:
        pass  # fall through to prefix rules with the registry vendor known
    # prefix fallback (unknown arch)
    if a.startswith("gfx103"):
        return "amd_rdna2"   # gfx1030/gfx1031/gfx1032/gfx1034 — RDNA2/RDNA 3.5
    if a.startswith("gfx11"):
        return "amd_rdna3"
    if a.startswith("gfx12"):
        return "amd_rdna4"
    if a.startswith("gfx9"):
        return "amd_cdna"   # cdna* — AMD data-center ISA
    if a.startswith("cdna2"):
        return "amd_cdna2"   # cdna2 — MI200 class
    if a.startswith("cdna3"):
        return "amd_cdna3"   # cdna3 — MI300 class
    if a.startswith("cdna"):
        return "amd_cdna"   # cdna* — AMD data-center ISA (non-gfx arch id)
    m = re.match(r"(?:sm|compute)_(\d+)", a)
    if m:
        return f"nvidia_sm{m.group(1)}"
    if a.startswith(("spir", "intel", "gen", "arc", "xe")):
        return "intel_xe"
    if vendor:
        return f"{vendor}_unknown"
    return "unknown"


def _capability_rule(api_id: str, arch: str) -> tuple[str, str, str] | None:
    """Resolve from capability flags when the table has no entry for this bucket.

    This is what keeps the map vendor-neutral: a target that lacks a capability
    gets the right answer without anyone enumerating its arch id in a table.
    """
    info = arch_info(arch)
    if not info:
        return None
    low = api_id.lower()
    has_matrix = bool(info.get("tensor_cores")) or bool(info.get("wmma")) \
        or bool(info.get("matrix_cores"))
    if "mma.sync" in low or "wmma" in low or "mfma" in low:
        if not has_matrix:
            return ("", "none", "target has no matrix-core capability (per ARCH_MAP)")
        return ("target matrix intrinsic (see ARCH_MAP capabilities)", "manual", "")
    if "cp.async" in low:
        if info.get("async_copy") is False:
            return ("", "none", "target has no async global->lds copy")
        return ("", "manual", "verify async copy availability on this target")
    if "dot4" in low or "dp4a" in low:
        if info.get("dot4") is False:
            return ("scalar expand, or dot2 if available", "manual",
                    "target has no 4-way int8 dot")
        return ("target dot4 intrinsic", "auto", "")
    if "fp8" in low or "e4m3" in low:
        if info.get("fp8_native") is False:
            return ("", "none", "target has no native FP8")
    if "bf16" in low:
        if info.get("bf16_native") is False:
            return ("__float2bfloat162_rn (storage only; math may emulate)", "partial", "")
    if "__shfl" in low:
        ws = info.get("wave_size")
        return ("__shfl_* (drop the mask)", "manual",
                f"wave size {ws}" if ws else "")
    return None


# --------------------------------------------------------------------------- runtime API table
# (source, hip replacement, base status, note)
RUNTIME_APIS: dict[str, tuple[str, str, str, str]] = {
    # ---- memory ----
    "cudaMalloc": ("hipMalloc", "auto", ""),
    "cudaFree": ("hipFree", "auto", ""),
    "cudaMallocAsync": ("hipMallocAsync", "auto", "mem-pool support varies by runtime"),
    "cudaFreeAsync": ("hipFreeAsync", "auto", ""),
    "cudaMallocHost": ("hipHostMalloc", "auto", ""),
    "cudaFreeHost": ("hipHostFree", "auto", ""),
    "cudaMallocManaged": ("hipMallocManaged", "auto", "HMM quality differs; verify"),
    "cudaMemcpy": ("hipMemcpy", "auto", ""),
    "cudaMemcpyAsync": ("hipMemcpyAsync", "auto", ""),
    "cudaMemcpyPeer": ("hipMemcpyPeer", "auto", ""),
    "cudaMemcpyPeerAsync": ("hipMemcpyPeerAsync", "auto", ""),
    "cudaMemset": ("hipMemset", "auto", ""),
    "cudaMemsetAsync": ("hipMemsetAsync", "auto", ""),
    "cudaMemGetInfo": ("hipMemGetInfo", "auto", ""),
    "cudaMemPoolCreate": ("hipMemPoolCreate", "auto", ""),
    "cudaMemPoolSetAccess": ("hipMemPoolSetAccess", "auto", ""),
    "cudaMemPoolPointerAttributes": ("hipMemPoolPointerAttributes", "auto", ""),
    # ---- device ----
    "cudaSetDevice": ("hipSetDevice", "auto", ""),
    "cudaGetDevice": ("hipGetDevice", "auto", ""),
    "cudaGetDeviceCount": ("hipGetDeviceCount", "auto", ""),
    "cudaGetDeviceProperties": ("hipGetDeviceProperties", "auto", "field names differ"),
    "cudaDeviceSynchronize": ("hipDeviceSynchronize", "auto", ""),
    "cudaDeviceSetLimit": ("hipDeviceSetLimit", "auto", ""),
    "cudaGetLastError": ("hipGetLastError", "auto", ""),
    "cudaPeekAtLastError": ("hipPeekAtLastError", "auto", ""),
    "cudaGetErrorName": ("hipGetErrorName", "partial", "not every code is covered"),
    "cudaGetErrorString": ("hipGetErrorString", "partial", ""),
    # ---- stream / event ----
    "cudaStreamCreate": ("hipStreamCreate", "auto", ""),
    "cudaStreamCreateWithFlags": ("hipStreamCreateWithFlags", "auto", ""),
    "cudaStreamDestroy": ("hipStreamDestroy", "auto", ""),
    "cudaStreamSynchronize": ("hipStreamSynchronize", "auto", ""),
    "cudaStreamWaitEvent": ("hipStreamWaitEvent", "auto", ""),
    "cudaEventCreate": ("hipEventCreate", "auto", ""),
    "cudaEventCreateWithFlags": ("hipEventCreateWithFlags", "auto", ""),
    "cudaEventRecord": ("hipEventRecord", "auto", ""),
    "cudaEventSynchronize": ("hipEventSynchronize", "auto", ""),
    "cudaEventQuery": ("hipEventQuery", "auto", ""),
    "cudaEventElapsedTime": ("hipEventElapsedTime", "auto", ""),
    "cudaEventDestroy": ("hipEventDestroy", "auto", ""),
    # ---- launch ----
    "cudaLaunchKernel": ("hipLaunchKernel", "auto", ""),
    "cudaFuncSetAttribute": ("hipFuncSetAttribute", "auto", "not all attributes exist"),
    "cudaFuncSetCacheConfig": ("hipFuncSetCacheConfig", "auto", "no effect on AMD"),
    # ---- graphs ----
    "cudaStreamBeginCapture": ("hipStreamBeginCapture", "auto", "ROCm 6.0+"),
    "cudaStreamEndCapture": ("hipStreamEndCapture", "auto", "ROCm 6.0+"),
    "cudaGraphInstantiate": ("hipGraphInstantiate", "auto", "ROCm 6.0+"),
    "cudaGraphLaunch": ("hipGraphLaunch", "auto", "ROCm 6.0+; verify on your runtime"),
    "cudaGraphExecUpdate": ("hipGraphExecUpdate", "auto", "ROCm 6.0+"),
    "cudaGraphDestroy": ("hipGraphDestroy", "auto", "ROCm 6.0+"),
    "cudaGraphExecDestroy": ("hipGraphExecDestroy", "auto", "ROCm 6.0+"),
    # ---- PDL / Blackwell-specific launch ----
    "cudaLaunchKernelEx": ("", "manual",
                          "PDL has no HIP equivalent; rebuild dependency with "
                          "hipEventRecord + hipStreamWaitEvent"),
    "cudaLaunchAttributeProgrammaticStreamSerialization": ("", "none",
                          "PDL attribute; no equivalent"),
    "cudaTriggerProgrammaticLaunchCompletion": ("", "none",
                          "PDL trigger; no equivalent"),
    "cudaGridDependencySynchronize": ("", "none",
                          "PDL grid sync; use stream synchronisation instead"),
    # ---- driver API ----
    "cuTensorMap": ("", "none",
                    "TMA descriptor; AMD tensor-load intrinsics need a target "
                    "feature absent on gfx1031/gfx1201"),
    "cuTensorMapEncodeTiled": ("", "none", "no AMD descriptor encoding; rewrite the kernel"),
    "cuGetErrorName": ("hipGetErrorName", "auto", ""),
    "cuLaunchKernel": ("hipModuleLaunchKernel", "auto",
                       "same argument shape: (f, gx,gy,gz, bx,by,bz, shared, stream, params, extra)"),
    "cuModuleLoad": ("hipModuleLoad", "auto", ""),
    "cuModuleGetFunction": ("hipModuleGetFunction", "auto", ""),
}

#: Per-architecture overrides where the base status is wrong for that chip.
API_ARCH_OVERRIDE: dict[str, dict[str, tuple[str, str]]] = {
    "cudaMallocAsync": {"amd_rdna2": ("partial", "async allocator support is limited")},
    "cudaGraphLaunch": {"amd_rdna2": ("manual", "graph support may be absent; gate the graph path")},
    "cudaMemPoolCreate": {"amd_rdna2": ("partial", "")},
}

# --------------------------------------------------------------------------- PTX table
# category -> {ptx: {bucket: (replacement, status, note)}}
PTX_MAP: dict[str, dict[str, Any]] = {
    "matrix": {
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32": {
            "amd_cdna": ("__builtin_amdgcn_mfma_f32_16x16x8bf16", "manual", "shape 16x16x8 differs from 16x8x16"),
            "amd_rdna4": ("__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12", "manual", "register layout differs"),
            "amd_rdna3": ("__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32", "manual", ""),
            "amd_rdna2": ("", "none", "no matrix core: use scalar FMA or dot fallbacks"),
            "nvidia": ("native", "auto", ""),
        },
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32": {
            "amd_cdna": ("__builtin_amdgcn_mfma_f32_16x16x16f16", "manual", ""),
            "amd_rdna4": ("__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12", "manual", ""),
            "amd_rdna3": ("__builtin_amdgcn_wmma_f32_16x16x16_f16_w32", "manual", ""),
            "amd_rdna2": ("", "none", "no matrix core"),
            "nvidia": ("native", "auto", ""),
        },
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32": {
            "amd_cdna": ("__builtin_amdgcn_mfma_i32_16x16x32_i8", "manual", ""),
            "amd_rdna4": ("__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12", "manual",
                          "IU8 is unsigned; sign-adjust operands"),
            "amd_rdna3": ("__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32", "manual", ""),
            "amd_rdna2": ("__builtin_amdgcn_sdot4 / sdot8", "manual", "no matrix core; scalar dot path"),
            "nvidia": ("native", "auto", ""),
        },
        "mma.sync.aligned.kind::f8f6f4": {
            "amd_cdna": ("__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8", "manual", "CDNA3+"),
            "amd_rdna4": ("__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12", "manual",
                          "verify the intrinsic name against your compiler's target features"),
            "amd_rdna3": ("", "none", "no FP8"),
            "amd_rdna2": ("", "none", "no FP8"),
            "nvidia": ("native", "auto", ""),
        },
        "mma.sync.aligned.kind::mxf4nvf4": {
            "amd_cdna": ("", "none", "FP4 MMA is CDNA4 (gfx1250) only"),
            "amd_rdna4": ("", "none", "no FP4 matrix path; nibble-extract + dot is ~4x slower"),
            "amd_rdna3": ("", "none", ""),
            "amd_rdna2": ("", "none", ""),
            "nvidia": ("native", "auto", ""),
        },
        "mma.sync.aligned.m8n8k4.row.col.s32.s4.s4.s32": {
            "amd_cdna": ("__builtin_amdgcn_sdot8", "manual", ""),
            "amd_rdna4": ("__builtin_amdgcn_sdot8 / udot8", "manual", "8 nibbles per instruction"),
            "amd_rdna3": ("__builtin_amdgcn_sdot8", "manual", ""),
            "amd_rdna2": ("__builtin_amdgcn_sdot8", "manual", ""),
            "nvidia": ("native", "auto", ""),
        },
    },
    "async-copy": {
        "cp.async.ca.shared.global": {
            "amd_cdna": ("__builtin_amdgcn_global_load_async_to_lds_b128", "manual",
                         "different barrier model; needs s_wait_asynccnt"),
            "amd_rdna4": ("__builtin_amdgcn_global_load_async_to_lds_b128", "manual",
                          "verify availability on your target"),
            "amd_rdna3": ("__builtin_amdgcn_global_load_async_to_lds_b128", "manual", ""),
            "amd_rdna2": ("", "none", "async LDS load not available; use sync copy"),
            "nvidia": ("native", "auto", ""),
        },
        "cp.async.commit_group": {
            "amd_rdna4": ("implicit in the async-load intrinsic", "auto", ""),
            "amd_rdna3": ("implicit", "auto", ""),
            "amd_rdna2": ("", "none", ""),
            "nvidia": ("native", "auto", ""),
        },
        "cp.async.wait_group": {
            "amd_rdna4": ("__builtin_amdgcn_s_wait_asynccnt", "manual", ""),
            "amd_rdna3": ("__builtin_amdgcn_s_wait_asynccnt", "manual", ""),
            "amd_rdna2": ("", "none", ""),
            "nvidia": ("native", "auto", ""),
        },
        "cp.async.bulk.tensor": {
            "amd_rdna4": ("", "none", "TMA needs a target feature absent on gfx1201"),
            "amd_rdna3": ("", "none", ""),
            "amd_rdna2": ("", "none", ""),
            "amd_cdna": ("", "none", "no descriptor-based TMA; rewrite as tiled async loads"),
            "nvidia": ("native", "auto", ""),
        },
    },
    "matrix-load": {
        "ldmatrix.sync.aligned.m8n8": {
            "amd_cdna": ("ds_read_tr_b64 / ds_read_tr_b128", "manual", "different tile shapes"),
            "amd_rdna4": ("ds_read_tr_b64 / ds_read_tr_b128", "manual", "verify on target"),
            "amd_rdna3": ("ds_read_tr_b64", "manual", ""),
            "amd_rdna2": ("", "manual", "use plain LDS loads with manual transpose"),
            "nvidia": ("native", "auto", ""),
        },
    },
    "barrier": {
        "mbarrier.init": {
            "amd_rdna4": ("__builtin_amdgcn_s_barrier_init", "manual", "verify availability"),
            "amd_rdna3": ("__builtin_amdgcn_s_barrier_init", "manual", ""),
            "amd_rdna2": ("", "none", ""),
            "amd_cdna": ("__builtin_amdgcn_s_barrier_init", "manual", ""),
            "nvidia": ("native", "auto", ""),
        },
        "mbarrier.try_wait": {
            "amd_rdna4": ("__builtin_amdgcn_s_barrier_wait", "manual", "blocking only, no try_wait"),
            "amd_rdna3": ("__builtin_amdgcn_s_barrier_wait", "manual", ""),
            "amd_rdna2": ("", "none", ""),
            "amd_cdna": ("__builtin_amdgcn_s_barrier_wait", "manual", ""),
            "nvidia": ("native", "auto", ""),
        },
        "mbarrier.arrive.expect_tx": {
            "amd_rdna4": ("", "none", "transaction-count expectation is TMA-specific"),
            "amd_rdna3": ("", "none", ""),
            "amd_rdna2": ("", "none", ""),
            "amd_cdna": ("", "none", ""),
            "nvidia": ("native", "auto", ""),
        },
        "bar.sync": {
            "amd_rdna4": ("__syncthreads() / __builtin_amdgcn_s_barrier_wait", "manual", ""),
            "amd_rdna3": ("__syncthreads()", "manual", ""),
            "amd_rdna2": ("__syncthreads()", "manual", ""),
            "amd_cdna": ("__syncthreads()", "manual", ""),
            "nvidia": ("native", "auto", ""),
        },
        "setmaxnreg": {
            "amd_rdna4": ("", "none", "AMD does not expose register-count control"),
            "amd_rdna3": ("", "none", ""),
            "amd_rdna2": ("", "none", ""),
            "amd_cdna": ("", "none", ""),
            "nvidia": ("native", "auto", ""),
        },
        "fence.mbarrier_init.release.cluster": {
            "amd_rdna4": ("", "none", "cluster-scoped fence; use s_waitcnt"),
            "amd_rdna3": ("", "none", ""),
            "amd_rdna2": ("", "none", ""),
            "amd_cdna": ("", "none", ""),
            "nvidia": ("native", "auto", ""),
        },
    },
    "math": {
        "ex2.approx.f32": {
            "amd_rdna4": ("__builtin_amdgcn_exp2f / exp2f", "auto", ""),
            "amd_rdna3": ("__builtin_amdgcn_exp2f", "auto", ""),
            "amd_rdna2": ("__builtin_amdgcn_exp2f", "auto", ""),
            "amd_cdna": ("__builtin_amdgcn_exp2f", "auto", ""),
            "nvidia": ("native", "auto", ""),
        },
        "cvt.rn.bf16x2.f32": {
            "amd_rdna4": ("__float2bfloat162_rn", "auto", ""),
            "amd_rdna3": ("__float2bfloat162_rn", "auto", ""),
            "amd_rdna2": ("__float2bfloat162_rn", "auto", "bf16 storage works; native math may not"),
            "amd_cdna": ("__float2bfloat162_rn", "auto", ""),
            "nvidia": ("native", "auto", ""),
        },
    },
}

# --------------------------------------------------------------------------- intrinsics
INTRINSICS_MAP: dict[str, tuple[str, str, str]] = {
    # (replacement, status, note)
    "__shfl_sync": ("__shfl(var, lane)", "manual", "HIP takes no mask; drop the mask argument"),
    "__shfl_up_sync": ("__shfl_up(var, delta)", "manual", "drop the mask argument"),
    "__shfl_down_sync": ("__shfl_down(var, delta)", "manual", "drop the mask argument"),
    "__shfl_xor_sync": ("__shfl_xor(var, lane_mask)", "manual", "drop the mask argument"),
    "__ballot_sync": ("__ballot(pred)", "manual", "drop the mask; 64-bit mask required on HIP *_sync"),
    "__any_sync": ("__any(pred)", "manual", ""),
    "__all_sync": ("__all(pred)", "manual", ""),
    "__activemask": ("0xFFFFFFFFu", "manual", "all lanes active by construction"),
    "__syncwarp": ("__syncwarp()", "auto", ""),
    "__syncthreads": ("__syncthreads()", "auto", ""),
    "__syncthreads_count": ("__syncthreads_count()", "auto", ""),
    "__ldg": ("__ldg()", "auto", ""),
    "__cvta_generic_to_shared": ("__cvta_generic_to_shared()", "auto", ""),
    "__threadfence": ("__threadfence()", "auto", ""),
    "__float2half_rn": ("__float2half_rn()", "auto", ""),
    "__half2float": ("__half2float()", "auto", ""),
    "__floats2half2_rn": ("__floats2half2_rn()", "auto", ""),
    "__float2bfloat16_rn": ("__float2bfloat16_rn()", "auto", ""),
    "__bfloat162float": ("__bfloat162float()", "auto", ""),
    "__floats2bfloat162_rn": ("__floats2bfloat162_rn()", "auto", ""),
    "__float_as_uint": ("__float_as_uint()", "auto", ""),
    "__uint_as_float": ("__uint_as_float()", "auto", ""),
    "atomicAdd": ("atomicAdd()", "auto", ""),
    "atomicCAS": ("atomicCAS()", "auto", ""),
}

# --------------------------------------------------------------------------- feature blockers
FEATURES: dict[str, dict[str, Any]] = {
    "TMA (Tensor Memory Accelerator)": {
        "pattern": r"cp\.async\.bulk\.tensor|cuTensorMap|CUtensorMap|__grid_constant__",
        "amd_rdna4": ("none", "needs gfx1250-insts; absent on gfx1201"),
        "amd_rdna3": ("none", ""),
        "amd_rdna2": ("none", ""),
        "amd_cdna": ("none", "descriptor TMA does not exist; rewrite as tiled async loads"),
        "nvidia": ("auto", ""),
        "blocker": True,
    },
    "PDL (Programmatic Dependent Launch)": {
        "pattern": r"cudaLaunchKernelEx|cudaTriggerProgrammaticLaunchCompletion|"
                   r"cudaGridDependencySynchronize|ProgrammaticStreamSerialization",
        "amd_rdna4": ("none", "no equivalent; rebuild with events"),
        "nvidia": ("auto", ""),
        "blocker": True,
    },
    "Register-count control (setmaxnreg)": {
        "pattern": r"\bsetmaxnreg\b",
        "amd_rdna4": ("none", "AMD does not expose register-count control; restructure warps"),
        "amd_rdna3": ("none", ""),
        "amd_rdna2": ("none", ""),
        "amd_cdna": ("none", ""),
        "nvidia": ("auto", ""),
        "blocker": True,
    },
    "Named barriers (bar.sync N)": {
        "pattern": r"\bbar\.sync\s+\d",
        "amd_rdna4": ("manual", "portable via __syncthreads or s_barrier; verify the named-barrier intrinsic"),
        "amd_rdna3": ("manual", ""),
        "amd_rdna2": ("manual", ""),
        "amd_cdna": ("manual", ""),
        "nvidia": ("auto", ""),
        "blocker": False,
    },
    "FP4 / NVFP4 matmul": {
        "pattern": r"mxf4nvf4|e2m1|nvfp4",
        "amd_rdna4": ("none", "no FP4 matrix path on RDNA"),
        "amd_cdna": ("partial", "CDNA4 gfx1250 only"),
        "nvidia": ("auto", ""),
        "blocker": True,
    },
    "FP8 matmul": {
        "pattern": r"f8f6f4|e4m3|fp8",
        "amd_rdna4": ("manual", "verify the WMMA FP8 intrinsic against your compiler"),
        "amd_rdna3": ("none", ""),
        "amd_rdna2": ("none", ""),
        "amd_cdna": ("manual", "CDNA3+"),
        "nvidia": ("auto", ""),
        "blocker": False,
    },
    "Thread-block clusters": {
        "pattern": r"cluster_dims|__cluster_dims__|cluster\.sync|mapa\.",
        "amd_rdna4": ("none", "no cluster launch"),
        "nvidia": ("auto", ""),
        "blocker": True,
    },
}

# --------------------------------------------------------------------------- scanning
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)

#: Qualifiers and preprocessor tokens that look like intrinsics but are not
#: porting targets. Without this list every kernel reports a wall of noise and
#: the real unverified calls drown.
_NOT_AN_INTRINSIC = {
    "__global__", "__device__", "__host__", "__shared__", "__constant__",
    "__managed__", "__restrict__", "__restrict", "__forceinline__", "__inline__",
    "__noinline__", "__launch_bounds__", "__align__", "__grid_constant__",
    "__builtin_expect", "__cplusplus", "__cuda_arch__", "__hip__",
    "__hip_device_compile__", "__has_include", "__attribute__", "__declspec",
    "__pragma", "__file__", "__line__", "__func__", "__va_args__", "__counter__",
    "__date__", "__time__", "__stdc__", "__clang__", "__gnuc__", "__alignof__",
}

_KNOWN_IDS = (set(RUNTIME_APIS) | set(INTRINSICS_MAP)
              | {k for _group in PTX_MAP.values() for k in _group})

#: Longest key first, with non-identifier boundaries, so `cudaMalloc` no longer
#: matches inside `cudaMallocAsync` and every call is counted once, as itself.
_KNOWN_SYMBOL_RE = re.compile(
    r"(?<![A-Za-z0-9_])("
    + "|".join(re.escape(s) for s in sorted(_KNOWN_IDS, key=len, reverse=True))
    + r")(?![A-Za-z0-9_])"
)

#: Calls that are used but absent from the tables. These become `unverified`
#: rows: the point is that an unknown instruction must be visible, not silent.
_API_CALL_RE = re.compile(r"(?<![A-Za-z0-9_])((?:cuda|cu|hip)[A-Z][A-Za-z0-9_]*)\s*\(")
_INTRINSIC_CALL_RE = re.compile(r"(?<![A-Za-z0-9_])(__[A-Za-z][A-Za-z0-9_]*)\s*\(")
_ASM_OPEN_RE = re.compile(r"\basm\s*(?:volatile\s*)?\(")
_ASM_LITERAL_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_PTX_MNEMONIC_RE = re.compile(r"^\s*([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)")


def _asm_templates(text: str):
    """Yield (body, offset) for each ``asm(...)``, string- and paren-aware.

    A semicolon inside the template is ordinary PTX, not the end of the
    statement, and an operand may itself contain parentheses (``"=r"(x)``).
    A flat regex stops at the first of either and silently loses the block,
    which is how an uninventoried instruction stays invisible.
    """
    for match in _ASM_OPEN_RE.finditer(text):
        i, start = match.end(), match.end()
        depth, quote = 1, ""
        while i < len(text) and depth:
            ch = text[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = ""
            elif ch in "\"'":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        if depth == 0:
            yield text[start:i - 1], start



def _strip(src: str) -> str:
    return _COMMENT_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), src)


def _unknown_ptx(text: str, newlines: list[int]) -> list[tuple[str, int]]:
    """Inline-asm mnemonics with no entry in PTX_MAP.

    Multi-line asm templates are joined first, because a shape string split over
    two C literals is still one instruction the port depends on.
    """
    out: list[tuple[str, int]] = []
    for template, offset in _asm_templates(text):
        template = template.split(":", 1)[0]
        body = " ".join(_ASM_LITERAL_RE.findall(template)).strip()
        if not body:
            continue
        m = _PTX_MNEMONIC_RE.match(body)
        if not m:
            continue
        mnemonic = m.group(1)
        if any(mnemonic == k or mnemonic.startswith(k) or k.startswith(mnemonic)
               for k in _KNOWN_IDS):
            continue
        out.append((mnemonic, bisect_right(newlines, offset) + 1))
    return out


def scan_usage(files: list[str], unreadable: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """api/intrinsic/ptx/feature id -> [{file, line}] usage sites.

    Known symbols are matched by token, so prefix families do not double-count.
    Symbols used but missing from the tables are reported as ``api:``,
    ``intrinsic:`` and ``ptx:`` ids, which ``resolve`` marks ``unverified``.
    """
    hits: dict[str, list[dict[str, Any]]] = {}

    def add(key: str, path: str, line: int) -> None:
        sites = hits.setdefault(key, [])
        if not sites or sites[-1]["line"] != line or sites[-1]["file"] != path:
            sites.append({"file": path, "line": line})

    for f in files:
        p = Path(f)
        if not p.is_file():
            if unreadable is not None:
                unreadable.append(str(p))
            continue
        try:
            text = _strip(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            if unreadable is not None:
                unreadable.append(str(p))
            continue
        newlines = [i for i, ch in enumerate(text) if ch == "\n"]

        def line_of(pos: int) -> int:
            return bisect_right(newlines, pos) + 1

        for m in _KNOWN_SYMBOL_RE.finditer(text):
            add(m.group(1), str(p), line_of(m.start()))

        for m in _API_CALL_RE.finditer(text):
            tok = m.group(1)
            if tok not in _KNOWN_IDS:
                add(f"api:{tok}", str(p), line_of(m.start()))

        for m in _INTRINSIC_CALL_RE.finditer(text):
            tok = m.group(1)
            if tok.lower() not in _NOT_AN_INTRINSIC and tok not in _KNOWN_IDS:
                add(f"intrinsic:{tok}", str(p), line_of(m.start()))

        for feat, spec in FEATURES.items():
            for m in re.finditer(spec["pattern"], text, re.I):
                add(feat, str(p), line_of(m.start()))

        for mnemonic, line in _unknown_ptx(text, newlines):
            add(f"ptx:{mnemonic}", str(p), line)

    return hits


# --------------------------------------------------------------------------- resolution
def _norm(t: Any) -> tuple[str, str, str]:
    """Normalise a mapping entry to (replacement, status, note)."""
    if isinstance(t, str):
        return t, "manual", ""
    t = tuple(t)
    repl = t[0] if len(t) > 0 else ""
    status = t[1] if len(t) > 1 else "manual"
    note = t[2] if len(t) > 2 else ""
    return repl, status, note


def _vendor(arch: str) -> str:
    return str(arch_info(arch).get("vendor", "")).lower()


def _bucket_keys(arch: str) -> list[str]:
    """Buckets to try, most specific first, without leaking NVIDIA fallbacks.

    Falling back to the ``nvidia`` entry for a non-NVIDIA target is how a
    blocker used to disappear: TMA on Intel resolved as ``auto/native`` only
    because the vendor bucket was missing from the table.
    """
    keys = [arch_bucket(arch)]
    if _vendor(arch) == "nvidia":
        keys.append("nvidia")
    return keys


def _pick(entry: dict[str, Any], arch: str) -> tuple[tuple[str, str, str] | None, bool]:
    """Mapping entry for the target, and whether a target-specific key was found."""
    for key in _bucket_keys(arch):
        if key in entry:
            return _norm(entry[key]), True
    return None, False


def resolve(api_id: str, target_arch: str) -> dict[str, Any]:
    """Replacement + status for one api/intrinsic/ptx/feature on a target."""
    bucket = arch_bucket(target_arch)
    out: dict[str, Any] = {"id": api_id, "bucket": bucket, "target": target_arch,
                           "replacement": "", "status": "manual", "note": "", "category": ""}

    if api_id.startswith(("api:", "intrinsic:", "ptx:", "feature:")):
        kind, _, sym = api_id.partition(":")
        out.update(replacement="—", status="manual", category="unverified",
                   note=(f"`{sym}` ({kind}) is used in the source but absent from the "
                         "reference table; probe it on the target before relying on it"))
        return out

    if api_id in RUNTIME_APIS:
        repl, status, note = _norm(RUNTIME_APIS[api_id])
        out.update(replacement=repl or "—", status=status, note=note,
                   category="runtime")
        ov = API_ARCH_OVERRIDE.get(api_id, {}).get(bucket)
        if ov:
            o = _norm(ov)
            out["status"] = o[1]
            out["note"] = o[2] or note
        return out

    if api_id in INTRINSICS_MAP:
        repl, status, note = _norm(INTRINSICS_MAP[api_id])
        out.update(replacement=repl, status=status, note=note, category="intrinsic")
        return out

    for cat, group in PTX_MAP.items():
        if api_id in group:
            pick, found = _pick(group[api_id], target_arch)
            if not found:
                out.update(replacement="—", status="manual", category=cat,
                           note=(f"no entry for `{bucket}`; verify the replacement on "
                                 "the target, do not assume native"))
                return out
            out.update(replacement=pick[0] or "—", status=pick[1],
                       note=pick[2], category=cat)
            return out

    if api_id in FEATURES:
        spec = FEATURES[api_id]
        raw = None
        for key in _bucket_keys(target_arch):
            if key in spec:
                raw = spec[key]
                break
        if raw is None:
            status, note = ("manual",
                            f"no entry for `{bucket}`; verify on the target")
        elif isinstance(raw, str):
            status, note = raw, ""
        else:
            raw = tuple(raw)
            status = raw[0] if raw else "manual"
            note = raw[1] if len(raw) > 1 else ""
        out.update(replacement="—", status=status, note=note,
                   category="feature", blocker=spec.get("blocker", False))
        return out

    out.update(replacement="—", status="manual", note="not in table; probe it")
    rule = _capability_rule(api_id, target_arch)
    if rule:
        out.update(replacement=rule[0] or "—", status=rule[1],
                   note=rule[2], category=out["category"] or "capability")
    return out


def build_reference(files: list[str], source_arch: str, target_arch: str) -> dict[str, Any]:
    """Assemble the pre-porting reference: inventory, tables, usage, blockers."""
    unreadable: list[str] = []
    hits = scan_usage(files, unreadable)
    rows = []
    for api_id, sites in hits.items():
        r = resolve(api_id, target_arch)
        r["sites"] = sites
        r["count"] = len(sites)
        r["files"] = sorted({Path(s["file"]).name for s in sites})
        rows.append(r)
    rows.sort(key=lambda r: (STATUS_RANK.get(r["status"], 9), -r["count"]))

    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    blockers = [r for r in rows if r.get("blocker") and r["status"] == "none"]
    unverified = [r for r in rows if r["category"] == "unverified"]

    inv: dict[str, int] = {}
    for f in files:
        ext = Path(f).suffix.lower()
        inv[ext] = inv.get(ext, 0) + 1

    return {
        "source_arch": source_arch,
        "target_arch": target_arch,
        "bucket": arch_bucket(target_arch),
        "file_inventory": dict(sorted(inv.items(), key=lambda kv: -kv[1])),
        "file_count": len(files),
        "unreadable_files": unreadable,
        "symbols_found": len(rows),
        "by_status": by_status,
        "unverified_count": len(unverified),
        "blockers": [{"id": b["id"], "note": b["note"], "files": b["files"]} for b in blockers],
        "rows": rows,
    }


# --------------------------------------------------------------------------- rendering
_STATUS_MARK = {"auto": "Auto", "partial": "**Partial**", "manual": "**Manual**", "none": "**No equivalent**"}


def render_reference(ref: dict[str, Any], max_rows_per_section: int = 60) -> str:
    L = [f"# CUDA backend function reference — `{ref['source_arch']}` → `{ref['target_arch']}`",
         f"Target bucket: `{ref['bucket']}`\n",
         f"{ref['file_count']} source file(s) scanned; "
         f"{ref['symbols_found']} distinct API/intrinsic/PTX/feature symbol(s) used.\n"]

    inv = ref["file_inventory"]
    if inv:
        L.append("## Source file inventory\n")
        L.append("| extension | files |")
        L.append("|---|---|")
        for ext, n in inv.items():
            L.append(f"| `{ext or '(none)'}` | {n} |")
        L.append("")

    bad = ref.get("unreadable_files") or []
    if bad:
        L.append("## Files that could not be read\n")
        for f in bad[:20]:
            L.append(f"- `{f}`")
        if len(bad) > 20:
            L.append(f"- ... and {len(bad) - 20} more")
        L.append("\nThe inventory below excludes these. A path typo produces a "
                 "clean-looking report, so fix them before trusting the counts.\n")

    if ref["file_count"] and not ref["symbols_found"] and not bad:
        L.append("**No known symbol was found in these files.** That is a statement "
                 "about the scanner, not about the code: check the paths and the "
                 "extensions before treating the tree as empty.\n")

    L.append("## Status summary\n")
    L.append("| status | symbols |")
    L.append("|---|---|")
    for s in STATUS:
        if s in ref["by_status"]:
            L.append(f"| {_STATUS_MARK[s]} | {ref['by_status'][s]} |")
    if ref.get("unverified_count"):
        L.append(f"| Unverified (not in table) | {ref['unverified_count']} |")
    L.append("")

    if ref["blockers"]:
        L.append("## Hard blockers — must be resolved before porting\n")
        L.append("| feature | why it blocks | files |")
        L.append("|---|---|---|")
        for b in ref["blockers"]:
            L.append(f"| **{b['id']}** | {b['note'] or '—'} | "
                     f"{', '.join(Path(f).name for f in b['files'][:4])} |")
        L.append("\nA blocker is not a partial gap: the feature has no equivalent on "
                 "this target and the kernel must be restructured or dropped.\n")

    SECTIONS = [("unverified", "Used in source — not in the reference table"),
                ("runtime", "CUDA runtime / driver API"),
                ("intrinsic", "CUDA device intrinsics"),
                ("matrix", "PTX — matrix multiply"),
                ("async-copy", "PTX — async copy"),
                ("matrix-load", "PTX — matrix load"),
                ("barrier", "PTX — barriers and register control"),
                ("math", "PTX — math"),
                ("feature", "Features")]
    for cat, title in SECTIONS:
        sub = [r for r in ref["rows"] if r["category"] == cat]
        if not sub:
            continue
        L.append(f"## {title}\n")
        L.append("| source | replacement | status | uses | files | notes |")
        L.append("|---|---|---|---|---|---|")
        for r in sub[:max_rows_per_section]:
            files_s = ", ".join(f"`{f}`" for f in r["files"][:3])
            if len(r["files"]) > 3:
                files_s += f" +{len(r['files']) - 3}"
            note = (r["note"] or "").replace("|", "/")
            L.append(f"| `{r['id']}` | `{r['replacement']}` | {_STATUS_MARK.get(r['status'], r['status'])} "
                     f"| {r['count']} | {files_s} | {note} |")
        if len(sub) > max_rows_per_section:
            L.append(f"\n... and {len(sub) - max_rows_per_section} more")
        L.append("")

    L.append("## Reading this table\n")
    L.append("- **Auto** — the compiler maps it; no source change.")
    L.append("- **Partial** — maps, but behaviour or coverage differs; verify.")
    L.append("- **Manual** — a replacement exists but the source must be edited.")
    L.append("- **No equivalent** — restructure the kernel; do not stub it.")
    L.append("- **Unverified** — used in the source but absent from the table. It is "
             "listed so it cannot pass as clean code; probe it before trusting it.")
    L.append("\nSame symbol can differ by architecture: `mma.sync` is manual on every "
             "AMD target, but *which* intrinsic replaces it changes at RDNA2/RDNA3/RDNA4 "
             "and again on CDNA.")
    return "\n".join(L)


def lookup_api(api_id: str, target_arch: str) -> dict[str, Any]:
    r = resolve(api_id, target_arch)
    r["known"] = api_id in RUNTIME_APIS or api_id in INTRINSICS_MAP or api_id in FEATURES or any(
        api_id in g for g in PTX_MAP.values())
    return r
