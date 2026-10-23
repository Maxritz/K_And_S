"""
isa_tools — vendor-agnostic instruction probing and build verification for K and S.

Why this module exists
----------------------
A kernel port was once designed around an instruction (`v_dot4_i32_i8`) that the
target architecture cannot encode. Nothing caught it: the compiler broke inside
the LLVM backend rather than at the assembler, and every reference document
suggested the instruction existed. The only reliable oracle is compiling the
instruction for the real target and reading the result.

Two rules this module enforces:

1. **One candidate, one translation unit.** Probing several candidates in one
   file means the first error masks the rest.
2. **Delete the artifact before building.** A stale binary is indistinguishable
   from a successful build to anything that only checks for the file's presence.

Nothing here assumes a vendor. Backends are described in a table; unknown
backends report what they need instead of guessing.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- toolchains

#: How to build and disassemble one probe for each backend.
#: `arch_flag` uses the literal token `{target}`.
TOOLCHAINS: dict[str, dict[str, Any]] = {
    "hip": {
        "compilers": ("hipcc.exe", "hipcc.bat", "hipcc"),
        "roots": ("ROCM_PATH", "ROCM_HOME", "HIP_PATH"),
        "default_roots": (r"C:\Program Files\AMD\ROCm", "/opt/rocm"),
        "arch_flag": "--offload-arch={target}",
        "language": ("-x", "hip"),
        "device_only": ("--cuda-device-only",),
        "include": "#include <hip/hip_runtime.h>\n#include <hip/amd_detail/amd_math_functions.h>\n",
        "objdump": ("llvm-objdump.exe", "llvm-objdump"),
        "extension": ".hip",
    },
    "cuda": {
        "compilers": ("nvcc.exe", "nvcc"),
        "roots": ("CUDA_PATH", "CUDA_HOME"),
        "default_roots": (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                          "/usr/local/cuda"),
        "arch_flag": "-arch={target}",
        "language": ("-x", "cu"),
        "device_only": ("--cubin",),
        "include": "#include <cuda_runtime.h>\n",
        "objdump": ("llvm-objdump.exe", "llvm-objdump"),
        "extension": ".cu",
    },
    "sycl": {
        "compilers": ("icpx.exe", "icpx", "dpcpp.exe", "dpcpp", "clang++.exe", "clang++"),
        "roots": ("ONEAPI_ROOT", "SYCL_HOME"),
        "default_roots": (r"C:\Program Files (x86)\Intel\oneAPI", "/opt/intel/oneapi"),
        "arch_flag": "-Xs",
        "language": ("-x", "c++"),
        "device_only": ("-fsycl", "-fsycl-device-only"),
        "include": "#include <sycl/sycl.hpp>\n",
        "objdump": ("llvm-objdump.exe", "llvm-objdump"),
        "extension": ".cpp",
    },
}

#: Mnemonics that exist in one vendor's ISA and must never reach another's
#: assembler. Callers may override; these are defaults, not truth.
FOREIGN_MNEMONICS: dict[str, tuple[str, ...]] = {
    # NVIDIA-only spellings that have no AMD encoding
    "hip": ("mma.sync.", "cp.async", "bar.sync", "ld.global", "st.global",
            "cvta.to.", "shfl.sync", "vote.sync", "cvt.rn.f16", "cvt.f32.f16",
            "dp4a.s32", "v_dot4", "v_dot8", "v_mac_f16", "v_mad_f16"),
    # AMD-only spellings that have no NVIDIA encoding
    "cuda": ("v_dot4", "v_dot8", "v_mac_f16", "v_mad_f16", "s_waitcnt",
             "ds_read", "ds_write", "buffer_load", "s_memtime", "v_cvt_pk_f16_f32"),
    "sycl": ("mma.sync.", "cp.async", "v_dot4", "v_dot8", "s_waitcnt"),
}

_ASM_STRING_RE = r'asm\s*(?:volatile\s*)?\(\s*"'


# --------------------------------------------------------------------------- discovery

def _env() -> dict[str, str]:
    """Process env with the host C++ toolchain prepended on Windows."""
    env = dict(os.environ)
    if os.name == "nt":
        pats = sorted(glob.glob(
            r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64"))
        if pats:
            env["PATH"] = pats[-1] + ";" + env.get("PATH", "")
    return env


def find_root(backend: str) -> Path | None:
    """Installation root for a backend, via env var then conventional paths."""
    spec = TOOLCHAINS.get(backend)
    if not spec:
        return None
    for var in spec["roots"]:
        val = os.environ.get(var)
        if val and Path(val).exists():
            return Path(val)
    for cand in spec["default_roots"]:
        p = Path(cand)
        if p.exists():
            return p
    exe = find_compiler(backend)
    return exe.resolve().parent.parent if exe else None


def find_compiler(backend: str) -> Path | None:
    """First available compiler for a backend."""
    spec = TOOLCHAINS.get(backend)
    if not spec:
        return None
    root = None
    for var in spec["roots"]:
        val = os.environ.get(var)
        if val and Path(val).exists():
            root = Path(val)
            break
    if root is None:
        for cand in spec["default_roots"]:
            if Path(cand).exists():
                root = Path(cand)
                break
    search_dirs = []
    if root:
        search_dirs += [root / "bin", root / "compiler" / "latest" / "bin"]
    for name in spec["compilers"]:
        for d in search_dirs:
            p = d / name
            if p.exists():
                return p
        h = shutil.which(name)
        if h:
            return Path(h)
    return None


def find_llvm_tool(*names: str) -> Path | None:
    """Locate an LLVM binary (objdump/nm/readelf) next to ROCm or on PATH."""
    root = find_root("hip") or find_root("cuda")
    for name in names:
        cands: list[Path] = []
        if root:
            cands += [root / "lib" / "llvm" / "bin" / name,
                      root / "llvm" / "bin" / name,
                      root / "bin" / name]
        h = shutil.which(name)
        if h:
            cands.append(Path(h))
        for c in cands:
            if c.exists():
                return c
    return None


# --------------------------------------------------------------------------- disassembly

_ELF_MAGIC = b"\x7fELF"


def unwrap_code_object(path: str | Path) -> Path | None:
    """Return a raw ELF code object, unbundling a clang offload bundle if needed.

    hipcc/nvcc write a *bundle* (fatbinary container), not a bare ELF, so
    objdump and nm reject it with "not recognized as a valid object file".
    Scan for the ELF magic and slice from there.
    """
    p = Path(path)
    try:
        data = p.read_bytes()
    except Exception:
        return None
    if data[:4] == _ELF_MAGIC:
        return p
    idx = data.find(_ELF_MAGIC)
    if idx < 0:
        return None
    out = p.with_name(p.name + ".elf")
    try:
        out.write_bytes(data[idx:])
    except Exception:
        return None
    return out


def disassemble_mnemonics(obj: Path) -> list[str]:
    """Instruction mnemonics in a compiled object/cubin/HSACO.

    Returns [] when no disassembler is available — the compile result is still
    authoritative, so this is enrichment, not the verdict.
    """
    tool = find_llvm_tool("llvm-objdump.exe", "llvm-objdump")
    if not tool:
        return []
    target = unwrap_code_object(obj)
    if target is None:
        return []
    try:
        r = subprocess.run([str(tool), "-d", "--no-show-raw-insn", str(target)],
                           capture_output=True, text=True, timeout=300, env=_env())
    except Exception:
        return []
    text = r.stdout or ""
    found: set[str] = set()
    # AMD:        "\tv_mov_b32_e32 v0, s0    // 000000015600: BF850001"
    # NVIDIA/alt: "  10: v_mov_b32_e32 ..."
    for line in text.splitlines():
        m = re.match(r"^\s+([a-z][a-z0-9_.]+)\b[^\n]*//\s*[0-9a-fA-F]+:", line)
        if m:
            found.add(m.group(1))
            continue
        m = re.match(r"^\s*[0-9a-fA-F]+:\s+([A-Za-z][A-Za-z0-9_.]+)", line)
        if m:
            found.add(m.group(1))
    return sorted(found)


def disassemble_text(obj: Path) -> str:
    tool = find_llvm_tool("llvm-objdump.exe", "llvm-objdump")
    if not tool:
        return ""
    target = unwrap_code_object(obj)
    if target is None:
        return ""
    try:
        r = subprocess.run([str(tool), "-d", "--no-show-raw-insn", str(target)],
                           capture_output=True, text=True, timeout=300, env=_env())
        return r.stdout or ""
    except Exception:
        return ""


# --------------------------------------------------------------------------- probing

_PROBE_BODY = r'''
extern "C" __global__ void probe(const int* __restrict__ ai,
                                 const int* __restrict__ bi,
                                 int* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int a = ai[i], b = bi[i], acc = out[i];
    char4 ca = *reinterpret_cast<char4*>(&a);
    char4 cb = *reinterpret_cast<char4*>(&b);
    (void)ca; (void)cb;
    %s
    out[i] = acc;
}
'''


def probe_instruction(target: str, stmt: str, name: str = "", *,
                      backend: str = "auto", prelude: str = "",
                      body: str | None = None,
                      extra_flags: list[str] | None = None) -> dict[str, Any]:
    """Compile ONE candidate statement for `target` and report whether it encodes.

    Parameters
    ----------
    target   : architecture id, e.g. ``gfx1201``, ``sm_80``
    stmt     : the statement to test, e.g. ``acc = __builtin_amdgcn_udot4(a,b,acc,false);``
    backend  : ``hip`` | ``cuda`` | ``sycl`` | ``auto`` (inferred from target)

    Returns ``{candidate, ok, error, mnemonics, size}``. ``ok`` means the
    compiler produced a device object for `target`; that is the only claim made.
    """
    if backend == "auto":
        backend = "cuda" if re.match(r"^(sm_|compute_)", target) else \
                  "sycl" if target.lower().startswith(("spir", "intel")) else "hip"
    spec = TOOLCHAINS.get(backend)
    result: dict[str, Any] = {
        "candidate": name or stmt.strip()[:60],
        "backend": backend, "target": target,
        "ok": False, "error": "", "mnemonics": [], "size": 0,
    }
    if not spec:
        result["error"] = (f"unsupported backend '{backend}'. "
                           f"known: {', '.join(sorted(TOOLCHAINS))}")
        return result
    cc = find_compiler(backend)
    if not cc:
        result["error"] = (f"no compiler for '{backend}' (tried "
                           f"{', '.join(spec['compilers'])}; set "
                           f"{spec['roots'][0]})")
        return result

    tmp = Path(tempfile.mkdtemp(prefix="kands_probe_"))
    src = tmp / ("probe" + spec["extension"])
    src.write_text(spec["include"] + prelude + (body % stmt if body else _PROBE_BODY % stmt),
                   encoding="utf-8")
    obj = tmp / "probe.o"

    cmd = [str(cc)]
    arch = spec["arch_flag"]
    if "{target}" in arch:
        cmd.append(arch.format(target=target))
    else:
        cmd.append(arch)
        cmd.append(f"target={target}")
    cmd += list(spec["device_only"]) + list(spec["language"])
    cmd += ["-O3", "-o", str(obj), str(src)]
    cmd += (extra_flags or [])

    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=600, env=_env())
    except subprocess.TimeoutExpired:
        result["error"] = "compile timed out"
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if r.returncode != 0:
        errs = [l.strip() for l in (r.stderr or "").splitlines() if "error" in l.lower()]
        result["error"] = (errs[0] if errs else (r.stderr or "")[:300])[:400]
        return result

    result["ok"] = True
    result["size"] = obj.stat().st_size if obj.exists() else 0
    result["mnemonics"] = disassemble_mnemonics(obj)
    return result


def probe_many(target: str, candidates: list[dict], *,
               backend: str = "auto") -> list[dict[str, Any]]:
    """Probe a list of ``{"name", "stmt"}`` dicts, each in its own TU."""
    return [probe_instruction(target, c["stmt"], c.get("name", ""),
                              backend=backend, prelude=c.get("prelude", ""))
            for c in candidates]


# --------------------------------------------------------------------------- scanning

def scan_residual_asm(files: list[str], mnemonics: list[str] | None = None, *,
                      backend: str | None = None) -> list[dict[str, Any]]:
    """Report live ``asm("...")`` strings naming a forbidden mnemonic.

    ``mnemonics`` wins if given; otherwise the backend's default foreign set is
    used. Only inline-asm *string literals* are matched, so explanatory comments
    mentioning an instruction do not produce false positives.
    """
    if mnemonics is None:
        if backend is None:
            backend = "hip"
        mnemonics = list(FOREIGN_MNEMONICS.get(backend, ()))
    if not mnemonics:
        return []
    alt = "|".join(re.escape(m) for m in mnemonics)
    pat = re.compile(_ASM_STRING_RE + "(?:" + alt + ")")
    hits: list[dict[str, Any]] = []
    for f in files:
        p = Path(f)
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for ln, line in enumerate(text.splitlines(), 1):
            if pat.search(line):
                hits.append({"file": str(p), "line": ln, "text": line.strip()[:160]})
    return hits


# --------------------------------------------------------------------------- building

def build_module(sources: list[str], target: str, out: str, *,
                 backend: str = "auto", includes: list[str] | None = None,
                 flags: list[str] | None = None,
                 concatenate: bool = True,
                 timeout: int = 1800) -> dict[str, Any]:
    """Delete-then-build a device module; report rc, size and kernel symbols.

    ``concatenate=True`` mirrors runtimes that build a single TU from several
    source files (kindi, many others). Set False to compile the first file only.
    """
    if backend == "auto":
        backend = "cuda" if re.match(r"^(sm_|compute_)", target) else "hip"
    spec = TOOLCHAINS.get(backend)
    res: dict[str, Any] = {"ok": False, "rc": -1, "size": 0, "errors": [],
                           "symbols": [], "out": out, "backend": backend}
    if not spec:
        res["errors"] = [f"unsupported backend '{backend}'"]
        return res
    cc = find_compiler(backend)
    if not cc:
        res["errors"] = [f"no compiler for '{backend}'"]
        return res

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        out_p.unlink()          # a stale artifact must never look like success

    texts = []
    for f in (sources if concatenate else sources[:1]):
        p = Path(f)
        if not p.is_file():
            res["errors"].append(f"missing source: {f}")
            return res
        texts.append(p.read_text(encoding="utf-8", errors="replace"))
    tmp = Path(tempfile.mkdtemp(prefix="kands_build_"))
    merged = tmp / ("module" + spec["extension"])
    merged.write_text("\n\n".join(texts), encoding="utf-8")

    cmd = [str(cc)]
    arch = spec["arch_flag"]
    if "{target}" in arch:
        cmd.append(arch.format(target=target))
    else:
        cmd.append(arch)
        cmd.append(f"target={target}")
    cmd += list(spec["device_only"]) + list(spec["language"])
    cmd += ["-O3", "-o", str(out_p), str(merged)]
    for inc in (includes or []):
        cmd.append("-I" + inc)
    cmd += (flags or [])

    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=_env())
    except subprocess.TimeoutExpired:
        res["errors"] = ["compile timed out"]
        return res
    res["rc"] = r.returncode
    if r.returncode != 0:
        res["errors"] = [l.strip() for l in (r.stderr or "").splitlines()
                         if "error" in l.lower()][:25]
        return res
    res["ok"] = True
    res["size"] = out_p.stat().st_size if out_p.exists() else 0
    res["symbols"] = module_symbols(str(out_p))
    return res


def module_symbols(obj: str) -> list[str]:
    """Kernel symbols exported by a compiled module (best effort, vendor-neutral)."""
    tool = find_llvm_tool("llvm-objdump.exe", "llvm-objdump")
    if not tool:
        return []
    target = unwrap_code_object(obj)
    if target is None:
        return []
    syms: set[str] = set()
    for flag in ("-t", "--syms"):
        try:
            r = subprocess.run([str(tool), flag, str(target)], capture_output=True,
                               text=True, timeout=300, env=_env())
        except Exception:
            continue
        for line in (r.stdout or "").splitlines():
            for tok in line.split():
                if tok.startswith(("k_", "d_k_", "_Z")):
                    # AMD emits a ".kd" descriptor twin for every kernel; only
                    # the plain name is loadable via hipModuleGetFunction.
                    syms.add(tok.split(".kd")[0])
        if syms:
            break
    return sorted(syms)


def diff_symbols(found: list[str], expected: list[str]) -> dict[str, list[str]]:
    """Compare produced kernel symbols against an expected list."""
    fset, eset = set(found), set(expected)
    return {
        "missing": sorted(eset - fset),
        "unexpected": sorted(fset - eset),
        "present": sorted(eset & fset),
    }


# --------------------------------------------------------------------------- capability report

def capability_report(target: str, backend: str = "auto") -> dict[str, Any]:
    """What this host can actually do for `target` — discovery, not assumption."""
    if backend == "auto":
        backend = "cuda" if re.match(r"^(sm_|compute_)", target) else "hip"
    spec = TOOLCHAINS.get(backend, {})
    return {
        "backend": backend,
        "target": target,
        "supported_backend": bool(spec),
        "compiler": str(find_compiler(backend) or ""),
        "root": str(find_root(backend) or ""),
        "objdump": str(find_llvm_tool("llvm-objdump.exe", "llvm-objdump") or ""),
        "known_backends": sorted(TOOLCHAINS),
    }
