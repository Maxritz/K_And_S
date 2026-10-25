#!/usr/bin/env python
"""Regression tests for discovery correctness + the backend-readiness step.

Run from any directory (script dir is added to sys.path, so the local ``mcp``
folder does not shadow the installed ``mcp`` package):

    python H:\\K_and_S\\mcp\\_test_discovery.py

Every check prints PASS or FAIL. Exit code is non-zero if any check failed.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(tempfile.gettempdir())

import server as s            # noqa: E402
import api_map                # noqa: E402
import backend_profile as bp  # noqa: E402

FAILS: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}" + (f" — {detail}" if detail else ""))
        FAILS.append(name)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- discovery
print("\n== discovery: content-aware classification ==")
with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "proj"
    write(root / "kern" / "hip_kernel.cu",
          '#include <hip/hip_runtime.h>\n__global__ void k(){ hipMalloc(0, 1); }\n')
    write(root / "kern" / "cuda_kernel.cu",
          '#include <cuda_runtime.h>\n__global__ void k(){ __shared__ int x; }\n')
    write(root / "kern" / "hdr.cuh",
          '#include <hip/hip_runtime.h>\n__device__ void d();\n')
    write(root / "d" / "avx512_gemm.cpp",
          '#include <immintrin.h>\nvoid g(){ __m512 a = _mm512_set1_ps(0.f); (void)a; }\n')
    write(root / "d" / "plain_host.cpp", "int main(){ return 0; }\n")
    write(root / "shaders" / "t.tesc", "#version 450\nvoid main(){}\n")
    write(root / "sycl" / "k.sycl", "#include <sycl/sycl.hpp>\nvoid k(){ sycl::queue q; }\n")
    write(root / "build" / "copy.cu",
          '#include <hip/hip_runtime.h>\n__global__ void k(){}\n')
    write(root / "node_modules" / "x" / "v.comp", "#version 450\n")

    findings = s._scan_sources(root)
    by_path = {f["path"]: f for f in findings}

    check("HIP in .cu is classified gpu_hip",
          by_path.get("kern/hip_kernel.cu", {}).get("backend") == "gpu_hip",
          str(by_path.get("kern/hip_kernel.cu")))
    check("CUDA in .cu stays gpu_cuda",
          by_path.get("kern/cuda_kernel.cu", {}).get("backend") == "gpu_cuda",
          str(by_path.get("kern/cuda_kernel.cu")))
    check("HIP header .cuh is gpu_hip",
          by_path.get("kern/hdr.cuh", {}).get("backend") == "gpu_hip",
          str(by_path.get("kern/hdr.cuh")))
    check("AVX-512 .cpp is a CPU kernel (no 'cpu' in path)",
          by_path.get("d/avx512_gemm.cpp", {}).get("backend") == "cpu_x86_avx512",
          str(by_path.get("d/avx512_gemm.cpp")))
    check("host-only .cpp is not reported",
          "d/plain_host.cpp" not in by_path)
    check(".tesc is discovered",
          by_path.get("shaders/t.tesc", {}).get("backend") == "gpu_vulkan",
          str(by_path.get("shaders/t.tesc")))
    check(".sycl is discovered",
          by_path.get("sycl/k.sycl", {}).get("backend") == "gpu_sycl",
          str(by_path.get("sycl/k.sycl")))
    check("build/ copy is pruned",
          not any(p.startswith("build/") for p in by_path), str(sorted(by_path)))
    check("node_modules/ is pruned",
          not any(p.startswith("node_modules/") for p in by_path), str(sorted(by_path)))

# --------------------------------------------------------------------------- api_map
print("\n== api_map: no phantom rows, unknowns surfaced ==")
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    write(root / "a.cu", """
#include <cuda_runtime.h>
__global__ void k(float* o) {
  void* p = 0;
  cudaMallocAsync(&p, 64, 0);
  unsigned x;
  asm volatile("mystery.op %0;" : "=r"(x));
  o[0] = (float)x;
}
""")
    ref = api_map.build_reference([str(root / "a.cu")], "sm_80", "gfx1201")
    ids = {r["id"] for r in ref["rows"]}
    check("cudaMallocAsync is inventoried", "cudaMallocAsync" in ids, str(sorted(ids)))
    check("cudaMalloc is not invented from cudaMallocAsync",
          "cudaMalloc" not in ids, str(sorted(ids)))
    check("unknown asm is surfaced as ptx:",
          any(i.startswith("ptx:") for i in ids), str(sorted(ids)))
    unverified = [r for r in ref["rows"] if r["category"] == "unverified"]
    check("unknown asm row is unverified, not clean",
          any(r["id"].startswith("ptx:") for r in unverified),
          str([r["id"] for r in ref["rows"]]))

    missing = api_map.build_reference([str(root / "nope.cu")], "sm_80", "gfx1201")
    check("missing file is reported",
          any("nope.cu" in f for f in missing.get("unreadable_files", [])),
          str(missing.get("unreadable_files")))
    check("missing file does not look like clean code",
          missing.get("symbols_found") == 0 and bool(missing.get("unreadable_files")))

    tma_intel = api_map.resolve("TMA (Tensor Memory Accelerator)", "gen12_hp")
    check("TMA on Intel is not resolved from the NVIDIA entry",
          tma_intel["status"] == "manual" and "intel" in tma_intel["note"].lower(),
          json.dumps(tma_intel))
    tma_nvidia = api_map.resolve("TMA (Tensor Memory Accelerator)", "sm_90")
    check("TMA on an NVIDIA target still resolves",
          tma_nvidia["status"] != "unverified", json.dumps(tma_nvidia))

# --------------------------------------------------------------------------- readiness
print("\n== backend readiness: probes, Pareto, ReBAR ==")
check("unknown backend is reported, not guessed",
      bp.diagnose("definitely-not-a-backend").get("known") is False)
check("aliases resolve", bp.resolve_backend("dx12") == "directx12")

hip = bp.diagnose("hip", target_arch="gfx1201", host_os="windows", probe_host=False)
statuses = {r["id"]: r["status"] for r in hip["rows"]}
check("probe_host=False leaves tool rows UNVALIDATED",
      statuses.get("toolchain.hipcc") == "UNVALIDATED", str(statuses))
check("documented capability still resolves with probe_host=False",
      statuses.get("capability.matrix") in ("PASS", "FAIL"), str(statuses))
check("ReBAR is never guessed PASS without a probe",
      hip["rebar"]["status"] == "UNVALIDATED", hip["rebar"]["status"])
check("ReBAR carries a fallback for hip",
      bool(hip["rebar"].get("fallback")), json.dumps(hip["rebar"]))
check("every row has a status from the evidence vocabulary",
      all(r["status"] in bp.LABELS for r in hip["rows"]),
      str([r["status"] for r in hip["rows"]]))

scores = [float(r["score"]) for r in hip["rows"]]
check("Pareto total equals the sum of scores",
      abs(hip["ranking"]["total"] - round(sum(scores), 2)) < 1e-9,
      f'{hip["ranking"]["total"]} vs {sum(scores)}')
check("vital few is non-empty and smaller than all rows",
      0 < len(hip["ranking"]["vital_few"]) <= len(hip["rows"]),
      str(hip["ranking"]))
cumulative_ok = True
total = hip["ranking"]["total"]
running = 0.0
for row in sorted(hip["rows"], key=lambda r: (-float(r["score"]), str(r["id"]))):
    if float(row["score"]) <= 0:
        continue
    before = running / total
    running += float(row["score"])
    if row.get("vital") != (before < 0.8):
        cumulative_ok = False
check("vital flag matches the 80% cumulative rule", cumulative_ok)

metal = bp.diagnose("metal", host_os="windows", probe_host=False)
check("Metal ReBAR is NOT RUN / not applicable, not PASS",
      metal["rebar"]["status"] == "NOT RUN" and metal["rebar"]["supported"] is False,
      json.dumps({k: metal["rebar"].get(k) for k in ("status", "supported")}))
check("metal on windows has a host blocker",
      "host.macos" in metal["blockers"], str(metal["blockers"]))

rendered = bp.render(hip)
check("render includes READINESS lines", "READINESS:" in rendered)
check("render includes the ReBAR gate", "ReBAR gate" in rendered)
check("render states the fallback decision when ReBAR is not PASS",
      "use the fallback" in rendered)

check("slim output is JSON-safe",
      json.loads(json.dumps(bp.slim(hip)))["backend"] == "hip")

# --------------------------------------------------------------------------- result
print()
if FAILS:
    print(f"RESULT: FAIL — {len(FAILS)} check(s) failed")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: PASS — all checks passed")
