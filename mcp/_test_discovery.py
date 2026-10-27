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
import preport_docs         # noqa: E402

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

# --------------------------------------------------------------------------- facts
print("\n== arch facts: flattening, provenance, feeds ==")
import arch_facts          # noqa: E402


def rows_of(diag: dict) -> dict:
    return {r["id"]: r for r in diag.get("rows", [])}


gfx1031 = bp.diagnose("hip", target_arch="gfx1031", host_os="linux", probe_host=False)
r1031 = rows_of(gfx1031)
check("ARCH_MAP's nested features block is read (gfx1031 wmma:false -> FAIL)",
      r1031["capability.matrix"]["status"] == "FAIL",
      str(r1031["capability.matrix"]["evidence"]))
check("a declared-absent matrix core gates the port",
      "capability.matrix" in gfx1031["blockers"], str(gfx1031["blockers"]))
cdna2 = rows_of(bp.diagnose("hip", target_arch="cdna2", probe_host=False))
check("nested features read in the PASS direction too (cdna2 wmma:true)",
      cdna2["capability.matrix"]["status"] == "PASS",
      str(cdna2["capability.matrix"]["evidence"]))
check("provenance is reported per fact",
      cdna2["capability.matrix"]["confidence"] == "documented",
      cdna2["capability.matrix"]["confidence"])

gen = bp.diagnose("sycl", target_arch="gen12_hp", probe_host=False)
check("an arch that contradicts its own sub_target is UNVALIDATED, not guessed",
      rows_of(gen)["capability.matrix"]["status"] == "UNVALIDATED"
      and any("disagree" in e for e in rows_of(gen)["capability.matrix"]["evidence"]),
      str(rows_of(gen)["capability.matrix"]["evidence"]))

with tempfile.TemporaryDirectory() as td:
    feed = Path(td) / "facts.json"
    feed.write_text(json.dumps({"arches": {"gfx1031": {"wmma": True}}}), encoding="utf-8")
    os.environ["KANDS_ARCH_FACTS"] = str(feed)
    fed = rows_of(bp.diagnose("hip", target_arch="gfx1031", probe_host=False))
    del os.environ["KANDS_ARCH_FACTS"]
check("a user-fed fact beats ARCH_MAP",
      fed["capability.matrix"]["status"] == "PASS"
      and fed["capability.matrix"]["confidence"] == "asserted",
      str(fed["capability.matrix"]))

unfed = bp._probe_capability({"field": "wmma", "expect": True},
                            {"arch": "custom_chip", "known": True, "facts": {}}, None)
check("a fact no source has is UNVALIDATED and names the key to feed",
      unfed[0] == "UNVALIDATED" and any("ARCH_FACTS" in e for e in unfed[2]), str(unfed))
check("no source at all is reported, not fabricated",
      bp.diagnose("cpu", target_arch="not_an_arch_xyz", probe_host=False)["facts"]["known"] is False)
check("ARCH_MAP's avx512 spelling resolves to the row's avx512f field",
      rows_of(bp.diagnose("cpu", target_arch="intel_13th", host_os="linux", probe_host=False))
      ["capability.avx512f"]["status"] == "FAIL",
      str(rows_of(bp.diagnose("cpu", target_arch="intel_13th", host_os="linux",
                             probe_host=False))["capability.avx512f"]["evidence"]))
check("a named CPU model does not borrow the build host's ISA probe",
      any("named CPU model" in n for n in
          bp.diagnose("cpu", target_arch="intel_13th", host_os="linux")["facts"]["notes"]),
      str(bp.diagnose("cpu", target_arch="intel_13th", host_os="linux")["facts"]["notes"]))

# --------------------------------------------------------------------------- cross-OS
print("\n== cross-OS and cross-backend coverage ==")
check("a CPU backend exists (it did not before)",
      bp.diagnose("cpu", target_arch="x86_64", host_os="linux").get("known") is True)
check("cpu aliases resolve", bp.resolve_backend("x86_64") == "cpu")
lin = rows_of(bp.diagnose("cpu", target_arch="x86_64", host_os="linux", probe_host=False))
win = rows_of(bp.diagnose("cpu", target_arch="x86_64", host_os="windows", probe_host=False))
check("a Windows-only row is NOT RUN on Linux, never FAIL",
      lin["toolchain.msvc"]["status"] == "NOT RUN" and lin["toolchain.msvc"]["applicable"] is False,
      str(lin["toolchain.msvc"]))
check("the same row applies on Windows",
      win["toolchain.msvc"]["applicable"] is True, str(win["toolchain.msvc"]))
check("a Linux-only row is NOT RUN on Windows",
      win["toolchain.gcc"]["status"] == "NOT RUN" and win["toolchain.gcc"]["applicable"] is False)
check("inapplicable rows are not marching-order steps",
      all(s["id"] != "toolchain.msvc" for s in
          bp.diagnose("cpu", target_arch="x86_64", host_os="linux", probe_host=False)["directive"]["steps"]))

# --------------------------------------------------------------------------- probe honesty
print("\n== probes: no false PASS, no cross-vendor evidence ==")
with tempfile.TemporaryDirectory() as td:
    os.environ["SOME_SDK_ROOT"] = td          # exists, but the tool is not inside it
    st = bp._probe_tool({"env": ["SOME_SDK_ROOT"], "names": ["definitely-not-a-tool-xyz"]})
    del os.environ["SOME_SDK_ROOT"]
check("an existing env root without the tool inside is not PASS",
      st[0] == "FAIL", str(st))

saved_probe = arch_facts.pci_display_devices
arch_facts.pci_display_devices = lambda *a, **k: ([], ["test: no pci devices"])
nogpu = bp.probe_rebar("vulkan", "linux")
arch_facts.pci_display_devices = saved_probe
check("ReBAR with no display device is UNVALIDATED, not FAIL",
      nogpu["status"] == "UNVALIDATED", str(nogpu["evidence"]))

saved_probe = arch_facts.pci_display_devices
arch_facts.pci_display_devices = lambda *a, **k: (
    [{"pci": "0000:01:00.0", "vendor": "0x10de", "class": "0x030000",
      "resize_files": ["resource0_resize"], "bar_bytes": 32 * 1024 ** 3,
      "bar_max_bytes": 32 * 1024 ** 3}], [])
hip_smi = bp._probe_rebar_nvidia_smi("hip")
arch_facts.pci_display_devices = saved_probe
check("nvidia-smi evidence is not attached to a HIP target", hip_smi is None, str(hip_smi))

# --------------------------------------------------------------------------- directive
print("\n== directive: the marching order ==")
blocked = bp.diagnose("hip", target_arch="gfx1201")
steps = [s for s in blocked["directive"]["steps"] if s["priority"] != "documentation"]
check("a probed-absent impact-5 row is a STOP, first in the order",
      blocked["directive"]["gate"] == "blocked" and steps[0]["priority"] == "stop",
      json.dumps(steps[:1]))
check("every non-PASS applicable row is a step",
      {r["id"] for r in blocked["rows"]
       if r["status"] != "PASS" and r.get("applicable", True)} == {s["id"] for s in steps},
      str([s["id"] for s in steps]))
validate = bp.diagnose("cpu", target_arch="zen3", host_os="windows", probe_host=False)
check("an impact-5 row with no evidence is validate-first, not a pass",
      validate["directive"]["gate"] == "validate-first"
      and "toolchain.msvc" in validate["must_validate"],
      json.dumps(validate["directive"])[:200])
check("an unmodelled backend still gets a directive instead of silence",
      bp.diagnose("not-a-backend")["directive"]["gate"] == "no-model")
metal = bp.diagnose("metal", host_os="macos")
tree = "\n".join(metal["fault_tree"])
statuses = {r["id"]: r["status"] for r in metal["rows"]}
check("the fault tree reports each row's real status",
      all(f"{rid} {status}" in tree for rid, status in statuses.items()), tree)
check("the fault tree no longer claims PASS for rows it never ran",
      "rows PASS" not in tree, tree)
check("slim carries the facts provenance and the directive",
      {"facts", "directive", "must_validate"} <= set(bp.slim(blocked)), str(list(bp.slim(blocked))))
rendered_cpu = bp.render(bp.diagnose("cpu", target_arch="x86_64", host_os="linux"))
check("render shows where the facts came from", "Where the facts came from" in rendered_cpu)
check("render shows the directive", "Directive — the marching order" in rendered_cpu)

# --------------------------------------------------------------------------- probe fixes
print("\n== probes: BAR window, nested API blocks, device precedence ==")
saved_pci = arch_facts.pci_display_devices


def _fake_pci(bar, mx):
    return lambda *a, **k: ([{"pci": "0000:03:00.0", "vendor": "0x1002",
                             "class": "0x030000",
                             "resize_files": ["resource0_resize"] if mx else [],
                             "bar_bytes": bar, "bar_max_bytes": mx}], [])


arch_facts.pci_display_devices = _fake_pci(16 * 1024 ** 3, 32 * 1024 ** 3)
big = bp.probe_rebar("hip", "linux")
arch_facts.pci_display_devices = _fake_pci(128 * 1024 ** 2, 16 * 1024 ** 3)
small = bp.probe_rebar("hip", "linux")
arch_facts.pci_display_devices = _fake_pci(64 * 1024 ** 2, 64 * 1024 ** 2)
tiny = bp.probe_rebar("hip", "linux")
arch_facts.pci_display_devices = saved_pci
check("a BAR window at or above the threshold is PASS",
      big["status"] == "PASS", str(big["evidence"]))
check("a resizable-but-still-small window is UNVALIDATED, never PASS",
      small["status"] == "UNVALIDATED" and small["window_mib"] == 128,
      str(small["evidence"]))
check("a device whose largest window is below the threshold is FAIL",
      tiny["status"] == "FAIL", str(tiny["evidence"]))

vk_amd = {r["id"]: r for r in bp.diagnose("vulkan", target_arch="gfx1030")["rows"]}
vk_intel = {r["id"]: r for r in bp.diagnose(
    "vulkan", target_arch="gen12_hpc_battlemage")["rows"]}
check("Vulkan cooperative matrix is answered from ARCH_MAP's nested vulkan block",
      vk_amd["capability.coopmat"]["status"] == "FAIL",
      str(vk_amd["capability.coopmat"]["evidence"]))
check("the same row passes where the table says the driver supports it",
      vk_intel["capability.coopmat"]["status"] == "PASS")
check("a capability row reads the first candidate field name a source has",
      "vulkan.cooperative_matrix" in vk_amd["capability.coopmat"]["evidence"][0])
check("Vulkan subgroup width is reported as info, not as a pass",
      vk_amd["capability.subgroup"]["status"] == "NOT RUN")

arm = {r["id"]: r for r in bp.diagnose(
    "cpu", target_arch="apple_m2", host_os="macos", probe_host=False)["rows"]}
check("an x86 ISA row is not applicable to an ARM target",
      arm["capability.avx2"]["applicable"] is False
      and arm["capability.avx2"]["status"] == "NOT RUN",
      str(arm["capability.avx2"]))
check("an ARM ISA row is applicable on that ARM target",
      arm["capability.neon"]["applicable"] is True, str(arm["capability.neon"]))

saved_dev = arch_facts.probe_device_capabilities
arch_facts.probe_device_capabilities = lambda backend, host_os="": {
    "backend": backend, "host_os": host_os, "probed": True, "arch": "gfx1030",
    "facts": {"wmma": True}, "evidence": {"wmma": "test device reports matrix cores"},
    "devices": [], "notes": []}
dev_same = bp.diagnose("hip", target_arch="gfx1030")
dev_other = bp.diagnose("hip", target_arch="gfx1031")
arch_facts.probe_device_capabilities = saved_dev
same_row = {r["id"]: r for r in dev_same["rows"]}["capability.matrix"]
other_row = {r["id"]: r for r in dev_other["rows"]}["capability.matrix"]
check("a measured device fact wins over the table for the same arch",
      dev_same["device"]["applies"] is True and same_row["status"] == "PASS"
      and "[device]" in same_row["evidence"][0], str(same_row["evidence"]))
check("a measured device fact is not applied to a different arch",
      dev_other["device"]["applies"] is False
      and "[device]" not in other_row["evidence"][0], str(other_row["evidence"]))

# --------------------------------------------------------------------------- documents
print("\n== pre-port analysis documents (mandatory) ==")
with tempfile.TemporaryDirectory() as td:
    d0 = bp.diagnose("vulkan", target_arch="gfx1030", host_os="linux", preport_dir=td)
    check("documents are required and reported missing before they exist",
          d0["documents"]["required"] is True and d0["documents"]["gate"] == "missing",
          json.dumps(d0["documents"])[:200])
    check("the directive opens with the documentation stop",
          d0["directive"]["steps"][0]["priority"] == "documentation"
          and d0["directive"]["docs_gate"] == "missing",
          json.dumps(d0["directive"]["steps"][:1]))
    check("render says STOP before the first kernel edit",
          "STOP-DOCS" in bp.render(d0) and "Pre-port analysis documents" in bp.render(d0))

    man = bp.preport_docs.write(d0, bp.render(d0), preport_dir=td) if hasattr(bp, "preport_docs") \
        else preport_docs.write(d0, bp.render(d0), preport_dir=td)
    check("the full set is written",
          len(man["files"]) == 7 and all(Path(f).is_file() for f in man["files"]),
          str(man["files"]))
    check("a written set reads back as complete",
          preport_docs.check(d0, preport_dir=td)["gate"] == "complete")

    waiver = Path(man["dir"]) / "05-waivers.md"
    waiver.write_text(waiver.read_text(encoding="utf-8") + "\nMY DECISION\n",
                      encoding="utf-8")
    again = bp.analyze("vulkan", target_arch="gfx1030", host_os="linux", preport_dir=td)
    check("analyze() writes the set and reports it complete",
          again["documents"]["gate"] == "complete", json.dumps(again["documents"])[:200])
    check("regeneration never overwrites the human waiver ledger",
          "MY DECISION" in waiver.read_text(encoding="utf-8"))
    check("the documentation stop disappears once the set exists",
          all(s["priority"] != "documentation" for s in again["directive"]["steps"]))

    drifted = bp.diagnose("vulkan", target_arch="gfx1030", host_os="linux", preport_dir=td)
    row0 = drifted["rows"][0]
    row0["status"] = "FAIL" if row0["status"] != "FAIL" else "PASS"
    stale = preport_docs.check(drifted, preport_dir=td)
    check("a document set that disagrees with the probe is stale, not done",
          stale["gate"] == "stale" and stale["drift"], json.dumps(stale)[:300])

# --------------------------------------------------------------------------- safeguards
print("\n== agnostic safeguards: advisory before the port, gating after ==")
import postport_check  # noqa: E402

declared = {r["id"] for r in (bp.load_requirements().get("safeguards") or [])}
# gate.correctness is the precondition for every rule, not one of the shipped
# safeguards, so it is not declared in the requirements file.
implemented = {rule({})["id"] for rule in postport_check.RULES} - {"gate.correctness"}
check("every declared safeguard has a recheck rule, and none is undeclared",
      declared == implemented, f"declared-only={sorted(declared - implemented)} "
                               f"rule-only={sorted(implemented - declared)}")

for backend in ("hip", "vulkan", "cpu", "metal"):
    d = bp.diagnose(backend, host_os="linux", probe_host=False)
    rows = {r["id"]: r for r in d["rows"]}
    missing = declared - set(rows)
    check(f"`{backend}` carries every safeguard row", not missing, f"missing={sorted(missing)}")
    check(f"`{backend}` safeguard rows are advisory, not blockers",
          all(rows[i]["advisory"] for i in declared)
          and not (declared & set(d["blockers"]))
          and not (declared & set(d["must_validate"])),
          str(d["blockers"]))
    steps = {s["id"]: s for s in d["directive"]["steps"]}
    check(f"`{backend}` lists them as RECHECK steps",
          all(steps.get(i, {}).get("priority") == "recheck" for i in declared),
          str([steps.get(i, {}).get("priority") for i in declared]))

meteor = postport_check.evaluate({
    "backend": "vulkan", "arch": "meteor_lake", "correctness_verified": True,
    "iterations_per_s": 20.65, "source_iterations_per_s": 33.0,
    "dispatches_per_iteration": 116, "source_dispatches_per_iteration": 116,
    "dispatch_overhead_us": 385.0, "workgroup_size": 256,
    "max_verified_workgroup": 256, "peak_utilisation_pct": 0.03,
})
rows = {r["id"]: r for r in meteor["rows"]}
check("the worked example reproduces: 385 us x 116 dispatches is 92% of the token",
      rows["efficiency.dispatch_overhead"]["status"] == "FAIL"
      and "92%" in rows["efficiency.dispatch_overhead"]["headline"],
      rows["efficiency.dispatch_overhead"]["headline"])
check("0.03% of peak is reported as not-arithmetic-bound",
      rows["efficiency.peak_utilisation"]["status"] == "FAIL")
check("33 -> 20.65 tok/s is reported as a source-parity regression",
      rows["efficiency.source_parity"]["status"] == "FAIL")

inconsistent = postport_check.evaluate({
    "backend": "vulkan", "correctness_verified": True, "iterations_per_s": 20.65,
    "dispatches_per_iteration": 263, "dispatch_overhead_us": 385.0})
check("numbers that cannot all be true are UNVALIDATED, not a confident FAIL",
      [r for r in inconsistent["rows"] if r["id"] == "efficiency.dispatch_overhead"][0]
      ["status"] == "UNVALIDATED")

empty = postport_check.evaluate({"backend": "hip"})
check("a recheck with no measurements reports NOT RUN, never PASS",
      empty["counts"]["PASS"] == 0 and empty["verdict"] == "incomplete",
      json.dumps(empty["counts"]))
check("a missing measurement names the number to go and get",
      "dispatch_overhead_us" in postport_check.render(empty))

with tempfile.TemporaryDirectory() as td:
    first = postport_check.write(meteor, td)
    again = postport_check.write(postport_check.evaluate({
        "backend": "vulkan", "arch": "meteor_lake", "correctness_verified": True,
        "iterations_per_s": 28.0, "source_iterations_per_s": 33.0,
        "dispatches_per_iteration": 92, "source_dispatches_per_iteration": 116,
        "dispatch_overhead_us": 385.0, "workgroup_size": 256,
        "max_verified_workgroup": 256, "peak_utilisation_pct": 0.05}), td)
    check("the recheck is written as markdown and json",
          Path(first["markdown"]).is_file() and Path(first["json"]).is_file())
    check("a second run reports the delta against the first",
          again["deltas"].get("iterations_per_s", {}).get("pct", 0) > 30,
          json.dumps(again["deltas"])[:200])
    check("the delta survives into the markdown",
          "Since the last recheck" in Path(again["markdown"]).read_text(encoding="utf-8"))

# --------------------------------------------------------------------------- result
print()
if FAILS:
    print(f"RESULT: FAIL — {len(FAILS)} check(s) failed")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: PASS — all checks passed")
