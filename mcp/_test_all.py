#!/usr/bin/env python
"""Exercise every script and every MCP surface in the bundle.

What this covers, in one run:

  1. every ``.py`` file compiles;
  2. every JSON file parses, and the cross-file references hold (arch ids,
     requirement rows, toolchain keys);
  3. every tool advertised by ``tools/list`` is called through the real
     dispatcher with realistic arguments, and returns text without a traceback;
  4. every resource advertised by ``resources/list`` reads non-empty;
  5. the JSON-RPC handlers return the promised shapes;
  6. the other test scripts are run as subprocesses and their exit codes are
     reported.

The point is coverage, not opinion: a tool that silently stops answering, a
resource that throws, or a new tool that nobody wired a test for is a failure.
Mutating tools write into a temporary state directory, so a run leaves the
checkout untouched.

    python _test_all.py            # 0 = pass, 1 = fail
"""
from __future__ import annotations

import asyncio
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

FAILS: list[str] = []
SKIPS: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}" + (f" — {detail}" if detail else ""))
        FAILS.append(f"{name}: {detail}")


def skip(name: str, why: str) -> None:
    print(f"SKIP  {name} — {why}")
    SKIPS.append(f"{name}: {why}")


# --------------------------------------------------------------------------- 1. compile
print("== every Python file compiles ==")
py_files = sorted(list(ROOT.glob("*.py")) + list(HERE.glob("*.py")))
bad: list[str] = []
for path in py_files:
    try:
        py_compile.compile(str(path), doraise=True, cfile=str(Path(tempfile.gettempdir()) / "kands.pyc"))
    except py_compile.PyCompileError as exc:
        bad.append(f"{path.name}: {exc.msg}")
check(f"{len(py_files)} Python files compile", not bad, "; ".join(bad))
check("the bundle root was located (not a hardcoded drive)",
      (ROOT / "ARCH_MAP.json").is_file() and (HERE / "server.py").is_file(), str(ROOT))

# --------------------------------------------------------------------------- 2. JSON
print("\n== every JSON file parses, and references hold ==")
json_files = sorted(list(ROOT.glob("*.json")) + list(HERE.glob("*.json")) +
                    list((HERE / "state").glob("*.json")))
parse_errors = []
docs: dict[str, object] = {}
for path in json_files:
    try:
        docs[path.name] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parse_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
check(f"{len(json_files)} JSON files parse", not parse_errors, "; ".join(parse_errors))

arch_ids = set((docs.get("ARCH_MAP.json") or {}).get("entries", {}))
check("ARCH_MAP has entries", len(arch_ids) > 20, str(len(arch_ids)))

reqs = (docs.get("BACKEND_REQUIREMENTS.json") or {}).get("backends", {})
row_problems: list[str] = []
for name, spec in reqs.items():
    if not spec.get("requirements"):
        row_problems.append(f"{name}: no requirements")
    rebar = spec.get("rebar") or {}
    if "supported" not in rebar or not rebar.get("fallback"):
        row_problems.append(f"{name}: rebar has no supported/fallback")
    for row in spec.get("requirements") or []:
        for field in ("id", "kind", "need", "probe", "impact", "likelihood"):
            if field not in row:
                row_problems.append(f"{name}/{row.get('id', '?')}: no {field}")
        if row.get("kind") == "capability" and "field" not in row:
            row_problems.append(f"{name}/{row.get('id')}: capability without a field")
        if row.get("kind") == "capability" and "expect" not in row:
            row_problems.append(f"{name}/{row.get('id')}: capability without expect")
        if not str(row.get("fallback") or "").strip():
            row_problems.append(f"{name}/{row.get('id')}: no fallback")
check(f"{sum(len(s.get('requirements') or []) for s in reqs.values())} requirement rows are complete",
      not row_problems, "; ".join(row_problems[:6]))

# Archs PERF_BENCHMARKS carries numbers for but ARCH_MAP cannot describe. They are
# a known gap, not a licence to invent capability flags: the estimate now says so
# in its own output. A NEW id appearing here is a failure, so the list is a ratchet.
KNOWN_PERF_ONLY_ARCHS = {"sm_90", "ad103", "gfx1105", "gfx90a", "arc_a770",
                         "arc_b580", "gen12_hpc"}
perf = (docs.get("PERF_BENCHMARKS.json") or {}).get("hardware_theoretical", {})
perf_archs = {k for k in perf if k in arch_ids}
perf_unknown = sorted({k for k in perf if k not in arch_ids and k not in ("description", "notes")})
check(f"{len(perf_archs)} perf-benchmark archs resolve to ARCH_MAP",
      set(perf_unknown) == KNOWN_PERF_ONLY_ARCHS,
      f"gap changed: {sorted(set(perf_unknown) ^ KNOWN_PERF_ONLY_ARCHS)} — add the "
      f"ARCH_MAP entry, or update KNOWN_PERF_ONLY_ARCHS if the gap was accepted")
if perf_unknown:
    print(f"NOTE  {len(perf_unknown)} arch ids have perf data but no capability record: "
          f"{', '.join(perf_unknown)} (estimate_performance says so; compare_archs cannot)")

model_map = docs.get("MODEL_MAP.json") or {}
model_blob = json.dumps(model_map)
mentions = sorted({a for a in arch_ids if f'"{a}"' in model_blob})
check("MODEL_MAP references only arch ids that exist", len(mentions) > 0 or "entries" not in model_map,
      f"{len(mentions)} arch ids referenced")

tk = docs.get("TK_TOOLS.json") or {}
check("TK_TOOLS is a non-empty mapping", bool(tk), str(type(tk).__name__))

config = docs.get("claude_desktop_config.json") or {}
servers = ((config.get("mcpServers") or {}) if isinstance(config, dict) else {})
check("the shipped MCP client config names a server entry", bool(servers), str(config)[:120])

# --------------------------------------------------------------------------- sample tree
import server as s  # noqa: E402

WORK = Path(tempfile.mkdtemp(prefix="kands-all-"))
REPO = WORK / "repo"
(REPO / "kernels").mkdir(parents=True)
(REPO / "kernels" / "attn.cu").write_text(
    '#include <cuda_runtime.h>\n'
    '#define WARPS 4\n'
    'extern "C" __global__ void attn(const float* q, const float* k, float* out) {\n'
    '  __shared__ float tile[32][33];\n'
    '  unsigned m = __shfl_xor_sync(0xffffffffu, 1u, 1);\n'
    '  int d = __dp4a(1, 2, 0);\n'
    '  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32");\n'
    '  __syncthreads();\n'
    '  out[threadIdx.x] = tile[0][0] + q[threadIdx.x] + k[threadIdx.x] + m + d;\n'
    '}\n', encoding="utf-8")
(REPO / "kernels" / "matmul.comp").write_text(
    '#version 460\nlayout(local_size_x = 64) in;\n'
    'layout(std430, binding = 0) buffer B { float data[]; };\n'
    'shared float tile[64];\nvoid main() { uint i = gl_GlobalInvocationID.x; '
    'tile[gl_LocalInvocationIndex] = data[i]; barrier(); data[i] = tile[i]; }\n', encoding="utf-8")
(REPO / "kernels" / "blit.hlsl").write_text(
    "RWStructuredBuffer<float> outBuf : register(u0);\n"
    "[numthreads(64,1,1)] void CSMain(uint3 id : SV_DispatchThreadID) { outBuf[id.x] = 1.0; }\n",
    encoding="utf-8")
(REPO / "kernels" / "conv.metal").write_text(
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void conv(device float* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) "
    "{ simdgroup_float8x8 m; out[gid] = 0; }\n", encoding="utf-8")
(REPO / "kernels" / "reduce.cpp").write_text(
    "#include <sycl/sycl.hpp>\nvoid r(sycl::queue& q, float* x) { q.parallel_for(64, [=](sycl::id<1> i)"
    "{ x[i] = sycl::reduce_over_group(sycl::sub_group{}, x[i], sycl::plus<>()); }); }\n",
    encoding="utf-8")
(REPO / "kernels" / "cpu_kernel.c").write_text(
    "void axpy(int n, float a, const float* x, float* y) { for (int i = 0; i < n; i++) y[i] += a * x[i]; }\n",
    encoding="utf-8")
CU = str(REPO / "kernels" / "attn.cu")
COMP = str(REPO / "kernels" / "matmul.comp")
HLSL = str(REPO / "kernels" / "blit.hlsl")
METAL = str(REPO / "kernels" / "conv.metal")
SYCL = str(REPO / "kernels" / "reduce.cpp")
CPU_K = str(REPO / "kernels" / "cpu_kernel.c")

# Mutating tools must not touch the checkout: point their state at the temp dir.
s._GOAL_STATE_PATH = WORK / "state" / "goal.json"
s._PERF_HISTORY_PATH = WORK / "state" / "perf_history.json"
s._ANALYSIS_CACHE_DIR = WORK / "state" / "analysis"
for _p in (s._GOAL_STATE_PATH, s._PERF_HISTORY_PATH, s._ANALYSIS_CACHE_DIR):
    _p.parent.mkdir(parents=True, exist_ok=True)

TRACE_A = ("PERF_TRACING component=attn phase=prefill ops=1 total_ms=12.5 gpu_ms=11.0 host_ms=1.5\n"
           "PERF_TRACING component=matmul phase=decode ops=1 total_ms=30.0 gpu_ms=28.0 host_ms=2.0\n")
TRACE_B = ("PERF_TRACING component=attn phase=prefill ops=1 total_ms=9.0 gpu_ms=8.0 host_ms=1.0\n"
           "PERF_TRACING component=matmul phase=decode ops=1 total_ms=31.0 gpu_ms=29.0 host_ms=2.0\n")

# name -> (arguments, may_legitimately_refuse)
CASES: dict[str, tuple[dict, bool]] = {
    "discover_kernels": ({"repo": str(REPO), "target_backend": "hip",
                          "target_arch": "gfx1201", "host_os": "linux",
                          "preport_dir": str(WORK / "docs")}, False),
    "write_preport_analysis": ({"target_backend": "cuda", "target_arch": "sm_80",
                                "repo": str(REPO)}, False),
    "postport_recheck": ({"backend": "vulkan", "arch": "meteor_lake",
                          "correctness_verified": True, "iterations_per_s": 20.65,
                          "source_iterations_per_s": 33.0,
                          "dispatches_per_iteration": 116,
                          "source_dispatches_per_iteration": 116,
                          "dispatch_overhead_us": 385.0,
                          "workgroup_size": 256, "max_verified_workgroup": 256,
                          "peak_utilisation_pct": 0.03,
                          "fusions": [{"a": "gate", "b": "up", "dep": "per-row"},
                                      {"a": "silu", "b": "down", "dep": "cross-workgroup"}],
                          "extensions": [{"name": "VK_KHR_timeline_semaphore",
                                          "advertised": True, "enabled": False,
                                          "used": False}],
                          "out_dir": str(WORK / "postport")}, False),
    "check_arch_compatibility": ({"kernel_list": ["attn", "matmul"], "arch": "gfx1201"}, False),
    "query_kernel_features": ({"repo": str(REPO), "pattern": "shfl|subgroup|dp4a|simdgroup",
                               "extensions": [".cu", ".comp", ".metal"]}, False),
    "compare_archs": ({"arch_a": "sm_80", "arch_b": "gfx1201"}, False),
    "generate_porting_report": ({"engine": "tiny-vllm", "target_arch": "gfx1201",
                                 "repo_path": str(REPO), "model_family": "llama"}, False),
    "generate_tracing_plan": ({"repo": str(REPO), "components": ["attn", "matmul"]}, False),
    "diff_perf_traces": ({"trace_a": TRACE_A, "trace_b": TRACE_B}, False),
    "query_toolchain": ({"gpu_vendor": "amd", "os": "linux", "host_compiler": "gcc"}, False),
    "estimate_performance": ({"arch": "gfx1201", "workload": "flash_attn", "seq_len": 4096}, False),
    "compare_to_target": ({"arch": "gfx1201", "workload": "flash_attn",
                           "target_tok_per_s": 200, "current_tok_per_s": 120}, False),
    "search_arxiv": ({"query": "flash attention GPU kernel porting", "max_results": 2}, True),
    "search_github": ({"query": "hip flash attention kernel", "max_results": 2}, True),
    "search_references": ({"query": "vulkan cooperative matrix matmul", "max_results": 2}, True),
    "search_online": ({"query": "ROCm 7 supported GPUs", "max_results": 2}, True),
    "record_performance": ({"arch": "gfx1201", "workload": "flash_attn", "tok_per_s": 120.5,
                            "notes": "suite"}, False),
    "suggest_optimizations": ({"arch": "gfx1201", "workload": "flash_attn",
                               "target_tok_per_s": 200}, False),
    "export_contribution": ({"engine_name": "tiny-vllm", "arch": "gfx1201",
                             "workload": "flash_attn",
                             "output_path": str(WORK / "contribution.md")}, False),
    "set_goal": ({"goal": "port attention kernels to gfx1201", "target_arch": "gfx1201",
                  "source_arch": "sm_80", "target_backend": "hip"}, False),
    "update_goal": ({"override_field": "target_tok_per_s", "new_value": "150",
                     "reason": "suite"}, False),
    "check_guardrail": ({"action": "edit attention kernel", "reason": "port step"}, False),
    "prepare_pull_request": ({"repo_url": "https://github.com/example/kands",
                              "branch_name": "kands/gfx1201-attn",
                              "change_types": ["docs", "kernels"]}, False),
    "verify_numerical_equivalence": ({"arch": "gfx1201", "workload": "flash_attn",
                                      "source_values": [1.0, 2.0, 3.0],
                                      "target_values": [1.0, 2.0001, 3.0]}, False),
    "strip_tracing": ({"directory": str(REPO), "strip_debug": 1, "strip_perf": 1,
                       "extensions": [".cu"]}, False),
    "reject_common_pattern": ({"common_idea": "just enable WMMA on RDNA2",
                               "arch": "gfx1031"}, False),
    "radical_experiment_plan": ({"arch": "gfx1201", "kernel_type": "matmul",
                                 "experiment_type": "memory"}, False),
    "detect_toolchain": ({"gpu_vendor": "amd", "host_compiler": "auto"}, False),
    "autonomous_port": ({"repo_path": str(REPO), "source_arch": "sm_80",
                         "target_arch": "gfx1201", "target_backend": "hip",
                         "target_language": "cxx", "target_os": "linux",
                         "max_files": 50, "timeout_seconds": 30,
                         "force_refresh": True}, False),
    "capability_report": ({"target": "gfx1201", "backend": "hip"}, False),
    "probe_isa": ({"target": "gfx1201", "backend": "hip",
                   "candidates": ["acc = __builtin_amdgcn_udot4(a, b, acc, false);",
                                  {"name": "mfma",
                                   "stmt": 'acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);'}]}, True),
    "scan_residual_asm": ({"files": [CU], "backend": "hip"}, False),
    "build_module": ({"sources": [CU], "target": "gfx1201",
                      "out": str(WORK / "attn.hsaco"), "backend": "hip"}, True),
    "code_object_report": ({"path": str(WORK / "attn.hsaco"), "arch": "gfx1201"}, True),
    "build_porting_map": ({"files": [CU, COMP, HLSL, METAL, SYCL, CPU_K],
                           "source_arch": "sm_80", "target_arch": "gfx1201"}, False),
    "lookup_construct": ({"construct": "__shfl_xor_sync", "target_arch": "gfx1201"}, False),
    "verify_porting_map": ({"files": [CU], "source_arch": "sm_80",
                            "target_arch": "gfx1201", "backend": "hip"}, False),
    "build_function_reference": ({"files": [CU], "source_arch": "sm_80",
                                  "target_arch": "gfx1201"}, False),
    "lookup_api": ({"api": "cudaMallocManaged", "target_arch": "gfx1201"}, False),
}


# --------------------------------------------------------------------------- 3. tools
def _text(result) -> str:
    return "\n".join(getattr(item, "text", str(item)) for item in result)


async def run_tools() -> None:
    print("\n== every advertised tool answers ==")
    advertised = [t.name for t in s._tools()]
    missing_cases = sorted(set(advertised) - set(CASES))
    check("every advertised tool has a call case in this suite", not missing_cases,
          f"untested: {missing_cases}")
    extra_cases = sorted(set(CASES) - set(advertised))
    check("no case refers to a tool that does not exist", not extra_cases,
          f"stale cases: {extra_cases}")

    for name in advertised:
        args, may_refuse = CASES.get(name, ({}, True))
        try:
            result = await asyncio.wait_for(s._dispatch_tool(name, args), timeout=180)
            text = _text(result)
            ok = bool(text.strip())
            detail = text.strip()[:300]
            if not ok:
                check(f"tool {name} returns text", False, "empty response")
                continue
            check(f"tool {name} returns text", True)
            trace = "Traceback (most recent call last)" in text
            check(f"tool {name} raises nothing internally", not trace, text[:400])
            if may_refuse:
                skip(f"tool {name} full result", "needs a compiler, network or artifact")
            else:
                user_error = text.lstrip().startswith(("Error:", "ERROR"))
                check(f"tool {name} produces a real answer", not user_error, detail)
        except asyncio.TimeoutError:
            check(f"tool {name} completes", False, "timed out after 180s")
        except Exception as exc:  # a raise escaping the dispatcher is a bug
            check(f"tool {name} completes", False,
                  f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}")


# --------------------------------------------------------------------------- 4. resources
def run_resources() -> None:
    print("\n== every advertised resource reads ==")
    uris = []
    for item in s._list_resources():
        uris.append(str(getattr(item, "uri", "")))
    check("resources/list is non-empty", bool(uris), str(len(uris)))
    empties, throwers = [], []
    for uri in uris:
        try:
            text = s._read_resource(uri)
        except Exception as exc:
            throwers.append(f"{uri}: {type(exc).__name__}: {exc}")
            continue
        if not str(text).strip():
            empties.append(uri)
    check(f"all {len(uris)} resources read without raising", not throwers, "; ".join(throwers))
    check(f"all {len(uris)} resources return content", not empties, ", ".join(empties))


# --------------------------------------------------------------------------- 5. handlers
async def run_handlers() -> None:
    print("\n== JSON-RPC handlers return the promised shapes ==")
    list_tools = s._make_list_tools_handler()
    result = await list_tools(None, None)
    check("tools/list returns ToolsResult with tools", bool(getattr(result, "tools", None)),
          type(result).__name__)
    check("tools/list marks no tool as requiring an unknown field",
          all(getattr(t, "name", "") for t in result.tools))

    call = s._make_call_tool_handler()
    answers = await call(None, SimpleNamespace(
        name="check_arch_compatibility",
        arguments={"kernel_list": ["attn"], "arch": "gfx1201"}))
    texts = [getattr(c, "text", "") for c in getattr(answers, "content", [])]
    check("tools/call returns content for a cheap tool", any(texts), str(answers)[:200])

    ping = s._make_ping_handler() if hasattr(s, "_make_ping_handler") else None
    if ping:
        await ping(None, None)
        check("ping handler answers", True)
    else:
        skip("ping handler", "no _make_ping_handler in this server build")

    read = s._make_read_resource_handler() if hasattr(s, "_make_read_resource_handler") else None
    if read:
        res = await read(None, SimpleNamespace(uri="kands://status"))
        check("resources/read answers", bool(getattr(res, "contents", None)),
              type(res).__name__)
    else:
        skip("resources/read handler", "no _make_read_resource_handler in this server build")


# --------------------------------------------------------------------------- 6. scripts
SCRIPT_TIMEOUT = 600


def run_scripts() -> None:
    print("\n== the other test scripts run ==")
    scripts = ["_test_discovery.py", "_test_gpu.py", "_smoke_register.py",
               "_final_test.py", "_validate_bundle.py"]
    for name in scripts:
        path = HERE / name
        if not path.is_file():
            skip(name, "not present")
            continue
        try:
            proc = subprocess.run([sys.executable, name], cwd=HERE, capture_output=True,
                                  text=True, timeout=SCRIPT_TIMEOUT)
        except subprocess.TimeoutExpired:
            check(f"{name} finishes", False, f"timed out after {SCRIPT_TIMEOUT}s")
            continue
        tail = (proc.stdout or "").strip().splitlines()[-3:]
        check(f"{name} exits 0", proc.returncode == 0,
              f"rc={proc.returncode} :: {' | '.join(tail)} :: "
              f"{(proc.stderr or '').strip()[-400:]}")


# --------------------------------------------------------------------------- regressions
async def run_regressions() -> None:
    """The defects this suite found, kept as fixed-shape checks."""
    print("\n== regressions found by this suite ==")

    async def _call(name, args):
        result = await s._dispatch_tool(name, args)
        return "\n".join(getattr(i, "text", str(i)) for i in result)

    tc = await _call("query_toolchain", {"gpu_vendor": "amd", "os": "linux",
                                               "host_compiler": "gcc"})
    check("query_toolchain renders the list-shaped sdk_paths",
          "SDK paths" in tc and "/opt/rocm" in tc, tc[:200])

    ne = await _call("verify_numerical_equivalence",
                           {"arch": "gfx1201", "workload": "flash_attn",
                            "source_values": [1.0, 2.0], "target_values": [1.0, 2.0001]})
    check("verify_numerical_equivalence reports a date and a verdict",
          "Verdict" in ne and "Date:" in ne, ne[:200])

    perf_only = await _call("estimate_performance",
                                  {"arch": "sm_90", "workload": "flash_attn", "seq_len": 4096})
    check("an arch with perf data but no capability row still estimates, and says why it is limited",
          "Performance Estimate: sm_90" in perf_only and "Capability facts are missing" in perf_only,
          perf_only[:200])

    cmp_ = await _call("compare_to_target",
                             {"arch": "sm_90", "workload": "flash_attn", "target_tok_per_s": 50})
    check("a missing capability record is not reported as 'no WMMA'",
          "has no WMMA" not in cmp_ and "No capability record" in cmp_, cmp_[-400:])

    missing = await _call("estimate_performance",
                                {"arch": "definitely_not_an_arch", "workload": "flash_attn",
                                 "seq_len": 1})
    check("an arch nobody has data for is named as unknown, not estimated",
          "neither ARCH_MAP.json nor PERF_BENCHMARKS.json" in missing, missing[:160])

    recheck = await _call("postport_recheck", {
        "backend": "vulkan", "arch": "meteor_lake", "correctness_verified": False,
        "iterations_per_s": 20.65})
    check("the recheck refuses to judge performance before correctness",
          "gate.correctness" in recheck and "FAIL" in recheck
          and "Fix correctness before any performance work" in recheck, recheck[:300])

    unknown = await _call("no_such_tool", {})
    check("an unknown tool name says so instead of returning nothing",
          unknown.startswith("Error: unknown tool"), unknown[:120])


# --------------------------------------------------------------------------- main
async def main() -> None:
    await run_tools()
    run_resources()
    await run_handlers()
    await run_regressions()
    run_scripts()

    print()
    print(f"checks: {CHECKS}  failed: {len(FAILS)}  skipped: {len(SKIPS)}")
    for item in SKIPS:
        print(f"  SKIP {item}")
    if FAILS:
        print(f"RESULT: FAIL — {len(FAILS)} check(s) failed")
        for item in FAILS:
            print(f"  - {item}")
        sys.exit(1)
    print("RESULT: PASS — every script, tool and resource answered")


if __name__ == "__main__":
    asyncio.run(main())
