"""
backend_profile — discovery-phase answer to "what does this port need for <backend>?"

Why this exists
---------------
A port used to start from the source inventory alone: how many kernels, which
CUDA calls, what the target arch supports. That answers *what is in the tree*,
not *what the target needs before the first kernel is edited*. The difference is
how a plan survives contact with the machine: the matrix instruction is there in
the table but the compiler that emits it is not installed, the host BAR is too
small for the working set, or the target OS cannot run the backend at all.

The method is the diagnostic method applied forward, not a bug hunt:

    enumerate requirements -> probe each one -> rank by impact x likelihood
    -> name the vital few -> state the fallback -> gate the port on evidence

Every row ends in one of four labels, never in a guess:

    PASS          probed and present
    FAIL          probed and absent; use the fallback or stop
    NOT RUN       needs a build or a manual step before it can pass
    UNVALIDATED   could not be probed; treat as absent until proven

ReBAR is a first-class row (``pcie.rebar``): a larger BAR is a real speedup for
weight streaming, and every backend that can use it also has a documented
fallback (staging + chunked copy). A FAIL or UNVALIDATED ReBAR never blocks the
port; it selects the fallback and records which path ran.

Requirements live in ``BACKEND_REQUIREMENTS.json``. Nothing here hardcodes a
vendor: an unknown backend reports what it needs to be described, and an
unlisted arch leaves capability rows UNVALIDATED instead of assuming support.
"""

from __future__ import annotations

import json
import glob
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- vocabulary

LABELS = ("PASS", "FAIL", "NOT RUN", "UNVALIDATED")

#: Coarse host names used by the requirements file and by ``host_os``.
_OS_ALIASES = {
    "win": "windows", "win32": "windows", "win64": "windows", "windows": "windows",
    "nt": "windows", "linux": "linux", "mac": "macos", "macos": "macos",
    "darwin": "macos", "osx": "macos",
}

#: PCI vendor ids, for the Linux sysfs ReBAR probe.
_PCI_VENDORS = {
    "hip": {"0x1002"},
    "cuda": {"0x10de"},
    "sycl": {"0x8086"},
    "vulkan": set(),
    "directx12": set(),
    "metal": set(),
}

#: Manual evidence paths when the host exposes no machine-readable ReBAR state.
_REBAR_MANUAL = {
    "windows": [
        "GPU-Z -> Graphics Card -> 'Resizable BAR' field",
        "vendor control panel (AMD Smart Access Memory / NVIDIA Resizable BAR / Intel ReBAR) toggle state",
        "`nvidia-smi -q` only if the driver prints a 'Resizable BAR' line",
    ],
    "linux": [
        "`lspci -vv | grep -i 'resizable bar'`",
        "`ls /sys/bus/pci/devices/*/resource*_resize` (a file means the capability is exposed)",
    ],
    "macos": ["not applicable: Apple silicon uses unified memory"],
}

_REQUIREMENTS_FILE = "BACKEND_REQUIREMENTS.json"


# --------------------------------------------------------------------------- data

def _bundle_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_requirements() -> dict[str, Any]:
    """Load BACKEND_REQUIREMENTS.json, or {} when it is missing."""
    for p in (_bundle_root() / _REQUIREMENTS_FILE, Path("H:/K_and_S") / _REQUIREMENTS_FILE):
        try:
            if p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {}


def _backends() -> dict[str, Any]:
    return load_requirements().get("backends", {}) or {}


def known_backends() -> list[str]:
    return sorted(_backends())


def resolve_backend(name: str) -> str | None:
    """Canonical backend key for an id, display name or alias; None if unknown."""
    if not name:
        return None
    low = name.strip().lower()
    table = _backends()
    if low in table:
        return low
    for key, spec in table.items():
        aliases = [key, str(spec.get("name", "")).lower()]
        aliases += [str(a).lower() for a in spec.get("aliases", [])]
        if low in aliases:
            return key
    return None


def _host_os() -> str:
    system = platform.system().lower()
    return _OS_ALIASES.get(system, system or "unknown")


def _norm_os(value: str) -> str:
    return _OS_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())


# --------------------------------------------------------------------------- probes

def probe_rebar(target_backend: str, host_os: str = "", *, probe_host: bool = True) -> dict[str, Any]:
    """ReBAR support for the target backend on this host.

    ReBAR is a property of the PCIe topology *and* of the backend API. The
    backend entry says whether the API can use a large BAR and what to do when
    it cannot; the host probe decides PASS/FAIL. When neither sysfs nor the
    driver reports a machine-readable answer the row stays UNVALIDATED: a
    guessed "yes" here is how a plan assumes zero-copy streaming it does not get.
    """
    backend = resolve_backend(target_backend) or target_backend
    spec = (_backends().get(backend, {}) or {}).get("rebar", {}) or {}
    os_name = _norm_os(host_os) if host_os else _host_os()

    out: dict[str, Any] = {
        "id": "pcie.rebar",
        "backend": backend,
        "supported": bool(spec.get("supported")),
        "api": spec.get("api", ""),
        "benefit": spec.get("benefit", ""),
        "fallback": spec.get("fallback", "staging buffer + chunked copy"),
        "status": "UNVALIDATED",
        "confidence": "none",
        "evidence": [],
        "host_os": os_name,
    }

    if not spec:
        out["evidence"].append(f"no ReBAR entry for backend `{backend}` in {_REQUIREMENTS_FILE}")
        return out

    if not out["supported"]:
        out.update(status="NOT RUN", confidence="documented",
                   evidence=[f"not applicable: {out['benefit'] or 'this backend has no discrete BAR window'}"])
        return out

    if not probe_host:
        out["evidence"].append("host probe disabled (probe_host=False)")
        out["manual"] = _REBAR_MANUAL.get(os_name, [])
        return out

    if os_name == "linux":
        res = _probe_rebar_linux(backend)
        if res:
            out.update(res)
            return out

    if os_name in ("linux", "windows"):
        res = _probe_rebar_nvidia_smi()
        if res:
            out.update(res)
            return out

    out["evidence"].append(f"no machine-readable ReBAR source on {os_name}")
    out["manual"] = _REBAR_MANUAL.get(os_name, [])
    return out


def _probe_rebar_linux(backend: str) -> dict[str, Any] | None:
    """Read the resizable-BAR capability from sysfs for the backend's vendor."""
    devs = Path("/sys/bus/pci/devices")
    if not devs.is_dir():
        return None
    try:
        devices = sorted(devs.iterdir())
    except OSError:
        return None
    vendors = _PCI_VENDORS.get(backend, set())
    seen: list[str] = []
    hits: list[str] = []
    for dev in devices:
        try:
            vendor = (dev / "vendor").read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if vendors and vendor not in vendors:
            continue
        if not vendors and vendor not in {"0x1002", "0x10de", "0x8086"}:
            continue
        sizes = sorted(p.name for p in dev.glob("resource*_resize"))
        seen.append(f"{dev.name} ({vendor})")
        if sizes:
            hits.append(f"{dev.name} exposes {', '.join(sizes)}")
    if not seen:
        return None
    if hits:
        return {"status": "PASS", "confidence": "probed",
                "evidence": hits[:5] + [f"devices checked: {', '.join(seen[:5])}"]}
    return {"status": "FAIL", "confidence": "probed",
            "evidence": [f"no resource*_resize file on {', '.join(seen[:5])}",
                         "the BAR is not resizable on this topology"]}


def _probe_rebar_nvidia_smi() -> dict[str, Any] | None:
    """Parse an explicit 'Resizable BAR' line if the installed driver prints one."""
    exe = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if not exe:
        return None
    try:
        proc = subprocess.run([exe, "-q"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"resizable\s+bar\s*:\s*([A-Za-z]+)", proc.stdout or "", re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).lower()
    ok = value in ("yes", "enabled", "supported", "on")
    return {"status": "PASS" if ok else "FAIL", "confidence": "probed",
            "evidence": [f"nvidia-smi reports Resizable BAR: {match.group(1)}"]}


def _probe_tool(req: dict[str, Any]) -> tuple[str, str, list[str]]:
    """(status, confidence, evidence) for a tool row, via PATH and declared env roots."""
    evidence: list[str] = []
    for var in req.get("env", []) or []:
        value = os.environ.get(str(var))
        if value:
            evidence.append(f"{var}={value}")
            if Path(value).exists():
                return "PASS", "probed", evidence + ["declared root exists"]
    for name in req.get("names", []) or []:
        found = shutil.which(str(name))
        if found:
            return "PASS", "probed", evidence + [f"found `{found}`"]
    # Some SDK tools are never on PATH (the DXC in the Windows SDK is the
    # common case); a declared glob avoids reporting a false FAIL.
    for pattern in req.get("paths", []) or []:
        matches = glob.glob(os.path.expandvars(str(pattern)))
        if matches:
            return "PASS", "probed", evidence + [f"found `{matches[0]}`"]
    if evidence:
        return "FAIL", "probed", evidence + ["declared root does not exist"]
    return "FAIL", "probed", [f"none of {', '.join(map(str, req.get('names', []))) or '(no names declared)'} on PATH"]


def _arch_record(arch: str) -> dict[str, Any]:
    try:
        import api_map  # local import: server imports this module before api_map in some paths
        return api_map.arch_info(arch) if arch else {}
    except Exception:
        return {}


def _probe_capability(req: dict[str, Any], target_arch: str,
                      source_arch: str) -> tuple[str, str, list[str]]:
    field = str(req.get("field", ""))
    if not field:
        return "NOT RUN", "none", ["no capability field declared; run the listed probe"]
    info = _arch_record(target_arch)
    if not info:
        return ("UNVALIDATED", "none",
                [f"`{target_arch or '(no target arch)'}` is not in ARCH_MAP.json; probe on the device"])
    if field not in info:
        return "UNVALIDATED", "none", [f"ARCH_MAP.json has no `{field}` for `{target_arch}`"]
    value = info.get(field)
    src = _arch_record(source_arch) if source_arch else {}
    evidence = [f"ARCH_MAP.json {target_arch}.{field} = {value}"]
    if src and field in src:
        evidence.append(f"source `{source_arch}.{field} = {src.get(field)}`")
    if "expect" not in req:
        # Informational flag (e.g. wave size): there is no pass/fail to report.
        return "NOT RUN", "documented", evidence
    expect = req["expect"]
    if value == expect:
        return "PASS", "documented", evidence
    return "FAIL", "documented", evidence


def _probe_requirement(req: dict[str, Any], *, target_arch: str, source_arch: str,
                       host_os: str, backend_os: list[str],
                       probe_host: bool) -> tuple[str, str, list[str]]:
    kind = str(req.get("kind", "")).lower()

    if kind == "tool":
        if not probe_host:
            return "UNVALIDATED", "none", ["host probe disabled (probe_host=False)"]
        return _probe_tool(req)

    if kind == "os":
        wanted = {_norm_os(o) for o in (req.get("host_os") or backend_os or [])}
        if not wanted:
            return "UNVALIDATED", "none", ["no host requirement declared"]
        if host_os in wanted:
            return "PASS", "probed", [f"host is `{host_os}`"]
        return "FAIL", "probed", [f"host is `{host_os}`, backend needs {', '.join(sorted(wanted))}"]

    if kind == "capability":
        return _probe_capability(req, target_arch, source_arch)

    if kind == "artifact":
        return ("NOT RUN", "none",
                [f"requires a build: {req.get('probe', 'inspect the produced artifact')}"])

    if kind == "manual":
        if not probe_host:
            return "NOT RUN", "none", ["manual probe not run"]
        return "NOT RUN", "none", [f"manual: {req.get('probe', 'run the listed probe')}"]

    if kind == "info":
        field = str(req.get("field", ""))
        value = _arch_record(target_arch).get(field) if field else None
        if value is None:
            return "NOT RUN", "none", [f"`{target_arch or '(no target arch)'}.{field}` not in ARCH_MAP.json"]
        return "NOT RUN", "documented", [f"ARCH_MAP.json {target_arch}.{field} = {value}"]

    if kind == "pcie":
        return "UNVALIDATED", "none", ["use probe_rebar()"]

    if kind == "env":
        var = str(req.get("var", ""))
        if os.environ.get(var):
            return "PASS", "probed", [f"{var} is set"]
        return "FAIL", "probed", [f"{var} is not set"]

    return "UNVALIDATED", "none", [f"unknown requirement kind `{kind}`"]


# --------------------------------------------------------------------------- scoring

def pareto(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank by impact x likelihood and cut the vital few at 80% cumulative score."""
    scored = [r for r in rows if float(r.get("score", 0)) > 0]
    scored.sort(key=lambda r: (-float(r["score"]), str(r["id"])))
    total = sum(float(r["score"]) for r in scored) or 1.0
    cumulative = 0.0
    vital: list[str] = []
    for row in scored:
        before = cumulative / total
        cumulative += float(row["score"])
        row["cumulative"] = round(cumulative / total, 3)
        row["vital"] = before < 0.8
        if row["vital"]:
            vital.append(str(row["id"]))
    return {"total": round(total, 2), "vital_few": vital,
            "tail": [str(r["id"]) for r in scored if not r.get("vital")]}


# --------------------------------------------------------------------------- diagnosis

def diagnose(target_backend: str, target_arch: str = "", host_os: str = "",
             findings: list[dict[str, Any]] | None = None, *, probe_host: bool = True,
             source_arch: str = "") -> dict[str, Any]:
    """Requirement model, probes and Pareto ranking for one target backend.

    ``findings`` is the discovery inventory (``path``/``backend``/``kind``). It is
    used for context only: it tells the reader which source backends the port is
    leaving, never to infer that a host requirement is satisfied.
    """
    backend = resolve_backend(target_backend)
    if backend is None:
        return {
            "known": False,
            "requested": target_backend,
            "known_backends": known_backends(),
            "error": (f"`{target_backend}` is not described in {_REQUIREMENTS_FILE}; "
                      "add its requirements before planning against it"),
            "rows": [],
        }

    spec = _backends()[backend]
    os_name = _norm_os(host_os) if host_os and _norm_os(host_os) != "auto" else _host_os()
    backend_os = [str(o) for o in spec.get("host_os", [])]

    rebar = probe_rebar(backend, os_name, probe_host=probe_host)
    rebar_row = {
        "id": rebar["id"], "kind": "pcie", "category": "memory",
        "need": "a BAR window large enough for the streaming working set",
        "probe": rebar.get("api") or "pcie rebar probe",
        "fallback": rebar.get("fallback", ""),
        "source": "BACKEND_REQUIREMENTS.json#rebar",
        "impact": 4, "likelihood": 0.5, "score": 4 * 0.5,
        "status": rebar["status"], "confidence": rebar["confidence"],
        "evidence": rebar["evidence"], "supported": rebar["supported"],
        "rebar": rebar,
    }

    rows: list[dict[str, Any]] = []
    for raw in spec.get("requirements", []) or []:
        req = dict(raw)
        status, confidence, evidence = _probe_requirement(
            req, target_arch=target_arch, source_arch=source_arch,
            host_os=os_name, backend_os=backend_os, probe_host=probe_host,
        )
        impact = float(req.get("impact", 3))
        likelihood = float(req.get("likelihood", 0.5))
        rows.append({
            "id": req.get("id", "?"), "kind": req.get("kind", "?"),
            "category": req.get("category", req.get("kind", "?")),
            "need": req.get("need", ""), "probe": req.get("probe", ""),
            "fallback": req.get("fallback", ""), "source": req.get("source", ""),
            "impact": impact, "likelihood": likelihood,
            "score": round(impact * likelihood, 2),
            "status": status, "confidence": confidence, "evidence": evidence,
        })
    rows.append(rebar_row)
    ranking = pareto(rows)

    counts = {label: 0 for label in LABELS}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    blockers = [r for r in rows if r["status"] == "FAIL" and float(r["impact"]) >= 5]
    fallbacks = [r for r in rows if r["status"] in ("FAIL", "UNVALIDATED")]

    return {
        "known": True,
        "backend": backend,
        "name": spec.get("name", backend),
        "vendor": spec.get("vendor", ""),
        "target_arch": target_arch,
        "source_arch": source_arch,
        "host_os": os_name,
        "probed": bool(probe_host),
        "rebar": rebar,
        "rows": rows,
        "ranking": ranking,
        "counts": counts,
        "blockers": [r["id"] for r in blockers],
        "fallback_rows": [r["id"] for r in fallbacks],
        "source": _source_context(findings),
        "fault_tree": _fault_tree(rows, backend),
    }


def _source_context(findings: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not findings:
        return {"file_count": 0, "by_backend": {}}
    by: dict[str, int] = {}
    for f in findings:
        key = str(f.get("backend", "?"))
        by[key] = by.get(key, 0) + 1
    return {"file_count": len(findings),
            "by_backend": dict(sorted(by.items(), key=lambda kv: -kv[1]))}


def _fault_tree(rows: list[dict[str, Any]], backend: str) -> list[str]:
    """Failure modes for the port, each gated by named requirement rows."""
    by_id = {r["id"]: r for r in rows}

    def ids(*prefixes: str) -> list[str]:
        return [r["id"] for r in rows if r["id"].startswith(prefixes)]

    def bad(keys: list[str]) -> list[str]:
        return [k for k in keys if by_id.get(k, {}).get("status") in ("FAIL", "UNVALIDATED")]

    tool = bad(ids("toolchain.", "sdk.", "host."))
    cap = bad(ids("capability.", "memory.", "sync."))
    art = bad(ids("artifact."))
    rebar = bad(ids("pcie."))
    lines = [f"port to `{backend}` does not build or load"]
    lines.append("  +-- " + (", ".join(tool) if tool else "toolchain/host rows PASS"))
    lines.append("port builds but the artifact is not what was intended")
    lines.append("  +-- " + (", ".join(art) if art else "artifact rows PASS (build first)"))
    lines.append("port builds but results are wrong or a kernel restructured")
    lines.append("  +-- " + (", ".join(cap) if cap else "capability rows PASS"))
    lines.append("port builds and is correct but streams slowly")
    lines.append("  +-- " + (", ".join(rebar) if rebar else "pcie.rebar PASS"))
    return lines


# --------------------------------------------------------------------------- rendering

_MARK = {"PASS": "✅", "FAIL": "❌", "NOT RUN": "⏳", "UNVALIDATED": "⚠️"}


def render(diag: dict[str, Any]) -> str:
    """Markdown readiness report: Pareto, per-requirement probes, ReBAR gate."""
    if not diag.get("known"):
        return (f"## Backend Readiness — `{diag.get('requested', '?')}`\n\n"
                f"**This backend is not described yet.** {diag.get('error', '')}\n\n"
                f"Known backends: {', '.join(diag.get('known_backends', [])) or 'none'}.\n")

    rows = diag["rows"]
    counts = diag["counts"]
    ranking = diag["ranking"]
    rebar = diag["rebar"]
    src = diag["source"]

    L: list[str] = []
    L.append(f"## Backend Readiness — `{diag['backend']}` ({diag.get('name', '')})\n")
    probe_state = "live host probe" if diag["probed"] else "host probe disabled"
    L.append(f"**Target arch:** `{diag['target_arch'] or '(not set)'}` · "
             f"**Host:** `{diag['host_os']}` · **Evidence:** {probe_state} · "
             f"**Source:** {src['file_count']} kernel/shader file(s)"
             + (f" ({', '.join(f'{k} x{v}' for k, v in src['by_backend'].items())})"
                if src["by_backend"] else ""))
    L.append("")
    L.append("**Verdict:** " + " · ".join(f"{_MARK[k]} {k} {counts.get(k, 0)}" for k in LABELS))
    if diag["blockers"]:
        L.append("")
        L.append(f"**Gate:** {len(diag['blockers'])} impact-5 blocker(s) — "
                 f"{', '.join(diag['blockers'])}. Resolve or explicitly waive before "
                 "editing kernels.")
    L.append("")

    L.append("### Pareto — the vital few\n")
    L.append("`score = impact (1-5) x likelihood (0-1)`; the vital few are the rows "
             "that reach 80% of the total. Fix order is this table, not the order "
             "the requirements were written.\n")
    L.append("| # | requirement | kind | I x L | score | cum. | status |")
    L.append("|---|---|---|---|---|---|---|")
    ordered = sorted(rows, key=lambda r: (-float(r["score"]), str(r["id"])))
    for i, r in enumerate(ordered, 1):
        vital = "**" if r.get("vital") else ""
        L.append(f"| {i} | {vital}`{r['id']}`{vital} | {r['kind']} | "
                 f"{r['impact']:g} x {r['likelihood']:g} | {r['score']:g} | "
                 f"{r.get('cumulative', 0):.0%} | {_MARK.get(r['status'], '')} {r['status']} |")
    L.append("")
    if ranking["tail"]:
        L.append(f"Beyond the 80% line: {', '.join(f'`{t}`' for t in ranking['tail'])} "
                 "— real, but not what you fix first.\n")

    L.append("### Requirement detail\n")
    L.append("| requirement | what it needs | probe | status | evidence | fallback |")
    L.append("|---|---|---|---|---|---|")
    for r in ordered:
        ev = "; ".join(r.get("evidence", [])) or "—"
        fb = r.get("fallback") or "—"
        L.append(f"| `{r['id']}` | {r['need']} | {r['probe']} | "
                 f"{_MARK.get(r['status'], '')} {r['status']} | {ev} | {fb} |")
    L.append("")
    L.append("`documented` evidence comes from `ARCH_MAP.json`, not from a run on "
             "the device; `probed` rows were measured on this host.\n")

    L.append("### ReBAR gate — PCIe resizable BAR (pcie-rebar)\n")
    rebar_state = rebar["status"]
    L.append(f"**Status:** {_MARK.get(rebar_state, '')} {rebar_state} · "
             f"**backend support:** {'yes' if rebar.get('supported') else 'no'} · "
             f"**confidence:** {rebar.get('confidence', 'none')}")
    if rebar.get("api"):
        L.append(f"- **API path:** {rebar['api']}")
    if rebar.get("benefit"):
        L.append(f"- **Benefit:** {rebar['benefit']}")
    if rebar.get("fallback"):
        L.append(f"- **Fallback (required unless PASS):** {rebar['fallback']}")
    for e in rebar.get("evidence", []):
        L.append(f"- Evidence: {e}")
    for m in rebar.get("manual", []) or []:
        L.append(f"- Manual probe: {m}")
    if rebar_state != "PASS":
        L.append(f"- **Decision:** use the fallback and record that the port ran "
                 f"without a large BAR; do not describe it as zero-copy.")
    L.append("")

    L.append("### READINESS\n")
    pending = [r for r in ordered if r["status"] != "PASS"]
    if not pending:
        L.append("- READINESS: all requirement rows PASS on this host and arch.")
    for r in pending:
        L.append(f"- READINESS: `{r['id']}` = **{r['status']}** — {r['need']}. "
                 f"probe: {r['probe']}. fallback: {r['fallback'] or 'none declared'}")
    L.append("")

    L.append("### Fault tree\n")
    L.append("```")
    L.extend(diag.get("fault_tree", []))
    L.append("```")
    return "\n".join(L) + "\n"


def slim(diag: dict[str, Any]) -> dict[str, Any]:
    """Compact, JSON-safe shape for a tool response or a plan artifact."""
    if not diag.get("known"):
        return {k: diag.get(k) for k in ("known", "requested", "error", "known_backends")}
    return {
        "known": True,
        "backend": diag["backend"],
        "name": diag["name"],
        "target_arch": diag["target_arch"],
        "source_arch": diag["source_arch"],
        "host_os": diag["host_os"],
        "probed": diag["probed"],
        "counts": diag["counts"],
        "blockers": diag["blockers"],
        "vital_few": diag["ranking"]["vital_few"],
        "rebar": {k: diag["rebar"].get(k) for k in
                  ("status", "supported", "confidence", "api", "fallback", "evidence")},
        "rows": [{k: r.get(k) for k in
                  ("id", "kind", "need", "probe", "status", "confidence", "impact",
                   "likelihood", "score", "vital", "fallback", "evidence")}
                 for r in diag["rows"]],
    }
