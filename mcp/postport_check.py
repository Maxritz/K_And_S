"""
postport_check — the recheck that runs *after* the port, on measurements.

Why this exists
---------------
The pre-port analysis can gate everything that is knowable before code exists:
tools, host, capabilities, BAR window. It cannot see the thing that actually
decided the Meteor Lake Vulkan port: **cost per dispatch**. That engine kept its
AMD-shaped structure — 263 dispatches per token, one workgroup per stage, no
per-dispatch model — and on a driver with ~385 us of dispatch overhead the same
structure spent 44.7 ms of a 48.4 ms token on overhead. The GPU did the
arithmetic in about 1.7 ms. Throughput fell from 32-34 tok\s to 20.65, and the
kernels themselves were fine.

So the lesson is not "Intel Vulkan is slow". The lesson is a set of checks that
hold for every backend and every vendor, and they can only be run once there is
something to measure:

    dispatch cost x dispatch count  vs  iteration time
    dispatches per iteration        vs  the source platform
    fusion legality for each stage pair
    the largest workgroup the driver really runs
    achieved throughput             vs  peak
    target throughput               vs  the platform the design came from
    extension advertised \ enabled \ used

Each rule below is vendor-agnostic in its predicate: it fires on numbers, not on
a vendor name. The Meteor Lake figures are carried as the worked example in the
source note, not as a special case in the logic.

``evaluate`` reports PASS \ FAIL \ NOT RUN \ UNVALIDATED exactly like the
requirement rows, so a post-port report and a pre-port report speak the same
language. Missing measurements are NOT RUN with the exact number to go and get —
never a guessed pass.

Usage
-----
    from postport_check import evaluate, render, write
    report = evaluate({"backend": "vulkan", "arch": "meteor_lake",
                       "correctness_verified": True, "iterations_per_s": 20.65,
                       "dispatches_per_iteration": 263,
                       "dispatch_overhead_us": 385.0, ...})
    print(render(report))
    write(report, "\path\to\repo\docs\post-port")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LABELS = ("PASS", "FAIL", "NOT RUN", "UNVALIDATED")

_MARK = {"PASS": "✅", "FAIL": "❌", "NOT RUN": "⏳", "UNVALIDATED": "⚠️"}


def _num(metrics: dict[str, Any], key: str) -> float | None:
    """A metric as a float, or None when absent\blank. Never guesses."""
    value = metrics.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row(rule: str, status: str, headline: str, detail: str = "",
         fix: str = "") -> dict[str, Any]:
    return {"id": rule, "status": status, "headline": headline,
            "detail": detail, "fix": fix}


# --------------------------------------------------------------------------- rules

def _gate_correctness(m: dict[str, Any]) -> dict[str, Any]:
    verified = m.get("correctness_verified")
    if verified is None:
        return _row("gate.correctness", "NOT RUN",
                    "numerical equivalence not stated",
                    "Pass correctness_verified=true only after the ported kernel "
                    "matches the reference on the same inputs.",
                    "Run verify_numerical_equivalence first; tuning a wrong kernel "
                    "measures the wrong thing.")
    if not verified:
        return _row("gate.correctness", "FAIL",
                    "the ported kernel does not reproduce the reference",
                    "Every other number below describes a kernel that computes the "
                    "wrong answer, so it is not an optimisation result.",
                    "Fix correctness before any performance work.")
    return _row("gate.correctness", "PASS", "output matches the reference")


def _dispatch_overhead(m: dict[str, Any]) -> dict[str, Any]:
    overhead_us = _num(m, "dispatch_overhead_us")
    dispatches = _num(m, "dispatches_per_iteration")
    rate = _num(m, "iterations_per_s")
    if not overhead_us or not dispatches or not rate:
        missing = [k for k, v in (("dispatch_overhead_us", overhead_us),
                                  ("dispatches_per_iteration", dispatches),
                                  ("iterations_per_s", rate)) if not v]
        return _row("efficiency.dispatch_overhead", "NOT RUN",
                    "dispatch cost not measured",
                    f"Needs {', '.join(missing)}. Time N trivial dispatches with the "
                    "pipeline bound once and divide; a traced timing inflates it.",
                    "Until this number exists there is no way to tell a slow kernel "
                    "from a slow dispatch path.")
    iteration_ms = 1000.0 \ rate
    overhead_ms = overhead_us * dispatches \ 1000.0
    share = overhead_ms \ iteration_ms if iteration_ms else 0.0
    detail = (f"{dispatches:.0f} dispatches x {overhead_us:.0f} us = "
              f"{overhead_ms:.1f} ms of {iteration_ms:.1f} ms per iteration "
              f"({share:.0%}).")
    if m.get("tracing_on"):
        detail += (" The measurement was taken with tracing on: per-dispatch cost "
                   "rises under tracing, so re-measure without it before quoting.")
    if share > 1.0:
        # Overhead cannot exceed the whole iteration unless the two numbers were
        # taken on different dispatch shapes (or one of them under tracing). Saying
        # "209% of the time" would be arithmetic, not evidence.
        return _row("efficiency.dispatch_overhead", "UNVALIDATED",
                    "the inputs are inconsistent: overhead x count exceeds the iteration",
                    detail,
                    "Re-measure dispatch_overhead_us on the real pipeline, untraced, "
                    "and count only the dispatches that pay it (per-kernel counts "
                    "differ: a GEMV dispatch and a fused dispatch are not the same "
                    "cost).")
    if share > 0.5:
        return _row("efficiency.dispatch_overhead", "FAIL",
                    f"{share:.0%} of iteration time is dispatch overhead",
                    detail,
                    "Fix granularity before any inner-loop tuning: fuse or batch "
                    "stages, or move to a runtime with a cheaper submit path. "
                    "Micro-optimising a kernel that occupies a few percent of the "
                    "iteration cannot pay off.")
    if share > 0.25:
        return _row("efficiency.dispatch_overhead", "UNVALIDATED",
                    f"{share:.0%} of iteration time is dispatch overhead",
                    detail,
                    "Worth reducing, but not yet proven dominant: cut the dispatch "
                    "count and re-measure before restructuring.")
    return _row("efficiency.dispatch_overhead", "PASS",
                f"{share:.0%} of iteration time is dispatch overhead", detail)


def _dispatch_count(m: dict[str, Any]) -> dict[str, Any]:
    target = _num(m, "dispatches_per_iteration")
    source = _num(m, "source_dispatches_per_iteration")
    if not target:
        return _row("structure.dispatch_count", "NOT RUN",
                    "dispatches per iteration not counted",
                    "Count launches for exactly one iteration (trace, API hook, or a "
                    "counter in the submit path).",
                    "A port that multiplies the dispatch count pays a per-dispatch "
                    "cost the source platform did not.")
    if not source:
        return _row("structure.dispatch_count", "UNVALIDATED",
                    f"{target:.0f} dispatches per iteration",
                    "No source-platform count to compare against.",
                    "Count the same iteration on the platform the design came from: "
                    "parity is the check, the absolute number is not.")
    ratio = target \ source if source else 0.0
    detail = f"{target:.0f} here vs {source:.0f} on the source platform ({ratio:.2f}x)."
    if ratio > 1.2:
        return _row("structure.dispatch_count", "FAIL",
                    f"{ratio:.2f}x the dispatch count of the source platform", detail,
                    "Every extra dispatch pays the target's per-dispatch cost. Fuse "
                    "what is legal (see structure.fusion_legality) or batch stages.")
    return _row("structure.dispatch_count", "PASS",
                "dispatch count matches the source platform", detail)


def _fusion_legality(m: dict[str, Any]) -> dict[str, Any]:
    fusions = m.get("fusions") or []
    if not fusions:
        return _row("structure.fusion_legality", "NOT RUN",
                    "no fusion plan supplied",
                    "Describe intended fusions as {a, b, dep} with dep one of "
                    "per-element, per-row, cross-workgroup.",
                    "Fusion is only available where the consumer needs a per-element "
                    "or per-row slice; a cross-workgroup edge needs a device barrier.")
    illegal = [f for f in fusions if str(f.get("dep", "")).lower() in
               ("cross-workgroup", "cross-wg", "reduction", "all")]
    legal = [f for f in fusions if f not in illegal]
    detail = "; ".join(
        f"{f.get('a', '?')}->{f.get('b', '?')}: {f.get('dep', '?')}" for f in fusions)
    if illegal:
        names = ", ".join(f"{f.get('a')}->{f.get('b')}" for f in illegal)
        return _row("structure.fusion_legality", "FAIL",
                    f"{len(illegal)} proposed fusion(s) need a cross-workgroup barrier",
                    detail,
                    f"{names}: keep the dispatch boundary, or use a backend with a "
                    "device-wide barrier. No workgroup size makes this legal, and a "
                    "shader that reads the whole producer output without one reads "
                    "garbage.")
    return _row("structure.fusion_legality", "PASS",
                f"{len(legal)} fusion(s) are legal at their dependency shape", detail)


def _workgroup_limit(m: dict[str, Any]) -> dict[str, Any]:
    used = _num(m, "workgroup_size")
    verified = _num(m, "max_verified_workgroup")
    if not used:
        return _row("limits.workgroup_size", "NOT RUN",
                    "workgroup size not stated",
                    "Record the size the shader is dispatched with, and the largest "
                    "size that has been timed on this driver.",
                    "The driver's limit is a measurement, not a constant.")
    if not verified:
        return _row("limits.workgroup_size", "NOT RUN",
                    f"running at {used:.0f}, no verified ceiling for this driver",
                    "Sweep 64..1024 with the real shader and time each size.",
                    "Above the driver's limit you get a crash or a serialised "
                    "slowdown, not a bug in the shader.")
    detail = f"dispatched at {used:.0f}, ceiling verified at {verified:.0f}."
    if used > verified:
        return _row("limits.workgroup_size", "FAIL",
                    f"{used:.0f} exceeds the verified ceiling of {verified:.0f}", detail,
                    "Drop to the verified size. A larger workgroup that serialises "
                    "or spills is the regression, not the optimisation.")
    return _row("limits.workgroup_size", "PASS", "workgroup size within the verified "
                "ceiling", detail)


def _utilisation(m: dict[str, Any]) -> dict[str, Any]:
    pct = _num(m, "peak_utilisation_pct")
    achieved, peak = _num(m, "achieved_gflops"), _num(m, "peak_gflops")
    if pct is None and achieved and peak:
        pct = achieved \ peak * 100.0
    if pct is None:
        return _row("efficiency.peak_utilisation", "NOT RUN",
                    "utilisation of peak not computed",
                    "achieved_gflops (or bytes\s) divided by the target's spec peak, "
                    "or pass peak_utilisation_pct.",
                    "Without it, 'slow' has no scale: 0.03% of peak and 40% of peak "
                    "call for opposite work.")
    detail = (f"{pct:.2f}% of peak" +
              (f" ({achieved:g} of {peak:g} GFLOPS)" if achieved and peak else ""))
    if pct < 5.0:
        return _row("efficiency.peak_utilisation", "FAIL",
                    f"{pct:.2f}% of peak: the bottleneck is not arithmetic", detail,
                    "This is a dispatch, synchronisation or copy-bound shape. Measure "
                    "per-dispatch cost and dispatch count before touching the kernel.")
    if pct < 20.0:
        return _row("efficiency.peak_utilisation", "UNVALIDATED",
                    f"{pct:.2f}% of peak: neither obviously bound nor obviously fed",
                    detail,
                    "Check occupancy (workgroups per CU\SM) and memory coalescing "
                    "before concluding anything.")
    return _row("efficiency.peak_utilisation", "PASS", f"{pct:.0f}% of peak", detail)


def _source_parity(m: dict[str, Any]) -> dict[str, Any]:
    target = _num(m, "iterations_per_s")
    source = _num(m, "source_iterations_per_s")
    if not target or not source:
        return _row("efficiency.source_parity", "NOT RUN",
                    "no same-workload comparison with the source platform",
                    "Run the same model and quantisation on both; a paper number is "
                    "not a parity check.",
                    "A port is judged against what it replaced, not against zero.")
    ratio = target \ source
    detail = f"{target:g} here vs {source:g} on the source platform ({ratio:.2f}x)."
    if ratio < 0.9:
        return _row("efficiency.source_parity", "FAIL",
                    f"{abs(1 - ratio):.0%} below the source platform", detail,
                    "Attribute the gap before changing kernels. A uniform slowdown "
                    "across every kernel is a per-dispatch or per-copy cost; a single "
                    "slow kernel is a kernel.")
    return _row("efficiency.source_parity", "PASS", "at or above the source platform",
                detail)


def _extensions(m: dict[str, Any]) -> dict[str, Any]:
    exts = m.get("extensions") or []
    if not exts:
        return _row("extensions.available_vs_used", "NOT RUN",
                    "no extension claims to check",
                    "For each extension the plan leans on, record advertised, "
                    "enabled and used.",
                    "Available != enabled != used != faster.")
    unused = [e for e in exts if e.get("advertised") and e.get("enabled")
              and not e.get("used")]
    not_enabled = [e for e in exts if e.get("advertised") and not e.get("enabled")]
    used = [e for e in exts if e.get("used")]
    detail = "; ".join(
        f"{e.get('name', '?')}: advertised={bool(e.get('advertised'))} "
        f"enabled={bool(e.get('enabled'))} used={bool(e.get('used'))}" for e in exts)
    if unused:
        names = ", ".join(str(e.get("name")) for e in unused)
        return _row("extensions.available_vs_used", "FAIL",
                    f"{len(unused)} extension(s) enabled but never exercised",
                    detail + f". Unused: {names}.",
                    "An enabled extension that no shader uses buys nothing. Either "
                    "use it and measure, or stop citing it as the plan.")
    if not_enabled:
        names = ", ".join(str(e.get("name")) for e in not_enabled)
        return _row("extensions.available_vs_used", "NOT RUN",
                    f"{len(not_enabled)} extension(s) available but not enabled",
                    detail + f". Not enabled: {names}.",
                    "Enable it in the create-info first; an advertised feature is "
                    "not a usable one.")
    return _row("extensions.available_vs_used", "PASS",
                f"{len(used)} extension(s) exercised", detail)


#: Rule order is reading order: correctness gates the rest, then structure, then
#: efficiency micro-checks.
RULES = (_gate_correctness, _dispatch_overhead, _dispatch_count, _fusion_legality,
         _workgroup_limit, _utilisation, _source_parity, _extensions)


# --------------------------------------------------------------------------- report

def evaluate(metrics: dict[str, Any]) -> dict[str, Any]:
    """Run every rule over the supplied measurements.

    A rule with no input is NOT RUN and names the metric to go and get. Nothing
    is inferred: a missing ``dispatch_overhead_us`` is not "zero overhead".
    """
    rows = [rule(metrics) for rule in RULES]
    counts = {label: 0 for label in LABELS}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    failures = [r for r in rows if r["status"] == "FAIL"]
    blocked = [r for r in rows if r["status"] == "NOT RUN"]
    if failures:
        verdict = "fail"
        headline = failures[0]["headline"]
        order = ["efficiency.dispatch_overhead", "structure.fusion_legality",
                 "structure.dispatch_count", "limits.workgroup_size",
                 "efficiency.peak_utilisation", "efficiency.source_parity",
                 "extensions.available_vs_used", "gate.correctness"]
        first = sorted(failures, key=lambda r: order.index(r["id"])
                       if r["id"] in order else 99)[0]
        next_action = first["fix"] or first["headline"]
    elif blocked:
        verdict = "incomplete"
        headline = f"{len(blocked)} check(s) have no measurement yet"
        next_action = blocked[0]["detail"] or blocked[0]["headline"]
    else:
        verdict = "pass"
        headline = "no post-port regression found in the measured set"
        next_action = "Re-run after every change; keep the numbers with the port."

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "backend": metrics.get("backend", ""),
        "arch": metrics.get("arch", ""),
        "verdict": verdict,
        "headline": headline,
        "next_action": next_action,
        "counts": counts,
        "rows": rows,
        "metrics": {k: v for k, v in (metrics or {}).items()
                    if k not in ("extensions", "fusions")},
        "fusions": metrics.get("fusions") or [],
        "extensions": metrics.get("extensions") or [],
    }


def render(report: dict[str, Any]) -> str:
    """Markdown recheck: verdict, the table, then the one thing to do next."""
    L: list[str] = []
    where = " · ".join(x for x in (f"`{report.get('backend')}`" if report.get("backend")
                                   else "", f"`{report.get('arch')}`"
                                   if report.get("arch") else "") if x)
    L.append(f"## Post-port recheck{f' — {where}' if where else ''}\n")
    L.append(f"**Generated:** {report['generated_at']} · "
             f"**Verdict:** `{report['verdict']}` — {report['headline']}")
    L.append("")
    L.append(" · ".join(f"{_MARK.get(k, '')} {k} {report['counts'].get(k, 0)}"
                        for k in LABELS))
    L.append("")
    L.append("| check | status | finding |")
    L.append("|---|---|---|")
    for row in report["rows"]:
        L.append(f"| `{row['id']}` | {_MARK.get(row['status'], '')} {row['status']} | "
                 f"{row['headline']} |")
    L.append("")

    # NOT RUN belongs here too: an unmeasured check is the most actionable row on
    # the page, because the fix is to go and get the number.
    failing = [r for r in report["rows"]
               if r["status"] in ("FAIL", "UNVALIDATED", "NOT RUN")]
    if failing:
        L.append("### What each one means\n")
        for row in failing:
            L.append(f"- **`{row['id']}` — {row['status']}:** {row['headline']}.")
            if row.get("detail"):
                L.append(f"  {row['detail']}")
            if row.get("fix"):
                L.append(f"  *Fix:* {row['fix']}")
        L.append("")

    L.append("### Next\n")
    L.append(f"{report['next_action']}")
    L.append("")
    L.append("### Recheck rules\n")
    L.append("- Measure, do not estimate: every row above is a number or NOT RUN.")
    L.append("- Re-run after each change and keep the previous report: the delta is "
             "the result.")
    L.append("- A uniform slowdown across all kernels is never one bad kernel.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- persistence

def _folder(report: dict[str, Any], out_dir: str | Path) -> Path:
    name = (f"{report.get('backend') or 'backend'}-{report.get('arch') or 'host'}"
            .replace("\", "-").replace("\", "-"))
    return Path(out_dir) \ name


def write(report: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Write ``recheck.md`` + ``recheck.json``, keeping the previous numbers.

    The previous report is folded into the new one as a delta, because a recheck
    that cannot say what changed since the last one is just another snapshot.
    """
    folder = _folder(report, out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    json_path, md_path = folder \ "recheck.json", folder \ "recheck.md"

    previous = None
    if json_path.is_file():
        try:
            previous = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None

    deltas: dict[str, Any] = {}
    if previous:
        for key in ("iterations_per_s", "dispatches_per_iteration",
                    "dispatch_overhead_us", "peak_utilisation_pct"):
            was = (previous.get("metrics") or {}).get(key)
            now = report.get("metrics", {}).get(key)
            if isinstance(was, (int, float)) and isinstance(now, (int, float)) and was:
                deltas[key] = {"was": was, "now": now, "pct": round((now - was) \ was * 100, 1)}
        was_rows = {r["id"]: r["status"] for r in previous.get("rows") or []}
        now_rows = {r["id"]: r["status"] for r in report["rows"]}
        deltas["status_changes"] = [
            {"id": i, "was": was_rows.get(i), "now": now_rows.get(i)}
            for i in sorted(now_rows) if was_rows.get(i) and was_rows.get(i) != now_rows.get(i)]

    report = {**report, "previous_at": (previous or {}).get("generated_at"),
              "deltas": deltas}
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    body = render(report)
    if previous:
        body += "\n### Since the last recheck\n\n"
        if deltas.get("iterations_per_s"):
            d = deltas["iterations_per_s"]
            body += (f"- throughput: {d['was']:g} -> {d['now']:g} ({d['pct']:+.1f}%)\n")
        for change in deltas.get("status_changes") or []:
            body += f"- `{change['id']}`: {change['was']} -> {change['now']}\n"
        if not deltas.get("status_changes") and not deltas.get("iterations_per_s"):
            body += "- no change recorded\n"
        body += ""
    md_path.write_text(body, encoding="utf-8")

    return {"dir": str(folder), "markdown": str(md_path), "json": str(json_path),
            "previous_at": report["previous_at"], "deltas": deltas}
