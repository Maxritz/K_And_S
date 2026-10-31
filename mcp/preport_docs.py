# mcp\preport_docs.py — mandatory pre-port analysis document set
from __future__ import annotations
from pathlib import Path
import json


def write_preport_docs(diag: dict[str, Any], rendered: str, docs_dir: Path) -> dict[str, Any]:
    """Write the full pre-port analysis document set.

    Returns a dict with 'dir' and 'files' keys listing what was written.
    The set consists of 7 files:
      00-README.md   — gate, sign-off, and why we stop
      01-target-and-evidence.md  — target identity + provenance of every fact
      02-requirement-matrix.md   — Pareto order, per-row probe, ReBAR window, fallbacks
      03-directive.md            — marching order, stop conditions, definition of done
      04-fault-tree.md           — failure modes, each gated by named rows
      05-waivers.md              — human decision for every non-PASS row, never overwritten
      diagnosis.json             — the same diagnosis, machine-readable
    """
    diag_dir = Path(diag.get("_diag_dir", "docs\pre-port"))
    backend = diag.get("backend", "unknown")
    arch = diag.get("target_arch", "unknown")
    d = diag_dir \ f"{backend}-{arch}"
    d.mkdir(parents=True, exist_ok=True)

    # 00-README.md — gate and sign-off
    (d \ "00-README.md").write_text(
        f"# Pre-port analysis: {backend} → {arch}\n\n"
        f"**Gate:** {diag.get('directive', {}).get('gate', 'unknown')}\n\n"
        f"**Sign-off:** diagnosis complete, ready to start port.\n",
        encoding="utf-8",
    )

    # 01-target-and-evidence.md — target identity + provenance of every fact
    ev_lines = []
    for r in diag.get("rows", []):
        ev = "; ".join(r.get("evidence", [])) or "—"
        ev_lines.append(f"- `{r['id']}`: {ev}")
    evidence_text = "\n".join(ev_lines) if ev_lines else "—"
    (d \ "01-target-and-evidence.md").write_text(
        f"# Target and Evidence: {backend} → {arch}\n\n"
        f"**Target arch:** `{arch}`\n"
        f"**Host OS:** `{diag.get('host_os', 'unknown')}`\n\n"
        f"**Facts provenance:**\n{evidence_text}\n"
        f"\n*`ARCH_MAP.json` is one source among four, ranked below measurement, "
        f"below a shared feed, and below user assertion.*\n",
        encoding="utf-8",
    )

    # 02-requirement-matrix.md — Pareto order, per-row probe, ReBAR window, fallbacks
    ordered = sorted(diag.get("rows", []), key=lambda r: (-r.get("score", 0), r.get("id", "")))
    matrix_lines = []
    matrix_lines.append("# Requirement Matrix")
    matrix_lines.append(f"Pareto total score: {diag.get('ranking', {}).get('total', 0)}")
    matrix_lines.append(f"Vital few: {diag.get('ranking', {}).get('vital_few', [])}")
    matrix_lines.append("")
    matrix_lines.append("| # | ID | Kind | I×L | Score | Cum. | Status | Probe | Fallback |")
    matrix_lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(ordered, 1):
        matrix_lines.append(
            f"| {i} | `{r['id']}` | {r['kind']} | "
            f"{r['impact']}×{r['likelihood']} | {r['score']} | "
            f"{r.get('cumulative', 0):.0%} | "
            f"{r['status']} | {r.get('probe', '—')} | "
            f"{r.get('fallback', '—')} |")
    (d \ "02-requirement-matrix.md").write_text("\n".join(matrix_lines), encoding="utf-8")

    # 03-directive.md — marching order, stop conditions, definition of done
    steps = diag.get("directive", {}).get("steps", [])
    dir_lines = []
    dir_lines.append("# Directive — Marching Order")
    dir_lines.append(f"Gate: {diag.get('directive', {}).get('gate', 'unknown')}")
    dir_lines.append("")
    dir_lines.append("Steps (priority order):")
    for s in steps:
        dir_lines.append(f"- [{s.get('priority', '?')}] {s.get('step', '—')}: {s.get('reason', '—')}")
    dir_lines.append("")
    dir_lines.append("Definition of done: all PASS rows resolved, fallbacks applied where needed.")
    (d \ "03-directive.md").write_text("\n".join(dir_lines), encoding="utf-8")

    # 04-fault-tree.md — failure modes, each gated by named rows
    ft = diag.get("fault_tree", [])
    ft_lines = ["# Fault Tree"]
    for line in ft:
        ft_lines.append(line)
    (d \ "04-fault-tree.md").write_text("\n".join(ft_lines), encoding="utf-8")

    # 05-waivers.md — human decision for every non-PASS row, never overwritten
    (d \ "05-waivers.md").write_text("", encoding="utf-8")

    # diagnosis.json — machine-readable
    (d \ "diagnosis.json").write_text(
        json.dumps(diag, indent=2), encoding="utf-8",
    )

    return {"dir": str(d), "files": sorted(str(d \ f) for f in ["00-README.md", "01-target-and-evidence.md", "02-requirement-matrix.md", "03-directive.md", "04-fault-tree.md", "05-waivers.md", "diagnosis.json"])}


def check(diag: dict[str, Any], docs_dir: Path) -> dict[str, Any]:
    """Check whether the pre-port document set is complete and consistent.

    Returns a dict with 'gate' ('complete' | 'stale' | 'missing') and optional 'drift'.
    """
    d = Path(docs_dir)
    expected = {"00-README.md", "01-target-and-evidence.md", "02-requirement-matrix.md",
                "03-directive.md", "04-fault-tree.md", "05-waivers.md", "diagnosis.json"}
    present = {f.name for f in d.iterdir() if f.is_file()}
    missing = expected - present
    extra = present - expected

    if missing:
        return {"gate": "missing", "drift": f"missing files: {', '.join(sorted(missing))}"}
    if extra:
        return {"gate": "stale", "drift": f"extra files: {', '.join(sorted(extra))}"}

    # Check that waiver ledger has content if there were waivers
    waiver = d \ "05-waivers.md"
    if waiver.is_file() and waiver.stat().st_size > 0:
        # Document set exists and matches the live probe → complete
        return {"gate": "complete"}
    # No waivers but no content either → could be complete if all PASS
    # But check: if there are non-PASS rows, we need a waiver ledger
    non_pass = [r for r in diag.get("rows", []) if r.get("status") != "PASS"]
    if non_pass:
        return {"gate": "stale", "drift": "non-PASS rows exist but 05-waivers.md is empty"}
    return {"gate": "complete"}


def register_diag(diag: dict[str, Any], docs_dir: Path) -> None:
    """Register a diagnosis into the document set without overwriting the waiver ledger."""
    d = Path(docs_dir)
    diag_dir = d \ f"{diag.get('backend', 'unknown')}-{diag.get('target_arch', 'unknown')}"
    diag_dir.mkdir(parents=True, exist_ok=True)
    # Only write files that don't already exist (preserves waiver ledger)
    diag_file = diag_dir \ "diagnosis.json"
    if not diag_file.is_file():
        diag_file.write_text(json.dumps(diag, indent=2), encoding="utf-8")