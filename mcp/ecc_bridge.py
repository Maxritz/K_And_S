"""
ECC Bridge for K_And_S — OPTIONAL enhancement layer.

Design contract
---------------
K_and_S is the core porting engine and works 100% standalone. ECC (Everything
Claude Code) is treated as an OPTIONAL, opportunistic resource on the host:

  1. If ECC is found (repo checkout, ~\\.claude install, or `ecc` CLI),
     K_and_S may READ agent role guidance and skill references from it to
     enrich its recommendations. Attribution travels with the output.
  2. If ECC is absent, every K_and_S code path works unchanged. Nothing here
     is a hard dependency; nothing is imported from ECC at module load.
  3. ECC assets are read-only. K_and_S never writes into an ECC install.

Availability is probed lazily and cached. Call `get_bridge()` from anywhere;
it never raises.
"""

from __future__ import annotations
import os
import re
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

#: Environment variable a user can set to point at an ECC checkout explicitly.
ECC_ROOT_ENV = "KANDS_ECC_ROOT"

#: Directories that mark a genuine ECC repo checkout (need more than one hit).
_REPO_MARKERS = (
    "agents", "skills", "commands", "rules", "hooks", "scripts",
    ".claude-plugin", "docs",
)

#: Home-relative install locations ECC uses (Claude plugin / manual install).
_HOME_INSTALL_DIRS = (
    Path.home() / ".claude" / "agents",
    Path.home() / ".claude" / "skills",
    Path.home() / ".claude" / "rules" / "ecc",
)

#: Candidate repo locations probed when the env var is unset.
_REPO_CANDIDATES = (
    Path.home() / "ECC",
    Path.home() / "ecc",
    Path("C:\\Users\\rr\\ECC"),          # observed checkout on this host
    Path("\\opt\\ecc"),
)


def _is_ecc_repo(p: Path) -> bool:
    """A directory counts as an ECC checkout when several known markers coexist."""
    if not p.is_dir():
        return False
    hits = sum(1 for m in _REPO_MARKERS if (p / m).is_dir())
    plugin = (p / ".claude-plugin" / "plugin.json").is_file()
    return hits >= 3 or (hits >= 1 and plugin)


def discover_ecc() -> Dict[str, Any]:
    """Probe the host for ECC. Never raises; returns a discovery record."""
    record: Dict[str, Any] = {
        "available": False,
        "source": None,
        "repo_root": None,
        "claude_agents_dir": None,
        "claude_skills_dir": None,
        "cli": None,
        "agents_count": 0,
        "skills_count": 0,
        "probed_at_errors": [],
    }

    # 1. Explicit override
    env_root = os.environ.get(ECC_ROOT_ENV, "")
    if env_root:
        p = Path(env_root)
        if _is_ecc_repo(p):
            record["available"] = True
            record["source"] = f"env:{ECC_ROOT_ENV}"
            record["repo_root"] = str(p)
        else:
            record["probed_at_errors"].append(
                f"{ECC_ROOT_ENV}={env_root} is set but does not look like an ECC checkout"
            )

    # 2. Known repo locations
    if not record["available"]:
        for cand in _REPO_CANDIDATES:
            if _is_ecc_repo(cand):
                record["available"] = True
                record["source"] = "repo"
                record["repo_root"] = str(cand)
                break

    # 3. Claude-style install (plugin or manual) — usable without a repo
    agents_dir = _HOME_INSTALL_DIRS[0]
    skills_dir = _HOME_INSTALL_DIRS[1]
    if agents_dir.is_dir():
        record["claude_agents_dir"] = str(agents_dir)
        record["agents_count"] = len(list(agents_dir.glob("*.md")))
        if not record["available"]:
            record["available"] = record["agents_count"] > 0
            record["source"] = "claude_install"
    if skills_dir.is_dir():
        record["claude_skills_dir"] = str(skills_dir)
        record["skills_count"] = sum(
            1 for d in skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").is_file()
        )
        if not record["available"] and record["skills_count"] > 0:
            record["available"] = True
            record["source"] = "claude_install"

    # 4. Repo-based counts when a repo was found
    if record["repo_root"]:
        root = Path(record["repo_root"])
        agents = root / "agents"
        skills = root / "skills"
        if agents.is_dir():
            record["agents_count"] = max(record["agents_count"], len(list(agents.glob("*.md"))))
        if skills.is_dir():
            record["skills_count"] = max(record["skills_count"], sum(
                1 for d in skills.iterdir() if d.is_dir() and (d / "SKILL.md").is_file()
            ))

    # 5. CLI presence is a weak signal — recorded but not sufficient alone
    cli = shutil.which("ecc") or shutil.which("ecc-universal")
    if cli:
        record["cli"] = cli

    return record


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class ECCBridge:
    """Read-only access to ECC assets when they exist on the host."""

    def __init__(self) -> None:
        self._record: Optional[Dict[str, Any]] = None
        self._agent_cache: Dict[str, str] = {}
        self._skill_cache: Dict[str, Dict[str, str]] = {}

    # -- availability -------------------------------------------------------

    @property
    def record(self) -> Dict[str, Any]:
        if self._record is None:
            self._record = discover_ecc()
        return self._record

    def refresh(self) -> Dict[str, Any]:
        self._record = discover_ecc()
        self._agent_cache.clear()
        self._skill_cache.clear()
        return self._record

    @property
    def available(self) -> bool:
        return bool(self.record.get("available"))

    def status(self) -> Dict[str, Any]:
        rec = self.record
        return {
            "ecc_available": rec.get("available", False),
            "kands_standalone": True,   # always true by design
            "source": rec.get("source"),
            "repo_root": rec.get("repo_root"),
            "agents_count": rec.get("agents_count", 0),
            "skills_count": rec.get("skills_count", 0),
            "cli": rec.get("cli"),
            "usage_policy": (
                "K_and_S runs standalone. When ECC is present it is used as a "
                "read-only reference (agent role guidance, skill lookups); every "
                "recommendation names its source."
            ),
        }

    # -- agents -------------------------------------------------------------

    def list_agents(self) -> List[Dict[str, str]]:
        """Summaries of ECC agent definitions (name + purpose line)."""
        rec = self.record
        agents: List[Dict[str, str]] = []
        roots = []
        if rec.get("claude_agents_dir"):
            roots.append(Path(rec["claude_agents_dir"]))
        if rec.get("repo_root"):
            roots.append(Path(rec["repo_root"]) / "agents")
        for root in roots:
            if not root.is_dir():
                continue
            for md in sorted(root.glob("*.md")):
                name = md.stem
                if name in self._agent_cache:
                    agents.append({"name": name, "purpose": self._agent_cache[name]})
                    continue
                purpose = ""
                try:
                    text = md.read_text(encoding="utf-8", errors="replace")
                    for line in text.splitlines():
                        s = line.strip()
                        if s and not s.startswith(("#", "---", "|", ">", "```")):
                            purpose = s[:160]
                            break
                except OSError:
                    purpose = "(unreadable)"
                self._agent_cache[name] = purpose
                agents.append({"name": name, "purpose": purpose})
        return agents

    def get_agent_guidance(self, role: str) -> Optional[str]:
        """Full instruction text for one ECC agent role, if present."""
        rec = self.record
        candidates = []
        if rec.get("claude_agents_dir"):
            candidates.append(Path(rec["claude_agents_dir"]) / f"{role}.md")
        if rec.get("repo_root"):
            candidates.append(Path(rec["repo_root"]) / "agents" / f"{role}.md")
        for p in candidates:
            if p.is_file():
                try:
                    return p.read_text(encoding="utf-8", errors="replace")[:8000]
                except OSError:
                    return None
        return None

    # -- skills -------------------------------------------------------------

    def list_skills(self) -> List[Dict[str, str]]:
        """Summaries of ECC skills (name + description from SKILL.md frontmatter)."""
        rec = self.record
        skills: List[Dict[str, str]] = []
        roots = []
        if rec.get("claude_skills_dir"):
            roots.append(Path(rec["claude_skills_dir"]))
        if rec.get("repo_root"):
            roots.append(Path(rec["repo_root"]) / "skills")
        desc_re = re.compile(r"^description:\s*(.+)$", re.MULTILINE)
        name_re = re.compile(r"^name:\s*(.+)$", re.MULTILINE)
        for root in roots:
            if not root.is_dir():
                continue
            for d in sorted(root.iterdir()):
                skill_md = d / "SKILL.md"
                if not (d.is_dir() and skill_md.is_file()):
                    continue
                key = str(skill_md)
                if key in self._skill_cache:
                    skills.append(dict(self._skill_cache[key]))
                    continue
                name, desc = d.name, ""
                try:
                    text = skill_md.read_text(encoding="utf-8", errors="replace")[:4000]
                    m = name_re.search(text)
                    if m:
                        name = m.group(1).strip().strip("\"'")
                    m = desc_re.search(text)
                    if m:
                        desc = m.group(1).strip().strip("\"'")[:200]
                except OSError:
                    pass
                entry = {"name": name, "description": desc, "path": str(d)}
                self._skill_cache[key] = entry
                skills.append(entry)
        return skills

    # -- porting integration --------------------------------------------------

    #: Mapping from K_and_S porting concerns to ECC agent roles that cover them.
    _ROLE_MAP = {
        "planning": "planner",
        "review": "code-reviewer",
        "cpp": "cpp-reviewer",
        "build": "cpp-build-resolver",
        "python": "python-reviewer",
        "security": "security-reviewer",
        "architecture": "architect",
        "autonomous_loop": "loop-operator",
        "ml_runtime": "mle-reviewer",
        "pytorch": "pytorch-build-resolver",
    }

    def guidance_for_porting(self, concerns: List[str]) -> Dict[str, Any]:
        """
        Opportunistic enrichment: for each K_and_S concern, pull the matching
        ECC agent's guidance if ECC exists. Returns {} when unavailable.
        Every item carries its source so attribution is explicit.
        """
        if not self.available:
            return {}
        out: Dict[str, Any] = {}
        for concern in concerns:
            role = self._ROLE_MAP.get(concern)
            if not role:
                continue
            text = self.get_agent_guidance(role)
            if text:
                out[concern] = {
                    "role": role,
                    "source": f"ECC agent `{role}` (read-only borrow)",
                    "excerpt": text[:2000],
                }
        return out


# ---------------------------------------------------------------------------
# Module-level singleton — the only entry point other modules should use.
# ---------------------------------------------------------------------------

_bridge: Optional[ECCBridge] = None


def get_bridge() -> ECCBridge:
    global _bridge
    if _bridge is None:
        _bridge = ECCBridge()
    return _bridge
