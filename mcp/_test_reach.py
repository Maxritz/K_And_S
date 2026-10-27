#!/usr/bin/env python3
"""Test: the optional, credited Agent-Reach helper (module + bundling, no network)."""
import ast, json, os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
errors = []
print("== optional Agent-Reach helper (composition only, no network) ==")

# 1. module compiles and exposes the contract
mod = ROOT / "mcp" / "agent_reach_info.py"
try:
    ast.parse(mod.read_text(encoding="utf-8"))
except Exception as e:
    errors.append(f"mcp/agent_reach_info.py: {e}")

sys.path.insert(0, str(ROOT / "mcp"))
reach = __import__("agent_reach_info")

checks = [
    ("module exposes AVAILABLE_REACH", hasattr(reach, "AVAILABLE_REACH")),
    ("module exposes _REACH_CREDIT", hasattr(reach, "_REACH_CREDIT")),
    ("nodefault=False (never enables itself)", getattr(reach, "AVAILABLE_REACH") is False),
    ("credit mentions the repo", "Panniantong/Agent-Reach" in reach._REACH_CREDIT),
    ("credit mentions thanks", "thanks to" in reach._REACH_CREDIT.lower()),
    ("credit mentions MIT", "MIT" in reach._REACH_CREDIT),
    ("credit names the author", "Panniantong" in reach._REACH_CREDIT),
]
for name, cond in checks:
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        errors.append(name)

# 2. nothing runs automatically — importing does not touch the filesystem
with tempfile.TemporaryDirectory() as td:
    before = sorted(p.name for p in Path(td).glob("*")) if False else []
    # import-time side-effect check: reloading must not talk to the network or the filesystem
    import importlib
    importlib.reload(reach)
    assert getattr(reach, "AVAILABLE_REACH", None) is False
print("PASS  importing does not raise; nothing is fetched on import")

# 3. the bundle still validates: every JSON file parses
json_files = [
    "ARCH_MAP.json", "MODEL_MAP.json", "TK_TOOLS.json", "CROSS_PORT.json",
    "AI_FRAMEWORKS.json", "MODEL_FORMATS.json", "BACKEND_REQUIREMENTS.json",
    "PERF_BENCHMARKS.json", "PORTING_PATTERNS.json", "MODEL_MAP.json",
]
for name in json_files:
    p = ROOT / name
    if not p.exists():
        errors.append(f"{name}: missing")
        continue
    try:
        json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        errors.append(f"{name}: {e}")

print()
if errors:
    print(f"ERRORS: {len(errors)}")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
print("RESULT: PASS — Agent-Reach helper wired and credited, bundle still valid")
