# mcp\arch_facts.py — arch facts with provenance
from __future__ import annotations
from pathlib import Path

def canon_field(name: str) -> str:
    return name

def normalize_entry(entry: dict) -> tuple[dict, list[dict]]:
    return entry, []

def probe_host_cpu(host_os: str = "") -> tuple[dict, list[str]]:
    return {}, []
