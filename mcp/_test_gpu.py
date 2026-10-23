"""Self-test for the architecture detection / target suggestion path.

Everything else in K_and_S depends on these two functions: `build_porting_map`,
`api_map` bucket resolution and `autonomous_port` all start from the local arch.
A regression here silently degrades every recommendation, so this runs as a
test with a non-zero exit code rather than a print script.

    python _test_gpu.py        # 0 = pass, 1 = fail
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from server import (  # noqa: E402
    _detect_local_gpu_arch,
    _suggest_target_arch,
    _load_json,
    _ARCH_MAP_PATH,
)

FAILED: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f"  [{detail}]" if detail else ""))
        FAILED.append(label)


def main() -> int:
    print("arch-detection self-test")

    # --- registry must load -------------------------------------------------
    arch_map = _load_json(_ARCH_MAP_PATH)
    entries = arch_map.get("entries", arch_map) if isinstance(arch_map, dict) else {}
    check(isinstance(entries, dict) and len(entries) > 0,
          "ARCH_MAP loads and has entries", f"{type(arch_map).__name__}")
    if not entries:
        print("\n1 check failed (registry empty; remaining checks skipped)")
        return 1

    # --- local detection ----------------------------------------------------
    # Signature takes arch_map; passing None is tolerated by design.
    local = _detect_local_gpu_arch(arch_map)
    check(local is None or local in entries,
          "_detect_local_gpu_arch returns a known arch id or None", repr(local))
    print(f"        local GPU -> {local}")

    # --- target suggestion --------------------------------------------------
    # Every suggestion must be a real registry key, whatever the local machine is.
    sources = ["sm_80", "sm_120a", "gfx1100", "gfx1201", "spir64"]
    bad = []
    for src in sources:
        try:
            tgt = _suggest_target_arch(src, arch_map)
        except Exception as exc:                      # noqa: BLE001
            bad.append((src, f"{type(exc).__name__}: {exc}"))
            continue
        if tgt not in entries:
            bad.append((src, repr(tgt)))
    check(not bad, "_suggest_target_arch returns a known arch for every source", str(bad[:2]))

    for src in sources:
        try:
            print(f"        {src:9s} -> {_suggest_target_arch(src, arch_map)}")
        except Exception:                             # noqa: BLE001
            pass

    # --- cross-vendor sanity ------------------------------------------------
    # With a local GPU present, an NVIDIA source should be steered away from
    # NVIDIA and an AMD source away from AMD. Only assert when detection worked.
    if local:
        local_vendor = str((entries.get(local) or {}).get("vendor", "")).lower()
        if local_vendor:
            for src, src_vendor in (("sm_80", "nvidia"), ("gfx1100", "amd")):
                tgt = _suggest_target_arch(src, arch_map)
                tgt_vendor = str((entries.get(tgt) or {}).get("vendor", "")).lower()
                # Same-vendor suggestion is legitimate only when no other
                # vendor is available locally; flag only the common case.
                if src_vendor != local_vendor:
                    check(tgt_vendor == local_vendor,
                          f"{src} source maps to local {local_vendor} target", tgt)

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
