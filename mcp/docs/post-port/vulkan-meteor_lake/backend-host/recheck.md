## Post-port recheck

**Generated:** 2026-10-31 15:07 UTC · **Verdict:** `fail` — 92% of iteration time is dispatch overhead

✅ PASS 3 · ❌ FAIL 4 · ⏳ NOT RUN 1 · ⚠️ UNVALIDATED 0

| check | status | finding |
|---|---|---|
| `gate.correctness` | ✅ PASS | output matches the reference |
| `efficiency.dispatch_overhead` | ❌ FAIL | 92% of iteration time is dispatch overhead |
| `structure.dispatch_count` | ✅ PASS | dispatch count matches the source platform |
| `structure.fusion_legality` | ❌ FAIL | 1 proposed fusion(s) need a cross-workgroup barrier |
| `limits.workgroup_size` | ✅ PASS | workgroup size within the verified ceiling |
| `efficiency.peak_utilisation` | ❌ FAIL | 0.03% of peak: the bottleneck is not arithmetic |
| `efficiency.source_parity` | ❌ FAIL | 37% below the source platform |
| `extensions.available_vs_used` | ⏳ NOT RUN | 1 extension(s) available but not enabled |

### What each one means

- **`efficiency.dispatch_overhead` — FAIL:** 92% of iteration time is dispatch overhead.
  116 dispatches x 385 us = 44.7 ms of 48.4 ms per iteration (92%).
  *Fix:* Fix granularity before any inner-loop tuning: fuse or batch stages, or move to a runtime with a cheaper submit path. Micro-optimising a kernel that occupies a few percent of the iteration cannot pay off.
- **`structure.fusion_legality` — FAIL:** 1 proposed fusion(s) need a cross-workgroup barrier.
  gate->up: per-row; silu->down: cross-workgroup
  *Fix:* silu->down: keep the dispatch boundary, or use a backend with a device-wide barrier. No workgroup size makes this legal, and a shader that reads the whole producer output without one reads garbage.
- **`efficiency.peak_utilisation` — FAIL:** 0.03% of peak: the bottleneck is not arithmetic.
  0.03% of peak
  *Fix:* This is a dispatch, synchronisation or copy-bound shape. Measure per-dispatch cost and dispatch count before touching the kernel.
- **`efficiency.source_parity` — FAIL:** 37% below the source platform.
  20.65 here vs 33 on the source platform (0.63x).
  *Fix:* Attribute the gap before changing kernels. A uniform slowdown across every kernel is a per-dispatch or per-copy cost; a single slow kernel is a kernel.
- **`extensions.available_vs_used` — NOT RUN:** 1 extension(s) available but not enabled.
  VK_KHR_timeline_semaphore: advertised=True enabled=False used=False. Not enabled: VK_KHR_timeline_semaphore.
  *Fix:* Enable it in the create-info first; an advertised feature is not a usable one.

### Next

Fix granularity before any inner-loop tuning: fuse or batch stages, or move to a runtime with a cheaper submit path. Micro-optimising a kernel that occupies a few percent of the iteration cannot pay off.

### Recheck rules

- Measure, do not estimate: every row above is a number or NOT RUN.
- Re-run after each change and keep the previous report: the delta is the result.
- A uniform slowdown across all kernels is never one bad kernel.

### Since the last recheck

- throughput: 20.65 -> 20.65 (+0.0%)
