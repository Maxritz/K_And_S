# K_and_S Goal Template — Ultimate Goal + Guardrails

> **Date:** `{{CURRENT_DATE}}`
> **Goal ID:** `{{GOAL_ID}}`

---

## Ultimate Goal

> _One sentence: what do you want the final result to be?_

**Example:** "Port llama.cpp's flash-attention kernel to run on AMD gfx1201 (RDNA4) using DP4a via Vulkan compute, achieving ≥25 tok/s at 4096-dim, with parity against the CPU AVX-512 oracle."

**Your goal:**

```
{{ULTIMATE_GOAL}}
```

---

## Measurable Success Criteria

Define what "done" looks like. Each criterion must be measurable.

| Criterion | Target | Measurement Method |
|---|---|---|
| Performance | `{{TARGET_TOK_PER_S}}` tok/s | `--perf-tracing` output |
| Correctness | Match CPU oracle | `ctest -R parity` passes |
| Numerical accuracy | < `{{TOLERANCE}}` max diff | `parity_fp16` test |
| Edge cases | All pass | `ctest -R edge` |
| Build | Compiles on target | `cmake --build` succeeds |

---

## Guardrails (what NOT to do)

These prevent the agent from going off on tangents.

### Code scope
- [ ] **ONLY modify:** `{{FILES_IN_SCOPE}}` (e.g., `src/kernels/*.cu`, `src/kernels/*.comp`)
- [ ] **DO NOT touch:** `{{FILES_OUT_OF_SCOPE}}` (e.g., tokenizer, Python scripts, config files)
- [ ] **No rewrites unless:** Required by arch incompatibility (document why)

### Architecture constraints
- [ ] **Target backend:** `{{TARGET_BACKEND}}` (e.g., Vulkan/HIP/CUDA)
- [ ] **Target arch:** `{{TARGET_ARCH}}` (e.g., gfx1201)
- [ ] **Source arch:** `{{SOURCE_ARCH}}` (e.g., sm_80)
- [ ] **Memory model:** Use explicit allocation (no managed memory) unless arch supports it
- [ ] **Vulkan note:** Use DP4a, NOT CooperativeMatrix (broken on AMD)

### Research boundaries
- [ ] **Search scope:** Only search ArXiv/GitHub for GPU kernel optimization papers/repos
- [ ] **Date filtering:** Only results updated after `{{CUTOFF_DATE}}`
- [ ] **No personal data:** Never store paths, emails, or tokens in output files

### Performance boundaries
- [ ] **Minimum viable:** Get it working at {{MIN_PERF}} tok/s first (loose mode)
- [ ] **Optimization target:** {{TARGET_TOK_PER_S}} tok/s (strict mode)
- [ ] **Budget:** Max {{MAX_ITERATIONS}} optimization iterations

### Verification boundaries
- [ ] **CPU oracle:** Every GPU change must pass parity test against CPU reference
- [ ] **No silent fallbacks:** If falling back to F32, document why
- [ ] **FMA contraction:** Control explicitly (`--fmad=false` or `-ffp-contract=off`)

---

## Workflow Phases

1. **Clone** — `git clone {{GITHUB_URL}} && cd $(basename {{GITHUB_URL}})`
2. **Discover** — `discover_kernels(repo=".")` → find all GPU kernels
3. **Analyze** — `check_arch_compatibility(...)` + `compare_archs(...)` → find gaps
4. **Research** — `search_references(query="...")` → find papers/code
5. **Plan** — `autonomous_port(...)` → full porting plan with Copy/Adapt/Rewrite per kernel
6. **Implement** — Make changes (guided by guardrails)
7. **Verify** — CPU oracle parity + numerical accuracy
8. **Optimize** — sherlock-it loop (trace → dissect → hypothesis → experiment → verify)
9. **Record** — `record_performance(...)` → track progress
10. **Export** — `export_contribution(...)` → share learnings

---

## Current Status

| Phase | Status | Date |
|---|---|---|
| 1. Clone | {{STATUS_CLONE}} | |
| 2. Discover | {{STATUS_DISCOVER}} | |
| 3. Analyze | {{STATUS_ANALYZE}} | |
| 4. Research | {{STATUS_RESEARCH}} | |
| 5. Plan | {{STATUS_PLAN}} | |
| 6. Implement | {{STATUS_IMPLEMENT}} | |
| 7. Verify | {{STATUS_VERIFY}} | |
| 8. Optimize | {{STATUS_OPTIMIZE}} | |
| 9. Record | {{STATUS_RECORD}} | |
| 10. Export | {{STATUS_EXPORT}} | |

---

## Tool Commands Cheat Sheet

```bash
# Set your goal (creates this template in kands://goal)
set_goal(goal="...", guardrails=["..."], target_arch="gfx1201")

# Clone + auto-analyze a repo
autonomous_port(github_url="https://github.com/ggerganov/llama.cpp", target_arch="gfx1201")

# Check if an action is within guardrails
check_guardrail(action="modify src/kernels/attention.cu", reason="adding DP4a fallback")

# Record a measurement
record_performance(arch="gfx1201", workload="flash_attn", tok_per_s=18.5)

# Check progress vs goal
compare_to_target(arch="gfx1201", workload="flash_attn", target_tok_per_s=25)
```
