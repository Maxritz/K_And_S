# Diagnostics & Verification Framework

> **Date context:** This bundle was last updated `2026-10-25`.
> ArXiv/GitHub reference searches use the **current system date** for date filtering.
> The MCP server exposes `kands://status` with live date/time for all tools and agents.

---

## Iterative Performance Optimisation Loop

> **"I need XYZ optimized. Where do I start?"**

This is the concrete loop: porter declares a bottleneck → sherlock-it traces it → dissect the detail → experiment → verify → repeat.

### The Loop

```
Declare (pain point) → Trace (sherlock-it) → Dissect (breakdown) → Hypothesize → Experiment (1 var) → Verify (CPU oracle + measure) → Repeat
```

### Step-by-step

#### 1. Declare — "What needs optimization?"

The porter identifies the slowest or most costly component:

| Pain point example | Typical bottleneck |
|---|---|
| "flash attention is 95% of GPU time" | attention kernel → check WMMA/tensor cores vs F32 fallback |
| "Q4_0 dequant eats 50% of decode" | dequant kernel → check load patterns, shared memory usage |
| "MoE router takes 8ms per layer" | router_topk kernel → check dot4 instruction efficiency |
| "KV cache update is 3ms/tok" | kv_q8 kernel → check memcopy vs in-place update |
| "bf16 GEMV slow on AMD" | bf16_gemv → check if using native bf16 or emulation |

**How to declare:** Run `--perf-tracing` on the target engine. Read the summary line:
```
PERF_TRACING component=engine phase=summary ops=12 gpu_ms=207.9 stages: attention=198.4(95%), projections=6.2(3%), ...
```
The highest `%dev` stage is your declared target.

#### 2. Trace — sherlock-it `trace` mode

Run sherlock-it on the declared component:

```bash
sherlock-it trace --component attention --repo <repo_path> --arch gfx1201
```

Output (from sherlock-it):
```
     component           ops     %dev    dev us     idle us     host us
     attention             24  95.4%   535880.80      377.62       49.20
     attention-qkv          24  40.2%    215400        0         12.00
     attention-score         24  35.1%    187720        0         10.00
     attention-dropout        1   0.0%      0.48      25.44        2.40
```

This tells you **which sub-component** of attention is the real bottleneck.

#### 3. Dissect — break down the sub-component

Use the diagnostic traps to break down further:

| Trap to use | Questions to ask |
|---|---|
| **Dispatch trap** | How many kernel launches per stage? Grid/workgroup size optimal? |
| **Memory trap** | Bytes read/written per op? Are we memory-bound? Shared memory utilization? |
| **Instruction trap** | What ISA is generated? Are tensor cores being used? Scalar fallback? |
| **Sync trap** | Are there barrier stalls or unnecessary synchronizations? |
| **Transfer trap** | Host-device copies per iteration? Staging buffer usage? |

**Ponytail-diag** helps here: when you see a mismatch between CPU oracle and GPU, use `ponytail-diag` to trace the call graph and find the root cause:

```
ROOT CAUSE: attention_flash.cu:118 — WMMA path produces NaN when seq_len mod 8 ≠ 0 | FIX: add alignment padding or F32 fallback for remainder | PROBE: 118 assert(seq_len % 8 == 0)
```

#### 4. Hypothesize — rank candidate root causes

Generate hypotheses (max 4), each with:
- A **claim** (what I think is happening)
- **Evidence for** (data that supports it)
- **Evidence against** (data that contradicts it)
- A **test** (experiment to confirm or disprove)

Example hypotheses for "attention is slow":
```
H1: WMMA instructions not being used — GPU falls back to F32 scalar | For: ISA dump shows `fadd` not `mma` | Against: arch_map says wmma=true | Test: check PTX/SASS
H2: Thread divergence in QKV projection | For: warp execution efficiency = 32% (low) | Against: memory bound, not compute | Test: profile warp execution
H3: KV cache not in shared memory (global loads dominate) | For: memory trap shows 90% bandwidth used | Against: LDS is 64KiB (sufficient) | Test: add shared mem prefetch
```

#### 5. Experiment — one variable at a time

Pick the highest-impact, lowest-cost hypothesis. Change ONE thing:

```bash
# Before (baseline)
cmake --build build-gfx1030 && ./engine --perf-tracing  # → attention: 535ms

# Change: add shared memory prefetch to attention_flash.cu
# Rebuild
cmake --build build-gfx1030 && ./engine --perf-tracing  # → attention: 420ms

# Diff
diff_perf_traces(before_trace, after_trace)  # → 21.4% improvement, PASS
```

Use `diff_perf_traces` to verify the improvement is real and not noise.

#### 6. Verify

- **CPU oracle check**: Run `ctest -R parity` to ensure the change didn't break correctness
- **Measurement**: Use 3+ runs, discard outliers, report min/avg/max
- **Regression check**: Ensure other components didn't get slower

#### 7. Repeat

If the target is still too slow, go back to **Step 2 (Trace)** with the new perf data.
The loop converges when either:
- The component is fast enough (meets performance budget)
- No more single-change improvements remain (needs algorithmic rethink)
- The bottleneck moves to a different component (target has shifted)

---

### Example: Full Loop for "Q4_0 dequant is slow"

```
Declare:  dequant = 50% of decode time
Trace:    sherlock-it trace dequant_q4.cu
Dissect:  Memory trap → 80% bandwidth, 20% compute → MEMORY-BOUND
          Instruction trap → scalar `and` ops dominate, no vectorisation
          Dispatch trap → 32 small launches per token (launch overhead = 2ms)
H1:       Launch overhead dominates (32 launches × 2ms = 64ms/token) | Test: batch launches
H2:       Scalar load not vectorised (scalar `and` vs `v_and` packed) | Test: use v_and_b32x4
Experiment: Batch 8 dequant launches into 1 → 64ms → 8ms (87% improvement)
Verify:   ctest -R parity_q4 → PASS | diff_perf_traces → 87% faster, no regression
Repeat:   Now attention is 60% → loop back to Declare
```

---

## CPU-as-Oracle Strategy

**A correct CPU implementation is the gold-standard oracle** for verifying GPU ports.
This is called **differential testing** — run the same input through both CPU and GPU
implementations and compare outputs bit-for-bit (or within tolerance for FP).

### How it works

1. **Write or use an existing CPU reference** — typically the engine's CPU kernel path
   (AVX-512, AVX2, portable C++). This is the "golden" implementation.

2. **Feed identical inputs** — same weights, same prompt, same KV cache state.
   Seed RNG identically across both paths.

3. **Compare outputs** — element-wise comparison with a tolerance for floating-point:
   ```python
   # Example comparison harness
   max_diff = np.max(np.abs(cpu_output - gpu_output))
   assert max_diff < 1e-3, f"Mismatch: max diff {max_diff}"
   ```

4. **For quantized formats** — the CPU quantized path (e.g., `q2_avx2.cpp`,
   `iq_avx512.cpp`) should produce results within tolerance of the GPU path.
   This is the **i-quant parity test**: run the ggml reference CPU decoder on one
   side, the GPU EXL3/dot4 kernel on the other.

### CPU oracle checklist

- [ ] **Float32 baseline** — `fp32` CPU path is the reference for all FP16/BF16 paths
- [ ] **Quantized parity** — `q4_0/q8_0/iq4_nl` CPU kernels must match GPU within tolerance
- [ ] **EXL3 parity** — CPU EXL3 fallback matches GPU EXL3 within 1e-4
- [ ] **Edge cases** — null experts, boundary tokens, empty KV cache
- [ ] **Deterministic mode** — `SEED=42`, `NTOK=12` for reproducible comparison

### Tools that use CPU-as-oracle

- `kands://validation/test_suites/` — standard test cases with CPU oracle references
- `PORTING_METHOD.md` Phase 7 — "Run `ctest -R _parity`" executes the parity harness
- `DIAGNOSTICS.md` — ponytail-diag for debugging parity failures

---

## Ponytail-Diag: Failure Diagnosis Method

Diagnose failures by building an internal model of the code (call graph,
control-flow branches, data-state transitions) and emitting only the verdict.

### Default (minimal output)

```
ROOT CAUSE: <file>:<line> — <one-line cause> | FIX: <one-line direction> | PROBE: <line> <expr>
```

Example:
```
ROOT CAUSE: shader.comp:42 — shared memory index overflow (array of 64, index 67) | FIX: clamp thread index or increase shared array size | PROBE: 42 assert(idx < 64)
```

### Deep Dive Format (trigger: "deep dive" or "show work")

```
FLOW: <entry> → fn_A[br1:F] → fn_B[br2:T] → crash
DATA: var_x=100 → fn_A → nil → fn_B  ← nil appears without reassignment
BRANCHES: br1: fn_A:27 if len(fields)<3 [taken=false] — static-resolvable
TRUTH TABLES:
  flA=F flB=T → path-D  ← (failing)
  flA=T flB=F → path-E  <untested>
  flA=T flB=T → path-F  <unreachable>
H1: <claim>  For:<ev> Against:<ev> Test:<exp> Cost:low Status:unknown
H2: <claim>  For:<ev> Against:<ev> Test:<exp> Cost:low Status:unknown
H3: <claim>  For:<ev> Against:<ev> Test:<exp> Cost:low Status:unknown
H4: <claim>  For:<ev> Against:<ev> Test:<exp> Cost:low Status:unknown
FISHBONE: 3 subsystems — Method:[], Machine:[], Material:[], Man:[]
5 WHYS: Q1: why? A: because. Q2: ... Q3: ...
BARRIER: <what should have stopped this> Existence:Y/N Bypass:Y/N
FTA: top-event → AND(gate1, gate2) OR(basic-event)
CHANGE: last-diff ≈ 7d ago — commit abc123 — fn_B:12 changed
TODO:<file>:<line>: [sev:P0|P1|P2] <issue> | fix: <candidate> | test: <repro>
```

### Diagnostic Traps (from sherlock-it)

| Trap | Captures |
|---|---|
| Dispatch trap | Launch count, grid size, workgroup dimensions, launch latency |
| Memory trap | Bytes read/written, alignment, bandwidth, cache misses |
| Instruction trap | Generated ISA, vectorisation, instruction mix, fallback paths |
| Synchronisation trap | Barriers, fences, queue waits, idle gaps |
| Allocation trap | Allocation frequency, memory pressure, temporary buffers |
| Transfer trap | Host-device copies, staging, transfer latency |
| Dependency trap | Serial execution, critical path, hidden synchronisation |
| Accuracy trap | Output differences, numerical error, precision-related performance |
| Regression trap | Before/after timings, workload sensitivity and performance variance |

---

## Ponytail-Diag, Forward: Porting Discovery

The diagnostic method is not only for failures. Run it *before* the port, with
"the port is not finished" as the top event, and it produces the requirement
model instead of a root cause.

| Diag element | Forward use in discovery |
|---|---|
| Data/state model | What the source depends on (runtime API, PTX, intrinsics, features) and what the target provides |
| Control/data flow | Build → artifact → load → launch → read; the requirement that gates each edge |
| Branches / truth tables | Requirement present/absent x fallback available/not: which combination ships |
| Hypotheses (H1-H4) | Candidate explanations for a missing capability, each with the probe that discriminates |
| Fishbone | Toolchain, host/OS, capability, memory/sync, artifact, PCIe topology |
| Barrier analysis | What should have stopped an unsupported plan (the gate row) |
| FTA | Port does not build / artifact is wrong / results differ / streams slowly |
| TODO with P0/P1/P2 | `READINESS:` rows below the Pareto line, ordered by score |

The verdict vocabulary is the evidence vocabulary from the quality gates:
**PASS**, **FAIL**, **NOT RUN**, **UNVALIDATED**. A requirement that cannot be
probed is UNVALIDATED, never assumed present.

```
READINESS: toolchain.hipcc = FAIL — no hipcc for gfx1201. probe: hipcc --version. fallback: build on a Linux ROCm host
READINESS: capability.matrix = PASS — ARCH_MAP.json gfx1201.wmma = True
READINESS: pcie.rebar = UNVALIDATED — no machine-readable source on windows. fallback: staging ring + chunked copy
```

A row whose status is not PASS either selects its declared fallback or stops the
port. For PCIe ReBAR the fallback is always available (staging buffer plus
chunked copy, hot tensors resident), so the port can proceed — but the plan
records that it ran without a large BAR, and no claim of zero-copy streaming is
allowed without a PASS.

Run it with `discover_kernels(repo=..., target_backend="hip", target_arch="gfx1201")`,
or let `autonomous_port` include it in Phase 0. The data lives in
`BACKEND_REQUIREMENTS.json`; the code is `mcp/backend_profile.py`.

---

## Sherlocking: Performance Investigation Method

1. **Baseline** — Build & run unmodified target. Record hardware, driver, compiler, flags, workload dimensions, throughput, latency, correctness.
2. **Decompose** — Map execution flow. Enumerate every kernel, dispatch, transfer, sync point. Record counts, dimensions, bytes moved.
3. **Instrument** — Add diagnostic traps for latency, memory, ISA, sync. Compare instrumented vs uninstrumented.
4. **Investigate** — Rank bottlenecks by cumulative cost. Compare against hardware ceilings. Form testable hypotheses.
5. **Optimise** — One variable at a time. Preserve known-good. Avoid speculative rewrites.
6. **Verify** — Rebuilt & rerun identical workloads. Compare raw per-kernel measurements. Reject regressions.

### Sample sherlock trace format

```
     component           ops   %dev      dev us     idle us     host us
     attention            24  95.4%   535880.80      377.62       49.20
     projections         187   3.2%    17723.32     2026.19      460.80
     recurrent            90   1.1%     6127.59     1170.02      143.80
     attention-mix        72   0.2%     1335.28     1102.27      111.60
     head+sample           1   0.0%       98.20       91.80       56.90
     bias                 18   0.0%       90.03      265.52       66.60
     transfer              1   0.0%       79.80     1087.10     1084.30
     norms                67   0.0%       72.98      784.24      139.40
     residual             48   0.0%       23.40      198.44       73.30
     ffn-activate         42   0.0%       21.52      372.84       67.70
     embed                 1   0.0%        0.48       25.44        2.40
   instrumentation floor, 256 empty ops timed through the same begin/end path: 0.70 us host, 19.54 us device each.
```

### Enforced Rules

- Never assume a kernel is optimal because it uses a specialised instruction.
- Never equate aggregate device time with individual invocation latency.
- Never confuse theoretical peak with achievable workload performance.
- Never accept optimisation based on a single noisy measurement.
- Never claim a root cause without evidence.
- Never stop at the first bottleneck if another significant one remains.
- Never discard the original baseline or correctness tests.
- Always redirect toward the highest-impact measurable performance gap.

---

## Validation Suite Reference

| Test name | Type | CPU oracle | GPU path | Tolerance |
|---|---|---|---|---|
| `parity_fp32` | Correctness | AVX-512 fp32 | CUDA FP32 | bit-identical |
| `parity_bf16` | Correctness | AVX512_BF16 | CUDA BF16 (mma) | 1e-3 |
| `parity_fp16` | Correctness | AVX512_FP16 | CUDA FP16 (mma) | 1e-3 |
| `parity_q4_0` | Quant | q4_avx2.cpp | GPU q4_0 | 1e-4 |
| `parity_q8_0` | Quant | q8_avx512 | GPU q8_0 | 1e-5 |
| `parity_exl3` | Quant | exl3_cpu.cpp | GPU EXL3 | 1e-4 |
| `parity_wmma` | Math | scalar FMA fallback | WMMA/MMA | 1e-5 |
| `parity_attention` | End-to-end | CPU attention | GPU flash_attn | 1e-3 |
| `parity_moe` | Routing | CPU router | GPU dot4 router | bit-exact (integer indices) |
| `parity_kvcache` | Memory | CPU KV update | GPU KV update | bit-identical |

### Running validation

```bash
# CPU oracle build (reference)
mkdir build-cpu && cmake -S . -B build-cpu -DENABLE_CUDA=OFF -DENABLE_HIP=OFF
cmake --build build-cpu -j$(nproc)
ctest --test-dir build-cpu -R parity -j$(nproc)

# GPU port build (target)
mkdir build-gpu && cmake -S . -B build-gpu -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build build-gpu -j$(nproc)

# Run differential test
./build-gpu/test_parity --compare-cpu ./build-cpu/test_parity --tol 1e-4
```

---

## Secure Autonomous Operations

> **Security for `autonomous_port(github_url=...)`**

The `autonomous_port` tool can clone GitHub repositories for analysis. The
following security measures are built into the MCP server:

### Built-in safeguards

| Safeguard | Mechanism | Default |
|---|---|---|
| URL allowlist | Only `https://github.com/` / `git+https://...` accepted | enforced |
| File count limit | Scans truncated at `max_files` | 1000 |
| Timeout | Clone + scan timeout | 60 seconds |
| No script execution | Build scripts, git hooks, cmake — never executed | guaranteed |
| Read-only | Cloned repo is not modified | guaranteed |

### Recommended operational security

1. **Clone into a sandbox directory** — e.g., `/tmp/kands_sandbox_<repo_name>/`
2. **Run in a container** — e.g., `docker run --rm -v $PWD:/workspace ...`
3. **Pre-scan for vulnerabilities** — run `npm audit` / `pip-audit` / `cargo audit` on deps
4. **Never run build scripts from untrusted repos** — the bundle won't do this, but your build system might
5. **Log the clone source** — record repo URL + commit hash (sanitized) in `mcp/state/analysis/`

### Risk levels

| Risk | Example | Action |
|---|---|---|
| **Low** | Well-known public repos (e.g., `ggerganov/llama.cpp`) | Standard scan is safe |
| **Medium** | New forks, unmaintained repos | Sandbox + vulnerability scan |
| **High** | Unknown repos, private forks | Manual review required before any execution |

---

## Numerical Equivalence Verification

> **Using the `verify_numerical_equivalence` tool**

This tool checks that ported GPU kernel output matches the CPU reference (oracle)
within tolerance. It detects FP contraction, reduction order, and transcendental
function differences.

### Usage

```
# After porting, extract a few output values from CPU oracle and GPU implementation
verify_numerical_equivalence(
    arch="gfx1201",
    workload="flash_attn",
    source_values=[0.123, -0.456, 0.789, ...],   # CPU reference output
    target_values=[0.122, -0.455, 0.788, ...],   # GPU ported output
    tolerance=1e-5,
    relative_tolerance=1e-4
)
```

### What it checks

1. **Array length match** — divergent lengths indicate wrong workgroup/shared memory config
2. **Max absolute error** — detects FP contraction differences
3. **Max relative error** — catches scale-dependent divergence
4. **Per-value divergence analysis** — shows exactly which indices differ

### Architecture-specific diagnostic suggestions

The tool automatically adds arch-specific hints:

- **AMD**: Check `-ffp-contract=off`, FMA contraction differences
- **NVIDIA**: Check `--fmad=false`, `@fast_math` vs `-ffast-math`
- **Intel**: Check `-fp-model precise` vs fast, DPAS rounding modes
- **Vulkan/Metal**: Check gl SL `layout(invocation` vs spirv optimization passes

### Common causes of divergence

| Cause | Symptom | Fix |
|---|---|---|
| FP contraction | Small errors in multiply-add sequences | Disable FMA: `--fmad=false` (NV), `-ffp-contract=off` (GCC/Clang) |
| Reduction order | Errors accumulate differently with different block sizes | Use Kahan summation, or fix block size to match reference |
| Transcendental fn | `sin/cos/exp/log` differ by 1-2 ULP | Use `@fast_math=0` or link libm explicitly |
| BF16 FP16 rounding | Half-precision accumulation differs | Use FP32 accumulation, or match rounding mode |
| WMMA fallback | FP32 fallback produces different results | Verify fallback path is correct (not a stub!) |

---

## Roofline Model Usage

> **Read `kands://perf-benchmarks` for per-arch roofline parameters**

The PERF_BENCHMARKS.json now includes roofline model data. To use:

1. **Compute kernel arithmetic intensity (AI)**:
   ```
   AI = total_FLOPs_in_kernel / total_bytes_loaded_from_memory
   ```

2. **Compare AI to ridge point**:
   ```
   ridge = peak_compute (GFLOPS) / peak_bandwidth (GB/s)
   ```

3. **Interpretation**:
   - AI < ridge → **Memory-bound** → optimize data reuse, caching, tiling
   - AI ≥ ridge → **Compute-bound** → optimize instruction throughput, occupancy

4. **Example for gfx1201**:
   - Ridge = 17.4 GFLOPS / 1152 GB/s = 0.015 GFLOP/GB
   - If your attention kernel has AI = 0.01 → memory-bound → use DP4a instead of F32 fallback
   - If your GEMM kernel has AI = 0.5 → compute-bound → maximize occupancy

---

## GPU Fence & TDR Prevention

> **Critical for Windows: Timeout Detection and Recovery (TDR) resets the GPU if a kernel runs > 2s**

### The Problem

If you port a GPU kernel and forget to add proper fence synchronization, you'll get:
- **TDR resets** on Windows (kernel runs > 2s without yielding)
- **GPU hangs** (no fence to signal completion)
- **Stale data** (reading GPU buffer before write completes)

### Per-Platform TDR Limits

| Platform | Default timeout | Safe kernel duration |
|---|---|---|
| Windows (all vendors) | 2 seconds | < 1.5s with fence polling |
| Linux (CUDA) | 60 seconds (soft) | < 30s |
| Linux (Vulkan/DirectX) | 60 seconds (soft) | < 30s |

### Fence Patterns (Read `kands://porting-patterns` → `sync_patterns`)

#### CUDA → HIP (Direct)
```cpp
// CUDA:
cudaEvent_t evt; cudaEventCreate(&evt);
cudaEventRecord(evt, stream);
cudaEventSynchronize(evt);  // BLOCKING — no timeout on Windows!

// SAFER on Windows — poll with timeout:
while (cudaEventQuery(evt) == cudaErrorNotReady) {
    Sleep(1);  // Yield to CPU
    timeout_counter++;
    if (timeout_counter > 2000) { /* handle error */ }
}
```

#### CUDA → Vulkan
```cpp
// CUDA: cudaEventRecord(stream, event)
// Vulkan:
vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, query_pool, 0);
// Then use VK_FENCE_TYPE_TIMELINE for multi-second work:
VkSemaphore timeline_sem;
// Create with VkSemaphoreTypeCreateInfo(VK_SEMAPHORE_TYPE_TIMELINE)
uint64_t signal_value;
// Signal: vkSignalSemaphore(device, &(VkSemaphoreSignalInfo){ .semaphore = timeline_sem, .value = ++signal_value });
// Wait with timeout: vkWaitSemaphores(device, &(VkSemaphoreWaitInfo){...}, timeout_ns);
```

#### CUDA → DirectX 12
```cpp
// CUDA: cudaEventRecord
// DX12: Use timeline fence (NOT legacy fence!)
// MUST use ID3D12Fence1 (timeline) for multi-second operations
ID3D12Fence* timelineFence;
UINT64 fenceValue = 0;
device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&timelineFence));
HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

// After queue submission:
commandQueue->Signal(timelineFence, ++fenceValue);

// Wait WITH timeout (prevents TDR):
if (timelineFence->GetCompletedValue() < fenceValue) {
    timelineFence ->SetEventOnCompletion(fenceValue, fenceEvent);
    WaitForSingleObject(fenceEvent, 1500);  // 1.5s timeout < 2s TDR limit
}

// Cleanup (no leak!):
CloseHandle(fenceEvent);
timelineFence->Release();
```

### STRACE Integration for Fence Debugging

```cpp
// Copy mcp/templates/breadcrumb.hpp into your project
#include "breadcrumb.hpp"

// Before kernel launch:
STRACE("compute", "dispatch", "gemm", "m=%d n=%d k=%d", M, N, K);
// Record fence AFTER launch:
STRACE("sync", "record", "fence_signal", "queue=main fence_val=%llu", fence_val);
// Before reading results:
STRACE("sync", "wait", "fence_wait", "fence_val=%llu timeout=1500ms", fence_val);
```

### TDR Checklist

- [ ] Every kernel launch has a fence/semaphore signal
- [ ] Every fence has a timeout (≤ 1500ms on Windows)
- [ ] Fence events are properly cleaned up (CloseHandle / vkDestroySemaphore / Release)
- [ ] Timeline fences used for multi-second GPU operations (Vulkan/DX12)
- [ ] No `cudaDeviceSynchronize()` in hot path on Windows (blocks CPU thread, may cause TDR)
- [ ] STRACE() calls instrument every fence signal/wait
