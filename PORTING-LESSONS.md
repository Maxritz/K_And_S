# PORTING-LESSONS — what actually breaks when moving an inference engine to another GPU

Derived from a full CUDA→HIP port of a ternary-quantised LLM engine to RDNA4
(gfx1201). Every rule below is tied to a failure that was observed and measured,
not to a theory. Apply them to any engine of the same class (ninfer, deepinfer,
llama.cpp derivatives, vLLM-like servers, custom quantised GEMV stacks).

The short version:

> **Plan from probes, not from memory. Compile the instruction. Read the
> disassembly. Run the model. Gate the build on invariants.**

---

## 1. Hard rules (gates that must exist before the first edit)

| # | Gate | Why it exists |
|---|---|---|
| G0 | `build_function_reference` before anything else | the pre-port inventory turns "where do I start" into a work list: every runtime API, PTX instruction, intrinsic and feature used, its replacement, and whether that is auto / partial / manual / no-equivalent |
| G1 | `probe_isa` every instruction the port depends on, one candidate per TU | a whole port was designed around `v_dot4_i32_i8`; the gfx12 backend *crashes* on it rather than rejecting it at the assembler |
| G2 | `scan_residual_asm` must return zero before compiling | foreign mnemonics survived four "fix" passes because each regex silently no-op'd |
| G3 | delete the output artifact before every build | a stale `.hsaco` made a probe report kernels the current source could not produce |
| G4 | assert the exported symbol inventory matches the expected kernel list | "it compiled" and "the runtime can call it" are different claims |
| G5 | run the model and check the text, not just the exit code | compiling and producing coherent output are different claims |

If a gate cannot be expressed, the port is not ready to start.

---

## 2. ISA verification protocol

1. **Compile the candidate on the real target.** Capability tables, vendor docs
   and model memory are all unreliable. One compile is decisive.
2. **One candidate per translation unit.** Probing several in one file means the
   first error masks the rest — this cost a full round trip.
3. **Read the disassembly, not just the exit code.** Measured on gfx1201:

   | source | compiler result | emitted |
   |---|---|---|
   | `asm("v_dot4_i32_i8 …")` | backend crash | — (gfx9-only encoding) |
   | `amd_mixed_dot(char4,…)` | ok | `v_dot4_i32_iu8` ✅ |
   | `__builtin_amdgcn_udot4(…)` | ok | `v_dot4_u32_u8` ✅ |
   | scalar expansion | ok | no dot instruction |

   The *mnemonic* is not the capability. Two of these are the same hardware op
   under different spellings; one does not exist.
4. **Confirm wave width empirically.** `-mwavefrontsize32` vs default measured
   identical for a 32-lane-shaped kernel — a plausible optimisation that
   changed nothing. Measure before believing.
5. **Treat every "portable intrinsic" as unproven until disassembled.** A builtin
   that compiles may lower to a scalar loop.

Probe recipe:

```python
from isa_tools import probe_many
probe_many("gfx1201", [
    {"name": "raw v_dot4",  "stmt": 'asm("v_dot4_i32_i8 %0,%1,%2,%0;" : "+r"(acc) : "r"(a),"r"(b));'},
    {"name": "hip helper",  "stmt": "acc = amd_mixed_dot(ca, cb, acc, false);"},
], backend="hip")
```

---

## 3. Source transformation — the six ways a rewrite silently fails

1. **Literal mismatch.** `cp.async.ca_shared.global` in the pattern vs
   `cp.async.ca.shared.global` in the source. Zero substitutions, no error.
2. **Prefix mismatch.** Pattern expects `asm(`; source says `asm volatile(`.
3. **Terminator mismatch.** Pattern expects `";`; a multi-line macro ends
   `};"\` (backslash continuation). The regex never matches.
4. **Cross-line matching.** `re.DOTALL` on a line-oriented pattern makes `.*?`
   consume hundreds of lines up to an unrelated `";`, deleting real code.
5. **Whole-line replacement.** `tc_commit(){ asm(...); }` on one line is
   destroyed when a sweep replaces the entire line instead of the asm statement.
6. **Ordering.** Rewriting asm constraints (`"l"`→`"r"`) *before* the pattern
   that keys on `"l"` guarantees the later pattern misses.

Two structural traps that are worse than any of the above:

- **`__launch_bounds__(N)`** is the first `(` after the qualifier, so any naive
  signature scan names the function `__launch_bounds__` and the kernel vanishes
  from the inventory. Blank it before scanning. This bit twice — once in the
  hipifier, once in the porting-map extractor.
- **ISA constructs hide in macros.** `#define TC_MMA … mma.sync …` means a
  body-only scan reports a clean bill of health on code built entirely from an
  unsupported instruction. Attribute macro constructs to their callers.

**Do not hand-write instruction mnemonics.** Use the vendor's helper
(`amd_mixed_dot`, `__dp4a`, `sycl::…`) so the compiler chooses the encoding.

**Decide the output extension before rewriting anything.** A port is not
"translate every file": some files are copied verbatim, some change extension
and must be rewritten, and a few change language entirely. For example:

| source | → HIP | → Vulkan | action |
|---|---|---|---|
| `.cu` | `.hip` | `.comp` | rewrite |
| `.cl` (OpenCL) | `.cpp` | `.comp` | rewrite |
| `.comp` (GLSL) | `.cpp` | `.comp` | **copy** when targeting Vulkan |
| `.hlsl` | `.cpp` | `.comp` | rewrite |
| `.metal` | `.cpp` | `.comp` | rewrite |

Getting this wrong produces a tree that builds nothing, or silently drops the
files that needed real work. Generate the plan (`build_porting_map` → `file_plan`)
and review the rewrite/copy split before writing code.

---

## 4. Runtime and dependency layer

Inference engines rarely isolate the GPU behind a clean seam. Expect all of:

| dependency | what it looks like | handling |
|---|---|---|
| vendor array/tensor lib | `import cupy`, `torch.cuda` | replace the **interface**, not just kernels (a shim) |
| raw driver calls | `ctypes.CDLL("libcuda.so.1")`, `cuLaunchKernel`, `cuFuncSetSharedMemConfig` | intercept; `cuLaunchKernel` argument shape matches `hipModuleLaunchKernel` |
| CUDA graphs | `cuGraphCreate` … used for decode replay | `hipGraph*` exists; otherwise return failure so the engine falls back |
| runtime-generated code | engine concatenates sources and **generates wrappers from kernel text at load time** | replicate the generation; otherwise `d_k_*` symbols are "missing" when they were never emitted |
| build-time extension | `torch.utils.cpp_extension`, `setup.py` with nvcc | hipcc + `HIP_PLATFORM=amd`, or port the source |
| graphs/streams/events | per-token sync, double buffering | map to HIP equivalents; measure the sync cost |

**Shim conformance is behavioural, not nominal.** Broken in one port:

- device name returned `str`, engine called `.decode()` → returns **bytes**
- `arr[i]` returned a host scalar; engine passes it to a kernel → returns a
  **0-d device view**
- strided view `.copy()` did a flat `memcpy` → copy **per contiguous run**
- `kernelParams[i]` was the device pointer instead of a pointer **to** the value
  → access violation reading a device address on the host
- missing `Device.compute_capability`, `pool.free_bytes()` → attribute cascade

Checklist: enumerate every vendor-stack attribute the engine touches
(`grep -o 'cp\.[A-Za-z_.]*'`) and confirm each one exists with the right type.

---

## 5. Correctness protocol

1. Unit-test the smallest kernel you can (vec-add, a 4-element dot) before
   trusting the model.
2. Run the model. Coherent output on several unrelated prompts (factual, math,
   code) is the minimum bar.
3. Prefer a reference oracle: the upstream vendor fork, a CPU implementation, or
   llama.cpp. For Bonsai-style models the vendor publishes a tested setup — use
   it to confirm formats and expected quality.
4. **Grade fidelity explicitly.** A matrix instruction reconstructed element-wise
   is *approximate*, not equivalent. Gate it off by default
   (`supported() == False`) instead of silently serving wrong logits, and let the
   numerically faithful path be canonical.
5. Verify the *format* too: the same tensor id can mean different layouts across
   runtime generations (e.g. legacy vs official group-size). Loading successfully
   does not mean loading correctly.

---

## 6. Performance method

Never quote a per-kernel number collected with a sync after every launch on
Windows/WDDM — submission batching inflates it. Separate:

```
wall = host-launch + queue-wait + device-exec + sync
```

Measured on the port: 331 launches/token, 9.4 ms host, 25 ms device,
34 ms wall. Roofline said ~0.7 ms for the weight bytes. So the gap was **not**
arithmetic — it was per-launch fixed cost.

Findings that generalise:

- **Launch count × fixed cost dominates** for small tensors. 1–3 MB GEMVs sat at
  20–70 GB/s while the 127 MB head reached 462 GB/s. Optimise by **fusion and
  fewer launches**, not by better math inside a 40 µs kernel.
- **Size the grid to the work, not to the CU count.** Sweep `gx`/`gy`; a tensor
  with `NG/32 = 4` useful blocks does not benefit from 72.
- **Check the roofline per tensor class.** Small tensors are latency-bound; large
  ones can saturate. Treat them as different optimisation problems.
- **Verify the instruction actually helps.** The hardware dot4 was equivalent to
  scalar for this workload because the hot loop used a lookup table, not dp4a.

---

## 7. Engine-class checklist (ninfer / deepinfer / llama.cpp-derived)

Before writing any kernel, answer these. Each "no" is a scoping risk.

**A. What already works on the target?**
- Does the vendor publish a tested HIP/ROCm build of a comparable engine? Use it
  as the reference oracle for numerics and formats.
- Is ROCm PyTorch already installed? If the engine is PyTorch-based, much of the
  port may be "install the right wheel", not "write kernels".
- Does the engine already have a Triton or ONNX path? Those may need no port at
  all.

**B. What is actually custom?**
- Identify the *quantised* kernels: int4/int8/fp8 GEMV, dequant, attention,
  KV-cache layout. These are the real work; everything else is plumbing.
- Map each custom kernel to a construct family and grade it (see §2/§5).

**C. What runtime machinery must be replaced?**
- Allocator / memory pool, streams, events, graphs, peer copies.
- Build system: extension compiled at install time? Cross-compiling for gfx1201?
- Multi-GPU/NCCL → RCCL. Scope single-GPU first.

**D. What are the named library dependencies?**
- `cuBLAS/cuBLASLt` → hipBLASLt; `cuDNN` → MIOpen/CK; `CUB/Thrust` →
  hipCUB/rocThrust; `FlashAttention` → CK or a Triton/torch SDPA fallback;
  `NCCL` → RCCL. Confirm each has a drop-in for the *specific* call used, not
  just "exists on AMD".

**E. What data types are required?**
- fp8 / bf16 / fp16 / int8 support differs per arch. Probe, don't assume —
  gfx12 rejected one packed-fp16 instruction outright while accepting its
  two-instruction replacement.

**F. What is the numerical contract?**
- Which kernels must be bit-exact and which may be approximate? Write it down
  before touching either. Attention softmax and output projection tolerate far
  less than an activation.

---

## 8. Process discipline

1. **One source of truth for generated files.** Four overlapping "fix" scripts
   caused two silent regressions. Keep one pipeline; if a second pass is needed,
   make it report counts of what it changed.
2. **Probe before planning.** A 20-second compile would have saved most of the
   first day.
3. **Record the measurement with the change.** "Faster" without a number is not
   a result.
4. **Never let a script mutate files without a summary.** Diff, count, or hash.
5. **Re-inspect after any edit that rewrites large regions.** A truncation to
   3 lines is invisible until the next build.

---

## 9. Tooling that enforces this

Available in this MCP:

| tool | use |
|---|---|
| `capability_report` | what this host can actually build for a target |
| `probe_isa` | compile candidates, one TU each, report encode + emitted ISA |
| `scan_residual_asm` | gate: no foreign mnemonic inside a live asm string |
| `build_module` | delete-then-build, report size + symbol inventory |
| `build_porting_map` | per-function construct/fidelity map derived from source |
| `lookup_construct` | one construct → target substitution + grade |
| `verify_porting_map` | probe every mapped replacement, fold results back |

Use them in that order: capability → probe → map → gate → build → run.
