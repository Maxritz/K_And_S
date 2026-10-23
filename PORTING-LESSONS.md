# PORTING-LESSONS: what actually breaks when moving an inference engine to another GPU

Derived from a full CUDA→HIP port of a ternary-quantised LLM engine to RDNA4
(gfx1201). Every rule below is tied to a failure that was observed and measured,
not to a theory. Apply them to any engine of the same class (ninfer, deepinfer,
llama.cpp derivatives, vLLM-like servers, custom quantised GEMV stacks).

**The rules are stated in vendor-neutral terms.** Any execution model with an
asynchronous host-side submission queue and a SIMD device obeys them: the
platform name, instruction or file extension that appears below is the *evidence*,
never the subject. The Terminology map translates between the vocabularies.

The short version:

> **Plan from probes, not from memory. Compile the instruction. Read the
> disassembly. Run the model. Gate the build on invariants.**

---

## Terminology

The rules below are stated in API-neutral terms. Where a platform name is needed
for evidence, it appears in parentheses. Any execution model with an
asynchronous host-side submission queue obeys the same rules.

| concept here | CUDA | HIP / ROCm | Vulkan / DirectX | Metal | SYCL / oneAPI |
|---|---|---|---|---|---|
| work queue | stream | stream | queue | command queue | queue |
| captured commands | graph | graph | secondary command buffer | indirect command buffer | command graph |
| execution batch | block | workgroup | workgroup | threadgroup | work-group |
| SIMD unit | warp (32) | wavefront (32/64) | subgroup | SIMD group | sub-group |
| lane | thread | lane | invocation | thread | work-item |
| fast scratch | shared memory (SMEM) | LDS | groupshared | threadgroup memory | local accessor |
| timestamp | cudaEvent | hipEvent | timestamp query | encoder sample | event |
| kernel artifact | cubin / fatbin | hsaco / code object | SPIR-V module | metallib | SPIR-V / binary |
| compute unit | SM | CU | CU / EU | GPU core | Xe-core / EU |
| register count | registers/thread | VGPR / SGPR | registers | registers | registers |

"Host", "device", "kernel" and "launch" are used throughout because every one of
these APIs shares the same model: the host queues work asynchronously and the
device executes it later.

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
   first error masks the rest. This cost a full round trip.
3. **Read the disassembly, not just the exit code.** Measured on gfx1201:

   | source | compiler result | emitted |
   |---|---|---|
   | `asm("v_dot4_i32_i8 …")` | backend crash | none (gfx9-only encoding) |
   | `amd_mixed_dot(char4,…)` | ok | `v_dot4_i32_iu8` ✅ |
   | `__builtin_amdgcn_udot4(…)` | ok | `v_dot4_u32_u8` ✅ |
   | scalar expansion | ok | no dot instruction |

   The *mnemonic* is not the capability. Two of these are the same hardware op
   under different spellings; one does not exist.
4. **Confirm the default empirically, and verify the artifact rather than the
   flag.** `-mwavefrontsize32` was rejected outright by the toolchain
   ("unknown argument") and the default was already 32 lanes: `__ballot(1)`
   returns `0x00000000ffffffff` and the code object records
   `.wavefront_size: 32`. Two separate lessons in one measurement. A flag that
   a document describes may not exist in your compiler, and a flag that *is*
   accepted may be silently ignored. Read the emitted object, or probe the
   hardware, before believing either. (A spelling this toolchain accepts is
   `-Xclang -target-feature -Xclang +wavefrontsize32`.)
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

## 3. Source transformation: the six ways a rewrite silently fails

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
  from the inventory. Blank it before scanning. This bit twice, once in the
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
   llama.cpp. For Bonsai-style models the vendor publishes a tested setup. Use
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

Never quote a per-kernel number collected with a sync after every launch. Any
platform whose driver batches submissions (Windows/WDDM is one) inflates it by an
order of magnitude. Separate:

```
wall = host-launch + queue-wait + device-exec + sync
```

**Correction:** an earlier version of this section reported "25 ms device" for
this token. That figure came from the sync-per-launch instrument and was wrong.
Re-measured properly: **device 1.94 ms, host 3.3 ms, wall 5.34 ms**. The
per-launch conclusion held; the device number was inflated ~13x, and it hid a
second, real problem, which was the device running at 39% of achievable bandwidth. A
wrong measurement that survives into documentation is worse than no measurement.

### 6.1 Calibrate the instrument before you believe the number

Four different ways of timing the *same* kernel gave answers spanning three
orders of magnitude:

| instrument | what it actually reports |
|---|---|
| one timestamp pair around a whole phase | the **host's submission span**. When the host is the bottleneck the device is idle and the interval still equals wall time, so it looks like "device = wall" |
| a timestamp pair around every operation | the **instrumentation floor**, which is per operation, not per kernel: ~19 us here, so 346 operations invent ~6.7 ms of device time out of nothing |
| launch followed by an immediate device sync | the **driver round trip** (submission, signal, wake-up), not the kernel |
| **burst replay, host-subtracted** | the kernel. Queue the same launch 200+ times with nothing in between so the queue stays deep, then `kernel = wall - host_queue_time` |

The numbers in this table are one measured example (a 346-launch token on a
discrete GPU under Windows); the ordering of the four instruments is what
transfers.

- **Validate the timer against a known workload first.** A streaming read of a
  known size gives a measurable bandwidth; if the timer disagrees with it, the
  timer is wrong. (Here: 512 MB in 871 us = 616 GB/s, matching wall clock.)
- **A timestamp pair isolates a kernel only if nothing else is queued behind
  it.** Otherwise it measures queue wait as well.
- **Prefer one big interval over many small ones** for totals: ten tokens inside a
  single pair is honest; ten tokens each with their own pair is ten times the floor,
  and the floor is not small.
- **Do not assume the queue drains in order with your host code.** The host can
  finish submitting long before the device finishes executing, which is what makes
  the first row of the table above so misleading.

### 6.2 Decompose, then ask which side the wall is on

```
wall = host-active (launch calls + engine Python) + exposed device + idle gaps
```

The decisive question is **does the host finish submitting before the device
finishes executing?**

- If yes, the device idles through the token and the wall is **host-shaped**:
  launch count, per-launch cost and engine Python are the targets.
- If no, the wall is **device-shaped**: go into the kernels.

This distinction flipped the whole diagnosis of one port. The host took 3.48 ms
to submit a token whose device work was 1.94 ms. The device had already finished
while the host was still submitting, so device optimisation alone could not have
helped.

### 6.3 Invariant: the parts cannot exceed the whole

If the sum of measured device time exceeds wall time, the instrument is
inflating. Stop and fix the measurement. This one check catches the floor,
double counting, and timestamps taken on a different queue than the work.

### 6.4 Findings that generalise

- **A workload is launch-shaped when the model is small relative to the GPU.**
  346 launches x ~10 us of host cost was 3.3 ms of a 5.3 ms token. Compute
  `launches x per-launch cost` and compare with device time; if it is larger,
  fusion or capture is the win, not better arithmetic. A design that was fine on
  a 1-SM, 28 GB/s card becomes launch-bound on a 32-CU, 616 GB/s one.
- **Batching is the strongest cure for launch overhead** where the workload
  allows it: the same weights on the same card, as one GEMM over 256 tokens
  instead of one GEMV per token, cost **6.1x less per token**. Autoregressive
  decode cannot batch across tokens, which is exactly why its per-launch tax has
  to be removed structurally.
- **Rank by cumulative cost, and re-rank after every change.** The bottleneck
  moves. After launch reduction the device became visible again, and the profile
  changed from host-bound to device-at-39%-of-achievable-bandwidth.
- **Size the grid to the work, not to the CU count.** Sweep `gx`/`gy`; a tensor
  with `NG/32 = 4` useful blocks does not benefit from 72.
- **Check the roofline per tensor class, against measured bandwidth.** Small
  tensors are latency-bound, large ones can saturate; different problems.
- **Confirm the instruction helps.** A hardware dot product was equivalent to
  scalar for a workload whose hot loop used a lookup table instead.

### 6.5 Accept a change only on reproducible, bit-exact evidence

- **Fast can mean wrong.** A block-size change measured 1.6x faster and produced
  all-zero tokens. Assert numerics bit-for-bit against a reference *before*
  accepting a speedup. Timing without verification is not a result.
- **Check that the comparison harness proves anything.** A differential test that
  "passed" was comparing NaN to NaN, because the reference itself was invalid (synthetic
  inputs the kernel never sees). Print the reference's magnitude, finiteness and
  a checksum; a degenerate reference makes every candidate look equal.
- **Enough samples, a median, and interleaved arms.** 24-token samples varied by
  +-8%, larger than the effects being chased; 100 tokens cut that to ~1%. Arms
  run back to back also drift with clock and thermal state, so run A, B, A, B. An
  improvement that does not reproduce is not an improvement, and retracting one
  is a normal outcome.
- **Keep the known-good configuration:** revert first, re-measure, then change
  one variable.

- **Launch count times fixed cost dominates** for small tensors. 1 to 3 MB GEMVs sat at
  20 to 70 GB/s while the 127 MB head reached 462 GB/s. Optimise by **fusion and
  fewer launches**, not by better math inside a 40 µs kernel.
- **Size the grid to the work, not to the CU count.** Sweep `gx`/`gy`; a tensor
  with `NG/32 = 4` useful blocks does not benefit from 72.
- **Check the roofline per tensor class.** Small tensors are latency-bound; large
  ones can saturate. Treat them as different optimisation problems.
- **Verify the instruction actually helps.** The hardware dot4 was equivalent to
  scalar for this workload because the hot loop used a lookup table, not dp4a.

---

## 7. Verify the artifact, not the flag

The compiler is not the oracle. **The emitted code object is.**

1. **Know your container.** `hipcc`/`nvcc` write a *bundle* (clang offload
   bundle or fatbinary), not a bare ELF, so `objdump`/`nm` reject it with "not
   recognized as a valid object file", and a bundle can hold several
   architectures. Slicing from the first ELF magic analyses whichever payload
   comes first, i.e. possibly an architecture that was never targeted. Parse the
   container and select by triple (`unwrap_code_object(path, arch=...)`).
2. **Read the metadata, do not infer it.** A kernel artifact states SIMD width,
   register counts (VGPR/SGPR, or plain registers), spill counts and segment
   sizes (shared/scratch memory per block). `code_object_report` extracts them
   per kernel and computes how many blocks that scratch memory allows to be
   resident on a compute unit.
3. **Distinguish "flag rejected" from "source broken".** A probe of six compiler
   flags "proved" all six were rejected; in fact the test source did not compile
   and every candidate failed for an unrelated reason. When a probe fails, print
   the whole error and test the null hypothesis (compile with no flag at all).
4. **Defaults are not what you assume.** The same probe showed one flag was
   unnecessary because the default was already the desired behaviour, while
   another toolchain defaults the other way. Measure the default.
5. **Multi-arch bundles mean symbol inventories can lie.** A module that builds
   and exports `k_g4` may export it for the wrong target. Delete the artifact
   before every build (G3) and confirm the payload triple.

### 7.1 Residency audit: port the reason, not the code

The highest-value reading in one port came from a single field:

| field (generic name) | value | consequence |
|---|---|---|
| shared memory per block (`group_segment_fixed_size`, `smem`) | 41 472 B | against 64 KiB usable per compute unit: **one block resident per CU** |
| bytes fetched from global memory | 96 B per iteration | the kernel is not streaming |
| shared-memory loads + vector bit-manipulation ops | 64 + 64 per iteration | issue slots are spent **decoding**, not moving data |
| SIMD width (`wavefront_size`, warp size) | 32 | (the assumed problem, already correct) |

That kernel was a ternary decode loop written for a GPU with **no integer dot
hardware** (a low-end Kepler part): hoisting lookup tables into shared memory was
the only way to decode.
On the new target the tables pin occupancy at one block per CU and burn the issue
slots. The design's *reason* had disappeared, and nothing in the port pointed at
it, because the code still compiled and still produced correct output.

**Rule: when a kernel hoists data into a scarce resource (shared memory,
registers, a texture, constant memory), ask what scarcity it was written for. If
the new target does not have that scarcity - or has the instruction that made the
trick unnecessary - the structure is now a tax, not an optimisation.**

---

## 8. Capture and replay: the semantics that bite

Recording a command sequence once and replaying it as a single unit (CUDA/HIP
graphs, secondary command buffers, command graphs) is the cheapest large win for
a launch-shaped engine: measured 186 -> 250-268 tok/s over 18 layers with no
numerical change. It is also easy to get subtly, silently wrong.

1. **Audit the captured region for capture-illegal operations.** Allocations and
   synchronising copies are illegal. Make transfers stream-ordered first (async
   copies are legal, sync ones are not) and look for hidden allocation in hot
   paths (`.copy()` on a view is a common one).
2. **A captured region bakes in host-side bookkeeping.** Anything the host would
   have done during the recorded calls, such as swapping buffers, resetting a flag or advancing a
   counter, does *not* happen on replay, and the next kernel may
   branch on it. Record the host state after the recording pass and re-apply it,
   or the replay silently uses stale state.
3. **A capture is only valid for the state and context it was recorded in.** It
   bakes in the buffers, strides and arguments that were live at record time.
   In one engine a double-buffered activation stream meant the buffer at a given
   layer depended on how many buffers had already been swapped, and that count
   differed between two entry paths (one performed an extra normalisation per
   step). Replaying a capture recorded on the other path read the wrong buffer.
   Key captures by a signature of the state they assume, and record a second one
   when the signature changes. The same applies to shapes, dtypes, and batch
   dimensions: a capture is not a function.
4. **A failed capture must re-run the work.** During capture kernels are
   *recorded*, not executed. Every failure path, whether capture was refused or end-capture or instantiate
   failed, has to fall back to running the region for real,
   and should blacklist the tag so it is not retried on every token.
5. **Verify capture against the un-captured path, including phase transitions.**
   Testing one phase in isolation passed while the transition from a batched
   phase into a token-at-a-time phase produced garbage. Compare exact outputs
   across the *transitions* between heterogeneous code paths, not only within one.
   This is also where a wrong buffer or a stale flag shows up first.

---

## 9. Measured numbers beat datasheet numbers

Every roofline in a port is divided by an assumed bandwidth and multiplied by an
assumed resource size. Both are usually wrong, and usually in the direction that
hides the real gap. One measured example, from a single card:

| quantity | reference value | measured value | consequence of the gap |
|---|---|---|---|
| streaming bandwidth | 1152 GB/s | **616 GB/s** | a "39% of peak" kernel is really 73% of what the machine can do: the target was 2x too optimistic |
| cache-resident bandwidth | - | **~1424 GB/s** | working sets that fit in cache are worth structuring for |
| scratch memory per compute unit | 128 KiB (in the table) | **64 KiB usable** | residency math is off by 2x |
| number of compute units | 72 (in the table) | **32 reported by the runtime** | the runtime property counted *workgroup processors*, not compute units (32 WGP = 64 CU) |

Rules:

- **Store the measured value with its provenance.** "1152 GB/s" is a spec;
  "616 GB/s measured by a 512 MB streaming read on this card" is a number you can
  divide by. Mark which is which in any data file.
- **Beware unit ambiguity in runtime properties.** A property named
  "multiprocessor count" or "compute unit count" may return execution batches,
  half-compute-units, or something else entirely. Confirm against the physical
  spec before using it to size a grid - a factor-of-two error here silently
  halves occupancy or doubles the grid.
- **Calibrate both ends of the memory hierarchy** if the workload has two
  regimes: streaming (DRAM) and cache-resident. They differ by 2x or more.

---

## 10. Engine-class checklist (ninfer / deepinfer / llama.cpp-derived)

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
- fp8 / bf16 / fp16 / int8 support differs per arch. Probe, don't assume:
  gfx12 rejected one packed-fp16 instruction outright while accepting its
  two-instruction replacement.

**F. What is the numerical contract?**
- Which kernels must be bit-exact and which may be approximate? Write it down
  before touching either. Attention softmax and output projection tolerate far
  less than an activation.

---

## 11. Process discipline

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

## 12. Tooling that enforces this

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
| `code_object_report` | artifact truth: wave size, registers, spills, LDS per block, residency, instruction mix |

Use them in that order: capability, probe, map, gate, build, run, in that order.
