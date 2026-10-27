<img width="960" height="410" alt="image" src="https://github.com/user-attachments/assets/8471da74-e628-4a74-b92d-b647e532d95d" />

**Kernels & Shaders**[^kand]

**Many architectures. Endless debugging.**

---


**K_and_S** (Kernels and Shaders) is an MCP server that helps you port GPU kernels and shaders across different hardware vendors and APIs. Instead of guessing how to translate a CUDA kernel to HIP for AMD, or how to get the best performance out of RDNA on Vulkan, K_and_S gives you the knowledge, patterns, and tools to do it properly.

---

## Why This Exists

Modern GPU development is complicated. You have NVIDIA with CUDA, AMD with HIP and ROCm, Intel with oneAPI and SYCL, and then there is Vulkan and DirectX for cross-platform compute. Most developers stick to one vendor because porting is hard. When you do try to port, you end up copying code blindly, hoping intrinsics translate, and debugging mysterious performance regressions.

K_and_S was built to fix that. It started as a collection of notes from real kernel porting work, documenting what actually works on real hardware. We tested on AMD RDNA2 and RDNA4 (gfx1031, gfx1201), and later will test on Intel Core Ultra 5 135H (meteor_lake). We learned which patterns fail, which optimizations help in practice, and which myths waste hours of development time.

This is not just a tool. It is a knowledge base that travels with you. When you fix a kernel on one project, you can save the solution as a template and reuse it on the next project. When the team grows, the knowledge stays.

---

## What You Can Do With It

### Discover and Audit
Point K_and_S at any codebase. It classifies every `.cu`, `.hip`, `.cuh`, `.sycl`, `.comp`, `.tesc`, `.hlsl`, `.metal` and `.cpp` file by **what is inside it**, not just by its extension: HIP that lives in a `.cu`, device code compiled through a `.cpp`, or AVX-512 intrinsics in a file whose path never mentions "cpu". Build output, `node_modules` and VCS directories are pruned, so a copy under `build/` is not counted twice.

### What the target needs before you port
Discovery can also answer the question that decides the schedule: *what does this port need for `hip`/`cuda`/`vulkan`/`directx12`/`sycl`/`metal`?* Pass `target_backend` to `discover_kernels` (optionally `target_arch`, `host_os`) and it appends a readiness diagnosis.

Each requirement gets one evidence label and never a guess:

| label | meaning |
|---|---|
| **PASS** | probed and present |
| **FAIL** | probed and absent — use the fallback or stop |
| **NOT RUN** | needs a build or a manual step |
| **UNVALIDATED** | could not be probed; treat it as absent until proven |

Rows are weighted `impact (1-5) x likelihood (0-1)` and ranked by Pareto: the rows that reach 80% of the total score are the vital few, and that is the order to resolve them in. Manual and host rows stay visible instead of passing silently. Every non-PASS row carries the fallback to take.

**PCIe ReBAR is a first-class row.** When the backend can use a large BAR (`hipHostMalloc` + `hipMemAdvise`, `cudaHostAllocMapped`, a host-visible device-local Vulkan heap, D3D12 GPU upload heaps, SYCL USM host allocations), the diagnosis probes the host's **display-class** PCI devices — a host bridge or a NIC that shares the vendor id is not a GPU — and reads both numbers that matter: `resource*_resize` lists the window sizes the device can become, and `resourceN` is the window enabled now. Capability alone is not a pass: a resizable BAR still sitting at 128 MiB does not fit the streaming working set, so that row is **UNVALIDATED** with the size printed and the firmware toggle named, and a maximum below the backend's threshold is **FAIL**. An NVIDIA driver report is only used for a device that driver actually speaks for. With no GPU present the row is UNVALIDATED, not FAIL. If ReBAR is not PASS the fallback is a staging buffer plus chunked copy — and the report says so, so the port is never described as zero-copy when it is not. Apple metal has no discrete BAR window, which is recorded as not applicable rather than as a failed probe.

**Capabilities are probed on the device when a backend has a tool that can answer.** `nvidia-smi --query-gpu=compute_cap` gives the SM number the capability is derived from, `rocminfo` gives the gfx target, wavefront width and LDS size, and `vulkaninfo` lists `VK_KHR_cooperative_matrix` and the integer-dot extension — which is exactly what the Vulkan rows ask. A device measurement outranks every table, but only for the arch it measured: a probe on an RX 9070 is not evidence about an MI300, and the report says when it withheld the result for that reason.
### Where the facts come from
A capability table is a claim, not a measurement, and tables rot: consumer CPUs ship with AVX-512 fused off, one AMD target has no matrix core while its neighbours do, and a hand-typed entry can disagree with itself. So every arch fact is resolved from four sources and **records which one it came from**:

| source | weight | what it means |
|---|---|---|
| `user` | highest | `ARCH_FACTS.local.json` (or `$KANDS_ARCH_FACTS`) — a human asserted it |
| `probe` | | measured on this machine: CPU ISA flags from `/proc/cpuinfo`, `sysctl`, or PCI identity |
| `remote` | | a shared JSON feed at `$KANDS_FACTS_URL` — opt-in, off by default, never required |
| `bundle` | lowest | `ARCH_MAP.json` — documented, never a measurement |

A fact no source has is **UNVALIDATED**, and the report names the exact key to feed rather than guessing it.

A capability row may name several candidate field names, because one capability has a different name per API (`wmma` in HIP, `vulkan.cooperative_matrix` in Vulkan), and nested blocks in the table are read instead of dropped. Above all four sources sits the device probe described above, applied only when it measured the target arch.

Rows can carry `host_arch` as well as `host_os`: an AVX2 row does not FAIL on an ARM host, and an ARM NEON row does not FAIL on x86. Both read as **NOT RUN — not applicable**, outside the gate and outside the marching order. A probe only answers for the machine it ran on: a named CPU model (`intel_13th`) does not borrow the build host's ISA flags, and a HIP row never gets its ReBAR answer from `nvidia-smi`.

The row set covers **GPU and CPU**. `cpu` is a backend like any other: MSVC on Windows, GCC on Linux, clang everywhere, AVX2/FMA/AVX-512 rows resolved from the probe, and a BLAS row that is honest about precision. A row may name its `host_os`; on any other host it is **NOT RUN — not applicable**, never FAIL, so one model covers Windows, Linux and macOS instead of three files.

### The directive
The report ends with a marching order, not a table: every non-PASS row in Pareto order, each with the probe to run, the fallback to take, and a gate — `blocked` (an impact-5 row probed absent), `validate-first` (an impact-5 row nobody has evidence for: absence is the assumption), `fallback` (the vital rows have evidence; the rest take fallbacks), or `clear`. Rows that do not apply to this host are not steps. An unmodelled backend still gets a directive: describe it first.

`autonomous_port` runs the same diagnosis automatically in its Phase 0.

### The pre-port analysis is a document, and it is mandatory
A readiness report that scrolls past in a tool response leaves no record: three days later nobody can reconstruct which rows were known, which were waived and who said so. So every run with a `target_backend` writes a fixed document set into `<repo>/docs/pre-port/<backend>-<arch>/`:

| file | what it answers |
|---|---|
| `00-README.md` | what this is, the gate, the sign-off block |
| `01-target-and-evidence.md` | target identity and where every fact came from |
| `02-requirement-matrix.md` | the Pareto order, per-row probe, ReBAR window, fallbacks |
| `03-directive.md` | the marching order, stop conditions, definition of done |
| `04-fault-tree.md` | failure modes, each gated by named rows |
| `05-waivers.md` | the human decision for every row that is not PASS |
| `diagnosis.json` | the same diagnosis, machine-readable |

`write_preport_analysis` regenerates the set on demand; `discover_kernels` and `autonomous_port` write it as part of Phase 0. It is a **gate, not an artefact**: until it exists and agrees with the live probe, the directive opens with a `STOP-DOCS` step and the port does not start. `05-waivers.md` is written once and never overwritten — it is the only place a non-PASS row becomes a decision with a reason, an owner and a date. A set that has drifted from the current probe reads as `stale` rather than as done, because a stale waiver is worse than no waiver.

### After the port: the recheck
A port can be correct, build clean, and still be three times too slow for a reason no kernel inspection finds. One real case: a Vulkan engine moved from AMD to an Intel iGPU, same kernels and quantisation — 32-34 tok/s became **20.65 tok/s**. Dispatches per token: ~263. Cost of one dispatch on the target: ~385 us (100 us on the source). Arithmetic the GPU needed: ~15 us. So 44.7 ms of a 48.4 ms token went to dispatch overhead, at **0.03% of peak**. The kernels were fine; the structure was designed where a dispatch was cheap.

`postport_recheck` turns that failure mode into checks that run on measurements, for any vendor and any backend:

| check | fires when |
|---|---|
| `efficiency.dispatch_overhead` | `overhead x dispatches` is more than half the iteration |
| `structure.dispatch_count` | the port emits noticeably more dispatches than the source platform |
| `structure.fusion_legality` | a proposed fusion needs a cross-workgroup barrier (not available in Vulkan compute at any workgroup size) |
| `limits.workgroup_size` | the dispatched size exceeds the largest size actually verified on the driver |
| `efficiency.peak_utilisation` | achieved work is a few percent of peak, i.e. the bottleneck is not arithmetic |
| `efficiency.source_parity` | the target is meaningfully below the platform the design came from |
| `extensions.available_vs_used` | an extension is enabled but no shader exercises it |

It takes only numbers; anything you have not measured is **NOT RUN** and names the metric to go and get. Numbers that cannot all be true together (overhead x count exceeding the whole iteration) come back UNVALIDATED rather than as a confident percentage, and correctness gates the rest: tuning a kernel that computes the wrong answer measures the wrong thing. Each run writes `docs/post-port/<backend>-<arch>/` and folds the previous numbers in as a delta, because the delta is the result.

The same seven checks are rows in the pre-port matrix, marked `advisory`: they cannot be answered before code exists, so they never gate the start of a port, and they are the first thing the recheck evaluates once it does.

### Check Compatibility
Before you start porting, see if your target architecture supports everything your source needs. Does your target have tensor cores? DP4a? Native BF16? Will Vulkan cooperative matrices work (they do not on AMD, use DP4a instead). K_and_S tells you upfront so you do not waste weeks on a dead end.

### Inventory before you port: the pre-port reference
Do not start from "where do I begin". Start from a complete list of what the tree actually depends on.

`build_function_reference` scans the sources and produces the vendor-function reference for your target: every CUDA runtime call, driver call, inline-PTX instruction, device intrinsic and framework feature that is genuinely used, each with

- the replacement on the target,
- a status: **Auto** (compiler maps it), **Partial** (maps but differs), **Manual** (source must change) or **No equivalent** (restructure),
- the call sites and the files that depend on it.

Status resolves **per architecture**, because the answer changes within a vendor family. `mma.sync` is manual on every AMD target, but MFMA covers CDNA, WMMA covers RDNA3/RDNA4, and RDNA2 has no matrix core at all. The table says so instead of flattening it all into "manual".

Anything with **no equivalent** is reported as a hard blocker. A blocker is not a partial gap. The kernel has to be redesigned or dropped, and that decision belongs before the first line of code, not after.

`lookup_api` gives the same answer for a single symbol while you are reading code.


Capability tables lie. A real port was designed around `v_dot4_i32_i8`, and that mnemonic has **no gfx12 encoding**. It crashes the LLVM backend instead of failing at the assembler. Its replacement, `amd_mixed_dot`, compiles fine and emits `v_dot4_i32_iu8`. Same hardware, different spelling.

`probe_isa` compiles each candidate **for your actual target**, one translation unit per candidate so a single failure cannot mask the rest, and returns both the verdict and the disassembled mnemonics. That is the only reliable oracle for "does this instruction exist on this chip".

### Build a function-to-function porting map
`build_porting_map` reads your source and, for every kernel or device function, reports:

- the constructs it uses (inline-asm mnemonics, warp intrinsics, barriers, local memory, id builtins)
- the **substitution** for your target, with an explicit fidelity grade: `exact`, `portable`, `approximate`, `lossy`, `manual` or `drop`
- a caveat where one applies (for example a matrix instruction that can only be reconstructed element-wise)

It understands CUDA/HIP, SYCL, OpenCL, GLSL, HLSL and Metal, and it follows ISA constructs that hide inside `#define` macros. Otherwise code built entirely from an unsupported instruction reports as clean.

It also emits the **extension-to-extension file plan** for the ported tree, so you know which files are rewritten and which are copied:

| source | to HIP | to CUDA | to SYCL | to Vulkan | to DirectX |
|---|---|---|---|---|---|
| `.cu` (CUDA) | `.hip` | `.cu` | `.cpp` | `.comp` | `.hlsl` |
| `.cl` (OpenCL) | `.cpp` | `.cu` | `.cpp` | `.comp` | `.hlsl` |
| `.comp` (GLSL) | `.cpp` | `.cu` | `.cpp` | `.comp` (copy) | `.hlsl` |
| `.hlsl` | `.cpp` | `.cu` | `.cpp` | `.comp` | `.hlsl` (copy) |
| `.metal` | `.cpp` | `.cu` | `.cpp` | `.comp` | `.hlsl` |
| SYCL `.cpp` | `.cpp` | `.cu` | `.cpp` (copy) | `.comp` | `.hlsl` |

`verify_porting_map` then closes the loop: it probes every mapped *replacement* on the real target and downgrades anything that does not encode.

### Gate the build
`scan_residual_asm` finds live `asm("...")` strings that still name a foreign mnemonic (comments are ignored, so no false positives), and `build_module` deletes the output before compiling, then reports the size plus the exported kernel symbols compared against what you expect. A module that compiles but lacks a kernel the runtime calls is not a successful build, and a stale artifact must never look like a fresh one.

### Inspect the compiled artifact
`code_object_report` opens a compiled kernel file (`.hsaco`, `.cubin`, `.co`, `.elf`) and tells you what the compiler actually produced.

It unpacks the container and selects the right target if the file holds more than one architecture, then reports for each kernel: SIMD width, register counts, spills, scratch memory per block, how many blocks that scratch allows to reside per compute unit, and the instruction mix (global loads with their widths, shared memory operations, bit operations, dot operations), with plain verdicts.

This is how a kernel that looks acceptable in a timing profile turns out to be pinned at one block per compute unit, or to be spending its issue slots decoding data instead of moving it. In one port it found both problems in a single kernel that had been assumed to be memory bound.

### Find Installed Toolchains
Instead of reading docs and guessing where hipcc or nvcc lives on your machine, K_and_S checks at runtime. It looks in your PATH, checks your environment variables (ROCM_PATH, CUDA_PATH, VULKAN_SDK, ONEAPI_ROOT), and scans common install locations. It even detects Strawberry Perl, which you need for some build scripts alongside MSVC.

### Estimate Performance
Got a target architecture and a workload (say, flash attention at 8192 sequence length)? K_and_S uses roofline models to tell you the theoretical ceiling and what real developers achieve in practice. You get a target tokens per second number before you write a single line of code.

Remember to divide by a bandwidth you measured on your own card. On one RDNA4 card the datasheet says 1152 GB/s and a streaming read measures 616 GB/s, so a kernel sitting at 39 percent of the datasheet figure was really at 73 percent of what the machine can do.

### Reject Common Myths
Before you follow the usual advice, check if it is wisdom or just cargo culting. Is hipify-perl enough to port a CUDA project to HIP? No. Can you copy CUDA code to HIP verbatim? No. Does Vulkan CooperativeMatrix work on AMD? No. K_and_S has a tool that calls out these patterns as wrong and explains the correct approach for your specific architecture.

### Plan Radical Experiments
RDNA has hundreds of extensions most developers never use because they copy CUDA habits. K_and_S generates experiments you can try instead: use DPP shuffles instead of LDS transposes, try register-only blocking for small tiles, compare wave32 vs wave64, write raw AMDGCN assembly for hot loops. Each experiment comes with a verification plan so you do not break correctness while chasing performance.

### Verify Correctness
Every port gets checked against a CPU reference (the CPU-as-oracle strategy). Feed it your reference output and your ported output, specify a tolerance, and it tells you if they match. No more shipping broken ports and finding out later that outputs diverged by 0.3 percent on half the test cases.

---

## What Users Share As Templates

One of the most powerful features is the ability to save and share learnings. After you successfully port something, you do not just move on. You save what worked.

### Porting Reports
Run the full porting workflow and K_and_S generates a complete report covering every phase: baseline capture, source analysis, arch comparison, toolchain setup, performance estimates, code tracing plan, and final verification checklist. Save this report and you have a blueprint for anyone else porting a similar engine.

### Tracing Plans
Every kernel gets a STRACE breadcrumb trail plan. This tells you exactly where to add tracing macros so you can measure stage-by-stage performance. Save these as templates so your team always instruments consistently.

### Optimization Experiments
Did you find that wave32 with DPP shuffles beats the naive port by 2.3x on gfx1201? Document it. The `radical_experiment_plan` tool creates an experimentation plan with hypotheses, implementation steps, and verification requirements. When you find something that works, save it as a contribution template.

### Toolchain Results
After running `detect_toolchain`, save the output. Next time someone on your team sets up a new machine, they know exactly which environment variables and paths to check.

### Sharing Format
All templates are standard markdown files. You can:
- Put them in a `findings/` directory in your project
- Share them with your team via Git or Slack
- Submit them as contributions back to the K_and_S bundle using `export_contribution`
- Load them later using `read kands://findings/engine-name`

---

## How It Makes Things Easier

### Before K_and_S
1. Google for how to install ROCm on Windows
2. Try hipify-perl and hope for the best
3. Debug kernel by kernel, discovering incompatibilities late
4. No way to know what performance to expect
5. Lose hard-won knowledge when team members leave

### With K_and_S
1. One command tells you what is actually installed (`detect_toolchain`)
2. `reject_common_pattern` stops you from following dead-end advice
3. `compare_archs` shows all feature gaps before you start coding
4. `estimate_performance` gives you a realistic target upfront
5. `verify_numerical_equivalence` catches correctness bugs immediately
6. `code_object_report` shows what the compiler really emitted, so you tune the kernel you have and not the one you assumed
7. Save successful approaches as templates for the whole team

---

## Supported Hardware

| Vendor | Architecture IDs | Notes |
|---|---|---|
| NVIDIA | sm_75, sm_80, sm_86, sm_89, sm_90, sm_120 | CUDA, cuBLAS, cuBLASLt |
| AMD | gfx908, gfx1030, gfx1031, gfx1100, gfx1150, gfx1200, gfx1201 | ROC 10.1, TheRock, Tensile, hipBLASLt |
| Intel | gen12, gen13, meteor_lake, arc_alchemist, arc_battlemage, arc_b580 | oneAPI DPC++, SYCL, Vulkan |

**Special note on AMD Vulkan**: CooperativeMatrix is broken on AMD Vulkan drivers. K_and_S automatically routes to DP4a (GL_KHR_shader_integer_dot_product) instead.

---

## Quick Start

### Install the MCP Server

The server speaks both `mcp` API generations (0.x `add_request_handler` and the
1.x/2.x `request_handlers` registry), so no version pin and no downgrade is needed.

```bash
cd mcp
pip install -e .
```

Then confirm it starts, rather than assuming it did:

```bash
python _smoke_register.py
```

That prints the `mcp` version, which API path was taken, the resource count, a large
resource read, and a read through a non-`str` URI. Expect **21 resources** and a
**~26,000-character** read from `kands://porting-patterns`. If it fails here it will
fail in the client too, and the output says which part broke.

### Configure Your MCP Client

#### OpenCode

Use the CLI: it writes the correct shape for you, and `opencode mcp list` is also the
first thing to check when a server does not appear.

```sh
opencode mcp add kands \
  --env PYTHONPATH=/absolute/path/to/K_and_S/mcp \
  -- python -m server
opencode mcp list          # expect: kands  connected
```

Add `--global` to make it available in every project; omit it for project-local.

By hand, server names go under **`mcp.servers`**. V2 does not accept them directly under
`mcp`, and a block OpenCode does not recognise is ignored **silently**:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "servers": {
      "kands": {
        "type": "local",
        "command": ["python", "-m", "server"],
        "environment": { "PYTHONPATH": "/absolute/path/to/K_and_S/mcp" },
      },
    },
  },
}
```

Field notes that matter here:

- `command` is the executable followed by its arguments, as one array.
- `environment` adds variables to OpenCode's own environment. `PYTHONPATH` is what makes
  the server's sibling imports (`isa_tools`, `porting_map`, `reference_search`) resolve,
  and it must be absolute.
- `cwd` is an alternative to `PYTHONPATH`: point it at the `mcp/` directory.
- `disabled: true` keeps a server configured without connecting it. There is no `enabled`.
- Use `{env:NAME}` for environment substitution; `$NAME` is not expanded in JSON.
- Config changes reload on save. There is no restart step.

#### Other clients

Claude Desktop and others use this shape instead. Note `mcpServers`, `env`, and the split
command/args: OpenCode does not accept any of those.

```json
{
  "mcpServers": {
    "kands": {
      "command": "python",
      "args": ["-m", "server"],
      "env": { "PYTHONPATH": "/path/to/K_and_S/mcp" }
    }
  }
}
```

#### If it does not show up in the server list

- Run `opencode mcp list`, or `/mcps` inside OpenCode. If it is not listed there, OpenCode
  never saw the configuration. The usual cause is the nesting: an entry under `mcp.<name>`
  instead of `mcp.servers.<name>`, or a `mcpServers` block, both of which are ignored
  without any error.
- Check the file is one OpenCode reads: `opencode.json(c)` at a project root or any
  ancestor, or the global `~/.config/opencode/opencode.json(c)`.
- Run `python _smoke_register.py` inside `mcp/`. A server that crashes at startup produces
  nothing useful in the client and simply never connects.
- On Windows, give `command` the full interpreter path (`C:/.../python.exe`): `python` may
  not be on the PATH that OpenCode's process inherits.
- Make sure the config file is valid JSON **as a whole**. A trailing comma or a mismatched
  brace anywhere in it stops the file loading at all, so every entry disappears and it
  looks like an MCP problem. Check it directly:

  ```sh
  python -c "import json,io,sys; json.load(io.open(sys.argv[1], encoding='utf-8')); print('valid')" path/to/opencode.json
  ```

#### If the server shows as failed

The client reports only `process exited with code 1`. Run it yourself to see why:

```sh
cd mcp
python -m server        # should wait on stdio, not exit
```

- A traceback here is the real error. Two that have been hit and are fixed in `main`: a
  dead `from mcp.server.lowlevel import NotificationOptions` (a module some `mcp` builds do
  not have), and `asyncio.run(stdio_server(server))`, which cannot work because
  `stdio_server` is an async context manager that takes no server argument. Pull the latest
  `main` before debugging either.
- If a source edit seems to have no effect, delete `mcp/__pycache__`: a stale `.pyc` keeps
  running the old code after the source changes.

#### If it will not start

- `ModuleNotFoundError: No module named 'mcp_types'`, or `'Server' object has no
  attribute 'add_request_handler'` — these were real bugs in this repo (a top-level
  `mcp_types` import and a 0.x-only registration call) and are fixed; pull the latest
  `main` before debugging your environment.
- `ModuleNotFoundError` for a sibling such as `isa_tools`, `porting_map` or
  `reference_search` — that is `PYTHONPATH` pointing somewhere other than `mcp/`, not a
  broken install. The server imports those modules from its own directory.
- A resource read returns `Unknown resource URI` — list the resources first
  (`_list_resources()` via the smoke test prints the real URIs); the scheme is
  `kands://`, and the names are not guesses.

### Typical Workflow

```
# 1. What is installed?
detect_toolchain(gpu_vendor="AMD", host_compiler="auto")

# 2. Compare source and target architectures
compare_archs(arch_a="sm_80", arch_b="gfx1201")

# 3. What performance can I realistically get?
estimate_performance(arch="gfx1201", workload="flash_attn", seq_len=8192)

# 4. Check a common idea before implementing
reject_common_pattern("hipify-perl handles everything", arch="gfx1201")

# 5. Verify every instruction the port depends on, before designing anything
probe_isa(target="gfx1201", candidates=[
    {"name": "raw v_dot4",     "stmt": 'asm("v_dot4_i32_i8 %0,%1,%2,%0;" : "+r"(acc) : "r"(a),"r"(b));'},
    {"name": "amd_mixed_dot",  "stmt": "acc = amd_mixed_dot(ca, cb, acc, false);"},
])
# -> raw v_dot4 FAIL (no gfx12 encoding); amd_mixed_dot OK (emits v_dot4_i32_iu8)

# 6. Build the function-to-function map from the source
build_porting_map(files=[...], source_arch="sm_80", target_arch="gfx1201")
# -> per function: role, constructs, target substitution, fidelity grade

# 7. Prove the map instead of trusting it
verify_porting_map(files=[...], target_arch="gfx1201")

# 8. Gate the build
scan_residual_asm(files=[...], backend="hip")
build_module(sources=[...], target="gfx1201", out="engine.hsaco",
             expected_symbols=["k_g4", "k_rmsw"])

# 9. Look at what the compiler actually emitted, before writing any tuning code
code_object_report(path="engine.hsaco", arch="gfx1201", lds_kib=64)
# -> k_g4: scratch 41472 B per block, 1 block per CU,
#    64 shared loads and 64 bit ops for every 96 bytes decoded

# 10. Port the full repo
autonomous_port(repo_path="/path/to/kernel/code", source_arch="sm_80", target_arch="gfx1201")

# 11. Verify correctness after each change
verify_numerical_equivalence(arch="gfx1201", workload="gemm", source_values=[...], target_values=[...])

# 12. Record what you learned
record_performance(arch="gfx1201", workload="gemm", tok_per_s=42500.0, notes="Used DPP shuffle, wave32")

# 13. Save as a template for next time
export_contribution(engine_name="MyEngine", arch="gfx1201", workload="gemm")
```

---

## Phase 0 to Phase 8 Methodology

K_and_S follows an eight phase methodology so you never miss a step:

**Phase 0**: Establish baseline with perf tracing before any changes
**Phase 1**: Environmental scan and toolchain detection
**Phase 2**: Kernel census and feature discovery
**Phase 3**: Architecture comparison and gap analysis
**Phase 4**: Instrumentation with breadcrumb tracing
**Phase 5**: Manual or assisted porting
**Phase 6**: Numerical equivalence verification against CPU oracle
**Phase 7**: Performance optimization loop (declare, trace, dissect, hypothesize, experiment, verify, record)
**Phase 8**: Share and contribute your findings back

---

## Field Notes: `kands://porting-lessons`

`PORTING-LESSONS.md` is compiled from a full port of a ternary quantised LLM engine from CUDA to RDNA4. Read it before starting any port. It is blunt about what went wrong, and it is written in vendor neutral language, so the rules apply to CUDA, HIP, Vulkan, DirectX, Metal and SYCL alike.

- **A terminology map** at the top, which lines up queue, captured commands, execution batch, SIMD unit, lane, fast scratch, timestamp, artifact, compute unit and registers across six ecosystems. A rule stated once can then be applied anywhere.
- **Six gates** that must exist before the first edit: probe the ISA, scan for leftover foreign asm, delete the output before building, check the exported symbols, and run the model (section 1).
- **The ISA verification protocol** (section 2). A mnemonic name is not a capability. `v_dot4_i32_i8` has no gfx12 encoding and breaks the backend, while `amd_mixed_dot` compiles and emits `v_dot4_i32_iu8`. The same section records that a wave size flag was rejected by the compiler while the default was already correct, which is a reminder to verify the artifact and not the flag.
- **Six ways a source rewrite fails silently**: literal mismatch, prefix mismatch, terminator mismatch, cross line regex, whole line replacement, and ordering (section 3).
- **Two structural traps**: `__launch_bounds__(N)` breaking every signature scan, and ISA constructs hiding inside `#define` macros (section 3).
- **Shim conformance** (section 4). A drop in replacement has to match behaviour and not just names: bytes against str, zero dimensional device views, strided copies, and argument marshalling.
- **A measurement doctrine** (section 6). Four different instruments that timed the same kernel gave answers three orders of magnitude apart, so the document now records what each one really reports. A timestamp pair around a whole phase measures the host submission span whenever the host is the bottleneck. A timestamp pair around every operation measures its own instrumentation floor, which is per operation and not small. A launch followed by an immediate device sync measures the driver round trip. Only burst replay with the host time subtracted measures the kernel. The section then gives the question that settles the diagnosis (does the host finish submitting before the device finishes executing?), the invariant that the parts cannot exceed the whole, and the conditions for accepting a change at all (bit exact output, reproducible numbers, arms interleaved).
- **Verify the artifact, not the flag** (section 7). Compiler output is usually a container that can hold more than one architecture, the metadata states the real SIMD width, register counts and scratch sizes, and a probe that appears to prove six compiler flags were rejected proves nothing if the test source never compiled in the first place.
- **Port the reason, not the code** (section 7.1). A kernel that hoists tables into a scarce resource was written for that scarcity. If the new target has the instruction that made the trick unnecessary, the structure has become a cost instead of an optimisation.
- **Capture and replay semantics** (section 8). What cannot be captured, why host side bookkeeping has to be replayed by hand, why a capture is valid only for the state it was recorded in, and why a failed capture must run the work again instead of silently dropping it.
- **Measured numbers beat datasheet numbers** (section 9). On one card: 616 GB/s achieved against 1152 GB/s on paper, 64 KiB of usable scratch against 128 KiB in the table, and a runtime "compute unit" count that was really workgroup processors.
- **An engine class checklist** (section 10) for ninfer, deepinfer and llama.cpp style ports: what already runs on the target, what is genuinely custom, which runtime machinery has to be replaced, which named libraries need call level answers, and how to fix the numerical contract before touching code.

Every claim in it is tied to a measurement that was taken, not advice that was read. That includes a **retraction** of the document's own earlier performance number, because a wrong measurement left sitting in documentation is worse than no measurement at all.

---
## An Example of how it caught an issue 

<img width="1194" height="1138" alt="image" src="https://github.com/user-attachments/assets/f962677b-bee9-4699-ae58-8bedad8838d5" />

---
<img width="1186" height="1041" alt="image" src="https://github.com/user-attachments/assets/03b10023-e6ec-47b2-a105-0b08b7eb2260" />

---
## Coming Soon

We are testing two real NVIDIA CUDA to HIP ports using K_and_S on actual GPU hardware. Results and templates from those experiments will be added to the `findings/` directory so the whole community can learn from them.

---

## License

Knowledge base and tools provided as documentation and reference implementations. See individual file headers for details.

---

[^kand]: <sub><i>Kand means some scam, or a problem someone sets in motion. Kernels And Shaders accidentally ended up as Kands. As we say in Hindi, "Kand kar diya".</i></sub>
