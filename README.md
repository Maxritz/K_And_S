<img width="960" height="410" alt="image" src="https://github.com/user-attachments/assets/8471da74-e628-4a74-b92d-b647e532d95d" />

**Kernels & Shaders**

**Many architectures. Endless debugging.**

---


**K_and_S** (Kernels and Shaders) is an MCP server that helps you port GPU kernels and shaders across different hardware vendors and APIs. Instead of guessing how to translate a CUDA kernel to HIP for AMD, or how to get the best performance out of RDNA on Vulkan, K_and_S gives you the knowledge, patterns, and tools to do it properly.

---

## Why This Exists

Modern GPU development is complicated. You have NVIDIA with CUDA, AMD with HIP and ROCm, Intel with oneAPI and SYCL, and then there is Vulkan and DirectX for cross-platform compute. Most developers stick to one vendor because porting is hard. When you do try to port, you end up copying code blindly, hoping intrinsics translate, and debugging mysterious performance regressions.

K_and_S was built to fix that. It started as a collection of notes from real kernel porting work, documenting what actually works on real hardware. We tested on AMD RDNA2 and RDNA4 (gfx1031, gfx1201), and later will test on Intel Core Ultra 5 135H (meteor_lake). We learned which patterns fail, which optimizations actually help, and which myths waste hours of development time.

This is not just a tool. It is a knowledge base that travels with you. When you fix a kernel on one project, you can save the solution as a template and reuse it on the next project. When the team grows, the knowledge stays.

---

## What You Can Do With It

### Discover and Audit
Point K_and_S at any codebase. It will scan every `.cu`, `.hip`, `.comp`, `.hlsl`, `.cpp` file and tell you exactly what GPU code exists and what backend it targets. No more hunting through thousands of files.

### Check Compatibility
Before you start porting, see if your target architecture supports everything your source needs. Does your target have tensor cores? DP4a? Native BF16? Will Vulkan cooperative matrices work (they do not on AMD, use DP4a instead). K_and_S tells you upfront so you do not waste weeks on a dead end.

### Inventory before you port — the pre-port reference
Do not start from "where do I begin". Start from a complete list of what the tree actually depends on.

`build_function_reference` scans the sources and produces the vendor-function reference for your target: every CUDA runtime call, driver call, inline-PTX instruction, device intrinsic and framework feature that is genuinely used, each with

- the replacement on the target,
- a status — **Auto** (compiler maps it), **Partial** (maps but differs), **Manual** (source must change) or **No equivalent** (restructure),
- the call sites and the files that depend on it.

Status resolves **per architecture**, because the answer changes within a vendor family: `mma.sync` is manual on every AMD target, but MFMA covers CDNA, WMMA covers RDNA3/RDNA4, and RDNA2 has no matrix core at all — the table says so rather than flattening it into "manual".

Anything with **no equivalent** is reported as a hard blocker. A blocker is not a partial gap: the kernel must be redesigned or dropped, and that decision belongs before the first line of code, not after.

`lookup_api` gives the same answer for a single symbol while you are reading code.


Capability tables lie. A real port was designed around `v_dot4_i32_i8`; that mnemonic has **no gfx12 encoding** and crashes the LLVM backend rather than failing at the assembler. Its replacement, `amd_mixed_dot`, compiles fine and emits `v_dot4_i32_iu8` — same hardware, different spelling.

`probe_isa` compiles each candidate **for your actual target**, one translation unit per candidate so a single failure cannot mask the rest, and returns both the verdict and the disassembled mnemonics. That is the only reliable oracle for "does this instruction exist on this chip".

### Build a function-to-function porting map
`build_porting_map` reads your source and, for every kernel/device function, reports:

- the constructs it uses (inline-asm mnemonics, warp intrinsics, barriers, local memory, id builtins)
- the **substitution** for your target, with an explicit fidelity grade: `exact`, `portable`, `approximate`, `lossy`, `manual` or `drop`
- a caveat where one applies (e.g. a matrix instruction that can only be reconstructed element-wise)

It understands CUDA/HIP, SYCL, OpenCL, GLSL, HLSL and Metal, and it follows ISA constructs that hide inside `#define` macros — otherwise code built entirely from an unsupported instruction reports as clean.

It also emits the **extension-to-extension file plan** for the ported tree, so you know which files are rewritten and which are copied:

| source | → HIP | → CUDA | → SYCL | → Vulkan | → DirectX |
|---|---|---|---|---|---|
| `.cu` (CUDA) | `.hip` | `.cu` | `.cpp` | `.comp` | `.hlsl` |
| `.cl` (OpenCL) | `.cpp` | `.cu` | `.cpp` | `.comp` | `.hlsl` |
| `.comp` (GLSL) | `.cpp` | `.cu` | `.cpp` | `.comp` (copy) | `.hlsl` |
| `.hlsl` | `.cpp` | `.cu` | `.cpp` | `.comp` | `.hlsl` (copy) |
| `.metal` | `.cpp` | `.cu` | `.cpp` | `.comp` | `.hlsl` |
| SYCL `.cpp` | `.cpp` | `.cu` | `.cpp` (copy) | `.comp` | `.hlsl` |

`verify_porting_map` then closes the loop: it probes every mapped *replacement* on the real target and downgrades anything that does not encode.

### Gate the build
`scan_residual_asm` finds live `asm("…")` strings that still name a foreign mnemonic (comments are ignored, so no false positives), and `build_module` deletes the output before compiling, then reports size plus the exported kernel symbols compared against what you expect. A module that compiles but lacks a kernel the runtime calls is not a successful build — and a stale artifact must never look like a fresh one.

### Find Installed Toolchains
Instead of reading docs and guessing where hipcc or nvcc lives on your machine, K_and_S checks at runtime. It looks in your PATH, checks your environment variables (ROCM_PATH, CUDA_PATH, VULKAN_SDK, ONEAPI_ROOT), and scans common install locations. It even detects Strawberry Perl, which you need for some build scripts alongside MSVC.

### Estimate Performance
Got a target architecture and a workload (say, flash attention at 8192 sequence length)? K_and_S uses roofline models to tell you the theoretical ceiling and what real developers achieve in practice. You get a target tokens per second number before you write a single line of code.

### Reject Common Myths
Before you follow the usual advice, check if it is actually wisdom or just cargo culting. Is hipify-perl enough to port a CUDA project to HIP? No. Can you copy CUDA code to HIP verbatim? No. Does Vulkan CooperativeMatrix work on AMD? No. K_and_S has a tool that calls out these patterns as wrong and explains the correct approach for your specific architecture.

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
6. Save successful approaches as templates for the whole team

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

```bash
cd mcp
pip install -e .
```

### Configure Your MCP Client

Use the included `claude_desktop_config.json` or add to your client config:

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

# 5. Verify every instruction the port depends on — before designing anything
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

# 9. Port the full repo
autonomous_port(repo_path="/path/to/kernel/code", source_arch="sm_80", target_arch="gfx1201")

# 10. Verify correctness after each change
verify_numerical_equivalence(arch="gfx1201", workload="gemm", source_values=[...], target_values=[...])

# 11. Record what you learned
record_performance(arch="gfx1201", workload="gemm", tok_per_s=42500.0, notes="Used DPP shuffle, wave32")

# 12. Save as a template for next time
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

`PORTING-LESSONS.md` is a resource distilled from an end-to-end port of a ternary-quantised LLM engine from CUDA to RDNA4. Read it before starting any port. It is deliberately blunt about what failed:

- **Nine gates** that must exist before the first edit (probe the ISA, scan for residual foreign asm, delete artifacts, assert the symbol inventory, run the model)
- **The ISA verification protocol**, including the measured table showing that a mnemonic name is not a capability — `v_dot4_i32_i8` fails while `amd_mixed_dot` emits `v_dot4_i32_iu8`
- **Six ways a source rewrite silently fails**: literal mismatch, prefix mismatch, terminator mismatch, cross-line regex, whole-line replacement, ordering
- **Two structural traps**: `__launch_bounds__(N)` corrupting every signature scan, and ISA constructs hiding inside `#define` macros
- **Shim conformance**: a drop-in replacement must match *behavioural* contracts (bytes vs str, 0-d device views, strided copies, argument marshalling)
- **A performance method** that separates host launch, queue wait, device execution and sync — and the finding that 331 launches/token cost more than the arithmetic ever did
- **An engine-class checklist** for ninfer/deepinfer/llama.cpp-style ports: what already runs on the target, what is genuinely custom, which runtime machinery must be replaced, which named libraries need call-level answers, and how to establish the numerical contract

Every claim in it is tied to a measurement that was taken, not to advice that was read.

---

## Coming Soon

We are testing two real NVIDIA CUDA to HIP ports using K_and_S on actual GPU hardware. Results and templates from those experiments will be added to the `findings/` directory so the whole community can learn from them.

---

## License

Knowledge base and tools provided as documentation and reference implementations. See individual file headers for details.
