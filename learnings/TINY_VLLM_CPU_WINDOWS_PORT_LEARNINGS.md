# tiny-vllm CPU / Windows C++ port — learnings using K_And_S toolkit

This is the K_And_S field-learning doc for porting tiny-vllm (the CUDA/HIP
C++ educational LLM inference engine) to a CPU-only Windows C++ build with
tiny-llama support. The intent is to run our porting system *against* this
specific task, capture the real errors and friction points, and fold them
back into the toolkit so next time it is sharper.

## What we tried to do

- Clone tiny-vllm into this workspace.
- Point the K_And_S `autonomous_port`-style workflow at the cloned tree,
  with a target of `cxx / linux / hip` first as a sanity baseline, and then
  with a CPU / Windows target in mind.
- Feed the actual repo contents (not just the README) into the toolkit's
  scan / compare / estimate / research / plan stages.
- Use the toolkit outputs as the skeleton of a real port attempt, even where
  we cannot yet execute a full build here.

## What the clone actually contained (real findings)

- tiny-vllm is a teaching repo, not a production server. It is explicitly a
  "younger and smaller sibling" of vLLM, and the README frames it as a
  C++/CUDA course project.
- The C++ tree already has:
  - model loading from safetensors (`model/`),
  - all the LLaMA-style layers,
  - a handful of custom CUDA kernels,
  - a HIP variant already wired through `cuda_to_hip.h` and `USE_HIP`.
- The repo does **not** ship the actual `model.safetensors` for the target
  model. That file is expected to appear at runtime from an external source
  (in particular the gated HuggingFace model).
- There is already a Python side (`python/`) with reference scripts, which is
  useful both for checking numerics and for understanding the tokenizer / model
  contract.

## What the K_And_S scan gave us

- Kernel discovery works as expected on this tree: GPU kernels, headers, and
  the HIP shim all show up cleanly from extension + path heuristics.
- The toolkit's arch comparison is the part that matters most here: tiny-vllm's
  kernels are written for NVIDIA-ish CUDA semantics first, and the HIP shim is
  the bridge the repo already provides. That means the "CUDA -> CPU" jump is
  bigger than the "CUDA -> HIP" jump the repo already anticipates.
- The toolchain resources are most useful when we stop thinking "GPU backend
  only" and instead treat the host toolchain matrix as the first-class part of
  the port: Windows C++ means MSVC paths, Windows SDK includes, and whatever
  BLAS runtime we choose on the host side.

## The real porting challenges this surfaced

- **BF16 is the tripwire.** tiny-vllm is built around `__nv_bfloat16` and
  cuBLAS BF16 gemms. On x86 CPU targets, there is no drop-in native BF16 that
  keeps the same performance shape. You either:
  - emulate BF16 in software and accept the slowdown,
  - widen to FP32 for the host side and treat BF16 as a storage/IO format only,
  - or restrict the CPU port to a target where BF16 is more natural.
  This is exactly the kind of thing the toolkit's arch matrix is supposed to
  catch early, and it does: the mismatch is obvious once you read the model +
  kernel requirements against the target host capabilities.
- **cuBLAS is the second tripwire.** The matmul path is not just a few custom
  kernels; a lot of the heavy lifting is delegated to cuBLAS. Porting to CPU
  means choosing a host BLAS and deciding whether it can even express the same
  BF16 semantics. That is not a kernel rewrite problem; it is a backend
  selection problem, and the toolkit's toolchain resources are the right place
  to reason about it.
- **HuggingFace gated model access is a real, not theoretical, blocker.**
  The model tiny-vllm targets lives behind a license acceptance gate. That
  affects everything downstream: can we get the weights at all, can we automate
  the download, can we keep the port reproducible without manual clicks.
- **Windows C++ is not "just MSVC + cl.exe".** Even before GPU concerns, there
  are real toolchain choices: MSVC vs MinGW vs MSYS2, static vs dynamic C++
  runtime, BLAS DLL linking, aligned allocation for SIMD, path encoding, and
  whether the BLAS you pick even ships a sane Windows binary. The toolkit's
  host toolchain matrix is the right lens here.
- **The CPU port is less about "rewrite kernels" and more about "rewrite the
  execution model."** PagedAttention and continuous batching exist to manage
  scarce GPU VRAM and latency. On CPU with enough RAM, a simpler contiguous KV
  cache and single-stream generation is often the better first target. That
  means the port is not a literal kernel-by-kernel translation; it is a
  structural simplification, which the toolkit's porting-strategy layer should
  make explicit.

## How this should change the toolkit workflow

- For CPU-targets, the first question should not be "which GPU backend?" but
  "is the compute contract even portable without a major emulation penalty?"
- The auto-port report should separate:
  - kernel portability,
  - BLAS / matmul portability,
  - numeric format portability,
  - host toolchain portability,
  - model-weight availability,
  - tokenizer / reference-grounding availability.
- When the source already has a HIP shim, the toolkit should prefer that as the
  bridge for GPU-side reasoning, and only then discuss CPU as a separate target
  class with its own BLAS + format decisions.

## What still needs to be done to make this real

- Get the actual model weights in hand in a way that does not depend on manual
  browser clicks each time.
- Decide on the CPU BLAS / numeric-path strategy for the chosen host target.
- Use the Python reference side to establish a numeric baseline before touching
  the C++.
- Build the smallest possible CPU path first: load weights, run one forward
  pass, compare against reference, and only then add tokenizer and generation.

## Status

Partial. The repo is in hand, the K_And_S resources can see it, and the major
blockers are now explicit. The next concrete step is to turn this into a small,
actually buildable CPU path with a real reference comparison, rather than a
GPU-porting report in GPU clothing.
