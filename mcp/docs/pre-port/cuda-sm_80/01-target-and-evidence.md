# Target and Evidence: cuda → sm_80

**Target arch:** `sm_80`
**Host OS:** `windows`

**Facts provenance:**
- `toolchain.nvcc`: CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4; declared root exists
- `toolchain.cuobjdump`: found `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\cuobjdump.EXE`
- `capability.matrix`: ARCH_MAP.json sm_80.tensor_cores = True
- `runtime.graphs`: manual: capture one launch graph
- `memory.managed`: manual: one oversubscribed allocation
- `artifact.cubin`: requires a build: code_object_report(path, arch)
- `pcie.rebar`: no machine-readable ReBAR source on windows

*`ARCH_MAP.json` is one source among four, ranked below measurement, below a shared feed, and below user assertion.*
