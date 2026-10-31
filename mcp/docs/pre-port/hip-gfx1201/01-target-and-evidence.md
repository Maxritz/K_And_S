# Target and Evidence: hip → gfx1201

**Target arch:** `gfx1201`
**Host OS:** `linux`

**Facts provenance:**
- `toolchain.hipcc`: ROCM_PATH=G:\ROCM10RT-gfx1201; declared root exists
- `toolchain.rocminfo`: none of rocminfo, rocminfo.exe on PATH
- `toolchain.objdump`: found `C:\Users\rr\AppData\Local\Programs\Swift\Toolchains\6.3.3+Asserts\usr\bin\llvm-objdump.EXE`
- `capability.dot4`: ARCH_MAP.json gfx1201.dot4 = True
- `capability.matrix`: ARCH_MAP.json gfx1201.wmma = True
- `capability.wave`: ARCH_MAP.json gfx1201.wave_size = 32
- `runtime.finegrained`: manual: run one host-mapped copy on target
- `artifact.hsaco`: requires a build: code_object_report(path, arch)
- `host.windows_rocm`: host is `linux`
- `pcie.rebar`: no machine-readable ReBAR source on linux

*`ARCH_MAP.json` is one source among four, ranked below measurement, below a shared feed, and below user assertion.*
