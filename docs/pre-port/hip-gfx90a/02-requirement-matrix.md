# Requirement Matrix
Pareto total score: 24.4
Vital few: ['toolchain.hipcc', 'capability.matrix', 'artifact.hsaco', 'capability.dot4', 'toolchain.objdump', 'capability.wave', 'pcie.rebar', 'runtime.finegrained']

| # | ID | Kind | I×L | Score | Cum. | Status | Probe | Fallback |
|---|---|---|---|---|---|---|---|---|
| 1 | `toolchain.hipcc` | tool | 5.0×0.9 | 4.5 | 18% | UNVALIDATED | hipcc --version | build the hsaco on a Linux ROCm host and ship the code object; never hand-encode ISA |
| 2 | `capability.matrix` | capability | 5.0×0.6 | 3.0 | 31% | UNVALIDATED | probe_isa(wmma/mfma candidate) | DP4a or scalar FMA reconstruction; a matrix kernel restructured is a design change, not a flag |
| 3 | `artifact.hsaco` | artifact | 4.0×0.7 | 2.8 | 42% | NOT RUN | code_object_report(path, arch) | a module that compiles but lacks the called symbol is a failure; check the symbol inventory |
| 4 | `capability.dot4` | capability | 4.0×0.6 | 2.4 | 52% | UNVALIDATED | probe_isa(amd_mixed_dot) | raw v_dot4_i32_i8 may have no gfx12 encoding; use amd_mixed_dot, else scalar expand |
| 5 | `toolchain.objdump` | tool | 4.0×0.6 | 2.4 | 62% | UNVALIDATED | roc-objdump --version | use llvm-objdump from the LLVM install; a build that cannot be inspected is not verified |
| 6 | `capability.wave` | info | 4.0×0.5 | 2.0 | 70% | NOT RUN | compare_archs(source, target) | rewrite warp intrinsics for the target width; do not pass a wave-size flag and assume it applied |
| 7 | `pcie.rebar` | pcie | 4×0.5 | 2.0 | 78% | UNVALIDATED | hipHostMalloc/hipHostGetDevicePointer (fine-grained, host-visible) + hipMemAdvise | page-locked staging ring + chunked hipMemcpyAsync, double-buffered; keep hot tensors resident in device memory |
| 8 | `runtime.finegrained` | manual | 4.0×0.5 | 2.0 | 86% | NOT RUN | run one host-mapped copy on target | fall back to coarse-grained + explicit copies; record which path ran |
| 9 | `host.windows_rocm` | os | 3.0×0.6 | 1.8 | 94% | PASS | hipcc --version on the build host | build on Linux and ship the code object, or use the Windows ROCm build that is actually installed |
| 10 | `toolchain.rocminfo` | tool | 3.0×0.5 | 1.5 | 100% | UNVALIDATED | rocminfo | read the PCI id and map it to gfx manually, then record it |