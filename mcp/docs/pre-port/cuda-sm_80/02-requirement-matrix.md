# Requirement Matrix
Pareto total score: 15.1
Vital few: ['toolchain.nvcc', 'artifact.cubin', 'capability.matrix', 'pcie.rebar', 'runtime.graphs']

| # | ID | Kind | I×L | Score | Cum. | Status | Probe | Fallback |
|---|---|---|---|---|---|---|---|---|
| 1 | `toolchain.nvcc` | tool | 5.0×0.9 | 4.5 | 30% | PASS | nvcc --version | match the toolkit to the driver; a newer toolkit than the driver supports produces a load failure, not a compile error |
| 2 | `artifact.cubin` | artifact | 4.0×0.6 | 2.4 | 46% | NOT RUN | code_object_report(path, arch) | assert the exported kernel list before trusting a successful build |
| 3 | `capability.matrix` | capability | 4.0×0.5 | 2.0 | 59% | PASS | probe_isa(mma candidate) | fp32 FMA path; keep it selectable, not a silent downgrade |
| 4 | `pcie.rebar` | pcie | 4×0.5 | 2.0 | 72% | UNVALIDATED | cudaHostAllocMapped / cudaHostRegister + cudaHostGetDevicePointer | pinned staging buffers + chunked cudaMemcpyAsync, hot tensors resident |
| 5 | `runtime.graphs` | manual | 3.0×0.5 | 1.5 | 82% | NOT RUN | capture one launch graph | rebuild with events; a failed capture must re-run the work, not drop it |
| 6 | `toolchain.cuobjdump` | tool | 3.0×0.5 | 1.5 | 92% | PASS | cuobjdump --version | nvdisasm on the extracted cubin |
| 7 | `memory.managed` | manual | 3.0×0.4 | 1.2 | 100% | NOT RUN | one oversubscribed allocation | explicit device/host copies with a residency plan |