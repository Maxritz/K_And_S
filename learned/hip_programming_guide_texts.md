# Hip Programming Guide-Texts

_Source: Distilled from PDF documentation_


## HIP


## PROGRAMMING GUIDE


```cpp

A structured engineering guide to the HIP C++ runtime and GPU kernel model

```

Based on the ROCm/hip project
Repository: github.com/ROCm/hip
Branch checked: develop
Prepared: 31 October 2026
Host runtime  Kernels  Memory  Streams  Graphs  Porting  Performance  Debugging
HIP Programming Guide | Upstream source: ROCm/hip
2
How to use this guide
Follow the chapters in order for a first implementation. Use the quick sheet when coding, and consult the
version-matched official HIP documentation for exact signatures, availability and platform constraints.
Contents
1. Orientation: what HIP is
2. Installation, toolchain and first build

```cpp

3. HIP execution model and kernel anatomy

```

4. Runtime lifecycle, device selection and errors
5. Memory management
6. Streams, events and asynchronous execution
7. Cooperative groups and synchronisation
8. Graph execution
9. Kernel language extensions and C++ support
10. CUDA-to-HIP porting strategy
11. Performance engineering
12. Debugging, logging and observability
13. Device properties and feature detection
14. Runtime compilation and external interoperability
15. A practical development workflow
16. Reference quick sheet
17. Versioning, support boundaries and source map
HIP Programming Guide | Upstream source: ROCm/hip
3
1. Orientation: what HIP is

```cpp

HIP is a C++ runtime API and kernel language for GPU programming. It provides a source-level programming model for
host code and device kernels, with implementations for AMD ROCm and NVIDIA CUDA platforms. It is CUDA-like, but it

```

is not a promise of automatic drop-in compatibility or identical performance.

## A HIP application typically contains


```cpp

 host-side C++ code that selects devices, allocates buffers, transfers data, launches kernels and synchronises work;
 device-side kernels compiled for the selected GPU architecture;

```

 runtime and compiler components supplied by the platform toolchain.
The important distinction is between source portability and performance portability. HIP can make a common source
base practical, but code still needs validation on each target, especially for architecture-specific features, memory
behaviour, atomics, launch limits and compiler support.
Repository branch note. The upstream repository identifies develop as the active development branch and
describes main/release branches as stable release lines. Do not treat an unpinned develop checkout as a release
contract. For production builds, pin the ROCm release and compiler/runtime versions.
2. Installation, toolchain and first build
Toolchain roles

```cpp

 HIP source/API: headers, runtime declarations and kernel-language extensions.

```

 hipcc: compiler driver that chooses the backend toolchain and passes appropriate include/library options.
 HIP-Clang: AMD-side compiler path used with ROCm.
 CUDA toolkit / nvcc: NVIDIA-side compiler path for HIP source targeting NVIDIA.
 CMake: project configuration and target wiring.
Establish the environment before building
Record the operating system, GPU model, driver, ROCm release, HIP package version, compiler version, CMake
version and target architecture. Run hipconfig --full and preserve the output with build logs. A successful compile
does not establish that the executable can run on the intended GPU.
Typical inspection commands (exact package names and paths vary by distribution and ROCm release):
hipconfig --full
hipcc --version
rocminfo
which hipcc
printenv | grep -E '^(HIP|ROCM|HSA|GPU|AMD)'
Minimal CMake project
cmake_minimum_required(VERSION 3.21)
project(hip_first LANGUAGES CXX)
find_package(hip CONFIG REQUIRED)
add_executable(hip_first main.cpp)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(hip_first PRIVATE hip::host)
Configure with the target architecture explicitly where supported by the installed ROCm toolchain. Verify the installed
package's CMake config and examples for the exact compiler and architecture-selection syntax. Do not copy flags from
a different ROCm generation without checking them.
Build checklist
Confirm the GPU is visible to the driver/runtime.
HIP Programming Guide | Upstream source: ROCm/hip
4
Confirm the compiler can discover the intended GPU architecture.
Configure a clean build directory.
Build with verbose output once, so compiler invocations are inspectable.
Run a device enumeration sample.

```cpp

Run a memory-copy and kernel-launch smoke test.

```

Capture hipconfig --full, compiler output and runtime errors in the build record.

```cpp

3. HIP execution model and kernel anatomy

```

A GPU launch is a grid of thread blocks. A block contains threads that cooperate through block-scoped shared memory
and synchronisation. The runtime schedules blocks onto GPU compute resources; a programmer should not assume a
particular block-to-compute-unit mapping.

## Common built-ins include

Built-in
Meaning

```cpp

threadIdx

```

Thread coordinate inside its block
blockIdx
Block coordinate inside the grid
blockDim
Dimensions of a block
gridDim
Dimensions of the grid
A grid-stride loop supports arbitrary problem sizes and can decouple the number of launched threads from the number

## of elements


```cpp

#include <hip/hip_runtime.h>
__global__ void square_kernel(float* out, const float* in, size_t n) {
const size_t first = blockIdx.x * blockDim.x + threadIdx.x;

```

const size_t stride = blockDim.x * gridDim.x;
for (size_t i = first; i < n; i += stride) {
out[i] = in[i] * in[i];
}
}
Launch configuration uses a grid dimension, block dimension, dynamic shared-memory byte count and stream. Keep

```cpp

launch dimensions validated against device and kernel limits. The runtime launch macro commonly used in HIP

```

examples is hipLaunchKernelGGL.
const unsigned threads = 256;
const unsigned blocks = (static_cast<unsigned>(n) + threads - 1) / threads;

```cpp

hipLaunchKernelGGL(square_kernel, dim3(blocks), dim3(threads), 0, stream,

```

d_out, d_in, n);
The arithmetic in the example is illustrative. For very large n, compute the block count in a wide integer type and clamp
it to a suitable launch size, relying on the grid-stride loop for the remainder.
4. Runtime lifecycle, device selection and errors
A disciplined program makes device selection and runtime failures explicit.

## Recommended lifecycle

Query the device count.
Select a device intentionally, rather than relying on an implicit default in a multi-GPU process.
Query device properties and validate required capabilities.
Create streams/events and allocate required memory.
HIP Programming Guide | Upstream source: ROCm/hip
5

```cpp

Submit transfers and kernels.

```

Synchronise at well-defined boundaries and check asynchronous errors.
Release resources in reverse order.
Use hipGetDeviceCount, hipSetDevice, hipGetDevice, hipGetDeviceProperties, and

```cpp

hipGetLastError/hipPeekAtLastError as appropriate. Check every HIP API status. A kernel launch can return
before the GPU has executed the kernel; an execution error may surface only at a later synchronisation or event query.

## A practical error macro

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

```

#define HIP_CHECK(expr) do {                                      \
hipError_t e_ = (expr);                                       \
if (e_ != hipSuccess) {                                       \
std::fprintf(stderr, "%s:%d: %s failed: %s\\n",          \
__FILE__, __LINE__, #expr, hipGetErrorString(e_));    \
std::exit(EXIT_FAILURE);                                  \
}                                                              \
} while (0)
For debugging, check launch/configuration errors immediately after a launch and synchronise at a diagnostic boundary

```cpp

to expose asynchronous execution errors. Avoid adding a global device synchronisation after every kernel in a

```

performance build.
5. Memory management
Device memory

```cpp

hipMalloc allocates device-accessible storage; hipFree releases it. hipMemcpy performs synchronous copies for
the selected direction. Use hipMemcpyAsync with a stream when the host/device memory and runtime conditions

```

permit asynchronous operation.
Host memory
Ordinary pageable host memory may require staging before a transfer. Pinned/page-locked host allocations can
improve transfer behaviour and enable asynchronous DMA paths, but they consume a constrained system resource.
Use them selectively, pool them where practical, and release them correctly.
Unified/managed memory
Managed allocations simplify addressability, but they do not remove the cost of data movement or locality. Measure
migration and page-fault behaviour. On discrete GPUs, explicit placement and transfer may produce more predictable
performance for latency-sensitive code.
Virtual memory and stream-ordered allocation
The newer virtual-memory APIs separate physical allocation, mapping and access permissions, enabling specialised
allocators and large virtual address layouts. Stream-ordered allocation APIs make allocation/free operations participate
in stream ordering, which can reduce global synchronisation and support asynchronous memory pools. These facilities
are version- and platform-dependent; confirm availability in the headers installed with the target ROCm release.
Memory correctness rules
 Track allocation size, ownership and lifetime explicitly.

```cpp

 Do not free a buffer while an in-flight kernel or transfer can still access it.

```

 Make stream dependencies explicit when a buffer crosses streams.
 Avoid host reads before device writes are complete.
HIP Programming Guide | Upstream source: ROCm/hip
6
 Check alignment and element-size arithmetic for overflow.
 Treat host/device pointer accessibility as a capability to verify, not an assumption.
6. Streams, events and asynchronous execution
A stream is an ordered queue of GPU work. Commands submitted to the same stream execute in order. Different
streams may overlap, but cross-stream ordering must be expressed with events or other supported dependency
mechanisms.

## Typical patterns

hipStream_t stream{};
HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

```cpp

HIP_CHECK(hipMemcpyAsync(d_in, h_in, bytes, hipMemcpyHostToDevice, stream));
hipLaunchKernelGGL(square_kernel, grid, block, 0, stream, d_out, d_in, n);
HIP_CHECK(hipMemcpyAsync(h_out, d_out, bytes, hipMemcpyDeviceToHost, stream));

```

HIP_CHECK(hipStreamSynchronize(stream));
HIP_CHECK(hipStreamDestroy(stream));
Use events to timestamp GPU work and create dependencies between streams. Avoid assuming that default-stream
semantics are identical across all configurations and platforms. Prefer explicit streams for libraries and subsystems that
must compose predictably.

```cpp

Asynchronous copies require suitable memory and valid lifetime management. hipMemcpyAsync alone does not

```

guarantee useful overlap. To obtain overlap, consider pinned host buffers, separate streams, device copy-engine
capability, enough independent work and non-conflicting memory dependencies.
For a pipelined workload, divide the input into chunks. While chunk N computes, transfer chunk N+1, provided the GPU
and memory system can execute both activities concurrently and the dependencies are correct.
7. Cooperative groups and synchronisation
Cooperative groups provide abstractions for groups of threads and collective operations beyond simple block-wide
primitives. Use them when the algorithm needs a well-defined participating group, group synchronisation or collective
operations.
The most important correctness rule is participation: all threads required by a collective must reach it in a convergent
manner. Divergent control flow around a barrier or collective can deadlock or produce undefined behaviour. Do not
assume that a block barrier synchronises different blocks. Grid-wide cooperation requires the appropriate
cooperative-launch support and its residency constraints.
Choose the narrowest synchronisation scope that correctly represents the algorithm. Warp/subgroup, block and
grid-level synchronisation have different costs and constraints. Validate the target architecture's supported features and

```cpp

the kernel's launch requirements.

```

8. Graph execution
HIP graph APIs represent a workflow as nodes and dependencies. A graph can reduce repeated host submission
overhead when the same dependency structure is replayed many times. Graphs are especially relevant for repeated

```cpp

multi-kernel pipelines, static transfer/compute sequences and workloads where CPU launch overhead is material.

## A practical graph workflow


```

Identify a repeated sequence and establish a non-graph baseline.
Capture or explicitly construct the sequence.
Instantiate the graph executable.
Replay it repeatedly with valid input/output buffers.
HIP Programming Guide | Upstream source: ROCm/hip
7
Update graph node parameters when supported and when doing so is cheaper than rebuilding.
Measure end-to-end throughput and latency, not merely graph-launch time.

```cpp

Graphs do not make a slow kernel fast. They primarily address submission and orchestration costs. The graph API's

```

available features and capture restrictions vary by release; consult the installed API reference.
9. Kernel language extensions and C++ support
HIP device code uses C++ with device attributes and built-ins. The language extension reference defines

```cpp

execution-space attributes such as __global__, __device__ and __host__, plus address-space, vector, atomic,

```

math and compiler-control features.

## Practical guidance

 Keep host-only headers and implementation details out of device compilation paths where possible.
 Use templates carefully; compile-time specialisation can remove runtime branches but can also increase code size.
 Verify which standard-library facilities are supported in device code for the specific compiler release.
 Do not include backend-internal headers such as implementation detail directories directly in portable application
code.
 Treat architecture intrinsics as specialised fast paths with a tested fallback.
The upstream reference is authoritative for exact attribute semantics and per-platform support. The portable subset
should be the default; backend-specific extensions should be isolated behind small interfaces.
10. CUDA-to-HIP porting strategy
HIPIFY can automate many mechanical CUDA source transformations, but a successful port is a staged engineering
task, not a text replacement exercise.
Recommended sequence
Inventory CUDA APIs, language extensions, inline PTX, libraries, build scripts and assumptions.
Establish a clean baseline on the original platform with correctness tests and performance measurements.
Run HIPIFY on a branch, not on the only copy of the source.
Compile with hipcc and resolve syntax/API gaps.
Replace unsupported libraries or backend-specific code with portable alternatives.
Validate memory ownership, stream ordering, atomics and synchronisation.
Run deterministic correctness and numerical-tolerance tests.
Profile on the target GPU and optimise the actual bottleneck.
Common sources of manual work
 CUDA-specific libraries and APIs that have no direct HIP equivalent;
 inline PTX or architecture-specific assembly;
 assumptions about warp width, scheduling or occupancy;
 different support for atomics and floating-point operations;
 stream/default-stream behaviour;
 compiler flags, build systems and device architecture selection;
 numerical differences arising from instruction selection or reduction order.
HIP Programming Guide | Upstream source: ROCm/hip
8
Do not equate successful HIPIFY output with a complete port. Keep a porting ledger with each changed API, reason,
test and performance impact. Isolate platform specialisations with narrow wrappers and compile-time selection.
11. Performance engineering

```cpp

Optimisation should begin with measurement. Separate host submission time, kernel execution time, transfer time,

```

allocation overhead and synchronisation. Use a profiler and timeline to identify whether the workload is compute-bound,
bandwidth-bound, latency-bound, occupancy-limited or launch-bound.
Core principles
 Minimise unnecessary host/device transfers.
 Coalesce global-memory accesses where the algorithm permits.
 Reuse data through registers, shared memory or caches when reuse offsets the extra resource cost.
 Avoid register pressure and shared-memory usage that reduce occupancy without sufficient benefit.

```cpp

 Select launch dimensions empirically for the target architecture and kernel.
 Fuse kernels when saved launch/traffic cost outweighs register and code-size costs.

```

 Use asynchronous copies and multiple streams only when the timeline shows real overlap.
 Benchmark warm and cold paths separately.
Timing correctly
Use HIP events for device-side elapsed time. For end-to-end timing, use a host clock around the full sequence and

```cpp

synchronise at the measurement boundary. Do not measure an asynchronous launch and label the result as kernel

```

execution time. Run enough iterations, include warm-up, report distributions, and avoid profiling instrumentation when
collecting final throughput numbers if it perturbs the workload.
Reproducibility record
Record GPU and architecture, driver/runtime version, compiler flags, source revision, input dimensions, data type,
launch configuration, warm-up count, iteration count, timing method and whether transfers are included. Compare like
with like.
12. Debugging, logging and observability

```cpp

Start by reducing the failing case. Check API return codes, kernel launch errors and asynchronous execution errors.

```

Use sanitising/debugging tools supported by the installed ROCm release, and preserve compiler output and runtime
diagnostics.

## A useful diagnostic sequence

Verify the process sees the expected GPU.
Verify the runtime and compiler versions match the intended installation.

```cpp

Run a minimal allocation/copy/kernel smoke test.

```

Check all API statuses.
Add a synchronisation boundary after the suspect operation.

```cpp

Reduce to one stream and one kernel, then reintroduce concurrency.

```

Validate bounds, alignment, data races and lifetime.
Compare CPU reference results with a defined numerical tolerance.
Re-enable optimisation flags one at a time if the failure is compiler- or optimisation-sensitive.
HIP Programming Guide | Upstream source: ROCm/hip
9
HIP logging and environment controls are release-dependent. Use only variables documented by the exact installed
version. Capture hipconfig --full, GPU information, application logs and a minimal reproducer when reporting an
issue.
13. Device properties and feature detection
Never hard-code assumptions based on a GPU product name alone. Query device properties at runtime and test the
exact feature the application requires. Properties such as compute capability-style identifiers, memory size,
multiprocessor/compute-unit counts, limits, concurrent execution and memory-pool support must be interpreted
according to HIP's documented structures and the target release.

## Feature detection should be explicit

 query the device and relevant attributes;
 verify that a feature is supported on the selected runtime/platform;
 compile optional code paths only when the toolchain supports them;
 retain a fallback path where portability is required;
 test the fallback and specialised path independently.
14. Runtime compilation and external interoperability

```cpp

HIP runtime compilation and external interoperability are advanced topics. Runtime compilation is useful when kernels

```

must be specialised or compiled dynamically, but it adds compilation latency, cache/version management, diagnostic
complexity and compatibility constraints. Cache generated binaries with keys that include source, options,
compiler/runtime version and target architecture.
External interoperability may connect HIP with graphics APIs or other runtimes. Such paths require explicit ownership
and synchronisation rules: determine who owns the resource, when it is mapped/unmapped, and which API establishes
completion. Never assume that a pointer or handle can be used concurrently by both APIs without a documented
interop contract.
15. A practical development workflow

## For an engineering project, adopt the following repeatable workflow

Design: Define target GPUs, operating systems, ROCm release, numerical requirements, throughput/latency goals and
fallback behaviour.

```cpp

Prototype: Implement a small kernel and CPU reference. Prove device selection, allocation, transfer, launch and

```

synchronisation.
Integrate: Introduce streams, events, memory pools and graphs only when they serve a measured need. Wrap HIP
calls behind project-level utilities for error handling and allocation ownership.
Validate: Test boundary sizes, non-multiple block sizes, zero/one element cases, large dimensions, multiple streams
and multiple devices where relevant. Compare numerical outputs with tolerances justified by the algorithm.
Optimise: Profile, identify the dominant cost, change one variable, and remeasure. Maintain a performance regression
suite per architecture.
Ship: Pin toolchain versions, record required runtime/driver versions, package the correct binaries, and include a
diagnostic mode that prints device/runtime/compiler information.
HIP Programming Guide | Upstream source: ROCm/hip
10
16. Reference quick sheet
Task
Common HIP API / facility
Enumerate GPUs
hipGetDeviceCount
Select a GPU
hipSetDevice
Query device
hipGetDeviceProperties
Allocate device memory

```cpp

hipMalloc

```

Free device memory
hipFree
Copy data

```cpp

hipMemcpy, hipMemcpyAsync

```

Create stream
hipStreamCreate, hipStreamCreateWithFlags
Wait for stream
hipStreamSynchronize
Create event
hipEventCreate
Record event
hipEventRecord
Wait for event
hipEventSynchronize

```cpp

Launch kernel

```

hipLaunchKernelGGL or supported launch syntax
Query last error
hipGetLastError, hipPeekAtLastError
Runtime/device info
hipconfig, rocminfo
This is a navigation aid, not a complete API signature reference. Use the official API documentation corresponding to
the installed ROCm release.
17. Versioning, support boundaries and source map
The upstream repository's develop branch is useful for current development documentation but may include
unreleased behaviour. Release branches and tags are the correct starting point for reproducible production work. The
API surface, compiler flags, environment variables, feature availability and supported hardware can change between
ROCm releases.
Official source: https://github.com/ROCm/hip
Official published documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/

## Relevant repository areas

 docs/install/: installation and build instructions;
 docs/how-to/: runtime, language, porting, debugging and performance guides;
 docs/tutorial/: worked programming examples and patterns;
 docs/understand/: programming model, compiler, hardware and optimisation explanations;
 docs/reference/: API, environment variables, feature and error references;
 docs/data/: figures and supporting assets.
The official repository states that HIP is not intended as a drop-in CUDA replacement and that manual coding and
performance tuning may be needed. The examples and API descriptions in this guide should therefore be treated as a
learning path, with exact API availability checked against the installed headers.