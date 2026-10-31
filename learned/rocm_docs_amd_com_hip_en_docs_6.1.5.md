# Rocm-Docs-Amd-Com-Hip-En-Docs-6.1.5

_Source: Distilled from PDF documentation_

HIP Documentation
Release 6.3.42134
Advanced Micro Devices, Inc.
Mar 13, 2025

## INSTALL

1 Overview 3
2 Install HIP 5
2.1 Prerequisites . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.2 Installation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.3 Verify your installation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3 Build HIP from source 7
3.1 Prerequisites . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.2 Building the HIP runtime . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.3 Build HIP tests . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
3.4 Run HIP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4 Understanding the HIP programming model 13
4.1 RDNA & CDNA Architecture Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
4.2 Single Instruction Multiple Threads . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
4.3 Heterogeneous Programming . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
5 Programming model reference 17
5.1 Threading Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
5.1.1 Inherent Thread Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
5.1.2 Cooperative Groups Thread Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
5.2 Memory Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
5.3 Execution Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
5.3.1 Host-side execution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
5.3.2 Device-side execution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
5.3.3 Kernel launch . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
6 Hardware Implementation 21
6.1 Compute Units . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

## 6.1.1 SIMD . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

6.1.2 Vector Cache . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
6.1.3 Local Data Share . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
6.1.4 Scalar Unit . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
6.2 CDNA Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
6.3 RDNA Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
6.4 Shader Engines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
7 AMD compute language runtime (CLR) 25
7.1 Project Organisation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
7.2 How to build/install . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
i
7.2.1 Prerequisites . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
7.2.2 Linux . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
7.2.3 Test . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
7.2.4 Release notes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
8 HIP Programming Manual 27
8.1 Host Memory . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
8.1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
8.1.2 Memory allocation flags . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
8.1.3 Numa-aware host memory allocation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
8.1.4 Coherency Controls . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
8.1.5 Visibility of Zero-Copy Host Memory . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
8.1.6 hipEventSynchronize . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
8.1.7 Summary and Recommendations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
8.1.8 Managed memory allocation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
8.1.9 HIP Stream Memory Operations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
8.2 Direct Dispatch . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
8.3 HIP Runtime Compilation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
8.4 HIP Graph . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
8.5 Device-Side Malloc . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
8.6 Use of Per-thread default stream . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
8.7 Use of Long Double Type . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
8.8 Use of _Float16 Type . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
8.9 FMA and contractions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
8.10 Math functions with special rounding modes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
8.11 Creating Static Libraries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
9 HIP Porting Guide 35
9.1 Porting a New CUDA Project . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
9.1.1 General Tips . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
9.1.2 Scanning existing CUDA code to scope the porting effort . . . . . . . . . . . . . . . . . . . 35
9.1.3 Converting a project “in-place” . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
9.1.4 Library Equivalents . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
9.2 Distinguishing Compiler Modes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
9.2.1 Identifying HIP Target Platform . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
9.2.2 Identifying the Compiler: hip-clang or NVCC . . . . . . . . . . . . . . . . . . . . . . . . . 38
9.2.3 Identifying Current Compilation Pass: Host or Device . . . . . . . . . . . . . . . . . . . . 38
9.2.4 Compiler Defines: Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
9.3 Identifying Architecture Features . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
9.3.1 HIP_ARCH Defines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
9.3.2 Device-Architecture Properties . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
9.3.3 Table of Architecture Properties . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
9.4 Finding HIP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
9.5 Identifying HIP Runtime . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
9.6 hipLaunchKernelGGL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
9.7 Compiler Options . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
9.7.1 Compiler options supported on AMD platforms . . . . . . . . . . . . . . . . . . . . . . . . 42
9.8 Linking Issues . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
9.8.1 Linking With hipcc . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
9.8.2 -lm Option . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
9.9 Linking Code With Other Compilers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
9.9.1 libc++ and libstdc++ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
9.9.2 HIP Headers ( hip_runtime.h,hip_runtime_api.h) . . . . . . . . . . . . . . . . . . . . 44
9.9.3 Using a Standard C++ Compiler . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
ii
9.9.3.1 cuda.h . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
9.9.4 Choosing HIP File Extensions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
9.10 Workarounds . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
9.10.1 warpSize . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
9.10.2 Kernel launch with group size > 256 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
9.11 memcpyToSymbol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45

## 9.12 CU_POINTER_ATTRIBUTE_MEMORY_TYPE . . . . . . . . . . . . . . . . . . . . . . . . . . . 46

9.13 threadfence_system . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
9.13.1 Textures and Cache Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
9.14 More Tips . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
9.14.1 HIP Logging . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
9.14.2 Debugging hipcc . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49
9.14.3 Editor Highlighting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49
10 Porting CUDA Driver API 51
10.1 Introduction to the CUDA Driver and Runtime APIs . . . . . . . . . . . . . . . . . . . . . . . . . . 51
10.1.1 cuModule API . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
10.1.2 cuCtx API . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
10.2 HIP Module andCtx APIs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
10.2.1 hipModule API . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
10.2.2 hipCtx API . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
10.2.3 hipify translation of CUDA Driver API . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
10.2.3.1 Address Spaces . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
10.2.3.2 Using hipModuleLaunchKernel . . . . . . . . . . . . . . . . . . . . . . . . . . 53
10.2.3.3 Additional Information . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
10.2.4 hip-clang Implementation Notes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
10.2.4.1 .hip_fatbin . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
10.2.4.2 Initialization and Termination Functions . . . . . . . . . . . . . . . . . . . . . . . 54
10.2.4.3 Kernel Launching . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 54
10.2.5 NVCC Implementation Notes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 54
10.2.5.1 Interoperation between HIP and CUDA Driver . . . . . . . . . . . . . . . . . . . . 54
10.2.5.2 Compilation Options . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55
10.3 HIP Module and Texture Driver API . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
11 Programming for HIP Runtime Compiler (RTC) 59
11.1 Compilation APIs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
11.2 HIPRTC specific options . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63
11.2.1 Bitcode . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
11.2.2 CU Mode vs WGP mode . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
11.3 Linker APIs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
11.3.1 Example . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
11.3.1.1 Note . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
11.3.2 Input Types . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 66
11.3.3 Backward Compatibility of LLVM Bitcode/IR . . . . . . . . . . . . . . . . . . . . . . . . . 66
11.3.4 Link Options . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 66
11.4 Error Handling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 67
11.5 HIPRTC General APIs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 67
11.6 Lowered Names (Mangled Names) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 68
11.6.1 Note . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 68
11.6.2 Example . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 68
11.7 Versioning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 69
11.8 HIP header support . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 70
11.9 Deprecation notice . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 70
iii
12 Performance Guidelines 71
12.1 Parallel execution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71
12.1.1 Application level . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71
12.1.2 Device level . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71
12.1.3 Multiprocessor level . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 72
12.2 Memory optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 72
12.2.1 Data Transfer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 72
12.2.2 Device Memory Access . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73
12.3 Optimization for maximum instruction throughput . . . . . . . . . . . . . . . . . . . . . . . . . . . 73
12.3.1 Arithmetic instructions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
12.3.2 Control flow instructions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
12.3.3 Synchronization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
12.4 Minimizing memory thrashing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 75
13 Debugging with HIP 77
13.1 Tracing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 77
13.2 Debugging . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 79
13.2.1 Debugging HIP applications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 79
13.3 Useful environment variables . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 81
13.3.1 Kernel enqueue serialization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 81
13.3.2 Making device visible . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 81
13.3.3 Dump code object . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 81
13.3.4 HSA-related environment variables (Linux) . . . . . . . . . . . . . . . . . . . . . . . . . . 82
13.3.5 HIP environment variable summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 82
13.4 General debugging tips . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 84
14 Logging HIP activity 85
14.1 Logging level . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 85
14.2 Logging mask . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 86
14.3 Logging command . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 86
14.4 Logging examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 87
15 Frequently asked questions 91
15.1 What APIs and features does HIP support? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 91
15.2 What is not supported? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 91
15.2.1 Runtime/Driver API features . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 91
15.2.2 Kernel language features . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 92
15.3 Is HIP a drop-in replacement for CUDA? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 92
15.4 What specific version of CUDA does HIP support? . . . . . . . . . . . . . . . . . . . . . . . . . . . 92
15.5 What libraries does HIP support? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 93
15.6 How does HIP compare with OpenCL? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 93
15.7 How does porting CUDA to HIP compare to porting CUDA to OpenCL? . . . . . . . . . . . . . . . 93
15.8 What hardware does HIP support? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 94
15.9 Do HIPIFY tools automatically convert all source code? . . . . . . . . . . . . . . . . . . . . . . . . 94
15.10 What is NVCC? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 94
15.11 What is HIP-Clang? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 94
15.12 Why use HIP rather than supporting CUDA directly? . . . . . . . . . . . . . . . . . . . . . . . . . . 94
15.13 Can I develop HIP code on an NVIDIA CUDA platform? . . . . . . . . . . . . . . . . . . . . . . . 95
15.14 Can I develop HIP code on an AMD HIP-Clang platform? . . . . . . . . . . . . . . . . . . . . . . . 95
15.15 How to use HIP-Clang to build HIP programs? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 95
15.16 What is AMD clr? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 95
15.17 What is hipother? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 96
15.18 Can I get HIP open source repository for Windows? . . . . . . . . . . . . . . . . . . . . . . . . . . 96
15.19 Can a HIP binary run on both AMD and NVIDIA platforms? . . . . . . . . . . . . . . . . . . . . . 96
iv
15.20 On HIP-Clang, can I link HIP code with host code compiled with another compiler such as gcc, icc,
or clang? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 96
15.21 Can HIP API support C style application? What is the difference between C and C++? . . . . . . . . 96
15.22 Can I install both CUDA SDK and HIP-Clang on the same machine? . . . . . . . . . . . . . . . . . 97
15.23 HIP detected my platform (HIP-Clang vs NVCC) incorrectly * what should I do? . . . . . . . . . . . 97
15.24 On CUDA, can I mix CUDA code with HIP code? . . . . . . . . . . . . . . . . . . . . . . . . . . . 98
15.25 How do I trace HIP application flow? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 98

```cpp

15.26 What are the maximum limits of kernel launch parameters? . . . . . . . . . . . . . . . . . . . . . . 98

```

15.27 Are__shfl_*_sync functions supported on HIP platform? . . . . . . . . . . . . . . . . . . . . . . 98
15.28 How to create a guard for code that is specific to the host or the GPU? . . . . . . . . . . . . . . . . . 98
15.29 Why _OpenMP is undefined when compiling with-fopenmp? . . . . . . . . . . . . . . . . . . . . . 99
15.30 Does the HIP-Clang compiler support extern shared declarations? . . . . . . . . . . . . . . . . . . . 99
15.31 I have multiple HIP enabled devices and I am getting an error code
hipErrorSharedObjectInitFailed with the message “Error: shared object initialization
failed”? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 99
15.32 How to use per-thread default stream in HIP? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 100
15.33 How to use complex multiplication and division operations? . . . . . . . . . . . . . . . . . . . . . . 100
15.34 Can I develop applications with HIP APIs on Windows the same on Linux? . . . . . . . . . . . . . . 100
15.35 Does HIP support LUID? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 101
15.36 How can I know the version of HIP? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 101
16 HIP Runtime API Reference 103
16.1 Related Pages . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.2 Modules . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3 Data Structures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.1 Data Structures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.2 Data Structure Index . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1 All . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.1 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.2 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.3 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.4 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.5 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.6 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.7 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.8 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.9 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.10 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.11 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.12 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.13 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.14 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.15 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.16 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.17 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.18 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.19 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.20 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.21 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.22 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.23 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.1.24 Data Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2 Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
v
16.3.3.2.1 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.2 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.3 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.4 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.5 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.6 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.7 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.8 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.9 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.10 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.11 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.12 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.13 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.14 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.15 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.16 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.17 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.18 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.19 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.20 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.21 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.22 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.23 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.3.3.2.24 Data Fields - Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4 Files . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.1 File List . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1 All . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.1 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.2 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.3 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.4 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.5 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.6 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.7 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.8 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.9 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.10 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.11 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.12 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.13 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.14 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.15 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.16 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.17 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.18 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.1.19 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2 Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.1 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.2 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.3 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.4 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.5 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.6 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
vi
16.4.2.2.7 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.8 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.9 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.10 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.11 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.12 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.13 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.14 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.15 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.2.16 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.3 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.4 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.5 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.6 Enumerator . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.6.1 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.6.2 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
16.4.2.7 Globals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
17 C++ Language Extensions 105
17.1 Function-type qualifiers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 105
17.1.1 __device__ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 105

```cpp

17.1.2 __global__ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 106

```

17.1.3 __host__ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 106

```cpp

17.2 Calling __global__ functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 106

```

17.3 Kernel launch example . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 107
17.4 Variable type qualifiers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
17.4.1 __constant__ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
17.4.2 __shared__ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
17.4.3 __managed__ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
17.4.4 __restrict__ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
17.5 Built-in variables . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
17.5.1 Coordinate built-ins . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
17.5.2 warpSize . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 109
17.6 Vector types . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 109
17.6.1 Short vector types . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 109

```cpp

17.6.2 dim3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 110

```

17.7 Memory fence instructions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 110
17.8 Synchronization functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 110
17.9 Math functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 110
17.9.1 Single precision mathematical functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . 110
17.9.2 Double precision mathematical functions . . . . . . . . . . . . . . . . . . . . . . . . . . . 120
17.9.3 Integer intrinsics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 130
17.9.4 Floating-point Intrinsics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 132
17.10 Texture functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 134
17.11 Surface functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 135
17.12 Timer functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 135
17.13 Atomic functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 135
17.13.1 Unsafe floating-point atomic RMW operations . . . . . . . . . . . . . . . . . . . . . . . . . 137
17.14 Warp cross-lane functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 137
17.14.1 Warp vote and ballot functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 138
17.14.2 Warp match functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 139
17.14.3 Warp shuffle functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 139
17.15 Cooperative groups functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 140
17.16 Warp matrix functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 140
vii
17.17 Independent thread scheduling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 141
17.18 Profiler Counter Function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 141
17.19 Assert . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 141
17.20 printf . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 142
17.21 Device-Side Dynamic Global Memory Allocation . . . . . . . . . . . . . . . . . . . . . . . . . . . 142
17.22 __launch_bounds__ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 142
17.22.1 Compiler Impact . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 143
17.22.2 CU and EU Definitions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 143
17.22.3 Porting from CUDA__launch_bounds . . . . . . . . . . . . . . . . . . . . . . . . . . . . 143
17.22.4 maxregcount . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 144
17.23 Asynchronous Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 144
17.23.1 Memory stream . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 144
17.23.2 Peer to peer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 159
17.23.3 Memory management . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 161
17.23.4 External Resource Interoperability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 190
17.24 Register Keyword . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 193
17.25 Pragma Unroll . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 193
17.26 In-Line Assembly . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 193
17.27 C++ Support . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 193
17.28 Kernel Compilation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 194

```cpp

17.29 gfx-arch-specific-kernel . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 194

```

18 Table Comparing Syntax for Different Compute APIs 195
18.1 Notes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 196
19 HSA Runtime API for ROCm 197
20 HIP deprecated runtime API functions 203
20.1 Context management . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 203
20.2 Memory management . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 204
20.3 Profiler control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 204
20.4 Texture management . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 204
21 FP8 Numbers 207
21.1 FP8 Definition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 207
21.2 HIP Header . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 207
21.3 Supported Devices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 207
21.4 Usage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 208
22 License 213
Index 215
viii
HIP Documentation, Release 6.3.42134

```cpp

The Heterogeneous-computing Interface for Portability (HIP) API is a C++ runtime API and kernel language that lets

```

developers create portable applications for AMD and NVIDIA GPUs from single source code.

## For HIP supported AMD GPUs on multiple operating systems, see

• Linux system requirements
• Microsoft Windows system requirements
The CUDA enabled NVIDIA GPUs are supported by HIP. For more information, see GPU Compute Capability.
OntheAMDROCmplatform,HIPprovidesheaderfilesandruntimelibrarybuiltontopofHIP-Clangcompilerinthe
repository Compute Language Runtime (CLR), which contains source codes for AMD’s compute languages runtimes.
On non-AMD platforms, like NVIDIA, HIP provides header files required to support non-AMD specific back-end
implementation in the repository ‘hipother’, which translates from the HIP runtime APIs to CUDA runtime APIs.

## INSTALL 1

HIP Documentation, Release 6.3.42134

## 2 INSTALL


## CHAPTER


## ONE


## OVERVIEW

Install
• Installing HIP
• Building HIP from source
Conceptual
• Understanding the HIP programming model
• Programming model reference
• Hardware Implementation
• AMD compute language runtime (CLR)
How to
• Programming Manual
• HIP Porting Guide
• HIP Porting: Driver API Guide
• Programming for HIP Runtime Compiler (RTC)
• Performance Guidelines
• Debugging with HIP
• Logging HIP activity
• Frequently asked questions
Reference
• HIP Runtime API Reference
• C++ language extensions
• Comparing Syntax for different APIs
• HSA Runtime API for ROCm
• List of deprecated APIs
• FP8 numbers in HIP
Tutorial
• HIP examples
• HIP test samples
3
HIP Documentation, Release 6.3.42134
Known issues are listed on the HIP GitHub repository.
TocontributefeaturesorfunctionstotheHIPproject,refertoContributingtoHIP.Tocontributetothedocumentation,
refer to Contributing to ROCm docs page.
You can find licensing information on the Licensing page.
4 Chapter 1. Overview

## CHAPTER


## TWO


## INSTALL HIP

HIP can be installed on AMD (ROCm with HIP-Clang) and NVIDIA (CUDA with NVCC) platforms.
Note: The version definition for the HIP runtime is different from CUDA. On an AMD platform, the
hipRuntimeGerVersion function returns the HIP runtime version; on an NVIDIA platform, this function returns
the CUDA runtime version.
2.1 Prerequisites

## AMD


## Refer to the Prerequisites section in the ROCm install guides

• rocm-install-on-linux:reference/system-requirements
• rocm-install-on-windows:reference/system-requirements

## NVIDIA

Check the system requirements in the NVIDIA CUDA Installation Guide.
2.2 Installation

## AMD

HIPisautomaticallyinstalledduringtheROCminstallation. Ifyouhaven’tyetinstalledROCm,youcanfindinstallation

## instructions here

• rocm-install-on-linux:index
• rocm-install-on-windows:index
By default, HIP is installed into/opt/rocm/hip.
Note: There is no autodetection for the HIP installation. If you choose to install it somewhere other than the default
location, you must set theHIP_PATH environment variable as explained in Build HIP from source.
5
HIP Documentation, Release 6.3.42134

## NVIDIA

1. Install the NVIDIA driver.
sudo apt-get install ubuntu-drivers-common && sudo ubuntu-drivers autoinstall
sudo reboot
Alternatively, you can download the latest CUDA Toolkit.
2. Install thehip-runtime-nvidiaand hip-dev packages. This installs the CUDA SDK and HIP porting layer.
apt-get install hip-runtime-nvidia hip-dev

## The default paths are

• CUDA SDK:/usr/local/cuda
• HIP:/opt/rocm/hip
You can optionally add/opt/rocm/binto your path, which can make it easier to use the tools.
2.3 Verify your installation
Runhipconfig in your installation path.
/opt/rocm/bin/hipconfig --full
6 Chapter 2. Install HIP

## CHAPTER


## THREE


## BUILD HIP FROM SOURCE

3.1 Prerequisites
HIP code can be developed either on AMD ROCm platform using HIP-Clang compiler, or a CUDA platform with
nvccinstalled. BeforebuildingandrunningHIP,makesuredriversandprebuiltpackagesareinstalledproperlyonthe
platform.
YoualsoneedtoinstallPython3,whichincludesthe CppHeaderParserpackage. InstallPython3usingthefollowing

## command

apt-get install python3

## Check and installCppHeaderParser package using the command

pip3 install CppHeaderParser

## InstallROCm LLVM package using the command

apt-get install rocm-llvm-dev
3.2 Building the HIP runtime
Set the repository branch using the variable:ROCM_BRANCH. For example, for ROCm 6.1, use:
export ROCM_BRANCH=rocm-6.1.x

## AMD

1. Get HIP source code.
Note: Starting in ROCM 5.6, CLR is a new repository that includes the former ROCclr, HIPAMD and OpenCl
repositories. OpenCL provides headers that ROCclr runtime depends on.
Note: Starting in ROCM 6.1, a new repositoryhipotheris added to ROCm, which is branched out from HIP.
hipother provides files required to support the HIP back-end implementation on some non-AMD platforms,
like NVIDIA.
7
HIP Documentation, Release 6.3.42134
git clone -b "$ROCM_BRANCH" https://github.com/ROCm/clr.git
git clone -b "$ROCM_BRANCH" https://github.com/ROCm/hip.git
CLR (Compute Language Runtime) repository includes ROCclr, HIPAMD and OpenCL.
ROCclr(ROCmComputeLanguageRuntime)isavirtualdeviceinterfacewhichisdefinedontheAMDplatform.
HIP runtime uses ROCclr to interact with different backends.
HIPAMD provides implementation specifically for HIP on the AMD platform.
OpenCLprovidesheadersthatROCclrruntimecurrentlydependson. hipotherprovidesheadersandimplemen-
tation specifically for non-AMD HIP platforms, like NVIDIA.
2. Set the environment variables.
export CLR_DIR="$(readlink -f clr)"
export HIP_DIR="$(readlink -f hip)"
3. Build HIP.
cd "$CLR_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=amd -DCMAKE_PREFIX_PATH="/opt/rocm/"␣
˓→-DCMAKE_INSTALL_PREFIX=$PWD/install -DHIP_CATCH_TEST=0 -DCLR_BUILD_HIP=ON -DCLR_

## ˓→BUILD_OCL=OFF ..

make -j$(nproc)
sudo make install
Note: Note, if you don’t specifyCMAKE_INSTALL_PREFIX, the HIP runtime is installed at<ROCM_PATH>.
By default, release version of HIP is built. If need debug version, you can put the option
CMAKE_BUILD_TYPE=Debugin the command line.

## Default paths and environment variables

• HIP is installed into<ROCM_PATH>. This can be overridden by setting theINSTALL_PREFIXas the
command option.
environment variable.
• HSA is in<ROCM_PATH>. This can be overridden by setting theHSA_PATH
environment variable.
• Clang is in<ROCM_PATH>/llvm/bin. This can be overridden by setting the
HIP_CLANG_PATHenvironment variable.
• The device library is in<ROCM_PATH>/lib. This can be overridden by setting the
DEVICE_LIB_PATHenvironment variable.
• Optionally, you can add<ROCM_PATH>/bin to yourPATH, which can make it easier to
use the tools.
• Optionally, you can setHIPCC_VERBOSE=7to output the command line for compilation.
After you run themake install command, HIP is installed to<ROCM_PATH> by default, or$PWD/install/
hipwhileINSTALL_PREFIX is defined.
1. Generate a profiling header after adding/changing a HIP API.
8 Chapter 3. Build HIP from source
HIP Documentation, Release 6.3.42134
WhenyouaddorchangeaHIPAPI,youmayneedtogenerateanew hip_prof_str.hheader. Thisheader
is used by ROCm tools to track HIP APIs, such asrocprofiler and roctracer.
To generate the header after your change, use thehip_prof_gen.pytool located inhipamd/src.

## Usage

‘hip_prof_gen.py [-v] <input HIP API .h file> <patched srcs path> <previous␣
˓→output> [<output>]‘

## Flags

• -v: Verbose messages
• -r: Process source directory recursively
• -t: API types matching check
• --priv: Private API check
• -e: On error exit mode
• -p: HIP_INIT_APImacro patching mode

## Example usage

hip_prof_gen.py -v -p -t --priv <hip>/include/hip/hip_runtime_api.h \
<hipamd>/src <hipamd>/include/hip/amd_detail/hip_prof_str.h \
<hipamd>/include/hip/amd_detail/hip_prof_str.h.new

## NVIDIA

1. Get the HIP source code.
git clone -b "$ROCM_BRANCH" https://github.com/ROCm/clr.git
git clone -b "$ROCM_BRANCH" https://github.com/ROCm/hip.git
git clone -b "$ROCM_BRANCH" https://github.com/ROCm/hipother.git
2. Set the environment variables.
export CLR_DIR="$(readlink -f clr)"
export HIP_DIR="$(readlink -f hip)"
export HIP_OTHER="$(readlink -f hipother)"
3. Build HIP.
cd "$CLR_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=nvidia -DCMAKE_INSTALL_PREFIX=$PWD/
˓→install -DHIP_CATCH_TEST=0 -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIPNV_DIR=
˓→$HIP_OTHER/hipnv ..
make -j$(nproc)
sudo make install
3.2. Building the HIP runtime 9
HIP Documentation, Release 6.3.42134
3.3 Build HIP tests

## AMD

• Build HIP catch tests.
HIP catch tests are separate from the HIP project and use Catch2.
– Get HIP tests source code.
git clone -b "$ROCM_BRANCH" https://github.com/ROCm/hip-tests.git
– Build HIP tests from source.
export HIPTESTS_DIR="$(readlink -f hip-tests)"
cd "$HIPTESTS_DIR"
mkdir -p build; cd build
cmake ../catch -DHIP_PLATFORM=amd -DHIP_PATH=$CLR_DIR/build/
˓→install # or any path where HIP is installed; for example: ``/
˓→opt/rocm``
make build_tests
ctest # run tests
HIP catch tests are built in$HIPTESTS_DIR/build.

## To run any single catch test, use this example

cd $HIPTESTS_DIR/build/catch_tests/unit/texture
./TextureTest
– Build a HIP Catch2 standalone test.
cd "$HIPTESTS_DIR"
hipcc $HIPTESTS_DIR/catch/unit/memory/hipPointerGetAttributes.cc \
-I ./catch/include ./catch/hipTestMain/standalone_main.cc \
-I ./catch/external/Catch2 -o hipPointerGetAttributes
./hipPointerGetAttributes
...
All tests passed

## NVIDIA

The commands to build HIP tests on an NVIDIA platform are the same as on an AMD platform. However, you must
first set-DHIP_PLATFORM=nvidia.
10 Chapter 3. Build HIP from source
HIP Documentation, Release 6.3.42134
3.4 Run HIP
After installation and building HIP, you can compile your application and run. A simple example is square sample.
3.4. Run HIP 11
HIP Documentation, Release 6.3.42134
12 Chapter 3. Build HIP from source

## CHAPTER


## FOUR


## UNDERSTANDING THE HIP PROGRAMMING MODEL

The HIP programming model makes it easy to map data-parallel C/C++ algorithms to massively parallel, wide single
instruction, multiple data (SIMD) architectures, such as GPUs. A basic understanding of the underlying device archi-
tecture helps you make efficient use of HIP and general purpose graphics processing unit (GPGPU) programming in
general.
4.1 RDNA & CDNA Architecture Summary
MostGPUarchitectures,likeRDNAandCDNA,haveahierarchicalstructure. TheinnermostpieceisaSIMD-enabled
vector Arithmetic Logical Unit(ALU). In addition to the vectorALUs, most recentGPUs also house matrix ALUs for
accelerating algorithms involving matrix multiply-accumulate operations. AMD GPUs also contain scalar ALUs, that
canbeusedtoreducetheloadonthevectorALUbyperformingoperationswhichareuniformforallthreadsofawarp.
A set of ALUs, together with register files, caches and shared memory, comprise a larger block, often referred to as a
compute unit (CU), e.g. in OpenCL and AMD block diagrams, or as streaming multiprocessor (SM).
Fig. 4.1: Block Diagram of an RDNA3 Compute Unit.
Fig. 4.2: Block Diagram of a CDNA3 Compute Unit.
Forimplementationinhardware,multipleCUsaregroupedtogetherintoaShaderEngineorComputeEngine,typically
sharing some fixed function units or memory subsystem resources.
13
HIP Documentation, Release 6.3.42134
Fig. 4.3: Block Diagram of a CDNA2 Graphics Compute Die.
14 Chapter 4. Understanding the HIP programming model
HIP Documentation, Release 6.3.42134
4.2 Single Instruction Multiple Threads
Thesingleinstruction,multiplethreads(SIMT)programmingmodelbehindtheHIPdevice-sideexecutionisamiddle-
groundbetweenSMT(SimultaneousMulti-Threading)programmingknownfrommulticoreCPUs,andSIMD(Single
Instruction,MultipleData)programmingmostlyknownfromexploitingrelevantinstructionsetsonCPUs(forexample
SSE/AVX/Neon).
A HIP device compiler maps SIMT code written in HIP C++ to an inherently SIMD architecture (like GPUs). This

```cpp

is done by scalarizing the entire kernel and issuing the scalar instructions of multiple kernel instances to each of the
SIMD engine lanes, rather than exploiting data parallelism within a single instance of a kernel and spreading identical

```

instructions over the available SIMD engines.

## Consider the following kernel


```cpp

__global__ void k(float4* a, const float4* b)

```

{

```cpp

int tid = threadIdx.x;

```

int bid = blockIdx.x;
int dim = blockDim.x;
a[tid] += (tid + bid - dim) * b[tid];
}
The incoming four-vector of floating-point valuesb is multiplied by a scalar and then added element-wise to the four-
vector floating-point values ofa. On modern SIMD-capable architectures the four-vector ops are expected to compile

```cpp

to a single SIMD instruction. GPU execution of this kernel however will typically look the following:

```

Fig. 4.4: Instruction flow of the sample SIMT program.
In HIP, lanes of a SIMD architecture are fed by mapping threads of a SIMT execution, one thread down each lane of
a SIMD engine. Execution parallelism usually isn’t exploited from the width of the built-in vector types, but via the

```cpp

thread id constantsthreadIdx.x,blockIdx.x, etc. For more details, refer toInherent Thread Model.

```

4.3 Heterogeneous Programming

```cpp

TheHIPprogrammingmodelassumestwoexecutioncontexts. Oneisreferredtoas hostwhilecomputekernelsexecute

```

on adevice. These contexts have different capabilities, therefor slightly different rules apply. Thehost execution is
definedbytheC++abstractmachine,while deviceexecutionfollowstheHIPmodel,primarilydefinedbySIMT.These
executioncontextsincodearesignifiedbythe __host__and__device__decorators. Thereareafewkeydifferences

## between the two

• The C++ abstract machine assumes a unified memory address space, meaning that one can always access any
givenaddressinmemory(assumingtheabsenceofdataraces). HIPhoweverintroducesseveralmemorynames-
paces, an address from one means nothing in another. Moreover, not all address spaces are accessible from all
contexts.
If one were to look atBlock Diagram of a CDNA2 Graphics Compute Die.and inside theBlock Diagram of a
CDNA3 Compute Unit., every Compute Unit has an instance of storage backing the namespace__shared__.
Even if the host were to have access to these regions of memory, the performance benefits of the segmented
memory subsystem are supported by the inability of asynchronous access from the host.
• Not all C++ language features map cleanly to typical device architectures, some are very expensive (meaning
slow) to implement on GPU devices, therefor they are forbidden in device contexts to avoid users tapping into
featuresthatunexpectedlydecimatetheirprogram’sperformance. OffloaddevicestargetedbyHIParen’tgeneral
4.2. Single Instruction Multiple Threads 15
HIP Documentation, Release 6.3.42134
purposedevices,atleastnotinthesenseaCPUis. HIPfocusesondataparallelcomputationsandassuchcaters
to throughput optimized architectures, such as GPUs or accelerators derived from GPU architectures.
• AsynchronyisattheforefrontoftheHIPAPI.Computationslaunchedonthedeviceexecuteasynchronouslywith
respect to the host, and it is the user’s responsibility to synchronize their data dispatch/fetch with computations
on thedevice. HIP does performimplicit synchronization onoccasions, more advanced than otherAPIs such as
OpenCL or SYCL, in which the responsibility of synchronization mostly depends on the user.
16 Chapter 4. Understanding the HIP programming model

## CHAPTER


## FIVE


## PROGRAMMING MODEL REFERENCE

HIP defines a model for mapping single instruction, multiple threads (SIMT) programs onto various architectures,
primarily GPUs. While the model may be expressed in most imperative languages, (for example Python via PyHIP)
this document will focus on the original C/C++ API of HIP.
5.1 Threading Model
TheSIMTnatureofHIPiscapturedbytheabilitytoexecuteuser-provideddeviceprograms,expressedassingle-source
C/C++ functions or sources compiled online/offline to binaries in bulk.

```cpp

Multiple instances of the device program (or kernel) are called threads and may execute in parallel. All uniquely

```

identified by a set of integral values, or thread IDs. The set of integers identifying a thread relate to the hierarchy in
which threads execute.
5.1.1 Inherent Thread Model
The thread hierarchy inherent to how AMD GPUs operate is depicted in Fig. 5.1.
Fig. 5.1: Hierarchy of thread groups.
Warp
Theinnermostgroupingiscalledawarp,orawavefrontinISAterms. Awarpisthemosttightlycoupledgroups
of threads, both physically and logically.
When referring to threads inside a warp, they may be called lanes, and the integral value identifying them the
lane ID. Lane IDs aren’t queried like other thread IDs, but are user-calculated. As a consequence they are only
as multidimensional as the user interprets the calculated values to be.
The size of a warp is architecture dependent and always fixed. Warps are signified by the set of communication
primitives at their disposal, as discussed inWarp cross-lane functions.
Block
The middle grouping is called a block or thread block. The defining feature of a block is that all threads in a
block will share an instance of memory which they may use to share data or synchronize with one another.
Thesizeofablockisuser-configurablebutismaximizedbythequeryablecapabilitiesoftheexecutinghardware.
The unique ID of the thread within a block is 3-dimensional as provided by the API. When linearizing thread
IDs within a block, assume the “fast index” being dimensionx, followed by theyand zdimensions.
Grid

```cpp

The outermost grouping is called a grid. A grid manifests as a single dispatch of kernels for execution. The

```

17
HIP Documentation, Release 6.3.42134
uniqueIDofeachblockwithinagridis3-dimensional, asprovidedbytheAPIandisqueryablebyeverythread
within the block.
5.1.2 Cooperative Groups Thread Model
The Cooperative Groups API introduces new APIs to launch, group, subdivide, synchronize and identify threads, as
well as some predefined group-collective algorithms, but most importantly a matching threading model to think in
terms of. It relaxes some restrictions of theInherent Thread Modelimposed by the strict 1:1 mapping of architectural
details to the programming model.
The rich set of APIs introduced by Cooperative Groups allow the programmer to define their own set of thread groups

```cpp

which may fit their user-cases better than those defined by the hardware. The set of implicit groups by kernel launch

```

parameters are still available.
The thread hierarchy abstraction of Cooperative Groups manifest as depicted in Fig. 5.2.
Fig. 5.2: Cooperative group thread hierarchy.
Multi Grid

```cpp

An abstraction of potentially multiple simultaneous launches of the same kernel over multiple devices. Grids
insideamultidevicekernellaunchneednotbeofuniformsize,thusallowingtakingintoaccountdifferentdevice

```

capabilities and preferences.
Deprecated since version 5.0.
Grid

```cpp

Same as theInherent Thread ModelGrid entity. The ability to synchronize over a grid requires the kernel to be

```

launched using the Cooperative Groups API.
Block
Same as theInherent Thread ModelBlock entity.
Note: Explicit warp-level thread handling is absent from the Cooperative Groups API. In order to exploit the known
hardware SIMD width on which built-in functionality translates to simpler logic, one may use the group partitioning
part of the API, such astiled_partition.
5.2 Memory Model
Thehierarchyofthreadsintroducedby InherentThreadModel isinducedbythememorysubsystemofGPUs. Fig.5.3
summarizes that memory namespaces and how they relate to the various levels of the threading model.
Fig. 5.3: Memory hierarchy.
Local or per-thread memory
Read-write storage only visible to the threads defining the given variables, also called per-thread memory. The

```cpp

size of a block for a given kernel, the number of concurrent warps are limited by local memory usage. This

```

relates to an important aspect: occupancy. This is the default memory namespace.
Shared memory
Read-write storage visible to all the threads in a given block.
18 Chapter 5. Programming model reference
HIP Documentation, Release 6.3.42134
Global
Read-write storage visible to all threads in a given grid. There are specialized versions of global memory with
different usage semantics which are typically backed by the same hardware storing global.
Constant
Read-only storage visible to all threads in a given grid. It is a limited segment of global with queryable
size.
Texture
Read-only storage visible to all threads in a given grid and accessible through additional APIs.
Surface
A read-write version of texture memory.
5.3 Execution Model

## HIP programs consist of two distinct scopes


## • The host-side API running on the host processor. There are to APIs available

– The HIP runtime API which enables use of the single-source programming model.
– The HIP driver API which sits at a lower level and most importantly differs by removing some facilities

```cpp

provided by the runtime API, most importantly around kernel launching and argument setting. It is geared

```

towards implementing abstractions atop, such as the runtime API itself.

```cpp

• The device-side kernels running on GPUs. Both the host and the device-side APIs have synchronous and asyn-

```

chronous functions in them.
5.3.1 Host-side execution
Thepartofthehost-sideAPIwhichdealswithdevicemanagementandtheirqueriesaresynchronous. Allasynchronous

```cpp

APIs,suchaskernelexecution,datamovementandpotentiallydataallocation/freeingallhappeninthecontextofdevice

```

streams.
Streams are FIFO buffers of commands to execute relating to a given device. Commands which enqueue tasks on a
stream all return promptly and the command is executed asynchronously. All side effects of a command on a stream
are visible to all subsequent commands on the same stream. Multiple streams may point to the same device and those
streams may be fed from multiple concurrent host-side threads. Execution on multiple streams may be concurrent but
isn’t required to be.
Asynchronous APIs involving a stream all return a stream event which may be used to synchronize the execution of
multiple streams. A user may enqueue a barrier onto a stream referencing an event. The barrier will block until the
command related to the event does not complete, at which point all side effects of the command shall be visible to
commands following the barrier, even if those side effects manifest on different devices.
Streams also support executing user-defined functions as callbacks on the host. The stream will not launch subsequent
commands until the callback completes.
5.3. Execution Model 19
HIP Documentation, Release 6.3.42134
5.3.2 Device-side execution
TheSIMTprogrammingmodelbehindtheHIPdevice-sideexecutionisamiddle-groundbetweenSMT(Simultaneous
Multi-Threading)programmingknownfrommulticoreCPUs,andSIMD(SingleInstruction,MultipleData)program-
ming mostly known from exploiting relevant instruction sets on CPUs (for example SSE/AVX/Neon).
A HIP device compiler maps our SIMT code written in HIP C++ to an inherently SIMD architecture (like GPUs) not

```cpp

by exploiting data parallelism within a single instance of a kernel and spreading identical instructions over the SIMD
engines at hand, but by scalarizing the entire kernel and issuing the scalar instructions of multiple kernel instances to

```

each of the SIMD engine lanes.
5.3.3 Kernel launch
Kernels may be launched in multiple ways all with different syntaxes and intended use-cases.

```cpp

• Using the triple-chevron<<<...>>> operator on a__global__annotated function.
• UsinghipLaunchKernelGGL() on a__global__ annotated function.

```

Tip: This name by default is a macro expanding to triple-chevron. In cases where language syn-

```cpp

tax extensions are undesirable, or where launching templated and/or overloaded kernel functions define the

```

HIP_TEMPLATE_KERNEL_LAUNCH preprocessor macro before including the HIP headers to turn it into a tem-
plated function.
• Using the launch APIs supporting the triple-chevron syntax directly.
Caution: These APIs are intended to be used/generated by tools such as the HIP compiler itself and not
intended towards end-user code. Should you be writing a tool having to launch device code using HIP,
consider using these over the alternatives.
20 Chapter 5. Programming model reference

## CHAPTER


## SIX


## HARDWARE IMPLEMENTATION

This chapter describes the typical hardware implementation of GPUs supported by HIP, and how theInherent Thread
Model maps to the hardware.
6.1 Compute Units
ThebasicbuildingblockofaGPUisacomputeunit(CU),alsoknownasstreamingmultiprocessor(SM)onNVIDIA
GPUs. ThethreadblocksmakingupagridarescheduledforexecutiononCUs. Eachblockisassignedtoanindividual
CU, and a CU can accommodate several blocks. Depending on their resource usage up to thousands of threads can
reside on a CU.
CUs contain an array of processing elements, referred to as vector ALU (VALU), that execute the actual instructions
of the threads according to theSIMT model, together with the necessary registers and caches.
Thethreadsareexecutedingroupingscalledwarps. Theamountofthreadsmakingupawarpisarchitecturedependent.
On AMD GPUs the warp size is commonly 64 threads, except in RDNA architectures which can utilize a warp size of
32or64respectively. ThewarpsizeofsupportedAMDGPUsislistedintherocm:reference/gpu-arch-specs. NVIDIA
GPUs have a warp size of 32.
In contrast to CPUs, GPUs generally do not employ complex cache structures or control logic, like branch prediction
or out-of-order execution, but instead rely on massive hardware multithreading to hide latency.
Context switching between warps residing on a CU incurs no overhead, as the context for the warps is stored on the
CU and does not need to be fetched from memory. If there are not enough free registers to accommodate all warps of
a block, the block can not be scheduled to that CU and it has to wait until other blocks finish execution.
TheamountofwarpsthatcanresideconcurrentlyonaCU,knownasoccupancy,isdeterminedbythewarp’sresource
usage of registers and shared memory.
Fig. 6.1: An AMD Graphics Core Next (GCN) CU. The CDNA and RDNA CUs are based on variations of the GCN

## CU.


## On AMD GCN GPUs the basic structure of a CU is

• four Single Instruction Multiple Data units (SIMDs)
• a vector cache
• a local data share
• and a scalar unit
21
HIP Documentation, Release 6.3.42134

## 6.1.1 SIMD

A SIMD consists of a VALU, that executes the instruction of a warp, together with a register file, that provides the
registers warps.
The size of the warp is inherently related to the width of the vector ALU of the SIMD. On GCN compute units the
width of the VALU is 16, so a warp can be issued to a SIMD every 4 cycles. Since a CU has 4 SIMDs it issues one
warp per cycle. The instructions of a warp are effectively executed in lock-step.
A SIMD always executes the same instruction for the whole VALU. If the control flow of a warp diverges, the perfor-
mance is decreased, as the results for the threads that do not participate in that branch have to be masked out, and the
instructions of the other branch have to be executed in the same way. The best performance can therefore be achieved
when thread divergence is kept to a warp level, i.e. when all threads in a warp take the same execution path.
6.1.2 Vector Cache
The usage of cache on a GPU differs from that on a CPU, as there is less cache available per thread. Its main purpose
is to coalesce memory accesses of the warps in order to reduce the amount of accesses to device memory, and make
that memory available for other warps that currently reside on the compute unit, that also need to load those values.
6.1.3 Local Data Share
Thelocaldatashareismemorythatisaccessibletoallthreadswithinablock. Itslatencyandbandwidthiscomparable
to that of the vector cache. It can be used to share memory between the threads in a block, or as a software managed
cache.
6.1.4 Scalar Unit
The scalar unit performs instructions that are uniform within a warp. It thereby improves efficiency and reduces the
pressure on the vector ALUs and the vector register file.
6.2 CDNA Architecture
The general structure of CUs stays mostly as it is in GCN architectures. The most prominent change is the addition of
matrix ALUs, which can greatly improve the performance of algorithms involving matrix multiply-accumulate opera-
tions for int8, float16, bfloat16 or float32.
Fig. 6.2: Block Diagram of a CDNA3 Compute Unit.
22 Chapter 6. Hardware Implementation
HIP Documentation, Release 6.3.42134
6.3 RDNA Architecture
RDNA makes a fundamental change to CU design, by changing the size of a warp to 32 threads. This is done by
effectively combining two GCN5 SIMDs, creating a VALU of width 32, so that a whole warp can be issued in one
cycle. The CU is also replaced by the work group processor (WGP), which encompasses two CUs. For backwards
compatibility the WGP can also run in wave64 mode, in which it issues a warp of size 64 in two cycles.
It also adds an extra layer of cache to the WGP, shared by the CUs within it. This cache is referred to as L1 cache,
promoting the per-CU cache to an L0 cache.
Fig. 6.3: Block Diagram of an RDNA3 work group processor.
6.4 Shader Engines
For hardware implementation’s sake, multiple CUs are grouped together into a Shader Engine or Compute Engine,
typically sharing some fixed function units or memory subsystem resources.
6.3. RDNA Architecture 23
HIP Documentation, Release 6.3.42134
Fig. 6.4: Block Diagram of a CDNA2 Graphics Compute Die.
24 Chapter 6. Hardware Implementation

## CHAPTER


## SEVEN


## AMD COMPUTE LANGUAGE RUNTIME (CLR)

CLRcontainssourcecodesforAMD’scomputelanguagesruntimes: HIPandOpenCL™.CLRisthepartofHIPruntime
which is supported on the AMD ROCm platform, it provides a header and runtime library built on top of HIP-Clang
compiler. Fordevelopersandusers, CLRimplementsHIPruntimeAPIsincludingstreams, events,andmemoryAPIs,
whichisaobjectlibrarythatislinkedwiththeapplication. Thesourcecodesforallheadersandthelibraryimplemen-
tation are available on GitHub in the clr repository.
7.1 Project Organisation
CLR includes the following source code,
• hipamd - contains implementation ofHIPlanguage on the AMD platform. It is hosted at clr/hipamd.
• opencl - contains implementation of OpenCL™ on AMD platform. It is hosted at clr/opencl.
• rocclr - contains ROCm compute runtime used inHIP and OpenCL™. This is hosted at clr/rocclr.
7.2 How to build/install
7.2.1 Prerequisites
Please refer to Quick Start Guide in ROCm Docs.
Building CLR requiresrocm-hip-librariesmeta package, which provides the pre-requisites for CLR.
7.2.2 Linux
• Clone this repository
cd clr && mkdir build && cd build
• ForHIP
cmake .. -DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=$HIP_COMMON_DIR
‘‘HIP_COMMON_DIR‘‘ points to ‘HIP <https://github.com/ROCm/HIP>‘_.
• ForOpenCL™
25
HIP Documentation, Release 6.3.42134
cmake .. -DCLR_BUILD_OCL=ON
make
make install
Users can also buildOCL and HIP at the same time by passing-DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=ON to
configure command.
For detail instructions, please refer to build HIP.
7.2.3 Test
hip-tests is a separate repository hosted at hip-tests.
To runhip-tests please go to the repository and follow the steps.
7.2.4 Release notes
HIP provides release notes in CLR change log, which has records of changes in each release.
26 Chapter 7. AMD compute language runtime (CLR)

## CHAPTER


## EIGHT


## HIP PROGRAMMING MANUAL

8.1 Host Memory
8.1.1 Introduction
hipHostMallocallocatespinnedhostmemorywhichismappedintotheaddressspaceofallGPUsinthesystem, the
memory can be accessed directly by the GPU device, and can be read or written with much higher bandwidth than
pageable memory obtained with functions such asmalloc(). There are two use cases for this host memory:
• FasterHostToDeviceandDeviceToHostDataTransfers: Theruntimetracksthe hipHostMallocallocations
and can avoid some of the setup required for regular unpinned memory. For exact measurements on a specific
system, experiment with--unpinned and --pinnedswitches for thehipBusBandwidth tool.
• Zero-Copy GPU Access: GPU can directly access the host memory over the CPU/GPU interconnect, without

```cpp

need to copy the data. This avoids the need for the copy, but during the kernel access each memory access must

```

traversetheinterconnect,whichcanbetensoftimesslowerthanaccessingtheGPU’slocaldevicememory. Zero-
copy memory can be a good choice when the memory accesses are infrequent (perhaps only once). Zero-copy
memory is typically “Coherent” and thus not cached by the GPU but this can be overridden if desired.
8.1.2 Memory allocation flags
There are flags parameter which can specify options how to allocate the memory, for example,
hipHostMallocPortable, the memory is considered allocated by all contexts, not just the one on which the
allocationismade. hipHostMallocMapped,willmaptheallocationintotheaddressspaceforthecurrentdevice,and
the device pointer can be obtained with the APIhipHostGetDevicePointer(). hipHostMallocNumaUser is the
flag to allow host memory allocation to follow Numa policy by user. Please note this flag is currently only applicable
on Linux, under development on Windows.
All allocation flags are independent, and can be used in any combination without restriction, for instance,
hipHostMalloc can be called with bothhipHostMallocPortable and hipHostMallocMapped flags set. Both
usagemodelsdescribedaboveusethesameallocationflags,andthedifferenceisinhowthesurroundingcodeusesthe
host memory.
27
HIP Documentation, Release 6.3.42134
8.1.3 Numa-aware host memory allocation
Numa policy determines how memory is allocated. Target of Numa policy is to select a CPU that is closest to each
GPU. Numa distance is the measurement of how far between GPU and CPU devices.
By default, each GPU selects a Numa CPU node that has the least Numa distance between them, that is, host mem-
ory will be automatically allocated closest on the memory pool of Numa node of the current GPU device. Using
hipSetDevice API to a different GPU will still be able to access the host allocation, but can have longer Numa dis-
tance. Note, Numa policy is so far implemented on Linux, and under development on Windows.
8.1.4 Coherency Controls

## ROCm defines two coherency options for host memory


```cpp

• Coherent memory : Supports fine-grain synchronization while the kernel is running. For example, a kernel can

```

performatomicoperationsthatarevisibletothehostCPUortoother(peer)GPUs. Synchronizationinstructions
includethreadfence_system and C++11-style atomic operations. In order to achieve this fine-grained coher-
ence, many AMD GPUs use a limited cache policy, such as leaving these allocations uncached by the GPU, or
making them read-only.

```cpp

• Non-coherent memory : Can be cached by GPU, but cannot support synchronization while the kernel is run-
ning. Non-coherent memory can be optionally synchronized only at command (end-of-kernel or copy com-

```

mand) boundaries. This memory isappropriate forhigh-performance access whenfine-grain synchronization is
not required.
HIP provides the developer with controls to select which type of memory is used via allocation flags passed
to hipHostMalloc and the HIP_HOST_COHERENT environment variable. By default, the environment variable
HIP_HOST_COHERENT is set to 0 in HIP. The control logic in the current version of HIP is as follows:
• Noflagsarepassedin: thehostmemoryallocationiscoherent,theHIP_HOST_COHERENTenvironmentvari-
able is ignored.
• hipHostMallocCoherent=1: Thehostmemoryallocationwillbecoherent,theHIP_HOST_COHERENTen-
vironment variable is ignored.
• hipHostMallocMapped=1: The host memory allocation will be coherent, the HIP_HOST_COHERENT envi-
ronment variable is ignored.
• hipHostMallocNonCoherent=1, hipHostMallocCoherent=0, and hipHostMallocMapped=0: The host
memory will be non-coherent, the HIP_HOST_COHERENT environment variable is ignored.
• hipHostMallocCoherent=0, hipHostMallocNonCoherent=0, hipHostMallocMapped=0, but one of the

## other HostMalloc flags is set

– If HIP_HOST_COHERENT is defined as 1, the host memory allocation is coherent.
– If HIP_HOST_COHERENT is not defined, or defined as 0, the host memory allocation is non-coherent.
• hipHostMallocCoherent=1,hipHostMallocNonCoherent=1: Illegal.
28 Chapter 8. HIP Programming Manual
HIP Documentation, Release 6.3.42134
8.1.5 Visibility of Zero-Copy Host Memory
Coherent host memory is automatically visible at synchronization points. Non-coherent
HIP API Synchronization Effect Fence Coherent Host
Memory Visibil-
ity
Non-Coherent
Host Memory Visi-
bility
hipStreamSynchronizehostwaitsforallcommandsinthespec-
ified stream to complete
system-
scope
release
yes yes
hipDeviceSynchronizehost waits for all commands in all
streams on the specified device to com-
plete
system-
scope
release
yes yes
hipEventSynchronizehostwaitsforthespecifiedeventtocom-
plete
device-
scope
release
yes depends - see below
hipStreamWaitEventstream waits for the specified event to
complete
none yes no
8.1.6 hipEventSynchronize

## Developers can control the release scope forhipEvents

• By default, the GPU performs a device-scope acquire and release operation with each recorded event. This will
make host and device memory visible to other commands executing on the same device.
A stronger system-level fence can be specified when the event is created withhipEventCreateWithFlags:
• hipEventReleaseToSystem: Perform a system-scope release operation when the event is recorded. This will
makebothCoherentandNon-Coherenthostmemoryvisibletootheragentsinthesystem,butmayinvolveheavy-

```cpp

weightoperationssuchascacheflushing. Coherentmemorywilltypicallyuselighter-weightin-kernelsynchro-

```

nization mechanisms such as an atomic operation and thus does not need to usehipEventReleaseToSystem.
• hipEventDisableTiming: Events created with this flag will not record profiling data and provide the best
performance if used for synchronization.
8.1.7 Summary and Recommendations
• CoherenthostmemoryisthedefaultandistheeasiesttousesincethememoryisvisibletotheCPUattypicalsyn-

```cpp

chronization points. This memory allows in-kernel synchronization commands such asthreadfence_system

```

to work transparently.
• HIP/ROCm also supports the ability to cache host memory in the GPU using the “Non-Coherent” host memory
allocations. This can provide performance benefit, but care must be taken to use the correct synchronization.
8.1. Host Memory 29
HIP Documentation, Release 6.3.42134
8.1.8 Managed memory allocation
Managed memory, including the__managed__ keyword, is supported in HIP combined host/device compilation, on
Linux, not on Windows (under development).
Managedmemory,viaunifiedmemoryallocation,allowsdatabesharedandaccessibletoboththeCPUandGPUusing
asinglepointer. TheallocationwillbemanagedbyAMDGPUdriverusingtheLinuxHMM(HeterogeneousMemory

```cpp

Management) mechanism, the user can call managed memory APIhipMallocManaged to allocate a large chunk of
HMM memory, execute kernels on device and fetch data between the host and device as needed.

```

In HIP application, it is recommended to do the capability check before calling the managed memory APIs. For

## example

int managed_memory = 0;
HIPCHECK(hipDeviceGetAttribute(&managed_memory,
hipDeviceAttributeManagedMemory,p_gpuDevice));
if (!managed_memory ) {
printf ("info: managed memory access not supported on the device %d\n Skipped\n", p_
˓→gpuDevice);
}
else {
HIPCHECK(hipSetDevice(p_gpuDevice));

```cpp

HIPCHECK(hipMallocManaged(&Hmm, N * sizeof(T)));

```

. . .
}
Pleasenote,themanagedmemorycapabilitycheckmaynotbenecessary,butifHMMisnotsupported,thenmanaged
malloc will fall back to using system memory and other managed memory API calls will have undefined behavior.
Note, managed memory management is implemented on Linux, not supported on Windows yet.
8.1.9 HIP Stream Memory Operations
HIP supports Stream Memory Operations to enable direct synchronization between Network Nodes and GPU.
Following new APIs are added, hipStreamWaitValue32 hipStreamWaitValue64 hipStreamWriteValue32
hipStreamWriteValue64
Note, CPU access to the semaphore’s memory requires volatile keyword to disable CPU compiler’s optimizations on
memory access. For more details, please check the documentationHIP-API.pdf.
Please note, HIP stream does not guarantee concurrency on AMD hardware for the case of multiple (at least 6) long-
running streams executing concurrently, usinghipStreamSynchronize(nullptr) for synchronization.
8.2 Direct Dispatch
HIPruntimehasDirectDispatchenabledbydefaultinROCM4.4onLinux. Withthisfeaturewemoveawayfromour
conventional producer-consumer model where the runtime creates a worker thread(consumer) for each HIP Stream,
and the host thread(producer) enqueues commands to a command queue(per stream).
For Direct Dispatch, HIP runtime would directly enqueue a packet to the AQL queue (user mode queue on GPU) on
the Dispatch API call from the application. That has shown to reduce the latency to launch the first wave on the idle
GPU and total time of tiny dispatches synchronized with the host.
30 Chapter 8. HIP Programming Manual
HIP Documentation, Release 6.3.42134
Inaddition,eliminatingthethreadsinruntimehasreducedthevarianceinthedispatchnumbersasthethreadscheduling
delays and atomics/locks synchronization latencies are reduced.
This feature can be disabled via setting the following environment variable, AMD_DIRECT_DISPATCH=0
Note, Direct Dispatch is implemented on Linux. It is currently not supported on Windows.
8.3 HIP Runtime Compilation
HIP now supports runtime compilation (HIP RTC), the usage of which will provide the possibility of optimizations
and performance improvement compared with other APIs via regular offline static compilation.
HIP RTC APIs accept HIP source files in character string format as input parameters and create handles of programs
by compiling the HIP source files without spawning separate processes.
For more details on HIP RTC APIs, refer to HIP Runtime API Reference.
For Linux developers, the link here shows an example how to program HIP application using runtime compilation
mechanism, and a detailedHIP RTC programming guideis also available.
8.4 HIP Graph
HIP graph is supported. For more details, refer to the HIP API Guide.
8.5 Device-Side Malloc
HIP-Clang now supports device-side malloc and free. This implementation does not require the use of
hipDeviceSetLimit(hipLimitMallocHeapSize,value)nor respects any setting. The heap is fully dynamic and
can grow until the available free memory on the device is consumed.
8.6 Use of Per-thread default stream
Theper-threaddefaultstreamissupportedinHIP.Itisanimplicitstreamlocaltoboththethreadandthecurrentdevice.
Thismeansthatthecommandissuedtotheper-threaddefaultstreambythethreaddoesnotimplicitlysynchronizewith
other streams (like explicitly created streams), or default per-thread stream on other threads. The per-thread default
stream is a blocking stream and will synchronize with the default null stream if both are used in a program. The
per-thread default stream can be enabled via adding a compilation option,-fgpu-default-stream=per-thread.
And users can explicitly usehipStreamPerThread as per-thread default stream handle as input in API commands.
There are test codes as examples in the link.
8.3. HIP Runtime Compilation 31
HIP Documentation, Release 6.3.42134
8.7 Use of Long Double Type
In HIP-Clang, long double type is 80-bit extended precision format for x86_64, which is not supported by AMDGPU.
HIP-ClangtreatslongdoubletypeasIEEEdoubletypeforAMDGPU.UsinglongdoubletypeinHIPsourcecodewill
not cause issue as long as data of long double type is not transferred between host and device. However, long double

```cpp

type should not be used as kernel argument type.

```

8.8 Use of_Float16 Type
If a host function is to be used between clang (or hipcc) and gcc for x86_64, i.e. its definition is compiled by one
compiler but the caller is compiled by a different compiler,_Float16 or aggregates containing_Float16 should not
be used as function argument or return type. This is due to lack of stable ABI for_Float16 on x86_64. Passing
_Float16 or aggregates containing_Float16 between clang and gcc could cause undefined behavior.
8.9 FMA and contractions
By default HIP-Clang assumes -ffp-contract=fast-honor-pragmas. Users can use #pragma clang fp
contract(on|off|fast) to controlfp contraction of a block of code. For x86_64, FMA is off by default since
the generic x86_64 target does not support FMA by default. To turn on FMA on x86_64, either use-mfma or
-march=nativeon CPU’s supporting FMA.
WhencontractionsareenabledandtheCPUhasnotenabledFMAinstructions,theGPUcanproducedifferentnumeri-
calresultsthantheCPUforexpressionsthatcanbecontracted. Toleranceshouldbeusedforfloatingpointcomparisons.
8.10 Math functions with special rounding modes
Note: Currently, HIP only supports basic math functions with rounding modern (round to nearest). HIP does not
support basic math functions with rounding modesru (round up),rd (round down), andrz(round towards zero).
8.11 Creating Static Libraries
HIP-Clang supports generating two types of static libraries. The first type of static library does not export device
functions, and only exports and launches host functions within the same library. The advantage of this type is the
ability to link with a non-hipcc compiler such as gcc. The second type exports device functions to be linked by other
code objects. However, this requires using hipcc as the linker.
In addition, the first type of library contains host objects with device code embedded as fat binaries. It is generated
using the flag –emit-static-lib. The second type of library contains relocatable device objects and is generated using
ar.

## Here is an example to create and use static libraries


## • Type 1 using--emit-static-lib

hipcc hipOptLibrary.cpp --emit-static-lib -fPIC -o libHipOptLibrary.a
gcc test.cpp -L. -lhipOptLibrary -L/path/to/hip/lib -lamdhip64 -o test.out

## • Type 2 using systemar

32 Chapter 8. HIP Programming Manual
HIP Documentation, Release 6.3.42134
hipcc hipDevice.cpp -c -fgpu-rdc -o hipDevice.o
ar rcsD libHipDevice.a hipDevice.o
hipcc libHipDevice.a test.cpp -fgpu-rdc -o test.out
For more information, please see HIP samples host functions and device_functions.
8.11. Creating Static Libraries 33
HIP Documentation, Release 6.3.42134
34 Chapter 8. HIP Programming Manual

## CHAPTER


## NINE


## HIP PORTING GUIDE

In addition to providing a portable C++ programming environment for GPUs, HIP is designed to ease the porting
of existing CUDA code into the HIP environment. This section describes the available tools and provides practical
suggestions on how to port CUDA code and work through common issues.
9.1 Porting a New CUDA Project
9.1.1 General Tips
• Starting the port on a CUDA machine is often the easiest approach, since you can incrementally port pieces of
the code to HIP while leaving the rest in CUDA. (Recall that on CUDA machines HIP is just a thin layer over
CUDA, so the two code types can interoperate on NVCC platforms.) Also, the HIP port can be compared with
the original CUDA code for function and performance.
• Once the CUDA code is ported to HIP and is running on the CUDA machine, compile the HIP code using the
HIP compiler on an AMD machine.
• HIPportscanreplaceCUDAversions: HIPcandeliverthesameperformanceasanativeCUDAimplementation,
with the benefit of portability to both NVIDIA and AMD architectures as well as a path to future C++ standard
support. You can handle platform-specific features through conditional compilation or by adding them to the
open-source HIP infrastructure.
• Use hipconvertinplace-perl.shto hipify all code files in the CUDA source directory.
9.1.2 Scanning existing CUDA code to scope the porting effort
Thehipexamine-perl.shtoolwillscanasourcedirectorytodeterminewhichfilescontainCUDAcodeandhowmuch
of that code can be automatically hipified.
> cd examples/rodinia_3.0/cuda/kmeans
> $HIP_DIR/bin/hipexamine-perl.sh.
info: hipify ./kmeans.h =====>
info: hipify ./unistd.h =====>
info: hipify ./kmeans.c =====>

```cpp

info: hipify ./kmeans_cuda_kernel.cu =====>

```

info: converted 40 CUDA->HIP refs( dev:0 mem:0 kern:0 builtin:37 math:0 stream:0␣
˓→event:0 err:0 def:0 tex:3 other:0 ) warn:0 LOC:185
info: hipify ./getopt.h =====>
info: hipify ./kmeans_cuda.cu =====>
info: converted 49 CUDA->HIP refs( dev:3 mem:32 kern:2 builtin:0 math:0 stream:0␣
(continues on next page)
35
HIP Documentation, Release 6.3.42134
(continued from previous page)
˓→event:0 err:0 def:0 tex:12 other:0 ) warn:0 LOC:311
info: hipify ./rmse.c =====>
info: hipify ./cluster.c =====>
info: hipify ./getopt.c =====>
info: hipify ./kmeans_clustering.c =====>
info: TOTAL-converted 89 CUDA->HIP refs( dev:3 mem:32 kern:2 builtin:37 math:0 stream:0␣
˓→event:0 err:0 def:0 tex:15 other:0 ) warn:0 LOC:3607

```cpp

kernels (1 total) : kmeansPoint(1)

```

hipexamine-perl scans each code file (cpp, c, h, hpp, etc.) found in the specified directory:
• Files with no CUDA code (kmeans.h) print one line summary just listing the source file name.

```cpp

• Files with CUDA code print a summary of what was found - for example thekmeans_cuda_kernel.cu file:
info: hipify ./kmeans_cuda_kernel.cu =====>

```

info: converted 40 CUDA->HIP refs( dev:0 mem:0 kern:0 builtin:37 math:0 stream:0␣
˓→event:0

## • Interesting information inkmeans_cuda_kernel.cu 

– How many CUDA calls were converted to HIP (40)
– Breakdown of the CUDA functionality used (dev:0 mem:0 etc). This file uses many CUDA builtins (37)
and texture functions (3).
– Warning for code that looks like CUDA API but was not converted (0 in this file).
– Count Lines-of-Code (LOC) - 185 for this file.
• hipexamine-perlalsopresentsasummaryattheendoftheprocessforthestatisticscollectedacrossallfiles. This

```cpp

has similar format to the per-file reporting, and also includes a list of all kernels which have been called. An

## example from above


```

info: TOTAL-converted 89 CUDA->HIP refs( dev:3 mem:32 kern:2 builtin:37 math:0 stream:0␣
˓→event:0 err:0 def:0 tex:15 other:0 ) warn:0 LOC:3607

```cpp

kernels (1 total) : kmeansPoint(1)

```

9.1.3 Converting a project “in-place”
> hipify-perl --inplace

## For each input file FILE, this script will

• If FILE.prehip file does not exist, copy the original code to a new file with extension.prehip. Then hipify
the code file.
• If FILE.prehipfile exists, hipifyFILE.prehipand save to FILE.
This is useful for testing improvements to the hipify toolset.
The hipconvertinplace-perl.sh script will perform inplace conversion for all code files in the specified directory. This
can be quite handy when dealing with an existing CUDA code base since the script preserves the existing directory
structure and filenames - and includes work. After converting in-place, you can review the code to add additional
parameters to directory names.
36 Chapter 9. HIP Porting Guide
HIP Documentation, Release 6.3.42134
> hipconvertinplace-perl.sh MY_SRC_DIR
9.1.4 Library Equivalents
Most CUDA libraries have a corresponding ROCm library with similar functionality and APIs. However, ROCm also
provides HIP marshalling libraries that greatly simplify the porting process because they more precisely reflect their
CUDAcounterpartsandcanbeusedwitheithertheAMDorNVIDIAplatforms(see“IdentifyingHIPTargetPlatform”

## below). There are a few notable exceptions

• MIOpen does not have a marshalling library interface to ease porting from cuDNN.
• RCCL is a drop-in replacement for NCCL and implements the NCCL APIs.
• hipBLASLt does not have a ROCm library but can still target the NVIDIA platform, as needed.
• EIGEN’s HIP support is part of the library.
CUDA Li-
brary
HIP Li-
brary
ROCm Li-
brary
Comment
cuBLAS hipBLAS rocBLAS Basic Linear Algebra Subroutines
cuBLASLt hip-
BLASLt
N/A BasicLinearAlgebraSubroutines,lightweightandnewflexibleAPI
cuFFT hipFFT rocFFT Fast Fourier Transfer Library
cuSPARSE hipSPARSE rocSPARSE Sparse BLAS + SPMV
cuSOLVER hip-

## SOLVER

rocSOLVER Lapack library
AmgX N/A rocALU-

## TION

Sparseiterativesolversandpreconditionerswithalgebraicmultigrid
Thrust N/A rocThrust C++ parallel algorithms library
CUB hipCUB rocPRIM Low Level Optimized Parallel Primitives
cuDNN N/A MIOpen Deep learning Solver Library
cuRAND hipRAND rocRAND Random Number Generator Library
EIGEN EIGEN N/A C++ template library for linear algebra: matrices, vectors, numeri-
cal solvers,
NCCL N/A RCCL Communications Primitives Library based on the MPI equivalents
9.2 Distinguishing Compiler Modes
9.2.1 Identifying HIP Target Platform
All HIP projects target either AMD or NVIDIA platform. The platform affects which headers are included and which
libraries are used for linking.
• HIP_PLATFORM_AMD is defined if the HIP platform targets AMD. Note,HIP_PLATFORM_HCC was previously
defined if the HIP platform targeted AMD, it is deprecated.
• HIP_PLATFORM_NVDIA is defined if the HIP platform targets NVIDIA. Note,HIP_PLATFORM_NVCC was previ-
ously defined if the HIP platform targeted NVIDIA, it is deprecated.
9.2. Distinguishing Compiler Modes 37
HIP Documentation, Release 6.3.42134
9.2.2 Identifying the Compiler: hip-clang or NVCC
Often,it’susefultoknowwhethertheunderlyingcompilerisHIP-ClangorNVCC.Thisknowledgecanguardplatform-
specific code or aid in platform-specific performance tuning.
#ifdef __HIP_PLATFORM_AMD__
// Compiled with HIP-Clang
#endif
#ifdef __HIP_PLATFORM_NVIDIA__
// Compiled with nvcc
// Could be compiling with CUDA language extensions enabled (for example, a ".cu file)
// Could be in pass-through mode to an underlying host compile OR (for example, a .cpp ␣
˓→file)
#ifdef __CUDACC__
// Compiled with nvcc (CUDA language extensions enabled)
Compiler directly generates the host code (using the Clang x86 target) and passes the code to another host compiler.
Thus, they have no equivalent of the__CUDACC__ define.
9.2.3 Identifying Current Compilation Pass: Host or Device
NVCCmakestwopassesoverthecode: oneforhostcodeandonefordevicecode. HIP-Clangwillhavemultiplepasses
over the code: one for the host code, and one for each architecture on the device code.__HIP_DEVICE_COMPILE__is

```cpp

set to a nonzero value when the compiler (HIP-Clang or NVCC) is compiling code for a device inside a__global__
kernel or for a device function.__HIP_DEVICE_COMPILE__ can replace#ifdef checks on the__CUDA_ARCH__

```

define.
// #ifdef __CUDA_ARCH__
#if __HIP_DEVICE_COMPILE__
Unlike__CUDA_ARCH__, the__HIP_DEVICE_COMPILE__ value is 1 or undefined, and it doesn’t represent the feature
capability of the target device.
38 Chapter 9. HIP Porting Guide
HIP Documentation, Release 6.3.42134
9.2.4 Compiler Defines: Summary
Define HIP-Clang NVCC Other (GCC, ICC,
Clang, etc.)
HIP-related

## defines

__HIP_PLATFORM_AMD__Defined Undefined DefinediftargetingAMD
platform; undefined oth-
erwise
__HIP_PLATFORM_NVIDIA__Undefined Defined Defined if targeting
NVIDIA platform; unde-
fined otherwise
__HIP_DEVICE_COMPILE__1ifcompilingfordevice;un-
defined if compiling for host
1 if compiling for device; undefined if
compiling for host
Undefined
__HIPCC__ Defined Defined Undefined
__HIP_ARCH_*0 or 1 depending on feature
support (see below)
0 or 1 depending on feature support
(see below)
0

## NVCC-

related

## defines

__CUDACC__ Defined if source code is
compiled by NVCC; unde-
fined otherwise
Undefined

## __NVCC__

Undefined
Defined Undefined
__CUDA_ARCH__Undefined Unsigned representing compute capa-
bility (e.g., “130”) if in device code; 0
if in host code
Undefined
hip-clang-
related

## defines

__HIP__ Defined Undefined Undefined
HIP-Clang
common

## defines

__clang__ Defined Defined Undefined
9.3 Identifying Architecture Features
9.3.1 HIP_ARCH Defines
Some CUDA code tests__CUDA_ARCH__ for a specific value to determine whether the machine supports a certain
architectural feature. For instance,
#if (__CUDA_ARCH__ >= 130)
// doubles are supported
This type of code requires special attention, since AMD and CUDA devices have different architectural capabilities.
Moreover, you can’t determine the presence of a feature using a simple comparison against an architecture’s version
9.3. Identifying Architecture Features 39
HIP Documentation, Release 6.3.42134
number. HIP provides a set of defines and device properties to query whether a specific architectural feature is sup-
ported.

## The__HIP_ARCH_*defines can replace comparisons of__CUDA_ARCH__ values

//#if (__CUDA_ARCH__ >= 130) // non-portable
if __HIP_ARCH_HAS_DOUBLES__ { // portable HIP feature query
// doubles are supported
}
Forhostcode,the __HIP_ARCH__*definesaresetto0. Youshouldonlyusethe __HIP_ARCH__fieldsindevicecode.
9.3.2 Device-Architecture Properties
Hostcodeshouldquerythearchitecturefeatureflagsinthedevicepropertiesthat hipGetDevicePropertiesreturns,

## rather than testing the “major” and “minor” fields directly

hipGetDeviceProperties(&deviceProp, device);
//if ((deviceProp.major == 1 && deviceProp.minor < 2)) // non-portable
if (deviceProp.arch.hasSharedInt32Atomics) { // portable HIP feature query
// has shared int32 atomic operations ...
}
9.3.3 Table of Architecture Properties
The table below shows the full set of architectural properties that HIP supports.
40 Chapter 9. HIP Porting Guide
HIP Documentation, Release 6.3.42134
Define (use only in device code) Device Property (run-
time query)
Comment

## 32-bit atomics

__HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__hasGlobalInt32Atomics 32-bit integer atomics for global memory
__HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__hasGlobalFloatAtomicExch32-bitfloatatomicexchangeforglobalmem-
ory
__HIP_ARCH_HAS_SHARED_INT32_ATOMICS__hasSharedInt32Atomics 32-bit integer atomics for shared memory
__HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__hasSharedFloatAtomicExch32-bitfloatatomicexchangeforsharedmem-
ory
__HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__hasFloatAtomicAdd 32-bit float atomic add in global and shared
memory

## 64-bit atomics

__HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__hasGlobalInt64Atomics 64-bit integer atomics for global memory
__HIP_ARCH_HAS_SHARED_INT64_ATOMICS__hasSharedInt64Atomics 64-bit integer atomics for shared memory

## Doubles

__HIP_ARCH_HAS_DOUBLES__ hasDoubles Double-precision floating point

## Warp cross-lane operations

__HIP_ARCH_HAS_WARP_VOTE__ hasWarpVote Warp vote instructions (any,all)
__HIP_ARCH_HAS_WARP_BALLOT__ hasWarpBallot Warp ballot instructions
__HIP_ARCH_HAS_WARP_SHUFFLE__ hasWarpShuffle Warp shuffle operations (shfl_*)
__HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__hasFunnelShift Funnel shift two input words into one

## Sync

__HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__hasThreadFenceSystem threadfence_system
__HIP_ARCH_HAS_SYNC_THREAD_EXT__hasSyncThreadsExt syncthreads_count, syncthreads_and,
syncthreads_or

## Miscellaneous

__HIP_ARCH_HAS_SURFACE_FUNCS__ hasSurfaceFuncs
__HIP_ARCH_HAS_3DGRID__ has3dGrid Grids and groups are 3D
__HIP_ARCH_HAS_DYNAMIC_PARALLEL__hasDynamicParallelism
9.4 Finding HIP
Makefiles can use the following syntax to conditionally provide a default HIP_PATH if one does not exist:
HIP_PATH ?= $(shell hipconfig --path)
9.4. Finding HIP 41
HIP Documentation, Release 6.3.42134
9.5 Identifying HIP Runtime
HIP can depend on rocclr, or CUDA as runtime
• AMD platform On AMD platform, HIP uses ROCm Compute Language Runtime, called ROCclr. ROCclr is
a virtual device interface that HIP runtimes interact with different backends which allows runtimes to work on
Linux , as well as Windows without much efforts.
• NVIDIA platform On NVIDIA platform, HIP is just a thin layer on top of CUDA. On non-AMD platform, HIP
runtime determines if CUDA is available and can be used. If available, HIP_PLATFORM is set tonvidia and
underneath CUDA path is used.
9.6 hipLaunchKernelGGL

```cpp

hipLaunchKernelGGL is a macro that can serve as an alternative way to launch kernel, which accepts parameters of
launch configurations (grid dims, group dims, stream, dynamic shared size) followed by a variable number of kernel

```

arguments. It can replace <<< >>>, if the user so desires.
9.7 Compiler Options
hipcc is a portable compiler driver that will call NVCC or HIP-Clang (depending on the target system) and attach all
requiredincludeandlibraryoptions. Itpassesoptionsthroughtothetargetcompiler. Toolsthatcallhipccmustensure
thecompileroptionsareappropriateforthetargetcompiler. The hipconfigscriptmayhelpfulinidentifyingthetarget
platform, compiler and runtime. It can also help set options appropriately.
9.7.1 Compiler options supported on AMD platforms
Here are the main compiler options supported on AMD platforms by HIP-Clang.
Option Description
--amdgpu-target=<gpu_arch>[DEPRECATED] This option is being replaced by--offload-arch=<target>. Generate code
for the given GPU target. Supported targets are gfx701, gfx801, gfx802, gfx803, gfx900, gfx906,
gfx908, gfx1010, gfx1011, gfx1012, gfx1030, gfx1031. This option could appear multiple times on
the same command line to generate a fat binary for multiple targets.

```cpp

--fgpu-rdc Generate relocatable device code, which allows kernels or device functions calling device functions

```

in different translation units.
-ggdb Equivalent to-g plus tuning for GDB. This is recommended when using ROCm’s GDB to debug
GPU code.
--gpu-max-threads-per-block=<num>Generate code to support up to the specified number of threads per block.
-O<n> Specify the optimization level.
-offload-arch=<target>Specify the AMD GPU target ID.
-save-tempsSave the compiler generated intermediate files.
-v Show the compilation steps.
42 Chapter 9. HIP Porting Guide
HIP Documentation, Release 6.3.42134
9.8 Linking Issues
9.8.1 Linking With hipcc
hipcc adds the necessary libraries for HIP as well as for the accelerator compiler (NVCC or AMD compiler). We
recommend linking with hipcc since it automatically links the binary to the necessary HIP runtime libraries. It also
has knowledge on how to link and to manage the GPU objects.
9.8.2 -lm Option
hipcc adds-lm by default to the link command.
9.9 Linking Code With Other Compilers

```cpp

CUDA code often uses NVCC for accelerator code (defining and launching kernels, typically defined in.cu or .cuh

```

files). It also uses a standard compiler (g++) for the rest of the application. NVCC is a preprocessor that employs a
standardhostcompiler(gcc)togeneratethehostcode. Codecompiledusingthistoolcanemployonlytheintersection
of language features supported by both NVCC and the host compiler. In some cases, you must take care to ensure the
data types and alignment of the host compiler are identical to those of the device compiler. Only some host compilers
are supported—for example, recent NVCC versions lack Clang host-compiler capability.
HIP-Clang generates both device and host code using the same Clang-based compiler. The code uses the same API
as gcc, which allows code generated by different gcc-compatible compilers to be linked together. For example, code
compiledusingHIP-Clangcanlinkwithcodecompiledusing“standard”compilers(suchasgcc,ICCandClang). Take
care to ensure all compilers use the same standard C++ header and library formats.
9.9.1 libc++ and libstdc++
hipcc links to libstdc++ by default. This provides better compatibility between g++ and HIP.
If you pass--stdlib=libc++ to hipcc, hipcc will use the libc++ library. Generally, libc++ provides a broader set of
C++ features while libstdc++ is the standard for more compilers (notably including g++).
When cross-linking C++ code, any C++ functions that use types from the C++ standard library (including std::string,
std::vector and other containers) must use the same standard-library implementation. They include the following:

```cpp

• Functions or kernels defined in HIP-Clang that are called from a standard compiler

```

• Functions defined in a standard compiler that are called from HIP-Clang.
Applications with these interfaces should use the default libstdc++ linking.
Applications which are compiled entirely with hipcc, and which benefit from advanced C++ features not supported in
libstdc++, and which do not require portability to NVCC, may choose to use libc++.
9.8. Linking Issues 43
HIP Documentation, Release 6.3.42134
9.9.2 HIP Headers (hip_runtime.h, hip_runtime_api.h)
Thehip_runtime.handhip_runtime_api.hfiles define the types, functions and enumerations needed to compile

## a HIP program


```cpp

• hip_runtime_api.h: defines all the HIP runtime APIs (e.g.,hipMalloc) and the types required to call
them. A source file that is only calling HIP APIs but neither defines nor launches any kernels can include

```

hip_runtime_api.h. hip_runtime_api.h uses no custom Heterogeneous Compute (HC) language features
and can be compiled using a standard C++ compiler.
• hip_runtime.h: included inhip_runtime_api.h. It additionally provides the types and defines required to

```cpp

create and launch kernels. hip_runtime.h can be compiled using a standard C++ compiler but will expose a

```

subset of the available functions.
CUDAhasslightlydifferentcontentsforthesetwofiles. Insomecasesyoumayneedtoconverthipifiedcodetoinclude
the richerhip_runtime.h instead ofhip_runtime_api.h.
9.9.3 Using a Standard C++ Compiler
You can compilehip_runtime_api.h using a standard C or C++ compiler (e.g., gcc or ICC). The HIP include
paths and defines (__HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__) must pass to the standard compiler;

## hipconfig then returns the necessary options

> hipconfig --cxx_config
-D__HIP_PLATFORM_AMD__ -I/home/user1/hip/include
You can capture thehipconfig output and passed it to the standard compiler; below is a sample makefile syntax:
CPPFLAGS += $(shell $(HIP_PATH)/bin/hipconfig --cpp_config)
NVCCincludessomeheadersbydefault. However,HIPdoesnotincludedefaultheaders,andinsteadallrequiredfiles

```cpp

mustbeexplicitlyincluded. Specifically,filesthatcallHIPrun-timeAPIsordefineHIPkernelsmustexplicitlyinclude

```

theappropriateHIPheaders. IfthecompilationprocessreportsthatitcannotfindnecessaryAPIs(forexample, error:
identifier hipSetDevice is undefined), ensure that the file includes hip_runtime.h (or hip_runtime_api.h, if
appropriate). The hipify-perl script automatically convertscuda_runtime.h to hip_runtime.h, and it converts
cuda_runtime_api.htohip_runtime_api.h, but it may miss nested headers or macros.
9.9.3.1 cuda.h
TheHIP-Clang pathprovides anempty cuda.hfile. Someexisting CUDAprograms includethis filebut don’trequire
any of the functions.
9.9.4 Choosing HIP File Extensions
Many existing CUDA projects use the.cu and .cuh file extensions to indicate code that should be run through the
NVCCcompiler. ForquickHIPports,leavingthesefileextensionsunchangedisofteneasier,asitminimizesthework

```cpp

required to change file names in the directory and #include statements in the files.

```

Fornewprojectsorportswhichcanbere-factored,werecommendtheuseoftheextension .hip.cppforsourcefiles,
and.hip.hor.hip.hppforheaderfiles. ThisindicatesthatthecodeisstandardC++code,butalsoprovidesaunique
indication for make tools to run hipcc when appropriate.
44 Chapter 9. HIP Porting Guide
HIP Documentation, Release 6.3.42134
9.10 Workarounds
9.10.1 warpSize
Code should not assume a warp size of 32 or 64. See Warp Cross-Lane Functions for information on how to write
portable wave-aware code.
9.10.2 Kernel launch with group size > 256
Kernel code should use__attribute__((amdgpu_flat_work_group_size(<min>,<max>))).

## For example


```cpp

__global__ void dot(double *a,double *b,const int n) __attribute__((amdgpu_flat_work_

```

˓→group_size(1, 512)))
9.11 memcpyToSymbol

```cpp

HIP support forhipMemcpyToSymbol is complete. This feature allows a kernel to define a device-side data symbol

```

which can be accessed on the host side. The symbol can be in __constant or device space.
Note that the symbol name needs to be encased in the HIP_SYMBOL macro, as shown in the code example below.

```cpp

This also applies tohipMemcpyFromSymbol,hipGetSymbolAddress, andhipGetSymbolSize.

## For example


## Device Code

#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include<iostream>

```

#define HIP_ASSERT(status) \
assert(status == hipSuccess)
#define LEN 512
#define SIZE 2048
__constant__ int Value[LEN];

```cpp

__global__ void Get(hipLaunchParm lp, int *Ad)

```

{

```cpp

int tid = threadIdx.x + blockIdx.x * blockDim.x;

```

Ad[tid] = Value[tid];
}
int main()
{
int *A, *B, *Ad;
A = new int [LEN];
B = new int [LEN];
for(unsigned i=0;i<LEN;i++)
(continues on next page)
9.10. Workarounds 45
HIP Documentation, Release 6.3.42134
(continued from previous page)
{
A[i] = -1*i;
B[i] = 0;
}

```cpp

HIP_ASSERT(hipMalloc((void**)&Ad, SIZE));
HIP_ASSERT(hipMemcpyToSymbol(HIP_SYMBOL(Value), A, SIZE, 0, hipMemcpyHostToDevice));
hipLaunchKernelGGL(Get, dim3(1,1,1), dim3(LEN,1,1), 0, 0, Ad);
HIP_ASSERT(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));

```

for(unsigned i=0;i<LEN;i++)
{
assert(A[i] == B[i]);
}
std::cout<<"Passed"<<std::endl;
}

## 9.12 CU_POINTER_ATTRIBUTE_MEMORY_TYPE

To get pointer’s memory type in HIP/HIP-Clang, developers should usehipPointerGetAttributes API. First pa-
rameter of the API ishipPointerAttribute_t which has ‘type’ as member variable. ‘type’ indicates input pointer
is allocated on device or host.

## For example

double * ptr;

```cpp

hipMalloc(reinterpret_cast<void**>(&ptr), sizeof(double));

```

hipPointerAttribute_t attr;
hipPointerGetAttributes(&attr, ptr); /*attr.type will have value as hipMemoryTypeDevice*/
double* ptrHost;
hipHostMalloc(&ptrHost, sizeof(double));
hipPointerAttribute_t attr;
hipPointerGetAttributes(&attr, ptrHost); /*attr.type will have value as ␣
˓→hipMemoryTypeHost*/
Please note,hipMemoryType enum values are different fromcudaMemoryType enum values.
For example, on AMD platform,hipMemoryType is defined inhip_runtime_api.h,
typedef enum hipMemoryType {
hipMemoryTypeHost = 0, ///< Memory is physically located on host
hipMemoryTypeDevice = 1, ///< Memory is physically located on device. (see deviceId ␣
˓→for specific device)
hipMemoryTypeArray = 2, ///< Array memory, physically located on device. (see ␣
˓→deviceId for specific device)
hipMemoryTypeUnified = 3, ///< Not used currently
hipMemoryTypeManaged = 4 ///< Managed memory, automaticallly managed by the unified ␣
˓→memory system
} hipMemoryType;
46 Chapter 9. HIP Porting Guide
HIP Documentation, Release 6.3.42134
Looking into CUDA toolkit, it definescudaMemoryType as following,
enum cudaMemoryType
{
cudaMemoryTypeUnregistered = 0, // Unregistered memory.
cudaMemoryTypeHost = 1, // Host memory.
cudaMemoryTypeDevice = 2, // Device memory.
cudaMemoryTypeManaged = 3, // Managed memory
}
In this case, memory type translation forhipPointerGetAttributes needs to be handled properly on NVIDIA
platform to get the correct memory type in CUDA, which is done in the filenvidia_hip_runtime_api.h.
So in any HIP applications which use HIP APIs involving memory types, developers should use#ifdef in order to
assign the correct enum values depending on NVIDIA or AMD platform.
As an example, please see the code from the link.
With the#ifdef condition, HIP APIs work as expected on both AMD and NVIDIA platforms.
Note,cudaMemoryTypeUnregstered is currently not supported inhipMemoryType enum, due to HIP functionality
backward compatibility.
9.13 threadfence_system
threadfence_system makes all device memory writes, all writes to mapped host memory, and all writes to peer
memoryvisibletoCPUandotherGPUdevices. SomeimplementationscanprovidethisbehaviorbyflushingtheGPU
L2cache. HIP/HIP-Clangdoesnotprovidethisfunctionality. Asaworkaround,userscansettheenvironmentvariable

```cpp

HSA_DISABLE_CACHE=1todisabletheGPUL2cache. Thiswillaffectallaccessesandforallkernelsandsomayhave

```

a performance impact.
9.13.1 Textures and Cache Control
Compute programs sometimes use textures either to access dedicated texture caches or to use the texture-sampling
hardware for interpolation and clamping. The former approach uses simple point samplers with linear interpolation,
essentially only reading a single point. The latter approach uses the sampler hardware to interpolate and combine
multiplesamples. AMDhardware,aswellasrecentcompetinghardware,hasaunifiedtexture/L1cache,soitnolonger
has a dedicated texture cache. But the NVCC path often caches global loads in the L2 cache, and some programs may
benefit from explicit control of the L1 cache contents. We recommend the__ldginstruction for this purpose.
AMD compilers currently load all data into both the L1 and L2 caches, so__ldg is treated as a no-op.

## We recommend the following for functional portability

• For programs that use textures only to benefit from improved caching, use the__ldginstruction
• Programs that use texture object and reference APIs, work well on HIP
9.13. threadfence_system 47
HIP Documentation, Release 6.3.42134
9.14 More Tips
9.14.1 HIP Logging
OnanAMDplatform,settheAMD_LOG_LEVELenvironmentvariabletologHIPapplicationexecutioninformation.
The value of the setting controls different logging level,
enum LogLevel {

## LOG_NONE = 0,


## LOG_ERROR = 1,


## LOG_WARNING = 2,


## LOG_INFO = 3,


## LOG_DEBUG = 4

};
Logging mask is used to print types of functionalities during the execution of HIP application. It can be set as one of
the following values,
enum LogMask {
LOG_API = 1, //!< (0x1) API call
LOG_CMD = 2, //!< (0x2) Kernel and Copy Commands and Barriers
LOG_WAIT = 4, //!< (0x4) Synchronization and waiting for commands to ␣
˓→finish
LOG_AQL = 8, //!< (0x8) Decode and display AQL packets
LOG_QUEUE = 16, //!< (0x10) Queue commands and queue contents
LOG_SIG = 32, //!< (0x20) Signal creation, allocation, pool
LOG_LOCK = 64, //!< (0x40) Locks and thread-safety code.
LOG_KERN = 128, //!< (0x80) Kernel creations and arguments, etc.
LOG_COPY = 256, //!< (0x100) Copy debug
LOG_COPY2 = 512, //!< (0x200) Detailed copy debug
LOG_RESOURCE = 1024, //!< (0x400) Resource allocation, performance-impacting ␣
˓→events.
LOG_INIT = 2048, //!< (0x800) Initialization and shutdown
LOG_MISC = 4096, //!< (0x1000) Misc debug, not yet classified
LOG_AQL2 = 8192, //!< (0x2000) Show raw bytes of AQL packet
LOG_CODE = 16384, //!< (0x4000) Show code creation debug
LOG_CMD2 = 32768, //!< (0x8000) More detailed command info, including barrier ␣
˓→commands
LOG_LOCATION = 65536, //!< (0x10000) Log message location
LOG_MEM = 131072, //!< (0x20000) Memory allocation
LOG_MEM_POOL = 262144, //!< (0x40000) Memory pool allocation, including memory in ␣
˓→graphs
LOG_ALWAYS = -1 //!< (0xFFFFFFFF) Log always even mask flag is zero
};
48 Chapter 9. HIP Porting Guide
HIP Documentation, Release 6.3.42134
9.14.2 Debugging hipcc
To see the detailed commands that hipcc issues, set the environment variable HIPCC_VERBOSE to 1. Doing so will
print tostderr the HIP-clang (or NVCC) commands that hipcc generates.
export HIPCC_VERBOSE=1
make
...
hipcc-cmd: /opt/rocm/bin/hipcc --offload-arch=native -x hip backprop_cuda.cu
9.14.3 Editor Highlighting
See the utils/vim or utils/gedit directories to add handy highlighting to hip files.
9.14. More Tips 49
HIP Documentation, Release 6.3.42134
50 Chapter 9. HIP Porting Guide

## CHAPTER


## TEN


## PORTING CUDA DRIVER API

10.1 Introduction to the CUDA Driver and Runtime APIs
CUDA provides a separate CUDA Driver and Runtime APIs. The two APIs have significant overlap in functionality:
• Both APIs support events, streams, memory management, memory copy, and error handling.
• Both APIs deliver similar performance.
• Driver APIs calls begin with the prefixcu while Runtime APIs begin with the prefixcuda. For example, the
Driver API API containscuEventCreate while the Runtime API containscudaEventCreate, with similar
functionality.
• The Driver API defines a different but largely overlapping error code space than the Runtime API, and uses a
differentcodingconvention. Forexample,DriverAPIdefines CUDA_ERROR_INVALID_VALUEwhiletheRuntime
API definescudaErrorInvalidValue
The Driver API offers two additional pieces of functionality not provided by the Runtime API:cuModule and cuCtx
APIs.
10.1.1 cuModule API
The Module section of the Driver API provides additional control over how and when accelerator code objects are
loaded. For example, the driver API allows code objects to be loaded from files or memory pointers. Symbols for

```cpp

kernelsorglobaldatacanbeextractedfromtheloadedcodeobjects. Incontrast,theRuntimeAPIautomaticallyloads
and (if necessary) compiles all of the kernels from an executable binary when run. In this mode, NVCC must be used
to compile kernel code so the automatic loading can function correctly.
Both Driver and Runtime APIs define a function for launching kernels (called cuLaunchKernel or
cudaLaunchKernel. The kernel arguments and the execution configuration (grid dimensions, group dimen-

```

sions,dynamicsharedmemory,andstream)arepassedasargumentstothelaunchfunction. TheRuntimeadditionally

```cpp

provides the<<< >>> syntax for launching kernels, which resembles a special function call and is easier to use than
explicit launch API (in particular with respect to handling of kernel arguments). However, this syntax is not standard

```

C++ and is available only when NVCC is used to compile the host code.
The Module features are useful in an environment which generates the code objects directly, such as a new accelerator

```cpp

languagefront-end. Here,NVCCisnotused. Instead,theenvironmentmayhaveadifferentkernellanguageordifferent
compilation flow. Other environments have many kernels and do not want them to be all loaded automatically. The
Modulefunctionscanbeusedtoloadthegeneratedcodeobjectsandlaunchkernels. Aswewillseebelow,HIPdefines

```

a Module API which provides similar explicit control over code object management.
51
HIP Documentation, Release 6.3.42134
10.1.2 cuCtx API
The Driver API defines “Context” and “Devices” as separate entities. Contexts contain a single device, and a device
cantheoreticallyhavemultiplecontexts. Eachcontextcontainsasetofstreamsandeventsspecifictothecontext. His-
torically contexts also defined a unique address space for the GPU, though this may no longer be the case in Unified
Memory platforms (since the CPU and all the devices in the same process share a single unified address space). The
Context APIs also provide a mechanism to switch between devices, which allowed a single CPU thread to send com-
mandstodifferentGPUs. HIPaswellasarecentversionsofCUDARuntimeprovideothermechanismstoaccomplish
this feat - for example using streams orcudaSetDevice.
The CUDA Runtime API unifies the Context API with the Device API. This simplifies the APIs and has little loss of
functionality since each Context can contain a single device, and the benefits of multiple contexts has been replaced
withotherinterfaces. HIPprovidesacontextAPItofacilitateeasyportingfromexistingDrivercodes. InHIP,the Ctx
functions largely provide an alternate syntax for changing the active device.
Most new applications will prefer to usehipSetDevice or the stream APIs , therefore HIP has markedhipCtx APIs
as deprecated. Support for these APIs may not be available in future releases. For more details on deprecated APIs
please refer HIP deprecated APIs.
10.2 HIP Module andCtx APIs
Rather than present two separate APIs, HIP extends the HIP API with new APIs for Modules andCtx control.
10.2.1 hipModule API
Like the CUDA Driver API, the Module API provides additional control over how code is loaded, including options
to load code from files or from in-memory pointers. NVCC and HIP-Clang target different architectures and use
differentcodeobjectformats: NVCCis cubinorptxfiles,whiletheHIP-Clangpathisthe hsacoformat. Theexternal
compilerswhichgeneratethesecodeobjectsareresponsibleforgeneratingandloadingthecorrectcodeobjectforeach
platform. Notably,thereisnotafatbinaryformatthatcancontaincodeforbothNVCCandHIP-Clangplatforms. The

## following table summarizes the formats used on each platform

Format APIs NVCC HIP-CLANG
Code Object hipModuleLoad,hipModuleLoadData .cubin or PTX text .hsaco
Fat Binary hipModuleLoadFatBin .fatbin .hip_fatbin
hipccusesHIP-ClangorNVCCtocompilehostcodes. Bothofthesemayembedcodeobjectsintothefinalexecutable,
and these code objects will be automatically loaded when the application starts. ThehipModule API can be used to
load additional code objects, and in this way provides an extended capability to the automatically loaded code objects.
HIP-Clang allows both of these capabilities to be used together, if desired. Of course it is possible to create a program

```cpp

with no kernels and thus no automatic loading.

```

52 Chapter 10. Porting CUDA Driver API
HIP Documentation, Release 6.3.42134
10.2.2 hipCtx API
HIP provides aCtxAPI as a thin layer over the existing Device functions. ThisCtxAPI can be used to set the current
context,ortoquerypropertiesofthedeviceassociatedwiththecontext. Thecurrentcontextisimplicitlyusedbyother
APIs such ashipStreamCreate.
10.2.3 hipify translation of CUDA Driver API
The HIPIFY tools convert CUDA Driver APIs for streams, events, modules, devices, memory management, context,
profiler to the equivalent HIP driver calls. For example,cuEventCreate will be translated tohipEventCreate.
HIPIFY tools also convert error codes from the Driver namespace and coding convention to the equivalent HIP error
code. Thus, HIP unifies the APIs for these common functions.
ThememorycopyAPIrequiresadditionalexplanation. TheCUDAdriverincludesthememorydirectioninthenameof
theAPI( cuMemcpyH2D)whiletheCUDAdriverAPIprovidesasinglememorycopyAPIwithaparameterthatspecifies
the direction and additionally supports a “default” direction where the runtime determines the direction automatically.

```cpp

HIPprovidesAPIswithbothstyles: forexample, hipMemcpyH2Daswellas hipMemcpy. Thefirstflavormaybefaster

```

in some cases since they avoid host overhead to detect the different memory directions.
HIP defines a single error space, and uses camel-case for all errors (i.e.hipErrorInvalidValue).
10.2.3.1 Address Spaces
HIP-Clangdefinesaprocess-wideaddressspacewheretheCPUandalldevicesallocateaddressesfromasingleunified
pool. Thusaddressesmaybesharedbetweencontexts,andunliketheoriginalCUDAdefinitionanewcontextdoesnot
create a new address space for the device.
10.2.3.2 Using hipModuleLaunchKernel
hipModuleLaunchKerneliscuLaunchKernel in HIP world. It takes the same arguments ascuLaunchKernel.
10.2.3.3 Additional Information
• HIP-Clang creates a primary context when the HIP API is called. So in a pure driver API code, HIP-Clang will
create a primary context while HIP/NVCC will have empty context stack. HIP-Clang will push primary context
to context stack when it is empty. This can have subtle differences on applications which mix the runtime and
driver APIs.
10.2.4 hip-clang Implementation Notes
10.2.4.1 .hip_fatbin
hip-clanglinksdevicecodefromdifferenttranslationunitstogether. Foreachdevicetarget,acodeobjectisgenerated.
Codeobjectsfordifferentdevicetargetsarebundledby clang-offload-bundlerasonefatbinary,whichisembeded
as a global symbol__hip_fatbinin the.hip_fatbinsection of the ELF file of the executable or shared object.
10.2. HIP Module andCtx APIs 53
HIP Documentation, Release 6.3.42134
10.2.4.2 Initialization and Termination Functions
hip-clanggeneratesinitializationandterminationfunctionsforeachtranslationunitforhostcodecompilation. Theini-
tialization functions call__hipRegisterFatBinaryto register the fatbinary embeded in the ELF file. They also call

```cpp

__hipRegisterFunctionand__hipRegisterVartoregisterkernelfunctionsanddevicesideglobalvariables. The

```

termination functions call__hipUnregisterFatBinary. hip-clang emits a global variable__hip_gpubin_handle
ofvoid**typewithlinkoncelinkageandinitalvalue0foreachhosttranslationunit. Eachinitializationfunctionchecks
__hip_gpubin_handle and register the fatbinary only if__hip_gpubin_handle is 0 and saves the return value of
__hip_gpubin_handle to __hip_gpubin_handle. This is to guarantee that the fatbinary is only registered once.
Similar check is done in the termination functions.
10.2.4.3 Kernel Launching

```cpp

hip-clang supports kernel launching by CUDA<<<>>> syntax, hipLaunchKernelGGL. The latter one is macro which

```

expand to CUDA<<<>>> syntax.
When the executable or shared library is loaded by the dynamic linker, the initialization functions are called. In the

```cpp

initializationfunctions,when __hipRegisterFatBinaryiscalled,thecodeobjectscontainingallkernelsareloaded;
when __hipRegisterFunction is called, the stub functions are associated with the corresponding kernels in code

```

objects.

```cpp

hip-clang implements two sets of kernel launching APIs.

```

By default, in the host code, for the<<<>>> statement, hip-clang first emits call ofhipConfigureCall to set
up the threads and grids, then emits call of the stub function with the given arguments. In the stub function,

```cpp

hipSetupArgument is called for each kernel argument, thenhipLaunchByPtr is called with a function pointer to
the stub function. InhipLaunchByPtr, the real kernel associated with the stub function is launched.

```

10.2.5 NVCC Implementation Notes
10.2.5.1 Interoperation between HIP and CUDA Driver
CUDA applications may want to mix CUDA driver code with HIP code (see example below). This table shows the
type equivalence to enable this interaction.
HIP Type CU Driver Type CUDA Runtime Type
hipModule_t CUmodule
hipFunction_t CUfunction
hipCtx_t CUcontext
hipDevice_t CUdevice
hipStream_t CUstream cudaStream_t
hipEvent_t CUevent cudaEvent_t
hipArray CUarray cudaArray
54 Chapter 10. Porting CUDA Driver API
HIP Documentation, Release 6.3.42134
10.2.5.2 Compilation Options
The hipModule_t interface does not supportcuModuleLoadDataEx function, which is used to control PTX compi-
lation options. HIP-Clang does not use PTX and does not support these compilation options. In fact, HIP-Clang code
objectsalwayscontainfullycompiledISAanddonotrequireadditionalcompilationasapartoftheloadstep. Thecor-
responding HIP functionhipModuleLoadDataEx behaves ashipModuleLoadData on HIP-Clang path (compilation

## options are not used) and ascuModuleLoadDataExon NVCC path. For example (CUDA)

CUmodule module;
void *imagePtr = ...; // Somehow populate data pointer with code object
const int numOptions = 1;
CUJit_option options[numOptions];
void * optionValues[numOptions];
options[0] = CU_JIT_MAX_REGISTERS;
unsigned maxRegs = 15;
optionValues[0] = (void*)(&maxRegs);
cuModuleLoadDataEx(module, imagePtr, numOptions, options, optionValues);
CUfunction k;
cuModuleGetFunction(&k, module, "myKernel");

## HIP

hipModule_t module;
void *imagePtr = ...; // Somehow populate data pointer with code object
const int numOptions = 1;
hipJitOption options[numOptions];
void * optionValues[numOptions];
options[0] = hipJitOptionMaxRegisters;
unsigned maxRegs = 15;
optionValues[0] = (void*)(&maxRegs);
// hipModuleLoadData(module, imagePtr) will be called on HIP-Clang path, JIT options ␣
˓→will not be used, and
// cupModuleLoadDataEx(module, imagePtr, numOptions, options, optionValues) will be ␣
˓→called on NVCC path
hipModuleLoadDataEx(module, imagePtr, numOptions, options, optionValues);
hipFunction_t k;
hipModuleGetFunction(&k, module, "myKernel");
The below sample shows how to usehipModuleGetFunction.

```cpp

#include<hip_runtime.h>
#include<hip_runtime_api.h>
#include<iostream>
#include<fstream>
#include<vector>

```

(continues on next page)
10.2. HIP Module andCtx APIs 55
HIP Documentation, Release 6.3.42134
(continued from previous page)
#define LEN 64
#define SIZE LEN<<2
#ifdef __HIP_PLATFORM_AMD__
#define fileName "vcpy_isa.co"
#endif
#ifdef __HIP_PLATFORM_NVIDIA__
#define fileName "vcpy_isa.ptx"
#endif

```cpp

#define kernel_name "hello_world"

```

int main(){
float *A, *B;
hipDeviceptr_t Ad, Bd;
A = new float [LEN];
B = new float [LEN];
for(uint32_t i=0;i<LEN;i++){
A[i] = i*1.0f;
B[i] = 0.0f;
std::cout<<A[i] << " "<<B[i]<<std::endl;
}
#ifdef __HIP_PLATFORM_NVIDIA__
hipInit(0);
hipDevice_t device;
hipCtx_t context;
hipDeviceGet(&device, 0);
hipCtxCreate(&context, 0, device);
#endif

```cpp

hipMalloc((void**)&Ad, SIZE);
hipMalloc((void**)&Bd, SIZE);
hipMemcpyHtoD(Ad, A, SIZE);
hipMemcpyHtoD(Bd, B, SIZE);

```

hipModule_t Module;
hipFunction_t Function;
hipModuleLoad(&Module, fileName);

```cpp

hipModuleGetFunction(&Function, Module, kernel_name);

```

std::vector<void*>argBuffer(2);
memcpy(&argBuffer[0], &Ad, sizeof(void*));
memcpy(&argBuffer[1], &Bd, sizeof(void*));
size_t size = argBuffer.size()*sizeof(void*);
void *config[] = {
HIP_LAUNCH_PARAM_BUFFER_POINTER, &argBuffer[0],
(continues on next page)
56 Chapter 10. Porting CUDA Driver API
HIP Documentation, Release 6.3.42134
(continued from previous page)
HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,

## HIP_LAUNCH_PARAM_END

};
hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, NULL, (void**)&config);

```cpp

hipMemcpyDtoH(B, Bd, SIZE);

```

for(uint32_t i=0;i<LEN;i++){
std::cout<<A[i]<<" - "<<B[i]<<std::endl;
}
#ifdef __HIP_PLATFORM_NVIDIA__
hipCtxDetach(context);
#endif
return 0;
}
10.3 HIP Module and Texture Driver API
HIP supportstexture driver APIshowever texture referenceshould bedeclared in hostscope. Following code explains
the use of texture reference for__HIP_PLATFORM_AMD__ platform.
// Code to generate code object

```cpp

#include "hip/hip_runtime.h"

```

extern texture<float, 2, hipReadModeElementType> tex;

```cpp

__global__ void tex2dKernel(hipLaunchParm lp, float* outputData,

```

int width,
int height)
{

```cpp

int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

```

outputData[y*width + x] = tex2D(tex, x, y);
}

## // Host code

texture<float, 2, hipReadModeElementType> tex;
void myFunc ()
{
// ...
textureReference* texref;
hipModuleGetTexRef(&texref, Module1, "tex");
hipTexRefSetAddressMode(texref, 0, hipAddressModeWrap);
hipTexRefSetAddressMode(texref, 1, hipAddressModeWrap);
hipTexRefSetFilterMode(texref, hipFilterModePoint);
(continues on next page)
10.3. HIP Module and Texture Driver API 57
HIP Documentation, Release 6.3.42134
(continued from previous page)
hipTexRefSetFlags(texref, 0);
hipTexRefSetFormat(texref, HIP_AD_FORMAT_FLOAT, 1);
hipTexRefSetArray(texref, array, HIP_TRSA_OVERRIDE_FORMAT);
// ...
}
58 Chapter 10. Porting CUDA Driver API

## CHAPTER


## ELEVEN


## PROGRAMMING FOR HIP RUNTIME COMPILER (RTC)


```cpp

HIP lets you compile kernels at runtime with thehiprtc* APIs. Kernels can be stored as a text string and can be

```

passed to HIPRTC APIs alongside options to guide the compilation.

## NOTE

• This library can be used on systems without HIP installed nor AMD GPU driver installed at all (offline compi-
lation). Therefore, it does not depend on any HIP runtime library.
• But it does depend on COMGr. You may try to statically link COMGr into HIPRTC to avoid any ambiguity.
• Developers can decide to bundle this library with their application.
11.1 Compilation APIs

```cpp

To use HIPRTC functionality, HIPRTC header needs to be included first.#include <hip/hiprtc.h>

## Kernels can be stored in a string

static constexpr auto kernel_source {

## R"(


```

extern "C"

```cpp

__global__ void vector_add(float* output, float* input1, float* input2, size_t size)

```

˓→{

```cpp

int i = threadIdx.x;

```

if (i < size) {
output[i] = input1[i] + input2[i];
}
}
)"};

```cpp

Now to compile this kernel, it needs to be associated withhiprtcProgram type, which is done by declaring

## hiprtcProgram prog; and associating the string of kernel with this program


```

hiprtcCreateProgram(&prog, // HIPRTC program handle

```cpp

kernel_source, // HIP kernel source string

```

"vector_add.cpp", // Name of the HIP program, can be null or an ␣
˓→empty string
0, // Number of headers
NULL, // Header sources
NULL); // Name of header files
59
HIP Documentation, Release 6.3.42134
hiprtcCreateProgram API also allows you to add headers which can be included in your RTC program. For on-
line compilation, the compiler pre-defines HIP device API functions, HIP specific types and macros for device com-
pilation, but does not include standard C/C++ headers by default. Users can only include header files provided to
hiprtcCreateProgram.

```cpp

After associating the kernel string withhiprtcProgram, you can now compile this program using:

```

hiprtcCompileProgram(prog, // hiprtcProgram
0, // Number of options
options); // Clang Options [Supported Clang Options](clang_options.
˓→md)
hiprtcCompileProgram returns a status value which can be converted to string viahiprtcGetErrorString. If
compilation is successful,hiprtcCompileProgram will returnHIPRTC_SUCCESS.

## If the compilation fails, you can look up the logs via

size_t logSize;
hiprtcGetProgramLogSize(prog, &logSize);
if (logSize) {
string log(logSize, '\0');
hiprtcGetProgramLog(prog, &log[0]);
// Corrective action with logs
}
If the compilation is successful, you can load the compiled binary in a local variable.
size_t codeSize;
hiprtcGetCodeSize(prog, &codeSize);

```cpp

vector<char> kernel_binary(codeSize);
hiprtcGetCode(prog, kernel_binary.data());

```

After loading the binary,hiprtcProgram can be destroyed.hiprtcDestroyProgram(&prog);

```cpp

The binary present inkernel_binarycan now be loaded viahipModuleLoadData API.

```

hipModule_t module;

```cpp

hipFunction_t kernel;
hipModuleLoadData(&module, kernel_binary.data());
hipModuleGetFunction(&kernel, module, "vector_add");
And now this kernel can be launched viahipModule APIs.

## The full example is below

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <iostream>
#include <string>
#include <vector>

```

#define CHECK_RET_CODE(call, ret_code) ␣
(continues on next page)
60 Chapter 11. Programming for HIP Runtime Compiler (RTC)
HIP Documentation, Release 6.3.42134
(continued from previous page)
˓→ \
{ ␣
˓→ \
if ((call) != ret_code) { ␣
˓→ \
std::cout << "Failed in call: " << #call << std::endl; ␣
˓→ \
std::abort(); ␣
˓→ \
} ␣
˓→ \
}
#define HIP_CHECK(call) CHECK_RET_CODE(call, hipSuccess)
#define HIPRTC_CHECK(call) CHECK_RET_CODE(call, HIPRTC_SUCCESS)
// source code for hiprtc

```cpp

static constexpr auto kernel_source{

## R"(


```

extern "C"

```cpp

__global__ void vector_add(float* output, float* input1, float* input2, size_t size)

```

˓→{

```cpp

int i = threadIdx.x;

```

if (i < size) {
output[i] = input1[i] + input2[i];
}
}
)"};
int main() {
hiprtcProgram prog;
auto rtc_ret_code = hiprtcCreateProgram(&prog, // HIPRTC program handle

```cpp

kernel_source, // kernel source string

```

"vector_add.cpp", // Name of the file
0, // Number of headers
NULL, // Header sources
NULL); // Name of header file
if (rtc_ret_code != HIPRTC_SUCCESS) {
std::cout << "Failed to create program" << std::endl;
std::abort();
}
hipDeviceProp_t props;
int device = 0;
HIP_CHECK(hipGetDeviceProperties(&props, device));
std::string sarg = std::string("--gpu-architecture=") +
props.gcnArchName; // device for which binary is to be generated
const char * options[] = {sarg.c_str()};
rtc_ret_code = hiprtcCompileProgram(prog, // hiprtcProgram
0, // Number of options
(continues on next page)
11.1. Compilation APIs 61
HIP Documentation, Release 6.3.42134
(continued from previous page)
options); // Clang Options
if (rtc_ret_code != HIPRTC_SUCCESS) {
std::cout << "Failed to create program" << std::endl;
std::abort();
}
size_t logSize;
HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
if (logSize) {
std::string log(logSize, '\0');
HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
std::cout << "Compilation failed with: " << log << std::endl;
std::abort();
}
size_t codeSize;
HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

```cpp

std::vector<char> kernel_binary(codeSize);
HIPRTC_CHECK(hiprtcGetCode(prog, kernel_binary.data()));

```

HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
hipModule_t module;

```cpp

hipFunction_t kernel;
HIP_CHECK(hipModuleLoadData(&module, kernel_binary.data()));
HIP_CHECK(hipModuleGetFunction(&kernel, module, "vector_add"));

```

constexpr size_t ele_size = 256; // total number of items to add
std::vector<float> hinput, output;
hinput.reserve(ele_size);
output.reserve(ele_size);
for (size_t i = 0; i < ele_size; i++) {
hinput.push_back(static_cast<float>(i + 1));
output.push_back(0.0f);
}
float *dinput1, *dinput2, *doutput;

```cpp

HIP_CHECK(hipMalloc(&dinput1, sizeof(float) * ele_size));
HIP_CHECK(hipMalloc(&dinput2, sizeof(float) * ele_size));
HIP_CHECK(hipMalloc(&doutput, sizeof(float) * ele_size));
HIP_CHECK(hipMemcpy(dinput1, hinput.data(), sizeof(float) * ele_size,␣
˓→hipMemcpyHostToDevice));
HIP_CHECK(hipMemcpy(dinput2, hinput.data(), sizeof(float) * ele_size,␣
˓→hipMemcpyHostToDevice));

```

struct {
float* output;
float* input1;
(continues on next page)
62 Chapter 11. Programming for HIP Runtime Compiler (RTC)
HIP Documentation, Release 6.3.42134
(continued from previous page)
float* input2;
size_t size;
} args{doutput, dinput1, dinput2, ele_size};
auto size = sizeof(args);
void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
˓→ &size,

## HIP_LAUNCH_PARAM_END};


```cpp

HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, ele_size, 1, 1, 0, nullptr, nullptr,␣

```

˓→config));

```cpp

HIP_CHECK(hipMemcpy(output.data(), doutput, sizeof(float) * ele_size,␣
˓→hipMemcpyDeviceToHost));

```

for (size_t i = 0; i < ele_size; i++) {
if ((hinput[i] + hinput[i]) != output[i]) {
std::cout << "Failed in validation: " << (hinput[i] + hinput[i]) << " - " <<␣
˓→output[i]
<< std::endl;
std::abort();
}
}
std::cout << "Passed" << std::endl;
HIP_CHECK(hipFree(dinput1));
HIP_CHECK(hipFree(dinput2));
HIP_CHECK(hipFree(doutput));
}
11.2 HIPRTC specific options
HIPRTC provides a few HIPRTC specific flags
• --gpu-architecture : This flag can guide the code object generation for a specific gpu arch. Example:
--gpu-architecture=gfx906:sramecc+:xnack-, its equivalent to--offload-arch.
– This option is compulsory if compilation is done on a system without AMD GPUs supported by HIP run-
time.
– Otherwise, HIPRTC will load the hip runtime and gather the current device and its architecture info and
use it as option.
• -fgpu-rdc : This flag when provided during thehiprtcCompileProgram generates the bitcode (HIPRTC
doesn’t convert this bitcode into ISA and binary). This bitcode can later be fetched usinghiprtcGetBitcode
and hiprtcGetBitcodeSize APIs.
11.2. HIPRTC specific options 63
HIP Documentation, Release 6.3.42134
11.2.1 Bitcode

```cpp

Intheusualscenario,thekernelassociatedwith hiprtcProgramiscompiledintothebinarywhichcanbeloadedand

```

run. However, if-fpu-rdc option is provided in the compile options, HIPRTC calls comgr and generates only the
LLVM bitcode. It doesn’t convert this bitcode to ISA and generate the final binary.
std::string sarg = std::string("-fgpu-rdc");
const char * options[] = {
sarg.c_str() };
hiprtcCompileProgram(prog, // hiprtcProgram
1, // Number of options
options);
Ifthecompilationissuccessful,onecanloadthebitcodeinalocalvariableusingthebitcodeAPIsprovidedbyHIPRTC.
size_t bitCodeSize;
hiprtcGetBitcodeSize(prog, &bitCodeSize);

```cpp

vector<char> kernel_bitcode(bitCodeSize);
hiprtcGetBitcode(prog, kernel_bitcode.data());

```

11.2.2 CU Mode vs WGP mode
AMD GPUs consist of an array of workgroup processors, each built with 2 compute units (CUs) capable of executing
SIMD32. All the CUs inside a workgroup processor use local data share (LDS).
gfx10+ support execution of wavefront in CU mode and work-group processor mode (WGP). Please refer to section
2.3 of RDNA3 ISA reference.
gfx9 and below only supports CU mode.
In WGP mode, 4 warps of a block can simultaneously be executed on the workgroup processor, where as in CU mode
only 2 warps of a block can simultaneously execute on a CU. In theory, WGP mode might help with occupancy and
increasetheperformanceofcertainHIPprograms(ifnotboundtointerwarpcommunication),butmightincurperfor-
mance penalty on other HIP programs which rely on atomics and inter warp communication. This also has effect of
how the LDS is split between warps, please refer to RDNA3 ISA reference for more information.
HIPRTCassumes WGPmodebydefault forgfx10+. Thiscanbeoverriddenbypassing -mcumodetoHIPRTCcompile
options inhiprtcCompileProgram.
11.3 Linker APIs
The bitcode generated using the HIPRTC Bitcode APIs can be loaded usinghipModule APIs and also can be linked
with other generated bitcodes with appropriate linker flags using the HIPRTC linker APIs. This also provides more
flexibility and optimizations to the applications who want to generate the binary dynamically according to their needs.
The input bitcodes can be generated only for a specific architecture or it can be a bundled bitcode which is generated
for multiple architectures.
64 Chapter 11. Programming for HIP Runtime Compiler (RTC)
HIP Documentation, Release 6.3.42134
11.3.1 Example
Firstly, HIPRTC link instance or a pending linker invocation must be created usinghiprtcLinkCreate, with the
appropriate linker options provided.
hiprtcLinkCreate( num_options, // number of options
options, // Array of options
option_vals, // Array of option values cast to void*
&rtc_link_state ); // HIPRTC link state created upon success
Followingwhich,thebitcodedatacanbeaddedtothislinkinstancevia hiprtcLinkAddData(ifthedataispresentas
a string) orhiprtcLinkAddFile(if the data is present as a file) with the appropriate input type according to the data
or the bitcode used.
hiprtcLinkAddData(rtc_link_state, // HIPRTC link state
input_type, // type of the input data or bitcode
bit_code_ptr, // input data which is null terminated
bit_code_size, // size of the input data
"a", // optional name for this input
0, // size of the options
0, // Array of options applied to this input
0); // Array of option values cast to void*
hiprtcLinkAddFile(rtc_link_state, // HIPRTC link state
input_type, // type of the input data or bitcode
bc_file_path.c_str(), // path to the input file where bitcode is ␣
˓→present
0, // size of the options
0, // Array of options applied to this input
0); // Array of option values cast to void*
Once the bitcodes for multiple architectures are added to the link instance, the linking of the device code must be
completed usinghiprtcLinkCompletewhich generates the final binary.
hiprtcLinkComplete(rtc_link_state, // HIPRTC link state
&binary, // upon success, points to the output binary
&binarySize); // size of the binary is stored (optional)
If thehiprtcLinkCompletereturns successfully, the generated binary can be loaded and run using thehipModule*
APIs.
hipModuleLoadData(&module, binary);
11.3.1.1 Note
• The compiled binary must be loaded before HIPRTC link instance is destroyed using thehiprtcLinkDestroy

## API.

hiprtcLinkDestroy(rtc_link_state);
• The correct sequence of calls is : hiprtcLinkCreate, hiprtcLinkAddData or hiprtcLinkAddFile,
hiprtcLinkComplete,hiprtcModuleLoadData,hiprtcLinkDestroy.
11.3. Linker APIs 65
HIP Documentation, Release 6.3.42134
11.3.2 Input Types
HIPRTC provides hiprtcJITInputType enumeration type which defines the input types accepted
by the Linker APIs. Here are the enum values of hiprtcJITInputType. However only the in-
put types HIPRTC_JIT_INPUT_LLVM_BITCODE, HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE and
HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE are supported currently.
HIPRTC_JIT_INPUT_LLVM_BITCODEcan be used to load both LLVM bitcode or LLVM IR assembly code. However,
HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODEandHIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE
are only for bundled bitcode and archive of bundled bitcode.

## HIPRTC_JIT_INPUT_CUBIN = 0,


## HIPRTC_JIT_INPUT_PTX,


## HIPRTC_JIT_INPUT_FATBINARY,


## HIPRTC_JIT_INPUT_OBJECT,


## HIPRTC_JIT_INPUT_LIBRARY,


## HIPRTC_JIT_INPUT_NVVM,


## HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES,


## HIPRTC_JIT_INPUT_LLVM_BITCODE = 100,


## HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE = 101,


## HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE = 102,


## HIPRTC_JIT_NUM_INPUT_TYPES = (HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES + 3)

11.3.3 Backward Compatibility of LLVM Bitcode/IR
ForHIPapplicationsutilizingHIPRTCtocompileLLVMbitcode/IR,compatibilityisassuredonlywhentheROCmor
HIP SDK version used for generating the LLVM bitcode/IR matches the version used during the runtime compilation.
When an application requires the ingestion of bitcode/IR not derived from the currently installed AMD compiler, it
must run with HIPRTC and comgr dynamic libraries that are compatible with the version of the bitcode/IR.
comgr, a shared library, incorporates the LLVM/Clang compiler that HIPRTC relies on. To identify the bitcode/IR
versionthatcomgriscompatiblewith,onecanexecute“clang-v”usingtheclangbinaryfromthesameROCmorHIP
SDK package. For instance, if compiling bitcode/IR version 14, the HIPRTC and comgr libraries released by AMD
around mid 2022 would be the best choice, assuming the LLVM/Clang version included in the package is also version
14.
Toensuresmoothoperationandcompatibility, anapplicationmaychoosetoshipthespecificversionsofHIPRTCand
comgr dynamic libraries, or it may opt to clearly specify the version requirements and dependencies. This approach
guarantees that the application can correctly compile the specified version of bitcode/IR.
11.3.4 Link Options
• HIPRTC_JIT_IR_TO_ISA_OPT_EXT - AMD Only. Options to be passed on to link step of compiler by
hiprtcLinkCreate.
• HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT- AMD Only. Count of options passed on to link step of compiler.

## Example

const char * isaopts[] = {"-mllvm", "-inline-threshold=1", "-mllvm", "-inlinehint-
˓→threshold=1"};
std::vector<hiprtcJIT_option> jit_options = {HIPRTC_JIT_IR_TO_ISA_OPT_EXT,

## HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};

size_t isaoptssize = 4;
(continues on next page)
66 Chapter 11. Programming for HIP Runtime Compiler (RTC)
HIP Documentation, Release 6.3.42134
(continued from previous page)
const void * lopts[] = {(void*)isaopts, (void*)(isaoptssize)};
hiprtcLinkState linkstate;
hiprtcLinkCreate(2, jit_options.data(), (void**)lopts, &linkstate);
11.4 Error Handling
HIPRTC defines thehiprtcResult enumeration type and a functionhiprtcGetErrorString for API call error
handling. hiprtcResult enum defines the API result codes. HIPRTC APIs returnhiprtcResult to indicate
the call result. hiprtcGetErrorString function returns a string describing the givenhiprtcResult code, e.g.,
HIPRTC_SUCCESS to “HIPRTC_SUCCESS”. For unrecognized enumeration values, it returns “Invalid HIPRTC
error code”.
hiprtcResult enumsupported values and thehiprtcGetErrorString usage are mentioned below.

## HIPRTC_SUCCESS = 0,


## HIPRTC_ERROR_OUT_OF_MEMORY = 1,


## HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,


## HIPRTC_ERROR_INVALID_INPUT = 3,


## HIPRTC_ERROR_INVALID_PROGRAM = 4,


## HIPRTC_ERROR_INVALID_OPTION = 5,


## HIPRTC_ERROR_COMPILATION = 6,


## HIPRTC_ERROR_LINKING = 7,


## HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 8,


## HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 9,


## HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 10,


## HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 11,


## HIPRTC_ERROR_INTERNAL_ERROR = 12

hiprtcResult result;
result = hiprtcCompileProgram(prog, 1, opts);
if (result != HIPRTC_SUCCESS) {
std::cout << "hiprtcCompileProgram fails with error " << hiprtcGetErrorString(result);
}
11.5 HIPRTC General APIs
HIPRTC provides the following API for querying the version.
hiprtcVersion(int* major, int* minor)-ThissetstheoutputparametersmajorandminorwiththeHIPRun-
time compilation major version and minor version number respectively.
Currently, it returns hardcoded value. This should be implemented to return HIP runtime major and minor version in
the future releases.
11.4. Error Handling 67
HIP Documentation, Release 6.3.42134
11.6 Lowered Names (Mangled Names)

```cpp

HIPRTC mangles the__global__ function names and names of__device__ and __constant__ variables. If the
generated binary is being loaded using the HIP Runtime API, the kernel function or__device__/__constant__

```

variable must be looked up by name, but this is very hard when the name has been mangled. To overcome this,

```cpp

HIPRTC provides API functions that map__global__ function or__device__/__constant__ variable names in

```

the source to the mangled names present in the generated binary.
The two APIshiprtcAddNameExpression and hiprtcGetLoweredName provide this functionality. First, a ‘name

```cpp

expression’stringdenotingtheaddressforthe __global__functionor __device__/__constant__variableispro-

```

vided tohiprtcAddNameExpression. Then, the program is compiled withhiprtcCompileProgram. During com-
pilation, HIPRTC will parse the name expression string as a C++ constant expression at the end of the user program.
Finally, the functionhiprtcGetLoweredName is called with the original name expression and it returns a pointer to

```cpp

the lowered name. The lowered name can be used to refer to the kernel or variable in the HIP Runtime API.

```

11.6.1 Note
• The identical name expression string must be provided on a subsequent call tohiprtcGetLoweredName to
extract the lowered name.
• The correct sequence of calls is : hiprtcAddNameExpression, hiprtcCompileProgram,
hiprtcGetLoweredName,hiprtcDestroyProgram.
• The lowered names must be fetched usinghiprtcGetLoweredName only after the HIPRTC program has been
compiled, and before it has been destroyed.
11.6.2 Example

```cpp

kernel containing various definitions__global__ functions/function templates and__device__/__constant__

```

variables can be stored in a string.
static constexpr const char gpu_program[] {

## R"(

__device__ int V1; // set from host code

```cpp

static __global__ void f1(int *result) { *result = V1 + 10; }

```

namespace N1 {
namespace N2 {
__constant__ int V2; // set from host code

```cpp

__global__ void f2(int *result) { *result = V2 + 20; }

```

}
}
template<typename T>

```cpp

__global__ void f3(int *result) { *result = sizeof(T); }

```

)"};

```cpp

hiprtcAddNameExpression is called with various name expressions referring to the address of__global__ func-

```

tions and__device__/__constant__variables.

```cpp

kernel_name_vec.push_back("&f1");
kernel_name_vec.push_back("N1::N2::f2");
kernel_name_vec.push_back("f3<int>");
for (auto&& x : kernel_name_vec) hiprtcAddNameExpression(prog, x.c_str());

```

variable_name_vec.push_back("&V1");
(continues on next page)
68 Chapter 11. Programming for HIP Runtime Compiler (RTC)
HIP Documentation, Release 6.3.42134
(continued from previous page)
variable_name_vec.push_back("&N1::N2::V2");
for (auto&& x : variable_name_vec) hiprtcAddNameExpression(prog, x.c_str());
After which, the program is compiled usinghiprtcCompileProgram and the generated binary is loaded using
hipModuleLoadData. And the mangled names can be fetched usinghirtcGetLoweredName.
for (decltype(variable_name_vec.size()) i = 0; i != variable_name_vec.size(); ++i) {
const char * name;
hiprtcGetLoweredName(prog, variable_name_vec[i].c_str(), &name);
}

```cpp

for (decltype(kernel_name_vec.size()) i = 0; i != kernel_name_vec.size(); ++i) {

```

const char * name;

```cpp

hiprtcGetLoweredName(prog, kernel_name_vec[i].c_str(), &name);

```

}
The mangled name of the variables are used to look up the variable in the module and update its value.
hipDeviceptr_t variable_addr;
size_t bytes{};
hipModuleGetGlobal(&variable_addr, &bytes, module, name);

```cpp

hipMemcpyHtoD(variable_addr, &initial_value, sizeof(initial_value));
Finally, the mangled name of the kernel is used to launch it using thehipModule APIs.
hipFunction_t kernel;
hipModuleGetFunction(&kernel, module, name);
hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, config);

```

Please have a look athiprtcGetLoweredName.cppfor the detailed example.
11.7 Versioning
HIPRTC follows the below versioning.
• Linux
– HIPRTC follows the same versioning as HIP runtime library.
– The so name field for the shared library is set to MAJOR version. For example, for HIP 5.3 theso name
is set to 5 (hiprtc.so.5).
• Windows
– HIPRTC dll is named ashiprtcXXYY.dllwhere XX is MAJOR version and YY is MINOR version. For
example, for HIP 5.3 the name ishiprtc0503.dll.
11.7. Versioning 69
HIP Documentation, Release 6.3.42134
11.8 HIP header support
• Added HIPRTC support for all the hip common header files such as library_types.h, hip_math_constants.h,
hip_complex.h, math_functions.h, surface_types.h etc. from 6.1. HIPRTC users need not include any HIP
macros or constants explicitly in their header files. All of these should get included via HIPRTC builtins when
the app links to HIPRTC library.
11.9 Deprecation notice
• CurrentlyHIPRTCAPIsareseparatedfromHIPAPIsandHIPRTCisavailableasaseparatelibrary libhiprtc.
so/libhiprtc.dll. But on Linux, HIPRTC symbols are also present inlibamdhip64.soin order to support
the existing applications. Gradually, these symbols will be removed from HIP library and applications using
HIPRTCwillberequiredtoexplicitlylinktoHIPRTClibrary. However,onWindows hiprtc.dllmustbeused
as theamdhip64.dlldoesn’t contain the HIPRTC symbols.
• Data types such asuint32_t, uint64_t, int32_t, int64_t defined in std namespace in HIPRTC are depre-
cated earlier and are being removed from ROCm release 6.1 since these can conflict with the standard C++ data
types. These data types are now prefixed with__hip__, e.g.__hip_uint32_t. Applications previously using
std::uint32_tor similar types can use__hip_prefixed types to avoid conflicts with standard std namespace
or application can have their own definitions for these types. Also, type_traits templates previously defined in
std namespace are moved to__hip_internal namespace as implementation details.
70 Chapter 11. Programming for HIP Runtime Compiler (RTC)

## CHAPTER


## TWELVE


## PERFORMANCE GUIDELINES

TheAMDHIPPerformanceGuidelinesareasetofbestpracticesdesignedtohelpdevelopersoptimizetheperformance
ofAMDGPUs. Theycoverestablishedparallelizationandoptimizationtechniques,codingmetaphors,andidiomsthat
can greatly simplify programming for HIP-capable GPU architectures.
By following four main cornerstones, we can exploit the performance optimization potential of HIP.
• parallel execution
• memory usage optimization
• optimization for maximum throughput
• minimizing memory thrashing
In the following chapters, we will show you their benefits and how to use them effectively.
12.1 Parallel execution
For optimal use, the application should reveal and efficiently imply as much parallelism as possible to keep all system
components active.
12.1.1 Application level
The application should optimize parallel execution across the host and devices using asynchronous calls and streams.
Workloads should be assigned based on efficiency: serial to the host, parallel to the devices.
For parallel workloads, when threads need to synchronize to share data, if they belong to the same block, they should

```cpp

use__syncthreads()(see: Synchronizationfunctions)withinthesamekernelinvocation. Iftheybelongtodifferent
blocks, they must use global memory with two separate kernel invocations. The latter should be minimized as it adds

```

overhead.
12.1.2 Device level
Device-level optimization primarily involves maximizing parallel execution across the multiprocessors of the device.

```cpp

This can be achieved by executing multiple kernels concurrently on a device. The management of these kernels is

```

facilitatedbystreams,whichallowfortheoverlappingofcomputationanddatatransfers,enhancingperformance. The

```cpp

aimistokeepallmultiprocessorsbusybyexecutingenoughkernelsconcurrently. However,launchingtoomanykernels

```

canleadtoresourcecontention,soabalancemustbefoundforoptimalperformance. Thisapproachhelpsinachieving
maximum utilization of the resources of the device.
71
HIP Documentation, Release 6.3.42134
12.1.3 Multiprocessor level
Multiprocessor-leveloptimizationinvolvesmaximizingparallelexecutionwithineachmultiprocessoronadevice. Each
multiprocessor can execute a number of threads concurrently, and the total number of threads that can run in parallel
is determined by the number of concurrent threads each multiprocessor can handle.
Thekeytomultiprocessor-leveloptimizationistoefficientlyutilizethevariousfunctionalunitswithinamultiprocessor.
This can be achieved by ensuring a sufficient number of resident warps, as at every instruction issue time, a warp
scheduler selects an instruction that is ready to execute. This instruction can be another independent instruction of the
samewarp,exploiting Optimizationformaximuminstructionthroughput ,ormorecommonlyaninstructionofanother
warp, exploiting thread-level parallelism.
Incomparison,device-leveloptimizationfocusesonthedeviceasawhole, aimingtokeepallmultiprocessorsbusyby

```cpp

executing enough kernels concurrently. Both levels of optimization are crucial for achieving maximum performance.

```

They work together to ensure efficient utilization of the resources of the GPU, from the individual multiprocessors to
the device as a whole.
12.2 Memory optimization
The first step in maximizing memory throughput is to minimize low-bandwidth data transfers. This involves reducing
data transfers between the host and the device, as these have lower bandwidth than transfers between global memory
and the device.
Additionally, data transfers between global memory and the device should be minimized by maximizing the use of
on-chip memory: shared memory and caches. Shared memory acts as a user-managed cache, where the application
explicitly allocates and accesses it. A common programming pattern is to stage data from device memory into shared
memory. Thisinvolveseachthreadofablockloadingdatafromdevicememorytosharedmemory,synchronizingwith
all other threads of the block, processing the data in shared memory, synchronizing again if necessary, and writing the
results back to device global memory.
For some applications, a traditional hardware-managed cache is more appropriate to exploit data locality. On devices
of certain compute capabilities, the same on-chip memory is used for both L1 and shared memory, and the amount

```cpp

dedicated to each is configurable for each kernel call.
Finally, the throughput of memory accesses by a kernel can vary significantly depending on the access pattern for

```

each type of memory. Therefore, the next step in maximizing memory throughput is to organize memory accesses as
optimally as possible. This is especially important for global memory accesses, as global memory bandwidth is low
compared to available on-chip bandwidths and arithmetic instruction throughput. Thus, non-optimal global memory
accesses generally have a high impact on performance.
12.2.1 Data Transfer
Applications should aim to minimize data transfers between the host and the device. This can be achieved by moving

```cpp

morecomputationsfromthehosttothedevice,evenifitmeansrunningkernelsthatdonotfullyutilizetheparallelism

```

for device. Intermediate data structures can be created, used, and discarded in device memory without being mapped
or copied to host memory.
Batchingsmalltransfersintoasinglelargetransfercanimproveperformanceduetotheoverheadassociatedwitheach
transfer. On systems with a front-side bus, using page-locked host memory can enhance data transfer performance.
When using mapped page-locked memory, there is no need to allocate device memory or explicitly copy data between

```cpp

deviceandhostmemory. Datatransfersoccurimplicitlyeachtimethekernelaccessesthemappedmemory. Foroptimal

```

performance, these memory accesses should be coalesced, similar to global memory accesses.
72 Chapter 12. Performance Guidelines
HIP Documentation, Release 6.3.42134
On integrated systems where device and host memory are physically the same, any copy operation between host and
device memory is unnecessary, and mapped page-locked memory should be used instead. Applications can check if a
device is integrated by querying the integrated device property.
12.2.2 Device Memory Access
Memory access instructions may be repeated due to the spread of memory addresses across warp threads. The impact
on throughput varies with memory type and is generally reduced when addresses are more scattered, especially in
global memory.
Devicememoryisaccessedvia32-,64-,or128-bytetransactionsthatmustbenaturallyaligned. Maximizingmemory
throughputinvolvescoalescingmemoryaccessesofthreadswithinawarpintominimaltransactions,followingoptimal
access patterns, using properly sized and aligned data types, and padding data when necessary.
Global memory instructions support reading or writing data of specific sizes (1, 2, 4, 8, or 16 bytes) that are naturally
aligned. If the size and alignment requirements are not met, it leads to multiple instructions, reducing performance.
Therefore,usingdatatypesthatmeettheserequirements,ensuringalignmentforstructures,andmaintainingalignment
for all values or arrays is crucial for correct results and optimal performance.
Threads often access 2D arrays at an address calculated asBaseAddress + xIndex + width * yIndex. For effi-
cient memory access, the array and thread block widths should be multiples of the warp size. If the array width is not
amultipleofthewarpsize,itisusuallymoreefficienttoallocateitwithawidthroundeduptothenearestmultipleand
pad the rows accordingly.
Local memory is used for certain automatic variables, such as arrays with non-constant indices, large structures or

```cpp

arrays, and any variable when the kernel uses more registers than available. Local memory resides in device memory,

```

leadingtohighlatencyandlowbandwidthsimilartoglobalmemoryaccesses. However,itisorganizedforconsecutive
32-bit words to be accessed by consecutive thread IDs, allowing full coalescing when all threads in a warp access the
same relative address.
Shared memory, located on-chip, provides higher bandwidth and lower latency than local or global memory. It is
divided into banks that can be simultaneously accessed, boosting bandwidth. However, bank conflicts, where two
addresses fall in the same bank, lead to serialized access and decreased throughput. Therefore, understanding how
memory addresses map to banks and scheduling requests to minimize conflicts is crucial for optimal performance.
Constantmemoryisindevicememoryandcachedintheconstantcache. Requestsaresplitbasedondifferentmemory
addresses,affectingthroughput,andareservicedatthethroughputoftheconstantcacheforcachehits,orthethroughput
of the device memory otherwise.
Textureandsurfacememoryarestoredindevicememoryandcachedintexturecache. Thissetupoptimizes2Dspatial
locality,leadingtobetterperformanceforthreadsreadingclose2Daddresses. Readingdevicememorythroughtexture
orsurfacefetchingcanbeadvantageous,offeringhigherbandwidthforlocaltexturefetchesorsurfacereads,offloading
addressing calculations, allowing data broadcasting, and optional conversion of 8-bit and 16-bit integer input data to
32-bit floating-point values on-the-fly.
12.3 Optimization for maximum instruction throughput

## To maximize instruction throughput

• minimize low throughput arithmetic instructions
• minimize divergent warps inflicted by control flow instructions
• minimize the number of instruction as possible
• maximize instruction parallelism
12.3. Optimization for maximum instruction throughput 73
HIP Documentation, Release 6.3.42134
12.3.1 Arithmetic instructions
Thetypeandcomplexityofarithmeticoperationscansignificantlyimpacttheperformanceofyourapplication. Weare
highlighting some hints how to maximize it.
Using efficient operations: Some arithmetic operations are more costly than others. For example, multiplication is
typically faster than division, and integer operations are usually faster than floating-point operations, especially with
double-precision.
Minimizing low-throughput instructions: This might involve trading precision for speed when it does not affect the
final result. For instance, consider using single-precision arithmetic instead of double-precision.
Leverage intrinsic functions: Intrinsic functions are pre-defined functions available in HIP that can often be executed
faster than equivalent arithmetic operations (subject to some input or accuracy restrictions). They can help optimize
performance by replacing more complex arithmetic operations.
Avoidingdivergentwarps: Divergentwarpsoccurwhenthreadswithinthesamewarpfollowdifferentexecutionpaths.
Thiscanhappenduetoconditionalstatementsthatleadtodifferentarithmeticoperationsbeingperformedbydifferent
threads. Divergent warps can significantly reduce instruction throughput, so try to structure your code to minimize
divergence.
Optimizingmemoryaccess: Theefficiencyofmemoryaccesscanimpactthespeedofarithmeticoperations. Coalesced
memory access, where threads in a warp access consecutive memory locations, can improve memory throughput and
thus the speed of arithmetic operations.
Maximizing instruction parallelism: Some GPU architectures could issue parallel independent instructions simultane-
ously, for example integer and floating point, or two operations with independent inputs and outputs. Mostly this is a
work for compiler, but expressing parallelism in the code explicitly can improve instructions throughput.
12.3.2 Control flow instructions
Flowcontrolinstructions( if,else,for,do,while,break,continue,switch)canimpactinstructionthroughputby
causingthreadswithinawarptodivergeandfollowdifferentexecutionpaths. Tooptimizeperformance,controlcondi-

```cpp

tionsshouldbewrittentominimizedivergentwarps. Forexample,whenthecontrolconditiondependson( threadIdx

```

/warpSize),nowarpdiverges. Thecompilermayoptimizeloopsorshortiforswitchblocksusingbranchpredication,
preventing warp divergence. With branch predication, instructions associated with a false predicate are scheduled but
not executed, avoiding unnecessary operations.
12.3.3 Synchronization
Synchronizationensuresthatallthreadswithinablockhavecompletedtheircomputationsandmemoryaccessesbefore
movingforward,whichiscriticalwhenthreadsaredependentontheresultsofotherthreads. However,synchronization
can also lead to performance overhead, as it requires threads to wait, potentially leading to idle GPU resources.
__syncthreads()is used to synchronize all threads in a block, ensuring that all threads have reached the same point
in the code and that shared memory is visible to all threads after the point of synchronization.
Analternativewaytosynchronizeisusingstreams. Differentstreamscanexecutecommandsoutoforderwithrespect
tooneanotherorconcurrently. Thisallowsformorefine-grainedcontrolovertheexecutionorderofcommands,which
can be beneficial in certain scenarios.
74 Chapter 12. Performance Guidelines
HIP Documentation, Release 6.3.42134
12.4 Minimizing memory thrashing
Applications frequently allocating and freeing memory may experience slower allocation calls over time. This is ex-
pectedasmemoryisreleasedbacktotheoperatingsystem. Tooptimizeperformanceinsuchscenarios,considersome

## recommendations


```cpp

• avoidallocatingallavailablememorywith hipMalloc/hipHostMalloc,asthisimmediatelyreservesmemory

```

andcanblockotherapplicationsfromusingit. Thiscouldstraintheoperatingsystemschedulersorevenprevent
other applications from running on the same GPU.
• aimtoallocatememoryinsuitablysizedblocksearlyinthelifecycleoftheapplicationanddeallocateonlywhen

```cpp

the application no longer needs it. Minimize the number ofhipMalloc and hipFree calls in your application,

```

particularly in areas critical to performance.
• if an application is unable to allocate sufficient device memory, consider resorting to other memory types such

```cpp

as hipHostMalloc or hipMallocManaged. While these may not offer the same performance, they can allow

```

the application to continue running.

```cpp

• For supported platforms,hipMallocManaged allows for oversubscription. With the right memory advise poli-
cies, it can maintain most, if not all, of the performance ofhipMalloc. hipMallocManaged does not require

```

an allocation to be resident until it is needed or prefetched, easing the load on the operating system schedulers
and facilitating multi-tenant scenarios.
12.4. Minimizing memory thrashing 75
HIP Documentation, Release 6.3.42134
76 Chapter 12. Performance Guidelines

## CHAPTER


## THIRTEEN


## DEBUGGING WITH HIP

AMDdebuggingtoolsinclude ltraceandROCgdb. Externaltoolsareavailableandcanbefoundonline. Forexample,
if you’re using Windows, you can useMicrosoft Visual Studioand WinGDB.
You can trace and debug your code using the following tools and techniques.
13.1 Tracing
You can use tracing to quickly observe the flow of an application before reviewing the detailed information provided
by a command-line debugger. Tracing can be used to identify issues ranging from accidental API calls to calls made
on a critical path.
ltrace is a standard Linux tool that provides a message tostderr on every dynamic library call. You can use ltrace to
visualize the runtime behavior of the entire ROCm software stack.
Here’s a simple command-line example that uses ltrace to trace HIP APIs and output:
$ ltrace -C -e "hip*" ./hipGetChanDesc
hipGetChanDesc->hipCreateChannelDesc(0x7ffdc4b66860, 32, 0, 0) = 0x7ffdc4b66860

```cpp

hipGetChanDesc->hipMallocArray(0x7ffdc4b66840, 0x7ffdc4b66860, 8, 8) = 0

```

hipGetChanDesc->hipGetChannelDesc(0x7ffdc4b66848, 0xa63990, 5, 1) = 0
hipGetChanDesc->hipFreeArray(0xa63990, 0, 0x7f8c7fe13778, 0x7ffdc4b66848) = 0

## PASSED!

+++ exited (status 0) +++

## Here’s another example that uses ltrace to trace hsa APIs and output

$ ltrace -C -e "hsa*" ./hipGetChanDesc
libamdhip64.so.4->hsa_init(0, 0x7fff325a69d0, 0x9c80e0, 0 <unfinished ...>
libhsa-runtime64.so.1->hsaKmtOpenKFD(0x7fff325a6590, 0x9c38c0, 0, 1) = 0
libhsa-runtime64.so.1->hsaKmtGetVersion(0x7fff325a6608, 0, 0, 0) = 0
libhsa-runtime64.so.1->hsaKmtReleaseSystemProperties(3, 0x80084b01, 0, 0) = 0
libhsa-runtime64.so.1->hsaKmtAcquireSystemProperties(0x7fff325a6610, 0, 0, 1) = 0
libhsa-runtime64.so.1->hsaKmtGetNodeProperties(0, 0x7fff325a66a0, 0, 0) = 0
libhsa-runtime64.so.1->hsaKmtGetNodeMemoryProperties(0, 1, 0x9c42b0, 0x936012) = 0
...
<... hsaKmtCreateEvent resumed> ) = 0
libhsa-runtime64.so.1->hsaKmtAllocMemory(0, 4096, 64, 0x7fff325a6690) = 0
libhsa-runtime64.so.1->hsaKmtMapMemoryToGPUNodes(0x7f1202749000, 4096, 0x7fff325a6690,␣
˓→0) = 0
libhsa-runtime64.so.1->hsaKmtCreateEvent(0x7fff325a6700, 0, 0, 0x7fff325a66f0) = 0
(continues on next page)
77
HIP Documentation, Release 6.3.42134
(continued from previous page)
libhsa-runtime64.so.1->hsaKmtAllocMemory(1, 0x100000000, 576, 0x7fff325a67d8) = 0
libhsa-runtime64.so.1->hsaKmtAllocMemory(0, 8192, 64, 0x7fff325a6790) = 0
libhsa-runtime64.so.1->hsaKmtMapMemoryToGPUNodes(0x7f120273c000, 8192, 0x7fff325a6790,␣
˓→0) = 0
libhsa-runtime64.so.1->hsaKmtAllocMemory(0, 4096, 4160, 0x7fff325a6450) = 0
libhsa-runtime64.so.1->hsaKmtMapMemoryToGPUNodes(0x7f120273a000, 4096, 0x7fff325a6450,␣
˓→0) = 0
libhsa-runtime64.so.1->hsaKmtSetTrapHandler(1, 0x7f120273a000, 4096, 0x7f120273c000) = 0
<... hsa_init resumed> ) = 0
libamdhip64.so.4->hsa_system_get_major_extension_table(513, 1, 24, 0x7f1202597930) = 0
libamdhip64.so.4->hsa_iterate_agents(0x7f120171f050, 0, 0x7fff325a67f8, 0 <unfinished ...
˓→>
libamdhip64.so.4->hsa_agent_get_info(0x94f110, 17, 0x7fff325a67e8, 0) = 0
libamdhip64.so.4->hsa_amd_agent_iterate_memory_pools(0x94f110, 0x7f1201722816,␣
˓→0x7fff325a67f0, 0x7f1201722816 <unfinished ...>
libamdhip64.so.4->hsa_amd_memory_pool_get_info(0x9c7fb0, 0, 0x7fff325a6744,␣
˓→0x7fff325a67f0) = 0
libamdhip64.so.4->hsa_amd_memory_pool_get_info(0x9c7fb0, 1, 0x7fff325a6748,␣
˓→0x7f1200d82df4) = 0
...
<... hsa_amd_agent_iterate_memory_pools resumed> ) = 0
libamdhip64.so.4->hsa_agent_get_info(0x9dbf30, 17, 0x7fff325a67e8, 0) = 0
<... hsa_iterate_agents resumed> ) = 0
libamdhip64.so.4->hsa_agent_get_info(0x9dbf30, 0, 0x7fff325a6850, 3) = 0
libamdhip64.so.4->hsa_agent_get_info(0x9dbf30, 0xa000, 0x9e7cd8, 0) = 0
libamdhip64.so.4->hsa_agent_iterate_isas(0x9dbf30, 0x7f1201720411, 0x7fff325a6760,␣
˓→0x7f1201720411) = 0
libamdhip64.so.4->hsa_isa_get_info_alt(0x94e7c8, 0, 0x7fff325a6728, 1) = 0
libamdhip64.so.4->hsa_isa_get_info_alt(0x94e7c8, 1, 0x9e7f90, 0) = 0
libamdhip64.so.4->hsa_agent_get_info(0x9dbf30, 4, 0x9e7ce8, 0) = 0
...
<... hsa_amd_memory_pool_allocate resumed> ) = 0
libamdhip64.so.4->hsa_ext_image_create(0x9dbf30, 0xa1c4c8, 0x7f10f2800000, 3 <unfinished␣
˓→...>
libhsa-runtime64.so.1->hsaKmtAllocMemory(0, 4096, 64, 0x7fff325a6740) = 0
libhsa-runtime64.so.1->hsaKmtQueryPointerInfo(0x7f1202736000, 0x7fff325a65e0, 0, 0) = 0
libhsa-runtime64.so.1->hsaKmtMapMemoryToGPUNodes(0x7f1202736000, 4096, 0x7fff325a66e8,␣
˓→0) = 0
<... hsa_ext_image_create resumed> ) = 0
libamdhip64.so.4->hsa_ext_image_destroy(0x9dbf30, 0x7f1202736000, 0x9dbf30, 0
˓→<unfinished ...>
libhsa-runtime64.so.1->hsaKmtUnmapMemoryToGPU(0x7f1202736000, 0x7f1202736000, 4096,␣
˓→0x9c8050) = 0
libhsa-runtime64.so.1->hsaKmtFreeMemory(0x7f1202736000, 4096, 0, 0) = 0
<... hsa_ext_image_destroy resumed> ) = 0
libamdhip64.so.4->hsa_amd_memory_pool_free(0x7f10f2800000, 0x7f10f2800000, 256,␣
˓→0x9e76f0) = 0

## PASSED!

78 Chapter 13. Debugging with HIP
HIP Documentation, Release 6.3.42134
13.2 Debugging
You can use ROCgdb for debugging and profiling.
ROCgdbistheROCmsource-leveldebuggerforLinuxandisbasedonGNUProjectdebugger(GDB).theGNUsource-
leveldebugger,equivalentofCUDA-GDB,canbeusedwithdebuggerfrontends,suchasEclipse,VisualStudioCode,
or GDB dashboard. For details, see (https://github.com/ROCm/ROCgdb).
Below is a sample how to use ROCgdb run and debug HIP application, ROCgdb is installed with ROCM package in
the folder /opt/rocm/bin.
$ export PATH=$PATH:/opt/rocm/bin
$ rocgdb ./hipTexObjPitch
GNU gdb (rocm-dkms-no-npi-hipclang-6549) 10.1
Copyright (C) 2020 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
...

## For bug reporting instructions, please see

<https://github.com/ROCm/ROCgdb/issues>.

## Find the GDB manual and other documentation resources online at

<http://www.gnu.org/software/gdb/documentation/>.
...
Reading symbols from ./hipTexObjPitch...
(gdb) break main
Breakpoint 1 at 0x4013d1: file /home/test/hip/tests/src/texture/hipTexObjPitch.cpp, line␣
˓→98.
(gdb) run
Starting program: /home/test/hip/build/directed_tests/texture/hipTexObjPitch
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
Breakpoint 1, main ()
at /home/test/hip/tests/src/texture/hipTexObjPitch.cpp:98
98 texture2Dtest<float>();
(gdb)c
13.2.1 Debugging HIP applications
ThefollowingLinuxexampleshowshowtogetusefulinformationfromthedebuggerwhilerunningasimplememory
copy test, which caused a segmentation fault issue.
test: simpleTest2<?> numElements=4194304 sizeElements=4194304 bytes
Segmentation fault (core dumped)
(gdb) run

```cpp

Starting program: /home/test/hipamd/build/directed_tests/runtimeApi/memory/hipMemcpy_

```

˓→simple
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
Breakpoint 1, main (argc=1, argv=0x7fffffffdea8)

```cpp

at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:147

```

(continues on next page)
13.2. Debugging 79
HIP Documentation, Release 6.3.42134
(continued from previous page)
147 int main(int argc, char* argv[]) {
(gdb) c
Continuing.
[New Thread 0x7ffff64c4700 (LWP 146066)]

```cpp

Thread 1 "hipMemcpy_simpl" received signal SIGSEGV, Segmentation fault.

```

0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)

```cpp

at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104

```

104 A_h1[i] = 3.14f + 1000 * i;
(gdb) bt
#0 0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)

```cpp

at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104

```

#1 0x000000000020e96c in main (argc=<optimized out>, argv=<optimized out>)

```cpp

at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:163

```

(gdb) info thread
Id Target Id Frame

```cpp

* 1 Thread 0x7ffff64c5880 (LWP 146060) "hipMemcpy_simpl" 0x000000000020f78e in␣

```

˓→simpleTest2<float> (numElements=4194304, usePinnedHost=true)

```cpp

at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104
2 Thread 0x7ffff64c4700 (LWP 146066) "hipMemcpy_simpl" 0x00007ffff6b0850b in ioctl

```

() from /lib/x86_64-linux-gnu/libc.so.6
(gdb) thread 2
[Switching to thread 2 (Thread 0x7ffff64c4700 (LWP 146066))]
#0 0x00007ffff6b0850b in ioctl () from /lib/x86_64-linux-gnu/libc.so.6
(gdb) bt
#0 0x00007ffff6b0850b in ioctl () from /lib/x86_64-linux-gnu/libc.so.6
#1 0x00007ffff6604568 in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
#2 0x00007ffff65fe73a in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
#3 0x00007ffff659e4d6 in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
#4 0x00007ffff65807de in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
#5 0x00007ffff65932a2 in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
#6 0x00007ffff654f547 in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
#7 0x00007ffff7f76609 in start_thread () from /lib/x86_64-linux-gnu/libpthread.so.0
#8 0x00007ffff6b13293 in clone () from /lib/x86_64-linux-gnu/libc.so.6
(gdb) thread 1
[Switching to thread 1 (Thread 0x7ffff64c5880 (LWP 146060))]
#0 0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)

```cpp

at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104

```

104 A_h1[i] = 3.14f + 1000 * i;
(gdb) bt
#0 0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)

```cpp

at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104

```

#1 0x000000000020e96c in main (argc=<optimized out>, argv=<optimized out>)

```cpp

at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:163

```

(gdb)
...
Debugging HIP applications using Windows tools can be more informative than on Linux. Windows tools provides
more visibility into debug codes, which makes it easier to inspect variables, watch multiple details, and examine call
stacks.
80 Chapter 13. Debugging with HIP
HIP Documentation, Release 6.3.42134
13.3 Useful environment variables
HIPprovidesenvironmentvariablesthatallowHIP,hip-clang,orHSAdriverstopreventcertainfeaturesandoptimiza-
tions. Thesearenotintendedforproduction,butcanbeusefultodiagnosesynchronizationproblemsintheapplication
(or driver).
Some of the more widely used environment variables are described in this section. These are supported on the Linux
ROCm path and Windows.
13.3.1 Kernel enqueue serialization

## You can control kernel command serialization from the host


```cpp

AMD_SERIALIZE_KERNEL, for serializing kernel enqueue

```

AMD_SERIALIZE_KERNEL = 1, Wait for completion before enqueueAMD_SERIALIZE_KERNEL = 2, Wait for
completion after enqueueAMD_SERIALIZE_KERNEL = 3, Both
Or
AMD_SERIALIZE_COPY, for serializing copies
AMD_SERIALIZE_COPY = 1, Wait for completion before enqueueAMD_SERIALIZE_COPY = 2, Wait for com-
pletion after enqueueAMD_SERIALIZE_COPY = 3, Both
So HIP runtime can wait for GPU idle before/after any GPU command depending on the environment setting.
13.3.2 Making device visible
For systems with multiple devices, you can choose to make only certain device(s) visible to HIP using
HIP_VISIBLE_DEVICES (or CUDA_VISIBLE_DEVICES on an NVIDIA platform). Once enabled, HIP can only view

## devices that have indices present in the sequence. For example


## $ HIP_VISIBLE_DEVICES=0,1


## Or in the application

if (totalDeviceNum > 2) {
setenv("HIP_VISIBLE_DEVICES", "0,1,2", 1);
assert(getDeviceNumber(false) == 3);
... ...
}
13.3.3 Dump code object
To analyze compiler-related issues, you can use the dump code object:GPU_DUMP_CODE_OBJECT.
13.3. Useful environment variables 81
HIP Documentation, Release 6.3.42134
13.3.4 HSA-related environment variables (Linux)
HSA provides environment variables that help analyze issues in drivers or hardware.
• To isolate issues with hardware copy engines, you can useHSA_ENABLE_SDMA.
HSA_ENABLE_SDMA=0causeshost-to-deviceanddevice-to-hostcopiestousecomputeshaderblitker-
nels,ratherthanthededicatedDMAcopyengines. Computeshadercopieshavelowlatency(typically
< 5 us) and can achieve approximately 80% of the bandwidth of the DMA copy engine.
• To diagnose interrupt storm issues in the driver, you can useHSA_ENABLE_INTERRUPT.
HSA_ENABLE_INTERRUPT=0 causes completion signals to be detected with memory-based polling,
rather than interrupts.
13.3.5 HIP environment variable summary

## Here are some of the more commonly used environment variables

82 Chapter 13. Debugging with HIP
HIP Documentation, Release 6.3.42134
Environment
variable
De-
fault
value
Usage

## AMD_LOG_LEVEL

Enable HIP log
on different
Level
0 0: Disable log. 1: Enable log on error level 2: Enable log on warning and below levels
0x3: Enable log on information and below levels 0x4: Decode and display AQL packets

## AMD_LOG_MASK

Enable HIP log
on different
Level
0x7FFFFFFF0x1: Log API calls 0x02: Kernel and Copy Commands and Barriers 0x4: Synchroniza-
tionandwaitingforcommandstofinish0x8: Enablelogoninformationandbelowlevels

## 0x20: Queuecommandsandqueuecontents0x40: Signalcreation,allocation,pool0x80

Locks and thread-safety code 0x100: Copy debug 0x200: Detailed copy debug 0x400:
Resource allocation, performance-impacting events 0x800: Initialization and shutdown
0x1000: Misc debug, not yet classified 0x2000: Show raw bytes of AQL packet 0x4000:
Showcodecreationdebug0x8000: Moredetailedcommandinfo,includingbarriercom-
mands0x10000: Logmessagelocation0xFFFFFFFF:Logalwaysevenmaskflagiszero

## HIP_LAUNCH_BLOCKING

Used for serial-

```cpp

izationonkernel

```

execution.

```cpp

0 0: Disable. Kernel executes normally. 1: Enable. Serializes kernel enqueue, behaves the

```

same as AMD_SERIALIZE_KERNEL.

## HIP_VISIBLE_DEVICES

(or

## CUDA_VISIBLE_DEVICES)

Only devices
whose index is
present in the
sequence are
visible to HIP
0,1,2: Depending on the number of devices on the system

## GPU_DUMP_CODE_OBJECT

Dump code ob-
ject
0 0: Disable 1: Enable

## AMD_SERIALIZE_KERNEL


```cpp

Serialize kernel

```

enqueue
0 1: Wait for completion before enqueue 2: Wait for completion after enqueue 3: Both

## AMD_SERIALIZE_COPY

Serialize copies
0 1: Wait for completion before enqueue 2: Wait for completion after enqueue 3: Both

## HIP_HOST_COHERENT

Coherent mem-
ory in hipHost-
Malloc
0 0: memory is not coherent between host and GPU 1: memory is coherent with host

## AMD_DIRECT_DISPATCH

Enable direct

```cpp

kernel dispatch

```

(Currently for
Linux; under
development for
Windows)
1 0: Disable 1: Enable

## GPU_MAX_HW_QUEUES

The maximum
number of hard-
ware queues
allocated per
device
4 The variable controls how many independent hardware queues HIP runtime can create
per process, per device. If an application allocates more HIP streams than this number,
then HIP runtime reuses the same hardware queues for the new streams in a round-robin
manner. Note that this maximum number does not apply to hardware queues that are
created for CU-masked HIP streams, or cooperative queues for HIP Cooperative Groups
(single queue per device).
13.3. Useful environment variables 83
HIP Documentation, Release 6.3.42134
13.4 General debugging tips
• gdb --argscan be used to pass the executable and arguments togdb.

## • You can set environment variables (set env) from within GDB on Linux

(gdb) set env AMD_SERIALIZE_KERNEL 3
Note: Thisgdbcommand does not use an equal (=) sign.
• The GDB backtrace shows a path in the runtime. This is because a fault is caught by the runtime, but it is
generated by an asynchronous command running on the GPU.

```cpp

• Todeterminethetruelocationofafault,youcanforcethekernelstorunsynchronouslybysettingtheenvironment

```

variables AMD_SERIALIZE_KERNEL=3 and AMD_SERIALIZE_COPY=3. This forces HIP runtime to wait for the

```cpp

kernel to finish running before returning. If the fault occurs when a kernel is running, you can see the code
that launched the kernel inside the backtrace. The thread that’s causing the issue is typically the one inside

```

libhsa-runtime64.so.

## • VM faults inside kernels can be caused by

– Incorrect code (e.g., a for loop that extends past array boundaries)

```cpp

– Memory issues, such as invalid kernel arguments (null pointers, unregistered host pointers, bad pointers)

```

– Synchronization issues
– Compiler issues (incorrect code generation from the compiler)
– Runtime issues
84 Chapter 13. Debugging with HIP

## CHAPTER


## FOURTEEN


## LOGGING HIP ACTIVITY

HIPprovidesaloggingmechanismthatallowsyoutotraceHIPAPIandruntimecodeswhenrunningaHIPapplication.
In addition to being useful to our users/developers, the HIP development team uses these logs to improve the HIP
runtime.
By adjusting the logging settings and logging mask, you can get different types of information for different function-

```cpp

alities, such as HIP APIs, executed kernels, queue commands, and queue contents. Refer to the following sections for

```

examples.
Tip: Logging works for the release and debug versions of HIP. If you want to save logging output in a file, define the

## file when running the application via command line. For example

user@user-test:~/hip/bin$ ./hipinfo > ~/hip_log.txt
14.1 Logging level
HIPloggingisdisabledbydefault. Youcanenableitviathe AMD_LOG_LEVELenvironmentvariable. Thevalueofthis

## variable controls your logging level. Levels are defined as follows

enum LogLevel {

## LOG_NONE = 0,


## LOG_ERROR = 1,


## LOG_WARNING = 2,


## LOG_INFO = 3,


## LOG_DEBUG = 4

};
Tip: You can call a logging function with different logging levels. All information under the value set for
AMD_LOG_LEVELis printed.
85
HIP Documentation, Release 6.3.42134
14.2 Logging mask
The logging mask is designed to print functionality types when you’re running a HIP application. Once you set
AMD_LOG_LEVEL, the logging mask is set as the default value (0x7FFFFFFF). You can change this to any of the valid

## values

enum LogMask {
LOG_API = 0x00000001, //!< API call
LOG_CMD = 0x00000002, //!< Kernel and Copy Commands and Barriers
LOG_WAIT = 0x00000004, //!< Synchronization and waiting for commands to finish
LOG_AQL = 0x00000008, //!< Decode and display AQL packets
LOG_QUEUE = 0x00000010, //!< Queue commands and queue contents
LOG_SIG = 0x00000020, //!< Signal creation, allocation, pool
LOG_LOCK = 0x00000040, //!< Locks and thread-safety code.

```cpp

LOG_KERN = 0x00000080, //!< kernel creations and arguments, etc.

```

LOG_COPY = 0x00000100, //!< Copy debug
LOG_COPY2 = 0x00000200, //!< Detailed copy debug
LOG_RESOURCE = 0x00000400, //!< Resource allocation, performance-impacting events.
LOG_INIT = 0x00000800, //!< Initialization and shutdown
LOG_MISC = 0x00001000, //!< misc debug, not yet classified
LOG_AQL2 = 0x00002000, //!< Show raw bytes of AQL packet
LOG_CODE = 0x00004000, //!< Show code creation debug
LOG_CMD2 = 0x00008000, //!< More detailed command info, including barrier commands
LOG_LOCATION = 0x00010000, //!< Log message location
LOG_MEM = 0x00020000, //!< Memory allocation
LOG_MEM_POOL = 0x00040000, //!< Memory pool allocation, including memory in graphs
LOG_ALWAYS = 0xFFFFFFFF, //!< Log always even mask flag is zero
};
You can also define the logging mask via theAMD_LOG_MASKenvironment variable.
14.3 Logging command

## You can use the following code to print HIP logging information

#define ClPrint(level, mask, format, ...) \
do { \
if (AMD_LOG_LEVEL >= level) { \
if (AMD_LOG_MASK & mask || mask == amd::LOG_ALWAYS) { \
if (AMD_LOG_MASK & amd::LOG_LOCATION) { \
amd::log_printf(level, __FILENAME__, __LINE__, format, ##__VA_ARGS__);\
} else { \
amd::log_printf(level, "", 0, format, ##__VA_ARGS__); \
} \
} \
} \
} while (false)
Using HIP code, call theClPrint() function with the desired input variables. For example:
ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Initializing HSA stack.");
86 Chapter 14. Logging HIP activity
HIP Documentation, Release 6.3.42134
14.4 Logging examples
On Linux, you can enable HIP logging and retrieve logging information when you runhipinfo.
user@user-test:~/hip/bin$ export AMD_LOG_LEVEL=4
user@user-test:~/hip/bin$ ./hipinfo
:3:rocdevice.cpp :453 : 23647210092: Initializing HSA stack.
:3:comgrctx.cpp :33 : 23647639336: Loading COMGR library.
:3:rocdevice.cpp :203 : 23647687108: Numa select cpu␣
˓→agent[0]=0x13407c0(fine=0x13409a0,coarse=0x1340ad0) for gpu agent=0x1346150
:4:runtime.cpp :82 : 23647698669: init
:3:hip_device_runtime.cpp :473 : 23647698869: 5617 : [7fad295dd840] hipGetDeviceCount:␣
˓→Returned hipSuccess
:3:hip_device_runtime.cpp :502 : 23647698990: 5617 : [7fad295dd840] hipSetDevice ( 0 )
:3:hip_device_runtime.cpp :507 : 23647699042: 5617 : [7fad295dd840] hipSetDevice:␣
˓→Returned hipSuccess
--------------------------------------------------------------------------------
device# 0
:3:hip_device.cpp :150 : 23647699276: 5617 : [7fad295dd840]␣
˓→hipGetDeviceProperties ( 0x7ffdbe7db730, 0 )
:3:hip_device.cpp :237 : 23647699335: 5617 : [7fad295dd840]␣
˓→hipGetDeviceProperties: Returned hipSuccess
Name: Device 7341
pciBusID: 3
pciDeviceID: 0
pciDomainID: 0
multiProcessorCount: 11
maxThreadsPerMultiProcessor: 2560
isMultiGpuBoard: 0
clockRate: 1900 Mhz
memoryClockRate: 875 Mhz
memoryBusWidth: 0
clockInstructionRate: 1000 Mhz
totalGlobalMem: 7.98 GB
maxSharedMemoryPerMultiProcessor: 64.00 KB
totalConstMem: 8573157376
sharedMemPerBlock: 64.00 KB
canMapHostMemory: 1
regsPerBlock: 0
warpSize: 32
l2CacheSize: 0
computeMode: 0
maxThreadsPerBlock: 1024
maxThreadsDim.x: 1024
maxThreadsDim.y: 1024
maxThreadsDim.z: 1024
maxGridSize.x: 2147483647
maxGridSize.y: 2147483647
maxGridSize.z: 2147483647
major: 10
minor: 12
concurrentKernels: 1
(continues on next page)
14.4. Logging examples 87
HIP Documentation, Release 6.3.42134
(continued from previous page)
cooperativeLaunch: 0
cooperativeMultiDeviceLaunch: 0
arch.hasGlobalInt32Atomics: 1
...
gcnArch: 1012
isIntegrated: 0
maxTexture1D: 65536
maxTexture2D.width: 16384
maxTexture2D.height: 16384
maxTexture3D.width: 2048
maxTexture3D.height: 2048
maxTexture3D.depth: 2048
isLargeBar: 0
:3:hip_device_runtime.cpp :471 : 23647701557: 5617 : [7fad295dd840] hipGetDeviceCount␣
˓→( 0x7ffdbe7db714 )
:3:hip_device_runtime.cpp :473 : 23647701608: 5617 : [7fad295dd840] hipGetDeviceCount:␣
˓→Returned hipSuccess
:3:hip_peer.cpp :76 : 23647701731: 5617 : [7fad295dd840]␣
˓→hipDeviceCanAccessPeer ( 0x7ffdbe7db728, 0, 0 )
:3:hip_peer.cpp :60 : 23647701784: 5617 : [7fad295dd840] canAccessPeer:␣
˓→Returned hipSuccess
:3:hip_peer.cpp :77 : 23647701831: 5617 : [7fad295dd840]␣
˓→hipDeviceCanAccessPeer: Returned hipSuccess

## peers

:3:hip_peer.cpp :76 : 23647701921: 5617 : [7fad295dd840]␣
˓→hipDeviceCanAccessPeer ( 0x7ffdbe7db728, 0, 0 )
:3:hip_peer.cpp :60 : 23647701965: 5617 : [7fad295dd840] canAccessPeer:␣
˓→Returned hipSuccess
:3:hip_peer.cpp :77 : 23647701998: 5617 : [7fad295dd840]␣
˓→hipDeviceCanAccessPeer: Returned hipSuccess
non-peers: device#0
:3:hip_memory.cpp :345 : 23647702191: 5617 : [7fad295dd840] hipMemGetInfo (␣
˓→0x7ffdbe7db718, 0x7ffdbe7db720 )
:3:hip_memory.cpp :360 : 23647702243: 5617 : [7fad295dd840] hipMemGetInfo:␣
˓→Returned hipSuccess
memInfo.total: 7.98 GB
memInfo.free: 7.98 GB (100%)
On Windows, you can setAMD_LOG_LEVEL via environment variable from the advanced system settings or the com-
mand prompt (when run as administrator). The following example shows debug log information when calling the
backend runtime.
C:\hip\bin>set AMD_LOG_LEVEL=4
C:\hip\bin>hipinfo
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\device\comgrctx.cpp:33 :␣
˓→605413686305 us: 29864: [tid:0x9298] Loading COMGR library.
:4:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\platform\runtime.cpp:83 :␣
˓→605413869411 us: 29864: [tid:0x9298] init
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_context.cpp:47 ␣
˓→: 605413869502 us: 29864: [tid:0x9298] Direct Dispatch: 0
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device_runtime.
(continues on next page)
88 Chapter 14. Logging HIP activity
HIP Documentation, Release 6.3.42134
(continued from previous page)
˓→cpp:543 : 605413870553 us: 29864: [tid:0x9298] hipGetDeviceCount: Returned hipSuccess :
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device_runtime.
˓→cpp:556 : 605413870631 us: 29864: [tid:0x9298] ←[32m hipSetDevice ( 0 ) ←[0m
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device_runtime.
˓→cpp:561 : 605413870848 us: 29864: [tid:0x9298] hipSetDevice: Returned hipSuccess :
--------------------------------------------------------------------------------
device# 0
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device.cpp:346 :␣
˓→605413871623 us: 29864: [tid:0x9298] ←[32m hipGetDeviceProperties ( 0000008AEBEFF8C8,␣
˓→0 ) ←[0m
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device.cpp:348 :␣
˓→605413871695 us: 29864: [tid:0x9298] hipGetDeviceProperties: Returned hipSuccess :
Name: AMD Radeon(TM) Graphics
pciBusID: 3
pciDeviceID: 0
pciDomainID: 0
multiProcessorCount: 7
maxThreadsPerMultiProcessor: 2560
isMultiGpuBoard: 0
clockRate: 1600 Mhz
memoryClockRate: 1333 Mhz
memoryBusWidth: 0
totalGlobalMem: 12.06 GB
totalConstMem: 2147483647
sharedMemPerBlock: 64.00 KB
...
gcnArchName: gfx90c:xnack-
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device_runtime.
˓→cpp:541 : 605413924779 us: 29864: [tid:0x9298] ←[32m hipGetDeviceCount (␣
˓→0000008AEBEFF8A4 ) ←[0m
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device_runtime.
˓→cpp:543 : 605413925075 us: 29864: [tid:0x9298] hipGetDeviceCount: Returned hipSuccess :
peers: :3:C: \constructicon\builds\gfx\two\22.40\drivers\
˓→compute\hipamd\src\hip_peer.cpp:176 : 605413928643 us: 29864: [tid:0x9298] ←[32m␣
˓→hipDeviceCanAccessPeer ( 0000008AEBEFF890, 0, 0 ) ←[0m
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_peer.cpp:177 :␣
˓→605413928743 us: 29864: [tid:0x9298] hipDeviceCanAccessPeer: Returned hipSuccess :
non-peers: :3:C: \constructicon\builds\gfx\two\22.40\drivers\
˓→compute\hipamd\src\hip_peer.cpp:176 : 605413930830 us: 29864: [tid:0x9298] ←[32m␣
˓→hipDeviceCanAccessPeer ( 0000008AEBEFF890, 0, 0 ) ←[0m
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_peer.cpp:177 :␣
˓→605413930882 us: 29864: [tid:0x9298] hipDeviceCanAccessPeer: Returned hipSuccess :
device#0
...
:4:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\device\pal\palmemory.
˓→cpp:430 : 605414517802 us: 29864: [tid:0x9298] Free-: 8000 bytes, VM[ 3007c8000, ␣
˓→3007d0000]
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\device\devprogram.cpp:2979:␣
˓→605414517893 us: 29864: [tid:0x9298] For Init/Fini: Kernel Name: __amd_rocclr_
˓→copyBufferToImage
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\device\devprogram.cpp:2979:␣
˓→605414518259 us: 29864: [tid:0x9298] For Init/Fini: Kernel Name: __amd_rocclr_
(continues on next page)
14.4. Logging examples 89
HIP Documentation, Release 6.3.42134
(continued from previous page)
˓→copyBuffer
...
:4:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\device\pal\palmemory.
˓→cpp:206 : 605414523422 us: 29864: [tid:0x9298] Alloc: 100000 bytes,␣
˓→ptr[00000003008D0000-00000003009D0000], obj[00000003007D0000-00000003047D0000]
:4:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\device\pal\palmemory.
˓→cpp:206 : 605414523767 us: 29864: [tid:0x9298] Alloc: 100000 bytes,␣
˓→ptr[00000003009D0000-0000000300AD0000], obj[00000003007D0000-00000003047D0000]
:3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_memory.cpp:681 :␣

## ˓→605414524092 us: 29864: [tid:0x9298] hipMemGetInfo: Returned hipSuccess 

memInfo.total: 12.06 GB
memInfo.free: 11.93 GB (99%)
90 Chapter 14. Logging HIP activity

## CHAPTER


## FIFTEEN


## FREQUENTLY ASKED QUESTIONS

15.1 What APIs and features does HIP support?

## HIP provides the following

• Devices (hipSetDevice(),hipGetDeviceProperties(), etc.)

```cpp

• Memory management (hipMalloc(),hipMemcpy(),hipFree(), etc.)

```

• Streams (hipStreamCreate(),hipStreamSynchronize(),hipStreamWaitEvent(), etc.)
• Events (hipEventRecord(),hipEventElapsedTime(), etc.)

```cpp

• Kernel launching (hipLaunchKernel/hipLaunchKernelGGL is the preferred way of launching kernels.
hipLaunchKernelGGL is a standard C/C++ macro that can serve as an alternative way to launch kernels, re-

```

placing the CUDA triple-chevron (<<< >>>) syntax).
• HIP Module API to control when and how code is loaded.

```cpp

• CUDA-style kernel coordinate functions (threadIdx,blockIdx,blockDim,gridDim)

```

• Cross-lane instructions includingshfl,ballot,any,all
• Most device-side math built-ins
• Error reporting (hipGetLastError(),hipGetErrorString())
The HIP API documentation describes each API and its limitations, if any, compared with the equivalent CUDA API.
15.2 What is not supported?
15.2.1 Runtime/Driver API features

## At a high-level, the following features are not supported

• Textures (partial support available)
• Dynamic parallelism (CUDA 5.0)
• Graphics interoperability with OpenGL or Direct3D
• CUDA IPC Functions (Under Development)
• CUDA array,mipmappedArray and pitched memory
• Queue priority controls
See the API Support Table for more detailed information.
91
HIP Documentation, Release 6.3.42134
15.2.2 Kernel language features
• C+±style device-side dynamic memory allocations (free, new, delete) (CUDA 4.0)
• Virtual functions, indirect functions and try/catch (CUDA 4.0)
• __prof_trigger
• PTX assembly (CUDA 4.0). HIP-Clang supports inline GCN assembly.

```cpp

• Several kernel features are under development. See theC++ Language Extensionsfor more information.

```

15.3 Is HIP a drop-in replacement for CUDA?
No. HIP provides porting tools which do most of the work to convert CUDA code into portable C++ code that uses
the HIP APIs. Most developers will port their code from CUDA to HIP and then maintain the HIP version. HIP code
provides the same performance as native CUDA code, plus the benefits of running on AMD platforms.
15.4 What specific version of CUDA does HIP support?
HIP APIs and features do not map to a specific CUDA version. HIP provides a strong subset of the functionality
provided in CUDA, and the hipify tools can scan code to identify any unsupported CUDA functions - this is useful for
identifying the specific features required by a given application.
However, we can provide a rough summary of the features included in each CUDA SDK and the support level in HIP.
EachbulletbelowliststhemajornewlanguagefeaturesineachCUDAreleaseandthenindicatewhicharesupported/not

## supported in HIP


## • CUDA 4.0 and earlier 

– HIP supports CUDA 4.0 except for the limitations described above.

## • CUDA 5.0 

– Dynamic Parallelism (not supported)
– cuIpc functions (under development).

## • CUDA 6.0 

– Managed memory (under development)

## • CUDA 6.5 

– __shflintrinsic (supported)

## • CUDA 7.0 

– Per-thread default streams (supported)
– C++11 (Hip-Clang supports all of C++11, all of C++14 and some C++17 features)

## • CUDA 7.5 

– float16 (supported)

## • CUDA 8.0 

– Page Migration includingcudaMemAdvise,cudaMemPrefetch, othercudaMem*APIs(not supported)

## • CUDA 9.0 

92 Chapter 15. Frequently asked questions
HIP Documentation, Release 6.3.42134
– Cooperative Launch, Surface Object Management, Version Management
15.5 What libraries does HIP support?
HIP includes growing support for the four key math libraries using hipBLAS, hipFFT, hipRAND and hipSPARSE, as
well as MIOpen for machine intelligence applications. These offer pointer-based memory interfaces (as opposed to
opaque buffers) and can be easily interfaced with other HIP applications. The hip interfaces support both ROCm and
CUDA paths, with familiar library interfaces.
• hipBLAS, which utilizes rocBlas.
• hipFFT
• hipsSPARSE
• hipRAND
• MIOpen
Additionally, some of the cuBLAS routines are automatically converted to hipblas equivalents by the HIPIFY tools.
These APIs use cuBLAS or hcBLAS depending on the platform and replace the need to use conditional compilation.
15.6 How does HIP compare with OpenCL?
Both AMD and NVIDIA support OpenCL 1.2 on their devices so that developers can write portable code. HIP offers

## several benefits over OpenCL

• Developers can code in C++ as well as mix host and device C++ code in their source files. HIP C++ code can
use templates, lambdas, classes and so on.
• The HIP API is less verbose than OpenCL and is familiar to CUDA developers.
• BecausebothCUDAandHIPareC++languages,portingfromCUDAtoHIPissignificantlyeasierthanporting
from CUDA to OpenCL.
• HIP uses the best available development tools on each platform: on NVIDIA GPUs, HIP code compiles using
NVCC and can employ the Nsight profiler and debugger (unlike OpenCL on NVIDIA GPUs).
• HIP provides pointers and host-side pointer arithmetic.
• HIP provides device-level control over memory allocation and placement.
• HIP offers an offline compilation model.
15.7 How does porting CUDA to HIP compare to porting CUDA to
OpenCL?
Both HIP and CUDA are dialects of C++, and thus porting between them is relatively straightforward. Both dialects
support templates, classes, lambdas, and other C++ constructs. As one example, the hipify-perl tool was originally a
Perl script that used simple text conversions from CUDA to HIP. HIP and CUDA provide similar math library calls as
well. In summary, the HIP philosophy was to make the HIP language close enough to CUDA that the porting effort is
relatively simple. This reduces the potential for error, and also makes it easy to automate the translation. HIP goal is
to quickly get the ported program running on both platforms with little manual intervention, so that the programmer
can focus on performance optimizations.
15.5. What libraries does HIP support? 93
HIP Documentation, Release 6.3.42134
There have been several tools that have attempted to convert CUDA into OpenCL, such as CU2CL. OpenCL is a C99-

```cpp

basedkernellanguage(ratherthanC++)andalsodoesnotsupportsingle-sourcecompilation. Asaresult,theOpenCL

```

syntax is different from CUDA, and the porting tools have to perform some heroic transformations to bridge this gap.
ThetoolsalsostrugglewithmorecomplexCUDAapplications,inparticular,thosethatusetemplates,classes,orother

```cpp

C++ features inside the kernel.

```

15.8 What hardware does HIP support?
• For AMD platforms, see the ROCm documentation for the list of supported platforms.
• ForNVIDIAplatforms,HIPrequiresunifiedmemoryandshouldrunonanydevicesupportingCUDASDK6.0
or newer. We have tested the NVIDIA Titan and Tesla K40.
15.9 Do HIPIFY tools automatically convert all source code?
Typically, HIPIFY tools can automatically convert almost all run-time code. Most device code needs no additional
conversion since HIP and CUDA have similar names for math and built-in functions. The hipify-clang tool will auto-

```cpp

maticallymodifythekernelsignatureasneeded(automatingastepthatusedtobedonemanually). Additionalporting

```

may be required to deal with architecture feature queries or with CUDA capabilities that HIP doesn’t support. In
general, developers should always expect to perform some platform-specific tuning and optimization.
15.10 What is NVCC?
NVCC is NVIDIA’s compiler driver for compiling “CUDA C++” code into PTX or device code for NVIDIA GPUs.
It’s a closed-source binary compiler that is provided by the CUDA SDK.
15.11 What is HIP-Clang?
HIP-Clang is a Clang/LLVM based compiler to compile HIP programs which can run on AMD platform.
15.12 Why use HIP rather than supporting CUDA directly?
While HIP is a strong subset of the CUDA, it is a subset. The HIP layer allows that subset to be clearly defined and
documented. DeveloperswhocodetotheHIPAPIcanbeassuredtheircodewillremainportableacrossNVIDIAand
AMD platforms. In addition, HIP defines portable mechanisms to query architectural features and supports a larger
64-bit WaveSize which expands the return type for cross-lane functions like ballot and shuffle from 32-bit integers to
64-bit integers.
94 Chapter 15. Frequently asked questions
HIP Documentation, Release 6.3.42134
15.13 Can I develop HIP code on an NVIDIA CUDA platform?
Yes. HIP’s CUDA path only exposes the APIs and functionality that work on both NVCC and AMDGPU back-ends.
“Extra”APIs,parameters,andfeatureswhichexistinCUDAbutnotinHIP-Clangwilltypicallyresultincompile-time
orrun-timeerrors. DevelopersneedtousetheHIPAPIformostacceleratorcodeandbracketanyCUDA-specificcode
with preprocessor conditionals. Developers concerned about portability should, of course, run on both platforms, and
should expect to tune for performance. In some cases, CUDA has a richer set of modes for some APIs, and some C++
capabilities such as virtual functions - see the HIP @API documentation for more details.
15.14 Can I develop HIP code on an AMD HIP-Clang platform?
Yes. HIP’s HIP-Clangpath only exposes theAPIs and functions thatwork on AMDruntime back ends. “Extra” APIs,
parameters and features that appear in HIP-Clang but not CUDA will typically cause compile- or run-time errors.
Developers must use the HIP API for most accelerator code and bracket any HIP-Clang specific code with preproces-
sor conditionals. Those concerned about portability should, of course, test their code on both platforms and should
tune it for performance. Typically, HIP-Clang supports a more modern set of C++11/C++14/C++17 features, so HIP
developers who want portability should be careful when using advanced C++ features on the HIP-Clang path.
15.15 How to use HIP-Clang to build HIP programs?

## The environment variable can be used to set compiler path

• HIP_CLANG_PATH: path to hip-clang. When set, this variable let hipcc to use hip-clang for compila-
tion/linking.

## There is an alternative environment variable to set compiler path

• HIP_ROCCLR_HOME: path to root directory of the HIP-ROCclr runtime. When set, this variable let hipcc
use hip-clang from the ROCclr distribution. NOTE: If HIP_ROCCLR_HOME is set, there is no need to set
HIP_CLANG_PATH since hipcc will deduce them from HIP_ROCCLR_HOME.
15.16 What is AMD clr?
AMD Compute Language Runtime (CLR) is a repository for the AMD platform, which contains source codes for
AMD’s compute languages runtimes as follows,
• hipamd - contains implementation of HIP language for AMD GPU.
• rocclr - contains virtual device interfaces that compute runtimes interact with backends, such as ROCr on Linux
and PAL on Windows.
• opencl - contains implementation of OpenCL™ on the AMD platform.
15.13. Can I develop HIP code on an NVIDIA CUDA platform? 95
HIP Documentation, Release 6.3.42134
15.17 What is hipother?
A new repository ‘hipother’ is added in the ROCm 6.1 release, which is branched out from HIP. hipother supports the
HIP back-end implementation on some non-AMD platforms, like NVIDIA.
15.18 Can I get HIP open source repository for Windows?
No, there is no HIP repository open publicly on Windows.
15.19 Can a HIP binary run on both AMD and NVIDIA platforms?
HIP is a source-portable language that can be compiled to run on either AMD or NVIDIA platform. HIP tools don’t
create a “fat binary” that can run on either platform, however.
15.20 OnHIP-Clang, canIlinkHIPcodewithhostcodecompiledwith
another compiler such as gcc, icc, or clang?
Yes. HIP generates the object code which conforms to the GCC ABI, and also links with libstdc++. This means you
can compile host code with the compiler of your choice and link the generated object code with GPU code compiled
withHIP.Largerprojectsoftencontainamixtureofacceleratorcode(initiallywritteninCUDAwithNVCC)andhost
code(compiledwithgcc,icc,orclang). TheseprojectscanconverttheacceleratorcodetoHIP,compilethatcodewith
hipcc, and link with object code from their preferred compiler.
15.21 CanHIPAPIsupportCstyleapplication? Whatisthedifference
between C and C++?
HIP is C++ runtime API that supports C style applications as well.
Some C style applications (and interfaces to other languages (FORTRAN, Python)) would call certain HIP APIs but

```cpp

not use kernel programming. They can be compiled with a C compiler and run correctly, however, small details must
beconsideredinthecode. Forexample,initialization,asshowninthesimpleapplicationbelow,usesHIPstructsdim3

```

with the file name “test.hip.cpp”

```cpp

#include "hip/hip_runtime_api.h"
#include "stdio.h"

```

int main(int argc, char** argv) {

```cpp

dim3 grid1;
printf("dim3 grid1; x=%d, y=%d, z=%d\n",grid1.x,grid1.y,grid1.z);
dim3 grid2 = {1,1,1};
printf("dim3 grid2 = {1,1,1}; x=%d, y=%d, z=%d\n",grid2.x,grid2.y,grid2.z);

```

return 0;
}
When using a C++ compiler,
96 Chapter 15. Frequently asked questions
HIP Documentation, Release 6.3.42134
$ gcc -x c++ $(hipconfig --cpp_config) test3.hip.cpp -o test
$ ./test

```cpp

dim3 grid1; x=1, y=1, z=1
dim3 grid2 = {1,1,1}; x=1, y=1, z=1
In which “dim3 grid1;” will yield a dim3 grid with all dimensional members x,y,z initialized to 1, as the default

## constructor behaves that way. Further, if written

dim3 grid(2); // yields {2,1,1}
dim3 grid(2,3); yields {2,3,1}

```

In comparison, when using the C compiler,
$ gcc -x c $(hipconfig --cpp_config) test.hip.cpp -o test
$ ./test

```cpp

dim3 grid1; x=646881376, y=21975, z=1517277280
dim3 grid2 = {1,1,1}; x=1, y=1, z=1
In which “dim3 grid;” does not imply any initialization, no constructor is called, and dimensional values x,y,z of grid

```

are undefined. NOTE: To get the C++ default behavior, C programmers must additionally specify the right-hand side
as shown below,

```cpp

dim3 grid = {1,1,1}; // initialized as in C++

```

15.22 Can I install both CUDA SDK and HIP-Clang on the same ma-
chine?
Yes. You can use HIP_PLATFORM to choose which path hipcc targets. This configuration can be useful when using
HIP to develop an application which is portable to both AMD and NVIDIA.
15.23 HIP detected my platform (HIP-Clang vs NVCC) incorrectly *
what should I do?
HIP will set the platform to AMD and use HIP-Clang as compiler if it sees that the AMD graphics driver is installed
and has detected an AMD GPU. Sometimes this isn’t what you want * you can force HIP to recognize the platform by
setting the following,
export HIP_PLATFORM=amd
HIP then set and use correct AMD compiler and runtime, HIP_COMPILER=clang HIP_RUNTIME=rocclr
To choose NVIDIA platform, you can set,
export HIP_PLATFORM=nvidia
In this case, HIP will set and use the following,
HIP_COMPILER=cuda
HIP_RUNTIME=nvcc
15.22. Can I install both CUDA SDK and HIP-Clang on the same machine? 97
HIP Documentation, Release 6.3.42134
Onesymptomofthisproblemisthemessage“error: ‘unknownerror’(11)at square.hipref.cpp:56. Thiscanoccur
if you have a CUDA installation on an AMD platform, and HIP incorrectly detects the platform as NVCC. HIP may
beabletocompiletheapplicationusingtheNVCCtool-chainbutwillgeneratethiserroratruntimesincetheplatform
does not have a CUDA device.
15.24 On CUDA, can I mix CUDA code with HIP code?
Yes. MostHIPdatastructures( hipStream_t,hipEvent_t)aretypedefstoCUDAequivalentsandcanbeintermixed.
Both CUDA and HIP use integer device ids. One notable exception is thathipError_t is a new type, and cannot be
used where acudaError_t is expected. In these cases, refactor the code to remove the expectation. Alternatively,
hip_runtime_api.h defines functions which convert between the error code spaces:
hipErrorToCudaError hipCUDAErrorTohipError hipCUResultTohipError
If platform portability is important, use#ifdef __HIP_PLATFORM_NVIDIA__ to guard the CUDA-specific code.
15.25 How do I trace HIP application flow?
See Logging HIP activityfor more information.

```cpp

15.26 What are the maximum limits of kernel launch parameters?
Productofblock.x,block.y,andblock.zshouldbelessthan1024. Pleasenote,HIPdoesnotsupportkernellaunchwith

```

total work items defined in dimension with sizegridDim x blockDim >= 2^32, sogridDim.x * blockDim.x,
gridDim.y * blockDim.y and gridDim.z * blockDim.z are always less than 2^32.
15.27 Are __shfl_*_sync functions supported on HIP platform?
__shfl_*_sync is not supported on HIP but for NVCC path CUDA 9.0 and above all shuffle calls get redirected to
it’s sync version.
15.28 Howtocreateaguardforcodethatisspecifictothehostorthe

## GPU?

The compiler defines the__HIP_DEVICE_COMPILE__ macro only when compiling the code for the GPU. It could be
used to guard code that is specific to the host or the GPU.
98 Chapter 15. Frequently asked questions
HIP Documentation, Release 6.3.42134
15.29 Why _OpenMP is undefined when compiling with-fopenmp?
WhencompilinganOpenMPsourcefilewith hipcc -fopenmp,thecompilermaygenerateerrorifthereisareference
tothe _OPENMPmacro. Thisisduetoalimitationinhipccthattreatsanysourcefiletype(forexample .cpp)asanHIP
translationunitleadingtosomeconflictswiththeOpenMPlanguageswitch. IftheOpenMPsourcefiledoesn’tcontain
any HIP language constructs you could work around this issue by adding the-x c++ switch to force the compiler to
treat the file as regular C++. Another approach would be to guard the OpenMP code with#ifdef _OPENMP so that
the code block is disabled when compiling for the GPU. The__HIP_DEVICE_COMPILE__ macro defined by the HIP
compiler when compiling GPU code could also be used for guarding code paths specific to the host or the GPU.
15.30 Does the HIP-Clang compiler support extern shared declara-
tions?
Previously, it was essential to declare dynamic shared memory using the HIP_DYNAMIC_SHARED macro for accu-

```cpp

racy, as using static shared memory in the same kernel could result in overlapping memory ranges and data-races.

```

Now, the HIP-Clang compiler provides support for extern shared declarations, and the HIP_DYNAMIC_SHARED
option is no longer required. You may use the standard extern definition: externsharedtype var[];
15.31 I have multiple HIP enabled devices and I am getting an error
code hipErrorSharedObjectInitFailed with the message “Er-
ror: shared object initialization failed”?
This error message is seen due to the fact that you do not have valid code object for all of your devices.
Ifyouhavecompiledtheapplicationyourself,makesureyouhavegiventhecorrectdevicename(s)anditsfeaturesvia:
--offload-arch. Ifyouarenotmentioningthe --offload-arch,makesurethat hipccisusingthecorrectoffload
arch by verifying the hipcc output generated by setting the environment variableHIPCC_VERBOSE=1.
If you have a precompiled application/library (like rocblas, TensorFlow etc) which gives you such error, there are one
of two possibilities.
• The application/library does not ship code object bundles forall of your device(s): in this case you need to
recompile the application/library yourself with correct--offload-arch.
• The application/library does not ship code object bundles forsome of your device(s), for example you have
a system with an APU + GPU and the library does not ship code objects for your APU. For this you can set
the environment variableHIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES on NVIDIA platform, to only
enable GPUs for which code object is available. This will limit the GPUs visible to your application and allow
it to run.
Note: In previous releases, the error code ishipErrorNoBinaryForGpu with message “Unable to find code ob-
ject for all current devices”. The error code handling behavior is changed. HIP runtime shows the error code
hipErrorSharedObjectInitFailedwithmessage“Error: sharedobjectinitializationfailed”onunsupportedGPU.
15.29. Why _OpenMP is undefined when compiling with-fopenmp? 99
HIP Documentation, Release 6.3.42134
15.32 How to use per-thread default stream in HIP?
The per-thread default stream is an implicit stream local to both the thread and the current device. It does not do
any implicit synchronization with other streams (like explicitly created streams), or default per-thread stream on other
threads.
Theper-threaddefaultstreamisablockingstreamandwillsynchronizewiththedefaultnullstreamifbothareusedin
a program.
InROCm,acompilationoptionshouldbeaddedinordertocompilethetranslationunitwithper-threaddefaultstream
enabled. -fgpu-default-stream=per-thread. Once source is compiled with per-thread default stream enabled,
all APIs will be executed on per thread default stream, hence there will not be any implicit synchronization with other
streams.
Besides, per-thread default stream be enabled per translation unit, users can compile some files with feature enabled
and some with feature disabled. Feature enabled translation unit will have default stream as per thread and there will
not be any implicit synchronization done but other modules will have legacy default stream which will do implicit
synchronization.
15.33 How to use complex multiplication and division operations?
In HIP,hipFloatComplexand hipDoubleComplex are defined as complex data types,
typedef float2 hipFloatComplex;
typedef double2 hipDoubleComplex;
Any application uses complex multiplication and division operations, need to replace ‘*’ and ‘/’ operators with the
following,
• hipCmulf()and hipCdivf()forhipFloatComplex
• hipCmul() and hipCdiv() forhipDoubleComplex
Note: These complex operations are equivalent to corresponding types/functions on the NVIDIA platform.
15.34 CanIdevelopapplicationswithHIPAPIsonWindowsthesame
on Linux?
Yes, HIP APIs are available to use on both Linux and Windows. Due to different working mechanisms on operating

```cpp

systemslikeWindowsvsLinux, HIPAPIscallcorrespondinglowerlevelbackendruntimelibrariesandkerneldrivers

```

for the OS, in order to control the executions on GPU hardware accordingly. There might be a few differences on the
relatedbackendsoftwareanddriversupport,whichmightaffectusageofHIPAPIs. SeeOSsupportdetailsinHIPAPI
document.
100 Chapter 15. Frequently asked questions
HIP Documentation, Release 6.3.42134
15.35 Does HIP support LUID?
Starting ROCm 6.0, HIP runtime supports Locally Unique Identifier (LUID). This feature enables the local physical
device(s) to interoperate with other devices. For example, DirectX 12.
HIP runtime sets device LUID properties so the driver can query LUID to identify each device for interoperability.
Note: HIP supports LUID only on Windows OS.
15.36 How can I know the version of HIP?
HIP version definition has been updated since ROCm 4.2 release as the following:

## HIP_VERSION=HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH

HIP version can be queried from HIP API call,
hipRuntimeGetVersion(&runtimeVersion);
The version returned will always be greater than the versions in previous ROCm releases.
Note: The version definition of HIP runtime is different from CUDA. On AMD platform, the function returns HIP
runtime version, while on NVIDIA platform, it returns CUDA runtime version. And there is no mapping/correlation
between HIP version and CUDA version.
15.35. Does HIP support LUID? 101
HIP Documentation, Release 6.3.42134
102 Chapter 15. Frequently asked questions
103
HIP Documentation, Release 6.3.42134

## CHAPTER


## SIXTEEN


## HIP RUNTIME API REFERENCE

16.1 Related Pages
16.2 Modules
16.3 Data Structures
16.3.1 Data Structures
16.3.2 Data Structure Index
16.3.3 Data Fields
16.3.3.1 All
16.3.3.1.1 Data Fields
16.3.3.1.2 Data Fields
16.3.3.1.3 Data Fields
16.3.3.1.4 Data Fields
16.3.3.1.5 Data Fields
16.3.3.1.6 Data Fields
16.3.3.1.7 Data Fields
16.3.3.1.8 Data Fields
16.3.3.1.9 Data Fields
16.3.3.1.10 Data Fields
16.3.3.1.11 Data Fields
16.3.3.1.12 Data Fields
16.3.3.1.13 Data Fields
16.3.3.1.14 Data Fields
16.3.3.1.15 Data Fields
16.3.3.1.16 Data Fields
16.3.3.1.17 Data Fields
16.3.3.1.18 Data Fields
16.3.3.1.19 Data Fields
16.3.3.1.20 Data Fields
16.3.3.1.21 Data Fields
16.3.3.1.22 Data Fields
16.3.3.1.23 Data Fields
16.3.3.1.24 Data Fields
16.3.3.2 Variables
16.3.3.2.1 Data Fields - Variables
16.3.3.2.2 Data Fields - Variables
16.3.3.2.3 Data Fields - Variables
16.3.3.2.4 Data Fields - Variables
16.3.3.2.5 Data Fields - Variables
16.3.3.2.6 Data Fields - Variables
16.3.3.2.7 Data Fields - Variables
16.3.3.2.8 Data Fields - Variables
16.3.3.2.9 Data Fields - Variables
16.3.3.2.10 Data Fields - Variables
16.3.3.2.11 Data Fields - Variables
16.3.3.2.12 Data Fields - Variables
16.3.3.2.13 Data Fields - Variables
16.3.3.2.14 Data Fields - Variables
16.3.3.2.15 Data Fields - Variables
16.3.3.2.16 Data Fields - Variables
16.3.3.2.17 Data Fields - Variables
16.3.3.2.18 Data Fields - Variables
16.3.3.2.19 Data Fields - Variables
16.3.3.2.20 Data Fields - Variables
16.3.3.2.21 Data Fields - Variables
16.3.3.2.22 Data Fields - Variables
16.3.3.2.23 Data Fields - Variables
16.3.3.2.24 Data Fields - Variables
16.4 Files
16.4.1 File List
16.4.2 Globals
16.4.2.1 All
16.4.2.1.1 Globals
16.4.2.1.2 Globals
16.4.2.1.3 Globals
16.4.2.1.4 Globals
16.4.2.1.5 Globals
16.4.2.1.6 Globals
16.4.2.1.7 Globals
16.4.2.1.8 Globals
16.4.2.1.9 Globals
16.4.2.1.10 Globals
16.4.2.1.11 Globals
16.4.2.1.12 Globals
16.4.2.1.13 Globals
16.4.2.1.14 Globals
16.4.2.1.15 Globals
16.4.2.1.16 Globals
16.4.2.1.17 Globals
16.4.2.1.18 Globals
16.4.2.1.19 Globals
16.4.2.2 Functions
16.4.2.2.1 Globals
16.4.2.2.2 Globals
16.4.2.2.3 Globals
16.4.2.2.4 Globals
16.4.2.2.5 Globals
16.4.2.2.6 Globals
16.4.2.2.7 Globals
16.4.2.2.8 Globals
16.4.2.2.9 Globals
16.4.2.2.10 Globals
16.4.2.2.11 Globals
16.4.2.2.12 Globals
16.4.2.2.13 Globals
16.4.2.2.14 Globals
16.4.2.2.15 Globals
16.4.2.2.16 Globals
16.4.2.3 Globals
16.4.2.4 Globals
16.4.2.5 Globals
16.4.2.6 Enumerator
16.4.2.6.1 Globals
16.4.2.6.2 Globals
16.4.2.7 Globals
104 Chapter 16. HIP Runtime API Reference

## CHAPTER


## SEVENTEEN


## C++ LANGUAGE EXTENSIONS


```cpp

HIPprovidesaC++syntaxthatissuitableforcompilingmostcodethatcommonlyappearsincomputekernels(classes,

```

namespaces, operator overloading, and templates). HIP also defines other language features that are designed to target

## accelerators, such as


```cpp

• A kernel-launch syntax that uses standard C++ (this resembles a function call and is portable to all HIP targets)

```

• Short-vector headers that can serve on a host or device
• Math functions that resemble those inmath.h, which is included with standard C++ compilers
• Built-in functions for accessing specific GPU hardware capabilities

```cpp

Note: Thischapterdescribesthebuilt-invariablesandfunctionsthatareaccessiblefromtheHIPkernel. It’sintended
for users who are familiar with CUDA kernel syntax and want to learn how HIP differs from CUDA.

## Features are labeled with one of the following keywords


```

• Supported: HIP supports the feature with a CUDA-equivalent function
• Not supported: HIP does not support the feature
• Under development: The feature is under development and not yet available
17.1 Function-type qualifiers
17.1.1 __device__

## Supported __device__ functions are

• Run on the device
• Called from the device only
You can combine__device__with the host keyword (__host__).
105
HIP Documentation, Release 6.3.42134

```cpp

17.1.2 __global__

## Supported __global__ functions are


```

• Run on the device
• Called (launched) from the host

```cpp

HIP __global__ functions must have avoid return type. The first parameter in a HIP__global__ function must

```

have the typehipLaunchParm. Refer toKernel launch exampleto see usage.

```cpp

HIP doesn’t support dynamic-parallelism, which means that you can’t call__global__functions from the device.

```

17.1.3 __host__

## Supported __host__ functions are

• Run on the host
• Called from the host
You can combine__host__ with __device__; in this case, the function compiles for the host and the device. Note

```cpp

that these functions can’t use the HIP grid coordinate functions (e.g.,threadIdx.x). If you need to use HIP grid

```

coordinate functions, you can pass the necessary coordinate information as an argument.

```cpp

You can’t combine__host__ with __global__.

```

HIP parses the__noinline__ and __forceinline__ keywords and converts them into the appropriate Clang at-
tributes.

```cpp

17.2 Calling __global__ functions
__global__ functions are often referred to askernels. When you call a global function, you’relaunching a kernel.
When launching a kernel, you must specify an execution configuration that includes the grid and block dimensions.

```

Theexecutionconfigurationcanalsoincludeotherinformationforthelaunch, suchastheamountofadditionalshared

```cpp

memory to allocate and the stream where you want to execute the kernel.
HIPintroducesastandardC++callingconvention( hipLaunchKernelGGL)topasstherunconfigurationtothekernel.

```

However, you can also use the CUDA<<< >>>syntax.

## When usinghipLaunchKernelGGL, your first five parameters must be


```cpp

• symbol kernelName: The name of the kernel you want to launch. To support template kernels that contain

```

",", use theHIP_KERNEL_NAMEmacro (HIPIFY tools insert this automatically).

```cpp

• dim3 gridDim: 3D-grid dimensions that specify the number of blocks to launch.
• dim3 blockDim: 3D-block dimensions that specify the number of threads in each block.

```

• size_t dynamicShared: The amount of additional shared memory that you want to allocate when launching

```cpp

the kernel (see__shared__).
• hipStream_t: The stream where you want to run the kernel. A value of0 corresponds to the NULL stream

```

(seeSynchronization functions).

```cpp

You can include your kernel arguments after these parameters.

```

106 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134

## // Example hipLaunchKernelGGL pseudocode


```cpp

__global__ MyKernel(hipLaunchParm lp, float *A, float *B, float *C, size_t N)

```

{
...
}

```cpp

MyKernel<<<dim3(gridDim), dim3(groupDim), 0, 0>>> (a,b,c,n);

## // Alternatively, you can launch the kernel using

// hipLaunchKernelGGL(MyKernel, dim3(gridDim), dim3(groupDim), 0/*dynamicShared*/, 0/

```

˓→*stream), a, b, c, n);
YoucanuseHIPIFYtoolstoconvertCUDAlaunchsyntaxto hipLaunchKernelGGL.Thisincludestheconversionof
optional<<< >>>arguments into the five requiredhipLaunchKernelGGLparameters.

```cpp

Note: HIP doesn’t support dimension sizes of𝑔𝑟𝑖𝑑𝐷𝑖𝑚* 𝑏𝑙𝑜𝑐𝑘𝐷𝑖𝑚≥ 232 when launching a kernel.

```

17.3 Kernel launch example
// Example showing device function, __device__ __host__
// <- compile for both device and host
float PlusOne(float x)
{
return x + 1.0;
}

```cpp

__global__

```

void
MyKernel (hipLaunchParm lp, /*lp parm for execution configuration */
const float *a, const float *b, float *c, unsigned N)
{

```cpp

unsigned gid = threadIdx.x; // <- coordinate index function

```

if (gid < N) {
c[gid] = a[gid] + PlusOne(b[gid]);
}
}
void callMyKernel()
{
float *a, *b, *c; // initialization not shown...
unsigned N = 1000000;
const unsigned blockSize = 256;

```cpp

MyKernel<<<dim3(gridDim), dim3(groupDim), 0, 0>>> (a,b,c,n);
// Alternatively, kernel can be launched by
// hipLaunchKernelGGL(MyKernel, dim3(N/blockSize), dim3(blockSize), 0, 0, a,b,c,N);

```

}
17.3. Kernel launch example 107
HIP Documentation, Release 6.3.42134
17.4 Variable type qualifiers
17.4.1 __constant__

```cpp

The host writes constant memory before launching the kernel. This memory is read-only from the GPU while the

## kernel is running. The functions for accessing constant memory are


```

• hipGetSymbolAddress()
• hipGetSymbolSize()

```cpp

• hipMemcpyToSymbol()
• hipMemcpyToSymbolAsync()
• hipMemcpyFromSymbol()
• hipMemcpyFromSymbolAsync()

```

17.4.2 __shared__
Toallowthehosttodynamicallyallocatesharedmemory,youcanspecify extern __shared__asalaunchparameter.
Note: PriortotheHIP-Clangcompiler,dynamicsharedmemoryhadtobedeclaredusingthe HIP_DYNAMIC_SHARED

```cpp

macro in order to ensure accuracy. This is because using static shared memory in the same kernel could’ve resulted

```

in overlapping memory ranges and data-races. The HIP-Clang compiler provides support forextern __shared_
declarations, soHIP_DYNAMIC_SHARED is no longer required.
17.4.3 __managed__
Managed memory, including the__managed__ keyword, is supported in HIP combined host/device compilation.
17.4.4 __restrict__

```cpp

__restrict__ tells the compiler that the associated memory pointer not to alias with any other pointer in the kernel

```

orfunction. Thiscanhelpthecompilergeneratebettercode. Inmostusecases,everypointerargumentshouldusethis
keyword in order to achieve the benefit.
17.5 Built-in variables
17.5.1 Coordinate built-ins

```cpp

The kernel uses coordinate built-ins (thread*,block*,grid*) to determine the coordinate index and bounds for the

```

active work item.
Built-ins are defined inamd_hip_runtime.h, rather than being implicitly defined by the compiler.

```cpp

Coordinatevariabledefinitionsforbuilt-insarethesameforHIPandCUDA.Forexample: threadIdx.x,blockIdx.

```

y, and gridDim.y. The products gridDim.x * blockDim.x, gridDim.y * blockDim.y, and gridDim.z *
blockDim.zare always less than2^32.
108 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Coordinate built-ins are implemented as structures for improved performance. When used withprintf, they must be
explicitly cast to integer types.
17.5.2 warpSize
ThewarpSizevariabletypeis int. Itcontainsthewarp size(inthreads)for thetargetdevice. warpSizeshouldonly
be used in device functions that develop portable wave-aware code.
Note: NVIDIA devices return 32 for this variable; AMD devices return 64 for gfx9 and 32 for gfx10 and above.
17.6 Vector types
The following vector types are defined inhip_runtime.h. They are not automatically provided by the compiler.
17.6.1 Short vector types
Short vector types derive from basic integer and floating-point types. These structures are defined in
hip_vector_types.h. The first, second, third, and fourth components of the vector are defined by thex,y,z, andw
fields, respectively. All short vector types support a constructor function of the formmake_<type_name>(). For ex-
ample, float4 make_float4(float x, float y, float z, float w) creates a vector with typefloat4 and
value(x,y,z,w).

## HIP supports the following short vector formats


## • Signed Integers

– char1,char2,char3,char4
– short1,short2,short3,short4
– int1,int2,int3,int4
– long1,long2,long3,long4
– longlong1,longlong2,longlong3,longlong4

## • Unsigned Integers

– uchar1,uchar2,uchar3,uchar4
– ushort1,ushort2,ushort3,ushort4
– uint1,uint2,uint3,uint4
– ulong1,ulong2,ulong3,ulong4
– ulonglong1,ulonglong2,ulonglong3,ulonglong4

## • Floating Points

– float1,float2,float3,float4
– double1,double2,double3,double4
17.6. Vector types 109
HIP Documentation, Release 6.3.42134

```cpp

17.6.2 dim3
dim3is a three-dimensional integer vector type that is commonly used to specify grid and group dimensions.
Thedim3constructoracceptsbetweenzeroandthreearguments. Bydefault,itinitializesunspecifieddimensionsto1.
typedef struct dim3 {

```

uint32_t x;
uint32_t y;
uint32_t z;

```cpp

dim3(uint32_t _x=1, uint32_t _y=1, uint32_t _z=1) : x(_x), y(_y), z(_z) {};

```

};
17.7 Memory fence instructions
HIP supports__threadfence() and __threadfence_block(). If you’re usingthreadfence_system() in the

## HIP-Clang path, you can use the following workaround

1. Build HIP with theHIP_COHERENT_HOST_ALLOCenvironment variable enabled.

## 2. Modify kernels that use__threadfence_system() as follows


```cpp

• Ensure the kernel operates only on fine-grained system memory, which should be allocated with

```

hipHostMalloc().
• Removememcpy for all allocated fine-grained system memory regions.
17.8 Synchronization functions
The __syncthreads() built-in function is supported in HIP. The __syncthreads_count(int),
__syncthreads_and(int), and__syncthreads_or(int) functions are under development.
17.9 Math functions
HIP-Clangsupportsasetofmathoperationsthatarecallablefromthedevice. HIPsupportsmostofthedevicefunctions
supported by CUDA. These are described in the following sections.
17.9.1 Single precision mathematical functions
Following is the list of supported single precision mathematical functions.
Table 17.1: Single precision mathematical functions
Function Supported on Host Supported on Device
float abs(float x)
Returns the absolute value of𝑥
✓ ✓
continues on next page
110 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.1 – continued from previous page
float acosf(float x)
Returns the arc cosine of𝑥.
✓ ✓
float acoshf(float x)
Returns the nonnegative arc
hyperbolic cosine of𝑥.
✓ ✓
float asinf(float x)
Returns the arc sine of𝑥.
✓ ✓
float asinhf(float x)
Returnsthearchyperbolicsineof 𝑥.
✓ ✓
float atanf(float x)
Returns the arc tangent of𝑥.
✓ ✓
float atan2f(float x, float
y)
Returns the arc tangent of the ratio
of 𝑥 and 𝑦.
✓ ✓
float atanhf(float x)
Returns the arc hyperbolic tangent
of 𝑥.
✓ ✓
float cbrtf(float x)
Returns the cube root of𝑥.
✓ ✓
float ceilf(float x)
Returns ceiling of𝑥.
✓ ✓
float copysignf(float x,
float y)
Create value with given magnitude,
copying sign of second value.
✓ ✓
float cosf(float x)
Returns the cosine of𝑥.
✓ ✓
continues on next page
17.9. Math functions 111
HIP Documentation, Release 6.3.42134
Table 17.1 – continued from previous page
float coshf(float x)
Returns the hyperbolic cosine of𝑥.
✓ ✓
float cospif(float x)
Returns the cosine of𝜋· 𝑥.
✓ ✓
float cyl_bessel_i0f(float
x)
Returns the value of the regular
modified cylindrical Bessel
function of order 0 for𝑥.
float cyl_bessel_i1f(float
x)
Returns the value of the regular
modified cylindrical Bessel
function of order 1 for𝑥.
float erff(float x)
Returns the error function of𝑥.
✓ ✓
float erfcf(float x)
Returns the complementary error
function of𝑥.
✓ ✓
float erfcinvf(float x)
Returns the inverse complementary
function of𝑥.
✓ ✓
float erfcxf(float x)
Returns the scaled complementary
error function of𝑥.
✓ ✓
float erfinvf(float x)
Returns the inverse error function
of 𝑥.
✓ ✓
float expf(float x)
Returns 𝑒𝑥.
✓ ✓
continues on next page
112 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.1 – continued from previous page
float exp10f(float x)
Returns 10𝑥.
✓ ✓
float exp2f( float x)
Returns 2𝑥.
✓ ✓
float expm1f(float x)
Returns 𝑙𝑛(𝑥− 1)
✓ ✓
float fabsf(float x)
Returns the absolute value ofx
✓ ✓
float fdimf(float x, float
y)
Returns the positive difference
between 𝑥 and 𝑦.
✓ ✓
float fdividef(float x,
float y)
Divide two floating point values.
✓ ✓
float floorf(float x)
Returns the largest integer less than
or equal to𝑥.
✓ ✓
float fmaf(float x, float
y, float z)
Returns 𝑥· 𝑦 + 𝑧 as a single
operation.
✓ ✓
float fmaxf(float x, float
y)
Determine the maximum numeric
value of𝑥 and 𝑦.
✓ ✓
float fminf(float x, float
y)
Determine the minimum numeric
value of𝑥 and 𝑦.
✓ ✓
continues on next page
17.9. Math functions 113
HIP Documentation, Release 6.3.42134
Table 17.1 – continued from previous page
float fmodf(float x, float
y)
Returns the floating-point
remainder of𝑥/𝑦.
✓ ✓
float modff(float x, float*
iptr)
Break down𝑥 into fractional and
integral parts.
✓
float frexpf(float x, int*
nptr)
Extract mantissa and exponent of𝑥.
✓
float hypotf(float x, float
y)
Returns the square root of the sum
of squares of𝑥 and 𝑦.
✓ ✓
int ilogbf(float x)
Returns the unbiased integer
exponent of𝑥.
✓ ✓
bool isfinite(float x)
Determine whether𝑥 is finite.
✓ ✓
bool isinf(float x)
Determine whether𝑥 is infinite.
✓ ✓
bool isnan(float x)
Determine whether𝑥 is aNAN.
✓ ✓
float j0f(float x)
Returns the value of the Bessel
function of the first kind of order 0
for 𝑥.
✓ ✓
continues on next page
114 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.1 – continued from previous page
float j1f(float x)
Returns the value of the Bessel
function of the first kind of order 1
for 𝑥.
✓ ✓
float jnf(int n, float x)
Returns the value of the Bessel
function of the first kind of order n
for 𝑥.
✓ ✓
float ldexpf(float x, int
exp)
Returns the natural logarithm of the
absolute value of the gamma
function of𝑥.
✓ ✓
float lgammaf(float x)
Returns the natural logarithm of the
absolute value of the gamma
function of𝑥.
✓
long int lrintf(float x)
Round 𝑥 to nearest integer value.
✓ ✓
long long int llrintf(float
x)
Round 𝑥 to nearest integer value.
✓ ✓
long int lroundf(float x)
Round to nearest integer value.
✓ ✓
long long int
llroundf(float x)
Round to nearest integer value.
✓ ✓
float log10f(float x)
Returns the base 10 logarithm of𝑥.
✓ ✓
continues on next page
17.9. Math functions 115
HIP Documentation, Release 6.3.42134
Table 17.1 – continued from previous page
float log1pf(float x)
Returns the natural logarithm of
𝑥 + 1.
✓ ✓
float log2f(float x)
Returns the base 2 logarithm of𝑥.
✓ ✓
float logf(float x)
Returns the natural logarithm of𝑥.
✓ ✓
float logbf(float x)
Returns the floating point
representation of the exponent of𝑥.
✓ ✓
float nanf(const char*
tagp)
Returns “Not a Number” value.
✓
float nearbyintf(float x)
Round 𝑥 to the nearest integer.
✓ ✓
float nextafterf(float x,
float y)
Returns next representable
single-precision floating-point value
after argument.
✓
float norm3df(float x,
float y, float z)
Returns the square root of the sum
of squares of𝑥, 𝑦 and 𝑧.
✓ ✓
float norm4df(float x,
float y, float z, float w)
Returns the square root of the sum
of squares of𝑥, 𝑦, 𝑧 and 𝑤.
✓ ✓
continues on next page
116 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.1 – continued from previous page
float normcdff(float y)
Returns the standard normal
cumulative distribution function.
✓ ✓
float normcdfinvf(float y)
Returns the inverse of the standard
normal cumulative distribution
function.
✓ ✓
float normf(int dim, const
float *a)
Returns the square root of the sum
of squares of any number of
coordinates.
✓ ✓
float powf(float x, float
y)
Returns 𝑥𝑦.
✓ ✓
float powif(float base, int
iexp)
Returns the value of first argument
to the power of second argument.
✓ ✓
float remainderf(float x,
float y)
Returns single-precision
floating-point remainder.
✓ ✓
float remquof(float x,
float y, int* quo)
Returns single-precision
floating-point remainder and part of
quotient.
✓ ✓
float roundf(float x)
Round to nearest integer value in
floating-point.
✓ ✓
continues on next page
17.9. Math functions 117
HIP Documentation, Release 6.3.42134
Table 17.1 – continued from previous page
float rcbrtf(float x)
Returns the reciprocal cube root
function.
✓ ✓
float rhypotf(float x,
float y)
Returns one over the square root of
the sum of squares of two
arguments.
✓ ✓
float rintf(float x)
Round input to nearest integer value
in floating-point.
✓ ✓
float rnorm3df(float x,
float y, float z)
Returns one over the square root of
the sum of squares of three
coordinates of the argument.
✓ ✓
float rnorm4df(float x,
float y, float z, float w)
Returns one over the square root of
the sum of squares of four
coordinates of the argument.
✓ ✓
float rnormf(int dim, const
float *a)
Returns the reciprocal of square
root of the sum of squares of any
number of coordinates.
✓ ✓
float scalblnf(float x,
long int n)
Scale 𝑥 by 2𝑛.
✓ ✓
float scalbnf(float x, int
n)
Scale 𝑥 by 2𝑛.
✓ ✓
continues on next page
118 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.1 – continued from previous page
bool signbit(float x)
Return the sign bit of𝑥.
✓ ✓
float sinf(float x)
Returns the sine of𝑥.
✓ ✓
float sinhf(float x)
Returns the hyperbolic sine of𝑥.
✓ ✓
float sinpif(float x)
Returns the hyperbolic sine of𝜋· 𝑥.
✓ ✓
void sincosf(float x, float
*sptr, float *cptr)
Returns the sine and cosine of𝑥.
✓ ✓
void sincospif(float x,
float *sptr, float *cptr)
Returns the sine and cosine of𝜋· 𝑥.
✓ ✓
float sqrtf(float x)
Returns the square root of𝑥.
✓ ✓
float rsqrtf(float x)
Returns the reciprocal of the square
root of𝑥.
✓
float tanf(float x)
Returns the tangent of𝑥.
✓ ✓
float tanhf(float x)
Returns the hyperbolic tangent of𝑥.
✓ ✓
float tgammaf(float x)
Returns the gamma function of𝑥.
✓ ✓
continues on next page
17.9. Math functions 119
HIP Documentation, Release 6.3.42134
Table 17.1 – continued from previous page
float truncf(float x)
Truncate 𝑥 to the integral part.
✓ ✓
float y0f(float x)
Returns the value of the Bessel
function of the second kind of order
0 for𝑥.
✓ ✓
float y1f(float x)
Returns the value of the Bessel
function of the second kind of order
1 for𝑥.
✓ ✓
float ynf(int n, float x)
Returns the value of the Bessel
function of the second kind of order
n for𝑥.
✓ ✓
17.9.2 Double precision mathematical functions
Following is the list of supported double precision mathematical functions.
Table 17.2: Double precision mathematical functions
Function Supported on Host Supported on Device
double abs(double x)
Returns the absolute value of𝑥
✓ ✓
double acos(double x)
Returns the arc cosine of𝑥.
✓ ✓
double acosh(double x)
Returns the nonnegative arc
hyperbolic cosine of𝑥.
✓ ✓
double asin(double x)
Returns the arc sine of𝑥.
✓ ✓
continues on next page
120 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.2 – continued from previous page
double asinh(double x)
Returnsthearchyperbolicsineof 𝑥.
✓ ✓
double atan(double x)
Returns the arc tangent of𝑥.
✓ ✓
double atan2(double x,
double y)
Returns the arc tangent of the ratio
of 𝑥 and 𝑦.
✓ ✓
double atanh(double x)
Returns the arc hyperbolic tangent
of 𝑥.
✓ ✓
double cbrt(double x)
Returns the cube root of𝑥.
✓ ✓
double ceil(double x)
Returns ceiling of𝑥.
✓ ✓
double copysign(double x,
double y)
Create value with given magnitude,
copying sign of second value.
✓ ✓
double cos(double x)
Returns the cosine of𝑥.
✓ ✓
double cosh(double x)
Returns the hyperbolic cosine of𝑥.
✓ ✓
double cospi(double x)
Returns the cosine of𝜋· 𝑥.
✓ ✓
continues on next page
17.9. Math functions 121
HIP Documentation, Release 6.3.42134
Table 17.2 – continued from previous page
double cyl_bessel_i0(double
x)
Returns the value of the regular
modified cylindrical Bessel
function of order 0 for𝑥.
double cyl_bessel_i1(double
x)
Returns the value of the regular
modified cylindrical Bessel
function of order 1 for𝑥.
double erf(double x)
Returns the error function of𝑥.
✓ ✓
double erfc(double x)
Returns the complementary error
function of𝑥.
✓ ✓
double erfcinv(double x)
Returns the inverse complementary
function of𝑥.
✓ ✓
double erfcx(double x)
Returns the scaled complementary
error function of𝑥.
✓ ✓
double erfinv(double x)
Returns the inverse error function
of 𝑥.
✓ ✓
double exp(double x)
Returns 𝑒𝑥.
✓ ✓
double exp10(double x)
Returns 10𝑥.
✓ ✓
double exp2( double x)
Returns 2𝑥.
✓ ✓
continues on next page
122 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.2 – continued from previous page
double expm1(double x)
Returns 𝑙𝑛(𝑥− 1)
✓ ✓
double fabs(double x)
Returns the absolute value ofx
✓ ✓
double fdim(double x,
double y)
Returns the positive difference
between 𝑥 and 𝑦.
✓ ✓
double floor(double x)
Returns the largest integer less than
or equal to𝑥.
✓ ✓
double fma(double x, double
y, double z)
Returns 𝑥· 𝑦 + 𝑧 as a single
operation.
✓ ✓
double fmax(double x,
double y)
Determine the maximum numeric
value of𝑥 and 𝑦.
✓ ✓
double fmin(double x,
double y)
Determine the minimum numeric
value of𝑥 and 𝑦.
✓ ✓
double fmod(double x,
double y)
Returns the floating-point
remainder of𝑥/𝑦.
✓ ✓
double modf(double x,
double* iptr)
Break down𝑥 into fractional and
integral parts.
✓
continues on next page
17.9. Math functions 123
HIP Documentation, Release 6.3.42134
Table 17.2 – continued from previous page
double frexp(double x, int*
nptr)
Extract mantissa and exponent of𝑥.
✓
double hypot(double x,
double y)
Returns the square root of the sum
of squares of𝑥 and 𝑦.
✓ ✓
int ilogb(double x)
Returns the unbiased integer
exponent of𝑥.
✓ ✓
bool isfinite(double x)
Determine whether𝑥 is finite.
✓ ✓
bool isin(double x)
Determine whether𝑥 is infinite.
✓ ✓
bool isnan(double x)
Determine whether𝑥 is aNAN.
✓ ✓
double j0(double x)
Returns the value of the Bessel
function of the first kind of order 0
for 𝑥.
✓ ✓
double j1(double x)
Returns the value of the Bessel
function of the first kind of order 1
for 𝑥.
✓ ✓
double jn(int n, double x)
Returns the value of the Bessel
function of the first kind of order n
for 𝑥.
✓ ✓
continues on next page
124 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.2 – continued from previous page
double ldexp(double x, int
exp)
Returns the natural logarithm of the
absolute value of the gamma
function of𝑥.
✓ ✓
double lgamma(double x)
Returns the natural logarithm of the
absolute value of the gamma
function of𝑥.
✓
long int lrint(double x)
Round 𝑥 to nearest integer value.
✓ ✓
long long int llrint(double
x)
Round 𝑥 to nearest integer value.
✓ ✓
long int lround(double x)
Round to nearest integer value.
✓ ✓
long long int
llround(double x)
Round to nearest integer value.
✓ ✓
double log10(double x)
Returns the base 10 logarithm of𝑥.
✓ ✓
double log1p(double x)
Returns the natural logarithm of
𝑥 + 1.
✓ ✓
double log2(double x)
Returns the base 2 logarithm of𝑥.
✓ ✓
double log(double x)
Returns the natural logarithm of𝑥.
✓ ✓
continues on next page
17.9. Math functions 125
HIP Documentation, Release 6.3.42134
Table 17.2 – continued from previous page
double logb(double x)
Returns the floating point
representation of the exponent of𝑥.
✓ ✓
double nan(const char*
tagp)
Returns “Not a Number” value.
✓
double nearbyint(double x)
Round 𝑥 to the nearest integer.
✓ ✓
double nextafter(double x,
double y)
Returns next representable
double-precision floating-point
value after argument.
✓ ✓
double norm3d(double x,
double y, double z)
Returns the square root of the sum
of squares of𝑥, 𝑦 and 𝑧.
✓ ✓
double norm4d(double x,
double y, double z, double
w)
Returns the square root of the sum
of squares of𝑥, 𝑦, 𝑧 and 𝑤.
✓ ✓
double normcdf(double y)
Returns the standard normal
cumulative distribution function.
✓ ✓
double normcdfinv(double y)
Returns the inverse of the standard
normal cumulative distribution
function.
✓ ✓
continues on next page
126 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.2 – continued from previous page
double norm(int dim, const
double *a)
Returns the square root of the sum
of squares of any number of
coordinates.
✓ ✓
double pow(double x, double
y)
Returns 𝑥𝑦.
✓ ✓
double powi(double base,
int iexp)
Returns the value of first argument
to the power of second argument.
✓ ✓
double remainder(double x,
double y)
Returns double-precision
floating-point remainder.
✓ ✓
double remquo(double x,
double y, int* quo)
Returns double-precision
floating-point remainder and part of
quotient.
✓
double round(double x)
Round to nearest integer value in
floating-point.
✓ ✓
double rcbrt(double x)
Returns the reciprocal cube root
function.
✓ ✓
double rhypot(double x,
double y)
Returns one over the square root of
the sum of squares of two
arguments.
✓ ✓
continues on next page
17.9. Math functions 127
HIP Documentation, Release 6.3.42134
Table 17.2 – continued from previous page
double rint(double x)
Round input to nearest integer value
in floating-point.
✓ ✓
double rnorm3d(double x,
double y, double z)
Returns one over the square root of
the sum of squares of three
coordinates of the argument.
✓ ✓
double rnorm4d(double x,
double y, double z, double
w)
Returns one over the square root of
the sum of squares of four
coordinates of the argument.
✓ ✓
double rnorm(int dim, const
double *a)
Returns the reciprocal of square
root of the sum of squares of any
number of coordinates.
✓ ✓
double scalbln(double x,
long int n)
Scale 𝑥 by 2𝑛.
✓ ✓
double scalbn(double x, int
n)
Scale 𝑥 by 2𝑛.
✓ ✓
bool signbit(double x)
Return the sign bit of𝑥.
✓ ✓
double sin(double x)
Returns the sine of𝑥.
✓ ✓
double sinh(double x)
Returns the hyperbolic sine of𝑥.
✓ ✓
continues on next page
128 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.2 – continued from previous page
double sinpi(double x)
Returns the hyperbolic sine of𝜋· 𝑥.
✓ ✓
void sincos(double x,
double *sptr, double *cptr)
Returns the sine and cosine of𝑥.
✓ ✓
void sincospi(double x,
double *sptr, double *cptr)
Returns the sine and cosine of𝜋· 𝑥.
✓ ✓
double sqrt(double x)
Returns the square root of𝑥.
✓ ✓
double rsqrt(double x)
Returns the reciprocal of the square
root of𝑥.
✓
double tan(double x)
Returns the tangent of𝑥.
✓ ✓
double tanh(double x)
Returns the hyperbolic tangent of𝑥.
✓ ✓
double tgamma(double x)
Returns the gamma function of𝑥.
✓ ✓
double trunc(double x)
Truncate 𝑥 to the integral part.
✓ ✓
double y0(double x)
Returns the value of the Bessel
function of the second kind of order
0 for𝑥.
✓ ✓
continues on next page
17.9. Math functions 129
HIP Documentation, Release 6.3.42134
Table 17.2 – continued from previous page
double y1(double x)
Returns the value of the Bessel
function of the second kind of order
1 for𝑥.
✓ ✓
double yn(int n, double x)
Returns the value of the Bessel
function of the second kind of order
n for𝑥.
✓ ✓
17.9.3 Integer intrinsics
Following is the list of supported integer intrinsics. Note that intrinsics are supported on device only.
130 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.3: Integer intrinsics mathematical functions
Function
unsigned int __brev(unsigned int x)
Reverse the bit order of a 32 bit unsigned integer.
unsigned long long int __brevll(unsigned long long int x)
Reverse the bit order of a 64 bit unsigned integer.
unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int z)
Return selected bytes from two 32-bit unsigned integers.
unsigned int __clz(int x)
Return the number of consecutive high-order zero bits in 32 bit integer.
unsigned int __clzll(long long int x)
Return the number of consecutive high-order zero bits in 64 bit integer.
unsigned int __ffs(int x)
Find the position of least significant bit set to 1 in a 32 bit integer.
unsigned int __ffsll(long long int x)
Find the position of least significant bit set to 1 in a 64 bit signed integer.
unsigned int __fns32(unsigned long long mask, unsigned int base, int offset)
Find the position of the n-th set to 1 bit in a 32-bit integer.
unsigned int __fns64(unsigned long long int mask, unsigned int base, int offset)
Find the position of the n-th set to 1 bit in a 64-bit integer.
unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift)
Concatenate ℎ𝑖 and 𝑙𝑜, shift left by shift & 31 bits, return the most significant 32 bits.
unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift)
Concatenate ℎ𝑖 and 𝑙𝑜, shift left by min(shift, 32) bits, return the most significant 32 bits.
unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)
Concatenate ℎ𝑖 and 𝑙𝑜, shift right by shift & 31 bits, return the least significant 32 bits.
unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift)
Concatenate ℎ𝑖 and 𝑙𝑜, shift right by min(shift, 32) bits, return the least significant 32 bits.
unsigned int __hadd(int x, int y)
Compute average of signed input arguments, avoiding overflow in the intermediate sum.
unsigned int __rhadd(int x, int y)
Compute rounded average of signed input arguments, avoiding overflow in the intermediate sum.
unsigned int __uhadd(int x, int y)
Compute average of unsigned input arguments, avoiding overflow in the intermediate sum.
unsigned int __urhadd (unsigned int x, unsigned int y)
Compute rounded average of unsigned input arguments, avoiding overflow in the intermediate sum.
int __sad(int x, int y, int z)
Returns|𝑥− 𝑦| + 𝑧, the sum of absolute difference.
unsigned int __usad(unsigned int x, unsigned int y, unsigned int z)
Returns|𝑥− 𝑦| + 𝑧, the sum of absolute difference.
unsigned int __popc(unsigned int x)
Count the number of bits that are set to 1 in a 32 bit integer.
unsigned int __popcll(unsigned long long int x)
Count the number of bits that are set to 1 in a 64 bit integer.
int __mul24(int x, int y)
Multiply two 24bit integers.
unsigned int __umul24(unsigned int x, unsigned int y)
Multiply two 24bit unsigned integers.
int __mulhi(int x, int y)
Returns the most significant 32 bits of the product of the two 32-bit integers.
unsigned int __umulhi(unsigned int x, unsigned int y)
Returns the most significant 32 bits of the product of the two 32-bit unsigned integers.
long long int __mul64hi(long long int x, long long int y)
Returns the most significant 64 bits of the product of the two 64-bit integers.
unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y)
Returns the most significant 64 bits of the product of the two 64 unsigned bit integers.
17.9. Math functions 131
HIP Documentation, Release 6.3.42134
The HIP-Clang implementation of__ffs() and __ffsll() contains code to add a constant +1 to produce theffs
result format. For the cases where this overhead is not acceptable and programmer is willing to specialize for the
platform,HIP-Clangprovides __lastbit_u32_u32(unsigned int input)and__lastbit_u32_u64(unsigned
long long int input). The index returned by__lastbit_instructions starts at -1, while forffsthe index starts
at 0.
17.9.4 Floating-point Intrinsics
Following is the list of supported floating-point intrinsics. Note that intrinsics are supported on device only.
Note: Only the nearest even rounding mode supported on AMD GPUs by defaults. The_rz, _ru and _rd suffixed
intrinsic functions are existing in HIP AMD backend, if theOCML_BASIC_ROUNDED_OPERATIONSmacro is defined.
132 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.4: Single precision intrinsics mathematical functions
Function
float __cosf(float x)
Returns the fast approximate cosine of𝑥.
float __exp10f(float x)
Returns the fast approximate for 10x.
float __expf(float x)
Returns the fast approximate for ex.
float __fadd_rn(float x, float y)
Add two floating-point values in round-to-nearest-even mode.
float __fdiv_rn(float x, float y)
Divide two floating point values in round-to-nearest-even mode.
float __fmaf_rn(float x, float y, float z)
Returnsx × y + zas a single operation in round-to-nearest-even mode.
float __fmul_rn(float x, float y)
Multiply two floating-point values in round-to-nearest-even mode.
float __frcp_rn(float x, float y)
Returns1 / x in round-to-nearest-even mode.
float __frsqrt_rn(float x)
Returns1 / x in round-to-nearest-even mode.
float __fsqrt_rn(float x)
Returnsxin round-to-nearest-even mode.
float __fsub_rn(float x, float y)
Subtract two floating-point values in round-to-nearest-even mode.
float __log10f(float x)
Returns the fast approximate for base 10 logarithm of𝑥.
float __log2f(float x)
Returns the fast approximate for base 2 logarithm of𝑥.
float __logf(float x)
Returns the fast approximate for natural logarithm of𝑥.
float __powf(float x, float y)
Returns the fast approximate of xy.
float __saturatef(float x)
Clamp 𝑥 to [+0.0, 1.0].
float __sincosf(float x, float* sinptr, float* cosptr)
Returns the fast approximate of sine and cosine of𝑥.
float __sinf(float x)
Returns the fast approximate sine of𝑥.
float __tanf(float x)
Returns the fast approximate tangent of𝑥.
17.9. Math functions 133
HIP Documentation, Release 6.3.42134
Table 17.5: Double precision intrinsics mathematical functions
Function
double __dadd_rn(double x, double y)
Add two floating-point values in round-to-nearest-even mode.
double __ddiv_rn(double x, double y)
Divide two floating-point values in round-to-nearest-even mode.
double __dmul_rn(double x, double y)
Multiply two floating-point values in round-to-nearest-even mode.
double __drcp_rn(double x, double y)
Returns1 / x in round-to-nearest-even mode.
double __dsqrt_rn(double x)
Returnsxin round-to-nearest-even mode.
double __dsub_rn(double x, double y)
Subtract two floating-point values in round-to-nearest-even mode.
double __fma_rn(double x, double y, double z)
Returnsx × y + zas a single operation in round-to-nearest-even mode.
17.10 Texture functions
Thesupportedtexturefunctionsarelistedin texture_fetch_functions.handtexture_indirect_functions.
hheader files in the HIP-AMD backend repository.
Texture functions are not supported on some devices. To determine if texture functions are supported on your device,
use Macro __HIP_NO_IMAGE_SUPPORT == 1. You can query the attributehipDeviceAttributeImageSupport
to check if texture functions are supported in the host runtime code.
134 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
17.11 Surface functions
Surface functions are not supported.
17.12 Timer functions
To read a high-resolution timer from the device, HIP provides the following built-in functions:

## • Returning the incremental counter value for every clock cycle on a device

clock_t clock()
long long int clock64()
The difference between the values that are returned represents the cycles used.

## • Returning the wall clock count at a constant frequency on the device

long long int wall_clock64()
This can be queried using the HIP API with thehipDeviceAttributeWallClockRateattribute of the device

## in HIP application code. For example

int wallClkRate = 0; //in kilohertz
HIPCHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate,␣
˓→deviceId));
Where hipDeviceAttributeWallClockRate is a device attribute. Note that wall clock frequency is a per-
device attribute.
17.13 Atomic functions
Atomic functions are run as read-modify-write (RMW) operations that reside in global or shared memory. No other
deviceorthreadcanobserveormodifythememorylocationduringanatomicoperation. Ifmultipleinstructionsfrom
different devices or threads target the same memory location, the instructions are serialized in an undefined order.
To support system scope atomic operations, you can use the HIP APIs that contain the_system suffix. For example:
• atomicAnd: This function is atomic and coherent within the GPU device running the function
• atomicAnd_system: ThisfunctionextendstheatomicoperationfromtheGPUdevicetootherCPUsandGPU
devices in the system.
HIP supports the following atomic operations.
Table 17.6: Atomic operations
Function Supported in HIP Supported in CUDA
int atomicAdd(int* address, int val) ✓ ✓
int atomicAdd_system(int* address, int val) ✓ ✓
unsigned int atomicAdd(unsigned int* address,unsigned int val) ✓ ✓
unsigned int atomicAdd_system(unsigned int* address, unsigned int val) ✓ ✓
unsigned long long atomicAdd(unsigned long long* address,unsigned long long val) ✓ ✓
unsigned long long atomicAdd_system(unsigned long long* address, unsigned long long val) ✓ ✓
continues on next page
17.11. Surface functions 135
HIP Documentation, Release 6.3.42134
Table 17.6 – continued from previous page
float atomicAdd(float* address, float val) ✓ ✓
float atomicAdd_system(float* address, float val) ✓ ✓
double atomicAdd(double* address, double val) ✓ ✓
double atomicAdd_system(double* address, double val) ✓ ✓
float unsafeAtomicAdd(float* address, float val) ✓
float safeAtomicAdd(float* address, float val) ✓
double unsafeAtomicAdd(double* address, double val) ✓
double safeAtomicAdd(double* address, double val) ✓
int atomicSub(int* address, int val) ✓ ✓
int atomicSub_system(int* address, int val) ✓ ✓
unsigned int atomicSub(unsigned int* address,unsigned int val) ✓ ✓
unsigned int atomicSub_system(unsigned int* address, unsigned int val) ✓ ✓
int atomicExch(int* address, int val) ✓ ✓
int atomicExch_system(int* address, int val) ✓ ✓
unsigned int atomicExch(unsigned int* address,unsigned int val) ✓ ✓
unsigned int atomicExch_system(unsigned int* address, unsigned int val) ✓ ✓
unsigned long long atomicExch(unsigned long long int* address,unsigned long long int val) ✓ ✓
unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val) ✓ ✓
unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val) ✓ ✓
float atomicExch(float* address, float val) ✓ ✓
int atomicMin(int* address, int val) ✓ ✓
int atomicMin_system(int* address, int val) ✓ ✓
unsigned int atomicMin(unsigned int* address,unsigned int val) ✓ ✓
unsigned int atomicMin_system(unsigned int* address, unsigned int val) ✓ ✓
unsigned long long atomicMin(unsigned long long* address,unsigned long long val) ✓ ✓
int atomicMax(int* address, int val) ✓ ✓
int atomicMax_system(int* address, int val) ✓ ✓
unsigned int atomicMax(unsigned int* address,unsigned int val) ✓ ✓
unsigned int atomicMax_system(unsigned int* address, unsigned int val) ✓ ✓
unsigned long long atomicMax(unsigned long long* address,unsigned long long val) ✓ ✓
unsigned int atomicInc(unsigned int* address) ✓
unsigned int atomicDec(unsigned int* address) ✓
int atomicCAS(int* address, int compare, int val) ✓ ✓
int atomicCAS_system(int* address, int compare, int val) ✓ ✓
unsigned int atomicCAS(unsigned int* address,unsigned int compare,unsigned int val) ✓ ✓
unsigned int atomicCAS_system(unsigned int* address, unsigned int compare, unsigned int val) ✓ ✓
unsigned long long atomicCAS(unsigned long long* address,unsigned long long compare,unsigned long long val) ✓ ✓
unsigned long long atomicCAS_system(unsigned long long* address, unsigned long long compare, unsigned long long val) ✓ ✓
int atomicAnd(int* address, int val) ✓ ✓
int atomicAnd_system(int* address, int val) ✓ ✓
unsigned int atomicAnd(unsigned int* address,unsigned int val) ✓ ✓
unsigned int atomicAnd_system(unsigned int* address, unsigned int val) ✓ ✓
unsigned long long atomicAnd(unsigned long long* address,unsigned long long val) ✓ ✓
unsigned long long atomicAnd_system(unsigned long long* address, unsigned long long val) ✓ ✓
int atomicOr(int* address, int val) ✓ ✓
int atomicOr_system(int* address, int val) ✓ ✓
unsigned int atomicOr(unsigned int* address,unsigned int val) ✓ ✓
unsigned int atomicOr_system(unsigned int* address, unsigned int val) ✓ ✓
unsigned int atomicOr_system(unsigned int* address, unsigned int val) ✓ ✓
unsigned long long atomicOr(unsigned long long int* address,unsigned long long val) ✓ ✓
unsigned long long atomicOr_system(unsigned long long* address, unsigned long long val) ✓ ✓
int atomicXor(int* address, int val) ✓ ✓
continues on next page
136 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.6 – continued from previous page
int atomicXor_system(int* address, int val) ✓ ✓
unsigned int atomicXor(unsigned int* address,unsigned int val) ✓ ✓
unsigned int atomicXor_system(unsigned int* address, unsigned int val) ✓ ✓
unsigned long long atomicXor(unsigned long long* address,unsigned long long val) ✓ ✓
unsigned long long atomicXor_system(unsigned long long* address, unsigned long long val) ✓ ✓
17.13.1 Unsafe floating-point atomic RMW operations
Some HIP devices supportfast atomic RMW operations on floating-point values. For example,atomicAddon single-
or double-precision floating-point values may generate a hardware RMW instruction that is faster than emulating the
atomic operation using an atomic compare-and-swap (CAS) loop.
On some devices, fast atomic RMW instructions can produce results that differ from the same functions implemented
with atomic CAS loops. For example, some devices will use different rounding or denormal modes, and some devices
produce incorrect answers if fast floating-point atomic RMW instructions target fine-grained memory allocations.
The HIP-Clang compiler offers a compile-time option, so you can choose fast–but potentially unsafe–atomic instruc-
tions for your code. On devices that support these instructions, you can include the-munsafe-fp-atomics option.
This flag indicates to the compiler that all floating-point atomic function calls are allowed to use an unsafe version, if
oneexists. Forexample,onsomedevices,thisflagindicatestothecompilerthatnofloating-point atomicAddfunction
can target fine-grained memory.
If you want to avoid using unsafe use a floating-point atomic RMW operations, you can use the
-mno-unsafe-fp-atomics option. Note that the compiler default is to not produce unsafe floating-point
atomic RMW instructions, so the-mno-unsafe-fp-atomics option is not necessarily required. However, passing
this option to the compiler is good practice.
When you pass-munsafe-fp-atomics or -mno-unsafe-fp-atomics to the compiler’s command line, the option
is applied globally for the entire compilation. Note that if some of the atomic RMW function calls cannot safely use
the faster floating-point atomic RMW instructions, you must use-mno-unsafe-fp-atomics in order to ensure that
your atomic RMW function calls produce correct results.
HIP has four extra functions that you can use to more precisely control which floating-point atomic RMW functions

## produce unsafe atomic RMW instructions

• float unsafeAtomicAdd(float* address, float val)
• double unsafeAtomicAdd(double* address, double val)(AlwaysproducesfastatomicRMWinstruc-
tions on devices that have them, even when-mno-unsafe-fp-atomics is used)
• float safeAtomicAdd(float* address, float val)
• double safeAtomicAdd(double* address, double val) (Always produces safe atomic RMW opera-
tions, even when-munsafe-fp-atomics is used)
17.14 Warp cross-lane functions
Threads in a warp are referred to aslanes and are numbered from0 to warpSize - 1. Warp cross-lane functions
operate across all lanes in a warp. The hardware guarantees that all warp lanes will execute in lockstep, so additional
synchronization is unnecessary, and the instructions use no shared memory.
Note that NVIDIA and AMD devices have different warp sizes. You can usewarpSizebuilt-ins in you portable code
to query the warp size.
17.14. Warp cross-lane functions 137
HIP Documentation, Release 6.3.42134
Tip: Be sure to review HIP code generated from the CUDA path to ensure that it doesn’t assume awaveSize of
32. “Wave-aware” code that assumes awaveSize of 32 can run on a wave-64 machine, but it only utilizes half of the
machine’s resources.
To get the default warp size of a GPU device, usehipGetDeviceProperties in you host functions.
cudaDeviceProp props;
cudaGetDeviceProperties(&props, deviceID);
int w = props.warpSize;
// implement portable algorithm based on w (rather than assume 32 or 64)
Only usewarpSize built-ins in device functions, and don’t assumewarpSize to be a compile-time constant.

```cpp

Notethatassemblykernelsmaybebuiltforawarpsizethatisdifferentfromthedefault. Allmaskvalueseitherreturned

```

or accepted by these builtins are 64-bit unsigned integer values, even when compiled for a wave-32 device, where all
the higher bits are unused. CUDA code ported to HIP requires changes to ensure that the correct type is used.
Note that the__sync variants are made available in ROCm 6.2, but disabled by default to help with the transition to
64-bit masks. They can be enabled by setting the preprocessor macroHIP_ENABLE_WARP_SYNC_BUILTINS. These
builtins will be enabled unconditionally in the next ROCm release. Wherever possible, the implementation includes a
static assert to check that the program source uses the correct type for the mask.
17.14.1 Warp vote and ballot functions
int __all(int predicate)
int __any(int predicate)
unsigned long long __ballot(int predicate)
unsigned long long __activemask()
int __all_sync(unsigned long long mask, int predicate)
int __any_sync(unsigned long long mask, int predicate)
unsigned long long __ballot_sync(unsigned long long mask, int predicate)
You can use__anyand __all to get a summary view of the predicates evaluated by the participating lanes.
• __any(): Returns 1 if the predicate is non-zero for any participating lane, otherwise it returns 0.
• __all(): Returns 1 if the predicate is non-zero for all participating lanes, otherwise it returns 0.
To determine if the target platform supports the any/all instruction, you can use thehasWarpVote device property or
theHIP_ARCH_HAS_WARP_VOTE compiler definition.
__ballotreturnsabitmaskcontainingthe1-bitpredicatevaluefromeachlane. Thenthbitoftheresultcontainsthe
1 bit contributed by the nth warp lane.
__activemask() returns a bit mask of currently active warp lanes. The nth bit of the result is 1 if the nth warp lane
is active.
Notethatthe __ballotand__activemaskbuiltinsinHIPhavea64-bitreturnvalue(unlikethe32-bitvaluereturned
by the CUDA builtins). Code ported from CUDA should be adapted to support the larger warp sizes that the HIP
version requires.
Applications can test whether the target platform supports the__ballot or __activemask instructions using the
hasWarpBallot device property in host code or theHIP_ARCH_HAS_WARP_BALLOT macro defined by the compiler
for device code.
138 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
The _sync variants require a 64-bit unsigned integer mask argument that specifies the lanes in the warp that will
participateincross-lanecommunicationwiththecallinglane. Eachparticipatingthreadmusthaveitsownbitsetinits
maskargument, andallactivethreadsspecifiedinanymaskargumentmustexecutethesamecallwiththesamemask,
otherwise the result is undefined.
17.14.2 Warp match functions
unsigned long long __match_any(T value)
unsigned long long __match_all(T value, int *pred)
unsigned long long __match_any_sync(unsigned long long mask, T value)
unsigned long long __match_all_sync(unsigned long long mask, T value, int *pred)
Tcan be a 32-bit integer type, 64-bit integer type or a single precision or double precision floating point type.
__match_any returns a bit mask containing a 1-bit for every participating lane if and only if that lane has the same
value invalueas the current lane, and a 0-bit for all other lanes.
__match_all returns a bit mask containing a 1-bit for every participating lane if and only if they all have the same
value invalue as the current lane, and a 0-bit for all other lanes. The predicatepred is set to true if and only if all
participating threads have the same value invalue.
The _sync variants require a 64-bit unsigned integer mask argument that specifies the lanes in the warp that will
participateincross-lanecommunicationwiththecallinglane. Eachparticipatingthreadmusthaveitsownbitsetinits
maskargument, andallactivethreadsspecifiedinanymaskargumentmustexecutethesamecallwiththesamemask,
otherwise the result is undefined.
17.14.3 Warp shuffle functions
The default width iswarpSize(see Warp cross-lane functions). Half-float shuffles are not supported.
T __shfl (T var, int srcLane, int width=warpSize);
T __shfl_up (T var, unsigned int delta, int width=warpSize);
T __shfl_down (T var, unsigned int delta, int width=warpSize);
T __shfl_xor (T var, int laneMask, int width=warpSize);
T __shfl_sync ( unsigned long long mask, T var, int srcLane, int width=warpSize);
T __shfl_up_sync ( unsigned long long mask, T var, unsigned int delta, int␣
˓→width=warpSize);
T __shfl_down_sync (unsigned long long mask, T var, unsigned int delta, int␣
˓→width=warpSize);
T __shfl_xor_sync ( unsigned long long mask, T var, int laneMask, int width=warpSize);
Tcan be a 32-bit integer type, 64-bit integer type or a single precision or double precision floating point type.
The _sync variants require a 64-bit unsigned integer mask argument that specifies the lanes in the warp that will
participateincross-lanecommunicationwiththecallinglane. Eachparticipatingthreadmusthaveitsownbitsetinits
maskargument, andallactivethreadsspecifiedinanymaskargumentmustexecutethesamecallwiththesamemask,
otherwise the result is undefined.
17.14. Warp cross-lane functions 139
HIP Documentation, Release 6.3.42134
17.15 Cooperative groups functions
You can use cooperative groups to synchronize groups of threads. Cooperative groups also provide a way of commu-
nicating between groups of threads at a granularity that is different from the block.

```cpp

HIP supports the following kernel language cooperative groups types and functions:

```

Table 17.7: Cooperative groups functions
Function Supported in HIP Supported in CUDA
void thread_group.sync(); ✓ ✓
unsigned thread_group.size(); ✓ ✓
unsigned thread_group.thread_rank() ✓ ✓
bool thread_group.is_valid(); ✓ ✓
grid_group this_grid() ✓ ✓
void grid_group.sync() ✓ ✓
unsigned grid_group.size() ✓ ✓
unsigned grid_group.thread_rank() ✓ ✓
bool grid_group.is_valid() ✓ ✓
multi_grid_group this_multi_grid() ✓ ✓
void multi_grid_group.sync() ✓ ✓
unsigned multi_grid_group.size() ✓ ✓
unsigned multi_grid_group.thread_rank() ✓ ✓
bool multi_grid_group.is_valid() ✓ ✓
unsigned multi_grid_group.num_grids() ✓ ✓
unsigned multi_grid_group.grid_rank() ✓ ✓
thread_block this_thread_block() ✓ ✓
multi_grid_group this_multi_grid() ✓ ✓
void multi_grid_group.sync() ✓ ✓
void thread_block.sync() ✓ ✓
unsigned thread_block.size() ✓ ✓
unsigned thread_block.thread_rank() ✓ ✓
bool thread_block.is_valid() ✓ ✓

```cpp

dim3 thread_block.group_index() ✓ ✓
dim3 thread_block.thread_index() ✓ ✓

```

17.16 Warp matrix functions
Warpmatrixfunctionsallowawarptocooperativelyoperateonsmallmatricesthathaveelementsspreadoverlanesin
an unspecified manner.

```cpp

HIP does not support kernel language warp matrix types or functions.

```

140 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Table 17.8: Warp matrix functions
Function Sup-
ported in

## HIP

Supported
in CUDA
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned
lda)
✓
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned
lda, layout_t layout)
✓
void store_matrix_sync(T* mptr, fragment<...> &a, unsigned lda,
layout_t layout)
✓
void fill_fragment(fragment<...> &a, const T &value) ✓
void mma_sync(fragment<...> &d, const fragment<...> &a, const
fragment<...> &b, const fragment<...> &c , bool sat)
✓
17.17 Independent thread scheduling
Certain architectures that support CUDA allow threads to progress independently of each other. This independent
thread scheduling makes intra-warp synchronization possible.
HIP does not support this type of scheduling.
17.18 Profiler Counter Function
The CUDA__prof_trigger() instruction is not supported.
17.19 Assert
The assert function is supported in HIP. Assert function is used for debugging purpose, when the input expression
equals to zero, the execution will be stopped.
void assert(int input)
There are two kinds of implementations for assert functions depending on the use sceneries, - One is for the host
version of assert, which is defined inassert.h, - Another is the device version of assert, which is implemented in
hip/hip_runtime.h. Users need to includeassert.h to useassert. For assert to work in both device and host
functions, users need to include"hip/hip_runtime.h".
HIPprovidesthefunction abort()whichcanbeusedtoterminatetheapplicationwhenterminalfailuresaredetected.
It is implemented using the__builtin_trap()function.
This function produces a similar effect of usingasm("trap") in the CUDA code.
Note: In HIP, the function terminates the entire application, while in CUDA,asm("trap") only terminates the
dispatch and the application continues to run.
17.17. Independent thread scheduling 141
HIP Documentation, Release 6.3.42134
17.20 printf

```cpp

printf function is supported in HIP. The following is a simple example to print information in the kernel.
#include <hip/hip_runtime.h>
__global__ void run_printf() { printf("Hello World\n"); }

```

int main() {

```cpp

run_printf<<<dim3(1), dim3(1), 0, 0>>>();

```

}
17.21 Device-Side Dynamic Global Memory Allocation
Device-sidedynamicglobalmemoryallocationisunderdevelopment. HIPnowincludesapreliminaryimplementation
of malloc and free that can be called from device functions.
17.22 __launch_bounds__
GPU multiprocessors have a fixed pool of resources (primarily registers and shared memory) which are shared by the

```cpp

actively running warps. Using more resources can increase IPC of the kernel but reduces the resources available for

```

otherwarpsandlimitsthenumberofwarpsthatcanbesimultaneouslyrunning. ThusGPUshaveacomplexrelationship
between resource usage and performance.
__launch_bounds__ allows the application to provide usage hints that influence the resources (primarily registers)

```cpp

used by the generated code. It is a function attribute that must be attached to a __global__ function:
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)

```

MyKernel(hipGridLaunch lp, ...)
...
__launch_bounds__ supports two parameters: - MAX_THREADS_PER_BLOCK - The programmers guarantees

```cpp

that kernel will be launched with threads less than MAX_THREADS_PER_BLOCK. (On NVCC this maps to the.

```

maxntid PTX directive). If no launch_bounds is specified, MAX_THREADS_PER_BLOCK is the maximum block
size supported by the device (typically 1024 or larger). Specifying MAX_THREADS_PER_BLOCK less than the
maximum effectively allows the compiler to use more resources than a default unconstrained compilation that sup-
ports all possible block sizes at launch time. The threads-per-block is the product of (blockDim.x * blockDim.
y * blockDim.z). - MIN_WARPS_PER_EXECUTION_UNIT - directs the compiler to minimize resource us-
age so that the requested number of warps can be simultaneously active on a multi-processor. Since active warps
compete for the same fixed pool of resources, the compiler must reduce resources required by each warp(primarily
registers). MIN_WARPS_PER_EXECUTION_UNIT is optional and defaults to 1 if not specified. Specifying a
MIN_WARPS_PER_EXECUTION_UNITgreaterthanthedefault1effectivelyconstrainsthecompiler’sresourceus-
age.

```cpp

When launch kernel with HIP APIs, for example,hipModuleLaunchKernel(), HIP will do validation to make sure
input kernel dimension size is not larger than specified launch_bounds. In case exceeded, HIP would return launch

```

failure, if AMD_LOG_LEVEL is set with proper value (for details, please refer todocs/markdown/hip_logging.

```cpp

md), detail information will be shown in the error log message, including launch parameters of kernel dim size, launch
bounds, and the name of the faulting kernel. It’s helpful to figure out which is the faulting kernel, besides, the kernel

```

dim size and launch bounds values will also assist in debugging such failures.
142 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
17.22.1 Compiler Impact
Thecompilerusestheseparametersasfollows: -Thecompilerusesthehintsonlytomanageregisterusage,anddoesnot

```cpp

automaticallyreducesharedmemoryorotherresources. -Compilationfailsifcompilercannotgenerateakernelwhich

```

meetstherequirementsofthespecifiedlaunchbounds. -FromMAX_THREADS_PER_BLOCK,thecompilerderives
themaximumnumberofwarps/blockthatcanbeusedatlaunchtime. ValuesofMAX_THREADS_PER_BLOCKless
than the default allows the compiler to use a larger pool of registers : each warp uses registers, and this hint constrains
the launch to a warps/block size which is less than maximum. - From MIN_WARPS_PER_EXECUTION_UNIT, the

```cpp

compiler derives a maximum number of registers that can be used by the kernel (to meet the required #simultaneous
active blocks). If MIN_WARPS_PER_EXECUTION_UNIT is 1, then the kernel can use all registers supported by
the multiprocessor. - The compiler ensures that the registers used in the kernel is less than both allowed maximums,

```

typically by spilling registers (to shared or global memory), or by using more instructions. - The compiler may use
heuristics to increase register usage, or may simply be able to avoid spilling. The MAX_THREADS_PER_BLOCK
is particularly useful in this cases, since it allows the compiler to use more registers and avoid situations where the
compiler constrains the register usage (potentially spilling) to meet the requirements of a large block size that is never
used at launch time.
17.22.2 CU and EU Definitions
A compute unit (CU) is responsible for executing the waves of a work-group. It is composed of one or more execution
units(EU)whichareresponsibleforexecutingwaves. AnEUcanhaveenoughresourcestomaintainthestateofmore
thanoneexecutingwave. ThisallowsanEUtohidelatencybyswitchingbetweenwavesinasimilarwaytosymmetric
multithreading on a CPU. In order to allow the state for multiple waves to fit on an EU, the resources used by a single
wavehavetobelimited. Limitingsuchresourcescanallowgreaterlatencyhiding,butcanresultinhavingtospillsome
register state to memory. This attribute allows an advanced developer to tune the number of waves that are capable of
fittingwithintheresourcesofanEU.Itcanbeusedtoensureatleastacertainnumberwillfittohelphidelatency, and
can also be used to ensure no more than a certain number will fit to limit cache thrashing.
17.22.3 Porting from CUDA__launch_bounds

## CUDA defines a__launch_boundswhich is also designed to control occupancy

__launch_bounds(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)
• The second parameter __launch_bounds parameters must be converted to the format used
__hip_launch_bounds, which uses warps and execution-units rather than blocks and multi-processors
(this conversion is performed automatically by HIPIFY tools).

## MIN_WARPS_PER_EXECUTION_UNIT = (MIN_BLOCKS_PER_MULTIPROCESSOR * MAX_THREADS_PER_BLOCK) /␣

˓→32
The key differences in the interface are: - Warps (rather than blocks): The developer is trying to tell the compiler
to control resource utilization to guarantee some amount of active Warps/EU for latency hiding. Specifying active
warps in terms of blocks appears to hide the micro-architectural details of the warp size, but makes the interface
more confusing since the developer ultimately needs to compute the number of warps to obtain the desired level of
control. -ExecutionUnits(ratherthanmultiprocessor): Theuseofexecutionunitsratherthanmultiprocessorsprovides
support for architectures with multiple execution units/multi-processor. For example, the AMD GCN architecture
has 4 execution units per multiprocessor. ThehipDeviceProps has a fieldexecutionUnitsPerMultiprocessor.
Platform-specific coding techniques such as#ifdef can be used to specify different launch_bounds for NVCC and
HIP-Clang platforms, if desired.
17.22. __launch_bounds__ 143
HIP Documentation, Release 6.3.42134
17.22.4 maxregcount
Unlike NVCC, HIP-Clang does not support the--maxregcount option. Instead, users are encouraged to use the
hip_launch_bounds directive since the parameters are more intuitive and portable than micro-architecture details like

```cpp

registers, and also the directive allows per-kernel control rather than an entire file. hip_launch_bounds works on both

```

HIP-Clang and NVCC targets.
17.23 Asynchronous Functions
17.23.1 Memory stream
typedef void (*hipStreamCallback_t)(hipStream_t stream, hipError_t status, void *userData)
Stream CallBack struct
hipError_t hipStreamCreate(hipStream_t *stream)
Create an asynchronous stream.
Create a new asynchronous stream.stream returns an opaque handle that can be used to reference the newly
created stream in subsequent hipStream* commands. The stream is allocated on the heap and will remain al-
located even if the handle goes out-of-scope. To release the memory used by the stream, application must call
hipStreamDestroy.

## See also

hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hip-
StreamDestroy
Parameters
stream– [inout]ValidpointertohipStream_t. Thisfunctionwritesthememorywiththenewly
created stream.
Returns
hipSuccess, hipErrorInvalidValue
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags)
Create an asynchronous stream.
Create a new asynchronous stream.stream returns an opaque handle that can be used to reference the newly
created stream in subsequent hipStream* commands. The stream is allocated on the heap and will remain al-
located even if the handle goes out-of-scope. To release the memory used by the stream, application must call
hipStreamDestroy. Flags controls behavior of the stream. See hipStreamDefault, hipStreamNonBlocking.

## See also

hipStreamCreate, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDe-
stroy
144 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Parameters
• stream – [inout] Pointer to new stream
• flags – [in] to control stream creation.
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipStreamCreateWithPriority(hipStream_t *stream, unsigned int flags, int priority)
Create an asynchronous stream with the specified priority.
Create a new asynchronous stream with the specified priority.stream returns an opaque handle that can be
used to reference the newly created stream in subsequent hipStream* commands. The stream is allocated on
the heap and will remain allocated even if the handle goes out-of-scope. To release the memory used by the
stream, application must call hipStreamDestroy. Flags controls behavior of the stream. See hipStreamDefault,
hipStreamNonBlocking.

## See also

hipStreamCreate,hipStreamSynchronize,hipStreamWaitEvent,hipStreamDestroy
Parameters
• stream – [inout] Pointer to new stream
• flags – [in]to control stream creation.
• priority – [in]of the stream. Lower numbers represent higher priorities.
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority)
Returns numerical values that correspond to the least and greatest stream priority.
Returns in *leastPriority and *greatestPriority the numerical values that correspond to the least and greatest
streampriorityrespectively. Streamprioritiesfollowaconventionwherelowernumbersimplygreaterpriorities.
The range of meaningful stream priorities is given by [*greatestPriority, *leastPriority]. If the user attempts to
create a stream with a priority value that is outside the meaningful range as specified by this API, the priority is
automatically clamped to within the valid range.
Parameters
• leastPriority–[inout]pointerinwhichvaluecorrespondingtoleastpriorityisreturned.
• greatestPriority – [inout] pointer in which value corresponding to greatest priority is
returned.
Returns
hipSuccess
hipError_t hipStreamDestroy(hipStream_t stream)
Destroys the specified stream.
Destroys the specified stream.
17.23. Asynchronous Functions 145
HIP Documentation, Release 6.3.42134
Ifcommandsarestillexecutingonthespecifiedstream,somemaycompleteexecutionbeforethequeueisdeleted.
The queue may be destroyed while some commands are still inflight, or may wait for all commands queued to
the stream before destroying it.

## See also

hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamQuery, hipStreamWait-
Event,hipStreamSynchronize
Parameters
stream – [in]stream identifier.
Returns
hipSuccess hipErrorInvalidHandle
hipError_t hipStreamQuery(hipStream_t stream)
Return hipSuccess if all of the operations in the specifiedstream have completed, or hipErrorNotReady if not.
This is thread-safe and returns a snapshot of the current state of the queue. However, if other host threads are
sending work to the stream, the status may change immediately after the function is called. It is typically used
for debug.

## See also

hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStream-
Synchronize,hipStreamDestroy
Parameters
stream – [in]stream to query
Returns
hipSuccess, hipErrorNotReady, hipErrorInvalidHandle
hipError_t hipStreamSynchronize(hipStream_t stream)
Wait for all commands in stream to complete.
This command is host-synchronous : the host will block until the specified stream is empty.
This command follows standard null-stream semantics. Specifically, specifying the null stream will cause the
command to wait for other streams on the same device to complete all pending operations.
ThiscommandhonorsthehipDeviceLaunchBlockingflag,whichcontrolswhetherthewaitisactiveorblocking.

## See also

hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hip-
StreamDestroy
Parameters
stream – [in]stream identifier.
146 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Returns
hipSuccess, hipErrorInvalidHandle
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags)
Make the specified compute stream wait for an event.
This function inserts a wait operation into the specified stream. All future work submitted tostream will wait
untileventreports completion before beginning execution.
Thisfunctiononlywaitsforcommandsinthecurrentstreamtocomplete. Notably,thisfunctiondoesnotimplic-
itly wait for commands in the default stream to complete, even if the specified stream is created with hipStream-
NonBlocking = 0.

## See also

hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hip-
StreamDestroy
Parameters
• stream – [in] stream to make wait.
• event – [in]event to wait on
• flags – [in]control operation [must be 0]
Returns
hipSuccess, hipErrorInvalidHandle
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags)
Return flags associated with this stream.
Return flags associated with this stream in *flags.

## See also

hipStreamCreateWithFlags
Parameters
• stream – [in] stream to be queried
• flags – [inout]Pointer to an unsigned integer in which the stream’s flags are returned
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidHandle
Returns
hipSuccess hipErrorInvalidValue hipErrorInvalidHandle
hipError_t hipStreamGetPriority(hipStream_t stream, int *priority)
Query the priority of a stream.
Query the priority of a stream. The priority is returned in in priority.
17.23. Asynchronous Functions 147
HIP Documentation, Release 6.3.42134

## See also

hipStreamCreateWithFlags
Parameters
• stream – [in] stream to be queried
• priority– [inout]Pointertoanunsignedintegerinwhichthestream’spriorityisreturned
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidHandle
Returns
hipSuccess hipErrorInvalidValue hipErrorInvalidHandle
hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t *device)
Get the device assocaited with the stream.

## See also

hipStreamCreate,hipStreamDestroy, hipDeviceGetStreamPriorityRange
Parameters
• stream – [in] stream to be queried
• device – [out] device associated with the stream
Returns
hipSuccess, hipErrorInvalidValue, hipErrorContextIsDestroyed, hipErrorInvalidHandle, hipEr-
rorNotInitialized, hipErrorDeinitialized, hipErrorInvalidContext
hipError_t hipExtStreamCreateWithCUMask(hipStream_t *stream, uint32_t cuMaskSize, const uint32_t
*cuMask)
Create an asynchronous stream with the specified CU mask.
Create a new asynchronous stream with the specified CU mask.stream returns an opaque handle that can be
used to reference the newly created stream in subsequent hipStream* commands. The stream is allocated on the
heap and will remain allocated even if the handle goes out-of-scope. To release the memory used by the stream,
application must call hipStreamDestroy.

## See also

hipStreamCreate,hipStreamSynchronize,hipStreamWaitEvent,hipStreamDestroy
Parameters
• stream – [inout] Pointer to new stream
• cuMaskSize – [in] Size of CU mask bit array passed in.
• cuMask – [in] Bit-vector representing the CU mask. Each active bit represents using one
CU. The first 32 bits represent the first 32 CUs, and so on. If its size is greater than physical
148 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
CU number (i.e., multiProcessorCount member of hipDeviceProp_t), the extra elements are
ignored. It is user’s responsibility to make sure the input is meaningful.
Returns
hipSuccess, hipErrorInvalidHandle, hipErrorInvalidValue
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t *cuMask)
Get CU mask associated with an asynchronous stream.

## See also

hipStreamCreate,hipStreamSynchronize,hipStreamWaitEvent,hipStreamDestroy
Parameters
• stream – [in] stream to be queried
• cuMaskSize– [in] number of the block of memories (uint32_t *) allocated by user
• cuMask – [out] Pointer to a pre-allocated block of memories (uint32_t *) in which the
stream’s CU mask is returned. The CU mask is returned in a chunck of 32 bits where each
active bit represents one active CU
Returns
hipSuccess, hipErrorInvalidHandle, hipErrorInvalidValue
hipError_t hipStreamAddCallback(hipStream_t stream,hipStreamCallback_tcallback, void *userData, unsigned
int flags)
Adds a callback to be called on the host after all currently enqueued items in the stream have completed. For
eachhipStreamAddCallbackcall,acallbackwillbeexecutedexactlyonce. Thecallbackwillblocklaterworkin
the stream until it is finished.

## See also

hipStreamCreate, hipStreamCreateWithFlags, hipStreamQuery, hipStreamSynchronize, hipStreamWaitEvent,
hipStreamDestroy,hipStreamCreateWithPriority
Parameters
• stream – [in] - Stream to add callback to
• callback – [in]- The function to call once preceding stream operations are complete
• userData – [in]- User specified data to be passed to the callback function
• flags – [in]- Reserved for future use, must be 0
Returns
hipSuccess, hipErrorInvalidHandle, hipErrorNotSupported

```cpp

hipError_t hipMallocAsync(void **dev_ptr, size_t size, hipStream_t stream)

```

Allocates memory with stream ordered semantics.
Inserts a memory allocation operation intostream. A pointer to the allocated memory is returned immediately
in *dptr. The allocation must not be accessed until the allocation operation completes. The allocation comes
from the memory pool associated with the stream’s device.
17.23. Asynchronous Functions 149
HIP Documentation, Release 6.3.42134

## See also


```cpp

hipMallocFromPoolAsync, hipFreeAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute, hipDeviceSetMem-

```

Pool,hipMemPoolSetAttribute,hipMemPoolSetAccess, hipMemPoolGetAccess
Note: The default memory pool of a device contains device memory from that device.
Note: Basic stream ordering allows future work submitted into the same stream to use the allocation. Stream
query,streamsynchronize,andHIPeventscanbeusedtoguaranteethattheallocationoperationcompletesbefore
work submitted in a separate stream runs.
Note: During stream capture, this function results in the creation of an allocation node. In this case, the
allocation is owned by the graph instead of the memory pool. The memory pool’s properties are used to set the
node’s creation parameters.
Note: This API is implemented on Linux, under development on Windows.
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• dev_ptr– [out] Returned device pointer of memory allocation
• size– [in] Number of bytes to allocate
• stream – [in]Thestreamestablishingthestreamorderingcontractandthememorypoolto
allocate from
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotSupported, hipErrorOutOfMemory
hipError_t hipFreeAsync(void *dev_ptr, hipStream_t stream)
Frees memory with stream ordered semantics.
Inserts a free operation intostream. The allocation must not be used after stream execution reaches the free.
After this API returns, accessing the memory from any subsequent work launched on the GPU or querying its
pointer attributes results in undefined behavior.

## See also


```cpp

hipMallocFromPoolAsync, hipMallocAsync, hipMemPoolTrimTo, hipMemPoolGetAttribute, hipDeviceSet-

```

MemPool,hipMemPoolSetAttribute,hipMemPoolSetAccess, hipMemPoolGetAccess
Note: During stream capture, this function results in the creation of a free node and must therefore be passed
the address of a graph allocation.
150 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Note: This API is implemented on Linux, under development on Windows.
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• dev_ptr– [in] Pointer to device memory to free
• stream–[in]Thestream,wherethedestrucitonwilloccuraccordingtotheexecutionorder
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotSupported
hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold)
Releases freed memory back to the OS.
Releases memory back to the OS until the pool contains fewer thanmin_bytes_to_keep reserved bytes, or
there is no more memory that the allocator can safely release. The allocator cannot release OS allocations that
back outstanding asynchronous allocations. The OS allocations may happen at different granularity from the
user allocations.

## See also


```cpp

hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipDeviceSetMemPool,

```

hipMemPoolSetAttribute,hipMemPoolSetAccess,hipMemPoolGetAccess
Note: : Allocations that have not been freed count as outstanding.
Note: : Allocations that have been asynchronously freed but whose completion has not been observed on the
host (eg. by a synchronize) can count as outstanding.
Note: This API is implemented on Linux, under development on Windows.
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• mem_pool – [in]The memory pool to trim allocations
• min_bytes_to_hold – [in] If the pool has less than min_bytes_to_hold reserved, then
the TrimTo operation is a no-op. Otherwise the memory pool will contain at least
min_bytes_to_hold bytes reserved after the operation.
Returns
hipSuccess, hipErrorInvalidValue
17.23. Asynchronous Functions 151
HIP Documentation, Release 6.3.42134
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void *value)
Sets attributes of a memory pool.

## Supported attributes are

• hipMemPoolAttrReleaseThreshold: (value type = cuuint64_t) Amount of reserved memory in bytes
to hold onto before trying to release memory back to the OS. When more than the release threshold bytes
of memory are held by the memory pool, the allocator will try to release memory back to the OS on the
next call to stream, event or context synchronize. (default 0)

```cpp

• hipMemPoolReuseFollowEventDependencies: (value type = int) AllowhipMallocAsync to use

```

memoryasynchronouslyfreedinanotherstreamaslongasastreamorderingdependencyoftheallocating
stream on the free action exists. HIP events and null stream interactions can create the required stream
ordered dependencies. (default enabled)
• hipMemPoolReuseAllowOpportunistic: (value type = int) Allow reuse of already completed frees
when there is no dependency between the free and allocation. (default enabled)

```cpp

• hipMemPoolReuseAllowInternalDependencies: (valuetype=int)Allow hipMallocAsynctoinsert

```

new stream dependencies in order to establish the stream ordering required to reuse a piece of memory
released byhipFreeAsync(default enabled).

## See also


```cpp

hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolTrimTo,

```

hipDeviceSetMemPool,hipMemPoolSetAccess, hipMemPoolGetAccess
Note: This API is implemented on Linux, under development on Windows.
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• mem_pool – [in]The memory pool to modify
• attr– [in] The attribute to modify
• value – [in] Pointer to the value to assign
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void *value)
Gets attributes of a memory pool.

## Supported attributes are

• hipMemPoolAttrReleaseThreshold: (value type = cuuint64_t) Amount of reserved memory in bytes
to hold onto before trying to release memory back to the OS. When more than the release threshold bytes
of memory are held by the memory pool, the allocator will try to release memory back to the OS on the
next call to stream, event or context synchronize. (default 0)

```cpp

• hipMemPoolReuseFollowEventDependencies: (value type = int) AllowhipMallocAsync to use

```

memoryasynchronouslyfreedinanotherstreamaslongasastreamorderingdependencyoftheallocating
152 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
stream on the free action exists. HIP events and null stream interactions can create the required stream
ordered dependencies. (default enabled)
• hipMemPoolReuseAllowOpportunistic: (value type = int) Allow reuse of already completed frees
when there is no dependency between the free and allocation. (default enabled)

```cpp

• hipMemPoolReuseAllowInternalDependencies: (valuetype=int)Allow hipMallocAsynctoinsert

```

new stream dependencies in order to establish the stream ordering required to reuse a piece of memory
released byhipFreeAsync(default enabled).

## See also


```cpp

hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolTrimTo, hipDeviceSetMemPool,hip-

```

MemPoolSetAttribute,hipMemPoolSetAccess, hipMemPoolGetAccess
Note: This API is implemented on Linux, under development on Windows.
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• mem_pool – [in]The memory pool to get attributes of
• attr– [in] The attribute to get
• value – [in] Retrieved value
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc *desc_list, size_t count)
Controls visibility of the specified pool between devices.

## See also


```cpp

hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolTrimTo,

```

hipDeviceSetMemPool,hipMemPoolSetAttribute, hipMemPoolGetAccess
Note: This API is implemented on Linux, under development on Windows.
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• mem_pool – [in]Memory pool for acccess change
• desc_list–[in]Arrayofaccessdescriptors. Eachdescriptorinstructstheaccesstoenable
for a single gpu
17.23. Asynchronous Functions 153
HIP Documentation, Release 6.3.42134
• count – [in] Number of descriptors in the map array.
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipMemPoolGetAccess(hipMemAccessFlags *flags, hipMemPool_t mem_pool, hipMemLocation
*location)
Returns the accessibility of a pool from a device.
Returns the accessibility of the pool’s memory from the specified location.

## See also


```cpp

hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolTrimTo,

```

hipDeviceSetMemPool,hipMemPoolSetAttribute, hipMemPoolSetAccess
Note: This API is implemented on Linux, under development on Windows.
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• flags – [out] Accessibility of the memory pool from the specified location/device
• mem_pool – [in]Memory pool being queried
• location – [in] Location/device for memory pool access
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipMemPoolCreate(hipMemPool_t *mem_pool, const hipMemPoolProps *pool_props)
Creates a memory pool.
Creates a HIP memory pool and returns the handle inmem_pool. Thepool_props determines the properties
of the pool such as the backing device and IPC capabilities.
By default, the memory pool will be accessible from the device it is allocated on.

## See also


```cpp

hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolDestroy,

```

hipMemPoolTrimTo,hipDeviceSetMemPool, hipMemPoolSetAttribute,hipMemPoolSetAccess,hipMemPoolGe-
tAccess
Note: Specifying hipMemHandleTypeNone creates a memory pool that will not support IPC.
Note: This API is implemented on Linux, under development on Windows.
154 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• mem_pool – [out]Contains createed memory pool
• pool_props– [in] Memory pool properties
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotSupported
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool)
Destroys the specified memory pool.
If any pointers obtained from this pool haven’t been freed or the pool has free operations that haven’t completed
when hipMemPoolDestroy is invoked, the function will return immediately and the resources associated with
the pool will be released automatically once there are no more outstanding allocations.
Destroying the current mempool of a device sets the default mempool of that device as the current mempool for
that device.

## See also


```cpp

hipMallocFromPoolAsync, hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolCreate hip-

```

MemPoolTrimTo,hipDeviceSetMemPool, hipMemPoolSetAttribute,hipMemPoolSetAccess,hipMemPoolGetAc-
cess
Note: A device’s default memory pool cannot be destroyed.
Note: This API is implemented on Linux, under development on Windows.
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
mem_pool– [in] Memory pool for destruction
Returns
hipSuccess, hipErrorInvalidValue

```cpp

hipError_t hipMallocFromPoolAsync(void **dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream)

```

Allocates memory from a specified pool with stream ordered semantics.
Inserts an allocation operation intostream. A pointer to the allocated memory is returned immediately in
dev_ptr. The allocation must not be accessed until the allocation operation completes. The allocation comes
from the specified memory pool.
17.23. Asynchronous Functions 155
HIP Documentation, Release 6.3.42134
Basic stream ordering allows future work submitted into the same stream to use the allocation. Stream query,
streamsynchronize,andHIPeventscanbeusedtoguaranteethattheallocationoperationcompletesbeforework
submitted in a separate stream runs.

## See also


```cpp

hipMallocAsync, hipFreeAsync, hipMemPoolGetAttribute, hipMemPoolCreate hipMemPoolTrimTo, hipDevice-

```

SetMemPool,hipMemPoolSetAttribute,hipMemPoolSetAccess,hipMemPoolGetAccess,
Note: The specified memory pool may be from a device different than that of the specifiedstream.
Note: During stream capture, this function results in the creation of an allocation node. In this case, the
allocation is owned by the graph instead of the memory pool. The memory pool’s properties are used to set the
node’s creation parameters.
Note: This API is implemented on Linux, under development on Windows.
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• dev_ptr– [out] Returned device pointer
• size– [in] Number of bytes to allocate
• mem_pool – [in]The pool to allocate from
• stream – [in] The stream establishing the stream ordering semantic
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotSupported, hipErrorOutOfMemory
hipError_t hipMemPoolExportToShareableHandle(void *shared_handle, hipMemPool_t mem_pool,
hipMemAllocationHandleType handle_type, unsigned int
flags)
Exports a memory pool to the requested handle type.
GivenanIPCcapablemempool,createanOShandletosharethepoolwithanotherprocess. Arecipientprocess
canconverttheshareablehandleintoamempoolwith hipMemPoolImportFromShareableHandle. Individual
pointerscanthenbesharedwiththe hipMemPoolExportPointerandhipMemPoolImportPointerAPIs. The
implementation of what the shareable handle is and how it can be transferred is defined by the requested handle
type.

## See also

hipMemPoolImportFromShareableHandle
156 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Note: : To create an IPC capable mempool, create a mempool with ahipMemAllocationHandleType other
than hipMemHandleTypeNone.
Note: This API is implemented on Linux, under development on Windows.
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• shared_handle– [out] Pointer to the location in which to store the requested handle
• mem_pool – [in]Pool to export
• handle_type – [in]The type of handle to create
• flags – [in] Must be 0
Returns
hipSuccess, hipErrorInvalidValue, hipErrorOutOfMemory
hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t *mem_pool, void *shared_handle,
hipMemAllocationHandleType handle_type, unsigned int
flags)
Imports a memory pool from a shared handle.
Specific allocations can be imported from the imported pool withhipMemPoolImportPointer.

## See also

hipMemPoolExportToShareableHandle
Note: Imported memory pools do not support creating new allocations. As such imported memory pools may

```cpp

not be used inhipDeviceSetMemPool or hipMallocFromPoolAsync calls.

```

Note: This API is implemented on Linux, under development on Windows.
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• mem_pool – [out]Returned memory pool
• shared_handle – [in] OS handle of the pool to open
• handle_type – [in]The type of handle being imported
• flags – [in] Must be 0
17.23. Asynchronous Functions 157
HIP Documentation, Release 6.3.42134
Returns
hipSuccess, hipErrorInvalidValue, hipErrorOutOfMemory
hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData *export_data, void *dev_ptr)
Export data to share a memory pool allocation between processes.
Constructs export_data for sharing a specific allocation from an already shared memory pool. The recipient
process can import the allocation with thehipMemPoolImportPointer api. The data is not a handle and may
be shared through any IPC mechanism.

## See also

hipMemPoolImportPointer
Note: This API is implemented on Linux, under development on Windows.
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• export_data – [out] Returned export data
• dev_ptr– [in] Pointer to memory being exported
Returns
hipSuccess, hipErrorInvalidValue, hipErrorOutOfMemory
hipError_t hipMemPoolImportPointer(void **dev_ptr, hipMemPool_t mem_pool, hipMemPoolPtrExportData
*export_data)
Import a memory pool allocation from another process.
Returns indev_ptr a pointer to the imported memory. The imported memory must not be accessed before the
allocationoperationcompletesintheexportingprocess. Theimportedmemorymustbefreedfromallimporting
processesbeforebeingfreedintheexportingprocess. Thepointermaybefreedwith hipFreeorhipFreeAsync.
If hipFreeAsyncis used, the free must be completed on the importing process before the free operation on the
exporting process.

## See also

hipMemPoolExportPointer
Note: The hipFreeAsync api may be used in the exporting process before thehipFreeAsync operation
completes in its stream as long as thehipFreeAsync in the exporting process specifies a stream with a stream
dependency on the importing process’shipFreeAsync.
Note: This API is implemented on Linux, under development on Windows.
158 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Warning: : This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• dev_ptr– [out] Pointer to imported memory
• mem_pool – [in]Memory pool from which to import a pointer
• export_data – [in]Data specifying the memory to import
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized, hipErrorOutOfMemory
17.23.2 Peer to peer
hipError_t hipDeviceCanAccessPeer(int *canAccessPeer, int deviceId, int peerDeviceId)
Determine if a device can access a peer’s memory.
Returns “1” incanAccessPeer if the specifieddevice is capable of directly accessing memory physically
located on peerDevice , or “0” if not.
Returns “0” incanAccessPeerif deviceId == peerDeviceId, and both are valid devices : a device is not a peer
of itself.
Parameters
• canAccessPeer – [out] Returns the peer access capability (0 or 1)
• deviceId – [in]- device from where memory may be accessed.
• peerDeviceId – [in] - device where memory is physically located
Returns
hipSuccess,
Returns
hipErrorInvalidDevice if deviceId or peerDeviceId are not valid devices
hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags)
Enable direct access from current device’s virtual address space to memory allocations physically located on a
peer device.
Memory which already allocated on peer device will be mapped into the address space of the current device. In
addition, all future memory allocations on peerDeviceId will be mapped into the address space of the current
device when the memory is allocated. The peer memory remains accessible from the current device until a call
to hipDeviceDisablePeerAccess or hipDeviceReset.
Returns hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue,
Parameters
• peerDeviceId – [in] Peer device to enable direct access to from the current device
• flags – [in]Reserved for future use, must be zero
Returns
hipErrorPeerAccessAlreadyEnabled if peer access is already enabled for this device.
17.23. Asynchronous Functions 159
HIP Documentation, Release 6.3.42134
hipError_t hipDeviceDisablePeerAccess(int peerDeviceId)
Disable direct access from current device’s virtual address space to memory allocations physically located on a
peer device.
ReturnshipErrorPeerAccessNotEnabledifdirectaccesstomemoryonpeerDevicehasnotyetbeenenabledfrom
the current device.
Parameters
peerDeviceId – [in]Peer device to disable direct access to
Returns
hipSuccess, hipErrorPeerAccessNotEnabled
hipError_t hipMemGetAddressRange(hipDeviceptr_t *pbase, size_t *psize, hipDeviceptr_t dptr)
Get information on memory allocations.

## See also

hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxSetCurrent, hipC-
txPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
Parameters
• pbase – [out]- BAse pointer address
• psize – [out]- Size of allocation
• dptr- – [in]Device Pointer
Returns
hipSuccess, hipErrorNotFound

```cpp

hipError_t hipMemcpyPeer(void *dst, int dstDeviceId, const void *src, int srcDeviceId, size_t sizeBytes)

```

Copies memory from one device to memory on another device.
Parameters
• dst – [out] - Destination device pointer.
• dstDeviceId – [in]- Destination device
• src – [in]- Source device pointer
• srcDeviceId – [in]- Source device
• sizeBytes – [in] - Size of memory copy in bytes
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidDevice

```cpp

hipError_t hipMemcpyPeerAsync(void *dst, int dstDeviceId, const void *src, int srcDevice, size_t sizeBytes,

```

hipStream_t stream)
Copies memory from one device to memory on another device.
Parameters
• dst – [out] - Destination device pointer.
• dstDeviceId – [in]- Destination device
• src – [in] - Source device pointer
• srcDevice – [in] - Source device
160 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
• sizeBytes – [in] - Size of memory copy in bytes
• stream – [in] - Stream identifier
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidDevice

## USE_PEER_NON_UNIFIED

17.23.3 Memory management
hipError_t hipPointerSetAttribute(const void *value, hipPointer_attribute attribute, hipDeviceptr_t ptr)
Sets information on the specified pointer.[BETA].
Warning: This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• value – [in] Sets pointer attribute value
• attribute – [in] Attribute to set
• ptr – [in]Pointer to set attributes for
Returns
hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes, const void *ptr)
Returns attributes for the specified pointer.
The output parameter ‘attributes’ has a member named ‘type’ that describes what memory the pointer is asso-
ciated with, such as device memory, host memory, managed memory, and others. Otherwise, the API cannot
handle the pointer and returns hipErrorInvalidValue.

## See also

hipPointerGetAttribute
Note: TheunrecognizedmemorytypeisunsupportedtokeeptheHIPfunctionalitybackwardcompatibilitydue
to hipMemoryType enum values.
Note: The current behavior of this HIP API corresponds to the CUDA API before version 11.0.
Parameters
• attributes– [out] attributes for the specified pointer
• ptr – [in]pointer to get attributes for
17.23. Asynchronous Functions 161
HIP Documentation, Release 6.3.42134
Returns
hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
hipError_t hipPointerGetAttribute(void *data, hipPointer_attribute attribute, hipDeviceptr_t ptr)
Returns information about the specified pointer.[BETA].

## See also

hipPointerGetAttributes
Warning: This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• data– [inout] Returned pointer attribute value
• attribute – [in] Attribute to query for
• ptr – [in]Pointer to get attributes for
Returns
hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute *attributes, void
**data, hipDeviceptr_t ptr)
Returns information about the specified pointer.[BETA].

## See also

hipPointerGetAttribute
Warning: This API is marked as beta, meaning, while this is feature complete, it is still open to changes
and may have outstanding issues.
Parameters
• numAttributes – [in] number of attributes to query for
• attributes– [in] attributes to query for
• data– [inout]atwo-dimensionalcontainingpointerstomemorylocationswheretheresult
of each attribute query will be written to
• ptr – [in]pointer to get attributes for
Returns
hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue

```cpp

hipError_t hipMalloc(void **ptr, size_t size)

```

Allocate memory on the default accelerator.
If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
162 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134

## See also


```cpp

hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray, hipHostFree,

```

hipHostMalloc
Parameters
• ptr – [out] Pointer to the allocated memory
• size– [in] Requested memory size
Returns
hipSuccess, hipErrorOutOfMemory, hipErrorInvalidValue (bad context, null *ptr)
hipError_t hipExtMallocWithFlags(void **ptr, size_t sizeBytes, unsigned int flags)
Allocate memory on the default accelerator.
If requested memory size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
ThememoryallocationflagshouldbeeitherhipDeviceMallocDefault,hipDeviceMallocFinegrained,hipDevice-

```cpp

MallocUncached, or hipMallocSignalMemory. If the flag is any other value, the API returns hipErrorInvalid-

```

Value.

## See also


```cpp

hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray, hipHostFree,

```

hipHostMalloc
Parameters
• ptr – [out] Pointer to the allocated memory
• sizeBytes – [in] Requested memory size
• flags – [in]Type of memory allocation
Returns
hipSuccess, hipErrorOutOfMemory, hipErrorInvalidValue (bad context, null *ptr)
hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags)
Allocates device accessible page locked (pinned) host memory.
This API allocates pinned host memory which is mapped into the address space of all GPUs in the system, the
memory can be accessed directly by the GPU device, and can be read or written with much higher bandwidth
than pageable memory obtained with functions such as malloc().
Using the pinned host memory, applications can implement faster data transfers for HostToDevice and Device-
ToHost. The runtime tracks the hipHostMalloc allocations and can avoid some of the setup required for regular
unpinned memory.
When the memory accesses are infrequent, zero-copy memory can be a good choice, for coherent allocation.
GPU can directly access the host memory over the CPU/GPU interconnect, without need to copy the data.
Currently the allocation granularity is 4KB for the API.
Developers need to choose proper allocation flag with consideration of synchronization.
17.23. Asynchronous Functions 163
HIP Documentation, Release 6.3.42134
If no input for flags, it will be the default pinned memory allocation on the host.

## See also

hipSetDeviceFlags, hipHostFree
Parameters
• ptr – [out] Pointer to the allocated host pinned memory
• size–[in]RequestedmemorysizeinbytesIfsizeis0,nomemoryisallocated,*ptrreturns
nullptr, and hipSuccess is returned.
• flags – [in] Type of host memory allocation
Returns
hipSuccess, hipErrorOutOfMemory
hipError_t hipHostAlloc(void **ptr, size_t size, unsigned int flags)
Allocate device accessible page locked host memory.
If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.

## Flags

• hipHostAllocDefault Default pinned memory allocation on the host.
• hipHostAllocPortable Memory is considered allocated by all contexts.
• hipHostAllocMapped Map the allocation into the address space for the current device.
• hipHostAllocWriteCombined Allocates the memory as write-combined.
Parameters
• ptr – [out] Pointer to the allocated host pinned memory
• size– [in] Requested memory size in bytes
• flags – [in] Type of host memory allocation see below
Returns
hipSuccess, hipErrorOutOfMemory, hipErrorInvalidValue
hipError_t hipHostGetDevicePointer(void **devPtr, void *hstPtr, unsigned int flags)
Get Device pointer from Host Pointer allocated through hipHostMalloc.

## See also

hipSetDeviceFlags,hipHostMalloc
Parameters
• devPtr – [out] Device Pointer mapped to passed host pointer
• hstPtr – [in] Host Pointer allocated through hipHostMalloc
• flags – [in]Flags to be passed for extension
164 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Returns
hipSuccess, hipErrorInvalidValue, hipErrorOutOfMemory
hipError_t hipHostGetFlags(unsigned int *flagsPtr, void *hostPtr)
Return flags associated with host pointer.

## See also

hipHostMalloc
Parameters
• flagsPtr – [out] Memory location to store flags
• hostPtr– [in] Host Pointer allocated through hipHostMalloc
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipHostRegister(void *hostPtr, size_t sizeBytes, unsigned int flags)
Register host memory so it can be accessed from the current device.

## Flags

• hipHostRegisterDefault Memory is Mapped and Portable
• hipHostRegisterPortable Memory is considered registered by all contexts. HIP only supports one context
so this is always assumed true.
• hipHostRegisterMappedMaptheallocationintotheaddressspaceforthecurrentdevice. Thedevicepointer
can be obtained withhipHostGetDevicePointer.
After registering the memory, usehipHostGetDevicePointer to obtain the mapped device pointer. On many
systems, the mapped device pointer will have a different value than the mapped host pointer. Applications must
use the device pointer in device code, and the host pointer in host code.
On some systems, registered memory is pinned. On some systems, registered memory may not be actually be
pinned but uses OS or hardware facilities to all GPU access to the host memory.
Developers are strongly encouraged to register memory blocks which are aligned to the host cache-line size.
(typically 64-bytes but can be obtains from the CPUID instruction).
Ifregisteringnon-alignedpointers,theapplicationmusttakecarewhenregisterpointersfromthesamecacheline
on different devices. HIP’s coarse-grained synchronization model does not guarantee correct results if different
devices write to different parts of the same cache block - typically one of the writes will “win” and overwrite
data from the other registered memory region.

## See also

hipHostUnregister,hipHostGetFlags,hipHostGetDevicePointer
Parameters
• hostPtr– [out] Pointer to host memory to be registered.
• sizeBytes – [in] Size of the host memory
• flags – [in]See below.
17.23. Asynchronous Functions 165
HIP Documentation, Release 6.3.42134
Returns
hipSuccess, hipErrorOutOfMemory
hipError_t hipHostUnregister(void *hostPtr)
Un-register host pointer.

## See also

hipHostRegister
Parameters
hostPtr– [in]Host pointer previously registered withhipHostRegister
Returns
Error code

```cpp

hipError_t hipMallocPitch(void **ptr, size_t *pitch, size_t width, size_t height)

```

Allocates at least width (in bytes) * height bytes of linear memory Padding may occur to ensure alighnment
requirements are met for the given row The change in width size due to padding will be returned in *pitch.
Currently the alignment is set to 128 bytes
If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.

## See also


```cpp

hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree,hipMalloc3D, hipMalloc3DArray, hipHost-

```

Malloc
Parameters
• ptr – [out] Pointer to the allocated device memory
• pitch – [out] Pitch for allocation (in bytes)
• width – [in] Requested pitched allocation width (in bytes)
• height – [in]Requested pitched allocation height
Returns
Error code
hipError_t hipMemAllocPitch(hipDeviceptr_t *dptr, size_t *pitch, size_t widthInBytes, size_t height, unsigned int
elementSizeBytes)
Allocates at least width (in bytes) * height bytes of linear memory Padding may occur to ensure alighnment
requirements are met for the given row The change in width size due to padding will be returned in *pitch.
Currently the alignment is set to 128 bytes
If size is 0, no memory is allocated,ptr returns nullptr, and hipSuccess is returned. The intended usage of pitch
is as a separate parameter of the allocation, used to compute addresses within the 2D array. Given the row and
column of an array element of type T, the address is computed as: TpElement = (T*)((char*)BaseAddress +
Row * Pitch) + Column;
166 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134

## See also


```cpp

hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree,hipMalloc3D, hipMalloc3DArray, hipHost-

```

Malloc
Parameters
• dptr– [out] Pointer to the allocated device memory
• pitch – [out] Pitch for allocation (in bytes)
• widthInBytes– [in] Requested pitched allocation width (in bytes)
• height – [in]Requested pitched allocation height
• elementSizeBytes – [in]The size of element bytes, should be 4, 8 or 16
Returns
Error code
hipError_t hipFree(void *ptr)
Free memory allocated by the hcc hip memory allocation API. This API performs an implicit hipDeviceSyn-
chronize() call. If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.

## See also


```cpp

hipMalloc, hipMallocPitch, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D, hipMalloc3DArray,

```

hipHostMalloc
Parameters
ptr– [in] Pointer to memory to be freed
Returns
hipSuccess
Returns
hipErrorInvalidDevicePointer (if pointer is invalid, including host pointers allocated with
hipHostMalloc)
hipError_t hipFreeHost(void *ptr)
Freespage-lockedmemoryThisAPIperformsanimplicithipDeviceSynchronize()call. IfpointerisNULL,the
hip runtime is initialized and hipSuccess is returned.
Parameters
ptr– [in] Pointer to memory to be freed
Returns
hipSuccess, hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with

```cpp

hipMalloc)
hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind)

```

Copy data from src to dst.
It supports memory from host to device, device to host, device to device and host to host The src and dst must
not overlap.

```cpp

ForhipMemcpy,thecopyisalwaysperformedbythecurrentdevice(setbyhipSetDevice). Formulti-gpuorpeer-

```

to-peer configurations, it is recommended to set the current device to the device where the src data is physically
located. Foroptimalpeer-to-peercopies,thecopydevicemustbeabletoaccessthesrcanddstpointers(bycalling
hipDeviceEnablePeerAccess with copy agent as the current device and src/dest as the peerDevice argument. if
17.23. Asynchronous Functions 167
HIP Documentation, Release 6.3.42134

```cpp

this is not done, the hipMemcpy will still work, but will perform the copy using a staging buffer on the host.
CallinghipMemcpywithdstandsrcpointersthatdonotmatchthehipMemcpyKindresultsinundefinedbehavior.

## See also


```

hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAl-

```cpp

locPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,hipMemcpyAtoA, hipMemcpyAtoD,
hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hip-
MemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hip-

```

MemFree,hipMemFreeHost, hipMemGetAddressRange,hipMemGetInfo,hipMemHostAlloc,hipMemHostGet-
DevicePointer
Parameters
• dst – [out] Data being copy to
• src – [in]Data being copy from
• sizeBytes – [in] Data size in bytes
• kind – [in] Kind of transfer
Returns
hipSuccess, hipErrorInvalidValue, hipErrorUnknown

```cpp

hipError_t hipMemcpyWithStream(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t

```

stream)
Memory copy on the stream. It allows single or multiple devices to do memory copy on single or multiple
streams.

## See also


```cpp

hipMemcpy,hipStreamCreate, hipStreamSynchronize, hipStreamDestroy, hipSetDevice, hipLaunchKernelGGL

```

Parameters
• dst – [out] Data being copy to
• src – [in]Data being copy from
• sizeBytes – [in] Data size in bytes
• kind – [in] Kind of transfer
• stream – [in] Valid stream
Returns
hipSuccess, hipErrorInvalidValue, hipErrorUnknown, hipErrorContextIsDestroyed

```cpp

hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void *src, size_t sizeBytes)

```

Copy data from Host to Device.

## See also

hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAl-

```cpp

locPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,hipMemcpyAtoA, hipMemcpyAtoD,

```

168 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134

```cpp

hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hip-
MemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hip-

```

MemFree,hipMemFreeHost, hipMemGetAddressRange,hipMemGetInfo,hipMemHostAlloc,hipMemHostGet-
DevicePointer
Parameters
• dst – [out] Data being copy to
• src – [in]Data being copy from
• sizeBytes – [in] Data size in bytes
Returns
hipSuccess, hipErrorDeinitialized, hipErrorNotInitialized, hipErrorInvalidContext, hipErrorIn-
validValue

```cpp

hipError_t hipMemcpyDtoH(void *dst, hipDeviceptr_t src, size_t sizeBytes)

```

Copy data from Device to Host.

## See also

hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAl-

```cpp

locPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,hipMemcpyAtoA, hipMemcpyAtoD,
hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hip-
MemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hip-

```

MemFree,hipMemFreeHost, hipMemGetAddressRange,hipMemGetInfo,hipMemHostAlloc,hipMemHostGet-
DevicePointer
Parameters
• dst – [out] Data being copy to
• src – [in]Data being copy from
• sizeBytes – [in] Data size in bytes
Returns
hipSuccess, hipErrorDeinitialized, hipErrorNotInitialized, hipErrorInvalidContext, hipErrorIn-
validValue

```cpp

hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes)

```

Copy data from Device to Device.

## See also

hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAl-

```cpp

locPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,hipMemcpyAtoA, hipMemcpyAtoD,
hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hip-
MemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hip-

```

MemFree,hipMemFreeHost, hipMemGetAddressRange,hipMemGetInfo,hipMemHostAlloc,hipMemHostGet-
DevicePointer
Parameters
• dst – [out] Data being copy to
17.23. Asynchronous Functions 169
HIP Documentation, Release 6.3.42134
• src – [in]Data being copy from
• sizeBytes – [in] Data size in bytes
Returns
hipSuccess, hipErrorDeinitialized, hipErrorNotInitialized, hipErrorInvalidContext, hipErrorIn-
validValue

```cpp

hipError_t hipMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray, size_t srcOffset, size_t ByteCount)

```

Copies from one 1D array to device memory.

## See also

hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAl-

```cpp

locPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,hipMemcpyAtoA, hipMemcpyAtoD,
hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hip-
MemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hip-

```

MemFree,hipMemFreeHost, hipMemGetAddressRange,hipMemGetInfo,hipMemHostAlloc,hipMemHostGet-
DevicePointer
Parameters
• dstDevice – [out] Destination device pointer
• srcArray – [in]Source array
• srcOffset – [in] Offset in bytes of source array
• ByteCount – [in] Size of memory copy in bytes
Returns
hipSuccess, hipErrorDeinitialized, hipErrorNotInitialized, hipErrorInvalidContext, hipErrorIn-
validValue

```cpp

hipError_t hipMemcpyDtoA(hipArray_t dstArray, size_t dstOffset, hipDeviceptr_t srcDevice, size_t ByteCount)

```

Copies from device memory to a 1D array.

## See also

hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAl-

```cpp

locPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,hipMemcpyAtoA, hipMemcpyAtoD,
hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hip-
MemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hip-

```

MemFree,hipMemFreeHost, hipMemGetAddressRange,hipMemGetInfo,hipMemHostAlloc,hipMemHostGet-
DevicePointer
Parameters
• dstArray – [out]Destination array
• dstOffset – [in] Offset in bytes of destination array
• srcDevice – [in] Source device pointer
• ByteCount – [in] Size of memory copy in bytes
170 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Returns
hipSuccess, hipErrorDeinitialized, hipErrorNotInitialized, hipErrorInvalidContext, hipErrorIn-
validValue

```cpp

hipError_t hipMemcpyAtoA(hipArray_t dstArray, size_t dstOffset, hipArray_t srcArray, size_t srcOffset, size_t

```

ByteCount)
Copies from one 1D array to another.

## See also

hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAl-

```cpp

locPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,hipMemcpyAtoA, hipMemcpyAtoD,
hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hip-
MemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hip-

```

MemFree,hipMemFreeHost, hipMemGetAddressRange,hipMemGetInfo,hipMemHostAlloc,hipMemHostGet-
DevicePointer
Parameters
• dstArray – [out]Destination array
• dstOffset – [in] Offset in bytes of destination array
• srcArray – [in]Source array
• srcOffset – [in] Offset in bytes of source array
• ByteCount – [in] Size of memory copy in bytes
Returns
hipSuccess, hipErrorDeinitialized, hipErrorNotInitialized, hipErrorInvalidContext, hipErrorIn-
validValue

```cpp

hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void *src, size_t sizeBytes, hipStream_t stream)

```

Copy data from Host to Device asynchronously.

## See also

hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAl-

```cpp

locPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,hipMemcpyAtoA, hipMemcpyAtoD,
hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hip-
MemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hip-

```

MemFree,hipMemFreeHost, hipMemGetAddressRange,hipMemGetInfo,hipMemHostAlloc,hipMemHostGet-
DevicePointer
Parameters
• dst – [out] Data being copy to
• src – [in]Data being copy from
• sizeBytes – [in] Data size in bytes
• stream – [in] Stream identifier
Returns
hipSuccess, hipErrorDeinitialized, hipErrorNotInitialized, hipErrorInvalidContext, hipErrorIn-
validValue
17.23. Asynchronous Functions 171
HIP Documentation, Release 6.3.42134

```cpp

hipError_t hipMemcpyDtoHAsync(void *dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)

```

Copy data from Device to Host asynchronously.

## See also

hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAl-

```cpp

locPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,hipMemcpyAtoA, hipMemcpyAtoD,
hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hip-
MemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hip-

```

MemFree,hipMemFreeHost, hipMemGetAddressRange,hipMemGetInfo,hipMemHostAlloc,hipMemHostGet-
DevicePointer
Parameters
• dst – [out] Data being copy to
• src – [in]Data being copy from
• sizeBytes – [in] Data size in bytes
• stream – [in] Stream identifier
Returns
hipSuccess, hipErrorDeinitialized, hipErrorNotInitialized, hipErrorInvalidContext, hipErrorIn-
validValue

```cpp

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)

```

Copy data from Device to Device asynchronously.

## See also

hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAl-

```cpp

locPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,hipMemcpyAtoA, hipMemcpyAtoD,
hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hip-
MemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hip-

```

MemFree,hipMemFreeHost, hipMemGetAddressRange,hipMemGetInfo,hipMemHostAlloc,hipMemHostGet-
DevicePointer
Parameters
• dst – [out] Data being copy to
• src – [in]Data being copy from
• sizeBytes – [in] Data size in bytes
• stream – [in] Stream identifier
Returns
hipSuccess, hipErrorDeinitialized, hipErrorNotInitialized, hipErrorInvalidContext, hipErrorIn-
validValue

```cpp

hipError_t hipMemcpyAtoHAsync(void *dstHost, hipArray_t srcArray, size_t srcOffset, size_t ByteCount,

```

hipStream_t stream)
Copies from one 1D array to host memory.
172 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134

## See also

hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAl-

```cpp

locPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,hipMemcpyAtoA, hipMemcpyAtoD,
hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hip-
MemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hip-

```

MemFree,hipMemFreeHost, hipMemGetAddressRange,hipMemGetInfo,hipMemHostAlloc,hipMemHostGet-
DevicePointer
Parameters
• dstHost– [out] Destination pointer
• srcArray – [in]Source array
• srcOffset – [in] Offset in bytes of source array
• ByteCount – [in] Size of memory copy in bytes
• stream – [in] Stream identifier
Returns
hipSuccess, hipErrorDeinitialized, hipErrorNotInitialized, hipErrorInvalidContext, hipErrorIn-
validValue

```cpp

hipError_t hipMemcpyHtoAAsync(hipArray_t dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount,

```

hipStream_t stream)
Copies from host memory to a 1D array.

## See also

hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAl-

```cpp

locPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned,hipMemcpyAtoA, hipMemcpyAtoD,
hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hip-
MemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hip-

```

MemFree,hipMemFreeHost, hipMemGetAddressRange,hipMemGetInfo,hipMemHostAlloc,hipMemHostGet-
DevicePointer
Parameters
• dstArray – [out]Destination array
• dstOffset – [in] Offset in bytes of destination array
• srcHost– [in] Source host pointer
• ByteCount – [in] Size of memory copy in bytes
• stream – [in] Stream identifier
Returns
hipSuccess, hipErrorDeinitialized, hipErrorNotInitialized, hipErrorInvalidContext, hipErrorIn-
validValue
hipError_t hipModuleGetGlobal(hipDeviceptr_t *dptr, size_t *bytes, hipModule_t hmod, const char *name)
Returns a global pointer from a module. Returns in *dptr and *bytes the pointer and size of the global of name
namelocatedinmodulehmod. Ifnovariableofthatnameexists,itreturnshipErrorNotFound. Bothparameters
dptr and bytes are optional. If one of them is NULL, it is ignored and hipSuccess is returned.
Parameters
17.23. Asynchronous Functions 173
HIP Documentation, Release 6.3.42134
• dptr– [out] Returns global device pointer
• bytes – [out] Returns global size in bytes
• hmod – [in] Module to retrieve global from
• name– [in] Name of global to retrieve
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotFound, hipErrorInvalidContext
hipError_t hipGetSymbolAddress(void **devPtr, const void *symbol)
Gets device pointer associated with symbol on the device.
Parameters
• devPtr – [out] pointer to the device associated the symbole
• symbol – [in] pointer to the symbole of the device
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipGetSymbolSize(size_t *size, const void *symbol)
Gets the size of the given symbol on the device.
Parameters
• symbol – [in] pointer to the device symbole
• size– [out] pointer to the size
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipGetProcAddress(const char *symbol, void **pfn, int hipVersion, uint64_t flags,
hipDriverProcAddressQueryResult *symbolStatus)
Gets the pointer of requested HIP driver function.
Returns hipSuccess if the returned pfn is addressed to the pointer of found driver function.
Parameters
• symbol – [in] The Symbol name of the driver function to request.
• pfn – [out] Output pointer to the requested driver function.
• hipVersion – [in] The HIP version for the requested driver function symbol. HIP version
is defined as 100*version_major + version_minor. For example, in HIP 6.1, the hipversion
is 601, for the symbol function “hipGetDeviceProperties”, the specified hipVersion 601 is
greater or equal to the version 600, the symbol function will be handle properly as backend
compatible function.
• flags – [in]Currently only default flag is suppported.
• symbolStatus – [out] Optional enumeration for returned status of searching for symbol
driver function based on the input hipVersion.
Returns
hipSuccess, hipErrorInvalidValue.
174 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134

```cpp

hipError_t hipMemcpyToSymbol(const void *symbol, const void *src, size_t sizeBytes, size_t offset,
hipMemcpyKind kind)
Copies data to the given symbol on the device. Symbol HIP APIs allow a kernel to define a device-side data

```

symbol which can be accessed on the host side. The symbol can be in __constant or device space. Note that the

```cpp

symbol name needs to be encased in the HIP_SYMBOL macro. This also applies to hipMemcpyFromSymbol,

```

hipGetSymbolAddress, and hipGetSymbolSize. For detailed usage, see the memcpyToSymbol example in the
HIP Porting Guide.
Parameters
• symbol – [out] pointer to the device symbole
• src – [in]pointer to the source address
• sizeBytes – [in] size in bytes to copy
• offset – [in] offset in bytes from start of symbole
• kind – [in] type of memory transfer
Returns
hipSuccess, hipErrorInvalidValue

```cpp

hipError_t hipMemcpyToSymbolAsync(const void *symbol, const void *src, size_t sizeBytes, size_t offset,
hipMemcpyKind kind, hipStream_t stream)

```

Copies data to the given symbol on the device asynchronously.
Parameters
• symbol – [out] pointer to the device symbole
• src – [in]pointer to the source address
• sizeBytes – [in] size in bytes to copy
• offset – [in] offset in bytes from start of symbole
• kind – [in] type of memory transfer
• stream – [in] stream identifier
Returns
hipSuccess, hipErrorInvalidValue

```cpp

hipError_t hipMemcpyFromSymbol(void *dst, const void *symbol, size_t sizeBytes, size_t offset, hipMemcpyKind

```

kind)
Copies data from the given symbol on the device.
Parameters
• dst – [out] Returns pointer to destinition memory address
• symbol – [in] Pointer to the symbole address on the device
• sizeBytes – [in] Size in bytes to copy
• offset – [in] Offset in bytes from the start of symbole
• kind – [in] Type of memory transfer
Returns
hipSuccess, hipErrorInvalidValue
17.23. Asynchronous Functions 175
HIP Documentation, Release 6.3.42134

```cpp

hipError_t hipMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t sizeBytes, size_t offset,
hipMemcpyKind kind, hipStream_t stream)

```

Copies data from the given symbol on the device asynchronously.
Parameters
• dst – [out] Returns pointer to destinition memory address
• symbol – [in] pointer to the symbole address on the device
• sizeBytes – [in] size in bytes to copy
• offset – [in] offset in bytes from the start of symbole
• kind – [in] type of memory transfer
• stream – [in] stream identifier
Returns
hipSuccess, hipErrorInvalidValue

```cpp

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t

```

stream)
Copy data from src to dst asynchronously.
Formulti-gpuorpeer-to-peerconfigurations,itisrecommendedtouseastreamwhichisaattachedtothedevice
where the src data is physically located. For optimal peer-to-peer copies, the copy device must be able to access
thesrcanddstpointers(bycallinghipDeviceEnablePeerAccesswithcopyagentasthecurrentdeviceandsrc/dest

```cpp

as the peerDevice argument. if this is not done, the hipMemcpy will still work, but will perform the copy using

```

a staging buffer on the host.

## See also


```cpp

hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMem-
cpy2DFromArray, hipMemcpyArrayToArray, hipMemcpy2DArrayToArray, hipMemcpyToSymbol, hipMem-
cpyFromSymbol, hipMemcpy2DAsync, hipMemcpyToArrayAsync, hipMemcpy2DToArrayAsync, hipMem-
cpyFromArrayAsync, hipMemcpy2DFromArrayAsync, hipMemcpyToSymbolAsync, hipMemcpyFromSymbol-

```

Async
Warning: If host or dest are not pinned, the memory copy will be performed synchronously. For best
performance, use hipHostMalloc to allocate host memory that is transferred asynchronously.

```cpp

Warning: onHCChipMemcpyAsyncdoesnotsupportoverlappedH2DandD2Hcopies. ForhipMemcpy,

```

the copy is always performed by the device associated with the specified stream.
Parameters
• dst – [out] Data being copy to
• src – [in]Data being copy from
• sizeBytes – [in] Data size in bytes
• kind – [in] Type of memory transfer
176 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
• stream – [in] Stream identifier
Returns
hipSuccess, hipErrorInvalidValue, hipErrorUnknown
hipError_t hipMemset(void *dst, int value, size_t sizeBytes)
Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant byte value value.
Parameters
• dst – [out] Data being filled
• value – [in]Value to be set
• sizeBytes – [in] Data size in bytes
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count)
Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant byte value value.
Parameters
• dest– [out] Data ptr to be filled
• value – [in]Value to be set
• count – [in]Number of values to be set
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count, hipStream_t stream)
Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant byte value value.
hipMemsetD8Async() is asynchronous with respect to the host, so the call may return before the memset is
complete. The operation can optionally be associated to a stream by passing a non-zero stream argument. If
stream is non-zero, the operation may overlap with operations in other streams.
Parameters
• dest– [out] Data ptr to be filled
• value – [in]Constant value to be set
• count – [in]Number of values to be set
• stream – [in] Stream identifier
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count)
Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant short value value.
Parameters
• dest– [out] Data ptr to be filled
• value – [in]Constant value to be set
• count – [in]Number of values to be set
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
17.23. Asynchronous Functions 177
HIP Documentation, Release 6.3.42134
hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count, hipStream_t stream)
Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant short value value.
hipMemsetD16Async() is asynchronous with respect to the host, so the call may return before the memset is
complete. The operation can optionally be associated to a stream by passing a non-zero stream argument. If
stream is non-zero, the operation may overlap with operations in other streams.
Parameters
• dest– [out] Data ptr to be filled
• value – [in]Constant value to be set
• count – [in]Number of values to be set
• stream – [in] Stream identifier
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count)
Fills the memory area pointed to by dest with the constant integer value for specified number of times.
Parameters
• dest– [out] Data being filled
• value – [in]Constant value to be set
• count – [in]Number of values to be set
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
hipError_t hipMemsetAsync(void *dst, int value, size_t sizeBytes, hipStream_t stream)
Fills the first sizeBytes bytes of the memory area pointed to by dev with the constant byte value value.
hipMemsetAsync()isasynchronouswithrespecttothehost,sothecallmayreturnbeforethememsetiscomplete.
Theoperationcanoptionallybeassociatedtoastreambypassinganon-zerostreamargument. Ifstreamisnon-
zero, the operation may overlap with operations in other streams.
Parameters
• dst – [out] Pointer to device memory
• value – [in]Value to set for each byte of specified memory
• sizeBytes – [in] Size in bytes to set
• stream – [in] Stream identifier
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream)
Fills the memory area pointed to by dev with the constant integer value for specified number of times.
hipMemsetD32Async() is asynchronous with respect to the host, so the call may return before the memset is
complete. The operation can optionally be associated to a stream by passing a non-zero stream argument. If
stream is non-zero, the operation may overlap with operations in other streams.
Parameters
• dst – [out] Pointer to device memory
• value – [in]Value to set for each byte of specified memory
178 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
• count – [in] Number of values to be set
• stream – [in] Stream identifier
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipMemset2D(void *dst, size_t pitch, int value, size_t width, size_t height)
Fills the memory area pointed to by dst with the constant value.
Parameters
• dst – [out] Pointer to device memory
• pitch – [in] Data size in bytes
• value – [in]Constant value to be set
• width – [in]
• height – [in]
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipMemset2DAsync(void *dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream)
Fills asynchronously the memory area pointed to by dst with the constant value.
Parameters
• dst – [in]Pointer to 2D device memory
• pitch – [in] Pitch size in bytes
• value – [in]Value to be set for each byte of specified memory
• width – [in] Width of matrix set columns in bytes
• height – [in] Height of matrix set rows in bytes
• stream – [in] Stream identifier
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent)
Fills synchronously the memory area pointed to by pitchedDevPtr with the constant value.
Parameters
• pitchedDevPtr – [in]Pointer to pitched device memory
• value – [in]Value to set for each byte of specified memory
• extent – [in] Size parameters for width field in bytes in device memory
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent, hipStream_t stream)
Fills asynchronously the memory area pointed to by pitchedDevPtr with the constant value.
Parameters
• pitchedDevPtr – [in]Pointer to pitched device memory
• value – [in]Value to set for each byte of specified memory
17.23. Asynchronous Functions 179
HIP Documentation, Release 6.3.42134
• extent – [in]Size parameters for width field in bytes in device memory
• stream – [in] Stream identifier
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipMemGetInfo(size_t *free, size_t *total)
Query memory info.
On ROCM, this function gets the actual free memory left on the current device, so supports the cases while
running multi-workload (such as multiple processes, multiple threads, and multiple GPUs).
Warning: On Windows, the free memory only accounts for memory allocated by this process and may be
optimistic.
Parameters
• free– [out] Returns free memory on the current device in bytes
• total – [out] Returns total allocatable memory on the current device in bytes
Returns
hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
hipError_t hipMemPtrGetInfo(void *ptr, size_t *size)
Get allocated memory size via memory pointer.
This function gets the allocated shared virtual memory size from memory pointer.
Parameters
• ptr – [in]Pointer to allocated memory
• size– [out] Returns the allocated memory size in bytes
Returns
hipSuccess, hipErrorInvalidValue

```cpp

hipError_t hipMallocArray(hipArray_t *array, const hipChannelFormatDesc *desc, size_t width, size_t height,

```

unsigned int flags)
Allocate an array on the device.

## See also


```cpp

hipMalloc,hipMallocPitch, hipFree, hipFreeArray,hipHostMalloc, hipHostFree

```

Parameters
• array – [out]Pointer to allocated array in device memory
• desc– [in] Requested channel format
• width – [in] Requested array allocation width
• height – [in] Requested array allocation height
• flags – [in]Requested properties of allocated array
Returns
hipSuccess, hipErrorOutOfMemory
180 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
hipError_t hipArrayCreate(hipArray_t *pHandle, const HIP_ARRAY_DESCRIPTOR *pAllocateArray)
Create an array memory pointer on the device.

## See also


```cpp

hipMallocArray, hipArrayDestroy,hipFreeArray

```

Parameters
• pHandle– [out] Pointer to the array memory
• pAllocateArray – [in]Requested array desciptor
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotSupported
hipError_t hipArrayDestroy(hipArray_t array)
Destroy an array memory pointer on the device.

## See also

hipArrayCreate,hipArrayDestroy, hipFreeArray
Parameters
array– [in] Pointer to the array memory
Returns
hipSuccess, hipErrorInvalidValue
hipError_t hipArray3DCreate(hipArray_t *array, const HIP_ARRAY3D_DESCRIPTOR *pAllocateArray)
Create a 3D array memory pointer on the device.

## See also


```cpp

hipMallocArray, hipArrayDestroy,hipFreeArray

```

Parameters
• array – [out]Pointer to the 3D array memory
• pAllocateArray – [in]Requested array desciptor
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotSupported

```cpp

hipError_t hipMalloc3D(hipPitchedPtr *pitchedDevPtr, hipExtent extent)

```

Create a 3D memory pointer on the device.

## See also


```cpp

hipMallocPitch,hipMemGetInfo,hipFree

```

Parameters
• pitchedDevPtr – [out] Pointer to the 3D memory
17.23. Asynchronous Functions 181
HIP Documentation, Release 6.3.42134
• extent – [in]Requested extent
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotSupported
hipError_t hipFreeArray(hipArray_t array)
Frees an array on the device.

## See also


```cpp

hipMalloc,hipMallocPitch, hipFree, hipMallocArray,hipHostMalloc, hipHostFree

```

Parameters
array– [in] Pointer to array to free
Returns
hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized

```cpp

hipError_t hipMalloc3DArray(hipArray_t *array, const struct hipChannelFormatDesc *desc, struct hipExtent

```

extent, unsigned int flags)
Allocate an array on the device.

## See also


```cpp

hipMalloc,hipMallocPitch, hipFree, hipFreeArray,hipHostMalloc, hipHostFree

```

Parameters
• array – [out]Pointer to allocated array in device memory
• desc– [in] Requested channel format
• extent – [in] Requested array allocation width, height and depth
• flags – [in]Requested properties of allocated array
Returns
hipSuccess, hipErrorOutOfMemory
hipError_t hipArrayGetInfo(hipChannelFormatDesc *desc, hipExtent *extent, unsigned int *flags, hipArray_t
array)
Gets info about the specified array.

## See also

hipArrayGetDescriptor,hipArray3DGetDescriptor
Parameters
• desc– [out] - Returned array type
• extent – [out] - Returned array shape. 2D arrays will have depth of zero
• flags – [out]- Returned array flags
• array – [in]- The HIP array to get info for
182 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Returns
hipSuccess, hipErrorInvalidValue hipErrorInvalidHandle
hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR *pArrayDescriptor, hipArray_t array)
Gets a 1D or 2D array descriptor.

## See also

hipArray3DCreate,hipArray3DGetDescriptor,hipArrayCreate,hipArrayDestroy,hipMemAlloc,hipMemAllo-

```cpp

cHost,hipMemAllocPitch,hipMemcpy2D,hipMemcpy2DAsync,hipMemcpy2DUnaligned, hipMemcpy3D,hip-
Memcpy3DAsync, hipMemcpyAtoA, hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpy-
DtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA,
hipMemcpyHtoAAsync,hipMemcpyHtoD,hipMemcpyHtoDAsync,hipMemFree,hipMemFreeHost, hipMemGe-

```

tAddressRange, hipMemGetInfo, hipMemHostAlloc, hipMemHostGetDevicePointer,hipMemsetD8, hipMem-
setD16,hipMemsetD32,hipArrayGetInfo
Parameters
• pArrayDescriptor – [out]- Returned array descriptor
• array – [in]- Array to get descriptor of
Returns
hipSuccess, hipErrorDeinitialized, hipErrorNotInitialized, hipErrorInvalidContext, hipErrorIn-
validValue hipErrorInvalidHandle
hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR *pArrayDescriptor, hipArray_t array)
Gets a 3D array descriptor.

## See also

hipArray3DCreate, hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllo-

```cpp

cHost,hipMemAllocPitch,hipMemcpy2D,hipMemcpy2DAsync,hipMemcpy2DUnaligned, hipMemcpy3D,hip-
Memcpy3DAsync, hipMemcpyAtoA, hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpy-
DtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA,
hipMemcpyHtoAAsync,hipMemcpyHtoD,hipMemcpyHtoDAsync,hipMemFree,hipMemFreeHost, hipMemGe-

```

tAddressRange, hipMemGetInfo, hipMemHostAlloc, hipMemHostGetDevicePointer,hipMemsetD8, hipMem-
setD16,hipMemsetD32,hipArrayGetInfo
Parameters
• pArrayDescriptor – [out]- Returned 3D array descriptor
• array – [in]- 3D array to get descriptor of
Returns
hipSuccess, hipErrorDeinitialized, hipErrorNotInitialized, hipErrorInvalidContext, hipErrorIn-
validValue hipErrorInvalidHandle, hipErrorContextIsDestroyed

```cpp

hipError_t hipMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
hipMemcpyKind kind)

```

Copies data between host and device.
17.23. Asynchronous Functions 183
HIP Documentation, Release 6.3.42134

## See also


```cpp

hipMemcpy, hipMemcpyToArray,hipMemcpy2DToArray, hipMemcpyFromArray,hipMemcpyToSymbol, hip-

```

MemcpyAsync
Parameters
• dst – [in]Destination memory address
• dpitch – [in]Pitch of destination memory
• src – [in]Source memory address
• spitch – [in]Pitch of source memory
• width – [in] Width of matrix transfer (columns in bytes)
• height – [in] Height of matrix transfer (rows)
• kind – [in] Type of transfer
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection

```cpp

hipError_t hipMemcpyParam2D(const hip_Memcpy2D *pCopy)

```

Copies memory for 2D arrays.

## See also


```cpp

hipMemcpy,hipMemcpy2D,hipMemcpyToArray, hipMemcpy2DToArray,hipMemcpyFromArray, hipMemcpy-
ToSymbol, hipMemcpyAsync

```

Parameters
pCopy– [in] Parameters for the memory copy
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection

```cpp

hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D *pCopy, hipStream_t stream)

```

Copies memory for 2D arrays.

## See also


```cpp

hipMemcpy,hipMemcpy2D,hipMemcpyToArray, hipMemcpy2DToArray,hipMemcpyFromArray, hipMemcpy-
ToSymbol, hipMemcpyAsync

```

Parameters
• pCopy – [in]Parameters for the memory copy
• stream – [in] Stream to use
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection
184 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134

```cpp

hipError_t hipMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
hipMemcpyKind kind, hipStream_t stream)

```

Copies data between host and device.

## See also


```cpp

hipMemcpy, hipMemcpyToArray,hipMemcpy2DToArray, hipMemcpyFromArray,hipMemcpyToSymbol, hip-

```

MemcpyAsync
Parameters
• dst – [in]Destination memory address
• dpitch – [in]Pitch of destination memory
• src – [in]Source memory address
• spitch – [in]Pitch of source memory
• width – [in] Width of matrix transfer (columns in bytes)
• height – [in] Height of matrix transfer (rows)
• kind – [in] Type of transfer
• stream – [in] Stream to use
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection

```cpp

hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch,
size_t width, size_t height, hipMemcpyKind kind)

```

Copies data between host and device.

## See also


```cpp

hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol, hipMem-

```

cpyAsync
Parameters
• dst – [in]Destination memory address
• wOffset– [in] Destination starting X offset
• hOffset– [in] Destination starting Y offset
• src – [in]Source memory address
• spitch – [in]Pitch of source memory
• width – [in] Width of matrix transfer (columns in bytes)
• height – [in] Height of matrix transfer (rows)
• kind – [in] Type of transfer
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection
17.23. Asynchronous Functions 185
HIP Documentation, Release 6.3.42134

```cpp

hipError_t hipMemcpy2DToArrayAsync(hipArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t
spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t

```

stream)
Copies data between host and device.

## See also


```cpp

hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol, hipMem-

```

cpyAsync
Parameters
• dst – [in]Destination memory address
• wOffset– [in] Destination starting X offset
• hOffset– [in] Destination starting Y offset
• src – [in]Source memory address
• spitch – [in]Pitch of source memory
• width – [in] Width of matrix transfer (columns in bytes)
• height – [in] Height of matrix transfer (rows)
• kind – [in] Type of transfer
• stream – [in] Accelerator view which the copy is being enqueued
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection

```cpp

hipError_t hipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst, size_t hOffsetDst, hipArray_const_t src,

```

size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height,

```cpp

hipMemcpyKind kind)

```

Copies data between host and device.

## See also


```cpp

hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol, hipMem-

```

cpyAsync
Parameters
• dst – [in]Destination memory address
• wOffsetDst – [in] Destination starting X offset
• hOffsetDst – [in] Destination starting Y offset
• src – [in] Source memory address
• wOffsetSrc – [in]Source starting X offset
• hOffsetSrc – [in]Source starting Y offset (columns in bytes)
• width – [in]Width of matrix transfer (columns in bytes)
• height – [in] Height of matrix transfer (rows)
186 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
• kind – [in] Type of transfer
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidMemcpyDirection

```cpp

hipError_t hipMemcpy2DFromArray(void *dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset,
size_t width, size_t height, hipMemcpyKind kind)

```

Copies data between host and device.

## See also


```cpp

hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray,hipMemcpyToSymbol, hipMem-

```

cpyAsync
Parameters
• dst – [in]Destination memory address
• dpitch – [in]Pitch of destination memory
• src – [in]Source memory address
• wOffset– [in] Source starting X offset
• hOffset– [in] Source starting Y offset
• width – [in] Width of matrix transfer (columns in bytes)
• height – [in] Height of matrix transfer (rows)
• kind – [in] Type of transfer
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection

```cpp

hipError_t hipMemcpy2DFromArrayAsync(void *dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t
hOffset, size_t width, size_t height, hipMemcpyKind kind, hipStream_t

```

stream)
Copies data between host and device asynchronously.

## See also


```cpp

hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray,hipMemcpyToSymbol, hipMem-

```

cpyAsync
Parameters
• dst – [in]Destination memory address
• dpitch – [in]Pitch of destination memory
• src – [in] Source memory address
• wOffset– [in] Source starting X offset
• hOffset– [in] Source starting Y offset
• width – [in] Width of matrix transfer (columns in bytes)
• height – [in] Height of matrix transfer (rows)
17.23. Asynchronous Functions 187
HIP Documentation, Release 6.3.42134
• kind – [in] Type of transfer
• stream – [in] Accelerator view which the copy is being enqueued
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection

```cpp

hipError_t hipMemcpyAtoH(void *dst, hipArray_t srcArray, size_t srcOffset, size_t count)

```

Copies data between host and device.

## See also


```cpp

hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray,hipMemcpyToSymbol, hipMem-

```

cpyAsync
Parameters
• dst – [in]Destination memory address
• srcArray – [in]Source array
• srcOffset – [in] Offset in bytes of source array
• count – [in]Size of memory copy in bytes
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection

```cpp

hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset, const void *srcHost, size_t count)

```

Copies data between host and device.

## See also


```cpp

hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray,hipMemcpyToSymbol, hipMem-

```

cpyAsync
Parameters
• dstArray – [in]Destination memory address
• dstOffset – [in] Offset in bytes of destination array
• srcHost– [in] Source host pointer
• count – [in]Size of memory copy in bytes
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection

```cpp

hipError_t hipMemcpy3D(const struct hipMemcpy3DParms *p)

```

Copies data between host and device.
188 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134

## See also


```cpp

hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray,hipMemcpyToSymbol, hipMem-

```

cpyAsync
Parameters
p– [in]3D memory copy parameters
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection

```cpp

hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms *p, hipStream_t stream)

```

Copies data between host and device asynchronously.

## See also


```cpp

hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray,hipMemcpyToSymbol, hipMem-

```

cpyAsync
Parameters
• p– [in] 3D memory copy parameters
• stream – [in] Stream to use
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection
hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D *pCopy)
Copies data between host and device.

## See also


```cpp

hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray,hipMemcpyToSymbol, hipMem-

```

cpyAsync
Parameters
pCopy– [in] 3D memory copy parameters
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection
hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D *pCopy, hipStream_t stream)
Copies data between host and device asynchronously.

## See also


```cpp

hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray,hipMemcpyToSymbol, hipMem-

```

cpyAsync
Parameters
17.23. Asynchronous Functions 189
HIP Documentation, Release 6.3.42134
• pCopy – [in] 3D memory copy parameters
• stream – [in] Stream to use
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue, hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection
17.23.4 External Resource Interoperability
hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t *extSem_out, const
hipExternalSemaphoreHandleDesc *semHandleDesc)
Imports an external semaphore.

## See also

Note: This API is currently not supported on Linux.
Parameters
• extSem_out– [out] External semaphores to be waited on
• semHandleDesc– [in] Semaphore import handle descriptor
Returns
hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t *extSemArray, const
hipExternalSemaphoreSignalParams *paramsArray, unsigned
int numExtSems, hipStream_t stream)
Signals a set of external semaphore objects.

## See also

Note: This API is currently not supported on Linux.
Parameters
• extSemArray – [in]External semaphores to be waited on
• paramsArray – [in]Array of semaphore parameters
• numExtSems– [in] Number of semaphores to wait on
• stream – [in] Stream to enqueue the wait operations in
Returns
hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
190 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t *extSemArray, const
hipExternalSemaphoreWaitParams *paramsArray, unsigned int
numExtSems, hipStream_t stream)
Waits on a set of external semaphore objects.

## See also

Note: This API is currently not supported on Linux.
Parameters
• extSemArray – [in]External semaphores to be waited on
• paramsArray – [in]Array of semaphore parameters
• numExtSems– [in] Number of semaphores to wait on
• stream – [in] Stream to enqueue the wait operations in
Returns
hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem)
Destroys an external semaphore object and releases any references to the underlying resource. Any outstanding
signals or waits must have completed before the semaphore is destroyed.

## See also

Note: This API is currently not supported on Linux.
Parameters
extSem – [in]handle to an external memory object
Returns
hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
hipError_t hipImportExternalMemory(hipExternalMemory_t *extMem_out, const
hipExternalMemoryHandleDesc *memHandleDesc)
Imports an external memory object.

## See also

Parameters
• extMem_out– [out] Returned handle to an external memory object
• memHandleDesc – [in] Memory import handle descriptor
17.23. Asynchronous Functions 191
HIP Documentation, Release 6.3.42134
Returns
hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
hipError_t hipExternalMemoryGetMappedBuffer(void **devPtr, hipExternalMemory_t extMem, const
hipExternalMemoryBufferDesc *bufferDesc)
Maps a buffer onto an imported memory object.

## See also

Parameters
• devPtr – [out] Returned device pointer to buffer
• extMem – [in] Handle to external memory object
• bufferDesc– [in] Buffer descriptor
Returns
hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem)
Destroys an external memory object.

## See also

Parameters
extMem – [in]External memory object to be destroyed
Returns
hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
hipError_t hipExternalMemoryGetMappedMipmappedArray(hipMipmappedArray_t *mipmap,
hipExternalMemory_t extMem, const
hipExternalMemoryMipmappedArrayDesc
*mipmapDesc)
Maps a mipmapped array onto an external memory object.
Returned mipmapped array must be freed using hipFreeMipmappedArray.

## See also

hipImportExternalMemory, hipDestroyExternalMemory, hipExternalMemoryGetMappedBuffer,
hipFreeMipmappedArray
Parameters
• mipmap – [out] mipmapped array to return
• extMem – [in] external memory object handle
• mipmapDesc – [in] external mipmapped array descriptor
192 Chapter 17. C++ Language Extensions
HIP Documentation, Release 6.3.42134
Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidResourceHandle
17.24 Register Keyword
The register keyword is deprecated in C++, and is silently ignored by both NVCC and HIP-Clang. You can pass the
option -Wdeprecated-registerthe compiler warning message.
17.25 Pragma Unroll

## Unroll with a bounds that is known at compile-time is supported. For example

#pragma unroll 16 /* hint to compiler to unroll next loop by 16 */
for (int i=0; i<16; i++) ...
#pragma unroll 1 /* tell compiler to never unroll the loop */
for (int i=0; i<16; i++) ...
#pragma unroll /* hint to compiler to completely unroll next loop. */
for (int i=0; i<16; i++) ...
17.26 In-Line Assembly
GCN ISA In-line assembly is supported.
There are some usage limitations in ROCm compiler for inline asm support, please refer to Inline ASM statements for
details.
Users can get related background resources on how to use inline assembly for any usage of inline assembly features.
A short example program including an inline assembly statement can be found at inline asm tutorial.
For further usage of special AMD GPU hardware features that are available through assembly, please refer to the ISA
manual for AMDGPU usage, in which AMD GCN is listed from gfx906 to RDNA 3.5.
17.27 C++ Support

## The following C++ features are not supported

• Run-time-type information (RTTI)
• Try/catch

## Partially supported features

• Virtual functions
Virtual functions are not supported if objects containing virtual function tables are passed between GPU’s of different
offload arch’s, e.g. between gfx906 and gfx1030. Otherwise virtual functions are supported.
17.24. Register Keyword 193
HIP Documentation, Release 6.3.42134
17.28 Kernel Compilation

```cpp

hipccnowsupportscompilingC++/HIPkernelstobinarycodeobjects. Thefileformatforbinaryis .cowhichmeans

```

Code Object. The following command builds the code object usinghipcc.
hipcc --genco --offload-arch=[TARGET GPU] [INPUT FILE] -o [OUTPUT FILE]
[TARGET GPU] = GPU architecture

```cpp

[INPUT FILE] = Name of the file containing kernels

```

[OUTPUT FILE] = Name of the generated code object file

```cpp

Note: When using binary code objects is that the number of arguments to the kernel is different on HIP-Clang and
NVCC path. Refer to the HIP module_api sample for differences in the arguments to be passed to the kernel.
17.29 gfx-arch-specific-kernel
Clangdefined‘__gfx*__’ macroscanbeused toexecutegfxarchspecific codesinsidethekernel. Refertothe sample

```

in HIP 14_gpu_arch sample.
194 Chapter 17. C++ Language Extensions

## CHAPTER


## EIGHTEEN


## TABLE COMPARING SYNTAX FOR DIFFERENT COMPUTE APIS

Term CUDA HIP OpenCL
Device int deviceId int deviceId cl_device
Queue cudaStream_t hipStream_t cl_command_queue
Event cudaEvent_t hipEvent_t cl_event
Memory void * void * cl_mem
grid grid NDRange
block block work-group
thread thread work-item
warp warp sub-group

```cpp

Thread-index threadIdx.x threadIdx.x get_local_id(0)

```

Block-index blockIdx.x blockIdx.x get_group_id(0)
Block-dim blockDim.x blockDim.x get_local_size(0)
Grid-dim gridDim.x gridDim.x get_num_groups(0)

```cpp

Device Kernel __global__ __global__ __kernel

```

Device Function __device__ __device__ Implied in device compilation
Host Function __host_ (default) __host_(default) Implied in host compilation
Host + Device Function __host__ __device__ __host__ __device__ No equivalent
Kernel Launch <<< >>> hipLaunchKernel/hipLaunchKernelGGL/<<< >>> clEnqueueNDRangeKernel

```cpp

Global Memory __global__ __global__ __global

```

Group Memory __shared__ __shared__ __local
Constant __constant__ __constant__ __constant
__syncthreads __syncthreads barrier(CLK_LOCAL_MEMFENCE)
Atomic Builtins atomicAdd atomicAdd atomic_add
Precise Math cos(f) cos(f) cos(f)
Fast Math __cos(f) __cos(f) native_cos(f)
Vector float4 float4 float4
195
HIP Documentation, Release 6.3.42134
18.1 Notes
The indexing functions (starting withthread-index) show the terminology for a 1D grid. Some APIs use reverse
order ofxyz / 012 indexing for 3D grids.
196 Chapter 18. Table Comparing Syntax for Different Compute APIs

## CHAPTER


## NINETEEN


## HSA RUNTIME API FOR ROCM

The following functions are located in the https://github.com/ROCm/ROCR-Runtime repository.
hsa_status_t hsa_amd_vmem_address_reserve(void **va, size_t size, uint64_t address, uint64_t flags)
Allocate a reserved address range.
Reserve a virtual address range. The size must be a multiple of the system page size. If it is not possible to
allocate the address specified byaddress, thenva will be a different address range. Address range should be
released by calling hsa_amd_vmem_address_free.
NotethatthisAPIwillbedeprecatedinafuturereleaseandreplacedbyhsa_amd_vmem_address_reserve_align
Parameters
• va– [out] virtual address allocated
• size– [in] of address range requested
• address– [in] requested
• flags – [in] currently unsupported
Return values
• ::HSA_STATUS_SUCCESS – Address range allocated successfully
• ::HSA_STATUS_ERROR_NOT_INITIALIZED– The HSA runtime has not been initialized.
• ::HSA_STATUS_ERROR_OUT_OF_RESOURCES–Insufficientresourcestoallocateanaddress
range of this size.
hsa_status_t hsa_amd_vmem_address_free(void *va, size_t size)
Free a reserved address range.
Free a previously allocated address range. The size must match the size of a previously allocated address range.
Parameters
• va– [out] virtual address to be freed
• size– [in] of address range
Return values
• ::HSA_STATUS_SUCCESS – Address range released successfully
• ::HSA_STATUS_ERROR_INVALID_ALLOCATION– Invalid va specified
• ::HSA_STATUS_ERROR_INVALID_ARGUMENT – Invalid size specified
• ::HSA_STATUS_ERROR_RESOURCE_FREE – Address range is still in use
197
HIP Documentation, Release 6.3.42134
• ::HSA_STATUS_ERROR – Internal unexpected error
hsa_status_t hsa_amd_vmem_handle_create(hsa_amd_memory_pool_t pool, size_t size,
hsa_amd_memory_type_t type, uint64_t flags,
hsa_amd_vmem_alloc_handle_t *memory_handle)
Create a virtual memory handle.
Create a virtual memory handle within this pool size must be a aligned to allocation granule size
for this memory pool, see HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE To min-
imize internal memory fragmentation, align the size to the recommended allocation granule size, see

## HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE

Parameters
• pool– [in] memory to use
• size– [in] of the memory allocation
• type– [in] of memory
• flags – [in] - currently unsupported
• memory_handle– [out] - handle for the allocation
Return values
• ::HSA_STATUS_SUCCESS – memory allocated successfully
• ::HSA_STATUS_ERROR_NOT_INITIALIZED– The HSA runtime has not been initialized.
• ::HSA_STATUS_ERROR_INVALID_ARGUMENT – Invalid arguments
• ::HSA_STATUS_ERROR_INVALID_ALLOCATION–Thismemorypooldoesnotsupportallo-
cations
• ::HSA_STATUS_ERROR_OUT_OF_RESOURCES– Insufficient resources to allocate this mem-
ory
hsa_status_t hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t memory_handle)
Release a virtual memory handle.
Parameters
memory– [in] handle that was previously allocated
Return values
• ::HSA_STATUS_SUCCESS – Address range allocated successfully
• ::HSA_STATUS_ERROR_INVALID_ALLOCATION– Invalid memory handle
hsa_status_t hsa_amd_vmem_map(void *va, size_t size, size_t in_offset, hsa_amd_vmem_alloc_handle_t
memory_handle, uint64_t flags)
Map a virtual memory handle.
Map a virtual memory handle to a reserved address range. The virtual address requested must be within a
previouslyreservedaddressrange. vaand(va+size)mustbemustbewithin(va+size)ofthepreviousallocated
addressrange. sizemustbeequaltosizeofthe memory_handlehsa_amd_vmem_set_accessneedstobecalled
to make the memory accessible to specific agents
Parameters
• va– [in] virtual address range where memory will be mapped
• size– [in] of memory mapping
• in_offset – [in] offset into memory. Currently unsupported
198 Chapter 19. HSA Runtime API for ROCm
HIP Documentation, Release 6.3.42134
• memory_handle– [in] virtual memory handle to be mapped
• flags. – [in]Currently unsupported
Return values
• ::HSA_STATUS_SUCCESS – Memory mapped successfully
• ::HSA_STATUS_ERROR_INVALID_ARGUMENT – va, size or memory_handle are invalid
• ::HSA_STATUS_ERROR_OUT_OF_RESOURCES – Insufficient resources
• ::HSA_STATUS_ERROR – Unexpected internal error
hsa_status_t hsa_amd_vmem_unmap(void *va, size_t size)
Unmap a virtual memory handle.
Unmap previously mapped virtual address range
Parameters
• va– [in] virtual address range where memory will be mapped
• size– [in] of memory mapping
Return values
• ::HSA_STATUS_SUCCESS – Memory backing unmapped successfully
• ::HSA_STATUS_ERROR_INVALID_ALLOCATION– memory_handle is invalid
• ::HSA_STATUS_ERROR_INVALID_ARGUMENT – size is invalid
• ::HSA_STATUS_ERROR – Unexpected internal error
hsa_status_t hsa_amd_vmem_set_access(void *va, size_t size, const hsa_amd_memory_access_desc_t *desc,
size_t desc_cnt)
Make a memory mapping accessible.
Make previously mapped virtual address accessible to specific agents.size must be equal to size of previ-
ously mapped virtual memory handle. Calling hsa_amd_vmem_set_access multiple times on the sameva will
overwrite previous permissions for all agents
Parameters
• va– [in] previously mapped virtual address
• size– [in] of memory mapping
• desc– [in] list of access permissions for each agent
• desc_cnt – [in]number of elements in desc
Return values

## • ::HSA_STATUS_SUCCESS –

• ::HSA_STATUS_ERROR_INVALID_ARGUMENT – va, size or memory_handle are invalid
• ::HSA_STATUS_ERROR_INVALID_ALLOCATION– memory_handle is invalid
• ::HSA_STATUS_ERROR_OUT_OF_RESOURCES – Insufficient resources
• ::HSA_STATUS_ERROR_INVALID_AGENT – Invalid agent in desc
• ::HSA_STATUS_ERROR – Unexpected internal error
199
HIP Documentation, Release 6.3.42134
hsa_status_t hsa_amd_vmem_get_access(void *va, hsa_access_permission_t *perms, hsa_agent_t agent_handle)
Get current access permissions for memory mapping.
Get access permissions for memory mapping for specific agent.
Parameters
• va– [in] previously mapped virtual address
• perms – [in] current permissions
• agent_handle– [in] agent
Return values

## • ::HSA_STATUS_SUCCESS –

• ::HSA_STATUS_ERROR_INVALID_AGENT – Invalid agent
• ::HSA_STATUS_ERROR_INVALID_ALLOCATION – va is not mapped or permissions never
set for this agent
• ::HSA_STATUS_ERROR – Unexpected internal error
hsa_status_t hsa_amd_vmem_export_shareable_handle(int *dmabuf_fd, hsa_amd_vmem_alloc_handle_t
handle, uint64_t flags)
Get an exportable shareable handle.
Getanexportableshareablehandleforamemory_handle. Thisshareablhandlecanthenbeusedtore-createavir-
tual memory handle using hsa_amd_vmem_import_shareable_handle. The shareable handle can be transferred
usingmechanismsthatsupportposixfiledescriptorsOnceallshareablehandlesareclosed, thememory_handle
is released.
Parameters
• dmabuf_fd – [out] shareable handle
• handle – [in] previously allocated virtual memory handle
• flags – [in]Currently unsupported
Return values

## • ::HSA_STATUS_SUCCESS –

• ::HSA_STATUS_ERROR_INVALID_ALLOCATION – Invalid memory handle
• ::HSA_STATUS_ERROR_OUT_OF_RESOURCES – Out of resources
• ::HSA_STATUS_ERROR – Unexpected internal error
hsa_status_t hsa_amd_vmem_import_shareable_handle(int dmabuf_fd, hsa_amd_vmem_alloc_handle_t
*handle)
Import a shareable handle.
Importashareablehandleforamemoryhandle. Importingashareablehandlethathasbeenclosedandreleased
results in undefined behavior.
Parameters
• dmabuf_fd–[in]shareablehandleexportedwithhsa_amd_vmem_export_shareable_handle
• handle – [out] virtual memory handle
Return values

## • ::HSA_STATUS_SUCCESS –

200 Chapter 19. HSA Runtime API for ROCm
HIP Documentation, Release 6.3.42134
• ::HSA_STATUS_ERROR_INVALID_ALLOCATION– Invalid memory handle
• ::HSA_STATUS_ERROR_OUT_OF_RESOURCES – Out of resources
• ::HSA_STATUS_ERROR – Unexpected internal error
hsa_status_t hsa_amd_vmem_retain_alloc_handle(hsa_amd_vmem_alloc_handle_t *memory_handle, void
*addr)
Returns memory handle for mapped memory.
Return a memory handle for previously mapped memory. The handle will be the same value of handle
used to map the memory. The returned handle must be released with corresponding number of calls to
hsa_amd_vmem_handle_release.
Parameters
• memory_handle – [out] memory handle for this mapped address
• mapped – [in]address
Return values

## • ::HSA_STATUS_SUCCESS –

• ::HSA_STATUS_ERROR_INVALID_ALLOCATION– Invalid address
hsa_status_t hsa_amd_vmem_get_alloc_properties_from_handle(hsa_amd_vmem_alloc_handle_t
memory_handle, hsa_amd_memory_pool_t
*pool, hsa_amd_memory_type_t *type)
Returns the current allocation properties of a handle.
Returns the allocation properties of an existing handle
Parameters
• memory_handle – [in] memory handle to be queried
• pool– [out] memory pool that owns this handle
• memory – [out] type
Return values

## • ::HSA_STATUS_SUCCESS –

• ::HSA_STATUS_ERROR_INVALID_ALLOCATION– Invalid memory_handle
201
HIP Documentation, Release 6.3.42134
202 Chapter 19. HSA Runtime API for ROCm

## CHAPTER


## TWENTY


## HIP DEPRECATED RUNTIME API FUNCTIONS

Several of our API functions have been flagged for deprecation. Using the following functions results in errors and
unexpected results, so we encourage you to update your code accordingly.
20.1 Context management
CUDAsupportscuCtxAPI,whichisthedriverAPIthatdefines“Context”and“Devices”asseparateentities. Context
contains a single device, and a device can theoretically have multiple contexts. HIP initially added limited support for
theseAPIsinordertofacilitateportingfromexistingdrivercodes. TheseAPIsarenowmarkedasdeprecatedbecause
there are better alternate interfaces (such ashipSetDeviceor the stream API) to achieve these functions.
• hipCtxCreate
• hipCtxDestroy
• hipCtxPopCurrent
• hipCtxPushCurrent
• hipCtxSetCurrent
• hipCtxGetCurrent
• hipCtxGetDevice
• hipCtxGetApiVersion
• hipCtxGetCacheConfig
• hipCtxSetCacheConfig
• hipCtxSetSharedMemConfig
• hipCtxGetSharedMemConfig
• hipCtxSynchronize
• hipCtxGetFlags
• hipCtxEnablePeerAccess
• hipCtxDisablePeerAccess
• hipDevicePrimaryCtxGetState
• hipDevicePrimaryCtxRelease
• hipDevicePrimaryCtxRetain
• hipDevicePrimaryCtxReset
203
HIP Documentation, Release 6.3.42134
• hipDevicePrimaryCtxSetFlags
20.2 Memory management

```cpp

• hipMallocHost(replaced withhipHostMalloc)

```

• hipMemAllocHost(replaced withhipHostMalloc)

```cpp

• hipMemcpyToArray
• hipMemcpyFromArray

```

20.3 Profiler control
• hipProfilerStart(use roctracer/rocTX)
• hipProfilerStop(use roctracer/rocTX)
20.4 Texture management
• hipGetTextureReference
• hipTexRefSetAddressMode
• hipTexRefSetArray
• hipTexRefSetFilterMode
• hipTexRefSetFlags
• hipTexRefSetFormat
• hipTexRefGetAddress
• hipTexRefGetAddressMode
• hipTexRefGetFilterMode
• hipTexRefGetFlags
• hipTexRefGetFormat
• hipTexRefGetMaxAnisotropy
• hipTexRefGetMipmapFilterMode
• hipTexRefGetMipmapLevelBias
• hipTexRefGetMipmapLevelClamp
• hipTexRefGetMipMappedArray
• hipTexRefSetAddress
• hipTexRefSetAddress2D
• hipTexRefSetMaxAnisotropy
• hipTexRefSetBorderColor
• hipTexRefSetMipmapFilterMode
204 Chapter 20. HIP deprecated runtime API functions
HIP Documentation, Release 6.3.42134
• hipTexRefSetMipmapLevelBias
• hipTexRefSetMipmapLevelClamp
• hipTexRefSetMipmappedArray
• hipTexRefGetBorderColor
• hipTexRefGetArray
• hipBindTexture
• hipBindTexture2D
• hipBindTextureToArray
• hipGetTextureAlignmentOffset
• hipUnbindTexture
• hipBindTextureToMipmappedArray
20.4. Texture management 205
HIP Documentation, Release 6.3.42134
206 Chapter 20. HIP deprecated runtime API functions

## CHAPTER


## TWENTYONE


## FP8 NUMBERS

FP8 numbers were introduced to accelerate deep learning inferencing. They provide higher throughput of matrix
operations because the smaller size allows more of them in the available fixed memory.
HIP has two FP8 number representations calledFP8-OCP and FP8-FNUZ.
Open Compute Project(OCP) number definition can be found here.
Definition of FNUZ: fnuz suffix means only finite and NaN values are supported. Unlike other types, Inf are not
supported. NaN is when sign bit is set and all other exponent and mantissa bits are 0. All other values are finite. This
provides one extra value of exponent and adds to the range of supported FP8 numbers.
21.1 FP8 Definition
FP8 numbers are composed of a sign, an exponent and a mantissa. Their sizes are dependent on the format. There are
two formats of FP8 numbers, E4M3 and E5M2.
• E4M3: 1 bit sign, 4 bit exponent, 3 bit mantissa
• E5M2: 1 bit sign, 5 bit exponent, 2 bit mantissa
21.2 HIP Header
HIP header defined the FP8 ocp/fnuz numbers here.
21.3 Supported Devices
Table 21.1: Supported devices for fp8 numbers
Device Type FNUZ FP8 OCP FP8
Host Yes Yes
gfx940/gfx941/gfx942 Yes No
gfx1200/gfx1201 No Yes
207
HIP Documentation, Release 6.3.42134
21.4 Usage
To use the FP8 numbers inside HIP programs.

```cpp

#include <hip/hip_fp8.h>

## FP8 numbers can be used on CPU side


```

__hip_fp8_storage_t convert_float_to_fp8(
float in, /* Input val */
__hip_fp8_interpretation_t interpret, /* interpretation of number E4M3/E5M2 */
__hip_saturation_t sat /* Saturation behavior */
) {
return __hip_cvt_float_to_fp8(in, sat, interpret);
}

```cpp

The same can be done in kernels as well.

```

__device__ __hip_fp8_storage_t d_convert_float_to_fp8(
float in,
__hip_fp8_interpretation_t interpret,
__hip_saturation_t sat) {
return __hip_cvt_float_to_fp8(in, sat, interpret);
}
An important thing to note here is if you use this on MI300 GPU, it will be fnuz number but on any other GPU it will
be an OCP number.
The following code example does roundtrip FP8 conversions on both the CPU and GPU and compares the results.

```cpp

#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

```

#define hip_check(hip_call) \
{ \
auto hip_res = hip_call; \
if (hip_res != hipSuccess) { \
std::cerr << "Failed in hip call: " << #hip_call \
<< " with error: " << hipGetErrorName(hip_res) << std::endl; \
std::abort(); \
} \
}
__device__ __hip_fp8_storage_t d_convert_float_to_fp8(
float in, __hip_fp8_interpretation_t interpret, __hip_saturation_t sat) {
return __hip_cvt_float_to_fp8(in, sat, interpret);
}
__device__ float d_convert_fp8_to_float(float in,
__hip_fp8_interpretation_t interpret) {
__half hf = __hip_cvt_fp8_to_halfraw(in, interpret);
return hf;
(continues on next page)
208 Chapter 21. FP8 Numbers
HIP Documentation, Release 6.3.42134
(continued from previous page)
}

```cpp

__global__ void float_to_fp8_to_float(float *in,

```

__hip_fp8_interpretation_t interpret,
__hip_saturation_t sat, float *out,
size_t size) {

```cpp

int i = threadIdx.x;

```

if (i < size) {
auto fp8 = d_convert_float_to_fp8(in[i], interpret, sat);
out[i] = d_convert_fp8_to_float(fp8, interpret);
}
}
__hip_fp8_storage_t
convert_float_to_fp8(float in, /* Input val */
__hip_fp8_interpretation_t
interpret, /* interpretation of number E4M3/E5M2 */
__hip_saturation_t sat /* Saturation behavior */
) {
return __hip_cvt_float_to_fp8(in, sat, interpret);
}
float convert_fp8_to_float(
__hip_fp8_storage_t in, /* Input val */
__hip_fp8_interpretation_t
interpret /* interpretation of number E4M3/E5M2 */
) {
__half hf = __hip_cvt_fp8_to_halfraw(in, interpret);
return hf;
}
int main() {
constexpr size_t size = 32;
hipDeviceProp_t prop;
hip_check(hipGetDeviceProperties(&prop, 0));
bool is_supported = (std::string(prop.gcnArchName).find("gfx94") !=␣
˓→std::string::npos) || // gfx94x
(std::string(prop.gcnArchName).find("gfx120") !=␣
˓→std::string::npos); // gfx120x
if(!is_supported) {
std::cerr << "Need a gfx94x or gfx120x, but found: " << prop.gcnArchName <<␣
˓→std::endl;
std::cerr << "No device conversions are supported, only host conversions are␣
˓→supported." << std::endl;
return -1;
}
const __hip_fp8_interpretation_t interpret = (std::string(prop.gcnArchName).find(
˓→"gfx94") != std::string::npos)
? __HIP_E4M3_FNUZ // gfx94x
: __HIP_E4M3; // gfx120x
constexpr __hip_saturation_t sat = __HIP_SATFINITE;
(continues on next page)
21.4. Usage 209
HIP Documentation, Release 6.3.42134
(continued from previous page)
std::vector<float> in;
in.reserve(size);
for (size_t i = 0; i < size; i++) {
in.push_back(i + 1.1f);
}
std::cout << "Converting float to fp8 and back..." << std::endl;
// CPU convert
std::vector<float> cpu_out;
cpu_out.reserve(size);
for (const auto &fval : in) {
auto fp8 = convert_float_to_fp8(fval, interpret, sat);
cpu_out.push_back(convert_fp8_to_float(fp8, interpret));
}
// GPU convert
float *d_in, *d_out;

```cpp

hip_check(hipMalloc(&d_in, sizeof(float) * size));
hip_check(hipMalloc(&d_out, sizeof(float) * size));
hip_check(hipMemcpy(d_in, in.data(), sizeof(float) * in.size(),
hipMemcpyHostToDevice));

```

float_to_fp8_to_float<<<1, size>>>(d_in, interpret, sat, d_out, size);
std::vector<float> gpu_out(size, 0.0f);

```cpp

hip_check(hipMemcpy(gpu_out.data(), d_out, sizeof(float) * gpu_out.size(),
hipMemcpyDeviceToHost));

```

hip_check(hipFree(d_in));
hip_check(hipFree(d_out));
// Validation
for (size_t i = 0; i < size; i++) {
if (cpu_out[i] != gpu_out[i]) {
std::cerr << "cpu round trip result: " << cpu_out[i]
<< " - gpu round trip result: " << gpu_out[i] << std::endl;
std::abort();
}
}
std::cout << "...CPU and GPU round trip convert matches." << std::endl;
}
There are C++ style classes available as well.
__hip_fp8_e4m3_fnuz fp8_val(1.1f); // gfx94x
__hip_fp8_e4m3 fp8_val(1.1f); // gfx120x

## Each type of FP8 number has its own class

• __hip_fp8_e4m3
• __hip_fp8_e5m2
210 Chapter 21. FP8 Numbers
HIP Documentation, Release 6.3.42134
• __hip_fp8_e4m3_fnuz
• __hip_fp8_e5m2_fnuz
There is support of vector of FP8 types.
• __hip_fp8x2_e4m3: holds 2 values of OCP FP8 e4m3 numbers
• __hip_fp8x4_e4m3: holds 4 values of OCP FP8 e4m3 numbers
• __hip_fp8x2_e5m2: holds 2 values of OCP FP8 e5m2 numbers
• __hip_fp8x4_e5m2: holds 4 values of OCP FP8 e5m2 numbers
• __hip_fp8x2_e4m3_fnuz: holds 2 values of FP8 fnuz e4m3 numbers
• __hip_fp8x4_e4m3_fnuz: holds 4 values of FP8 fnuz e4m3 numbers
• __hip_fp8x2_e5m2_fnuz: holds 2 values of FP8 fnuz e5m2 numbers
• __hip_fp8x4_e5m2_fnuz: holds 4 values of FP8 fnuz e5m2 numbers
FNUZ extensions will be available on MI300 only.
21.4. Usage 211
HIP Documentation, Release 6.3.42134
212 Chapter 21. FP8 Numbers

## CHAPTER


## TWENTYTWO


## LICENSE

Copyright © 2008 - 2024 Advanced Micro Devices, Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documen-
tation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use,
copy,modify,merge,publish,distribute,sublicense,and/orsellcopiesoftheSoftware,andtopermitpersonstowhom

## the Software is furnished to do so, subject to the following conditions

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

## THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,


## INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PAR-


## TICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT


## HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION


## OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFT-


## WARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

213
HIP Documentation, Release 6.3.42134
214 Chapter 22. License

## INDEX


## H

hipArray3DCreate (C++ function), 181
hipArray3DGetDescriptor(C++ function), 183
hipArrayCreate (C++ function), 181
hipArrayDestroy(C++ function), 181
hipArrayGetDescriptor(C++ function), 183
hipArrayGetInfo(C++ function), 182
hipDestroyExternalMemory (C++ function), 192
hipDestroyExternalSemaphore (C++ function), 191
hipDeviceCanAccessPeer (C++ function), 159
hipDeviceDisablePeerAccess(C++ function), 160
hipDeviceEnablePeerAccess (C++ function), 159
hipDeviceGetStreamPriorityRange (C++ function),
145
hipDrvMemcpy3D (C++ function), 189
hipDrvMemcpy3DAsync (C++ function), 189
hipDrvPointerGetAttributes(C++ function), 162
hipExternalMemoryGetMappedBuffer (C++ func-
tion), 192
hipExternalMemoryGetMappedMipmappedArray
(C++ function), 192
hipExtMallocWithFlags(C++ function), 163
hipExtStreamCreateWithCUMask(C++function),148
hipExtStreamGetCUMask(C++ function), 149
hipFree(C++ function), 167
hipFreeArray(C++ function), 182
hipFreeAsync(C++ function), 150
hipFreeHost (C++ function), 167
hipGetProcAddress (C++ function), 174
hipGetSymbolAddress (C++ function), 174
hipGetSymbolSize (C++ function), 174
hipHostAlloc(C++ function), 164
hipHostGetDevicePointer(C++ function), 164
hipHostGetFlags(C++ function), 165
hipHostMalloc (C++ function), 163
hipHostRegister(C++ function), 165
hipHostUnregister (C++ function), 166
hipImportExternalMemory(C++ function), 191
hipImportExternalSemaphore(C++ function), 190

```cpp

hipMalloc (C++ function), 162
hipMalloc3D (C++ function), 181
hipMalloc3DArray (C++ function), 182
hipMallocArray (C++ function), 180
hipMallocAsync (C++ function), 149
hipMallocFromPoolAsync (C++ function), 155
hipMallocPitch (C++ function), 166

```

hipMemAllocPitch (C++ function), 166

```cpp

hipMemcpy (C++ function), 167
hipMemcpy2D (C++ function), 183
hipMemcpy2DArrayToArray (C++ function), 186
hipMemcpy2DAsync (C++ function), 184
hipMemcpy2DFromArray (C++ function), 187
hipMemcpy2DFromArrayAsync (C++ function), 187
hipMemcpy2DToArray (C++ function), 185
hipMemcpy2DToArrayAsync (C++ function), 185
hipMemcpy3D (C++ function), 188
hipMemcpy3DAsync (C++ function), 189
hipMemcpyAsync (C++ function), 176
hipMemcpyAtoA (C++ function), 171
hipMemcpyAtoD (C++ function), 170
hipMemcpyAtoH (C++ function), 188
hipMemcpyAtoHAsync (C++ function), 172
hipMemcpyDtoA (C++ function), 170
hipMemcpyDtoD (C++ function), 169
hipMemcpyDtoDAsync (C++ function), 172
hipMemcpyDtoH (C++ function), 169
hipMemcpyDtoHAsync (C++ function), 172
hipMemcpyFromSymbol (C++ function), 175
hipMemcpyFromSymbolAsync (C++ function), 175
hipMemcpyHtoA (C++ function), 188
hipMemcpyHtoAAsync (C++ function), 173
hipMemcpyHtoD (C++ function), 168
hipMemcpyHtoDAsync (C++ function), 171
hipMemcpyParam2D (C++ function), 184
hipMemcpyParam2DAsync (C++ function), 184
hipMemcpyPeer (C++ function), 160
hipMemcpyPeerAsync (C++ function), 160
hipMemcpyToSymbol (C++ function), 174
hipMemcpyToSymbolAsync (C++ function), 175
hipMemcpyWithStream (C++ function), 168

```

hipMemGetAddressRange (C++ function), 160
hipMemGetInfo (C++ function), 180
hipMemPoolCreate (C++ function), 154
hipMemPoolDestroy (C++ function), 155
215
HIP Documentation, Release 6.3.42134
hipMemPoolExportPointer(C++ function), 158
hipMemPoolExportToShareableHandle (C++ func-
tion), 156
hipMemPoolGetAccess (C++ function), 154
hipMemPoolGetAttribute (C++ function), 152
hipMemPoolImportFromShareableHandle (C++
function), 157
hipMemPoolImportPointer(C++ function), 158
hipMemPoolSetAccess (C++ function), 153
hipMemPoolSetAttribute (C++ function), 151
hipMemPoolTrimTo (C++ function), 151
hipMemPtrGetInfo (C++ function), 180
hipMemset (C++ function), 177
hipMemset2D (C++ function), 179
hipMemset2DAsync (C++ function), 179
hipMemset3D (C++ function), 179
hipMemset3DAsync (C++ function), 179
hipMemsetAsync (C++ function), 178
hipMemsetD16(C++ function), 177
hipMemsetD16Async (C++ function), 177
hipMemsetD32(C++ function), 178
hipMemsetD32Async (C++ function), 178
hipMemsetD8 (C++ function), 177
hipMemsetD8Async (C++ function), 177
hipModuleGetGlobal(C++ function), 173
hipPointerGetAttribute (C++ function), 162
hipPointerGetAttributes(C++ function), 161
hipPointerSetAttribute (C++ function), 161
hipSignalExternalSemaphoresAsync (C++ func-
tion), 190
hipStreamAddCallback (C++ function), 149
hipStreamCallback_t (C++ type), 144
hipStreamCreate(C++ function), 144
hipStreamCreateWithFlags (C++ function), 144
hipStreamCreateWithPriority (C++ function), 145
hipStreamDestroy (C++ function), 145
hipStreamGetDevice(C++ function), 148
hipStreamGetFlags (C++ function), 147
hipStreamGetPriority (C++ function), 147
hipStreamQuery (C++ function), 146
hipStreamSynchronize (C++ function), 146
hipStreamWaitEvent(C++ function), 147
hipWaitExternalSemaphoresAsync (C++ function),
190
hsa_amd_vmem_address_free (C++ function), 197
hsa_amd_vmem_address_reserve(C++function),197
hsa_amd_vmem_export_shareable_handle (C++
function), 200
hsa_amd_vmem_get_access(C++ function), 199
hsa_amd_vmem_get_alloc_properties_from_handle
(C++ function), 201
hsa_amd_vmem_handle_create(C++ function), 198
hsa_amd_vmem_handle_release (C++ function), 198
hsa_amd_vmem_import_shareable_handle (C++
function), 200
hsa_amd_vmem_map (C++ function), 198
hsa_amd_vmem_retain_alloc_handle (C++ func-
tion), 201
hsa_amd_vmem_set_access (C++ function), 199
hsa_amd_vmem_unmap (C++ function), 199

## U

USE_PEER_NON_UNIFIED (C macro), 161
216 Index