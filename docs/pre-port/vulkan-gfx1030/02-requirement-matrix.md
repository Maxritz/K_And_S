# Requirement Matrix
Pareto total score: 18.7
Vital few: ['capability.coopmat', 'toolchain.glslc', 'capability.subgroup', 'pcie.rebar', 'runtime.loader', 'artifact.spirv']

| # | ID | Kind | I×L | Score | Cum. | Status | Probe | Fallback |
|---|---|---|---|---|---|---|---|---|
| 1 | `capability.coopmat` | capability | 5.0×0.9 | 4.5 | 24% | UNVALIDATED | vulkaninfo --summary: VK_KHR_cooperative_matrix; else ARCH_MAP vulkan.cooperative_matrix | DP4a-style SPIR-V (VK_KHR_shader_integer_dot_product) matmul |
| 2 | `toolchain.glslc` | tool | 5.0×0.8 | 4.0 | 46% | UNVALIDATED | glslc --version | glslangValidator; both must target the same SPIR-V version the driver accepts |
| 3 | `capability.subgroup` | info | 4.0×0.5 | 2.0 | 56% | NOT RUN | vulkaninfo subgroupSize, or the device's own report | declare SUBGROUP_SIZE at build time and assert it at load |
| 4 | `pcie.rebar` | pcie | 4×0.5 | 2.0 | 67% | UNVALIDATED | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | HOST_VISIBLE_BIT heap | host-visible staging buffer + device-local copy, one chunk in flight per queue slot |
| 5 | `runtime.loader` | tool | 4.0×0.5 | 2.0 | 78% | UNVALIDATED | vulkaninfo --summary | install the ICD for the target vendor; a missing ICD looks like an unsupported feature |
| 6 | `artifact.spirv` | artifact | 3.0×0.5 | 1.5 | 86% | NOT RUN | spirv-dis + spirv-val | treat an unvalidated module as unverified |
| 7 | `toolchain.spirvval` | tool | 3.0×0.5 | 1.5 | 94% | UNVALIDATED | spirv-val --version | let the driver report; a driver error names no line, so keep the module small |
| 8 | `sync.timeline` | manual | 3.0×0.4 | 1.2 | 100% | NOT RUN | one capture with two queues | single queue with explicit barriers; slower but ordered |