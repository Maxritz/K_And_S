# Target and Evidence: vulkan → gfx1030

**Target arch:** `gfx1030`
**Host OS:** `linux`

**Facts provenance:**
- `toolchain.glslc`: host probe disabled (probe_host=False)
- `runtime.loader`: host probe disabled (probe_host=False)
- `toolchain.spirvval`: host probe disabled (probe_host=False)
- `capability.coopmat`: `gfx1030` is not in ARCH_MAP.json; probe on the device
- `capability.subgroup`: `gfx1030.['subgroup_size', 'vulkan.subgroup_size']` not in ARCH_MAP.json
- `sync.timeline`: manual probe not run
- `artifact.spirv`: requires a build: spirv-dis + spirv-val
- `pcie.rebar`: host probe disabled (probe_host=False)

*`ARCH_MAP.json` is one source among four, ranked below measurement, below a shared feed, and below user assertion.*
