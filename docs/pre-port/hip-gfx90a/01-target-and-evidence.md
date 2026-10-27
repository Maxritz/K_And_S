# Target and Evidence: hip → gfx90a

**Target arch:** `gfx90a`
**Host OS:** `linux`

**Facts provenance:**
- `toolchain.hipcc`: host probe disabled (probe_host=False)
- `toolchain.rocminfo`: host probe disabled (probe_host=False)
- `toolchain.objdump`: host probe disabled (probe_host=False)
- `capability.dot4`: `gfx90a` is not in ARCH_MAP.json; probe on the device
- `capability.matrix`: `gfx90a` is not in ARCH_MAP.json; probe on the device
- `capability.wave`: `gfx90a.wave_size` not in ARCH_MAP.json
- `runtime.finegrained`: manual probe not run
- `artifact.hsaco`: requires a build: code_object_report(path, arch)
- `host.windows_rocm`: host is `linux`
- `pcie.rebar`: host probe disabled (probe_host=False)

*`ARCH_MAP.json` is one source among four, ranked below measurement, below a shared feed, and below user assertion.*
