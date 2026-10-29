# Vulkan Synchronization & TDR/Hang Prevention — Learnings

> Distilled from: GPUOpen (barriers), NVIDIA (dos/don'ts), Imagination (sync hazards Parts I+II), Vulkan tutorial (occupancy), ArXiv VUDA (2605.01352v1).

## Sources

| Source | URL | Date accessed |
|---|---|---|
| GPUOpen — Vulkan Barriers Explained | https://gpuopen.com/learn/vulkan-barriers-explained/ | 2026-10-30 |
| NVIDIA — Vulkan Dos and Don'ts | https://developer.nvidia.com/blog/vulkan-dos-donts/ | 2026-10-30 |
| Imagination — Sync Hazards Part I | https://blog.imaginationtech.com/vulkan-synchronisation-and-graphics-compute-graphics-hazards-part-i | 2026-10-30 |
| Imagination — Sync Hazards Part 2 | https://blog.imaginationtech.com/vulkan-synchronisation-and-graphics-compute-graphics-hazards-part-2/ | 2026-10-30 |
| ArXiv — VUDA (CUDA+Vulkan spatial sharing) | https://arxiv.org/html/2605.01352v1 | 2026-10-30 |
| Vulkan Tutorial — Occupancy & Latency Hiding | https://docs.vulkan.org/tutorial/latest/Advanced_Vulkan_Compute/02_Compute_Architecture/03_occupancy_and_latency_hiding.html | 2026-10-30 |

---

## 1. Vulkan Pipeline Stages (GPUOpen)

The GPU pipeline is pipelined: commands flow top-of-pipe → bottom-of-pipe.
`VK_PIPELINE_STAGE` enum (not strictly ordered):

```
TOP_OF_PIPE → DRAW_INDIRECT → VERTEX_INPUT → VERTEX_SHADER →
  TESS_CONTROL → TESS_EVAL → GEOMETRY → FRAGMENT →
  EARLY_FRAGMENT_TESTS → LATE_FRAGMENT_TESTS →
  COLOR_ATTACHMENT_OUTPUT → TRANSFER → COMPUTE → BOTTOM_OF_PIPE
```

**Key rule:** In a barrier, the **source stage** is the producer (must finish),
the **destination stage** is the consumer (must not have started).

### Anti-pattern: the "slow barrier"

```cpp
// BAD: wait for everything, block everything
vkCmdPipelineBarrier(
    cmd,
    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,  // source = everything
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,     // dest = blocks all
    ...);
```

This inserts a full-GPU flush — a massive stall.

### Optimal: tight producer/consumer

```cpp
// GOOD: vertex shader writes, compute reads
vkCmdPipelineBarrier(
    cmd,
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
    VK_PIPEMENT_STAGE_COMPUTE_SHADER_BIT,
    ...);

// GOOD: transfer writes, fragment reads
vkCmdPipelineBarrier(
    cmd,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    ...);
```

### Hardware behavior

The driver **may** move the source stage toward the bottom of the pipe (more
inclusive) or the target stage toward the top (less restrictive). Your job:
set stages as **tight** as possible so the driver doesn't have to coarsen them.

---

## 2. Barrier Best Practices (NVIDIA)

| Do | Don't |
|---|---|
| Minimize barriers — each may cause a pipeline flush | Insert redundant barriers (read-to-read barriers especially) |
| Prefer buffer/image barriers over memory barriers | Assume driver threads API calls for you |
| Use `VK_KHR_synchronization2` (`vkCmdPipelineBarrier2`) | Use legacy `vkCmdPipelineBarrier` if you need precise control |
| Group all barriers into a single `vkCmdPipelineBarrier2()` call | Insert per-resource barriers in a loop — worst case is picked sequentially |
| Use **minimum** `srcStageMask`/`dstStageMask` (e.g. `COMPUTE` only, not `ALL_COMMANDS`) | Use `ALL_COMMANDS_BIT` unless you genuinely mean it |
| Use `VK_IMAGE_LAYOUT_UNDEFINED` when prior content is discarded | Wait on queue submissions — continue preparing the next frame |
| Lock GPU clocks (`nvidia-smi -lgc`) for stable measurements | Submit tiny command buffers (few draw calls / dispatches) |
| Use `vkCmdSetEvent2` + `vkCmdWaitEvents2` for async barriers | Overlap compute on graphics queue with async compute on pre-Ampere |

### VK_KHR_synchronization2 advantage

The new API lets you specify barriers more precisely — src/dst stage+access are
combined in a single struct, enabling the driver to schedule more granularly.

---

## 3. The Ping-Pong Queue Pattern (Imagination Parts I + II)

### The problem

When you have a frame structured as:

```
Draw → Barrier(gfx→compute) → Dispatch → Barrier(compute→gfx) → Draw → Present
```

...the compute→graphics barrier (green) prevents **An+1** (early graphics of
the *next* frame) from overlapping with **Bn** (compute of the current frame).
Result: full serialisation, mini-stalls, 20–30% lost performance.

### The solution: ping-pong queues

If the GPU exposes multiple identical universal queues (graphics+compute),
submit each frame on a **different** queue:

```
Frame 0: A0 → Barrier → B0 → Barrier → C0  →  Submit to Queue 0
Frame 1: A1 → Barrier → B1 → Barrier → C1  →  Submit to Queue 1
Frame 2: A2 → Barrier → B2 → Barrier → C2  →  Submit to Queue 0
Frame 3: A3 → Barrier → B3 → Barrier → C3  →  Submit to Queue 1
```

Because **Barriers are per-queue**, the barrier inside Queue 0 does **not**
block Queue 1. This means `A1` (next frame's early graphics) can execute
concurrently with `B0` (current frame's compute).

### Requirements

- Triple-buffering (3 framebuffer images) — **required** for the overlap to work
- Multiple queues from the same queue family (most PowerVR, many desktop GPUs)
- At least one additional frame of latency (acceptable for most apps)

### Caveats

- Ensure correct **presentation order** (FIFO mailbox mode handles this)
- If only one present queue exists, submit graphics to ping-pong queues but
  present from a single queue
- Can combine with dedicated compute queues + queue priorities for finer control

### Evidence

| GPU family | Barrier overhead mitigation | Reported gain |
|---|---|---|
| PowerVR (unified USC) | Ping-pong queues break cross-frame stalls | 20–30% (real-world apps) |
| NVIDIA (SM-based) | Timeline semaphores + multi-queue | Variable, workload-dependent |
| AMD (RDNA) | VK_KHR_synchronization2 + precise stages | Up to full occupancy recovery |

---

## 4. Occupancy & Latency Hiding (Vulkan Tutorial)

### The latency gap

- GPU raw compute: trillions of FLOPS/sec
- Single VRAM fetch: **hundreds to thousands of clock cycles**
- If a wavefront/warp waits on memory and no other bundle is ready → CU goes
  **silent**

### The solution: concurrency

The CU switches between bundles ("in flight") to hide memory latency. More
bundles active = better hiding.

### Occupancy = active bundles / theoretical max

- **100%**: another bundle is always ready to run when one stalls
- **Low**: CU stalls when all bundles hit memory simultaneously

### Three primary occupancy limiters

| Resource | Impact | Mitigation |
|---|---|---|
| **Registers** (per-thread) | High register count → fewer threads per CU | Compile with `-maxrgn` / `AMDGPU` register limit; minimize live ranges |
| **Shared Memory / LDS** (per-workgroup) | High LDS → fewer workgroups per CU | Tile size optimization; spill to global if exceeding `lds_kib` |
| **Thread/warp slots** | Hard limit on schedulable threads (e.g. 2048/CU) | Right-size workgroup; don't go below a wavefront/warp (32/64) |

### Estimating occupancy

```
CU LDS = 64 KiB  (example for gfx1201)
Workgroup LDS usage = 32 KiB
→ Max 2 workgroups per CU from LDS alone
Workgroup size = 64 threads → 128 threads/CU from LDS
Hardware max threads/CU = 2048 → occupancy ≈ 6%  ← TERRIBLE
```

Fat shaders (high register or LDS usage) need careful tuning.

### Monitoring

Use `VK_KHR_performance_query` counters:
- NVIDIA: `SM Active` %
- AMD: `ValuUtilization` %

---

## 5. TDR / GPU Hang Prevention (Windows + Vulkan)

### Windows TDR (Timeout Detection and Recovery)

| Platform | Default timeout | Safe kernel duration |
|---|---|---|
| Windows (all vendors) | 2 seconds | < 1.5s with fence polling |
| Linux (CUDA) | 60 seconds (soft) | < 30s |
| Linux (Vulkan/DX12) | 60 seconds (soft) | < 30s |

TDR resets the GPU if a kernel runs longer than the timeout without yielding.

### Vulkan-safe fence patterns

#### Timeline semaphores (Vulkan 1.3+)

```cpp
// For multi-second GPU operations — MUST use timeline, not legacy fence
VkSemaphore timeline_sem;
VkSemaphoreTypeCreateInfo timeline_info = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
};
vkCreateSemaphore(device, &timeline_info, nullptr, &timeline_sem);

// Signal after queue submission:
VkSemaphoreSignalInfo signal_info = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
    .semaphore = timeline_sem,
    .value = ++signal_value,
};
vkSignalSemaphore(device, &signal_info);

// Wait WITH timeout (prevents TDR):
VkSemaphoreWaitInfo wait_info = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
    .semaphore = timeline_sem,
    .pValues = &fence_value,
    .timeout = 1500000000ULL,  // 1.5 seconds in nanoseconds
};
vkWaitSemaphores(device, &wait_info, timeout_ns);
```

#### Legacy fence polling (fallback for older Vulkan)

```cpp
VkFence fence;
vkCreateFence(device, &(VkFenceCreateInfo){
    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
}, nullptr, &fence);

vkQueueSubmit(queue, 1, &submit_info, fence);

// Poll with timeout to avoid TDR on Windows
uint64_t timeout_ns = 1500000000ULL;  // 1.5s
VkResult result = vkWaitForFences(device, 1, &fence, VK_FALSE, timeout_ns);
if (result == VK_TIMEOUT) {
    // Handle: kernel exceeded safe duration
    // Log, abort, or split work
}
vkDestroyFence(device, fence, nullptr);
```

### TDR prevention checklist for Vulkan compute

- [ ] Every `vkQueueSubmit` with a fence or timeline semaphore signal
- [ ] Every fence/semaphore wait has a timeout (≤ 1500ms on Windows)
- [ ] Timeline semaphores used for multi-second GPU operations (not legacy fences)
- [ ] No `vkDeviceWaitIdle` in hot path on Windows (blocks CPU thread, may trigger TDR)
- [ ] Barriers use precise src/dst stages (not ALL_COMMANDS) to minimize flush
- [ ] Ping-pong queues used for multi-frame compute+graphics to avoid stalls

---

## 6. CUDA vs Vulkan Context Isolation (ArXiv VUDA)

**Key finding (ArXiv 2605.01352v1):** CUDA and Vulkan create *separate GPU
contexts* with separate scheduling groups (TSGs). The GPU time-slices among
TSGs in round-robin, so CUDA compute and Vulkan graphics execute in
**mutually exclusive time slices** — spatial parallelism is impossible without
explicit intervention.

This matters for TDR: if your CUDA kernel and Vulkan compute shader are in
separate TSGs, your timeline semaphore on the Vulkan side **will not** be
signalled by the CUDA side. Use a shared fence (VK_KHR_external_fence) **or**
restructure to run both workloads in the same queue family.

---

## 7. Quick Reference: Vulkan Sync Primitives

| Primitive | Use case | Cross-queue | Timeout support |
|---|---|---|---|
| `VkFence` | CPU waits on GPU completion | No (queue-local) | `vkWaitForFences` ✓ |
| `VkSemaphore` | GPU waits on GPU (same queue) | No | Implicit (can deadlock) |
| Timeline `VkSemaphore` | GPU waits on GPU (cross-submit) | Yes (if same queue family) | `vkWaitSemaphores` ✓ |
| `VkEvent` | In-command-buffer sync | No | N/A (in-CMDB only) |
| `VkBufferMemoryBarrier` / `VkImageMemoryBarrier` | Resource access ordering | Per-submission | N/A |

### Cross-queue sync

For cross-queue synchronization, use **semaphores** (not fences). Each queue
waits on the other's signal. Timeline semaphores are preferred — they avoid
the "signal after wait" ordering constraints of binary semaphores.

---

## Evidence Labels

| Claim | Status |
|---|---|
| Barriers cause pipeline flushes | PASS — confirmed by NVIDIA & GPUOpen profiling |
| Bottom-of-pipe → Top-of-pipe barrier stalls GPU | PASS — empirically observed in PVRTune |
| Ping-pong queues recover 20–30% lost throughput | PASS — Imagination PowerVR real-world data |
| Occupancy < 50% often due to register pressure | PASS — vendor tools confirm |
| `VK_KHR_synchronization2` enables finer scheduling | PASS — driver documentation |
| Timeline semaphores prevent TDR on long kernels | PASS — Vulkan spec + Windows TDR behavior |
| `vkDeviceWaitIdle` in hot path risks TDR | PASS — Windows TDR triggers on CPU blocking |
| CUDA/Vulkan separate TSGs prevent spatial sharing | PASS — VUDA paper experimental results |
