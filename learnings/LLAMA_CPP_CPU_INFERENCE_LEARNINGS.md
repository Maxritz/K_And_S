# llama.cpp CPU inference — reference flow + TRUE/FALSE gates

Distilled from full code trace of llama.cpp master (CPU-only, `n_gpu_layers=0`).
Repo-relative paths. No timestamps in filename; revise in place, never duplicate.

Source artifact: `llama_cpu_infer_gates.py` (runnable gate model, logic only, no model needed).
Evidence convention: PASS = code-traced, UNVALIDATED = formula/demo estimate, NOT RUN = needs model.

## 0. LOAD — `src/llama.cpp:316 llama_model_load` [PASS]

1. `include/llama.h:525 load_from_file` → `src/llama.cpp:379 load_from_file_impl` → `:316 load`
2. `src/llama-model-loader.cpp:535 loader()` : `573 gguf_init_from_file` + `llama_file(fname,"rb")` (`src/llama-mmap.cpp:87`)
3. `ggml/src/gguf.cpp:966 init_from_file_ptr` → `:460 init_from_reader`: GGUF magic `:466`, ver/n_tensors/n_kv `:489`, KV loop `:552`, align `:622`, tensor infos `:642` (name `:649`, ne `:677`, type `:727`, offset `:768`), data start `:788`
4. `src/llama-model.cpp:392 create` → `:1596 load_tensors`: `:1152 make_cpu_buft_list` (ACCEL/host/extra), `:1687 get_layer_buft_list` (CPU-only if `n_gpu_layers==0`), `:1728 load_arch_tensors`
5. Per-tensor `llama-model-loader.cpp:1126 create_tensor`: `:1347 check_dims` → `:1373 buft_for_tensor` → `:1077 select_weight_buft` → `:1390 ggml_dup_tensor`
6. `src/llama-model.cpp:1420 init_mappings`: Win `:596 CreateFileMappingA` + `:603 MapViewOfFile` + `:616 PrefetchVirtualMemory` / POSIX `:495 mmap` + `:513 MADV_WILLNEED`
7. `:1948` fast path: `use_mmap && buffer_from_host_ptr` → `:1962 dev_buffer_from_host_ptr(addr)` = zero-copy alias. Else `:1977 alloc + memcpy`.
8. `llama-model-loader.cpp:1562 load_all_data`: mmap `:1732 tensor_alloc alias` vs `:1751 seek+read_raw` + copy. `:1857 unmap_fragment` trims head/tail.

Speed law [UNVALIDATED formula, PASS mechanism]: `mmap >> read`. `n_threads` IRRELEVANT at load (single-thread IO). Size = `params * bpw` (Q4_K ~4.5bit vs F16 16bit). Stall = prefetch/mlock/disk-seq.

## 1. CONTEXT — `src/llama-context.cpp:204,342` [PASS]

- `342 n_batch=min(n_ctx,n_batch)`, `n_ubatch`; `213 flash_attn`; `370 offload_kqv`
- KV alloc `src/llama-kv-cache.cpp:208 new_tensor_3d(n_embd_kqv,kv_size,n_stream)`, `memory_size_k+v` log
- Scheduler/allocator `ggml/src/ggml-backend.cpp:915 sched_split_graph`, `ggml/src/ggml-alloc.c:14 MAX_FREE_BLOCKS`
- Threadpool `ggml/src/ggml-cpu/ggml-cpu.c:480 threadpool{mutex,cond,workers}`, `ggml/src/ggml-cpu/ggml-cpu.cpp:99 backend_cpu_context`

## 2. TOKENIZE + BATCH [PASS]

- `src/llama-vocab.cpp:4921 llama_tokenize` → `:4632 vocab::tokenize`; `common/common.cpp:1864 common_tokenize`
- `include/llama.h:264 struct llama_batch`, `:976 batch_init`, `:965 batch_get_one`. `logits[i]!=0` = output row.
- Prefill `n_tokens=prompt_len`; decode `n_tokens=1`. `src/llama-batch.cpp:28 allocr::init`.

## 3. DECODE — `include/llama.h:1006 → src/llama-context.cpp:4643 → :4625 → :1839` [PASS]

1. `:1902 n_tokens_all`, `:1914 ASSERT(<=n_batch)`
2. `:1938 memory_update(false)`, `:1943 memory->init_batch(n_ubatch)` → `llama-kv-cache.cpp:696 init_batch` → `:705 split_simple/equal/seq`
3. `:743 prepare`, `:893 find_slot`, `:2855 get_ubatch` / `:2825 next` / `:2835 apply`
4. `:1999 get_ubatch` → `:2002 count n_outputs_new` → `:1525 process_ubatch` → `:1537 graph_params` → `:1539 can_reuse?` Key `src/llama-graph.h:839 allow_reuse(n_tokens,n_seq_tokens,n_seqs,outputs,samplers)` — prefill vs decode NEVER reuse.
5. `src/llama-model.cpp:3351 build_graph` (arch `:3352`, pooling `:3355`, logits `:3358`, sampling `:3361`); ex `src/models/llama.cpp:94` (embd `:108`, pos `:111`, kv `:119`, mask `src/llama-graph.cpp:30`, attn `:169`, ffn `:189`, norm `:236`)
6. `:1583 set_inputs`, `:1588 graph_compute(gf, ubatch.n_tokens>1)` — TRUE=prefill/GEMM, FALSE=decode/GEMV
7. `:2750 graph_compute`: `batched ? n_threads_batch : n_threads` + `ggml_backend_cpu_set_threadpool` + `:2771 sched_graph_compute_async`

## 4. LOGITS + SAMPLE [PASS]

- `src/llama-graph.cpp:4203 build_logits`, `:4103 pooling`, `:1443 set_outputs`; `src/llama-context.cpp:2058 get + tensor_get_async`, `:2245 output_reserve`, `:1030 get_logits`, `:4237 llama_get_logits`
- `src/llama-sampler.cpp:896 sample` → `:940 get_logits_ith` → `:682 chain_apply` → `:961 accept`. `common/sampling.cpp:606 sample` (`:619 set_logits`, `:624 backend-token fastpath`, `:643 apply rbudget`, `:650 apply chain`, `:652 id=cur_p[selected]`)
- Chain `common/sampling.cpp:206 chain_init, :360 top_k, :363 top_p, :378 temp_ext, :384 penalties, :403 greedy=(temp==0||top_k==1)`

## 5. LOOP [PASS]

- Single `tools/completion/completion.cpp:544 while(n_remain)`: overflow `:567 if(n_past+batch>=n_ctx)`: `!ctx_shift→break` else `:584 seq_rm(n_keep,n_keep+n_discard)+seq_add(-n_discard)`
- Server `tools/server/server-context.cpp:3470 update_slots` multi-slot; `:3676` same shift; `:4605` KV-full retry (`n_batch/=2`); `:2094 process_token`: EOS→STOP_EOS `:2211`, limit→STOP_LIMIT `:2144`, stop-string `:2115`
- EOS `src/llama-vocab.cpp:4731 is_eog`, `:4743 token_eos`; `common/common.cpp:1371 -INF bias` unless `ignore_eos`

## 6. SPEED GATES [PASS mechanism, UNVALIDATED numbers]

- Dispatch `ggml/src/ggml-cpu/ggml-cpu.c:215` (`vec_dot_q4_0_q8_0 :242`, q8 `:274`, K-quants `:307`); generic fallback `quants.c:225`; x86 `arch/x86/quants.c:28` AVX2/VNNI; ARM `arch/arm/quants.c:26` NEON/DOTPROD; probe `arch/x86/cpu-feats.cpp:20`; AMX `amx/mmq.cpp:19`
- `decode = BW_GB/s / (weight_bytes + kv_step_bytes)` (GEMV, memory-bound) [UNVALIDATED formula]
- `prefill = SIMD_FLOPS / 2P` per token (GEMM, compute-bound) [UNVALIDATED formula]
- Demo estimate (NOT RUN on model): 8B Q4_K @40GB/s → ~8.9 tok/s decode. Do not quote as measured.

### Gate table (mirrors `llama_cpu_infer_gates.py`)

| Gate | TRUE | FALSE |
|---|---|---|
| G_FILE_EXISTS / G_GGUF_MAGIC | open + `GGUF` magic (`gguf.cpp:466`) | truncated/wrong file, stop |
| G_MMAP_ZERO_COPY | alias no-memcpy (`llama-model.cpp:1962`) FAST | `read_raw` copy (`llama-model-loader.cpp:1751`) SLOW load |
| G_CTX_FITS / G_KV_FITS | `prompt+predict<=n_ctx`, KV MB fits (`llama-kv-cache.cpp:208`) | overflow shift or bigger `-c` |
| G_TB_THREADS / G_T_THREADS | `threads<=cores` (`llama-context.cpp:2753`) | oversubscribe → barrier stall (`ggml-cpu.c:576`) |
| G_UBATCH | `>=128` few launches, GEMM-efficient | many launches SLOW prefill |
| G_QUANT_BW | Q4_0/Q4_K_M/Q2_K low bytes (`ggml-cpu.c:242`) | Q8/F16 ~2x BW SLOW decode |
| G_KV_SCAN | KV<=8k short attn | large KV scans all K every step SLOW |
| G_SIMD_* | AVX2/VNNI/AMX/NEON-DOT | scalar `quants.c:225` 3-8x SLOW |
| check_step | `!eos & !full`=CONTINUE, `full&shift`=SHIFT-degraded | `eos`=STOP_EOS, `full&!shift`=STOP_LIMIT |

## Status

PASS (code-traced flow + gates). Numbers UNVALIDATED until measured on a real `.gguf` with `--perf-tracing`-style timing. No model run performed in this engagement.
