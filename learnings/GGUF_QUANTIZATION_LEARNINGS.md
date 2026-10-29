# GGUF Quantization — Learnings Distilled

> Source: https://github.com/iuliaturc/gguf-docs (unofficial docs by iuliaturc)
> Distilled from k-quants.md, i-quants.md, importance-matrix.md, legacy-quants.md.

## Overview

GGUF quantization = Post-Training Quantization (PTQ) for llama.cpp models.
Reduces weight bit-width after training to shrink model size for consumer hardware.
Has NO cost during inference — quantization constants are baked into the GGUF file.

## Generations of GGUF Quants

| Gen | Name | Vector/Scalar | Key Innovation |
|---|---|---|---|
| 1 | Legacy (Q4_0, Q8_0) | Scalar | 32-weight blocks, affine quantization with per-block scale |
| 2 | K-Quants (Q4_K, Q5_K, Q6_K) | Scalar | Double quantization: quantize the quantization constants (QLoRA-style) |
| 3 | I-Quants (IQ2_S, IQ3_S, etc.) | Vector | Vector quantization over 8-weight groups, importance matrix-aware |

---

## Legacy Quants (Gen 1)

### Block Quantization

Each quant type groups weights into fixed-size blocks and stores affine quantization constants per block.

| Type | Block size | Bits/weight | Scale | Offset/zero-point | Notes |
|---|---|---|---|---|---|
| Q4_0 | 32 | 4.5 | FP16 (1 per block) | None | Simple, fast |
| Q4_1 | 32 | 4.5 | FP16 | FP16 | Adds zero-point for better accuracy |
| Q5_0 | 32 | 5.5 | FP16 | None | 5-bit values |
| Q5_1 | 32 | 5.5 | FP16 | FP16 | 5-bit + zero-point |
| Q8_0 | 32 | 8.5 | FP16 | None | Near-lossless, used for KV cache |

### Math

```
# Affine quantization per block of N=32:
S = (max - min) / 15        # scale (fp16)
q_i = round((w_i / S) + zero) # zero = 0 for Q4_0, stored for Q4_1
dequant: w_i' = S * (q_i - zero)
```

### File layout (struct)

```c
// Q4_0: 4-bit values + FP16 scale per 32 weights
struct block_q4_0 {
    ggml_half d;             // FP16 scale (2 bytes)
    uint8_t  qs[16];         // 16 bytes = 32 x 4-bit values
}; // 18 bytes per 32 weights = 4.5 bits/weight (verified: ggml-common.h block_q4_0)
```

### Porting note for legacy quants

- No vector alignment tricks — straightforward per-block dequant.
- On Kepler (sm_30): dp4a not available; must unpack 4-bit values into
  fp32 arrays in shared memory before the GEMV kernel.
- Kindi uses 5-trit ternary (PTQ1_0) or 4-trit packing instead of these standard
  formats, so the dequant paths differ — but unpacking is a stage INSIDE the GEMV
  kernels (k_g4 / k_mm_g4 / k_g5n4 / k_mm_g5n); there are no separate unpack kernels.

---

## K-Quants (Gen 2: K = Kawrakow or "kernels")

### Double Quantization

Instead of storing scale/zero-point as FP16 per regular block, K-quants
group 8 regular blocks into a **super-block** and quantize the constants:

1. **Regular block constants (scale, zero-point)**: stored as INT8 (quantized!)
2. **Super-block constants**: 1 FP16 pair to dequantize the INT8 block constants

```
Q4_K super-block (verified: ggml-common.h block_q4_K, static_assert = 144 bytes):
  - 1 * FP16 super-scale (d)  = 2 bytes
  - 1 * FP16 super-min    (dmin) = 2 bytes
  - 6-bit packed scales/mins for 8 sub-blocks = 12 bytes (K-scale-size, NOT INT8)
  - 128 bytes of 4-bit quants (256 weights x 4 bits)
  Total: 144 bytes per 256 weights = 4.5 bits/weight

vs Legacy Q4_0: 8 blocks * 18 bytes = 144 bytes per 256 weights = 4.5 bits/weight.
(Q4_K's win over Q4_0 at equal bpw is the 6-bit-quantized per-sub-block scale
granularity, not a smaller byte count.)
```

### Size modifiers (S/M/L)

K-quants use size modifiers that control which weights get higher precision:

- **S (Small)**: more aggressive quantization, smaller files
- **M (Medium)**: balanced, most common choice
- **L (Large)**: highest quality, largest files (q5_K, q6_K)

The modifier controls mixed-precision allocation:
- Layer norm weights: often kept in FP16
- Attention weights: q5_k or q6_k
- Output layer: sometimes higher precision

### K-Quant types

| Type | Bits/weight | Notes |
|---|---|---|
| Q2_K | ~2.0 | 2-bit values, 4+4 INT4 block constants |
| Q3_K_S/M/L | ~3.0-3.3 | 3-bit values + super-block |
| Q4_K_S/M/L | ~4.0-4.5 | 4-bit values + INT8 scale, FP16 super-block |
| Q5_K_S/M/L | ~5.0-5.5 | 5-bit values |
| Q6_K | ~6.0 | 6-bit values, near-lossless for many models |
| Q8_K | ~8.0 | 8-bit, higher-precision alternative to Q8_0 |

### Porting note for K-quants

- The super-block structure means dequant spans two levels: super-block
  FP16 -> INT8 block constants -> quantized weight values.
- llama.cpp kernels handle this in `convert.cuh` (dequant) and `mmq.cu`
  (matmul with in-tile dequantization).
- On AMD Vega/RDNA: no INT8 dot product (no dp4a equivalent); must unpack
  in shared memory like the legacy path.

---

## I-Quants (Gen 3: I = Importance)

### Vector Quantization

I-quants break from scalar quantization entirely:

- Treats groups of 8 weights as a vector: w_vec = [w_1, ..., w_8]
- Maps to a codebook entry (8D reference vector) via nearest-neighbor search
- Stores: code (index into codebook), scale S (shared 256 vectors), signs (1 byte)

```
Storage per 8 weights:
  - 1 byte code (256-entry codebook)     = 1 bit/weight (code portion)
  - 1 byte signs (sign trick)            = 1 bit/weight
  - FP16 scale S shared across 256 vectors = 0.008 bits/weight
  Total: ~2 bits/weight (IQ2_S)
```

### The Sign Trick

Codebook vectors are exclusively positive in all dimensions.
During quantization, search for the nearest neighbor of |w_vec| (absolute value),
then store a byte of signs encoding the original signs.

This increases effective codebook size by 2^8 = 256x without storing more codes.

### Codebook

- Hard-coded in [ggml-quants.h](https://github.com/ggml-org/llama.cpp/blob/0d9226763c82562186122f3b827fa3862864a19c/ggml/src/ggml-common.h#L482)
- Codes are encoded as hexadecimals (obfuscated)
- May be derived from E8 lattice (240 8D vectors), though unconfirmed in code
- Codebook lookup is a table of reference vectors; nearest-neighbor search
  adds computational overhead but can be precomputed/cached

### I-Quant types

| Type | Bits/weight | Requires imatrix? |
|---|---|---|
| IQ1_S | ~1.5 | Yes |
| IQ1_S / IQ1_M | ~1.5-1.8 | Yes (grid-based) |
| IQ2_XXS / IQ2_XS / IQ2_S | ~2.0-2.7 | Yes (2-byte codes per 8 weights + sign mask) |
| IQ3_XXS / IQ3_S | ~3.0-3.4 | Yes |
| IQ4_XS / IQ4_NL | ~4.0-4.5 | Yes (NL = non-linear codebook) |

(Real ggml enum has GGML_TYPE_COUNT = 40 incl. TQ1_0/TQ2_0 ternary; IQ types
verified against gguf.md + ggml-common.h. Invented names IQ1_B, IQ2_1D, IQ2_B,
IQ3_1D, IQ3_B, IQ3_G16, IQ4_KS, IQ5_KS removed 2026-10-30 — none exist in ggml.)

### Porting note for I-quants

- The codebook is architecture-dependent in spirit (8D vectors may map to
  different SIMD widths on different CPUs).
- **CORRECTED 2026-10-30 (audit):** llama.cpp CUDA DOES support I-quants —
  IQ1_S/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS/IQ4_NL all present in mmq.cu
  dispatch (lines 48-69) and mmvq.cu vec-dot selection (lines 56-64). The Vulkan
  backend also ships per-type mul_mat_vec_iq*_comp pipelines. The grid/sign
  tables live in ggml-common.h as `__device__` tables — they compile into GPU
  code. Do NOT claim GPU backends lack I-quant support.

---

## Importance Matrix (imatrix)

### What it is

A per-weight score matrix I[i,j] that rates how much each weight matters to
the model's output. NOT a quantization method — it's a quality enhancer that
can be applied to ANY quant type.

### Computation

Uses a calibration dataset (typically Wikitext, ~200-500 examples):

1. Run a forward pass over the calibration data.
2. Accumulate per-column activation statistics: sums and squared sums of the
   input activations each weight column multiplies (imatrix.cpp computes
   running sums/sumsq; the saved matrix holds per-column mean/variance data).
3. The quantizer then weights reconstruction error by those activation
   statistics when choosing scales.

(CORRECTED 2026-10-30: the earlier closed-form "I_ij = I_i + sqrt(sigma^2 +
w_ij^2)" appears nowhere in ggml — fabricated. The real mechanism is activation
sum/sumsq accumulation in tools/imatrix.cpp.)

### How it influences quantization

Bias the de-quantization constants (S', Z') toward important weights:
- When an important weight is dequantized, its reconstruction error is lower.
- Loss to minimize: L = sum_ij (w_ij - dequant(quant(w_ij)))^2
- This is a quadratic in S', Z' with a closed-form solution.
- Additional grid search: 36 values around fixed S to tune clipping range.

### Key properties

- **ZERO inference overhead**: no extra data stored in GGUF, no runtime cost
- **Perplexity improvement**: UNVALIDATED range — the earlier "10-30% common"
  claim had no source. Measure per model; gains are largest at low bit rates.
- **Most beneficial for Q4 and below** (aggressive quantization)
- **Does NOT change the quant format**: a q4_K_GGGUF with imatrix looks
  identical to one without — impossible to tell from the file alone

### Calibration best practices

- Use representative data matching your intended use case
- 200-500 high-quality examples typically sufficient
- DON'T use your evaluation/test data for calibration (overfitting risk)
- Larger models see bigger relative improvements

---

## Kindi Ternary/2-bit vs GGUF Quants

### Kindi's quant formats

| Feature | Kindi (Kepler) | GGUF Q4_0 | GGUF Q8_0 |
|---|---|---|---|
| Bit width | 2-bit or 5-trit ternary | 4-bit scalar | 8-bit scalar |
| Block structure | 128-column groups | 32-weight blocks | 32-weight blocks |
| Scale storage | Per-group fp16 | Per-block fp16 | Per-block fp32 |
| Double quantization | No | No (legacy) | No |
| Vector quantization | No | No | No |
| Unpack method | Lane-based bit manipulation | Bit-shift extract | Direct cast |
| dp4a support | No (sm_30) | No (unpack in shared mem) | Yes (CUDA dp4a) |

### Kindi k_g4 (2-bit) format

```
Weights are 2-bit ternary values packed 4-per-byte.
No zero-point — ternary {0, +1, -1} via bit pairs.
Unpack: k_g4 kernel extracts bits in registers, no dp4a.
Runs on sm_30 (K420) and sm_50 (K1200) — no tensor cores needed.
```

### Kindi k_g4 (4-trit ternary) format

```
Internal layout: base-3 trits packed 4 per byte (values 0-80, t0 + 3*t1 + 9*t2 + 27*t3).
Kernel: 81-entry float LUT in shared memory (T[2*81*64] incl. sign), one LDS load
+ one FMA per weight; weight = (trit - 1) * d.
Runs on sm_30 (K420) and sm_50 (K1200) — no tensor cores needed.
```

### Kindi k_g5 / PTQ1_0 (5-trit ternary) format

```
GGUF PTQ1_0 (type 143): per 128 weights = qs[24] | qh[2] | fp16 scale d
  = 28-byte blocks, 5 trits per byte in qs (24*5 + 2*4 = 128 trits), 1.75 bpw.
  (Verified: convert.py:3, kindi.cu:1725.)
Internal g5_blob (tp.py:38): repacked form — 24 B x 5 trits + 2 B x 4 trits
  + fp16 scale per 128-column group, byte-position-major lane layout for k_g5n4.
```

### Porting implications

- GGUF HAS native ternary types: TQ1_0 (type 34), TQ2_0 (type 35) — and the
  kindi models themselves are GGUF files with the Prism-fork PTQ1_0 extension
  (type 143). "No equivalent in GGUF" was wrong (corrected 2026-10-30).
- To port Kindi to AMDx: reuse the in-GEMV unpack structure (k_g4/k_g5n4
  decode weights inside the matmul), matching GGUF block layouts where they
  align.
- K-quant double-dequantization ports as a dequant shader (Vulkan) or in-tile
  unpack (HIP/CUDA) — llama.cpp does both.
- I-quants DO port to GPU: grid/sign tables compile as device constant tables
  (verified in mmq.cu, mmvq.cu, mul_mat_vec_iq*.comp).
- KV cache format: the single-card kindi path uses fp16 K/V; only the 27B
  tensor-parallel path uses q8_0 (kindi.cu:977). The q8_0 block layout
  (int8 + fp16 scale per 32) matches llama.cpp's -ctk q8_0 — portable.

---

## Quantization Summary for Porting

| Quant | Bits/weight | Vector | imatrix | CPU kernel | CUDA kernel | Vulkan shader | Kindi support |
|---|---|---|---|---|---|---|---|
| Q4_0 | 4.5 | No | No | type_traits_cpu vec_dot | mmq.cu | mul_mmq.comp | No (custom PTQ1_0) |
| Q5_0 | 5.5 | No | No | type_traits_cpu vec_dot | mmq.cu | mul_mmq.comp | No |
| Q8_0 | 8.5 | No | No | type_traits_cpu vec_dot | mmq.cu | mul_mmq.comp | Yes (27B TP KV) |
| Q4_K | 4.5 | No | Optional | type_traits_cpu vec_dot | mmq.cu | mul_mmq.comp | No |
| Q5_K | ~5.5 | No | Optional | type_traits_cpu vec_dot | mmq.cu | mul_mmq.comp | No |
| Q6_K | ~6.0 | No | Optional | type_traits_cpu vec_dot | mmq.cu | mul_mmq.comp | No |
| Q8_K | ~8.0 | No | Optional | type_traits_cpu vec_dot | mmq.cu | mul_mmq.comp | No |
| IQ2_XXS/XS/S | 2.06-2.56 | Yes | Recommended | type_traits_cpu (grid LUT) | mmq.cu + mmvq.cu | mul_mat_vec_iq2_*.comp | No |
| IQ3_XXS/S | ~3.0-3.4 | Yes | Recommended | type_traits_cpu (grid LUT) | mmq.cu + mmvq.cu | mul_mat_vec_iq3_*.comp | No |
| IQ4_XS/NL | ~4.0-4.5 | Yes | Optional | type_traits_cpu (grid LUT) | mmq.cu + mmvq.cu | mul_mat_vec_iq4_*.comp | No |
| TQ1_0/TQ2_0 | ~1.7-2.1 | No | No | type_traits_cpu | mmq.cu (tq) | mul_mat_vec_tq*_comp | PTQ1_0 ≈ TQ family |
| k_g4 (internal) | 2.0 | No | No | N/A | k_g4 (mm_core) | N/A | Yes (native) |
| k_g5 / PTQ1_0 | 1.75 | No | No | N/A | k_g5n / k_g5r1 | N/A | Yes (native) |

(CPU kernel column corrected 2026-10-30: there is no `ggml_cpu_mul_mat` symbol;
CPU mul_mat is generic code driven by the `type_traits_cpu` table — see
GPU_PIPELINE_DATAFLOW_METHODOLOGY.md §3. IQ CUDA/Vulkan support corrected —
"IQ kernels missing" was false.)

---

## Practical Porting Checklist

1. **Q4_K/Q5_K/Q6_K**: Port the double-dequantization super-block logic
   from `convert.cuh` into a dequant shader/kernel. Use fp16 for
   super-block constants, INT8 for block constants.

2. **I-Quants**: PORTABLE to GPU (corrected 2026-10-30 — earlier "codebook is
   CPU-only" claim was false). Grid/sign tables compile as device constants;
   llama.cpp ships IQ kernels on CUDA, Vulkan, and Metal.

3. **Importance matrix**: No porting needed — it only changes the
   scale/zero-point constants in the GGUF file. Once dequantized to
   fp32/f16, the inference kernels are identical.

4. **KV cache (q8_0)**: Fully portable. Kindi's k_store_q8 writes int8
   + fp16 scale per 32-value block — same as llama.cpp's q8_0 cache.

5. **Kindi k_g4/k_g5**: Custom formats with no GGUF equivalent. If porting
   Kindi weights to a different GPU, implement the same lane-based
   unpack kernels (bit manipulation, not dp4a).

6. **Memory alignment**: K-quant super-blocks (256-weight groups) align
   well with 128-column tile layouts. I-quant 8-weight vectors align
   with int8 SIMD widths.
