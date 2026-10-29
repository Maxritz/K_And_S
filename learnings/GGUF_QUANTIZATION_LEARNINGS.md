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
| Q8_0 | 32 | 8.5 | FP32 | None | Near-lossless, used for KV cache |

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
    float   scale;           // FP32 scale (or fp16 stored as uint16)
    uint8_t  qs[16];         // 16 bytes = 32 x 4-bit values
}; // 20 bytes per 32 weights = 5 bits/weight
```

### Porting note for legacy quants

- No vector alignment tricks — straightforward per-block dequant.
- On Kepler (sm_30): dp4a not available; must unpack 4-bit values into
  fp32 arrays in shared memory before the GEMV kernel.
- Kindi uses 5-trit ternary or 2-bit instead of these standard formats,
  so the dequant kernels are completely different (k_g4_unpack, k_g5n_unpack).

---

## K-Quants (Gen 2: K = Kawrakow or "kernels")

### Double Quantization

Instead of storing scale/zero-point as FP16 per regular block, K-quants
group 8 regular blocks into a **super-block** and quantize the constants:

1. **Regular block constants (scale, zero-point)**: stored as INT8 (quantized!)
2. **Super-block constants**: 1 FP16 pair to dequantize the INT8 block constants

```
Storage per super-block (8 regular blocks of 32 weights = 256 weights):
  - 8 * (16 x 4-bit values) = 512 bits = 64 bytes for weights
  - 8 * 1 INT8 scale = 8 bytes
  - 1 * FP16 super-scale + 1 * FP16 super-zero-point = 4 bytes
  Total: 76 bytes for 256 weights = ~2.38 bits/weight (for q4_K)
```

vs Legacy Q4_0: 8 blocks * (20 bytes each) = 160 bytes for 256 weights = 5 bits/weight.
K-quants roughly halve the metadata overhead via double quantization.

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
| IQ1_B | ~1.6 | Yes |
| IQ2_S / IQ2_1D / IQ2_B | ~2.0-2.5 | Yes |
| IQ2_G16 | ~2.3 | Yes |
| IQ3_S / IQ3_1D / IQ3_B / IQ3_G16 | ~3.0-3.6 | Yes |
| IQ4_NL | ~4.0 | No (non-linear) |
| IQ4_KS | ~4.0 | Yes |
| IQ5_KS | ~5.0 | Yes |

### Porting note for I-quants

- The codebook is architecture-dependent in spirit (8D vectors may map to
  different SIMD widths on different CPUs).
- GPU ports struggle because the codebook is CPU-optimized (cache-friendly
  lookup tables, branch-heavy nearest-neighbor search).
- llama.cpp CUDA does NOT support I-quants (only K-quants and legacy).
- Vulkan backend also lacks I-quant support — falls back to K-quants or fp16.

---

## Importance Matrix (imatrix)

### What it is

A per-weight score matrix I[i,j] that rates how much each weight matters to
the model's output. NOT a quantization method — it's a quality enhancer that
can be applied to ANY quant type.

### Computation

Uses a calibration dataset (typically Wikitext, ~200-500 examples):

1. Run a forward pass: y = W * x (y_i = output for row i)
2. Per-row importance: I_i = y_i^2
3. Per-weight importance: I_ij = I_i + sqrt(sigma^2 + w_ij^2)
   (sigma = row weight variation, prevents zero for zero weights)

### How it influences quantization

Bias the de-quantization constants (S', Z') toward important weights:
- When an important weight is dequantized, its reconstruction error is lower.
- Loss to minimize: L = sum_ij (w_ij - dequant(quant(w_ij)))^2
- This is a quadratic in S', Z' with a closed-form solution.
- Additional grid search: 36 values around fixed S to tune clipping range.

### Key properties

- **ZERO inference overhead**: no extra data stored in GGUF, no runtime cost
- **10-30% perplexity improvement** common, more at low bit rates
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

### Kindi k_g5 (5-trit ternary) format

```
Weights use a base-3 encoding (trits: {-1, 0, +1, +2... wait, ternary is 3-level}).
Packed 4 trits per byte as t0 + 3*t1 + 9*t2 + 27*t3.
Lane layout: byte position major (optimized for k_g5n4 lane kernel).
5 trits encode 3^5 = 243 values, stored in 28 bytes per group.
```

### Porting implications

- Kindi's k_g4/k_g5 have NO equivalent in GGUF — they are custom formats.
- To port Kindi to AMDx: need to implement k_g4/k_g5 unpack kernels using
  the same bit-manipulation approach (no dp4a needed).
- GGUF Q4_K double-quantization can be ported as a dequant shader (Vulkan)
  or unpack kernel (HIP), but I-quants cannot (codebook is CPU-only).
- KV cache format: Kindi uses q8_0 (matching llama.cpp's -ctk q8_0) —
  this IS portable between engines.

---

## Quantization Summary for Porting

| Quant | Bits/weight | Vector | imatrix | CPU kernel | CUDA kernel | Vulkan shader | Kindi support |
|---|---|---|---|---|---|---|---|
| Q4_0 | 4.5 | No | No | ggml_cpu_mul_mat | mmq.cu | mul_mmq.comp | No (custom k_g4) |
| Q5_0 | 5.5 | No | No | ggml_cpu_mul_mat | mmq.cu | mul_mmq.comp | No (custom k_g5) |
| Q8_0 | 8.5 | No | No | ggml_cpu_mul_mat | mmq.cu | mul_mmq.comp | Yes (KV cache) |
| Q4_K | ~4.5 | No | Optional | ggml_cpu_mul_mat | mmq.cu | mul_mmq.comp | No |
| Q5_K | ~5.5 | No | Optional | ggml_cpu_mul_mat | mmq.cu | mul_mmq.comp | No |
| Q6_K | ~6.0 | No | Optional | ggml_cpu_mul_mat | mmq.cu | mul_mmq.comp | No |
| Q8_K | ~8.0 | No | Optional | ggml_cpu_mul_mat | mmq.cu | mul_mmq.comp | No |
| IQ2_S | ~2.0 | Yes | Required | ggml_cpu_mul_mat (codebook) | No | No | No (custom k_g4) |
| IQ3_S | ~3.0 | Yes | Required | ggml_cpu_mul_mat (codebook) | No | No | No |
| IQ4_NL | ~4.0 | Yes | No | ggml_cpu_mul_mat (codebook) | No | No | No |
| k_g4 | 2.0 | No | No | N/A | k_g4 (mm_core) | N/A | Yes (native) |
| k_g5 | 5-trit | No | No | N/A | k_g5n / k_g5r1 | N/A | Yes (native) |

---

## Practical Porting Checklist

1. **Q4_K/Q5_K/Q6_K**: Port the double-dequantization super-block logic
   from `convert.cuh` into a dequant shader/kernel. Use fp16 for
   super-block constants, INT8 for block constants.

2. **I-Quants**: NOT portable to GPU. The codebook lookup is a CPU
   optimization. Use Q4_K or Q5_K instead for GPU inference.

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
