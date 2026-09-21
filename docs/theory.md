# The theory behind this project

This project fine-tunes a 7B-parameter language model on a single free
Colab GPU. That's only possible because of two ideas working together:
**quantization** (shrink the numbers) and **LoRA** (shrink the number of
weights you actually train). This note explains both, and how they combine
into QLoRA, the technique this repo actually uses.

## 1. Quantization: storing weights in fewer bits

A model's weights are just numbers. By default they're stored as 32-bit or
16-bit floating point (FP32 / FP16 / BF16) — enough precision that a 7B
parameter model needs ~14-28GB just to hold the weights, before any
activations, gradients, or optimizer state.

Quantization converts those numbers into a lower-precision format — most
commonly 8-bit or 4-bit integers — so the same weights take a fraction of
the memory and every matrix multiply is cheaper.

**How the conversion works.** For a tensor of values, you pick a `scale`
(and, for asymmetric ranges, a `zero_point`) so that the full range of real
values maps onto the full range of the target integer type:

```
scale = (max_val - min_val) / (quant_max - quant_min)
quantized_value = round(real_value / scale) + zero_point
```

This mapping process — deciding the scale and zero-point from a sample of
the real weight distribution — is called **calibration**. Do it once after
training on a frozen model and you get **post-training quantization
(PTQ)**: fast, but it bakes in some accuracy loss because rounding is lossy.
Do it *during* training instead — simulate the rounding in the forward
pass so the model learns weights that are robust to it — and you get
**quantization-aware training (QAT)**. QLoRA (below) sits closer to the
QAT end of this spectrum: it quantizes the frozen base model but keeps the
newly trained weights in full precision, so quantization error doesn't
accumulate during fine-tuning.

**Why 4-bit and not 2-bit or 1-bit?** Precision loss grows as bit-width
shrinks. QLoRA's authors picked a 4-bit scheme (`NF4`, "NormalFloat4" —
a quantile-based 4-bit type tuned for the roughly-Gaussian shape of neural
network weights, rather than a uniform int4) because it preserves enough
information that the fine-tuning result is statistically indistinguishable
from full 16-bit fine-tuning in their benchmarks. This project uses
`bnb_4bit_quant_type="nf4"` for exactly that reason (see `src/config.py`).

## 2. LoRA: fine-tune a lot fewer numbers

Full fine-tuning updates every one of a model's weight matrices. For a 7B
model that's 7 billion numbers to keep gradients and optimizer state for —
the actual reason fine-tuning needs so much VRAM (usually far more than
inference on the same model).

LoRA's ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) observation:
you don't need to update the full weight matrix `W₀` (shape `d × k`) to
adapt it — a much lower-rank update captures most of what fine-tuning
needs. Instead of learning a full `ΔW` (also `d × k`), LoRA factors it into
two small matrices:

```
ΔW = B · A          where B is (d × r), A is (r × k), r << min(d, k)
W = W₀ + (alpha / r) · B·A
```

`W₀` stays frozen. Only `A` and `B` are trained. If `d = k = 4096` and
`r = 16`, a full update would be `4096 × 4096 ≈ 16.8M` parameters; the LoRA
version is `2 × (4096 × 16) ≈ 131K` — about 128x fewer trainable
parameters for that one matrix, and the memory saved on optimizer state
(Adam keeps two extra copies of every trainable parameter) is what makes
fine-tuning a 7B model on a 16GB GPU possible at all.

Two hyperparameters matter:
- **`r` (rank)** — the bottleneck dimension. Higher `r` = more trainable
  parameters = more capacity to learn complex behavior, at the cost of
  more memory and slower training. This project uses `r=16` (see
  `LORA_R` in `src/config.py`), a common default for instruction-style
  fine-tuning; the LoRA paper found `r=8` sufficient for many tasks.
- **`alpha`** — a scaling factor on the update (`alpha / r` above). It
  controls how strongly the LoRA update perturbs the frozen base weights.
  A common convention (used here) is `alpha = 2 × r`.

At inference time you can either keep `A`/`B` as a separate small adapter
(what this project does — see `config.ADAPTER_DIR`) or merge `B·A` back
into `W₀` for a single dense checkpoint with zero extra latency.

## 3. QLoRA: combining both

[QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023) is simply:
load the frozen base model in 4-bit (NF4), and train LoRA adapters on top
of it in a higher-precision compute dtype (bf16 here). The frozen weights
being 4-bit is what makes the 7B model fit in ~4-5GB instead of ~14-28GB;
the LoRA adapters being few and full-precision is what makes training
stable and effective despite the base model being heavily compressed.

Two more details this project's config reflects:
- **Double quantization** (`bnb_4bit_use_double_quant=True`) — quantizes
  the quantization *constants* (the per-block scales) themselves, saving
  a further ~0.4 bits/parameter with negligible accuracy impact.
- **Compute dtype vs. storage dtype** — weights are *stored* in 4-bit but
  *computed* in bf16 (`bnb_4bit_compute_dtype`): each 4-bit block is
  dequantized on the fly for the actual matrix multiply, then discarded.
  This is why QLoRA needs no custom low-precision kernels for the compute
  itself.

## Why this matters for the resource constraints in this repo

Free-tier Colab gives a T4 with ~15GB of VRAM. Full fine-tuning of a 7B
model needs the weights (14-28GB) plus gradients plus Adam's optimizer
state (2x parameters) — already impossible before a single batch of data
is loaded. QLoRA's combination — 4-bit frozen weights (~4-5GB) + a LoRA
adapter with a few million trainable parameters (optimizer state on the
order of tens of MB, not tens of GB) — is what makes `src/finetune.py`
actually runnable on the hardware this project targets.

## References

- Hu, E. J., et al. (2021). [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685).
- Dettmers, T., et al. (2023). [*QLoRA: Efficient Finetuning of Quantized LLMs*](https://arxiv.org/abs/2305.14314).
- Dettmers, T., et al. (2022). [*LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*](https://arxiv.org/abs/2208.07339) — the earlier 8-bit quantization work bitsandbytes builds on.
