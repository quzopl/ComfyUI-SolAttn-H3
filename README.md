# ComfyUI-SolAttn-H3

**Sol-Attn** — NVIDIA's training-free sparse attention from
[Sol Engine](https://github.com/NVlabs/Sana/tree/sol-engine) — wired into
ComfyUI's native **MiniMax-H3**.

One node. No training, no LoRA, no offline calibration. When the kernel contract
isn't met, the node falls back to dense attention **with a named reason** — never
silently.

![The Sol-Attn node in ComfyUI](docs/images/node-comfyui.png)

---

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/benchmark-dark.png">
  <img alt="Kernel time per attention call on RTX 4070 Ti and L40 (cute_sm89), RTX 5080 and RTX PRO 6000 Blackwell (cute_sm120), Sol-Attn against SDPA and SageAttention at three sequence lengths" src="docs/images/benchmark-light.png">
</picture>

The chart is `selftest.py` on four cards at identical settings: kernel time per
attention call, `tau=1.0`, on the CuTe backend each one resolves to. The grid is
architecture × segment — Ada on top, Blackwell below; consumer on the left,
datacentre on the right — so either axis can be read on its own. The L40 has no
bar at 30 976 rows because 48 GB could not hold the tensors; the gap is drawn
rather than dropped.

Read the speedup labels across the grid and the obvious explanation fails.
Against SageAttention the two consumer cards land in the same place — **1.31×,
1.45×, 1.65×** on the 4070 Ti and **1.27×, 1.50×, 1.66×** on the 5080 — while the
datacentre Blackwell reaches **1.84×, 2.26×, 2.55×**. Moving from Ada to
Blackwell on a consumer board bought a large drop in absolute time and almost
nothing in *ratio*. What separates the panels is the memory system, not the
architecture, so "it's a Blackwell card" does not predict what Sol-Attn will give
you.

Every table below states its card **and its backend**, because both change the
answer. Do not transfer any of these numbers to your own GPU — run `selftest.py`
on it instead.

> **Upstream update, August 2026.** NVIDIA added a **CuTe DSL kernel for SM89**
> ([`9cfdd07`](https://github.com/NVlabs/Sana/commit/9cfdd07)), so Ada is no
> longer restricted to the Triton research implementation. The SM89 sections
> below were measured on Triton and are kept because they still describe what
> you get when the CuTe runtime is missing — which is the default, since it is
> not installed automatically. Re-measured on the same card with `cute_sm89`:
>
> | seq 20 530, sink 1 495, `tau=1.0` | kernel | full sparse path | vs SageAttention |
> |---|---:|---:|---:|
> | Triton | 57.3 ms | 83.4 ms | **0.78×** (slower) |
> | **CuTe SM89** | **39.9 ms** | **53.3 ms** | **1.06×** |
>
> First-call compilation drops from 13.4 s to 3.2 s, and the margin widens with
> sequence length: **1.21×** at 31 650 rows, **1.32×** at 45 241.
>
> End-to-end on the real model with the CuTe backend (seq 17 504, 20 steps,
> SageAttention baseline) came out at 161.3 s → 151.0 s. **Do not read that 1.07×
> as the kernel being faster**: the attention accounting does not support it. The `off` run spent
> 1000 × 43.40 ms = 43.4 s in attention, the `on` run 768 × 42.54 + 232 × 43.28 +
> 3.4 s of compilation = 46.1 s — *more*, not less. At this shape, with a 494-row
> sink, the CuTe kernel buys parity with SageAttention (42.54 vs 43.28 ms per
> call), so the win is not arithmetic.
>
> **That last sentence used to read "the 10 s difference in wall clock is
> run-to-run noise on a memory-pressured card." That was wrong**, and
> [the 16 GB section](#end-to-end-when-the-model-fits-in-vram-and-when-it-streams)
> shows why: the gap reproduces across eight runs on two cards, and it vanishes
> the moment the weights fit in VRAM. It is not noise — it is streaming relief,
> and it is worth more than the kernel is.

> ### ⚠️ Read this before you expect a speedup
>
> **The baseline you compare against decides the outcome.** Everything in this
> section is measured against ComfyUI's `pytorch attention` (SDPA). Against
> **SageAttention** on the same card, at the default `tau=1.0`, this node is a
> **net loss** — this is the **Triton** backend, i.e. what you get on any
> architecture when the CuTe runtime is not installed. On the CuTe DSL path the
> result reverses on every card measured: 1.06× on
> [SM89](#results), 1.27–1.66× on a
> [consumer SM120 board](#sm120--rtx-5080-16-gb-cute-dsl) and 1.85–2.55× on a
> [datacentre one](#sm120--rtx-pro-6000-blackwell-cute-dsl):
>
> | Baseline (seq 17 504, 20 steps, SM89) | ms per attention call | End-to-end |
> |---|---:|---:|
> | SageAttention, node off | 43.20 | 157.5 s |
> | Sol-Attn, node on | 51.77 | 169.6 s (**0.93×**) |
>
> The kernel itself wins — 57.3 ms against SageAttention's 65.3 ms in isolation
> at seq 20 530, i.e. 1.14×. What eats the win is everything around it: three
> contiguous BTHD copies (~840 MiB of temporaries per call) and the dense
> recomputation of the prefix query rows. On a card that is already streaming a
> 20 GB model through 12 GB of VRAM, those temporaries cost more wall clock than
> an isolated benchmark suggests.
>
> **Gains at `tau=1.0` are not guaranteed. Check `attn_ms_per_call` in your own
> log before assuming any.** The levers that do produce a win — a higher `tau`,
> or `sink_mode=text` — trade quality; see [Tuning](#tuning).

### SM120 — RTX PRO 6000 Blackwell (CuTe DSL)

Kernel only, `selftest.py`, 56 heads, head_dim 128, `tau=1.0`, `thresh_type=diag`,
`sink_mode=prefix`. torch 2.13.0+cu130, CUDA 13.0, SageAttention 2.2.0 built from
source for `sm_120` (no prebuilt wheel exists for Blackwell on CUDA 13). Backend
picked automatically: `cute_sm120`.

| Sequence | Gate | Density | Sol-Attn | SDPA | SageAttention | vs SDPA | vs Sage |
|---:|:--:|---:|---:|---:|---:|---:|---:|
| 8 192 | PASS | 0.271 | 1.97 ms | 5.88 ms | 3.63 ms | 3.00× | 1.85× |
| 16 384 | PASS | 0.214 | 5.53 ms | 21.99 ms | 12.50 ms | 3.98× | 2.26× |
| 30 976 | PASS | 0.186 | 16.58 ms | 77.45 ms | 42.23 ms | 4.67× | **2.55×** |

**The warning above does not hold on this backend.** On Triton, Sol-Attn loses to
SageAttention at the default `tau=1.0`; on CuTe DSL it wins by 1.85–2.55×, and
the margin widens with sequence length. Nothing was tuned to get that — it is the
stock `tau=1.0` policy, the one the SM89 section has to trade quality to beat.

The correctness gate passes at every length: `rel_l2` ≈ 0.0031 against a limit of
0.005, `max_abs` ≤ 1.2e-4. First call costs 2.9–3.3 s of kernel compilation. The
BTHD copies that dominate the SM89 analysis cost 0.53–1.97 ms here, shrinking from
27 % of kernel time at 8 192 rows to 12 % at 30 976.

Two back-to-back runs on an otherwise idle card; the table is their mean and the
raw timings agreed to within 2 % on every cell. Repeating the same benchmark while
an unrelated training job shared the GPU roughly doubled every absolute time while
leaving the ratios intact — so measure on a quiet card if you want the absolute
numbers to mean anything.

These are kernel-only figures; no end-to-end ComfyUI run was measured on this card.

### SM120 — RTX 5080, 16 GB (CuTe DSL)

The same backend on a consumer Blackwell board. torch 2.12.0+cu130, CUDA 13.0,
driver 610.57.04, SageAttention 2.2.0 built from source for `sm_120`.

| Sequence | Gate | Density | Sol-Attn | SDPA | SageAttention | vs SDPA | vs Sage |
|---:|:--:|---:|---:|---:|---:|---:|---:|
| 8 192 | PASS | 0.271 | 5.47 ms | 18.84 ms | 6.94 ms | 3.45× | 1.27× |
| 16 384 | PASS | 0.214 | 17.14 ms | 75.39 ms | 25.66 ms | 4.40× | 1.50× |
| 30 976 | PASS | 0.186 | 52.85 ms | 267.33 ms | 87.62 ms | 5.06× | 1.66× |

**Densities are identical to the RTX PRO 6000 to three decimals** — 0.271, 0.214,
0.186 on both. Routing is a property of the data, not of the hardware, which is
the control that says these two panels are measuring the same thing.

The absolute times are not close: this card needs 2.8–3.2× longer per call than
the datacentre part. More useful is what happens to the *ratio*. Against
SageAttention the 5080 gives 1.27–1.66×, and the 4070 Ti — a different
architecture, a different kernel — gives 1.31–1.65×. Practically the same.
Against the RTX PRO 6000's 1.85–2.55× that is a large gap, and it does not follow
the architecture line; it follows the memory system. Sol-Attn reads about a fifth
of the K/V blocks, so what it converts into speed is spare bandwidth, and a
consumer board has less of it to give relative to its compute.

All three lengths fit in 16 GB — peak 6 811 MiB at 30 976 rows. That is the
kernel's working set on an otherwise empty card, with the model not loaded; a
real generation adds the weights on top.

### End-to-end on SM120, across weight profiles

A second RTX PRO 6000 Blackwell, 96 GB. MiniMax-H3 at 1344×768, 107 frames,
20 steps, `res_multistep`, same seed. Each row is a full generation; the service
was restarted between rows so peak VRAM is comparable.

| Weights | SDPA | SageAttention | Sol-Attn | Peak VRAM |
|---|---:|---:|---:|---:|
| `pruned_bf16` + int8 text encoder | 170.1 s | 135.0 s | **125.1 s** | 72–76 GB |
| `pruned_int8_convrot` + NVFP4 text encoder | 140.9 s | 105.0 s | **95.0 s** | 43–47 GB |

Two more combinations, measured against SageAttention only, for the memory
picture: `pruned_int8_convrot` + int8 text encoder 105.0 s at 53.9 GB, and
`pruned_bf16` + NVFP4 text encoder 135.0 s at 61.4 GB.

Read across those four and the split is clean: **the transformer sets the time,
the text encoder sets the memory.** Both bf16 rows take 135 s and both int8 rows
105 s regardless of which encoder is paired with them, while swapping the encoder
from int8 to NVFP4 returns about 10 GB either way. That is what you would expect
from a text encoder that runs once against a transformer that runs twenty times,
and it means the encoder choice is free speed-wise — pick it on VRAM alone.

Sol-Attn is worth 1.08× over SageAttention on the bf16 weights and 1.11× on int8,
against 2.26× in the kernel table at the same sequence length. Attention is a
smaller share of wall clock on a 96 GB card than the kernel numbers suggest, and
the sparse path costs about 4 GB more VRAM than SageAttention.

The kernel-only numbers from this card reproduce the SM120 table above closely
(1.98/5.45 ms against 1.97/5.53 at 8 192 and 16 384) — a different machine, the
same measurement.

### End-to-end: when the model fits in VRAM, and when it streams

This is the one section with a controlled variable. The same box, the same
ComfyUI, the same SageAttention commit (`v2.2.0-38-gd1a57a5`), the same graph and
seed — first on an RTX 4070 Ti (12 GB), then on an RTX 5080 (16 GB) after nothing
but a card swap. 864×480, 125 frames (**17 504**-row sequence), 8 steps,
`res_multistep`, seed 42, `fl2va` weights, measurement pass after a warm-up pass.

| Card | Weights | DiT staged | Node off | Node on | Ratio |
|---|---|---:|---:|---:|---:|
| 4070 Ti (`cute_sm89`) | `int8_convrot` | 19 995 MB | 83.2 s | 75.1 s | 1.11× |
| 4070 Ti (`cute_sm89`) | NVFP4 *(emulated)* | 11 944 MB | 117.4 s | 116.5 s | 1.00× |
| 5080 (`cute_sm120`) | `int8_convrot` | 19 995 MB | 58.2 s | **45.1 s** | **1.29×** |
| 5080 (`cute_sm120`) | NVFP4 *(native)* | 11 944 MB | **40.2 s** | **40.0 s** | 1.00× |

#### The node's value is set by VRAM pressure, not by the kernel

Attention time on the 5080 does not depend on the weight format at all. Four
runs, four readings: dense 29.69 / 29.71 / 29.74 / 29.94 ms per call, sparse
27.58 / 27.60 / 27.61 ms. Q/K/V reach the kernel as bf16 whatever the weights are
stored as. So Sol-Attn saves the same ~0.4 s of attention in every row below —
and the wall clock does something else entirely:

| 5080, 15.5 GB usable | DiT | Fits? | Attention saved | Wall clock saved |
|---|---:|---|---:|---:|
| `int8_convrot` | 19 995 MB | no → streams over PCIe | 0.4 s | **13.1 s** |
| NVFP4 | 11 944 MB | yes | 0.4 s | 0.2 s |

When the weights fit, the end-to-end gain **equals** the attention saving, to
within the noise. When they stream, it is 33× larger than the attention saving.
The sparse path touches about a quarter of the K/V blocks, and on a card that is
pulling 20 GB of weights across PCIe every step, that spare bandwidth is worth
far more than the arithmetic it skips.

This also explains the 96 GB table above, where nothing streams and Sol-Attn is
worth 1.08–1.11× — its honest attention share, and no more. **The node pays best
on the cards that can least afford the model, which is the opposite of what a
kernel benchmark suggests.**

#### NVFP4 reverses direction between the two cards

On the 4070 Ti, NVFP4 was **1.41× slower** than int8 (117.4 s vs 83.2 s). On the
5080 it is **1.45× faster** (40.2 s vs 58.2 s). Same files, same loader; the
difference is one line of ComfyUI's startup log, because
`supports_nvfp4_compute()` rejects `props.major < 10`:

```
4070 Ti (SM89) :  emulated ops: mxfp8, nvfp4
5080   (SM120) :  Native ops: …, nvfp4          ← no "emulated" line at all
```

Emulation keeps the 4-bit weights in VRAM but dequantizes to bf16 before every
matmul, and on Ada that costs more than the 8 GB of streaming it saves. Step rate
went 12.0 s/it → 3.10 s/it across the swap. **Do not judge NVFP4 on a
pre-Blackwell card — you are not measuring the format, you are measuring its
fallback.**

Best configuration on each card, which is what a buyer actually feels: 75.1 s →
**40.0 s**, i.e. **1.88×**.

#### Correctness and output

The gate passed 8/8 on the 5080 with values indistinguishable from Ada:
`max_rel` 0.00385–0.00388 against a 0.02 limit, `rel_l2` 0.00105 against 0.005.
Routing density 0.2573–0.2582 against 0.2575–0.2584 on the 4070 Ti — data, not
hardware, as it should be.

| PSNR | dB |
|---|---:|
| Node off vs on — int8, 5080 | 25.13 |
| Node off vs on — NVFP4, 5080 | 23.92 |
| int8 vs NVFP4, both node-off, 5080 | **19.06** |
| Same config, 4070 Ti vs 5080 — int8 | 30.84 |
| Same config, 4070 Ti vs 5080 — NVFP4 | 24.65 |

Two things worth reading off that. **Choosing the weight format moves the image
more than enabling this node does** — 19 dB between int8 and NVFP4, against 25 dB
for switching Sol-Attn on. And the same NVFP4 file scores 24.65 dB across the two
cards while int8 scores 30.84 dB: emulated and native 4-bit arithmetic genuinely
differ, so an NVFP4 preview rendered on an Ada card is not what the format
produces.

As [Quality](#quality--and-why-off-vs-on-psnr-misleads) explains at length, PSNR
between different kernels or quantizations measures *divergence*, not quality —
matched seeds do not survive a changed trajectory. None of these numbers say
which frame looks better.

**Scope.** Seq 17 504 is the short end. Attention is a minority of step time
here, the kernel tables above show the margin widening with length, and a 15 s
clip runs several times longer a sequence. Nothing in this section transfers to
that without measuring it.

### End-to-end, MiniMax-H3 in ComfyUI — SDPA baseline, Triton backend

**Scope:** this is the oldest measurement in the file and the most flattering
one, because both its baseline and its backend are the weak options: ComfyUI's
default `pytorch attention` against the Triton kernel. Kept because it is the
only full end-to-end A/B with per-call instrumentation. For the same run against
SageAttention see the box at the top; for the CuTe backend see the tables above.

864×480, 125 frames (**17 504**-row packed sequence), 8 steps, `res_multistep`,
int8 `fl2va` weights. Measurement pass after warm-up:

| Metric | Node off | Node on | Ratio |
|---|---:|---:|---:|
| **End-to-end wall clock** | 116.1 s | **84.2 s** | **1.38×** |
| Attention, total | 45.5 s | 26.8 s | 1.70× |
| Attention, per dense call | 113.65 ms | 113.32 ms | — |
| Attention, per sparse call | — | **47.87 ms** | **2.37×** |
| Sparse / dense calls | 0 / 400 | 288 / 112 | — |
| Routing density (effective) | — | 0.258 | — |

Dense-call latency across four independent runs: 113.18 / 113.27 / 113.65 /
113.32 ms. That the `off` and `on` runs agree on the *same* code path is the
control for the measurement itself — the instrumentation does not skew results.

Attention accounts for **59 %** of step time here (45.5 s of 76 s sampling), so
kernel speedup translates to wall clock in a sane proportion.

### SM89 — NVIDIA L40 (CuTe DSL)

`selftest.py` on a datacentre Ada card, 48 GB, driver 580.126.20. torch
2.14.0+cu130, CUDA 13.0, SageAttention 2.2.0 built from source for `sm_89`.
Backend picked automatically: `cute_sm89`.

| Sequence | Gate | Density | Sol-Attn | SDPA | SageAttention | vs SDPA | vs Sage |
|---:|:--:|---:|---:|---:|---:|---:|---:|
| 8 192 | PASS | 0.271 | 3.93 ms | 13.02 ms | 5.77 ms | 3.32× | 1.47× |
| 16 384 | PASS | 0.214 | 13.40 ms | 50.00 ms | 24.63 ms | 3.73× | **1.84×** |

Densities match the 4070 Ti to three decimals, as they must — routing is a
property of the model, not the card. First call 7.7 s. Gate `max_abs` 1.2e-4,
`rel_l2` 0.0032. BTHD copies 1.81 ms, 13.5 % of kernel time at 16 384 rows.

**End-to-end on the same card**, MiniMax-H3 at 1344×768, 107 frames, 20 steps,
`res_multistep`, int8 `fl2va` + int8 text encoder:

| Attention backend | Wall clock | vs SDPA |
|---|---:|---:|
| PyTorch SDPA | 353.0 s | — |
| SageAttention (`--use-sage-attention`) | 251.5 s | 1.40× |
| **Sol-Attn node** | **235.3 s** | **1.50×** |

The end-to-end margin over SageAttention (1.07×) is much smaller than the
kernel-only one (1.84×), and the reason is this card rather than the kernel:
peak VRAM was 47.1 GB of 48, so the run is offload-bound and attention is a
smaller share of wall clock than it is on a card with headroom. A kernel table
is not a prediction of end-to-end gain — measure both.

### SM89 — RTX 4070 Ti, both backends

Same measurement as the SM120 table above, so the two are directly comparable.
`selftest.py`, kernel only, 56 heads, head_dim 128, `tau=1.0`, `thresh_type=diag`.

**`cute_sm89`** — what you get with the CuTe runtime installed:

| Sequence | Gate | Density | Sol-Attn | SDPA | SageAttention | vs SDPA | vs Sage |
|---:|:--:|---:|---:|---:|---:|---:|---:|
| 8 192 | PASS | 0.272 | 8.70 ms | 26.06 ms | 11.40 ms | 3.00× | 1.31× |
| 16 384 | PASS | 0.214 | 27.42 ms | 106.49 ms | 39.81 ms | 3.88× | 1.45× |
| 30 976 | PASS | 0.186 | 79.55 ms | 373.51 ms | 131.57 ms | **4.70×** | **1.65×** |

**`triton`** — the fallback when any of the three CuTe packages is missing:

| Sequence | Gate | Density | Sol-Attn | SDPA | SageAttention | vs SDPA | vs Sage |
|---:|:--:|---:|---:|---:|---:|---:|---:|
| 8 192 | PASS | 0.271 | 10.2 ms | 26.4 ms | 11.2 ms | 2.58× | 1.09× |
| 16 384 | PASS | 0.214 | 31.7 ms | 105.4 ms | 40.0 ms | 3.32× | 1.26× |
| 30 976 | PASS | 0.186 | 99.6 ms | 380.0 ms | 133.6 ms | 3.82× | 1.34× |

Two things to read off these tables. Routing density falls as the sequence grows,
so **the longer the video, the more Sol-Attn pays off** — on both backends and
both cards. And the speedup against **SDPA is nearly identical on SM89 and SM120**
(3.00/3.88/4.70 against 3.00/3.98/4.67): that ratio measures the sparsity itself,
which is architecture-independent. The gap against SageAttention differs only
because Sage is relatively stronger on Ada than on Blackwell.

### One-off costs

First sparse call in a process, including the correctness gate and the density
probe: **3.2 s** on `cute_sm89`, **13.4 s** on `triton`. With a warm compile cache
both drop to well under a second (measured 0.05–0.38 s).

**Triton compiles per sequence shape, not once per process.** Every new
resolution or frame count pays that cost again, and on short runs it can eat the
entire gain. Second measurement point — 640×384, 73 frames (**5 548** rows),
20 steps:

| Metric | Node off | Node on | Ratio |
|---|---:|---:|---:|
| Attention, per call | 11.88 ms | **7.33 ms** | 1.62× |
| Attention, excluding one-off | 11.9 s | 8.4 s | 1.41× |
| Compilation for the new shape | — | 3.4 s | — |
| **End-to-end wall clock** | 40.3 s | 39.1 s | **1.03×** |

Shorter sequence → smaller per-call win *and* a smaller attention share of the
step *and* the same fixed compile spread over less work.

### Composing with `ComfyUI-MiniMaxH3-Cache`

Sol-Engine's H3 line composes Sol-Attn with FirstBlockCache, so this is intended.
Verified with `strict=True`, 20 steps, 5 548-row sequence:

```
sparse_calls: 336   dense_calls: 114   last_step: 19   total_steps: 20
```

450 attention calls instead of 1 000 — the cache skipped 11 of 20 forwards — yet
the step number and schedule length stayed correct, `strict` raised nothing, and
the sparse path still ran 336 times. Wall clock: 39.1 s → **20.0 s**.

This is precisely why the step index is read from `sample_sigmas` rather than
counted: **a forward counter would drift on every skipped step.**

#### The cache is the bigger lever — and `max_steps` is not what tunes it

NVIDIA's validated H3 recipe is identical for SM89 and SM120
(`config/minimax_h3/rtx4090_fullopt.toml`, `rtx5090_fullopt.toml`) and it matches
this node's defaults exactly: `tau=1.0`, `thresh_type=diag`, 10 dense steps out
of 50, 2 dense layers, gate on. Its cache arm is TeaCache with threshold `0.10`,
5 retained steps and 1 cooldown step. In their controlled attribution run on a
4090 the cache was worth **3.18×** and Sol-Attn a further **1.22×** — the cache
does the heavy lifting.

`ComfyUI-MiniMaxH3-Cache` maps onto that recipe one-to-one: `resuse_threshold`
is the threshold (already 0.10), `max_steps` counts consecutive skipped forwards
and resets after each real one, so the cooldown of 1 is implicit. Only
`max_steps` differs — the node ships 2 against NVIDIA's 5. Measured here at
864×480, 125 frames, 20 steps, Sol-Attn on the CuTe backend throughout:

| `max_steps` | forwards skipped | attention calls | end-to-end |
|---:|---:|---:|---:|
| 2 (node default) | 11 of 20 | 450 | 101.2 s |
| 5 (NVIDIA) | 12 of 20 | 400 | **79.1 s** |

**Read that carefully.** Raising `max_steps` bought exactly *one* extra skipped
forward — the binding constraint is the accumulated-`rel_l1` threshold, not the
consecutive-skip cap. Of the 22 s difference, roughly 6 s is that forward, 3 s is
a warm kernel-compile cache in the second run and 2 s is attention; the rest is
the same run-to-run noise this card shows everywhere. Treat 1.28× as an upper
bound, not a result.

Frames between the two settings compare at 23.4 dB — the same order of divergence
the sparse path itself produces. More aggressive caching is not free.

---

## Tuning

**Scope:** the table below was measured on the **Triton** backend, where the
default configuration loses to SageAttention. On `cute_sm89` the default already
wins (1.06× at seq 20 530, 1.32× at 45 241) and these levers are optional. Reach
for them when you are stuck on Triton, or when you want more than the default on
a short sequence.

Two levers move the cost materially. Both trade quality, so neither is a default.
Measured in isolation at seq 20 530, 56 heads, against SageAttention (~56.5 ms):

| `sink_mode` | `tau` | kernel | prefix recompute | total | vs Sage |
|---|---:|---:|---:|---:|---:|
| prefix (1 495 rows) | 1.0 | 57.3 ms | 14.1 ms | 83.4 ms | 0.78× |
| prefix | 1.5 | 33.7 ms | 12.4 ms | 50.1 ms | 1.13× |
| prefix | 2.0 | 26.0 ms | 12.4 ms | 42.4 ms | 1.34× |
| text (537 rows) | 1.0 | 42.3 ms | 5.2 ms | 51.5 ms | 1.10× |
| text | 1.5 | 25.3 ms | 5.2 ms | 34.5 ms | 1.64× |

**`sink_mode` is the lever most people should try first, and its cost is
shape-dependent in a way the reference does not spell out.** NVIDIA's H3
integration estimates `prefix` at "about 1 % density" over `text` — but that was
measured with a 951-row prefix in a 38 247-row sequence, i.e. **2.5 %** of the
sequence. Put the same prefix in a 20 530-row sequence and it is **7.3 %**, and
the cost scales with it: the sink forces those blocks exact for *every* query
*and* has to be recomputed densely. `text` is the policy the kernel's own README
describes; `prefix` was added by the H3 team as an audio-quality safeguard, and
that safeguard gets three times more expensive at short sequence lengths.

The risk is concrete and named: NVIDIA recorded a prompt whose picture scored
best of its set while its dialogue fell apart. If your outputs carry speech,
verify on your own material before keeping `text`.

`tau` behaves predictably and is shape-independent: the routing threshold yields
the same `threshold_density` (0.155 at `tau=1.0`, 0.102 at 1.25, 0.064 at 1.5)
regardless of sequence length. Only the sink's contribution varies.

## Quality — and why off-vs-on PSNR misleads

Sol-Attn is an **approximation**, not a lossless path. Same seed, 20 steps,
5 548-row sequence: decoded frames compare at **22.4 dB** PSNR.

That number alone is misleading, as the control run shows. At `tau = -1000` the
router admits **every** block (measured density: exactly 1.0), so the kernel
computes full attention and skips nothing. PSNR against the dense path is still
only **30.7 dB**:

| Configuration | Routing density | PSNR vs dense |
|---|---:|---:|
| `tau = -1000` (nothing skipped) | 1.000 | 30.7 dB |
| `tau = 1.0` (production policy) | 0.311 | 22.4 dB |

In other words the **floor is ~31 dB**, and it comes from swapping the attention
implementation at all. Per-call relative error is 0.1 % (`rel_l2` from the gate)
and the worst element is exactly half a bf16 ulp — as good as the format allows.
Applied ~960 times (20 steps × 48 sparse layers) inside a non-linear sampler,
trajectory divergence becomes macroscopic. Routing adds roughly 8 dB on top.

This is not an integration defect: the correctness gate passes, prefix rows match
dense attention, and the decline path returns the original backend's output
bit-for-bit — all covered by GPU tests. NVIDIA's own reference warns about it —
*"a visual metric alone will rate it too highly on this model"* — and reports
LPIPS 0.293 for H3.

**Practical takeaway:** judge it on your own material, not on PSNR. For work that
must match the native trajectory, flip `enabled` off and compare the same prompt
at the same seed.

---

## Requirements

| | |
|---|---|
| ComfyUI | with native MiniMax-H3 (`comfy.ldm.minimax`) |
| GPU | NVIDIA, compute capability ≥ 8.0 |
| PyTorch | ≥ 2.10 |
| CUDA | ≥ 12.8 |
| Triton | ≥ 3.6 |
| CuTe DSL | **strongly recommended**: `nvidia-cutlass-dsl` ≥ 4.5, `cuda-python`, `apache-tvm-ffi` |

Backend is selected automatically from the GPU architecture:

| Architecture | Example | Backend |
|---|---|---|
| SM89 | RTX 4090, RTX 4070 Ti | CuTe DSL |
| SM90 | H100 | CuTe DSL |
| SM100 | B200 / GB200 | CuTe DSL |
| SM120 | RTX 5080, RTX 5090, RTX PRO 6000 Blackwell | CuTe DSL |
| SM80 / SM86 | A100, RTX 3090 | Triton |

Missing any of `cutlass.cute`, `cuda-python` or `tvm_ffi` falls back to Triton
regardless of architecture — and that fallback is the difference between a win
and a loss on SM89. The node prints the selected backend when it mounts;
`backend=triton` on an SM89+ card means one of those three packages is absent.

### After a GPU upgrade, rebuild SageAttention

Sol-Attn resolves its own backend from the live device, so it needs nothing after
a card swap. **SageAttention does not.** A source build compiles for the card that
was installed at the time, and a binary holding only `sm_89` gives no PTX to JIT
from, so ComfyUI started with `--use-sage-attention` fails on a Blackwell card
with:

```
Error running sage attention: CUDA error: no kernel image is available for execution on the device
```

Check the binary rather than the import — `import sageattention` succeeds either
way, and so does a plain `sageattn()` call, because it can dispatch to a variant
that happens to exist:

```bash
cuobjdump --list-elf .../site-packages/sageattention/_qattn_sm89*.so | grep -o 'sm_[0-9]*'
```

Rebuilding for `sm_120` on CUDA 13.3 needs two flags that are not obvious:

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
export CC=/usr/bin/clang++ CXX=/usr/bin/clang++   # NOT gcc — see below
export CPATH=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1/include   # clang has no omp.h
pip install --no-build-isolation .
```

`CC` rather than `NVCC_PREPEND_FLAGS`, because `torch/utils/cpp_extension.py`
reads `-ccbin` from `CC` and appends its own — pass it any other way and you get
two `-ccbin` flags, of which nvcc uses the last. And clang rather than gcc
because nvcc 13.3 drops the `typename` that `ATen/core/List_inl.h:202` already
has when it generates host code, which every gcc then rejects. The device pass is
fine on either; only the host pass fails.

## Installation

### One command

From your ComfyUI directory:

```bash
curl -sSLO https://raw.githubusercontent.com/quzopl/ComfyUI-SolAttn-H3/master/install.sh
bash install.sh
```

The script checks torch, CUDA, Triton and the GPU's compute capability **before**
changing anything, then fetches the node, fetches the Sol-Attn kernel from
NVlabs/Sana, installs it plus the CuTe DSL runtime into ComfyUI's own Python
environment, and prints the backend you ended up with. `--no-cute` skips the CuTe
runtime, `--skip-selftest` skips the closing benchmark, `--comfyui PATH` points it
at a ComfyUI it cannot autodetect.

### By hand

```bash
git clone https://github.com/quzopl/ComfyUI-SolAttn-H3 \
  ComfyUI/custom_nodes/ComfyUI-SolAttn-H3

# The kernel is installed from NVlabs, not vendored here
git clone --branch sol-engine --depth 1 https://github.com/NVlabs/Sana.git ~/sana-sol-engine
uv pip install --python ComfyUI/venv/bin/python \
  -e ~/sana-sol-engine/techniques/sparse_backends

# The CuTe DSL runtime. Skipping it drops every architecture to Triton, which on
# SM89 turns a 1.06x win into a 0.78x loss. apache-tvm-ffi is needed too and the
# upstream docs do not mention it.
uv pip install --python ComfyUI/venv/bin/python \
  "nvidia-cutlass-dsl>=4.5" cuda-python apache-tvm-ffi
```

Check your environment without launching ComfyUI:

```bash
ComfyUI/venv/bin/python ComfyUI/custom_nodes/ComfyUI-SolAttn-H3/selftest.py
```

It prints the GPU, selected backend, correctness-gate verdict, routing density,
and a speed comparison against SDPA and SageAttention at the sequence lengths you
pass in. If the gate fails on your card you find out in a minute instead of after
a week of odd artifacts.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `enabled` | on | Turns everything off without rewiring the graph. |
| `tau` | 1.0 | Higher = fewer K/V blocks computed exactly. This is the validated H3 policy; per-shape calibration in the reference returned an empty route set. |
| `thresh_type` | `diag` | `exact` uses the full-covariance threshold — more accurate, more expensive. |
| `first_dense_steps` | 0.2 | Below 1 it is a fraction of the schedule, 1 and above a fixed step count. 0.2 matches the reference's 10-of-50. |
| `first_dense_layers` | 2 | First N DiT blocks stay dense. Counted from zero. |
| `sink_mode` | `prefix` | `prefix` keeps text + conditioning + audio exact. `text` reproduces the reference policy. |
| `correctness_gate` | on | Once per shape, compares the kernel against SDPA on real QKV. A failure aborts generation. |
| `strict` | off | Turns every unintended decline into an exception. For validation, not daily use. |
| `kv_splits` | 1 | How many pieces the kernel splits the K/V axis into. **SM90 only** — see below. |

### `kv_splits` is an H100 option

The kernel takes it, but accepts anything above 1 on SM90 alone; elsewhere it
raises `kv_splits=2/4 is currently available on SM90 only`. It raises from
inside the sampler, on the first sparse call, with the text encoder and the
transformer already resident — so the node checks the architecture when it
mounts and refuses there instead. Every Sol-Engine config leaves this at 1.

The correctness gate gets the same value as the production path. Above 1 the
kernel reduces across split accumulators, which changes the summation order and
so the rounding; gating at 1 would clear arithmetic that never runs.

### Overriding from the environment

Sol-Engine drives every setting from environment variables, so a config under
`config/minimax_h3/` transfers here unchanged. When set, these win over the
widgets:

`SOL_ATTN_TAU`, `SOL_ATTN_THRESH_TYPE`, `SOL_ATTN_FIRST_DENSE_STEPS`,
`SOL_ATTN_FIRST_DENSE_LAYERS`, `H3_SOL_SINK_MODE`, `SOL_ATTN_CORRECTNESS_GATE`,
`SOL_ATTN_STRICT`, `SOL_ATTN_KV_SPLITS`.

An unset variable changes nothing. A malformed one raises rather than leaving
the widget's value quietly in place — a typo in `SOL_ATTN_THRESH_TYPE` that fell
back to the widget would be the same class of bug as a sparse configuration
silently running dense. Each override is logged with the value it replaced:

```
[sol-attn-h3] environment override SOL_ATTN_TAU=2.5 (widget had 1.0)
```

### Why `sink_mode=prefix` rather than `text`

The sink is a contiguous K/V range kept exact for all queries. The reference
policy covers only the text rows. Here the default covers the whole prefix,
audio rows included — because those are **generated** (the model returns a
velocity for them), and NVIDIA's handoff recorded a prompt whose picture scored
best of its set while its dialogue fell apart. Cost versus the reference policy:
about 1 % density and 1 % extra dense query rows.

## How it is wired

Four public ModelPatcher APIs. **No file under `comfy/` is modified** — unlike
some other H3 acceleration nodes, which patch core files.

| Mechanism | Role |
|---|---|
| `transformer_options["optimized_attention_override"]` | intercepts attention; returning `func(...)` gives the dense fallback |
| `WrappersMP.DIFFUSION_MODEL` wrapper | once per forward: layout, sink range, step number |
| `patches_replace["dit"][("double_block", i)]` | stamps the real block index |
| `WrappersMP.OUTER_SAMPLE` wrapper | run boundaries and the aggregate check |

The block index is **stamped, not counted**, because `token_refiner` also calls
`Attention` with head_dim 128 — counting calls would shift `first_dense_layers`.

## Diagnostics

The node logs three things, all aimed at catching a configuration that asks for
sparse attention and quietly computes dense:

```
[sol-attn-h3] SM89, CuTe DSL unavailable, backend=triton, DiT blocks: 50
[sol-attn-h3] correctness gate PASS max_abs=0.12500 mean_abs=0.000305 rel_l2=0.00111
              ref_max=33.000 max_rel=0.00379 over_1e2=4.14e-03
              limits={'max_rel': 0.02, 'mean_abs': 0.002, 'rel_l2': 0.005}
[sol-attn-h3] routing density {'blocks': 87, 'sink_blocks': 5,
              'threshold_density': 0.24153, 'effective_density': 0.30748}
[sol-attn-h3] {'backend': 'triton', 'sparse_calls': 288, 'dense_calls': 112,
              'sparse_fraction': 0.72,
              'attn_ms_per_call': {'sparse': 7.29, 'dense': 12.09, ...}, ...}
```

- **correctness gate** — once per shape; `tau=-1000` admits every block, so the
  comparison against SDPA measures the kernel's arithmetic, not the routing policy
- **routing density** — near 1.0 means the router isn't routing; 0.0 means it
  collapsed
- **run statistics** — sparse and dense call counts, decline reasons, attention
  time on both paths

Decline reasons `disabled`, `warmup_step` and `dense_layer` are intended and
silent. `kernel_unavailable`, `oom`, `mask_present`, `layout_unknown`, `batch`,
`dtype`, `head_dim`, `no_layout` and `seq_mismatch` are logged once each, and
raise under `strict`. On top of that, a run that clears warm-up without a single
sparse call ends with a loud warning — the reason that does the damage
(`warmup_step`) is itself legitimate, so the error exists only in aggregate.

### Divergences from the NVIDIA reference

Three, each because ComfyUI exposes information the SGLang runtime did not:

1. **Step number from `sample_sigmas`**, not guessed from the direction of
   timestep change. The reference documents two failures on that mechanism — a
   reset that never fired, and one that fired every step; both reported dense
   attention under a sparse label.
2. **Sink range from `PackedLayout.segments`**, not inferred from discontinuities
   in `video_indices`.
3. **Relative gate criterion.** The reference's absolute `max_abs ≤ 0.08` assumes
   an activation distribution. On real H3 QKV, `max_abs = 0.125` at `ref_max = 33`
   is exactly half a bf16 ulp — the theoretical minimum representation error at
   that magnitude — while `mean_abs` and `rel_l2` had 6.5× and 4.5× headroom.
   Only the max criterion changed, to `max_rel ≤ 0.02`; `mean_abs` and `rel_l2`
   are NVIDIA's, untouched. Three tests verify the loosening did not disarm the
   gate.

## Known limitations

- **MiniMax-H3 only.** The node refuses to mount on any other model, with an
  explicit error.
- **Kernel contract:** bf16, head_dim exactly 128, contiguous BTHD, batch 1.
  A violation means dense attention, not an exception.
- **`torch.compile`:** the override is a callable in `transformer_options`, so a
  graph break occurs. Not addressed.
- **Per-shape compilation:** 3–13 s each time the resolution or frame count
  changes.
- **Contiguous copies:** H3 hands over Q/K/V as views into the packed `qkv_proj`
  buffer, so one copy is unavoidable — 3 × 424 MiB at 31 k rows, about 6.5 % of
  kernel time. Out of memory latches the run onto the dense path.
- **`kv_splits` above 1 needs SM90.** The node refuses to mount elsewhere.
- **Environment overrides do not invalidate ComfyUI's cache.** A node is cached
  by its input values, and environment variables are not inputs. After changing
  one, touch a widget or reload the workflow, or the previous patch is reused.

## Testing

The suite does not run in place from `custom_nodes`: the repo has an
`__init__.py` (ComfyUI requires one), so pytest tries to import the directory as
a package and every test errors on a relative import. Mount it under a valid
module name instead:

```bash
rm -rf /tmp/h && mkdir -p /tmp/h
cp -r ComfyUI/custom_nodes/ComfyUI-SolAttn-H3 /tmp/h/solattn_h3
rm -rf /tmp/h/solattn_h3/.git
cd /tmp/h && ComfyUI/venv/bin/python -m pytest solattn_h3/tests -q   # 79 tests
```

`layout.py` and `state.py` are CUDA-free and unit-tested; `attention.py` and
`kernel.py` have GPU integration tests that skip when no CUDA is present.
`bench/ab_bench.py` drives the ComfyUI API for the A/B measurements above.

## License and attribution

Apache-2.0 — see [LICENSE](LICENSE). The kernel and the acceleration policy come
from NVlabs/Sana; details in [NOTICE](NOTICE).

Paper: [Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention
Sparsification](https://arxiv.org/abs/2607.24027) (arXiv:2607.24027).
