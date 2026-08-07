# Sol-Attn for MiniMax-H3 in ComfyUI — design

Date: 2026-08-06
Status: approved; implementation plan written and executed

## 1. Goal

Wire Sol-Attn — the training-free sparse attention from NVIDIA Sol Engine — into
ComfyUI's native MiniMax-H3 implementation, as an optional custom node.

Sol Engine composes five acceleration techniques. For H3 the validated line is
context parallelism + kernel fusion + Sol-Attn + FirstBlockCache. Of that set,
only Sol-Attn was missing from this installation:

| Sol Engine pillar | Status in `~/comfyX/ComfyUI` |
|---|---|
| Cross-step cache | `ComfyUI-MiniMaxH3-Cache`, or `ComfyUI-Spectrum-MiniMax-H3` |
| Quantization / kernel fusion | `int8-fast` plus `minimax_h3_*_pruned_int8_convrot` weights |
| **Sparse attention (Sol-Attn)** | **missing — the scope of this document** |
| Context parallelism | not applicable (single GPU) |

## 2. Scope

In scope:

- one custom node `SolAttnH3` (MODEL → MODEL) that wires Sol-Attn into H3
- automatic kernel-backend selection by GPU architecture
- graceful degradation to dense attention with an explicit, counted reason
- a correctness gate and a routing-density probe
- a selftest script to run on the target machine

Out of scope:

- other video models (WAN, HunyuanVideo, LTX) — Sol-Engine has validated lines
  for them, but each has a different sequence layout and a different place for
  the sink
- our own cache or kernel-fusion implementation (already covered)
- `tau` autotuning
- `torch.compile` interoperability (the override is a callable in
  `transformer_options`, so a graph break occurs — note it in the README, do not
  fix it now)
- publishing the node (possible later; v1 targets working on the author's setup)

## 3. Environment

Development machine (build and correctness testing):

- ComfyUI v0.30.1 in `~/comfyX/ComfyUI`, a Python 3.13 venv
- torch 2.12.0+cu130, Triton 3.7.0, `sageattention` and `flash_attn` present
- RTX 4070 Ti 12 GB — **SM89 → Triton backend**
- 62 GB RAM; H3 weights: 20 GB int8 DiT, 4.9 GB `qwen3vl_4b_fp8` as a light encoder

Target machine (generation):

- RTX PRO 6000 Blackwell — **SM120 → CuTe DSL backend**
- the node must degrade correctly on weaker consumer cards
  (Ada SM89, Ampere SM86/SM80 → Triton)

The `sol-attn` 0.5.0 requirements hold on both: Python ≥3.10, torch ≥2.10,
CUDA ≥12.8, Triton ≥3.6.

## 4. Contract compatibility

The kernel's contract (`sol_attn/interface.py::_validate_inputs`) against what
ComfyUI provides in `comfy/ldm/minimax/model.py`:

| Kernel requirement | H3 in ComfyUI | Verdict |
|---|---|---|
| head_dim == 128 | `attention_head_dim=128` (`model.py:414`) | matches |
| dtype bfloat16 | `supported_inference_dtypes = [bfloat16, float32]` (`supported_models.py:973`) | matches |
| contiguous BTHD | supplied as BHSD, needs one copy | copy required |
| no mask | `mask=None` (`model.py:181`) | matches |
| compute capability ≥ 8.0 | machine-dependent | checked at runtime |

The BTHD copy is 3 × 424 MiB at ~31 k rows. NVIDIA's reference measures an
equivalent copy at <0.1 ms against 6.7 ms of attention.

## 5. Integration points

All supported — no ComfyUI core file is patched.

| Mechanism | Location | Role |
|---|---|---|
| `transformer_options["optimized_attention_override"]` | `comfy/ldm/modules/attention.py:148` (`wrap_attn`) | intercept attention; `func(*args, **kwargs)` gives the dense fallback |
| `WrappersMP.DIFFUSION_MODEL` | `comfy/ldm/minimax/model.py:502` | once per forward: layout, sink, step index, layer reset |
| `patches_replace["dit"][("double_block", i)]` | `comfy/ldm/minimax/model.py:620` | stamp the real block index |

ModelPatcher API: `clone()`, `add_wrapper_with_key()`,
`set_model_patch_replace(patch, "dit", "double_block", i)`.

## 6. Architecture

```
~/sol/                          # repo, symlinked as custom_nodes/ComfyUI-SolAttn-H3
├── __init__.py    # NODE_CLASS_MAPPINGS for ComfyUI
├── nodes.py       # the SolAttnH3 node — UI and mounting only
├── state.py       # step/layer clock, decline reasons, counters
├── layout.py      # sink range from layout.segments (a pure function)
├── kernel.py      # lazy sol_attn import, backend detection and reporting
├── attention.py   # the override adapter: contract, kernel, dense fallback
├── selftest.py    # standalone diagnostic script (target machine)
└── tests/         # CUDA-free unit tests
```

The split is driven by testability: `layout.py` and `state.py` never touch CUDA
and take unit tests. `attention.py` and `kernel.py` need a GPU.

### Flow of one sampler step

1. The `DIFFUSION_MODEL` wrapper — once per forward:
   - reads `minimax_payload["layout"]`
   - sink = `(0, start of the "video" segment)` for `sink_mode="prefix"`
   - step index = position of `timestep` in `transformer_options["sample_sigmas"]`
   - resets the layer counter
2. `patches_replace["dit"][("double_block", i)]` — 50×: stamps
   `transformer_options["solattn_block"] = i`, calls the original block
3. `Attention.forward` → `optimized_attention` → `wrap_attn` → the adapter:
   - decline → `func(*args, **kwargs)`
   - accept → BHSD→BTHD contiguous bf16 → `sol_attn(...)` with the sink →
     dense recomputation of the prefix query rows → `reshape(1, s, heads*128)`

### Divergences from NVIDIA's reference

Both follow from ComfyUI exposing information the SGLang runtime used in
`models/minimax_h3/optimized/sol_attn_h3.py` did not have.

**Step index from the schedule, not from the direction of timestep change.** The
reference detects the start of a request by watching for a reversal in timestep
direction, and documents two failures on that mechanism: a reset that never
fired (measurement started at step 49 and ran fully sparse while reporting ten
dense steps), and a reset that fired every step (everything declined as
`warmup_step`, so both configurations measured dense attention under a sparse
label). `sample_sigmas` gives the step index directly.

**Sink from explicit segments.** The reference infers the start of the video tail
from discontinuities in `video_indices`. `PackedLayout.segments` provides
labelled `(a, b, kind)` tuples for `text / cond / ref_img / audio / video`.

**Block index stamped, not counted.** `token_refiner` (`model.py:584`) also calls
`Attention` with head_dim 128, on the text rows alone. Counting calls would shift
`first_dense_layers`. The refiner also falls out on the sequence-length check,
but correctness must not depend on two mechanisms at once.

## 7. Sparse policy

Defaults from the validated H3 line:

| Parameter | Default | Rationale |
|---|---|---|
| `enabled` | on | one switch that disables everything without rewiring the graph |
| `tau` | 1.0 | the reference passes it directly; its earlier per-shape calibration returned an empty route set on H3 and the configuration silently ran dense |
| `thresh_type` | `diag` | `exact` is the full-covariance threshold, more expensive |
| `first_dense_steps` | 0.2 of the schedule | see below |
| `first_dense_layers` | 2 | counted from zero |
| `sink_mode` | `prefix` | see below |
| `correctness_gate` | on | |
| `strict` | off | |

**`first_dense_steps` as a fraction.** The reference's `10` comes from a 50-step
schedule. At 20 steps that would mean running half the schedule dense. Since
`sample_sigmas` is available, a value below 1 is interpreted as a fraction of the
schedule and a value of 1 or more as a fixed step count.

**`sink_mode="prefix"`, not `text`.** The sink covers text, conditioning rows and
audio — everything before the video tail. The audio rows are generated (the model
returns a velocity for them), and NVIDIA's handoff recorded a prompt whose
picture scored best of its set while its dialogue fell apart. Cost against the
reference policy: about 1 % density and 1 % additional dense query rows. `text`
remains available to reproduce the reference exactly.

**The sink applies to keys, not queries.** The kernel makes the sink range exact
as K/V, but its own query rows still route sparsely. The kernel's README is
explicit that an MMDiT integration must recompute them densely. Exactness is
applied at 64-token block granularity, rounding outward.

**Composition with a cache.** Sol-Engine's H3 line composes Sol-Attn with
FirstBlockCache, so working alongside `ComfyUI-MiniMaxH3-Cache` is intended. When
the cache skips a step, the wrapper simply does not fire; the `sample_sigmas`
clock stays correct where a call counter would drift.

## 8. Error handling

Principle: a configuration that asked for sparse attention and quietly got dense
is a wrong measurement wearing the right label. Every decline carries a reason,
not a boolean.

| Reason | Behaviour |
|---|---|
| `warmup_step`, `dense_layer` | intended — silent, counted only |
| `disabled` | intended — the `enabled` switch is off, silent |
| `no_layout` | the wrapper never saw `minimax_payload["layout"]` — logged once |
| `seq_mismatch` | `token_refiner` and anything unexpected — logged once |
| `dtype`, `head_dim`, `mask_present` | kernel contract — logged once |
| `arch_unsupported` (<SM80), `kernel_import` | logged once at mount |
| `kernel_error` | exception from the kernel → dense, traceback logged once |
| `oom` | latches onto dense for the rest of the run |

**Aggregate check at end of run.** If sampling passed `first_dense_steps` and made
not a single sparse call — a loud warning. Per-call reasons cannot catch this,
because the reason that does the damage (`warmup_step`) is itself legitimate. The
error exists only in aggregate, so the check must be in aggregate too.

`strict=True` turns every unintended reason into an exception. For validation,
not for daily use.

**Correctness gate** — once per sequence shape. `tau=-1000` admits every block, so
the comparison against SDPA measures the kernel's arithmetic, not the routing
policy. On real QKV and all heads: a probe on random tensors answers a question
about the kernel, not about this model at this shape. Running the gate at the
production head count also warms Triton's autotuning in `preprocess.prepare`,
which keys on `T` alone — a first call at one head would cache a configuration
chosen for a single-head grid.

Limits: `max_rel ≤ 0.02`, `mean_abs ≤ 0.002`, `rel_l2 ≤ 0.005`. A failure raises;
silently accepting a broken kernel is worse than no speedup.

**Divergence from the reference, established by measurement.** The original design
adopted NVIDIA's limits wholesale, including the absolute `max_abs ≤ 0.08` (or
0.15 above 32 k tokens). The first run on real H3 failed it: `max_abs = 0.125`
with `mean_abs = 0.000305` (6.5× headroom) and `rel_l2 = 0.00111` (4.5× headroom).

Two hypotheses were tested:

1. *Partial last block* — `T = 5548` is 86 full 64-token blocks plus 44, whereas
   every shape in the spike was a multiple of 64. Disproved: the same `T` on
   synthetic tensors gives `max_abs = 0.00012`, identical to full-block shapes.
2. *Activation scale* — confirmed. Measured `ref_max = 33.0`. bfloat16 has
   7 mantissa bits, so for `|x| ∈ [32, 64)` one ulp is `32·2⁻⁷ = 0.25` and the
   correct-rounding bound is half an ulp, i.e. **exactly 0.125**. The worst
   element was the theoretical minimum representation error at that magnitude.

Conclusion: the threshold was not too tight, the criterion was non-transferable —
an absolute limit in output space assumes the activation distribution it was
measured on. **Only** the maximum criterion changed, to the relative
`max_rel = max_abs / ref_max`. `mean_abs` and `rel_l2` remain exactly as NVIDIA
set them. Three tests guard against the loosening having disarmed the gate:
rejection of a kernel perturbed by 5 %, acceptance of a perfect one, and
independence of the verdict from output scale.

**Density probe** — once. Reports `threshold_density` and `effective_density`. A
density near 1.0 means the routing is not routing; 0.0 means it collapsed.

Two things deliberately left out of v1: `torch.compile` interaction (the override
is a callable in `transformer_options`, so a graph break occurs — note it in the
README rather than fix it) and any form of `tau` autotuning.

## 9. Validation plan

**Step 0 — installation spike, before writing the node.**
`pip install -e techniques/sparse_backends` into the comfyX venv, then a
standalone script calling `sol_attn()` on synthetic tensors at the real shape
(~31 k × 56 × 128 bf16) against SDPA. It answers the four questions that would
invalidate the rest of the plan: does Triton compile, does the gate pass, what
density comes out, what do the contiguous copies cost.

**Step 1 — unit tests, no GPU.** Only where they guard against silent regression —
i.e. the three places where the reference had documented failures:

- `layout.py`: sink for t2va, fl2va with keyframes, ref2va with refs
- `state.py`: the clock under skipped steps, reset between runs, the
  `first_dense_layers` off-by-one
- the backend-selection table per compute capability

**Step 2 — integration on real H3, locally.** Headless ComfyUI, a minimal H3 graph
at low resolution, `strict=True`. Evidence to collect:

- `gate PASS` on real QKV
- `sparse_calls > 0` in the measured run
- `declined` containing only intended reasons plus `seq_mismatch` from the refiner
- density inside (0, 1) — neither 1.0 (routing not routing) nor 0.0 (collapsed)
- A/B at the same seed: off vs on — frame difference and wall clock
- the same run with `MiniMaxH3-Cache` enabled

**What this machine cannot prove:** the numerics and performance of the CuTe SM120
path, or production-scale speedup. Hence the selftest script for the target
machine: it prints the selected backend, the gate verdict, density, and
attention's share of step time.

## 10. Risks

| Risk | Assessment |
|---|---|
| The Triton path (Ada, Ampere) is described in the kernel's README as a research implementation "for portability, kernel studies"; NVIDIA benchmarked SM90/100/120 | Sol-Attn may come out worse than the current SageAttention on weaker cards. The node must be switchable with one toggle and must report its backend. *(Disproved in practice: 1.09–1.34× against SageAttention across the sweep.)* |
| ~1.3 GB of transients for contiguous copies at 31 k rows | Irrelevant on a PRO 6000 (96 GB), significant on 12–16 GB cards. Hence the `oom` latch. |
| Composition with `MiniMaxH3-Cache`, which patches core files | Untested at design time. Verified explicitly in step 2. |
| ComfyUI version | The integration rests on public APIs, but `PackedLayout.segments` and the `minimax_payload` signature are H3 implementation details. The contract is checked at mount, with an explicit error rather than silent degradation. |

## 11. License and attribution

The `sol-attn` kernel comes from NVlabs/Sana (Apache-2.0) and is **installed**,
not vendored — v1 creates no redistribution obligation. The integration draws on
`models/minimax_h3/optimized/sol_attn_h3.py` (Apache-2.0); the node's README
points at the source, the paper (arXiv 2607.24027) and the license. If the node
is ever published, the vendoring decision has to be revisited.
