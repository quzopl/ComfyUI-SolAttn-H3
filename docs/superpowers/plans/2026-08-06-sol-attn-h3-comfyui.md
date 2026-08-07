# Sol-Attn for MiniMax-H3 in ComfyUI — implementation plan

> **Status: executed.** Every task below shipped. This document was originally
> written in Polish with full test code inlined; it has since been translated,
> and the test bodies now point at the files they became rather than duplicating
> them, since those files are the authoritative version. The reasoning behind the
> design lives in
> [the spec](../specs/2026-08-06-sol-attn-h3-comfyui-design.md).

**Goal:** a custom node that wires the training-free sparse attention Sol-Attn
(NVIDIA Sol Engine) into ComfyUI's native MiniMax-H3, with a dense fallback and
hard correctness validation.

**Architecture:** the node clones the ModelPatcher and mounts three things on
public ComfyUI APIs: the attention override
(`transformer_options["optimized_attention_override"]`), a `DIFFUSION_MODEL`
wrapper that refreshes layout/sink/step index once per forward, and
`patches_replace["dit"]` to stamp the block index. CUDA-free logic (`layout.py`,
`state.py`) is separated from CUDA logic (`attention.py`, `kernel.py`) so it can
be unit-tested.

**Stack:** Python 3.13, torch 2.12+cu130, Triton 3.7, `sol-attn` 0.5.0
(Apache-2.0), ComfyUI v0.30.1, pytest.

## Global constraints

- The `sol_attn` kernel requires: contiguous **BF16**, **BTHD** layout
  `[B, T, H, 128]`, **head_dim exactly 128**, CUDA, compute capability **≥ 8.0**.
  Violating any of these means a dense fallback with a named reason — never an
  exception on the production path.
- **Zero modifications to files under `~/comfyX/ComfyUI/comfy/`.** Integration
  goes exclusively through public ModelPatcher APIs.
- Every decline of the sparse path carries a **reason (string)**, is counted, and
  is logged once. Silent degradation to dense attention is treated as a bug.
- Policy defaults come from
  `vendor/sana-sol-engine/models/minimax_h3/optimized/sol_attn_h3.py` and **must
  not be changed without measurement**.
- Correctness-gate limits: `max_abs ≤ 0.15` (T ≥ 32768) or `≤ 0.08`,
  `mean_abs ≤ 0.002`, `rel_l2 ≤ 0.005`.
  *(Superseded during execution — see the spec, §8: the max criterion became
  relative, `max_rel ≤ 0.02`, on measured evidence.)*
- Repo: `~/sol`. Local git identity must not use the author's personal data.

## File structure

| File | Responsibility | CUDA? |
|---|---|---|
| `__init__.py` | `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS` | no |
| `kernel.py` | architecture and backend detection, lazy `sol_attn` import | probe only |
| `layout.py` | sink range from `PackedLayout.segments` | no |
| `state.py` | step/layer clock, decline reasons, counters, gate bookkeeping | no |
| `attention.py` | the override adapter: tensor contract, kernel, dense fallback | yes |
| `nodes.py` | the `SolAttnH3` node — UI and mounting | no |
| `selftest.py` | diagnostics for the target machine | yes |
| `tests/` | CUDA-free unit tests | no |
| `bench/ab_bench.py` | A/B measurement with and without the node, via the ComfyUI API | yes |

---

### Task 1: `kernel.py` — backend detection

**Files:** create `kernel.py`, `tests/test_kernel.py`, `pyproject.toml`, `README.md`

**Produces:** `Probe` (dataclass: `arch`, `cute_available`, `backend`,
`available`, `error`), `probe(device=None) -> Probe`,
`load_sol_attn() -> Callable`, `backend_for_arch(arch, cute_available) -> str`.

- [x] **Step 1: write the failing test** — parametrized over
  (SM90/SM100/SM120 with and without CuTe, SM89/SM86/SM80), plus a `RuntimeError`
  below SM80. Shipped as `tests/test_kernel.py`.
- [x] **Step 2: run it, confirm FAIL** — `ModuleNotFoundError: kernel`.
- [x] **Step 3: implement.** `backend_for_arch` reproduces the released table
  (`{(9,0): "cute_sm90", (10,0): "cute_sm100", (12,0): "cute_sm120"}`, everything
  else ≥8.0 → `"triton"`, below 8.0 → `RuntimeError`) locally, so the tests run
  without a GPU; a comparison test pins it against
  `sol_attn.interface._CUTE_BACKENDS`. `probe()` reads
  `torch.cuda.get_device_capability()`, checks the `cutlass.cute` +
  `cuda.bindings.driver` imports, and caches the result. `load_sol_attn()` does
  `from sol_attn import sol_attn`.
- [x] **Step 4: run, confirm PASS.**
- [x] **Step 5: commit.**

---

### Task 2: `layout.py` — sink range

**Files:** create `layout.py`, `tests/test_layout.py`

**Produces:** `SinkRange` (dataclass: `start`, `tokens`, `seq_len`,
`video_start`), `sink_from_segments(segments, seq_len, sink_mode) -> SinkRange`,
`SINK_MODES = ("prefix", "text")`.

The input contract mirrors `PackedLayout.segments` from
`comfy/ldm/minimax/model.py`: a list of `(a, b, kind)`, kinds
`text | cond | ref_img | audio | ref_audio | video`, segments contiguous and
sorted, with exactly one `video` (the target segment) — the same assumption
`model.py:634` makes.

- [x] **Step 1: write the failing tests** — prefix and text sinks across t2va,
  fl2va, ref2va and multi-keyframe layouts; errors for a missing `video`
  segment, more than one `video` segment, an unknown mode, and non-contiguous
  segments. Shipped as `tests/test_layout.py`.
- [x] **Step 2: run, confirm FAIL.**
- [x] **Step 3: implement.** `sink_from_segments` finds `video_start` as the `a`
  of the single `kind == "video"` segment (absent → `ValueError`). `prefix` →
  `SinkRange(0, video_start, seq_len, video_start)`. `text` → the text segment;
  its absence yields an empty sink, not an error. Unknown mode → `ValueError`.
- [x] **Step 4: run, confirm PASS.**
- [x] **Step 5: commit.**

---

### Task 3: `state.py` — clock and decline reasons

**Files:** create `state.py`, `tests/test_state.py`

**Produces:** `Policy` (dataclass with `enabled`, `tau`, `thresh_type`,
`first_dense_steps`, `first_dense_layers`, `sink_mode`, `correctness_gate`,
`strict`), `SolAttnState`, the `DECLINE_*` constants,
`resolve_step(sigmas, sample_sigmas) -> int|None`,
`dense_step_count(first_dense_steps, total_steps) -> int`.

`SolAttnState` API: `begin_run()`, `begin_forward(sink, step, total_steps)`,
`next_block()`, `decline(**kwargs) -> str|None`, `note(reason)`, `note_sparse()`,
`end_run()`, `stats() -> dict`.

- [x] **Step 1: write the failing tests.** Shipped as `tests/test_state.py`. The
  three groups map to the three documented failures in NVIDIA's reference:
  the fraction-vs-fixed dense step count, the schedule-derived step index under
  skipped forwards, and the `first_dense_layers` off-by-one. Plus the kernel
  contract, the decline table, and the aggregate end-of-run check.
- [x] **Step 2: run, confirm FAIL.**
- [x] **Step 3: implement.**
  `dense_step_count(v, total)`: `round(v * total)` for `v < 1`, else `int(v)`.
  `resolve_step`: `torch.isclose(sample_sigmas, sigmas[0], rtol=1e-4)` → the
  first match, `None` if there is none (the pattern from
  `context_windows.py:558`).
  `decline()` checks in cheapest-first order: `disabled` → `kernel_unavailable`
  → `oom` → `mask_present` → `layout_unknown` (when `skip_reshape` is not `True`)
  → `batch` (when `batch != 1`) → `dtype` → `head_dim` → `no_layout` →
  `seq_mismatch` → `warmup_step` → `dense_layer`. All arguments are keyword-only
  with defaults, so Task 4 can call with a superset.
  `note(reason)` increments `declined[reason]`; reasons outside
  `{warmup_step, dense_layer, disabled}` are logged once and raise under `strict`.
  `end_run()`: if `total_steps` exceeded `dense_step_count` and
  `sparse_calls == 0` → `RuntimeError` under `strict`, otherwise a loud warning.
- [x] **Step 4: run, confirm PASS.**
- [x] **Step 5: commit.**

---

### Task 4: `attention.py` — the override adapter

**Files:** create `attention.py`, `tests/test_attention_gpu.py`

**Consumes:** `kernel.load_sol_attn`, `kernel.probe`, `layout.SinkRange`,
`state.SolAttnState`, `state.Policy`.
**Produces:** `make_override(state, policy) -> Callable`,
`dense_bthd(q, k, v) -> Tensor`,
`run_gate(sol_attn, q, k, v, thresh_type) -> dict`, `route_density(...) -> dict`.

Call contract: `wrap_attn` (`comfy/ldm/modules/attention.py:148`) invokes
`override(func, q, k, v, heads, mask=None, skip_reshape=True,
transformer_options=..., _inside_attn_wrapper=True)`. Input is BHSD
`(1, H, S, 128)`; output is `(1, S, H*128)` when `skip_output_reshape` is falsy.

- [x] **Step 1: write the failing GPU tests** — output shape and dtype, the
  sparse path actually being taken, a decline returning exactly the dense
  result, `token_refiner` never reaching the kernel, prefix rows recomputed
  densely, and the video tail differing from dense attention. Shipped as
  `tests/test_attention_gpu.py`.
- [x] **Step 2: run, confirm FAIL.**
- [x] **Step 3: implement.** The override resolves the block index (stamp first,
  internal counter as fallback), asks `state.decline(...)`, and on a decline
  returns `func(...)` unchanged. Otherwise: BHSD → contiguous BTHD, the gate and
  density probe once each, `sol_attn(...)` with the sink, dense recomputation of
  the prefix query rows, then `reshape` back to `(B, S, heads*dim)`.
  `torch.OutOfMemoryError` latches the run onto dense.
- [x] **Step 4: run, confirm PASS.**
- [x] **Step 5: commit.**

---

### Task 5: `nodes.py` — the node and its mounting

**Files:** create `nodes.py`, `__init__.py`

**Produces:** class `SolAttnH3` with `INPUT_TYPES` / `RETURN_TYPES = ("MODEL",)` /
`FUNCTION = "patch"` / `CATEGORY = "model_patches/attention"`.

Inputs: `model` (MODEL), `enabled` (BOOLEAN, `True`), `tau` (FLOAT, `1.0`,
−1000..10, step 0.05), `thresh_type` (`["diag", "exact"]`), `first_dense_steps`
(FLOAT, `0.2`, 0..50), `first_dense_layers` (INT, `2`, 0..50), `sink_mode`
(`["prefix", "text"]`), `correctness_gate` (BOOLEAN, `True`), `strict` (BOOLEAN,
`False`).

- [x] **Step 1: mount onto a ModelPatcher clone** — the override into
  `model_options["transformer_options"]`, an `OUTER_SAMPLE` wrapper, a
  `DIFFUSION_MODEL` wrapper, and one `set_model_patch_replace(..., "dit",
  "double_block", i)` per block.
- [x] **Step 2: the forward wrapper** — reads `minimax_payload["layout"]`
  (rebuilding `PackedLayout` exactly as `model.py:520-524` does when absent),
  computes the sink via `sink_from_segments`, resolves the step index via
  `resolve_step`, and calls `state.begin_forward(...)`. A missing layout yields
  the `no_layout` decline.
- [x] **Step 3: the block-index stamp** — writes
  `args["transformer_options"]["solattn_block"]` and delegates to
  `extra["original_block"]`.
- [x] **Step 4: the run wrapper** — `OUTER_SAMPLE`: `state.begin_run()`, then
  `state.end_run()` and a `state.stats()` dump in a `finally`.
- [x] **Step 5: smoke test** — the node appears in
  `curl localhost:8188/object_info/SolAttnH3`.
- [x] **Step 6: commit.**

---

### Task 6: integration on real H3

**Files:** create `bench/ab_bench.py`

- [x] **Step 1: symlink into `custom_nodes`.**
- [x] **Step 2: a minimal H3 graph** at low resolution, a light encoder, a fixed
  seed and a small step count, in API format.
- [x] **Step 3: a run with `strict=True`** — evidence to collect from the log:
  `gate PASS` on real QKV, `sparse_calls > 0`, `declined` containing only the
  intended reasons plus `seq_mismatch` from the refiner, and an effective density
  strictly inside (0, 1).
- [x] **Step 4: A/B at the same seed** — `bench/ab_bench.py` queues the same
  graph once with `enabled=False` and once with `True`, times both through
  `/history`, and compares the decoded frames.
- [x] **Step 5: a run with `MiniMaxH3-Cache` enabled** — confirming that skipped
  steps do not break the clock.
- [x] **Step 6: commit.**

---

### Task 7: `selftest.py`, README, attribution

**Files:** create `selftest.py`, fill in `README.md`, `NOTICE`

- [x] **Step 1: `selftest.py`** — a standalone script for the target machine
  printing GPU, compute capability, CuTe availability, the selected backend, the
  gate verdict and the density at shapes given on the command line.
- [x] **Step 2: README** — installation, the parameter table with rationales, the
  measured numbers from this machine explicitly labelled "SM89/Triton, not
  SM120/CuTe", and the known limitations.
- [x] **Step 3: NOTICE** — Apache-2.0 attribution: NVlabs/Sana, the paper
  (arXiv 2607.24027), and the note that the kernel is installed rather than
  vendored.
- [x] **Step 4: the full test suite**, with and without a GPU.
- [x] **Step 5: commit.**
