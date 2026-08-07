"""The SolAttnH3 node — UI and mounting onto the ModelPatcher.

Four things are mounted, all on public ComfyUI APIs:

  1. `transformer_options["optimized_attention_override"]` — intercepts attention
  2. a `WrappersMP.DIFFUSION_MODEL` wrapper — per forward: layout, sink, step
  3. `patches_replace["dit"][("double_block", i)]` — stamps the block index
  4. a `WrappersMP.OUTER_SAMPLE` wrapper — run boundaries and the aggregate check

No file under `comfy/` is modified.
"""
from __future__ import annotations

from comfy.patcher_extension import WrappersMP

from .attention import make_override
from .kernel import probe
from .layout import SINK_MODES, sink_from_segments
from .state import LOG, Policy, SolAttnState, resolve_step

KEY = "solattn_h3"
STAMP = "solattn_block"


class SolAttnH3:
    """Training-free sparse attention (Sol-Attn) for native MiniMax-H3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True,
                            "tooltip": "Turns the node off without rewiring the graph."}),
                "tau": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 10.0, "step": 0.05,
                                  "tooltip": "Higher values select fewer K/V blocks for exact "
                                             "attention. 1.0 is the validated H3 policy."}),
                "thresh_type": (["diag", "exact"], {"default": "diag",
                                "tooltip": "exact uses the full-covariance threshold: more "
                                           "accurate, more expensive."}),
                "first_dense_steps": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 50.0, "step": 0.05,
                                      "tooltip": "Below 1 this is a fraction of the schedule; "
                                                 "1 and above is a fixed step count. 0.2 matches "
                                                 "the reference's 10 steps out of 50."}),
                "first_dense_layers": ("INT", {"default": 2, "min": 0, "max": 50,
                                       "tooltip": "The first N DiT blocks stay dense. Counted "
                                                  "from zero."}),
                "sink_mode": (list(SINK_MODES), {"default": "prefix",
                              "tooltip": "prefix keeps text, conditioning and audio rows exact. "
                                         "text reproduces the reference policy."}),
                "correctness_gate": ("BOOLEAN", {"default": True,
                                     "tooltip": "Once per shape, compares the kernel against SDPA "
                                                "on real QKV. A failure aborts generation."}),
                "strict": ("BOOLEAN", {"default": False,
                           "tooltip": "Turns every unintended decline of the sparse path into an "
                                      "exception. For validation, not for daily use."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = ("Sol-Attn (NVIDIA Sol Engine) for MiniMax-H3. Training-free sparse "
                   "attention; falls back to dense attention with a named reason whenever "
                   "the kernel contract is not met.")

    def patch(self, model, enabled, tau, thresh_type, first_dense_steps,
              first_dense_layers, sink_mode, correctness_gate, strict):
        blocks = _block_count(model)

        policy = Policy(enabled=enabled, tau=tau, thresh_type=thresh_type,
                        first_dense_steps=first_dense_steps,
                        first_dense_layers=first_dense_layers, sink_mode=sink_mode,
                        correctness_gate=correctness_gate, strict=strict)
        state = SolAttnState(policy)

        found = probe()
        state.backend = found.backend
        if not found.available:
            state.kernel_error = found.error
            print(f"{LOG} kernel {found.describe()} — the model will run dense attention",
                  flush=True)
        else:
            print(f"{LOG} {found.describe()}, DiT blocks: {blocks}", flush=True)

        patched = model.clone()
        options = patched.model_options.setdefault("transformer_options", {})
        options["optimized_attention_override"] = make_override(state, policy)
        patched.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, KEY, _make_run_wrapper(state))
        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, KEY,
                                     _make_forward_wrapper(state, policy))
        for index in range(blocks):
            patched.set_model_patch_replace(_make_stamp(index), "dit", "double_block", index)
        return (patched,)


def _block_count(model) -> int:
    """Number of DiT blocks; doubles as the check that this really is MiniMax-H3."""
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    blocks = getattr(diffusion, "blocks", None)
    if blocks is None or type(diffusion).__name__ != "MiniMaxH3Model":
        raise ValueError(
            f"{LOG} SolAttnH3 only works with MiniMax-H3; got "
            f"{type(diffusion).__name__}. The node was not mounted."
        )
    return len(blocks)


def _make_stamp(index: int):
    """Stamp the real block index, then hand control to the original block.

    Counting attention calls instead of stamping would shift first_dense_layers,
    because token_refiner (model.py:584) also calls Attention with head_dim 128.
    """
    def stamp(args, extra):
        args["transformer_options"][STAMP] = index
        return extra["original_block"](args)
    return stamp


def _make_run_wrapper(state):
    """Boundaries of one sampler run: counter reset and the aggregate check."""
    def wrapper(executor, *args, **kwargs):
        state.begin_run()
        try:
            return executor(*args, **kwargs)
        finally:
            state.end_run()
            if state.sparse_calls or state.dense_calls:
                print(f"{LOG} {state.stats()}", flush=True)
    return wrapper


def _make_forward_wrapper(state, policy):
    """Once per forward: layout, sink range and the step index from the schedule."""
    def wrapper(executor, x, timestep, context, transformer_options, **kwargs):
        sink = None
        try:
            layout = _resolve_layout(executor.class_obj, kwargs.get("minimax_payload"),
                                     x, context)
            if layout is not None:
                sink = sink_from_segments(layout.segments, layout.seq_len, policy.sink_mode)
        except Exception as exc:
            _warn_once(state, f"could not determine the sink range: "
                              f"{type(exc).__name__}: {exc}")

        sample_sigmas = transformer_options.get("sample_sigmas")
        step = resolve_step(transformer_options.get("sigmas"), sample_sigmas)
        total = int(sample_sigmas.shape[0]) - 1 if sample_sigmas is not None else None
        state.begin_forward(sink, step, total)
        return executor(x, timestep, context, transformer_options, **kwargs)
    return wrapper


def _resolve_layout(model, payload, x, context):
    """The packed-sequence layout: from the payload, or rebuilt when absent.

    `extra_conds` builds it once per run, but only when it knows latent_shapes.
    The rebuild mirrors `model.py:506-524`. A stale layout cannot produce a wrong
    result: the `rows != sink.seq_len` check in `decline()` catches it and the
    call falls through to dense attention.
    """
    payload = payload or {}
    layout = payload.get("layout")
    if layout is not None:
        return layout

    import comfy.ldm.common_dit
    from comfy.ldm.minimax.model import PackedLayout

    video_x = comfy.ldm.common_dit.pad_to_patch_size(x[0], model.patch_size)
    return PackedLayout(context.shape[1], video_x.shape[2], video_x.shape[3], video_x.shape[4],
                        x[1].shape[-1], keyframes=payload.get("keyframes"),
                        refs=payload.get("refs"), frame_count=payload.get("frame_count"))


def _warn_once(state, message: str) -> None:
    if message not in state._logged:
        state._logged.add(message)
        print(f"{LOG} {message}", flush=True)


NODE_CLASS_MAPPINGS = {"SolAttnH3": SolAttnH3}
NODE_DISPLAY_NAME_MAPPINGS = {"SolAttnH3": "Sol-Attn (MiniMax-H3)"}
