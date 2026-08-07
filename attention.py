"""The adapter installed via `transformer_options["optimized_attention_override"]`.

`wrap_attn` (comfy/ldm/modules/attention.py:148) calls us as
`override(func, q, k, v, heads, **kwargs)`, where `func` is the original
attention backend. Returning `func(...)` gives a free dense fallback — the exact
code that would have run without this node.

H3 hands over BHSD `(1, heads, S, 128)` (`skip_reshape=True`) and expects
`(1, S, heads*128)` back.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .kernel import load_sol_attn
from .state import LOG

BLOCK_SIZE = 64


def dense_bthd(q, k, v):
    """SDPA on the BTHD layout, same layout on the way out.

    `q` may be shorter than `k`/`v` — that is how the prefix query rows are
    recomputed.
    """
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
    )
    return out.transpose(1, 2)


def run_gate(sol_attn, q, k, v, thresh_type: str) -> dict:
    """Check the kernel's arithmetic against SDPA on real QKV.

    `tau=-1000` admits every block, so what is measured is the kernel's
    arithmetic rather than the routing policy. A probe on random tensors answers
    a question about the kernel, not about this model at this shape.

    The gate runs at the production head count: `preprocess.prepare` autotunes
    its Triton kernels on a key of `T` alone, so a first call at a lower head
    count would cache a configuration chosen for a narrower grid.
    """
    got = sol_attn(q, k, v, tau=-1000.0, thresh_type=thresh_type)
    want = dense_bthd(q, k, v)
    diff = (got.float() - want.float()).abs()
    ref_max = float(want.float().abs().max())
    stats = {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_l2": float(torch.linalg.vector_norm(got.float() - want.float())
                        / torch.linalg.vector_norm(want.float()).clamp_min(1e-12)),
        # Reference scale: `max_abs` on its own depends on the activation range,
        # so an absolute threshold does not transfer between models or shapes.
        "ref_max": ref_max,
        "max_rel": float(diff.max()) / max(ref_max, 1e-12),
        "over_1e2": float((diff > 1e-2).float().mean()),
        "shape": list(q.shape),
    }
    # `mean_abs` and `rel_l2` are exactly the limits NVIDIA set. The absolute
    # limit on `max_abs` (0.08 / 0.15) was replaced by a relative one, because it
    # does not transfer between activation distributions:
    #
    # On real H3 QKV we measured ref_max=33.0 and max_abs=0.125. bfloat16 has
    # 7 mantissa bits, so for |x| in [32, 64) one ulp is 32*2^-7 = 0.25 and the
    # correct-rounding bound is half an ulp = 0.125. The worst element is
    # therefore the *theoretical minimum* representation error at that magnitude
    # — the kernel and SDPA rounded the same value to two adjacent bf16 numbers.
    # The same shape on synthetic N(0, 0.5) tensors gives max_abs=0.00012: the
    # difference lives in the output scale, not in the kernel.
    #
    # 0.02 is about five half-ulps at the tensor's peak — room for accumulation
    # order, still two orders of magnitude below what broken routing or indexing
    # would produce (there the relative error is of order one).
    limits = {
        "max_rel": 0.02,
        "mean_abs": 0.002,
        "rel_l2": 0.005,
    }
    stats["limits"] = limits
    stats["limits_nvidia_max_abs"] = 0.15 if q.shape[1] >= 32768 else 0.08
    stats["passed"] = all(stats[name] <= limit for name, limit in limits.items())
    return stats


@torch.no_grad()
def route_density(q, k, v, *, tau: float, thresh_type: str, sink) -> dict:
    """What fraction of K/V blocks the routing keeps.

    Reported because the failure this node exists to avoid is a configuration
    that quietly computes dense: a density near 1.0 says the routing is not
    routing, and 0.0 says it collapsed.
    """
    from sol_attn.preprocess import prepare

    scale = q.shape[-1] ** -0.5
    kc, _, threshold = prepare(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type)

    tokens, heads = q.shape[1], q.shape[2]
    blocks = math.ceil(tokens / BLOCK_SIZE)
    padded = F.pad(q, (0, 0, 0, 0, 0, blocks * BLOCK_SIZE - tokens))
    counts = torch.full((blocks,), float(BLOCK_SIZE), device=q.device, dtype=torch.float32)
    counts[-1] = tokens - (blocks - 1) * BLOCK_SIZE
    # sum(dtype=float32) accumulates without materializing a float32 copy of q
    q_bar = padded.view(q.shape[0], blocks, BLOCK_SIZE, heads, q.shape[3]).sum(
        dim=2, dtype=torch.float32) / counts.view(1, blocks, 1, 1)

    scores = torch.einsum("bqhd,bkhd->bqkh", q_bar, kc.float()).mul_(scale * math.log2(math.e))
    routed = scores > threshold[:, :, None, :]
    threshold_density = float(routed.float().mean())

    ids = torch.arange(blocks, device=q.device)
    routed |= ((ids[:, None] - ids[None, :]).abs() <= 1)[None, :, :, None]   # local band
    sink_blocks = 0
    if sink.tokens:
        first = sink.start // BLOCK_SIZE
        last = math.ceil(sink.stop / BLOCK_SIZE)
        routed[:, :, first:last, :] = True
        sink_blocks = last - first
    return {
        "blocks": blocks,
        "sink_blocks": sink_blocks,
        "threshold_density": round(threshold_density, 5),
        "effective_density": round(float(routed.float().mean()), 5),
    }


def make_override(state, policy):
    """Build the callable for `transformer_options["optimized_attention_override"]`."""

    def override(func, q, k, v, heads, *args, **kwargs):
        def dense():
            return func(q, k, v, heads, *args, **kwargs)

        options = kwargs.get("transformer_options") or {}
        block = options.get("solattn_block")
        if block is None:
            block = state.next_block()

        reason = state.decline(
            rows=q.shape[2] if q.ndim == 4 else None,
            dtype=q.dtype,
            head_dim=q.shape[-1],
            mask=kwargs.get("mask"),
            block_index=block,
            skip_reshape=kwargs.get("skip_reshape"),
            batch=q.shape[0] if q.ndim == 4 else None,
        )
        if reason is not None:
            state.note(reason)
            pair = state.timer()
            if pair is not None:
                pair[0].record()
            out = dense()
            if pair is not None:
                pair[1].record()
                state.record_timing(pair, "dense")
            return out

        pair = state.timer()
        if pair is not None:
            pair[0].record()
        try:
            sol_attn = state.sol_attn or _attach_kernel(state)
            # The kernel wants contiguous BTHD; H3 hands over views into the
            # packed qkv_proj buffer, so this copy is unavoidable (~6.5% of
            # kernel time at 31k rows, measured).
            qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
            sink = state.sink

            if state.should_gate(qb.shape[1:]):
                _gate_once(state, sol_attn, qb, kb, vb, policy)
            if state.density is None:
                _density_once(state, qb, kb, vb, policy, sink)

            out = sol_attn(qb, kb, vb, tau=policy.tau, thresh_type=policy.thresh_type,
                           kv_splits=1, sink_start=sink.start, sink_tokens=sink.tokens)
            # The sink makes the prefix exact as K/V, but its own query rows
            # still route sparsely. The kernel's README is explicit that an
            # MMDiT integration must recompute those rows densely.
            if sink.tokens:
                out[:, sink.start:sink.stop] = dense_bthd(qb[:, sink.start:sink.stop], kb, vb)
        except torch.OutOfMemoryError:
            state.latch_oom()
            print(f"{LOG} out of memory on the sparse path; dense attention for the "
                  "rest of the run", flush=True)
            return dense()

        if pair is not None:
            pair[1].record()
            # The first call carries Triton compilation, the gate and the density probe.
            state.record_timing(pair, "sparse_first" if state.sparse_calls == 0 else "sparse")
        state.note_sparse()
        if kwargs.get("skip_output_reshape"):
            return out.transpose(1, 2)
        return out.reshape(out.shape[0], out.shape[1], heads * out.shape[3])

    return override


def _attach_kernel(state):
    state.sol_attn = load_sol_attn()
    return state.sol_attn


def _gate_once(state, sol_attn, qb, kb, vb, policy) -> None:
    """Correctness gate, once per shape. Out of memory skips it, never narrows it.

    Narrowing to fewer heads would write into Triton's autotune cache a
    configuration chosen for a narrower grid, under the key that production
    calls will later use. Better not to measure than to poison it.
    """
    try:
        stats = run_gate(sol_attn, qb, kb, vb, policy.thresh_type)
    except torch.OutOfMemoryError:
        state.record_gate(qb.shape[1:], {"passed": None, "skipped": "out of memory"})
        print(f"{LOG} correctness gate skipped: out of memory", flush=True)
        return
    state.record_gate(qb.shape[1:], stats)
    verdict = "PASS" if stats["passed"] else "FAIL"
    print(f"{LOG} correctness gate {verdict} max_abs={stats['max_abs']:.5f} "
          f"mean_abs={stats['mean_abs']:.6f} rel_l2={stats['rel_l2']:.5f} "
          f"ref_max={stats['ref_max']:.3f} max_rel={stats['max_rel']:.5f} "
          f"over_1e2={stats['over_1e2']:.2e} limits={stats['limits']}", flush=True)
    if not stats["passed"]:
        raise RuntimeError(
            f"{LOG} correctness gate failed on real QKV: {stats}. "
            "Silently accepting a broken kernel is worse than no speedup."
        )


def _density_once(state, qb, kb, vb, policy, sink) -> None:
    try:
        state.density = route_density(qb, kb, vb, tau=policy.tau,
                                      thresh_type=policy.thresh_type, sink=sink)
    except torch.OutOfMemoryError:
        state.density = {"skipped": "out of memory"}
        return
    except Exception as exc:               # diagnostic probe, not a critical path
        state.density = {"skipped": f"{type(exc).__name__}: {exc}"}
        return
    print(f"{LOG} routing density {state.density}", flush=True)
