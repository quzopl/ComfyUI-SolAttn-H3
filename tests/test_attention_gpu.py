"""Adapter tests on GPU. Require CUDA and an installed sol-attn."""
import pytest
import torch

from solattn_h3.attention import dense_bthd, make_override, route_density, run_gate
from solattn_h3.kernel import load_sol_attn, probe
from solattn_h3.layout import SinkRange
from solattn_h3.state import DECLINE_SEQ_MISMATCH, Policy, SolAttnState

pytestmark = pytest.mark.skipif(not probe().available,
                                reason=f"kernel unavailable: {probe().error}")

H, D, T = 8, 128, 4096
SINK = SinkRange(start=0, tokens=512, seq_len=T, video_start=512)


def _bhsd():
    """QKV in the layout H3 hands over: views into the packed qkv_proj buffer."""
    torch.manual_seed(0)
    packed = torch.randn(T, 3 * H * D, device="cuda", dtype=torch.bfloat16) * 0.5
    return [x.view(T, H, D).transpose(0, 1).unsqueeze(0) for x in packed.split(H * D, dim=-1)]


def _fake_func(q, k, v, heads, skip_output_reshape=False, **kw):
    """Stand-in for ComfyUI's original attention backend.

    Honours skip_output_reshape, because the adapter uses it to recompute the
    prefix query rows without a round trip through the flattened layout.
    """
    out = dense_bthd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    if skip_output_reshape:
        return out.transpose(1, 2)
    return out.reshape(out.shape[0], out.shape[1], heads * out.shape[3])


def _ready_state(**kw):
    policy = Policy(first_dense_steps=0, first_dense_layers=0, correctness_gate=False, **kw)
    state = SolAttnState(policy)
    state.begin_run()
    state.begin_forward(SINK, step=5, total_steps=50)
    return state


def _run(state, q, k, v, **to):
    return make_override(state, state.policy)(
        _fake_func, q, k, v, H, mask=None, skip_reshape=True,
        transformer_options={"solattn_block": 4, **to})


def test_output_shape_and_dtype():
    q, k, v = _bhsd()
    out = _run(_ready_state(), q, k, v)
    assert out.shape == (1, T, H * D)
    assert out.dtype == torch.bfloat16


def test_the_sparse_path_was_actually_used():
    state = _ready_state()
    _run(state, *_bhsd())
    assert state.stats()["sparse_calls"] == 1
    assert state.stats()["declined"] == {}


def test_a_decline_returns_exactly_the_dense_result():
    q, k, v = _bhsd()
    state = SolAttnState(Policy(enabled=False))
    state.begin_run()
    out = _run(state, q, k, v)
    torch.testing.assert_close(out, _fake_func(q, k, v, H))


def test_token_refiner_never_reaches_the_kernel():
    """A shorter sequence is a refiner call; it must fall through to dense."""
    state = _ready_state()
    packed = torch.randn(537, 3 * H * D, device="cuda", dtype=torch.bfloat16)
    q, k, v = (x.view(537, H, D).transpose(0, 1).unsqueeze(0)
               for x in packed.split(H * D, dim=-1))
    _run(state, q, k, v)
    assert state.stats()["declined"] == {DECLINE_SEQ_MISMATCH: 1}


def test_prefix_rows_are_recomputed_densely():
    """The sink makes the prefix exact as K/V, but its queries must be dense."""
    q, k, v = _bhsd()
    out = _run(_ready_state(), q, k, v).reshape(1, T, H, D)
    want = dense_bthd(q.transpose(1, 2)[:, :SINK.tokens], k.transpose(1, 2), v.transpose(1, 2))
    torch.testing.assert_close(out[:, :SINK.tokens], want, rtol=2e-2, atol=2e-2)


def test_the_video_tail_differs_from_dense_attention():
    """If the router were not routing, the result would be identical to dense —
    which would mean the configuration computes dense under a sparse label."""
    q, k, v = _bhsd()
    out = _run(_ready_state(), q, k, v).reshape(1, T, H, D)
    want = dense_bthd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    tail_diff = (out[:, SINK.tokens:].float() - want[:, SINK.tokens:].float()).abs().max()
    assert tail_diff > 0, "video tail identical to dense: the router is not routing"


def test_gate_passes_on_a_real_shape():
    q, k, v = _bhsd()
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    stats = run_gate(load_sol_attn(), qb, kb, vb, "diag")
    assert stats["passed"], stats


def test_density_is_within_band():
    q, k, v = _bhsd()
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    d = route_density(qb, kb, vb, tau=1.0, thresh_type="diag", sink=SINK)
    assert 0.0 < d["effective_density"] < 1.0, d
    assert d["sink_blocks"] == SINK.tokens // 64


def test_fallback_block_counter_when_no_stamp():
    """Without the patches_replace stamp the block index comes from the counter."""
    state = SolAttnState(Policy(first_dense_steps=0, first_dense_layers=2,
                                correctness_gate=False))
    state.begin_run()
    state.begin_forward(SINK, step=5, total_steps=50)
    q, k, v = _bhsd()
    override = make_override(state, state.policy)
    for _ in range(3):
        override(_fake_func, q, k, v, H, mask=None, skip_reshape=True,
                 transformer_options={})
    assert state.stats()["declined"] == {"dense_layer": 2}
    assert state.stats()["sparse_calls"] == 1


# --- the gate criterion -----------------------------------------------------

def test_gate_rejects_a_broken_kernel():
    """After swapping the absolute threshold for a relative one, the gate must
    still catch a real error.

    Broken routing or indexing produces relative errors of a percent and up; 5%
    slips past `mean_abs` but not past `max_rel`.
    """
    q, k, v = _bhsd()
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))

    def broken(a, b, c, **kw):
        return dense_bthd(a, b, c) * 1.05

    stats = run_gate(broken, qb, kb, vb, "diag")
    assert not stats["passed"], stats
    assert stats["max_rel"] > stats["limits"]["max_rel"]


def test_gate_accepts_a_perfect_kernel():
    q, k, v = _bhsd()
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    stats = run_gate(lambda a, b, c, **kw: dense_bthd(a, b, c), qb, kb, vb, "diag")
    assert stats["passed"] and stats["max_rel"] == 0.0


def test_gate_is_independent_of_output_scale():
    """The point of the change: the same relative perturbation must get the same
    verdict regardless of how large the activations are. An absolute threshold
    did not have that property."""
    q, k, v = _bhsd()
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    verdicts = []
    for scale in (1.0, 50.0):
        big = [x * scale for x in (qb, kb, vb)]
        stats = run_gate(lambda a, b, c, **kw: dense_bthd(a, b, c) * 1.05, *big, "diag")
        verdicts.append(stats["passed"])
    assert verdicts == [False, False], verdicts



def test_prefix_recompute_goes_through_the_model_backend():
    """The prefix rows must be recomputed with `func`, not with SDPA.

    That is what makes the recomputation run on whatever fast attention ComfyUI
    is configured with. The probe also pins the call shape: queries sliced to the
    sink, keys and values full length, BHSD in and BHSD out.
    """
    q, k, v = _bhsd()
    seen = []

    def probe(qq, kk, vv, heads, skip_output_reshape=False, **kw):
        seen.append((tuple(qq.shape), tuple(kk.shape), skip_output_reshape))
        return _fake_func(qq, kk, vv, heads,
                          skip_output_reshape=skip_output_reshape, **kw)

    state = _ready_state()
    make_override(state, state.policy)(
        probe, q, k, v, H, mask=None, skip_reshape=True,
        transformer_options={"solattn_block": 4})

    assert seen == [((1, H, SINK.tokens, D), (1, H, T, D), True)], seen
