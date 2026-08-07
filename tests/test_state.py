"""Tests for the step clock, the block counter and the decline reasons.

Each of the three blocks below corresponds to a documented failure in NVIDIA's
reference integration (`models/minimax_h3/optimized/sol_attn_h3.py`).
"""
import pytest
import torch

from solattn_h3.layout import SinkRange
from solattn_h3.state import (DECLINE_BATCH, DECLINE_DENSE_LAYER, DECLINE_DISABLED,
                              DECLINE_DTYPE, DECLINE_HEAD_DIM, DECLINE_LAYOUT_UNKNOWN,
                              DECLINE_MASK, DECLINE_NO_LAYOUT, DECLINE_OOM,
                              DECLINE_SEQ_MISMATCH, DECLINE_WARMUP, Policy, SolAttnState,
                              dense_step_count, resolve_step)

SINK = SinkRange(start=0, tokens=951, seq_len=31000, video_start=951)


def _state(**kw):
    s = SolAttnState(Policy(first_dense_steps=0.2, first_dense_layers=2, **kw))
    s.begin_run()
    return s


def _call(s, *, rows=31000, dtype=torch.bfloat16, head_dim=128, mask=None,
          block=5, skip_reshape=True, batch=1):
    return s.decline(rows=rows, dtype=dtype, head_dim=head_dim, mask=mask,
                     block_index=block, skip_reshape=skip_reshape, batch=batch)


# --- dense-step fraction ----------------------------------------------------

def test_fraction_scales_with_schedule_length():
    """The reference's 10 comes from a 50-step schedule."""
    assert dense_step_count(0.2, 50) == 10
    assert dense_step_count(0.2, 20) == 4


def test_value_of_one_or_more_is_a_fixed_step_count():
    assert dense_step_count(10, 50) == 10
    assert dense_step_count(10, 20) == 10


def test_zero_means_no_dense_steps():
    assert dense_step_count(0, 50) == 0


# --- step index from the schedule -------------------------------------------

def test_step_index_from_schedule():
    sample = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    assert resolve_step(torch.tensor([0.5]), sample) == 2
    assert resolve_step(torch.tensor([1.0]), sample) == 0
    assert resolve_step(torch.tensor([0.0]), sample) == 4


def test_step_index_when_sigma_is_off_schedule():
    assert resolve_step(torch.tensor([0.31]), torch.tensor([1.0, 0.5, 0.0])) is None


def test_step_index_without_a_schedule():
    assert resolve_step(torch.tensor([0.5]), None) is None


def test_skipped_steps_do_not_break_the_clock():
    """A cache skips forwards; the step index comes from the schedule, not a counter.

    The reference counted forwards, so with a warm-up plus a measured request the
    measured run started at step 49 while still reporting ten dense steps.
    """
    s = _state()
    for step in (10, 13, 14, 19):
        s.begin_forward(SINK, step=step, total_steps=50)
        assert _call(s) is None
    assert s.stats()["last_step"] == 19
    assert s.stats()["sparse_calls"] == 0  # decline() does not count; note_sparse() does


# --- block counter ----------------------------------------------------------

def test_dense_layers_are_counted_from_zero():
    """The reference had an off-by-one here: dense_layers=2 left only block 0 dense."""
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s, block=0) == DECLINE_DENSE_LAYER
    assert _call(s, block=1) == DECLINE_DENSE_LAYER
    assert _call(s, block=2) is None


def test_block_counter_resets_every_forward():
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    assert [s.next_block() for _ in range(3)] == [0, 1, 2]
    s.begin_forward(SINK, step=11, total_steps=50)
    assert s.next_block() == 0


# --- decline reasons --------------------------------------------------------

def test_warmup_steps_decline_and_the_rest_do_not():
    s = _state()
    s.begin_forward(SINK, step=3, total_steps=50)      # 3 < 10
    assert _call(s) == DECLINE_WARMUP
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s) is None


def test_kernel_contract():
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s, dtype=torch.float16) == DECLINE_DTYPE
    assert _call(s, head_dim=64) == DECLINE_HEAD_DIM
    assert _call(s, mask=object()) == DECLINE_MASK
    assert _call(s, batch=2) == DECLINE_BATCH
    assert _call(s, skip_reshape=False) == DECLINE_LAYOUT_UNKNOWN


def test_token_refiner_falls_out_on_sequence_length():
    """model.py:584 calls Attention with head_dim 128 on the text rows alone."""
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s, rows=537) == DECLINE_SEQ_MISMATCH


def test_missing_layout():
    s = _state()
    assert _call(s) == DECLINE_NO_LAYOUT       # no begin_forward


def test_disabled_node():
    s = _state(enabled=False)
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s) == DECLINE_DISABLED


def test_oom_latches_onto_dense():
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s) is None
    s.latch_oom()
    assert _call(s) == DECLINE_OOM


def test_unavailable_kernel():
    s = _state()
    s.kernel_error = "sol_attn missing"
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s) == "kernel_unavailable"


# --- aggregate check --------------------------------------------------------

def test_a_run_without_a_single_sparse_call_is_an_error():
    """The reason that does the damage (warmup_step) is itself legitimate.

    The error exists only in aggregate, so the check has to be in aggregate too.
    """
    s = SolAttnState(Policy(strict=True))
    s.begin_run()
    for step in range(50):
        s.begin_forward(SINK, step=step, total_steps=50)
        s.note(DECLINE_WARMUP)
    with pytest.raises(RuntimeError, match="not a single sparse call"):
        s.end_run()


def test_a_run_with_sparse_calls_passes():
    s = SolAttnState(Policy(strict=True))
    s.begin_run()
    s.begin_forward(SINK, step=20, total_steps=50)
    s.note_sparse()
    s.end_run()


def test_a_run_shorter_than_warmup_does_not_alarm():
    """5 steps against 10 dense ones is a short run, not an error."""
    s = SolAttnState(Policy(strict=True, first_dense_steps=0.2))
    s.begin_run()
    for step in range(5):
        s.begin_forward(SINK, step=step, total_steps=50)
        s.note(DECLINE_WARMUP)
    s.end_run()


def test_strict_raises_on_an_unintended_reason():
    s = SolAttnState(Policy(strict=True))
    s.begin_run()
    with pytest.raises(RuntimeError, match=DECLINE_SEQ_MISMATCH):
        s.note(DECLINE_SEQ_MISMATCH)


def test_intended_reasons_do_not_raise_under_strict():
    s = SolAttnState(Policy(strict=True))
    s.begin_run()
    for reason in (DECLINE_WARMUP, DECLINE_DENSE_LAYER, DECLINE_DISABLED):
        s.note(reason)
    assert s.stats()["declined"] == {DECLINE_WARMUP: 1, DECLINE_DENSE_LAYER: 1,
                                     DECLINE_DISABLED: 1}


def test_begin_run_resets_counters_between_runs():
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    s.note_sparse()
    s.note(DECLINE_WARMUP)
    s.begin_run()
    assert s.stats()["sparse_calls"] == 0
    assert s.stats()["declined"] == {}


def test_disabled_node_does_not_alarm_at_end_of_run():
    """The benchmark's `off` variant has zero sparse calls by definition."""
    s = SolAttnState(Policy(strict=True, enabled=False))
    s.begin_run()
    for step in range(8):
        s.begin_forward(SINK, step=step, total_steps=8)
        s.note(DECLINE_DISABLED)
    s.end_run()


# --- gate, once per shape ---------------------------------------------------

def test_gate_fires_once_per_shape():
    s = _state()
    assert s.should_gate((31000, 56, 128)) is True
    s.record_gate((31000, 56, 128), {"passed": True})
    assert s.should_gate((31000, 56, 128)) is False
    assert s.should_gate((16000, 56, 128)) is True


def test_gate_disabled_by_policy():
    s = _state(correctness_gate=False)
    assert s.should_gate((31000, 56, 128)) is False
