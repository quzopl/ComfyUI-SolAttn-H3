"""Step clock, block counter, decline reasons and the per-run aggregate check.

The principle behind this whole module: a configuration that asked for sparse
attention and quietly got dense attention is a wrong measurement wearing the
right label. So every decline carries a reason (a string), never a boolean, and
every decline is counted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .layout import SinkRange

# Intended reasons — silent, counted only.
DECLINE_DISABLED = "disabled"
DECLINE_WARMUP = "warmup_step"
DECLINE_DENSE_LAYER = "dense_layer"
# Unintended reasons — logged once, raised under strict.
DECLINE_KERNEL = "kernel_unavailable"
DECLINE_OOM = "oom"
DECLINE_MASK = "mask_present"
DECLINE_LAYOUT_UNKNOWN = "layout_unknown"
DECLINE_BATCH = "batch"
DECLINE_DTYPE = "dtype"
DECLINE_HEAD_DIM = "head_dim"
DECLINE_NO_LAYOUT = "no_layout"
DECLINE_SEQ_MISMATCH = "seq_mismatch"

INTENTIONAL = frozenset({DECLINE_DISABLED, DECLINE_WARMUP, DECLINE_DENSE_LAYER})

LOG = "[sol-attn-h3]"


@dataclass
class Policy:
    """Sparse-attention policy parameters.

    Defaults come from the validated H3 line in Sol-Engine
    (`models/minimax_h3/optimized/sol_attn_h3.py`) and should not be changed
    without measuring.
    """

    enabled: bool = True
    tau: float = 1.0
    thresh_type: str = "diag"
    first_dense_steps: float = 0.2
    first_dense_layers: int = 2
    sink_mode: str = "prefix"
    correctness_gate: bool = True
    strict: bool = False


def dense_step_count(first_dense_steps: float, total_steps: int | None) -> int:
    """How many leading steps run dense.

    A value below 1 is a fraction of the schedule; 1 and above is a fixed step
    count. The reference's 10 comes from a 50-step schedule, which at 20 steps
    would mean running half the schedule dense.
    """
    if first_dense_steps >= 1:
        return int(first_dense_steps)
    if not total_steps:
        return 0
    return int(round(first_dense_steps * total_steps))


def resolve_step(sigmas, sample_sigmas) -> int | None:
    """The current step index, read off the sampler's own schedule.

    Pattern taken from `comfy/context_windows.py:558-560`. NVIDIA's reference had
    to guess where a request started from the direction of timestep change, and
    got it wrong twice; ComfyUI hands over the whole schedule, so there is
    nothing to guess.
    """
    if sigmas is None or sample_sigmas is None:
        return None
    import torch

    current = sigmas.flatten()[0].to(sample_sigmas.dtype)
    hits = torch.isclose(sample_sigmas, current, rtol=1e-4).nonzero()
    if hits.numel() == 0:
        return None
    return int(hits[0].item())


@dataclass
class SolAttnState:
    """State of one mounted node, shared across the wrappers."""

    policy: Policy
    kernel_error: str | None = None

    sink: SinkRange | None = field(default=None, init=False)
    step: int | None = field(default=None, init=False)
    total_steps: int | None = field(default=None, init=False)

    sparse_calls: int = field(default=0, init=False)
    dense_calls: int = field(default=0, init=False)
    declined: dict = field(default_factory=dict, init=False)
    gate_stats: dict | None = field(default=None, init=False)
    density: dict | None = field(default=None, init=False)
    backend: str | None = field(default=None, init=False)

    sol_attn: object | None = field(default=None, init=False)
    attn_ms: dict = field(
        default_factory=lambda: {"sparse": 0.0, "dense": 0.0, "sparse_first": 0.0},
        init=False)

    _events: list = field(default_factory=list, init=False)

    _block: int = field(default=0, init=False)
    _max_step: int = field(default=-1, init=False)
    _oom: bool = field(default=False, init=False)
    _logged: set = field(default_factory=set, init=False)
    _gated: set = field(default_factory=set, init=False)

    # -- lifecycle ------------------------------------------------------------

    def begin_run(self) -> None:
        """Start of one sampler run."""
        self.sparse_calls = 0
        self.dense_calls = 0
        self.declined = {}
        self.step = None
        self.total_steps = None
        self._max_step = -1
        self._block = 0
        self._oom = False
        self._logged = set()
        self.attn_ms = {"sparse": 0.0, "dense": 0.0, "sparse_first": 0.0}
        self._events = []

    def begin_forward(self, sink: SinkRange | None, step: int | None,
                      total_steps: int | None) -> None:
        """Start of one model forward."""
        self.sink = sink
        self.step = step
        self.total_steps = total_steps
        self._block = 0
        if step is not None:
            self._max_step = max(self._max_step, step)

    def next_block(self) -> int:
        """Fallback block counter, used when the patches_replace stamp is absent."""
        index = self._block
        self._block += 1
        return index

    def latch_oom(self) -> None:
        self._oom = True

    # -- decision -------------------------------------------------------------

    def decline(self, *, rows: int | None = None, dtype=None, head_dim: int | None = None,
                mask=None, block_index: int | None = None, skip_reshape=None,
                batch: int | None = None) -> str | None:
        """Why this call cannot take the sparse path, or None if it can.

        Ordered from the cheapest check to the most expensive.
        """
        import torch

        if not self.policy.enabled:
            return DECLINE_DISABLED
        if self.kernel_error is not None:
            return DECLINE_KERNEL
        if self._oom:
            return DECLINE_OOM
        if mask is not None:
            return DECLINE_MASK
        if skip_reshape is not None and not skip_reshape:
            return DECLINE_LAYOUT_UNKNOWN
        if batch is not None and batch != 1:
            return DECLINE_BATCH
        if dtype is not None and dtype != torch.bfloat16:
            return DECLINE_DTYPE
        if head_dim is not None and head_dim != 128:
            return DECLINE_HEAD_DIM
        if self.sink is None:
            return DECLINE_NO_LAYOUT
        if rows is not None and rows != self.sink.seq_len:
            return DECLINE_SEQ_MISMATCH
        if self.step is None:
            return DECLINE_NO_LAYOUT
        if self.step < dense_step_count(self.policy.first_dense_steps, self.total_steps):
            return DECLINE_WARMUP
        if block_index is not None and block_index < self.policy.first_dense_layers:
            return DECLINE_DENSE_LAYER
        return None

    # -- bookkeeping ----------------------------------------------------------

    def note(self, reason: str) -> None:
        self.declined[reason] = self.declined.get(reason, 0) + 1
        self.dense_calls += 1
        if reason in INTENTIONAL:
            return
        if self.policy.strict:
            raise RuntimeError(f"{LOG} sparse path declined: {reason}")
        if reason not in self._logged:
            self._logged.add(reason)
            print(f"{LOG} running dense, reason: {reason}", flush=True)

    def note_sparse(self) -> None:
        self.sparse_calls += 1

    # -- attention timing -----------------------------------------------------

    def timer(self):
        """A pair of CUDA events for timing one attention call.

        The events are recorded asynchronously; synchronization happens once, in
        `flush_timing()`, so the measurement never serializes the stream.
        """
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def record_timing(self, pair, kind: str) -> None:
        if pair is not None:
            self._events.append((pair, kind))

    def flush_timing(self) -> None:
        """Sum the recorded times. One synchronization for the whole run."""
        if not self._events:
            return
        import torch

        torch.cuda.synchronize()
        for (start, end), kind in self._events:
            try:
                self.attn_ms[kind] += start.elapsed_time(end)
            except RuntimeError:      # the event never made it onto the stream
                continue
        self._events = []

    def end_run(self) -> None:
        """Aggregate check: a run that cleared warm-up without touching the kernel.

        Per-call reasons cannot catch this, because the reason that does the
        damage (warmup_step) is itself legitimate. The error exists only in
        aggregate.
        """
        self.flush_timing()
        # A disabled node is not a measurement failure — the benchmark's `off`
        # variant has zero sparse calls by definition.
        if not self.policy.enabled or self._max_step < 0:
            return
        warmup = dense_step_count(self.policy.first_dense_steps, self.total_steps)
        if self._max_step + 1 <= warmup or self.sparse_calls > 0:
            return
        message = (f"{LOG} run passed {self._max_step + 1} steps with "
                   f"first_dense_steps={warmup} and made not a single sparse call; "
                   f"declines={self.declined}")
        if self.policy.strict:
            raise RuntimeError(message)
        print(f"{LOG} WARNING: {message}", flush=True)

    # -- gate, once per shape -------------------------------------------------

    def should_gate(self, shape) -> bool:
        return self.policy.correctness_gate and tuple(shape) not in self._gated

    def record_gate(self, shape, stats: dict) -> None:
        self._gated.add(tuple(shape))
        self.gate_stats = stats

    # -- reporting ------------------------------------------------------------

    def _per_call(self) -> dict:
        """Per-call time, with the first sparse call excluded.

        That first call carries Triton kernel compilation, the correctness gate
        and the density probe — one-off costs that on a short run swamp the
        actual measurement (across 288 calls, a 4 s compile is 14 ms per call,
        i.e. more than the kernel itself).
        """
        steady = max(self.sparse_calls - 1, 0)
        return {
            "sparse": round(self.attn_ms["sparse"] / steady, 2) if steady else None,
            "dense": round(self.attn_ms["dense"] / self.dense_calls, 2)
                     if self.dense_calls else None,
            "sparse_first_ms": round(self.attn_ms["sparse_first"], 1),
        }

    def stats(self) -> dict:
        total = self.sparse_calls + self.dense_calls
        return {
            "backend": self.backend,
            "sparse_calls": self.sparse_calls,
            "dense_calls": self.dense_calls,
            "sparse_fraction": round(self.sparse_calls / total, 4) if total else None,
            "attn_ms": {k: round(v, 1) for k, v in self.attn_ms.items()},
            "attn_ms_per_call": self._per_call(),
            "last_step": self.step,
            "total_steps": self.total_steps,
            "dense_steps": dense_step_count(self.policy.first_dense_steps, self.total_steps),
            "dense_layers": self.policy.first_dense_layers,
            "sink": None if self.sink is None else
                    {"start": self.sink.start, "tokens": self.sink.tokens,
                     "seq_len": self.sink.seq_len, "video_start": self.sink.video_start},
            "tau": self.policy.tau,
            "thresh_type": self.policy.thresh_type,
            "sink_mode": self.policy.sink_mode,
            "gate": self.gate_stats,
            "density": self.density,
            "declined": dict(self.declined),
        }
