"""Zegar kroku, licznik blokow, powody odmowy i kontrola zbiorcza przebiegu.

Zasada calego modulu: konfiguracja, ktora poprosila o rzadka uwage i po cichu
dostala gesta, to bledny pomiar w prawidlowej etykiecie. Dlatego kazda odmowa
niesie powod (string), a nie boolean, i jest liczona.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .layout import SinkRange

# Powody zamierzone — cicho, tylko licznik.
DECLINE_DISABLED = "disabled"
DECLINE_WARMUP = "warmup_step"
DECLINE_DENSE_LAYER = "dense_layer"
# Powody niezamierzone — log raz, w trybie strict wyjatek.
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
    """Parametry polityki rzadkiej uwagi.

    Domyslne wartosci pochodza ze zwalidowanej linii H3 w Sol-Engine
    (`models/minimax_h3/optimized/sol_attn_h3.py`) i nie powinny byc zmieniane
    bez pomiaru.
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
    """Liczba poczatkowych krokow liczonych gesto.

    Wartosc < 1 jest ulamkiem dlugosci harmonogramu, >= 1 sztywna liczba krokow.
    Referencyjne 10 pochodzi z harmonogramu 50-krokowego; przy 20 krokach
    oznaczaloby polowe przebiegu gesto.
    """
    if first_dense_steps >= 1:
        return int(first_dense_steps)
    if not total_steps:
        return 0
    return int(round(first_dense_steps * total_steps))


def resolve_step(sigmas, sample_sigmas) -> int | None:
    """Numer biezacego kroku odczytany z harmonogramu samplera.

    Wzorzec z `comfy/context_windows.py:558-560`. Referencja NVIDII musiala
    zgadywac poczatek requestu z kierunku zmian timestepu i pomylila sie na tym
    dwa razy; ComfyUI podaje caly harmonogram, wiec nie ma czego zgadywac.
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
    """Stan jednego zamontowanego node'a, wspoldzielony miedzy wrapperami."""

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

    _block: int = field(default=0, init=False)
    _max_step: int = field(default=-1, init=False)
    _oom: bool = field(default=False, init=False)
    _logged: set = field(default_factory=set, init=False)
    _gated: set = field(default_factory=set, init=False)

    # -- cykl zycia -----------------------------------------------------------

    def begin_run(self) -> None:
        """Poczatek jednego przebiegu samplera."""
        self.sparse_calls = 0
        self.dense_calls = 0
        self.declined = {}
        self.step = None
        self.total_steps = None
        self._max_step = -1
        self._block = 0
        self._oom = False
        self._logged = set()

    def begin_forward(self, sink: SinkRange | None, step: int | None,
                      total_steps: int | None) -> None:
        """Poczatek jednego forwardu modelu."""
        self.sink = sink
        self.step = step
        self.total_steps = total_steps
        self._block = 0
        if step is not None:
            self._max_step = max(self._max_step, step)

    def next_block(self) -> int:
        """Zapasowy licznik blokow, gdy stempel z patches_replace niedostepny."""
        index = self._block
        self._block += 1
        return index

    def latch_oom(self) -> None:
        self._oom = True

    # -- decyzja --------------------------------------------------------------

    def decline(self, *, rows: int | None = None, dtype=None, head_dim: int | None = None,
                mask=None, block_index: int | None = None, skip_reshape=None,
                batch: int | None = None) -> str | None:
        """Powod, dla ktorego to wywolanie nie moze isc sciezka rzadka, albo None.

        Kolejnosc od najtanszego sprawdzenia do najdrozszego.
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

    # -- ksiegowanie ----------------------------------------------------------

    def note(self, reason: str) -> None:
        self.declined[reason] = self.declined.get(reason, 0) + 1
        self.dense_calls += 1
        if reason in INTENTIONAL:
            return
        if self.policy.strict:
            raise RuntimeError(f"{LOG} odmowa sciezki rzadkiej: {reason}")
        if reason not in self._logged:
            self._logged.add(reason)
            print(f"{LOG} gesta uwaga, powod: {reason}", flush=True)

    def note_sparse(self) -> None:
        self.sparse_calls += 1

    def end_run(self) -> None:
        """Kontrola zbiorcza: przebieg, ktory minal rozgrzewke i nie tknal kernela.

        Powody per-call tego nie wylapia, bo powod robiacy szkode (warmup_step)
        jest sam w sobie legalny. Blad istnieje tylko w agregacie.
        """
        if self._max_step < 0:
            return
        warmup = dense_step_count(self.policy.first_dense_steps, self.total_steps)
        if self._max_step + 1 <= warmup or self.sparse_calls > 0:
            return
        message = (f"{LOG} przebieg przeszedl {self._max_step + 1} krokow przy "
                   f"first_dense_steps={warmup} i nie wykonal ani jednego wywolania "
                   f"rzadkiego; odmowy={self.declined}")
        if self.policy.strict:
            raise RuntimeError(message)
        print(f"{LOG} UWAGA: {message}", flush=True)

    # -- gate raz na ksztalt --------------------------------------------------

    def should_gate(self, shape) -> bool:
        return self.policy.correctness_gate and tuple(shape) not in self._gated

    def record_gate(self, shape, stats: dict) -> None:
        self._gated.add(tuple(shape))
        self.gate_stats = stats

    # -- raport ---------------------------------------------------------------

    def stats(self) -> dict:
        total = self.sparse_calls + self.dense_calls
        return {
            "backend": self.backend,
            "sparse_calls": self.sparse_calls,
            "dense_calls": self.dense_calls,
            "sparse_fraction": round(self.sparse_calls / total, 4) if total else None,
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
