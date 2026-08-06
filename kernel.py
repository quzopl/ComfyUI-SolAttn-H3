"""Wykrycie architektury GPU, wybor backendu i leniwy import kernela sol_attn.

Kernel wybiera backend sam, po `torch.cuda.get_device_capability()`. Ten modul
odtwarza te decyzje po to, zeby dalo sie ja *zaraportowac* przed pierwszym
wywolaniem i przetestowac bez GPU. Rozjazd wzgledem wydanego pakietu wychwytuje
`test_tablica_zgodna_z_wydanym_kernelem`.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

# Architektury z wydanym kernelem CuTe DSL. Reszta >= SM80 idzie na Tritona.
CUTE_BACKENDS = {
    (9, 0): "cute_sm90",     # H100
    (10, 0): "cute_sm100",   # B200 / GB200
    (12, 0): "cute_sm120",   # RTX 5090, RTX PRO 6000 Blackwell
}
TRITON = "triton"


@dataclass(frozen=True)
class Probe:
    """Co wiadomo o tym GPU zanim kernel zostanie wywolany."""

    arch: tuple[int, int] | None
    cute_available: bool
    backend: str | None
    available: bool
    error: str | None = None

    def describe(self) -> str:
        if not self.available:
            return f"niedostepny: {self.error}"
        arch = f"SM{self.arch[0]}{self.arch[1]}"
        cute = "CuTe DSL dostepny" if self.cute_available else "CuTe DSL niedostepny"
        return f"{arch}, {cute}, backend={self.backend}"


def backend_for_arch(arch: tuple[int, int], cute_available: bool) -> str:
    """Backend, ktory `sol_attn()` wybierze dla tej architektury.

    CuTe wygrywa tylko gdy architektura ma wyspecjalizowany kernel *i* runtime
    da sie zaimportowac; inaczej Triton. Ponizej SM80 kernel nie ma zadnej
    sciezki i podnosi wyjatek, wiec my tez.
    """
    if arch[0] < 8:
        raise RuntimeError(
            f"Sol-Attn wymaga compute capability >= 8.0; wykryto SM{arch[0]}{arch[1]}"
        )
    if cute_available and arch in CUTE_BACKENDS:
        return CUTE_BACKENDS[arch]
    return TRITON


def cute_runtime_available() -> bool:
    """Czy opcjonalny runtime CuTe DSL da sie zaimportowac."""
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return True


@functools.lru_cache(maxsize=None)
def probe(device=None) -> Probe:
    """Jednorazowe rozpoznanie srodowiska. Nie podnosi wyjatkow."""
    try:
        import torch
    except ImportError as exc:
        return Probe(None, False, None, False, f"brak torch: {exc}")

    if not torch.cuda.is_available():
        return Probe(None, False, None, False, "CUDA niedostepna")

    try:
        import sol_attn  # noqa: F401
    except ImportError as exc:
        return Probe(None, False, None, False,
                     f"pakiet sol-attn niezainstalowany ({exc}); "
                     "uv pip install -e vendor/sana-sol-engine/techniques/sparse_backends")

    arch = tuple(torch.cuda.get_device_capability(device))
    cute = cute_runtime_available()
    try:
        backend = backend_for_arch(arch, cute)
    except RuntimeError as exc:
        return Probe(arch, cute, None, False, str(exc))
    return Probe(arch, cute, backend, True)


def load_sol_attn():
    """Publiczne API kernela. Importowane leniwie — pierwsze wywolanie kompiluje."""
    from sol_attn import sol_attn

    return sol_attn
