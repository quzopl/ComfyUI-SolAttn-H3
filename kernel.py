"""GPU architecture detection, backend selection and lazy import of the kernel.

`sol_attn()` picks its own backend from `torch.cuda.get_device_capability()`.
This module reproduces that decision so it can be *reported* before the first
call and tested without a GPU. Drift against the released package is caught by
`test_table_matches_released_kernel`.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

# Architectures with a released CuTe DSL kernel. Everything else >= SM80 uses Triton.
CUTE_BACKENDS = {
    (9, 0): "cute_sm90",     # H100
    (10, 0): "cute_sm100",   # B200 / GB200
    (12, 0): "cute_sm120",   # RTX 5090, RTX PRO 6000 Blackwell
}
TRITON = "triton"


@dataclass(frozen=True)
class Probe:
    """What is known about this GPU before the kernel is ever called."""

    arch: tuple[int, int] | None
    cute_available: bool
    backend: str | None
    available: bool
    error: str | None = None

    def describe(self) -> str:
        if not self.available:
            return f"unavailable: {self.error}"
        arch = f"SM{self.arch[0]}{self.arch[1]}"
        cute = "CuTe DSL available" if self.cute_available else "CuTe DSL unavailable"
        return f"{arch}, {cute}, backend={self.backend}"


def backend_for_arch(arch: tuple[int, int], cute_available: bool) -> str:
    """The backend `sol_attn()` will pick for this architecture.

    CuTe wins only when the architecture has a specialized kernel *and* the
    runtime imports; otherwise Triton. Below SM80 the kernel has no path at all
    and raises, so we do too.
    """
    if arch[0] < 8:
        raise RuntimeError(
            f"Sol-Attn requires compute capability >= 8.0; detected SM{arch[0]}{arch[1]}"
        )
    if cute_available and arch in CUTE_BACKENDS:
        return CUTE_BACKENDS[arch]
    return TRITON


def cute_runtime_available() -> bool:
    """Whether the optional CuTe DSL runtime can be imported."""
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return True


@functools.lru_cache(maxsize=None)
def probe(device=None) -> Probe:
    """One-off environment detection. Never raises."""
    try:
        import torch
    except ImportError as exc:
        return Probe(None, False, None, False, f"torch missing: {exc}")

    if not torch.cuda.is_available():
        return Probe(None, False, None, False, "CUDA unavailable")

    try:
        import sol_attn  # noqa: F401
    except ImportError as exc:
        return Probe(None, False, None, False,
                     f"the sol-attn package is not installed ({exc}); "
                     "uv pip install -e <sana>/techniques/sparse_backends")

    arch = tuple(torch.cuda.get_device_capability(device))
    cute = cute_runtime_available()
    try:
        backend = backend_for_arch(arch, cute)
    except RuntimeError as exc:
        return Probe(arch, cute, None, False, str(exc))
    return Probe(arch, cute, backend, True)


def load_sol_attn():
    """The kernel's public API. Imported lazily — the first call compiles."""
    from sol_attn import sol_attn

    return sol_attn
