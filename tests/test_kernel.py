import pytest

from solattn_h3.kernel import CUTE_BACKENDS, backend_for_arch


@pytest.mark.parametrize("arch,cute,expected", [
    ((9, 0), True, "cute_sm90"),
    ((10, 0), True, "cute_sm100"),
    ((10, 3), True, "triton"),      # B300: no row yet in the installable kernel
    ((12, 0), True, "cute_sm120"),
    ((8, 9), True, "cute_sm89"),    # Ada gained a CuTe kernel upstream in Aug 2026
    ((9, 0), False, "triton"),      # no CuTe -> Triton, specialized architecture or not
    ((12, 0), False, "triton"),
    ((8, 9), False, "triton"),
    ((8, 6), True, "triton"),       # Ampere consumer has no CuTe kernel
    ((8, 0), True, "triton"),
])
def test_backend_for_arch(arch, cute, expected):
    assert backend_for_arch(arch, cute) == expected


def test_arch_below_sm80_unsupported():
    with pytest.raises(RuntimeError, match="8.0"):
        backend_for_arch((7, 5), True)


def test_table_matches_released_kernel():
    """The local table must agree with the one in sol_attn.interface.

    We reproduce it so the tests run without a GPU and without the package
    installed. If NVIDIA adds an architecture, this test catches it instead of
    letting backend selection drift silently.
    """
    interface = pytest.importorskip("sol_attn.interface")
    for arch, expected in interface._CUTE_BACKENDS.items():
        assert backend_for_arch(arch, True) == expected


def test_table_claims_no_architecture_the_kernel_lacks():
    """The other direction: an entry we have and the kernel does not.

    The check above iterates over *upstream's* table, so it only sees rows we
    are missing. A row we invented — or one that survived a version of the
    package where it existed — passes it untouched, and the node then advertises
    a CuTe backend while the kernel quietly dispatches Triton. That is the same
    failure the SM103 comment in `kernel.py` describes, arrived at from the
    opposite side.
    """
    interface = pytest.importorskip("sol_attn.interface")
    extra = set(CUTE_BACKENDS) - set(interface._CUTE_BACKENDS)
    assert not extra, (
        f"the local table claims {sorted(extra)}, which the installed sol-attn "
        f"({getattr(interface, '__version__', 'unknown version')}) does not have. "
        "Either the package is older than the table and should be upgraded, or "
        "the row is wrong and backend reporting is lying about those cards."
    )


def test_probe_agrees_with_the_public_resolver():
    """`probe()` must report what the kernel will actually dispatch to.

    Upstream exposes `get_sol_attn_backend`; our own table is only the offline
    fallback. If the two ever disagree, the node would advertise one backend and
    run another.
    """
    from solattn_h3.kernel import probe

    found = probe()
    if not found.available:
        pytest.skip(f"kernel unavailable: {found.error}")
    sol_attn = pytest.importorskip("sol_attn")
    resolver = getattr(sol_attn, "get_sol_attn_backend", None)
    if resolver is None:
        pytest.skip("this sol-attn release has no public resolver")
    assert found.backend == resolver()
