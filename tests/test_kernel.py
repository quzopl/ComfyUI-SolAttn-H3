import pytest

from solattn_h3.kernel import backend_for_arch


@pytest.mark.parametrize("arch,cute,expected", [
    ((9, 0), True, "cute_sm90"),
    ((10, 0), True, "cute_sm100"),
    ((12, 0), True, "cute_sm120"),
    ((9, 0), False, "triton"),      # no CuTe -> Triton, specialized architecture or not
    ((12, 0), False, "triton"),
    ((8, 9), True, "triton"),       # Ada has no CuTe kernel
    ((8, 6), True, "triton"),
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
