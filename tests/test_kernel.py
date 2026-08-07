import pytest

from solattn_h3.kernel import backend_for_arch


@pytest.mark.parametrize("arch,cute,expected", [
    ((9, 0), True, "cute_sm90"),
    ((10, 0), True, "cute_sm100"),
    ((12, 0), True, "cute_sm120"),
    ((9, 0), False, "triton"),      # brak CuTe -> Triton, mimo wyspecjalizowanej architektury
    ((12, 0), False, "triton"),
    ((8, 9), True, "triton"),       # Ada nie ma kernela CuTe
    ((8, 6), True, "triton"),
    ((8, 0), True, "triton"),
])
def test_backend_for_arch(arch, cute, expected):
    assert backend_for_arch(arch, cute) == expected


def test_arch_below_sm80_unsupported():
    with pytest.raises(RuntimeError, match="8.0"):
        backend_for_arch((7, 5), True)


def test_tablica_zgodna_z_wydanym_kernelem():
    """Lokalna tablica musi zgadzac sie z ta w sol_attn.interface.

    Odtwarzamy ja, zeby testy dzialaly bez GPU i bez zainstalowanego pakietu.
    Jesli NVIDIA doda architekture, ten test to wychwyci zamiast cichego
    rozjechania sie wyboru backendu.
    """
    interface = pytest.importorskip("sol_attn.interface")
    for arch, expected in interface._CUTE_BACKENDS.items():
        assert backend_for_arch(arch, True) == expected
