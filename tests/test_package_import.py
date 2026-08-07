"""Pakiet musi dac sie zaimportowac bez ComfyUI.

Katalog repo ma __init__.py, bo tego wymaga ComfyUI. To sprawia, ze pytest
tworzy dla niego kolektor Package i importuje go w setup() — jesli import
wymaga `comfy`, przewraca sie cala suite, a nie jeden test.
"""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name):
    """Zaladuj repo jako pakiet pod podana nazwa.

    Rejestracja w sys.modules przed exec_module jest konieczna: bez niej import
    wzgledny `.nodes` w __init__.py nie ma jak rozwiazac pakietu nadrzednego.
    """
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_import_bez_comfy_daje_puste_mapowania():
    module = _load("solattn_h3_probe")
    assert module.NODE_CLASS_MAPPINGS == {}


def test_blad_niezwiazany_z_comfy_propaguje(monkeypatch, tmp_path):
    """Literowka albo brakujaca zaleznosc nie moze zamienic sie w pusta liste wezlow."""
    import builtins
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.endswith("nodes") or name == "nodes":
            raise ImportError("No module named 'cos_zupelnie_innego'",
                              name="cos_zupelnie_innego")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="cos_zupelnie_innego"):
        _load("solattn_h3_probe_fail")
