"""The package must be importable without ComfyUI.

The repo directory has an __init__.py because ComfyUI requires one. That makes
pytest build a Package collector for it and import it during setup — and if that
import needs `comfy`, the whole suite goes down rather than a single test.
"""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name):
    """Load the repo as a package under the given name.

    Registering it in sys.modules before exec_module is required: without it the
    relative `.nodes` import in __init__.py has no parent package to resolve.
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


def test_import_without_comfy_yields_empty_mappings():
    module = _load("solattn_h3_probe")
    assert module.NODE_CLASS_MAPPINGS == {}


def test_an_unrelated_error_propagates(monkeypatch):
    """A typo or a missing dependency must not turn into an empty node list."""
    import builtins
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.endswith("nodes") or name == "nodes":
            raise ImportError("No module named 'something_entirely_different'",
                              name="something_entirely_different")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="something_entirely_different"):
        _load("solattn_h3_probe_fail")
