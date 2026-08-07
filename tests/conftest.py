"""Register the repo as a `solattn_h3` package without executing __init__.py.

ComfyUI imports a custom node as a package, so the modules use relative imports.
Unit tests, however, must not execute __init__.py, because that pulls in `comfy`
which does not exist outside ComfyUI. Registering just the __path__ makes
`from solattn_h3.layout import ...` work with zero dependencies.
"""
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
if "solattn_h3" not in sys.modules:
    pkg = types.ModuleType("solattn_h3")
    pkg.__path__ = [str(ROOT)]
    sys.modules["solattn_h3"] = pkg
