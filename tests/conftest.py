"""Rejestruje repo jako pakiet `solattn_h3` bez wykonywania __init__.py.

ComfyUI importuje custom node jako pakiet, wiec moduly uzywaja importow
wzglednych. Testy jednostkowe nie moga jednak wykonac __init__.py, bo ten
ciagnie za soba `comfy`, ktorego poza ComfyUI nie ma. Rejestracja samego
__path__ daje dzialajace `from solattn_h3.layout import ...` i zero zaleznosci.
"""
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
if "solattn_h3" not in sys.modules:
    pkg = types.ModuleType("solattn_h3")
    pkg.__path__ = [str(ROOT)]
    sys.modules["solattn_h3"] = pkg
