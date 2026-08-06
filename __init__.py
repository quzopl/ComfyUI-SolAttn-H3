"""Sol-Attn dla MiniMax-H3 w ComfyUI.

Poza ComfyUI ten pakiet importuje sie do pustych mapowan zamiast wybuchac na
braku `comfy` — inaczej pytest, ktory tworzy kolektor Package dla katalogu repo
(ma `__init__.py`, bo tego wymaga ComfyUI), przewracalby cala suite.

Lapany jest wylacznie brak samego `comfy`. Kazdy inny ImportError — literowka w
nazwie modulu, brakujaca zaleznosc, blad w `nodes.py` — propaguje, zeby wewnatrz
ComfyUI nie zamienil sie w cicho pusta liste wezlow.
"""
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as exc:
    if (exc.name or "").split(".")[0] != "comfy":
        raise
    NODE_CLASS_MAPPINGS: dict = {}
    NODE_DISPLAY_NAME_MAPPINGS: dict = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
