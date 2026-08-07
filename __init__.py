"""Sol-Attn for MiniMax-H3 in ComfyUI.

Outside ComfyUI this package imports to empty mappings instead of blowing up on
a missing `comfy` — otherwise pytest, which builds a Package collector for the
repo directory (it has an __init__.py because ComfyUI requires one), would take
down the whole suite.

Only a missing `comfy` is caught. Any other ImportError — a typo in a module
name, a missing dependency, a bug in `nodes.py` — propagates, so that inside
ComfyUI it never turns into a silently empty node list.
"""
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as exc:
    if (exc.name or "").split(".")[0] != "comfy":
        raise
    NODE_CLASS_MAPPINGS: dict = {}
    NODE_DISPLAY_NAME_MAPPINGS: dict = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
