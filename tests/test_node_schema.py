"""The node's declared inputs, checked without ComfyUI.

INPUT_TYPES is pure data, so it can be exercised by importing nodes.py with a
stub `comfy` in place. Worth doing, because a change here breaks every saved
workflow rather than one code path, and nothing else in the suite would notice.
"""
import sys
import types

import pytest


def _input_types():
    """Import nodes.py against a stub comfy and return its INPUT_TYPES()."""
    stub = types.ModuleType("comfy")
    ext = types.ModuleType("comfy.patcher_extension")
    ext.WrappersMP = types.SimpleNamespace(OUTER_SAMPLE="outer_sample",
                                           DIFFUSION_MODEL="diffusion_model")
    stub.patcher_extension = ext
    added = {"comfy": stub, "comfy.patcher_extension": ext}
    saved = {k: sys.modules.get(k) for k in added}
    sys.modules.update(added)
    try:
        from solattn_h3.nodes import SolAttnH3
        return SolAttnH3.INPUT_TYPES()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_kv_splits_is_optional_so_older_graphs_still_load():
    """`kv_splits` arrived after the shipped workflows were written.

    ComfyUI rejects a graph that omits a *required* input, so declaring it
    required breaks every previously saved workflow and every API graph in
    `workflows/` — verified against a live server before this test existed.
    `patch()` defaults it to 1, which is the reference line anyway.
    """
    schema = _input_types()
    assert "kv_splits" not in schema.get("required", {})
    assert "kv_splits" in schema.get("optional", {})


def test_the_documented_widgets_are_all_present():
    schema = _input_types()
    names = set(schema.get("required", {})) | set(schema.get("optional", {}))
    assert names == {"model", "enabled", "tau", "thresh_type", "first_dense_steps",
                     "first_dense_layers", "sink_mode", "correctness_gate",
                     "strict", "kv_splits"}


# Widget-typed inputs render as controls; anything else is a socket you must
# wire. "uppercase means a link" is wrong — BOOLEAN, INT and COMBO are widgets.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def _is_link(definition):
    kind = definition[0]
    if isinstance(kind, list):        # combo, given as its list of choices
        return False
    return isinstance(kind, str) and kind not in WIDGET_TYPES and "COMBO" not in kind


def test_model_is_the_only_link_input():
    """Everything else must stay a widget, or the node stops being configurable
    from the graph without extra wiring."""
    schema = _input_types()
    every = {**schema.get("required", {}), **schema.get("optional", {})}
    assert [n for n, d in every.items() if _is_link(d)] == ["model"]
