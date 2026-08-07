"""Wezel SolAttnH3 — UI i montaz na ModelPatcherze.

Montowane sa trzy rzeczy, wszystkie na publicznych API ComfyUI:

  1. `transformer_options["optimized_attention_override"]` — przechwycenie uwagi
  2. wrapper `WrappersMP.DIFFUSION_MODEL` — raz na forward: layout, sink, krok
  3. `patches_replace["dit"][("double_block", i)]` — stempel indeksu bloku
  4. wrapper `WrappersMP.OUTER_SAMPLE` — granice przebiegu i kontrola zbiorcza

Zaden plik w `comfy/` nie jest modyfikowany.
"""
from __future__ import annotations

from comfy.patcher_extension import WrappersMP

from .attention import make_override
from .kernel import probe
from .layout import SINK_MODES, sink_from_segments
from .state import LOG, Policy, SolAttnState, resolve_step

KEY = "solattn_h3"
STAMP = "solattn_block"


class SolAttnH3:
    """Training-free rzadka uwaga Sol-Attn dla natywnego MiniMax-H3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "tau": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 10.0, "step": 0.05,
                                  "tooltip": "Wyzsze wartosci wybieraja mniej blokow K/V do "
                                             "dokladnej uwagi. 1.0 to zwalidowana polityka H3."}),
                "thresh_type": (["diag", "exact"], {"default": "diag",
                                "tooltip": "exact = prog pelnokowariancyjny, dokladniejszy i drozszy."}),
                "first_dense_steps": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 50.0, "step": 0.05,
                                      "tooltip": "Ponizej 1 to ulamek harmonogramu, od 1 sztywna "
                                                 "liczba krokow. 0.2 odpowiada referencyjnym 10/50."}),
                "first_dense_layers": ("INT", {"default": 2, "min": 0, "max": 50,
                                       "tooltip": "Pierwsze N blokow DiT liczonych gesto."}),
                "sink_mode": (list(SINK_MODES), {"default": "prefix",
                              "tooltip": "prefix = tekst + warunkowanie + audio trzymane dokladnie. "
                                         "text = polityka referencyjna, sam tekst."}),
                "correctness_gate": ("BOOLEAN", {"default": True,
                                     "tooltip": "Raz na ksztalt porownuje kernel z SDPA na "
                                                "prawdziwych QKV. Fail przerywa generowanie."}),
                "strict": ("BOOLEAN", {"default": False,
                           "tooltip": "Zamienia kazda niezamierzona odmowe sciezki rzadkiej "
                                      "w wyjatek. Do walidacji, nie do codziennej pracy."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = ("Sol-Attn (NVIDIA Sol Engine) dla MiniMax-H3. Training-free rzadka uwaga; "
                   "degraduje sie do gestej uwagi z nazwanym powodem, gdy kontrakt kernela "
                   "nie jest spelniony.")

    def patch(self, model, enabled, tau, thresh_type, first_dense_steps,
              first_dense_layers, sink_mode, correctness_gate, strict):
        blocks = _block_count(model)

        policy = Policy(enabled=enabled, tau=tau, thresh_type=thresh_type,
                        first_dense_steps=first_dense_steps,
                        first_dense_layers=first_dense_layers, sink_mode=sink_mode,
                        correctness_gate=correctness_gate, strict=strict)
        state = SolAttnState(policy)

        found = probe()
        state.backend = found.backend
        if not found.available:
            state.kernel_error = found.error
            print(f"{LOG} kernel {found.describe()} — model przejdzie na gesta uwage", flush=True)
        else:
            print(f"{LOG} {found.describe()}, blokow DiT: {blocks}", flush=True)

        patched = model.clone()
        options = patched.model_options.setdefault("transformer_options", {})
        options["optimized_attention_override"] = make_override(state, policy)
        patched.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, KEY, _make_run_wrapper(state))
        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, KEY,
                                     _make_forward_wrapper(state, policy))
        for index in range(blocks):
            patched.set_model_patch_replace(_make_stamp(index), "dit", "double_block", index)
        return (patched,)


def _block_count(model) -> int:
    """Liczba blokow DiT; sluzy tez jako sprawdzenie, ze to na pewno MiniMax-H3."""
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    blocks = getattr(diffusion, "blocks", None)
    if blocks is None or type(diffusion).__name__ != "MiniMaxH3Model":
        raise ValueError(
            f"{LOG} SolAttnH3 dziala tylko z MiniMax-H3; dostano "
            f"{type(diffusion).__name__}. Wezel nie zostal zamontowany."
        )
    return len(blocks)


def _make_stamp(index: int):
    """Stempluje prawdziwy indeks bloku i oddaje sterowanie oryginalnemu blokowi.

    Liczenie wywolan uwagi zamiast stemplowania przesuneloby first_dense_layers,
    bo token_refiner (model.py:584) tez wola Attention z head_dim 128.
    """
    def stamp(args, extra):
        args["transformer_options"][STAMP] = index
        return extra["original_block"](args)
    return stamp


def _make_run_wrapper(state):
    """Granice jednego przebiegu samplera: reset licznikow i kontrola zbiorcza."""
    def wrapper(executor, *args, **kwargs):
        state.begin_run()
        try:
            return executor(*args, **kwargs)
        finally:
            state.end_run()
            if state.sparse_calls or state.dense_calls:
                print(f"{LOG} {state.stats()}", flush=True)
    return wrapper


def _make_forward_wrapper(state, policy):
    """Raz na forward: layout, zakres sinka i numer kroku z harmonogramu."""
    def wrapper(executor, x, timestep, context, transformer_options, **kwargs):
        sink = None
        try:
            layout = _resolve_layout(executor.class_obj, kwargs.get("minimax_payload"),
                                     x, context)
            if layout is not None:
                sink = sink_from_segments(layout.segments, layout.seq_len, policy.sink_mode)
        except Exception as exc:
            _warn_once(state, f"nie udalo sie wyznaczyc sinka: {type(exc).__name__}: {exc}")

        sample_sigmas = transformer_options.get("sample_sigmas")
        step = resolve_step(transformer_options.get("sigmas"), sample_sigmas)
        total = int(sample_sigmas.shape[0]) - 1 if sample_sigmas is not None else None
        state.begin_forward(sink, step, total)
        return executor(x, timestep, context, transformer_options, **kwargs)
    return wrapper


def _resolve_layout(model, payload, x, context):
    """Layout spakowanej sekwencji: z payloadu, a gdy go brak — odtworzony.

    `extra_conds` buduje go raz na przebieg, ale tylko gdy zna latent_shapes.
    Odtworzenie odwzorowuje `model.py:506-524`. Nieaktualny layout nie moze dac
    zlego wyniku: sprawdzenie `rows != sink.seq_len` w `decline()` zlapie go
    i wywolanie spadnie na gesta uwage.
    """
    payload = payload or {}
    layout = payload.get("layout")
    if layout is not None:
        return layout

    import comfy.ldm.common_dit
    from comfy.ldm.minimax.model import PackedLayout

    video_x = comfy.ldm.common_dit.pad_to_patch_size(x[0], model.patch_size)
    return PackedLayout(context.shape[1], video_x.shape[2], video_x.shape[3], video_x.shape[4],
                        x[1].shape[-1], keyframes=payload.get("keyframes"),
                        refs=payload.get("refs"), frame_count=payload.get("frame_count"))


def _warn_once(state, message: str) -> None:
    if message not in state._logged:
        state._logged.add(message)
        print(f"{LOG} {message}", flush=True)


NODE_CLASS_MAPPINGS = {"SolAttnH3": SolAttnH3}
NODE_DISPLAY_NAME_MAPPINGS = {"SolAttnH3": "Sol-Attn (MiniMax-H3)"}
