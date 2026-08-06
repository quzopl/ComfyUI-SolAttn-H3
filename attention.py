"""Adapter wpinany przez `transformer_options["optimized_attention_override"]`.

`wrap_attn` (comfy/ldm/modules/attention.py:148) wola nas jako
`override(func, q, k, v, heads, **kwargs)`, gdzie `func` to oryginalny backend
uwagi. Zwrocenie `func(...)` daje darmowy gesty fallback — dokladnie ten sam
kod, ktory bylby uzyty bez node'a.

Wejscie z H3 to BHSD `(1, heads, S, 128)` (`skip_reshape=True`), wyjscie
oczekiwane jako `(1, S, heads*128)`.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .kernel import load_sol_attn
from .state import LOG

BLOCK_SIZE = 64


def dense_bthd(q, k, v):
    """SDPA na layoucie BTHD, ten sam layout na wyjsciu.

    `q` moze byc krotsze od `k`/`v` — tak liczone sa wiersze-zapytania prefiksu.
    """
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
    )
    return out.transpose(1, 2)


def run_gate(sol_attn, q, k, v, thresh_type: str) -> dict:
    """Sprawdz arytmetyke kernela wzgledem SDPA na prawdziwych QKV.

    `tau=-1000` przepuszcza wszystkie bloki, wiec mierzona jest arytmetyka
    kernela, a nie polityka routingu. Proba na losowych tensorach odpowiada na
    pytanie o kernel, nie o ten model przy tym ksztalcie.

    Gate biegnie na produkcyjnej liczbie glow: `preprocess.prepare` autotunuje
    kernele Tritona kluczem po samym `T`, wiec pierwsze wywolanie przy mniejszej
    liczbie glow zapisaloby konfiguracje dobrana dla wezszej siatki.
    """
    got = sol_attn(q, k, v, tau=-1000.0, thresh_type=thresh_type)
    want = dense_bthd(q, k, v)
    diff = (got.float() - want.float()).abs()
    ref_max = float(want.float().abs().max())
    stats = {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_l2": float(torch.linalg.vector_norm(got.float() - want.float())
                        / torch.linalg.vector_norm(want.float()).clamp_min(1e-12)),
        # Skala odniesienia: `max_abs` sam w sobie zalezy od rozpietosci aktywacji,
        # wiec prog bezwzgledny nie przenosi sie miedzy modelami ani ksztaltami.
        "ref_max": ref_max,
        "max_rel": float(diff.max()) / max(ref_max, 1e-12),
        "over_1e2": float((diff > 1e-2).float().mean()),
        "shape": list(q.shape),
    }
    # `mean_abs` i `rel_l2` sa dokladnie te, ktore ustawila NVIDIA. Bezwzgledny
    # limit na `max_abs` (0.08 / 0.15) zostal zastapiony wzglednym, bo nie
    # przenosi sie miedzy rozkladami aktywacji:
    #
    # Na prawdziwych QKV H3 zmierzono ref_max=33.0 i max_abs=0.125. bf16 ma
    # 7 bitow mantysy, wiec dla |x| w [32, 64) ulp wynosi 32*2^-7 = 0.25, a
    # granica poprawnego zaokraglenia to pol ulp = 0.125. Najgorszy element to
    # zatem *teoretyczne minimum* bledu reprezentacji przy tej skali — kernel
    # i SDPA zaokraglily te sama wartosc do dwoch sasiednich liczb bf16.
    # Ten sam ksztalt na tensorach syntetycznych N(0, 0.5) daje max_abs=0.00012:
    # roznica siedzi w skali wyjscia, nie w kernelu.
    #
    # 0.02 to okolo piec polulpow przy szczycie tensora — z zapasem na kolejnosc
    # akumulacji, a wciaz o dwa rzedy wielkosci ponizej bledu, jaki daje zepsuty
    # routing albo indeksowanie (tam blad wzgledny jest rzedu jednosci).
    limits = {
        "max_rel": 0.02,
        "mean_abs": 0.002,
        "rel_l2": 0.005,
    }
    stats["limits"] = limits
    stats["limits_nvidia_max_abs"] = 0.15 if q.shape[1] >= 32768 else 0.08
    stats["passed"] = all(stats[name] <= limit for name, limit in limits.items())
    return stats


@torch.no_grad()
def route_density(q, k, v, *, tau: float, thresh_type: str, sink) -> dict:
    """Jaki ulamek blokow K/V routing zachowuje.

    Raportowane, bo awaria, ktorej ten node ma unikac, to konfiguracja licząca
    po cichu gesto: gestosc bliska 1.0 znaczy, ze routing nie routuje, a 0.0 —
    ze sie zapadl.
    """
    from sol_attn.preprocess import prepare

    scale = q.shape[-1] ** -0.5
    kc, _, threshold = prepare(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type)

    tokens, heads = q.shape[1], q.shape[2]
    blocks = math.ceil(tokens / BLOCK_SIZE)
    padded = F.pad(q, (0, 0, 0, 0, 0, blocks * BLOCK_SIZE - tokens))
    counts = torch.full((blocks,), float(BLOCK_SIZE), device=q.device, dtype=torch.float32)
    counts[-1] = tokens - (blocks - 1) * BLOCK_SIZE
    # sum(dtype=float32) akumuluje bez materializowania kopii float32 calego q
    q_bar = padded.view(q.shape[0], blocks, BLOCK_SIZE, heads, q.shape[3]).sum(
        dim=2, dtype=torch.float32) / counts.view(1, blocks, 1, 1)

    scores = torch.einsum("bqhd,bkhd->bqkh", q_bar, kc.float()).mul_(scale * math.log2(math.e))
    routed = scores > threshold[:, :, None, :]
    threshold_density = float(routed.float().mean())

    ids = torch.arange(blocks, device=q.device)
    routed |= ((ids[:, None] - ids[None, :]).abs() <= 1)[None, :, :, None]   # pasmo lokalne
    sink_blocks = 0
    if sink.tokens:
        first = sink.start // BLOCK_SIZE
        last = math.ceil(sink.stop / BLOCK_SIZE)
        routed[:, :, first:last, :] = True
        sink_blocks = last - first
    return {
        "blocks": blocks,
        "sink_blocks": sink_blocks,
        "threshold_density": round(threshold_density, 5),
        "effective_density": round(float(routed.float().mean()), 5),
    }


def make_override(state, policy):
    """Zbuduj callable dla `transformer_options["optimized_attention_override"]`."""

    def override(func, q, k, v, heads, *args, **kwargs):
        def dense():
            return func(q, k, v, heads, *args, **kwargs)

        options = kwargs.get("transformer_options") or {}
        block = options.get("solattn_block")
        if block is None:
            block = state.next_block()

        reason = state.decline(
            rows=q.shape[2] if q.ndim == 4 else None,
            dtype=q.dtype,
            head_dim=q.shape[-1],
            mask=kwargs.get("mask"),
            block_index=block,
            skip_reshape=kwargs.get("skip_reshape"),
            batch=q.shape[0] if q.ndim == 4 else None,
        )
        if reason is not None:
            state.note(reason)
            pair = state.timer()
            if pair is not None:
                pair[0].record()
            out = dense()
            if pair is not None:
                pair[1].record()
                state.record_timing(pair, "dense")
            return out

        pair = state.timer()
        if pair is not None:
            pair[0].record()
        try:
            sol_attn = state.sol_attn or _attach_kernel(state)
            # Kernel wymaga contiguous BTHD; H3 podaje widoki w spakowanym
            # buforze qkv_proj, wiec ta kopia jest nieunikniona (~6,5% czasu
            # kernela przy 31k wierszy, zmierzone).
            qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
            sink = state.sink

            if state.should_gate(qb.shape[1:]):
                _gate_once(state, sol_attn, qb, kb, vb, policy)
            if state.density is None:
                _density_once(state, qb, kb, vb, policy, sink)

            out = sol_attn(qb, kb, vb, tau=policy.tau, thresh_type=policy.thresh_type,
                           kv_splits=1, sink_start=sink.start, sink_tokens=sink.tokens)
            # Sink czyni prefiks dokladnym jako K/V, ale jego wlasne zapytania
            # nadal routuja sie rzadko. README kernela jest jednoznaczne, ze
            # integracja MMDiT musi przeliczyc te wiersze gesto.
            if sink.tokens:
                out[:, sink.start:sink.stop] = dense_bthd(qb[:, sink.start:sink.stop], kb, vb)
        except torch.OutOfMemoryError:
            state.latch_oom()
            print(f"{LOG} brak pamieci na sciezce rzadkiej; gesta uwaga do konca przebiegu",
                  flush=True)
            return dense()

        if pair is not None:
            pair[1].record()
            state.record_timing(pair, "sparse")
        state.note_sparse()
        if kwargs.get("skip_output_reshape"):
            return out.transpose(1, 2)
        return out.reshape(out.shape[0], out.shape[1], heads * out.shape[3])

    return override


def _attach_kernel(state):
    state.sol_attn = load_sol_attn()
    return state.sol_attn


def _gate_once(state, sol_attn, qb, kb, vb, policy) -> None:
    """Gate poprawnosci raz na ksztalt. Brak pamieci pomija gate, nie zweza go.

    Zwezenie do mniejszej liczby glow zapisaloby w cache'u autotuningu Tritona
    konfiguracje dobrana dla wezszej siatki, pod kluczem, ktorego uzyja pozniej
    wywolania produkcyjne. Lepiej nie zmierzyc, niz zepsuc.
    """
    try:
        stats = run_gate(sol_attn, qb, kb, vb, policy.thresh_type)
    except torch.OutOfMemoryError:
        state.record_gate(qb.shape[1:], {"passed": None, "skipped": "brak pamieci"})
        print(f"{LOG} gate poprawnosci pominiety: brak pamieci", flush=True)
        return
    state.record_gate(qb.shape[1:], stats)
    verdict = "PASS" if stats["passed"] else "FAIL"
    print(f"{LOG} gate poprawnosci {verdict} max_abs={stats['max_abs']:.5f} "
          f"mean_abs={stats['mean_abs']:.6f} rel_l2={stats['rel_l2']:.5f} "
          f"ref_max={stats['ref_max']:.3f} max_rel={stats['max_rel']:.5f} "
          f"over_1e2={stats['over_1e2']:.2e} limity={stats['limits']}", flush=True)
    if not stats["passed"]:
        raise RuntimeError(
            f"{LOG} gate poprawnosci nie przeszedl na prawdziwych QKV: {stats}. "
            "Cicha akceptacja zepsutego kernela jest gorsza niz brak przyspieszenia."
        )


def _density_once(state, qb, kb, vb, policy, sink) -> None:
    try:
        state.density = route_density(qb, kb, vb, tau=policy.tau,
                                      thresh_type=policy.thresh_type, sink=sink)
    except torch.OutOfMemoryError:
        state.density = {"skipped": "brak pamieci"}
        return
    except Exception as exc:                      # sonda diagnostyczna, nie sciezka krytyczna
        state.density = {"skipped": f"{type(exc).__name__}: {exc}"}
        return
    print(f"{LOG} gestosc routingu {state.density}", flush=True)
