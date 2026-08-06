"""Diagnostyka Sol-Attn do uruchomienia na maszynie docelowej.

Odpowiada na cztery pytania, ktore decyduja, czy node ma tu sens:

  1. jaki backend wybierze kernel na tym GPU
  2. czy gate poprawnosci przechodzi (tau=-1000 vs SDPA)
  3. jaka wychodzi gestosc routingu przy produkcyjnym tau
  4. ile kernel daje wzgledem SDPA i SageAttention na realnych ksztaltach

Uruchamiane samodzielnie, bez ComfyUI:

    python selftest.py                      # domyslny przemiat
    python selftest.py --tokens 30976       # jeden ksztalt
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

if __package__ in (None, ""):
    # Uruchomienie jako skrypt: moduly uzywaja importow wzglednych (wymog
    # ComfyUI), wiec rejestrujemy katalog jako pakiet zamiast dokladac go do
    # sys.path. __init__.py nie jest wykonywany, wiec `comfy` nie jest potrzebne.
    import pathlib
    import types

    _root = pathlib.Path(__file__).resolve().parent
    _pkg = types.ModuleType("solattn_h3")
    _pkg.__path__ = [str(_root)]
    sys.modules.setdefault("solattn_h3", _pkg)
    from solattn_h3.attention import dense_bthd, route_density, run_gate
    from solattn_h3.kernel import load_sol_attn, probe
    from solattn_h3.layout import SinkRange
else:
    from .attention import dense_bthd, route_density, run_gate
    from .kernel import load_sol_attn, probe
    from .layout import SinkRange

HEADS = 56          # MiniMax-H3
DIM = 128


def timed(fn, iters=5, warmup=2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def make_qkv(tokens: int, heads: int, device):
    """QKV w layoucie, w jakim podaje je H3: widoki w spakowanym buforze qkv_proj."""
    torch.manual_seed(0)
    packed = torch.randn(tokens, 3 * heads * DIM, device=device, dtype=torch.bfloat16) * 0.5
    return [x.view(tokens, heads, DIM).transpose(0, 1).unsqueeze(0)
            for x in packed.split(heads * DIM, dim=-1)]


def run_case(tokens: int, heads: int, sink_tokens: int, thresh_type: str, tau: float) -> dict:
    device = torch.device("cuda")
    sol_attn = load_sol_attn()
    sink = SinkRange(0, sink_tokens, tokens, sink_tokens)
    print(f"\n{'=' * 74}\nT={tokens}  heads={heads}  sink={sink_tokens}  "
          f"thresh_type={thresh_type}  tau={tau}\n{'=' * 74}")

    q, k, v = make_qkv(tokens, heads, device)
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    per = qb.numel() * qb.element_size() / 2**20
    print(f"[mem ] tensor QKV: {per:.0f} MiB, trojka: {3 * per:.0f} MiB")

    started = time.perf_counter()
    try:
        sol_attn(qb, kb, vb, tau=tau, thresh_type=thresh_type, kv_splits=1,
                 sink_start=sink.start, sink_tokens=sink.tokens)
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"[FAIL] kernel podniosl wyjatek: {type(exc).__name__}: {exc}")
        return {"tokens": tokens, "failed": str(exc)}
    print(f"[ok  ] pierwsze wywolanie (z kompilacja): {time.perf_counter() - started:.1f} s")

    gate = run_gate(sol_attn, qb, kb, vb, thresh_type)
    print(f"[gate] {'PASS' if gate['passed'] else 'FAIL'}  "
          f"max_abs={gate['max_abs']:.5f}  mean_abs={gate['mean_abs']:.6f}  "
          f"rel_l2={gate['rel_l2']:.5f}  limity={gate['limits']}")

    density = route_density(qb, kb, vb, tau=tau, thresh_type=thresh_type, sink=sink)
    print(f"[dens] {density}")

    ms_sol = timed(lambda: sol_attn(qb, kb, vb, tau=tau, thresh_type=thresh_type, kv_splits=1,
                                    sink_start=sink.start, sink_tokens=sink.tokens))
    ms_sdpa = timed(lambda: dense_bthd(qb, kb, vb))
    ms_copy = timed(lambda: [x.transpose(1, 2).contiguous() for x in (q, k, v)])
    try:
        from sageattention import sageattn
        ms_sage = timed(lambda: sageattn(q, k, v, tensor_layout="HND", is_causal=False))
    except Exception as exc:
        print(f"[sage] niedostepna: {type(exc).__name__}")
        ms_sage = None

    print(f"[time] sol_attn={ms_sol:.2f} ms   sdpa={ms_sdpa:.2f} ms"
          + (f"   sage={ms_sage:.2f} ms" if ms_sage else ""))
    print(f"[time] kopia BHSD->BTHD (3 tensory)={ms_copy:.2f} ms "
          f"({100 * ms_copy / ms_sol:.1f}% czasu kernela)")
    print(f"[spd ] vs sdpa={ms_sdpa / ms_sol:.2f}x"
          + (f"   vs sage={ms_sage / ms_sol:.2f}x" if ms_sage else ""))
    print(f"[mem ] peak={torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    torch.cuda.reset_peak_memory_stats()

    return {"tokens": tokens, "gate": gate["passed"],
            "density": density.get("effective_density"),
            "sol": ms_sol, "sdpa": ms_sdpa, "sage": ms_sage, "copy": ms_copy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="*",
                        default=[8192, 16384, 30976],
                        help="dlugosci sekwencji do sprawdzenia")
    parser.add_argument("--heads", type=int, default=HEADS)
    parser.add_argument("--sink-tokens", type=int, default=960,
                        help="rozmiar prefiksu (tekst + warunkowanie + audio)")
    parser.add_argument("--thresh-type", default="diag", choices=["diag", "exact"])
    parser.add_argument("--tau", type=float, default=1.0)
    args = parser.parse_args()

    found = probe()
    print(f"GPU        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
    print(f"kernel     : {found.describe()}")
    print(f"torch      : {torch.__version__}  CUDA {torch.version.cuda}")
    if not found.available:
        print("\nSelftest przerwany: kernel niedostepny.")
        raise SystemExit(1)

    rows = [run_case(t, args.heads, args.sink_tokens, args.thresh_type, args.tau)
            for t in args.tokens]

    print(f"\n{'=' * 74}\nPODSUMOWANIE ({found.backend})\n{'=' * 74}")
    print(f"{'T':>7} {'gate':>5} {'gestosc':>9} {'sol ms':>9} {'sdpa ms':>9} "
          f"{'sage ms':>9} {'vs sdpa':>9} {'vs sage':>9}")
    for row in rows:
        if row.get("failed"):
            print(f"{row['tokens']:>7} {'FAIL':>5}  {row['failed'][:50]}")
            continue
        sage = f"{row['sage']:9.2f}" if row["sage"] else "        -"
        vsage = f"{row['sage'] / row['sol']:8.2f}x" if row["sage"] else "        -"
        dens = f"{row['density']:9.4f}" if row["density"] is not None else "        -"
        print(f"{row['tokens']:>7} {'PASS' if row['gate'] else 'FAIL':>5} {dens} "
              f"{row['sol']:9.2f} {row['sdpa']:9.2f} {sage} "
              f"{row['sdpa'] / row['sol']:8.2f}x {vsage}")

    if not all(r.get("gate") for r in rows if not r.get("failed")):
        raise SystemExit("gate poprawnosci nie przeszedl na co najmniej jednym ksztalcie")


if __name__ == "__main__":
    main()
