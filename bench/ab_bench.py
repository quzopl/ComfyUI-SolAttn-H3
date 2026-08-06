"""Pomiar A/B: ten sam graf H3 z wezlem SolAttnH3 i bez niego.

Porownywane sa dwie rzeczy naraz, bo mierza co innego:

  * **czas end-to-end** — to, co widzi uzytkownik, ale na maszynie, gdzie model
    nie miesci sie w VRAM, zdominuje go streaming wag przez PCIe, a nie uwaga;
  * **czas samej uwagi** — czesc, na ktora Sol-Attn faktycznie wplywa, zbierana
    zdarzeniami CUDA wewnatrz node'a i raportowana w `stats()`.

Porownanie wyniku idzie po latentach, nie po klatkach: VAE jest deterministyczne,
wiec latent jest scislejszy i nie wymaga ladowania dekodera.

Uzycie:
    python bench/ab_bench.py --port 8199 --steps 8 --width 640 --height 384 --length 73
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid

PROMPT = ("A slow cinematic push-in on a lighthouse at dusk, waves breaking against "
          "the rocks, seagulls calling overhead.")


def build_graph(args, *, enabled: bool) -> dict:
    """Graf w formacie API. `enabled=False` daje ten sam graf ze sciezka gesta."""
    return {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": args.unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": args.clip, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": args.vae}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo",
                 "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": PROMPT,
                            "width": args.width, "height": args.height, "length": args.length}},
        "solattn": {"class_type": "SolAttnH3",
                    "inputs": {"model": ["unet", 0], "enabled": enabled,
                               "tau": args.tau, "thresh_type": args.thresh_type,
                               "first_dense_steps": args.first_dense_steps,
                               "first_dense_layers": args.first_dense_layers,
                               "sink_mode": args.sink_mode,
                               "correctness_gate": args.gate, "strict": args.strict}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        "guider": {"class_type": "BasicGuider",
                   "inputs": {"model": ["solattn", 0], "conditioning": ["cond", 0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": args.sampler}},
        "sigmas": {"class_type": "BasicScheduler",
                   "inputs": {"model": ["solattn", 0], "scheduler": "simple",
                              "steps": args.steps, "denoise": 1.0}},
        "sample": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0],
                              "sampler": ["sampler", 0], "sigmas": ["sigmas", 0],
                              "latent_image": ["cond", 1]}},
        "save": {"class_type": "SaveLatent",
                 "inputs": {"samples": ["sample", 0],
                            "filename_prefix": f"solattn_ab/{'on' if enabled else 'off'}"}},
    }


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def get(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def run_once(base: str, graph: dict, label: str, timeout: float) -> dict:
    client = str(uuid.uuid4())
    started = time.perf_counter()
    result = post(f"{base}/prompt", {"prompt": graph, "client_id": client})
    prompt_id = result["prompt_id"]
    print(f"[{label}] zakolejkowane, prompt_id={prompt_id}", flush=True)

    deadline = started + timeout
    while time.perf_counter() < deadline:
        history = get(f"{base}/history/{prompt_id}")
        entry = history.get(prompt_id)
        if entry and entry.get("status", {}).get("completed") is not None:
            elapsed = time.perf_counter() - started
            status = entry["status"]
            if not status.get("completed"):
                raise RuntimeError(f"[{label}] wykonanie nieudane: "
                                   f"{json.dumps(status)[:800]}")
            return {"label": label, "wall_s": elapsed, "prompt_id": prompt_id,
                    "outputs": entry.get("outputs", {})}
        time.sleep(1.0)
    raise TimeoutError(f"[{label}] przekroczono {timeout:.0f} s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--length", type=int, default=73)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler", default="res_multistep")
    parser.add_argument("--unet", default="minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    parser.add_argument("--clip", default="qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    parser.add_argument("--vae", default="minimax_h3_video_vae_fp16.safetensors")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--thresh-type", default="diag")
    parser.add_argument("--first-dense-steps", type=float, default=0.2)
    parser.add_argument("--first-dense-layers", type=int, default=2)
    parser.add_argument("--sink-mode", default="prefix")
    parser.add_argument("--gate", action="store_true", default=True)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--timeout", type=float, default=5400)
    parser.add_argument("--only", choices=["on", "off"], default=None,
                        help="uruchom tylko jeden wariant")
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    variants = [("off", False), ("on", True)]
    if args.only:
        variants = [v for v in variants if v[0] == args.only]

    results = []
    for label, enabled in variants:
        results.append(run_once(base, build_graph(args, enabled=enabled), label, args.timeout))
        print(f"[{label}] czas end-to-end: {results[-1]['wall_s']:.1f} s", flush=True)

    print("\n=== WYNIK ===")
    for row in results:
        print(f"{row['label']:>4}: {row['wall_s']:8.1f} s   {json.dumps(row['outputs'])[:200]}")
    if len(results) == 2:
        off, on = results[0]["wall_s"], results[1]["wall_s"]
        print(f"\nend-to-end: {off / on:.3f}x "
              f"({'szybciej' if on < off else 'wolniej'} z nodem)")
    print("\nCzas samej uwagi i statystyki routingu sa w logu ComfyUI "
          "(linie [sol-attn-h3]).")


if __name__ == "__main__":
    main()
