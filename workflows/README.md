# Workflows

## `minimax-h3-i2v-solattn-sage`

MiniMax-H3 **image-to-video**, Sol-Attn alone, with SageAttention underneath it:

```
UNETLoader → SolAttnH3 → BasicGuider / BasicScheduler
```

| File | Use |
|---|---|
| `minimax-h3-i2v-solattn-sage.json` | drag into the ComfyUI canvas |
| `minimax-h3-i2v-solattn-sage.api.json` | POST to `/prompt` |

### SageAttention is a launch flag, not a node

```bash
python main.py --use-sage-attention
```

There is no Sage node in this graph and there should not be. ComfyUI's `wrap_attn`
calls `optimized_attention_override(func, …)` (`comfy/ldm/modules/attention.py:193`)
where `func` is whatever backend the process was started with — so with the flag
set, SageAttention *is* what Sol-Attn falls back to when it declines a call, and
what recomputes the sink's query rows. Without the flag you get SDPA, which is
roughly 2.5× slower at these shapes, and every measurement in the main README
stops applying.

### Do not add `Patch Sage Attention KJ`

It is the obvious node to reach for, and it silently breaks this graph.
KJNodes' `PathchSageAttentionKJ` assigns
`model_options["transformer_options"]["optimized_attention_override"]` — the same
key `SolAttnH3` uses. The two overwrite each other, last node in the chain wins,
and the loser does nothing at all: no warning, no log line, just the speedup
quietly gone. Use the launch flag instead; it composes, the node does not.

### Before running

1. Point `LoadImage` at your own first frame — the shipped value is a placeholder.
2. The loaders ship with the NVFP4 filenames; switch them in the dropdowns if you
   run int8. Nothing else in the graph depends on that choice — attention time is
   identical either way (measured: dense 29.69–29.94 ms per call across both).
3. `width`/`height` are independent of the input image. Match your frame's aspect
   ratio or the result is stretched.
4. `length` snaps **up** to the model's 17k+5 grid: 124 (~5 s), 141, 158, …,
   362 (~15 s). Off-grid values are rounded silently — 125 gives you 141 frames.

### It ships at 362 frames on purpose

RTX 5080, real i2v with a first frame, NVFP4 weights, 8 steps, `res_multistep`,
SageAttention baseline, measurement pass after warm-up, nothing else on the GPU:

| Frames | Sequence | Node off | `sink_mode=prefix` | `sink_mode=text` |
|---:|---:|---:|---:|---:|
| 124 | 16 421 | 35.2 s | 39.0 s (**0.90×**) | 34.2 s (1.03×) |
| **362** | 45 563 | 150.2 s | **137.1 s (1.10×)** | 138.3 s (1.09×) |

**A first frame makes the sink 1436 rows, and the sink is recomputed densely on
every sparse call.** At 124 frames that is 8.7 % of the sequence and the node
cannot earn it back — it costs you 10 %. At 362 frames the same sink is 4.9 %
while attention has grown quadratically, and the node pays.

Note the ordering flips with it. At 124 frames the only way to win is
`sink_mode=text`, which stops keeping the reference-image rows exact — the rows
that hold identity in i2v. At 362 frames `prefix` is not just viable, it is
*faster* than `text` (137.1 vs 138.3 s), so the quality-preserving default is
also the quick one. There is no trade to make at length.

If you drop to 124 to iterate on a prompt, set `enabled` to false or switch to
`sink_mode=text`, then switch back for the final render.

15 s at 864×480 fits in 16 GB — 150.2 s with the node off, 137.1 s with it on.

### The prompt is not prose

H3 was trained on the structured output of H3-Context-IR, and ComfyUI passes your
string through untouched. The shipped prompt keeps that structure —
`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music` —
and it is worth editing rather than replacing. Dialogue has to be written out
verbatim inside `<d>…</d>`; saying *that* someone speaks gives you correct mouth
shapes with no words. See MiniMax's
[prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).

### Confirm it is actually running

Once per run, the console should print:

```
[sol-attn-h3] SM120, CuTe DSL available, backend=cute_sm120, DiT blocks: 50
[sol-attn-h3] correctness gate PASS max_rel=0.00385 … 
[sol-attn-h3] routing density {… 'effective_density': 0.258}
[sol-attn-h3] {… 'sparse_calls': 288, 'dense_calls': 112, 'attn_ms_per_call': …}
```

`backend=triton` on an SM89+ card means the CuTe runtime is missing. `sparse_calls: 0`
means every call declined — the named reason is in `declined`.

## `minimax-h3-ref2va-solattn-sage`

MiniMax-H3 **reference-to-video** — up to 9 reference images drive the identity,
rather than a first frame driving the geometry. Same chain as the i2v graph:

```
UNETLoader → SolAttnH3 → BasicGuider / BasicScheduler
```

| File | Use |
|---|---|
| `minimax-h3-ref2va-solattn-sage.json` | drag into the ComfyUI canvas |
| `minimax-h3-ref2va-solattn-sage.api.json` | POST to `/prompt` |

The SageAttention and `Patch Sage Attention KJ` notes from the i2v section apply
here unchanged — the flag is what wires Sage under Sol-Attn, and the KJ node
silently disables one of the two.

### What differs from the i2v graph

`MiniMaxH3ReferenceToVideo` takes a third link the i2v node does not: **`audio_vae`**.
The audio VAE therefore feeds both the conditioning node and the decode, and a
graph missing that wire will not run.

References arrive on Autogrow sockets. The canonical input id carries the group
name — `ref_images.ref_image_0`, not `ref_image_0` — because
`finalize_prefix` (`comfy_api/latest/_io.py:1074`) joins the group and the
per-slot name with a dot, and both the UI and the API format use that id. In the
UI graph each one also needs `"label"` (what the node draws) and `"shape": 7`
(optional socket); emit the bare name instead and the frontend treats it as a
stray socket and adds its own template one beside it, so the node shows
**two** `ref_image_0`.

**Only one free socket per group is shown at a time.** The graph ships three
images connected plus one empty `ref_image_3`, and a single empty slot for each
of `ref_video_0`, `ref_video_audio_0` and `ref_audio_0`. `ref_audio_1` and
`ref_audio_2` are not missing — Autogrow reveals the next slot as you fill the
current one. The node accepts nine images, three reference videos, three
soundtracks for those videos, and three standalone audios.

**Connect them in the order your `<Picture N>` tags use** — `<Picture 1>` is
`ref_image_0`. The tokenizer presents references in connection order, so a
mismatch silently points your prompt at the wrong image. Three or four varied
shots hold identity far better than one.

`ref_image_size` is `match` here, which scales each reference to the
generation's pixel area. `max` uses the reference pipeline's 2048 px short edge
for the best identity fidelity and is several times slower — reference tokens
ride through **every** sampling step, not just the first.

### It ships at 124 frames, and that is the length where the node did not pay

Read this before assuming the node is helping. On the **i2v** sibling, measured
on an RTX 5080 with NVFP4 weights at 8 steps:

| Frames | Node off | Node on |
|---:|---:|---:|
| 124 | 35.2 s | 39.0 s (**0.90×**) |
| 362 | 150.2 s | 137.1 s (**1.10×**) |

The sink is recomputed densely on every sparse call, so a short sequence cannot
earn it back. Reference images make the sink **larger** than a single first
frame, so ref2va should need the longer length even more.

**That last sentence is reasoning from the i2v measurement, not a ref2va
measurement — this graph has not been timed.** 124 is the default here only
because it is the safe choice for VRAM: 362 frames plus reference tokens on a
16 GB card is untested and may not fit. Raise the length for real renders, and
while iterating at 124 either set `enabled` to false or measure it yourself.

## `minimax-h3-i2v-solattn-spectrum`

MiniMax-H3 **image-to-video** with both accelerators chained:

```
UNETLoader → SolAttnH3 → SpectrumApplyMiniMaxH3 → BasicGuider / BasicScheduler
```

Two files, same graph:

| File | Use |
|---|---|
| `minimax-h3-i2v-solattn-spectrum.json` | drag into the ComfyUI canvas |
| `minimax-h3-i2v-solattn-spectrum.api.json` | POST to `/prompt` |

### Before running

1. Set `LoadImage` to your own first frame — the shipped value is a placeholder.
2. Check the model filenames against your `models/` directory. The graph assumes
   `minimax_h3_fl2va_pruned_int8_convrot`, `qwen3vl_32b_minimax_h3_int8_convrot`,
   and the fp16 video / fp32 audio VAEs.
3. `width`/`height` are set independently of the input image; match them to your
   frame's aspect ratio or the result will be stretched.

### Why the two nodes compose

They occupy different levels: **Spectrum skips whole forwards** (spectral
forecasting of post-transformer features), **Sol-Attn speeds up the forwards that
do run** (sparse attention inside each block).

The one place they could have collided is `patches_replace["dit"]` — Sol-Attn
stamps the block index on all 50 blocks, Spectrum replaces the last one. Spectrum
handles this correctly (`minimax_h3.py:285`): it reads any existing replacement
and calls it rather than overwriting. Node order in the graph does not matter,
because Spectrum resolves this per forward rather than at mount time.

Verified on a real run, 20 steps, 6 034-row sequence:

```
sparse_calls: 480   dense_calls: 220   last_step: 19   total_steps: 20
```

700 attention calls instead of 1 000 — Spectrum forecast 6 of 20 forwards — while
Sol-Attn's step clock stayed correct throughout. The arithmetic is exact:
4 warm-up forwards × 50 blocks + 10 forwards × (48 sparse + 2 dense).

### Savings compete, they do not add up

Spectrum removes forwards; Sol-Attn accelerates what is left. The more Spectrum
skips, the less there is for Sol-Attn to accelerate. Both are approximations with
different failure modes — Spectrum diverges on fast motion, Sol-Attn has a
floor of roughly 31 dB from the kernel swap alone — so enable them one at a time
first and judge the quality cost separately.

**On the Triton backend, against SageAttention, Sol-Attn is a net loss at
`tau=1.0`** — that is what you get on any card when the CuTe DSL runtime is not
installed. On `cute_sm89` and `cute_sm120` it wins; see the main README for the
per-card numbers. Stack it under Spectrum only once it is a win on its own.
