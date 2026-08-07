# Workflows

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

**On SM89 against SageAttention, Sol-Attn is currently a net loss at `tau=1.0`;
see the main README.** Stack it under Spectrum only once it is a win on its own.
