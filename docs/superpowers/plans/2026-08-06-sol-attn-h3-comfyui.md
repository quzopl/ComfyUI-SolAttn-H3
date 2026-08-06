# Sol-Attn dla MiniMax-H3 w ComfyUI — plan implementacji

> **Dla agentów:** WYMAGANY SUB-SKILL: `superpowers:executing-plans`. Kroki mają składnię checkboxów (`- [ ]`).

**Cel:** Custom node wpinający training-free rzadką uwagę Sol-Attn (NVIDIA Sol Engine) w natywny MiniMax-H3 w ComfyUI, z gęstym fallbackiem i twardą walidacją poprawności.

**Architektura:** Węzeł klonuje ModelPatcher i montuje trzy rzeczy na publicznych API ComfyUI: override uwagi (`transformer_options["optimized_attention_override"]`), wrapper `DIFFUSION_MODEL` odświeżający raz na krok layout/sink/numer kroku oraz `patches_replace["dit"]` stemplujący indeks bloku. Logika bez CUDA (`layout.py`, `state.py`) jest oddzielona od logiki z CUDA (`attention.py`, `kernel.py`), żeby dała się testować jednostkowo.

**Stack:** Python 3.13, torch 2.12+cu130, Triton 3.7, `sol-attn` 0.5.0 (Apache-2.0), ComfyUI v0.30.1, pytest.

## Ograniczenia globalne

- Kernel `sol_attn` wymaga: contiguous **BF16**, layout **BTHD** `[B, T, H, 128]`, **head_dim dokładnie 128**, CUDA, compute capability **≥ 8.0**. Naruszenie któregokolwiek → gęsty fallback z nazwanym powodem, nigdy wyjątek w ścieżce produkcyjnej.
- **Zero modyfikacji plików w `~/comfyX/ComfyUI/comfy/`.** Wpięcie wyłącznie przez publiczne API ModelPatchera.
- Każda odmowa ścieżki rzadkiej niesie **powód (string)**, jest liczona i logowana raz. Cicha degradacja do gęstej uwagi jest traktowana jako błąd.
- Wartości domyślne polityki pochodzą z `vendor/sana-sol-engine/models/minimax_h3/optimized/sol_attn_h3.py` i **nie wolno ich zmieniać bez pomiaru**.
- Progi gate'u poprawności: `max_abs ≤ 0.15` (T ≥ 32768) lub `≤ 0.08`, `mean_abs ≤ 0.002`, `rel_l2 ≤ 0.005`.
- Kod i komentarze po polsku bez znaków diakrytycznych w commitach; identyfikatory po angielsku.
- Repo: `~/sol`. Tożsamość gita lokalnie: `sol-attn-h3 <noreply@localhost>`. Nie używać danych osobowych autora.

## Struktura plików

| Plik | Odpowiedzialność | CUDA? |
|---|---|---|
| `__init__.py` | `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS` | nie |
| `kernel.py` | wykrycie architektury i backendu, leniwy import `sol_attn` | tylko probe |
| `layout.py` | zakres sinka z `PackedLayout.segments` | nie |
| `state.py` | zegar kroku/warstwy, powody odmowy, liczniki, gate | nie |
| `attention.py` | adapter override: kontrakt tensorów, kernel, gęsty fallback | tak |
| `nodes.py` | węzeł `SolAttnH3` — UI i montaż na ModelPatcherze | nie |
| `selftest.py` | diagnostyka do odpalenia na maszynie docelowej | tak |
| `tests/` | testy jednostkowe bez GPU | nie |
| `bench/ab_bench.py` | pomiar A/B z nodem i bez, przez API ComfyUI | tak |

---

### Task 1: `kernel.py` — wykrycie backendu

**Pliki:**
- Utwórz: `kernel.py`, `tests/test_kernel.py`, `pyproject.toml`, `README.md`

**Interfejsy:**
- Produkuje: `Probe` (dataclass: `arch: tuple[int,int]|None`, `cute_available: bool`, `backend: str`, `available: bool`, `error: str|None`), `probe(device=None) -> Probe`, `load_sol_attn() -> Callable`, `backend_for_arch(arch, cute_available) -> str`.

- [ ] **Krok 1: Test wyboru backendu per architektura**

```python
# tests/test_kernel.py
import pytest
from kernel import backend_for_arch

@pytest.mark.parametrize("arch,cute,expected", [
    ((9, 0), True, "cute_sm90"), ((10, 0), True, "cute_sm100"), ((12, 0), True, "cute_sm120"),
    ((9, 0), False, "triton"), ((12, 0), False, "triton"),   # brak CuTe -> Triton
    ((8, 9), True, "triton"), ((8, 6), True, "triton"), ((8, 0), True, "triton"),
])
def test_backend_for_arch(arch, cute, expected):
    assert backend_for_arch(arch, cute) == expected

def test_arch_below_sm80_unsupported():
    with pytest.raises(RuntimeError, match="8.0"):
        backend_for_arch((7, 5), True)
```

- [ ] **Krok 2: Uruchom, potwierdź FAIL** — `pytest tests/test_kernel.py -v`, oczekiwane `ModuleNotFoundError: kernel`.

- [ ] **Krok 3: Implementacja**

`backend_for_arch` deleguje do `sol_attn.interface._backend_for_arch`, jeśli pakiet jest dostępny, inaczej odtwarza jego tablicę (`{(9,0): "cute_sm90", (10,0): "cute_sm100", (12,0): "cute_sm120"}`, reszta ≥8.0 → `"triton"`, <8.0 → `RuntimeError`). Test ma przechodzić bez GPU, więc tablica lokalna jest ścieżką podstawową.

`probe()` czyta `torch.cuda.get_device_capability()`, sprawdza import `cutlass.cute` + `cuda.bindings.driver`, zwraca `Probe`. Wynik cache'owany (`functools.lru_cache`). Brak CUDA lub brak `sol_attn` → `available=False` z `error`.

`load_sol_attn()` robi `from sol_attn import sol_attn` i zwraca funkcję; `ImportError` propaguje.

- [ ] **Krok 4: Uruchom, potwierdź PASS.**

- [ ] **Krok 5: Commit** — `git add kernel.py tests/ pyproject.toml README.md && git commit -m "feat(kernel): wykrywanie architektury i backendu sol-attn"`

---

### Task 2: `layout.py` — zakres sinka

**Pliki:** Utwórz `layout.py`, `tests/test_layout.py`

**Interfejsy:**
- Produkuje: `SinkRange` (dataclass: `start: int`, `tokens: int`, `seq_len: int`, `video_start: int`), `sink_from_segments(segments, seq_len, sink_mode) -> SinkRange`, `SINK_MODES = ("prefix", "text")`.

Kontrakt wejścia odwzorowuje `PackedLayout.segments` z `comfy/ldm/minimax/model.py`: lista `(a, b, kind)`, kindy `text | cond | ref_img | audio | ref_audio | video`, segmenty ciągłe i posortowane, dokładnie jeden `video` (segment docelowy) — tak jak zakłada `model.py:634`.

- [ ] **Krok 1: Testy**

```python
# tests/test_layout.py
import pytest
from layout import SinkRange, sink_from_segments

T2VA = [(0, 537, "text"), (537, 951, "audio"), (951, 31000, "video")]
FL2VA = [(0, 537, "text"), (537, 800, "cond"), (800, 1200, "audio"), (1200, 31000, "video")]
REF2VA = [(0, 537, "text"), (537, 700, "ref_img"), (700, 900, "ref_audio"),
          (900, 1300, "audio"), (1300, 31000, "video")]

@pytest.mark.parametrize("segs,video_start", [(T2VA, 951), (FL2VA, 1200), (REF2VA, 1300)])
def test_prefix_sink_konczy_sie_na_ogonie_wideo(segs, video_start):
    s = sink_from_segments(segs, 31000, "prefix")
    assert (s.start, s.tokens, s.video_start) == (0, video_start, video_start)

@pytest.mark.parametrize("segs", [T2VA, FL2VA, REF2VA])
def test_text_sink_obejmuje_tylko_tekst(segs):
    s = sink_from_segments(segs, 31000, "text")
    assert (s.start, s.tokens) == (0, 537)

def test_brak_segmentu_video_to_blad():
    with pytest.raises(ValueError, match="video"):
        sink_from_segments([(0, 537, "text")], 537, "prefix")

def test_nieznany_tryb_sinka():
    with pytest.raises(ValueError, match="sink_mode"):
        sink_from_segments(T2VA, 31000, "cokolwiek")

def test_sink_nigdy_nie_wykracza_poza_sekwencje():
    s = sink_from_segments(T2VA, 31000, "prefix")
    assert 0 <= s.start and s.start + s.tokens <= s.seq_len
```

- [ ] **Krok 2: Uruchom, potwierdź FAIL.**

- [ ] **Krok 3: Implementacja** — `sink_from_segments` znajduje `video_start` jako `a` jedynego segmentu `kind == "video"` (brak → `ValueError`). Tryb `prefix` → `SinkRange(0, video_start, seq_len, video_start)`. Tryb `text` → segment `text`; jego brak → `SinkRange(0, 0, ...)` (sink pusty, nie błąd). Nieznany tryb → `ValueError`.

- [ ] **Krok 4: Uruchom, potwierdź PASS.**

- [ ] **Krok 5: Commit** — `git commit -m "feat(layout): zakres sinka z segmentow PackedLayout"`

---

### Task 3: `state.py` — zegar i powody odmowy

**Pliki:** Utwórz `state.py`, `tests/test_state.py`

**Interfejsy:**
- Produkuje: `Policy` (dataclass: `enabled: bool = True`, `tau: float = 1.0`, `thresh_type: str = "diag"`, `first_dense_steps: float = 0.2`, `first_dense_layers: int = 2`, `sink_mode: str = "prefix"`, `correctness_gate: bool = True`, `strict: bool = False`), `SolAttnState`, stałe powodów `DECLINE_*`, `resolve_step(sigmas, sample_sigmas) -> int|None`, `dense_step_count(first_dense_steps, total_steps) -> int`.
- `SolAttnState` API: `begin_run()`, `begin_forward(sink, step, total_steps)`, `next_block()`, `decline(**kwargs) -> str|None`, `note(reason)`, `note_sparse()`, `end_run()`, `stats() -> dict`.

- [ ] **Krok 1: Testy**

```python
# tests/test_state.py
import pytest
import torch
from layout import SinkRange
from state import (DECLINE_DENSE_LAYER, DECLINE_DTYPE, DECLINE_HEAD_DIM, DECLINE_NO_LAYOUT,
                   DECLINE_SEQ_MISMATCH, DECLINE_WARMUP, Policy, SolAttnState,
                   dense_step_count, resolve_step)

SINK = SinkRange(start=0, tokens=951, seq_len=31000, video_start=951)

def test_ulamek_skaluje_sie_z_dlugoscia_harmonogramu():
    assert dense_step_count(0.2, 50) == 10        # odpowiednik referencyjnych 10/50
    assert dense_step_count(0.2, 20) == 4
def test_wartosc_co_najmniej_1_jest_sztywna_liczba_krokow():
    assert dense_step_count(10, 50) == 10
    assert dense_step_count(10, 20) == 10

def test_numer_kroku_z_harmonogramu():
    sample = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    assert resolve_step(torch.tensor([0.5]), sample) == 2
    assert resolve_step(torch.tensor([1.0]), sample) == 0
def test_numer_kroku_gdy_sigma_spoza_harmonogramu():
    assert resolve_step(torch.tensor([0.31]), torch.tensor([1.0, 0.5, 0.0])) is None

def _state(**kw):
    s = SolAttnState(Policy(first_dense_steps=0.2, first_dense_layers=2, **kw))
    s.begin_run()
    return s

def _call(s, *, rows=31000, dtype=torch.bfloat16, head_dim=128, mask=None, block=5):
    return s.decline(rows=rows, dtype=dtype, head_dim=head_dim, mask=mask, block_index=block)

def test_kroki_rozgrzewkowe_odmawiaja_reszta_nie():
    s = _state()
    s.begin_forward(SINK, step=3, total_steps=50)     # 3 < 10
    assert _call(s) == DECLINE_WARMUP
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s) is None

def test_dense_layers_liczone_od_zera():
    """Referencja miala tu off-by-one: dense_layers=2 zostawialo gesty tylko blok 0."""
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s, block=0) == DECLINE_DENSE_LAYER
    assert _call(s, block=1) == DECLINE_DENSE_LAYER
    assert _call(s, block=2) is None

def test_kontrakt_kernela():
    s = _state(); s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s, dtype=torch.float16) == DECLINE_DTYPE
    assert _call(s, head_dim=64) == DECLINE_HEAD_DIM
    assert _call(s, rows=537) == DECLINE_SEQ_MISMATCH      # token_refiner
def test_brak_layoutu():
    s = _state()
    assert _call(s) == DECLINE_NO_LAYOUT                   # bez begin_forward

def test_pominiete_kroki_nie_psuja_zegara():
    """Cache pomija forwardy; numer kroku pochodzi z harmonogramu, nie z licznika."""
    s = _state()
    for step in (10, 13, 14, 19):
        s.begin_forward(SINK, step=step, total_steps=50)
        assert _call(s) is None
    assert s.stats()["last_step"] == 19

def test_licznik_blokow_zeruje_sie_co_forward():
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    assert [s.next_block() for _ in range(3)] == [0, 1, 2]
    s.begin_forward(SINK, step=11, total_steps=50)
    assert s.next_block() == 0

def test_przebieg_bez_ani_jednego_wywolania_rzadkiego_jest_bledem():
    """Powod, ktory robi szkode (warmup_step), jest sam w sobie legalny.
    Blad istnieje tylko w agregacie, wiec i sprawdzenie musi byc w agregacie."""
    s = SolAttnState(Policy(strict=True)); s.begin_run()
    for step in range(50):
        s.begin_forward(SINK, step=step, total_steps=50)
        s.note(DECLINE_WARMUP)
    with pytest.raises(RuntimeError, match="ani jednego"):
        s.end_run()

def test_przebieg_z_wywolaniami_rzadkimi_przechodzi():
    s = SolAttnState(Policy(strict=True)); s.begin_run()
    s.begin_forward(SINK, step=20, total_steps=50); s.note_sparse()
    s.end_run()
```

- [ ] **Krok 2: Uruchom, potwierdź FAIL.**

- [ ] **Krok 3: Implementacja**

`dense_step_count(v, total)`: `int(round(v * total))` dla `v < 1`, inaczej `int(v)`.

`resolve_step(sigmas, sample_sigmas)`: `torch.isclose(sample_sigmas, sigmas[0], rtol=1e-4)` → indeks pierwszego dopasowania, brak → `None` (wzorzec z `context_windows.py:558`).

`decline()` sprawdza w kolejności od najtańszego: `disabled` → `kernel_unavailable` → `oom` → `mask_present` → `layout_unknown` (gdy `skip_reshape` nie jest `True`) → `batch` (gdy `batch != 1`; H3 i tak wymusza 1 w `model.py:509`) → `dtype` → `head_dim` → `no_layout` → `seq_mismatch` → `warmup_step` → `dense_layer`. Zwraca stałą lub `None`. Sygnatura przyjmuje wszystkie te argumenty jako keyword-only z wartościami domyślnymi, żeby Task 4 mógł wołać nadzbiorem.

`note(reason)` inkrementuje `declined[reason]`; powody spoza `{warmup_step, dense_layer, disabled}` logowane raz i — przy `strict` — podnoszą `RuntimeError`.

`end_run()`: jeśli `total_steps` przekroczyło `dense_step_count` i `sparse_calls == 0` → `RuntimeError` przy `strict`, inaczej głośny `print` z rozbiciem `declined`.

- [ ] **Krok 4: Uruchom, potwierdź PASS.**

- [ ] **Krok 5: Commit** — `git commit -m "feat(state): zegar kroku z harmonogramu i powody odmowy"`

---

### Task 4: `attention.py` — adapter override

**Pliki:** Utwórz `attention.py`, `tests/test_attention_gpu.py` (oznaczone `@pytest.mark.gpu`)

**Interfejsy:**
- Konsumuje: `kernel.load_sol_attn`, `kernel.probe`, `layout.SinkRange`, `state.SolAttnState`, `state.Policy`.
- Produkuje: `make_override(state, policy) -> Callable`, `dense_bthd(q, k, v) -> Tensor`, `run_gate(sol_attn, q, k, v, thresh_type) -> dict`, `route_density(...) -> dict`.

Kontrakt wywołania: `wrap_attn` (`comfy/ldm/modules/attention.py:148`) woła `override(func, q, k, v, heads, mask=None, skip_reshape=True, transformer_options=..., _inside_attn_wrapper=True)`. Wejście BHSD `(1, H, S, 128)`, wyjście `(1, S, H*128)` gdy `skip_output_reshape` jest fałszywe.

- [ ] **Krok 1: Test GPU — równoważność z gęstą uwagą przy tau przepuszczającym wszystko**

```python
# tests/test_attention_gpu.py
import pytest, torch
from attention import dense_bthd, make_override
from layout import SinkRange
from state import Policy, SolAttnState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="wymaga CUDA")
H, D, T = 8, 128, 4096

def _bhsd():
    packed = torch.randn(T, 3 * H * D, device="cuda", dtype=torch.bfloat16) * 0.5
    return [x.view(T, H, D).transpose(0, 1).unsqueeze(0) for x in packed.split(H * D, dim=-1)]

def _fake_func(q, k, v, heads, **kw):
    return dense_bthd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).reshape(1, T, heads * D)

def test_ksztalt_i_dtype_wyjscia():
    q, k, v = _bhsd()
    st = SolAttnState(Policy(first_dense_steps=0, first_dense_layers=0, correctness_gate=False))
    st.begin_run(); st.begin_forward(SinkRange(0, 512, T, 512), step=5, total_steps=50)
    out = make_override(st, st.policy)(_fake_func, q, k, v, H, mask=None, skip_reshape=True,
                                       transformer_options={"solattn_block": 4})
    assert out.shape == (1, T, H * D) and out.dtype == torch.bfloat16

def test_odmowa_zwraca_dokladnie_wynik_gestej_sciezki():
    q, k, v = _bhsd()
    st = SolAttnState(Policy(enabled=False)); st.begin_run()
    out = make_override(st, st.policy)(_fake_func, q, k, v, H, mask=None, skip_reshape=True,
                                       transformer_options={})
    torch.testing.assert_close(out, _fake_func(q, k, v, H))

def test_wiersze_prefiksu_sa_gesto_przeliczone():
    """Sink czyni prefiks dokladnym jako K/V, ale jego wlasne zapytania musza byc geste."""
    q, k, v = _bhsd()
    sink = SinkRange(0, 512, T, 512)
    st = SolAttnState(Policy(first_dense_steps=0, first_dense_layers=0, correctness_gate=False))
    st.begin_run(); st.begin_forward(sink, step=5, total_steps=50)
    out = make_override(st, st.policy)(_fake_func, q, k, v, H, mask=None, skip_reshape=True,
                                       transformer_options={"solattn_block": 4})
    want = dense_bthd(q.transpose(1, 2)[:, :512], k.transpose(1, 2), v.transpose(1, 2))
    torch.testing.assert_close(out[:, :512].reshape(1, 512, H, D), want, rtol=2e-2, atol=2e-2)
```

- [ ] **Krok 2: Uruchom, potwierdź FAIL.**

- [ ] **Krok 3: Implementacja**

```python
def make_override(state, policy):
    def override(func, q, k, v, heads, *args, **kwargs):
        def dense():
            return func(q, k, v, heads, *args, **kwargs)
        block = kwargs.get("transformer_options", {}).get("solattn_block")
        if block is None:
            block = state.next_block()          # zapasowy licznik, gdy stempel niedostepny
        reason = state.decline(rows=q.shape[2], dtype=q.dtype, head_dim=q.shape[-1],
                               mask=kwargs.get("mask"), block_index=block,
                               skip_reshape=kwargs.get("skip_reshape"), batch=q.shape[0])
        if reason is not None:
            state.note(reason)
            return dense()
        try:
            qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
            if policy.correctness_gate:
                state.gate_once(qb, kb, vb)     # tau=-1000 vs SDPA, raz na ksztalt
            sink = state.sink
            out = state.sol_attn(qb, kb, vb, tau=policy.tau, thresh_type=policy.thresh_type,
                                 kv_splits=1, sink_start=sink.start, sink_tokens=sink.tokens)
            if sink.tokens:
                lo, hi = sink.start, sink.start + sink.tokens
                out[:, lo:hi] = dense_bthd(qb[:, lo:hi], kb, vb)
        except torch.OutOfMemoryError:
            state.latch_oom(); return dense()
        state.note_sparse()
        if kwargs.get("skip_output_reshape"):
            return out.transpose(1, 2)
        return out.reshape(out.shape[0], out.shape[1], heads * out.shape[3])
    return override
```

`dense_bthd(q, k, v)` — SDPA na BTHD, ten sam layout na wyjściu (transpozycja do BHSD, `scaled_dot_product_attention`, transpozycja z powrotem). `run_gate` i `route_density` przeniesione 1:1 z `spike_solattn.py`, który je już zwalidował.

- [ ] **Krok 4: Uruchom, potwierdź PASS** — `pytest tests/test_attention_gpu.py -v`

- [ ] **Krok 5: Commit** — `git commit -m "feat(attention): adapter override z sinkiem i gestym fallbackiem"`

---

### Task 5: `nodes.py` — węzeł i montaż

**Pliki:** Utwórz `nodes.py`, `__init__.py`

**Interfejsy:**
- Produkuje: klasa `SolAttnH3` z `INPUT_TYPES` / `RETURN_TYPES = ("MODEL",)` / `FUNCTION = "patch"` / `CATEGORY = "model_patches/attention"`.

Wejścia: `model` (MODEL), `enabled` (BOOLEAN, `True`), `tau` (FLOAT, `1.0`, −1000..10, krok 0.05), `thresh_type` (`["diag", "exact"]`), `first_dense_steps` (FLOAT, `0.2`, 0..50), `first_dense_layers` (INT, `2`, 0..50), `sink_mode` (`["prefix", "text"]`), `correctness_gate` (BOOLEAN, `True`), `strict` (BOOLEAN, `False`).

- [ ] **Krok 1: Montaż na klonie ModelPatchera**

```python
def patch(self, model, enabled, tau, thresh_type, first_dense_steps,
          first_dense_layers, sink_mode, correctness_gate, strict):
    policy = Policy(enabled=enabled, tau=tau, thresh_type=thresh_type,
                    first_dense_steps=first_dense_steps, first_dense_layers=first_dense_layers,
                    sink_mode=sink_mode, correctness_gate=correctness_gate, strict=strict)
    state = SolAttnState(policy)
    m = model.clone()
    m.model_options["transformer_options"]["optimized_attention_override"] = make_override(state, policy)
    m.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, KEY, make_run_wrapper(state))
    m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, KEY, make_forward_wrapper(state, policy))
    for i in range(_block_count(m)):
        m.set_model_patch_replace(_make_stamp(i), "dit", "double_block", i)
    return (m,)
```

- [ ] **Krok 2: Wrapper forwardu**

Raz na forward: odczytuje `minimax_payload["layout"]` (gdy brak — odtwarza `PackedLayout` dokładnie jak `model.py:520-524`), liczy sink przez `sink_from_segments`, ustala numer kroku przez `resolve_step(transformer_options["sigmas"], transformer_options["sample_sigmas"])`, woła `state.begin_forward(...)`. Brak layoutu → `state.begin_forward(None, ...)`, co daje powód `no_layout`.

- [ ] **Krok 3: Stempel indeksu bloku**

```python
def _make_stamp(index):
    def stamp(args, extra):
        args["transformer_options"]["solattn_block"] = index
        return extra["original_block"](args)
    return stamp
```

- [ ] **Krok 4: Wrapper przebiegu** — `OUTER_SAMPLE`: `state.begin_run()`, `try: executor(...) finally: state.end_run()` plus wypis `state.stats()`.

- [ ] **Krok 5: Test dymny** — `python -c "import nodes"` w venvie comfyX, następnie start ComfyUI i sprawdzenie, że węzeł pojawia się w `object_info`:

```bash
curl -s localhost:8188/object_info/SolAttnH3 | head -c 400
```
Oczekiwane: JSON z definicją węzła, nie `{}`.

- [ ] **Krok 6: Commit** — `git commit -m "feat(nodes): wezel SolAttnH3 i montaz na ModelPatcherze"`

---

### Task 6: Integracja na prawdziwym H3

**Pliki:** Utwórz `bench/h3_workflow.json`, `bench/ab_bench.py`

- [ ] **Krok 1: Symlink do custom_nodes**

```bash
ln -sfn ~/sol ~/comfyX/ComfyUI/custom_nodes/ComfyUI-SolAttn-H3
```

- [ ] **Krok 2: Minimalny graf H3** — mała rozdzielczość (np. 480×288, 49 klatek), `qwen3vl_4b_fp8` jako encoder, stały seed, mała liczba kroków. Zapisany jako format API (`/prompt`).

- [ ] **Krok 3: Przebieg z `strict=True`** — dowody do zebrania z logu:
  - `gate PASS` na prawdziwych QKV
  - `sparse_calls > 0` w mierzonym przebiegu
  - w `declined` wyłącznie `warmup_step`, `dense_layer` i `seq_mismatch` (token_refiner)
  - `effective_density` w paśmie (0, 1) — nie 1,0 (routing nie routuje) i nie 0,0 (routing się zapadł)

- [ ] **Krok 4: A/B na tym samym seedzie** — `bench/ab_bench.py` kolejkuje ten sam graf raz z `enabled=False`, raz z `True`, mierzy czas przez `/history` i porównuje zdekodowane klatki (PSNR + max abs). Odchylenie jest oczekiwane — Sol-Attn to aproksymacja — ale trajektoria ma pozostać rozpoznawalnie ta sama.

- [ ] **Krok 5: Przebieg z włączonym `MiniMaxH3-Cache`** — potwierdzenie, że pominięte kroki nie psują zegara (`last_step` zgodny z harmonogramem, `sparse_calls > 0`).

- [ ] **Krok 6: Commit** — `git commit -m "test: integracja i pomiar A/B na prawdziwym H3"`

---

### Task 7: `selftest.py`, README, atrybucja

**Pliki:** Utwórz `selftest.py`, uzupełnij `README.md`, `NOTICE`

- [ ] **Krok 1: `selftest.py`** — samodzielny skrypt dla maszyny docelowej: wypisuje GPU, compute capability, dostępność CuTe, wybrany backend, wynik gate'u i gęstość na kształtach zadanych z linii poleceń. Baza: zwalidowany `spike_solattn.py`.

- [ ] **Krok 2: README** — instalacja (`uv pip install -e vendor/sana-sol-engine/techniques/sparse_backends`), tabela parametrów z uzasadnieniami, zmierzone liczby z tej maszyny z jawną etykietą „SM89/Triton, nie SM120/CuTe", znane ograniczenia (graph-break przy `torch.compile`, tylko MiniMax-H3, head_dim 128, bf16).

- [ ] **Krok 3: NOTICE** — atrybucja Apache-2.0: NVlabs/Sana, paper arXiv 2607.24027, wskazanie że kernel jest instalowany, nie vendorowany.

- [ ] **Krok 4: Pełny zestaw testów** — `pytest tests/ -v` bez GPU i z GPU.

- [ ] **Krok 5: Commit** — `git commit -m "docs: selftest, README i atrybucja"`

---

## Self-review planu

**Pokrycie specyfikacji:** §4 kontrakty → Task 3 (`decline`) i Task 4. §5 punkty wpięcia → Task 5. §6 architektura → Taski 1–5, po pliku na task. §7 polityka → Task 3 (`Policy`, `dense_step_count`) i Task 2 (`sink_mode`). §8 obsługa błędów → Task 3 (powody, `end_run`) i Task 4 (gate, gęstość, `oom`). §9 walidacja → Task 6. §10 ryzyka → Task 6 krok 5 (kompozycja z cache), Task 7 krok 2 (etykieta backendu). §11 atrybucja → Task 7 krok 3. Bez luk.

**Skan placeholderów:** brak TBD/TODO; każdy krok kodowy ma blok kodu lub konkretną komendę z oczekiwanym wynikiem.

**Spójność typów:** `SinkRange(start, tokens, seq_len, video_start)` używane identycznie w Taskach 2–5. `Policy` z Taska 3 konsumowane w Taskach 4–5 z tym samym zestawem pól. `state.decline()` przyjmuje pełny zestaw argumentów (`rows`, `dtype`, `head_dim`, `mask`, `block_index`, `skip_reshape`, `batch`) jako keyword-only z domyślnymi — spójne między Taskiem 3 a 4.
