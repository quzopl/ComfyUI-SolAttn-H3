# Sol-Attn dla MiniMax-H3 w ComfyUI — design

Data: 2026-08-06
Status: zatwierdzony do rozpisania planu implementacji

## 1. Cel

Wpiąć Sol-Attn — training-free rzadką uwagę z NVIDIA Sol Engine — w natywną
implementację MiniMax-H3 w ComfyUI, jako opcjonalny custom node.

Sol Engine składa pięć technik akceleracji. Dla H3 zwalidowana linia to
context parallelism + fuzje kerneli + Sol-Attn + FirstBlockCache. Z tego
zestawu w tej instalacji brakuje wyłącznie Sol-Attn:

| Filar Sol Engine | Stan w `~/comfyX/ComfyUI` |
|---|---|
| Cross-step cache | `ComfyUI-MiniMaxH3-Cache`, alternatywnie `ComfyUI-Spectrum-MiniMax-H3` |
| Kwantyzacja / fuzje kerneli | `int8-fast` + wagi `minimax_h3_*_pruned_int8_convrot` |
| **Rzadka uwaga (Sol-Attn)** | **brak — zakres tego dokumentu** |
| Context parallelism | nie dotyczy (pojedyncze GPU) |

## 2. Zakres

W zakresie:

- jeden custom node `SolAttnH3` (MODEL → MODEL) wpinający Sol-Attn w H3
- automatyczny wybór backendu kernela wg architektury GPU
- degradacja do gęstej uwagi z jawnym, liczonym powodem
- gate poprawności i sonda gęstości routingu
- skrypt selftestu do uruchomienia na maszynie docelowej

Poza zakresem:

- inne modele wideo (WAN, HunyuanVideo, LTX) — Sol-Engine ma dla nich
  zwalidowane linie, ale każdy ma inny layout sekwencji i inne miejsce na sink
- własna implementacja cache lub fuzji kerneli (już pokryte)
- autotuning `tau`
- współpraca z `torch.compile` (override to callable w `transformer_options`,
  więc wystąpi graph-break — do odnotowania w README, nie do naprawy w v1)
- publikacja node'a (możliwa później; v1 celuje w działanie u autora)

## 3. Środowisko

Maszyna deweloperska (build + testy poprawnościowe):

- ComfyUI v0.30.1 w `~/comfyX/ComfyUI`, venv z Pythonem 3.13
- torch 2.12.0+cu130, Triton 3.7.0, obecne `sageattention` i `flash_attn`
- RTX 4070 Ti 12 GB — **SM89 → backend Triton**
- 62 GB RAM, wagi H3: DiT int8 20 GB, `qwen3vl_4b_fp8` 4,9 GB jako lekki encoder

Maszyna docelowa (generowanie):

- RTX PRO 6000 Blackwell — **SM120 → backend CuTe DSL**
- node musi degradować się poprawnie na słabsze karty konsumenckie
  (Ada SM89, Ampere SM86/SM80 → Triton)

Wymagania kernela `sol-attn` 0.5.0 są spełnione na obu: Python ≥3.10,
torch ≥2.10, CUDA ≥12.8, Triton ≥3.6.

## 4. Zgodność kontraktów

Kontrakt kernela (`sol_attn/interface.py::_validate_inputs`) kontra to, co
podaje ComfyUI w `comfy/ldm/minimax/model.py`:

| Wymóg kernela | H3 w ComfyUI | Wynik |
|---|---|---|
| head_dim == 128 | `attention_head_dim=128` (`model.py:414`) | zgodne |
| dtype bfloat16 | `supported_inference_dtypes = [bfloat16, float32]` (`supported_models.py:973`) | zgodne |
| contiguous BTHD | podawane BHSD, wymaga jednej kopii | kopia konieczna |
| brak maski | `mask=None` (`model.py:181`) | zgodne |
| compute capability ≥ 8.0 | zależne od maszyny | sprawdzane w runtime |

Kopia BTHD: przy ~31 tys. wierszy to 3 × 444 MB. Referencja NVIDII mierzy
analogiczną kopię na <0,1 ms przeciw 6,7 ms uwagi.

## 5. Punkty wpięcia

Wszystkie wspierane — bez patchowania plików core ComfyUI.

| Mechanizm | Miejsce | Rola |
|---|---|---|
| `transformer_options["optimized_attention_override"]` | `comfy/ldm/modules/attention.py:148` (`wrap_attn`) | przechwycenie uwagi; `func(*args, **kwargs)` daje gęsty fallback |
| `WrappersMP.DIFFUSION_MODEL` | `comfy/ldm/minimax/model.py:502` | raz na forward: layout, sink, numer kroku, reset warstw |
| `patches_replace["dit"][("double_block", i)]` | `comfy/ldm/minimax/model.py:620` | stemplowanie prawdziwego indeksu bloku |

API ModelPatcher: `clone()`, `add_wrapper_with_key()`, `set_model_patch_replace(patch, "dit", "double_block", i)`.

## 6. Architektura

```
~/sol/                          # repo, symlink → custom_nodes/ComfyUI-SolAttn-H3
├── __init__.py    # NODE_CLASS_MAPPINGS dla ComfyUI
├── nodes.py       # węzeł SolAttnH3 — wyłącznie UI i montaż
├── state.py       # zegar kroku/warstwy, powody odmowy, liczniki
├── layout.py      # zakres sinka z layout.segments (czysta funkcja)
├── kernel.py      # leniwy import sol_attn, wykrycie i raport backendu
├── attention.py   # adapter override: kontrakt, kernel, dense fallback
├── selftest.py    # samodzielny skrypt diagnostyczny (maszyna docelowa)
└── tests/         # testy jednostkowe bez GPU
```

Podział podyktowany testowalnością: `layout.py` i `state.py` nie dotykają CUDA
i idą pod testy jednostkowe. `attention.py` i `kernel.py` wymagają GPU.

### Przepływ jednego kroku samplera

1. Wrapper `DIFFUSION_MODEL` — raz na forward:
   - czyta `minimax_payload["layout"]`
   - sink = `(0, start segmentu "video")` dla `sink_mode="prefix"`
   - numer kroku = pozycja `timestep` w `transformer_options["sample_sigmas"]`
   - zeruje licznik warstw
2. `patches_replace["dit"][("double_block", i)]` — 50×: stempluje
   `transformer_options["solattn_block"] = i`, woła oryginalny blok
3. `Attention.forward` → `optimized_attention` → `wrap_attn` → adapter:
   - odmowa → `func(*args, **kwargs)`
   - zgoda → BHSD→BTHD contiguous bf16 → `sol_attn(...)` z sinkiem →
     gęste przeliczenie wierszy-zapytań prefiksu → `reshape(1, s, heads*128)`

### Odejścia od referencji NVIDII

Oba wynikają z tego, że ComfyUI udostępnia informacje, których nie miał
runtime SGLang użyty w `models/minimax_h3/optimized/sol_attn_h3.py`.

**Numer kroku z harmonogramu, nie z kierunku timestepu.** Referencja wykrywa
początek requestu po odwróceniu kierunku zmian timestepu i dokumentuje dwie
wpadki na tym mechanizmie: reset, który nie odpalał wcale (pomiar zaczynał się
od kroku 49 i biegł w pełni rzadko, raportując dziesięć kroków gęstych), oraz
reset odpalający co krok (wszystko odmawiało jako `warmup_step`, obie
konfiguracje mierzyły gęstą uwagę pod etykietą rzadkiej). `sample_sigmas` daje
numer kroku wprost.

**Sink z jawnych segmentów.** Referencja wnioskuje początek ogona wideo z
nieciągłości `video_indices`. `PackedLayout.segments` podaje etykietowane
`(a, b, kind)` dla `text / cond / ref_img / audio / video`.

**Indeks bloku stemplowany, nie liczony.** `token_refiner` (`model.py:584`)
również woła `Attention` z head_dim 128, na samych wierszach tekstu. Liczenie
wywołań przesunęłoby `first_dense_layers`. Refiner odpada też na sprawdzeniu
długości sekwencji, ale poprawność nie ma zależeć od dwóch mechanizmów naraz.

## 7. Polityka sparse

Wartości domyślne z zwalidowanej linii H3:

| Parametr | Domyślnie | Uzasadnienie |
|---|---|---|
| `enabled` | włączony | jeden przełącznik wyłączający całość bez przebudowy grafu |
| `tau` | 1.0 | referencja przekazuje wprost; jej wcześniejsza kalibracja per-kształt zwracała pusty zbiór tras na H3 i konfiguracja po cichu leciała gęsto |
| `thresh_type` | `diag` | `exact` = próg pełnokowariancyjny, droższy |
| `first_dense_steps` | 0.2 harmonogramu | patrz niżej |
| `first_dense_layers` | 2 | liczone od zera |
| `sink_mode` | `prefix` | patrz niżej |
| `correctness_gate` | włączony | |
| `strict` | wyłączony | |

**`first_dense_steps` jako ułamek.** Referencyjne `10` pochodzi z harmonogramu
50-krokowego. Przy 20 krokach oznaczałoby połowę przebiegu gęsto. Znając
`sample_sigmas` interpretujemy wartość <1 jako ułamek długości harmonogramu,
wartość ≥1 jako sztywną liczbę kroków.

**`sink_mode="prefix"`, nie `text`.** Sink obejmuje tekst, wiersze
warunkujące i audio — wszystko przed ogonem wideo. Wiersze audio są
generowane (model zwraca dla nich prędkość), a handoff NVIDII odnotował
prompt, w którym obraz wyszedł najlepiej w zestawie, podczas gdy dialog się
rozpadł. Koszt wobec polityki referencyjnej: ~1% gęstości i ~1% dodatkowych
gęstych wierszy-zapytań. `text` pozostaje dostępny do odtworzenia referencji.

**Sink dotyczy kluczy, nie zapytań.** Kernel czyni zakres sinka dokładnym jako
K/V, ale jego własne wiersze-zapytania nadal routują się rzadko. README
kernela jest jednoznaczne, że integracja MMDiT musi przeliczyć je gęsto.
Dokładność stosowana jest z granulacją bloków 64-tokenowych, zaokrąglając na
zewnątrz.

**Kompozycja z cache.** Sol-Engine dla H3 składa Sol-Attn z FirstBlockCache,
więc współpraca z `ComfyUI-MiniMaxH3-Cache` jest zamierzona. Gdy cache pominie
krok, wrapper po prostu się nie odpali; zegar oparty o `sample_sigmas`
pozostaje poprawny, licznik wywołań by się rozjechał.

## 8. Obsługa błędów

Zasada: konfiguracja, która poprosiła o rzadką uwagę i po cichu dostała
gęstą, to błędny pomiar w prawidłowej etykiecie. Każda odmowa niesie powód,
nie boolean.

| Powód | Zachowanie |
|---|---|
| `warmup_step`, `dense_layer` | zamierzone — cicho, tylko licznik |
| `disabled` | zamierzone — przełącznik `enabled` wyłączony, cicho |
| `no_layout` | wrapper nie zobaczył `minimax_payload["layout"]` — log raz |
| `seq_mismatch` | `token_refiner` i wszystko nieoczekiwane — log raz |
| `dtype`, `head_dim`, `mask_present` | kontrakt kernela — log raz |
| `arch_unsupported` (<SM80), `kernel_import` | log raz przy montażu |
| `kernel_error` | wyjątek z kernela → gęsto, log z tracebackiem raz |
| `oom` | zatrzaskuje gęsto do końca przebiegu |

**Kontrola zbiorcza na koniec przebiegu.** Jeśli sampling przekroczył
`first_dense_steps` i nie wykonał ani jednego wywołania rzadkiego — głośny
warning. Powody per-call tego nie wyłapią, bo powód, który robi szkodę
(`warmup_step`), jest sam w sobie legalny. Błąd istnieje tylko w agregacie.

`strict=True` zamienia każdy niezamierzony powód w wyjątek. Do walidacji, nie
do codziennej pracy.

**Gate poprawności** — raz na kształt sekwencji. `tau=-1000` przepuszcza
wszystkie bloki, więc porównanie z SDPA mierzy arytmetykę kernela, nie
politykę routingu. Na prawdziwych QKV i wszystkich głowach: próba na losowych
tensorach odpowiada na pytanie o kernel, nie o ten model przy tym kształcie.
Gate na produkcyjnej liczbie głów rozgrzewa też autotuning Tritona w
`preprocess.prepare`, który kluczuje po samym `T` — pierwsze wywołanie przy
jednej głowie zapisałoby konfigurację dobraną dla siatki jednogłowowej.

Progi: `max_abs ≤ 0,15` (≥32k tokenów, inaczej 0,08), `mean_abs ≤ 0,002`,
`rel_l2 ≤ 0,005`. Fail → wyjątek; cicha akceptacja zepsutego kernela jest
gorsza niż brak przyspieszenia.

**Sonda gęstości** — raz. Raportuje `threshold_density` i `effective_density`.
Gęstość bliska 1,0 znaczy, że routing nie routuje; 0,0 — że się zapadł.

## 9. Plan walidacji

**Krok 0 — spike instalacyjny, przed pisaniem node'a.**
`pip install -e techniques/sparse_backends` do venva comfyX, potem samodzielny
skrypt wołający `sol_attn()` na syntetycznych tensorach w realnym kształcie
(~31k × 56 × 128 bf16) kontra SDPA. Odpowiada na cztery pytania unieważniające
resztę planu: czy Triton się kompiluje, czy gate przechodzi, jaka wychodzi
gęstość, ile kosztują kopie contiguous.

**Krok 1 — testy jednostkowe, bez GPU.** Wyłącznie tam, gdzie chronią przed
cichą regresją — czyli w trzech miejscach, w których referencja miała
udokumentowane wpadki:

- `layout.py`: sink dla t2va, fl2va z keyframe'ami, ref2va z refs
- `state.py`: zegar przy pominiętych krokach, reset między przebiegami,
  off-by-one na `first_dense_layers`
- tablica wyboru backendu per compute capability

**Krok 2 — integracja na prawdziwym H3, lokalnie.** ComfyUI headless,
minimalny graf H3 w małej rozdzielczości, `strict=True`. Dowody:

- gate PASS na prawdziwych QKV
- `sparse_calls > 0` w mierzonym przebiegu
- w `declined` wyłącznie zamierzone powody plus `seq_mismatch` z refinera
- gęstość w sensownym paśmie, nie 1,0 i nie 0,0
- A/B na tym samym seedzie: off vs on — różnica na zdekodowanych klatkach i czas
- ten sam przebieg z włączonym `MiniMaxH3-Cache`

**Czego ta maszyna nie udowodni:** numeryki i wydajności ścieżki CuTe SM120
ani przyspieszenia w skali produkcyjnej. Stąd skrypt selftestu do uruchomienia
na maszynie docelowej: wypisuje wybrany backend, wynik gate'u, gęstość i
udział uwagi w czasie kroku.

## 10. Ryzyka

| Ryzyko | Ocena |
|---|---|
| Ścieżka Triton (Ada, Ampere) jest w README kernela opisana jako implementacja badawcza „for portability, kernel studies"; NVIDIA benchmarkowała SM90/100/120 | Na słabszych kartach Sol-Attn może wypaść gorzej od obecnego SageAttention. Node musi dać się wyłączyć jednym przełącznikiem i raportować backend. |
| ~1,3 GB tymczasowych na kopie contiguous przy 31k wierszy | Bez znaczenia na PRO 6000 (96 GB), istotne na kartach 12–16 GB. Stąd zatrzask `oom`. |
| Kompozycja z `MiniMaxH3-Cache`, który patchuje pliki core | Niesprawdzona. Testowana jawnie w kroku 2. |
| Wersja ComfyUI | Wpięcie stoi na publicznych API, ale `PackedLayout.segments` i sygnatura `minimax_payload` to szczegóły implementacji H3. Kontrakt sprawdzany przy montażu, z jawnym błędem zamiast cichej degradacji. |

## 11. Licencja i atrybucja

Kernel `sol-attn` pochodzi z NVlabs/Sana (Apache-2.0) i jest **instalowany**,
nie vendorowany — w v1 nie powstaje obowiązek redystrybucji. Integracja czerpie
z `models/minimax_h3/optimized/sol_attn_h3.py` (Apache-2.0); README node'a
wskazuje źródło, paper (arXiv 2607.24027) i licencję. Przy ewentualnej
publikacji decyzję o vendorowaniu trzeba podjąć ponownie.
