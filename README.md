# ComfyUI-SolAttn-H3

Training-free rzadka uwaga **Sol-Attn** z [NVIDIA Sol Engine](https://github.com/NVlabs/Sana/tree/sol-engine)
dla natywnego **MiniMax-H3** w ComfyUI.

Jeden węzeł, `Sol-Attn (MiniMax-H3)`, wpinany między loader modelu a resztę
grafu. Bez treningu, bez LoRA, bez kalibracji offline. Gdy kontrakt kernela nie
jest spełniony, węzeł schodzi na gęstą uwagę z **nazwanym powodem** — nigdy po
cichu.

## Wymagania

| | |
|---|---|
| ComfyUI | z natywnym MiniMax-H3 (`comfy.ldm.minimax`) |
| GPU | NVIDIA, compute capability ≥ 8.0 |
| PyTorch | ≥ 2.10 |
| CUDA | ≥ 12.8 |
| Triton | ≥ 3.6 |
| CuTe DSL | opcjonalnie: `cutlass-python` ≥ 4.5 + `cuda-python` |

Backend wybierany jest automatycznie po architekturze GPU:

| Architektura | Przykład | Backend |
|---|---|---|
| SM90 | H100 | CuTe DSL |
| SM100 | B200 / GB200 | CuTe DSL |
| SM120 | RTX 5090, RTX PRO 6000 Blackwell | CuTe DSL |
| SM80 / SM86 / SM89 | A100, RTX 3090, RTX 4090 | Triton |

Brak `cutlass.cute` lub `cuda-python` → automatyczny spadek na Tritona,
niezależnie od architektury. Węzeł wypisuje wybrany backend przy montażu.

## Instalacja

```bash
git clone <to-repo> ~/sol
ln -s ~/sol ComfyUI/custom_nodes/ComfyUI-SolAttn-H3

# kernel: instalowany z repozytorium NVlabs, nie vendorowany
git clone --branch sol-engine --depth 1 https://github.com/NVlabs/Sana.git ~/sol/vendor/sana-sol-engine
uv pip install --python ComfyUI/venv/bin/python -e ~/sol/vendor/sana-sol-engine/techniques/sparse_backends
```

Sprawdzenie środowiska bez uruchamiania ComfyUI:

```bash
ComfyUI/venv/bin/python ~/sol/selftest.py
```

Wypisuje GPU, wybrany backend, wynik gate'u poprawności, gęstość routingu oraz
porównanie prędkości z SDPA i SageAttention na zadanych długościach sekwencji.

## Parametry

| Parametr | Domyślnie | Znaczenie |
|---|---|---|
| `enabled` | on | Wyłącza całość bez przebudowy grafu. |
| `tau` | 1.0 | Wyższe = mniej bloków K/V liczonych dokładnie. Wartość ze zwalidowanej linii H3; kalibracja per-kształt w referencji zwracała pusty zbiór tras. |
| `thresh_type` | `diag` | `exact` = próg pełnokowariancyjny, dokładniejszy i droższy. |
| `first_dense_steps` | 0.2 | Poniżej 1 = ułamek harmonogramu, od 1 = sztywna liczba kroków. 0.2 odpowiada referencyjnym 10 krokom z 50. |
| `first_dense_layers` | 2 | Pierwsze N bloków DiT liczonych gęsto. Liczone od zera. |
| `sink_mode` | `prefix` | `prefix` = tekst + warunkowanie + audio trzymane dokładnie. `text` = polityka referencyjna. |
| `correctness_gate` | on | Raz na kształt porównuje kernel z SDPA na prawdziwych QKV. Fail przerywa generowanie. |
| `strict` | off | Każda niezamierzona odmowa ścieżki rzadkiej staje się wyjątkiem. Do walidacji. |

### Dlaczego `sink_mode=prefix`, a nie `text`

Sink to ciągły zakres K/V trzymany dokładnie dla wszystkich zapytań. Polityka
referencyjna obejmuje nim sam tekst. Tutaj domyślnie obejmuje cały prefiks —
także wiersze audio, bo te są **generowane** (model zwraca dla nich prędkość),
a handoff NVIDII odnotował prompt, w którym obraz wyszedł najlepiej w zestawie,
podczas gdy dialog się rozpadł. Koszt wobec polityki referencyjnej to około 1%
gęstości i 1% dodatkowych gęstych wierszy-zapytań.

## Jak to jest wpięte

Cztery publiczne API ModelPatchera. **Żaden plik w `comfy/` nie jest
modyfikowany** — w przeciwieństwie do niektórych innych węzłów akceleracyjnych
dla H3, które patchują pliki core.

| Mechanizm | Rola |
|---|---|
| `transformer_options["optimized_attention_override"]` | przechwycenie uwagi; zwrot `func(...)` daje gęsty fallback |
| wrapper `WrappersMP.DIFFUSION_MODEL` | raz na forward: layout, zakres sinka, numer kroku |
| `patches_replace["dit"][("double_block", i)]` | stempel prawdziwego indeksu bloku |
| wrapper `WrappersMP.OUTER_SAMPLE` | granice przebiegu i kontrola zbiorcza |

Indeks bloku jest **stemplowany, nie liczony**, bo `token_refiner` również woła
`Attention` z head_dim 128 — liczenie wywołań przesunęłoby `first_dense_layers`.

## Diagnostyka

Węzeł loguje trzy rzeczy, wszystkie po to, żeby wykryć konfigurację, która
prosi o rzadką uwagę i po cichu liczy gęstą:

- **gate poprawności** — raz na kształt, `tau=-1000` przepuszcza wszystkie
  bloki, więc porównanie z SDPA mierzy arytmetykę kernela, nie politykę routingu
- **gęstość routingu** — bliska 1,0 znaczy, że routing nie routuje; 0,0 że się
  zapadł
- **statystyki przebiegu** — liczba wywołań rzadkich i gęstych, rozbicie powodów
  odmowy, czas uwagi w obu ścieżkach

Powody odmowy: `disabled`, `warmup_step`, `dense_layer` są zamierzone i ciche.
`kernel_unavailable`, `oom`, `mask_present`, `layout_unknown`, `batch`, `dtype`,
`head_dim`, `no_layout`, `seq_mismatch` są logowane raz, a w trybie `strict`
podnoszą wyjątek. Dodatkowo przebieg, który minął rozgrzewkę i nie wykonał ani
jednego wywołania rzadkiego, kończy się głośnym ostrzeżeniem — powód, który
robi szkodę (`warmup_step`), jest sam w sobie legalny, więc błąd istnieje
wyłącznie w agregacie.

## Znane ograniczenia

- **Tylko MiniMax-H3.** Węzeł odmawia montażu na innym modelu z jawnym błędem.
- **Kontrakt kernela:** bf16, head_dim dokładnie 128, contiguous BTHD, batch 1.
  Naruszenie → gęsta uwaga, nie wyjątek.
- **`torch.compile`:** override jest callable w `transformer_options`, więc
  wystąpi graph-break. Nie naprawiane.
- **Kopie contiguous:** H3 podaje Q/K/V jako widoki w spakowanym buforze
  `qkv_proj`, więc kernel wymaga jednej kopii — 3 × 424 MiB przy 31 tys.
  wierszy, około 6,5% czasu kernela. Przy braku pamięci ścieżka rzadka zatrzaskuje
  się na gęstą do końca przebiegu.

## Licencja i atrybucja

Kernel i polityka pochodzą z NVlabs/Sana (Apache-2.0). Szczegóły w [NOTICE](NOTICE).
