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

## Zmierzone

**Uwaga: to są liczby z RTX 4070 Ti, czyli ze ścieżki Triton (SM89) — najsłabszej
ze wspieranych.** Architektury SM90/SM100/SM120 dostają kernele CuTe DSL, których
tutaj nie da się zmierzyć. Przenoszenie tych liczb na inne GPU jest nieuprawnione;
do pomiaru u siebie służy `selftest.py`.

### End-to-end, MiniMax-H3 w ComfyUI

864×480, 125 klatek (sekwencja **17 504** wierszy), 8 kroków, `res_multistep`,
model int8 `fl2va`. Linia bazowa to domyślna uwaga ComfyUI w tej instalacji —
`pytorch attention` (SDPA). Przebieg pomiarowy po rozgrzewce:

| | `off` | `on` | |
|---|---:|---:|---|
| czas end-to-end | 116,1 s | **84,2 s** | **1,38×** |
| czas uwagi łącznie | 45,5 s | 26,8 s | 1,70× |
| ms na wywołanie gęste | 113,65 | 113,32 | — |
| ms na wywołanie rzadkie | — | **47,87** | **2,37×** |
| wywołania rzadkie / gęste | 0 / 400 | 288 / 112 | |

Wywołania gęste zmierzone w czterech przebiegach: 113,18 / 113,27 / 113,65 /
113,32 ms. Zgodność wariantu `off` i `on` na tej samej ścieżce jest kontrolą
poprawności samego pomiaru — instrumentacja nie przekrzywia wyniku.

Uwaga stanowi tu **59% czasu kroku** (45,5 s uwagi na 76 s samplowania), więc
przyspieszenie kernela przekłada się na czas całkowity w rozsądnej proporcji.

**Koszt jednorazowy:** pierwsze wywołanie rzadkie to 13,4 s (kompilacja kerneli
Tritona) plus gate i sonda gęstości. Przy ciepłym cache'u spada do 0,38 s.

**Triton kompiluje per kształt sekwencji**, a nie raz na proces. Każda nowa
rozdzielczość lub długość płaci ten koszt ponownie — i przy krótkich przebiegach
potrafi zjeść cały zysk. Drugi punkt pomiarowy, 640×384, 73 klatki (sekwencja
**5 548**), 20 kroków:

| | `off` | `on` | |
|---|---:|---:|---|
| ms na wywołanie | 11,88 | **7,33** | 1,62× |
| czas uwagi bez kosztu jednorazowego | 11,9 s | 8,4 s | 1,41× |
| kompilacja dla nowego kształtu | — | 3,4 s | |
| czas end-to-end | 40,3 s | 39,1 s | **1,03×** |

Czyli: krótsza sekwencja to mniejszy zysk na wywołaniu (1,62× wobec 2,37× przy
17 504) i mniejszy udział uwagi w kroku, a do tego jednorazowa kompilacja
rozłożona na mniej pracy. Sol-Attn opłaca się tym bardziej, im dłuższe wideo.

### Kompozycja z `ComfyUI-MiniMaxH3-Cache`

Sol-Engine dla H3 składa Sol-Attn z FirstBlockCache, więc współpraca jest
zamierzona. Zweryfikowana z `strict=True`, 20 kroków, sekwencja 5 548:

```
sparse_calls: 336   dense_calls: 114   last_step: 19   total_steps: 20
```

Razem 450 wywołań zamiast 1 000 — cache pominął 11 z 20 forwardów, a mimo to
numer kroku i długość harmonogramu pozostały poprawne, `strict` nie podniósł
żadnego niezamierzonego powodu, a ścieżka rzadka wykonała 336 wywołań.
Czas: 39,1 s → **20,0 s**.

To jest uzasadnienie dla czytania numeru kroku z `sample_sigmas` zamiast liczenia
forwardów: **licznik rozjechałby się przy każdym pominiętym kroku.**

### Kernel na syntetycznych kształtach (`selftest.py`)

56 głów, head_dim 128, `tau=1.0`, `thresh_type=diag`:

| T | gate | gęstość | sol_attn | SDPA | SageAttention | vs SDPA | vs Sage |
|---:|:--:|---:|---:|---:|---:|---:|---:|
| 5 548 | PASS | 0,231 | — | — | — | 2,75× | 1,27× |
| 8 192 | PASS | 0,271 | 10,2 ms | 26,4 ms | 11,2 ms | 2,58× | 1,09× |
| 16 384 | PASS | 0,214 | 31,7 ms | 105,4 ms | 40,0 ms | 3,32× | 1,26× |
| 30 976 | PASS | 0,186 | 99,6 ms | 380,0 ms | 133,6 ms | 3,82× | 1,34× |

Gęstość spada wraz z długością sekwencji, czyli routing tym bardziej się opłaca,
im dłuższe wideo.

### Jakość — i dlaczego PSNR off-vs-on wprowadza w błąd

Sol-Attn jest **aproksymacją**, nie ścieżką bezstratną. Przy identycznym seedzie,
20 krokach i sekwencji 5 548 porównanie zdekodowanych klatek daje **22,4 dB**.

Sama ta liczba jest jednak myląca, co widać w teście kontrolnym. Przy
`tau = -1000` routing przepuszcza **wszystkie** bloki (zmierzona gęstość: dokładnie
1,0), więc kernel liczy pełną uwagę i nic nie pomija. Mimo to PSNR wobec ścieżki
gęstej wynosi wtedy **30,7 dB**, a nie kilkadziesiąt:

| konfiguracja | gęstość routingu | PSNR vs gęsta |
|---|---:|---:|
| `tau = -1000` (nic nie pomijane) | 1,000 | 30,7 dB |
| `tau = 1.0` (polityka produkcyjna) | 0,311 | 22,4 dB |

Innymi słowy **podłoga wynosi tu ~31 dB** i bierze się z samej podmiany
implementacji uwagi. Błąd względny kernela to 0,1% na wywołanie (`rel_l2`
z gate'u), a najgorszy element to co do wartości pół ulpa bf16 — czyli tyle,
ile format w ogóle pozwala. Przy 20 krokach × 48 warstwach rzadkich to około
960 zastosowań tego zaburzenia w nieliniowym samplerze, więc rozjazd trajektorii
jest makroskopowy. Routing dokłada do tego około 8 dB.

To nie jest usterka integracji: gate poprawności przechodzi, wiersze prefiksu
zgadzają się z gęstą uwagą, a ścieżka odmowy zwraca bit w bit wynik oryginalnego
backendu (pilnują tego testy GPU). Referencja NVIDII ostrzega o tym wprost —
*„a visual metric alone will rate it too highly on this model"* — i sama raportuje
dla H3 LPIPS 0,293.

**Praktyczny wniosek:** oceniaj na własnym materiale, nie po PSNR. Do pracy
wymagającej wierności wobec natywnej trajektorii wyłącz węzeł przełącznikiem
`enabled` i porównaj ten sam prompt z tym samym seedem.

## Znane ograniczenia

- **Tylko MiniMax-H3.** Węzeł odmawia montażu na innym modelu z jawnym błędem.
- **Kontrakt kernela:** bf16, head_dim dokładnie 128, contiguous BTHD, batch 1.
  Naruszenie → gęsta uwaga, nie wyjątek.
- **`torch.compile`:** override jest callable w `transformer_options`, więc
  wystąpi graph-break. Nie naprawiane.
- **Kompilacja per kształt:** Triton kompiluje kernele osobno dla każdej długości
  sekwencji. Zmiana rozdzielczości lub liczby klatek to jednorazowe 3–13 s.
  Przy krótkich przebiegach potrafi to zniwelować cały zysk.
- **Kopie contiguous:** H3 podaje Q/K/V jako widoki w spakowanym buforze
  `qkv_proj`, więc kernel wymaga jednej kopii — 3 × 424 MiB przy 31 tys.
  wierszy, około 6,5% czasu kernela. Przy braku pamięci ścieżka rzadka zatrzaskuje
  się na gęstą do końca przebiegu.

## Licencja i atrybucja

Kernel i polityka pochodzą z NVlabs/Sana (Apache-2.0). Szczegóły w [NOTICE](NOTICE).
