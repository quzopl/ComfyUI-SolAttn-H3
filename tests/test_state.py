"""Testy zegara kroku, licznika blokow i powodow odmowy.

Kazdy z trzech blokow ponizej odpowiada udokumentowanej wpadce z referencyjnej
integracji NVIDII (`models/minimax_h3/optimized/sol_attn_h3.py`).
"""
import pytest
import torch

from solattn_h3.layout import SinkRange
from solattn_h3.state import (DECLINE_BATCH, DECLINE_DENSE_LAYER, DECLINE_DISABLED,
                              DECLINE_DTYPE, DECLINE_HEAD_DIM, DECLINE_LAYOUT_UNKNOWN,
                              DECLINE_MASK, DECLINE_NO_LAYOUT, DECLINE_OOM,
                              DECLINE_SEQ_MISMATCH, DECLINE_WARMUP, Policy, SolAttnState,
                              dense_step_count, resolve_step)

SINK = SinkRange(start=0, tokens=951, seq_len=31000, video_start=951)


def _state(**kw):
    s = SolAttnState(Policy(first_dense_steps=0.2, first_dense_layers=2, **kw))
    s.begin_run()
    return s


def _call(s, *, rows=31000, dtype=torch.bfloat16, head_dim=128, mask=None,
          block=5, skip_reshape=True, batch=1):
    return s.decline(rows=rows, dtype=dtype, head_dim=head_dim, mask=mask,
                     block_index=block, skip_reshape=skip_reshape, batch=batch)


# --- ulamek kroków gestych --------------------------------------------------

def test_ulamek_skaluje_sie_z_dlugoscia_harmonogramu():
    """Referencyjne 10 pochodzi z harmonogramu 50-krokowego."""
    assert dense_step_count(0.2, 50) == 10
    assert dense_step_count(0.2, 20) == 4


def test_wartosc_co_najmniej_1_jest_sztywna_liczba_krokow():
    assert dense_step_count(10, 50) == 10
    assert dense_step_count(10, 20) == 10


def test_zero_znaczy_zero_krokow_gestych():
    assert dense_step_count(0, 50) == 0


# --- numer kroku z harmonogramu ---------------------------------------------

def test_numer_kroku_z_harmonogramu():
    sample = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    assert resolve_step(torch.tensor([0.5]), sample) == 2
    assert resolve_step(torch.tensor([1.0]), sample) == 0
    assert resolve_step(torch.tensor([0.0]), sample) == 4


def test_numer_kroku_gdy_sigma_spoza_harmonogramu():
    assert resolve_step(torch.tensor([0.31]), torch.tensor([1.0, 0.5, 0.0])) is None


def test_numer_kroku_bez_harmonogramu():
    assert resolve_step(torch.tensor([0.5]), None) is None


def test_pominiete_kroki_nie_psuja_zegara():
    """Cache pomija forwardy; numer kroku pochodzi z harmonogramu, nie z licznika.

    Referencja liczyla forwardy i przy warmupie + pomiarze zaczynala mierzony
    przebieg od kroku 49, raportujac przy tym dziesiec krokow gestych.
    """
    s = _state()
    for step in (10, 13, 14, 19):
        s.begin_forward(SINK, step=step, total_steps=50)
        assert _call(s) is None
    assert s.stats()["last_step"] == 19
    assert s.stats()["sparse_calls"] == 0  # decline() nie liczy, liczy note_sparse()


# --- licznik blokow ---------------------------------------------------------

def test_dense_layers_liczone_od_zera():
    """Referencja miala tu off-by-one: dense_layers=2 zostawialo gesty tylko blok 0."""
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s, block=0) == DECLINE_DENSE_LAYER
    assert _call(s, block=1) == DECLINE_DENSE_LAYER
    assert _call(s, block=2) is None


def test_licznik_blokow_zeruje_sie_co_forward():
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    assert [s.next_block() for _ in range(3)] == [0, 1, 2]
    s.begin_forward(SINK, step=11, total_steps=50)
    assert s.next_block() == 0


# --- powody odmowy ----------------------------------------------------------

def test_kroki_rozgrzewkowe_odmawiaja_reszta_nie():
    s = _state()
    s.begin_forward(SINK, step=3, total_steps=50)      # 3 < 10
    assert _call(s) == DECLINE_WARMUP
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s) is None


def test_kontrakt_kernela():
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s, dtype=torch.float16) == DECLINE_DTYPE
    assert _call(s, head_dim=64) == DECLINE_HEAD_DIM
    assert _call(s, mask=object()) == DECLINE_MASK
    assert _call(s, batch=2) == DECLINE_BATCH
    assert _call(s, skip_reshape=False) == DECLINE_LAYOUT_UNKNOWN


def test_token_refiner_odpada_na_dlugosci_sekwencji():
    """model.py:584 wola Attention z head_dim 128 na samych wierszach tekstu."""
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s, rows=537) == DECLINE_SEQ_MISMATCH


def test_brak_layoutu():
    s = _state()
    assert _call(s) == DECLINE_NO_LAYOUT       # bez begin_forward


def test_wylaczony_node():
    s = _state(enabled=False)
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s) == DECLINE_DISABLED


def test_oom_zatrzaskuje_gesto():
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s) is None
    s.latch_oom()
    assert _call(s) == DECLINE_OOM


def test_niedostepny_kernel():
    s = _state()
    s.kernel_error = "brak sol_attn"
    s.begin_forward(SINK, step=10, total_steps=50)
    assert _call(s) == "kernel_unavailable"


# --- kontrola zbiorcza ------------------------------------------------------

def test_przebieg_bez_ani_jednego_wywolania_rzadkiego_jest_bledem():
    """Powod, ktory robi szkode (warmup_step), jest sam w sobie legalny.

    Blad istnieje tylko w agregacie, wiec i sprawdzenie musi byc w agregacie.
    """
    s = SolAttnState(Policy(strict=True))
    s.begin_run()
    for step in range(50):
        s.begin_forward(SINK, step=step, total_steps=50)
        s.note(DECLINE_WARMUP)
    with pytest.raises(RuntimeError, match="ani jednego"):
        s.end_run()


def test_przebieg_z_wywolaniami_rzadkimi_przechodzi():
    s = SolAttnState(Policy(strict=True))
    s.begin_run()
    s.begin_forward(SINK, step=20, total_steps=50)
    s.note_sparse()
    s.end_run()


def test_przebieg_krotszy_niz_rozgrzewka_nie_alarmuje():
    """5 krokow przy 10 gestych to nie blad, tylko krotki przebieg."""
    s = SolAttnState(Policy(strict=True, first_dense_steps=0.2))
    s.begin_run()
    for step in range(5):
        s.begin_forward(SINK, step=step, total_steps=50)
        s.note(DECLINE_WARMUP)
    s.end_run()


def test_strict_podnosi_na_niezamierzonym_powodzie():
    s = SolAttnState(Policy(strict=True))
    s.begin_run()
    with pytest.raises(RuntimeError, match=DECLINE_SEQ_MISMATCH):
        s.note(DECLINE_SEQ_MISMATCH)


def test_zamierzone_powody_nie_podnosza_w_strict():
    s = SolAttnState(Policy(strict=True))
    s.begin_run()
    for reason in (DECLINE_WARMUP, DECLINE_DENSE_LAYER, DECLINE_DISABLED):
        s.note(reason)
    assert s.stats()["declined"] == {DECLINE_WARMUP: 1, DECLINE_DENSE_LAYER: 1,
                                     DECLINE_DISABLED: 1}


def test_begin_run_zeruje_liczniki_miedzy_przebiegami():
    s = _state()
    s.begin_forward(SINK, step=10, total_steps=50)
    s.note_sparse()
    s.note(DECLINE_WARMUP)
    s.begin_run()
    assert s.stats()["sparse_calls"] == 0
    assert s.stats()["declined"] == {}


# --- gate raz na ksztalt ----------------------------------------------------

def test_gate_odpala_sie_raz_na_ksztalt():
    s = _state()
    assert s.should_gate((31000, 56, 128)) is True
    s.record_gate((31000, 56, 128), {"passed": True})
    assert s.should_gate((31000, 56, 128)) is False
    assert s.should_gate((16000, 56, 128)) is True


def test_gate_wylaczony_polityka():
    s = _state(correctness_gate=False)
    assert s.should_gate((31000, 56, 128)) is False


def test_wylaczony_wezel_nie_alarmuje_na_koniec_przebiegu():
    """Wariant `off` w benchmarku ma zero wywolan rzadkich z definicji."""
    s = SolAttnState(Policy(strict=True, enabled=False))
    s.begin_run()
    for step in range(8):
        s.begin_forward(SINK, step=step, total_steps=8)
        s.note(DECLINE_DISABLED)
    s.end_run()
