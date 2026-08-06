"""Testy adaptera na GPU. Wymagaja CUDA i zainstalowanego sol-attn."""
import pytest
import torch

from solattn_h3.attention import dense_bthd, make_override, route_density, run_gate
from solattn_h3.kernel import load_sol_attn, probe
from solattn_h3.layout import SinkRange
from solattn_h3.state import DECLINE_SEQ_MISMATCH, Policy, SolAttnState

pytestmark = pytest.mark.skipif(not probe().available,
                                reason=f"kernel niedostepny: {probe().error}")

H, D, T = 8, 128, 4096
SINK = SinkRange(start=0, tokens=512, seq_len=T, video_start=512)


def _bhsd():
    """QKV w layoucie, w jakim podaje je H3: widoki w spakowanym buforze qkv_proj."""
    torch.manual_seed(0)
    packed = torch.randn(T, 3 * H * D, device="cuda", dtype=torch.bfloat16) * 0.5
    return [x.view(T, H, D).transpose(0, 1).unsqueeze(0) for x in packed.split(H * D, dim=-1)]


def _fake_func(q, k, v, heads, **kw):
    """Zastepnik oryginalnego backendu uwagi ComfyUI."""
    out = dense_bthd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.reshape(out.shape[0], out.shape[1], heads * out.shape[3])


def _ready_state(**kw):
    policy = Policy(first_dense_steps=0, first_dense_layers=0, correctness_gate=False, **kw)
    state = SolAttnState(policy)
    state.begin_run()
    state.begin_forward(SINK, step=5, total_steps=50)
    return state


def _run(state, q, k, v, **to):
    return make_override(state, state.policy)(
        _fake_func, q, k, v, H, mask=None, skip_reshape=True,
        transformer_options={"solattn_block": 4, **to})


def test_ksztalt_i_dtype_wyjscia():
    q, k, v = _bhsd()
    out = _run(_ready_state(), q, k, v)
    assert out.shape == (1, T, H * D)
    assert out.dtype == torch.bfloat16


def test_sciezka_rzadka_zostala_uzyta():
    state = _ready_state()
    _run(state, *_bhsd())
    assert state.stats()["sparse_calls"] == 1
    assert state.stats()["declined"] == {}


def test_odmowa_zwraca_dokladnie_wynik_gestej_sciezki():
    q, k, v = _bhsd()
    state = SolAttnState(Policy(enabled=False))
    state.begin_run()
    out = _run(state, q, k, v)
    torch.testing.assert_close(out, _fake_func(q, k, v, H))


def test_token_refiner_nie_trafia_do_kernela():
    """Krotsza sekwencja to wywolanie refinera; ma spasc na gesta sciezke."""
    state = _ready_state()
    packed = torch.randn(537, 3 * H * D, device="cuda", dtype=torch.bfloat16)
    q, k, v = (x.view(537, H, D).transpose(0, 1).unsqueeze(0)
               for x in packed.split(H * D, dim=-1))
    _run(state, q, k, v)
    assert state.stats()["declined"] == {DECLINE_SEQ_MISMATCH: 1}


def test_wiersze_prefiksu_sa_gesto_przeliczone():
    """Sink czyni prefiks dokladnym jako K/V, ale jego zapytania musza byc geste."""
    q, k, v = _bhsd()
    out = _run(_ready_state(), q, k, v).reshape(1, T, H, D)
    want = dense_bthd(q.transpose(1, 2)[:, :SINK.tokens], k.transpose(1, 2), v.transpose(1, 2))
    torch.testing.assert_close(out[:, :SINK.tokens], want, rtol=2e-2, atol=2e-2)


def test_ogon_wideo_rozni_sie_od_gestej_uwagi():
    """Gdyby routing nie routowal, wynik bylby identyczny z gestym — a to znaczyloby,
    ze konfiguracja liczy gesto pod etykieta rzadkiej."""
    q, k, v = _bhsd()
    out = _run(_ready_state(), q, k, v).reshape(1, T, H, D)
    want = dense_bthd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    tail_diff = (out[:, SINK.tokens:].float() - want[:, SINK.tokens:].float()).abs().max()
    assert tail_diff > 0, "ogon wideo identyczny z gestym: routing nie routuje"


def test_gate_przechodzi_na_prawdziwym_ksztalcie():
    q, k, v = _bhsd()
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    stats = run_gate(load_sol_attn(), qb, kb, vb, "diag")
    assert stats["passed"], stats


def test_gestosc_jest_w_pasmie():
    q, k, v = _bhsd()
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    d = route_density(qb, kb, vb, tau=1.0, thresh_type="diag", sink=SINK)
    assert 0.0 < d["effective_density"] < 1.0, d
    assert d["sink_blocks"] == SINK.tokens // 64


def test_zapasowy_licznik_blokow_gdy_brak_stempla():
    """Bez stempla z patches_replace indeks bloku pochodzi z licznika."""
    state = SolAttnState(Policy(first_dense_steps=0, first_dense_layers=2,
                                correctness_gate=False))
    state.begin_run()
    state.begin_forward(SINK, step=5, total_steps=50)
    q, k, v = _bhsd()
    override = make_override(state, state.policy)
    for _ in range(3):
        override(_fake_func, q, k, v, H, mask=None, skip_reshape=True,
                 transformer_options={})
    assert state.stats()["declined"] == {"dense_layer": 2}
    assert state.stats()["sparse_calls"] == 1


# --- kryterium gate'u -------------------------------------------------------

def test_gate_odrzuca_zepsuty_kernel():
    """Po zamianie progu bezwzglednego na wzgledny gate musi nadal lapac blad.

    Zepsuty routing albo indeksowanie daje blad wzgledny rzedu procentow i wyzej;
    5% przechodzi przez `mean_abs`, ale nie przez `max_rel`.
    """
    q, k, v = _bhsd()
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))

    def zepsuty(a, b, c, **kw):
        return dense_bthd(a, b, c) * 1.05

    stats = run_gate(zepsuty, qb, kb, vb, "diag")
    assert not stats["passed"], stats
    assert stats["max_rel"] > stats["limits"]["max_rel"]


def test_gate_przepuszcza_kernel_idealny():
    q, k, v = _bhsd()
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    stats = run_gate(lambda a, b, c, **kw: dense_bthd(a, b, c), qb, kb, vb, "diag")
    assert stats["passed"] and stats["max_rel"] == 0.0


def test_gate_nie_zalezy_od_skali_wyjscia():
    """Sedno zmiany: to samo zaburzenie wzgledne ma dac ten sam werdykt
    niezaleznie od tego, jak duze sa aktywacje. Prog bezwzgledny tego nie mial."""
    q, k, v = _bhsd()
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    verdicts = []
    for scale in (1.0, 50.0):
        big = [x * scale for x in (qb, kb, vb)]
        stats = run_gate(lambda a, b, c, **kw: dense_bthd(a, b, c) * 1.05, *big, "diag")
        verdicts.append(stats["passed"])
    assert verdicts == [False, False], verdicts
