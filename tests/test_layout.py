"""Testy wyznaczania sinka z tabeli segmentow PackedLayout.

Fixture'y odwzorowuja `comfy/ldm/minimax/model.py:396-409`: lista (start, stop,
kind), kindy text/cond/ref_img/ref_audio/audio/video, a docelowe audio i wideo
zawsze jako dwa ostatnie segmenty.
"""
import pytest

from solattn_h3.layout import sink_from_segments

SEQ = 31000
# t2va: sam tekst, potem docelowe audio i wideo
T2VA = [(0, 537, "text"), (537, 951, "audio"), (951, SEQ, "video")]
# fl2va: keyframe jako segment cond
FL2VA = [(0, 537, "text"), (537, 800, "cond"), (800, 1200, "audio"), (1200, SEQ, "video")]
# ref2va: blok referencyjny wnosi ref_audio + ref_img
REF2VA = [(0, 537, "text"), (537, 700, "ref_audio"), (700, 900, "ref_img"),
          (900, 1300, "audio"), (1300, SEQ, "video")]
# wiele keyframe'ow -> wiele segmentow cond
MULTI_COND = [(0, 537, "text"), (537, 700, "cond"), (700, 863, "cond"),
              (863, 1100, "audio"), (1100, SEQ, "video")]

ALL = [T2VA, FL2VA, REF2VA, MULTI_COND]


@pytest.mark.parametrize("segs,video_start", [
    (T2VA, 951), (FL2VA, 1200), (REF2VA, 1300), (MULTI_COND, 1100),
])
def test_prefix_sink_konczy_sie_na_ogonie_wideo(segs, video_start):
    """Sink w trybie prefix obejmuje wszystko przed docelowym wideo."""
    s = sink_from_segments(segs, SEQ, "prefix")
    assert (s.start, s.tokens, s.video_start) == (0, video_start, video_start)


@pytest.mark.parametrize("segs", ALL)
def test_text_sink_obejmuje_tylko_tekst(segs):
    s = sink_from_segments(segs, SEQ, "text")
    assert (s.start, s.tokens) == (0, 537)
    assert s.video_start == segs[-1][0]


@pytest.mark.parametrize("segs", ALL)
@pytest.mark.parametrize("mode", ["prefix", "text"])
def test_sink_nigdy_nie_wykracza_poza_sekwencje(segs, mode):
    s = sink_from_segments(segs, SEQ, mode)
    assert 0 <= s.start
    assert s.start + s.tokens <= s.seq_len == SEQ


@pytest.mark.parametrize("segs", ALL)
def test_prefix_jest_nadzbiorem_text(segs):
    """Prefix dokłada audio i wiersze warunkujace do tego, co obejmuje text."""
    assert sink_from_segments(segs, SEQ, "prefix").tokens >= \
           sink_from_segments(segs, SEQ, "text").tokens


def test_brak_segmentu_video_to_blad():
    with pytest.raises(ValueError, match="video"):
        sink_from_segments([(0, 537, "text")], 537, "prefix")


def test_wiecej_niz_jeden_segment_video_to_blad():
    """model.py:634 zaklada dokladnie jeden docelowy segment wideo."""
    segs = [(0, 100, "text"), (100, 200, "video"), (200, 300, "video")]
    with pytest.raises(ValueError, match="video"):
        sink_from_segments(segs, 300, "prefix")


def test_nieznany_tryb_sinka():
    with pytest.raises(ValueError, match="sink_mode"):
        sink_from_segments(T2VA, SEQ, "cokolwiek")


def test_brak_tekstu_daje_pusty_sink_a_nie_blad():
    """Sink pusty jest legalny: kernel po prostu nie ma czego trzymac dokladnie."""
    segs = [(0, 400, "audio"), (400, SEQ, "video")]
    s = sink_from_segments(segs, SEQ, "text")
    assert s.tokens == 0


def test_nieciagle_segmenty_to_blad():
    """Dziura miedzy segmentami znaczy, ze kontrakt PackedLayout sie zmienil."""
    with pytest.raises(ValueError, match="ciag"):
        sink_from_segments([(0, 100, "text"), (150, SEQ, "video")], SEQ, "prefix")
