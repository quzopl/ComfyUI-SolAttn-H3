"""Tests for deriving the sink range from PackedLayout's segment table.

The fixtures mirror `comfy/ldm/minimax/model.py:396-409`: a list of
(start, stop, kind), kinds text/cond/ref_img/ref_audio/audio/video, with target
audio and target video always the last two segments.
"""
import pytest

from solattn_h3.layout import sink_from_segments

SEQ = 31000
# t2va: text only, then target audio and video
T2VA = [(0, 537, "text"), (537, 951, "audio"), (951, SEQ, "video")]
# fl2va: a keyframe arrives as a cond segment
FL2VA = [(0, 537, "text"), (537, 800, "cond"), (800, 1200, "audio"), (1200, SEQ, "video")]
# ref2va: a reference block contributes ref_audio + ref_img
REF2VA = [(0, 537, "text"), (537, 700, "ref_audio"), (700, 900, "ref_img"),
          (900, 1300, "audio"), (1300, SEQ, "video")]
# several keyframes -> several cond segments
MULTI_COND = [(0, 537, "text"), (537, 700, "cond"), (700, 863, "cond"),
              (863, 1100, "audio"), (1100, SEQ, "video")]

ALL = [T2VA, FL2VA, REF2VA, MULTI_COND]


@pytest.mark.parametrize("segs,video_start", [
    (T2VA, 951), (FL2VA, 1200), (REF2VA, 1300), (MULTI_COND, 1100),
])
def test_prefix_sink_ends_at_the_video_tail(segs, video_start):
    """In prefix mode the sink covers everything before the target video."""
    s = sink_from_segments(segs, SEQ, "prefix")
    assert (s.start, s.tokens, s.video_start) == (0, video_start, video_start)


@pytest.mark.parametrize("segs", ALL)
def test_text_sink_covers_only_the_text(segs):
    s = sink_from_segments(segs, SEQ, "text")
    assert (s.start, s.tokens) == (0, 537)
    assert s.video_start == segs[-1][0]


@pytest.mark.parametrize("segs", ALL)
@pytest.mark.parametrize("mode", ["prefix", "text"])
def test_sink_never_runs_past_the_sequence(segs, mode):
    s = sink_from_segments(segs, SEQ, mode)
    assert 0 <= s.start
    assert s.start + s.tokens <= s.seq_len == SEQ


@pytest.mark.parametrize("segs", ALL)
def test_prefix_is_a_superset_of_text(segs):
    """Prefix adds the audio and conditioning rows on top of what text covers."""
    assert sink_from_segments(segs, SEQ, "prefix").tokens >= \
           sink_from_segments(segs, SEQ, "text").tokens


def test_missing_video_segment_is_an_error():
    with pytest.raises(ValueError, match="video"):
        sink_from_segments([(0, 537, "text")], 537, "prefix")


def test_more_than_one_video_segment_is_an_error():
    """model.py:634 assumes exactly one target video segment."""
    segs = [(0, 100, "text"), (100, 200, "video"), (200, 300, "video")]
    with pytest.raises(ValueError, match="video"):
        sink_from_segments(segs, 300, "prefix")


def test_unknown_sink_mode():
    with pytest.raises(ValueError, match="sink_mode"):
        sink_from_segments(T2VA, SEQ, "whatever")


def test_missing_text_yields_an_empty_sink_not_an_error():
    """An empty sink is legal: the kernel simply has nothing to keep exact."""
    segs = [(0, 400, "audio"), (400, SEQ, "video")]
    s = sink_from_segments(segs, SEQ, "text")
    assert s.tokens == 0


def test_non_contiguous_segments_are_an_error():
    """A gap between segments means the PackedLayout contract has changed."""
    with pytest.raises(ValueError, match="contiguous"):
        sink_from_segments([(0, 100, "text"), (150, SEQ, "video")], SEQ, "prefix")
