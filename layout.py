"""The sink range, derived from PackedLayout's segment table.

The sink is a contiguous K/V range the kernel keeps exact for every query, at
64-token block granularity (rounding outward). NVIDIA's reference had to infer
the start of the video tail from discontinuities in `video_indices`; ComfyUI
hands over labelled segments, so we read it off directly.
"""
from __future__ import annotations

from dataclasses import dataclass

SINK_MODES = ("prefix", "text")


@dataclass(frozen=True)
class SinkRange:
    """A contiguous K/V range kept exact, plus the target-video boundary."""

    start: int
    tokens: int
    seq_len: int
    video_start: int

    @property
    def stop(self) -> int:
        return self.start + self.tokens


def sink_from_segments(segments, seq_len: int, sink_mode: str) -> SinkRange:
    """Determine the sink for the given mode.

    `segments` is the list of (start, stop, kind) from `PackedLayout.segments`.
    Target audio and target video are always the last two segments, and there
    may be several `cond` / `ref_img` segments.

    `prefix` covers everything before the target video — text, conditioning rows
    and audio. The audio rows are *generated* (the model returns a velocity for
    them), and NVIDIA's handoff recorded a prompt whose picture scored best of
    its set while its dialogue fell apart. Hence prefix by default, even though
    the reference policy sinks only the text rows.
    """
    if sink_mode not in SINK_MODES:
        raise ValueError(f"unknown sink_mode {sink_mode!r}; allowed: {SINK_MODES}")

    _check_contiguous(segments, seq_len)

    video = [seg for seg in segments if seg[2] == "video"]
    if len(video) != 1:
        raise ValueError(
            f"expected exactly one 'video' segment, found {len(video)}; "
            "the PackedLayout contract has changed"
        )
    video_start = int(video[0][0])

    if sink_mode == "prefix":
        return SinkRange(0, video_start, seq_len, video_start)

    text = [seg for seg in segments if seg[2] == "text"]
    if not text:
        # An empty sink is legal — the kernel simply has nothing to keep exact.
        return SinkRange(0, 0, seq_len, video_start)
    start, stop = int(text[0][0]), int(text[-1][1])
    return SinkRange(start, stop - start, seq_len, video_start)


def _check_contiguous(segments, seq_len: int) -> None:
    """Segments must be contiguous and cover the whole sequence."""
    if not segments:
        raise ValueError("empty segment table")
    offset = 0
    for start, stop, kind in segments:
        if int(start) != offset:
            raise ValueError(
                f"segments are not contiguous: {kind!r} starts at {start}, expected {offset}"
            )
        offset = int(stop)
    if offset != seq_len:
        raise ValueError(f"segments end at {offset} but seq_len is {seq_len}")
