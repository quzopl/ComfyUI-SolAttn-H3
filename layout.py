"""Zakres sinka wyliczony z tabeli segmentow PackedLayout.

Sink to ciagly zakres K/V, ktory kernel trzyma dokladnym dla wszystkich zapytan,
z granulacja blokow 64-tokenowych (zaokraglajac na zewnatrz). Referencja NVIDII
musiala wnioskowac poczatek ogona wideo z nieciaglosci `video_indices`; ComfyUI
podaje etykietowane segmenty, wiec bierzemy go wprost.
"""
from __future__ import annotations

from dataclasses import dataclass

SINK_MODES = ("prefix", "text")


@dataclass(frozen=True)
class SinkRange:
    """Ciagly zakres K/V trzymany dokladnie, plus granica ogona wideo."""

    start: int
    tokens: int
    seq_len: int
    video_start: int

    @property
    def stop(self) -> int:
        return self.start + self.tokens


def sink_from_segments(segments, seq_len: int, sink_mode: str) -> SinkRange:
    """Wyznacz sink dla podanego trybu.

    `segments` to lista (start, stop, kind) z `PackedLayout.segments`. Docelowe
    audio i wideo sa zawsze dwoma ostatnimi segmentami, a segmentow `cond` /
    `ref_img` moze byc wiele.

    `prefix` obejmuje wszystko przed docelowym wideo — tekst, wiersze
    warunkujace i audio. Wiersze audio sa *generowane* (model zwraca dla nich
    predkosc), a handoff NVIDII odnotowal prompt, w ktorym obraz wyszedl
    najlepiej w zestawie, podczas gdy dialog sie rozpadl. Stad prefix domyslnie,
    mimo ze polityka referencyjna obejmuje sinkiem sam tekst.
    """
    if sink_mode not in SINK_MODES:
        raise ValueError(f"nieznany sink_mode {sink_mode!r}; dozwolone: {SINK_MODES}")

    _check_contiguous(segments, seq_len)

    video = [seg for seg in segments if seg[2] == "video"]
    if len(video) != 1:
        raise ValueError(
            f"oczekiwano dokladnie jednego segmentu 'video', znaleziono {len(video)}; "
            "kontrakt PackedLayout sie zmienil"
        )
    video_start = int(video[0][0])

    if sink_mode == "prefix":
        return SinkRange(0, video_start, seq_len, video_start)

    text = [seg for seg in segments if seg[2] == "text"]
    if not text:
        # Sink pusty jest legalny — kernel nie ma czego trzymac dokladnie.
        return SinkRange(0, 0, seq_len, video_start)
    start, stop = int(text[0][0]), int(text[-1][1])
    return SinkRange(start, stop - start, seq_len, video_start)


def _check_contiguous(segments, seq_len: int) -> None:
    """Segmenty musza byc ciagle i pokrywac cala sekwencje."""
    if not segments:
        raise ValueError("pusta tabela segmentow")
    offset = 0
    for start, stop, kind in segments:
        if int(start) != offset:
            raise ValueError(
                f"segmenty nie sa ciagle: {kind!r} zaczyna sie na {start}, oczekiwano {offset}"
            )
        offset = int(stop)
    if offset != seq_len:
        raise ValueError(f"segmenty koncza sie na {offset}, a seq_len to {seq_len}")
