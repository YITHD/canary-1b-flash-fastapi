"""Render an :class:`InferenceResult` into the various OpenAI response formats."""

from __future__ import annotations

from .model import InferenceResult, TimedSpan
from .schemas import Segment, VerboseTranscriptionResponse, Word


def _fmt_timestamp(seconds: float, sep: str) -> str:
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def _segments_for_subtitles(result: InferenceResult) -> list[TimedSpan]:
    if result.segments:
        return result.segments
    # Fall back to a single span covering the whole utterance.
    end = result.words[-1].end if result.words else 0.0
    return [TimedSpan(text=result.text, start=0.0, end=end)]


def to_srt(result: InferenceResult) -> str:
    lines: list[str] = []
    for idx, span in enumerate(_segments_for_subtitles(result), start=1):
        start = _fmt_timestamp(span.start, ",")
        end = _fmt_timestamp(span.end, ",")
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(span.text.strip())
        lines.append("")
    return "\n".join(lines)


def to_vtt(result: InferenceResult) -> str:
    lines = ["WEBVTT", ""]
    for span in _segments_for_subtitles(result):
        start = _fmt_timestamp(span.start, ".")
        end = _fmt_timestamp(span.end, ".")
        lines.append(f"{start} --> {end}")
        lines.append(span.text.strip())
        lines.append("")
    return "\n".join(lines)


def to_verbose(
    result: InferenceResult,
    *,
    task: str,
    language: str,
    duration: float,
    include_words: bool,
    include_segments: bool,
) -> VerboseTranscriptionResponse:
    words = None
    if include_words and result.words:
        words = [Word(word=w.text, start=w.start, end=w.end) for w in result.words]

    segments = None
    if include_segments:
        spans = _segments_for_subtitles(result)
        segments = [
            Segment(id=i, start=s.start, end=s.end, text=s.text.strip())
            for i, s in enumerate(spans)
        ]

    return VerboseTranscriptionResponse(
        task=task,
        language=language,
        duration=duration,
        text=result.text,
        words=words,
        segments=segments,
    )
