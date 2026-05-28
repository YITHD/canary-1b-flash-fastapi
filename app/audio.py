"""Audio decoding / resampling helpers.

Uploaded audio can be any format ffmpeg/libsndfile understands. Canary wants
16 kHz mono float32 PCM, so everything is normalised to a temporary wav file
before being handed to NeMo.
"""

from __future__ import annotations

import os
import tempfile

import librosa
import soundfile as sf


class AudioError(ValueError):
    """Raised when an upload cannot be decoded as audio."""


def load_and_resample(raw: bytes, suffix: str, target_sr: int) -> tuple[str, float]:
    """Decode ``raw`` bytes, resample to mono ``target_sr``, write a temp wav.

    Returns ``(wav_path, duration_seconds)``. The caller owns the returned file
    and must delete it (see :func:`cleanup`).
    """
    # librosa.load goes through soundfile, then audioread/ffmpeg as a fallback,
    # so it handles wav/flac/ogg natively and mp3/m4a/webm via ffmpeg.
    src_path = _spill_to_disk(raw, suffix)
    try:
        try:
            audio, _ = librosa.load(src_path, sr=target_sr, mono=True)
        except Exception as exc:  # noqa: BLE001 - surface a clean 400
            raise AudioError(f"Could not decode audio file: {exc}") from exc

        if audio.size == 0:
            raise AudioError("Audio file contains no samples.")

        duration = float(len(audio) / target_sr)

        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(wav_path, audio, target_sr, subtype="PCM_16")
        return wav_path, duration
    finally:
        cleanup(src_path)


def _spill_to_disk(raw: bytes, suffix: str) -> str:
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}" if suffix else ""
    fd, path = tempfile.mkstemp(suffix=safe_suffix)
    with os.fdopen(fd, "wb") as fh:
        fh.write(raw)
    return path


def cleanup(*paths: str) -> None:
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
