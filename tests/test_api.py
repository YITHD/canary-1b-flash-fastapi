"""Smoke tests that exercise the HTTP layer with the model mocked out.

Run with: CANARY_SKIP_MODEL_LOAD=1 pytest
These do not require NeMo, torch, or a GPU.
"""

from __future__ import annotations

import io
import os
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

os.environ["CANARY_SKIP_MODEL_LOAD"] = "1"

from app import main  # noqa: E402
from app.model import InferenceResult, TimedSpan  # noqa: E402


class FakeModel:
    is_loaded = True

    def transcribe(self, wav_path, *, source_lang, target_lang, pnc, timestamps):
        words = [TimedSpan("hello", 0.0, 0.5), TimedSpan("world", 0.5, 1.0)]
        segments = [TimedSpan("hello world", 0.0, 1.0)]
        return InferenceResult(
            text="hello world",
            words=words if timestamps else [],
            segments=segments if timestamps else [],
        )


def _wav_bytes(seconds: float = 1.0, sr: int = 16000) -> bytes:
    samples = (np.random.rand(int(seconds * sr)) * 0.1).astype(np.float32)
    pcm = (samples * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        main.app.state.model = FakeModel()
        yield c


def _upload(name="audio.wav"):
    return {"file": (name, _wav_bytes(), "audio/wav")}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "canary-1b-flash"


def test_transcription_json(client):
    r = client.post("/v1/audio/transcriptions", files=_upload())
    assert r.status_code == 200
    assert r.json() == {"text": "hello world"}


def test_transcription_text(client):
    r = client.post(
        "/v1/audio/transcriptions",
        files=_upload(),
        data={"response_format": "text"},
    )
    assert r.status_code == 200
    assert r.text == "hello world"


def test_transcription_verbose_json_words(client):
    r = client.post(
        "/v1/audio/transcriptions",
        files=_upload(),
        data={"response_format": "verbose_json", "timestamp_granularities[]": "word"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "transcribe"
    assert body["words"][0]["word"] == "hello"


def test_srt(client):
    r = client.post(
        "/v1/audio/transcriptions",
        files=_upload(),
        data={"response_format": "srt"},
    )
    assert r.status_code == 200
    assert "00:00:00,000 --> 00:00:01,000" in r.text


def test_vtt(client):
    r = client.post(
        "/v1/audio/translations",
        files=_upload(),
        data={"response_format": "vtt", "language": "es"},
    )
    assert r.status_code == 200
    assert r.text.startswith("WEBVTT")


def test_unsupported_language(client):
    r = client.post(
        "/v1/audio/transcriptions",
        files=_upload(),
        data={"language": "zz"},
    )
    assert r.status_code == 400


def test_bad_response_format(client):
    r = client.post(
        "/v1/audio/transcriptions",
        files=_upload(),
        data={"response_format": "bogus"},
    )
    assert r.status_code == 400


def test_empty_upload(client):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert r.status_code == 400
