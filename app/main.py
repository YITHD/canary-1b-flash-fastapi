"""OpenAI-compatible HTTP server for NVIDIA Canary-1B-Flash.

Implements the subset of the OpenAI audio API that Canary can serve:

    POST /v1/audio/transcriptions   (ASR: speech -> text, same language)
    POST /v1/audio/translations     (AST: speech -> text in another language)
    GET  /v1/models
    GET  /health

The server is a drop-in target for OpenAI SDKs/clients: point ``base_url`` at
this host and set ``model`` to ``canary-1b-flash``.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__, audio, formatters
from .config import Settings, get_settings
from .model import CanaryModel
from .schemas import (
    ModelCard,
    ModelList,
    ResponseFormat,
    TranscriptionResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("canary.server")

VALID_FORMATS = {"json", "text", "srt", "verbose_json", "vtt"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    model = CanaryModel(settings)
    if os.getenv("CANARY_SKIP_MODEL_LOAD") == "1":
        logger.warning("CANARY_SKIP_MODEL_LOAD=1 set; model will NOT be loaded.")
    else:
        model.load()
    app.state.model = model
    app.state.settings = settings
    yield


app = FastAPI(
    title="Canary-1B-Flash OpenAI-compatible API",
    version=__version__,
    lifespan=lifespan,
)


def require_auth(
    authorization: Optional[str] = Header(default=None),
) -> None:
    settings: Settings = app.state.settings
    if not settings.api_key:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key.")


def _model() -> CanaryModel:
    model: CanaryModel = getattr(app.state, "model", None)
    if model is None or not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return model


def _normalise_lang(value: Optional[str], settings: Settings, default: str) -> str:
    lang = (value or default).strip().lower()
    # OpenAI clients may pass ISO-639-1 ("en") or full names; take the prefix.
    lang = lang.split("-")[0][:2] if lang else default
    if lang not in settings.supported_languages:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported language '{value}'. Supported: "
                f"{', '.join(settings.supported_languages)}."
            ),
        )
    return lang


def _check_format(response_format: str) -> ResponseFormat:
    if response_format not in VALID_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown response_format '{response_format}'.",
        )
    return response_format  # type: ignore[return-value]


async def _read_upload(file: UploadFile, settings: Settings) -> bytes:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Audio file too large.")
    return raw


def _suffix(filename: Optional[str]) -> str:
    if not filename:
        return ".wav"
    _, ext = os.path.splitext(filename)
    return ext or ".wav"


def _run(
    *,
    file: UploadFile,
    raw: bytes,
    source_lang: str,
    target_lang: str,
    response_format: ResponseFormat,
    timestamp_granularities: list[str],
    task: str,
    settings: Settings,
):
    want_words = "word" in timestamp_granularities
    want_segments = "segment" in timestamp_granularities or not want_words
    need_timestamps = response_format in {"srt", "vtt"} or (
        response_format == "verbose_json"
    )

    wav_path, duration = audio.load_and_resample(
        raw, _suffix(file.filename), settings.target_sample_rate
    )
    try:
        result = _model().transcribe(
            wav_path,
            source_lang=source_lang,
            target_lang=target_lang,
            pnc=True,
            timestamps=need_timestamps,
        )
    finally:
        audio.cleanup(wav_path)

    if response_format == "text":
        return PlainTextResponse(result.text)
    if response_format == "srt":
        return PlainTextResponse(formatters.to_srt(result), media_type="text/plain")
    if response_format == "vtt":
        return PlainTextResponse(formatters.to_vtt(result), media_type="text/vtt")
    if response_format == "verbose_json":
        verbose = formatters.to_verbose(
            result,
            task="transcribe" if task == "asr" else "translate",
            language=source_lang,
            duration=duration,
            include_words=want_words,
            include_segments=want_segments,
        )
        return JSONResponse(verbose.model_dump(exclude_none=True))

    return JSONResponse(TranscriptionResponse(text=result.text).model_dump())


@app.get("/health")
def health():
    model: CanaryModel = getattr(app.state, "model", None)
    return {"status": "ok", "model_loaded": bool(model and model.is_loaded)}


@app.get("/v1/models", response_model=ModelList)
def list_models(_: None = Depends(require_auth)):
    settings: Settings = app.state.settings
    card = ModelCard(id=settings.served_model_name, created=int(time.time()))
    return ModelList(data=[card])


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    _: None = Depends(require_auth),
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str = Form(default="en"),
    prompt: str = Form(default=""),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: list[str] = Form(default=[], alias="timestamp_granularities[]"),
):
    """Transcribe audio in its source language (OpenAI `transcriptions`)."""
    settings: Settings = app.state.settings
    fmt = _check_format(response_format)
    src = _normalise_lang(language, settings, default="en")
    raw = await _read_upload(file, settings)
    return _run(
        file=file,
        raw=raw,
        source_lang=src,
        target_lang=src,
        response_format=fmt,
        timestamp_granularities=timestamp_granularities,
        task="asr",
        settings=settings,
    )


@app.post("/v1/audio/translations")
async def translations(
    _: None = Depends(require_auth),
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str = Form(default=""),
    target_lang: str = Form(default="en"),
    prompt: str = Form(default=""),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: list[str] = Form(default=[], alias="timestamp_granularities[]"),
):
    """Translate speech into another language.

    OpenAI's endpoint always targets English. Canary supports en<->{de,fr,es},
    so the source language must be supplied via the (non-standard but
    compatible) ``language`` form field; ``target_lang`` defaults to English.
    """
    settings: Settings = app.state.settings
    fmt = _check_format(response_format)
    # Source defaults to a non-English language since target is English.
    default_src = next(
        (l for l in settings.supported_languages if l != "en"),
        "en",
    )
    src = _normalise_lang(language, settings, default=default_src)
    tgt = _normalise_lang(target_lang, settings, default="en")
    raw = await _read_upload(file, settings)
    return _run(
        file=file,
        raw=raw,
        source_lang=src,
        target_lang=tgt,
        response_format=fmt,
        timestamp_granularities=timestamp_granularities,
        task="ast",
        settings=settings,
    )


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
