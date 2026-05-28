"""Pydantic schemas mirroring the OpenAI audio API surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]


class Word(BaseModel):
    word: str
    start: float
    end: float


class Segment(BaseModel):
    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    # The fields below are part of the OpenAI schema. Canary does not produce
    # them, so we emit OpenAI-compatible defaults.
    tokens: list[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class TranscriptionResponse(BaseModel):
    """`json` response_format."""

    text: str


class VerboseTranscriptionResponse(BaseModel):
    """`verbose_json` response_format."""

    task: str
    language: str
    duration: float
    text: str
    words: list[Word] | None = None
    segments: list[Segment] | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "nvidia"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
