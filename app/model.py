"""Thin wrapper around the NeMo Canary-1B-Flash model.

Loading NeMo is expensive and the model is not thread-safe for concurrent
GPU calls, so we load once at startup and serialise inference behind a lock.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from .config import Settings

logger = logging.getLogger("canary.model")


@dataclass
class TimedSpan:
    text: str
    start: float
    end: float


@dataclass
class InferenceResult:
    text: str
    words: list[TimedSpan] = field(default_factory=list)
    segments: list[TimedSpan] = field(default_factory=list)


def _coerce_time(entry: dict, *keys: str) -> float:
    for key in keys:
        if key in entry and entry[key] is not None:
            try:
                return float(entry[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _parse_spans(entries, text_keys: tuple[str, ...]) -> list[TimedSpan]:
    spans: list[TimedSpan] = []
    if not entries:
        return spans
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = ""
        for key in text_keys:
            if entry.get(key):
                text = str(entry[key])
                break
        spans.append(
            TimedSpan(
                text=text,
                start=_coerce_time(entry, "start", "start_time", "start_offset"),
                end=_coerce_time(entry, "end", "end_time", "end_offset"),
            )
        )
    return spans


class CanaryModel:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        from nemo.collections.asr.models import EncDecMultiTaskModel

        device = self._settings.resolved_device
        logger.info(
            "Loading %s on %s ...", self._settings.model_name, device
        )
        model = EncDecMultiTaskModel.from_pretrained(self._settings.model_name)

        # Configure decoding (greedy by default for speed/determinism).
        try:
            decode_cfg = model.cfg.decoding
            decode_cfg.beam.beam_size = self._settings.beam_size
            model.change_decoding_strategy(decode_cfg)
        except Exception as exc:  # noqa: BLE001 - non-fatal
            logger.warning("Could not set decoding strategy: %s", exc)

        model = model.to(device)
        model.eval()
        self._apply_dtype(model, device)

        self._model = model
        logger.info("Model loaded.")

    def _apply_dtype(self, model, device: str) -> None:
        dtype = self._settings.compute_dtype.lower()
        if device != "cuda" or dtype == "float32":
            return
        try:
            import torch

            target = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            model.to(dtype=target)
        except Exception as exc:  # noqa: BLE001 - non-fatal
            logger.warning("Could not cast model to %s: %s", dtype, exc)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe(
        self,
        wav_path: str,
        *,
        source_lang: str,
        target_lang: str,
        pnc: bool,
        timestamps: bool,
    ) -> InferenceResult:
        if self._model is None:
            raise RuntimeError("Model is not loaded.")

        task = "asr" if source_lang == target_lang else "ast"
        kwargs = dict(
            audio=[wav_path],
            batch_size=self._settings.batch_size,
            task=task,
            source_lang=source_lang,
            target_lang=target_lang,
            pnc="yes" if pnc else "no",
            timestamps="yes" if timestamps else "no",
        )

        with self._lock:
            outputs = self._model.transcribe(**kwargs)

        return self._to_result(outputs, timestamps)

    @staticmethod
    def _to_result(outputs, timestamps: bool) -> InferenceResult:
        if not outputs:
            return InferenceResult(text="")
        hyp = outputs[0]

        text = getattr(hyp, "text", None)
        if text is None:
            text = str(hyp)
        result = InferenceResult(text=text.strip())

        if timestamps:
            ts = getattr(hyp, "timestamp", None) or {}
            if isinstance(ts, dict):
                result.words = _parse_spans(ts.get("word"), ("word", "char", "text"))
                result.segments = _parse_spans(
                    ts.get("segment"), ("segment", "text")
                )
        return result
