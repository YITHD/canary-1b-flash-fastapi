"""Runtime configuration, loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CANARY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model
    model_name: str = "nvidia/canary-1b-flash"
    # Identifier reported through the OpenAI-style API (the `model` field).
    served_model_name: str = "canary-1b-flash"
    # "cuda", "cpu", or "auto" (cuda if available, else cpu).
    device: str = "auto"
    # Cast the model to float16 / bfloat16 on GPU to save memory.
    compute_dtype: str = "float32"  # one of: float32, float16, bfloat16

    # Inference
    beam_size: int = 1
    batch_size: int = 1
    # Sample rate Canary expects. Uploaded audio is resampled to this.
    target_sample_rate: int = 16000
    # Max upload size (bytes). OpenAI's limit is 25 MB; default a bit higher.
    max_upload_bytes: int = 100 * 1024 * 1024

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    # Optional bearer token. When set, requests must send
    # `Authorization: Bearer <token>`. Empty disables auth.
    api_key: str = ""

    # Languages Canary-1b-flash supports (ASR + X<->X translation).
    supported_languages: tuple[str, ...] = ("en", "de", "fr", "es")

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()
