# Canary-1B-Flash — OpenAI-compatible API

A [FastAPI](https://fastapi.tiangolo.com/) server that exposes NVIDIA's
[`nvidia/canary-1b-flash`](https://huggingface.co/nvidia/canary-1b-flash)
speech model behind the **OpenAI audio API**. Point any OpenAI client at it and
get speech-to-text transcription and speech translation.

Canary-1B-Flash is a multitask encoder-decoder model that does automatic
speech recognition (ASR) and speech-to-text translation (AST) across
**English, German, French, and Spanish**, with word- and segment-level
timestamps.

## Endpoints

| Method | Path                       | OpenAI equivalent | Description |
|--------|----------------------------|-------------------|-------------|
| `POST` | `/v1/audio/transcriptions` | transcriptions    | Speech → text in the **same** language |
| `POST` | `/v1/audio/translations`   | translations      | Speech → text in **another** language (defaults to English) |
| `GET`  | `/v1/models`               | models            | List the served model |
| `GET`  | `/health`                  | —                 | Liveness + model-loaded probe |

### Supported request fields

`file` (required), `model`, `language`, `response_format`, `temperature`,
`prompt`, and `timestamp_granularities[]`. `temperature` and `prompt` are
accepted for compatibility but are not used by the model.

`response_format` supports `json` (default), `text`, `verbose_json`, `srt`,
and `vtt`. Word timestamps appear in `verbose_json` when you request
`timestamp_granularities[]=word`; segment timestamps drive `srt`/`vtt`.

## Quick start

```bash
pip install -r requirements.txt          # installs NeMo + torch (large)
# System deps for audio decoding:
#   apt-get install ffmpeg libsndfile1
python -m app.main                        # serves on 0.0.0.0:8000
```

First start downloads the model from Hugging Face (~1B params). A CUDA GPU is
strongly recommended; CPU works but is slow.

### Docker

```bash
docker build -t canary-flash .
docker run --gpus all -p 8000:8000 canary-flash
```

## Usage

### curl

```bash
# Transcribe English audio
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=canary-1b-flash \
  -F language=en

# Translate Spanish speech to English with subtitles
curl http://localhost:8000/v1/audio/translations \
  -F file=@spanish.wav \
  -F language=es \
  -F response_format=srt
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with open("sample.wav", "rb") as f:
    out = client.audio.transcriptions.create(
        model="canary-1b-flash",
        file=f,
        language="en",
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
print(out.text)
```

## Configuration

All settings are environment variables prefixed with `CANARY_` (see
[`.env.example`](.env.example)). Notable ones:

| Variable | Default | Notes |
|----------|---------|-------|
| `CANARY_DEVICE` | `auto` | `auto` \| `cuda` \| `cpu` |
| `CANARY_COMPUTE_DTYPE` | `float32` | `float16`/`bfloat16` on GPU saves memory |
| `CANARY_BEAM_SIZE` | `1` | greedy decoding by default |
| `CANARY_API_KEY` | _(empty)_ | when set, requires `Authorization: Bearer <key>` |
| `CANARY_PORT` | `8000` | listen port |

## Notes on translation

OpenAI's `/translations` endpoint always targets English and auto-detects the
source language. Canary needs the source language explicitly, so pass it via
the `language` form field (e.g. `language=de`). Set `target_lang` to translate
into a non-English language (en ↔ de/fr/es is supported).

## Tests

The HTTP layer can be tested without NeMo, torch, or a GPU — the model is
mocked and loading is skipped:

```bash
pip install fastapi uvicorn python-multipart pydantic pydantic-settings \
            soundfile librosa numpy pytest httpx
CANARY_SKIP_MODEL_LOAD=1 pytest
```

## License

The model is governed by NVIDIA's license on the Hugging Face model card.
This server code is provided as-is.
