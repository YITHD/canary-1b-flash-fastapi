# NeMo needs a recent CUDA + PyTorch base for GPU inference.
FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg + libsndfile let librosa/soundfile decode mp3/m4a/ogg/etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["python", "-m", "app.main"]
