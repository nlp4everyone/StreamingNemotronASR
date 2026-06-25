# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build — không cần NGC account
#
# Stage 1 · builder  : cuda-devel  — nvcc để compile CUDA extensions của NeMo
# Stage 2 · runtime  : cuda-runtime — chỉ CUDA runtime libs, không có compiler
#
# Tiết kiệm ~4-5 GB so với single-stage devel image.
# ─────────────────────────────────────────────────────────────────────────────

# ══ Stage 1: builder ══════════════════════════════════════════════════════════
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Dev packages: nvcc (có sẵn trong devel), Python headers, libsndfile headers
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Virtualenv tại /venv — sẽ được copy nguyên sang runtime stage
RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# PyTorch với CUDA 12.4 (pre-built wheel, không cần nvcc)
#RUN pip install \
#    torch==2.4.0 torchaudio==2.12.0 \
#    --index-url https://download.pytorch.org/whl/cu124

# NeMo ASR — cần nvcc để build một số CUDA kernels
# nemotron-3.5-asr-streaming-0.6b yêu cầu NeMo 26.06+, chưa có trên PyPI → cài từ main
RUN pip install Cython packaging
RUN pip install \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"
#RUN pip install nemo_toolkit['asr']==2.7.2
RUN pip install \
    torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124



# ══ Stage 2: runtime ══════════════════════════════════════════════════════════
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/venv/bin:$PATH" \
    # libtorchaudio.so links against torch bundled libs — phải có trong LD_LIBRARY_PATH
    LD_LIBRARY_PATH="/venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH" \
    HF_HOME=/cache/huggingface \
    NEMO_CACHE_DIR=/cache/nemo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime system deps — không có nvcc, CUDA headers, libsndfile-dev, git
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy toàn bộ virtualenv từ builder — .so đã compile với CUDA 12.4, tương thích runtime
COPY --from=builder /venv /venv

# Rebuild ldconfig cache sau khi copy venv — dynamic linker tìm được các .so mới
RUN ldconfig

# App dependencies — pure Python, không cần nvcc, cài thẳng vào venv đã copy
COPY requirements.txt /tmp/requirements.txt
RUN /venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY . .

EXPOSE 8000

# start_period 180s: lần đầu chạy tải model từ HuggingFace
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["/venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--ws-ping-interval", "20", "--ws-ping-timeout", "30"]
