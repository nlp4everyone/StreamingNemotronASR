# 🎙️ StreamingNenotronASR

A production-ready multi-user streaming Speech-to-Text framework built on FastAPI and WebSocket with low-latency real-time transcription.

<br />

## Features

- **Real-time streaming** — WebSocket endpoint; each session fully isolated; returns partial and final transcripts continuously after each inference chunk
- **Cache-aware inference** — model maintains acoustic context via `att_cache` and `conv_cache`; no re-encoding from scratch per chunk; significantly lower latency than stateless approaches
- **Streaming presets** — 5 profiles from `ultra_low` (80ms) to `high` (1120ms); switch by changing a single env var, no code changes needed
- **Batch scheduling** — supports `per_session` (infer immediately, minimum latency) and `dynamic` (batch multiple sessions into one GPU call, higher throughput)
- **Non-blocking pipeline** — asyncio event loop handles all I/O; GPU inference offloaded to ThreadPoolExecutor, does not block WebSocket connections
- **Multi-user** — each session holds its own cache tensors; VRAM released immediately on disconnect, no waiting for GC; hard limit via `ASR_MAX_SESSIONS`
- **True GPU batching** — `dynamic` mode stacks sessions into a single `[B, D, T]` forward pass; groups by language and decoder state for homogeneous batches
- **Idle session cleanup** — background sweeper evicts ghost sessions (TCP alive, no audio) after `idle_timeout_s`; complements WS ping/pong for dead TCP
- **Bounded inference queue** — non-final chunks beyond `max_pending_per_session` are dropped; prevents latency spiral under GPU load
- **bfloat16 precision** — optional `use_bf16` mode halves model VRAM on Ampere+ GPUs; preprocessor stays float32, mel cast at encoder boundary; falls back gracefully if GPU does not support it
- **Runtime metrics** — `GET /health/stats` returns queue depth, drop count, inference count, rolling `avg_batch_size`, `avg_gpu_batch_size` (post group-split), and `batch_latency_ms` (p50/p99)

<br />

## Setup

```bash
git clone https://github.com/nlp4everyone/StreamingNenotronASR.git
cd StreamingNenotronASR/
cp .env.example .env
```

Configure in `.env`:
```
APP_PORT=8010

ASR_STREAMING_PRESET=balanced
ASR_BATCH_MODE=dynamic
ASR_MAX_SESSIONS=200
ASR_DEVICE=cuda
ASR_DEFAULT_LANG=auto
```

> Algorithm parameters (preset, batch mode, session limit) go in `.env`. Model weights are downloaded automatically from HuggingFace on first startup.

Pull model weights (run once):
```bash
make pull-model
```

Build and run with Docker:
```bash
make build   # ~20 min on first run, cached after
make run
```

Server is available at `ws://localhost:8010/ws/stream`. Health check: `GET http://localhost:8010/health`.

<br />

## Quick Example

`scripts/stream_client.py` streams a WAV file to the server in 20ms packets and prints partial/final transcripts as they arrive.

```bash
# Vietnamese
python scripts/stream_client.py --file resources/sample_vi.wav

# English
python scripts/stream_client.py --file resources/sample_en.wav

# Explicit language and URL
python scripts/stream_client.py --file resources/sample_vi.wav --lang vi --url ws://localhost:8010/ws/stream
```

Sample output:
```
[connected] session=a1b2c3d4  preset=balanced  chunk=560ms  packets/chunk=28
[partial] 'hello'
[partial] 'hello world'
[FINAL  ] 'hello world how are you'
```

Or via Makefile:
```bash
make test-client WAV=resources/sample_vi.wav
```

<br />

## Integration

- **API**: FastAPI + WebSocket
- **Runtime**: Docker Compose (multi-stage CUDA build)
- **ASR**: [NVIDIA Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) — cache-aware streaming model, runs directly in container
- **Inference**: NeMo Framework + PyTorch (CUDA)
- **Audio I/O**: 16kHz mono int16 PCM, base64-encoded JSON frames

<br />

## Documentation

- [Architecture Overview](ARCHITECTURE.md) — architecture diagram, processing pipeline, WebSocket protocol
- [Design Decisions](DESIGN_DECISIONS.md) — rationale behind each major architectural choice
- [Detailed Components](DETAILED_COMPONENTS.md) — per-module behavior, interfaces, and edge cases
- [Project Structure](../PROJECT_STRUCTURE.md) — directory tree and per-file roles

<br />

## To-Do / Roadmap

### 🔌 Transport & API
- [x] WebSocket endpoint with typed JSON frames (`start` / `audio` / `end`)
- [x] Health check HTTP endpoint (`/health`)
- [x] Runtime metrics endpoint (`/health/stats`) — queue depth, drop/inference counts, batch size and latency histograms
- [x] Session capacity limit with WebSocket close code 1008

### 🤖 ASR & Inference
- [x] Cache-aware streaming inference (NeMo `att_cache` + `conv_cache`)
- [x] Per-session cache isolation + deterministic VRAM release
- [x] Streaming presets (`ultra_low` → `high`) — switch without rebuild
- [x] Batch scheduler: `per_session` and `dynamic` mode
- [x] True GPU batching in `dynamic` mode (`[B, D, T]` forward pass)
- [x] Model pool — eliminates lang-prompt race condition for multi-language
- [x] bfloat16 precision mode — optional VRAM reduction on Ampere+; graceful fallback
- [ ] INT8 quantization to reduce VRAM per session

### 🎤 Audio Pipeline
- [x] AudioChunkBuffer — accumulator with zero-padding
- [x] EOS timeout auto-flushes final chunk after silence
- [ ] VAD (Silero) to skip inference on silence, save GPU

### 🛡️ Resilience
- [x] Graceful shutdown + VRAM cleanup on SIGTERM
- [x] Multi-stage Docker build (builder `cuda-devel` → runtime `cuda-runtime`)
- [x] WS ping/pong for dead TCP detection + idle sweeper for ghost sessions
- [x] Bounded inference queue per session (`max_pending_per_session`)
- [ ] Persist completed transcripts to storage

<br />

## Model Citation

This project uses **NVIDIA Nemotron 3.5 ASR Streaming 0.6B**:
➡️ https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
