# 🎙️ StreamingNenotronASR

Framework nhận dạng giọng nói theo thời gian thực (Speech-to-Text), hỗ trợ nhiều người dùng đồng thời, xây dựng trên FastAPI và WebSocket với độ trễ thấp.

<br />

## Tính năng nổi bật

- **Streaming thời gian thực** — endpoint WebSocket, mỗi session độc lập hoàn toàn, trả về partial và final transcript liên tục sau mỗi chunk inference
- **Cache-aware inference** — model giữ acoustic context qua `att_cache` và `conv_cache`; không re-encode lại từ đầu mỗi chunk, latency thấp hơn đáng kể so với stateless approach
- **Streaming presets** — 5 profile từ `ultra_low` (80ms) đến `high` (1120ms); đổi preset chỉ cần thay một biến môi trường, không cần sửa code
- **Batch scheduling** — hỗ trợ `per_session` (inference ngay, latency tối thiểu) và `dynamic` (gom nhiều session vào một GPU call, tăng throughput)
- **Non-blocking pipeline** — asyncio event loop xử lý toàn bộ I/O; GPU inference offload sang ThreadPoolExecutor, không block WebSocket connections
- **Đa người dùng** — mỗi session giữ riêng cache tensors; VRAM được giải phóng ngay khi disconnect, không chờ GC; hard limit qua `ASR_MAX_SESSIONS`
- **True GPU batching** — `dynamic` mode stack sessions vào một forward pass `[B, D, T]`; nhóm theo ngôn ngữ và decoder state
- **Idle session cleanup** — sweeper evict ghost sessions (TCP alive, không có audio) sau `idle_timeout_s`; bổ sung WS ping/pong
- **Bounded inference queue** — non-final chunks vượt `max_pending_per_session` bị drop; ngăn latency spiral
- **Chế độ bfloat16** — tùy chọn `use_bf16` giảm một nửa VRAM model trên GPU Ampere+; preprocessor giữ float32, mel cast tại ranh giới encoder; fallback tự động nếu GPU không hỗ trợ
- **Runtime metrics** — `GET /health/stats` trả về queue depth, drop count, inference count, rolling `avg_batch_size`, `avg_gpu_batch_size` (sau group-split), và `batch_latency_ms` (p50/p99)

<br />

## Cài đặt

```bash
git clone https://github.com/nlp4everyone/StreamingNenotronASR.git
cd StreamingNenotronASR/
cp .env.example .env
```

Cấu hình trong `.env`:
```
APP_PORT=8010

ASR_STREAMING_PRESET=balanced
ASR_BATCH_MODE=dynamic
ASR_MAX_SESSIONS=200
ASR_DEVICE=cuda
ASR_DEFAULT_LANG=auto
```

> Các tham số thuật toán (preset, batch mode, session limit) đặt trong `.env`. Model weights được tải tự động từ HuggingFace khi khởi động lần đầu.

Tải model weights trước (chỉ chạy một lần):
```bash
make pull-model
```

Build và chạy bằng Docker:
```bash
make build   # ~20 phút lần đầu, cached sau đó
make run
```

Server sẵn sàng tại `ws://localhost:8010/ws/stream`. Health check: `GET http://localhost:8010/health`.

<br />

## Ví dụ nhanh

`scripts/stream_client.py` stream một file WAV đến server theo từng packet 20ms và in partial/final transcript khi nhận được.

```bash
# Tiếng Việt
python scripts/stream_client.py --file resources/sample_vi.wav

# Tiếng Anh
python scripts/stream_client.py --file resources/sample_en.wav

# Chỉ định ngôn ngữ và URL tường minh
python scripts/stream_client.py --file resources/sample_vi.wav --lang vi --url ws://localhost:8010/ws/stream
```

Output mẫu:
```
[connected] session=a1b2c3d4  preset=balanced  chunk=560ms  packets/chunk=28
[partial] 'xin'
[partial] 'xin chào'
[FINAL  ] 'xin chào bạn'
```

Hoặc dùng Makefile:
```bash
make test-client WAV=resources/sample_vi.wav
```

<br />

## Tích hợp

- **API**: FastAPI + WebSocket
- **Runtime**: Docker Compose (multi-stage CUDA build)
- **ASR**: [NVIDIA Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) — cache-aware streaming model, chạy trực tiếp trong container
- **Inference**: NeMo Framework + PyTorch (CUDA)
- **Audio I/O**: 16kHz mono int16 PCM, base64-encoded JSON frames

<br />

## Tài liệu

- [Kiến trúc tổng quan](ARCHITECTURE_vi.md) — sơ đồ kiến trúc, pipeline xử lý, WebSocket protocol
- [Quyết định kỹ thuật](DESIGN_DECISIONS_vi.md) — lý do đằng sau từng lựa chọn kiến trúc quan trọng
- [Mô tả component chi tiết](DETAILED_COMPONENTS_vi.md) — hành vi từng module, interface, và edge cases
- [Cấu trúc project](../PROJECT_STRUCTURE_vi.md) — sơ đồ thư mục và vai trò từng file

<br />

## To-Do / Roadmap

### 🔌 Transport & API
- [x] WebSocket endpoint với typed JSON frames (`start` / `audio` / `end`)
- [x] Health check HTTP endpoint (`/health`)
- [x] Runtime metrics endpoint (`/health/stats`) — queue depth, drop/inference counts, batch size và latency histograms
- [x] Session capacity limit với WebSocket close code 1008

### 🤖 ASR & Inference
- [x] Cache-aware streaming inference (NeMo `att_cache` + `conv_cache`)
- [x] Per-session cache isolation + deterministic VRAM release
- [x] Streaming presets (`ultra_low` → `high`) — đổi không cần rebuild
- [x] Batch scheduler: `per_session` và `dynamic` mode
- [x] True GPU batching trong `dynamic` mode (forward pass `[B, D, T]`)
- [x] Model pool — loại bỏ race condition lang-prompt đa ngôn ngữ
- [x] Chế độ bfloat16 — tùy chọn giảm VRAM trên Ampere+; fallback tự động
- [ ] INT8 quantization để giảm VRAM per session

### 🎤 Audio Pipeline
- [x] AudioChunkBuffer — accumulator với zero-padding
- [x] EOS timeout tự động flush chunk cuối sau khoảng lặng
- [ ] VAD (Silero) để skip inference khi silence, tiết kiệm GPU

### 🛡️ Khả năng chịu lỗi
- [x] Graceful shutdown + VRAM cleanup khi nhận SIGTERM
- [x] Multi-stage Docker build (builder `cuda-devel` → runtime `cuda-runtime`)
- [x] WS ping/pong cho dead TCP + idle sweeper cho ghost sessions
- [x] Bounded inference queue per session (`max_pending_per_session`)
- [ ] Lưu transcript đã hoàn thành xuống storage

<br />

## Trích dẫn mô hình

Dự án sử dụng mô hình **NVIDIA Nemotron 3.5 ASR Streaming 0.6B**:
➡️ https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
