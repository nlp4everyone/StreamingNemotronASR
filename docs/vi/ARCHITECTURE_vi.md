# Kiến trúc hệ thống — StreamingNenotronASR

> **Trạng thái:** Draft v1.0 · Tài liệu thiết kế trước khi triển khai

---

## 1. Tổng quan

StreamingNenotronASR là một WebSocket server nhận luồng âm thanh thời gian thực và trả về transcript liên tục (partial/final) bằng cách sử dụng model **Nemotron 3.5 ASR Streaming 0.6B** của NVIDIA.

**Model:** [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)  
**Kiến trúc model:** FastConformer-CacheAware-RNNT · 600M params · 24 encoder layers · D=1024  
**Ngôn ngữ:** 40 language-locales · hỗ trợ auto-detect  
**Giao thức:** WebSocket · JSON text frames · PCM int16 base64-encoded  
**Runtime:** Python ≥ 3.11 · NeMo 26.06 · PyTorch · CUDA (Ampere+)

---

## 2. Nguyên lý thiết kế

| Nguyên tắc | Áp dụng |
|---|---|
| **Session isolation** | Mỗi WebSocket connection có cache GPU riêng, không shared mutable state |
| **Non-blocking I/O** | asyncio event loop xử lý toàn bộ WebSocket; GPU inference chạy trong ThreadPoolExecutor |
| **Config-driven latency** | Một biến `streaming_preset` điều khiển toàn bộ hành vi streaming; không hardcode |
| **Derive, đừng duplicate** | `chunk_ms`, `packets_per_chunk` được tính từ `att_context_size`, không khai báo riêng |
| **Graceful lifecycle** | Disconnect → giải phóng GPU cache ngay; shutdown → flush tất cả sessions trước |

---

## 3. Kiến trúc phân lớp

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1 · Transport                                      │
│  FastAPI + uvicorn · asyncio event loop                   │
│  Endpoint: ws://host:8000/ws/stream                       │
└────────────────────┬─────────────────────────────────────┘
                     │ JSON text frames
┌────────────────────▼─────────────────────────────────────┐
│  Layer 2 · Session                                        │
│  SessionManager · StreamingSession per connection         │
│  Giữ: AudioBuffer, ASRCacheState, TranscriptState         │
└────────────────────┬─────────────────────────────────────┘
                     │ PCM bytes
┌────────────────────▼─────────────────────────────────────┐
│  Layer 3 · Audio Pipeline                                 │
│  AudioChunkBuffer → Feature Extraction (log-mel 80-dim)   │
│  Tích lũy 20ms packets → flush khi đủ chunk_ms           │
└────────────────────┬─────────────────────────────────────┘
                     │ float32 features [1, T, 80]
┌────────────────────▼─────────────────────────────────────┐
│  Layer 4 · ASR Engine                                     │
│  NemoStreamingEngine · conformer_stream_step()            │
│  Cập nhật att_cache + conv_cache sau mỗi chunk            │
│  Chạy trong ThreadPoolExecutor (GPU blocking call)        │
└────────────────────┬─────────────────────────────────────┘
                     │ text tokens
┌────────────────────▼─────────────────────────────────────┐
│  Layer 5 · Emit                                           │
│  TranscriptEmitter · partial (is_final=false) mỗi chunk   │
│  final (is_final=true) khi nhận "end" hoặc timeout 3s    │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Luồng dữ liệu

```
Client                     Server
  │                           │
  │──── WS connect ──────────►│ tạo session_id
  │◄─── session_info ─────────│ {session_id, preset, chunk_ms, ...}
  │                           │
  │──── audio (20ms) ────────►│ base64 decode → PCM int16
  │──── audio (20ms) ────────►│ AudioChunkBuffer.push()
  │──── ...                   │   (tích lũy N packets)
  │──── audio (20ms) ────────►│ buffer đủ → flush chunk
  │                           │ Feature extraction (log-mel)
  │                           │ run_in_executor → GPU inference
  │                           │ cập nhật att_cache, conv_cache
  │◄─── transcript (partial) ─│ {text, is_final: false}
  │                           │
  │──── audio (20ms) ────────►│ (lặp lại...)
  │◄─── transcript (partial) ─│
  │                           │
  │──── {"type": "end"} ─────►│ flush buffer còn lại (zero-pad)
  │                           │ inference lần cuối
  │◄─── transcript (final) ───│ {text, is_final: true}
  │                           │
  │──── WS close ────────────►│ giải phóng GPU cache
  │                           │
```

---

## 5. Các thành phần

> Xem mô tả chi tiết từng thành phần: [DETAILED_COMPONENTS.md](DETAILED_COMPONENTS.md)

---

## 6. Giao thức WebSocket

### Message flow

```
Client                          Server
  │                               │
  │═══ WS Handshake ═════════════►│
  │◄══ {"type":"session_info"} ═══│
  │                               │
  │──► {"type":"audio", ...} ─────│
  │         (lặp lại N lần)       │
  │◄── {"type":"transcript",      │  partial, is_final=false
  │         is_final:false} ──────│  (emit mỗi khi inference xong)
  │                               │
  │──► {"type":"end"} ────────────│
  │◄── {"type":"transcript",      │  final, is_final=true
  │         is_final:true} ───────│
  │                               │
  │═══ WS Close ════════════════► │
```

### Message schemas

**Server → Client: session_info**
```json
{
  "type": "session_info",
  "session_id": "a3f2c1d9-4b5e-...",
  "preset": "balanced",
  "chunk_ms": 560,
  "att_context_size": [56, 6],
  "packets_per_chunk": 28
}
```

**Client → Server: audio** *(mỗi 20ms)*
```json
{
  "type": "audio",
  "data": "<base64 PCM int16, 320 samples = 640 bytes>",
  "sample_rate": 16000,
  "lang": "vi-VN"
}
```

**Server → Client: transcript (partial)**
```json
{
  "type": "transcript",
  "session_id": "a3f2c1d9-...",
  "text": "Xin chào, đây là",
  "is_final": false,
  "lang_detected": "vi-VN"
}
```

**Server → Client: transcript (final)**
```json
{
  "type": "transcript",
  "session_id": "a3f2c1d9-...",
  "text": "Xin chào, đây là kết quả cuối cùng.",
  "is_final": true,
  "lang_detected": "vi-VN",
  "duration_ms": 4820
}
```

**Client → Server: end**
```json
{ "type": "end" }
```

### Quy tắc emit partial

- Emit partial **chỉ khi** `text != session.last_partial` — tránh gửi no-op update
- Nếu inference trả về chuỗi rỗng (silence) → không emit
- End-of-speech timeout: nếu 3s không có packet mới → tự trigger `handle_end()`

---

## 7. Cấu hình Streaming

### Preset system

`att_context_size = [left_frames, right_frames]` — left cố định 56 (4.48s left context).

```
Preset      att_context_size   chunk_frames   chunk_ms   packets/chunk   WER
─────────────────────────────────────────────────────────────────────────────
ultra_low   [56,  0]                1            80ms          4         cao nhất
low         [56,  1]                2           160ms          8          ↑
medium      [56,  3]                4           320ms         16          │
balanced    [56,  6]                7           560ms         28          ↓
high        [56, 13]               14          1120ms         56        ~8.84% avg
```

> **Cách tính chunk_frames:** `right_frames + 1`  
> **Cách tính chunk_ms:** `chunk_frames × 80ms` (mỗi subsampled frame = 10ms shift × 8x subsampling)  
> **Cách tính packets_per_chunk:** `chunk_ms / 20ms` (client gửi 20ms/packet)

### Cách switch

```bash
# Env var — không cần rebuild
ASR_STREAMING_PRESET=ultra_low uvicorn app.main:app

# Hoặc config/settings.local.yaml (gitignored)
streaming_preset: high
```

Tất cả code đều đọc `settings.preset.*` — thay preset ở một chỗ, toàn bộ pipeline tự điều chỉnh.

---

## 8. Concurrency Model

```
asyncio event loop (main thread)
│
├── WS connection 1 ──► handle_audio() ──► run_in_executor() ──┐
├── WS connection 2 ──► handle_audio() ──► run_in_executor() ──┤
├── WS connection 3 ──► handle_audio() ──► run_in_executor() ──┤
│                                                               │
└─────────── (non-blocking, event loop free) ──────────────────┘
                                                               │
                        ThreadPoolExecutor (N threads)         │
                        ├── thread-1: GPU inference ◄──────────┘
                        ├── thread-2: GPU inference
                        └── thread-N: GPU inference
```

**Nguyên tắc:**
- `asyncio` xử lý toàn bộ I/O (WebSocket recv/send) — không blocking
- GPU inference là synchronous CUDA call → phải chạy trong thread pool
- `loop.run_in_executor(thread_pool, engine.stream_step, ...)` — non-blocking await
- Thread pool size = `thread_pool_workers` (default 2)
- Model pool (`model_pool_size` instances) — mỗi thread acquire một instance độc quyền; không cần `threading.Lock`

**Không dùng `asyncio.Queue` per-session** (như DETAILED_COMPONENTS.md) vì không có VAD/adaptive pacing. Flow đơn giản: packet đến → đủ chunk → inference ngay.

---

## 9. Session Lifecycle

```
                    ┌─────────────┐
                    │   CREATED   │ ← WS connect, cache khởi tạo zeros
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
              ┌────►│  STREAMING  │ ← nhận audio packets, emit partials
              │     └──────┬──────┘
              │            │ {"type":"end"} hoặc timeout 3s
              │     ┌──────▼──────┐
              │     │ FINALIZING  │ ← flush buffer, inference cuối, emit final
              │     └──────┬──────┘
              │            │ reset cache, clear buffer
              └────────────┘ ← sẵn sàng utterance tiếp theo

                    ┌──────▼──────┐
                    │   CLOSED    │ ← WS disconnect HOẶC idle sweeper evict
                    └─────────────┘  GPU cache giải phóng ngay
```

**Lưu ý:** Một session có thể có nhiều utterances (nhiều chu kỳ STREAMING→FINALIZING) trong vòng đời của một WebSocket connection. Cache được **reset sau mỗi final**, không phải sau mỗi disconnect — đây là hành vi đúng cho continuous speech.

---

## 10. Ràng buộc hệ thống

### GPU VRAM

| Thành phần | VRAM |
|---|---|
| Model weights (fp16) | ~1.2 GB |
| Cache per session (att + conv, 24 layers) | ~50 MB |
| **Tối đa sessions trên A100 80GB** | ~(80 - 1.2) / 0.05 ≈ **1,570** (lý thuyết) |
| **Thực tế** (fragmentation + overhead) | ~400–600 sessions |

Nên set `max_sessions` trong config để tránh OOM — server trả `503` khi đạt giới hạn.

### CPU

- Feature extraction (log-mel) chạy trên CPU, có thể là bottleneck khi nhiều sessions
- `torchaudio` dùng C++ backend — khá nhanh nhưng cần benchmark thực tế
- Nếu bottleneck: move feature extraction lên GPU (`torchaudio.transforms.MelSpectrogram().cuda()`)

### Network

- Client gửi 20ms × 640 bytes = 32 KB/s per session (raw)
- Sau base64: ~43 KB/s per session
- 100 sessions: ~4.3 MB/s inbound — không đáng kể

### Latency budget

```
Client packet (20ms)
→ Buffer tích lũy [chunk_ms]          ← phụ thuộc preset
→ Feature extraction (~2–5ms CPU)
→ GPU inference (~10–50ms on A100)
→ Token decode (~1ms)
→ WS send (~1ms)
─────────────────────────────────────
Total: preset_latency + 15–60ms overhead
```

Với preset `balanced` (560ms): user thấy transcript sau ~575–620ms kể từ đầu chunk.

---

## 11. Cấu trúc dự án

> Xem chi tiết: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

## 12. Quyết định thiết kế — ghi lại lý do

| Quyết định | Lý do | Thay thế đã xem xét |
|---|---|---|
| FastAPI + uvicorn | WebSocket native, asyncio, HTTP endpoint đi kèm | aiohttp: ít ergonomic; bare websockets: thiếu routing |
| asyncio + ThreadPoolExecutor | GPU call là blocking; executor giải phóng event loop | asyncio.Queue per-session: phức tạp hơn, không cần thiết khi không có VAD |
| JSON text frames + base64 | Browser-friendly, debuggable; overhead 33% chấp nhận được | Binary WS frames: tối ưu hơn nhưng phức tạp hơn; thêm sau nếu cần |
| Per-session cache tensors | Model cache-aware by design; không thể share att_cache | Stateless inference (re-encode từ đầu mỗi chunk): sẽ mất left context |
| Không dùng VAD | Cache-aware model xử lý silence tự nhiên (output rỗng); giảm phức tạp | Silero VAD: cần nếu muốn tối ưu GPU — thêm sau như optimization |
| Không dùng ring buffer | Không cần replay audio; model tự giữ context qua att_cache | Ring buffer 12s: cần cho VAD trim/final window extraction |
| Preset system (named) | Đổi một env var, toàn pipeline điều chỉnh; tránh config không nhất quán | Trực tiếp cấu hình att_context_size: dễ làm sai chunk_size tương ứng |
| Emit partial chỉ khi text thay đổi | Tránh no-op WebSocket messages; giảm tải client render | Emit mỗi chunk bất kể: đơn giản hơn nhưng lãng phí bandwidth |
| Model pool (không phải singleton) | Pool-based exclusive acquisition loại bỏ race condition lang-prompt | Single model + Lock: vẫn có race window giữa set_inference_prompt và conformer_stream_step |
| Bounded inference queue per session | Drop chunk lỗi thời, ngăn latency spiral khi GPU chậm | Unbounded queue: queue phình to, latency tăng dần |
| Idle session sweeper | Thu hồi VRAM từ ghost sessions (TCP alive, không có audio); bổ sung WS ping/pong | Chỉ dùng ping/pong: không xử lý được soft "client im lặng" |

---
