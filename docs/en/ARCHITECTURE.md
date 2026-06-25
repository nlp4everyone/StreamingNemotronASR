# System Architecture — StreamingNenotronASR

> **Status:** Draft v1.0 · Pre-implementation design document

---

## 1. Overview

StreamingNenotronASR is a WebSocket server that receives real-time audio streams and returns continuous transcripts (partial/final) using NVIDIA's **Nemotron 3.5 ASR Streaming 0.6B** model.

**Model:** [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)  
**Model architecture:** FastConformer-CacheAware-RNNT · 600M params · 24 encoder layers · D=1024  
**Languages:** 40 language-locales · auto-detect supported  
**Protocol:** WebSocket · JSON text frames · PCM int16 base64-encoded  
**Runtime:** Python ≥ 3.11 · NeMo 26.06 · PyTorch · CUDA (Ampere+)

---

## 2. Design Principles

| Principle | Application |
|---|---|
| **Session isolation** | Each WebSocket connection has its own GPU cache; no shared mutable state |
| **Non-blocking I/O** | asyncio event loop handles all WebSocket I/O; GPU inference runs in ThreadPoolExecutor |
| **Config-driven latency** | A single `streaming_preset` variable controls all streaming behavior; nothing hardcoded |
| **Derive, don't duplicate** | `chunk_ms`, `packets_per_chunk` are computed from `att_context_size`, not declared separately |
| **Graceful lifecycle** | Disconnect → release GPU cache immediately; shutdown → flush all sessions first |

---

## 3. Layered Architecture

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
│  Holds: AudioBuffer, ASRCacheState, TranscriptState       │
└────────────────────┬─────────────────────────────────────┘
                     │ PCM bytes
┌────────────────────▼─────────────────────────────────────┐
│  Layer 3 · Audio Pipeline                                 │
│  AudioChunkBuffer → Feature Extraction (log-mel 80-dim)   │
│  Accumulates 20ms packets → flushes when chunk_ms is met  │
└────────────────────┬─────────────────────────────────────┘
                     │ float32 features [1, T, 80]
┌────────────────────▼─────────────────────────────────────┐
│  Layer 4 · ASR Engine                                     │
│  NemoStreamingEngine · conformer_stream_step()            │
│  Updates att_cache + conv_cache after each chunk          │
│  Runs in ThreadPoolExecutor (GPU blocking call)           │
└────────────────────┬─────────────────────────────────────┘
                     │ text tokens
┌────────────────────▼─────────────────────────────────────┐
│  Layer 5 · Emit                                           │
│  TranscriptEmitter · partial (is_final=false) each chunk  │
│  final (is_final=true) on "end" message or 3s timeout    │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow

```
Client                     Server
  │                           │
  │──── WS connect ──────────►│ create session_id
  │◄─── session_info ─────────│ {session_id, preset, chunk_ms, ...}
  │                           │
  │──── audio (20ms) ────────►│ base64 decode → PCM int16
  │──── audio (20ms) ────────►│ AudioChunkBuffer.push()
  │──── ...                   │   (accumulate N packets)
  │──── audio (20ms) ────────►│ buffer full → flush chunk
  │                           │ Feature extraction (log-mel)
  │                           │ run_in_executor → GPU inference
  │                           │ update att_cache, conv_cache
  │◄─── transcript (partial) ─│ {text, is_final: false}
  │                           │
  │──── audio (20ms) ────────►│ (repeat...)
  │◄─── transcript (partial) ─│
  │                           │
  │──── {"type": "end"} ─────►│ flush remaining buffer (zero-pad)
  │                           │ final inference
  │◄─── transcript (final) ───│ {text, is_final: true}
  │                           │
  │──── WS close ────────────►│ release GPU cache
  │                           │
```

---

## 5. Components

> See per-component detail: [DETAILED_COMPONENTS.md](DETAILED_COMPONENTS.md)

---

## 6. WebSocket Protocol

### Message flow

```
Client                          Server
  │                               │
  │═══ WS Handshake ═════════════►│
  │◄══ {"type":"session_info"} ═══│
  │                               │
  │──► {"type":"audio", ...} ─────│
  │         (repeat N times)      │
  │◄── {"type":"transcript",      │  partial, is_final=false
  │         is_final:false} ──────│  (emitted after each inference)
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

**Client → Server: audio** *(every 20ms)*
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
  "text": "Hello, this is",
  "is_final": false,
  "lang_detected": "en-US"
}
```

**Server → Client: transcript (final)**
```json
{
  "type": "transcript",
  "session_id": "a3f2c1d9-...",
  "text": "Hello, this is the final result.",
  "is_final": true,
  "lang_detected": "en-US",
  "duration_ms": 4820
}
```

**Client → Server: end**
```json
{ "type": "end" }
```

### Partial emit rules

- Emit partial **only when** `text != session.last_partial` — avoids sending no-op updates
- If inference returns an empty string (silence) → do not emit
- End-of-speech timeout: if no new packet arrives for 3s → auto-trigger `handle_end()`

---

## 7. Streaming Configuration

### Preset system

`att_context_size = [left_frames, right_frames]` — left is fixed at 56 (4.48s left context).

```
Preset      att_context_size   chunk_frames   chunk_ms   packets/chunk   WER
─────────────────────────────────────────────────────────────────────────────
ultra_low   [56,  0]                1            80ms          4         highest
low         [56,  1]                2           160ms          8          ↑
medium      [56,  3]                4           320ms         16          │
balanced    [56,  6]                7           560ms         28          ↓
high        [56, 13]               14          1120ms         56        ~8.84% avg
```

> **chunk_frames:** `right_frames + 1`  
> **chunk_ms:** `chunk_frames × 80ms` (each subsampled frame = 10ms shift × 8x subsampling)  
> **packets_per_chunk:** `chunk_ms / 20ms` (client sends 20ms/packet)

### Switching

```bash
# Env var — no rebuild needed
ASR_STREAMING_PRESET=ultra_low uvicorn app.main:app

# Or config/settings.local.yaml (gitignored)
streaming_preset: high
```

All code reads `settings.preset.*` — change the preset in one place and the entire pipeline adjusts automatically.

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

**Principles:**
- `asyncio` handles all I/O (WebSocket recv/send) — non-blocking
- GPU inference is a synchronous CUDA call → must run in thread pool
- `loop.run_in_executor(thread_pool, engine.stream_step, ...)` — non-blocking await
- Thread pool size = `thread_pool_workers` (default 2; one thread can preprocess while another holds the model)
- Model pool (`model_pool_size` instances) — each thread acquires an instance exclusively; no `threading.Lock` needed

**No `asyncio.Queue` per-session** (unlike DETAILED_COMPONENTS.md) because there is no VAD or adaptive pacing. Simple flow: packet arrives → chunk full → infer immediately.

---

## 9. Session Lifecycle

```
                    ┌─────────────┐
                    │   CREATED   │ ← WS connect, cache initialized to zeros
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
              ┌────►│  STREAMING  │ ← receiving audio packets, emitting partials
              │     └──────┬──────┘
              │            │ {"type":"end"} or 3s timeout
              │     ┌──────▼──────┐
              │     │ FINALIZING  │ ← flush buffer, final inference, emit final
              │     └──────┬──────┘
              │            │ reset cache, clear buffer
              └────────────┘ ← ready for next utterance

                    ┌──────▼──────┐
                    │   CLOSED    │ ← WS disconnect OR idle sweeper eviction
                    └─────────────┘  GPU cache released immediately
```

**Note:** A single session can have multiple utterances (multiple STREAMING→FINALIZING cycles) within one WebSocket connection lifetime. Cache is **reset after each final**, not after each disconnect — this is the correct behavior for continuous speech.

---

## 10. System Constraints

### GPU VRAM

| Component | VRAM |
|---|---|
| Model weights (fp16) | ~1.2 GB |
| Cache per session (att + conv, 24 layers) | ~50 MB |
| **Max sessions on A100 80GB** | ~(80 - 1.2) / 0.05 ≈ **1,570** (theoretical) |
| **Practical** (fragmentation + overhead) | ~400–600 sessions |

Set `max_sessions` in config to avoid OOM — server returns `503` when the limit is reached.

### CPU

- Feature extraction (log-mel) runs on CPU; may become a bottleneck at high session counts
- `torchaudio` uses a C++ backend — reasonably fast but needs real benchmarking
- If bottleneck: move feature extraction to GPU (`torchaudio.transforms.MelSpectrogram().cuda()`)

### Network

- Client sends 20ms × 640 bytes = 32 KB/s per session (raw)
- After base64: ~43 KB/s per session
- 100 sessions: ~4.3 MB/s inbound — negligible

### Latency budget

```
Client packet (20ms)
→ Buffer accumulation [chunk_ms]       ← depends on preset
→ Feature extraction (~2–5ms CPU)
→ GPU inference (~10–50ms on A100)
→ Token decode (~1ms)
→ WS send (~1ms)
─────────────────────────────────────
Total: preset_latency + 15–60ms overhead
```

With `balanced` preset (560ms): user sees transcript ~575–620ms after the start of the chunk.

---

## 11. Project Structure

> See details: [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)

---

## 12. Design Decisions — rationale

| Decision | Reason | Alternatives considered |
|---|---|---|
| FastAPI + uvicorn | Native WebSocket, asyncio, HTTP endpoint included | aiohttp: less ergonomic; bare websockets: no routing |
| asyncio + ThreadPoolExecutor | GPU call is blocking; executor frees event loop | asyncio.Queue per-session: more complex, unnecessary without VAD |
| Model pool (not singleton) | Pool-based exclusive acquisition eliminates lang-prompt race condition for multi-lang | Single shared model + Lock: race window between set_inference_prompt and conformer_stream_step |
| Bounded inference queue per session | Drops stale non-final chunks; prevents queue bloat under slow GPU | Unbounded queue: back-pressure causes latency spiral |
| Idle session sweeper | Reclaims VRAM from ghost sessions (TCP alive, no audio); complements WS ping/pong | Rely on ping/pong only: doesn't handle soft "client went silent" case |
| JSON text frames + base64 | Browser-friendly, debuggable; 33% overhead acceptable | Binary WS frames: more efficient but more complex; add later if needed |
| Per-session cache tensors | Model is cache-aware by design; att_cache cannot be shared | Stateless inference (re-encode from scratch each chunk): loses left context |
| No VAD | Cache-aware model handles silence naturally (empty output); reduces complexity | Silero VAD: add later as GPU optimization |
| No ring buffer | No audio replay needed; model maintains context via att_cache | 12s ring buffer: needed for VAD trim/final window extraction |
| Named preset system | Change one env var, entire pipeline adjusts; avoids inconsistent config | Direct att_context_size config: easy to set mismatched chunk_size |
| Emit partial only on text change | Avoids no-op WebSocket messages; reduces client render load | Emit every chunk regardless: simpler but wastes bandwidth |

---
