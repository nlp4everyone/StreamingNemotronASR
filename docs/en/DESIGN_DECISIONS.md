# Design Decisions — StreamingNenotronASR

This document explains the rationale behind each major architectural choice, including trade-offs and alternatives that were considered.

---

## 1. Transport — WebSocket with JSON text frames

All audio flows over a single persistent WebSocket connection. The client sends typed JSON frames (`start` / `audio` / `end`); the server streams transcript fragments back in real time. Audio binary is base64-encoded to stay within the JSON frame. A `"start"` message resets decoder state within the same connection rather than closing and reopening it — encoder cache is preserved to maintain acoustic continuity between utterances.

### Pros

- **Full-duplex:** Client streams audio up while server streams results back — no polling, lowest possible latency.
- **State tied to connection:** No session token or re-authentication between utterances; state lives as long as the connection does.
- **Easy to debug:** JSON frames are human-readable in `wscat` or browser DevTools without special tooling.

### Cons

- **Base64 overhead ~33%:** Audio binary must be base64-encoded to fit in JSON — at 43 KB/s per session this is acceptable, but compounds with many concurrent sessions.
- **Not cache-friendly:** WebSocket does not pass through CDN or HTTP cache — worth noting when designing load balancing.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| HTTP chunked transfer / SSE | One-directional — cannot multiplex audio upload and result stream in a single connection |
| gRPC bidirectional streaming | Binary-native, more bandwidth-efficient, but requires a proto schema and HTTP/2 infrastructure — setup overhead not justified at current scale |
| REST polling | High latency due to polling interval; not suitable for real-time streaming |

---

## 2. Audio Buffer — Simple accumulator, not a ring buffer

Incoming audio packets accumulate in `AudioChunkBuffer` — a simple accumulator — until `chunk_size` bytes are reached, then flush to inference. The final chunk is padded with `\x00` bytes (silence) rather than truncated, to keep tensor shapes consistent. No ring buffer is used because the cache-aware model maintains acoustic context through `att_cache` — no arbitrary audio window replay is needed.

### Pros

- **Consistent tensor shape:** The preprocessor always receives the same input shape — no special code path for short chunks.
- **Saves VRAM:** Avoids ~384 KB per session for a ring buffer that would never be used.
- **Zero-padding is acoustically safe:** NeMo does not emit spurious tokens for a short silence at the end of a chunk — padding does not affect the transcript.

### Cons

- **No audio replay:** Cannot re-read an audio window after it has been flushed — if VAD trimming or final window extraction is needed later, this will require a refactor.
- **Slight added latency at end of utterance:** The final chunk must wait to be padded to full size rather than being processed immediately.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Fixed ring buffer (12s) | Useful when arbitrary audio window access is needed; with a cache-aware model, it is unnecessary overhead (~384 KB per session) |
| Queue of raw audio packets | Cannot query by time range; memory overhead scales with packet rate |

---

## 3. Inference Engine — NeMo Cache-Aware Streaming

The NeMo cache-aware model processes audio in fixed-size chunks. Each session holds its own `att_cache` and `conv_cache` — the cache encodes the acoustic context of that speaker and is updated after each chunk inference. When a new utterance begins (`"start"`), decoder cache (`hypotheses`, `pred_out`) is reset but encoder cache is preserved so acoustic continuity is not broken. Feature extraction is fully delegated to NeMo's `model.preprocessor()` rather than computed manually — this guarantees no feature mismatch with training.

### Pros

- **High quality:** Left context is preserved through cache — the model "remembers" what it has heard; no feature mismatch between train and inference.
- **Low latency:** Each chunk is a small incremental forward pass — no re-encoding of the full audio from scratch.
- **Acoustic continuity between utterances:** Encoder cache is not reset between consecutive sentences — the model is not "surprised" by the start of the next utterance.

### Cons

- **VRAM scales linearly with sessions:** Each session holds its own cache tensors — cannot be shared across sessions. `max_sessions` must be tuned to actual VRAM.
- **Cache is not portable:** Cache tensors are tied to a specific session; cannot migrate a session to another GPU or serialize for resume.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Stateless inference (re-encode full audio each chunk) | Simpler but loses left context — quality degrades significantly, especially at the end of long sentences |
| Offline batch inference | Cannot achieve low latency; not suitable for real-time streaming |
| Shared encoder cache across sessions | Acoustically incorrect — one speaker's cache corrupts another speaker's context |

---

## 4. Concurrency — asyncio + ThreadPoolExecutor

FastAPI's asyncio event loop handles all WebSocket I/O (non-blocking). GPU inference is a blocking CUDA call — offloaded to a thread pool via `run_in_executor` so it does not block the event loop. The two layers are completely separate: I/O and compute run in parallel. Shared model state is protected by a small `threading.Lock` that wraps only the state-mutation call, not the entire inference.

### Pros

- **Many concurrent connections:** Event loop never blocks on inference — dozens of WebSocket connections process I/O simultaneously.
- **Shared memory:** Threads share process memory — per-session cache tensors need no serialization and have no IPC overhead.
- **CUDA releases the GIL:** The Python GIL does not block GPU work — the thread pool genuinely parallelizes inference across sessions.

### Cons

- **Manual thread safety:** Shared state must be explicitly protected with locks — easy to miss when extending code.
- **CUDA error handling is less ergonomic:** Exceptions in threads must be propagated back to the main coroutine — less clean than handling directly in the event loop.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| `multiprocessing` | Requires tensor serialization over IPC — impractical with per-session cache tensors; high overhead |
| Synchronous server (Flask + gunicorn) | Incompatible with asyncio WebSocket model; poorly scalable under I/O-heavy load |
| Separate inference microservice via queue | Adds a network hop and serialization overhead; overcomplicates deployment for no clear benefit at current scale |

---

## 5. Session Management & VRAM Lifecycle

Each WebSocket connection creates a `StreamingSession` with its own cache tensors. `SessionManager` tracks all active sessions, enforces the `max_sessions` limit (closes with WebSocket close code 1008 when full), and releases VRAM immediately on disconnect by nulling out tensor references — no waiting for GC. `ServiceRegistry` raises `RuntimeError` rather than returning `None` if accessed before `startup()` completes.

### Pros

- **Deterministic VRAM release:** Does not depend on GC timing — VRAM is returned as soon as the session ends, preventing OOM under high connect/disconnect workloads.
- **Hard capacity limit:** Server never exceeds its VRAM budget; clients receive a clear signal (code 1008) rather than hanging connections.
- **Fail loud:** `RuntimeError` from the registry surfaces immediately at the source, not at a random call site where `None` is dereferenced.

### Cons

- **No queuing:** Clients are rejected immediately when full rather than waiting for a free slot — retry logic must be implemented on the client side.
- **`max_sessions` requires manual tuning:** No auto-scaling to actual remaining VRAM — profiling is needed to set the right value.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Let GC release VRAM naturally | Non-deterministic — under high load, VRAM may be exhausted before a GC cycle runs, causing OOM |
| Queue waiting for a free slot | VRAM is still held while waiting; increases server complexity without addressing the root cause |
| `__del__` method on session object | Timing not guaranteed in CPython, especially with circular references |

---

## 6. Batch Scheduler — `per_session` and `dynamic`

`BatchScheduler` supports two modes configured via env var. `per_session` infers immediately when a session's chunk is ready — lowest latency, suitable when session count is low. `dynamic` batches requests from multiple sessions within a time window and issues a single GPU call — higher throughput, accepts an additional `batch_timeout_ms` of latency. Switching mode requires no code changes or image rebuild.

### Pros

- **Flexible per workload:** Same codebase serves both single-user latency-sensitive and multi-user throughput-optimized scenarios.
- **No rebuild needed:** Switching mode is just an env var change — suited for deployments with multiple environments.

### Cons

- **`dynamic` mode adds fixed latency:** Each inference step waits `batch_timeout_ms` to collect a batch — not suitable when latency is the top priority.
- **`dynamic` mode is harder to debug:** Multiple sessions are merged into one GPU call — harder to isolate issues for a specific session.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| `per_session` only | Wastes GPU when many sessions are active — each session occupies its own GPU call even when the GPU could handle more |
| `dynamic` only | Adds unnecessary latency in single-user or low-session workloads |
| Adaptive batching (self-adjusting window) | Increases scheduler complexity; no real profiling data yet to calibrate |
