# Core Components

## FastAPI Application (`app/main.py`)

Application entry point. Initializes the FastAPI app, wires the lifespan, registers the router and health endpoint.

- `lifespan(app)` — calls `startup()` before accepting requests, `shutdown()` when the server stops
- Registers `ws_router` containing the `/ws/stream` endpoint
- `/health` — returns `model_ready`, `active_sessions`, `preset`, `batch_mode`

## WebSocket Router (`app/routers/websocket.py`)

The sole entry point for WebSocket connections. Endpoint: `/ws/stream`.

- Rejects connections if `session_count >= max_sessions` (close code 1008)
- Creates a `StreamingSession`, registers it with `SessionManager`, sends `session_info`
- Dispatches messages by `type`:
  - `"audio"` → `StreamingHandler.handle_audio()`
  - `"end"` → `StreamingHandler.handle_end()`
  - `"start"` → reset `cache`, `audio_buffer`, `transcript` for a new utterance in the same connection
  - unknown type → `send_error("UNKNOWN_TYPE", ...)`
- `finally` → `handler.cleanup(session_id)` whether disconnect or exception

Does not handle audio logic or call the model directly.

## StreamingHandler (`app/websocket/handlers.py`)

Per-message orchestrator. Stateless — all state lives in `StreamingSession`.

`handle_audio(session, msg)`:
- Updates `session.lang`, `session.last_sample_rate`, resets EOS timer
- Decodes base64 → PCM bytes; decode error → `send_error("DECODE_ERROR", ...)`
- `session.audio_buffer.push()` — if buffer has a full chunk, calls `_infer_and_emit(is_final=False)`
- Partial transcript is only sent when `text != last_partial` (avoids no-op frames)

`handle_end(session)`:
- Cancels EOS timer
- Flushes remaining buffer → `_infer_and_emit(is_final=True)`
- If buffer is empty: promotes `last_partial` to final with `duration_ms`
- Resets `cache`, `audio_buffer`, `transcript` for the next utterance

EOS timeout:
- Each audio packet cancels and recreates an `asyncio.Task` (`_reschedule_eos`)
- After `end_of_speech_timeout_s` seconds with no new packet, `_eos_timeout` calls `handle_end()` automatically

## ConnectionManager (`app/websocket/manager.py`)

Stateless send helpers. All `ws.send_json()` calls go through here to ensure consistent schemas. Send errors are swallowed — client disconnect is not an application error.

- `send_session_info(ws, session)` — sends `preset`, `chunk_ms`, `att_context_size`, `packets_per_chunk`, `batch_mode`
- `send_transcript(ws, session_id, text, is_final, ...)` — sends partial or final transcript; `duration_ms` is included only when `is_final=True`
- `send_error(ws, code, message)` — sends `{"type": "error", "code": ..., "message": ...}`

## ServiceRegistry (`app/services/registry.py`)

Singleton container holding all application-level services. Properties raise `RuntimeError` if accessed before `startup()` runs — surfaces wiring errors early rather than as silent `AttributeError` at a random call site.

- `engine` → `NemoStreamingEngine`
- `scheduler` → `BatchScheduler`
- `session_manager` → `SessionManager`
- `is_ready()` — `True` when all three services are initialized and the model has finished loading

## Startup / Shutdown (`app/startup/initializer.py`)

Initializes services in dependency order and releases resources on shutdown.

`startup()` — three sequential steps:
1. `SessionManager()` — lightweight, no I/O; must run first so engine/scheduler can reference it
2. `NemoStreamingEngine().load()` — downloads (~1–2 GB on first run) and loads model weights onto device; this is the most time-consuming step
3. `BatchScheduler(engine).start()` — initializes ThreadPoolExecutor; if `batch_mode=dynamic`, starts a background batch worker task

`shutdown()`:
- `scheduler.stop()` — cancels worker task, shuts down thread pool (cancels pending futures)

## SessionManager (`app/session/manager.py`)

A `dict[UUID, StreamingSession]` registry. Thread-safe within the asyncio single-thread model.

- `create(session)` — registers a new session in the registry
- `get(session_id)` → `StreamingSession | None`
- `remove(session_id)` — removes the session and **immediately releases GPU cache** (nulls out all tensor fields in `ASRCacheState` before GC has a chance to run)
- `count()` → number of currently active sessions (used to check capacity limit)

## StreamingSession (`app/session/state.py`)

All mutable state for a single WebSocket connection. Three nested dataclasses:

`ASRCacheState` — NeMo streaming cache per session:

```python
att_cache: torch.Tensor | None      # encoder attention cache [layers, B, heads, cache_len, dim]
conv_cache: torch.Tensor | None     # encoder convolution cache [layers, B, dim, kernel-1]
att_cache_len: torch.Tensor | None  # valid cache lengths per layer [layers, B]
hypotheses: list | None             # RNNT decoder carry-over (previous_hypotheses)
pred_out: torch.Tensor | None       # RNNT decoder previous token output
```

`reset()` — preserves encoder cache (acoustic continuity), clears decoder state for a new utterance.

`TranscriptState` — transcript state within one utterance:
- `last_partial` — last partial text that was emitted
- `utterance_start` — monotonic timestamp for computing `duration_ms`
- `reset()` — clears `last_partial` and resets utterance timer

`StreamingSession`:
- `audio_buffer`, `cache`, `transcript` — three core session components
- `lang`, `last_sample_rate` — metadata from the most recent audio packet
- `eos_task` — pending EOS timeout task; cancelled and recreated on each packet
- `touch()` — updates `last_activity` (called on every packet)
- `idle_seconds` — time elapsed since the last packet

## AudioChunkBuffer (`app/audio/buffer.py`)

Accumulates 20ms PCM int16 packets from the client; flushes when one inference chunk is full.

`target_bytes = packets_per_chunk × 320 samples × 2 bytes` — computed from preset, not hardcoded.

- `push(pcm_bytes)` → `bytes | None` — returns a chunk when `target_bytes` are reached, `None` otherwise; remainder is kept for the next call
- `flush()` → `bytes | None` — returns remaining bytes zero-padded to `target_bytes` (zero = silence; NeMo does not emit spurious tokens); `None` if buffer is empty
- `clear()` — clears the buffer without returning data
- `buffered_ms` — milliseconds of audio currently held in the buffer (assumes 16 kHz mono int16)

## NemoStreamingEngine (`app/asr/engine.py`)

Wrapper around `nvidia/nemotron-3.5-asr-streaming-0.6b`. Singleton, loaded once at startup; weights are read-only thereafter. Called from a `ThreadPoolExecutor` (blocking CUDA call).

`load()` — initializes the model in 5 steps:
1. `ASRModel.from_pretrained()` on the configured device
2. `.eval()` — disables dropout/batch-norm
3. `set_default_att_context_size()` — applies look-ahead frames from the preset
4. `change_decoding_strategy()` if not already `greedy_batch`
5. `set_inference_prompt()` — sets the default language (**required** — omitting this causes the model to return empty strings)

`stream_step(pcm_bytes, sample_rate, cache, lang)` → `(text, new_cache)`:
1. Preprocess: int16 → float32 → resample if needed → log-mel via model's preprocessor
2. Set language prompt (uses `threading.Lock` because this is shared model state)
3. `conformer_stream_step()` → returns 6 values: greedy predictions, transcripts, new att_cache, new conv_cache, new att_cache_len, best_hyp
4. Extract text (RNNT returns a `Hypothesis` object, CTC returns a `str`)
5. Pack new `ASRCacheState` and return

`stream_step_batch(requests)` — runs multiple sessions sequentially in one call (placeholder for true GPU batching B > 1).

## BatchScheduler (`app/asr/scheduler.py`)

Routes inference requests to the engine. Mode is configured via `ASR_BATCH_MODE`:

`per_session` — each request is dispatched immediately via `loop.run_in_executor(thread_pool, engine.stream_step, ...)`.

`dynamic` — requests are placed in an `asyncio.Queue`; a background worker collects up to `max_batch_size` requests within `batch_timeout_ms` then calls `engine.stream_step_batch()` once.

- `submit(pcm_bytes, sample_rate, cache, lang)` → `(text, new_cache)` — unified interface for both modes; the caller just `await`s without knowing the mode
- `start()` / `stop()` — lifecycle; `stop()` cancels the worker task and shuts down the thread pool

## Schema (`app/schema/`)

Pydantic models for all message types:

- `audio.py` — `AudioMessage`: `type="audio"`, `data` (base64 PCM int16), `sample_rate`, `lang`
- `session.py` — `SessionInfoMessage`: `session_id`, `preset`, `chunk_ms`, `att_context_size`, `packets_per_chunk`, `batch_mode`
- `transcript.py` — `TranscriptMessage`: `session_id`, `text`, `is_final`, `lang_detected`, `duration_ms`
