# Các Thành Phần Chính

## FastAPI Application (`app/main.py`)

Entry point của ứng dụng. Khởi tạo FastAPI app, kết nối lifespan, đăng ký router và health endpoint.

- `lifespan(app)` — gọi `startup()` trước khi nhận request, `shutdown()` khi dừng server
- Đăng ký `ws_router` chứa endpoint `/ws/stream`
- `/health` — trả về `model_ready`, `active_sessions`, `preset`, `batch_mode`

## WebSocket Router (`app/routers/websocket.py`)

Entry point duy nhất cho WebSocket connections. Endpoint: `/ws/stream`.

- Từ chối kết nối nếu `session_count >= max_sessions` (close code 1008)
- Tạo `StreamingSession`, đăng ký với `SessionManager`, gửi `session_info`
- Dispatch message theo `type`:
  - `"audio"` → `StreamingHandler.handle_audio()`
  - `"end"` → `StreamingHandler.handle_end()`
  - `"start"` → reset `cache`, `audio_buffer`, `transcript` cho utterance mới trong cùng connection
  - type không xác định → `send_error("UNKNOWN_TYPE", ...)`
- `finally` → `handler.cleanup(session_id)` dù disconnect hay exception

Không xử lý logic audio, không gọi model trực tiếp.

## StreamingHandler (`app/websocket/handlers.py`)

Orchestrator per-message. Stateless — toàn bộ state nằm trong `StreamingSession`.

`handle_audio(session, msg)`:
- Cập nhật `session.lang`, `session.last_sample_rate`, reset EOS timer
- Decode base64 → PCM bytes; lỗi decode → `send_error("DECODE_ERROR", ...)`
- `session.audio_buffer.push()` — nếu buffer đủ chunk, gọi `_infer_and_emit(is_final=False)`
- Partial transcript chỉ được gửi khi `text != last_partial` (tránh no-op frame)

`handle_end(session)`:
- Hủy EOS timer
- Flush buffer còn lại → `_infer_and_emit(is_final=True)`
- Nếu buffer rỗng: promote `last_partial` làm final kèm `duration_ms`
- Reset `cache`, `audio_buffer`, `transcript` cho utterance tiếp theo

EOS timeout:
- Mỗi audio packet hủy và tạo lại `asyncio.Task` (`_reschedule_eos`)
- Sau `end_of_speech_timeout_s` giây không có packet, `_eos_timeout` tự gọi `handle_end()`

## ConnectionManager (`app/websocket/manager.py`)

Stateless send helpers. Tất cả `ws.send_json()` đi qua đây để đảm bảo schema nhất quán. Lỗi gửi bị nuốt — client disconnect không phải lỗi ứng dụng.

- `send_session_info(ws, session)` — gửi `preset`, `chunk_ms`, `att_context_size`, `packets_per_chunk`, `batch_mode`
- `send_transcript(ws, session_id, text, is_final, ...)` — gửi partial hoặc final transcript; `duration_ms` chỉ có khi `is_final=True`
- `send_error(ws, code, message)` — gửi `{"type": "error", "code": ..., "message": ...}`

## ServiceRegistry (`app/services/registry.py`)

Container singleton giữ tất cả application-level services. Properties raise `RuntimeError` nếu được truy cập trước khi `startup()` chạy — phát hiện lỗi wiring sớm thay vì `AttributeError` ngầm.

- `engine` → `NemoStreamingEngine`
- `scheduler` → `BatchScheduler`
- `session_manager` → `SessionManager`
- `is_ready()` — `True` khi cả ba service đã initialized và model đã load xong

## Startup / Shutdown (`app/startup/initializer.py`)

Khởi tạo services theo thứ tự dependency và giải phóng tài nguyên khi shutdown.

`startup()` — ba bước tuần tự:
1. `SessionManager()` — lightweight, không I/O; phải chạy trước để engine/scheduler có thể tham chiếu
2. `NemoStreamingEngine().load()` — download (~1–2 GB lần đầu) và load model weights lên device; đây là bước tốn thời gian nhất
3. `BatchScheduler(engine).start()` — khởi tạo ThreadPoolExecutor; nếu `batch_mode=dynamic`, khởi động background batch worker task

`shutdown()`:
- `scheduler.stop()` — hủy worker task, shutdown thread pool (cancel pending futures)

## SessionManager (`app/session/manager.py`)

Registry `dict[UUID, StreamingSession]`. Thread-safe trong asyncio single-thread model.

- `create(session)` — đăng ký session mới vào registry
- `get(session_id)` → `StreamingSession | None`
- `remove(session_id)` — xóa session và **giải phóng GPU cache ngay** (null out toàn bộ tensor fields trong `ASRCacheState` trước khi GC có cơ hội chạy)
- `count()` → số session đang active (dùng để kiểm tra capacity limit)

## StreamingSession (`app/session/state.py`)

Toàn bộ mutable state của một WebSocket connection. Ba dataclass lồng nhau:

`ASRCacheState` — NeMo streaming cache per-session:

```python
att_cache: torch.Tensor | None      # encoder attention cache [layers, B, heads, cache_len, dim]
conv_cache: torch.Tensor | None     # encoder convolution cache [layers, B, dim, kernel-1]
att_cache_len: torch.Tensor | None  # valid cache lengths per layer [layers, B]
hypotheses: list | None             # RNNT decoder carry-over (previous_hypotheses)
pred_out: torch.Tensor | None       # RNNT decoder previous token output
```

`reset()` — giữ encoder cache (acoustic continuity), xóa decoder state cho utterance mới.

`TranscriptState` — trạng thái transcript trong một utterance:
- `last_partial` — text partial cuối cùng đã emit
- `utterance_start` — timestamp monotonic để tính `duration_ms`
- `reset()` — xóa `last_partial` và reset utterance timer

`StreamingSession`:
- `audio_buffer`, `cache`, `transcript` — ba thành phần core của session
- `lang`, `last_sample_rate` — metadata từ audio packet gần nhất
- `eos_task` — pending EOS timeout task; bị hủy và tạo lại mỗi packet
- `touch()` — cập nhật `last_activity` (gọi trên mỗi packet)
- `idle_seconds` — thời gian kể từ packet cuối cùng

## AudioChunkBuffer (`app/audio/buffer.py`)

Tích lũy PCM int16 packets 20ms từ client; flush khi đủ một inference chunk.

`target_bytes = packets_per_chunk × 320 samples × 2 bytes` — tính từ preset, không hardcode.

- `push(pcm_bytes)` → `bytes | None` — trả về chunk khi đủ `target_bytes`, `None` nếu chưa; phần dư giữ lại cho lần tiếp
- `flush()` → `bytes | None` — trả về bytes còn lại zero-padded đến `target_bytes` (zero = silence, NeMo không emit token giả); `None` nếu buffer rỗng
- `clear()` — xóa buffer không trả dữ liệu
- `buffered_ms` — milliseconds audio đang giữ trong buffer (assumes 16 kHz mono int16)

## NemoStreamingEngine (`app/asr/engine.py`)

Wrapper quanh `nvidia/nemotron-3.5-asr-streaming-0.6b`. Singleton, load một lần khi startup, weights read-only sau đó. Gọi từ `ThreadPoolExecutor` (blocking CUDA call).

`load()` — khởi tạo model theo 5 bước:
1. `ASRModel.from_pretrained()` trên configured device
2. `.eval()` — tắt dropout/batch-norm
3. `set_default_att_context_size()` — áp dụng look-ahead frames từ preset
4. `change_decoding_strategy()` nếu khác `greedy_batch`
5. `set_inference_prompt()` — set ngôn ngữ mặc định (**bắt buộc** — thiếu thì model trả chuỗi rỗng)

`stream_step(pcm_bytes, sample_rate, cache, lang)` → `(text, new_cache)`:
1. Preprocess: int16 → float32 → resample nếu cần → log-mel qua model's preprocessor
2. Set language prompt (dùng `threading.Lock` vì là shared model state)
3. `conformer_stream_step()` → trả về 6 giá trị: greedy predictions, transcripts, att_cache mới, conv_cache mới, att_cache_len mới, best_hyp
4. Extract text (RNNT trả `Hypothesis` object, CTC trả `str`)
5. Pack `ASRCacheState` mới và trả về

`stream_step_batch(requests)` — chạy nhiều session tuần tự trong một call (placeholder cho true GPU batching B > 1).

## BatchScheduler (`app/asr/scheduler.py`)

Định tuyến inference requests đến engine. Mode được cấu hình qua `ASR_BATCH_MODE`:

`per_session` — mỗi request được dispatch ngay qua `loop.run_in_executor(thread_pool, engine.stream_step, ...)`.

`dynamic` — requests được đưa vào `asyncio.Queue`; background worker gom tối đa `max_batch_size` requests trong `batch_timeout_ms` rồi gọi `engine.stream_step_batch()` một lần.

- `submit(pcm_bytes, sample_rate, cache, lang)` → `(text, new_cache)` — interface chung cho cả hai mode; caller chỉ cần `await` mà không cần biết mode
- `start()` / `stop()` — lifecycle; `stop()` hủy worker task và shutdown thread pool

## Schema (`app/schema/`)

Pydantic models cho tất cả message types:

- `audio.py` — `AudioMessage`: `type="audio"`, `data` (base64 PCM int16), `sample_rate`, `lang`
- `session.py` — `SessionInfoMessage`: `session_id`, `preset`, `chunk_ms`, `att_context_size`, `packets_per_chunk`, `batch_mode`
- `transcript.py` — `TranscriptMessage`: `session_id`, `text`, `is_final`, `lang_detected`, `duration_ms`
