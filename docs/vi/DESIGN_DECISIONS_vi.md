# Quyết Định Kỹ Thuật — StreamingNenotronASR

Tài liệu này giải thích lý do đằng sau từng lựa chọn kiến trúc quan trọng, bao gồm các đánh đổi và các phương án đã được cân nhắc.

---

## 1. Transport — WebSocket với JSON text frames

Toàn bộ luồng audio đi qua một WebSocket connection duy nhất và bền vững. Client gửi typed JSON frames (`start` / `audio` / `end`); server stream transcript fragments về real-time theo chiều ngược lại. Audio binary được encode thành base64 để giữ trong JSON frame. Message `"start"` reset decoder state trong cùng connection thay vì đóng/mở lại — encoder cache được giữ nguyên để đảm bảo acoustic continuity giữa các utterances.

### Pros

- **Full-duplex:** Client stream audio lên trong khi server stream kết quả về — không cần polling, latency thấp nhất có thể.
- **State gắn với connection:** Không cần session token hay re-authentication giữa các utterances; connection tồn tại thì state tồn tại.
- **Debug dễ:** JSON frame đọc được bằng mắt thường trong `wscat` hoặc browser DevTools mà không cần tool đặc biệt.

### Cons

- **Base64 overhead ~33%:** Audio binary phải encode thành base64 để nằm trong JSON — ở 43 KB/s per session là chấp nhận được, nhưng cộng dồn khi nhiều sessions đồng thời.
- **Không cache-friendly:** WebSocket không đi qua CDN hay HTTP cache — cần lưu ý khi thiết kế load balancing.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| HTTP chunked transfer / SSE | Một chiều — không thể multiplex audio upload và result stream trong cùng một connection |
| gRPC bidirectional streaming | Binary-native, hiệu quả hơn về bandwidth, nhưng yêu cầu proto schema và HTTP/2 infrastructure — overhead thiết lập không xứng với quy mô hiện tại |
| REST polling | Latency cao do polling interval; không phù hợp với real-time streaming |

---

## 2. Audio Buffer — Accumulator đơn giản, không phải ring buffer

Incoming audio packets tích lũy trong `AudioChunkBuffer` — một accumulator đơn giản — cho đến khi đủ `chunk_size` thì flush sang inference. Chunk cuối được pad bằng byte `\x00` (silence) thay vì truncate để giữ tensor shape nhất quán. Không dùng ring buffer vì cache-aware model tự giữ acoustic context qua `att_cache` — không cần truy cập lại audio window tùy ý.

### Pros

- **Tensor shape nhất quán:** Preprocessor luôn nhận input cùng shape — không có code path đặc biệt cho chunk ngắn.
- **Tiết kiệm VRAM:** Không tốn ~384 KB per session cho ring buffer mà không dùng đến.
- **Zero-padding acoustic safe:** NeMo không emit token giả cho đoạn silence ngắn cuối chunk — padding không ảnh hưởng transcript.

### Cons

- **Không thể replay audio:** Không thể re-read audio window tùy ý sau khi đã flush — nếu sau này cần VAD trimming hoặc final window extraction, phải refactor.
- **Thêm latency nhẹ ở cuối utterance:** Chunk cuối phải chờ pad đủ kích thước thay vì xử lý ngay.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Ring buffer cố định (12s) | Hữu ích khi cần truy cập audio window tùy ý; với cache-aware model, là overhead không cần thiết (~384 KB per session) |
| Queue của raw audio packets | Không thể query theo time range; memory overhead tỷ lệ với packet rate |

---

## 3. Inference Engine — NeMo Cache-Aware Streaming

NeMo cache-aware model xử lý audio theo từng chunk cố định. Mỗi session giữ riêng `att_cache` và `conv_cache` — cache encode acoustic context của speaker đó và được cập nhật sau mỗi chunk inference. Khi utterance mới bắt đầu (`"start"`), decoder cache (`hypotheses`, `pred_out`) được reset nhưng encoder cache được giữ nguyên để acoustic continuity không bị gián đoạn. Feature extraction delegate hoàn toàn cho `model.preprocessor()` của NeMo thay vì tự tính log-mel — đảm bảo không có mismatch với lúc train.

### Pros

- **Chất lượng cao:** Left context được giữ qua cache — model "nhớ" những gì đã nghe; không có feature mismatch giữa train và inference.
- **Latency thấp:** Mỗi chunk là một incremental forward pass nhỏ — không re-encode toàn bộ audio từ đầu.
- **Acoustic continuity giữa utterances:** Encoder cache không bị reset giữa các câu liên tiếp — model không bị "ngạc nhiên" với âm đầu utterance tiếp theo.

### Cons

- **VRAM tỷ lệ tuyến tính với sessions:** Mỗi session giữ riêng cache tensors — không thể share giữa các sessions. Cần tuning `max_sessions` theo VRAM thực tế.
- **Cache không portable:** Cache tensors gắn với một session cụ thể; không thể migrate session sang GPU khác hay serialize để resume.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Stateless inference (re-encode full audio mỗi chunk) | Đơn giản hơn nhưng mất left context — quality giảm đáng kể, đặc biệt ở cuối câu dài |
| Offline batch inference | Latency thấp không đạt được; không phù hợp với real-time streaming |
| Share encoder cache giữa sessions | Sai về mặt acoustic — cache của speaker này corrupt context của speaker khác |

---

## 4. Concurrency — asyncio + ThreadPoolExecutor

FastAPI's asyncio event loop xử lý toàn bộ WebSocket I/O (non-blocking). GPU inference là blocking CUDA call — được offload sang thread pool qua `run_in_executor` để không block event loop. Engine duy trì pool `model_pool_size` instances độc lập; mỗi thread acquire một instance độc quyền — không cần `threading.Lock` cho language-prompt writes.

### Pros

- **Nhiều connections đồng thời:** Event loop không bao giờ block vì inference — hàng chục WebSocket connections xử lý I/O đồng thời.
- **Shared memory:** Threads chia sẻ process memory — cache tensors per-session không cần serialize, không có IPC overhead.
- **CUDA release GIL:** Python GIL không block GPU work — thread pool thực sự song song hóa inference across sessions.
- **Không có race condition lang-prompt:** Pool isolation đảm bảo `set_inference_prompt` và `conformer_stream_step` luôn chạy trên cùng một object.

### Cons

- **CUDA error handling phức tạp hơn:** Exception trong thread phải được propagate về coroutine chính — kém ergonomic hơn so với xử lý trực tiếp trong event loop.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| `multiprocessing` | Cần serialize tensors qua IPC — không thực tế với cache tensors per-session; overhead cao |
| Synchronous server (Flask + gunicorn) | Không tương thích với asyncio WebSocket model; kém scalable khi I/O-heavy |
| Separate inference microservice qua queue | Thêm network hop và serialization overhead; phức tạp hóa deployment cho lợi ích không rõ ở quy mô hiện tại |

---

## 5. Session Management & VRAM Lifecycle

Mỗi WebSocket connection tạo một `StreamingSession` với cache tensors riêng. `SessionManager` theo dõi tất cả active sessions, enforce giới hạn `max_sessions` (đóng với WebSocket close code 1008 khi đầy), và giải phóng VRAM ngay lập tức khi disconnect bằng cách null out các tensor references — không chờ GC. `ServiceRegistry` raise `RuntimeError` thay vì trả `None` nếu bị truy cập trước khi `startup()` hoàn thành.

### Pros

- **VRAM release deterministic:** Không phụ thuộc GC timing — VRAM được trả về ngay khi session kết thúc, tránh OOM trong workload nhiều connect/disconnect liên tục.
- **Hard capacity limit:** Server không bao giờ vượt quá VRAM budget; client nhận signal rõ ràng (code 1008) thay vì connection treo.
- **Fail loud:** `RuntimeError` từ registry xuất hiện ngay tại nguồn, không phải ở vị trí ngẫu nhiên khi `None` bị dereference.

### Cons

- **Không có queuing:** Client bị reject ngay khi đầy thay vì chờ slot trống — retry logic phải implement phía client.
- **`max_sessions` cần tuning thủ công:** Không có auto-scaling theo VRAM thực tế còn lại — cần profiling để set giá trị phù hợp.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Để GC giải phóng VRAM tự nhiên | Non-deterministic — dưới high load, VRAM có thể bị exhausted trước khi GC cycle chạy, gây OOM |
| Queue chờ slot trống khi đầy | VRAM vẫn bị giữ trong khi chờ; tăng complexity server mà không giải quyết root cause |
| `__del__` method trên session object | Timing không guaranteed trong CPython, đặc biệt khi có circular references |

---

## 6. Batch Scheduler — `per_session` và `dynamic`

`BatchScheduler` hỗ trợ hai mode cấu hình qua env var. `per_session` inference ngay khi chunk của một session sẵn sàng — latency thấp nhất, phù hợp khi số sessions thấp. `dynamic` gom requests từ nhiều sessions trong một time window rồi thực hiện một GPU call — throughput cao hơn, chấp nhận thêm `batch_timeout_ms` latency. `_collect_batch()` dùng two-phase drain: sau khi block trên item đầu tiên, ngay lập tức drain hết items đã có sẵn bằng `get_nowait()` trước khi async wait các item còn lại — tránh event-loop round-trip dư thừa khi queue đã có sẵn nhiều requests. Chuyển mode không cần sửa code hay rebuild image.

### Pros

- **Linh hoạt theo workload:** Cùng codebase phục vụ cả trường hợp single-user latency-sensitive và multi-user throughput-optimized.
- **Không cần rebuild:** Đổi mode chỉ cần thay đổi env var — phù hợp với deployment có nhiều môi trường khác nhau.

### Cons

- **`dynamic` mode thêm latency cố định:** Mỗi inference step phải chờ `batch_timeout_ms` để gom batch — không phù hợp khi latency là ưu tiên số một.
- **`dynamic` mode phức tạp hơn để debug:** Nhiều sessions gộp chung vào một GPU call — khó isolate vấn đề của một session cụ thể.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Chỉ `per_session` | Lãng phí GPU khi nhiều sessions đồng thời — mỗi session chiếm một GPU call riêng dù GPU có thể xử lý nhiều hơn |
| Chỉ `dynamic` | Thêm latency không cần thiết trong workload single-user hoặc session thấp |
| Adaptive batching (tự điều chỉnh window) | Tăng complexity scheduler; chưa có dữ liệu profiling thực tế để calibrate |

---

## 7. Model Pool — pool-based instance isolation

Pool `model_pool_size` instances độc lập (`queue.Queue`) thay thế thiết kế cũ (single instance + `threading.Lock`). Giữa `set_inference_prompt` và `conformer_stream_step` có một window mà thread khác có thể ghi đè language prompt lên shared instance — pool acquisition đóng window đó mà không cần lock.

### Pros
- **Không có race condition:** `set_inference_prompt` và `conformer_stream_step` chạy trên cùng object, không có write đồng thời.
- **Code đơn giản hơn:** Không cần quản lý lock; pool semantics tường minh.

### Cons
- **VRAM nhân lên:** Mỗi instance tốn ~1.2 GB; `pool_size=2` tăng gấp đôi model VRAM.
- **CUDA stream hazard khi pool_size > 1:** Concurrent kernels trên default stream gây illegal memory access — cần per-instance stream isolation trước khi bật.

---

## 8. Bounded inference queue per session

Mỗi session theo dõi `pending_infer` (số request đang in-flight). Non-final chunks vượt `max_pending_per_session` bị drop. Ngăn queue depth tăng khi GPU throughput thấp hơn audio rate — giữ session current thay vì xử lý audio cũ.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Unbounded queue | Queue phình to khi GPU chậm; latency tăng dần; xử lý audio cũ sau khi audio mới đã đến |
| Block sender | Sẽ stall asyncio event loop và delay toàn bộ sessions khác |

---

## 10. Chế độ bfloat16

Khi `use_bf16=true` và GPU hỗ trợ bfloat16 (Ampere+), toàn bộ model được cast sang `bfloat16` và preprocessor được restore về `float32`. Mel features được cast sang bfloat16 qua `_to_encoder_dtype()` ngay trước encoder forward pass. Cast toàn model một lần tránh mismatch dtype giữa các submodule (encoder, decoder, joint network). Preprocessor giữ float32 vì tính toán filterbank và log-mel cần độ chính xác float32; chỉ cần cast tường minh tại ranh giới đầu vào encoder.

### Pros
- **Giảm VRAM per instance:** bfloat16 giảm một nửa bộ nhớ model weight (~600 MB vs ~1.2 GB), dùng được trên GPU nhỏ hơn.
- **Không có overflow risk:** bfloat16 có cùng dải exponent với float32 — attention softmax không thể overflow, khác với float16.

### Cons
- **Chỉ hỗ trợ Ampere+:** Fallback về float32 kèm warning trên GPU cũ hơn; `_resolve_bf16()` kiểm tra `torch.cuda.is_bf16_supported()` khi load.
- **Chi phí cast tại ranh giới preprocessor:** Một lần cast tường minh per batch ở đầu vào encoder; chi phí không đáng kể trong thực tế.

---

## 11. Idle session sweeper

Background asyncio task poll mỗi `session_sweep_interval_s` (default 30s), evict sessions im lặng quá `idle_timeout_s` (default 60s). Bổ sung uvicorn WS ping/pong (xử lý dead TCP); sweeper xử lý trường hợp "ghost session" — TCP alive nhưng client không gửi audio.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Chỉ dùng WS ping/pong | Ping/pong phát hiện dead TCP; không phát hiện được TCP alive nhưng không có audio |
| Client tự đóng session | Yêu cầu mọi client implement teardown đúng; không thể enforce |
