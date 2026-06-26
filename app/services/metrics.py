"""
Server-wide metrics collected in memory and exposed via /health/stats.

All state lives in the module-level `stats` singleton — import and use directly.
Thread-safe for reads; counters use simple int addition which is GIL-protected.
"""
import statistics
from collections import deque


class _Stats:
    def __init__(self) -> None:
        self.drop_count: int = 0
        self.inference_count: int = 0
        # rolling window of (scheduler_batch_size, latency_ms) tuples
        self._batch_samples: deque[tuple[int, float]] = deque(maxlen=200)
        # rolling window of actual GPU forward-pass batch sizes (post group-split)
        self._gpu_batch_samples: deque[int] = deque(maxlen=500)

    def record_drop(self) -> None:
        self.drop_count += 1

    def record_batch(self, batch_size: int, latency_ms: float) -> None:
        self.inference_count += 1
        self._batch_samples.append((batch_size, latency_ms))

    def record_gpu_batch(self, batch_size: int) -> None:
        """Record the actual B used in one conformer_stream_step call."""
        self._gpu_batch_samples.append(batch_size)

    def snapshot(self) -> dict:
        samples = list(self._batch_samples)
        latencies = [s[1] for s in samples]
        avg_batch = (
            sum(s[0] for s in samples) / len(samples) if samples else 0.0
        )
        gpu_samples = list(self._gpu_batch_samples)
        avg_gpu_batch = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0.0

        latency_stats: dict = {"samples": len(latencies)}
        if latencies:
            latency_stats["p50"] = round(statistics.median(latencies), 1)
            latency_stats["p99"] = round(
                sorted(latencies)[int(len(latencies) * 0.99)], 1
            )
        return {
            "avg_batch_size": round(avg_batch, 1),        # scheduler-level (before group split)
            "avg_gpu_batch_size": round(avg_gpu_batch, 1), # GPU-level (actual conformer_stream_step B)
            "batch_latency_ms": latency_stats,
        }


stats = _Stats()
