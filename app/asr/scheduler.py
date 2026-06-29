"""
BatchScheduler — routes inference requests either immediately (per_session)
or in time-bounded batches (dynamic).

Switching modes requires only one config change:
  ASR_BATCH_MODE=dynamic   or   batch_mode: dynamic  in settings.yaml

per_session:  each request triggers its own run_in_executor call immediately.
dynamic    :  requests are queued; a background worker collects up to
              max_batch_size within batch_timeout_ms, then dispatches them
              together as a single engine.stream_step_batch() call.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from config import settings

if TYPE_CHECKING:
    from app.asr.engine import NemoStreamingEngine
    from app.session.state import ASRCacheState

logger = logging.getLogger(__name__)


@dataclass
class _Request:
    """One queued inference job; future receives (text, new_cache) when done."""

    pcm_bytes: bytes
    sample_rate: int
    cache: Any          # ASRCacheState
    future: asyncio.Future
    lang: str = "auto"


class BatchScheduler:
    """Routes inference requests to the engine, either immediately or as time-bounded batches."""

    def __init__(self,
                 engine: "NemoStreamingEngine") -> None:
        self._engine = engine
        self._thread_pool = ThreadPoolExecutor(
            max_workers=settings.thread_pool_workers,
            thread_name_prefix="nemo-infer",
        )
        self._queue: asyncio.Queue[_Request] = asyncio.Queue(
            maxsize=settings.max_sessions * settings.max_pending_per_session
        )
        self._worker: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def queue_depth(self) -> int:
        """Current number of inference requests waiting in the queue."""
        return self._queue.qsize()

    async def start(self) -> None:
        """Capture the event loop and, in dynamic mode, launch the batch worker task."""
        self._loop = asyncio.get_running_loop()
        if settings.batch_mode == "dynamic":
            self._worker = asyncio.create_task(
                self._batch_worker(), name="batch-scheduler"
            )
            logger.info(
                "batch scheduler started — max_batch=%d timeout=%dms",
                settings.max_batch_size,
                settings.batch_timeout_ms,
            )
        else:
            logger.info("inference mode: per_session (no batching)")

    async def stop(self) -> None:
        """Cancel the batch worker and shut down the thread pool, discarding pending jobs."""
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._thread_pool.shutdown(wait=False, cancel_futures=True)

    # ── Public interface ───────────────────────────────────────────────────────

    async def submit(self,
                     pcm_bytes: bytes,
                     sample_rate: int,
                     cache: "ASRCacheState",
                     lang: str = "auto") -> tuple[str, "ASRCacheState"]:
        """Submit one inference request and await its result.

        Args:
            pcm_bytes: Raw audio chunk encoded as int16 PCM.
            sample_rate: Sample rate of the input audio in Hz.
            cache: Per-session cache state.
            lang: BCP-47 language tag or "auto".

        Returns:
            Tuple of (transcript_text, updated_cache).
        """
        if settings.batch_mode == "per_session":
            return await self._dispatch_single(pcm_bytes, sample_rate, cache, lang)

        future = self._loop.create_future()
        await self._queue.put(_Request(pcm_bytes, sample_rate, cache, future, lang))
        return await future

    # ── per_session path ───────────────────────────────────────────────────────

    async def _dispatch_single(self,
                               pcm_bytes: bytes,
                               sample_rate: int,
                               cache: "ASRCacheState",
                               lang: str = "auto") -> tuple[str, "ASRCacheState"]:
        """Offload one blocking engine call to the thread pool.

        Args:
            pcm_bytes: Raw audio chunk encoded as int16 PCM.
            sample_rate: Sample rate of the input audio in Hz.
            cache: Per-session cache state.
            lang: BCP-47 language tag or "auto".

        Returns:
            Tuple of (transcript_text, updated_cache).
        """
        import time
        from app.services.metrics import stats

        t0 = time.perf_counter()
        result = await self._loop.run_in_executor(
            self._thread_pool,
            self._engine.stream_step,
            pcm_bytes,
            sample_rate,
            cache,
            lang,
        )
        stats.record_batch(1, (time.perf_counter() - t0) * 1000)
        return result

    # ── dynamic batching path ──────────────────────────────────────────────────

    async def _batch_worker(self) -> None:
        """Collect and dispatch batches in a loop; each dispatch is fire-and-forget."""
        while True:
            try:
                batch = await self._collect_batch()
                asyncio.create_task(
                    self._dispatch_batch(batch),
                    name=f"batch-{len(batch)}",
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("batch worker error")

    async def _collect_batch(self) -> list[_Request]:
        """Wait for the first request, then collect more until timeout or max size.

        Returns:
            List of collected _Request objects ready for batch dispatch.
        """
        # 1. block until at least one request is available
        batch = [await self._queue.get()]

        # 2. deadline anchored to first item — same window regardless of drain speed
        deadline = self._loop.time() + settings.batch_timeout_ms / 1000.0

        # 3. drain any already-queued requests immediately (no event-loop overhead)
        while len(batch) < settings.max_batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        # 4. wait for stragglers within the remaining window
        while len(batch) < settings.max_batch_size:
            remaining = deadline - self._loop.time()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(req)
            except asyncio.TimeoutError:
                break

        return batch

    async def _dispatch_batch(self,
                              batch: list[_Request]) -> None:
        """Split batch by (lang, has_decoder_state) and run each group in parallel.

        Each group acquires an independent model instance from the pool, so groups
        with different languages or decoder states run concurrently instead of
        sequentially. Exceptions in one group are forwarded only to that group's
        futures; other groups are unaffected.
        """
        import time
        from collections import defaultdict
        from app.services.metrics import stats

        t0 = time.perf_counter()

        # Requests with different lang or decoder-state shape can't share a forward pass.
        groups: dict[tuple, list[tuple[int, _Request]]] = defaultdict(list)
        for i, req in enumerate(batch):
            has_dec = req.cache.hypotheses is not None or req.cache.pred_out is not None
            groups[(req.lang, has_dec)].append((i, req))

        # Indices carry original batch positions so futures resolve correctly after gather.
        outcomes = await asyncio.gather(
            *[self._run_group(g) for g in groups.values()]
        )
        stats.record_batch(len(batch), (time.perf_counter() - t0) * 1000)

        for i, res in (p for grp in outcomes for p in grp):
            try:
                if isinstance(res, Exception):
                    batch[i].future.set_exception(res)
                else:
                    batch[i].future.set_result(res)
            except asyncio.InvalidStateError:
                pass

    async def _run_group(self,
                         indexed: list[tuple[int, _Request]]) -> list[tuple[int, object]]:
        """Infer one homogeneous sub-batch; return (original_index, result_or_exc) pairs."""
        indices = [i for i, _ in indexed]
        engine_reqs = [(r.pcm_bytes, r.sample_rate, r.cache, r.lang) for _, r in indexed]
        try:
            group_results = await self._loop.run_in_executor(
                self._thread_pool,
                self._engine._batch_infer,
                engine_reqs,
            )
            return list(zip(indices, group_results))
        except Exception as exc:
            logger.exception(
                "group inference failed (size=%d lang=%s)",
                len(engine_reqs),
                indexed[0][1].lang,
            )
            return [(i, exc) for i in indices]


