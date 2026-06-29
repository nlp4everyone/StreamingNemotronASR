import logging
import queue
import time
from collections import defaultdict
import numpy as np
import torch
import torchaudio
from config import settings
logger = logging.getLogger(__name__)

class NemoStreamingEngine:
    """
    Wraps nvidia/nemotron-3.5-asr-streaming-0.6b via NeMo.

    Maintains a pool of model_pool_size independent model instances.
    Each inference call acquires one instance exclusively — set_inference_prompt
    and conformer_stream_step execute on the same object with no concurrent
    access, eliminating the lang-prompt race condition for multi-language use.
    """

    def __init__(self) -> None:
        self._pool: queue.Queue = queue.Queue()
        self._device = settings.device
        self._loaded = False
        self._use_bf16: bool = False
        # Zero-fill tensors for new sessions joining a batch that already has
        # established sessions. Allocated once on first mixed batch, reused forever
        # — avoids a GPU malloc per new session (burst-connect CUDA alloc storm).
        self._zero_att: torch.Tensor | None = None
        self._zero_conv: torch.Tensor | None = None
        self._zero_len: torch.Tensor | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model_pool_size instances onto the configured device."""
        n = settings.model_pool_size
        self._use_bf16 = self._resolve_bf16()
        logger.info(
            "loading %d instance(s) of %s on %s ...", n, settings.model_name, self._device
        )
        t0 = time.perf_counter()
        for i in range(n):
            self._pool.put(self._load_one(i, n))
        self._loaded = True
        logger.info(
            "model pool ready (%d instances) in %.1fs — preset=%s att_context=%s lang=%s bf16=%s",
            n,
            time.perf_counter() - t0,
            settings.preset.name,
            settings.preset.att_context_size,
            settings.default_lang,
            self._use_bf16,
        )

    def _resolve_bf16(self) -> bool:
        """Return True only when bf16 is requested and the device supports it."""
        if not settings.use_bf16 or self._device == "cpu":
            return False
        if not torch.cuda.is_bf16_supported():
            logger.warning(
                "use_bf16=true but GPU does not support bfloat16 — falling back to float32"
            )
            return False
        return True

    def _load_one(self, idx: int, total: int):
        import nemo.collections.asr as nemo_asr
        from nemo.utils import logging as nemo_logging

        nemo_logging.set_verbosity(nemo_logging.ERROR)

        t1 = time.perf_counter()
        logger.info("loading instance %d/%d ...", idx + 1, total)

        # 1. download/load weights onto target device
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=settings.model_name,
            map_location=self._device,
        )
        model.eval()

        # 2. configure streaming context window
        model.encoder.set_default_att_context_size(list(settings.preset.att_context_size))

        # 3. apply decoding overrides
        if settings.decoding_strategy != "greedy_batch":
            model.change_decoding_strategy(
                decoding_cfg={"strategy": settings.decoding_strategy}
            )
        if settings.max_symbols_per_step != 10:
            model.decoding.cfg.greedy.max_symbols = settings.max_symbols_per_step

        # 4. set default language prompt
        if hasattr(model, "set_inference_prompt"):
            model.set_inference_prompt(settings.default_lang)

        if self._use_bf16:
            # Cast the full model to bfloat16, then restore preprocessor to float32.
            # Casting the full model avoids dtype mismatches across submodules (encoder,
            # decoder, joint network, etc.). Preprocessor (filterbank + log-mel) must
            # stay float32; mel is cast to bfloat16 just before the encoder forward pass.
            # bfloat16 is safe: same exponent range as float32, attention softmax cannot overflow.
            model.bfloat16()
            model.preprocessor.float()
            logger.info("instance %d/%d: model cast to bfloat16 (preprocessor kept float32)", idx + 1, total)

        logger.info("instance %d/%d ready in %.1fs", idx + 1, total, time.perf_counter() - t1)
        return model

    def is_ready(self) -> bool:
        return self._loaded

    @property
    def bf16_enabled(self) -> bool:
        return self._use_bf16

    def _to_encoder_dtype(self,
                          mel: torch.Tensor) -> torch.Tensor:
        """Cast mel features to match encoder weight dtype when bf16 is enabled."""
        return mel.bfloat16() if self._use_bf16 else mel

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocess(self,
                    pcm_bytes: bytes,
                    sample_rate: int,
                    model) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw PCM audio to log-mel spectrogram features.

        Args:
            pcm_bytes: Raw audio samples encoded as int16.
            sample_rate: Sample rate of the input audio in Hz.
            model: The model instance whose preprocessor to use.

        Returns:
            Tuple of (mel, mel_len) where mel has shape [1, D, T] and mel_len has shape [1].
        """
        # 1. decode int16 PCM → normalized float32, convert to torch once
        waveform = torch.from_numpy(
            np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        ) / 32768.0

        # 2. resample to 16 kHz if needed — stays in torch, no roundtrip back to numpy
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0), sample_rate, 16000
            ).squeeze(0)

        # 3. move to device and extract log-mel
        audio_t = waveform.unsqueeze(0).to(self._device)
        audio_len = torch.tensor([waveform.shape[0]], dtype=torch.long, device=self._device)

        with torch.inference_mode():
            mel, mel_len = model.preprocessor(input_signal=audio_t, length=audio_len)
        return mel, mel_len

    def _preprocess_batch(self,
                          requests: list[tuple[bytes, int, "ASRCacheState", str]],
                          model) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert all PCM chunks to log-mel in one batched GPU call.

        Returns:
            (mel_batch [B, D, T], mel_len_batch [B]) cast to encoder dtype.
        """
        B = len(requests)
        max_chunk_samples = settings.preset.chunk_frames * 1280

        if all(sr == 16000 for _, sr, _, _ in requests):
            # Fast path: all 16 kHz — one numpy stack, one torch conversion, no padding.
            # Buffer guarantees every chunk is exactly target_bytes long, so all arrays
            # share the same shape and np.stack never needs to pad.
            arrays = [np.frombuffer(pb, dtype=np.int16) for pb, _, _, _ in requests]
            audio_np = np.stack(arrays).astype(np.float32)  # [B, T], one alloc
            audio_np /= 32768.0                             # in-place, no extra alloc
            T = min(audio_np.shape[1], max_chunk_samples)
            src = audio_np if T == audio_np.shape[1] else np.ascontiguousarray(audio_np[:, :T])
            audio_batch = torch.from_numpy(src)             # zero-copy share
            clamped_lengths = [T] * B
        else:
            # Slow path: mixed sample rates — decode per chunk with resampling, then
            # preallocate with torch.empty (no upfront zero-fill) and fill in-place;
            # only the tail-padding region is explicitly zeroed.
            waveforms: list[torch.Tensor] = []
            for pcm_bytes, sample_rate, _, _ in requests:
                waveform = torch.from_numpy(
                    np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                ) / 32768.0
                if sample_rate != 16000:
                    waveform = torchaudio.functional.resample(
                        waveform.unsqueeze(0), sample_rate, 16000
                    ).squeeze(0)
                waveforms.append(waveform)

            T_max = min(max(w.shape[0] for w in waveforms), max_chunk_samples)
            audio_batch = torch.empty(B, T_max, dtype=torch.float32)
            clamped_lengths = []
            for i, w in enumerate(waveforms):
                n = min(w.shape[0], T_max)
                audio_batch[i, :n] = w[:n]
                if n < T_max:
                    audio_batch[i, n:].zero_()
                clamped_lengths.append(n)

        # 3. move batch to device and extract log-mel
        audio_t = audio_batch.to(self._device)
        audio_len = torch.tensor(clamped_lengths, dtype=torch.long, device=self._device)

        with torch.inference_mode():
            mel_batch, mel_len_batch = model.preprocessor(
                input_signal=audio_t, length=audio_len
            )

        return self._to_encoder_dtype(mel_batch), mel_len_batch

    # ── Single-session inference ───────────────────────────────────────────────

    def stream_step(self,
                    pcm_bytes: bytes,
                    sample_rate: int,
                    cache: "ASRCacheState",
                    lang: str = "auto") -> tuple[str, "ASRCacheState"]:  # noqa: F821
        """Run one streaming inference step for a single session.

        Acquires an exclusive model instance from the pool — set_inference_prompt
        and conformer_stream_step are guaranteed to run on the same object with no
        concurrent writes from other threads.

        Args:
            pcm_bytes: Raw audio chunk encoded as int16 PCM.
            sample_rate: Sample rate of the input audio in Hz.
            cache: Per-session attention and convolution cache state.
            lang: BCP-47 language tag or "auto" for automatic detection.

        Returns:
            Tuple of (transcript_text, updated_cache).
        """
        from app.session.state import ASRCacheState

        model = self._pool.get()
        try:
            # 1. preprocess → [1, D, T], cast dtype
            mel, mel_len = self._preprocess(pcm_bytes, sample_rate, model)
            mel = self._to_encoder_dtype(mel)

            # 2. set language — safe: this instance is not shared with any other thread
            if hasattr(model, "set_inference_prompt"):
                model.set_inference_prompt(lang)

            # 3. single GPU forward pass
            with torch.inference_mode():
                (
                    greedy_predictions,
                    all_hyp_or_transcribed_texts,
                    new_att_cache,
                    new_conv_cache,
                    new_att_cache_len,
                    best_hyp,
                ) = model.conformer_stream_step(
                    processed_signal=mel,
                    processed_signal_length=mel_len,
                    cache_last_channel=cache.att_cache,
                    cache_last_time=cache.conv_cache,
                    cache_last_channel_len=cache.att_cache_len,
                    keep_all_outputs=False,
                    previous_hypotheses=cache.hypotheses,
                    previous_pred_out=cache.pred_out,
                    drop_extra_pre_encoded=None,
                    return_transcription=True,
                )
        finally:
            self._pool.put(model)

        first = all_hyp_or_transcribed_texts[0] if all_hyp_or_transcribed_texts else None
        text = _extract_text(first)
        logger.debug("inference: type=%s text=%r", type(first).__name__, text)

        return text, ASRCacheState(
            att_cache=new_att_cache,
            conv_cache=new_conv_cache,
            att_cache_len=new_att_cache_len,
            hypotheses=best_hyp,
            pred_out=greedy_predictions,
        )

    # ── Batch inference ────────────────────────────────────────────────────────

    def stream_step_batch(self,
                          requests: list[tuple[bytes, int, "ASRCacheState", str]]) -> list[tuple[str, "ASRCacheState"]]:  # noqa: F821
        """Run inference for multiple sessions in one GPU call.

        Groups requests by (lang, has_decoder_state) so each sub-batch has
        homogeneous cache shapes. Each group acquires an exclusive model instance
        from the pool — no lang-prompt race condition.

        Args:
            requests: List of (pcm_bytes, sample_rate, cache, lang) tuples.

        Returns:
            List of (transcript_text, updated_cache) in the same order as requests.
        """
        if len(requests) == 1:
            pcm, sr, cache, lang = requests[0]
            return [self.stream_step(pcm, sr, cache, lang)]

        # group by (lang, has_decoder_state) — each sub-batch must be homogeneous
        groups: dict[tuple, list[tuple[int, tuple]]] = defaultdict(list)
        for i, req in enumerate(requests):
            _, _, cache, lang = req
            has_dec = cache.hypotheses is not None or cache.pred_out is not None
            groups[(lang, has_dec)].append((i, req))

        results: list = [None] * len(requests)
        for group in groups.values():
            indices = [i for i, _ in group]
            group_reqs = [req for _, req in group]
            for idx, res in zip(indices, self._batch_infer(group_reqs)):
                results[idx] = res

        return results

    def _batch_infer(self,
                     requests: list[tuple[bytes, int, "ASRCacheState", str]]) -> list[tuple[str, "ASRCacheState"]]:  # noqa: F821
        """True GPU batch inference for a homogeneous group of requests.

        Acquires an exclusive model instance — set_inference_prompt and
        conformer_stream_step execute on the same object, safe from concurrent writes.

        Args:
            requests: Homogeneous list of (pcm_bytes, sample_rate, cache, lang).

        Returns:
            List of (transcript_text, updated_cache) in the same order.
        """
        from app.services.metrics import stats
        from app.session.state import ASRCacheState

        B = len(requests)
        stats.record_gpu_batch(B)
        model = self._pool.get()
        try:
            # 1. batch preprocess → [B, D, T_max], [B]
            mel_batch, mel_len_batch = self._preprocess_batch(requests, model)

            # 2. stack encoder caches → [layers, B, D, T]
            caches = [r[2] for r in requests]
            att_batch, conv_batch, att_len_batch = self._stack_encoder_caches(caches)

            # 3. decoder state — None if all sessions are on first chunk
            all_hyps = [c.hypotheses for c in caches]
            previous_hypotheses = (
                None if all(h is None for h in all_hyps) else [h[0] for h in all_hyps]
            )
            all_preds = [c.pred_out for c in caches]
            previous_pred_out = (
                None if all(p is None for p in all_preds) else [p[0] for p in all_preds]
            )

            # 4. set language — safe: this instance is not shared with any other thread
            lang = requests[0][3]
            if hasattr(model, "set_inference_prompt"):
                model.set_inference_prompt(lang)

            # 5. single GPU forward pass for the whole batch
            with torch.inference_mode():
                (
                    greedy_predictions,
                    all_hyp_or_texts,
                    new_att_batch,
                    new_conv_batch,
                    new_att_len_batch,
                    best_hyp_batch,
                ) = model.conformer_stream_step(
                    processed_signal=mel_batch,
                    processed_signal_length=mel_len_batch,
                    cache_last_channel=att_batch,
                    cache_last_time=conv_batch,
                    cache_last_channel_len=att_len_batch,
                    keep_all_outputs=False,
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=previous_pred_out,
                    drop_extra_pre_encoded=None,
                    return_transcription=True,
                )

            # 6. split batch outputs back into per-session (text, cache) tuples
            # Kept inside the pool critical section so pool.put() only happens after
            # all copy_() ops are submitted — prevents Thread B from grabbing the model
            # mid-group (between en and vi sub-batches in stream_step_batch) which would
            # delay the second group's inference and desynchronize sessions.
            logger.debug("batch inference B=%d lang=%s", B, lang)
            results = []
            for i in range(B):
                item = all_hyp_or_texts[i] if all_hyp_or_texts else None
                src = caches[i]
                results.append((
                    _extract_text(item),
                    ASRCacheState(
                        att_cache=_slice_inplace(src.att_cache, new_att_batch, i) if new_att_batch is not None else None,
                        conv_cache=_slice_inplace(src.conv_cache, new_conv_batch, i) if new_conv_batch is not None else None,
                        att_cache_len=_slice_inplace(src.att_cache_len, new_att_len_batch, i) if new_att_len_batch is not None else None,
                        hypotheses=[best_hyp_batch[i]] if best_hyp_batch is not None else None,
                        pred_out=[greedy_predictions[i]],
                    ),
                ))
        finally:
            self._pool.put(model)

        return results

    def _stack_encoder_caches(self,
                              caches: list["ASRCacheState"]) -> tuple["torch.Tensor | None", "torch.Tensor | None", "torch.Tensor | None"]:
        """Concatenate per-session encoder caches along the batch dimension (dim=1).

        Returns (None, None, None) when all caches are None so NeMo initialises
        its own zero caches. Slots that are None (new sessions joining an ongoing
        batch) are filled with zeros matching the first non-None cache shape.

        Args:
            caches: Per-session ASRCacheState objects.

        Returns:
            Tuple of (att_cache, conv_cache, att_cache_len), each None or batched.
        """
        # 1. all-None → let NeMo init its own zero caches
        att_caches = [c.att_cache for c in caches]
        if all(a is None for a in att_caches):
            return None, None, None

        conv_caches = [c.conv_cache for c in caches]
        att_lens = [c.att_cache_len for c in caches]

        # 2. find reference shapes for zero-filling new sessions
        ref_att = next(a for a in att_caches if a is not None)
        ref_conv = next(c for c in conv_caches if c is not None)
        ref_len = next(l for l in att_lens if l is not None)

        # 3. lazy-init zero-fill tensors once; reuse on every subsequent call.
        # torch.cat only reads its inputs, so sharing the same zero tensor across
        # multiple None slots in the same call is safe.
        if self._zero_att is None:
            self._zero_att = torch.zeros_like(ref_att)
            self._zero_conv = torch.zeros_like(ref_conv)
            self._zero_len = torch.zeros_like(ref_len)

        # 4. fill None slots with zeros, then cat along batch dim (dim=1)
        att_list = [a if a is not None else self._zero_att for a in att_caches]
        conv_list = [c if c is not None else self._zero_conv for c in conv_caches]
        len_list = [l if l is not None else self._zero_len for l in att_lens]

        return (
            torch.cat(att_list, dim=1),
            torch.cat(conv_list, dim=1),
            torch.cat(len_list, dim=1),
        )


def _slice_inplace(
    existing: "torch.Tensor | None",
    batch: torch.Tensor,
    idx: int,
) -> torch.Tensor:
    """Extract session idx's slice from a batch output tensor.

    - Established session (existing tensor, shape matches): copy_ in-place.
      No new GPU allocation; batch tensor reference not retained.
    - First chunk (existing is None): clone() to allocate a fresh tensor.
      The clone breaks the view reference so batch can be freed immediately.

    dim=1 is the batch dimension for att_cache [layers, B, ...],
    conv_cache [layers, B, ...], and att_cache_len [layers, B].
    """
    src = batch.narrow(1, idx, 1)          # view — zero copy, no refcount bump on batch data
    if existing is not None and existing.shape == src.shape:
        return existing.copy_(src)         # in-place: reuse buffer, no allocation
    return src.clone()                     # first chunk: allocate once, break batch ref


def _extract_text(item) -> str:
    """Extract plain text from a NeMo RNNT Hypothesis or CTC string."""
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    return getattr(item, "text", "") or ""
