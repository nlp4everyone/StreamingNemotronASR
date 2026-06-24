import logging
import time
import threading
import numpy as np
import torch
import torchaudio
from config import settings

logger = logging.getLogger(__name__)

class NemoStreamingEngine:
    """
    Wraps nvidia/nemotron-3.5-asr-streaming-0.6b via NeMo.
    Singleton — loaded once at startup, weights are read-only.
    All mutable state lives in ASRCacheState (per-session).

    Called from ThreadPoolExecutor threads — blocking CUDA calls are expected here.
    """

    def __init__(self) -> None:
        """Model is lazy-loaded; call load() before any inference."""
        self._model = None
        self._device = settings.device
        self._prompt_lock = threading.Lock()  # set_inference_prompt writes shared model state

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Download (or cache) model weights and load directly onto the configured device."""
        import nemo.collections.asr as nemo_asr
        from nemo.utils import logging as nemo_logging

        # Silence NeMo's verbose load-time messages (tokenizer init, training-config warnings).
        # Must be called after the NeMo import (which initialises the Singleton at INFO level)
        # but before from_pretrained(), which emits the warning during __init__.
        nemo_logging.set_verbosity(nemo_logging.ERROR)

        t0 = time.perf_counter()
        logger.info("loading %s on %s ...", settings.model_name, self._device)

        # 1. load weights
        t1 = time.perf_counter()
        self._model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=settings.model_name,
            map_location=self._device,
        )
        logger.info("weights loaded in %.1fs", time.perf_counter() - t1)

        # 2. eval mode — disables dropout / batch-norm updates
        self._model.eval()

        # 3. apply attention context size from preset (controls look-ahead frames)
        self._model.encoder.set_default_att_context_size(
            list(settings.preset.att_context_size)
        )

        # 4. apply decoding strategy / max symbols per step
        if settings.decoding_strategy != "greedy_batch":
            self._model.change_decoding_strategy(
                decoding_cfg={"strategy": settings.decoding_strategy}
            )
        if settings.max_symbols_per_step != 10:
            self._model.decoding.cfg.greedy.max_symbols = settings.max_symbols_per_step

        # 5. set default language prompt — required; model returns empty strings without it
        if hasattr(self._model, "set_inference_prompt"):
            self._model.set_inference_prompt(settings.default_lang)

        logger.info(
            "model ready in %.1fs — preset=%s att_context=%s lang=%s",
            time.perf_counter() - t0,
            settings.preset.name,
            settings.preset.att_context_size,
            settings.default_lang,
        )

    def is_ready(self) -> bool:
        """Check whether the model has been loaded.

        Returns:
            True once load() has completed successfully.
        """
        return self._model is not None

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocess(self,
                    pcm_bytes: bytes,
                    sample_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw PCM audio to log-mel spectrogram features.

        Args:
            pcm_bytes: Raw audio samples encoded as int16.
            sample_rate: Sample rate of the input audio in Hz.

        Returns:
            Tuple of (mel, mel_len) where mel has shape [1, D, T]
            and mel_len has shape [1].
        """
        # 1. decode int16 → float32 in [-1, 1]
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # 2. resample to 16 kHz if needed
        if sample_rate != 16000:
            waveform = torch.from_numpy(audio).unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            audio = waveform.squeeze(0).numpy()

        # 3. move to device and build length tensor
        audio_t = torch.from_numpy(audio).unsqueeze(0).to(self._device)
        audio_len = torch.tensor([len(audio)], dtype=torch.long, device=self._device)

        # 4. extract log-mel features via model's own preprocessor
        with torch.inference_mode():
            mel, mel_len = self._model.preprocessor(
                input_signal=audio_t,
                length=audio_len,
            )
        return mel, mel_len  # [1, D, T], [1]

    # ── Single-session inference ───────────────────────────────────────────────

    def stream_step(self,
                    pcm_bytes: bytes,
                    sample_rate: int,
                    cache: "ASRCacheState",
                    lang: str = "auto") -> tuple[str, "ASRCacheState"]:  # noqa: F821
        """Run one streaming inference step for a single session.

        Args:
            pcm_bytes: Raw audio chunk encoded as int16 PCM.
            sample_rate: Sample rate of the input audio in Hz.
            cache: Per-session attention and convolution cache state.
            lang: BCP-47 language tag or "auto" for automatic detection.

        Returns:
            Tuple of (transcript_text, updated_cache).
        """
        from app.session.state import ASRCacheState

        # 1. convert raw audio to log-mel features
        mel, mel_len = self._preprocess(pcm_bytes, sample_rate)

        # 2. set language prompt (locked — shared model state)
        if hasattr(self._model, "set_inference_prompt"):
            with self._prompt_lock:
                self._model.set_inference_prompt(lang)

        # 3. run conformer streaming inference
        with torch.inference_mode():
            # NeMo return order (mixins.py):
            #   [0] greedy_predictions          — list[Tensor], token-id sequences
            #   [1] all_hyp_or_transcribed_texts — list[str] (CTC) or list[Hypothesis] (RNNT)
            #   [2] cache_last_channel_next     — attention cache Tensor
            #   [3] cache_last_time_next        — convolution cache Tensor
            #   [4] cache_last_channel_next_len — length Tensor  shape (batch,)
            #   [5] best_hyp                    — list[Hypothesis] (RNNT) or None (CTC)
            (
                greedy_predictions,
                all_hyp_or_transcribed_texts,
                new_att_cache,
                new_conv_cache,
                new_att_cache_len,
                best_hyp,
            ) = self._model.conformer_stream_step(
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

        # 4. extract text — RNNT yields Hypothesis objects, CTC yields plain strings
        first = all_hyp_or_transcribed_texts[0] if all_hyp_or_transcribed_texts else None
        if first is None:
            text = ""
        elif isinstance(first, str):
            text = first
        else:
            text = getattr(first, "text", "") or ""

        logger.info("DEBUG inference: type=%s text=%r hyp_text=%r y_seq_len=%s",
                    type(first).__name__, text,
                    getattr(first, "text", "N/A"),
                    len(getattr(first, "y_sequence", [])) if first is not None else 0)

        # 5. pack updated cache for next step
        new_cache = ASRCacheState(
            att_cache=new_att_cache,
            conv_cache=new_conv_cache,
            att_cache_len=new_att_cache_len,
            hypotheses=best_hyp,          # passed back as previous_hypotheses (RNNT)
            pred_out=greedy_predictions,  # passed back as previous_pred_out (CTC)
        )
        return text, new_cache

    # ── Batch inference ────────────────────────────────────────────────────────

    def stream_step_batch(self,
                          requests: list[tuple[bytes, int, "ASRCacheState", str]]) -> list[tuple[str, "ASRCacheState"]]:  # noqa: F821
        """Run inference for multiple sessions in one call.

        Args:
            requests: List of (pcm_bytes, sample_rate, cache, lang) tuples,
                one per session.

        Returns:
            List of (transcript_text, updated_cache) in the same order as requests.

        Note:
            Currently runs sessions sequentially. TODO: stack [B, D, T] tensors
            for true GPU batching with conformer_stream_step(B > 1).
        """
        return [self.stream_step(pcm, sr, cache, lang) for pcm, sr, cache, lang in requests]


