from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4
import torch, time, asyncio
from fastapi import WebSocket
from app.audio.buffer import AudioChunkBuffer

@dataclass
class ASRCacheState:
    """Per-session NeMo streaming cache tensors.

    NeMo's conformer_stream_step is stateful: it returns updated attention and
    convolution caches that must be fed back into the next call for the same
    session. All fields default to None — the model treats None as an empty
    (zero-initialised) cache on the first step.
    """

    att_cache: torch.Tensor | None = None       # encoder attention cache  [layers, B, heads, cache_len, dim]
    conv_cache: torch.Tensor | None = None      # encoder convolution cache [layers, B, dim, kernel-1]
    att_cache_len: torch.Tensor | None = None   # valid cache lengths per layer [layers, B]
    hypotheses: list[Any] | None = None         # RNNT decoder hypothesis carry-over (previous_hypotheses)
    pred_out: torch.Tensor | None = None        # RNNT decoder previous token output (previous_pred_out)

    def reset(self) -> None:
        """Clear decoder state between utterances while keeping the encoder cache.

        Resetting hypotheses and pred_out starts a fresh decoding sequence for
        the next utterance. The encoder caches (att_cache, conv_cache) are kept
        so the model retains acoustic context across utterance boundaries.
        """
        # Encoder cache intentionally preserved — gives the model acoustic
        # continuity so it doesn't mis-hear the start of the next utterance.
        self.hypotheses = None  # start decoding fresh
        self.pred_out = None    # no previous token to condition on


@dataclass
class TranscriptState:
    """Tracks the evolving transcript text within a single utterance."""

    last_partial: str = ""
    # Monotonic timestamp of when the current utterance started — used to
    # compute duration_ms reported in the final transcript event.
    utterance_start: float = field(default_factory=time.monotonic)

    def reset(self) -> None:
        """Discard the partial text and restart the utterance timer."""
        self.last_partial = ""
        self.utterance_start = time.monotonic()  # anchor for the next duration_ms calculation


@dataclass(kw_only=True)
class StreamingSession:
    """All mutable state for one connected WebSocket client.

    Kept in a single dataclass so SessionManager and StreamingHandler never
    need to coordinate through multiple separate stores — one object per client,
    passed around by reference.
    """

    websocket: WebSocket = field(repr=False)
    session_id: UUID = field(default_factory=uuid4)
    audio_buffer: AudioChunkBuffer = field(default_factory=AudioChunkBuffer)
    cache: ASRCacheState = field(default_factory=ASRCacheState)
    transcript: TranscriptState = field(default_factory=TranscriptState)
    lang: str = "auto"
    last_sample_rate: int = 16000
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    # Number of inference requests currently queued or in-flight for this session.
    # Incremented before submit(), decremented in finally — asyncio single-thread
    # guarantees no race between the check and the increment.
    pending_infer: int = 0
    # Pending EOS-timeout task; cancelled and recreated on every audio packet.
    eos_task: asyncio.Task | None = field(default=None, repr=False)

    def touch(self) -> None:
        """Update last_activity to now — called on every incoming audio packet."""
        self.last_activity = time.monotonic()

    @property
    def idle_seconds(self) -> float:
        """Seconds since the last audio packet was received."""
        return time.monotonic() - self.last_activity
