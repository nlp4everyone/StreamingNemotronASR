import asyncio
import base64
import logging
import time
from uuid import UUID
from config import settings
from app.schema.audio import AudioMessage
from app.services.registry import services
from app.session.state import StreamingSession
from app.websocket.manager import ConnectionManager

logger = logging.getLogger(__name__)


class StreamingHandler:
    """Per-message orchestration. Owns the EOS timeout lifecycle.

    One instance is shared across all sessions (stateless — all state is in StreamingSession).
    """

    def __init__(self, conn: ConnectionManager) -> None:
        self._conn = conn

    # ── Public entry points ────────────────────────────────────────────────────

    async def handle_audio(self,
                           session: StreamingSession,
                           msg: AudioMessage) -> None:
        """Buffer one audio packet and run inference when a full chunk is ready.

        EOS timeout is reset on every packet so silence detection only fires
        after a real gap in audio, not mid-speech.

        Args:
            session: The active streaming session that owns the audio buffer and cache.
            msg: Incoming audio message containing base64-encoded int16 PCM and metadata.
        """
        session.touch()
        session.lang = msg.lang
        session.last_sample_rate = msg.sample_rate
        self._reschedule_eos(session)

        try:
            pcm_bytes = base64.b64decode(msg.data)
        except Exception:
            await self._conn.send_error(
                session.websocket, "DECODE_ERROR", "Invalid base64 audio data"
            )
            return

        chunk = session.audio_buffer.push(pcm_bytes)
        if chunk is None:
            return  # buffer not full yet

        await self._infer_and_emit(session, chunk, is_final=False)

    async def handle_end(self,
                         session: StreamingSession) -> None:
        """Finalise the current utterance: flush remaining audio and emit a final transcript.

        Args:
            session: The active streaming session to finalise.
        """
        self._cancel_eos(session)

        chunk = session.audio_buffer.flush()
        if chunk:
            await self._infer_and_emit(session, chunk, is_final=True)
        else:
            # No buffered audio — promote last partial as final
            duration_ms = int((time.monotonic() - session.transcript.utterance_start) * 1000)
            await self._conn.send_transcript(
                ws=session.websocket,
                session_id=str(session.session_id),
                text=session.transcript.last_partial,
                is_final=True,
                duration_ms=duration_ms,
            )

        # Reset decoder state; keep encoder cache for acoustic context continuity
        session.cache.reset()
        session.audio_buffer.clear()
        session.transcript.reset()

    def cleanup(self,
                session_id: UUID) -> None:
        """Cancel the pending EOS task and remove the session from the manager.

        Args:
            session_id: UUID of the session to tear down.
        """
        sm = services.session_manager
        session = sm.get(session_id)
        if session:
            self._cancel_eos(session)
        sm.remove(session_id)

    # ── Inference ─────────────────────────────────────────────────────────────

    async def _infer_and_emit(self,
                              session: StreamingSession,
                              chunk: bytes,
                              *,
                              is_final: bool) -> None:
        """Submit a chunk to the scheduler and emit the result if the transcript changed.

        Partials are suppressed when the text is identical to the previous partial
        to avoid redundant WebSocket frames. Non-final chunks are dropped when the
        session already has max_pending_per_session requests in-flight, keeping the
        queue bounded and the session current without blocking other sessions.

        Args:
            session: The active streaming session owning the ASR cache.
            chunk: Raw int16 PCM audio to transcribe.
            is_final: If True, emit a final transcript and include utterance duration.
        """
        if not is_final and session.pending_infer >= settings.max_pending_per_session:
            logger.debug("dropping chunk — session=%s pending=%d", session.session_id, session.pending_infer)
            from app.services.metrics import stats
            stats.record_drop()
            return

        session.pending_infer += 1
        try:
            text, new_cache = await services.scheduler.submit(
                chunk, session.last_sample_rate, session.cache, session.lang
            )
        except Exception:
            logger.exception("inference error session=%s", session.session_id)
            return
        finally:
            session.pending_infer -= 1

        session.cache = new_cache

        # Normalize: inference may return a Tensor instead of str when output is empty
        if not isinstance(text, str):
            text = ""

        if is_final:
            duration_ms = int((time.monotonic() - session.transcript.utterance_start) * 1000)
            await self._conn.send_transcript(
                ws=session.websocket,
                session_id=str(session.session_id),
                text=text or session.transcript.last_partial,
                is_final=True,
                duration_ms=duration_ms,
            )
        elif text and text != session.transcript.last_partial:
            session.transcript.last_partial = text
            await self._conn.send_transcript(
                ws=session.websocket,
                session_id=str(session.session_id),
                text=text,
                is_final=False,
            )

    # ── EOS timeout ───────────────────────────────────────────────────────────

    def _reschedule_eos(self,
                        session: StreamingSession) -> None:
        """Cancel the existing EOS timer and start a fresh one.

        Always cancels before creating so each audio packet resets the full timeout window.

        Args:
            session: The active streaming session whose EOS task will be replaced.
        """
        self._cancel_eos(session)
        session.eos_task = asyncio.create_task(
            self._eos_timeout(session), name=f"eos-{session.session_id}"
        )

    def _cancel_eos(self,
                    session: StreamingSession) -> None:
        """Cancel the pending EOS timeout task if one is running.

        Args:
            session: The active streaming session whose EOS task will be cancelled.
        """
        if session.eos_task and not session.eos_task.done():
            session.eos_task.cancel()
            session.eos_task = None

    async def _eos_timeout(self,
                           session: StreamingSession) -> None:
        """Wait for the configured silence timeout then trigger end-of-utterance.

        Args:
            session: The active streaming session to finalise after the timeout.
        """
        await asyncio.sleep(settings.end_of_speech_timeout_s)
        logger.debug("EOS timeout — session=%s", session.session_id)
        await self.handle_end(session)
