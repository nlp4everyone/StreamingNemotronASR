import logging
from uuid import UUID
from app.session.state import StreamingSession

logger = logging.getLogger(__name__)

class SessionManager:
    """In-memory registry of all active WebSocket sessions.

    Keyed by session UUID. The manager is also responsible for releasing
    GPU tensor references when a session ends — without this, PyTorch's
    CUDA allocator holds onto VRAM until the next GC cycle.
    """

    def __init__(self) -> None:
        self._sessions: dict[UUID, StreamingSession] = {}

    def create(self, session: StreamingSession) -> None:
        """Register a newly accepted session."""
        self._sessions[session.session_id] = session
        logger.info("session created: %s (total: %d)", session.session_id, len(self._sessions))

    def get(self, session_id: UUID) -> StreamingSession | None:
        """Return the session for *session_id*, or None if it no longer exists."""
        return self._sessions.get(session_id)

    def remove(self, session_id: UUID) -> None:
        """Unregister a session and immediately free its GPU cache tensors."""
        # pop() returns None if the session was already removed (e.g. double-disconnect).
        session = self._sessions.pop(session_id, None)
        if session:
            # Release GPU memory before the session object is garbage collected.
            self._release_cache(session)
            logger.info("session removed: %s (total: %d)", session_id, len(self._sessions))

    def count(self) -> int:
        """Number of currently active sessions."""
        return len(self._sessions)

    @staticmethod
    def _release_cache(session: StreamingSession) -> None:
        # Null out every tensor field so the CUDA allocator can reclaim VRAM
        # immediately rather than waiting for Python's garbage collector.
        c = session.cache
        c.att_cache = None      # encoder attention cache
        c.conv_cache = None     # encoder convolution cache
        c.att_cache_len = None  # valid lengths per layer
        c.hypotheses = None     # RNNT decoder hypotheses
        c.pred_out = None       # RNNT decoder previous output


