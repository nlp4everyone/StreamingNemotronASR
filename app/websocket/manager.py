import logging

from fastapi import WebSocket

from app.schema.session import SessionInfoMessage
from app.schema.transcript import TranscriptMessage
from config import settings

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Stateless send helpers. All outbound WebSocket messages go through here.

    Swallowing send errors is intentional — a disconnected client is not an
    application error and should not pollute logs or crash the handler.
    """

    async def send_session_info(self,
                                ws: WebSocket,
                                session: "StreamingSession") -> None:
        """Send preset and chunk parameters to the client immediately after connection.

        Args:
            ws: The active WebSocket connection.
            session: The newly created session whose ID is included in the message.
        """
        p = settings.preset
        msg = SessionInfoMessage(
            session_id=str(session.session_id),
            preset=p.name,
            chunk_ms=p.chunk_ms,
            att_context_size=list(p.att_context_size),
            packets_per_chunk=p.packets_per_chunk,
            batch_mode=settings.batch_mode,
        )
        await self._send(ws, msg.model_dump())

    async def send_transcript(self,
                              ws: WebSocket,
                              session_id: str,
                              text: str,
                              is_final: bool,
                              lang_detected: str | None = None,
                              duration_ms: int | None = None) -> None:
        """Send a partial or final transcript message to the client.

        Args:
            ws: The active WebSocket connection.
            session_id: UUID string of the session this transcript belongs to.
            text: Recognised text for this chunk or utterance.
            is_final: True when the utterance has ended; False for streaming partials.
            lang_detected: BCP-47 language tag if detected, otherwise None.
            duration_ms: Utterance duration in milliseconds; only sent when is_final is True.
        """
        msg = TranscriptMessage(
            session_id=session_id,
            text=text,
            is_final=is_final,
            lang_detected=lang_detected,
            duration_ms=duration_ms if is_final else None,
        )
        await self._send(ws, msg.model_dump(exclude_none=True))

    async def send_error(self,
                         ws: WebSocket,
                         code: str,
                         message: str) -> None:
        """Send a structured error message to the client.

        Args:
            ws: The active WebSocket connection.
            code: Short machine-readable error code (e.g. "DECODE_ERROR").
            message: Human-readable description of the error.
        """
        await self._send(ws, {"type": "error", "code": code, "message": message})

    @staticmethod
    async def _send(ws: WebSocket,
                    data: dict) -> None:
        """Serialise and send a JSON payload, silently ignoring send failures.

        Args:
            ws: The active WebSocket connection.
            data: Payload to serialise and transmit.
        """
        try:
            await ws.send_json(data)
        except Exception:
            logger.debug("send failed (client likely disconnected)")


conn_manager = ConnectionManager()
