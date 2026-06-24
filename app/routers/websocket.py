"""WebSocket router — /ws/stream endpoint for streaming ASR."""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.schema.audio import AudioMessage
from app.services.registry import services
from app.session.state import StreamingSession
from app.websocket.handlers import StreamingHandler
from app.websocket.manager import conn_manager
from config import settings
logger = logging.getLogger(__name__)

router = APIRouter()
handler = StreamingHandler(conn=conn_manager)

@router.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket) -> None:
    """Accept and drive a streaming ASR WebSocket session.

    Args:
        ws: The incoming WebSocket connection.

    Message types:
        start  — reset cache and transcript for a new utterance.
        audio  — stream an audio chunk (see AudioMessage schema).
        end    — signal end of utterance; flushes remaining audio.
    """
    await ws.accept()

    # 1. reject if server is at session capacity
    sm = services.session_manager
    if sm.count() >= settings.max_sessions:
        await ws.close(code=1008, reason="Server at capacity")
        return

    # 2. register session and notify client
    session = StreamingSession(websocket=ws)
    sm.create(session)
    await conn_manager.send_session_info(ws, session)
    logger.info("WS connected — session=%s", session.session_id)

    # 3. message dispatch loop
    try:
        while True:
            raw = await ws.receive_json()
            msg_type = raw.get("type")

            if msg_type == "audio":
                await handler.handle_audio(session, AudioMessage.model_validate(raw))

            elif msg_type == "end":
                await handler.handle_end(session)

            elif msg_type == "start":
                # Reset state for a new utterance within the same connection
                session.cache.reset()
                session.audio_buffer.clear()
                session.transcript.reset()

            else:
                await conn_manager.send_error(
                    ws, "UNKNOWN_TYPE", f"Unknown message type: {msg_type!r}"
                )

    except WebSocketDisconnect:
        logger.info("WS disconnected — session=%s", session.session_id)
    except Exception:
        logger.exception("WS error — session=%s", session.session_id)
    finally:
        handler.cleanup(session.session_id)
