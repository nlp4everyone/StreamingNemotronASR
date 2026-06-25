"""Service initialization and teardown called by the FastAPI lifespan."""
import asyncio
import logging

from app.asr.engine import NemoStreamingEngine
from app.asr.scheduler import BatchScheduler
from app.session.manager import SessionManager
from app.services.registry import services
from config import settings

logger = logging.getLogger(__name__)

_sweep_task: asyncio.Task | None = None


async def startup() -> None:
    """Initialize all services in dependency order (session manager → engine → scheduler).

    Raises:
        Exception: Propagates any error from model loading or scheduler startup.
    """
    global _sweep_task
    logger.info("initializing services...")

    # 1. Session manager — lightweight, no I/O; must come first so the engine
    #    and scheduler can reference it during their own init if needed.
    services.session_manager = SessionManager()
    logger.info("session_manager ready")

    # 2. ASR engine — downloads model weights on first run (~1–2 GB), then loads
    #    them onto the configured device. This is the dominant startup cost.
    services.engine = NemoStreamingEngine()
    services.engine.load()
    logger.info("asr engine ready")

    # 3. Batch scheduler — wraps the engine with a ThreadPoolExecutor and,
    #    in dynamic mode, starts the background batching worker task.
    services.scheduler = BatchScheduler(services.engine)
    await services.scheduler.start()
    logger.info("scheduler ready")

    # 4. Idle session sweeper — evicts ghost sessions that stopped sending audio
    #    but kept the TCP connection alive (ping/pong handles hard network drops;
    #    the sweeper handles the soft "client went silent" case).
    _sweep_task = asyncio.create_task(_sweep_idle_sessions(), name="session-sweeper")
    logger.info(
        "session sweeper started — idle_timeout=%.0fs interval=%.0fs",
        settings.idle_timeout_s,
        settings.session_sweep_interval_s,
    )

    logger.info("all services initialized")


async def shutdown() -> None:
    """Stop the idle sweeper, batch scheduler, and release thread pool resources."""
    global _sweep_task
    logger.info("shutting down services...")

    if _sweep_task and not _sweep_task.done():
        _sweep_task.cancel()
        try:
            await _sweep_task
        except asyncio.CancelledError:
            pass

    if services._scheduler:
        await services._scheduler.stop()

    logger.info("shutdown complete")


async def _sweep_idle_sessions() -> None:
    """Periodically evict sessions that have been silent longer than idle_timeout_s."""
    while True:
        await asyncio.sleep(settings.session_sweep_interval_s)
        sm = services.session_manager
        for sid in list(sm._sessions.keys()):
            session = sm.get(sid)
            if session and session.idle_seconds > settings.idle_timeout_s:
                logger.info(
                    "evicting idle session=%s idle=%.1fs", sid, session.idle_seconds
                )
                sm.evict(sid)
