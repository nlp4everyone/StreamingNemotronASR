"""Service initialization and teardown called by the FastAPI lifespan."""
import logging

from app.asr.engine import NemoStreamingEngine
from app.asr.scheduler import BatchScheduler
from app.session.manager import SessionManager
from app.services.registry import services

logger = logging.getLogger(__name__)


async def startup() -> None:
    """Initialize all services in dependency order (session manager → engine → scheduler).

    Raises:
        Exception: Propagates any error from model loading or scheduler startup.
    """
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

    logger.info("all services initialized")


async def shutdown() -> None:
    """Stop the batch scheduler and release thread pool resources."""
    logger.info("shutting down services...")
    if services._scheduler:
        await services._scheduler.stop()
    logger.info("shutdown complete")
