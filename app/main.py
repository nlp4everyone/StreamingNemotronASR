"""FastAPI application entry point — wires lifespan, routes, and logging."""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import websocket as ws_router
from app.routers import health as health_router
from app.startup import shutdown, startup
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("nv_one_logger").setLevel(logging.ERROR)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown.

    Args:
        app: The FastAPI application instance (injected by FastAPI).
    """
    t0 = time.perf_counter()
    await startup()
    logger.info(
        "server ready in %.1fs — preset=%s batch_mode=%s host=%s port=%d",
        time.perf_counter() - t0,
        settings.streaming_preset,
        settings.batch_mode,
        settings.host,
        settings.port,
    )

    yield

    await shutdown()


app = FastAPI(
    title="StreamingNenotronASR",
    description="Cache-aware streaming ASR via WebSocket — nvidia/nemotron-3.5-asr-streaming-0.6b",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ws_router.router)
app.include_router(health_router.router)
