from fastapi import APIRouter

from app.services.metrics import stats
from app.services.registry import services
from config import settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Return current service health status."""
    return {
        "status": "ok",
        "model_ready": services.engine.is_ready(),
        "active_sessions": services.session_manager.count(),
        "preset": settings.streaming_preset,
        "batch_mode": settings.batch_mode,
    }


@router.get("/health/stats")
async def health_stats() -> dict:
    """Runtime metrics for queue health, drop rate, and batch latency."""
    return {
        "active_sessions": services.session_manager.count(),
        "max_sessions": settings.max_sessions,
        "queue_depth": services.scheduler.queue_depth(),
        "drop_count": stats.drop_count,
        "inference_count": stats.inference_count,
        **stats.snapshot(),
        "bf16_enabled": services.engine.bf16_enabled,
    }
