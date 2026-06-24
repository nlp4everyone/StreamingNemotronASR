"""Public re-exports for the startup and shutdown lifecycle hooks."""
from app.startup.initializer import shutdown, startup

__all__ = ["startup", "shutdown"]
