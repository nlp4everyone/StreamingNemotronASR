from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.asr.engine import NemoStreamingEngine
    from app.asr.scheduler import BatchScheduler
    from app.session.manager import SessionManager


class ServiceRegistry:
    """Central registry holding all application-level singleton services.

    Properties raise RuntimeError on access before startup() runs, which surfaces
    wiring bugs as loud failures rather than silent None-attribute errors.
    """

    def __init__(self) -> None:
        self._engine: NemoStreamingEngine | None = None
        self._scheduler: BatchScheduler | None = None
        self._session_manager: SessionManager | None = None

    @property
    def engine(self) -> NemoStreamingEngine:
        """Return the ASR engine, raising if startup() has not run.

        Returns:
            The initialised NemoStreamingEngine instance.

        Raises:
            RuntimeError: If the engine has not been assigned yet.
        """
        if self._engine is None:
            raise RuntimeError("engine not initialized — call startup() first")
        return self._engine

    @engine.setter
    def engine(self, value: NemoStreamingEngine) -> None:
        self._engine = value

    @property
    def scheduler(self) -> BatchScheduler:
        """Return the batch scheduler, raising if startup() has not run.

        Returns:
            The initialised BatchScheduler instance.

        Raises:
            RuntimeError: If the scheduler has not been assigned yet.
        """
        if self._scheduler is None:
            raise RuntimeError("scheduler not initialized — call startup() first")
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value: BatchScheduler) -> None:
        self._scheduler = value

    @property
    def session_manager(self) -> SessionManager:
        """Return the session manager, raising if startup() has not run.

        Returns:
            The initialised SessionManager instance.

        Raises:
            RuntimeError: If the session manager has not been assigned yet.
        """
        if self._session_manager is None:
            raise RuntimeError("session_manager not initialized — call startup() first")
        return self._session_manager

    @session_manager.setter
    def session_manager(self, value: SessionManager) -> None:
        self._session_manager = value

    def is_ready(self) -> bool:
        """Check whether all services are initialised and the model is loaded.

        Returns:
            True only when engine, scheduler, and session_manager are all set
            and the engine has finished loading its model weights.
        """
        return (
            self._engine is not None
            and self._engine.is_ready()
            and self._scheduler is not None
            and self._session_manager is not None
        )


services = ServiceRegistry()
