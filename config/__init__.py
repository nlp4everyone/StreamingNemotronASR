from functools import lru_cache

from .settings import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    lru_cache ensures a single instance across imports without relying on a
    module-level global that is awkward to patch in tests.

    Returns:
        The application Settings instance, constructed once on first call.
    """
    return Settings()


settings = get_settings()
