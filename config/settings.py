import os
from typing import Any, Tuple, Type

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from .presets import PRESETS, StreamingPreset


class _YamlSource(PydanticBaseSettingsSource):
    """Loads config/settings.yaml then config/settings.local.yaml (latter overrides)."""

    def get_field_value(self, field: Any, field_name: str) -> Any:  # type: ignore[override]
        # Not used — __call__ returns the full merged dict directly.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Merge base and local YAML files into a single settings dict.

        Returns:
            Merged dict of all YAML settings; local overrides base on key collision.
        """
        merged: dict[str, Any] = {}
        for path in ("config/settings.yaml", "config/settings.local.yaml"):
            if os.path.exists(path):
                with open(path) as fh:
                    data = yaml.safe_load(fh) or {}
                merged.update(data)
        return merged

    def field_is_complex(self, field: Any) -> bool:  # type: ignore[override]
        return False


class Settings(BaseSettings):
    """Application configuration.

    Resolution priority (highest → lowest):
      1. Environment variables (prefix ASR_)
      2. .env file
      3. config/settings.yaml + config/settings.local.yaml (local overrides yaml)
    """

    model_config = SettingsConfigDict(
        env_prefix="ASR_",
        env_file=".env",
        extra="ignore",
    )

    # ── Streaming ──────────────────────────────────────────────
    streaming_preset: str = "balanced"
    batch_mode: str = "per_session"   # "per_session" | "dynamic"
    max_batch_size: int = 8
    batch_timeout_ms: int = 20

    # ── Server ─────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    max_sessions: int = 50

    # ── Model ──────────────────────────────────────────────────
    model_name: str = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    device: str = "cuda"
    default_lang: str = "auto"
    # "greedy_batch" (default, faster) or "beam" (more accurate, slower)
    decoding_strategy: str = "greedy_batch"
    # Max tokens the RNNT decoder emits per encoder frame. Raise if words get cut off.
    max_symbols_per_step: int = 10

    # ── Behaviour ──────────────────────────────────────────────
    end_of_speech_timeout_s: float = 3.0
    thread_pool_workers: int = 4

    @field_validator("streaming_preset")
    @classmethod
    def _check_preset(cls, v: str) -> str:
        """Validate that the preset name exists in the PRESETS registry.

        Args:
            v: Raw preset name from config.

        Returns:
            The validated preset name unchanged.

        Raises:
            ValueError: If the name is not a key in PRESETS.
        """
        if v not in PRESETS:
            raise ValueError(f"Unknown preset '{v}'. Valid: {list(PRESETS)}")
        return v

    @field_validator("batch_mode")
    @classmethod
    def _check_batch_mode(cls, v: str) -> str:
        """Validate that batch_mode is one of the supported scheduling strategies.

        Args:
            v: Raw batch_mode value from config.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If the value is not 'per_session' or 'dynamic'.
        """
        if v not in ("per_session", "dynamic"):
            raise ValueError("batch_mode must be 'per_session' or 'dynamic'")
        return v

    @property
    def preset(self) -> StreamingPreset:
        """Resolve streaming_preset name to its StreamingPreset dataclass.

        Returns:
            The StreamingPreset corresponding to the configured streaming_preset name.
        """
        return PRESETS[self.streaming_preset]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
        **kwargs: Any,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Wire in the YAML source at lowest priority.

        Args:
            settings_cls: The Settings class being constructed.
            init_settings: Values passed directly to __init__.
            env_settings: Values from environment variables.
            dotenv_settings: Values from the .env file.
            file_secret_settings: Values from secret files (unused).
            **kwargs: Forward-compatibility placeholder.

        Returns:
            Source tuple ordered highest → lowest priority.
        """
        # Priority: env vars > .env file > YAML
        return init_settings, env_settings, dotenv_settings, _YamlSource(settings_cls)
