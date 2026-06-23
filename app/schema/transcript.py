from typing import Literal
from pydantic import BaseModel

class TranscriptMessage(BaseModel):
    type: Literal["transcript"] = "transcript"
    session_id: str
    text: str
    is_final: bool
    lang_detected: str | None = None
    duration_ms: int | None = None      # only present when is_final=True