from typing import Literal
from pydantic import BaseModel

class SessionInfoMessage(BaseModel):
    type: Literal["session_info"] = "session_info"
    session_id: str
    preset: str
    chunk_ms: int
    att_context_size: list[int]
    packets_per_chunk: int
    batch_mode: str