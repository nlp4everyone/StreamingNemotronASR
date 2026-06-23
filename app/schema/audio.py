from typing import Literal
from pydantic import BaseModel

class AudioMessage(BaseModel):
    type: Literal["audio"] = "audio"
    data: str           # base64-encoded PCM int16
    sample_rate: int = 16000
    lang: str = "auto"

class ControlMessage(BaseModel):
    type: Literal["end", "start"]