from config import settings

class AudioChunkBuffer:
    """
    Accumulates raw PCM int16 packets until a full inference chunk is ready.

    The WebSocket client sends 20 ms packets (320 samples @ 16 kHz).
    NeMo's streaming conformer expects a fixed number of those packets per
    call (``packets_per_chunk`` from the active preset), so this buffer holds
    incoming bytes until enough have arrived, then hands them off in one go.
    All sizing derives from the preset — switching preset is the only change needed.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        p = settings.preset
        # 20 ms packet = 320 samples @ 16 kHz; each sample is 2 bytes (int16).
        self._target_bytes: int = p.packets_per_chunk * 320 * 2

    def push(self,
             pcm_bytes: bytes) -> bytes | None:
        """Append one 20 ms packet. Returns a full inference chunk when ready, else None."""
        # Accumulate incoming packet into internal buffer.
        self._buf.extend(pcm_bytes)

        if len(self._buf) >= self._target_bytes:
            # Slice exactly one chunk from the front — leave the remainder for the next call.
            chunk = bytes(self._buf[: self._target_bytes])
            del self._buf[: self._target_bytes]
            return chunk

        # Not enough data yet — signal the caller to keep sending.
        return None

    def flush(self) -> bytes | None:
        """Return remaining audio, zero-padded to a full chunk size.

        Zero-padding appends silence, which is acoustically neutral — NeMo will
        not emit spurious tokens for a short tail of zeros at end-of-speech.
        Returns None if the buffer is already empty.
        """
        if not self._buf:
            return None

        # Pad with silence bytes so the tensor shape matches what NeMo expects.
        padded = bytes(self._buf).ljust(self._target_bytes, b"\x00")
        self._buf.clear()
        return padded

    def clear(self) -> None:
        """Discard all buffered audio without returning it."""
        self._buf.clear()

    @property
    def buffered_ms(self) -> int:
        """Milliseconds of audio currently held in the buffer (assumes 16 kHz mono int16)."""
        samples = len(self._buf) // 2  # 2 bytes per int16 sample
        return samples * 1000 // 16000
