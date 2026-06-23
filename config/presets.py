from dataclasses import dataclass


@dataclass(frozen=True)
class StreamingPreset:
    """Maps a named latency profile to NeMo attention-context parameters.

    att_context_size is (left_frames, right_frames) in subsampled space.
    right_frames controls look-ahead: 0 = truly online, higher = lower WER but more latency.
    Each subsampled frame covers 80 ms of audio (8× subsampling at 10 ms stride).

    Attributes:
        name: Human-readable preset identifier (e.g. "balanced").
        att_context_size: (left_frames, right_frames) passed to the NeMo encoder.
    """

    name: str
    att_context_size: tuple[int, int]

    @property
    def chunk_frames(self) -> int:
        """Number of subsampled frames per inference call.

        Returns:
            right_frames + 1; the minimum chunk the encoder needs to produce output.
        """
        return self.att_context_size[1] + 1

    @property
    def chunk_ms(self) -> int:
        """Audio duration covered by one inference call in milliseconds.

        Returns:
            chunk_frames × 80 ms (one subsampled frame = 80 ms at 16 kHz / stride 10 ms / factor 8).
        """
        return self.chunk_frames * 80

    @property
    def packets_per_chunk(self) -> int:
        """Number of 20 ms client packets that fill one inference chunk.

        Returns:
            chunk_ms // 20; the client must accumulate this many packets before sending.
        """
        return self.chunk_ms // 20


PRESETS: dict[str, StreamingPreset] = {
    "ultra_low": StreamingPreset("ultra_low", (56, 0)),   #  80ms |  4 packets
    "low":       StreamingPreset("low",       (56, 1)),   # 160ms |  8 packets
    "medium":    StreamingPreset("medium",    (56, 3)),   # 320ms | 16 packets
    "balanced":  StreamingPreset("balanced",  (56, 6)),   # 560ms | 28 packets
    "high":      StreamingPreset("high",      (56, 13)),  # 1120ms| 56 packets
}
