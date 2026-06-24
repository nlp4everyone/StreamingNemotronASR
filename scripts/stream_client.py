"""
Test client: streams a WAV file (or mic) to the server in 20ms chunks
and prints partial / final transcripts as they arrive.

Usage:
    python scripts/stream_client.py --file audio.wav
    python scripts/stream_client.py --file audio.wav --url ws://localhost:8000/ws/stream
    python scripts/stream_client.py --file audio.wav --lang vi-VN
"""
import argparse
import asyncio
import base64
import json
import wave
from pathlib import Path

import numpy as np
import websockets


PACKET_MS = 20
SAMPLE_RATE = 16000
SAMPLES_PER_PACKET = SAMPLE_RATE * PACKET_MS // 1000  # 320


async def stream_wav(url: str, wav_path: str, lang: str) -> None:
    with wave.open(wav_path) as wf:
        assert wf.getsampwidth() == 2, "WAV must be 16-bit PCM"
        assert wf.getnchannels() == 1, "WAV must be mono"
        src_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16)

    # Simple resample if needed
    if src_rate != SAMPLE_RATE:
        import torchaudio, torch
        t = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0)
        t = torchaudio.functional.resample(t, src_rate, SAMPLE_RATE)
        audio = (t.squeeze(0).numpy() * 32768).astype(np.int16)

    async with websockets.connect(url) as ws:
        # Receive session info
        info = json.loads(await ws.recv())
        print(f"[connected] session={info['session_id']}  preset={info['preset']}  "
              f"chunk={info['chunk_ms']}ms  packets/chunk={info['packets_per_chunk']}")

        # Stream audio in 20ms packets
        recv_task = asyncio.create_task(_recv_loop(ws))

        for i in range(0, len(audio), SAMPLES_PER_PACKET):
            chunk = audio[i: i + SAMPLES_PER_PACKET]
            if len(chunk) < SAMPLES_PER_PACKET:
                chunk = np.pad(chunk, (0, SAMPLES_PER_PACKET - len(chunk)))
            payload = base64.b64encode(chunk.tobytes()).decode()
            await ws.send(json.dumps({
                "type": "audio",
                "data": payload,
                "sample_rate": SAMPLE_RATE,
                "lang": lang,
            }))
            await asyncio.sleep(PACKET_MS / 1000)

        # Signal end of speech
        await ws.send(json.dumps({"type": "end"}))

        # Wait for final transcript
        await asyncio.wait_for(recv_task, timeout=10.0)


async def _recv_loop(ws) -> None:
    async for raw in ws:
        msg = json.loads(raw)
        if msg.get("type") != "transcript":
            continue
        label = "FINAL  " if msg["is_final"] else "partial"
        lang = msg.get("lang_detected", "")
        dur = f"  dur={msg['duration_ms']}ms" if msg.get("duration_ms") else ""
        print(f"[{label}] {msg['text']!r}  {lang}{dur}")
        if msg["is_final"]:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to WAV file (mono, 16-bit PCM)")
    parser.add_argument("--url", default="ws://localhost:8010/ws/stream")
    parser.add_argument("--lang", default="auto")
    args = parser.parse_args()

    asyncio.run(stream_wav(args.url, args.file, args.lang))


if __name__ == "__main__":
    main()
