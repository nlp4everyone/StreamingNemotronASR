# Project Structure — StreamingNenotronASR

```
StreamingNenotronASR/
├── app/
│   ├── main.py                    # FastAPI app, lifespan (model load/unload)
│   ├── routers/
│   │   └── websocket.py           # /ws/stream, /health
│   ├── websocket/
│   │   ├── handlers.py            # StreamingHandler
│   │   └── manager.py             # ConnectionManager
│   ├── session/
│   │   ├── state.py               # StreamingSession, ASRCacheState
│   │   └── manager.py             # SessionManager singleton
│   ├── audio/
│   │   ├── buffer.py              # AudioChunkBuffer
│   │   └── features.py            # log-mel extraction
│   ├── asr/
│   │   └── engine.py              # NemoStreamingEngine
│   └── schema/
│       ├── audio.py
│       ├── transcript.py
│       └── session.py
│
├── config/
│   ├── __init__.py                # settings singleton
│   ├── presets.py                 # StreamingPreset, PRESETS dict
│   ├── settings.py                # Pydantic Settings (env + yaml)
│   ├── settings.yaml              # defaults (committed)
│   └── settings.local.yaml        # local overrides (gitignored)
│
├── scripts/
│   └── stream_client.py           # test client: stream WAV → ws → print transcript
│
├── docker/
│   ├── Dockerfile                 # multi-stage CUDA build (devel → runtime)
│   └── docker-compose.yml         # GPU service, named volumes, port via APP_PORT
│
├── docs/
│   ├── ARCHITECTURE.md            # system architecture overview
│   ├── DETAILED_COMPONENTS.md     # per-component reference
│   └── PROJECT_STRUCTURE.md       # this file
│
├── .env.example
├── .gitignore
├── Makefile
└── requirements.txt
```