.PHONY: build run stop logs shell pull-model test-client

## Build image — multi-stage CUDA build (~20min first time, cached after)
build:
	docker compose -f docker/docker-compose.yml build

## Start the server (detached)
run:
	docker compose -f docker/docker-compose.yml up -d

## Stop the server
stop:
	docker compose -f docker/docker-compose.yml down

## Stream server logs
logs:
	docker compose -f docker/docker-compose.yml logs -f asr-server

## Open a shell inside the running container
shell:
	docker compose -f docker/docker-compose.yml exec asr-server bash

## Pre-download model weights into the named volume (run once before first start)
## Uses huggingface_hub directly — no GPU needed, avoids the torchaudio CUDA dlopen issue.
## NeMo's from_pretrained() at server startup will find these cached files automatically.
pull-model:
	docker compose -f docker/docker-compose.yml run --rm asr-server python3 -c "\
from huggingface_hub import snapshot_download; \
path = snapshot_download('nvidia/nemotron-3.5-asr-streaming-0.6b'); \
print('model cached at:', path)"

## Quick smoke test — stream a WAV file from host
## Usage: make test-client WAV=audio.wav
test-client:
	python3 scripts/stream_client.py --file $(WAV) --url ws://localhost:$${APP_PORT:-8010}/ws/stream
