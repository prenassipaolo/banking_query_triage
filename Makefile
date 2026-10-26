.PHONY: lint lint-fix test api predict

HOST ?= 127.0.0.1
PORT ?= 8000
CONFIG ?= config/serving.toml
PAYLOAD ?= examples/predict.json

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright src/
	uv run pydoclint src/

lint-fix:
	uv run ruff check --fix .
	uv run ruff format .
	uv run pyright src/
	uv run pydoclint src/

test:
	uv run pytest $(ARGS)

api:
	uv run banking-query-triage --config "$(CONFIG)" --host "$(HOST)" --port "$(PORT)"

predict:
	uv run python examples/call_api.py --url "http://$(HOST):$(PORT)/predict" --payload "$(PAYLOAD)"

