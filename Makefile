.PHONY: lint lint-fix test

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
	uv run pytest tests/unit/ tests/integration/

