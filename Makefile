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

# Coverage unit and functional test with floor
test:
	uv run pytest tests/unit/ tests/functional/ --cov=src --cov-report=term-missing --cov-fail-under=70

