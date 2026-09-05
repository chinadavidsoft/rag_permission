.PHONY: up down logs test fixtures demo

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api qdrant

test:
	uv run pytest -q

fixtures:
	uv run python scripts/create_fixtures.py

demo:
	uv run python scripts/demo_retrieval.py
