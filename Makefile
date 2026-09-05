.PHONY: up down logs test fixtures demo evaluate profile

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

evaluate:
	uv run python scripts/evaluate_retrieval.py --output reports/retrieval_evaluation.json

profile:
	uv run python scripts/profile_runtime.py --requests 20 --output reports/runtime_profile.json
