.PHONY: help install dev-install test lint format clean run docker-build docker-up docker-down

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install production dependencies"
	@echo "  dev-install - Install development dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  clean       - Clean up generated files"
	@echo "  run         - Run the application"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-up   - Start Docker services"
	@echo "  docker-down - Stop Docker services"

# Installation
install:
	pip install -r requirements.txt

dev-install: install
	pip install pytest pytest-asyncio pytest-cov black flake8 mypy isort pre-commit
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage .pytest_cache/ .mypy_cache/

# Development
run:
	python -m src.main

# Docker
docker-build:
	docker build -t lm-training-control-panel .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f app

# DVC
dvc-init:
	dvc init

dvc-add-data:
	dvc add data/
	git add data.dvc .gitignore

dvc-push:
	dvc push

dvc-pull:
	dvc pull

# Database
db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-migration:
	alembic revision --autogenerate -m "$(message)"

# MLflow
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000
