.PHONY: typecheck demo-drift docker-run install setup-dev format lint test train run-api clean

PYTHON = python3
PIP = pip

install:
	$(PIP) install -r requirements.txt

setup-dev:
	$(PIP) install -r requirements-dev.txt

format:
	ruff format src tests

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

typecheck:
	mypy

demo-drift:
	PYTHONPATH=. $(PYTHON) -m scripts.make_demo_batches
	PYTHONPATH=. $(PYTHON) -m src.monitoring.cli drift data/demo/stable.csv
	PYTHONPATH=. $(PYTHON) -m src.monitoring.cli drift data/demo/drifted.csv

docker-run:
	docker compose up --build

test:
	PYTHONPATH=. pytest --cov

train:
	PYTHONPATH=. $(PYTHON) -m src.training.train

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
