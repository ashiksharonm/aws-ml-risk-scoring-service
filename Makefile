.PHONY: install setup-dev format lint test train run-api clean

PYTHON = python3
PIP = pip

install:
	$(PIP) install -r requirements.txt

setup-dev: install
	$(PIP) install -r requirements.txt

format:
	ruff format src tests

lint:
	ruff check src tests

test:
	PYTHONPATH=. pytest tests/

train:
	PYTHONPATH=. $(PYTHON) -m src.training.train

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
