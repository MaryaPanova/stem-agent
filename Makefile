.PHONY: install test eval eval-mock evolve docker-build docker-eval-mock docker-eval clean

VENV ?= .venv
PY := $(VENV)/bin/python

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

# Tokenless end-to-end smoke test (no API key needed).
eval-mock:
	$(PY) main.py eval --mock

# Real evaluation across all domains (spends tokens; needs ANTHROPIC_API_KEY).
eval:
	$(PY) main.py eval --domain all -g 3

evolve:
	$(PY) main.py evolve --domain trading -g 3

docker-build:
	docker build -t stem-agent .

# Prove the pipeline inside the container with no key.
docker-eval-mock: docker-build
	docker run --rm stem-agent

# Real eval inside the container.
docker-eval: docker-build
	docker run --rm -e ANTHROPIC_API_KEY=$$ANTHROPIC_API_KEY \
		-e STEM_MODEL=$${STEM_MODEL:-claude-sonnet-4-6} \
		-v $$(pwd)/results:/app/results \
		stem-agent python main.py eval --domain all -g 3

clean:
	rm -rf results checkpoints __pycache__ .pytest_cache *.egg-info
