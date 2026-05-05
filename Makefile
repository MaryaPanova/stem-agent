.PHONY: venv install test run clean

PYTHON ?= python3.12
VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

PROVIDER ?= openai

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e ".[$(PROVIDER),dev]"

test: install
	$(VENV)/bin/pytest

run: install
	$(PY) main.py run

clean:
	rm -rf $(VENV) __pycache__ stem/__pycache__ tests/__pycache__ *.egg-info
