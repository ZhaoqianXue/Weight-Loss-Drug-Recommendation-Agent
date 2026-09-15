export PYTHONDONTWRITEBYTECODE := 1
PYTHON ?= .venv/bin/python
NODE ?= node
PORT ?= 5001
OUTPUT ?= .cache/fixture-$(shell date -u +%Y%m%dT%H%M%S)

.PHONY: install test reproduce demo validate build-web reproduce-fixture preview check test-research
install:
	uv venv --python 3.11 .venv
	uv pip sync requirements/offline.lock --python $(PYTHON)
	uv pip install --python $(PYTHON) --no-deps -e .
validate:
	$(PYTHON) -m weightloss validate
build-web:
	$(PYTHON) -m weightloss build-web
test: build-web
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
	PYTHON="$$(command -v $(PYTHON))" $(NODE) tests/frontend/review-assistant.test.js
reproduce:
	$(PYTHON) -m weightloss reproduce-fixture --output-dir $(OUTPUT)
demo: build-web
	$(PYTHON) -m weightloss serve --port $(PORT)
reproduce-fixture: reproduce
preview: demo
test-research:
	.venv-research/bin/python -m unittest discover -s tests/integration/research -p 'test_*.py'
check: validate test
