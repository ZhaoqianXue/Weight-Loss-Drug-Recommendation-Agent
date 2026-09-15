export PYTHONDONTWRITEBYTECODE := 1
PYTHON ?= .venv/bin/python
NODE ?= node
ROOT := $(abspath .)

.PHONY: install validate test build-web reproduce-fixture preview check check-structure test-research
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
	PYTHON="$(abspath $(PYTHON))" $(NODE) tests/frontend/review-assistant.test.js
reproduce-fixture:
	$(PYTHON) -m weightloss reproduce-fixture --output-dir .cache/fixture-$$(date -u +%Y%m%dT%H%M%S)
preview: build-web
	$(PYTHON) -m weightloss serve
check-structure:
	$(PYTHON) scripts/audit_structure.py
test-research:
	.venv-research/bin/python -m unittest discover -s tests/integration/research -p 'test_*.py'
check: validate test
	$(PYTHON) scripts/audit_migration.py
	$(PYTHON) scripts/audit_structure.py
