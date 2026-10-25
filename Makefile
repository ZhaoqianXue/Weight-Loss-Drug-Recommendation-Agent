export PYTHONDONTWRITEBYTECODE := 1
PYTHON ?= .venv/bin/python
WEB := apps/web
PORT ?= 5001
OUTPUT ?= .cache/fixture-$(shell date -u +%Y%m%dT%H%M%S)

.PHONY: install test reproduce demo validate build-web reproduce-fixture preview check test-research web-install web-build web-dev web-check web-api
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
reproduce:
	$(PYTHON) -m weightloss reproduce-fixture --output-dir $(OUTPUT)
demo: web-build
	$(PYTHON) -m weightloss serve --port $(PORT)
# One-time setup of the website on a fresh checkout: packages, frontend, then the site data bundle (graph and FAERS
# tables, from the project Google Drive folder) imported into the local Neo4j. Usage: make site-setup BUNDLE=<zip>
site-setup: install web-install web-build
	$(PYTHON) -m weightloss setup-site $(if $(BUNDLE),--bundle "$(BUNDLE)",)
# Website frontend (Vite + React in apps/web); the FastAPI backend is weightloss.web.
web-install:
	pnpm --dir $(WEB) install --frozen-lockfile
web-build:
	pnpm --dir $(WEB) build
web-check:
	pnpm --dir $(WEB) lint
	pnpm --dir $(WEB) build
# Regenerate the frontend API types after changing src/weightloss/web/schemas.py.
web-api:
	PYTHON="$(abspath $(PYTHON))" pnpm --dir $(WEB) gen:api
# API with auto-reload on PORT, frontend with hot reload on http://localhost:5173.
web-dev:
	$(PYTHON) -m weightloss serve --port $(PORT) --reload & API=$$!; trap "kill $$API" EXIT; WEIGHTLOSS_API=http://127.0.0.1:$(PORT) pnpm --dir $(WEB) dev
reproduce-fixture: reproduce
preview: demo
test-research:
	.venv-research/bin/python -m unittest discover -s tests/integration/research -p 'test_*.py'
check: validate test web-check
