# Contributing

1. Follow the locked offline installation in README and run `make check`.
2. Put reusable code in `src/weightloss/`; keep notebooks and experimental configurations outside the package. Use `get_settings()` inside operations rather than assuming the working directory. Module import must not download models, read study CSVs or call services.
3. Create a new run before changing derived data. Raw snapshots and frozen runs are immutable; generated web data are build outputs. Keep exact annotation-seed, terminology and model provenance.
4. Add tests for scientific invariants and failure cases, including pending/unknown annotations, cross-drug attribution, duplicate IDs and stale retrieval indexes. Use synthetic fixtures for offline integration tests.
5. Separate structural changes from method changes. TableRAG removal, vocabulary choice, graph schema and study scope require their own documented decisions.
6. Update README/data documentation and record new results in `results/`. Do not treat historical results as current benchmarks.
7. Do not commit credentials, local provider event payloads or regenerated cache files. Do not change data licensing or publish datasets merely as part of a code change.

## Verification levels

- `make check`: offline tests, existing dataset consistency, frontend checks and the migration byte audit and the required directory layout.
- `weightloss reproduce-fixture --output-dir PATH`: fresh synthetic replay through a frozen result and static build.
- Research environment tests: model adapters with mocked clients and import checks; no paid requests.
- Live model/retrieval/database checks: explicit operations with configured services. A skipped live check is not a passed check.

## Releases

Before a public release, maintainers must supply approved author/citation metadata and select a software license. Raw/derived review data, third-party terminology and reference PDFs have separate source permissions. No author list or license is inferred from local Git configuration. See TODO for the existing review-text release proposal.

## Test organization

`tests/unit/` covers synthetic collection parsing, annotation reuse and index fingerprints. `tests/integration/` covers the frozen snapshot, isolated runs and the web build. `tests/integration/research/` contains explicitly invoked model-adapter tests and is excluded from offline discovery. Frontend checks and synthetic input files remain in `tests/frontend/` and `tests/fixtures/`.
