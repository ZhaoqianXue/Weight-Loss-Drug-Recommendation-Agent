# Environments

The core package uses the Python standard library. Three extras are exposed:

| Extra | Purpose |
| --- | --- |
| `demo` | Collection parsing and the website backend (FastAPI, Uvicorn); no Torch, FAISS or PDF stack |
| `research` | Demo plus extraction (official OpenAI SDK, Pydantic), standardization and Neo4j graph import; no LangChain |
| `dev` | Demo plus build tools, offline checks and the API test client (httpx2) |

[Historical environments](../archive/environments/) and the previous two locks are archived. The unused PI scripts and PDF dependencies are outside the maintained research install.

## Offline environment

`offline.lock` is generated from the `dev` group (demo and build tools), with exact versions and distribution hashes. Local checks use Python 3.11.4 and Node.js 22.15.0 with pnpm 10. Website frontend packages are locked separately in `apps/web/pnpm-lock.yaml`. CI also declares Python 3.11 / Node 22 for Ubuntu and macOS; CI execution is separate from local verification.

```sh
uv venv --python 3.11 .venv
uv pip sync requirements/offline.lock --python .venv/bin/python
uv pip install --python .venv/bin/python --no-deps -e .
```

Regenerate intentionally, then retest:

```sh
uv pip compile pyproject.toml --extra dev --generate-hashes --python-version 3.11 -o requirements/offline.lock
```

## Research environment: Intel macOS 14+, Python 3.11

`research-macos-x86_64-py311.lock` includes all maintained optional groups. It is explicitly a platform lock, not a universal cross-platform environment. Intel macOS uses Torch 2.2.2 because Torch 2.7 has no Intel macOS wheel; other platforms retain the declared 2.7 constraint. The LangChain packages, the `faiss-cpu` package and transitive packages that only they required were removed from this lock on 2026-09-27 by offline pruning (no other pin changed; no index data was touched). No model weights are downloaded by installation/import tests.

On the matching platform:

```sh
uv venv --python 3.11 .venv-research
MACOSX_DEPLOYMENT_TARGET=14.0 uv pip sync requirements/research-macos-x86_64-py311.lock --python .venv-research/bin/python --python-platform x86_64-apple-darwin --only-binary :all:
uv pip install --python .venv-research/bin/python --no-deps -e .
.venv-research/bin/python -m unittest discover -s tests/integration/research -p 'test_*.py'
```

The explicit macOS target avoids old Anaconda/uv reporting this macOS 14 host as macOS 10.16 and attempting unnecessary source builds. It must not be used to install on a genuinely older operating system.

Regeneration command:

```sh
MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile pyproject.toml --extra dev --extra research --generate-hashes --only-binary :all: --python-version 3.11 --python-platform x86_64-apple-darwin -o requirements/research-macos-x86_64-py311.lock
```

Other platforms should resolve their own lock from the same extras and run the checks before claiming support. Research adapter tests mock clients; live OpenAI calls, model weight downloads, full FAISS rebuilds and Neo4j integration remain explicitly separate. Legacy UMLS experiments may need dependencies such as nltk/fuzzywuzzy from their original environments; they are not part of the maintained execution pipeline.

Package-definition and lock commands follow [Python Packaging](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) and [uv environment locking](https://docs.astral.sh/uv/pip/compile/).
