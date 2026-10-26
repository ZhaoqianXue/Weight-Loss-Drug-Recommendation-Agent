# AskGLP1

Code, prompts, and evaluation questions for the AskGLP1 paper. This review-v2 snapshot includes the research pipeline, question-answering agent, website, and evaluation scripts.

## Contents

| Material | Location |
| --- | --- |
| Complete 120-question list, TSV, and field definitions | [evaluation/questions](evaluation/questions/README.md) |
| Full methods, schema, extended tables and reproducibility documentation | [paper-supplement](paper-supplement/README.md) |
| Screening, extraction, term-selection, agent, answer-checking, and baseline prompts | [prompts](prompts/README.md) |
| Python pipeline, graph tools, agent, and API | [src/weightloss](src/weightloss) |
| Website frontend | [apps/web](apps/web/README.md) |
| Frozen dependency files | [requirements](requirements/README.md) |
| Evaluation runners and human-rating analysis | [results/human-eval-20261020/scripts](results/human-eval-20261020/scripts) |
| Re-extraction and FAERS processing scripts | [results](results) |
| Synthetic reviews for an offline pipeline check | [tests/fixtures/reviews.json](tests/fixtures/reviews.json) |

The complete question list preserves the question IDs, text, order, evaluation groups, and intended response types. The groups describe the study's evaluation tasks and can overlap in content. The question-set README distinguishes intended responses from system outputs and human ratings. Representative questions and the human-rating rules remain in the paper’s annotation appendix.

## Setup and offline check

Python requires version 3.11 or 3.12. The frozen offline environment uses Python 3.11. Install with uv from the repository root.

```sh
uv venv --python 3.11 .venv
uv pip sync requirements/offline.lock --python .venv/bin/python
uv pip install --python .venv/bin/python --no-deps -e .
make reproduce OUTPUT=.cache/synthetic-check
```

The offline check uses synthetic reviews and frozen annotation responses. It does not call a model or reproduce the paper's empirical results. Research dependencies are declared in pyproject.toml and documented in requirements/README.md. The website uses Node.js 22 and pnpm 10.

## Research workflow

The study's extraction prompt is v9-prompt-7 in src/weightloss/extraction/reextract.py. The question-answering agent is agent-0.8.5 in src/weightloss/agent. The prompts directory provides readable copies of the instructions used at each stage and links to the corresponding schemas and input builders.

The research workflow requires source records, extracted statements, licensed terminology resources, a local Neo4j graph, and model API credentials supplied by the operator. They are not bundled in this snapshot. File paths in the research runners identify their expected inputs. The code for MedDRA candidate selection retains its restricted gateway checks. Anonymized endpoint text is not an operational endpoint.

The original user posts and reviews, account identifiers, credentials, and licensed MedDRA vocabulary files are not distributed here. Code availability does not grant permission to redistribute third-party data or terminology.

## License

The code and accompanying author-created materials are distributed under the [MIT License](LICENSE). Third-party dependencies and resources retain their own licenses.
