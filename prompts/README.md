# Prompts

These files are readable copies of the instructions used by the study's processing and question-answering components. Prompt text is copied without language editing. Request builders and strict output schemas remain in the linked source files.

| Stage | Prompt file | Source |
| --- | --- | --- |
| Reddit screening | [screening.json](screening.json) | [Jev worker](../results/jev-full-2026-09-29/scripts/full_run.py), model jev-1.13.0, keep threshold 0.5 |
| Reddit extraction | [extraction-reddit.txt](extraction-reddit.txt) | [reextract.py](../src/weightloss/extraction/reextract.py), v9-prompt-7 |
| WebMD extraction | [extraction-webmd.txt](extraction-webmd.txt) | [reextract.py](../src/weightloss/extraction/reextract.py), v9-prompt-7 |
| MedDRA candidate selection | [term-selection.txt](term-selection.txt) | [select.py](../src/weightloss/standardization/coding/select.py), select-1.1.0 |
| Question-answering agent | [agent.txt](agent.txt) | [agent.py](../src/weightloss/agent/agent.py), agent-0.8.5 |
| Independent answer check | [critic.txt](critic.txt) | [agent.py](../src/weightloss/agent/agent.py) |
| Repair instruction template | [critic-repair.txt](critic-repair.txt) | [agent.py](../src/weightloss/agent/agent.py) |
| LLM-only baseline | [baseline-llm-only.txt](baseline-llm-only.txt) | [baselines.py](../results/human-eval-20261020/scripts/baselines.py) |
| Text-retrieval baseline | [baseline-text-retrieval.txt](baseline-text-retrieval.txt) | [baselines.py](../results/human-eval-20261020/scripts/baselines.py) |

The screening input is one cleaned Reddit post or comment. Extraction inputs use numbered sentences and a source header. Term selection receives a mention, its evidence sentences, and candidate LLTs with their PTs. The agent and baselines receive the evaluation question and the tools defined for that system. Tool definitions, response schemas, and dynamically constructed repair messages are part of the source code rather than these instruction files.

The [paper supplement](../paper-supplement/README.md) also preserves the complete paper prompt descriptions, readable field definitions and exported strict request schemas.
