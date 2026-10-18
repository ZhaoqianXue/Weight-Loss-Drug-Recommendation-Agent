# Agent conventions

- This project does not use LangChain or derived frameworks (langchain*, langgraph, langsmith). Model calls use the official provider SDK (OpenAI Python SDK, Responses API with Structured Outputs `strict`) and local schema validation (Pydantic v2), as in `src/weightloss/extraction/structured.py`.
- Read model answers from the raw response JSON: take the single `final_answer` message (or the only message when none has a phase), never concatenate messages, and treat refusals, truncation and incomplete output as anomalies rather than empty results.
- Production implementation work is done by Claude Opus 5.5 at max effort; Codex supervises and accepts.
- Paid API use: low-cost, bounded sample and debugging runs are pre-authorized by the user and need no per-run request; prefer synchronous calls; estimate Batch-equivalent token cost at 50% of the Standard rate. Full-corpus inference needs separate approval.
- Keep raw data, archives and historical outputs unchanged; never print or commit credentials.

## Website

- Backend: FastAPI in `src/weightloss/web/`. `schemas.py` is the HTTP contract; `kg.py` shapes views from the agent's graph tools (`weightloss.agent.tools`), so counts follow the same rules as the agent's and are never recomputed in the web layer; `ask.py` runs the agent unchanged and streams its progress; `app.py` only wires routes. Statistics are computed in the backend, never in the browser. The web never edits `weightloss.agent` or `weightloss.graph`.
- Frontend: `apps/web/` (Vite, React, TypeScript, Tailwind CSS 4, shadcn/ui, TanStack Query, D3), managed with pnpm. One folder per page section under `src/features/`.
- After changing `schemas.py`, run `make web-api`. Never edit `apps/web/src/api/openapi.json` or `schema.d.ts` by hand; a test fails when they are stale.
- Add shadcn components with `pnpm dlx shadcn@4.22.0 add <name>` in `apps/web`, then edit them in place.
- Checks: `make test` (Python, including API tests; the real-graph tests skip when Neo4j is down) and `make web-check` (eslint, tsc, build). Development server: `make web-dev` (start Neo4j first with `python -m weightloss.graph.neo4j_db start`).
