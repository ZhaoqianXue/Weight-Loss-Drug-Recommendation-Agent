# Website frontend

AskGLP1: an explorer for the knowledge graph of patient-reported GLP-1 purposes and side effects (Reddit, WebMD), its comparison with FAERS, and the AskGLP1 agent, whose answers cite every number to a graph query.

Vite + React 19 + TypeScript, styled with Tailwind CSS 4 and shadcn/ui (Radix primitives). The graph is drawn with D3. The visual language (cream canvas, hairline cards, navy accent, Newsreader / Hanken Grotesk / IBM Plex Mono) follows ReviewPilot; its tokens live in `src/index.css`. The backend is the FastAPI app in `src/weightloss/web/`; this package only renders what the API returns.

```text
src/
  api/          openapi.json and schema.d.ts (generated), client.ts (also the agent's event stream), queries.ts
  features/     one folder per page section:
                filters/ and graph/  the knowledge-graph explorer and its evidence panel
                faers/               patient records next to FAERS reports
                patterns/            distributions per platform
                about/               versions and counting rules
                ask/                 the AskGLP1 agent panel
  components/   shared pieces (statement view, segmented control) and shadcn components in ui/
  state.ts      explorer filters
```

From the repository root, with the graph running (`python -m weightloss.graph.neo4j_db start`):

```sh
make web-install   # pnpm install --frozen-lockfile
make web-dev       # API on :5001 with reload, frontend with hot reload on http://localhost:5173
make web-api       # after changing src/weightloss/web/schemas.py
make web-check     # eslint + tsc + vite build
make demo          # build, serve both on http://127.0.0.1:5001
```

Questions to the agent need `OPENAI_API_KEY` (environment or the repository `.env`) and stop at `WEIGHTLOSS_ASK_BUDGET_USD` (default 5) of model spend per server process. Each question runs the agent once; the full trace is kept in `.cache/agent-runs/web/`. Graph views are cached in `.cache/web-kg/` under the loaded graph's manifest.

Add a shadcn component with `pnpm dlx shadcn@4.22.0 add <name>` from this directory.

TypeScript stays on 6.0, as in the official shadcn Vite template: TypeScript 7 has no JavaScript compiler API yet, which typescript-eslint and openapi-typescript need.
