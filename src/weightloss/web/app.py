"""FastAPI application: JSON API under /api and the built React frontend (apps/web/dist) at /.

The API reads the local Neo4j knowledge graph through the agent's graph tools (web/kg.py) and runs the AskGLP1 agent
(web/ask.py). Run with `weightloss serve`, or during frontend work with `make web-dev` (Vite proxies /api here).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import schemas as s
from .ask import AskService
from .kg import GraphUnavailable, KGService

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / '.cache/web-kg'
UNAVAILABLE = {503: {'model': s.ErrorResponse, 'description': 'The knowledge graph database is not running'}}


class Frontend(StaticFiles):
    """The built frontend. index.html is revalidated on every load, so a rebuilt site reaches browsers at once; the files
    under assets/ carry a content hash in their names and are cached for good."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers['Cache-Control'] = ('public, max-age=31536000, immutable' if path.startswith('assets/')
                                             else 'no-cache')
        return response


def default_frontend():
    from weightloss.settings import get_settings
    return get_settings().path('web_source') / 'dist'


def create_app(frontend_dir: Optional[Path] = None, kg_service: Optional[KGService] = None,
               ask_service: Optional[AskService] = None, warm: bool = True) -> FastAPI:
    frontend_dir = default_frontend() if frontend_dir is None else frontend_dir

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.kg = kg_service or KGService(cache_dir=CACHE_DIR)
        app.state.ask = ask_service or AskService(app.state.kg)
        if warm:  # compute the default views in the background; requests for them wait instead of recomputing
            threading.Thread(target=_warm, args=(app.state.kg,), daemon=True, name='kg-warm').start()
        yield
        app.state.ask.close()
        app.state.kg.close()

    app = FastAPI(title='AskGLP1 website API', version='2.0.0', lifespan=lifespan,
                  generate_unique_id_function=lambda route: route.name)

    @app.exception_handler(GraphUnavailable)
    async def graph_unavailable(request: Request, error: GraphUnavailable):
        return JSONResponse({'detail': str(error)}, status_code=503)

    def service(request: Request) -> KGService:
        return request.app.state.kg

    @app.get('/api/overview', response_model=s.Overview, responses=UNAVAILABLE)
    def overview(request: Request):
        kg = service(request)
        return {**kg.overview(), 'socs': kg.socs()}

    @app.post('/api/graph', response_model=s.GraphView, responses=UNAVAILABLE)
    def graph(query: s.GraphQuery, request: Request):
        return service(request).graph_view(query.platform, query.level, query.show, query.drugs, query.age_groups, query.sexes,
                                           query.soc, query.top_n)

    @app.get('/api/patterns', response_model=s.Patterns, responses=UNAVAILABLE)
    def patterns(request: Request):
        return service(request).patterns()

    @app.post('/api/faers', response_model=s.FaersView, responses=UNAVAILABLE)
    def faers(query: s.FaersQuery, request: Request):
        return service(request).faers(query.scope, query.rank_by, query.top_n, query.sex, query.age_group)

    @app.post('/api/evidence', response_model=s.EvidenceView, responses=UNAVAILABLE)
    def evidence(query: s.EvidenceQuery, request: Request):
        return service(request).evidence(query.target, query.platform, query.level, query.drug, query.code, query.limit,
                                         query.age_groups, query.sexes, query.soc)

    @app.get('/api/ask/status', response_model=s.AskStatus)
    def ask_status(request: Request):
        return request.app.state.ask.status()

    @app.post('/api/ask', response_class=StreamingResponse,
              responses={200: {'model': s.AskEvent, 'description': 'Server-sent events; each data line is one AskEvent as JSON',
                               'content': {'text/event-stream': {}}}})
    async def ask(body: s.AskRequest, request: Request):
        async def events():
            async for event in request.app.state.ask.stream(body.question.strip()):
                try:
                    payload = s.AskEvent.model_validate(event).model_dump_json(exclude_none=False)
                except ValidationError as error:
                    payload = json.dumps({'type': 'error', 'message': f'Malformed agent event ({error.error_count()} errors).'})
                yield f'data: {payload}\n\n'
        return StreamingResponse(events(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    if (Path(frontend_dir) / 'index.html').is_file():
        app.mount('/', Frontend(directory=frontend_dir, html=True), name='frontend')
    else:
        @app.get('/', include_in_schema=False)
        def frontend_missing():
            return HTMLResponse('<p>The frontend is not built. Run <code>make web-build</code>; the API is under /api.</p>', status_code=503)

    return app


def _warm(kg: KGService):
    try:
        kg.warm()
    except Exception:  # the graph may be down; requests report that themselves
        pass


def openapi_document() -> dict:
    """The API schema, without connecting to the graph."""
    return create_app(frontend_dir=Path('.'), warm=False).openapi()
