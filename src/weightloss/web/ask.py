"""The AskGLP1 agent behind the website: one question per request, progress streamed as server-sent events.

The agent (weightloss.agent.agent.run) is used unchanged. Progress comes from two thin wrappers passed through run():
the graph wrapper reports each tool call and its result size, the client wrapper reports each model response. The full
trace is written by run() to .cache/agent-runs/web/<date>/<id>.jsonl; the final event resolves every citation to the tool
row it cites and every quote to the statement it comes from, so the page can show the evidence behind each number.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from weightloss.agent import VERSION as AGENT_VERSION

from .kg import GraphUnavailable, KGService, statement

ROOT = Path(__file__).resolve().parents[3]
TRACE_DIR = ROOT / '.cache/agent-runs/web'
TOOLS = ('link_terms', 'query_graph', 'get_evidence', 'compare_faers')
Emit = Callable[[dict], None]


def _brief(arguments):
    """The tool arguments that say what was asked (unset filters left out)."""
    return {k: v for k, v in (arguments or {}).items() if v not in (None, [], False, '')}


def _rows(output):
    rows = {}
    for value in output.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get('row'), int):
                    rows[item['row']] = item
    return rows


class ProgressGraph:
    """Passes tool calls to the knowledge graph and reports them."""

    def __init__(self, graph, emit: Emit):
        self._graph = graph
        self._emit = emit
        self.manifest = getattr(graph, 'manifest', {}) or {}

    def overview(self):
        return self._graph.overview()

    def __getattr__(self, name):
        method = getattr(self._graph, name)
        if name not in TOOLS:
            return method

        def call(**arguments):
            step = uuid.uuid4().hex[:8]
            self._emit({'type': 'tool', 'step': step, 'tool': name, 'status': 'running', 'arguments': _brief(arguments)})
            started = time.time()
            try:
                output = method(**arguments)
            except Exception as error:
                self._emit({'type': 'tool', 'step': step, 'tool': name, 'status': 'error', 'arguments': _brief(arguments),
                            'seconds': round(time.time() - started, 2), 'detail': str(error)[:300]})
                raise
            self._emit({'type': 'tool', 'step': step, 'tool': name, 'status': 'done', 'arguments': _brief(arguments),
                        'seconds': round(time.time() - started, 2), 'rows': len(_rows(output))})
            return output
        return call

    def close(self):
        pass


class _Responses:
    def __init__(self, responses, emit: Emit):
        self._responses = responses
        self._emit = emit
        self._round = 0

    def create(self, **body):
        self._round += 1
        self._emit({'type': 'model', 'round': self._round, 'status': 'running'})
        response = self._responses.create(**body)
        output = getattr(response, 'output', None) or []
        calls = sum(1 for item in output if getattr(item, 'type', None) == 'function_call')
        self._emit({'type': 'model', 'round': self._round, 'status': 'done', 'tool_calls': calls})
        return response


class ProgressClient:
    """Passes Responses API calls to the OpenAI client and reports them."""

    def __init__(self, client, emit: Emit):
        self.responses = _Responses(client.responses, emit)


def has_api_key():
    if os.environ.get('OPENAI_API_KEY'):
        return True
    env = ROOT / '.env'
    return env.is_file() and any(line.startswith('OPENAI_API_KEY=') and line.split('=', 1)[1].strip()
                                 for line in env.read_text().splitlines())


def resolve(result, trace):
    """The run result with each citation resolved to its tool row and each quote to its source statement."""
    calls = {c['call']: c for c in trace.get('tool_calls') or []}
    answer = result.get('answer') or {}
    citations = []
    for c in answer.get('citations') or []:
        call = calls.get(c['call']) or {}
        row = _rows(call.get('output') or {}).get(c['row'])
        citations.append({**c, 'tool': call.get('tool'), 'arguments': _brief(call.get('arguments') if isinstance(call.get('arguments'), dict)
                                                                              else _json(call.get('arguments'))),
                          'source': row})
    statements = {}
    for call in calls.values():
        if call.get('tool') == 'get_evidence':
            for row in (call.get('output') or {}).get('rows') or []:
                statements[row.get('edge_id')] = statement(row)
    quotes = [{**q, 'statement': statements.get(q['edge_id'])} for q in answer.get('quotes') or []]
    tool_calls = [{'call': c['call'], 'tool': c['tool'], 'seconds': c.get('seconds'),
                   'arguments': _brief(c['arguments'] if isinstance(c['arguments'], dict) else _json(c['arguments'])),
                   'error': (c.get('output') or {}).get('error'), 'rows': len(_rows(c.get('output') or {}))}
                  for c in trace.get('tool_calls') or [] if c.get('tool') != 'overview']
    return {'run_id': result.get('run_id'), 'question': result.get('question'), 'verified': bool(result.get('verified')),
            'answer_type': answer.get('answer_type'), 'answer': answer.get('answer') or '',
            'limitations': answer.get('limitations') or [], 'citations': citations, 'quotes': quotes,
            'tool_calls': tool_calls, 'problems': result.get('problems') or [], 'repair_used': bool(result.get('repair_used')),
            'stop_reason': result.get('stop_reason'), 'cost_usd': result.get('cost_usd') or 0.0,
            'model': trace.get('model'), 'reasoning_effort': trace.get('reasoning_effort'), 'agent_version': trace.get('agent_version')}


def _json(text):
    try:
        value = json.loads(text) if isinstance(text, str) else {}
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class AskService:
    """Runs agent questions in worker threads, within a spending cap for the server process."""

    def __init__(self, kg: KGService, *, budget_usd: Optional[float] = None, workers: int = 2, trace_dir: Path = TRACE_DIR,
                 run: Optional[Callable] = None, client_factory: Optional[Callable] = None, key_check: Callable[[], bool] = has_api_key):
        self.kg = kg
        self.budget_usd = float(os.environ.get('WEIGHTLOSS_ASK_BUDGET_USD', 5.0) if budget_usd is None else budget_usd)
        self.spent_usd = 0.0
        self.trace_dir = Path(trace_dir)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ask')
        self._lock = threading.Lock()
        self._run = run
        self._client_factory = client_factory
        self._key_check = key_check

    def status(self):
        if not self._key_check():
            return {'available': False, 'reason': 'No OpenAI API key is configured for the agent.', 'agent_version': AGENT_VERSION,
                    'spent_usd': round(self.spent_usd, 4), 'budget_usd': self.budget_usd}
        if self.spent_usd >= self.budget_usd:
            return {'available': False, 'reason': f'The spending cap of ${self.budget_usd:.2f} for this server has been reached.',
                    'agent_version': AGENT_VERSION, 'spent_usd': round(self.spent_usd, 4), 'budget_usd': self.budget_usd}
        return {'available': True, 'reason': None, 'agent_version': AGENT_VERSION, 'spent_usd': round(self.spent_usd, 4),
                'budget_usd': self.budget_usd}

    def answer(self, question: str, emit: Emit):
        """Blocking: run the agent once and return the resolved result."""
        from weightloss.agent import agent as A
        run = self._run or A.run
        if self._client_factory is not None:
            client = self._client_factory()
        else:
            A.load_key()
            from openai import OpenAI
            client = OpenAI(max_retries=2, timeout=300)
        graph = ProgressGraph(self.kg.graph, emit)
        day = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d')
        trace_path = self.trace_dir / day / f'{uuid.uuid4().hex}.jsonl'
        started = time.time()
        result = run(question, graph=graph, client=ProgressClient(client, emit), trace_path=trace_path, label='web')
        with self._lock:
            self.spent_usd += float(result.get('cost_usd') or 0.0)
        trace = {}
        if trace_path.is_file():
            lines = trace_path.read_text().splitlines()
            trace = json.loads(lines[-1]) if lines else {}
        resolved = resolve(result, trace)
        resolved['seconds'] = round(time.time() - started, 1)
        return resolved

    async def stream(self, question: str):
        """Events for one question: model and tool progress, then 'result' or 'error'."""
        status = self.status()
        if not status['available']:
            yield {'type': 'error', 'message': status['reason']}
            return
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def emit(event):
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def work():
            try:
                emit({'type': 'result', 'result': self.answer(question, emit)})
            except GraphUnavailable as error:
                emit({'type': 'error', 'message': str(error)})
            except BaseException as error:  # SystemExit from a missing key included; never leave the stream open
                emit({'type': 'error', 'message': f'The agent could not answer ({type(error).__name__}).'})
            finally:
                emit(None)

        yield {'type': 'start', 'question': question}
        loop.run_in_executor(self._pool, work)
        while True:
            event = await queue.get()
            if event is None:
                return
            yield event

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
