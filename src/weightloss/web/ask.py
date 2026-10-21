"""The AskGLP1 agent behind the website: one question per request, progress streamed as server-sent events.

The agent (weightloss.agent.agent.run) is used unchanged. Progress comes from two thin wrappers passed through run():
the graph wrapper reports each graph tool call and its result size, the client wrapper reports each model response and
the code cells (run_python) it asks for, which run between two responses. The full
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
        self._cells = {}  # code cells asked for by the last response: step -> (arguments, start time)

    def _finish_cells(self):
        """The agent runs the code cells of a response before it sends the next request; mark them done."""
        for step, (arguments, started) in self._cells.items():
            self._emit({'type': 'tool', 'step': step, 'tool': 'run_python', 'status': 'done', 'arguments': arguments,
                        'seconds': round(time.time() - started, 2)})
        self._cells = {}

    def create(self, **body):
        self._finish_cells()
        self._round += 1
        self._emit({'type': 'model', 'round': self._round, 'status': 'running'})
        response = self._responses.create(**body)
        output = getattr(response, 'output', None) or []
        calls = [item for item in output if getattr(item, 'type', None) == 'function_call']
        self._emit({'type': 'model', 'round': self._round, 'status': 'done', 'tool_calls': len(calls)})
        for item in calls:  # graph tools report themselves through ProgressGraph; code cells are reported here
            if getattr(item, 'name', None) == 'run_python':
                arguments = {'code': _code(getattr(item, 'arguments', '') or '')}
                step = (getattr(item, 'call_id', None) or uuid.uuid4().hex)[-12:]
                self._cells[step] = (arguments, time.time())
                self._emit({'type': 'tool', 'step': step, 'tool': 'run_python', 'status': 'running', 'arguments': arguments})
        return response


class ProgressClient:
    """Passes Responses API calls to the OpenAI client and reports them."""

    def __init__(self, client, emit: Emit):
        self.responses = _Responses(client.responses, emit)


def _code(arguments, limit=4000):
    """The code of a run_python call (its JSON arguments or a dict), shortened for display."""
    code = arguments.get('code') if isinstance(arguments, dict) else _json(arguments).get('code')
    code = code or ''
    return code if len(code) <= limit else code[:limit] + '\n# ... (shortened)'


def _output_lines(value, output, limit=3):
    """The lines of a code output that hold the cited value (as the verifier matched it)."""
    from weightloss.agent.agent import _number, _norm, number_in_output
    lines = [line for line in (output or '').splitlines() if line.strip()]
    if _number(value) is not None:
        hits = [line for line in lines if number_in_output(value, line)]
    else:
        hits = [line for line in lines if _norm(value) in _norm(line)]
    return [line[:300] for line in hits[:limit]]


def has_api_key():
    from weightloss.agent.agent import env_file_key
    return bool(env_file_key() or os.environ.get('OPENAI_API_KEY'))


def resolve(result, trace):
    """The run result with each citation resolved to its tool row and each quote to its source statement."""
    calls = {c['call']: c for c in trace.get('tool_calls') or []}
    answer = result.get('answer') or {}
    citations = []
    for c in answer.get('citations') or []:
        call = calls.get(c['call']) or {}
        if call.get('tool') == 'run_python':  # code output is text: show the code and the lines that hold the value
            output = (call.get('output') or {}).get('output') or ''
            citations.append({**c, 'tool': 'run_python', 'arguments': {},
                              'source': {'code': _code(call.get('arguments') or {}), 'output_lines': _output_lines(c['value'], output)}})
            continue
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
                   'arguments': ({'code': _code(c['arguments'])} if c['tool'] == 'run_python' else
                                 _brief(c['arguments'] if isinstance(c['arguments'], dict) else _json(c['arguments']))),
                   'error': (c.get('output') or {}).get('error'), 'rows': len(_rows(c.get('output') or {}))}
                  for c in trace.get('tool_calls') or [] if c.get('tool') != 'overview']
    return {'run_id': result.get('run_id'), 'question': result.get('question'), 'verified': bool(result.get('verified')),
            'answer_type': answer.get('answer_type'), 'answer': answer.get('answer') or '',
            'limitations': answer.get('limitations') or [], 'citations': citations, 'quotes': quotes,
            'tool_calls': tool_calls, 'problems': result.get('problems') or [], 'repair_used': bool(result.get('repair_used')),
            'repairs': int(result.get('repairs') or (1 if result.get('repair_used') else 0)),
            'stop_reason': result.get('stop_reason'), 'cost_usd': result.get('cost_usd') or 0.0,
            'model': trace.get('model'), 'reasoning_effort': trace.get('reasoning_effort'), 'agent_version': trace.get('agent_version')}


def _json(text):
    try:
        value = json.loads(text) if isinstance(text, str) else {}
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class NoAnswer(Exception):
    """The agent stopped without an answer (the model service could not be reached, or a limit was hit)."""


class AskService:
    """Runs agent questions in worker threads, within a spending cap for the server process. Several visitors can ask at
    once (WEIGHTLOSS_ASK_WORKERS, default 8); a question beyond that waits and is told how many start before it."""

    def __init__(self, kg: KGService, *, budget_usd: Optional[float] = None, workers: Optional[int] = None,
                 trace_dir: Path = TRACE_DIR, run: Optional[Callable] = None, client_factory: Optional[Callable] = None,
                 key_check: Callable[[], bool] = has_api_key):
        self.kg = kg
        self.budget_usd = float(os.environ.get('WEIGHTLOSS_ASK_BUDGET_USD', 5.0) if budget_usd is None else budget_usd)
        self.spent_usd = 0.0
        self.trace_dir = Path(trace_dir)
        self.workers = int(os.environ.get('WEIGHTLOSS_ASK_WORKERS', 8) if workers is None else workers)
        self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='ask')
        self._busy = 0  # questions accepted and not finished, running or waiting
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
        started = time.time()
        for attempt in range(2):  # a dropped connection to the model service is retried once from the start
            trace_path = self.trace_dir / day / f'{uuid.uuid4().hex}.jsonl'
            result = run(question, graph=graph, client=ProgressClient(client, emit), trace_path=trace_path, label='web')
            with self._lock:
                self.spent_usd += float(result.get('cost_usd') or 0.0)
            if result.get('answer') or not str(result.get('stop_reason') or '').startswith('transport_error') or attempt:
                break
            emit({'type': 'model', 'round': 0, 'status': 'running'})  # shown as "Reconnecting to the model service"
        if not result.get('answer'):
            reason = str(result.get('stop_reason') or '')
            raise NoAnswer('The model service could not be reached, so no answer was written. Please ask the question again.'
                           if reason.startswith('transport_error') else
                           'The agent stopped before it could write an answer. Please ask again, or ask a narrower question.')
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
            except (GraphUnavailable, NoAnswer) as error:
                emit({'type': 'error', 'message': str(error)})
            except BaseException as error:  # SystemExit from a missing key included; never leave the stream open
                emit({'type': 'error', 'message': f'The agent could not answer ({type(error).__name__}).'})
            finally:
                with self._lock:
                    self._busy -= 1
                emit(None)

        with self._lock:
            ahead = max(0, self._busy - self.workers + 1)
            self._busy += 1
        yield {'type': 'start', 'question': question}
        if ahead:
            yield {'type': 'queued', 'ahead': ahead}
        loop.run_in_executor(self._pool, work)
        while True:
            event = await queue.get()
            if event is None:
                return
            yield event

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
