"""Candidate selection by gpt-6-luna through the University of Arizona AI Assistant (Responses API, strict schema).

Compliance (docs/meddra-processing-compliance.md): requests carrying MedDRA candidates go only to UA_BASE, use
store=false, and every response must report a Bedrock route; anything else stops the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

UA_BASE = 'https://ai-assistant.responsibleai.arizona.edu/v1'
MODEL = 'gpt-6-luna'
PROMPT_VERSION = 'select-1.1.0'
INSTRUCTIONS = """You code one patient-reported mention to MedDRA. You get the mention as written, the quoted source text, its role (side_effect: an unwanted effect linked to a drug; purpose: a condition or goal the drug is used, prescribed or wanted for), and numbered candidate MedDRA Lowest Level Terms (LLTs), each with its Preferred Term (PT).
Choose the single candidate that most accurately represents what the mention reports, read in the context of the quote. Follow these principles:
- Code what is reported; do not add, infer or upgrade information (a symptom is not a diagnosis).
- Prefer the most specific candidate that is still fully supported by the text; prefer the LLT closest to the reported wording when several fit the same meaning.
- A candidate must match the meaning, not just share words.
- For a purpose, the same term is used whether it is a condition or a desired outcome.
- Code the concept the mention names even when the text says it did not happen, is feared, hypothetical or asked about, or concerns another person. Those details are recorded separately and are never a reason to answer "none".
- Answer "none" only when the mention names no specific medical concept (e.g. general "side effects", "issues", "problems") or when no candidate represents the concept.
Return only the JSON object required by the schema."""
RETRY_HTTP = {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}


class RoutingError(RuntimeError):
    """The gateway reported a route other than AWS Bedrock: stop, do not continue sending MedDRA content."""


def load_key(env_path):
    key = os.environ.get('UA_AI_ASSISTANT_API_KEY')
    if key:
        return key
    for line in Path(env_path).read_text().splitlines():
        if line.startswith('UA_AI_ASSISTANT_API_KEY='):
            return line.split('=', 1)[1].strip().strip('"\'')
    raise RuntimeError('UA_AI_ASSISTANT_API_KEY not found')


def candidate_ids(k):
    return [f'c{i + 1:02d}' for i in range(k)]


def schema(k):
    return {'type': 'object', 'properties': {'choice': {'type': 'string', 'enum': candidate_ids(k) + ['none']}},
            'required': ['choice'], 'additionalProperties': False}


def user_text(mention, candidates):
    lines = [f'{cid}: {c["llt_name"]} (PT: {c["pt_name"]})' for cid, c in zip(candidate_ids(len(candidates)), candidates)]
    quote = ' | '.join(mention['quotes'])
    return f"Role: {mention['role']}\nMention: {mention['verbatim']}\nQuote: {quote}\nCandidates:\n" + '\n'.join(lines)


def build_body(mention, candidates, effort):
    return {'model': MODEL, 'instructions': INSTRUCTIONS, 'input': user_text(mention, candidates), 'store': False,
            'reasoning': {'effort': effort},
            'text': {'format': {'type': 'json_schema', 'name': 'meddra_choice', 'schema': schema(len(candidates)),
                                'strict': True}}}


def prompt_sha256():
    return hashlib.sha256((PROMPT_VERSION + INSTRUCTIONS).encode()).hexdigest()


def post(body, key, timeout=60):
    url = UA_BASE + '/responses'
    if not url.startswith('https://ai-assistant.responsibleai.arizona.edu/'):
        raise RoutingError('MedDRA payloads may only be sent to the UA AI Assistant')
    if body.get('store') is not False:
        raise RoutingError('store must be false')
    req = urllib.request.Request(url, method='POST', data=json.dumps(body).encode(),
                                 headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return {k.lower(): v for k, v in r.headers.items()}, json.loads(r.read().decode())


def parse(resp, k):
    """(choice, anomaly) from the raw response: the single final_answer message, never a concatenation."""
    status = resp.get('status')
    if status != 'completed':
        return None, f'status_{status}'
    msgs = [o for o in resp.get('output', []) if o.get('type') == 'message']
    finals = [o for o in msgs if o.get('phase') == 'final_answer'] or (msgs if len(msgs) == 1 and not msgs[0].get('phase') else [])
    if len(finals) != 1:
        return None, 'no_final_answer' if not finals else 'ambiguous_final'
    parts = finals[0].get('content', [])
    if any(p.get('type') == 'refusal' for p in parts):
        return None, 'refusal'
    text = ''.join(p.get('text', '') for p in parts if p.get('type') == 'output_text')
    try:
        choice = json.loads(text)['choice']
    except (ValueError, KeyError, TypeError):
        return None, 'invalid_json'
    if choice not in candidate_ids(k) + ['none']:
        return None, 'choice_out_of_schema'
    return choice, None


def call(mention, candidates, effort, key, max_attempts=4):
    """One mention -> result dict. Transient HTTP errors are retried; semantic anomalies are recorded, not retried."""
    body = build_body(mention, candidates, effort)
    last = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        try:
            headers, resp = post(body, key)
        except urllib.error.HTTPError as e:
            last = f'http_{e.code}'
            if e.code in RETRY_HTTP and attempt < max_attempts:
                time.sleep(min(60, 2 ** attempt))
                continue
            return {'anomaly': last, 'attempts': attempt}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = f'transport_{type(e).__name__}'
            if attempt < max_attempts:
                time.sleep(min(60, 2 ** attempt))
                continue
            return {'anomaly': last, 'attempts': attempt}
        route = headers.get('x-litellm-model-name', '')
        if not route.startswith('bedrock'):
            raise RoutingError(f'unexpected route {route!r}')
        choice, anomaly = parse(resp, len(candidates))
        usage = resp.get('usage') or {}
        return {'choice': choice, 'anomaly': anomaly, 'attempts': attempt, 'latency_s': round(time.time() - t0, 3),
                'response_id': resp.get('id'), 'route': route, 'api_base': headers.get('x-litellm-model-api-base'),
                'cost_usd': float(headers.get('x-litellm-response-cost') or 0),
                'input_tokens': usage.get('input_tokens'), 'output_tokens': usage.get('output_tokens'),
                'reasoning_tokens': (usage.get('output_tokens_details') or {}).get('reasoning_tokens'),
                'cached_tokens': (usage.get('input_tokens_details') or {}).get('cached_tokens')}
    return {'anomaly': last, 'attempts': max_attempts}
