"""Native OpenAI Responses + Structured Outputs (strict) core shared by synchronous calls and the Batch API.

Sync and Batch send the same request body (`build_body`) and judge answers with the same function
(`final_output_text`). It reads the raw response JSON instead of SDK conveniences: `output_text` joins every message
and `output_parsed` takes the first parseable one, so neither separates a commentary message from the final answer.
Anything other than one clean final message (refusal, truncation, an incomplete or failed response, several candidate
final messages) raises ExtractionAnomaly and is never turned into an empty result.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

# Keywords kept in wire schemas: a deliberately small subset of what strict mode accepts. String lengths, list sizes
# and patterns are enforced locally by the Pydantic models, not by the server.
WIRE_KEYWORDS = frozenset({'type', 'properties', 'required', 'additionalProperties', 'items', 'enum', 'anyOf', 'description'})

# Anomaly kind -> retried automatically (within the runner's attempt cap) or left for review.
ANOMALY_RETRY = {
    'refusal': False,            # a content part of type "refusal"
    'truncated': True,           # status incomplete, reason max_output_tokens; retried once because the observed cause
                                 # was stochastic run-on repetition of a complete answer (never accepted as a result)
    'content_filter': False,     # status incomplete, reason content_filter
    'incomplete': False,         # status incomplete, other or missing reason
    'response_failed': True,     # status failed
    'not_completed': True,       # queued / in_progress / cancelled / no status
    'no_message': True,          # no message item in the output
    'no_final_answer': True,     # messages carry phases but none is final_answer
    'ambiguous_final': True,     # several final_answer messages, or several messages without a phase
    'ambiguous_content': True,   # the final message holds several output_text parts
    'empty_output': True,        # the final message holds no text
    'invalid_json': True,
    'schema_invalid': True,      # Pydantic validation failed (fields, enums, accepted schema conditions)
    'drug_ref_invalid': True,
    'surface_not_found': True,   # a drug surface, indication or effect that does not occur verbatim in the input
    'evidence_not_found': True,
    'evidence_ambiguous': True,
    'field_not_found': True,     # compact-contract-2.0.0: an as-written field (dose, onset, ...) that is no source literal
    'qualifier_unsupported': True,  # compact-contract-2.0.0: severity/event_dose/onset on a not_occurred/not_asserted effect
                                 # that its own quotes do not support
    'http_error': False,         # HTTP status that a retry cannot fix (see http_retriable)
    'transport_error': True,     # timeout or connection loss; the request may have been processed and billed
    'batch_error': True,         # error line for this request in a Batch output or error file
    'missing_result': True,      # a finished Batch returned no line for this request
}

SECRET = re.compile(r'(sk-[A-Za-z0-9_\-]{6,}|Bearer\s+\S+)')

# USD per 1M tokens on the standard tier, as used by the 2026-09-27 pilot. Batch = x BATCH_FACTOR; Batch-mode cache
# pricing has not been measured. Unknown models are logged with tokens only.
PRICES_PER_1M = {
    'gpt-6-luna': {'input': 0.10, 'cached': 0.01, 'cache_write': 0.125, 'output': 0.50},
    'gpt-5-nano': {'input': 0.05, 'cached': 0.005, 'cache_write': 0.0625, 'output': 0.40},
}
BATCH_FACTOR = 0.5


# Stage of an anomaly (2026-09-30 taxonomy): where it arose, so every class gets its own bounded action.
#   provider  no usable reply: transport, HTTP, status failed/incomplete/truncated, refusal, content filter
#   protocol  a reply without exactly one parseable final JSON object
#   structure the JSON violates the wire/canonical field shapes (types, enums, lengths, patterns)
#   contract  a cross-field or source-support rule of the canonical contract
#   reference an item points to a drug handle that does not exist (or handles repeat)
#   lexical   a surface, quote or as-written field is not an exact, token-bounded literal of the input
STAGE_OF = {'refusal': 'provider', 'truncated': 'provider', 'content_filter': 'provider', 'incomplete': 'provider',
            'response_failed': 'provider', 'not_completed': 'provider', 'http_error': 'provider', 'transport_error': 'provider',
            'batch_error': 'provider', 'missing_result': 'provider', 'no_message': 'protocol', 'no_final_answer': 'protocol',
            'ambiguous_final': 'protocol', 'ambiguous_content': 'protocol', 'empty_output': 'protocol', 'invalid_json': 'protocol',
            'schema_invalid': 'structure', 'drug_ref_invalid': 'reference', 'surface_not_found': 'lexical',
            'evidence_not_found': 'lexical', 'evidence_ambiguous': 'lexical', 'field_not_found': 'lexical',
            'qualifier_unsupported': 'contract'}
PROVIDER_KINDS = frozenset(k for k, v in STAGE_OF.items() if v in ('provider', 'protocol'))


class ExtractionAnomaly(Exception):
    """A response that must not be accepted; `kind` is one of ANOMALY_RETRY. `stage` and `paths` are optional
    (older call sites leave them out; as_dict then omits paths and derives the stage from the kind)."""

    def __init__(self, kind, detail='', retriable=None, stage=None, paths=None):
        if kind not in ANOMALY_RETRY:
            raise ValueError(f'unknown anomaly kind: {kind}')
        super().__init__(f'{kind}: {detail}' if detail else kind)
        self.kind, self.detail = kind, detail
        self.retriable = ANOMALY_RETRY[kind] if retriable is None else retriable
        self.stage, self.paths = stage, paths

    def as_dict(self):
        out = {'kind': self.kind, 'detail': self.detail, 'retriable': self.retriable}
        if self.stage is not None:
            out['stage'] = self.stage
        if self.paths:
            out['paths'] = list(self.paths)
        return out


def mask(text):
    return SECRET.sub('[masked]', str(text or ''))


def http_retriable(status):
    return status in (408, 409, 429) or status >= 500


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def schema_sha256(schema):
    """Hash of the schema exactly as sent (property order included, since it shapes generation)."""
    return sha256_text(compact_json(schema))


def strict_json_schema(source):
    """Pydantic model (or JSON schema dict) -> Structured Outputs strict wire schema.

    $refs are inlined (the models are not recursive); every object is closed and lists all of its properties as
    required; `const` becomes a one-element enum; only WIRE_KEYWORDS survive, so titles, defaults and local-only
    constraints (lengths, sizes, patterns) are dropped.
    """
    schema = source.model_json_schema() if hasattr(source, 'model_json_schema') else copy.deepcopy(source)
    defs = schema.get('$defs', {})

    def walk(node):
        if '$ref' in node:
            target = defs[node['$ref'].rsplit('/', 1)[-1]]
            return walk({**target, **{k: v for k, v in node.items() if k != '$ref'}})
        out = {}
        for key, value in node.items():
            if key == 'const':  # single-value Literal: keep the constraint as a one-element enum
                out['enum'] = [copy.deepcopy(value)]
            if key not in WIRE_KEYWORDS:
                continue
            if key == 'properties':
                out[key] = {name: walk(sub) for name, sub in value.items()}
            elif key == 'items':
                out[key] = walk(value)
            elif key == 'anyOf':
                out[key] = [walk(sub) for sub in value]
            elif key != 'required':
                out[key] = copy.deepcopy(value)
        if 'properties' in out:
            out['type'] = 'object'
            out['additionalProperties'] = False
            out['required'] = list(out['properties'])
        return out

    root = walk(schema)
    check_strict(root)
    return root


def check_strict(schema):
    """Raise ValueError unless the schema satisfies the strict-mode shape rules this project relies on."""
    if schema.get('type') != 'object' or 'anyOf' in schema:
        raise ValueError('strict schema root must be a plain object')

    def walk(node, path):
        extra = set(node) - WIRE_KEYWORDS
        if extra:
            raise ValueError(f'{path}: unsupported keywords {sorted(extra)}')
        if node.get('type') == 'object':
            if node.get('additionalProperties') is not False or node.get('required') != list(node.get('properties', {})):
                raise ValueError(f'{path}: objects must be closed and require every property')
        for name, sub in node.get('properties', {}).items():
            walk(sub, f'{path}.{name}')
        if 'items' in node:
            walk(node['items'], f'{path}[]')
        for i, sub in enumerate(node.get('anyOf', [])):
            walk(sub, f'{path}|{i}')

    walk(schema, '$')


def build_body(*, model, instructions, user_content, schema_name, schema, max_output_tokens, reasoning_effort=None,
               temperature=None, store=False, text_verbosity=None):
    """The one request body used by both the synchronous endpoint and Batch lines."""
    body = {'model': model, 'instructions': instructions, 'input': [{'role': 'user', 'content': user_content}]}
    if reasoning_effort is not None:
        body['reasoning'] = {'effort': reasoning_effort}
    if temperature is not None:
        body['temperature'] = temperature
    body['text'] = {'format': {'type': 'json_schema', 'name': schema_name, 'strict': True, 'schema': schema}}
    if text_verbosity is not None:
        body['text']['verbosity'] = text_verbosity
    body['max_output_tokens'] = max_output_tokens
    body['store'] = store
    return body


def batch_line(custom_id, body):
    return compact_json({'custom_id': custom_id, 'method': 'POST', 'url': '/v1/responses', 'body': body}) + '\n'


def final_output_text(response):
    """Text of the single final message of a raw Responses JSON object, or ExtractionAnomaly."""
    status = response.get('status')
    if status == 'incomplete':
        reason = (response.get('incomplete_details') or {}).get('reason')
        kind = {'max_output_tokens': 'truncated', 'content_filter': 'content_filter'}.get(reason, 'incomplete')
        raise ExtractionAnomaly(kind, f'status incomplete, reason {reason}')
    if status == 'failed':
        error = response.get('error') or {}
        raise ExtractionAnomaly('response_failed', f"{error.get('code')}: {mask(error.get('message'))[:200]}")
    if status != 'completed':
        raise ExtractionAnomaly('not_completed', f'status {status}')
    messages = [item for item in response.get('output') or [] if item.get('type') == 'message']
    for message in messages:
        for part in message.get('content') or []:
            if part.get('type') == 'refusal':
                raise ExtractionAnomaly('refusal', mask(part.get('refusal'))[:200])
    if not messages:
        raise ExtractionAnomaly('no_message', 'output has no message item')
    finals = [m for m in messages if m.get('phase') == 'final_answer']
    if len(finals) > 1:
        raise ExtractionAnomaly('ambiguous_final', f'{len(finals)} final_answer messages')
    if not finals:
        phases = [m.get('phase') for m in messages]
        if any(p is not None for p in phases):
            raise ExtractionAnomaly('no_final_answer', f'message phases {phases}')
        if len(messages) > 1:
            raise ExtractionAnomaly('ambiguous_final', f'{len(messages)} messages without phase')
        finals = messages
    texts = [p for p in finals[0].get('content') or [] if p.get('type') == 'output_text']
    if len(texts) > 1:
        raise ExtractionAnomaly('ambiguous_content', f'{len(texts)} output_text parts in the final message')
    if not texts or not texts[0].get('text'):
        raise ExtractionAnomaly('empty_output', 'final message has no text')
    return texts[0]['text']


def parse_json_object(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ExtractionAnomaly('invalid_json', f'{error.msg} at position {error.pos}') from None
    if not isinstance(value, dict):
        raise ExtractionAnomaly('invalid_json', 'top level is not an object')
    return value


def validate_model(model, value):
    """Pydantic validation with an error summary that never echoes input text."""
    from pydantic import ValidationError
    try:
        return model.model_validate(value)
    except ValidationError as error:
        errors = error.errors(include_url=False, include_input=False, include_context=False)
        problems = [f"{'.'.join(map(str, e['loc'])) or '$'}: {e['msg']}" for e in errors]
        # value_error comes from the model validators (cross-field rules); anything else is a field-shape violation
        stage = 'contract' if errors and all(e['type'] == 'value_error' for e in errors) else 'structure'
        raise ExtractionAnomaly('schema_invalid', '; '.join(problems[:5]) + (f' (+{len(problems) - 5} more)' if len(problems) > 5 else ''),
                                stage=stage, paths=['.'.join(map(str, e['loc'])) for e in errors][:20]) from None


def token_split(usage):
    """Plain input, cached input, cache-write input, output and reasoning tokens from a Responses usage object."""
    usage = usage or {}
    details = usage.get('input_tokens_details') or {}
    cached = details.get('cached_tokens') or 0
    write = details.get('cache_write_tokens') or details.get('cache_creation_tokens') or usage.get('cache_write_tokens') or 0
    total = usage.get('input_tokens') or 0
    return {'input_tokens': total, 'plain_input_tokens': max(total - cached - write, 0), 'cached_tokens': cached,
            'cache_write_tokens': write, 'output_tokens': usage.get('output_tokens') or 0,
            'reasoning_tokens': (usage.get('output_tokens_details') or {}).get('reasoning_tokens') or 0}


def cost_usd(model, usage, batch=False):
    prices = PRICES_PER_1M.get(model)
    if prices is None:
        return None
    t = token_split(usage)
    usd = (t['plain_input_tokens'] * prices['input'] + t['cached_tokens'] * prices['cached']
           + t['cache_write_tokens'] * prices['cache_write'] + t['output_tokens'] * prices['output']) / 1e6
    return round(usd * (BATCH_FACTOR if batch else 1), 9)


def request_cap_usd(model, body, batch=False, size=None):
    """Conservative per-request cap: UTF-8 bytes of the body as an input-token bound (an assumption, not an API limit)
    at the dearest input rate, plus max_output_tokens (the only part the API enforces) at the output rate. `size` (the
    UTF-8 length of compact_json(body)) may be passed when the caller already serialized the body."""
    prices = PRICES_PER_1M.get(model)
    if prices is None:
        raise ValueError(f'no price table for {model}; a budget cannot be enforced')
    if size is None:
        size = len(compact_json(body).encode())
    usd = (size * max(prices['input'], prices['cache_write']) + body['max_output_tokens'] * prices['output']) / 1e6
    return usd * (BATCH_FACTOR if batch else 1)
