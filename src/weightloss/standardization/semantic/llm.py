"""Optional, bounded model proposals for unresolved components (official SDK, Responses + strict Structured Outputs).

The default mode sends only the node's own wording and quotes: the model rewrites them as plain standard terms,
splits coordinated items and may say not_target / unsure. No dictionary content is sent; local retrieval turns the
proposed terms into candidates (method llm_proposal), which are never accepted without review. The answer is the
single final message (weightloss.extraction.structured.final_output_text); refusals, truncation, incomplete output,
invalid JSON and schema violations are anomalies, never empty proposals. A budget (requests, tokens, USD) is checked
before every request with a conservative cap and settled with the reported usage.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from weightloss.extraction.structured import (ExtractionAnomaly, build_body, cost_usd, final_output_text, parse_json_object,
                                              request_cap_usd, strict_json_schema, token_split, validate_model)

PROMPT_VERSION = 'semantic-propose-0.2.0'
INSTRUCTIONS = {
    'side_effect': ('You normalize one side-effect phrase from a patient narrative. Using only the phrase and its quotes, '
                    'rewrite each distinct symptom or event it names as a plain, standard clinical term in English. Split '
                    'items only when the text states both; if the phrase gives alternatives or is unclear, mark unsure. '
                    'Do not add diagnoses, causes or codes. If it is not a symptom or event, mark not_target.'),
    'indication': ('You normalize one purpose of medication use from a patient narrative. Using only the phrase and its '
                   'quotes, rewrite each distinct condition, symptom or goal as a plain standard term in English and give '
                   'the intent the text states (treat_condition, relieve_symptom, prevent, goal, other_explicit or '
                   'unclear) and, for goals, the direction. Never infer a diagnosis from a goal (weight loss is not '
                   'obesity; diabetes is not type 2 diabetes). A planned or considered purpose is still a purpose. Mark '
                   'not_target only for a dose, a dosing phase or an outcome stated by itself. Never attribute a purpose to '
                   'a drug other than the one in the phrase context.'),
}


class Closed(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ProposedComponent(Closed):
    text: str = Field(min_length=1, max_length=200, description='verbatim words of the phrase this item covers')
    normalized_term: str = Field(min_length=1, max_length=120)
    role_fit: Literal['fits', 'not_target', 'unsure']
    intent: Optional[Literal['treat_condition', 'relieve_symptom', 'prevent', 'goal', 'other_explicit', 'unclear']]
    goal_direction: Optional[Literal['decrease', 'maintain', 'increase', 'unspecified']]

    @model_validator(mode='after')
    def _direction(self):
        if self.goal_direction is not None and self.intent != 'goal':
            raise ValueError('goal_direction only with intent goal')
        return self


class Proposal(Closed):
    components: list[ProposedComponent] = Field(min_length=1, max_length=6)
    unresolved_reason: Optional[str] = Field(max_length=300)


@dataclass
class Budget:
    """Caps on requests, input/output tokens (output includes reasoning tokens) and USD. Before each request the
    conservative cap is reserved; it is settled with reported usage, or kept as an unknown charge (and the budget
    stops) when usage is missing or the transport failed, so an uncertain spend is never treated as free."""
    model: str
    max_requests: int
    max_input_tokens: int
    max_output_tokens: int
    max_usd: float
    requests: int = 0
    known_input_tokens: int = 0
    known_output_tokens: int = 0
    known_usd: float = 0.0
    unknown_requests: int = 0
    unknown_reserved_usd: float = 0.0
    unknown_reserved_input_tokens: int = 0
    unknown_reserved_output_tokens: int = 0
    stopped: Optional[str] = None

    def _input_bound(self, body):
        return len(json.dumps(body, ensure_ascii=False).encode())  # UTF-8 bytes bound the input tokens

    def admit(self, body):
        if self.stopped:
            return None
        cap = request_cap_usd(self.model, body)  # raises without a price table: no unbounded spend
        size = self._input_bound(body)
        used_in = self.known_input_tokens + self.unknown_reserved_input_tokens
        used_out = self.known_output_tokens + self.unknown_reserved_output_tokens
        for name, over in (('requests', self.requests + 1 > self.max_requests),
                           ('input_tokens', used_in + size > self.max_input_tokens),
                           ('output_tokens', used_out + body['max_output_tokens'] > self.max_output_tokens),
                           ('usd', self.known_usd + self.unknown_reserved_usd + cap > self.max_usd)):
            if over:
                self.stopped = name
                return None
        self.requests += 1
        return {'usd': cap, 'input_tokens': size, 'output_tokens': body['max_output_tokens']}

    @staticmethod
    def usable(usage):
        """Usage counts only when input and output tokens are present as non-negative integers (bool excluded)."""
        if not isinstance(usage, dict):
            return False
        return all(isinstance(usage.get(k), int) and not isinstance(usage.get(k), bool) and usage[k] >= 0
                   for k in ('input_tokens', 'output_tokens'))

    def settle(self, reservation, usage):
        if not self.usable(usage):
            self.unknown_requests += 1
            self.unknown_reserved_usd += reservation['usd']
            self.unknown_reserved_input_tokens += reservation['input_tokens']
            self.unknown_reserved_output_tokens += reservation['output_tokens']
            self.stopped = 'charge_unknown'
            return
        t = token_split(usage)
        self.known_input_tokens += t['input_tokens']
        self.known_output_tokens += t['output_tokens']
        self.known_usd += cost_usd(self.model, usage) or 0.0

    def summary(self):
        return {'model': self.model, 'requests': self.requests, 'known_input_tokens': self.known_input_tokens,
                'known_output_tokens': self.known_output_tokens, 'known_usd': round(self.known_usd, 6),
                'unknown_requests': self.unknown_requests, 'unknown_reserved_usd': round(self.unknown_reserved_usd, 6),
                'unknown_reserved_input_tokens': self.unknown_reserved_input_tokens,
                'unknown_reserved_output_tokens': self.unknown_reserved_output_tokens, 'stopped': self.stopped,
                'caps': {'requests': self.max_requests, 'input_tokens': self.max_input_tokens,
                         'output_tokens': self.max_output_tokens, 'usd': self.max_usd}}


def user_content(binding, record_text=None):
    payload = {'role': binding.role, 'phrase': binding.node['surface'], 'drug_in_context': binding.drug['surface'],
               'quotes': binding.quotes}
    if record_text is not None:
        payload['full_record'] = record_text  # the frozen source record, for pronouns / maintenance readings only
    return json.dumps(payload, ensure_ascii=False)


def request_body(binding, model, max_output_tokens=400, reasoning_effort=None, record_text=None):
    schema = strict_json_schema(Proposal)
    return build_body(model=model, instructions=INSTRUCTIONS[binding.role], user_content=user_content(binding, record_text),
                      schema_name='proposal', schema=schema, max_output_tokens=max_output_tokens,
                      reasoning_effort=reasoning_effort)


def check_proposal(proposal, binding, record_text=None):
    """Each item must quote the phrase, its own quotes or the frozen record verbatim."""
    sources = [binding.node['surface']] + binding.quotes + ([record_text] if record_text else [])
    for item in proposal.components:
        if not any(item.text in s for s in sources):
            raise ExtractionAnomaly('surface_not_found', 'a proposed item is not verbatim in the phrase or its sources')


def context_sha256(binding, record_text=None):
    return hashlib.sha256(user_content(binding, record_text).encode()).hexdigest()


def binding_fields(binding, kg_sha256, record_text=None):
    return {'node_id': binding.node['id'], 'record_id': binding.record_id, 'item_path': binding.node['item_path'],
            'kg_sha256': kg_sha256, 'carrier_sha256': binding.carrier_sha256,
            'context_sha256': context_sha256(binding, record_text), 'record_context': record_text is not None,
            'source_sha256': binding.source_sha256 if record_text is not None else None, 'prompt_version': PROMPT_VERSION}


def propose(bindings, send, budget, *, kg_sha256, records=None, max_output_tokens=400, reasoning_effort=None):
    """One request per binding (no automatic retry) while the budget admits it. Returns result rows bound to the KG
    hash, carrier edge, request context and prompt version."""
    from weightloss.extraction.runner import TransportError
    rows = []
    for b in bindings:
        record_text = None
        if records is not None:
            fields = records[b.record_id]
            record_text = '\n'.join(v for v in fields.values() if isinstance(v, str))
        body = request_body(b, budget.model, max_output_tokens, reasoning_effort, record_text)
        meta = binding_fields(b, kg_sha256, record_text)
        meta['model'] = budget.model
        reservation = budget.admit(body)
        if reservation is None:
            rows.append({**meta, 'status': 'not_sent', 'reason': f'budget:{budget.stopped}'})
            continue
        usage = None
        try:
            status, _, text = send(body)
            if status != 200:
                raise ExtractionAnomaly('http_error', f'HTTP {status}')
            response = json.loads(text)
            usage = response.get('usage')
            proposal = validate_model(Proposal, parse_json_object(final_output_text(response)))
            check_proposal(proposal, b, record_text)
            rows.append({**meta, 'status': 'ok', 'proposal': proposal.model_dump(mode='json'),
                         'usage': token_split(usage) if Budget.usable(usage) else None})
        except ExtractionAnomaly as anomaly:
            rows.append({**meta, 'status': 'anomaly', 'anomaly': anomaly.as_dict(),
                         'usage': token_split(usage) if Budget.usable(usage) else None})
        except (json.JSONDecodeError, TransportError) as error:
            kind = 'transport_error' if isinstance(error, TransportError) else 'invalid_json'
            rows.append({**meta, 'status': 'anomaly', 'anomaly': {'kind': kind, 'detail': type(error).__name__}, 'usage': None})
        finally:
            budget.settle(reservation, usage)
    return rows
