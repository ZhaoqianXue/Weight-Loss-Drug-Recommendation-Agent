"""Opt-in purpose pass: candidate purpose-pass-1.0.0 with merge purpose-merge-1.0.0 (stage 2 of a two-call design).

Stage 1 is the accepted compact-p7t4 chain (localize-1.3.0, literal-select-1.4.0, reextract-once-1.4.0); its validated
final record is the base. Stage 2 is one evidence-first, purpose-only request on the full source text (GPT-6 Luna,
reasoning low): every purpose the text states, each with an exact reason quote that holds the reason words and, when
the reason sentence does not name the drug, an exact anchor quote that does. The merge replaces only the purposes:
side_effects and no_side_effects (and the drug entries they use) stay byte-identical to stage 1, stage-1 purposes are
kept in the provenance record, new drug referents get new handles, and nothing is filled in from gold or from
neighbouring text. This module sends nothing: the runner (owned by Codex) sends the bodies built here, stores the raw
Responses JSON, and this module evaluates and merges them offline.

  python -m weightloss.extraction.purpose_pass fingerprint
  python -m weightloss.extraction.purpose_pass bodies --inputs INPUTS --stage1 EXPORT --out BODIES.jsonl
  python -m weightloss.extraction.purpose_pass merge --inputs INPUTS --stage1 EXPORT --responses RAW.jsonl --run RUN_ID
                                                       --out MERGED.jsonl --provenance PROV.jsonl --summary SUMMARY.json

Each stage-2 item is checked deterministically and kept only if every check passes (else it is listed as rejected with
its reasons, never repaired or re-asked): the reason quote and the anchor quote are exact, unique, token-bounded passages
of the source; the indication is copied from inside the reason quote and is not a pronoun; the drug is a listed handle
or a declared new drug whose surface is written in the source; and the drug's surface occurs in the reason or anchor
quote outside a condition label ("Ozempic face" is not a medication mention). A record whose answer is a refusal,
truncated, incomplete, not valid JSON or not schema-valid gets no purposes (purpose_pass_status "failed"): there is no
fallback to stage-1 purposes, and the record stays in every recall denominator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import Field

from . import compact, contract2, core_export
from .localize import splits_token_v2
from .compact import Label, Quote, StatementKind, Subject, Surface, UseStatus, Wire
from .structured import (ExtractionAnomaly, build_body, check_strict, compact_json, final_output_text, parse_json_object, request_cap_usd,
                         schema_sha256, sha256_text, strict_json_schema, validate_model)

VERSION = 'purpose-pass-1.0.0'
MERGE_VERSION = 'purpose-merge-1.0.0'
MODEL = 'gpt-6-luna'
REASONING = 'low'
TEXT_VERBOSITY = 'low'
MAX_OUTPUT_TOKENS = 4096
SCHEMA_NAME = 'weightloss_purpose_pass'
BASE_SPEC = 'compact-p7t4'

INSTRUCTIONS = """You list the purposes of drugs that one social-media post or health review states, for a knowledge graph. Use only the given text: no outside or drug knowledge, no guesses.

A purpose is the reason the text gives for taking, being prescribed or covered for, wanting, continuing, switching to or changing the dose of a drug. It applies to any drug: weight-loss or diabetes drugs, remedies or supplements taken for a symptom or side effect, and other medicines. Record a purpose only when the text gives that reason:
- "for X", "to X", "because of X", "prescribed/covered for X", "so I can X", "I take it when X";
- a remedy taken for a symptom ("took Zofran for the nausea", "Miralax when I'm backed up", "took 2 Advil and it controlled the headache");
- a switch or dose decision with its stated reason ("switched to Zep because the food noise came back" -> the food noise; "went up because I stalled" -> the stall);
- maintenance given as the reason ("staying on 2.5 to maintain", "maintenance dose").
Not a purpose: a result or benefit the writer only noticed (lost weight, A1C dropped, less food noise, pain improved, "it helped my mood"), unless the text says the drug is taken, kept or chosen for it; a diagnosis or condition only listed next to a drug; a drug class name alone; a dose schedule with no stated goal; a side effect.

For each purpose, one item per drug and reason:
- reason_quote: one exact passage from the text, copied character for character, that holds the reason words. Never join, shorten inside or edit passages.
- anchor_quote: null when reason_quote names the drug; otherwise one exact passage that names that same drug referent (e.g. reason_quote "I take it for nausea", anchor_quote "my doctor prescribed Zofran").
- indication: the reason itself, copied from inside reason_quote (short; never a pronoun such as "it" or "that").
- basis: "stated_reason", "symptom_relief", "dose_or_switch_reason", "maintenance" or "prescribed_or_covered_for".
- polarity: "affirmed"; "negated" when the text denies that reason ("not taking it for weight loss"); "uncertain" when it is hoped, tried or asked ("hoping it helps my migraines").
- subject: "self" (the writer), "other" (a specific person; subject_label as written, e.g. "my husband", "you") or "general" (people in general, hearsay, studies). Never move another person's purpose to the writer.
- statement_kind: "report", "question" or "hypothetical"; use_status: "current", "past", "planned", "considering", "not_using" or "unknown" as stated, null when not stated.
- drug: a listed handle when the purpose is about that referent. Otherwise declare it in new_drugs (handle n1, n2, ...; surface copied as written; ref "name", "class" or "unspecified") and use that handle. Never tie a reason to a drug the text does not tie it to. A drug name inside a condition or symptom label ("Ozempic face", "Ozempic butt") is not a medication mention.

List every stated purpose, including all drugs and all reasons in the text. Return empty lists when the text states none. Answer once, with the final JSON only."""

NewHandle = Annotated[str, Field(pattern=r'^n[1-9][0-9]*$')]
AnyHandle = Annotated[str, Field(pattern=r'^[dn][1-9][0-9]*$')]
BASIS = ('stated_reason', 'symptom_relief', 'dose_or_switch_reason', 'maintenance', 'prescribed_or_covered_for')


class WireNewDrug(Wire):
    handle: NewHandle
    surface: Surface
    ref: Literal['name', 'class', 'unspecified']


class WirePurposeItem(Wire):
    drug: AnyHandle
    reason_quote: Quote
    anchor_quote: Optional[Quote]
    indication: Surface
    basis: Literal['stated_reason', 'symptom_relief', 'dose_or_switch_reason', 'maintenance', 'prescribed_or_covered_for']
    polarity: Literal['affirmed', 'negated', 'uncertain']
    subject: Subject
    subject_label: Optional[Label]
    statement_kind: StatementKind
    use_status: UseStatus


class WirePurposePass(Wire):
    new_drugs: list[WireNewDrug]
    purposes: list[WirePurposeItem]


SCHEMA = strict_json_schema(WirePurposePass)
PRONOUNS = frozenset('it that this these those them they which one'.split())
RULES = {'version': VERSION, 'merge': MERGE_VERSION,
         'quote': 'unique exact token-bounded passage; else unique token-bounded passage differing only in whitespace runs (every word, case kept)',
         'indication': 'exact token-bounded phrase inside its own reason span; a case/whitespace variant becomes that one original phrase (logged)',
         'subject_label': 'as written (contract 2.x); one original case/whitespace variant (logged); else dropped (logged)',
         'new_drug': 'unique handle (all duplicates invalid), surface written in the text outside condition labels, alias surfaces ref name',
         'pronouns': sorted(PRONOUNS),
         'drug': 'listed or declared handle; surface outside a condition label in reason or anchor quote',
         'condition_label': core_export.CONDITION_LABEL.pattern, 'dedupe': '(drug, indication lower, subject class, polarity)',
         'failed_record': 'no purposes, no stage-1 fallback'}


def fingerprint():
    return {'candidate': VERSION, 'merge': MERGE_VERSION, 'base_spec': BASE_SPEC, 'model': MODEL, 'reasoning_effort': REASONING,
            'text_verbosity': TEXT_VERBOSITY, 'max_output_tokens': MAX_OUTPUT_TOKENS, 'schema_name': SCHEMA_NAME,
            'schema_sha256': schema_sha256(SCHEMA), 'prompt_sha256': sha256_text(INSTRUCTIONS),
            'rules_sha256': hashlib.sha256(json.dumps(RULES, sort_keys=True).encode()).hexdigest()}


def user_content(inp, stage1):
    listed = [f'- {h}: {json.dumps(d["surface"], ensure_ascii=False)} [{d["ref"]}]' for h, d in stage1['drugs'].items()] or ['- (none)']
    return (compact.render_input(inp) + '\n\nDrug referents already listed for this text (handle: surface [ref]):\n' + '\n'.join(listed)
            + '\nUse a listed handle for these referents; declare a new drug (n1, n2, ...) only for a referent not listed.')


def request_body(inp, stage1):
    """The exact Responses API body of the stage-2 request (sent by the runner, never here)."""
    if inp.record_id != stage1['record_id']:
        raise ValueError(f'{inp.record_id}: stage-1 record {stage1["record_id"]} does not match')
    return build_body(model=MODEL, instructions=INSTRUCTIONS, user_content=user_content(inp, stage1), schema_name=SCHEMA_NAME, schema=SCHEMA,
                      max_output_tokens=MAX_OUTPUT_TOKENS, reasoning_effort=REASONING, store=False, text_verbosity=TEXT_VERBOSITY)


def _bounded_hits(pattern, fields, start=0, end=None):
    """Regex hits that respect the version-2 token/number boundaries of the stage-1 contract."""
    return [(name, m.start(), m.end()) for name, text in fields.items()
            for m in pattern.finditer(text, start, len(text) if end is None else end) if not splits_token_v2(text, m.start(), m.end())]


def _spaced(phrase, flags=0):
    return re.compile(r'\s+'.join(re.escape(t) for t in phrase.split()), flags)


def locate(quote, fields):
    """(span, how) for a quote: the unique exact token-bounded passage, else the unique token-bounded passage that differs
    only in whitespace runs (every word kept, case kept); (None, 'ambiguous' | 'not_found') otherwise."""
    hits = contract2.literal_hits(quote, fields)
    if len(hits) == 1:
        return hits[0], 'exact'
    if hits or not quote.split():
        return None, 'ambiguous' if hits else 'not_found'
    loose = _bounded_hits(_spaced(quote), fields)
    if len(loose) == 1:
        return loose[0], 'whitespace'
    return None, 'ambiguous' if loose else 'not_found'


def exact_indication(indication, fields, span):
    """(phrase, how): the indication as an exact token-bounded phrase inside its own reason span; a case/whitespace variant
    is replaced by the one original phrase it matches there (logged); (None, reason) otherwise."""
    if _norm(indication) in PRONOUNS or not indication.split():
        return None, 'indication_pronoun'
    name, a, b = span
    text = fields[name]
    i = text.find(indication, a)
    while i != -1 and i + len(indication) <= b:
        if not splits_token_v2(text, i, i + len(indication)):
            return indication, 'exact'
        i = text.find(indication, i + 1)
    found = {text[s:e] for _, s, e in _bounded_hits(_spaced(indication, re.I), {name: text}, a, b)}
    if len(found) == 1:
        return found.pop(), 'normalized'
    return None, 'indication_ambiguous' if found else 'indication_not_in_reason_quote'


def exact_label(label, fields):
    """(label, note): an as-written subject_label (contract 2.x); a case/whitespace variant becomes the one original
    phrase (logged); a label that is not written in the text is dropped (logged), never invented."""
    if label is None or contract2.literal_hits(label, fields):
        return label, None
    found = {fields[n][s:e] for n, s, e in _bounded_hits(_spaced(label, re.I), fields)} if label.split() else set()
    return (found.pop(), 'subject_label_normalized') if len(found) == 1 else (None, 'subject_label_dropped')


def _norm(s):
    return ' '.join((s or '').lower().replace('’', "'").split())


def _surface_outside_label(surface, quotes):
    """The drug surface occurs in one of the quotes and at least once outside a condition label."""
    pattern = core_export.bounded(surface)
    if core_export.CONDITION_SURFACE.search(surface):
        return False
    return any(not core_export.CONDITION_LABEL.match(q[m.end():m.end() + 24]) for q in quotes for m in pattern.finditer(q))


def evaluate(response, inp, stage1):
    """Raw Responses JSON -> {'status', 'anomaly', 'new_drugs', 'accepted', 'rejected'} (deterministic, offline)."""
    out = {'record_id': inp.record_id, 'status': 'ok', 'anomaly': None, 'new_drugs': [], 'accepted': [], 'rejected': []}
    try:
        wire = validate_model(WirePurposePass, parse_json_object(final_output_text(response)))
    except ExtractionAnomaly as anomaly:
        out.update(status='failed', anomaly=anomaly.as_dict())
        return out
    fields = dict(inp.fields)
    drugs = {h: dict(d) for h, d in stage1['drugs'].items()}
    counts = Counter(nd.handle for nd in wire.new_drugs)
    for nd in wire.new_drugs:
        problems, notes, ref = [], [], nd.ref
        if counts[nd.handle] > 1:
            problems.append('duplicate_handle')  # every copy is invalid: a reference to it would be ambiguous
        if not contract2.literal_hits(nd.surface, fields):
            problems.append('surface_not_in_text')
        elif core_export.drug_identity({'surface': nd.surface, 'ref': nd.ref}, fields)['status'] != 'as_extracted':
            problems.append('surface_only_in_condition_label')
        if ref != 'name' and contract2.ALIAS_OF.get(contract2._alias_key(nd.surface)):
            ref = 'name'  # compact-contract-2.1.0 identity rule
            notes.append('ref_alias_to_name')
        out['new_drugs'].append({'handle': nd.handle, 'surface': nd.surface, 'ref': ref, 'problems': problems, 'notes': notes})
        if not problems:
            drugs[nd.handle] = {'surface': nd.surface, 'ref': ref}
    keys = set()
    for k, item in enumerate(wire.purposes):
        it = item.model_dump()
        reasons, notes = [], []
        if it['drug'] not in drugs:
            reasons.append('unknown_or_invalid_drug')
        reason_span, how = locate(it['reason_quote'], fields)
        if reason_span is None:
            reasons.append(f'reason_quote_{how}')
        elif how != 'exact':
            notes.append('reason_quote_whitespace')
        anchor_span = None
        if it['anchor_quote'] is not None:
            anchor_span, ahow = locate(it['anchor_quote'], fields)
            if anchor_span is None:
                reasons.append(f'anchor_quote_{ahow}')
            elif ahow != 'exact':
                notes.append('anchor_quote_whitespace')
        indication = None
        if reason_span is not None:
            indication, ihow = exact_indication(it['indication'], fields, reason_span)
            if indication is None:
                reasons.append(ihow)
            elif ihow != 'exact':
                notes.append('indication_normalized')
        if it['drug'] in drugs:
            quotes = [fields[sp[0]][sp[1]:sp[2]] for sp in (reason_span, anchor_span) if sp] or [it['reason_quote']]
            if not _surface_outside_label(drugs[it['drug']]['surface'], quotes):
                reasons.append('drug_not_named_in_quotes')
        label, lnote = exact_label(it['subject_label'], fields) if it['subject'] == 'other' else (it['subject_label'], None)
        if lnote:
            notes.append(lnote)
        key = (it['drug'], _norm(indication or it['indication']), 'self' if it['subject'] == 'self' else 'nonself', it['polarity'])
        if not reasons and key in keys:
            reasons.append('duplicate')
        if reasons:
            out['rejected'].append({'index': k, 'item': it, 'reasons': reasons})
            continue
        keys.add(key)
        spans = [f'{reason_span[0]}:{reason_span[1]}-{reason_span[2]}']
        if anchor_span and not (anchor_span[0] == reason_span[0] and reason_span[1] <= anchor_span[1] and anchor_span[2] <= reason_span[2]):
            spans.append(f'{anchor_span[0]}:{anchor_span[1]}-{anchor_span[2]}')
        out['accepted'].append({'index': k, 'item': it, 'indication': indication, 'subject_label': label, 'ev': spans, 'notes': notes})
    return out


def merge(stage1, evaluation, run_id):
    """(merged compact record, provenance). Only purposes change; non-P groups and the drug entries they use stay identical."""
    drugs = {h: dict(d) for h, d in stage1['drugs'].items()}
    handle_map, k = {}, max([int(h[1:]) for h in drugs] + [0]) + 1
    for nd in evaluation['new_drugs']:
        if nd['problems']:
            continue
        same = [h for h, d in drugs.items() if d['surface'].lower() == nd['surface'].lower() and d['ref'] == nd['ref']]
        if same:
            handle_map[nd['handle']] = same[0]
        else:
            handle_map[nd['handle']] = f'd{k}'
            drugs[f'd{k}'] = {'surface': nd['surface'], 'ref': nd['ref']}
            k += 1
    purposes = []
    for acc in evaluation['accepted']:
        it = acc['item']
        entry = {'drug': handle_map.get(it['drug'], it['drug']), 'indication': acc['indication'], 'subject': it['subject'],
                 'subject_label': acc['subject_label'], 'use_status': it['use_status'], 'statement_kind': it['statement_kind'],
                 'polarity': it['polarity'], 'ev': acc['ev']}
        purposes.append({key: v for key, v in entry.items() if v is not None})
    merged = {'record_id': stage1['record_id'], 'run': run_id, 'drugs': {}, 'purposes': purposes,
              'side_effects': copy.deepcopy(stage1['side_effects']), 'no_side_effects': copy.deepcopy(stage1['no_side_effects'])}
    used = {it['drug'] for g in compact.GROUPS for it in merged[g]}
    merged['drugs'] = {h: d for h, d in drugs.items() if h in used}
    for g in ('side_effects', 'no_side_effects'):
        if json.dumps(merged[g], sort_keys=True) != json.dumps(stage1[g], sort_keys=True):
            raise AssertionError(f'{stage1["record_id"]}: {g} changed')
        for it in stage1[g]:
            if merged['drugs'][it['drug']] != stage1['drugs'][it['drug']]:
                raise AssertionError(f'{stage1["record_id"]}: drug {it["drug"]} of {g} changed')
    provenance = {'record_id': stage1['record_id'], 'candidate': VERSION, 'merge': MERGE_VERSION, 'run': run_id, 'stage1_run': stage1.get('run'),
                  'purpose_pass_status': evaluation['status'], 'anomaly': evaluation['anomaly'], 'handle_map': handle_map,
                  'new_drugs': evaluation['new_drugs'],
                  'accepted': [{'answer': a['item'], 'merged_index': i, 'indication': a['indication'], 'subject_label': a['subject_label'],
                                'ev': a['ev'], 'notes': a['notes']} for i, a in enumerate(evaluation['accepted'])],
                  'rejected': evaluation['rejected'], 'stage1_purposes': stage1['purposes'],
                  'stage1_drugs_dropped': {h: d for h, d in stage1['drugs'].items() if h not in merged['drugs']},
                  'stage1_record_sha256': hashlib.sha256(json.dumps(stage1, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                  'stage1_record': copy.deepcopy(stage1)}
    return merged, provenance


def validate_merged(merged, inp):
    """The merged record must pass the stage-1 spec's own compact acceptance check (canonical wire, compact-contract-2.1.0)."""
    spec = compact.get_spec(BASE_SPEC)
    return compact.validate_compact(merged, inp, canonical_wire=spec.canonical_wire, contract=spec.contract)


def _jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _unique(rows, what):
    dup = sorted(k for k, n in Counter(r.get('record_id') for r in rows).items() if n > 1)
    if dup:
        raise SystemExit(f'duplicate record ids in {what}: {dup[:5]}')
    return {r['record_id']: r for r in rows}


def _load(inputs_path, stage1_path):
    inputs = _unique(_jl(inputs_path), 'inputs')
    stage1 = _unique(_jl(stage1_path), 'stage-1 export')
    problems = [f'{rid}: no input' for rid in stage1 if rid not in inputs]
    problems += [f'{rid}: {p}' for rid, rec in stage1.items() if rid in inputs for p in core_export.validate(rec, inputs[rid]['fields'])]
    if problems:
        raise SystemExit(f'invalid stage-1 export ({len(problems)} problems): {problems[:5]}')
    return {rid: compact.ExtractionInput(r['record_id'], r['fields'], r['kind']) for rid, r in inputs.items()}, stage1


def _write_once(inputs, outputs):
    resolved = [Path(p).resolve() for p in inputs + outputs]
    if len(set(resolved)) != len(resolved):
        raise SystemExit('input and output paths must all differ')
    for p in outputs:
        if Path(p).exists():
            raise SystemExit(f'{p} exists; outputs are written once')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('fingerprint')
    b = sub.add_parser('bodies')
    b.add_argument('--inputs', required=True)
    b.add_argument('--stage1', required=True)
    b.add_argument('--out', required=True)
    m = sub.add_parser('merge')
    for name in ('--inputs', '--stage1', '--responses', '--run', '--out', '--provenance', '--summary'):
        m.add_argument(name, required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'fingerprint':
        print(json.dumps(fingerprint(), indent=1))
        return
    if a.cmd == 'bodies':
        _write_once([a.inputs, a.stage1], [a.out])
        inputs, stage1 = _load(a.inputs, a.stage1)
        lines = []
        for rid, rec in stage1.items():
            body = request_body(inputs[rid], rec)
            lines.append({'record_id': rid, 'custom_id': f'{rid}#purpose@1', 'candidate': VERSION, 'body_sha256': sha256_text(compact_json(body)),
                          'cap_usd': round(request_cap_usd(MODEL, body), 6), 'body': body})
        Path(a.out).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in lines))
        print(json.dumps({'bodies': len(lines), 'cap_usd_sum': round(sum(x['cap_usd'] for x in lines), 6), **fingerprint()}))
        return
    _write_once([a.inputs, a.stage1, a.responses], [a.out, a.provenance, a.summary])
    inputs, stage1 = _load(a.inputs, a.stage1)
    raw = _jl(a.responses)
    responses = {}
    for r in raw:
        rid = r.get('record_id') or str(r.get('custom_id', '')).split('#', 1)[0]
        if rid in responses:
            raise SystemExit(f'duplicate response for {rid}')
        if rid not in stage1:
            raise SystemExit(f'response for {rid} without a stage-1 record')
        responses[rid] = r['response']
    merged, prov = [], []
    for rid, rec in stage1.items():
        if rid in responses:
            ev = evaluate(responses[rid], inputs[rid], rec)
        else:
            ev = {'record_id': rid, 'status': 'failed', 'anomaly': {'kind': 'no_response'}, 'new_drugs': [], 'accepted': [], 'rejected': []}
        mrec, p = merge(rec, ev, a.run)
        try:
            validate_merged(mrec, inputs[rid])
        except (ValueError, ExtractionAnomaly) as error:
            raise SystemExit(f'{rid}: merged record fails the {BASE_SPEC} compact contract, nothing written: {error}')
        merged.append(mrec)
        prov.append(p)
    Path(a.out).write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in merged))
    Path(a.provenance).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in prov))
    summary = {**fingerprint(), 'run': a.run, 'records': len(merged), 'responses': len(responses),
               'status': dict(Counter(p['purpose_pass_status'] for p in prov)),
               'anomalies': dict(Counter((p['anomaly'] or {}).get('kind') for p in prov if p['anomaly'])),
               'purposes_stage1': sum(len(p['stage1_purposes']) for p in prov), 'purposes_merged': sum(len(r['purposes']) for r in merged),
               'rejected_items': sum(len(p['rejected']) for p in prov),
               'rejected_reasons': dict(Counter(x for p in prov for r in p['rejected'] for x in r['reasons'])),
               'new_drugs_added': sum(1 for p in prov for h in p['handle_map'].values() if h not in stage1[p['record_id']]['drugs']),
               'accepted_notes': dict(Counter(n for p in prov for a_ in p['accepted'] for n in a_['notes'])),
               'non_P_identical': True, 'api_calls_here': 0,
               'sha256': {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in
                          (('inputs', a.inputs), ('stage1', a.stage1), ('responses', a.responses), ('merged', a.out), ('provenance', a.provenance))}}
    Path(a.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('records', 'status', 'purposes_stage1', 'purposes_merged', 'rejected_items', 'rejected_reasons')}))


check_strict(SCHEMA)

if __name__ == '__main__':
    sys.exit(main())
