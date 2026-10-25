"""Opt-in OFFLINE derived core export, core-export-1.1.0 (no API call, no gold, no change to any extraction spec).

  python -m weightloss.extraction.core_export --export EXPORT.jsonl --inputs INPUTS.jsonl --out CORE.jsonl --summary SUMMARY.json

Reads a validated compact export and the extraction inputs (source fields) and writes one derived record per exported
record. Nothing is rewritten: each derived record keeps the original prediction ("original"), each derived relation keeps
its item verbatim ("original_item") and points back to it (group, index), and its evidence is exactly the item's own
spans rendered from the source (no neighbouring sentence is appended, no field is filled in). Invalid input fails closed:
duplicate record ids, unknown drug handles, missing core fields, or empty, malformed or out-of-range evidence spans stop
the run before anything is written.

Derivations (narrow and auditable):
1. Quantity-limited generic denials ("almost no side effects", "nearly zero", "little to no", "very few", "minimal")
   become relation "limited_ae_statement" with occurrence meaning "limited" (neither an absence nor a definite
   occurrence); they are never no-AE negative-control candidates. The kind changes only when the qualifier is anchored:
   in a no-AE claim's scope_text, inside a generic effect, or right before the generic effect's own occurrence in its
   quotes. A qualifier elsewhere in a broad quote (another clause, person or drug) only raises a review flag.
   A generic not_occurred side effect ("no major side effects") is a no-AE claim, as in the frozen scorer.
2. Occurrence role of named drugs: a drug surface whose every occurrence in the source sits inside a narrow condition
   label ("Ozempic face", "Ozempic-face", "OZEMPIC FACE!", "Ozempic butt/teeth/feet/babies/burps") is not evidence of
   taking that drug: its identity is held (unresolved) and its relations need source review; no other drug is ever
   assigned. A surface not found in the source is held too (fail closed). One ordinary mention keeps the drug.
3. Review flags (triage before any analytic use; they never change content):
   purpose_reason_not_in_quotes     no content word of the indication appears in the purpose's own quotes;
   denial_exception_outside_scope   an exception marker ("except", "other than", "aside from", ", only ...") stands in
                                    the same sentence as a denial while its scope carries no exception word;
   effect_negator_outside_value     an occurred/uncertain effect whose every occurrence in its own quotes directly
                                    follows a negator in the same clause ("without being able to keep any water down");
   quantity_qualifier_unanchored    a quantity qualifier appears in the item's quotes but is not anchored to it.
   quantity_qualifier_contradictory a negated quantity word ("not had minimal side effects, they have been severe"):
                                    the kind is not changed.
   1.1.0 (full-batch-p7c1-luna-low-20261001-01a0f4c2) adds the following. Like every flag above, they hold a relation
   for review and never change its content:
   - "almost 0" counts as a near-zero quantity, and a denial hedged by "to speak of" / "worth mentioning" right after
     its side-effect noun is a limited statement (val200 n184, n109).
   - The exception marker also matches the typo "other that" (val200 n165).
   - denial_symptom_adjacent: the sentence right after a denial's quotes names a symptom that the denial's scope does
     not carry ("Havent had side effects yet either. A little constipation, ..."; val200 n57).
   - off_drug_recurrence: an occurred or uncertain effect whose own quotes say the drug is not being taken ("off the
     med", "since I stopped", "7 weeks without"), or say the effect is back next to an on-drug contrast ("are back ...
     gone while being on it"): a possible off-drug return or lack of effect (val200 n23, n134). A symptom that came back
     on re-injection, or that made the writer skip or lower a dose, is not flagged.
   - combination_attribution: the same occurred or uncertain effect, with overlapping evidence, attributed to two or
     more drugs of the record (a regimen change the text does not split; val200 n120).
Eligibility is explicit and conservative: precheck_clear means only that no heuristic flag fired; source_review_status
is "unreviewed" and analytic_eligible / negative_control_eligible are false for every relation until a separately
recorded source review accepts them. negative_control_candidate (absolute no-AE claim: no scope, dose or duration,
precheck clear) only nominates a claim for that review. No relation here carries population-analysis acceptance; an
absent purpose means unknown, never "not taken for X".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from . import compact
from .path_repair import GENERIC_EFFECT

VERSION = 'core-export-1.1.0'
SE_NOUN = r"side[\s-]?e?ffects?|side[\s-]?affects?|sides|symptoms?|issues|problems|reactions?"
NEAR_ZERO = (r"(?:almost|nearly|virtually|practically|basically|essentially)\s+(?:no|zero|0|none(?:\s+of\s+the)?)"
             r"|little\s+(?:to|or)\s+no|next\s+to\s+no|(?:barely|hardly|scarcely)\s+any")
FEW = r"(?:very\s+)?(?:few|minimal|minimum)|not\s+(?:many|much)|very\s+little"
# a FEW-type word must govern the side-effect noun ("for the first few weeks" is not a quantity of side effects)
LIMITED = re.compile(rf"\b(?:(?:{NEAR_ZERO})\b|(?:{FEW})\s+(?:[\w'’-]+\s+){{0,2}}?(?:{SE_NOUN})\b)", re.I)
HEDGE_BEFORE = re.compile(rf"\b(?:{NEAR_ZERO}|{FEW})\s+(?:[\w'’-]+\s+){{0,2}}$", re.I)
# a negated quantity word ("I have not had minimal side effects, they have been severe", "not almost no side effects") is
# a contradiction, not a limit; "not many/much side effects" is itself a finite quantity and is not negated here
NEGATED_QUANTITY = re.compile(rf"(?:\bnot|\bnever|n['’]t)\b(?:\s+[\w'’-]+){{0,2}}?\s+(?:(?:very\s+)?(?:few|minimal|minimum)|{NEAR_ZERO})\b",
                              re.I)
ABSOLUTE_GENERIC = re.compile(rf"^(?:any\s+)?(?:{SE_NOUN})$", re.I)
CONDITION_HEADS = ('face', 'butt', 'booty', 'teeth', 'feet', 'baby', 'babies', 'burp', 'burps')
CONDITION_LABEL = re.compile(r"^[\s\-‐‑–—_'’\"“”]*(?:" + '|'.join(CONDITION_HEADS) + r")\b", re.I)
CONDITION_SURFACE = re.compile(r"\b(?:" + '|'.join(CONDITION_HEADS) + r")\s*$", re.I)
EXCEPTION = re.compile(r"\b(?:except(?:\s+for)?|other\s+th[ae]n|other\s+that|aside\s+from|apart\s+from|besides|but\s+for|save\s+for)\b"
                       r"|,\s*(?:just|only)\b", re.I)
# 1.1.0: a denial hedged right after its side-effect noun ("no side effects ever to speak of") is a limited statement
SPEAK_OF = re.compile(rf"\b(?:{SE_NOUN})\b(?:\s+[\w'’-]+){{0,3}}?\s+(?:to\s+speak\s+of|worth\s+(?:mentioning|noting|talking\s+about))\b", re.I)
SPEAK_OF_SCOPE = re.compile(r"\b(?:to\s+speak\s+of|worth\s+(?:mentioning|noting|talking\s+about))\b", re.I)
# 1.1.0 review routing: a symptom named in the sentence right after a denial; an effect tied to stopping/missing/reducing
SYMPTOM = re.compile(r"\b(?:nause\w*|vomit\w*|throw\w*\s+up|puk\w*|constipat\w*|diarrh\w*|headache\w*|migraine\w*|fatigue\w*|tired\w*|"
                     r"exhaust\w*|dizz\w*|light[\s-]?headed\w*|heartburn|reflux|burp\w*|sulfur\w*|bloat\w*|gas(?:sy)?|cramp\w*|"
                     r"stomach\s*(?:ache|pain|issues?|upset)|ache\w*|pain\w*|insomnia|anxi\w*|depress\w*|mood\w*|hair\s+(?:loss|falling|thinning)|"
                     r"rash\w*|itch\w*|bruis\w*|sore\w*|chills|fever\w*|palpitat\w*|hypoglyc\w*|dehydrat\w*|dry\s+mouth|"
                     r"injection\s+site|indigestion|upset\s+stomach)\b", re.I)
# an effect present while the drug is not being taken: an absence cue, or a return cue next to an on-drug contrast
ABSENCE = re.compile(r"\b(?:off\s+(?:of\s+)?(?:it|them|the\s+(?:shot|med\w*|drug|injections?)|my\s+(?:shot|med\w*|dose))"
                     r"|(?:since|after|when|once)\s+(?:i\s+|we\s+|she\s+|he\s+)?(?:stopp?(?:ed|ing)|quit(?:ting)?|came\s+off|went\s+off|got\s+off|ran\s+out)"
                     r"|(?:stopp?(?:ed|ing)|quit(?:ting)?)\s+(?:taking\s+|using\s+)?(?:it|them|the\s+(?:shot|med\w*|drug|injections?)|my\s+(?:shot|med\w*|dose))"
                     r"|(?:been|weeks?|days?|months?)\s+without(?:\s+(?:it|them|the\s+(?:shot|med\w*|drug|injections?)|my\s+(?:shot|med\w*|dose)))?(?=\s*(?:[.!?,;)]|$))"
                     r"|without\s+(?:it|them|the\s+(?:shot|med\w*|drug))\b|ran\s+out|cold\s+turkey"
                     r"|(?:haven['’]?t|have\s+not|hasn['’]?t|not)\s+(?:been\s+)?tak(?:en|ing)\s+(?:it|them|the|my))\b", re.I)
RETURN = re.compile(r"\b(?:(?:are|is|was|were|been|came|come|coming|comes)\s+back|(?:['’]re|['’]s)\s+back|back\s+again|returned|returning|recurr\w*)\b", re.I)
ON_CONTRAST = re.compile(r"\b(?:gone|went\s+away|disappeared|go\s+away|stopped|resolved|improved|better)\s+(?:while|when)\s+(?:i\s+was\s+|being\s+)?"
                         r"on\s+(?:it|them|the|my)\b", re.I)
SCOPE_EXCEPTION = re.compile(r"\b(?:except|excepting|other|aside|apart|besides|only|just|save|but)\b", re.I)
SENTENCE_END = re.compile(r"[.!?…]+(?=\s|$)|\n")
CLAUSE_BREAK = re.compile(r"[,;:()\[\]–—]")
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
NEGATORS = frozenset({'without', 'not', 'no', 'never', 'cannot', 'unable', 'nor', 'none'})
NEG_BRIDGE = frozenset({'really', 'even', 'ever', 'being', 'getting', 'having', 'any', 'much', 'a', 'the'})
STOP = frozenset('the a an my your his her their our its it to for of and or in on with from that this these those be is was were are been '
                 'being i me you he she they we us them more less some any get got help helps helped make makes keep so as at by up down '
                 'out about just also very really all'.split())
REVIEW_FLAGS = ('drug_identity_held', 'purpose_reason_not_in_quotes', 'denial_exception_outside_scope', 'effect_negator_outside_value',
                'quantity_qualifier_unanchored', 'quantity_qualifier_contradictory', 'denial_symptom_adjacent', 'off_drug_recurrence',
                'combination_attribution')
RULES = {'version': VERSION, 'limited': LIMITED.pattern, 'hedge_before': HEDGE_BEFORE.pattern, 'negated_quantity': NEGATED_QUANTITY.pattern,
         'absolute_generic': ABSOLUTE_GENERIC.pattern,
         'condition_heads': CONDITION_HEADS, 'condition_label': CONDITION_LABEL.pattern, 'exception': EXCEPTION.pattern,
         'scope_exception': SCOPE_EXCEPTION.pattern, 'sentence_end': SENTENCE_END.pattern, 'clause_break': CLAUSE_BREAK.pattern,
         'negators': sorted(NEGATORS), 'neg_bridge': sorted(NEG_BRIDGE), 'stop': sorted(STOP), 'generic_effect': GENERIC_EFFECT.pattern,
         'stem': 5, 'review_flags': REVIEW_FLAGS, 'speak_of': SPEAK_OF.pattern, 'speak_of_scope': SPEAK_OF_SCOPE.pattern,
         'symptom': SYMPTOM.pattern, 'absence': ABSENCE.pattern, 'return': RETURN.pattern,
         'on_contrast': ON_CONTRAST.pattern}
REQUIRED = {'purposes': ('drug', 'indication'), 'side_effects': ('drug', 'effect', 'occurrence'), 'no_side_effects': ('drug',)}


def rules_sha256():
    return hashlib.sha256(json.dumps(RULES, sort_keys=True).encode()).hexdigest()


def bounded(value):
    """Case-insensitive literal of value on token boundaries (a boundary is only required next to a letter or digit)."""
    pre = r'(?<![A-Za-z0-9])' if value[:1].isalnum() else ''
    post = r'(?![A-Za-z0-9])' if value[-1:].isalnum() else ''
    return re.compile(pre + re.escape(value) + post, re.I)


def validate(record, fields):
    """Problems that make a record unusable for a derived export (empty list when valid)."""
    out = []
    drugs = record.get('drugs')
    if not isinstance(drugs, dict):
        return ['drugs is not a mapping']
    for h, d in drugs.items():
        if not isinstance(d, dict) or not d.get('surface') or d.get('ref') not in ('name', 'class', 'unspecified'):
            out.append(f'drugs.{h}: surface/ref')
    for group in compact.GROUPS:
        items = record.get(group)
        if not isinstance(items, list):
            out.append(f'{group} is not a list')
            continue
        for i, item in enumerate(items):
            where = f'{group}[{i}]'
            missing = [f for f in REQUIRED[group] if not item.get(f)]
            if missing:
                out.append(f'{where}: missing {missing}')
            if item.get('drug') not in drugs:
                out.append(f'{where}: unknown drug handle {item.get("drug")!r}')
            ev = item.get('ev')
            if not isinstance(ev, list) or not ev:
                out.append(f'{where}: no evidence')
                continue
            for s in ev:
                try:
                    if not compact.render_span(s, fields).strip():
                        out.append(f'{where}: empty evidence {s}')
                except (ValueError, AttributeError):
                    out.append(f'{where}: invalid evidence span {s!r}')
    return out


def _spans(item):
    out = []
    for s in item['ev']:
        name, _, offsets = s.rpartition(':')
        start, end = (int(x) for x in offsets.split('-'))
        out.append((name, start, end))
    return out


def drug_identity(drug, fields):
    """Occurrence roles of a named drug surface: every hit inside a condition label, or no hit at all -> held."""
    if drug['ref'] != 'name':
        return {'status': 'as_extracted', 'mentions': None, 'condition_labels': None}
    pattern = bounded(drug['surface'])
    hits = [(name, m.end()) for name, text in fields.items() for m in pattern.finditer(text)]
    if CONDITION_SURFACE.search(drug['surface']):
        roles = ['condition_label'] * len(hits)
    else:
        roles = ['condition_label' if CONDITION_LABEL.match(fields[name][end:end + 24]) else 'mention' for name, end in hits]
    status = ('held_surface_not_found' if not hits else 'held_condition_label_only' if all(r == 'condition_label' for r in roles)
              else 'as_extracted')
    return {'status': status, 'mentions': roles.count('mention'), 'condition_labels': roles.count('condition_label')}


def _tokens(s):
    return [t for t in re.findall(r"[a-z0-9]+", (s or '').lower().replace('’', "'")) if t not in STOP and len(t) > 1]


def _stem(t):
    return t[:5] if len(t) >= 5 else t


def reason_in_quotes(indication, quotes):
    """A content word of the indication (5-letter stems) appears in the purpose's own quotes."""
    want = {_stem(t) for t in _tokens(indication)}
    return not want or bool(want & {_stem(t) for q in quotes for t in _tokens(q)})


def _sentence(text, start, end):
    s = 0
    for m in SENTENCE_END.finditer(text, 0, start):
        s = m.end()
    m = SENTENCE_END.search(text, max(start, end - 1))
    return s, (m.start() if m else len(text))


def exception_outside_scope(fields, spans, scope):
    """An exception marker in the sentence(s) of the denial's quotes while its scope has no exception word."""
    if scope and SCOPE_EXCEPTION.search(scope):
        return False
    for name, start, end in spans:
        text = fields[name]
        s, e = _sentence(text, start, end)
        if EXCEPTION.search(text[s:max(e, end)]):
            return True
    return False


def symptom_after(fields, spans, scope):
    """1.1.0: a symptom named in the sentence right after the denial's quotes (same field) and not carried by its scope."""
    carried = {m.group(0).lower() for m in SYMPTOM.finditer(scope or '')}
    for name, start, end in spans:
        text = fields[name]
        _, e = _sentence(text, start, end)
        nxt = SENTENCE_END.search(text, e + 1)
        following = text[e:nxt.start() if nxt else len(text)]
        if any(m.group(0).lower() not in carried for m in SYMPTOM.finditer(following)):
            return True
    return False


def _norm_effect(value):
    return ' '.join(re.findall(r"[a-z0-9]+", (value or '').lower()))


def combination_items(record):
    """1.1.0: indices of occurred/uncertain side effects that share an effect and overlapping evidence with an item bound
    to another drug of the record (a regimen change the text does not split)."""
    items = [(i, item, _norm_effect(item['effect']), _spans(item)) for i, item in enumerate(record['side_effects'])
             if item.get('occurrence') in ('occurred', 'uncertain')]
    out = set()
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            (i, x, ex, sx), (j, y, ey, sy) = items[a], items[b]
            if x['drug'] == y['drug'] or not ex or ex != ey:
                continue
            if any(n1 == n2 and s1 < e2 and s2 < e1 for n1, s1, e1 in sx for n2, s2, e2 in sy):
                out.update((i, j))
    return out


def _occurrences(fields, spans, value):
    pattern = bounded(value)
    return [(name, start, m) for name, start, end in spans for m in pattern.finditer(fields[name], start, end)]


def quantity_anchor(fields, spans, value):
    """How a quantity qualifier governs the value: 'contradictory' (negated: "not had minimal side effects"), 'limited'
    (right before an occurrence of the value in the item's own quotes: "almost no side effects") or None."""
    if NEGATED_QUANTITY.search(value):
        return 'contradictory'
    if LIMITED.search(value):
        return 'limited'
    found = None
    for name, start, m in _occurrences(fields, spans, value):
        window = fields[name][max(start, m.start() - 60):m.start()]
        if HEDGE_BEFORE.search(window):
            if NEGATED_QUANTITY.search(window + m.group(0)):
                return 'contradictory'
            found = 'limited'
    return found


def negator_before_value(fields, spans, value):
    """Every occurrence of the value in the item's own quotes directly follows a negator in the same clause ("without X",
    "not X", "never really X"); review triage only, never a rewrite."""
    found = []
    for name, start, m in _occurrences(fields, spans, value):
        text = fields[name]
        s, _ = _sentence(text, m.start(), m.end())
        window = text[max(s, m.start() - 60):m.start()]
        cut = max((x.end() for x in CLAUSE_BREAK.finditer(window)), default=0)
        last = [w.lower() for w in WORD.findall(window[cut:])[-2:]]
        neg = lambda w: w in NEGATORS or w.endswith(("n't", "n’t"))
        found.append(bool(last) and (neg(last[-1]) or (len(last) == 2 and neg(last[0]) and last[1] in NEG_BRIDGE)))
    return bool(found) and all(found)


def derive_record(record, fields):
    """One derived core record: a pure function of a valid exported record and its source fields."""
    problems = validate(record, fields)
    if problems:
        raise ValueError(f"{record.get('record_id')}: {problems[:5]}")
    drugs = {h: {**d, 'identity': drug_identity(d, fields)} for h, d in record['drugs'].items()}
    combined = combination_items(record)
    relations = []
    for group in compact.GROUPS:
        for i, item in enumerate(record[group]):
            spans = _spans(item)
            quotes = [compact.render_span(s, fields) for s in item['ev']]
            drug = drugs[item['drug']]
            rel = {'id': f'{group}[{i}]', 'source': {'group': group, 'index': i}, 'drug': item['drug'], 'drug_surface': drug['surface'],
                   'drug_ref': drug['ref'], 'drug_identity': drug['identity']['status'], 'subject': item.get('subject'),
                   'subject_label': item.get('subject_label'), 'statement_kind': item.get('statement_kind'), 'use_status': item.get('use_status'),
                   'evidence': [{'span': s, 'quote': q} for s, q in zip(item['ev'], quotes)], 'review_flags': [], 'notes': []}
            quote_hedge = any(LIMITED.search(q) for q in quotes)
            speak_of = any(SPEAK_OF.search(q) for q in quotes)
            if group == 'purposes':
                rel.update(type='purpose', indication=item['indication'], polarity=item.get('polarity'))
                if not reason_in_quotes(item['indication'], quotes):
                    rel['review_flags'].append('purpose_reason_not_in_quotes')
            elif group == 'side_effects' and (GENERIC_EFFECT.match(item['effect'].strip()) or LIMITED.search(item['effect'])):
                anchor = quantity_anchor(fields, spans, item['effect'])
                anchored = anchor == 'limited'
                rel['notes'].append('generic_side_effect')
                if anchor == 'contradictory':
                    rel['review_flags'].append('quantity_qualifier_contradictory')
                if anchored:
                    rel.update(type='limited_ae_statement', occurrence_meaning='limited', limited_anchor='effect', statement_text=item['effect'],
                               absolute=False)
                elif item['occurrence'] == 'not_occurred' and speak_of and anchor is None:  # 1.1.0: "no side effects to speak of"
                    rel.update(type='limited_ae_statement', occurrence_meaning='limited', limited_anchor='quote_hedge', statement_text=item['effect'],
                               absolute=False)
                elif item['occurrence'] == 'not_occurred':
                    rel.update(type='no_ae_claim', statement_text=item['effect'], scope_text=None,
                               absolute=bool(ABSOLUTE_GENERIC.match(item['effect'].strip())))
                else:
                    rel.update(type='adverse_event', effect=item['effect'], occurrence=item['occurrence'], attribution=item.get('attribution'))
                    if item['occurrence'] in ('occurred', 'uncertain') and negator_before_value(fields, spans, item['effect']):
                        rel['review_flags'].append('effect_negator_outside_value')
                if quote_hedge and anchor is None:
                    rel['review_flags'].append('quantity_qualifier_unanchored')
                if rel['type'] != 'adverse_event' and exception_outside_scope(fields, spans, item['effect']):
                    rel['review_flags'].append('denial_exception_outside_scope')
                if rel['type'] != 'adverse_event' and symptom_after(fields, spans, item['effect']):
                    rel['review_flags'].append('denial_symptom_adjacent')
            elif group == 'side_effects':
                rel.update(type='adverse_event', effect=item['effect'], occurrence=item['occurrence'], attribution=item.get('attribution'))
                if item['occurrence'] in ('occurred', 'uncertain') and negator_before_value(fields, spans, item['effect']):
                    rel['review_flags'].append('effect_negator_outside_value')
            else:
                scope = item.get('scope_text')
                if scope and NEGATED_QUANTITY.search(scope):
                    rel.update(type='no_ae_claim', scope_text=scope, absolute=False)
                    rel['review_flags'].append('quantity_qualifier_contradictory')
                elif scope and (LIMITED.search(scope) or SPEAK_OF_SCOPE.search(scope)):
                    rel.update(type='limited_ae_statement', occurrence_meaning='limited', limited_anchor='scope_text', scope_text=scope,
                               absolute=False)
                elif speak_of:  # 1.1.0: "no side effects ever to speak of" with the hedge only in the quote
                    rel.update(type='limited_ae_statement', occurrence_meaning='limited', limited_anchor='quote_hedge', scope_text=scope,
                               absolute=False)
                else:
                    rel.update(type='no_ae_claim', scope_text=scope, absolute=scope is None)
                    if quote_hedge:
                        rel['review_flags'].append('quantity_qualifier_unanchored')
                if exception_outside_scope(fields, spans, scope):
                    rel['review_flags'].append('denial_exception_outside_scope')
                if symptom_after(fields, spans, scope):
                    rel['review_flags'].append('denial_symptom_adjacent')
            if group == 'side_effects' and rel['type'] == 'adverse_event' and item.get('occurrence') in ('occurred', 'uncertain'):  # 1.1.0 review routing
                joined = ' '.join(quotes)
                if ABSENCE.search(joined) or (RETURN.search(joined) and ON_CONTRAST.search(joined)):
                    rel['review_flags'].append('off_drug_recurrence')
                if i in combined:
                    rel['review_flags'].append('combination_attribution')
            if drug['identity']['status'] != 'as_extracted':
                rel['review_flags'].insert(0, 'drug_identity_held')
            rel['precheck_clear'] = not rel['review_flags']  # heuristic triage only
            rel['source_review_status'] = 'unreviewed'
            rel['analytic_eligible'] = False  # acceptance: only a separately recorded source review can grant it
            rel['negative_control_candidate'] = (rel['type'] == 'no_ae_claim' and rel['absolute'] and rel['precheck_clear']
                                                 and not item.get('dose') and not item.get('duration'))
            rel['negative_control_eligible'] = False  # acceptance as a negative control: source review only
            rel['original_item'] = item
            relations.append(rel)
    return {'record_id': record['record_id'], 'run': record.get('run'), 'core_export': VERSION, 'drugs': drugs, 'relations': relations,
            'purpose_absent_means': 'unknown', 'original': record}


def _jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def summarize(derived, missing):
    rels = [r for d in derived for r in d['relations']]
    count = lambda pred: sum(1 for r in rels if pred(r))
    return {'records': len(derived), 'records_without_export': missing,
            'items_in': {g: sum(len(d['original'][g]) for d in derived) for g in compact.GROUPS}, 'relations': len(rels),
            'by_type': {t: count(lambda r, t=t: r['type'] == t) for t in ('purpose', 'adverse_event', 'no_ae_claim', 'limited_ae_statement')},
            'precheck_clear': count(lambda r: r['precheck_clear']), 'review_required': count(lambda r: not r['precheck_clear']),
            'source_accepted': 0, 'analytic_eligible': count(lambda r: r['analytic_eligible']),
            'negative_control_eligible': count(lambda r: r['negative_control_eligible']),
            'review_flags': {f: count(lambda r, f=f: f in r['review_flags']) for f in REVIEW_FLAGS},
            'limited_anchor': {s: count(lambda r, s=s: r.get('limited_anchor') == s) for s in ('scope_text', 'effect', 'quote_hedge')},
            'no_ae_absolute': count(lambda r: r['type'] == 'no_ae_claim' and r['absolute']),
            'negative_control_candidates': count(lambda r: r['negative_control_candidate']),
            'held_drugs': sorted(f"{d['record_id']}:{h}:{v['surface']}:{v['identity']['status']}" for d in derived for h, v in d['drugs'].items()
                                 if v['identity']['status'] != 'as_extracted')}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--export', required=True)
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--summary', required=True)
    a = ap.parse_args(argv)
    out, summary = Path(a.out), Path(a.summary)
    if len({out.resolve(), summary.resolve(), Path(a.export).resolve(), Path(a.inputs).resolve()}) < 4:
        raise SystemExit('--out, --summary, --export and --inputs must be four different paths')
    for p in (out, summary):
        if p.exists():
            raise SystemExit(f'{p} exists; derived exports are written once')
    input_rows, records = _jl(a.inputs), _jl(a.export)
    for name, rows in (('inputs', input_rows), ('export', records)):
        dup = sorted(k for k, n in Counter(r.get('record_id') for r in rows).items() if n > 1)
        if dup:
            raise SystemExit(f'duplicate record ids in {name}: {dup[:5]}')
    inputs = {r['record_id']: r for r in input_rows}
    problems = [f"{r.get('record_id')}: no input" for r in records if r.get('record_id') not in inputs]
    problems += [f"{r['record_id']}: {p}" for r in records if r.get('record_id') in inputs for p in validate(r, inputs[r['record_id']]['fields'])]
    if problems:
        raise SystemExit(f'invalid export ({len(problems)} problems), nothing written: {problems[:5]}')
    derived = [derive_record(r, inputs[r['record_id']]['fields']) for r in records]
    exported = {r['record_id'] for r in records}
    out.write_text(''.join(json.dumps(d, ensure_ascii=False, sort_keys=True) + '\n' for d in derived))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta = {'core_export': VERSION, 'rules_sha256': rules_sha256(), 'module_sha256': sha(__file__),
            'export': {'path': a.export, 'sha256': sha(a.export)}, 'inputs': {'path': a.inputs, 'sha256': sha(a.inputs)},
            'out': {'path': str(out), 'sha256': sha(out)}, 'api_calls': 0, 'gold_used': False,
            'acceptance': 'none: precheck_clear is a heuristic triage result; analytic_eligible stays false until a recorded source review',
            **summarize(derived, sorted(r for r in inputs if r not in exported))}
    summary.write_text(json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: meta[k] for k in ('records', 'relations', 'by_type', 'precheck_clear', 'review_flags', 'held_drugs')}, ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main())
