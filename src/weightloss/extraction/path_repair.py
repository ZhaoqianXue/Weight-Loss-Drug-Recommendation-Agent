"""Path-only correction ``literal-select-1.3.0`` (2026-09-30 engineering review) for runs under contract 2.0.0 (compact-p7t3).

**When it runs.** The run's retry policy is provider-only, so a reply that fails local validation is never re-extracted.
This step corrects that FIRST reply at the failing paths only, and nothing else.

**Tasks.** Each task is one field. The strict output is an enum of exact candidates or null, and each request is sent
once and never retried.
- **quote** (evidence_not_found or evidence_ambiguous): exact source passages. These are single sentences, or two
  consecutive sentences, that occur exactly once, keep decimals and negations whole, and share content words with the
  rejected quote. An ambiguous quote gets the unique passages that contain it. The instruction is the literal-select-1.2
  "near copy" instruction. Quotes are corrected first; the other tasks are then computed on the corrected reply.
- **surface** (effect or indication, surface_not_found): token-bounded n-grams of the item's own verified quotes, with
  the 4-letter-prefix inflection fallback.
  - Only when these give nothing, the one sentence before or after each quote in the same field is used (anchored
    antecedent context).
  - A phrase chosen from it appends that exact sentence to the item's quotes as recorded added evidence.
- **drug** (drug surface, surface_not_found): token-bounded n-grams of the input that share content words with the
  rejected surface, ranked by overlap and then by length.
- **field** (an as-written field, field_not_found): token-bounded n-grams of the item's own quotes, or null. Null
  removes the value, which is recorded.
- **contract** (event fields filled on a not_occurred or not_asserted effect, schema_invalid at stage "contract"):
  deterministic, with no request. The offending event fields of that item are emptied, as the p7 contract requires, and
  the removed values are recorded in the change log. Occurrence, polarity, the effect, subject and evidence are never
  touched.

**Preservation gate.** The corrected canonical wire must equal the first reply's localized wire everywhere except the
task paths and the recorded appended quotes. That covers every drug, item, field, evidence quote and multiplicity.
Anything else rejects the correction. The record is then fully re-evaluated under the run's spec and contract, and
export-final re-checks it.

**Fails closed.** Structure or cross-field schema failures, reference failures, too many candidates, a null answer where
a value is required, refusal, truncation, HTTP or transport failures, budget, a gate violation, or a failed
re-evaluation all leave the record unresolved.

**literal-select-1.3.1** is 1.3.0 with two changes to field tasks and nothing else. It runs as a successor ledger over
the records a 1.3.0 ledger left unresolved (its manifest names the predecessor and freezes that scope).
- **Over-cap pools are ranked and packed.** An exact anchored pool larger than the existing caps is ranked by
  - shared digits, then shared content words (or pronoun forms) with the rejected value;
  - then closeness in token count;
  - then first occurrence.
  Its top-ranked prefix that fits both caps is offered. The caps are unchanged, and a pool within them is unchanged.
- **Pronoun values keep their forms.** A rejected value with no digit and no content word relates through its exact
  English pronoun and possessive forms, for example "you" to you/your/yours/yourself/yourselves. The source's "your" is
  therefore a candidate, where 1.3.0 had none because stopwords are discarded.

The answer is still one exact candidate or null. Code draws no semantic inference.

**literal-select-1.3.2** replaces 1.3.1 as the successor policy. Root rejected 1.3.1's selections: "your food noise" as a
subject label, and "titrate up to 7.5mg and hit the **12" as an onset. 1.3.2 is 1.3.1 with typed candidate boundaries:
- **Single-pronoun values.** A value that is one pronoun ("you") gets only the exact pronoun/possessive TOKENS of its
  family from the anchored text ("your"), never n-grams that contain them.
- **Temporal fields** (onset, event_duration, duration). A candidate is dropped when it
  - has unbalanced emphasis, quote or bracket marks ("mark** total");
  - starts or ends with a range connector;
  - at every bounded occurrence, starts or ends inside a numeric range ("hit the **12" before " to 14-week", "14-week
    mark" after "12 to ").
  The complete phrase ("12 to 14-week mark") stays selectable.
Other fields keep 1.3.1's rules. 1.3.1 ledgers stay archived as they are and are not part of a 1.3.2 export chain.

**literal-select-1.3.3** is 1.3.2 with two more boundaries on temporal phrases, which must be plain and complete. It
drops candidates that
- contain any emphasis or code mark (* _ ~ `);
- end on a function word (an article, preposition, conjunction, possessive or demonstrative).
Under 1.3.2 the model picked "hit the **12 to 14-week mark** total on": balanced marks, but a dangling "on" and extra
words, because trimming had removed the leading "**" of the complete phrase.

**literal-select-1.4.0** (iteration-20260930-01a0f4c2; compact-contract-2.1.0 runs only, every record, no successor scope).
It runs after localize-1.3.0, which already maps whitespace/case-only field values and splits unmarked joins without a
request. Differences from 1.3.3, all failing closed:
- **Own evidence only.** Surface and field candidates come only from the item's own verified quotes: no adjacent-sentence
  fallback, no appended evidence, no whole-input anchor sentences.
- **Complete phrases everywhere.** The 1.3.3 rule (no emphasis/code mark, no trailing function word) applies to every
  surface, drug and field candidate; temporal fields keep the 1.3.2 range rules and single pronouns their tokens.
- **Quote replacement keeps every word.** A replacement passage must contain every word token of the rejected quote, in
  the same order (case, whitespace and punctuation aside): it may only be a more complete original passage, so a dropped
  qualifier, negation, number, subject or drug name can never pass. Otherwise no passage is offered and the record stays
  unresolved.
- **Drug surfaces from their own evidence.** Candidates are complete n-grams of the verified quotes of the items bound to
  that drug; no sentence elsewhere in the input is used.
- **Fields without an admissible literal.** A non-core as-written field (dose, duration, event_dose, onset,
  event_duration, subject_label) is removed and logged without a request; a scope_text is never removed (its task is not
  nullable, and an empty pool leaves the record unresolved).
- **Qualifiers of negations survive the contract.** When the p7 contract empties severity on a not_occurred effect:
  - a generic effect ("side effects", "symptoms", ...) becomes a no-side-effect claim whose scope_text is the literal
    qualifier its own quote negates ("avoid major side effects" -> "major"), with drug, subject, use and quotes kept;
  - a specific effect keeps its qualifier in the effect when its own quote writes it right before the effect
    ("no severe pain" -> "severe pain");
  - otherwise the record stays unresolved (qualifier_unsupported).
  Other contract removals are unchanged and logged.
- **Re-extraction once** (``reextract_once``): a record that the correction pass leaves unresolved on a lexical or
  contract failure gets one ordinary request of the run's own spec, evaluated exactly like a first attempt, never retried
  and never path-corrected again; its result, ok or not, is final.

**literal-select-1.5.0** (full-batch-p7c1-luna-low-20261001-01a0f4c2) is 1.4.0 plus one rule for field values that
carry a number. A field candidate must state the same numbers as the rejected value, in the same order and compared as
exact decimals with their sign ("0.25" ~ ".25", "two" ~ "2", "10 mg" ~ "10mg"). Each number must also keep its unit when
the rejected value writes one after it (mg, ml, units, days, weeks, ...), so "10 mg daily for 5 days" never admits
"5 mg daily for 10 days".
Otherwise the candidate is not offered. A non-core field left without a candidate is removed without a request, and
its original value is kept in the change log (``field_removed``). A scope_text left without a candidate stays
unresolved. Origin: val200 n120, where 1.4.0 offered the later "5 mg daily" for a rejected "10 mg daily" and the
model chose it, which moved a dose across episodes. Every other 1.4.0 rule is unchanged, including re-extraction once
(``reextract-once-1.5.0``).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation

from . import compact, contract2, localize
from .structured import ExtractionAnomaly, compact_json, final_output_text, parse_json_object, request_cap_usd, sha256_text, validate_model
from .surface_repair import (EDGE, MAX_CANDIDATE_CHARS, MAX_CANDIDATES, QUOTE_INSTRUCTIONS_1_2, _send_once, _words, build_body,
                             candidates, check_strict, now, quote_candidates, schema, sentences)

POLICY = 'literal-select-1.3.0'
POLICY_1_3_1 = 'literal-select-1.3.1'  # successor ledger over 1.3.0's unresolved records (see the module docstring)
POLICY_1_3_2 = 'literal-select-1.3.2'  # 1.3.1 + typed candidate boundaries (pronoun tokens; well-formed temporal phrases)
POLICY_1_3_3 = 'literal-select-1.3.3'  # 1.3.2 + plain, complete temporal phrases (no marks, no trailing function word)
POLICY_1_4 = 'literal-select-1.4.0'  # contract 2.1.0 runs: own evidence only, complete phrases, quote anchors, kept qualifiers
POLICY_1_5 = 'literal-select-1.5.0'  # 1.4.0 + numeric fields keep their numbers and units (see the module docstring)
POLICIES = (POLICY, POLICY_1_3_1, POLICY_1_3_2, POLICY_1_3_3, POLICY_1_4, POLICY_1_5)
SUCCESSORS = (POLICY_1_3_1, POLICY_1_3_2, POLICY_1_3_3, POLICY_1_4, POLICY_1_5)  # policies with the 1.3.1 field-pool changes
TYPED = (POLICY_1_3_2, POLICY_1_3_3, POLICY_1_4, POLICY_1_5)  # policies with typed candidate boundaries
PLAIN = (POLICY_1_3_3, POLICY_1_4, POLICY_1_5)  # plain, complete temporal phrases
OWN_ONLY = (POLICY_1_4, POLICY_1_5)  # contract 2.1.0 policies: own evidence only, kept qualifiers, re-extraction once
SUFFIX = {POLICY: '@1.3', POLICY_1_3_1: '@1.3.1', POLICY_1_3_2: '@1.3.2', POLICY_1_3_3: '@1.3.3', POLICY_1_4: '@1.4',
          POLICY_1_5: '@1.5'}  # request custom_id suffix
CONTRACT_OF = {POLICY: contract2.CONTRACT_2, POLICY_1_3_1: contract2.CONTRACT_2, POLICY_1_3_2: contract2.CONTRACT_2,
               POLICY_1_3_3: contract2.CONTRACT_2, POLICY_1_4: contract2.CONTRACT_2_1, POLICY_1_5: contract2.CONTRACT_2_1}
REEXTRACT = 'reextract-once-1.4.0'
REEXTRACT_OF = {POLICY_1_4: REEXTRACT, POLICY_1_5: 'reextract-once-1.5.0'}
GENERIC_NOUN = r'side[\s-]?e?ffects?|side[\s-]?affects?|sides|symptoms?|issues|problems|reactions?'
QUALIFIERS = r'major|serious|severe|bad|significant|negative|real|big|extreme|noticeable|terrible|horrible|awful|intense'
GENERIC_EFFECT = re.compile(rf'^(?:(?:any|the|other|some)\s+)?(?:(?:{QUALIFIERS})\s+)?(?:{GENERIC_NOUN})$', re.I)
NEG_QUALIFIED = re.compile(rf"\b(?:no|not|n['’]t|without|avoid(?:ed|ing|s)?|never|zero|free of|none of the)\b[^.!?\n]{{0,40}}?"
                           rf"\b(?P<q>{QUALIFIERS})\s+(?:{GENERIC_NOUN})\b", re.I)
FUNCTION_TAIL = frozenset('the a an of on in at to for by with from into onto over under after before until till since around about through '
                          'during within per and or but than as my your his her their our its this that these those'.split())
TEMPORAL_FIELDS = ('onset', 'event_duration', 'duration')
# 1.4.0 quote replacement: words a replacement passage must not add to the rejected quote (polarity, exception, severity,
# quantity, temporal and modality scope); "t" is the negation of "didn't"/"can't" after tokenization
GUARDED_TOKENS = frozenset('no not never none nothing nobody nowhere neither nor without zero cannot t except excepting besides beside aside '
                           'apart other others only but additional extra more less major serious severe bad significant mild minor slight '
                           'slightly few fewer minimal yet until since far barely hardly almost nearly maybe probably possibly might may if '
                           'unless usually typical typically'.split())
RANGE = r'(?:to|through|-|–|—)'  # numeric range connectors ("12 to 14-week", "5-10")
PRONOUNS = (('i', 'me', 'my', 'mine', 'myself'), ('you', 'your', 'yours', 'yourself', 'yourselves'), ('he', 'him', 'his', 'himself'),
            ('she', 'her', 'hers', 'herself'), ('we', 'us', 'our', 'ours', 'ourselves'), ('they', 'them', 'their', 'theirs', 'themselves'))
FORMS = {form: frozenset(family) for family in PRONOUNS for form in family}
KINDS ={'surface_not_found', 'evidence_not_found', 'evidence_ambiguous', 'field_not_found', 'schema_invalid'}  # schema: stage contract only
MAX_OUTPUT_TOKENS = 1024
FIELD_MEANING = {'effect': 'the symptom or bodily change, as written in the text', 'indication': 'the condition or goal the drug is used for, as written in the text',
                 'surface': 'the words the text uses for the drug, as written'}
SURFACE_INSTRUCTIONS = ('A drug-statement extraction was rejected because one field value is not copied verbatim from the source text. '
                        'You get the field, the rejected value and exact source text. Choose the candidate that states the same '
                        'thing as the rejected value in the words of the text, preferring the shortest candidate that covers all '
                        'of it. Answer null if no candidate states the same thing. Do not reinterpret the item.')
VALUE_INSTRUCTIONS = ('A drug-statement extraction was rejected because one field value of an item is not supported by that '
                      "item's own source quotes. You get the field, the rejected value and the quotes. Choose the candidate "
                      'that the quotes state for that field, or null if the quotes state no such value. Do not reinterpret '
                      'the item or change any other field.')
RULE = {'policy': POLICY, 'quote': 'unique single/two-sentence passages sharing >= min(2, n) content words (ambiguous: unique passages containing it)',
        'surface': 'item-quote n-grams <= 10 tokens (content-word filter, 4-letter prefix fallback); else adjacent sentence (added as evidence)',
        'drug': 'n-grams <= 6 tokens of the sentences containing a content word or 4-letter prefix of the rejected surface (anchor), ranked by overlap then length; the request shows the cited quotes of that drug\'s items plus the anchor sentences',
        'field': 'item-quote token-bounded n-grams <= 8 tokens sharing a digit or a content word with the rejected value; if none, the same from the input sentences that contain such a digit/word (anchored context shown); or null; dose fields keep compound units whole',
        'trim': 'edge punctuation trimmed except a decimal mark or sign that starts a number',
        'contract': 'event fields of a not_occurred/not_asserted effect are emptied deterministically (p7 rule); removed values logged',
        'limits': {'max_candidates': MAX_CANDIDATES, 'max_candidate_chars': MAX_CANDIDATE_CHARS}}
RULE_1_3_1 = {**RULE, 'policy': POLICY_1_3_1,
              'field': RULE['field'] + '; 1.3.1: a value with no digit and no content word relates through its exact pronoun/possessive forms '
                       '(you ~ you, your, yours, yourself, yourselves); a pool over the caps is ranked (shared digits, shared content words '
                       'or pronoun forms, closeness in token count to the rejected value, first occurrence) and its top-ranked prefix '
                       'within both caps is offered; a pool within the caps is unchanged'}
RULE_1_3_2 = {**RULE_1_3_1, 'policy': POLICY_1_3_2,
              'pronoun': 'a value that is a single pronoun gets only the exact tokens of its pronoun/possessive family in the item quotes '
                         '(else in the input sentences containing one), never n-grams containing them',
              'temporal': 'onset/event_duration/duration candidates drop unbalanced emphasis, quote or bracket marks, a leading or '
                          'trailing range connector, and phrases that at every bounded occurrence start or end inside a numeric range'}
RULE_1_3_3 = {**RULE_1_3_2, 'policy': POLICY_1_3_3,
              'temporal': RULE_1_3_2['temporal'] + '; 1.3.3: also drop candidates containing an emphasis or code mark (* _ ~ `) '
                          'or ending on a function word (article, preposition, conjunction, possessive, demonstrative)'}
RULE_1_4 = {**RULE_1_3_3, 'policy': POLICY_1_4,
            'surface': "item-quote n-grams <= 10 tokens (content-word filter, 4-letter prefix fallback); no adjacent or remote sentence; complete phrases only",
            'drug': 'complete n-grams <= 6 tokens of the verified quotes of the items bound to that drug, sharing a content word or 4-letter prefix with the rejected surface; no remote sentence',
            'field': "item-quote token-bounded n-grams <= 8 tokens sharing a digit, content word or pronoun form with the rejected value; "
                     "no remote fallback; complete phrases only (temporal fields also the 1.3.2 range rules, single pronouns their tokens); "
                     "an empty pool removes a non-core field without a request and leaves a scope_text unresolved; scope_text tasks are not nullable",
            'quote_tokens': 'a replacement passage contains every word token of the rejected quote in order (case, whitespace, punctuation aside) '
                            'and no more negation or scope words (GUARDED_TOKENS: negations, exception, severity, quantity, temporal and '
                            'modality words) than the rejected quote; otherwise none is offered',
            'contract': "event fields of a not_occurred/not_asserted effect are emptied (p7 rule), except that an emptied severity on a not_occurred "
                        "effect keeps its qualifier: a generic effect becomes a no-side-effect claim scoped by the qualifier its own quote negates; "
                        "a specific effect takes the qualifier written right before it in its own quote; otherwise unresolved"}
RULE_1_5 = {**RULE_1_4, 'policy': POLICY_1_5,
            'numeric': 'a field value with a number offers only candidates whose ordered (number, unit) signature matches it: '
                       'the same numbers in the same order as exact signed decimals (number words zero-twelve, fifteen, twenty, '
                       'thirty), and at each position the unit the value writes after that number, if any; an empty pool removes a '
                       'non-core field without a request (original value logged) and leaves a scope_text unresolved'}
RULES = {POLICY: RULE, POLICY_1_3_1: RULE_1_3_1, POLICY_1_3_2: RULE_1_3_2, POLICY_1_3_3: RULE_1_3_3, POLICY_1_4: RULE_1_4, POLICY_1_5: RULE_1_5}
NUMBER_WORDS = {'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
                'nine': '9', 'ten': '10', 'eleven': '11', 'twelve': '12', 'fifteen': '15', 'twenty': '20', 'thirty': '30'}
UNIT_OF = {**dict.fromkeys(('mg', 'mgs', 'milligram', 'milligrams'), 'mg'), **dict.fromkeys(('mcg', 'ug', 'µg', 'microgram', 'micrograms'), 'mcg'),
           **dict.fromkeys(('ml', 'mls', 'cc', 'milliliter', 'milliliters', 'millilitre', 'millilitres'), 'ml'),
           **dict.fromkeys(('unit', 'units', 'iu', 'u'), 'unit'), **dict.fromkeys(('g', 'gram', 'grams'), 'g'),
           **dict.fromkeys(('kg', 'kgs', 'kilo', 'kilos'), 'kg'), **dict.fromkeys(('lb', 'lbs', 'pound', 'pounds'), 'lb'),
           **dict.fromkeys(('day', 'days'), 'day'), **dict.fromkeys(('week', 'weeks', 'wk', 'wks'), 'week'),
           **dict.fromkeys(('month', 'months', 'mo', 'mos'), 'month'), **dict.fromkeys(('year', 'years', 'yr', 'yrs'), 'year'),
           **dict.fromkeys(('hour', 'hours', 'hr', 'hrs'), 'hour'), **dict.fromkeys(('minute', 'minutes', 'min', 'mins'), 'minute'),
           '%': 'percent', 'percent': 'percent'}
QUANTITY_TOKEN = re.compile(r'((?:(?<![\w.,])[-+−])?(?:\d+(?:[.,]\d+)*|[.,]\d+))|([^\W\d_]+|%)')
INSTRUCTIONS_SHA = {'quote': sha256_text(QUOTE_INSTRUCTIONS_1_2), 'surface': sha256_text(SURFACE_INSTRUCTIONS), 'value': sha256_text(VALUE_INSTRUCTIONS)}


def _decimal(number):
    """Exact signed decimal of a written number ("1,000" -> 1000, "2,5" -> 2.5, ".25" -> 0.25, "−5" -> -5)."""
    sign = '-' if number[:1] in '-−' else ''
    digits = number.lstrip('-+−')
    plain = digits.replace(',', '') if re.fullmatch(r'\d{1,3}(?:,\d{3})+', digits) else digits.replace(',', '.')
    try:
        return format(Decimal(sign + plain).normalize(), 'f')
    except InvalidOperation:
        return sign + plain


def _quantities(text):
    """1.5.0: the ordered (number, unit) signature of a phrase: each number as an exact signed decimal (number words
    included), with the first unit written after it and before the next number, else None."""
    out = []
    for number, word in QUANTITY_TOKEN.findall(text or ''):
        w = word.lower()
        if number or w in NUMBER_WORDS:
            out.append([_decimal(number) if number else NUMBER_WORDS[w], None])
        elif w in UNIT_OF and out and out[-1][1] is None:
            out[-1][1] = UNIT_OF[w]
    return [tuple(x) for x in out]


def _same_quantities(candidate, rejected):
    """1.5.0: the candidate writes the rejected value's numbers in the same order, each with the unit the rejected value
    writes after it (a rejected number without a unit accepts any)."""
    want, have = _quantities(rejected), _quantities(candidate)
    return len(want) == len(have) and all(n == m and (u is None or u == v) for (n, u), (m, v) in zip(want, have))


def _trim(phrase):
    """Edge punctuation trimmed, except a decimal mark or sign that starts a number (".25mg", "-.5", "+5")."""
    start, end = 0, len(phrase)
    while start < end and phrase[start] in EDGE and not (phrase[start] in '.,+-−–' and start + 1 < end and (phrase[start + 1].isdigit()
                                                                                                          or phrase[start + 1] in '.,')):
        start += 1
    while end > start and phrase[end - 1] in EDGE:
        end -= 1
    return phrase[start:end]


def _ngrams(text, max_tokens, compound_units=False):
    """Token-bounded exact phrases of text (edge punctuation trimmed, number signs kept), in first-occurrence order."""
    seen, out = set(), []
    tokens = text.split()
    for n in range(1, max_tokens + 1):
        for i in range(len(tokens) - n + 1):
            phrase = _trim(' '.join(tokens[i:i + n]))
            if phrase and phrase not in seen and phrase in text and contract2.literal_hits(phrase, {'t': text}, compound_units):
                seen.add(phrase)
                out.append(phrase)
    return out


def _surface_candidates(quotes, rejected):
    """Item-quote n-grams (<= 10 tokens) sharing a content word with the rejected value, else a 4-letter-prefix relative."""
    words = _words(rejected)
    stems = {w[:4] for w in words if len(w) >= 4}
    pool = list(dict.fromkeys(c for q in quotes for c in _ngrams(q, 10)))
    if not words:
        return pool
    kept = [c for c in pool if _words(c) & words]
    return kept or [c for c in pool if any(len(w) >= 4 and w[:4] in stems for w in _words(c))]


def _capped(cands):
    return bool(cands) and len(cands) <= MAX_CANDIDATES and sum(map(len, cands)) <= MAX_CANDIDATE_CHARS


def _tokens(s):
    return set(re.findall(r'[a-z]+', s.lower()))


def _forms(value):
    """1.3.1: the exact pronoun/possessive forms of a value's pronoun tokens ("you" -> you, your, yours, ...)."""
    return frozenset().union(*(FORMS.get(t, frozenset()) for t in _tokens(value)))


def _single_pronoun(value):
    """1.3.2: the pronoun family of a value that is exactly one pronoun token ("you"), else None."""
    tokens = re.findall(r'[A-Za-z]+', value)
    return FORMS.get(tokens[0].lower()) if len(tokens) == 1 and value.strip().lower() == tokens[0].lower() else None


def _pronoun_tokens(texts, family, fields):
    """1.3.2: exact tokens of the pronoun family in texts (as written), bounded literals of the input, first occurrence."""
    return list(dict.fromkeys(m.group(0) for q in texts for m in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)?", q)
                              if m.group(0).lower() in family and contract2.literal_hits(m.group(0), fields)))


def _unbalanced(c):
    """Emphasis, quote or bracket marks that do not pair up inside the phrase ("mark** total", "(week 3")."""
    rest = c.replace('**', '')
    if c.count('**') % 2 or rest.count('*') % 2 or c.count('__') % 2 or c.count('~~') % 2 or c.count('"') % 2:
        return True
    return any(c.count(a) != c.count(b) for a, b in ('()', '[]', '{}', '“”'))


def _range_fragment(c, fields):
    """1.3.2 temporal: a piece of a numeric range. The phrase starts or ends with a range connector, or at every bounded
    occurrence it ends on a number that the source continues into a range ("hit the **12" + " to 14-week") or starts on a
    number that ends a range ("14-week mark" after "12 to ")."""
    if re.match(rf'^{RANGE}\s*\W*\d', c, re.I) or re.search(rf'\d\W*\s*{RANGE}$', c, re.I):
        return True
    ends_num, starts_num = bool(re.search(r'\d\W*$', c)), bool(re.match(r'^\W*\d', c))
    if not (ends_num or starts_num):
        return False

    def inside(name, a, b):
        text = fields[name]
        after, before = text[b:b + 16], text[max(0, a - 16):a]
        return bool((ends_num and re.match(rf'^\W*\s*{RANGE}\s*\W*\d', after, re.I))
                    or (starts_num and re.search(rf'\d\W*\s*{RANGE}\s*\W*$', before, re.I)))
    hits = contract2.literal_hits(c, fields)
    return bool(hits) and all(inside(*h) for h in hits)


def _incomplete(c):
    """1.3.3 temporal: not a plain, complete phrase (an emphasis/code mark anywhere, or a trailing function word)."""
    tokens = re.findall(r"[a-z]+", c.lower())
    return bool(re.search(r'[*_~`]', c)) or bool(tokens and tokens[-1] in FUNCTION_TAIL and re.search(r'[A-Za-z]\W*$', c))


def _word_tokens(s):
    """Case-, whitespace- and punctuation-insensitive word tokens (curly apostrophes as straight ones)."""
    return re.findall(r'[a-z0-9]+', s.lower().replace('’', "'").replace('‘', "'"))


def _keeps_all_tokens(passage, rejected):
    """1.4.0 quote replacement: every word token of the rejected quote occurs in the passage, in the same order; the passage
    may only add source words around or between them (a more complete original passage), never drop or change one."""
    want, have = _word_tokens(rejected), iter(_word_tokens(passage))
    return bool(want) and all(any(w == h for h in have) for w in want)


def _adds_no_guarded_token(passage, rejected):
    """1.4.0 quote replacement: the passage adds no negation or scope word to the rejected quote ("I had nausea" must not
    become "I had no nausea", "no side effects" not "no major side effects"); fails closed."""
    have, want = Counter(_word_tokens(passage)), Counter(_word_tokens(rejected))
    return all(have[t] <= want[t] for t in GUARDED_TOKENS)


def _kept_qualifier(item):
    """1.4.0: how an emptied severity of a not_occurred effect keeps its qualifier, from the item's own quotes:
    ('noae', qualifier) for a generic effect, ('effect', qualified effect) for a specific one, or None."""
    quotes = item['quotes']
    if GENERIC_EFFECT.match(item['effect'].strip()):
        for quote in quotes:
            m = NEG_QUALIFIED.search(quote)
            if m:
                return 'noae', m.group('q')
        return None
    pattern = re.compile(rf'\b(?:{QUALIFIERS}|mild|slight|moderate)\s+{re.escape(item["effect"])}(?!\w)', re.I)
    for quote in quotes:
        m = pattern.search(quote)
        if m:
            return 'effect', m.group(0)
    return None


def _rank_pack(cands, rejected, score):
    """1.3.1, over-cap pools only: rank by score (larger first), then closeness in token count to the rejected value, then
    first occurrence; return the top-ranked prefix that fits both caps (MAX_CANDIDATES, MAX_CANDIDATE_CHARS)."""
    n = len(rejected.split())
    order = sorted(range(len(cands)), key=lambda k: (tuple(-x for x in score(cands[k])), abs(len(cands[k].split()) - n), k))
    out, chars = [], 0
    for k in order:
        if len(out) == MAX_CANDIDATES or chars + len(cands[k]) > MAX_CANDIDATE_CHARS:
            break
        out.append(cands[k])
        chars += len(cands[k])
    return out


def _shares(words, stems, phrase):
    cw = _words(phrase)
    return len(cw & words) or sum(1 for w in cw if len(w) >= 4 and w[:4] in stems)


def _drug_candidates(fields, rejected):
    """(candidates, anchor sentences): n-grams of the sentences that contain a content word (or 4-letter prefix) of the
    rejected surface, ranked by overlap then length. The anchor is the model's own referring words, never the whole text."""
    words = _words(rejected)
    stems = {w[:4] for w in words if len(w) >= 4}
    scored, anchors = {}, []
    for text in fields.values():
        for a, b in sentences(text):
            sentence = text[a:b]
            if not _shares(words, stems, sentence):
                continue
            anchors.append(sentence)
            for c in _ngrams(sentence, 6):
                share = _shares(words, stems, c)
                if share and contract2.literal_hits(c, fields):
                    scored[c] = max(scored.get(c, 0), share)
    ranked = sorted(scored, key=lambda c: (-scored[c], len(c)))
    return ranked[:MAX_CANDIDATES], anchors


def _own_drug_candidates(quotes, rejected, fields):
    """1.4.0 drug surface: complete n-grams (<= 6 tokens) of the bound items' own quotes that share a content word or a
    4-letter prefix with the rejected surface, ranked by overlap then length."""
    words = _words(rejected)
    stems = {w[:4] for w in words if len(w) >= 4}
    scored = {}
    for quote in quotes:
        for c in _ngrams(quote, 6):
            share = _shares(words, stems, c)
            if share and contract2.literal_hits(c, fields) and not _incomplete(c):
                scored[c] = max(scored.get(c, 0), share)
    return sorted(scored, key=lambda c: (-scored[c], len(c)))[:MAX_CANDIDATES]


def _adjacent(fields, spans):
    """{field: [(start, end)]} of the sentence right before and after each quote span (same field, not overlapping it)."""
    out = []
    for span in spans:
        name, _, off = span.rpartition(':')
        a, b = map(int, off.split('-'))
        sents = sentences(fields[name])
        before = [s for s in sents if s[1] <= a]
        after = [s for s in sents if s[0] >= b]
        for s in (before[-1:] + after[:1]):
            out.append((name, s[0], s[1]))
    return out


def _paths_from(anomaly):
    return list(getattr(anomaly, 'paths', None) or [])


def analyse(value, inp, spec, policy):
    """Localized wire of a reply in the analysis shape (the p7 item without its occurrence rule, so contract failures are
    located per path); ExtractionAnomaly for any structural or reference failure, which is not correctable here."""
    if spec.contract not in contract2.CONTRACTS:
        raise ExtractionAnomaly('schema_invalid', f'{POLICY}: contract 2.x runs only', retriable=False)
    wire = validate_model(contract2.WireExtractionLoose, value)
    compact.check_references(wire)
    if policy is not None:
        wire, _ = localize.apply(wire, inp.fields, policy)
        wire = validate_model(contract2.WireExtractionLoose, wire.model_dump())
    return wire


def quote_tasks(wire, fields, policy=POLICY):
    """Tasks for quotes with 0 or several token-bounded occurrences (one task per distinct rejected string)."""
    rejected = {}
    for group, i, item in compact.items(wire):
        for q, quote in enumerate(item.quotes):
            hits = contract2.literal_hits(quote, fields)
            if len(hits) != 1:
                rejected.setdefault((quote, len(hits) > 1), []).append((group, i, q))
    tasks = []
    dump = wire.model_dump()
    for (quote, ambiguous), targets in rejected.items():
        pool = [p for p in quote_candidates(fields, quote) if len(contract2.literal_hits(p, fields)) == 1]
        cands = [p for p in pool if quote in p] if ambiguous else pool
        if policy in OWN_ONLY:  # 1.4.0: only passages that keep every word of the rejected quote, in order, adding no guarded word
            cands = [c for c in cands if _keeps_all_tokens(c, quote) and _adds_no_guarded_token(c, quote)]
        if not _capped(cands):
            raise ExtractionAnomaly('evidence_not_found', f'{policy}: rejected quote has {len(cands)} passages', retriable=False)
        g, i, q = targets[0]
        tasks.append({'kind': 'quote', 'path': f'{g}[{i}].quotes[{q}]', 'targets': targets, 'rejected': quote, 'ambiguous': ambiguous,
                      'items': [{'path': f'{gg}[{jj}]', 'drug': dump[gg][jj]['drug'], 'value': dump[gg][jj].get('effect') or dump[gg][jj].get('indication')}
                                for gg, jj, _ in targets], 'candidates': cands})
    return tasks


def value_tasks(wire, fields, policy=POLICY):
    """Surface, drug, field and qualifier tasks for a reply whose quotes all resolve; returns (tasks, deterministic changes)."""
    spans = contract2.locate_evidence(wire, fields)  # quotes must be fixed first
    tasks, fixed = [], []
    own_only = policy in OWN_ONLY  # 1.4.0: own evidence only, complete phrases everywhere
    for i, d in enumerate(wire.drugs):
        if not contract2.literal_hits(d.surface, fields):
            cited = [compact.render_span(s, fields) for group, j, item in compact.items(wire) if item.drug == d.handle for s in spans[group, j]]
            if own_only:  # 1.4.0: n-grams of the verified quotes of the items bound to this drug only; no remote sentence
                cands, anchors = _own_drug_candidates(cited, d.surface, fields), []
            else:
                cands, anchors = _drug_candidates(fields, d.surface)
            if not _capped(cands):
                raise ExtractionAnomaly('surface_not_found', f'{policy}: drugs[{i}].surface has {len(cands)} candidates', retriable=False)
            tasks.append({'kind': 'drug', 'path': f'drugs[{i}].surface', 'set': ('drugs', i, 'surface'), 'field': 'surface', 'rejected': d.surface,
                          'context': list(dict.fromkeys(cited + anchors))[:12], 'candidates': cands})
    for group, name in (('purposes', 'indication'), ('side_effects', 'effect')):
        for i, item in enumerate(getattr(wire, group)):
            rejected = getattr(item, name)
            if contract2.literal_hits(rejected, fields):
                continue
            own = [compact.render_span(s, fields) for s in spans[group, i]]
            cands = [c for c in _surface_candidates(own, rejected) if contract2.literal_hits(c, fields) and not (own_only and _incomplete(c))]
            added = None
            if not cands and not own_only:
                pool = {}
                for f, a, b in _adjacent(fields, spans[group, i]):
                    sentence = fields[f][a:b]
                    for c in _surface_candidates([sentence], rejected):
                        if contract2.literal_hits(c, fields):
                            pool.setdefault(c, sentence)
                cands, added = list(pool), pool
            if not _capped(cands):
                raise ExtractionAnomaly('surface_not_found', f'{policy}: {group}[{i}].{name} has {len(cands)} candidates', retriable=False)
            tasks.append({'kind': 'surface', 'path': f'{group}[{i}].{name}', 'set': (group, i, name), 'field': name, 'rejected': rejected,
                          'context': own + sorted(set((added or {}).values())), 'candidates': cands, 'adds_evidence': added})
    for group, i, item in compact.items(wire):
        for name in contract2.AS_WRITTEN[group]:
            v = getattr(item, name, None)
            if v in (None, '') or contract2.literal_hits(v, fields, name in contract2.DOSE_FIELDS):  # the acceptance predicate
                continue
            own = [compact.render_span(s, fields) for s in spans[group, i]]
            digits, words = set(re.findall(r'\d', v)), _words(v)
            forms = _forms(v) if policy == POLICY_1_3_1 and not digits and not words else frozenset()  # 1.3.1: pronoun values
            family = _single_pronoun(v) if policy in TYPED else None  # 1.3.2+: pronoun tokens only
            typed = policy in TYPED and name in TEMPORAL_FIELDS  # 1.3.2+: well-formed temporal phrases only
            plain = (policy in PLAIN and name in TEMPORAL_FIELDS) or own_only  # 1.3.3: temporal phrases; 1.4.0: every field
            cu = name in contract2.DOSE_FIELDS

            numeric = policy == POLICY_1_5 and bool(_quantities(v))  # 1.5.0: a value with a number keeps its numbers and units

            def related(c):  # shares a digit or a content word with the rejected value ("3 months" ~ "three months")
                if numeric and not _same_quantities(c, v):
                    return False
                return bool(set(re.findall(r'\d', c)) & digits or _words(c) & words or forms & _tokens(c))

            def pool(texts):
                if family is not None:
                    return _pronoun_tokens(texts, family, fields)
                return list(dict.fromkeys(c for q in texts for c in _ngrams(q, 8, cu) if related(c) and contract2.literal_hits(c, fields, cu)
                                          and not (typed and (_unbalanced(c) or _range_fragment(c, fields))) and not (plain and _incomplete(c))))
            cands = pool(own)
            context = own
            if not cands and not own_only:  # anchored fallback: sentences of the input that contain a related digit or content word (or pronoun)
                anchors = [text[a:b] for text in fields.values() for a, b in sentences(text)
                           if (family & _tokens(text[a:b]) if family is not None else related(text[a:b]))]
                cands = pool(anchors)
                context = own + anchors[:8]
            if policy in SUCCESSORS and cands and not _capped(cands):  # 1.3.1+: rank and pack an over-cap pool within the caps
                cands = _rank_pack(cands, v, lambda c: (len(set(re.findall(r'\d', c)) & digits), len(_words(c) & words), len(forms & _tokens(c))))
            if own_only and not cands and name != 'scope_text':  # 1.4.0: a non-core field without an admissible literal is removed
                fixed.append({'path': f'{group}[{i}].{name}', 'kind': 'field_removed', 'removed': v, 'requests': 0,
                              'reason': "no admissible literal in the item's own quotes"})
                continue
            if not _capped(cands):  # all caps apply; an over-large or empty pool is not corrected here
                raise ExtractionAnomaly('field_not_found', f'{policy}: {group}[{i}].{name} has {len(cands)} candidates', retriable=False)
            tasks.append({'kind': 'field', 'path': f'{group}[{i}].{name}', 'set': (group, i, name), 'field': name, 'rejected': v,
                          'context': context, 'candidates': cands, 'nullable': not (own_only and name == 'scope_text')})
    seen, unique = set(), []
    for task in tasks:  # one task per path (a later check must never re-request a path)
        if task['path'] not in seen:
            seen.add(task['path'])
            unique.append(task)
    return unique, fixed


def request_body(task, model, reasoning_effort):
    if task['kind'] == 'quote':
        items = '\n'.join(f"- {it['path']}: drug {it['drug']}, {json.dumps(it['value'], ensure_ascii=False)}" for it in task['items'])
        content = f"Rejected quote: {json.dumps(task['rejected'], ensure_ascii=False)}\nItems that cite it:\n{items}"
        instructions, name = QUOTE_INSTRUCTIONS_1_2, 'quote_select'
    else:
        ctx = '\n'.join(f'<<<\n{q}\n>>>' for q in task['context'])
        meaning = FIELD_MEANING.get(task['field'], f"the {task['field']} of the item, as written in the text")
        content = (f"Field: {task['path']} ({meaning})\nRejected value: {json.dumps(task['rejected'], ensure_ascii=False)}\n"
                   + (f'Source text:\n{ctx}' if ctx else 'Source text: the candidates are exact phrases of the input.'))
        instructions = SURFACE_INSTRUCTIONS if task['kind'] in ('surface', 'drug') else VALUE_INSTRUCTIONS
        name = 'literal_select'
    return build_body(model=model, instructions=instructions, user_content=content, schema_name=name,
                      schema=schema(task['candidates']), reasoning_effort=reasoning_effort, max_output_tokens=MAX_OUTPUT_TOKENS)


def diff_paths(a, b, prefix=''):
    """Paths where two canonical wire dicts differ (a list of different length differs as a whole)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = set()
        for k in set(a) | set(b):
            out |= diff_paths(a.get(k), b.get(k), f'{prefix}.{k}' if prefix else k)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = set()
        for i, (x, y) in enumerate(zip(a, b)):
            out |= diff_paths(x, y, f'{prefix}[{i}]')
        return out
    return set() if a == b else {prefix}


def _path_set(value, path, new):
    """Set value at a canonical path such as 'side_effects[2].effect' or 'drugs[0].surface' or 'purposes[1].quotes[0]'."""
    parts = re.findall(r'(\w+)(?:\[(\d+)\])?', path)
    node = value
    for n, (key, idx) in enumerate(parts):
        last = n == len(parts) - 1
        if idx:
            if last:
                node[key][int(idx)] = new
            else:
                node = node[key][int(idx)]
        elif last:
            node[key] = new
        else:
            node = node[key]


def expected_from(original, changes):
    """The corrected wire the change log allows: the original with exactly each logged patch applied (target values,
    contract removals, quote replacements on their targets, and appended antecedent evidence at the end of its item)."""
    out = json.loads(json.dumps(original))
    for c in changes:
        if c['kind'] in ('contract', 'field_removed'):
            name = c['path'].rsplit('.', 1)[1]
            _path_set(out, c['path'], [] if name == 'clinical_consequences' else None)
        elif c['kind'] == 'contract_to_noae':  # 1.4.0: a generic negation moves to no_side_effects with its qualifier as scope
            i = int(re.match(r'side_effects\[(\d+)\]$', c['path']).group(1))
            out['no_side_effects'].append(_noae_from(out['side_effects'][i], c['scope_text']))
            del out['side_effects'][i]
        elif c['kind'] == 'quote':
            for target in c['targets']:
                _path_set(out, target, c['to'])
        else:
            _path_set(out, c['path'], c['to'])
            if c.get('added_evidence'):
                group, i = re.match(r'(\w+)\[(\d+)\]', c['path']).groups()
                out[group][int(i)]['quotes'].append(c['added_evidence'])
    return out


def _noae_from(item, scope_text):
    """The no-side-effect claim of a generic not_occurred effect (drug, subject, use and quotes kept; event fields dropped)."""
    return {'drug': item['drug'], 'subject': item['subject'], 'subject_label': item['subject_label'], 'scope_text': scope_text,
            'use_status': item['use_status'], 'dose': item['dose'], 'duration': item['duration'], 'quotes': list(item['quotes'])}


def _contract_stage(value, pol):
    """Deterministic p7 contract corrections of a reply: [(change)] applied to value in place, in the logged order.
    1.4.0 keeps the qualifier of an emptied severity on a not_occurred effect or fails closed (see the module docstring)."""
    changes, moves = [], []
    viol = contract2.event_violations(validate_model(contract2.WireExtractionLoose, value))
    by_item = {}
    for path, name, v in viol:
        by_item.setdefault(int(re.search(r'\[(\d+)\]', path).group(1)), []).append((path, name, v))
    for i, found in sorted(by_item.items()):
        item = value['side_effects'][i]
        keep = None
        if pol in OWN_ONLY and item['occurrence'] == 'not_occurred' and any(name == 'severity' for _, name, _ in found):
            keep = _kept_qualifier(item)
            if keep is None:
                raise ExtractionAnomaly('qualifier_unsupported', f'{pol}: side_effects[{i}] severity qualifies a negation its own quotes do not '
                                        'write before the effect', stage='contract', paths=[f'side_effects[{i}].severity'], retriable=False)
        for path, name, v in found:
            item[name] = [] if name == 'clinical_consequences' else None
            changes.append({'path': path, 'kind': 'contract', 'removed': v, 'requests': 0,
                            'reason': 'p7 contract: event fields stay empty for not_occurred/not_asserted effects'})
        if keep and keep[0] == 'effect':
            changes.append({'path': f'side_effects[{i}].effect', 'kind': 'contract_qualifier', 'from': item['effect'], 'to': keep[1], 'requests': 0})
            item['effect'] = keep[1]
        elif keep:
            moves.append((i, keep[1]))
    for i, scope in sorted(moves, reverse=True):
        value['no_side_effects'].append(_noae_from(value['side_effects'][i], scope))
        del value['side_effects'][i]
        changes.append({'path': f'side_effects[{i}]', 'kind': 'contract_to_noae', 'scope_text': scope, 'requests': 0,
                        'to': f'no_side_effects[{len(value["no_side_effects"]) - 1}]'})
    return changes


def _apply(value, task, choice, changes):
    if task['kind'] == 'quote':
        for group, i, q in task['targets']:
            value[group][i]['quotes'][q] = choice
        changes.append({'path': task['path'], 'kind': 'quote', 'targets': [f'{g}[{i}].quotes[{q}]' for g, i, q in task['targets']],
                        'from': task['rejected'], 'to': choice, 'candidates': len(task['candidates'])})
        return [f'{g}[{i}].quotes[{q}]' for g, i, q in task['targets']]
    group, i, name = task['set']
    value[group][i][name] = choice
    allowed = [task['path']]
    entry = {'path': task['path'], 'kind': task['kind'], 'from': task['rejected'], 'to': choice, 'candidates': len(task['candidates'])}
    added = (task.get('adds_evidence') or {}).get(choice)
    if added and added not in value[group][i]['quotes']:
        if len(value[group][i]['quotes']) >= 8:
            raise ExtractionAnomaly('surface_not_found', f"{task.get('policy', POLICY)}: no room to add the antecedent evidence", retriable=False)
        value[group][i]['quotes'].append(added)
        entry['added_evidence'] = added
        allowed.append(f'{group}[{i}].quotes')
    changes.append(entry)
    return allowed


def in_scope(repair, rid):
    """A 1.3.1 successor ledger handles only the records frozen in its manifest scope; a 1.3.0 ledger handles all."""
    scope = repair.manifest.get('scope')
    return scope is None or rid in scope


def plan(ledger, repair):
    """Offline preview: [(record_id, custom_id, max requests or the not-eligible reason)]. Quote tasks are counted
    exactly; value tasks after a quote correction are bounded by one per failing value path."""
    responses = {r['custom_id']: r['response'] for r in ledger.rows('responses')}
    pol = repair.policy
    out = []
    for rid, state in ledger.states().items():
        last = state['last']
        if state['ok'] or state['held'] or not last or last['anomaly']['kind'] not in KINDS or rid in repair.results() or not in_scope(repair, rid):
            continue
        if last['anomaly']['kind'] == 'schema_invalid' and last['anomaly'].get('stage') != 'contract':
            out.append((rid, last['custom_id'], 'structural failure (not path-correctable)'))
            continue
        try:
            wire = analyse(parse_json_object(final_output_text(responses[last['custom_id']])), ledger.inputs()[rid], ledger.spec, ledger.manifest.get('localize'))
            value = wire.model_dump()
            if pol in OWN_ONLY:
                _contract_stage(value, pol)
            else:
                for path, name, _ in contract2.event_violations(wire):
                    value['side_effects'][int(re.search(r'\[(\d+)\]', path).group(1))][name] = [] if name == 'clinical_consequences' else None
            wire = validate_model(contract2.WireExtractionLoose, value)
            q = quote_tasks(wire, ledger.inputs()[rid].fields, pol)
            n = len(q) + (len(value_tasks(wire, ledger.inputs()[rid].fields, pol)[0]) if not q else len(wire.drugs) + sum(1 for _ in compact.items(wire)) * 2)
            out.append((rid, last['custom_id'], n))
        except (ExtractionAnomaly, KeyError) as error:
            out.append((rid, last['custom_id'], str(error)[:160]))
    return out


def recover(ledger, repair):
    """Crash recovery for this ledger's literal-select-1.3.x requests, by custom_id. Nothing is ever re-sent.
    - raw reply saved: add a missing usage row from the raw usage; the run then replays the saved choice through the same
      gate and re-validation;
    - HTTP 200 recorded but no raw reply saved: book the cap once as an uncertain charge and hold the record;
    - no HTTP outcome at all: book the cap once as an uncertain charge and hold the record.
    Returns {record_id: 'replay' | 'held'} for the records whose repair was interrupted."""
    from .runner import now as _now
    from .structured import cost_usd, token_split
    bodies = {r['custom_id']: r['body'] for r in repair.rows('request_bodies')}
    attempts = {}
    for a in repair.rows('attempts'):
        attempts.setdefault(a['custom_id'], []).append(a)
    saved = {r['custom_id']: r['response'] for r in repair.rows('responses')}
    usage = {r['custom_id'] for r in repair.rows('usage')}
    finished = {r['record_id'] for r in repair.rows('results')}
    out = {}
    for req in repair.rows('requests'):
        cid, rid = req['custom_id'], req['record_id']
        if req.get('policy') != repair.policy or rid in finished:
            continue
        if cid in saved:
            if cid not in usage:
                u = saved[cid].get('usage') or {}
                repair.append('usage', {'custom_id': cid, 'response_id': saved[cid].get('id'), 'status': saved[cid].get('status'), **token_split(u),
                                        'cost_usd': cost_usd(repair.manifest['model'], u) or 0.0, 'at': _now(), 'recovered': True})
                usage.add(cid)
            out.setdefault(rid, 'replay')
            continue
        if any(a.get('recovered') for a in attempts.get(cid, [])):
            out[rid] = 'held'
            continue
        http200 = any(a.get('http_status') == 200 for a in attempts.get(cid, []))
        if attempts.get(cid) and not http200:
            continue  # a recorded non-200 outcome: the original pass already booked it; nothing to recover
        cap = request_cap_usd(ledger.manifest['model'], bodies[cid]) if cid in bodies else 0.0
        repair.append('attempts', {'custom_id': cid, 'recovered': True, 'uncertain_charge_usd': round(cap, 9), 'at': _now(),
                                   'error': 'HTTP 200 recorded but body not saved' if http200 else 'no outcome recorded (interrupted)'})
        repair.append('held', {'custom_id': cid, 'reason': 'interrupted repair request; held, not re-sent', 'at': _now()})
        repair.append('results', {'record_id': rid, 'policy': repair.policy, 'outcome': 'held', 'at': _now(), 'reason': f'{cid} interrupted; held'})
        finished.add(rid)
        out[rid] = 'held'
    return out


def run(ledger, repair, send, *, budget_usd):
    """literal-select-1.3.x (the ledger's policy) over every eligible record of a contract 2.0.0 run in the ledger's scope,
    holding the run's driver lock; returns [(record_id, outcome)]."""
    from .runner import acquire_driver_lock, release_driver_lock
    lock = acquire_driver_lock(ledger.dir)
    try:
        recover(ledger, repair)
        return _run(ledger, repair, send, budget_usd=budget_usd)
    finally:
        release_driver_lock(lock)


def _run(ledger, repair, send, *, budget_usd):
    spec, policy, model = ledger.spec, ledger.manifest.get('localize'), ledger.manifest['model']
    pol = repair.policy
    if pol not in POLICIES:
        raise ValueError(f'path_repair: unknown policy {pol!r}')
    if spec.contract != CONTRACT_OF[pol]:
        raise ValueError(f'path_repair: {pol} corrects {CONTRACT_OF[pol]} runs only, not {spec.contract}')
    responses = {r['custom_id']: r['response'] for r in ledger.rows('responses')}
    saved_replies = {r['custom_id']: r['response'] for r in repair.rows('responses')}
    sent = {r['custom_id'] for r in repair.rows('requests')}
    done = set(repair.results())
    spent, uncertain = repair.spent_and_uncertain()
    outcomes = []
    for rid, state in ledger.states().items():
        last = state['last']
        if state['ok'] or state['held'] or not last or last['anomaly']['kind'] not in KINDS:
            continue
        if last['anomaly']['kind'] == 'schema_invalid' and last['anomaly'].get('stage') != 'contract':
            continue  # a structural (shape) failure is not a path-correctable contract failure
        if not in_scope(repair, rid):
            continue
        if rid in done:
            outcomes.append((rid, 'already_tried'))
            continue
        cid, inp = last['custom_id'], ledger.inputs()[rid]
        base = {'record_id': rid, 'from_custom_id': cid, 'policy': pol, 'at': now(),
                'reply_sha256': sha256_text(compact_json(responses[cid])) if cid in responses else None}
        changes, allowed, failure = [], set(), None
        try:
            if cid not in responses:
                raise ExtractionAnomaly('surface_not_found', f'{pol}: no saved reply for {cid}', retriable=False)
            wire = analyse(parse_json_object(final_output_text(responses[cid])), inp, spec, policy)
            original = wire.model_dump()
            value = json.loads(json.dumps(original))
            if pol in OWN_ONLY:  # 1.4.0: an emptied severity of a negation keeps its qualifier, or the record fails closed
                changes.extend(_contract_stage(value, pol))
            else:
                for path, name, v in contract2.event_violations(validate_model(contract2.WireExtractionLoose, value)):
                    i = int(re.search(r'\[(\d+)\]', path).group(1))
                    value['side_effects'][i][name] = [] if name == 'clinical_consequences' else None
                    allowed.add(path)
                    changes.append({'path': path, 'kind': 'contract', 'removed': v, 'requests': 0,
                                    'reason': 'p7 contract: event fields stay empty for not_occurred/not_asserted effects'})
            for stage in ('quotes', 'values'):
                current = validate_model(contract2.WireExtractionLoose, value)
                if stage == 'quotes':
                    tasks = quote_tasks(current, inp.fields, pol)
                else:
                    tasks, fixed = value_tasks(current, inp.fields, pol)
                    for f in fixed:  # 1.4.0: non-core fields without an admissible literal, removed without a request
                        _path_set(value, f['path'], None)
                        changes.append(f)
                for task in tasks:
                    rcid = f"{rid}#{task['path']}{SUFFIX[pol]}"
                    body = request_body(task, model, ledger.manifest['reasoning_effort'])
                    if rcid in sent:  # interrupted earlier: replay the saved raw choice, never re-send
                        if rcid not in saved_replies:
                            raise ExtractionAnomaly('surface_not_found', f'{pol}: {task["path"]} was already requested once', retriable=False)
                        frozen = {r['custom_id']: r['body_sha256'] for r in repair.rows('requests')}
                        if frozen.get(rcid) != sha256_text(compact_json(body)):
                            raise ExtractionAnomaly('schema_invalid', f'{pol}: {task["path"]}: rebuilt body differs from the sent one', retriable=False)
                        response = saved_replies[rcid]
                        choice = parse_json_object(final_output_text(response))
                        if set(choice) != {'choice'} or (choice['choice'] is not None and choice['choice'] not in task['candidates']):
                            raise ExtractionAnomaly('schema_invalid', f'{pol}: reply is not one enum candidate', retriable=False)
                        if choice['choice'] is None and not task.get('nullable'):
                            raise ExtractionAnomaly('evidence_not_found' if task['kind'] == 'quote' else 'surface_not_found',
                                                    f'{pol}: {task["path"]}: no candidate selected (null)', retriable=False)
                        allowed.update(_apply(value, dict(task, policy=pol), choice['choice'], changes))
                        changes[-1].update(custom_id=rcid, replayed=True)
                        continue
                    cap = request_cap_usd(model, body)
                    if spent + uncertain + cap > budget_usd:
                        raise ExtractionAnomaly('surface_not_found', f'{pol}: budget', retriable=False)
                    body_sha = sha256_text(compact_json(body))
                    repair.append('request_bodies', {'custom_id': rcid, 'body_sha256': body_sha, 'candidate_rule': RULES[pol], 'policy': pol,
                                                     'instructions_sha256': sha256_text(body['instructions']), 'body': body})
                    repair.append('requests', {'custom_id': rcid, 'record_id': rid, 'path': task['path'], 'kind': task['kind'], 'policy': pol,
                                               'body_sha256': body_sha, 'candidates': len(task['candidates']), 'at': now()})
                    sent.add(rcid)
                    response, sfail, _ = _send_once(repair, send, rcid, body, cap, pol)
                    spent, uncertain = repair.spent_and_uncertain()
                    if sfail is not None:
                        failure = sfail
                        raise StopIteration
                    choice = parse_json_object(final_output_text(response))
                    if set(choice) != {'choice'} or (choice['choice'] is not None and choice['choice'] not in task['candidates']):
                        raise ExtractionAnomaly('schema_invalid', f'{pol}: reply is not one enum candidate', retriable=False)
                    if choice['choice'] is None and not task.get('nullable'):
                        raise ExtractionAnomaly('evidence_not_found' if task['kind'] == 'quote' else 'surface_not_found',
                                                f'{pol}: {task["path"]}: no candidate selected (null)', retriable=False)
                    allowed.update(_apply(value, dict(task, policy=pol), choice['choice'], changes))
                    changes[-1]['custom_id'] = rcid
            expected = expected_from(original, changes)
            if expected != value:  # full equality with the original plus exactly the logged patches
                extra = sorted(diff_paths(expected, value))
                raise ExtractionAnomaly('schema_invalid', f'{pol}: preservation gate: unexpected changes at {extra[:5]}', retriable=False)
            record = compact.evaluate_value(value, inp, ledger.run_id, spec, localize_policy=policy, repairs=[])
        except StopIteration:
            pass
        except ExtractionAnomaly as anomaly:
            failure = anomaly
        if failure == 'held':
            repair.append('results', {**base, 'outcome': 'held', 'reason': 'HTTP 200 body is not a JSON object', 'changes': changes})
            outcomes.append((rid, 'held'))
        elif failure is not None:
            repair.append('results', {**base, 'outcome': 'anomaly', 'anomaly': failure.as_dict(), 'changes': changes})
            outcomes.append((rid, failure.kind))
        else:
            repair.append('results', {**base, 'outcome': 'ok', 'record': record, 'changes': changes, 'preserved': 'all paths except the listed changes',
                                      'canonical_version': spec.canonical[0] if spec.canonical else None})
            outcomes.append((rid, 'ok'))
    return outcomes


def reextract_eligible(ledger, repair):
    """1.4.0: records left without a validated result after the correction pass, on a non-provider failure: a 1.4.0
    correction that ended as an anomaly, or a structure/reference failure that path correction does not handle."""
    from .structured import STAGE_OF
    results, out = repair.results(), []
    for rid, state in ledger.states().items():
        last = state['last']
        if state['ok'] or state['held'] or not last:
            continue
        mine = results.get(rid)
        stage = last['anomaly'].get('stage') or STAGE_OF.get(last['anomaly']['kind'])
        if mine is not None and mine.get('policy') in REEXTRACT_OF.values():
            continue
        if (mine is not None and mine.get('policy') in OWN_ONLY and mine['outcome'] == 'anomaly') or \
                (mine is None and stage in ('structure', 'reference')):
            out.append(rid)
    return out


def reextract_once(ledger, repair, send, *, budget_usd):
    """reextract-once-1.4.0 (literal-select-1.4.0 ledgers): one ordinary request of the run's own spec for each eligible
    record, evaluated exactly like a first attempt and never retried or corrected again; a saved reply of an interrupted
    request is evaluated, never re-sent. Returns [(record_id, outcome)]."""
    if repair.policy not in OWN_ONLY:
        return []
    label = REEXTRACT_OF[repair.policy]
    from .runner import acquire_driver_lock, release_driver_lock
    lock = acquire_driver_lock(ledger.dir)
    try:
        spec, policy, model = ledger.spec, ledger.manifest.get('localize'), ledger.manifest['model']
        sent = {r['custom_id'] for r in repair.rows('requests')}
        saved = {r['custom_id']: r['response'] for r in repair.rows('responses')}
        spent, uncertain = repair.spent_and_uncertain()
        outcomes = []
        for rid in reextract_eligible(ledger, repair):
            inp, rcid = ledger.inputs()[rid], f'{rid}#reextract{SUFFIX[repair.policy]}'
            base = {'record_id': rid, 'policy': label, 'from_custom_id': rcid, 'at': now()}
            body = ledger.request_body(inp)
            failure = response = None
            if rcid in sent:
                if rcid not in saved:
                    continue  # interrupted without a saved reply: recover() holds it; never re-sent
                response = saved[rcid]
            else:
                cap = request_cap_usd(model, body)
                if spent + uncertain + cap > budget_usd:
                    outcomes.append((rid, 'budget'))
                    continue
                body_sha = sha256_text(compact_json(body))
                repair.append('request_bodies', {'custom_id': rcid, 'body_sha256': body_sha, 'policy': label, 'body': body})
                repair.append('requests', {'custom_id': rcid, 'record_id': rid, 'path': None, 'kind': 'reextract', 'policy': repair.policy,
                                           'body_sha256': body_sha, 'at': now()})
                sent.add(rcid)
                response, failure, _ = _send_once(repair, send, rcid, body, cap, label)
                spent, uncertain = repair.spent_and_uncertain()
            if failure is None:
                try:
                    record = compact.evaluate_response(response, inp, ledger.run_id, spec, localize_policy=policy, repairs=[])
                except ExtractionAnomaly as anomaly:
                    failure = anomaly
            if failure == 'held':
                repair.append('results', {**base, 'outcome': 'held', 'reason': 'HTTP 200 body is not a JSON object'})
                outcomes.append((rid, 'held'))
            elif failure is not None:
                repair.append('results', {**base, 'outcome': 'anomaly', 'anomaly': failure.as_dict()})
                outcomes.append((rid, failure.kind))
            else:
                repair.append('results', {**base, 'outcome': 'ok', 'record': record, 'changes': [{'kind': 'reextract', 'custom_id': rcid}],
                                          'canonical_version': spec.canonical[0] if spec.canonical else None})
                outcomes.append((rid, 'ok'))
        return outcomes
    finally:
        release_driver_lock(lock)
