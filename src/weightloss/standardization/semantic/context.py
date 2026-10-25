"""Deterministic context guards: source support, context-required terms, qualifiers, coordination and purpose cues.

Guards only ever withhold acceptance (candidate / unclear / flagged); none of them adds a concept or an intent that
the record does not state, and none changes a KG field.
"""
from __future__ import annotations

from functools import lru_cache
import re

from .textnorm import fold_key, tokens

RULES_VERSION = 'semantic-context-0.1.4'

# Lookup keys whose meaning depends on the record; a rule hit on them never becomes accepted. 0.1.4 removes only burp /
# burps / burping: they name eructation unambiguously and are reviewed keys of the certified eructation rules (knowledge
# v0.1.2+). Vague words (sick, gas, gassy, stomach issues, ...) stay here even if a rule names them; check-knowledge only
# reports such a certified key as a conflict for review and never lets a certification override this list.
CONTEXT_REQUIRED = {
    'side_effect': frozenset({'sick', 'sickness', 'gas', 'gassy', 'stomach issues', 'stomach problems',
                              'gi issues', 'gi problems', 'tummy issues', 'stomach stuff', 'it', 'this', 'that', 'stuff', 'weird',
                              'off', 'bad', 'rough', 'awful', 'not great', 'the shot', 'injection', 'reaction', 'reactions'}),
    'indication': frozenset({'maintenance', 'maintain', 'maintaining', 'maintenance phase', 'weight', 'lose', 'losing', 'my weight',
                             'sleep', 'relief', 'go', 'blood sugar', 'blood sugars', 'sugar', 'sugars', 'numbers', 'my numbers',
                             'appetite', 'health', 'my health', 'energy', 'gut health', 'stomach', 'it', 'this', 'help',
                             'side effects', 'the side effects', 'symptoms', 'everything'}),
}

GENERIC_AE = re.compile(r'^(?:(?:any|the|other|some|all|all the|my|these|those|bad|major|minor|mild|terrible)\s+)*'
                        r'(?:side[\s-]?e?ffects?|side[\s-]?affects?|sides|symptoms?|issues|problems|reactions?|effects)$', re.I)
DOSE_PHASE = re.compile(r'^(?:(?:the|my|a|her|his)\s+)?(?:(?:maintenance|starting|starter|titration|max(?:imum)?|top|loading|current|'
                        r'lowest|highest|low|high|next)\s+)?dos(?:e|es|age|ing)(?:\s+(?:phase|level))?$'
                        r'|^\d+(?:\.\d+)?\s*(?:mg|ml|mcg|units?|clicks?)$', re.I)
TOLERABILITY = re.compile(r"\b(?:tolerat\w*|couldn'?t\s+handle|can'?t\s+handle|didn'?t\s+agree\s+with)\b", re.I)

# Leading modifiers kept as typed residuals (never dropped). Tried only after the whole surface found nothing decisive.
PREFIX_QUALIFIERS = (
    ('determiner', ('my', 'the', 'her', 'his', 'their', 'our', 'your', 'a', 'an', 'this', 'that', 'these', 'those')),
    ('quantity', ('a little bit of', 'a bit of', 'a lot of', 'lots of', 'tons of', 'a little', 'some', 'little', 'more', 'less')),
    ('severity', ('really bad', 'very bad', 'super bad', 'so much', 'mild', 'mildly', 'light', 'slight', 'slightly', 'bad', 'severe',
                  'severely', 'extreme', 'extremely', 'terrible', 'horrible', 'awful', 'intense', 'major', 'minor', 'crazy',
                  'insane', 'massive', 'huge', 'super', 'really', 'very', 'so')),
    ('frequency', ('constant', 'constantly', 'occasional', 'occasionally', 'frequent', 'frequently', 'daily', 'persistent',
                   'random', 'chronic', 'recurring')),
    ('laterality', ('left', 'right', 'bilateral')),
    ('temporal', ('acute', 'sudden', 'initial', 'early')),
)
# a numeric severity score before the term ("8/10 pain") is a severity residual
SCORE = re.compile(r'(\d{1,2}\s*/\s*10|\d{1,2}\s+out\s+of\s+10)(?:\s+|-)(?=\S)', re.I)
COMPATIBLE_RESIDUALS = frozenset({'determiner', 'quantity', 'severity', 'frequency', 'laterality', 'temporal'})
NEVER_STRIP = frozenset({'light headed', 'lightheaded', 'light-headed'})

CONJUNCTIVE = re.compile(r'\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*|\s*\+\s*|\s+plus\s+', re.I)
DISJUNCTIVE = re.compile(r'\s*/\s*|\s+or\s+', re.I)
# a slash between numbers (8/10, 8 / 10, 1/2) is a score or fraction, never a connector; it is masked (same length)
NUMERIC_SLASH = re.compile(r'(?<=[0-9])(\s*)/(\s*)(?=[0-9])')
MAX_PART_TOKENS = 6

# Purpose relation: only a governing phrase immediately left of a mention of the purpose surface (determiners allowed
# in between) counts. Words elsewhere in the quote ("take", "on", a "for" that governs another noun) never do.
_DET = (r"(?:(?:my|your|her|his|their|our|the|this|that|any|some|more|further|getting|developing|having|really|actually|"
        r"finally|also|just|even|still)\s+)*")
# coordinated siblings between a governor and the mention ("for PCOS and IR" governs IR); "but", "have" etc. stop it
_COORD = r"(?:[\w'’-]+(?:\s+[\w'’-]+){0,3}\s*(?:,\s*(?:and\s+)?|\s+and\s+|\s*&\s*|\s*/\s*|\s+or\s+))*"
_TAIL = r"\s+" + _DET + _COORD + _DET + r"$"
# (signal, direction, pattern, needs_purpose_verb). An infinitival marker ("to", "want to", "help me") states a purpose
# only before a base-form purpose verb ("to lose weight"); before a gerund or a noun "to" is a preposition ("back to
# losing weight", "contributes to weight loss") and states a result, not a purpose.
LOCAL_SIGNALS = (
    ('prevent', None, re.compile(r"(?:prevent(?:s|ed|ing)?|avoid(?:s|ed|ing)?|ward\s+off|keep\s+from|"
                                 r"(?:reduce|lower|cut)(?:s|d|ing)?\s+(?:(?:my|her|his|the|their)\s+)?risk\s+of|"
                                 r"(?:don'?t|do\s+not|won'?t|never)\s+(?:get|develop|end\s+up\s+with))" + _TAIL, re.I), False),
    ('goal', 'decrease', re.compile(r"(?:to|helps?|helped|helping|trying\s+to|want(?:ed|s)?\s+to|need(?:ed|s)?\s+to)\s+(?:me\s+)?"
                                    r"(?:lose|drop|shed|reduce|lower|curb|suppress|quiet|cut)" + _TAIL, re.I), False),
    ('goal', 'maintain', re.compile(r"(?:to|helps?|helped|helping|trying\s+to|want(?:ed|s)?\s+to)\s+(?:me\s+)?(?:maintain|keep)"
                                    + _TAIL, re.I), False),
    ('goal', 'increase', re.compile(r"(?:to|helps?|helped|helping|trying\s+to|want(?:ed|s)?\s+to)\s+(?:me\s+)?(?:gain|increase|boost)"
                                    + _TAIL, re.I), False),
    ('relieve_symptom', None, re.compile(r"(?:eliminat(?:e|es|ed|ing)|reliev(?:e|es|ed|ing)|eas(?:e|es|ed|ing)|sooth(?:e|es|ed|ing)|"
                                         r"get\s+rid\s+of|stop(?:s|ped|ping)?|combat(?:s|ted|ting)?|counteract(?:s|ed|ing)?)"
                                         + _TAIL, re.I), False),
    ('purpose_for', None, re.compile(r"(?:\bfor|\baid(?:s|ed|ing)?(?:\s+(?:in|for|with))?|treat(?:s|ed|ing)?|"
                                     r"to\s+(?:treat|help(?:\s+with)?|manage|control|fix|address|deal\s+with)|helps?(?:\s+with)?|"
                                     r"helped(?:\s+with)?|helping(?:\s+with)?|because\s+of|due\s+to|manag(?:e|es|ed|ing)|"
                                     r"control(?:s|led|ling)?)" + _TAIL, re.I), False),
    ('purpose_marker', None, re.compile(r"(?:(?<!\bused\s)\bto|in\s+order\s+to|trying\s+to|hoping\s+to|want(?:ed|s)?\s+to|"
                                        r"need(?:ed|s)?\s+to|so\s+(?:that\s+)?(?:i|we|she|he|they)\s+(?:can|could|would)|"
                                        r"help(?:s|ed|ing)?(?:\s+me)?|goal\s+(?:is|was)\s+to)" + _TAIL, re.I), True),
    # "my goal is weight loss": the goal statement itself, with a noun or a verb
    ('purpose_marker', None, re.compile(r"\bgoals?\s+(?:is|was|are|were)" + _TAIL, re.I), False),
)
# base-form verbs that can head a purpose infinitive in these records ("to lose weight", "help me sleep")
PURPOSE_VERBS = frozenset({
    'lose', 'loose', 'drop', 'shed', 'reduce', 'lower', 'curb', 'suppress', 'quiet', 'quieten', 'cut', 'maintain', 'keep',
    'gain', 'increase', 'boost', 'treat', 'manage', 'control', 'prevent', 'avoid', 'stop', 'help', 'reach', 'hit', 'slim',
    'regulate', 'improve', 'fix', 'address', 'ease', 'relieve', 'combat', 'counteract', 'reverse', 'kickstart',
    'jumpstart', 'break', 'get', 'stay', 'feel', 'sleep', 'eat', 'fight', 'beat', 'tackle', 'normalize', 'normalise',
    'balance', 'stabilize', 'stabilise', 'bring', 'correct', 'resolve', 'heal', 'cure', 'deal', 'shrink', 'tame', 'calm'})
# A bare "to" that completes an aspect, ability, difficulty or evaluation predicate ("started to lose weight", "continued
# to", "was able to", "hard to") or a causative with its object ("caused me to", "allowed me to") reports a result or a
# capacity, not a purpose; "take Wegovy to", "on it to", "want to", "need to" stay purposes.
COMPLEMENT_HEADS = frozenset({
    'start', 'starts', 'started', 'starting', 'begin', 'begins', 'began', 'begun', 'beginning', 'continue', 'continues',
    'continued', 'continuing', 'proceed', 'proceeded', 'come', 'comes', 'came', 'cease', 'ceased', 'seem', 'seems',
    'seemed', 'appear', 'appears', 'appeared', 'tend', 'tends', 'tended', 'happen', 'happens', 'happened', 'able', 'unable',
    'manage', 'manages', 'managed', 'fail', 'fails', 'failed', 'struggle', 'struggles', 'struggled', 'struggling', 'going',
    'gonna', 'about', 'supposed', 'likely', 'unlikely', 'used', 'get', 'gets', 'got', 'getting', 'hard', 'harder', 'hardest',
    'easy', 'easier', 'easiest', 'difficult', 'impossible', 'possible', 'tough', 'nice', 'good', 'great', 'glad', 'happy',
    'sad', 'enough'})
CAUSATIVES = frozenset({'cause', 'causes', 'caused', 'causing', 'allow', 'allows', 'allowed', 'allowing', 'enable', 'enables',
                        'enabled', 'enabling', 'force', 'forces', 'forced', 'lead', 'leads', 'led', 'push', 'pushed', 'get',
                        'gets', 'got', 'make', 'makes', 'made'})
OBJECTS = frozenset({'me', 'you', 'him', 'her', 'us', 'them', 'people', 'everyone', 'it', 'someone', 'somebody'})
# "grateful for the weight loss", "known for weight loss", "credit it for my weight loss": evaluation or attribution of
# a result, not a purpose of use
NON_PURPOSE_FOR = frozenset({'grateful', 'thankful', 'happy', 'glad', 'excited', 'proud', 'credit', 'credits', 'credited',
                             'known', 'famous', 'responsible', 'blame', 'blamed', 'blames', 'thank', 'thanks', 'thanked',
                             'praise', 'praised', 'sorry', 'apologize', 'apologise'})
# Intent the surface itself states ("lose weight", "weight maintenance", "prevent diabetes").
SURFACE_SIGNALS = (
    ('prevent', None, re.compile(r"\b(?:prevent\w*|avoid\w*|prophyla\w*)\b", re.I)),
    ('goal', 'maintain', re.compile(r"\b(?:maintain\w*|maintenance|keep\s+(?:the\s+|my\s+)?weight\s+off)\b", re.I)),
    ('goal', 'decrease', re.compile(r"\b(?:lose|losing|loose|lost|drop(?:ping)?|shed(?:ding)?|weight\s*loss|slim\w*|"
                                    r"suppress\w*|reduc\w*|curb\w*|quiet\w*)\b", re.I)),
    ('goal', 'increase', re.compile(r"\b(?:gain\w*|put\s+on\s+weight)\b", re.I)),
)
# Intent of a generic purpose relation ("for X") by concept kind; other/unknown kinds stay unclear.
FOR_INTENT = {'condition': 'treat_condition', 'symptom': 'relieve_symptom', 'goal': 'goal', 'finding': 'treat_condition',
              'procedure': 'other_explicit'}
KIND_INTENTS = {'condition': {'treat_condition', 'prevent'}, 'symptom': {'relieve_symptom', 'prevent'},
                'finding': {'treat_condition', 'prevent', 'goal'}, 'goal': {'goal'},
                'procedure': {'other_explicit', 'treat_condition', 'prevent', 'goal'},
                'other': {'other_explicit', 'treat_condition', 'prevent', 'goal', 'relieve_symptom'}}


def _bounded(needle, ignore_case=True):
    pre = r'(?<![A-Za-z0-9])' if needle[:1].isalnum() else ''
    post = r'(?![A-Za-z0-9])' if needle[-1:].isalnum() else ''
    return re.compile(pre + re.escape(needle) + post, re.I if ignore_case else 0)


def mentions(surface, text):
    """Token-bounded, case-insensitive occurrences (start, end) of the surface in a text."""
    needle = surface.strip()
    return [(m.start(), m.end()) for m in _bounded(needle).finditer(text)] if needle else []


def source_support(surface, quotes, record_text=None):
    """own_evidence_literal | own_evidence_casefold | record_literal | not_literal; token boundaries always apply
    ("pain" is not supported by "painful")."""
    needle = surface.strip()
    if not needle:
        return 'not_literal'
    if any(_bounded(needle, ignore_case=False).search(q) for q in quotes):
        return 'own_evidence_literal'
    if any(_bounded(needle).search(q) for q in quotes):
        return 'own_evidence_casefold'
    if record_text and _bounded(needle).search(record_text):
        return 'record_literal'
    return 'not_literal'


def deterministic_support(support):
    return support in ('own_evidence_literal', 'own_evidence_casefold')


def context_required(role, text):
    return fold_key(text) in CONTEXT_REQUIRED[role]


def generic_ae(text):
    return bool(GENERIC_AE.match(fold_key(text)))


def dose_phase(text):
    return bool(DOSE_PHASE.match(fold_key(text)))


def tolerability(text):
    return bool(TOLERABILITY.search(text))


def strip_qualifiers(text):
    """(core text, residuals) with leading modifiers removed; residual offsets are relative to `text`."""
    if fold_key(text) in NEVER_STRIP:
        return text, []
    residuals, start = [], 0
    while True:
        rest = text[start:]
        lead = len(rest) - len(rest.lstrip())
        hit = None
        score = SCORE.match(rest[lead:])
        if score:
            hit = ('severity', start + lead, start + lead + len(score.group(1)), start + lead + score.end())
        for kind, words in (() if hit else PREFIX_QUALIFIERS):
            for word in sorted(words, key=len, reverse=True):
                m = re.match(r'(' + re.escape(word) + r')(?:\s+|-)(?=\S)', rest[lead:], re.I)
                if m:
                    hit = (kind, start + lead, start + lead + len(m.group(1)), start + lead + m.end())
                    break
            if hit:
                break
        if not hit:
            break
        kind, s, e, nxt = hit
        residuals.append({'kind': kind, 'text': text[s:e], 'start': s, 'end': e})
        start = nxt
    return text[start:].strip(), residuals


def split_components(text):
    """Return (reading, parts): reading is whole | conjunctive | ambiguous | descriptive; parts are (start, end).
    Connectors are found on a copy where numeric slashes are masked, so offsets stay those of `text`."""
    if len(tokens(text)) <= 1:
        return 'whole', [(0, len(text))]
    masked = NUMERIC_SLASH.sub(lambda m: m.group(1) + '\x00' + m.group(2), text)
    disj = bool(DISJUNCTIVE.search(masked))
    conj = bool(CONJUNCTIVE.search(masked))
    if not disj and not conj:
        return 'whole', [(0, len(text))]
    splitter = re.compile(f'(?:{CONJUNCTIVE.pattern})|(?:{DISJUNCTIVE.pattern})', re.I) if disj else CONJUNCTIVE
    parts, pos = [], 0
    for m in splitter.finditer(masked):
        if m.start() > pos:
            parts.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        parts.append((pos, len(text)))
    parts = [(s + (len(text[s:e]) - len(text[s:e].lstrip())), e - (len(text[s:e]) - len(text[s:e].rstrip())))
             for s, e in parts if text[s:e].strip()]
    if len(parts) < 2 or any(len(tokens(text[s:e])) > MAX_PART_TOKENS for s, e in parts):
        return 'descriptive', [(0, len(text))]
    return ('ambiguous' if disj else 'conjunctive'), parts


# A medication noun right after the surface ("adhd medication", "diabetes meds") states a purpose relation.
RIGHT_PURPOSE = re.compile(r"^\s*(?:medications?|meds?|medicines?|drugs?|pills?|treatments?|therapy|prescriptions?|rx)\b", re.I)
# The surface used as a dose name ("maintenance dose", "maintenance dosage"): a dosing phase, not a purpose.
RIGHT_DOSE = re.compile(r"^\s*(?:dos(?:e|es|age|ing)|level)\b", re.I)


def dose_use(surface, quotes):
    """True when every mention of the surface in its own quotes names a dose ("a topped out maintenance dose")."""
    hits = [quote[end:end + 20] for quote in quotes for _, end in mentions(surface, quote)]
    return bool(hits) and all(RIGHT_DOSE.match(h) for h in hits)


# --- relation scope ---------------------------------------------------------------------------------------------------
# A purpose relation belongs to the carrier only when the source says so. Clause boundaries: terminal punctuation before
# a capitalised / quoted start or the end ("Oz. for diabetes" is not a boundary), a semicolon, a line break, a spaced
# dash. Sub-clauses (", so", "but", ...) bound the predicate that governs the cue.
CLAUSE_BREAK = re.compile(r"[.!?]+(?=\s+[A-Z0-9\"“'‘(\[]|\s*$)|;|\n|\s[-–—]+\s")
SUB_BREAK = re.compile(r",\s*(?:so|but|and\s+then|then)\b|\bbut\b|\bthough\b|\bhowever\b", re.I)
# paying, allowing or qualifying for a drug is a reason for coverage, not a use relation
COVERAGE = re.compile(r"\b(?:insurance|insurer|covers?|covered|covering|coverage|approv(?:e|es|ed|al|ing)|den(?:y|ies|ied|ial)|"
                      r"prior\s+auth\w*|pre-?auth\w*|authori[sz]\w*|eligib\w*|qualif(?:y|ies|ied|ying)|formulary|reimburs\w*)\b", re.I)
USE = re.compile(r"\b(?:tak(?:e|es|ing|en)|took|us(?:e|es|ed|ing)|start(?:s|ed|ing)?|prescrib\w*|inject\w*|switch\w*|"
                 r"put\s+(?:me|her|him|them|us)\s+on|(?:am|i['’]?m|was|been|be|is|are|were|get|got|stay(?:ed|ing)?)\s+on|"
                 r"tr(?:y|ies|ied|ying)|need(?:s|ed)?|recommend\w*|suggest\w*)\b", re.I)
NON_MEDICATION = re.compile(r"\b(?:diet(?:s|ing)?|keto|fasting|exercis\w*|workouts?|gym|surgery|surgeries|sleeve|bypass|clinic|"
                            r"program(?:me)?s?|coach(?:es|ing)?|nutritionist|dietitian|therapist|counsel\w*)\b", re.I)
# A provider, consultation or service noun that the purpose cue attaches to: directly ("my doctor for weight loss",
# "an appointment for my diabetes") or as the fronted object of a relative clause ("the doc I use for weight loss", "the
# clinic my sister goes to for weight loss"). The purpose then belongs to that head, not to a drug. A person who only
# wants or asks something ("my doctor wants me to lose weight") is not such a governor.
_NON_DRUG_HEADS = (r"doc|docs|doctors?|dr|physicians?|providers?|pcp|gp|np|nurses?|practitioners?|endos?|endocrinologists?|"
                   r"specialists?|surgeons?|derm|dermatologists?|psychiatrists?|psych|gyno|gynecologists?|obgyn|cardiologists?|"
                   r"dietitians?|dieticians?|nutritionists?|coach(?:es)?|trainers?|therapists?|counsell?ors?|team|office|"
                   r"practice|clinics?|cent(?:er|re)s?|program(?:me)?s?|services?|compan(?:y|ies)|telehealth|telemedicine|apps?|"
                   r"pharmac(?:y|ies)|pharmacists?|hospitals?|appointments?|visits?|consult(?:ation)?s?|check-?ups?|"
                   r"follow-?ups?|sessions?")
_REL_SUBJECT = r"(?:i|we|you|she|he|they|(?:my|the|a|an|our|his|her|their|your)\s+[\w'’-]+(?:\s+[\w'’-]+)?)"
_REL_AUX = (r"(?:(?:have|has|had|am|is|are|was|were|be|been|being|do|does|did|currently|usually|now|also|"
            r"just|still|always|regularly|recently|finally|first|eventually)\s+)*")
_REL_VERB = (r"(?:us(?:e|es|ed|ing)|see(?:s|n|ing)?|saw|go(?:es|ing)?\s+to|went\s+to|gone\s+to|visit(?:s|ed|ing)?|ha(?:ve|s|d)|"
             r"f(?:ou|i)nd|work(?:s|ed|ing)?\s+with|switch(?:ed|ing)?\s+to|start(?:s|ed|ing)?\s+(?:with|seeing)|"
             r"pa(?:y|ys|id|ying)|sign(?:ed)?\s+up\s+(?:with|for)|join(?:s|ed)?|cho(?:se|ose)|pick(?:ed)?|liked?|trust|love|"
             r"recommend(?:s|ed)?|consult(?:s|ed)?|referr?(?:ed)?\s+to|spoke\s+(?:to|with)|talk(?:s|ed)?\s+(?:to|with)|"
             r"ask(?:s|ed)?|book(?:ed)?|schedul(?:e|ed)|attend(?:s|ed)?|follow(?:s|ed)?\s+up\s+with)")
NON_DRUG_GOVERNOR = re.compile(
    r"\b(?:" + _NON_DRUG_HEADS + r")\s+(?:(?:that|who|whom|which)\s+)?" + _REL_SUBJECT + r"(?:['’](?:ve|m|d|re|s|ll))?\s+" + _REL_AUX
    + _REL_VERB + r"\s*$"
    r"|\b(?:" + _NON_DRUG_HEADS + r")(?:['’]s)?\s*$", re.I)
# the surface modifies another head ("weight loss surgery", "a weight loss clinic"): its left cue governs that head
RIGHT_NON_MED = re.compile(r"^\s*(?:surgery|surgeries|clinics?|cent(?:er|re)s?|program(?:me)?s?|coach(?:es)?|doctors?|"
                           r"specialists?|communit(?:y|ies)|sub(?:reddit)?s?|groups?|apps?|challenges?)\b", re.I)
# text that joins two drug names into one governing group ("Mounjaro and metformin", "Wegovy (semaglutide)")
JOINER = re.compile(r"(?:[\s,/&+()]|\band\b|\bor\b|\bplus\b|\bwith\b|\bas\s+well\s+as\b|\balong\s+with\b|"
                    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|units?)\b|\b(?:er|xr|sr)\b)*", re.I)
SCOPE_PRIORITY = ('other_drug', 'ambiguous_drug', 'coverage', 'non_medication')


@lru_cache(maxsize=4096)
def _bounded_cached(needle):
    return _bounded(needle)


def clause_bounds(text, start, end):
    """(clause start, clause end) of the clause holding text[start:end]. Breaks are found on the whole text so the
    look-ahead sees what follows a period ("Oz. for" is not a break)."""
    cs, ce = 0, len(text)
    for m in CLAUSE_BREAK.finditer(text):
        if m.end() <= start:
            cs = m.end()
        elif m.start() >= end:
            ce = m.start()
            break
    return cs, ce


def scan_drugs(text, names):
    """Non-overlapping token-bounded mentions (start, end, class) of scope names in text; longer names win."""
    taken, out = [], []
    for name, cls in names:
        for m in _bounded_cached(name.strip()).finditer(text):
            s, e = m.start(), m.end()
            if not any(s < te and ts < e for ts, te in taken):
                taken.append((s, e))
                out.append((s, e, cls))
    return sorted(out)


def _groups(text, hits):
    """Drug mentions joined by JOINER text form one group (list of hits, in order)."""
    groups = []
    for hit in hits:
        if groups and JOINER.fullmatch(text[groups[-1][-1][1]:hit[0]]):
            groups[-1].append(hit)
        else:
            groups.append([hit])
    return groups


def _group_class(group):
    classes = {h[2] for h in group}
    return 'carrier' if 'carrier' in classes else ('related' if 'related' in classes else 'other')


APPOSITION = re.compile(r"\s*(?:[,(:\-–—]\s*)?(?:(?:is|was|as|being|aka|i\.e\.)\s+)?(?:(?:my|the|a|an|her|his|their|our|your|this)\s+)?",
                        re.I)


def _predicate(sub):
    """coverage | non_medication | None: the nearest predicate before a purpose cue within its sub-clause. A use verb
    after an insurance word ("insurance covers it and I take it for ...") makes it a use relation again."""
    last = lambda pattern: max((m.end() for m in pattern.finditer(sub)), default=-1)
    coverage, use, nonmed = last(COVERAGE), last(USE), last(NON_MEDICATION)
    if coverage > use:
        return 'coverage'
    if nonmed > max(use, coverage):
        return 'non_medication'
    return None


def relation_scope(quote, cue_start, mention_start, mention_end, names, quotes, nominal=False):
    """Who governs one purpose relation: 'carrier', or why it cannot be used for this carrier: other_drug (another named
    drug governs it), ambiguous_drug (no unique governor), coverage (insurance / approval / eligibility reason only),
    non_medication (a diet, surgery, clinic ... governs it). Without scope names only the predicate is checked."""
    cs, ce = clause_bounds(quote, cue_start, mention_end)
    left = quote[cs:cue_start]
    sub = left
    for m in SUB_BREAK.finditer(left):
        sub = left[m.end():]
    predicate = _predicate(sub)
    if predicate == 'non_medication' and names:
        # a drug named after the non-medication head is the nearer governor ("keto and Ozempic for weight loss")
        nonmed_end = max(m.end() for m in NON_MEDICATION.finditer(sub))
        if scan_drugs(sub[nonmed_end:], names):
            predicate = None
    if predicate is not None:
        return predicate
    if not nominal and NON_DRUG_GOVERNOR.search(sub):
        return 'non_medication'  # the cue belongs to a provider / consultation / service, not to a drug
    if not names:
        return 'carrier'
    if nominal:  # "<purpose> medication": the medication must be the carrier itself or stand in apposition to it
        clause = quote[cs:ce]
        rel_s, rel_e = mention_start - cs, mention_end - cs
        hits = scan_drugs(clause, names)
        inside = [h for h in hits if h[0] <= rel_s and rel_e <= h[1]]
        if inside:
            return 'carrier' if inside[0][2] == 'carrier' else ('ambiguous_drug' if inside[0][2] == 'related' else 'other_drug')
        noun = RIGHT_PURPOSE.match(clause[rel_e:])
        span_end = rel_e + noun.end() if noun else rel_e
        near = [h for h in hits if h[1] <= rel_s and APPOSITION.fullmatch(clause[h[1]:rel_s])] + \
               [h for h in hits if h[0] >= span_end and APPOSITION.fullmatch(clause[span_end:h[0]])]
        if not near:
            return 'ambiguous_drug'
        cls = {h[2] for h in near}
        return 'carrier' if cls == {'carrier'} else ('other_drug' if 'carrier' not in cls and 'other' in cls else 'ambiguous_drug')
    hits_left = scan_drugs(left, names)
    if hits_left:
        cls = _group_class(_groups(left, hits_left)[-1])
        return {'carrier': 'carrier', 'related': 'ambiguous_drug', 'other': 'other_drug'}[cls]
    hits_right = scan_drugs(quote[cue_start:ce], names)
    if hits_right:
        classes = {h[2] for h in hits_right}
        if classes == {'carrier'}:
            return 'carrier'
        return 'other_drug' if 'carrier' not in classes and 'related' not in classes else 'ambiguous_drug'
    # no drug named in the clause: unique only if no other drug is named anywhere in this node's evidence
    for q in quotes:
        for group in _groups(q, scan_drugs(q, names)):
            if any(h[2] == 'other' for h in group) or (_group_class(group) == 'related'):
                return 'ambiguous_drug'
    return 'carrier'


def _cue_is_purpose(quote, cue_start, cue_text, signal):
    """False for a cue that only looks like a purpose relation: an evaluative / attributive "for", or a bare "to" that
    completes a result, ability, aspect or causative predicate."""
    cue = fold_key(cue_text)
    before = tokens(quote[max(0, cue_start - 60):cue_start])
    if signal == 'purpose_for' and re.match(r'for\b', cue) and any(w in NON_PURPOSE_FOR for w in before[-3:]):
        return False
    if re.match(r'to\b', cue) and before:
        if before[-1] in COMPLEMENT_HEADS:
            return False
        if len(before) >= 2 and before[-2] in CAUSATIVES and before[-1] in OBJECTS:
            return False
    return True


def local_signals(surface, quotes, names=None):
    """(local, surface_signals, mentions, out_of_scope): local are (signal, direction) found immediately left (or, for a
    medication noun, right) of a mention of the surface in its own quotes and governed by the carrier; out_of_scope
    lists why other relations were not used; surface_signals are what the surface itself states."""
    found, local, dropped = 0, [], []
    for quote in quotes:
        for start, end in mentions(surface, quote):
            found += 1
            if RIGHT_NON_MED.match(quote[end:end + 30]):
                continue  # the mention modifies another head; no relation of its own
            base = max(0, start - 80)
            left = quote[base:start]
            words = tokens(quote[start:end])
            first = words[0] if words else ''
            for signal, direction, pattern, needs_verb in LOCAL_SIGNALS:
                m = pattern.search(left)
                if m is None or (needs_verb and first not in PURPOSE_VERBS):
                    continue
                cue = base + m.start()
                if not _cue_is_purpose(quote, cue, m.group(0), signal):
                    continue
                scope = relation_scope(quote, cue, start, end, names, quotes)
                if scope == 'carrier':
                    local.append((signal, direction))
                else:
                    dropped.append(scope)
            if RIGHT_PURPOSE.match(quote[end:end + 30]):
                scope = relation_scope(quote, start, start, end, names, quotes, nominal=True)
                if scope == 'carrier':
                    local.append(('purpose_for', None))
                else:
                    dropped.append(scope)
    own = [(signal, direction) for signal, direction, pattern in SURFACE_SIGNALS if pattern.search(surface)]
    return local, own, found, dropped


def decide_intent(concept_kind, policy, surface, quotes, inherent_direction=None, names=None):
    """(intent, direction, basis, conflict) from explicit relations of this purpose only (MTS:PTC 3.26.1: the words
    "weight loss" alone do not tell a goal from a problem being treated).

    A relation is a governing phrase right before a mention of the surface in its own quotes: a purpose marker
    ("for", "to", "want to", ...) confirms the intent the surface states ("to lose weight") or, for a surface stating
    none, the kind's intent ("for PCOS" -> treat_condition); a verb ("prevent", "to lose") states it directly. The
    surface wording without such a relation sets nothing. Contradictions (prevent vs goal, opposite directions, a
    direction against the concept's, an intent the kind cannot take) give unclear with conflict=True. A relation
    governed by another drug, by no unique drug, by an insurance / approval predicate or by a non-medication head is
    not this carrier's (`names`: scope names of the carrier, see source.scope_names); when only such relations exist
    the basis is relation_out_of_scope:<why>."""
    mode = (policy or {}).get('mode', 'context_required')
    local, own, found, dropped = local_signals(surface, quotes, names)
    if not found:
        return 'unclear', None, 'surface_not_in_evidence', False
    kind = concept_kind or 'other'
    intents = set()
    for signal, direction in local:
        if signal in ('purpose_for', 'purpose_marker'):
            if own:
                intents.update((s, d if s == 'goal' else None) for s, d in own)
            elif signal == 'purpose_for' or kind in ('goal', 'symptom', 'condition', 'finding'):
                intent = FOR_INTENT.get(kind)
                if intent is not None:
                    intents.add((intent, inherent_direction if intent == 'goal' else None))
        else:
            intents.add((signal, direction if signal == 'goal' else None))
    explicit = {i for i, _ in intents}
    goal_dirs = {d for i, d in intents if i == 'goal' and d}
    if len(explicit) > 1 or len(goal_dirs) > 1:
        return 'unclear', None, 'conflicting_relations', True
    if inherent_direction is not None and goal_dirs and goal_dirs != {inherent_direction}:
        return 'unclear', None, 'direction_contradicts_concept', True
    if explicit and not explicit <= KIND_INTENTS.get(kind, KIND_INTENTS['other']):
        return 'unclear', None, 'intent_incompatible_with_concept', True
    if mode == 'context_required':
        return 'unclear', None, 'context_required', False
    if not explicit:
        if dropped:
            return 'unclear', None, 'relation_out_of_scope:' + next(p for p in SCOPE_PRIORITY if p in dropped), False
        return 'unclear', None, 'no_local_purpose_relation', False
    intent = next(iter(explicit))
    direction = None
    if intent == 'goal':
        direction = inherent_direction or (next(iter(goal_dirs)) if goal_dirs else 'unspecified')
    if mode == 'surface_cue':
        wanted_intent, wanted_dir = policy['intent'], policy.get('direction')
        if intent != wanted_intent or (wanted_dir is not None and direction != wanted_dir):
            return 'unclear', None, 'surface_cue_not_confirmed', True
        return intent, direction, 'surface_cue_confirmed', False
    return intent, direction, 'local_relation', False
