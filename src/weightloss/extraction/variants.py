"""Experimental extraction specs (results/extraction-optimization-2026-09-27). A spec is frozen once a run used it;
changes get a new name, so earlier runs stay collectable and comparable. Every spec's answers are validated and stored
through the canonical wire model, so quality comparisons use one storage format.

compact-p2   prompt v2 on the canonical wire. Hypotheses from the dev baseline (compact-v1): purposes were over-extracted
             from outcomes/benefits/plans; clear references ("this medicine", "the pill") were left unspecified instead
             of the named drug or WebMD drug_field; WebMD condition_field was not used as the stated purpose; symptoms
             presented as side effects got attribution not_stated; symptoms were attached to a drug taken to relieve them
             when the causing drug was unnamed; limited none-claims became not_occurred items with a severity.
compact-p2n  the same rules on a nested wire (use and event details as optional objects) to cut output tokens; answers
             are flattened deterministically to the canonical wire.
compact-p3 / compact-p3n  prompt v3 (canonical / nested wire). From the p2/p2n dev errors and Codex review: WebMD
             condition_field is the review's associated condition (subject and use only from the text); a goal stated as the
             reason for use (e.g. quieting food noise) is a purpose, results or listed conditions are not; relief medicines
             never receive the symptoms they relieve, an unnamed causing drug gets an unspecified surface; effect must be a
             symptom (not foods, worries, dosing mistakes, wearing off, benefits or generic advice); "very minimal" still
             occurred; suspected other causes make attribution uncertain; limited none-claims stay no_side_effects; repeated
             phrases need a longer quote. (Registered but never run: superseded by p4 before execution.)
compact-p4 / compact-p4n  prompt v4 = v3 plus, from Codex's smoke review of p2/p2n: a question about the cause keeps the
             occurrence of a symptom the text says happens; severity is never inferred from descriptive words (rough, awful,
             bad); items of the same use episode, questions and hypotheticals included, all repeat that episode's use.
compact-p5   prompt v5 on the canonical wire = v4 plus: indication and effect name the condition or symptom, never a pronoun
             ("that", "it") standing for it; a symptom described as minimal ("it's been very minimal") occurred. The nested
             wire is not continued: under v4 it saved ~3% output tokens, showed no quality gain beyond run-to-run
             variation and produced the only two run-on repetition truncations.
compact-p6   prompt v5 + one closing sentence (answer once with the final JSON; no drafts, commentary or repeated answers)
             + request text.verbosity "low" (official output-length control; the API echoed "medium" by default). Codex
             traced both p4n truncations to a single commentary-phase message repeating the complete JSON until
             max_output_tokens.
compact-p7   p6 (flat wire, verbosity low, answer once) with three rule clarifications from Codex's reading of p6 outputs,
             aimed at field-constraint confusion rather than single records: use values belong to one subject's own use
             episode (general statements and other people never inherit the writer's use); questions or remarks about
             people in general stay subject general / not_asserted unless the writer says it happens to them; a relief
             medicine keeps its own purpose item and never receives the symptom it relieves (the unnamed causing drug does).
compact-p8   concise rewrite (about half of v7's length) with the same information needs, stating the two central
             distinctions directly (which drug causes a symptom vs a medicine taken to relieve it; the writer's own
             experience vs statements about people in general) to test whether the long accumulated rules interfere.
             Flat canonical wire, text.verbosity low.
compact-s1 / compact-s2  structural candidates for reasoning none (see the block at the end of this module): evidence
             first, cause + timing instead of attribution (derived deterministically), relief flag, purpose basis, event
             object; s2 also states drug, subject and use once per use context.
"""
from pathlib import Path
from typing import Optional

from .compact import (SCHEMA_VERSION, Consequence, DoseText, Handle, Label, Quotes, ShortText, Spec, StatementKind, Subject,
                      Surface, UseStatus, Wire, WireDrug, WireExtraction, WireNoSideEffect, WirePurpose, WireSideEffect, register)
from typing import Literal

_COMMON = """You extract drug statements from one social-media post or health review for a knowledge graph. Use only the given text: no outside knowledge, no guesses. Nothing is pre-filled and nothing has a default; never assume the writer, current use, an affirmed purpose or an attributed effect.

Drugs (every key present; null when the text does not say; [] when a group is empty):
- drugs: one entry per medication that an item below refers to. handle: d1, d2, ... in order of first mention. surface: the drug's name as written in the input (for a WebMD review the drug_field value may be used). ref: "name" (brand or generic name), "class" (a drug class only, e.g. "GLP-1") or "unspecified".
- Resolve references such as "it", "this med", "this one", "the pill", "the shot" or "these meds" to the named drug they clearly refer to and use that drug's handle. In a WebMD review, drug_field is the reviewed drug and the review is about it unless another drug is named. Use ref "unspecified" only for a drug that is never named for that referent in the input; then surface is the words used (e.g. "my shot"). Never upgrade a class or unspecified reference to a brand, and never replace one named drug with another.
- A medicine taken to relieve a symptom (e.g. Zofran for nausea) is not the cause of that symptom unless the text says so; attach the symptom to the drug the text links it to, or to an unspecified drug when that drug is never named.

Groups:
- purposes: a condition or goal the text presents as what a drug is used, prescribed or wanted for, including a condition it is said to treat or help (e.g. "for my type 2 diabetes", "prescribed due to PCOS", "I started it to lose weight", "it helps my inflammation"). In a WebMD review, condition_field (unless it is "Other") is the condition the reviewer takes the drug in drug_field for: add it as a report with polarity affirmed, quoting condition_field. Not purposes: weight or lab results, reduced appetite, cravings or food noise, dosing or diet plans, and other outcomes or benefits. indication: the condition or goal as written.
- side_effects: unwanted effects the text links to a drug, whether they occurred, did not occur, are uncertain, feared or asked about. effect: the symptom as written. The intended appetite reduction, fullness, fewer cravings or less food noise are not side effects unless the text presents them as a problem (e.g. "I can't eat enough"); neither are feelings about injecting, costs or intended weight loss.
- no_side_effects: explicit statements that a drug caused no side effects ("no side effects", "haven't had any side effects"). "Tolerated it well" or "doing fine" is not explicit. scope_text: the limiting words as written when the claim is narrower than none at all (e.g. "no major side effects", "none so far"), else null; never widen the scope, and never also add the claim as a side_effects item or give it a severity. A specific negated symptom ("no nausea") is a side_effects item with occurrence "not_occurred".
- surface, indication and effect are copied character for character from the input (same spelling and capitalization), never normalized or translated.

Item fields (independent of each other; never fill one from another):
- drug: the handle of the drug the statement is about.
- subject: "self" (the writer), "other" (a specific other person), "general" (people in general), or null when unclear. subject_label: a short label from the text only when subject is "other" (e.g. "sister"), else null.
- statement_kind: "report" (declarative), "question", or "hypothetical" (planned, considered, feared or conditional).
- polarity (purposes): "affirmed", "negated" or "uncertain"; null only for a pure question that asserts nothing. A report always has a polarity.
- occurrence (side effects): "occurred", "not_occurred", "uncertain", or "not_asserted" (no evidence either way, e.g. a pure fear or a question about a future effect). "I have nausea; could Ozempic be causing it?" is occurred + question + uncertain attribution.
- attribution (side effects; the author's view, not medical causality): "attributed" when the text presents the effect as caused by or a side effect of the drug ("side effects like", "gave me", "makes me", "from the shot", "caused"); "temporal_only" when only timing links them ("after each shot", "since starting"); "uncertain" when the author wonders; "other_cause" when the author blames something else; "not_stated" only when the effect is mentioned without any link to the drug.
- use_status: "current", "past", "planned", "considering", "not_using" or "unknown" as stated for this drug and subject; null when the text does not say. Planned or considering is not actual use.
- dose: the general dose as written (e.g. "0.5 mg"). duration: the total time on the drug as written (e.g. "six weeks"); it never fills onset or event_duration. Repeat use_status, dose and duration on every item of the same use episode (same person, same drug, explicitly the same episode); keep different episodes apart.
- severity only from a severity word in the text (mild or slight -> "mild"; moderate -> "moderate"; severe, extreme or intense -> "severe"), else null. event_dose (a dose the text ties to the event), onset (e.g. "two weeks after stopping"), event_duration, clinical_consequences ("medical_visit", "emergency_visit", "hospitalization", "other") and outcome ("resolved", "ongoing", "unknown"): only when stated for that effect. With occurrence "not_occurred" or "not_asserted" all of them stay null or [].
- quotes: 1 to 8 passages copied exactly from the input, character for character, that support the item and its fields; each long enough to occur only once in the input. Do not paraphrase, abbreviate with "..." or join separate passages.

If the input has no purpose, side effect or no-side-effect statement, return empty groups."""

PROMPT_V2 = _COMMON

COMPACT_P2 = register(Spec('compact-p2', PROMPT_V2, WireExtraction, SCHEMA_VERSION, 'compact-prompt-2.0.0'))


class UseN(Wire):
    status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]


class EventN(Wire):
    severity: Optional[Literal['mild', 'moderate', 'severe']]
    event_dose: Optional[DoseText]
    onset: Optional[ShortText]
    event_duration: Optional[ShortText]
    clinical_consequences: list[Consequence]
    outcome: Optional[Literal['resolved', 'ongoing', 'unknown']]


class PurposeN(Wire):
    drug: Handle
    indication: Surface
    subject: Subject
    subject_label: Optional[Label]
    statement_kind: StatementKind
    polarity: Optional[Literal['affirmed', 'negated', 'uncertain']]
    use: Optional[UseN]
    quotes: Quotes


class SideEffectN(Wire):
    drug: Handle
    effect: Surface
    subject: Subject
    subject_label: Optional[Label]
    statement_kind: StatementKind
    occurrence: Literal['occurred', 'not_occurred', 'uncertain', 'not_asserted']
    attribution: Literal['attributed', 'temporal_only', 'uncertain', 'other_cause', 'not_stated']
    use: Optional[UseN]
    event: Optional[EventN]
    quotes: Quotes


class NoSideEffectN(Wire):
    drug: Handle
    subject: Subject
    subject_label: Optional[Label]
    scope_text: Optional[ShortText]
    use: Optional[UseN]
    quotes: Quotes


class ExtractionN(Wire):
    drugs: list[WireDrug]
    purposes: list[PurposeN]
    side_effects: list[SideEffectN]
    no_side_effects: list[NoSideEffectN]


def _use(item):
    u = item.use
    return {'use_status': u.status if u else None, 'dose': u.dose if u else None, 'duration': u.duration if u else None}


def nested_to_canonical(answer):
    """Deterministic flattening of the nested wire to the canonical wire dict (absent objects become nulls / [])."""
    out = {'drugs': [d.model_dump() for d in answer.drugs], 'purposes': [], 'side_effects': [], 'no_side_effects': []}
    for p in answer.purposes:
        out['purposes'].append({'drug': p.drug, 'indication': p.indication, 'subject': p.subject, 'subject_label': p.subject_label,
                                **_use(p), 'statement_kind': p.statement_kind, 'polarity': p.polarity, 'quotes': p.quotes})
    for s in answer.side_effects:
        e = s.event
        out['side_effects'].append({'drug': s.drug, 'effect': s.effect, 'subject': s.subject, 'subject_label': s.subject_label,
                                    **_use(s), 'statement_kind': s.statement_kind, 'occurrence': s.occurrence,
                                    'attribution': s.attribution, 'severity': e.severity if e else None,
                                    'event_dose': e.event_dose if e else None, 'onset': e.onset if e else None,
                                    'event_duration': e.event_duration if e else None,
                                    'clinical_consequences': list(e.clinical_consequences) if e else [],
                                    'outcome': e.outcome if e else None, 'quotes': s.quotes})
    for n in answer.no_side_effects:
        out['no_side_effects'].append({'drug': n.drug, 'subject': n.subject, 'subject_label': n.subject_label,
                                       'scope_text': n.scope_text, **_use(n), 'quotes': n.quotes})
    return out


PROMPT_V2N = _COMMON.replace(
    "- use_status: ", "- use: an object {status, dose, duration} for the drug's use by this subject, or null when the text states none of them. status (use_status): ").replace(
    "- dose: the general dose", "- dose (inside use): the general dose").replace(
    "Repeat use_status, dose and duration on every item", "Repeat use on every item").replace(
    "- severity only from", "- event: an object {severity, event_dose, onset, event_duration, clinical_consequences, outcome} for that effect, or null when none of them is stated; it is always null when occurrence is \"not_occurred\" or \"not_asserted\". Inside event: severity only from").replace(
    " With occurrence \"not_occurred\" or \"not_asserted\" all of them stay null or [].", "")

COMPACT_P2N = register(Spec('compact-p2n', PROMPT_V2N, ExtractionN, 'compact-wire-nested-1.0.0', 'compact-prompt-2.0.0-nested',
                            schema_name='weightloss_compact_extraction_nested', to_canonical=nested_to_canonical))


def _v3(text):
    reps = [
        ("In a WebMD review, drug_field is the reviewed drug and the review is about it unless another drug is named.",
         "In a WebMD review, drug_field is the reviewed drug and the review is about it unless another drug is named."),
        ("- A medicine taken to relieve a symptom (e.g. Zofran for nausea) is not the cause of that symptom unless the text says so; attach the symptom to the drug the text links it to, or to an unspecified drug when that drug is never named.",
         "- A medicine taken to relieve symptoms (e.g. Zofran, Pepto, Imodium) is not their cause unless the text says so: \"throwing up even after taking zofran\" is a side effect of the drug taken for weight or diabetes, not of Zofran. When that causing drug is never named, create an unspecified drug whose surface is the words that refer to it (e.g. \"the shot\", \"my dose\", \".25\")."),
        ("- purposes: a condition or goal the text presents as what a drug is used, prescribed or wanted for, including a condition it is said to treat or help (e.g. \"for my type 2 diabetes\", \"prescribed due to PCOS\", \"I started it to lose weight\", \"it helps my inflammation\"). In a WebMD review, condition_field (unless it is \"Other\") is the condition the reviewer takes the drug in drug_field for: add it as a report with polarity affirmed, quoting condition_field. Not purposes: weight or lab results, reduced appetite, cravings or food noise, dosing or diet plans, and other outcomes or benefits. indication: the condition or goal as written.",
         "- purposes: what a drug is used, prescribed or wanted for, as the text states it: a condition or goal given as the reason for use, or a condition the drug is said to treat or help (e.g. \"for my type 2 diabetes\", \"prescribed due to PCOS\", \"I started it to lose weight\", \"I take it to quiet the food noise\", \"it helps my inflammation\"). indication: that condition or goal as written (e.g. \"type 2 diabetes\", \"lose weight\"), never a result, amount, duration or dose. Not purposes: results by themselves (e.g. \"lost 9 lbs\", \"A1C is 5.5\", \"the food noise is gone\"), conditions that are only listed (e.g. \"I also have PCOS\"), and dosing or diet plans. In a WebMD review, condition_field (unless it is \"Other\") is the condition the review is associated with for the drug in drug_field: add it as a purpose (report, polarity affirmed) quoting condition_field; take subject and use_status only from the review text, and follow the text if it negates the condition or is about another person."),
        ("effect: the symptom as written. The intended appetite reduction, fullness, fewer cravings or less food noise are not side effects unless the text presents them as a problem (e.g. \"I can't eat enough\"); neither are feelings about injecting, costs or intended weight loss.",
         "effect: the symptom or bodily change as written (e.g. \"nausea\", \"hair thinning\", \"red at the injection spot\"); only when no symptom is named, the words used for the side effects (e.g. \"side effects\"). The intended appetite reduction, fullness, fewer cravings or less food noise are not side effects unless the text presents them as a problem (e.g. \"I can't eat enough\"). Not side effects: foods or remedies, emotions or worries (e.g. \"nervous\"), dosing mistakes, the drug wearing off, benefits, intended weight loss, and side effects that are only asked about or mentioned in general advice without a named symptom."),
        ("scope_text: the limiting words as written when the claim is narrower than none at all (e.g. \"no major side effects\", \"none so far\"), else null;",
         "scope_text: the limiting words as written when the claim is narrower than none at all (e.g. \"no major side effects\", \"none so far\"; \"I haven't had the severe side effects others have\" -> scope_text \"the severe side effects\"), else null;"),
        ("\"I have nausea; could Ozempic be causing it?\" is occurred + question + uncertain attribution.",
         "\"I have nausea; could Ozempic be causing it?\" is occurred + question + uncertain attribution. A \"very minimal\", \"mild\" or \"a bit of\" symptom occurred."),
        ("\"uncertain\" when the author wonders;", "\"uncertain\" when the author wonders or also suspects another cause (e.g. \"maybe the needle\");"),
        ("each long enough to occur only once in the input.", "each long enough to occur only once in the input (if a phrase repeats, quote a longer passage around the intended occurrence)."),
    ]
    for a, b in reps:
        if a not in text:
            raise ValueError('prompt v3 replacement target missing: ' + a[:60])
        text = text.replace(a, b)
    return text


PROMPT_V3 = _v3(PROMPT_V2)
COMPACT_P3 = register(Spec('compact-p3', PROMPT_V3, WireExtraction, SCHEMA_VERSION, 'compact-prompt-3.0.0'))
PROMPT_V3N = _v3(_COMMON).replace(
    "- use_status: ", "- use: an object {status, dose, duration} for the drug's use by this subject, or null when the text states none of them. status (use_status): ").replace(
    "- dose: the general dose", "- dose (inside use): the general dose").replace(
    "Repeat use_status, dose and duration on every item", "Repeat use on every item").replace(
    "- severity only from", "- event: an object {severity, event_dose, onset, event_duration, clinical_consequences, outcome} for that effect, or null when none of them is stated; it is always null when occurrence is \"not_occurred\" or \"not_asserted\". Inside event: severity only from").replace(
    " With occurrence \"not_occurred\" or \"not_asserted\" all of them stay null or [].", "")
COMPACT_P3N = register(Spec('compact-p3n', PROMPT_V3N, ExtractionN, 'compact-wire-nested-1.0.0', 'compact-prompt-3.0.0-nested',
                            schema_name='weightloss_compact_extraction_nested', to_canonical=nested_to_canonical))


def _v4(text):
    reps = [
        ("A \"very minimal\", \"mild\" or \"a bit of\" symptom occurred.",
         "A \"very minimal\", \"mild\" or \"a bit of\" symptom occurred. occurrence says whether the symptom happens; asking whether the drug causes it does not change that (\"My hair is thinning; could X be causing it?\" is occurred + question + attribution uncertain)."),
        ("severe, extreme or intense -> \"severe\"), else null.",
         "severe, extreme or intense -> \"severe\"), else null; never infer it from descriptive words such as \"rough\", \"awful\" or \"bad\"."),
    ]
    for a, b in reps:
        if a not in text:
            raise ValueError('prompt v4 replacement target missing: ' + a[:60])
        text = text.replace(a, b)
    return text


_EPISODE = ("Repeat use_status, dose and duration on every item of the same use episode (same person, same drug, explicitly the same episode); keep different episodes apart.",
            "Repeat use_status, dose and duration on every item of the same use episode (same person, same drug, explicitly the same episode), questions and hypotheticals about that drug included; keep different episodes apart.")
PROMPT_V4 = _v4(PROMPT_V3).replace(*_EPISODE)
assert _EPISODE[1] in PROMPT_V4
COMPACT_P4 = register(Spec('compact-p4', PROMPT_V4, WireExtraction, SCHEMA_VERSION, 'compact-prompt-4.0.0'))
PROMPT_V4N = _v4(_v3(_COMMON).replace(*_EPISODE)).replace(
    "- use_status: ", "- use: an object {status, dose, duration} for the drug's use by this subject, or null when the text states none of them. status (use_status): ").replace(
    "- dose: the general dose", "- dose (inside use): the general dose").replace(
    "Repeat use_status, dose and duration on every item", "Repeat use on every item").replace(
    "- severity only from", "- event: an object {severity, event_dose, onset, event_duration, clinical_consequences, outcome} for that effect, or null when none of them is stated; it is always null when occurrence is \"not_occurred\" or \"not_asserted\". Inside event: severity only from").replace(
    " With occurrence \"not_occurred\" or \"not_asserted\" all of them stay null or [].", "")
assert "Repeat use on every item of the same use episode" in PROMPT_V4N and "questions and hypotheticals about that drug included" in PROMPT_V4N
COMPACT_P4N = register(Spec('compact-p4n', PROMPT_V4N, ExtractionN, 'compact-wire-nested-1.0.0', 'compact-prompt-4.0.0-nested',
                            schema_name='weightloss_compact_extraction_nested', to_canonical=nested_to_canonical))


def _v5(text):
    reps = [
        ("A \"very minimal\", \"mild\" or \"a bit of\" symptom occurred.",
         "A symptom described as \"very minimal\", \"mild\" or \"a bit of\" occurred (e.g. \"nausea ... it's been very minimal\" is occurred with severity null unless a severity word is used)."),
        ("- surface, indication and effect are copied character for character from the input (same spelling and capitalization), never normalized or translated.",
         "- surface, indication and effect are copied character for character from the input (same spelling and capitalization), never normalized or translated; indication and effect name the condition, goal or symptom itself, never a pronoun such as \"that\" or \"it\" standing for it."),
    ]
    for a, b in reps:
        if a not in text:
            raise ValueError('prompt v5 replacement target missing: ' + a[:60])
        text = text.replace(a, b)
    return text


PROMPT_V5 = _v5(PROMPT_V4)
COMPACT_P5 = register(Spec('compact-p5', PROMPT_V5, WireExtraction, SCHEMA_VERSION, 'compact-prompt-5.0.0'))


PROMPT_V6 = PROMPT_V5.replace(
    "If the input has no purpose, side effect or no-side-effect statement, return empty groups.",
    "If the input has no purpose, side effect or no-side-effect statement, return empty groups. Answer once, with the final JSON only: no drafts, commentary or repeated answers.")
assert PROMPT_V6 != PROMPT_V5
COMPACT_P6 = register(Spec('compact-p6', PROMPT_V6, WireExtraction, SCHEMA_VERSION, 'compact-prompt-6.0.0', text_verbosity='low'))


def _v7(text):
    reps = [
        ("Repeat use_status, dose and duration on every item of the same use episode (same person, same drug, explicitly the same episode), questions and hypotheticals about that drug included; keep different episodes apart.",
         "use_status, dose and duration describe one subject's own use of the drug: repeat them on every item of that same use episode (same person, same drug, explicitly the same episode), the writer's own questions and hypotheticals about that drug included; keep different episodes apart. Statements about people in general, or about another person, never take the writer's use values; give them only use values stated for them."),
        ("- A medicine taken to relieve symptoms (e.g. Zofran, Pepto, Imodium) is not their cause unless the text says so:",
         "- A medicine taken to relieve symptoms (e.g. Zofran, Pepto, Imodium) is not their cause unless the text says so; its own use is a purpose item (e.g. Pepto -> \"stomach issues\", Imodium -> \"side effects from metformin\"):"),
        ("- subject: \"self\" (the writer), \"other\" (a specific other person), \"general\" (people in general), or null when unclear.",
         "- subject: \"self\" (the writer), \"other\" (a specific other person), \"general\" (people in general), or null when unclear. A question or remark about what people in general experience (e.g. \"Is this how people lose weight? They feel so bad (acid reflux, headaches)\") is subject general with occurrence \"not_asserted\", unless the writer says it also happens to them."),
    ]
    for a, b in reps:
        if a not in text:
            raise ValueError('prompt v7 replacement target missing: ' + a[:60])
        text = text.replace(a, b)
    return text


PROMPT_V7 = _v7(PROMPT_V6)
COMPACT_P7 = register(Spec('compact-p7', PROMPT_V7, WireExtraction, SCHEMA_VERSION, 'compact-prompt-7.0.0', text_verbosity='low'))
# compact-p7t1 (2026-09-30 technical repair): the compact-p7 request unchanged plus the opt-in canonical
# compact-canonical-negdose-1.0.0; its runs use localize-1.1.0. Registered here so every native entry point resolves it.
from .compact import WireExtractionNegDose  # noqa: E402
COMPACT_P7T1 = register(Spec('compact-p7t1', PROMPT_V7, WireExtraction, SCHEMA_VERSION, 'compact-prompt-7.0.0', text_verbosity='low',
                             canonical=('compact-canonical-negdose-1.0.0', WireExtractionNegDose)))
# compact-p7t2 (2026-09-30, dev400 technical regression): the same request again, with canonical
# compact-canonical-negdose-sev-1.0.0 (negdose + severity-v2). Used only for the audited, versioned offline revalidation of
# saved replies (surface_repair.revalidate); no run sends it.
from .compact import WireExtractionNegDoseSev  # noqa: E402
COMPACT_P7T2 = register(Spec('compact-p7t2', PROMPT_V7, WireExtraction, SCHEMA_VERSION, 'compact-prompt-7.0.0', text_verbosity='low',
                             canonical=('compact-canonical-negdose-sev-1.0.0', WireExtractionNegDoseSev)))
# compact-p7t3 (2026-09-30 engineering review): the compact-p7 request unchanged, under the versioned field contract
# compact-contract-2.0.0 (contract2.py): p7's own event semantics (canonical compact-wire-1.0.0, no negdose/severity
# relaxation), literal as-written fields, version-2 token/number boundaries; its runs use localize-1.2.0.
from . import contract2  # noqa: E402
COMPACT_P7T3 = register(Spec('compact-p7t3', PROMPT_V7, WireExtraction, SCHEMA_VERSION, 'compact-prompt-7.0.0', text_verbosity='low',
                             contract=contract2.CONTRACT_2))
# iteration-20260930-01a0f4c2 (two arms on fresh100, both under compact-contract-2.1.0 with localize-1.3.0 and literal-select-1.4.0):
# compact-p7t4 is the compact-p7 request unchanged (control); compact-p7c1 is the single candidate request, PROMPT_V7 with the four
# replacements frozen in results/extraction-postjev-2026-09-29/iteration-20260930-01a0f4c2/candidate-prompt-p7c1.txt.
P7C1_REPLACEMENTS = [
    ('Use ref "unspecified" only for a drug that is never named for that referent in the input; then surface is the words used (e.g. "my shot").',
     'Use ref "unspecified" only for a drug that is never named for that referent in the input; then surface is the words used (e.g. "my shot", '
     '"the medication", "2.5"). A post that never names its drug still gets every item it states, bound to such an unspecified drug.'),
    ('- purposes: what a drug is used, prescribed or wanted for, as the text states it: a condition or goal given as the reason for use, or a '
     'condition the drug is said to treat or help (e.g. "for my type 2 diabetes", "prescribed due to PCOS", "I started it to lose weight", "I take '
     'it to quiet the food noise", "it helps my inflammation"). indication: that condition or goal as written (e.g. "type 2 diabetes", "lose '
     'weight"), never a result, amount, duration or dose. Not purposes: results by themselves (e.g. "lost 9 lbs", "A1C is 5.5", "the food noise '
     'is gone"), conditions that are only listed (e.g. "I also have PCOS"), and dosing or diet plans.',
     '- purposes: why a drug is taken, prescribed, wanted or continued, as the text states it: a condition or goal given as the reason (e.g. '
     '"for my type 2 diabetes", "prescribed due to PCOS", "I started it to lose weight", "I take it to quiet the food noise", "staying on 2.5 '
     'for maintenance"). A medicine, supplement or remedy taken for a symptom gets a purpose with that symptom (e.g. "took Pepto for the '
     'nausea", "I take magnesium if things get backed up", "I take 2 Advil and it controls it" -> the soreness it controls). indication: that '
     'condition or goal as written (e.g. "type 2 diabetes", "lose weight", "maintenance"), never a result, amount, duration or dose. Not '
     'purposes: results or benefits the writer only noticed (e.g. "lost 9 lbs", "A1C is 5.5", "the food noise is gone", "it helps me feel '
     'full") unless given as a reason for taking it, conditions that are only listed (e.g. "I also have PCOS"), and dosing or diet plans '
     'without a stated goal.'),
    ('The intended appetite reduction, fullness, fewer cravings or less food noise are not side effects unless the text presents them as a '
     'problem (e.g. "I can\'t eat enough").',
     'Eating or drinking too little because of the drug is a side effect when the text presents it as a problem (e.g. "I can\'t eat enough", '
     '"forcing myself to eat", "losing weight too fast"); welcomed or expected appetite reduction, fullness, fewer cravings or less food noise '
     'are not. Hunger, cravings or food noise that return or persist, and weight or blood sugar that do not improve, mean the drug is not '
     'working: not side effects.'),
    ('- quotes: 1 to 8 passages copied exactly from the input, character for character, that support the item and its fields; each long enough',
     '- quotes: 1 to 8 passages copied exactly from the input, character for character, that support the item and its fields, including the '
     'words that give a purpose\'s reason and the words that tie an effect to its drug; each long enough'),
]
P7C1_SHA256 = '9a974b162e9fac4e31a5d27f4c344035ae86cb5f57168c0c1db1b5cc4a69a078'


def _p7c1(text):
    for old, new in P7C1_REPLACEMENTS:
        if text.count(old) != 1:
            raise ValueError('prompt p7c1 replacement target must occur exactly once: ' + old[:60])
        text = text.replace(old, new)
    return text


PROMPT_P7C1 = _p7c1(PROMPT_V7)
assert __import__('hashlib').sha256(PROMPT_P7C1.encode()).hexdigest() == P7C1_SHA256, 'compact-p7c1 differs from the frozen candidate text'
COMPACT_P7T4 = register(Spec('compact-p7t4', PROMPT_V7, WireExtraction, SCHEMA_VERSION, 'compact-prompt-7.0.0', text_verbosity='low',
                             contract=contract2.CONTRACT_2_1))
COMPACT_P7C1 = register(Spec('compact-p7c1', PROMPT_P7C1, WireExtraction, SCHEMA_VERSION, 'compact-prompt-7c1-1.0.0', text_verbosity='low',
                             contract=contract2.CONTRACT_2_1))


PROMPT_V8 = """Extract drug statements from one social-media post or health review for a knowledge graph. Use only the text: no outside knowledge, no guesses, no defaults (never assume the writer, current use, an affirmed purpose or an attributed effect). Answer once with the final JSON only; every key present, null when not stated, [] when a group is empty.

drugs: each drug an item refers to. handle d1, d2, ... in order of first mention; surface = its name as written (for a WebMD review the drug_field value may be used); ref = "name" (brand or generic), "class" (e.g. "GLP-1") or "unspecified" (a drug never named, e.g. "my shot", "the 0.25"). Resolve "it", "this med", "the shot" to the named drug they clearly refer to. Never turn a class or unspecified drug into a brand.

purposes: what a drug is taken, prescribed or wanted for, as the text states it ("for my type 2 diabetes", "to lose weight", "to quiet the food noise", "it helps my inflammation"). Not purposes: results (weight lost, lab values, the food noise is gone), conditions that are only listed, dosing or diet plans. WebMD: condition_field (unless "Other") is the review's condition for the drug in drug_field: add it as a report, polarity affirmed, quoting condition_field; subject and use only from the review text.
side_effects: unwanted symptoms or bodily changes the text links to a drug, whether they occurred, did not occur, are uncertain, feared or asked about. Not side effects: foods, remedies, worries, dislikes (e.g. of injecting), dosing mistakes, benefits, the drug wearing off. Use the words "side effects" as the effect only when no symptom is named.
no_side_effects: explicit claims of no side effects ("no side effects", "haven't had any"); "tolerated it well" is not one. scope_text keeps limiting words as written ("no major side effects", "so far"), else null; never also add it as a side_effects item. "No nausea" is a side_effects item with occurrence "not_occurred".

Which drug: a symptom belongs to the drug the text says causes it, never to a medicine taken to relieve it (Zofran, Pepto, Imodium); the relief medicine gets its own purpose item (e.g. Pepto -> "stomach issues"). When the causing drug is never named, use an unspecified drug.
Whose experience: subject "self" (the writer), "other" (a specific person; subject_label from the text, e.g. "sister"), "general" (people in general) or null when unclear. A question or remark about what people generally experience is subject "general" with occurrence "not_asserted", not the writer's own symptom.

Item fields (independent of each other; never fill one from another):
- statement_kind: "report", "question" or "hypothetical" (planned, feared, conditional).
- polarity (purposes): "affirmed", "negated" or "uncertain"; null only for a pure question that asserts nothing.
- occurrence: "occurred" (mild or "very minimal" included), "not_occurred", "uncertain" or "not_asserted" (no evidence either way). Asking whether the drug causes a symptom that happens keeps "occurred".
- attribution (the author's view, not medical causality): "attributed" ("gave me", "a side effect of", "makes me"), "temporal_only" (only timing, e.g. "after the shot"), "uncertain" (wonders, or also suspects another cause), "other_cause" or "not_stated" (no link at all).
- use_status ("current", "past", "planned", "considering", "not_using", "unknown"), dose (as written) and duration (total time on the drug, as written): this subject's own use, null when not stated; repeat them on every item of the same use episode; never copy the writer's use onto statements about people in general or another person.
- severity: "mild" (mild, slight), "moderate", "severe" (severe, extreme, intense) only from such words, else null. event_dose, onset, event_duration, clinical_consequences ("medical_visit", "emergency_visit", "hospitalization", "other") and outcome ("resolved", "ongoing", "unknown") only when stated for that effect. All of these stay null or [] when occurrence is "not_occurred" or "not_asserted".
- surface, indication and effect are copied exactly from the text (same spelling and capitalization), never normalized and never a pronoun.
- quotes: 1 to 8 passages copied exactly from the text that support the item and its fields, each occurring only once in the text (quote a longer passage if a phrase repeats); no paraphrase, "..." or joined passages."""
COMPACT_P8 = register(Spec('compact-p8', PROMPT_V8, WireExtraction, SCHEMA_VERSION, 'compact-prompt-8.0.0', text_verbosity='low'))


# --- Structural candidates for reasoning none (2026-09-27, after the user fixed reasoning none) -------------------------
# Diagnosis of compact-p7 at reasoning none on dev-128 (corrections-v3): the dominant error is attribution (29 dimension
# errors per run, mostly not_stated or uncertain for symptoms the writer reports while on the drug, where the references
# accept attributed or temporal_only), then false positives (outcomes as purposes, non-symptoms as side effects),
# duplicate items with inconsistent states, and occasional relief-drug confusion. Reasoning low fixed most of these, so
# the none candidates restructure the one-shot answer instead of adding prompt patches:
#   - evidence first: every item starts with its quotes, so the fields are written after the supporting passage;
#   - the five-way attribution becomes two concrete questions, cause (what the writer says causes it) and timing
#     (whether the text ties it to the drug's use); attribution is derived from them deterministically (s1_attribution);
#   - a drug-level relief flag and a purpose-level basis make the relief-drug rule and the purpose definition explicit;
#   - event details form one optional object (null when absent, and always for not_occurred / not_asserted);
#   - statement_kind and occurrence stay separate, so every existing state combination remains expressible.
# compact-s1 keeps flat item lists; compact-s2 also states drug, subject and use once per use context.
# Wire-only aids (relief, basis, cause, timing) are not stored: answers convert to the canonical wire and compact record.

CAUSE_ATTRIBUTION = {'drug': 'attributed', 'drug_unsure': 'uncertain', 'drug_and_other': 'uncertain', 'other': 'other_cause'}


def s1_attribution(cause, timing):
    """Deterministic re-encoding of the writer's stated cause and time relation as the canonical attribution. A stated
    cause decides it; without one, a time relation the text states between the effect and the drug's use gives
    temporal_only (p7: 'only timing links them'), for any occurrence, so a negated or asked-about effect keeps its stated
    time relation (e.g. "no nausea since starting"); otherwise not_stated. No text is read here."""
    if cause in CAUSE_ATTRIBUTION:
        return CAUSE_ATTRIBUTION[cause]
    return 'temporal_only' if timing == 'tied_to_use' else 'not_stated'


Occurrence = Literal['occurred', 'not_occurred', 'uncertain', 'not_asserted']
Cause = Literal['drug', 'drug_unsure', 'drug_and_other', 'other', 'none']
Timing = Literal['tied_to_use', 'not_tied', 'not_stated']
Basis = Literal['reason', 'helps', 'condition_field']
Polarity = Optional[Literal['affirmed', 'negated', 'uncertain']]


class DrugS(Wire):
    handle: Handle
    surface: Surface
    ref: Literal['name', 'class', 'unspecified']
    relief: bool


class PurposeS1(Wire):
    quotes: Quotes
    drug: Handle
    indication: Surface
    basis: Basis
    subject: Subject
    subject_label: Optional[Label]
    statement_kind: StatementKind
    polarity: Polarity
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]


class SideEffectS1(Wire):
    quotes: Quotes
    drug: Handle
    effect: Surface
    subject: Subject
    subject_label: Optional[Label]
    statement_kind: StatementKind
    occurrence: Occurrence
    cause: Cause
    timing: Timing
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    event: Optional[EventN]


class NoSideEffectS1(Wire):
    quotes: Quotes
    drug: Handle
    subject: Subject
    subject_label: Optional[Label]
    scope_text: Optional[ShortText]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]


class ExtractionS1(Wire):
    drugs: list[DrugS]
    purposes: list[PurposeS1]
    side_effects: list[SideEffectS1]
    no_side_effects: list[NoSideEffectS1]


class PurposeS2(Wire):
    quotes: Quotes
    indication: Surface
    basis: Basis
    statement_kind: StatementKind
    polarity: Polarity


class SideEffectS2(Wire):
    quotes: Quotes
    effect: Surface
    statement_kind: StatementKind
    occurrence: Occurrence
    cause: Cause
    timing: Timing
    event: Optional[EventN]


class NoSideEffectS2(Wire):
    quotes: Quotes
    scope_text: Optional[ShortText]


class ContextS2(Wire):
    drug: Handle
    subject: Subject
    subject_label: Optional[Label]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    purposes: list[PurposeS2]
    side_effects: list[SideEffectS2]
    no_side_effects: list[NoSideEffectS2]


class ExtractionS2(Wire):
    drugs: list[DrugS]
    contexts: list[ContextS2]


def _drugs(answer):
    return [{'handle': d.handle, 'surface': d.surface, 'ref': d.ref} for d in answer.drugs]


def _event(e):
    return {'severity': e.severity if e else None, 'event_dose': e.event_dose if e else None, 'onset': e.onset if e else None,
            'event_duration': e.event_duration if e else None,
            'clinical_consequences': list(e.clinical_consequences) if e else [], 'outcome': e.outcome if e else None}


def _side(drug, subject, subject_label, use, s):
    return {'drug': drug, 'effect': s.effect, 'subject': subject, 'subject_label': subject_label, **use,
            'statement_kind': s.statement_kind, 'occurrence': s.occurrence,
            'attribution': s1_attribution(s.cause, s.timing), **_event(s.event), 'quotes': s.quotes}


def s1_to_canonical(answer):
    """Flat structural wire -> canonical wire dict: fields copied, attribution from s1_attribution, event object spread
    (absent -> nulls / []), relief and basis dropped."""
    use = lambda x: {'use_status': x.use_status, 'dose': x.dose, 'duration': x.duration}  # noqa: E731
    return {'drugs': _drugs(answer),
            'purposes': [{'drug': p.drug, 'indication': p.indication, 'subject': p.subject, 'subject_label': p.subject_label,
                          **use(p), 'statement_kind': p.statement_kind, 'polarity': p.polarity, 'quotes': p.quotes}
                         for p in answer.purposes],
            'side_effects': [_side(s.drug, s.subject, s.subject_label, use(s), s) for s in answer.side_effects],
            'no_side_effects': [{'drug': n.drug, 'subject': n.subject, 'subject_label': n.subject_label, 'scope_text': n.scope_text,
                                 **use(n), 'quotes': n.quotes} for n in answer.no_side_effects]}


def s2_to_canonical(answer):
    """Context wire -> canonical wire dict: each item takes its context's drug, subject and use values; otherwise as s1.
    Items are listed context by context (item order carries no meaning in the compact record)."""
    out = {'drugs': _drugs(answer), 'purposes': [], 'side_effects': [], 'no_side_effects': []}
    for c in answer.contexts:
        use = {'use_status': c.use_status, 'dose': c.dose, 'duration': c.duration}
        out['purposes'] += [{'drug': c.drug, 'indication': p.indication, 'subject': c.subject, 'subject_label': c.subject_label,
                             **use, 'statement_kind': p.statement_kind, 'polarity': p.polarity, 'quotes': p.quotes}
                            for p in c.purposes]
        out['side_effects'] += [_side(c.drug, c.subject, c.subject_label, use, s) for s in c.side_effects]
        out['no_side_effects'] += [{'drug': c.drug, 'subject': c.subject, 'subject_label': c.subject_label, 'scope_text': n.scope_text,
                                    **use, 'quotes': n.quotes} for n in c.no_side_effects]
    return out


_S_INTRO = """You extract drug statements from one social-media post or health review for a knowledge graph. Use only the given text: no outside knowledge, no guesses. Nothing is pre-filled and nothing has a default; never assume the writer, current use, an affirmed purpose or the cause of an effect.

Drugs (every key present; null when the text does not say; [] when a group is empty):
- drugs: one entry per medication that an item below refers to. handle: d1, d2, ... in order of first mention. surface: the drug's name as written in the input (for a WebMD review the drug_field value may be used). ref: "name" (brand or generic name), "class" (a drug class only, e.g. "GLP-1") or "unspecified". relief: true when the text presents the drug as taken to relieve symptoms or side effects (e.g. Zofran, Pepto, Imodium, stool softeners), else false.
- Resolve references such as "it", "this med", "this one", "the pill", "the shot" or "these meds" to the named drug they clearly refer to and use that drug's handle. In a WebMD review, drug_field is the reviewed drug and the review is about it unless another drug is named. Use ref "unspecified" only for a drug that is never named for that referent in the input; then surface is the words used (e.g. "my shot"). Never upgrade a class or unspecified reference to a brand, and never replace one named drug with another.
- A relief drug is not the cause of the symptoms it relieves unless the text says so. Its own use is a purpose item (e.g. Pepto -> "stomach issues", Imodium -> "side effects from metformin"), and the symptoms belong to the drug that causes them: "throwing up even after taking zofran" is a side effect of the drug taken for weight or diabetes, not of Zofran. When that causing drug is never named, create an unspecified drug whose surface is the words that refer to it (e.g. "the shot", "my dose", ".25").
"""

_S_GROUPS = """- purposes: what a drug is used, prescribed or wanted for, as the text states it: a condition or goal given as the reason for use, or a condition the drug is said to treat or help (e.g. "for my type 2 diabetes", "prescribed due to PCOS", "I started it to lose weight", "I take it to quiet the food noise", "it helps my inflammation"). indication: that condition or goal as written (e.g. "type 2 diabetes", "lose weight"), never a result, amount, duration or dose. basis: "reason" when the text gives it as the reason the drug is used, prescribed or wanted; "helps" when the text says the drug treats or helps it; "condition_field" for the WebMD purpose below. A statement that fits none of these is not a purpose: results by themselves (e.g. "lost 9 lbs", "A1C is 5.5", "the food noise is gone"), conditions that are only listed (e.g. "I also have PCOS"), and dosing or diet plans. In a WebMD review, condition_field (unless it is "Other") is the condition the review is associated with for the drug in drug_field: add it as a purpose (basis "condition_field", report, polarity affirmed) quoting condition_field; take subject and use_status only from the review text, and follow the text if it negates the condition or is about another person.
- side_effects: unwanted effects the text links to a drug, whether they occurred, did not occur, are uncertain, feared or asked about. effect: the symptom or bodily change as written (e.g. "nausea", "hair thinning", "red at the injection spot"); only when no symptom is named, the words used for the side effects (e.g. "side effects"). The intended appetite reduction, fullness, fewer cravings or less food noise are not side effects unless the text presents them as a problem (e.g. "I can't eat enough"). Not side effects: foods or remedies, emotions or worries (e.g. "nervous"), dosing mistakes, the drug wearing off, benefits, intended weight loss, and side effects that are only asked about or mentioned in general advice without a named symptom.
- no_side_effects: explicit statements that a drug caused no side effects ("no side effects", "haven't had any side effects"). "Tolerated it well" or "doing fine" is not explicit. scope_text: the limiting words as written when the claim is narrower than none at all (e.g. "no major side effects", "none so far"; "I haven't had the severe side effects others have" -> scope_text "the severe side effects"), else null; never widen the scope, and never also add the claim as a side_effects item or give it an event. A specific negated symptom ("no nausea") is a side_effects item with occurrence "not_occurred".
- surface, indication and effect are copied character for character from the input (same spelling and capitalization), never normalized or translated; indication and effect name the condition, goal or symptom itself, never a pronoun such as "that" or "it" standing for it.
"""

_S_QUOTES = """- quotes (write them first): 1 to 8 passages copied exactly from the input, character for character, that state the item, including the words that give its reason, timing or cause; each long enough to occur only once in the input (if a phrase repeats, quote a longer passage around the intended occurrence). Do not paraphrase, abbreviate with "..." or join separate passages."""

_S_SUBJECT = """- subject: "self" (the writer), "other" (a specific other person), "general" (people in general), or null when unclear. A question or remark about what people in general experience (e.g. "Is this how people lose weight? They feel so bad (acid reflux, headaches)") is subject general with occurrence "not_asserted", unless the writer says it also happens to them. subject_label: a short label from the text only when subject is "other" (e.g. "sister"), else null."""

_S_STATE = """- statement_kind: "report" (declarative), "question", or "hypothetical" (planned, considered, feared or conditional).
- polarity (purposes): "affirmed", "negated" or "uncertain"; null only for a pure question that asserts nothing. A report always has a polarity.
- occurrence (side effects): "occurred", "not_occurred", "uncertain", or "not_asserted" (no evidence either way, e.g. a pure fear or a question about a future effect). A symptom described as "very minimal", "mild" or "a bit of" occurred. occurrence says whether the symptom happens; asking whether the drug causes it does not change that ("My hair is thinning; could X be causing it?" is occurred + question).
- cause (side effects; the writer's view, not medical causality): "drug" when the text presents the effect as caused by or a side effect of the drug ("side effects like", "gave me", "makes me", "from the shot", "caused"); "drug_unsure" when the writer wonders whether the drug causes it ("could Ozempic be causing it?"); "drug_and_other" when the writer suspects the drug and also something else (e.g. "maybe the needle"); "other" when the writer blames something else (e.g. "from the carbs"); "none" when the text says nothing about what causes it.
- timing (side effects): "tied_to_use" when the text itself places the effect (or its absence) in time relative to the subject's use of the drug: while taking it, after a dose, or after starting, raising, restarting or stopping it (e.g. "since starting", "after each shot", "my first week on it", "on Zep I get", "no nausea so far"); "not_tied" when the text places it before or apart from the drug's use (e.g. "I've had migraines for years"); "not_stated" when the text gives no time relation to the drug. Appearing in a review of the drug or in a post about it is not by itself a time relation."""

_S_USE = """use_status: "current", "past", "planned", "considering", "not_using" or "unknown" as stated for this drug and subject; null when the text does not say. Planned or considering is not actual use. dose: the general dose as written (e.g. "0.5 mg"). duration: the total time on the drug as written (e.g. "six weeks"); it never fills onset or event_duration."""

_S_EVENT = """- event (side effects): an object {severity, event_dose, onset, event_duration, clinical_consequences, outcome} for details stated for that effect, or null when none is stated; always null when occurrence is "not_occurred" or "not_asserted". severity only from a severity word in the text (mild or slight -> "mild"; moderate -> "moderate"; severe, extreme or intense -> "severe"), else null; never infer it from descriptive words such as "rough", "awful" or "bad". event_dose (a dose the text ties to the event), onset (e.g. "two weeks after stopping"), event_duration, clinical_consequences ("medical_visit", "emergency_visit", "hospitalization", "other") and outcome ("resolved", "ongoing", "unknown"): only when stated for that effect."""

_S_END = """
If the input has no purpose, side effect or no-side-effect statement, return empty groups. Answer once, with the final JSON only: no drafts, commentary or repeated answers."""

PROMPT_S1 = (_S_INTRO + "\nGroups:\n" + _S_GROUPS + "\nItem fields, in the order written (independent of each other; never fill one from another):\n"
             + _S_QUOTES + "\n- drug: the handle of the drug the statement is about.\n" + _S_SUBJECT + "\n" + _S_STATE
             + "\n- " + _S_USE + " use_status, dose and duration describe one subject's own use of the drug: repeat them on every item of that same use episode (same person, same drug, explicitly the same episode), the writer's own questions and hypotheticals about that drug included; keep different episodes apart. Statements about people in general, or about another person, never take the writer's use values; give them only use values stated for them.\n"
             + _S_EVENT + "\n" + _S_END)

PROMPT_S2 = (_S_INTRO + """
Contexts: group the statements by use context. One context per drug and subject, and per use episode when the text keeps episodes apart (e.g. a past course and the current one); statements about people in general, and about another person, get their own context. Each context holds the purposes, side_effects and no_side_effects about that drug for that subject.

Context fields (stated once per context; independent of each other):
- drug: the handle of the drug the context is about.
""" + _S_SUBJECT + "\n- " + _S_USE + """ These describe the subject's own use in this context. A context about people in general, or about another person, never takes the writer's use values; give it only use values stated for it.

Groups inside a context:
""" + _S_GROUPS + "\nItem fields, in the order written (independent of each other; never fill one from another):\n" + _S_QUOTES + "\n"
             + _S_STATE + "\n" + _S_EVENT + "\n" + _S_END)

COMPACT_S1 = register(Spec('compact-s1', PROMPT_S1, ExtractionS1, 'compact-wire-s1-1.0.0', 'compact-prompt-s1-1.1.0',
                           schema_name='weightloss_compact_extraction_s1', to_canonical=s1_to_canonical, text_verbosity='low'))
COMPACT_S2 = register(Spec('compact-s2', PROMPT_S2, ExtractionS2, 'compact-wire-s2-1.0.0', 'compact-prompt-s2-1.1.0',
                           schema_name='weightloss_compact_extraction_s2', to_canonical=s2_to_canonical, text_verbosity='low'))


# --- compact-s3: p7 plus only the part of s1 that measurably helped (2026-09-27) --------------------------------------
# Dev-128 (corrections-v3), one run each against p7-none a/b: s1 cut side-effect attribution errors from 29 to 13 without
# over-using attributed (3 wrong attributed, as in p7) and raised WebMD side-effect precision from 0.71-0.72 to 0.87, but
# its purpose basis field turned benefits and outcomes into purposes (false positives 15 -> 24; 13 of them with basis
# "helps") and added paraphrased indications; s2's shared contexts spread timing (10 wrong temporal_only) and were worse
# on purposes. compact-s3 therefore keeps p7 exactly (drugs, purposes, no_side_effects, field order, quotes last) and only
# replaces the side-effect attribution with the two observations cause and timing (derived by s1_attribution) and the
# six event fields with one optional event object. No relief flag, no purpose basis, no contexts.

class SideEffectS3(Wire):
    drug: Handle
    effect: Surface
    subject: Subject
    subject_label: Optional[Label]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    statement_kind: StatementKind
    occurrence: Occurrence
    cause: Cause
    timing: Timing
    event: Optional[EventN]
    quotes: Quotes


class ExtractionS3(Wire):
    drugs: list[WireDrug]
    purposes: list[WirePurpose]
    side_effects: list[SideEffectS3]
    no_side_effects: list[WireNoSideEffect]


def s3_to_canonical(answer):
    """p7 wire with cause/timing and an event object -> canonical wire dict; drugs, purposes and no_side_effects unchanged."""
    return {'drugs': [d.model_dump() for d in answer.drugs], 'purposes': [p.model_dump() for p in answer.purposes],
            'side_effects': [_side(s.drug, s.subject, s.subject_label, {'use_status': s.use_status, 'dose': s.dose, 'duration': s.duration}, s)
                             for s in answer.side_effects],
            'no_side_effects': [n.model_dump() for n in answer.no_side_effects]}


def _s3(text):
    reps = [
        ("never assume the writer, current use, an affirmed purpose or an attributed effect.",
         "never assume the writer, current use, an affirmed purpose or the cause of an effect."),
        ("\"I have nausea; could Ozempic be causing it?\" is occurred + question + uncertain attribution.",
         "\"I have nausea; could Ozempic be causing it?\" is occurred + question + cause \"drug_unsure\"."),
        ("(\"My hair is thinning; could X be causing it?\" is occurred + question + attribution uncertain).",
         "(\"My hair is thinning; could X be causing it?\" is occurred + question)."),
        ("- attribution (side effects; the author's view, not medical causality): \"attributed\" when the text presents the effect as caused by or a side effect of the drug (\"side effects like\", \"gave me\", \"makes me\", \"from the shot\", \"caused\"); \"temporal_only\" when only timing links them (\"after each shot\", \"since starting\"); \"uncertain\" when the author wonders or also suspects another cause (e.g. \"maybe the needle\"); \"other_cause\" when the author blames something else; \"not_stated\" only when the effect is mentioned without any link to the drug.",
         "- cause (side effects; the writer's view, not medical causality): \"drug\" when the text presents the effect as caused by or a side effect of the drug (\"side effects like\", \"gave me\", \"makes me\", \"from the shot\", \"caused\"); \"drug_unsure\" when the writer wonders whether the drug causes it (\"could Ozempic be causing it?\"); \"drug_and_other\" when the writer suspects the drug and also something else (e.g. \"maybe the needle\"); \"other\" when the writer blames something else (e.g. \"from the carbs\"); \"none\" when the text says nothing about what causes it.\n"
         "- timing (side effects): \"tied_to_use\" when the text itself places the effect (or its absence) in time relative to the subject's use of the drug: while taking it, after a dose, or after starting, raising, restarting or stopping it (e.g. \"since starting\", \"after each shot\", \"my first week on it\", \"on Zep I get\", \"no nausea so far\"); \"not_tied\" when the text places it before or apart from the drug's use (e.g. \"I've had migraines for years\"); \"not_stated\" when the text gives no time relation to the drug. Appearing in a review of the drug or in a post about it is not by itself a time relation."),
        ("- severity only from a severity word in the text (mild or slight -> \"mild\"; moderate -> \"moderate\"; severe, extreme or intense -> \"severe\"), else null; never infer it from descriptive words such as \"rough\", \"awful\" or \"bad\". event_dose (a dose the text ties to the event), onset (e.g. \"two weeks after stopping\"), event_duration, clinical_consequences (\"medical_visit\", \"emergency_visit\", \"hospitalization\", \"other\") and outcome (\"resolved\", \"ongoing\", \"unknown\"): only when stated for that effect. With occurrence \"not_occurred\" or \"not_asserted\" all of them stay null or [].",
         "- event (side effects): an object {severity, event_dose, onset, event_duration, clinical_consequences, outcome} for details stated for that effect, or null when none is stated; always null when occurrence is \"not_occurred\" or \"not_asserted\". severity only from a severity word in the text (mild or slight -> \"mild\"; moderate -> \"moderate\"; severe, extreme or intense -> \"severe\"), else null; never infer it from descriptive words such as \"rough\", \"awful\" or \"bad\". event_dose (a dose the text ties to the event), onset (e.g. \"two weeks after stopping\"), event_duration, clinical_consequences (\"medical_visit\", \"emergency_visit\", \"hospitalization\", \"other\") and outcome (\"resolved\", \"ongoing\", \"unknown\"): only when stated for that effect."),
        ("never also add the claim as a side_effects item or give it a severity.",
         "never also add the claim as a side_effects item or give it an event."),
    ]
    for a, b in reps:
        if text.count(a) != 1:
            raise ValueError('prompt s3 replacement target missing or repeated: ' + a[:60])
        text = text.replace(a, b)
    return text


PROMPT_S3 = _s3(PROMPT_V7)
COMPACT_S3 = register(Spec('compact-s3', PROMPT_S3, ExtractionS3, 'compact-wire-s3-1.0.0', 'compact-prompt-s3-1.0.0',
                           schema_name='weightloss_compact_extraction_s3', to_canonical=s3_to_canonical, text_verbosity='low'))


# --- compact-s4: inline drug per item instead of a global drug list with handles (2026-09-27) --------------------------
# A different axis from s1-s3 (which kept the drug list and split judgment fields): the model no longer enumerates drugs
# first and then links items to handles. Each purpose / side effect / no-side-effect item names its drug inline as
# {surface, ref}, decided from that item's own statement. Everything else is p7: same categories and fields (attribution
# included), same rules, quotes last. The conversion numbers handles by exact (surface, ref) in order of appearance;
# it never merges different surfaces (no drug knowledge), so one drug written two ways becomes two drug entries.

class InlineDrug(Wire):
    surface: Surface
    ref: Literal['name', 'class', 'unspecified']


class PurposeS4(Wire):
    indication: Surface
    drug: InlineDrug
    subject: Subject
    subject_label: Optional[Label]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    statement_kind: StatementKind
    polarity: Polarity
    quotes: Quotes


class SideEffectS4(Wire):
    effect: Surface
    drug: InlineDrug
    subject: Subject
    subject_label: Optional[Label]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    statement_kind: StatementKind
    occurrence: Occurrence
    attribution: Literal['attributed', 'temporal_only', 'uncertain', 'other_cause', 'not_stated']
    severity: Optional[Literal['mild', 'moderate', 'severe']]
    event_dose: Optional[DoseText]
    onset: Optional[ShortText]
    event_duration: Optional[ShortText]
    clinical_consequences: list[Consequence]
    outcome: Optional[Literal['resolved', 'ongoing', 'unknown']]
    quotes: Quotes


class NoSideEffectS4(Wire):
    drug: InlineDrug
    subject: Subject
    subject_label: Optional[Label]
    scope_text: Optional[ShortText]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    quotes: Quotes


class ExtractionS4(Wire):
    purposes: list[PurposeS4]
    side_effects: list[SideEffectS4]
    no_side_effects: list[NoSideEffectS4]


def s4_to_canonical(answer):
    """Inline drugs -> canonical drug list: one handle per exact (surface, ref), numbered d1, d2, ... in order of appearance
    (purposes, then side effects, then no-side-effect items); every other field is copied unchanged."""
    handles = {}

    def handle(drug):
        return handles.setdefault((drug.surface, drug.ref), f'd{len(handles) + 1}')

    out = {'purposes': [], 'side_effects': [], 'no_side_effects': []}
    for group in out:
        for item in getattr(answer, group):
            out[group].append({**item.model_dump(exclude={'drug'}), 'drug': handle(item.drug)})
    return {'drugs': [{'handle': h, 'surface': s, 'ref': r} for (s, r), h in handles.items()], **out}


def _s4(text):
    reps = [
        ("Drugs (every key present; null when the text does not say; [] when a group is empty):\n- drugs: one entry per medication that an item below refers to. handle: d1, d2, ... in order of first mention. surface: the drug's name as written in the input (for a WebMD review the drug_field value may be used). ref: \"name\" (brand or generic name), \"class\" (a drug class only, e.g. \"GLP-1\") or \"unspecified\".\n- Resolve references such as \"it\", \"this med\", \"this one\", \"the pill\", \"the shot\" or \"these meds\" to the named drug they clearly refer to and use that drug's handle.",
         "Every key present; null when the text does not say; [] when a group is empty.\n\nDrugs (named inside each item):\n- Each item states its own drug as {surface, ref}, decided from that item's statement. surface: the drug's name as written in the input where it is first named (for a WebMD review the drug_field value may be used); use the same surface for the same drug in every item. ref: \"name\" (brand or generic name), \"class\" (a drug class only, e.g. \"GLP-1\") or \"unspecified\".\n- Resolve references such as \"it\", \"this med\", \"this one\", \"the pill\", \"the shot\" or \"these meds\" to the named drug they clearly refer to and give that drug's name as the surface."),
        ("Use ref \"unspecified\" only for a drug that is never named for that referent in the input; then surface is the words used (e.g. \"my shot\").",
         "Use ref \"unspecified\" only for a drug that is never named for that referent in the input; then surface is the words used where it is first mentioned (e.g. \"my shot\"), the same words in every item about it."),
        ("When that causing drug is never named, create an unspecified drug whose surface is the words that refer to it (e.g. \"the shot\", \"my dose\", \".25\").",
         "When that causing drug is never named, give the item an unspecified drug whose surface is the words that refer to it (e.g. \"the shot\", \"my dose\", \".25\")."),
        ("- drug: the handle of the drug the statement is about.",
         "- drug: {surface, ref} of the drug the statement is about (see Drugs)."),
    ]
    for a, b in reps:
        if text.count(a) != 1:
            raise ValueError('prompt s4 replacement target missing or repeated: ' + a[:60])
        text = text.replace(a, b)
    return text


PROMPT_S4 = _s4(PROMPT_V7)
COMPACT_S4 = register(Spec('compact-s4', PROMPT_S4, ExtractionS4, 'compact-wire-s4-1.0.0', 'compact-prompt-s4-1.0.0',
                           schema_name='weightloss_compact_extraction_s4', to_canonical=s4_to_canonical, text_verbosity='low'))


# --- compact-t1p / compact-t1a: two narrow single-call tasks, merged deterministically (restart, 2026-09-27) ----------
# s1-s4 changed the structure inside one call; each traded one error class for another. T1 splits the job across two
# calls on the same text: t1p extracts drugs + purposes only, t1a drugs + side effects + no-side-effect claims only.
# Each prompt is p7 with the other groups removed (same fields, categories and rules) plus one scope sentence (every
# medication, not only GLP-1 drugs) and one direction rule (a condition a drug causes is never its purpose; a condition
# it is taken for or helps is never its side effect). The two compact records merge by merge_compact (exact drug
# surface/ref, handles renumbered; no item is rewritten, dropped or reconciled).

class ExtractionT1P(Wire):
    drugs: list[WireDrug]
    purposes: list[WirePurpose]


class ExtractionT1A(Wire):
    drugs: list[WireDrug]
    side_effects: list[WireSideEffect]
    no_side_effects: list[WireNoSideEffect]


def t1p_to_canonical(answer):
    return {'drugs': [d.model_dump() for d in answer.drugs], 'purposes': [p.model_dump() for p in answer.purposes],
            'side_effects': [], 'no_side_effects': []}


def t1a_to_canonical(answer):
    return {'drugs': [d.model_dump() for d in answer.drugs], 'purposes': [],
            'side_effects': [x.model_dump() for x in answer.side_effects], 'no_side_effects': [x.model_dump() for x in answer.no_side_effects]}


_SCOPE = "- Cover every medication the text mentions (prescriptions, over-the-counter and relief medicines), not only GLP-1 drugs.\n"


def _cut(text, a, b=None):
    if text.count(a) != 1 or (b is not None and text.count(b) != 1):
        raise ValueError('prompt t1 cut target missing or repeated: ' + a[:60])
    start = text.index(a)
    end = text.index(b) if b is not None else start + len(a)
    return text[:start] + text[end:]


def _rep(text, a, b):
    if text.count(a) != 1:
        raise ValueError('prompt t1 replacement target missing or repeated: ' + a[:60])
    return text.replace(a, b)


def _t1p(text):
    text = _rep(text, "You extract drug statements from one social-media post or health review for a knowledge graph.",
                "You extract the purposes of drugs (what each drug is used, prescribed or wanted for) from one social-media post or health review for a knowledge graph. Side effects are extracted separately and are not part of this task.")
    text = _rep(text, "Groups:\n", "Groups:\n" + _SCOPE)
    text = _cut(text, "- side_effects: unwanted effects the text links to a drug", "- surface, indication and effect are copied")
    text = _rep(text, "- surface, indication and effect are copied character for character from the input (same spelling and capitalization), never normalized or translated; indication and effect name the condition, goal or symptom itself, never a pronoun such as \"that\" or \"it\" standing for it.",
                "- A condition or symptom the text says a drug causes, worsens or might cause is never a purpose of that drug.\n- surface and indication are copied character for character from the input (same spelling and capitalization), never normalized or translated; indication names the condition or goal itself, never a pronoun such as \"that\" or \"it\" standing for it.")
    text = _cut(text, "- occurrence (side effects):", "- use_status:")
    text = _cut(text, "- severity only from a severity word in the text", "- quotes:")
    text = _rep(text, "If the input has no purpose, side effect or no-side-effect statement, return empty groups.",
                "If the input has no purpose statement, return empty groups.")
    return text


def _t1a(text):
    text = _rep(text, "You extract drug statements from one social-media post or health review for a knowledge graph.",
                "You extract the side effects of drugs and explicit no-side-effect statements from one social-media post or health review for a knowledge graph. Purposes are extracted separately and are not part of this task.")
    text = _rep(text, "its own use is a purpose item (e.g. Pepto -> \"stomach issues\", Imodium -> \"side effects from metformin\"):",
                "what it is taken for is not its side effect:")
    text = _rep(text, "Groups:\n", "Groups:\n" + _SCOPE)
    text = _cut(text, "- purposes: what a drug is used, prescribed or wanted for", "- side_effects: unwanted effects the text links to a drug")
    text = _rep(text, "- surface, indication and effect are copied character for character from the input (same spelling and capitalization), never normalized or translated; indication and effect name the condition, goal or symptom itself, never a pronoun such as \"that\" or \"it\" standing for it.",
                "- A condition a drug is taken for, prescribed for or said to treat or help is never a side effect of that drug.\n- surface and effect are copied character for character from the input (same spelling and capitalization), never normalized or translated; effect names the symptom itself, never a pronoun such as \"that\" or \"it\" standing for it.")
    text = _cut(text, "- polarity (purposes):", "- occurrence (side effects):")
    text = _rep(text, "If the input has no purpose, side effect or no-side-effect statement, return empty groups.",
                "If the input has no side effect or no-side-effect statement, return empty groups.")
    return text


PROMPT_T1P = _t1p(PROMPT_V7)
PROMPT_T1A = _t1a(PROMPT_V7)
COMPACT_T1P = register(Spec('compact-t1p', PROMPT_T1P, ExtractionT1P, 'compact-wire-t1p-1.0.0', 'compact-prompt-t1p-1.0.0',
                            schema_name='weightloss_compact_extraction_t1p', to_canonical=t1p_to_canonical, text_verbosity='low'))
COMPACT_T1A = register(Spec('compact-t1a', PROMPT_T1A, ExtractionT1A, 'compact-wire-t1a-1.0.0', 'compact-prompt-t1a-1.0.0',
                            schema_name='weightloss_compact_extraction_t1a', to_canonical=t1a_to_canonical, text_verbosity='low'))


def merge_compact(parts, run):
    """Deterministic union of per-task compact records of one input: drugs keyed by exact (surface, ref) and renumbered
    d1, d2, ... in order of appearance across parts; items copied unchanged except their handle; nothing is reconciled."""
    ids = {p['record_id'] for p in parts}
    if len(ids) != 1:
        raise ValueError(f'merge_compact needs records of one input, got {sorted(ids)}')
    handles, out = {}, {'record_id': ids.pop(), 'run': run, 'drugs': {}, 'purposes': [], 'side_effects': [], 'no_side_effects': []}
    for part in parts:
        local = {}
        for h, d in part['drugs'].items():
            key = (d['surface'], d['ref'])
            if key not in handles:
                handles[key] = f'd{len(handles) + 1}'
                out['drugs'][handles[key]] = dict(d)
            local[h] = handles[key]
        for group in ('purposes', 'side_effects', 'no_side_effects'):
            out[group] += [{**item, 'drug': local[item['drug']]} for item in part.get(group, [])]
    return out


def select_groups(record, groups):
    """A task record reduced to the groups another task does not provide (e.g. purposes from the one-call p7 answer,
    side effects from compact-t1a): other groups emptied, drugs no kept item refers to dropped; items unchanged."""
    out = {**record, 'purposes': [], 'side_effects': [], 'no_side_effects': []}
    for group in groups:
        out[group] = list(record.get(group, []))
    used = {item['drug'] for group in groups for item in out[group]}
    out['drugs'] = {h: d for h, d in record['drugs'].items() if h in used}
    return out


# --- compact-r1: short review of relations and drug binding (restart control, 2026-09-27) --------------------------
# One extra none call per record that sees the source text and a numbered list of the items another run extracted. It
# judges only the drug and the relation of each item (correct / wrong_drug / wrong_group / not_in_text), never the
# item's other fields, and adds complete items for statements that are missing or were judged wrong. The list is prompt
# context only: the record's evidence fields are the source text, so added items are validated and located in the
# source alone. Each review spec is bound to one frozen context file (its sha256 is part of the prompt version).

REVIEW_CONTEXT = Path(__file__).resolve().parents[3]/'.cache/extraction-optimization/review-context'


class ReviewCheck(Wire):
    item: Label
    verdict: Literal['correct', 'wrong_drug', 'wrong_group', 'not_in_text']


class ReviewR1(Wire):
    checks: list[ReviewCheck]
    drugs: list[WireDrug]
    purposes: list[WirePurpose]
    side_effects: list[WireSideEffect]
    no_side_effects: list[WireNoSideEffect]


def r1_to_canonical(answer):
    """The added items only (checks are applied by apply_review)."""
    return {'drugs': [d.model_dump() for d in answer.drugs], 'purposes': [p.model_dump() for p in answer.purposes],
            'side_effects': [x.model_dump() for x in answer.side_effects], 'no_side_effects': [x.model_dump() for x in answer.no_side_effects]}


def _r1(text):
    head = ("You review an automatic extraction of drug statements from one social-media post or health review. The input gives the source text "
            "and, under \"Extraction to review\", numbered items (P = purpose, A = side effect, N = no-side-effect statement).\n\n"
            "Review (reasoning about the source text only):\n"
            "- checks: one entry per numbered item. verdict: \"correct\" when the source text states it for that drug and in that group; "
            "\"wrong_drug\" when the statement is about another drug (e.g. a relief medicine given the symptoms it relieves, or another brand); "
            "\"wrong_group\" when it is not that kind of statement (e.g. a symptom the drug causes listed as a purpose, a condition it treats listed as a side effect, an outcome, benefit, plan, device or supplement listed as a purpose or side effect); "
            "\"not_in_text\" when the source text does not state it. Judge only the drug and the group; ignore the item's other fields.\n"
            "- drugs, purposes, side_effects, no_side_effects: add complete items for every statement the extraction missed, and the corrected version of every item you did not judge correct, following the rules below. "
            "Never repeat an item judged correct. Quotes and surfaces come from the source text only, never from the extraction list.\n\n"
            "Rules for added items:\n")
    return head + text


PROMPT_R1 = _r1(PROMPT_V7)


def render_context(record, fields):
    """Numbered list of a compact record's items for the review prompt (spans rendered from the source)."""
    from .compact import render_span
    lines = []
    for group, tag, key in (('purposes', 'P', 'indication'), ('side_effects', 'A', 'effect'), ('no_side_effects', 'N', 'scope_text')):
        for i, item in enumerate(record.get(group, []), start=1):
            d = record['drugs'][item['drug']]
            what = item.get(key) or ('no side effects' if group == 'no_side_effects' else '')
            quotes = ' | '.join(render_span(s, fields) for s in item['ev'])
            lines.append(f'{tag}{i}: drug "{d["surface"]}" ({d["ref"]}); {group[:-1].replace("_", " ")}: "{what}"; evidence: "{quotes}"')
    return '\n'.join(lines) or '(no items)'


def review_spec(context_name):
    """Register compact-r1@<context_name>: the review prompt with the frozen context file review-context/<name>.jsonl
    ({record_id, context} lines) rendered after the source text."""
    import hashlib
    import json as _json
    from .compact import render_input
    path = REVIEW_CONTEXT/f'{context_name}.jsonl'
    data = path.read_bytes()
    ctx = {r['record_id']: r['context'] for r in (_json.loads(x) for x in data.decode().splitlines() if x.strip())}
    sha = hashlib.sha256(data).hexdigest()[:16]

    def render(inp):
        return render_input(inp) + '\nExtraction to review:\n<<<\n' + ctx.get(inp.record_id, '(no items)') + '\n>>>'

    return register(Spec(f'compact-r1@{context_name}', PROMPT_R1, ReviewR1, 'compact-wire-r1-1.0.0', f'compact-prompt-r1-1.0.0+ctx-{sha}',
                         schema_name='weightloss_compact_review_r1', to_canonical=r1_to_canonical, render=render, text_verbosity='low'))


def apply_review(record, checks, additions, run):
    """Deterministic application of a review: items the review judged anything but correct are removed, the review's
    added items (validated in the source) are appended; checks naming no item are ignored and reported by the caller."""
    tags = {}
    for group, tag in (('purposes', 'P'), ('side_effects', 'A'), ('no_side_effects', 'N')):
        for i in range(len(record.get(group, []))):
            tags[f'{tag}{i + 1}'] = (group, i)
    drop = {tags[c['item']] for c in checks if c['verdict'] != 'correct' and c['item'] in tags}
    kept = {**record, **{g: [it for i, it in enumerate(record.get(g, [])) if (g, i) not in drop] for g in ('purposes', 'side_effects', 'no_side_effects')}}
    kept = select_groups(kept, ['purposes', 'side_effects', 'no_side_effects'])
    return merge_compact([kept, {**additions, 'record_id': record['record_id']}], run)


# --- compact-t2p: purpose task with a non-purpose outlet (restart round T3, 2026-09-27) -----------------------------
# t1p (purposes only) turned implicit weight outcomes, benefits and plans into purposes (57 false positives on dev-128):
# with a single group the model has nowhere to put a drug statement that is not a purpose. t2p keeps t1p's purposes and
# adds other_statements, a short list for drug-linked statements that are not purposes (what the drug caused or might
# cause, results and benefits, plans, doses, costs). It is wire-only: dropped by the conversion, never stored. Side
# effects still come from compact-t1a.

class OtherStatement(Wire):
    drug: Handle
    words: Surface
    kind: Literal['caused_or_might_cause', 'result_or_benefit', 'plan_dose_or_cost', 'other']


class ExtractionT2P(Wire):
    drugs: list[WireDrug]
    purposes: list[WirePurpose]
    other_statements: list[OtherStatement]


def t2p_to_canonical(answer):
    return t1p_to_canonical(answer)


def _t2p(text):
    text = _rep(text, "- A condition or symptom the text says a drug causes, worsens or might cause is never a purpose of that drug.\n",
                "- A condition or symptom the text says a drug causes, worsens or might cause is never a purpose of that drug.\n"
                "- other_statements: every other statement about a drug that is not a purpose goes here instead, with the drug's handle, "
                "the words as written (words) and kind: \"caused_or_might_cause\" (a symptom or effect the drug causes or might cause), "
                "\"result_or_benefit\" (e.g. weight lost, less appetite, the food noise is gone), \"plan_dose_or_cost\" (dosing, switching, "
                "insurance, price) or \"other\". A statement is either a purpose or an other_statement, never both.\n")
    return text


PROMPT_T2P = _t2p(PROMPT_T1P)
COMPACT_T2P = register(Spec('compact-t2p', PROMPT_T2P, ExtractionT2P, 'compact-wire-t2p-1.0.0', 'compact-prompt-t2p-1.0.0',
                            schema_name='weightloss_compact_extraction_t2p', to_canonical=t2p_to_canonical, text_verbosity='low'))


# --- compact-s5: one call, side effects generated before purposes, plus the two direction rules (restart, 2026-09-27) --
# Measured: the side-effect-only task (t1a) extracts side effects better than p7's combined call, but every purpose-only
# task over-extracts purposes, and two calls cost about as much as reasoning low while gaining less. s5 stays one call:
# p7's fields and rules, with the groups generated in the order drugs, side_effects, no_side_effects, purposes, plus the
# scope sentence and the two direction rules from T1 (a caused condition is never a purpose; a treated condition is never
# a side effect).

class ExtractionS5(Wire):
    drugs: list[WireDrug]
    side_effects: list[WireSideEffect]
    no_side_effects: list[WireNoSideEffect]
    purposes: list[WirePurpose]


def s5_to_canonical(answer):
    return {'drugs': [d.model_dump() for d in answer.drugs], 'purposes': [p.model_dump() for p in answer.purposes],
            'side_effects': [x.model_dump() for x in answer.side_effects], 'no_side_effects': [x.model_dump() for x in answer.no_side_effects]}


def _s5(text):
    start = text.index("- purposes: what a drug is used, prescribed or wanted for")
    mid = text.index("- side_effects: unwanted effects the text links to a drug")
    end = text.index("- surface, indication and effect are copied")
    purposes, rest = text[start:mid], text[mid:end]
    text = text[:start] + rest + purposes + text[end:]
    text = _rep(text, "Groups:\n", "Groups (in this order):\n" + _SCOPE)
    text = _rep(text, "- surface, indication and effect are copied",
                "- A condition a drug is taken for, prescribed for or said to treat or help is never a side effect of that drug, and a condition or symptom the text says a drug causes, worsens or might cause is never a purpose of that drug.\n- surface, indication and effect are copied")
    return text


PROMPT_S5 = _s5(PROMPT_V7)
COMPACT_S5 = register(Spec('compact-s5', PROMPT_S5, ExtractionS5, 'compact-wire-s5-1.0.0', 'compact-prompt-s5-1.0.0',
                           schema_name='weightloss_compact_extraction_s5', to_canonical=s5_to_canonical, text_verbosity='low'))


# --- compact-p9: one call, p7 with self-describing attribution values (single-call route, 2026-09-27) ------------------
# The user rejected every combination route (two-stage, split, review, routing): one Luna none call per record, only the
# schema and prompt change. Measured on four p7-none runs (dev-128 a/b, fresh-64b, fresh-64c): attribution is the largest
# residual class (97 errors), 60 of them not_stated where the author links the effect to the drug (cause or time) and 20
# uncertain from questions about easing; over-attribution is 8. p9 keeps p7's three lists, every field and every other
# enum; only the wire values of attribution are renamed so that each says what it means (one-to-one with the canonical
# values, so no information is added or lost and a time link is never promoted to a cause), and the prompt moves the
# relation direction rules first (relief-drug sentence moved, not repeated), states the attribution values with the
# author's time link made explicit, and names dose aliases in the coreference rule. The prompt is no longer than p7.
# (compact-p8, a concise rewrite at half of p7's length, lost side-effect recall at none; hence edits, not a rewrite.)

P9_ATTRIBUTION = {'caused': 'attributed', 'linked_in_time': 'temporal_only', 'unsure': 'uncertain',
                  'other_cause': 'other_cause', 'no_link': 'not_stated'}


class WireSideEffectP9(WireSideEffect):
    attribution: Literal['caused', 'linked_in_time', 'unsure', 'other_cause', 'no_link']


class ExtractionP9(Wire):
    drugs: list[WireDrug]
    purposes: list[WirePurpose]
    side_effects: list[WireSideEffectP9]
    no_side_effects: list[WireNoSideEffect]


def p9_to_canonical(answer):
    return {'drugs': [d.model_dump() for d in answer.drugs], 'purposes': [p.model_dump() for p in answer.purposes],
            'side_effects': [{**x.model_dump(), 'attribution': P9_ATTRIBUTION[x.attribution]} for x in answer.side_effects],
            'no_side_effects': [x.model_dump() for x in answer.no_side_effects]}


_P7_RELIEF = ("- A medicine taken to relieve symptoms (e.g. Zofran, Pepto, Imodium) is not their cause unless the text says so; its own use is a purpose item (e.g. Pepto -> \"stomach issues\", Imodium -> \"side effects from metformin\"): \"throwing up even after taking zofran\" is a side effect of the drug taken for weight or diabetes, not of Zofran. When that causing drug is never named, create an unspecified drug whose surface is the words that refer to it (e.g. \"the shot\", \"my dose\", \".25\").\n")


def _p9(text):
    text = _rep(text, "never assume the writer, current use, an affirmed purpose or an attributed effect.\n",
                "never assume the writer, current use, an affirmed purpose or a cause the text does not give.\n\n"
                "Direction:\n"
                "- A symptom the author links to a drug is a side effect of that drug.\n"
                "- What a drug is taken for or said to treat or help is its purpose. So a medicine taken to relieve symptoms (e.g. Zofran, Pepto, Imodium) has them as its purpose (e.g. Pepto -> \"stomach issues\") and is not their cause unless the text says so: \"throwing up even after taking zofran\" is a side effect of the drug taken for weight or diabetes (unspecified if never named, e.g. \"the shot\", \".25\").\n"
                "- A result or benefit by itself (e.g. \"lost 9 lbs\", \"A1C is 5.5\", \"the food noise is gone\") is neither; a stated reason for use is a purpose in any wording.\n")
    text = _cut(text, _P7_RELIEF)
    text = _rep(text, "- Resolve references such as \"it\", \"this med\", \"this one\", \"the pill\", \"the shot\" or \"these meds\" to",
                "- Resolve references such as \"it\", \"this med\", \"this one\", \"the pill\", \"the shot\", \"these meds\" or a bare dose (e.g. \"the 5\") to")
    text = _rep(text, "Not purposes: results by themselves (e.g. \"lost 9 lbs\", \"A1C is 5.5\", \"the food noise is gone\"), conditions that are only listed (e.g. \"I also have PCOS\"), and dosing or diet plans.",
                "Not purposes: results by themselves, conditions that are only listed (e.g. \"I also have PCOS\"), and dosing or diet plans.")
    text = _rep(text, "the drug wearing off, benefits, intended weight loss, and side effects", "the drug wearing off, and side effects")
    text = _rep(text, " \"I have nausea; could Ozempic be causing it?\" is occurred + question + uncertain attribution.", "")
    text = _rep(text, " (e.g. \"nausea ... it's been very minimal\" is occurred with severity null unless a severity word is used)", "")
    text = _rep(text, "Statements about people in general, or about another person, never take the writer's use values; give them only use values stated for them.",
                "Statements about people in general or another person take only use values stated for them.")
    text = _rep(text, "- surface, indication and effect are copied character for character from the input (same spelling and capitalization), never normalized or translated; indication and effect name the condition, goal or symptom itself, never a pronoun such as \"that\" or \"it\" standing for it.",
                "- surface, indication and effect are copied exactly from the input (same spelling and capitalization), never normalized; indication and effect name the condition, goal or symptom itself, never a pronoun such as \"that\" or \"it\".")
    text = _rep(text, "occurrence says whether the symptom happens; asking whether the drug causes it does not change that (\"My hair is thinning; could X be causing it?\" is occurred + question + attribution uncertain).",
                "occurrence says only whether the symptom happens; asking whether the drug causes it does not change that (\"I have nausea; could Ozempic be causing it?\" is occurred + question + attribution unsure).")
    start = text.index("- attribution (side effects;")
    end = text.index("- use_status:")
    text = (text[:start] +
            "- attribution (side effects; the author's view, not medical causality): \"caused\" when the author presents the effect as caused by or a side effect of the drug (\"side effects like\", \"gave me\", \"makes me\", \"from the shot\", \"caused\"); \"linked_in_time\" when the author only ties when it happened to the drug's use, a dose or stopping (\"after each shot\", \"since starting\"), not merely because both appear in the same passage; \"unsure\" when the author doubts the drug caused it or also suspects another cause (e.g. \"maybe the needle\"); \"other_cause\" when the author blames something else; \"no_link\" only when the text gives neither a cause nor a time link. Questions about whether or when it will ease never change attribution.\n"
            + text[end:])
    return text


PROMPT_P9 = _p9(PROMPT_V7)
COMPACT_P9 = register(Spec('compact-p9', PROMPT_P9, ExtractionP9, 'compact-wire-p9-1.0.0', 'compact-prompt-9.0.0',
                           schema_name='weightloss_compact_extraction_p9', to_canonical=p9_to_canonical, text_verbosity='low'))


# --- compact-p10: p7 + only the attribution change of p9 (2026-09-27) ---------------------------------------------------
# dev-p9-none-a (dev-128, corrections-v3): attribution errors 29 -> 19, but the Direction block pulled benefits, goals and
# support into purposes (false positives 15 -> 29), bound a symptom to five relief medicines (side-effect misattribution
# 1 -> 7) and the moved relief sentence lost "surface is the words that refer to it" (invented unspecified surfaces; 4
# records lost). p10 isolates the one schema change: p9's attribution wire values and their definition (the author's
# stated time link; appearing in the same passage is not one; questions about easing never change attribution), the
# occurrence example renamed to the new value (duplicate example dropped) and the first line no longer telling the model
# never to assume an attributed effect. Every other character of p7 is unchanged.

def _p10(text):
    text = _rep(text, "never assume the writer, current use, an affirmed purpose or an attributed effect.",
                "never assume the writer, current use, an affirmed purpose or a cause the text does not give.")
    text = _rep(text, " \"I have nausea; could Ozempic be causing it?\" is occurred + question + uncertain attribution.", "")
    text = _rep(text, "(\"My hair is thinning; could X be causing it?\" is occurred + question + attribution uncertain).",
                "(\"I have nausea; could Ozempic be causing it?\" is occurred + question + attribution unsure).")
    start = text.index("- attribution (side effects;")
    end = text.index("- use_status:")
    return (text[:start] +
            "- attribution (side effects; the author's view, not medical causality): \"caused\" when the author presents the effect as caused by or a side effect of the drug (\"side effects like\", \"gave me\", \"makes me\", \"from the shot\", \"caused\"); \"linked_in_time\" when the author only ties when it happened to the drug's use, a dose or stopping (\"after each shot\", \"since starting\"), not merely because both appear in the same passage; \"unsure\" when the author doubts the drug caused it or also suspects another cause (e.g. \"maybe the needle\"); \"other_cause\" when the author blames something else; \"no_link\" only when the text gives neither a cause nor a time link. Questions about whether or when it will ease never change attribution.\n"
            + text[end:])


PROMPT_P10 = _p10(PROMPT_V7)
COMPACT_P10 = register(Spec('compact-p10', PROMPT_P10, ExtractionP9, 'compact-wire-p9-1.0.0', 'compact-prompt-10.0.0',
                            schema_name='weightloss_compact_extraction_p9', to_canonical=p9_to_canonical, text_verbosity='low'))


# --- compact-p11: p10 with p7's first line (2026-09-27) -----------------------------------------------------------------
# dev-p10-none-a: side effects improved (attribution errors 29 -> 22, misattributed 1 -> 0, precision 0.61-0.63 -> 0.66)
# while purposes, whose rules are byte-identical to p7, regressed (false positives 15/15 -> 23; 14 records vs 9-11, up to 7
# per record); p9 showed the same shift. p11 separates the two p10 edits: p7's first line ("never assume ... an affirmed
# purpose or an attributed effect") is restored, only the attribution value names/definition and the occurrence example
# stay changed.

def _p11(text):
    text = _rep(text, " \"I have nausea; could Ozempic be causing it?\" is occurred + question + uncertain attribution.", "")
    text = _rep(text, "(\"My hair is thinning; could X be causing it?\" is occurred + question + attribution uncertain).",
                "(\"I have nausea; could Ozempic be causing it?\" is occurred + question + attribution unsure).")
    start = text.index("- attribution (side effects;")
    end = text.index("- use_status:")
    p10 = PROMPT_P10
    line = p10[p10.index("- attribution (side effects;"):p10.index("- use_status:")]
    return text[:start] + line + text[end:]


PROMPT_P11 = _p11(PROMPT_V7)
COMPACT_P11 = register(Spec('compact-p11', PROMPT_P11, ExtractionP9, 'compact-wire-p9-1.0.0', 'compact-prompt-11.0.0',
                            schema_name='weightloss_compact_extraction_p9', to_canonical=p9_to_canonical, text_verbosity='low'))


# --- compact-p12: p11 + one non-GLP-1 purpose example (2026-09-27) ------------------------------------------------------
# fresh-64d (validation, two runs each): p11 keeps the side-effect gain (0.80/0.77 vs p7 0.71-0.73/0.61-0.64) but both specs
# miss purposes of other medicines, and p11 skipped co-medications p7 found (metformin "T2D Medication", a statin, Myralex:
# 4 of 6 chances vs 0 of 6). Every purpose example in p7 is about the weight or diabetes drug. p12 adds one example of
# another medicine's stated purpose, taken from no sample ("lisinopril for my blood pressure" occurs in no evaluation
# input); no rule or field changes. Tested on dev-128, fresh-64b and fresh-64c only (fresh-64d stays the validation
# sample and is not used to tune p12).

def _p12(text):
    return _rep(text, "\"I take it to quiet the food noise\", \"it helps my inflammation\")",
                "\"I take it to quiet the food noise\", \"it helps my inflammation\", \"lisinopril for my blood pressure\")")


PROMPT_P12 = _p12(PROMPT_P11)
COMPACT_P12 = register(Spec('compact-p12', PROMPT_P12, ExtractionP9, 'compact-wire-p9-1.0.0', 'compact-prompt-12.0.0',
                            schema_name='weightloss_compact_extraction_p9', to_canonical=p9_to_canonical, text_verbosity='low'))


# --- compact-p13: p11 + Codex's fresh-64d review rules (2026-09-27) -----------------------------------------------------
# Codex's independent review of 24 fresh-64d records kept p11's side-effect gain but found: purposes of other medicines
# missed (antidepressants -> depression, Myralex -> constipation), "not much as far as side effects go" stored as a
# no-side-effect claim, a coordinated phrase ("stomach and breathing problems") shortened to words not in the text so the
# whole record was rejected, and quotes that do not state the item (a no-side-effect claim quoted with weight text). p13
# states each as a short rule in p11's existing lines (no new field, no example taken from a sample): purposes cover every
# medicine in the text; few, minimal or not much side effects are side effects, not a no-side-effect claim; words shared
# by several symptoms keep the whole phrase as one effect; quotes state the item. The "very minimal" parenthetical is
# dropped as covered by the severity rule.

def _p13(text):
    text = _rep(text, "- purposes: what a drug is used, prescribed or wanted for, as the text states it:",
                "- purposes: what each medicine in the text (the weight or diabetes drug and any other or relief medicine) is used, prescribed or wanted for, as the text states it:")
    text = _rep(text, "\"Tolerated it well\" or \"doing fine\" is not explicit.",
                "\"Tolerated it well\" or \"doing fine\" is not explicit, and few, minimal or not much side effects are not none: they are a side_effects item (e.g. effect \"minimal side effects\").")
    text = _rep(text, "never a pronoun such as \"that\" or \"it\" standing for it.",
                "never a pronoun such as \"that\" or \"it\" standing for it. When symptoms share words (e.g. \"joint and muscle pain\"), copy the whole phrase as one effect; never build words that are not in the text.")
    text = _rep(text, " (e.g. \"nausea ... it's been very minimal\" is occurred with severity null unless a severity word is used)", "")
    text = _rep(text, "that support the item and its fields;", "that state the item and support its fields;")
    return text


PROMPT_P13 = _p13(PROMPT_P11)
COMPACT_P13 = register(Spec('compact-p13', PROMPT_P13, ExtractionP9, 'compact-wire-p9-1.0.0', 'compact-prompt-13.0.0',
                            schema_name='weightloss_compact_extraction_p9', to_canonical=p9_to_canonical, text_verbosity='low'))


# --- compact-p14: p13 + the negated/feared severity clause (2026-09-27) --------------------------------------------------
# The records lost in every attempt (both p7 and p13) are mostly schema_invalid: a severity word inside a negated or feared
# symptom ("No severe stomach pain", "severe risks") is put into severity although the item is not_occurred or
# not_asserted, and the validator (correctly) rejects the whole record. p14 adds one clause to the existing event-field
# rule: the severity word stays in the effect words and severity stays null. Nothing else changes; the validator is not
# relaxed and no output is rewritten.

def _p14(text):
    return _rep(text, "With occurrence \"not_occurred\" or \"not_asserted\" all of them stay null or [].",
                "With occurrence \"not_occurred\" or \"not_asserted\" all of them stay null or [], even when the effect words contain a severity word (e.g. \"no severe pain\": effect \"severe pain\", severity null).")


PROMPT_P14 = _p14(PROMPT_P13)
COMPACT_P14 = register(Spec('compact-p14', PROMPT_P14, ExtractionP9, 'compact-wire-p9-1.0.0', 'compact-prompt-14.0.0',
                            schema_name='weightloss_compact_extraction_p9', to_canonical=p9_to_canonical, text_verbosity='low'))


# --- compact-d1: drug-centric single call (2026-09-27, after Codex's fresh-64e review) -----------------------------------
# Codex's review of fresh-64e (p13 not accepted): purposes of other medicines are still missed ("I take medication for my
# anxiety disorder"; a current second GLP-1 and blood-pressure/psoriasis medicines left out while only the reviewed drug
# was covered), welcome or expected effects become side effects (calm, fullness), an ineffective relief medicine becomes
# a side effect, and a good effect becomes a purpose. d1 makes the drug the outer object: every medicine the text mentions
# for someone's use gets one entry that carries its own purposes, side_effects and no_side_effects, with exactly p13's
# item fields (p11 attribution values) and no handles. Not a repeat of s2 (drug x subject x episode contexts whose items
# inherited subject and use, plus s1's cause/timing aids) or s4 (inline drug on every item): here nothing is shared or
# inherited, only the grouping changes. d1_to_canonical assigns d1, d2, ... in order, merges entries with the exact same
# surface and ref (same drug named the same way), and leaves out entries with no statement: they carry no assertion and
# the compact record never stores a drug no item refers to. Items are copied unchanged apart from the attribution mapping.

class PurposeD1(Wire):
    indication: Surface
    subject: Subject
    subject_label: Optional[Label]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    statement_kind: StatementKind
    polarity: Optional[Literal['affirmed', 'negated', 'uncertain']]
    quotes: Quotes


class SideEffectD1(Wire):
    effect: Surface
    subject: Subject
    subject_label: Optional[Label]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    statement_kind: StatementKind
    occurrence: Literal['occurred', 'not_occurred', 'uncertain', 'not_asserted']
    attribution: Literal['caused', 'linked_in_time', 'unsure', 'other_cause', 'no_link']
    severity: Optional[Literal['mild', 'moderate', 'severe']]
    event_dose: Optional[DoseText]
    onset: Optional[ShortText]
    event_duration: Optional[ShortText]
    clinical_consequences: list[Consequence]
    outcome: Optional[Literal['resolved', 'ongoing', 'unknown']]
    quotes: Quotes


class NoSideEffectD1(Wire):
    subject: Subject
    subject_label: Optional[Label]
    scope_text: Optional[ShortText]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    quotes: Quotes


class DrugD1(Wire):
    surface: Surface
    ref: Literal['name', 'class', 'unspecified']
    purposes: list[PurposeD1]
    side_effects: list[SideEffectD1]
    no_side_effects: list[NoSideEffectD1]


class ExtractionD1(Wire):
    drugs: list[DrugD1]


def d1_to_canonical(answer):
    handles, out = {}, {'drugs': [], 'purposes': [], 'side_effects': [], 'no_side_effects': []}
    for d in answer.drugs:
        if not (d.purposes or d.side_effects or d.no_side_effects):
            continue
        key = (d.surface, d.ref)
        if key not in handles:
            handles[key] = f'd{len(handles) + 1}'
            out['drugs'].append({'handle': handles[key], 'surface': d.surface, 'ref': d.ref})
        h = handles[key]
        out['purposes'] += [{'drug': h, **p.model_dump()} for p in d.purposes]
        out['side_effects'] += [{'drug': h, **x.model_dump(), 'attribution': P9_ATTRIBUTION[x.attribution]} for x in d.side_effects]
        out['no_side_effects'] += [{'drug': h, **x.model_dump()} for x in d.no_side_effects]
    return out


def _d1(text):
    text = _rep(text, "Drugs (every key present; null when the text does not say; [] when a group is empty):\n- drugs: one entry per medication that an item below refers to. handle: d1, d2, ... in order of first mention. surface:",
                "Drugs (every key present; null when the text does not say; [] when a list is empty):\n- drugs: one entry for each medicine the text mentions for someone's use (the weight or diabetes drug and any other or relief medicine), each holding its own purposes, side_effects and no_side_effects (any of them may be []); put every statement under the drug it is about. surface:")
    text = _rep(text, "to the named drug they clearly refer to and use that drug's handle.", "to the named drug they clearly refer to and put the statement under that drug.")
    text = _rep(text, "Groups:\n", "Each drug's lists:\n")
    text = _rep(text, "- purposes: what each medicine in the text (the weight or diabetes drug and any other or relief medicine) is used, prescribed or wanted for,",
                "- purposes: what the drug is used, prescribed or wanted for,")
    text = _rep(text, "Not purposes: results by themselves (e.g. \"lost 9 lbs\", \"A1C is 5.5\", \"the food noise is gone\"),",
                "Not purposes: results or welcome effects by themselves (e.g. \"lost 9 lbs\", \"A1C is 5.5\", \"the food noise is gone\", sleeping better), unless given as the reason for use,")
    text = _rep(text, "Not side effects: foods or remedies, emotions or worries (e.g. \"nervous\"),",
                "Not side effects: welcome or expected effects (e.g. feeling calm, sleeping well, feeling full), a medicine that did not work, foods or remedies, emotions or worries (e.g. \"nervous\"),")
    text = _cut(text, "- drug: the handle of the drug the statement is about.\n")
    text = _rep(text, "If the input has no purpose, side effect or no-side-effect statement, return empty groups.",
                "If the input mentions no medicine, drugs is [].")
    return text


PROMPT_D1 = _d1(PROMPT_P13)
COMPACT_D1 = register(Spec('compact-d1', PROMPT_D1, ExtractionD1, 'compact-wire-d1-1.0.0', 'compact-prompt-d1-1.0.0',
                           schema_name='weightloss_compact_extraction_d1', to_canonical=d1_to_canonical, text_verbosity='low'))


# --- compact-d2: d1 + dose aliases are not drug entries (2026-09-27) ----------------------------------------------------
# d1 on fresh-64e: purposes 0.77/0.71 (p13 0.50-0.55, p7 0.45-0.68) and Codex's review cases mostly fixed, but the
# drug-centric list invites doses as containers: one record got entries "2.5 mg" and "5 mg" (8 statements left
# unresolved) although the drug is named. d2 adds one clause to the existing coreference line; nothing else changes.

def _d2(text):
    return _rep(text, "- Resolve references such as \"it\", \"this med\", \"this one\", \"the pill\", \"the shot\" or \"these meds\" to the named drug they clearly refer to and put the statement under that drug.",
                "- Resolve references such as \"it\", \"this med\", \"this one\", \"the pill\", \"the shot\" or \"these meds\" to the named drug they clearly refer to and put the statement under that drug; a dose written alone (e.g. \"2.5 mg\") is not a drug entry but belongs to the drug it is a dose of.")


PROMPT_D2 = _d2(PROMPT_D1)
COMPACT_D2 = register(Spec('compact-d2', PROMPT_D2, ExtractionD1, 'compact-wire-d1-1.0.0', 'compact-prompt-d2-1.0.0',
                           schema_name='weightloss_compact_extraction_d1', to_canonical=d1_to_canonical, text_verbosity='low'))


# --- compact-d3: d2 + severity-v2 semantics (separate schema candidate, 2026-09-27) -------------------------------------
# Codex's validity decision: a severity word inside a negated or feared effect describes that effect, so the rule that
# every event field is null for not_occurred / not_asserted items conflicts with the text ("No severe stomach pain",
# "severe risks"): those records were rejected in every attempt. d3 keeps d2 unchanged except the severity line and uses
# the opt-in canonical version compact-canonical-sev-1.0.0 (compact.WireSideEffectSev): severity may stand with any
# occurrence and never makes the effect occur; event dose, onset, event duration, clinical consequences and outcome
# still stay empty for those items. compact-wire-1.0.0 and every earlier spec and stored record are unchanged.

from .compact import WireExtractionSev  # noqa: E402


def _d3(text):
    text = _rep(text, "never infer it from descriptive words such as \"rough\", \"awful\" or \"bad\".",
                "never infer it from descriptive words such as \"rough\", \"awful\" or \"bad\". It is the intensity the text gives the effect, also when the effect did not occur or is only feared (e.g. \"no severe pain\": occurrence \"not_occurred\", severity \"severe\"), and never makes it occur.")
    text = _rep(text, "With occurrence \"not_occurred\" or \"not_asserted\" all of them stay null or [].",
                "With occurrence \"not_occurred\" or \"not_asserted\" event_dose, onset, event_duration, clinical_consequences and outcome stay null or [].")
    return text


PROMPT_D3 = _d3(PROMPT_D2)
COMPACT_D3 = register(Spec('compact-d3', PROMPT_D3, ExtractionD1, 'compact-wire-d1-1.0.0', 'compact-prompt-d3-1.0.0',
                           schema_name='weightloss_compact_extraction_d1', to_canonical=d1_to_canonical, text_verbosity='low',
                           canonical=('compact-canonical-sev-1.0.0', WireExtractionSev)))


# --- compact-u1: d3 with the groups named for their meaning (2026-09-27, after Codex's fresh-64f review) ----------------
# Codex's fresh-64f review (d2/d3 not accepted): the residual errors are relation direction - a condition the drug helps
# given as its side effect ("alcohol dependency"), recovery ("finally able to eat again") given as a new side effect, a
# negated use ("does not fix disordered eating") turned into a side effect, an opinion about the drug given as a purpose,
# and one negated symptom widened into a no-side-effect claim. As with the attribution values (the one robust gain), u1
# names the model-facing groups for what they mean: therapeutic_uses (canonical purposes) and unwanted_effects (canonical
# side_effects), one-to-one, every field kept; the definitions state that relief or recovery is not a new unwanted
# effect, that a condition the drug treats or helps is a use, that a negated treatment claim stays a use with polarity
# negated, and that a negated specific symptom is never a no-side-effect claim. Base: d3 (accepted severity canonical).

class DrugU1(Wire):
    surface: Surface
    ref: Literal['name', 'class', 'unspecified']
    therapeutic_uses: list[PurposeD1]
    unwanted_effects: list[SideEffectD1]
    no_side_effects: list[NoSideEffectD1]


class ExtractionU1(Wire):
    drugs: list[DrugU1]


def u1_to_canonical(answer):
    handles, out = {}, {'drugs': [], 'purposes': [], 'side_effects': [], 'no_side_effects': []}
    for d in answer.drugs:
        if not (d.therapeutic_uses or d.unwanted_effects or d.no_side_effects):
            continue
        key = (d.surface, d.ref)
        if key not in handles:
            handles[key] = f'd{len(handles) + 1}'
            out['drugs'].append({'handle': handles[key], 'surface': d.surface, 'ref': d.ref})
        h = handles[key]
        out['purposes'] += [{'drug': h, **p.model_dump()} for p in d.therapeutic_uses]
        out['side_effects'] += [{'drug': h, **x.model_dump(), 'attribution': P9_ATTRIBUTION[x.attribution]} for x in d.unwanted_effects]
        out['no_side_effects'] += [{'drug': h, **x.model_dump()} for x in d.no_side_effects]
    return out


def _u1(text):
    reps = [
        ("each holding its own purposes, side_effects and no_side_effects", "each holding its own therapeutic_uses, unwanted_effects and no_side_effects"),
        ("its own use is a purpose item", "its own use is a therapeutic_uses item"),
        ("is a side effect of the drug taken for weight or diabetes, not of Zofran", "is an unwanted effect of the drug taken for weight or diabetes, not of Zofran"),
        ("- purposes: what the drug is used, prescribed or wanted for, as the text states it: a condition or goal given as the reason for use, or a condition the drug is said to treat or help (e.g.",
         "- therapeutic_uses: what the drug is used, prescribed or wanted for, as the text states it: a condition or goal given as the reason for use, or a condition, habit or symptom the text says the drug treats, helps or reduces (e.g."),
        ("\"it helps my inflammation\"). indication:",
         "\"it helps my inflammation\"); when the text says the drug does not treat or help a condition, it is still a therapeutic_uses item, with polarity \"negated\". indication:"),
        ("Not purposes: results or welcome effects by themselves", "Not therapeutic uses: opinions about the drug (e.g. whether it is good for someone), results or welcome effects by themselves"),
        ("add it as a purpose (report, polarity affirmed) quoting condition_field", "add it as a therapeutic_uses item (report, polarity affirmed) quoting condition_field"),
        ("- side_effects: unwanted effects the text links to a drug,", "- unwanted_effects: symptoms or bodily changes that the text presents as unwanted and links to a drug,"),
        ("less food noise are not side effects unless the text presents them as a problem", "less food noise are not unwanted effects unless the text presents them as a problem"),
        ("Not side effects: welcome or expected effects", "Not unwanted effects: relief, recovery or improvement (e.g. being able to eat again, a symptom easing), a condition the drug treats or helps (a therapeutic use), welcome or expected effects"),
        ("they are a side_effects item (e.g. effect", "they are an unwanted_effects item (e.g. effect"),
        ("never also add the claim as a side_effects item or give it a severity", "never also add the claim as an unwanted_effects item or give it a severity"),
        ("A specific negated symptom (\"no nausea\") is a side_effects item with occurrence \"not_occurred\".",
         "A specific negated symptom (\"no nausea\") is an unwanted_effects item with occurrence \"not_occurred\", never a no_side_effects claim."),
        ("the drug wearing off, benefits, intended weight loss, and side effects", "the drug wearing off, and side effects"),
        ("- polarity (purposes):", "- polarity (therapeutic uses):"),
        ("- occurrence (side effects):", "- occurrence (unwanted effects):"),
        ("- attribution (side effects;", "- attribution (unwanted effects;"),
    ]
    for a, b in reps:
        text = _rep(text, a, b)
    return text


PROMPT_U1 = _u1(PROMPT_D3)
COMPACT_U1 = register(Spec('compact-u1', PROMPT_U1, ExtractionU1, 'compact-wire-u1-1.0.0', 'compact-prompt-u1-1.0.0',
                           schema_name='weightloss_compact_extraction_u1', to_canonical=u1_to_canonical, text_verbosity='low',
                           canonical=('compact-canonical-sev-1.0.0', WireExtractionSev)))


# --- compact-u2: u1 with p7's use scope (2026-09-27) --------------------------------------------------------------------
# u1 (two runs each): on fresh-64f the direction errors Codex listed nearly vanished (a use recorded as a side effect
# 1/1 vs d3 4/3; recovery as a side effect 0/1 vs 1/2; the negated use found in both runs vs none; widened
# no-side-effect claims 0/1 vs 1/3), but on dev-128 purposes gained false positives in both runs that d3 never made
# ("appetite", "hunger suppression", "helping me", other people's "weight loss"): intended effects read as uses because
# u1 widened the scope to "a condition, habit or symptom the text says the drug treats, helps or reduces". u2 restores
# p7's scope ("a condition the drug is said to treat or help"); the renamed groups, the unwanted-effect definition, the
# negated-use rule and the opinion exclusion stay.

def _u2(text):
    return _rep(text, "or a condition, habit or symptom the text says the drug treats, helps or reduces (e.g.",
                "or a condition the drug is said to treat or help (e.g.")


PROMPT_U2 = _u2(PROMPT_U1)
COMPACT_U2 = register(Spec('compact-u2', PROMPT_U2, ExtractionU1, 'compact-wire-u1-1.0.0', 'compact-prompt-u2-1.0.0',
                           schema_name='weightloss_compact_extraction_u1', to_canonical=u1_to_canonical, text_verbosity='low',
                           canonical=('compact-canonical-sev-1.0.0', WireExtractionSev)))


# --- compact-d4: d3 + binding of each field to its own drug and claim (2026-09-27, after Codex's fresh-64g review) -------
# Codex's fresh-64g review (u1 rejected, d3 kept as the structural baseline) lists persistent binding failures: a refusal
# of insurance coverage for weight loss stored as a negated use; "half" of a 25 mg pill stored as 25 mg; forty years of
# injections in general stored as the reviewed drug's duration; a leaking pen stored as a bodily side effect; "no side
# effects, except ..." stored as a global none; a condition_field purpose quoted with review text that does not state it.
# d4 edits d3's existing lines only (no field, no rename, no code filtering): fields describe only their own item's drug
# and claim; "negated" means the drug does not treat or is not taken for the condition, never a coverage/access/price
# refusal; dose and duration belong to this drug only (words that change the dose kept); device problems are not side
# effects; an exception narrows a no-side-effect claim and the excepted effect is its own item; quotes state the item's
# own claim (condition_field for a condition_field purpose).

def _d4(text):
    reps = [
        ("Item fields (independent of each other; never fill one from another):",
         "Item fields (each describes only this item's own drug and claim, as the words attached to them say; never fill one field from another):"),
        ("- polarity (purposes): \"affirmed\", \"negated\" or \"uncertain\"; null only for a pure question that asserts nothing. A report always has a polarity.",
         "- polarity (purposes): \"affirmed\", \"negated\" (the text says the drug does not treat or is not taken for it) or \"uncertain\"; null only for a pure question that asserts nothing. A report always has a polarity. A refusal of insurance coverage, access or price for a use (e.g. \"insurance won't cover it for weight loss\") does not negate the use."),
        ("- dose: the general dose as written (e.g. \"0.5 mg\"). duration: the total time on the drug as written (e.g. \"six weeks\"); it never fills onset or event_duration.",
         "- dose: the dose this subject takes of this drug as written, with any words that change it (e.g. \"0.5 mg\", \"half of the 25 mg\"). duration: the total time on this drug as written (e.g. \"six weeks\"), never time on other medicines or on injections in general; it never fills onset or event_duration."),
        ("dosing mistakes, the drug wearing off, benefits, intended weight loss, and side effects", "dosing mistakes or device problems (e.g. a leaking pen), the drug wearing off, and side effects"),
        ("\"I haven't had the severe side effects others have\" -> scope_text \"the severe side effects\"), else null;",
         "\"I haven't had the severe side effects others have\" -> scope_text \"the severe side effects\"; \"no side effects except the headaches\" -> scope_text \"except the headaches\", and the headaches are their own side_effects item), else null;"),
        ("that state the item and support its fields;", "that state this item's own claim and support its fields (a condition_field purpose quotes condition_field);"),
    ]
    for a, b in reps:
        text = _rep(text, a, b)
    return text


PROMPT_D4 = _d4(PROMPT_D3)
COMPACT_D4 = register(Spec('compact-d4', PROMPT_D4, ExtractionD1, 'compact-wire-d1-1.0.0', 'compact-prompt-d4-1.0.0',
                           schema_name='weightloss_compact_extraction_d1', to_canonical=d1_to_canonical, text_verbosity='low',
                           canonical=('compact-canonical-sev-1.0.0', WireExtractionSev)))


# --- compact-d5: d3 with evidence first in every item (2026-09-27) ------------------------------------------------------
# d4 (binding edits) fixed Codex's targeted fresh-64g cases but was a trade-off overall (two runs each): side effects more
# conservative (dev FP 12 vs 18, core found 62 vs 66.5), purposes with more false positives (dev 23 vs 19.5, fresh-64g 9.5
# vs 4.5), no significant record-level difference. d5 is a different narrow schema edit on d3: the quotes field comes first
# in every item (purposes, side effects, no-side-effect claims), so the model writes the passage that states the claim
# before the claim and its fields; the prompt only lists quotes first. Never tested alone: s1 bundled it with cause/timing,
# a relief flag and a purpose basis (the basis field caused s1's purpose false positives). No field added or removed; the
# converter, canonical record and severity canonical are d3's.

class PurposeD5(Wire):
    quotes: Quotes
    indication: Surface
    subject: Subject
    subject_label: Optional[Label]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    statement_kind: StatementKind
    polarity: Optional[Literal['affirmed', 'negated', 'uncertain']]


class SideEffectD5(Wire):
    quotes: Quotes
    effect: Surface
    subject: Subject
    subject_label: Optional[Label]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    statement_kind: StatementKind
    occurrence: Literal['occurred', 'not_occurred', 'uncertain', 'not_asserted']
    attribution: Literal['caused', 'linked_in_time', 'unsure', 'other_cause', 'no_link']
    severity: Optional[Literal['mild', 'moderate', 'severe']]
    event_dose: Optional[DoseText]
    onset: Optional[ShortText]
    event_duration: Optional[ShortText]
    clinical_consequences: list[Consequence]
    outcome: Optional[Literal['resolved', 'ongoing', 'unknown']]


class NoSideEffectD5(Wire):
    quotes: Quotes
    subject: Subject
    subject_label: Optional[Label]
    scope_text: Optional[ShortText]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]


class DrugD5(Wire):
    surface: Surface
    ref: Literal['name', 'class', 'unspecified']
    purposes: list[PurposeD5]
    side_effects: list[SideEffectD5]
    no_side_effects: list[NoSideEffectD5]


class ExtractionD5(Wire):
    drugs: list[DrugD5]


def _d5(text):
    quotes = text[text.index("- quotes: "):]
    quotes = quotes[:quotes.index("\n") + 1]
    text = _cut(text, quotes)
    return _rep(text, "Item fields (independent of each other; never fill one from another):\n",
                "Item fields (independent of each other; never fill one from another):\n" + quotes)


PROMPT_D5 = _d5(PROMPT_D3)
COMPACT_D5 = register(Spec('compact-d5', PROMPT_D5, ExtractionD5, 'compact-wire-d5-1.0.0', 'compact-prompt-d5-1.0.0',
                           schema_name='weightloss_compact_extraction_d5', to_canonical=d1_to_canonical, text_verbosity='low',
                           canonical=('compact-canonical-sev-1.0.0', WireExtractionSev)))


# --- compact-l1: p7 with the drug-centric structure only, for reasoning low (2026-09-27, user decision: p7 + low) -------
# The user moved the line to GPT-6 Luna reasoning low with compact-p7. At low the matched dev-128 baseline (p7-low a/b,
# brand-v1) is far above every none variant; its largest residual is co-medication purposes (fresh-64: 3 of 4 missed
# purposes are other medicines). l1 makes one change to p7: each medicine the text mentions for someone's use is one entry
# holding its own purposes, side_effects and no_side_effects. p7's item fields, canonical attribution values, canonical
# wire and rules are otherwise unchanged (no none-era rename, rules or severity canonical).

class PurposeL1(Wire):
    indication: Surface
    subject: Subject
    subject_label: Optional[Label]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    statement_kind: StatementKind
    polarity: Optional[Literal['affirmed', 'negated', 'uncertain']]
    quotes: Quotes


class SideEffectL1(Wire):
    effect: Surface
    subject: Subject
    subject_label: Optional[Label]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    statement_kind: StatementKind
    occurrence: Literal['occurred', 'not_occurred', 'uncertain', 'not_asserted']
    attribution: Literal['attributed', 'temporal_only', 'uncertain', 'other_cause', 'not_stated']
    severity: Optional[Literal['mild', 'moderate', 'severe']]
    event_dose: Optional[DoseText]
    onset: Optional[ShortText]
    event_duration: Optional[ShortText]
    clinical_consequences: list[Consequence]
    outcome: Optional[Literal['resolved', 'ongoing', 'unknown']]
    quotes: Quotes


class DrugL1(Wire):
    surface: Surface
    ref: Literal['name', 'class', 'unspecified']
    purposes: list[PurposeL1]
    side_effects: list[SideEffectL1]
    no_side_effects: list[NoSideEffectD1]


class ExtractionL1(Wire):
    drugs: list[DrugL1]


def l1_to_canonical(answer):
    handles, out = {}, {'drugs': [], 'purposes': [], 'side_effects': [], 'no_side_effects': []}
    for d in answer.drugs:
        if not (d.purposes or d.side_effects or d.no_side_effects):
            continue
        key = (d.surface, d.ref)
        if key not in handles:
            handles[key] = f'd{len(handles) + 1}'
            out['drugs'].append({'handle': handles[key], 'surface': d.surface, 'ref': d.ref})
        h = handles[key]
        for group in ('purposes', 'side_effects', 'no_side_effects'):
            out[group] += [{'drug': h, **x.model_dump()} for x in getattr(d, group)]
    return out


def _l1(text):
    text = _rep(text, "Drugs (every key present; null when the text does not say; [] when a group is empty):\n- drugs: one entry per medication that an item below refers to. handle: d1, d2, ... in order of first mention. surface:",
                "Drugs (every key present; null when the text does not say; [] when a list is empty):\n- drugs: one entry for each medicine the text mentions for someone's use (the weight or diabetes drug and any other or relief medicine), each holding its own purposes, side_effects and no_side_effects (any of them may be []); put every statement under the drug it is about. surface:")
    text = _rep(text, "to the named drug they clearly refer to and use that drug's handle.", "to the named drug they clearly refer to and put the statement under that drug.")
    text = _rep(text, "Groups:\n", "Each drug's lists:\n")
    text = _cut(text, "- drug: the handle of the drug the statement is about.\n")
    text = _rep(text, "If the input has no purpose, side effect or no-side-effect statement, return empty groups.",
                "If the input mentions no medicine, drugs is [].")
    return text


PROMPT_L1 = _l1(PROMPT_V7)
COMPACT_L1 = register(Spec('compact-l1', PROMPT_L1, ExtractionL1, 'compact-wire-l1-1.0.0', 'compact-prompt-l1-1.0.0',
                           schema_name='weightloss_compact_extraction_l1', to_canonical=l1_to_canonical, text_verbosity='low'))
