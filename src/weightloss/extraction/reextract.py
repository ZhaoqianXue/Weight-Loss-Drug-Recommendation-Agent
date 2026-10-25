"""Re-extraction (schema v9, 2026-10-16): sentence numbering, model input, wire schema and local checks.

One call per record. Code numbers the sentences of the source text; the model cites evidence by sentence number and
writes purposes and symptoms as worded in the text. Brand and ingredient are closed lists filled only when the text
names the drug. Reddit and WebMD share the item schema; only Reddit returns the writer's age group and sex (WebMD takes
them from the review profile). Field meanings are documented in docs/project-document.md, Section 3.3.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

AGE_GROUPS = ('13-18', '19-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75 or over')
BRANDS = ('Ozempic', 'Wegovy', 'Rybelsus', 'Mounjaro', 'Zepbound', 'Victoza', 'Saxenda', 'Trulicity')
INGREDIENTS = ('semaglutide', 'tirzepatide', 'liraglutide', 'dulaglutide')
BRAND_INGREDIENT = {'Ozempic': 'semaglutide', 'Wegovy': 'semaglutide', 'Rybelsus': 'semaglutide',
                    'Mounjaro': 'tirzepatide', 'Zepbound': 'tirzepatide', 'Victoza': 'liraglutide',
                    'Saxenda': 'liraglutide', 'Trulicity': 'dulaglutide'}
IMPACTS = ('daily_life', 'medical_visit', 'emergency_visit', 'hospitalization')
ACTIONS = ('dose_reduced', 'escalation_paused', 'dose_skipped', 'stopped')

# A sentence ends at . ! ? (optionally followed by closing quotes or brackets) before whitespace, or at a line break.
_BOUNDARY = re.compile(r'[.!?]+["\'’”)\]]*(?=\s)')
_ABBREV = re.compile(r'(?:^|\W)(?:dr|mr|mrs|ms|vs|etc|e\.g|i\.e|approx|appt|wk|wks|hr|hrs|lbs|lb|oz|st|jr|sr|no)\.$', re.I)
_WORDY = re.compile(r'[^\W_]')


def split_sentences(text, field='text'):
    """Numbered sentences of one text field: [{'id', 'field', 'start', 'end', 'text'}], offsets into `text`."""
    out = []
    for line in re.finditer(r'[^\n]+', text):
        start = line.start()
        cuts = [m.end() + line.start() for m in _BOUNDARY.finditer(line.group())
                if not _ABBREV.search(line.group()[:m.end()])]
        for cut in cuts + [line.end()]:
            segment = text[start:cut]
            lead = len(segment) - len(segment.lstrip())
            body = segment.strip()
            if body and _WORDY.search(body):
                s = start + lead
                out.append({'field': field, 'start': s, 'end': s + len(body), 'text': body})
            start = cut
    for i, sentence in enumerate(out, 1):
        sentence['id'] = i
    return out


T2D_COMMUNITIES = frozenset({'diabetes_t2', 'type2diabetes'})


def render(platform, kind, sentences, reviewed=None, community=None):
    """Model input: a one-line header and the numbered sentences. `reviewed` = (brand, ingredient) for WebMD;
    `community` = the Reddit community, shown only as its type (GLP-1 or type 2 diabetes), never by name."""
    if platform == 'webmd':
        brand, ingredient = reviewed
        head = f'WebMD review. Reviewed drug (from the review profile): {brand} ({ingredient})'
    elif community is None:
        head = f'Reddit {kind}'
    else:
        kind_of = 'a type 2 diabetes community' if community in T2D_COMMUNITIES else 'a GLP-1 medication community'
        head = f'Reddit {kind} in {kind_of}'
    return head + '\n' + '\n'.join(f"[{s['id']}] {s['text']}" for s in sentences)


class Wire(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


Evidence = list[int]


class Writer(Wire):
    age_group: Optional[Literal[AGE_GROUPS]]
    sex: Optional[Literal['female', 'male', 'other']]
    evidence: Evidence


class Purpose(Wire):
    purpose: str = Field(min_length=1, max_length=200)
    evidence: Evidence = Field(min_length=1)


class SideEffect(Wire):
    effect: str = Field(min_length=1, max_length=200)
    dose: Optional[str]
    duration: Optional[str]
    intensity: Optional[Literal['mild', 'moderate', 'severe']]
    impact: Optional[Literal[IMPACTS]]
    action_taken: list[Literal[ACTIONS]]
    evidence: Evidence = Field(min_length=1)


class Drug(Wire):
    brand: Optional[Literal[BRANDS]]
    ingredient: Optional[Literal[INGREDIENTS]]
    purposes: list[Purpose]
    side_effects: list[SideEffect]

    @model_validator(mode='after')
    def _levels_agree(self):
        if self.brand is not None and self.ingredient != BRAND_INGREDIENT[self.brand]:
            raise ValueError(f'ingredient {self.ingredient} does not match brand {self.brand}')
        return self


class WebmdOutput(Wire):
    drugs: list[Drug]


class RedditOutput(Wire):
    writer: Writer
    drugs: list[Drug]


def output_model(platform):
    return RedditOutput if platform == 'reddit' else WebmdOutput


def _norm(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def _words(text):
    """Lower-cased words; apostrophes (straight or curly) are removed so "PVC's" and "PVCs" give the same word."""
    return re.findall(r"[^\W_]+", re.sub(r"['\u2019\u2018]", '', text.casefold()))


def _same_word(w, p):
    """Exact, a shared stem (common prefix of at least 4 letters covering 60% of the shorter word: nauseous/nausea,
    vomited/vomiting, dizzy/dizziness), or a slight misspelling of a long word (dizzyness/dizziness)."""
    import difflib
    if w == p:
        return True
    k = 0
    while k < min(len(w), len(p)) and w[k] == p[k]:
        k += 1
    if k >= 4 and k >= 0.6 * min(len(w), len(p)):
        return True
    return len(w) >= 5 and difflib.SequenceMatcher(None, w, p).ratio() >= 0.8


def _worded_in(phrase, sentence_texts):
    """True when every word of `phrase` occurs in the cited sentences (words may come from a split phrase, e.g. "dry
    throat" from "dry and sore throat"); a word may differ in its ending or slightly in spelling (see _same_word)."""
    words, pool = _words(phrase), {w for t in sentence_texts for w in _words(t)}
    return bool(words) and all(any(_same_word(w, p) for p in pool) for w in words)


def drop_items(out, problems):
    """Copy of `out` without the items named in `problems` (paths like drugs[0].side_effects[2]) and without drug
    groups left empty; returns (new_output_dict, dropped_paths)."""
    bad = {p.split(':', 1)[0] for p in problems if p.startswith('drugs[')}
    data = out.model_dump()
    for d, drug in enumerate(data['drugs']):
        for group in ('purposes', 'side_effects'):
            drug[group] = [it for i, it in enumerate(drug[group]) if f'drugs[{d}].{group}[{i}]' not in bad]
    removed = [{'path': f'drugs[{d}].{group}[{i}]', **it} for d, drug in enumerate(out.model_dump()['drugs'])
               for group in ('purposes', 'side_effects') for i, it in enumerate(drug[group]) if f'drugs[{d}].{group}[{i}]' in bad]
    data['drugs'] = [d for d in data['drugs'] if d['purposes'] or d['side_effects']]
    data['_removed'] = removed
    if 'writer' in data and any(p.startswith('writer') for p in problems):
        data['writer'] = {'age_group': None, 'sex': None, 'evidence': []}
    return data, sorted(bad | {p.split(':', 1)[0] for p in problems if p.startswith('writer')})


def check_output(out, sentences):
    """Local checks after schema validation. Returns a list of problems (empty when the answer is acceptable):
    every cited sentence exists, and every word of each purpose and symptom occurs in one of its cited sentences."""
    by_id = {s['id']: _norm(s['text']) for s in sentences}
    problems = []

    def cited(path, ids):
        missing = [i for i in ids if i not in by_id]
        if missing:
            problems.append(f'{path}: unknown sentence {missing}')
        return [by_id[i] for i in ids if i in by_id]

    if isinstance(out, RedditOutput):
        cited('writer.evidence', out.writer.evidence)
        if (out.writer.age_group or out.writer.sex) and not out.writer.evidence:
            problems.append('writer: age or sex without evidence')
    for d, drug in enumerate(out.drugs):
        for k, item in enumerate(drug.purposes):
            texts = cited(f'drugs[{d}].purposes[{k}]', item.evidence)
            if texts and not _worded_in(item.purpose, texts):
                problems.append(f'drugs[{d}].purposes[{k}]: purpose not worded as in its sentences')
        for k, item in enumerate(drug.side_effects):
            texts = cited(f'drugs[{d}].side_effects[{k}]', item.evidence)
            if texts and not _worded_in(item.effect, texts):
                problems.append(f'drugs[{d}].side_effects[{k}]: effect not worded as in its sentences')
    return problems


PROMPT_VERSION = 'v9-prompt-7'

_CORE = 'The input is numbered sentences. Use only the text: no outside knowledge, no guesses. When a case is unclear, leave it out.\n\nScope:\n- Only the writer\'s own experience. Leave out what happens to other people (a spouse, parent, friend), even when compared with the writer\'s, people in general, advice and quotes. A drug named only for someone else (the person the writer replies to) does not name the writer\'s drug.\n- Only these drugs: semaglutide (Ozempic, Wegovy, Rybelsus), tirzepatide (Mounjaro, Zepbound), liraglutide (Victoza, Saxenda), dulaglutide (Trulicity), including generic names, compounded versions, nicknames and misspellings ("Zep", "sema", "tirz"), and references that name no drug but clearly mean the GLP-1 the writer takes ("my shot", "the 2.5", "this med"). In a GLP-1 medication community, a medication the writer takes that the text does not name (a dose such as "first dose" or "4 doses of .25", "injection day", "the side effects") is their GLP-1 unless the text points to another drug. Leave out every other medicine, including medicines taken to relieve a side effect.\n\ndrugs: one entry per study drug the writer uses or used that has at least one purpose or side effect.\n- brand: the brand, when the text names it for this drug (nicknames and misspellings count; a later "it" that clearly refers to a named brand counts); otherwise null.\n- ingredient: the ingredient, when the text names the drug by brand or generic name (a brand implies its ingredient); otherwise null. Compounded semaglutide: brand null, ingredient semaglutide. A reference that names no drug: both null.\n\npurposes: reasons the writer gives for using, starting, continuing or planning to use the drug: a condition or a goal, stated with an explicit reason ("for", "to" + verb, "because", "due to", "since", "so", "hence", "as", "that\'s why", "prescribed for", "my doctor suggested it when/because"; e.g. "I have insulin resistance, so I should be on this", "I am diabetic hence the shots"). Examples: "for my type 2 diabetes" -> "type 2 diabetes"; "started Zepbound to lose weight" -> "lose weight"; "when my A1C showed prediabetes, my doctor suggested Zepbound" -> "prediabetes"; "I use the 5 mg for my maintenance" -> "maintenance". purpose: the shortest words that name it, in the text\'s own words; one condition or goal per item. Not purposes: outcomes or effects ("working great for weight loss", "helped my PCOS", "worked well enough for my diabetes", "not sure it\'s helping my A1C", "reached my goal"); goals or conditions without a reason tying them to the drug ("I have 60 pounds to lose", "just wanting to lose 20lbs", "53 yo T2", "GW 70kg"); a maintenance phase or dose ("in maintenance", "my maintenance dose", "back down to 2.5mg maintenance", "maintaining on the shot", "I chose maintenance"); hopes, expectations or praise ("I thought it would be a cure-all for my weight loss", "this is going to help my binge eating", "a miracle drug for diabetics"); a prescription with no reason; negated reasons.\n\nside_effects: symptoms or unwanted bodily changes the writer says they had, even rarely or occasionally, and links to the drug, by cause ("Ozempic gave me nausea", "side effects were fatigue") or by timing ("since starting Zep I get headaches", "the day after my shot"). effect: the shortest words naming the symptom, in the text\'s own words; one symptom per item, and one item per symptom per drug with all its evidence. When one phrase names several symptoms, take each one\'s words from it ("dry and sore throat" -> "dry throat", "sore throat"; "red and itchy at the injection site" -> "red injection site", "itchy injection site"). Words such as "sick" or "GI issues" name a symptom; "side effects" or "issues" alone do not. A symptom the writer attributes to an effect of the drug counts ("hair shedding from losing weight so fast"). Leave out effects only feared or asked about (including "is this a known thing with Ozempic?" for symptoms the writer has), effects in hypothetical or conditional statements ("not high enough to deal with nausea"), symptoms while off the drug or after stopping it, effects the writer is unsure come from the drug (a question mark, "not sure", "don\'t know if", "could be", "maybe", "(psychological?)") or blames on something else, symptoms not linked to the drug, negated symptoms, statements of no side effects, "side effects" with no symptom named, emotions about the treatment, dosing mistakes, and the drug not working. Appetite loss, not feeling hungry, eating less, fullness, less food noise and intended weight loss are expected effects: leave them out unless the writer explicitly presents them as a problem for themselves (has to force themselves to eat, cannot eat enough or struggles to, worries about not eating enough or nutrition, calls it a side effect or a problem, says it makes them feel unwell); food aversion or food tasting bad counts ("cheese puts me off"). Hunger or cravings, new or returning, are not side effects. If the writer cannot eat because of another symptom ("so nauseous I could barely eat"), that is the impact of that symptom, not a separate item.\nFor each side effect, null or [] unless the text states it for this symptom:\n- dose: the dose the writer was on when it occurred, as written.\n- duration: how long the symptom lasted, as written; not the time on the drug and not when it started.\n- intensity: the writer\'s own rating of this symptom: "mild" (mild, slight), "moderate", "severe" (severe, extreme, intense, unbearable, excruciating), adverb forms included (mildly, slightly, severely, extremely); the highest if several. A rating of another symptom in the same sentence does not carry over. Words such as a little, minor, very, so, bad, awful, terrible, rough or worst do not count.\n- impact: the most serious consequence stated: "daily_life" (could not work, stayed in bed, could not eat or drink, could not function, debilitating), "medical_visit" (saw, called or messaged a doctor or nurse; urgent care; an infusion clinic), "emergency_visit" (ER, or hospital treatment without admission), "hospitalization" (admitted, hospitalized, in hospital, ICU). An impact stated for a list of symptoms applies to each.\n- action_taken: what the writer did with the drug because of this symptom: "dose_reduced", "escalation_paused" (stayed on the current dose instead of increasing), "dose_skipped" (skipped or delayed a dose), "stopped" (including switching to another drug).\n\nevidence: the numbers of the sentences that state the item and its details.\n\nWording: copy purpose and effect words exactly as they appear in the cited sentences, keeping the writer\'s form and spelling ("vomited", "nauseous", "dizzyness", "T2D", "sight of injection"). Do not replace them with medical or standard terms ("fatigue" for "dragging", "constipation" for "not as regular", "diarrhea" for "the runs"); if the writer describes a symptom without naming it, use the writer\'s own words ("really dragging", "not as regular").'

_WRITER = '\n\nwriter: the writer\'s own age and sex, only when stated explicitly about themselves ("I\'m 52", "52F", "(45M)", "as a woman").\n- age_group: the group that contains the stated age: 13-18, 19-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75 or over; null when the stated age spans two groups ("in my 50s") or is under 13.\n- sex: "female", "male" or "other"; never inferred from "my husband", pregnancy or menopause.\n- evidence: the sentences that state them; [] when both are null.'

PROMPTS = {
    'reddit': "You extract one Reddit writer's own experience with GLP-1 drugs for a knowledge graph. " + _CORE + _WRITER
              + "\n\nReturn the JSON only.",
    'webmd': "You extract one WebMD reviewer's own experience with GLP-1 drugs for a knowledge graph. The review is about the "
             "reviewed drug named in the header unless another drug is named; references that name no drug mean the reviewed drug. "
             + _CORE + "\n\nReturn the JSON only.",
}
