"""Evidence-grounded question answering over the knowledge graph (project document Section 3.7).

One Responses API conversation per question: strict function tools over the graph tools (tools.py: link_terms,
query_graph, get_evidence, compare_faers on the local Neo4j graph), reasoning items passed back encrypted (store=False),
and a strict JSON final answer whose numbers cite tool-output rows and whose quotes cite returned source sentences.
A deterministic verifier checks every citation and quote against the tool outputs; on failure the model gets one repair
round with the verifier messages, and the result is marked verified true or false. Every run appends a full trace
(request bodies, tool inputs and outputs, raw responses, usage, cost; never the API key) to a JSONL file.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from weightloss.extraction.structured import (ExtractionAnomaly, compact_json, final_output_text, mask, parse_json_object,
                                              sha256_text, strict_json_schema, token_split, validate_model)
from weightloss.extraction.structured import cost_usd as _shared_cost_usd
from weightloss.graph import neo4j_db

from . import VERSION
from .tools import TOOL_ARGS, KnowledgeGraph, QueryError

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / '.cache/agent-runs'
TRACE = RUN_DIR / 'traces.jsonl'
MODEL, EFFORT = 'gpt-6-luna', 'medium'
MAX_TOOL_ROUNDS = 12          # responses that call tools before the answer is forced
REPAIR_TOOL_ROUNDS = 2        # extra tool rounds allowed after each failed verification
MAX_REPAIRS = 2               # repair rounds after a failed verification
MAX_OUTPUT_TOKENS = 16000
BUDGET_USD = 0.15             # per question; the loop stops before a request once this is spent
# Standard USD per 1M tokens for models the shared price table (structured.py) does not hold: gpt-6-sol, official prices
# as recorded in weightloss.extraction.role_single_sol (confirmed 2026-10-01).
EXTRA_PRICES_PER_1M = {'gpt-6-sol': {'input': 2.0, 'cached': 0.2, 'cache_write': 2.5, 'output': 10.0}}


def cost_usd(model, usage):
    prices = EXTRA_PRICES_PER_1M.get(model)
    if prices is None:
        return _shared_cost_usd(model, usage)
    t = token_split(usage)
    return round((t['plain_input_tokens'] * prices['input'] + t['cached_tokens'] * prices['cached']
                  + t['cache_write_tokens'] * prices['cache_write'] + t['output_tokens'] * prices['output']) / 1e6, 9)
TRACE_LOCK = threading.Lock()  # concurrent runs append whole trace lines

ANALYSIS_TABLES = (
    "statement(role 'side_effect'|'purpose', edge_id, platform 'reddit'|'webmd', record_id, record_key, date 'YYYY-MM-DD', year, "
    "drug_id, ingredient, brand ('unspecified' = no brand named), ingredient_basis 'name'|'profile'|'community', "
    "writer_age_group, writer_sex 'female'|'male'|'other' (NULL = not stated), source 'profile'|'text' (purposes), wording, "
    'pt_code, pt (NULL = uncoded), hlt_code, hlt, hlgt_code, hlgt, soc_code, soc, nonclinical_group (NULL = clinical event), '
    'intensity, impact, action_taken (JSON list), stopped, dose_reduced, dose_skipped, escalation_paused (0/1), dose_text, '
    'dose_mg, duration_text, duration_bucket); faers_report(report_id, receivedate, year, sex, age_group, hospitalization 0/1, '
    "in_primary_analysis 0/1, has_clinical_pt 0/1); faers_stratum(report_id, level 'brand'|'ingredient'|'unspecified', "
    'ingredient, brand, in_primary 0/1, actions, action_known 0/1, withdrawn 0/1, dose_reduced 0/1, window_start); '
    'faers_reaction(report_id, pt_code, pt, hlt_code, hlt, hlgt_code, hlgt, soc_code, soc, nonclinical_group, clinical 0/1); '
    'meddra(pt_code, pt, hlt_code, hlt, hlgt_code, hlgt, soc_code, soc, nonclinical_group); approval(brand, ingredient, '
    'approval_date).')

TOOL_DESCRIPTIONS = {
    'link_terms': 'Entity linking. Find the graph nodes for a condition, goal or symptom: MedDRA PT nodes whose name or whose '
                  'statement wording matches the words, with record counts per platform, example wordings and FAERS report '
                  'counts, plus the MedDRA groups (HLT, HLGT, SOC) that match or sit above the best PTs. Call it before using '
                  'any code.',
    'query_graph': 'Graph pattern query over the patient statements. Match statements of the target role by drug, platform, '
                   'writer age group and sex, statements of the same drug in the same record (require_purposes / '
                   'require_side_effects) and target concepts (a PT or a whole MedDRA group), then count records: a concept '
                   'counts once per record; the denominator of a row is the number of records with at least one target '
                   'statement in the filtered universe and the same population group. Platforms are always split. Summary rows '
                   'give the universe per platform and how many records an age or sex filter excluded because the value was missing.',
    'get_evidence': 'Example statements for the same pattern: statement id, record, drug, PT and its SOC, wording, writer age and '
                    'sex, side-effect details and the cited source sentences. Fixed pseudo-random order, one statement per record. '
                    'wording filters by the extracted wording; sentence by a word or phrase in the cited source sentences. The '
                    'summary counts every matching statement and record per platform, so it also counts statements by what '
                    'their sentences say.',
    'compare_faers': 'The same side-effect concepts and drug scopes in the validated FAERS reports, side by side with the '
                     'Reddit and WebMD records under the RQ2 comparison rules (window from FDA approval to 2026-06-30, '
                     'clinical-event PTs only). Hospitalization and the action taken with the drug (withdrawn = the drug was stopped; '
                     'dose reduced) are outcomes: set include_outcomes=true for them and never look them up as PTs (Drug withdrawal '
                     'syndrome is a different, clinical event). Without targets it ranks PTs. With targets it also reviews the '
                     'concepts on both sides: PTs of the patient records and of the FAERS reports that no member counts.',
    'run_python': 'Python in a notebook-like sandbox (variables persist between calls; no network; at most 120 s and about '
                  '12,000 characters of output per call). Defined: db, a read-only SQLite connection to the analysis tables, and '
                  'pd (pandas). Tables: ' + ANALYSIS_TABLES + ' Views that apply the FAERS comparison rules: faers_scope(scope, '
                  "report_id, has_clinical_pt) with scope 'all', 'brand:<Brand>', 'ingredient:<ingredient>' or "
                  "'unspecified_brand:<ingredient>' (primary configuration), and comparable_statement(scope, every statement "
                  'column): side-effect statements of that scope dated from its approval date to 2026-06-30 with clinical-event '
                  'PTs or none (uncoded). SQL functions: wilson_low(k, n) and wilson_high(k, n) give the 95% '
                  "interval in percent; x REGEXP 'pattern' matches case-insensitively. residual(where, counted, "
                  "table='statement') reviews a concept count: where is the SQL condition (on the table's columns) that "
                  'selects the denominator\'s statements, counted the condition that selects every counted member (all '
                  'concepts of the question joined with OR); it prints the PTs and uncoded wordings of the denominator that no '
                  'member counts, marks with * those related in wording to the counted members that add enough records, and '
                  'lists the uncoded wordings the patterns counted (! = matched only inside a longer word). There are no '
                  'source sentences (use get_evidence for quotes). In statement, an uncoded mention whose wording misspells a '
                  'PT name ("nausa") carries that PT, as in query_graph; comparable_statement keeps coded PTs only. Print every '
                  'number you will cite exactly as you will write it (counts, denominators, percentages with one decimal, '
                  'interval bounds with the two decimals wilson_low and wilson_high return): the answer may cite only numbers '
                  'that appear in an output.',
}
TOOLS = [{'type': 'function', 'name': name, 'description': TOOL_DESCRIPTIONS[name], 'strict': True,
          'parameters': strict_json_schema(model)} for name, model in TOOL_ARGS.items()] + [
    {'type': 'function', 'name': 'run_python', 'description': TOOL_DESCRIPTIONS['run_python'], 'strict': True,
     'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['code'],
                    'properties': {'code': {'type': 'string', 'description': 'Python code to run'}}}}]


# ---------------------------------------------------------------------------------------------------- final answer
class Citation(BaseModel):
    id: str = Field(description='Tag id used in the answer, c1, c2, ...')
    call: str = Field(description='Label of the tool output, e.g. t3 (the "call" field of the output)')
    row: int = Field(description='The "row" number inside that output')
    field: str = Field(description='Field name in that row, e.g. records, denominator, percent, faers_reports')
    value: str = Field(description='The field value exactly as in the row, written as text; for a run_python output, the one '
                                   'number as printed (never a whole line or several numbers)')


class Quote(BaseModel):
    edge_id: str = Field(description='edge_id of the statement, from a get_evidence output')
    text: str = Field(description='The quoted words, copied exactly from that statement\'s evidence sentences')


class Comparison(BaseModel):
    text: str = Field(description='The words of the answer that state the comparison, copied exactly without tags and '
                                  'without the numbers, e.g. "the Reddit share is higher than the FAERS share"')
    first: str = Field(description='Citation id of the percentage of the share the text names first, e.g. c3')
    second: str = Field(description='Citation id of the percentage it is compared with, e.g. c7')
    claim: Literal['higher', 'lower', 'similar'] = Field(description='What the text says of the first share against the second')


class FinalAnswer(BaseModel):
    answer: str = Field(description='Markdown answer; every number from a tool output is followed by its tag, e.g. 8 [c1]')
    answer_type: Literal['answered', 'limited', 'declined'] = Field(
        description='answered: the data answer the question; limited: only part of it can be answered from the data, or the '
                    'data are insufficient; declined: medical advice or another question the data cannot answer')
    citations: list[Citation]
    quotes: list[Quote] = Field(description='Every passage quoted in the answer, with its statement; [] if none')
    comparisons: list[Comparison] = Field(description='Every statement in the answer that one share is higher, lower or '
                                                      'similar to another; [] if none')
    limitations: list[str] = Field(description='Short statements of data limits relevant to this answer')


FINAL_SCHEMA = strict_json_schema(FinalAnswer)

INSTRUCTIONS = """You answer questions about patient-reported purposes and side effects of GLP-1 receptor agonists (ingredients semaglutide, tirzepatide, liraglutide, dulaglutide; brands Ozempic, Wegovy, Rybelsus, Mounjaro, Zepbound, Victoza, Saxenda, Trulicity). The data are a knowledge graph built from Reddit posts and comments and WebMD reviews, linked to the MedDRA 29.1 hierarchy and to validated FAERS reports, and you reach it only through the tools. Use only tool outputs for any statement about the data; never fill gaps with outside knowledge.

The graph: Drug nodes (ingredient and brand), Purpose and SideEffect nodes at the MedDRA Preferred Term (PT) level, each PT on its MedDRA primary path PT -> HLT -> HLGT -> SOC, and one statement edge per thing a writer said: Drug -USED_FOR-> Purpose and Drug -SIDE_EFFECT-> SideEffect, carrying the record, the writer's age group and sex, the wording, the cited sentences and, for side effects, dose, duration, intensity, impact and action taken. FAERS reports link to the same Drug and SideEffect nodes.

Data facts (state the ones that matter for the question):
- Reddit and WebMD are separate sources with different populations; report them separately and never add them together.
- Counts are records (one Reddit post or comment, or one WebMD review), not patients; writers are not verified patients. A concept counts once per record. Percentages are shares of records that report something; they are not incidence, prevalence or risk, and no causal claim can be made.
- Side effects are symptoms the writer reports having and links to the drug; purposes are the writer's stated reasons for using it. Most WebMD purposes are the condition the reviewer chose from the site's list in the profile, while Reddit purposes all come from free text, so purpose shares measure different things on the two platforms. A condition such as "people who take it for X" means a purpose statement for the same drug in the same record.
- A purpose PT names either the goal (Weight decreased, Weight control, Blood glucose decreased) or the condition the drug is taken for (Obesity, Overweight, Weight, Weight increased, Type 2 diabetes mellitus, Glycosylated haemoglobin increased); a goal and the condition it addresses are one reason and belong to one concept.
- Writer age group and sex/gender: on Reddit they are known only when writers state them about themselves, so those records are a self-disclosed subset; on WebMD they come from the reviewer profile. Missing values are "not stated". An age or sex filter drops records with missing values.
- Ingredient-level results include references that name no drug but come from a single-ingredient Reddit community (ingredient_basis "community"); named_drugs_only=true restricts to named drugs. Keep named_drugs_only false unless the question asks to count only posts that name the drug, or you run it as a separately labelled robustness check; whenever you set it to true, say in the answer that community-assigned records were left out. A brand filter already counts named brands only. Brand-level results include only brands named in the text or the WebMD profile; brand "unspecified" is never a brand.
- The "[uncoded ...]" node collects mentions without an adequate MedDRA term; it is not a PT and must not be ranked as one. When uncoded_wordings (query_graph) or the uncoded row of link_terms show wordings that plainly name the asked concept (for example "t2d" or "t2" for type 2 diabetes, "bed" for binge eating disorder, "pcos symptoms", "food noise"), add each such wording to the concept as a member with level "wording" and the word as code (it counts only uncoded mentions containing those words) and say so in the answer. Universe-level uncoded counts cover every mention in the universe, not only the asked concept: when they are a tenth or more of universe_records, say that PT-based counts can miss reasons or effects, without attributing them to the concept.
- FAERS numbers are spontaneous reports (validated, primary configuration) from patients, clinicians and manufacturers. Of the FAERS seriousness outcomes, this extract holds only hospitalization (with the action taken with the drug); death and the other outcomes exist in FAERS but are not part of these data, so say that the data here do not include them, never that FAERS lacks them. Describe FAERS, Reddit and WebMD shares side by side as different reporting channels; never compute ratios or differences between them and never call any of them a rate.
- Rows flagged sparse (denominator below 10) and empty groups support no conclusion: say the data are insufficient. Rows flagged small (denominator below 30) must be called small in words.
- Comparisons: call one share higher, lower or different from another (between groups, drugs, platforms or FAERS) only when their 95% intervals do not overlap; when they overlap, say the shares are similar or that the data show no clear difference. The intervals come from the tool labels (clear_differences of query_graph; faers_vs_reddit, faers_vs_webmd and reddit_vs_webmd of compare_faers) or from wilson_low and wilson_high computed in code.

Medical questions: you inform about what writers report; you do not give medical advice. If the question asks what someone should do (take, stop, switch, dose or combine a drug, how to treat a symptom, whether something is safe for them) or asks about risk, incidence or causation, do not advise or estimate. Set answer_type "declined" when nothing in the question can be answered from the data, and say plainly that you cannot give medical advice and that a doctor or pharmacist can help; for severe symptoms (for example severe abdominal pain, fainting, signs of an allergic reaction) say to seek medical care promptly. Set answer_type "limited" when part of the question can be answered: answer that part from the data and state what cannot be answered. Never recommend a drug, dose or action. When the question rests on a premise the data contradict (a count, a direction, a fact about the drugs), say so with the numbers.

How to work:
1. Concepts. For every specific condition, goal or symptom in the question, call link_terms (kind purpose or side_effect) to see how writers' wordings were coded: the PT rows with example wordings and matching-wording counts, related PTs of the same MedDRA HLT worded differently, the MedDRA groups, and the uncoded row. Call it also with common lay variants or synonyms (gallbladder and gallstone; vision, eyesight and blurry; period and menstrual), and for a subtype also with the general term ("type 2 diabetes" and "diabetes"). Do not call it for generic words ("side effects", "complaints", "problems") or drug names. Then define each concept explicitly: the PT codes (or the MedDRA group that matches a broad category as a whole, plus any PT outside it whose wordings clearly belong) and, when uncoded wordings plainly name the concept ("t2d", "bed", "food noise", "hypos", "lows"), those uncoded statements by REGEXP on wording with pt_code IS NULL. Before counting, search the wording of all statements of the role with REGEXP for the concept's words and their variants, and look at which PTs those wordings were coded to and which were left uncoded, so that no large group of matching statements is missed; then count the union of the members (a record once). Writers' wordings for one meaning are often split over several PTs (weight loss: Weight decreased, Weight control, Obesity, Overweight, Weight loss poor, Weight; diabetes: Diabetes mellitus, Type 2 diabetes mellitus, Diabetes mellitus management, glucose and HbA1c terms); include them all. Check every member in code by listing its most frequent wordings, and leave out PTs whose wordings mean something else (Food aversion is not a taste change; Malaise is mostly "sick"). When the question names a subtype and the data hold an unspecified term for the same condition, report the unspecified term as its own concept. Write wording patterns with word starts (\\bgas, \\bfart) so that a short word does not match inside another word (gas in orgasmic). PT counts in query_graph and in the statement table already include uncoded mentions that misspell the PT name ("nausa", "diahrrea"); compare_faers counts coded PTs only. Then review what the concepts leave out: query_graph and compare_faers (patient and FAERS side) return not_counted for their concepts, and in run_python call residual(where, counted) once your counting code is final (and again after you change a concept). Count every listed row that expresses the asked meaning and count again; fix any pattern marked !. The review also lists the counted PTs with their most frequent wordings (counted_members in query_graph and compare_faers): drop any member whose wordings mean something else, such as Gastrointestinal pain ("gut cramps") for constipation or Throat irritation ("itchy throat") for reflux. The review also lists PTs close in meaning (BioLORD-2023 closeness): close is not the same (Nausea and Vomiting are close), so judge each from its wordings. Name every concept's members in the answer, and say which large candidate you left out and why: code checks every large matching row of link_terms and every row the review marks check (*). Name only those left-out rows and the ones a reader would expect; never list unrelated rows. link_terms counts cover all drugs, platforms and populations; they guide the choice of concepts and are never cited as results.
2. Counting. run_python is the main analysis tool: query the analysis tables with SQL from Python and print every number you will cite. Follow the counting rules: count DISTINCT record_key; a share's denominator is the records with at least one statement of the same role under the same filters (drug, platform, age, sex, detail filters); keep Reddit and WebMD apart; brand results use named brands only; "people who take it for X" means a purpose statement for the same drug in the same record (join on record_key and drug_id). When the question asks about reasons in general, rank purpose PTs, then merge those that express the same reason into named concepts and report the concepts. For "which side effects" in a domain, rank within the domain rather than checking only the examples named. For a detail (impact, intensity, action, dose, duration), say how many of the group's records have it, with the group's records as the denominator; a count for one detail value (for example intensity "not stated") is never the total for a PT. When age or sex is involved, report for each platform how many records state the value out of all records; for the distribution of an attribute (sex, age group), compare shares among records or reports where it is stated and give the missing count separately. For who the writers of a drug are (their sex or age make-up), count all records of the drug with any statement, purpose or side effect, not only side-effect records, unless the question names one role; say which records the shares cover. Doses (dose_mg) and durations (duration_bucket) are parsed from free text and present for part of the statements: say so when you use them; when the buckets are too coarse for the question (more than two weeks), read duration_text in code and say how you classified it. Compare years by shares, not counts, and treat the current year as partial.
3. Comparisons. Compute wilson_low and wilson_high for every share you compare and call one share higher or lower only when the intervals do not overlap; otherwise say the shares are similar. When the question asks how groups or drugs differ, test every PT or concept with enough records, not only the ones you expected, and report the clear differences found, largest first; say "no clear difference" only when none is found. With more than two groups, give a small table. Let the comparison word carry the result and give intervals for the one or two shares a conclusion rests on. Describe a trend across ordered groups (rising with age, a peak) or a ranking only where the intervals separate the groups that carry it; otherwise say the differences are within uncertainty. When comparing the make-up of two groups (age bands, sex), compare every band, not one, and say which bands differ.
4. FAERS. compare_faers applies the RQ2 rules and labels each comparison; use it when it fits (targets, rank_by, include_outcomes for hospitalization, withdrawal, dose reduction). Otherwise compute in code from the views that carry the same rules: FAERS reports of a scope from faers_scope (denominators: has_clinical_pt = 1; numerators: reactions with clinical = 1; FAERS outcomes from faers_report.hospitalization and faers_stratum withdrawn, dose_reduced and action_known for the scope's level), and Reddit and WebMD statements from comparable_statement for the same scope (never from statement directly, which has no comparison window); this holds for every Reddit or WebMD number you set next to a FAERS number, including counts of writers who were hospitalized or stopped. For distributions within FAERS (sex, age group), use the scope's reports with has_clinical_pt = 1 and compare among reports where the value is stated. FAERS hospitalization is a report-level seriousness flag; FAERS withdrawn and dose reduced are given as shares of all reports in scope and of reports with a known action; Reddit and WebMD count records in which a side effect led to hospitalization or to stopping; say how the measures differ. Never compute ratios or differences between FAERS and Reddit or WebMD shares. When a reaction common in FAERS is rare online, consider whether it is a lab value, a lack of effect or an effect writers welcome before calling it under-reported.
5. query_graph counts with the same rules built in and lists clear differences between population groups; use it when it fits and you want those labels.
6. Quotes and sentences. When the question asks what people say or for examples, call get_evidence (wording finds statements phrased a particular way) and quote from the returned sentences only; there are no sentences in the analysis tables. When the question turns on context that the extracted wording does not carry (a cause such as another medicine or a surgery, "no rash", "after stopping"), count with get_evidence sentence, whose summary gives the matching statements and records under the same filters, and try the common variants (antidepressant, SSRI, Lexapro). Each quote must show exactly what the question asks (a passage about knee surgery does not show a knee replacement): leave out near misses, and say so when fewer fitting passages exist than asked for.
7. Finish the analysis the question needs before answering: never report a count you know to be partial or a list you know to be incomplete; run the missing step instead. Count a residual row only when its own wordings mean the asked concept; a symptom that often comes with it (pain with constipation, choking with vomiting) stays out.
8. Stop calling tools once you have what you need, or when the data are clearly insufficient.

Every tool output has a "call" label (t0 is the graph overview given with the question) and numbered rows. Final answer (JSON):
- answer: concise markdown that answers the question directly, separated by platform, with missing or sparse population attributes called out. Answer every part of the question, and never withhold a number it asks for because a concept or count is imperfect: give the best count you can compute and state its limit. When it asks which drugs, brands or groups, report every one with a count above zero and name those with none, never a selection. A yes/no or headline sentence must hold on every platform; when the platforms differ, the first sentence says so. Give numbers, not just names: for every group and concept you report, state its records and denominator (and percent when useful), and say in plain words what each denominator counts (for example "Wegovy FAERS reports with at least one clinical reaction"), never "the comparison denominator". For long rankings, give the top few per group and say that more rows exist. Write PT and group names exactly as in the tool output; when a PT name may not tell a reader what writers said, add its common wording in parentheses, for example Malaise ("sick").
- Wording: name the unit (Reddit posts or comments, WebMD reviews) and phrase shares with their group, for example "Of 3,728 tirzepatide side-effect posts from writers who said they are women, 1,207 (32.4%) mention Nausea"; write "Reddit writers report" or "on Reddit", never "Reddit reports"; never call writers patients. Never write record or statement ids, user names or links in the answer. Quote a short clause (about 5 to 15 words) that shows the experience asked about, not a two-word fragment and not a whole post, never dates, ages, body measurements, places, family details or a writer's statement of their own gender identity.
- Every number taken from a tool output is written in digits, exactly as it appears in the row or code output (counts of 1,000 or more with thousands separators), and is followed immediately by a tag such as [c1]; for "8 of 30 records (26.7%)" write 8 [c1] of 30 [c2] records (26.7% [c3]), never 8 of 30 records (26.7%) [c1][c2][c3] or 8 records (26.7% [c3]) [c1]. The same holds in tables: each cell number carries its own tag. Tags are for numbers only. Do not compute new numbers (no sums across platforms or groups, no differences or ratios).
- citations: one entry per tag: id, call label, row number, field name, and the value exactly as in that row. For a run_python output, use row 1, field "output" and the single number as printed; every number you cite must have been printed (compute it in code, never in your head; run code again to print a number you need).
- quotes: every passage you put in quotation marks, with the edge_id it comes from and the exact words; [] if none. Never put a [c] tag after a quote or a name; quotes are checked through this list.
- comparisons: every statement in the answer that one share is higher, lower or similar to another (between groups, drugs, platforms or FAERS): the words of the statement copied exactly from the answer (without tags), the citation ids of the two percentages (first = the share the statement names first) and the claim. Code checks the claim against the cited numbers and their intervals; [] if the answer compares nothing.
- answer_type: "answered" whenever the data answer the question, even with caveats such as missing values, a self-disclosed subset or a few sparse groups; "limited" only when part of the question cannot be answered from the data at all (a risk, a cause, a cross-channel rate comparison, a field the graph lacks) or every relevant group is too sparse to say anything; "declined" for medical advice or when nothing can be answered.
- Length and caveats: keep the answer under about 220 words. Put the general caveats (records are not patients; shares are not incidence, risk or causation; sources are separate) in limitations, not in the answer; keep in the answer only the caveats specific to this question (sparse or small groups, missing values, a restriction you applied, concepts or rows you left out, uncoded wordings that matter, the WebMD profile list when comparing purposes)."""

REPAIR = ('Verification of your final answer failed:\n{problems}\n\nReturn a corrected final answer in the same JSON format. For a '
          'concept-review problem, count the row (query again, or change the condition and call residual() again) or name it '
          'in the answer as left out with the reason. Cite only '
          'values present in tool outputs (call label, row number, field, value exactly as in the row), write each cited number '
          'exactly as cited immediately before its own tag (one tag per number, never several tags after one number; write '
          '"5851 [c1] records (36.7% [c3])", not "5851 records (36.7% [c3]) [c1]"), give '
          'every tag a citation, and quote only exact words from evidence '
          'sentences returned by get_evidence, listing each quote with its edge_id. For a run_python output, cite each number '
          'separately (row 1, field "output", the single number as printed). You may call tools again if a needed number or '
          'sentence is missing, for example run_python to print a number you computed in your head.')


CRITIC_REPAIR = ('An independent check of your answer against the tool log raised these points:\n{issues}\n\nCheck each point; fix '
                 'what is right (call tools again if needed) and keep what is already correct. Then return the final answer in the '
                 'same JSON format, with every cited number from a tool output.')


# ---------------------------------------------------------------------------------------------------- session and tools
class Session:
    """Executes tool calls against the graph and keeps every output by its call label. Keys starting with '_' (full source
    sentences for quote checking) stay in the session and are never sent to the model."""

    def __init__(self, graph):
        self.graph = graph
        self.outputs = {}
        self.calls = []
        self.next_call = 1
        self.sandbox = None

    def close(self):
        if self.sandbox is not None:
            self.sandbox.close()
            self.sandbox = None

    def _python(self, arguments):
        from .code_baseline import Sandbox, agent_setup
        code = (json.loads(arguments) if isinstance(arguments, str) else arguments)['code']
        if self.sandbox is None:
            self.sandbox = Sandbox(agent_setup(), deny_read=[str(neo4j_db.KG)])
        res = self.sandbox.run(code)
        out = collections.OrderedDict(tool='run_python', ok=res['ok'], output=res['output'])
        out['_counted'] = counted_in_code(code)
        out['_code'] = code
        out['_reviews'] = res.get('reviews') or []
        return out

    def _store(self, label, name, arguments, output, seconds):
        out = collections.OrderedDict(call=label)
        out.update(output)
        self.outputs[label] = out
        self.calls.append({'call': label, 'tool': name, 'arguments': arguments, 'seconds': round(seconds, 3),
                           'output': public(out)})
        return out

    def overview(self):
        t0 = time.time()
        return self._store('t0', 'overview', {}, self.graph.overview(), time.time() - t0)

    def run_tool(self, name, arguments):
        label = f't{self.next_call}'
        self.next_call += 1
        t0 = time.time()
        try:
            if name == 'run_python':
                output = self._python(arguments)
            elif name not in TOOL_ARGS:
                raise QueryError(f'unknown tool {name}')
            else:
                args = validate_model(TOOL_ARGS[name], json.loads(arguments) if isinstance(arguments, str) else arguments).model_dump()
                output = getattr(self.graph, name)(**args)
        except (QueryError, ExtractionAnomaly, json.JSONDecodeError, TypeError, KeyError) as error:
            output = {'tool': name, 'error': str(error)}
        return self._store(label, name, arguments, output, time.time() - t0)


_MEDDRA = None


def counted_in_code(code):
    """PT codes a code cell counts: PT codes and quoted PT names in it, and the PTs under any HLT/HLGT/SOC code in it."""
    global _MEDDRA
    if _MEDDRA is None:
        from weightloss.graph import analysis_db
        db = analysis_db.connect()
        rows = db.execute('SELECT pt_code, lower(pt), hlt_code, hlgt_code, soc_code FROM meddra').fetchall()
        db.close()
        names = {n: c for c, n, *_ in rows}
        groups = collections.defaultdict(set)
        for c, _, h, g, so in rows:
            for x in (h, g, so):
                groups[x].add(c)
        _MEDDRA = ({c for c, *_ in rows}, names, groups)
    pts, names, groups = _MEDDRA
    found = set()
    for c in re.findall(r'\b1\d{7}\b', code or ''):
        found |= {c} if c in pts else groups.get(c, set())
    literals = re.findall(r"'([^'\n]{3,80})'", code or '') + re.findall(r'"([^"\n]{3,80})"', code or '')
    for literal in literals:
        if literal.lower() in names:
            found.add(names[literal.lower()])
    codes = sorted(found)
    return {'purpose': codes, 'side_effect': codes}


_MINOR_ONLY = None


def minor_only_wordings():
    """Statement wordings (lower case, trimmed) written only by writers aged 13-18; they are never quoted."""
    global _MINOR_ONLY
    if _MINOR_ONLY is None:
        from weightloss.graph import analysis_db
        try:
            db = analysis_db.connect()
            _MINOR_ONLY = {_norm(w) for (w,) in db.execute(
                "SELECT lower(trim(wording)) w FROM statement GROUP BY w "
                "HAVING SUM(writer_age_group = '13-18') = COUNT(*)")}
            db.close()
        except Exception:  # no analysis database (tests without data): nothing to check against
            _MINOR_ONLY = set()
    return _MINOR_ONLY


def number_in_output(cited, text):
    """A cited number appears in a code output: some number written there equals it, or rounds to it at the cited
    precision (a percent may be printed with more decimals or as a fraction of 1)."""
    c = _number(cited)
    if c is None:
        return False
    bare = str(cited).replace(',', '').rstrip('%').strip()
    decimals = len(bare.split('.', 1)[1]) if '.' in bare else 0
    for tok in re.findall(r'-?\d[\d,]*(?:\.\d+)?(?:[eE]-?\d+)?', text or ''):
        try:
            v = float(tok.replace(',', ''))
        except ValueError:
            continue
        if v == c or round(v, decimals) == c or (decimals and abs(v) <= 1 and round(100 * v, decimals) == c):
            return True
    return False


def public(output):
    """The part of a tool output the model sees."""
    return collections.OrderedDict((k, v) for k, v in output.items() if not k.startswith('_'))


# ---------------------------------------------------------------------------------------------------- verifier
TAG = re.compile(r'\[(c\d+)\]')
NUMBER = re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?')
NOT_DATA_NUMBERS = re.compile(r'\b95 ?% ?(?:CIs?|confidence|intervals?)\b|\b\d{2}[-–]\d{2}\b|\b75 or over\b|\btype [12]\b|\bMedDRA \d+(?:\.\d+)?|\b(?:19|20)\d{2}\b|\b\d{8}\b|'
                              r'\bt\d+\b|\bc\d+\b|\bGLP-1\b|\bPT\b|\[c\d+\]|\b\d+(?:\.\d+)? ?mg\b|'
                              r'\bunder 1 day\b|\b1 ?[-–] ?6[- ]days?\b|\b1 ?[-–] ?4[- ]weeks?\b|\bover 4 weeks?\b', re.I)
QUOTED = re.compile(r'["“]([^"”]{8,})["”]')
# Numbers that are not data counts: ages and age bands written out, ordinals, list sizes, numbered items.
EXEMPT_NUMBERS = re.compile(r'\b(?:aged?|ages)\s+\d{1,3}\b|\b\d{1,3}\s*(?:\+|-and-over|and over|or over|or older|years?\b|year-olds?|yo\b)|'
                            r'#\d+|\btop\s+\d+\b|\b\d+(?:st|nd|rd|th)\b', re.I)
NUMBER_WORDS = {w: str(i) for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen '
                                                  'fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
NUMBER_WORD = re.compile(r'\b(' + '|'.join(NUMBER_WORDS) + r')\b', re.I)


def _number(text):
    text = str(text).strip().replace(',', '').rstrip('%').strip()
    try:
        return float(text)
    except ValueError:
        return None


def same_value(cited, actual):
    """A cited value (text) agrees with the row value; a number may be cited rounded to fewer decimals."""
    cited = str(cited).strip()
    if actual is None:
        return cited.lower() in ('null', 'none', '')
    if isinstance(actual, bool):
        return cited.lower() == str(actual).lower()
    if isinstance(actual, (int, float)):
        c = _number(cited)
        if c is None:
            return False
        if c == float(actual):
            return True
        bare = cited.replace(',', '').rstrip('%').strip()
        decimals = len(bare.split('.', 1)[1]) if '.' in bare else 0
        return round(float(actual), decimals) == c
    if isinstance(actual, list):
        return cited.casefold() in {json.dumps(actual).casefold(), ', '.join(map(str, actual)).casefold()} | {str(a).casefold() for a in actual}
    return cited.casefold() == str(actual).strip().casefold()


def _segment(text, pos):
    """The clause before position pos: back to the last line break, table cell, or sentence or clause end."""
    start = max(text.rfind('\n', 0, pos), text.rfind('|', 0, pos))
    for sep in ('. ', '; ', '? ', '! '):
        start = max(start, text.rfind(sep, 0, pos))
    return text[start + 1:pos]


def _rows(output):
    rows = {}
    for value in output.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get('row'), int):
                    rows[item['row']] = item
    return rows


def _norm(text):
    return re.sub(r'\s+', ' ', str(text).replace('’', "'").replace('‘', "'")).strip().casefold()


def uncited_numbers(text, question=''):
    """Problems for numbers in the answer that no citation tag follows before the next number on the same line. Numbers
    inside quotation marks, numbers repeated from the question, doses, age bands, years, codes and ordinals are exempt."""
    blank = lambda m: ' ' * len(m.group(0))
    masked = TAG.sub(blank, text)
    for pattern in (QUOTED, NOT_DATA_NUMBERS, EXEMPT_NUMBERS):
        masked = pattern.sub(blank, masked)
    asked = {_number(x) for x in NUMBER.findall(question or '')}
    tags = [m.start() for m in TAG.finditer(text)]
    numbers = list(NUMBER.finditer(masked))
    problems = []
    for i, m in enumerate(numbers):
        if _number(m.group()) in asked:
            continue
        limit = numbers[i + 1].start() if i + 1 < len(numbers) else len(text)
        cited = any(m.end() <= t < limit and '\n' not in text[m.end():t] and '|' not in text[m.end():t] for t in tags)
        if not cited:
            context = text[max(0, m.start() - 50):m.end() + 30].replace('\n', ' ')
            problems.append(f'the number {m.group()} has no citation tag after it ("...{context}..."); cite it or leave it out')
    return problems


COVER_MIN_RECORDS = 20   # a link_terms row this large (records whose wording matched) ...
COVER_SHARE = 0.10       # ... and at least this share of the best row's matching records must be counted or mentioned
GENERIC_TAIL = re.compile(r'\s+(?:mellitus|syndrome|disorders?|disease|nec)$')
RECORD_ID = re.compile(r'\bt[13]_[a-z0-9]{5,}\b|\breddit:t[13]_|\bwebmd:\d{5,}')


def concept_gaps(text, outputs):
    """Problems for large link_terms rows that no counted concept includes and the answer does not name. A link_terms
    output is checked once one of its nodes is counted somewhere (as a target, a required co-statement or a FAERS concept)."""
    counted = {'purpose': set(), 'side_effect': set()}
    groups = {'purpose': set(), 'side_effect': set()}
    for out in outputs.values():
        for kind, codes in (out.get('_counted') or {}).items():
            counted[kind] |= set(codes)
        rc = out.get('resolved_concepts')
        entries = []
        if isinstance(rc, dict):
            target = (out.get('filters') or {}).get('target')
            entries = [(target, x) for x in rc.get('targets', [])] + [('purpose', x) for x in rc.get('require_purposes', [])] + \
                      [('side_effect', x) for x in rc.get('require_side_effects', [])]
        elif isinstance(rc, list):
            entries = [('side_effect', x) for x in rc]
        for kind, x in entries:
            for m in x.get('members', []):
                if kind in groups and m.get('level') != 'pt':
                    groups[kind].add(m['code'])
    low = _norm(text)
    named = lambda name: any(v and v in low for v in {_norm(name), GENERIC_TAIL.sub('', _norm(name)).strip()})
    problems = []
    for label, out in outputs.items():
        if out.get('tool') != 'link_terms' or out.get('_kind') not in counted:
            continue
        kind = out['_kind']
        rows = [r for r in out.get('rows', []) if r.get('code') not in (None, 'uncoded')]
        used = any(r['code'] in counted[kind] for r in rows) or any(g.get('code') in groups[kind] for g in out.get('groups', []))
        if not used or not rows:
            continue
        size = lambda r: r.get('matching_wording_records_reddit', 0) + r.get('matching_wording_records_webmd', 0)
        top = max(size(r) for r in rows)
        for r in rows:
            m = size(r)
            if (m >= max(COVER_MIN_RECORDS, COVER_SHARE * top) and r['code'] not in counted[kind]
                    and not named(r['name'])):
                problems.append(f"link_terms \"{out.get('text')}\" ({label}) returned {r['name']} with {m} records whose wording "
                                f"matched, which no counted concept includes: give it the same concept name if it expresses the "
                                f"asked meaning, or say in the answer that {r['name']} was not counted and why")
    return problems


# Code that counts a concept: a wording pattern or a PT / MedDRA group condition on statements.
CONCEPT_CODE = re.compile(r"\bREGEXP\b|\bpt(?:_code)?\s*(?:=|IN\s*\()|\b(?:hlt|hlgt|soc)(?:_code)?\s*(?:=|IN\s*\()", re.I)


PLACEHOLDER = re.compile(r'\[(?!c\d+\])[A-Za-z][A-Za-z _-]{1,30}\](?!\()')  # "[comparison]", "[quote]"; not [c1] or [text](link)
LEFT_OUT = re.compile(r"\b(?:not (?:counted|included|count|include)|did(?: not|n't) (?:count|include)|left out|leave out|"
                      r"leaving out|excluded?|excluding|omitted|counted separately|kept separate|set aside|rather than|instead of|"
                      r"(?:is|are|was|were|does|do) not|isn't|aren't|doesn't|don't|opposite|different (?:concept|meaning|symptom|reason))\b",
                      re.I)


def _named(name, low):
    """The answer says the row was left out: a sentence naming it (as a whole term) says it was not counted."""
    for v in {_norm(name), GENERIC_TAIL.sub('', _norm(name)).strip()}:
        if not v:
            continue
        for sentence in re.split(r'(?<=[.!?])\s+|\n', low):
            if re.search(r'(?<![\w-])' + re.escape(v) + r'(?![\w-])', sentence) and LEFT_OUT.search(sentence):
                return True
    return False


def concept_review(text, outputs, cited):
    """Problems from the concept review (vocabulary.py): a row marked check (related in wording to the counted members and
    adding enough records) that is neither counted nor named in the answer, in the query_graph outputs the answer cites or
    in the last residual() review of each role; concept counts in run_python without any residual() review; and uncoded
    wordings a pattern counts only through a match inside a longer word."""
    low = _norm(text)
    counted = {'purpose': set(), 'side_effect': set()}
    counted_wordings = set()
    for out in outputs.values():
        for kind, codes in (out.get('_counted') or {}).items():
            counted.setdefault(kind, set()).update(codes)
        counted_wordings |= {_norm(r.get('wording')) for r in out.get('counted_uncoded_wordings') or []}
    problems = []
    for label in sorted(cited):
        out = outputs.get(label) or {}
        if out.get('tool') not in ('query_graph', 'compare_faers'):
            continue
        for r in out.get('not_counted') or []:
            if not r.get('check'):
                continue
            if 'pt' in r and not (r['code'] in counted.get(r['kind'], set()) or _named(r['pt'], low)):
                problems.append(f"the concept review of {label} lists {r['pt']} ({r['new_records']} more {r['platform']} records, "
                                f"related through \"{r['related_to']}\"), which no counted concept includes: count it if it "
                                f"expresses the asked meaning, or say in the answer that {r['pt']} was not counted and why")
            elif 'uncoded_wording' in r and _norm(r['uncoded_wording']) not in counted_wordings | {''} and \
                    not _named(r['uncoded_wording'], low):
                problems.append(f"the concept review of {label} lists the uncoded wording \"{r['uncoded_wording']}\" "
                                f"({r['new_records']} more {r['platform']} records), which no counted concept includes: add it "
                                'as a wording member if it expresses the asked meaning, or say in the answer that it was not counted and why')
    code_outputs = [o for o in outputs.values() if o.get('tool') == 'run_python']
    if any((outputs.get(label) or {}).get('tool') == 'run_python' for label in cited):
        reviews = [rv for o in code_outputs for rv in o.get('_reviews') or []]
        if not reviews and any(CONCEPT_CODE.search(o.get('_code') or '') and 'statement' in (o.get('_code') or '')
                               for o in code_outputs):
            problems.append('your run_python counts use concept conditions (PT, MedDRA group or wording) but residual() was never '
                            'called: call residual(where, counted) with the condition of the denominator\'s statements and the '
                            'condition of every counted member, then count or name each row it marks with *')
        last = {}
        for rv in reviews:
            for role in rv.get('roles') or []:
                last[role] = rv
        for rv in {id(v): v for v in last.values()}.values():
            for r in rv.get('rows') or []:
                if r.get('check') and not _named(r['name'], low):
                    what = r['name'] if r['kind'] == 'pt' else f"the uncoded wording \"{r['name']}\""
                    problems.append(f"residual() lists {what} ({r['new_records']} more {r['platform']} records, related through "
                                    f"\"{r['related_to']}\"), which your counted condition leaves out: count it if it expresses "
                                    'the asked meaning (then call residual() again), or say in the answer that it was not counted and why')
            for u in rv.get('counted_uncoded') or []:
                if u.get('inside_word_only'):
                    problems.append(f"residual() shows that a pattern counts the uncoded wording \"{u['wording']}\" only through a "
                                    'match inside a longer word; fix the pattern (for example with \\b), count again and call residual() again')
    return problems


HIGHER_WORDS = re.compile(r'\b(?:higher|larger|bigger|greater|more|exceeds?|above|leads?|most)\b', re.I)
LOWER_WORDS = re.compile(r'\b(?:lower|smaller|less|fewer|below|least)\b', re.I)
PARTIAL = re.compile(r'\b(?:not (?:an? )?(?:exhaustive|comprehensive|complete)|(?:is|are) partial|partial (?:count|counts|figures?|results?|list)|'
                     r'(?:did|could) not (?:include|count) .{0,60}\bin the (?:recount|brand counts?|counts?))', re.I)


# The reader never sees the tools: an answer reports results, not the steps taken to reach them ("the concept review",
# "my earlier count"), and never holds back a number for want of a corrected or verified version.
INTERNAL = re.compile(r"\b(?:concept review|residual (?:review|rows?)|residual\(\)|the review (?:flagged|found|showed|lists?)|"
                      r"(?:earlier|previous|first|corrected|verified) (?:count|counts|figures?|numbers?|breakdown|results?)|"
                      r"my (?:earlier|previous|own) |tool outputs?|run_python|query_graph|compare_faers|get_evidence|link_terms)",
                      re.I)


def _interval(out, row_no, field):
    """(low, high) of the share a citation points to, when its row carries the interval."""
    row = _rows(out).get(row_no) or {}
    prefix = field[:-len('percent')] if field.endswith('percent') else None
    if prefix is None:
        return None
    lo, hi = row.get(f'{prefix}ci95_low'), row.get(f'{prefix}ci95_high')
    return (lo, hi) if lo is not None and hi is not None else None


# An interval written after a share, each bound with its own tag: "30.12% [c4]–34.81% [c5]" or "30.1 [c4] to 34.8 [c5]".
STATED_INTERVAL = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*%?\s*\[(c\d+)\]\s*(?:–|—|-|to)\s*(\d[\d,]*(?:\.\d+)?)\s*%?\s*\[(c\d+)\]')
CITED_SHARE = re.compile(r'%\s*\[c\d+\]')


def _stated_interval(answer, cites, cid):
    """(low, high) of the interval the answer writes after the share tagged cid, from the cited bounds: in the same
    sentence (or table row) and before any other cited share; None otherwise."""
    m = re.search(r'\[' + re.escape(cid) + r'\]', answer.answer)
    if not m:
        return None
    window = re.split(r'\.(?=\s|$)|\n', answer.answer[m.end():m.end() + 240], maxsplit=1)[0]  # same sentence or table row
    found = STATED_INTERVAL.search(window)
    if not found or CITED_SHARE.search(window[:found.start()]):
        return None
    lo, hi = cites.get(found.group(2)), cites.get(found.group(4))
    lo, hi = (_number(lo.value) if lo else None), (_number(hi.value) if hi else None)
    return (lo, hi) if lo is not None and hi is not None and lo <= hi else None


def comparison_problems(answer, outputs):
    """Each listed comparison: its text is in the answer and its words say the claim, the claim agrees with the two cited
    percentages, and, when both cited rows carry 95% intervals, higher/lower needs separate intervals and similar
    overlapping ones."""
    problems = []
    plain = _norm(TAG.sub(' ', answer.answer))
    cites = {c.id: c for c in answer.citations}
    for cmp in answer.comparisons:
        label = cmp.text.strip()[:90]
        if _norm(cmp.text).strip(' .,;:') not in plain:
            problems.append(f'comparison "{label}" is not text of the answer; copy the words exactly (without tags)')
            continue
        if cmp.claim != 'similar':
            said = HIGHER_WORDS.search(cmp.text), LOWER_WORDS.search(cmp.text)
            if (cmp.claim == 'higher' and said[1] and not said[0]) or (cmp.claim == 'lower' and said[0] and not said[1]):
                problems.append(f'comparison "{label}": the words say the opposite of the claim {cmp.claim!r}')
        a, b = cites.get(cmp.first), cites.get(cmp.second)
        if a is None or b is None:
            problems.append(f'comparison "{label}": {cmp.first} and {cmp.second} must be citation ids of the two percentages')
            continue
        va, vb = _number(a.value), _number(b.value)
        if va is None or vb is None:
            problems.append(f'comparison "{label}": cite the two percentages, not {a.value!r} and {b.value!r}')
            continue
        if (cmp.claim == 'higher' and not va > vb) or (cmp.claim == 'lower' and not va < vb):
            problems.append(f'comparison "{label}" says the first share ({a.value}) is {cmp.claim} than the second '
                            f'({b.value}); the numbers say otherwise: correct the statement')
            continue
        ia = _interval(outputs.get(a.call) or {}, a.row, a.field)
        ib = _interval(outputs.get(b.call) or {}, b.row, b.field)
        in_code = [c for c in (a, b) if (outputs.get(c.call) or {}).get('tool') == 'run_python']
        if in_code:  # a share printed by code carries no interval row: the answer must state and cite its interval
            stated = {c.id: _stated_interval(answer, cites, c.id) for c in in_code}
            missing = [c.value for c in in_code if stated[c.id] is None]
            if missing:
                problems.append(f'comparison "{label}" rests on shares computed in code ({", ".join(missing)}); in the answer text '
                                'write each one\'s 95% interval right after it, in the same sentence (in a table, the same cell), each '
                                'bound printed by wilson_low/wilson_high and cited, for example 32.4% [c3] (30.12% [c4]–34.81% [c5]); '
                                'the comparison text itself stays words only')
                continue
            ia = ia or stated.get(a.id)
            ib = ib or stated.get(b.id)
            overlap = not (ia[0] > ib[1] or ia[1] < ib[0])
            if cmp.claim != 'similar' and overlap:
                problems.append(f'comparison "{label}": the 95% intervals overlap ({ia[0]}-{ia[1]} and {ib[0]}-{ib[1]}), '
                                'so say the shares are similar or show no clear difference')
            elif cmp.claim == 'similar' and not overlap:
                problems.append(f'comparison "{label}": the 95% intervals do not overlap ({ia[0]}-{ia[1]} and '
                                f'{ib[0]}-{ib[1]}), so the shares are not similar')
            continue
        if ia and ib:
            overlap = not (ia[0] > ib[1] or ia[1] < ib[0])
            if cmp.claim != 'similar' and overlap:
                problems.append(f'comparison "{label}": the 95% intervals overlap ({ia[0]}-{ia[1]} and {ib[0]}-{ib[1]}), '
                                'so say the shares are similar or show no clear difference')
            elif cmp.claim == 'similar' and not overlap:
                problems.append(f'comparison "{label}": the 95% intervals do not overlap ({ia[0]}-{ia[1]} and '
                                f'{ib[0]}-{ib[1]}), so the shares are not similar')
            shown = {(c.call, c.row, c.field) for c in answer.citations}
            missing = [f'{c.call} row {c.row} {c.field[:-len("percent")]}ci95_{end}' for c in (a, b) for end in ('low', 'high')
                       if (c.call, c.row, f'{c.field[:-len("percent")]}ci95_{end}') not in shown]
            if missing:
                problems.append(f'comparison "{label}" rests on two shares: give both 95% intervals in the answer and cite them '
                                f'({", ".join(missing)})')
    if not answer.comparisons and re.search(r'\b(?:share|percent|proportion)', answer.answer, re.I) and re.search(
            r'\b(?:higher|lower|larger|smaller|bigger|greater) (?:share|percent|proportion)|\b(?:higher|lower|larger|smaller) than\b',
            answer.answer, re.I):
        problems.append('the answer compares shares but lists no comparisons; list each comparison with its two percentages')
    return problems


def leaked_ids(text, outputs):
    ids = {i for out in outputs.values() for i in out.get('_record_ids') or []}
    found = sorted({i for i in ids if i and i in text} | set(RECORD_ID.findall(text)))
    return [f'the answer shows record or statement ids ({", ".join(found[:3])}); remove them (ids belong only in the quotes list)'] if found else []


def verify(answer, outputs, question=''):
    """(problems, warnings) for a FinalAnswer against the tool outputs keyed by call label. Problems fail verification.
    A passage in quotation marks must be a listed quote, unless it repeats the question's words or a name or wording the tools
    returned (a PT, a MedDRA group, a statement's wording)."""
    problems, warnings = [], []
    cites = {}
    for c in answer.citations:
        if c.id in cites:
            problems.append(f'citation id {c.id} is used twice')
            continue
        cites[c.id] = c
        out = outputs.get(c.call)
        if out is None:
            problems.append(f'{c.id}: no tool output labelled {c.call!r}')
            continue
        if out.get('tool') == 'link_terms':
            problems.append(f'{c.id}: link_terms counts cover all drugs, platforms and populations and only guide the choice of '
                            'concepts; count the concept with query_graph or run_python and cite that')
            continue
        if out.get('tool') == 'run_python':  # code output is text: the number (or label) must appear in it
            found = (number_in_output(c.value, out.get('output')) if _number(c.value) is not None
                     else _norm(c.value) in _norm(out.get('output') or ''))
            if not found:
                problems.append(f'{c.id}: the value {c.value!r} does not appear in the output of {c.call}')
            continue
        row = _rows(out).get(c.row)
        if row is None:
            problems.append(f'{c.id}: {c.call} has no row {c.row}')
            continue
        if c.field not in row:
            problems.append(f'{c.id}: {c.call} row {c.row} has no field {c.field!r} (fields: {", ".join(k for k in row if k != "row")})')
            continue
        if not same_value(c.value, row[c.field]):
            problems.append(f'{c.id}: cited value {c.value!r} but {c.call} row {c.row} {c.field} = {row[c.field]!r}')
    text = answer.answer
    tags = TAG.findall(text)
    for c in answer.citations:
        if not re.fullmatch(r'c\d+', c.id):
            problems.append(f'citation id {c.id!r} is not of the form c1, c2, ...')
    order = lambda x: (0, int(x[1:])) if re.fullmatch(r'c\d+', x) else (1, x)
    for t in sorted(set(tags) - set(cites), key=order):
        problems.append(f'tag [{t}] has no citation')
    for t in sorted(set(cites) - set(tags), key=order):
        warnings.append(f'citation {t} is not used in the answer')
    for m in TAG.finditer(text):
        c = cites.get(m.group(1))
        if c is None:
            continue
        num = _number(c.value)
        if num is None:  # tags are for numbers; a tag on a name or passage is only flagged (its value was checked above)
            warnings.append(f'[{c.id}] tags a non-numeric value {c.value[:60]!r}')
            continue
        seg = NUMBER_WORD.sub(lambda w: NUMBER_WORDS[w.group(1).lower()], TAG.sub(' ', _segment(text, m.start())))
        nums = NUMBER.findall(seg)  # the number written last before the tag must be the cited one: each tag follows its own number
        if not (nums and _number(nums[-1]) == num):
            problems.append(f'[{c.id}]: the cited value {c.value!r} is not the number written just before the tag')
    # quotes: each must be exact words from the evidence sentences of a statement returned by get_evidence
    sentences = {}
    for out in outputs.values():
        for edge_id, spans in (out.get('_full_evidence') or {}).items():
            sentences.setdefault(edge_id, []).extend(_norm(s) for s in spans)
    quoted = []
    for q in answer.quotes:
        if q.edge_id not in sentences:
            problems.append(f'quote from {q.edge_id!r}: that statement was not returned by get_evidence')
            continue
        text_q = _norm(q.text)
        own = sentences[q.edge_id]
        if not (any(text_q in x for x in own) or text_q in ' '.join(own)):  # one sentence, or consecutive ones of the statement
            elsewhere = [e for e, xs in sentences.items() if e != q.edge_id and (any(text_q in x for x in xs) or text_q in ' '.join(xs))]
            hint = f'; it is a passage of {elsewhere[0]}, cite that edge_id' if elsewhere else ''
            problems.append(f'quote {q.text[:80]!r} is not an exact passage of the evidence sentences of {q.edge_id}{hint}')
            continue
        quoted.append(text_q)
        if text_q.strip(' .,;:') not in _norm(text):
            problems.append(f'quote {q.text[:80]!r} is listed in quotes but not shown in the answer; put it in the answer in '
                            'quotation marks or drop it')
    names ={_norm(v) for out in outputs.values() for row in _rows(out).values() for v in row.values() if isinstance(v, str)}
    names |= {_norm(x) for out in outputs.values() for row in _rows(out).values() for v in row.values() if isinstance(v, list)
              for x in v if isinstance(x, str)}
    asked = _norm(question)
    minors = minor_only_wordings()
    for m in QUOTED.finditer(text):
        passage = _norm(m.group(1)).strip(' .,;:')
        if passage in minors:
            problems.append(f'"{m.group(1)[:60]}" is the wording of a writer aged 13-18 only; describe it in your own words '
                            'without quoting it')
            continue
        if passage in names or (asked and passage in asked) or any(
                passage in _norm(o.get('output', '')) for o in outputs.values() if o.get('tool') == 'run_python'):
            continue
        if len(passage.split()) >= 3 and not any(passage in q or q in passage for q in quoted):
            problems.append(f'the answer quotes "{m.group(1)[:80]}" but quotes lists no exact source passage for it')
    problems += uncited_numbers(text, question)
    patient = any(out.get('tool') in ('query_graph', 'get_evidence') or (
        out.get('tool') == 'run_python' and 'statement' in (out.get('_code') or ''))
        for out in (outputs.get(c.call) or {} for c in answer.citations))
    if patient and not re.search(r'\b(?:reddit|webmd)\b', text, re.I):
        problems.append('the answer cites patient records but never says which platform a number comes from: report Reddit and '
                        'WebMD separately and name them (they are never pooled)')
    plain = TAG.sub(' ', text) + ' ' + ' '.join(answer.limitations)  # names are checked without citation tags
    for m in PLACEHOLDER.finditer(text):
        problems.append(f'the answer contains the placeholder "{m.group(0)}"; remove it (the only tags are [cN] right after numbers)')
    problems += comparison_problems(answer, outputs)
    if PARTIAL.search(plain):
        problems.append(f'the answer says its own count or list is incomplete ("{PARTIAL.search(plain).group(0)}"); run the '
                        'missing step and report the complete result')
    problems += concept_gaps(plain, outputs)
    problems += concept_review(plain, outputs, {c.call for c in answer.citations})
    m = INTERNAL.search(TAG.sub(' ', text))
    if m:
        problems.append(f'the answer describes your working steps ("{m.group(0)}"); report the results the question asks for '
                        '(the best counts you computed, with the members counted and any row left out named in plain words), '
                        'never a missing "corrected" or "verified" result')
    problems += leaked_ids(text, outputs)
    for item in answer.limitations:
        if NUMBER.search(NOT_DATA_NUMBERS.sub(' ', item)):
            warnings.append(f'number in limitations (not verified): {item.strip()[:120]!r}')
    return problems, warnings[:15]


# ---------------------------------------------------------------------------------------------------- critic
CRITIC_TOOL_ROUNDS = 3


class Critique(BaseModel):
    verdict: Literal['ok', 'revise']
    issues: list[str] = Field(description='Each substantive problem, with what to check or change; [] when verdict is ok')


CRITIQUE_SCHEMA = strict_json_schema(Critique)
CRITIC_INSTRUCTIONS = """You check an answer that an analyst agent wrote from tool outputs over a knowledge graph of GLP-1 drug experiences (Reddit posts and comments, WebMD reviews, MedDRA-coded purposes and side effects) and validated FAERS reports. You see the question, the answer and the log of tool calls with their outputs. Flag only substantive problems that a careful peer reviewer would raise and that the log shows or makes likely; do not flag style, length or caveats already present, and do not ask for analyses the question does not need.

Check:
- Concept: do the counted PTs, MedDRA groups and uncoded wordings capture what the question asks? Look for wordings in the outputs that name the concept but sit on PTs or uncoded statements left out, members whose wordings mean something else, and a broad category counted through only part of its MedDRA group.
- Numbers and denominators: does each share use the records with at least one statement of the same role under the same filters? Is a row for one detail value (for example intensity "not stated") presented as a total? For the distribution of an attribute (sex, age group), are shares compared among records or reports where it is stated?
- FAERS: are FAERS numbers taken from compare_faers or from faers_scope with has_clinical_pt = 1 and clinical reactions, and Reddit/WebMD numbers for the comparison from comparable_statement (approval date to 2026-06-30)? No ratio or difference between FAERS and Reddit/WebMD shares.
- Conclusions: is every "higher", "more", "clear" or "no difference" supported by non-overlapping (or overlapping) 95% intervals? When the question asks how groups differ, were all clear differences in the outputs reported? Does the answer address what was asked (an either/or question answered directly; every platform and group the question names)?
- Scope: no medical advice, no personal risk, incidence or causal claim, no outside knowledge presented as data, no quote with personal details.

Return verdict "revise" with the issues when at least one substantive problem exists, otherwise "ok" with no issues."""


def critic_input(question, answer, session):
    log = []
    for c in session.calls[1:]:
        args = c['arguments']
        args = json.loads(args) if isinstance(args, str) else args
        out = c['output']
        text = out.get('output') if c['tool'] == 'run_python' else compact_json(out)
        log.append({'call': c['call'], 'tool': c['tool'], 'arguments': compact_json(args)[:2000], 'output': (text or '')[:3500]})
    return compact_json({'question': question, 'answer': {k: answer[k] for k in ('answer', 'answer_type', 'limitations')},
                         'tool_log': log})


# ---------------------------------------------------------------------------------------------------- loop
def env_file_key():
    """The OPENAI_API_KEY value in the repository .env, or None; never printed."""
    env = ROOT / '.env'
    if env.exists():
        for line in open(env):
            if line.startswith('OPENAI_API_KEY='):
                value = line.split('=', 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    return None


def load_key():
    """OPENAI_API_KEY from the repository .env, else the environment; never printed. The .env file wins, so a stale or
    placeholder variable exported by a shell profile cannot shadow the key set up for this project."""
    key = env_file_key()
    if key:
        os.environ['OPENAI_API_KEY'] = key
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY is not set (environment or .env)')


def _strip_none(value):
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in value]
    return value


def request_body(items, model, effort, allow_tools):
    return {'model': model, 'instructions': INSTRUCTIONS, 'input': items, 'tools': TOOLS,
            'tool_choice': 'auto' if allow_tools else 'none', 'parallel_tool_calls': True,
            'text': {'format': {'type': 'json_schema', 'name': 'agent_answer', 'strict': True, 'schema': FINAL_SCHEMA}},
            'reasoning': {'effort': effort}, 'include': ['reasoning.encrypted_content'], 'store': False,
            'max_output_tokens': MAX_OUTPUT_TOKENS}


def run(question, graph=None, *, model=MODEL, effort=EFFORT, max_tool_rounds=MAX_TOOL_ROUNDS, budget_usd=BUDGET_USD,
        client=None, trace_path=TRACE, label=None, critic=False):
    """Answer one question; returns the result dict (also appended, with the full trace, to trace_path)."""
    own_graph = graph is None
    graph = graph or KnowledgeGraph()
    session = Session(graph)
    overview = session.overview()
    items = [{'role': 'user', 'content': f'Question: {question}\n\nGraph overview (call t0, citable like any tool output):\n'
                                         f'{compact_json(public(overview))}'}]
    started = dt.datetime.now(dt.timezone.utc)
    manifest_sha = hashlib.sha256(neo4j_db.MANIFEST.read_bytes()).hexdigest() if neo4j_db.MANIFEST.exists() else None
    trace = {'run_id': f"{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}", 'label': label,
             'started_utc': started.isoformat(timespec='seconds'), 'agent_version': VERSION, 'question': question,
             'graph': {'uri': neo4j_db.URI, 'manifest_sha256': manifest_sha, 'kg_version': graph.manifest.get('kg_version'),
                       'loaded_utc': graph.manifest.get('loaded_utc')},
             'model': model, 'reasoning_effort': effort, 'max_tool_rounds': max_tool_rounds, 'budget_usd': budget_usd,
             'instructions_sha256': sha256_text(INSTRUCTIONS), 'tools_sha256': sha256_text(compact_json(TOOLS)),
             'final_schema_sha256': sha256_text(compact_json(FINAL_SCHEMA)), 'instructions': INSTRUCTIONS, 'tools': TOOLS,
             'final_schema': FINAL_SCHEMA, 'requests': [], 'responses': [], 'attempts': []}
    usage_total, cost, tool_rounds, repairs, sent = collections.Counter(), 0.0, 0, 0, 0
    stop_reason, final, critic_used, best = None, None, False, None
    trace['critiques'] = []
    try:
        if client is None:
            load_key()
            from openai import OpenAI
            client = OpenAI(max_retries=5, timeout=300)
        while True:
            if cost >= budget_usd:
                stop_reason = f'budget reached (${cost:.4f} of ${budget_usd})'
                break
            if len(trace['responses']) >= max_tool_rounds + MAX_REPAIRS * (REPAIR_TOOL_ROUNDS + 1) + CRITIC_TOOL_ROUNDS + 4:
                stop_reason = f"response cap reached ({len(trace['responses'])} responses)"
                break
            allow_tools = tool_rounds < max_tool_rounds + REPAIR_TOOL_ROUNDS * repairs + (
                CRITIC_TOOL_ROUNDS if critic_used else 0)
            body = request_body(items, model, effort, allow_tools)
            trace['requests'].append({**{k: v for k, v in body.items() if k not in ('input', 'instructions', 'tools')},
                                      'input_items': len(items), 'input_new': items[sent:]})
            sent = len(items)
            t0 = time.time()
            try:
                raw = client.responses.create(**body).model_dump()
            except Exception as error:  # transport or HTTP error; the SDK already retried
                stop_reason = f'transport_error: {type(error).__name__}: {mask(error)[:300]}'
                break
            usage = raw.get('usage') or {}
            usd = cost_usd(model, usage) or 0.0
            cost += usd
            for k, v in token_split(usage).items():
                usage_total[k] += v
            trace['responses'].append({'seconds': round(time.time() - t0, 2), 'usd': usd, 'raw': raw})
            calls = [i for i in raw.get('output') or [] if i.get('type') == 'function_call']
            if calls and raw.get('status') == 'completed':
                tool_rounds += 1
                items.extend(_strip_none(i) for i in raw['output'])
                for c in calls:
                    out = session.run_tool(c['name'], c['arguments'])
                    items.append({'type': 'function_call_output', 'call_id': c['call_id'], 'output': compact_json(public(out))})
                continue
            attempt = {'raw_text': None, 'parsed': None, 'problems': [], 'warnings': [], 'anomaly': None}
            try:
                attempt['raw_text'] = final_output_text(raw)
                parsed = validate_model(FinalAnswer, parse_json_object(attempt['raw_text']))
                attempt['parsed'] = parsed.model_dump()
                attempt['problems'], attempt['warnings'] = verify(parsed, session.outputs, question)
            except ExtractionAnomaly as error:
                attempt['anomaly'] = error.as_dict()
                attempt['problems'] = [f'answer not accepted ({error.kind}): {error.detail}']
            trace['attempts'].append(attempt)
            final = attempt
            if not attempt['problems']:
                best = attempt
                if critic and not critic_used and cost < budget_usd:
                    critic_used = True
                    cbody = {'model': model, 'instructions': CRITIC_INSTRUCTIONS,
                             'input': [{'role': 'user', 'content': critic_input(question, attempt['parsed'], session)}],
                             'text': {'format': {'type': 'json_schema', 'name': 'critique', 'strict': True,
                                                 'schema': CRITIQUE_SCHEMA}},
                             'reasoning': {'effort': effort}, 'store': False, 'max_output_tokens': 8000}
                    try:
                        craw = client.responses.create(**cbody).model_dump()
                        cusd = cost_usd(model, craw.get('usage') or {}) or 0.0
                        cost += cusd
                        for k, v in token_split(craw.get('usage') or {}).items():
                            usage_total[k] += v
                        crit = validate_model(Critique, parse_json_object(final_output_text(craw)))
                        trace['critiques'].append({'usd': cusd, 'verdict': crit.verdict, 'issues': crit.issues})
                    except Exception as error:  # a failed critique never blocks a verified answer
                        trace['critiques'].append({'error': mask(f'{type(error).__name__}: {error}')[:300]})
                        break
                    if crit.verdict == 'revise' and crit.issues:
                        if raw.get('status') == 'completed':
                            items.extend(_strip_none(i) for i in raw['output'])
                        items.append({'role': 'user', 'content': CRITIC_REPAIR.format(
                            issues='\n'.join(f'- {x}' for x in crit.issues))})
                        continue
                break
            if repairs >= MAX_REPAIRS or (attempt['anomaly'] or {}).get('kind') in ('refusal', 'content_filter'):
                break
            repairs += 1
            if raw.get('status') == 'completed':
                items.extend(_strip_none(i) for i in raw['output'])
            items.append({'role': 'user', 'content': REPAIR.format(problems='\n'.join(f'- {p}' for p in attempt['problems']))})
    finally:
        if best is not None and (final is None or final['problems']):  # a failed revision never replaces a verified answer
            final = best
        verified = bool(final and final['parsed'] is not None and not final['problems'])
        result = {'run_id': trace['run_id'], 'label': label, 'question': question, 'verified': verified,
                  'answer': (final or {}).get('parsed'), 'problems': (final or {}).get('problems', []),
                  'warnings': (final or {}).get('warnings', []), 'repair_used': repairs > 0, 'repairs': repairs, 'stop_reason': stop_reason,
                  'critique': trace['critiques'][-1] if trace['critiques'] else None,
                  'tool_rounds': tool_rounds, 'responses': len(trace['responses']),
                  'tool_calls': [{'call': c['call'], 'tool': c['tool'], 'arguments': c['arguments'],
                                  'seconds': c['seconds'], 'error': c['output'].get('error')} for c in session.calls],
                  'usage': dict(usage_total), 'cost_usd': round(cost, 6)}
        trace.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), tool_calls=session.calls,
                     result=result)
        if trace_path:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            line = mask(json.dumps(trace, ensure_ascii=False, default=str)) + '\n'
            with TRACE_LOCK, open(trace_path, 'a') as f:
                f.write(line)
        session.close()
        if own_graph:
            graph.close()
    return result
