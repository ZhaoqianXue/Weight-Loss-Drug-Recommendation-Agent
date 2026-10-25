"""Blind rating workbook for the RQ3 human evaluation: AskGLP1 vs B1 (LLM only) vs B2 (text retrieval) on 120 questions.

Layout and styling follow the project's other annotation sheets (Arial 10, gray header, gray number and id columns, yellow
example rows labelled "Example n", column titles written as questions with notes on hover, dropdowns that reject other
values): an Instructions tab from rater-guide.md and one tab per rater. Every question takes three rows, one per system,
labelled A, B and C in an order shuffled per question; the question and its reference answer are merged over the three rows.
Four raters work in two pairs: pair A (Junting, Zhongyin) rates questions 1-60 and pair B (Anay, Siddhartha) questions
61-120, so every question is rated by two people. Which letter is
which system, the order and the assignment are written only to the local key file.

Inputs: eval_questions_human120.json; the first run of each system (.cache/human-eval-20261020/runs/final-{askglp1,b1,b2}-h120);
the reference answers (.cache/human-eval-20261020/references-h120/part-*.jsonl); examples.json.
Writes .cache/human-eval-20261020/sheet/{answer-evaluation.xlsx, key.json, manifest.json}.
"""
import argparse
import glob
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils.indexed_list import IndexedList
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from weightloss.web.ask import resolve  # noqa: E402  (the website's own citation resolver)

STAGE = ROOT / 'results/human-eval-20261020'
CACHE = ROOT / '.cache/human-eval-20261020'
QUESTIONS = STAGE / 'eval_questions_human120.json'
RUNS = {'askglp1': CACHE / 'runs/final-askglp1-h120', 'b1': CACHE / 'runs/final-b1-h120', 'b2': CACHE / 'runs/final-b2-h120'}
EXAMPLE_RUNS = {'askglp1': ROOT / '.cache/agent-runs/dev084-v8-20261020',
                'b1': CACHE / 'runs/dev3-b1-v8', 'b2': CACHE / 'runs/dev3-b2-v8'}
REFERENCES = CACHE / 'references-h120'
ORDER_SEED = 20261022  # assignment of questions to the two pairs and their order in the sheet
EXAMPLES = STAGE / 'examples.json'
GUIDE = STAGE / 'rater-guide.md'
OUT = CACHE / 'sheet'
SEED = 20261021
SYSTEMS = ('askglp1', 'b1', 'b2')

FONT, BOLD, TITLE = Font(name='Arial', size=10), Font(name='Arial', size=10, bold=True), Font(name='Arial', size=13, bold=True)
GRAY_FONT = Font(name='Arial', size=10, color='FF666666')
HEAD_FILL, ID_FILL = PatternFill('solid', fgColor='FFD9D9D9'), PatternFill('solid', fgColor='FFF3F3F3')  # as in the project's other annotation sheets
EXAMPLE_FILL = PatternFill('solid', fgColor='FFFFF2CC')
TOP_WRAP = Alignment(vertical='top', wrap_text=True)
SEPARATOR = Border(top=Side(style='thin', color='FFBFBFBF'))
PAIRS = {'A': ('Junting', 'Zhongyin'), 'B': ('Anay', 'Siddhartha')}  # each question is rated by both members of one pair

Q1 = ('Does the answer agree with the reference answer? (dropdown)', ['No error', 'Minor error', 'Major error'])
Q2 = ('Is every claim backed by the evidence it cites? (dropdown)', ['All supported', 'Some unsupported', 'None supported', 'No data claims'])
COLUMNS = [('No.', 8, 'Question number.'),
           ('Question ID', 9, None),
           ('Answer', 7, 'Which of the three answers this row shows. The letters are shuffled for every question and do not tell '
                         'you which system wrote the answer.'),
           ('Question', 24, None),
           ('Reference answer (what the data say)', 38, 'Worked out directly from the data. The first line says whether a good '
            'answer answers, answers only in part, or declines. Key facts are what a correct answer must get right.'),
           ("The system's answer (type, text, limitations)", 44, 'Tags such as [c1] point to the next column.'),
           ('Evidence the answer cites', 44, 'What each tag points to: a count from the data, a computed number, a search result '
            'or a sentence a writer wrote. Some answers cite nothing.'),
           (Q1[0], 15, 'No error: same conclusion and key facts. Minor error: conclusion right, something off. Major error: '
                       'conclusion wrong or missing, a refusal the reference does not support, or, where the reference says to '
                       'decline or limit, medical advice or a claim the data cannot support.'),
           (Q2[0], 15, 'Judge against the evidence column only, not against the reference. No data claims: the answer says '
                       'nothing about the data (for example, a plain refusal).'),
           ('Comment (optional)', 20, 'One short sentence, especially for Minor and Major errors.')]
TOOL_NAMES = {'query_graph': 'count in the knowledge graph', 'compare_faers': 'comparison with FAERS',
              'get_evidence': "search of writers' sentences", 'link_terms': 'lookup of medical terms',
              'run_python': 'calculation', 'search_records': 'keyword search of the posts and reviews'}
EXPECTED = {'answer': 'should answer', 'limited': 'should answer only in part (limited)', 'decline': 'should decline'}


# ---------------------------------------------------------------------------------------------------- loading
def latest_traces(run_dir):
    """id -> trace of the run recorded in results.jsonl (a resumed run can hold an earlier failed trace too)."""
    wanted = {r['id']: r['run_id'] for r in map(json.loads, open(run_dir / 'results.jsonl'))}
    out = {}
    for line in open(run_dir / 'traces.jsonl'):
        t = json.loads(line)
        qid = (t.get('label') or '').rsplit(':', 1)[-1]
        if wanted.get(qid) == t['run_id']:
            out[qid] = t
    return out


def load_references():
    refs = {}
    for path in sorted(glob.glob(str(REFERENCES / 'part-*.jsonl'))):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                refs[r['id']] = r
    return refs


# ---------------------------------------------------------------------------------------------------- rendering
def plain(text):
    """Markdown to sheet text: bold and heading marks dropped; each table row becomes one line
    '• first cell — header: value; header: value', which reads far better in a cell than pipes."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text or '')
    text = re.sub(r'(?m)^#+\s*', '', text)
    out, headers = [], None
    for line in text.splitlines():
        if line.strip().startswith('|'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if all(re.fullmatch(r':?-{2,}:?', c) for c in cells if c):
                continue
            if headers is None:
                headers = cells
                continue
            pairs = [f'{h}: {v}' for h, v in zip(headers[1:], cells[1:]) if v]
            out.append(f"• {cells[0]} — " + '; '.join(pairs) if pairs else f'• {cells[0]}')
        else:
            headers = None
            out.append(line)
    return '\n'.join(out).strip()


def fmt(value):
    if isinstance(value, bool) or value is None:
        return str(value).lower()
    if isinstance(value, int):
        return f'{value:,}'
    if isinstance(value, float):
        return f'{value:,.2f}'.rstrip('0').rstrip('.')
    if isinstance(value, list):
        shown = [fmt(v) if not isinstance(v, dict) else compact_dict(v) for v in value[:4]]
        return ', '.join(shown) + (f' (+{len(value) - 4} more)' if len(value) > 4 else '')
    if isinstance(value, dict):
        return compact_dict(value)
    return str(value)


SKIP = re.compile(r'^(row|.*_code|code|edge_id|record_id|record_key|call|_.*)$')
WORDS = {'ci95': '95% interval', 'wilson': '95% interval', 'faers': 'FAERS', 'reddit': 'Reddit', 'webmd': 'WebMD',
         'pt': 'medical term', 'pts': 'medical terms', 'pct': '%', 'percent': '%', 'denominator': 'out of',
         'records': 'records', 'vs': 'vs', 'group': 'grouped', 'by': 'by', 'top': 'top', 'n': 'n', 'soc': 'body-system group',
         'hlt': 'term group', 'hlgt': 'term group', 'uncoded': 'without a medical term', 'wordings': "writers' words",
         'wording': "writer's words", 'common': 'common', 'sparse': '(fewer than 10 records)', 'small': '(fewer than 30 records)',
         'universe': 'all', 'scope': 'scope', 'scopes': 'scopes', 'targets': 'looked for', 'target': 'looking at',
         'platforms': 'platforms', 'brands': 'brands', 'ingredients': 'ingredients', 'sexes': 'sex', 'age': 'age',
         'groups': 'groups', 'clear': 'clear', 'differences': 'differences', 'level': 'level', 'concept': 'concept'}


def label(key):
    return ' '.join(WORDS.get(w, w) for w in key.split('_'))


def compact_dict(d, limit=None):
    """One tool row in plain words: 'Reddit records 256; out of 67,542; % 0.4; 95% CI 0.34-0.43'."""
    parts, done = [], set()
    for k, v in d.items():
        if k in done or SKIP.match(k) or v in (None, [], '', {}) or v is False:
            continue
        for lo, hi in (('ci95_low', 'ci95_high'), ('wilson_low', 'wilson_high')):
            if k.endswith(lo) and k[:-len(lo)] + hi in d:
                partner = k[:-len(lo)] + hi
                parts.append(f"{label(k[:-len(lo)] + 'ci95')} {fmt(v)}–{fmt(d[partner])}".strip())
                done.add(partner)
                break
        else:
            parts.append(label(k) if v is True else f'{label(k)} {fmt(v)}')
    text = '; '.join(parts)
    return text if limit is None or len(text) <= limit else text[:limit - 1] + '…'


def answer_cell(answer):
    if not answer:
        return 'No answer (the system stopped without one).'
    lims = answer.get('limitations') or []
    return (f"Answer type: {answer.get('answer_type')}\n\n{plain(answer.get('answer'))}" +
            ('\n\nLimitations:\n' + '\n'.join(f'- {x}' for x in lims) if lims else ''))


def output_lines(value, printed, limit=5):
    """The whole lines of a code output that hold the cited value (as the verifier matches it), untruncated."""
    from weightloss.agent.agent import _norm, _number, number_in_output
    lines = [line for line in printed.splitlines() if line.strip()]
    if _number(value) is not None:
        hits = [line for line in lines if number_in_output(value, line)]
    else:
        hits = [line for line in lines if _norm(value) in _norm(line)]
    return hits[:limit]


def askglp1_evidence(trace):
    r = resolve(trace['result'], trace)
    calls = {c['call']: c for c in trace.get('tool_calls') or []}
    groups = {}
    for c in r['citations']:
        groups.setdefault((c['call'], c['row'] if c['tool'] != 'run_python' else 0, c['tool']), []).append(c)
    lines, seen_calls = [], set()
    for (call, row, tool), cs in groups.items():
        tags = ''.join(f"[{c['id']}]" for c in cs)
        if call not in seen_calls:
            seen_calls.add(call)
            args = cs[0].get('arguments') or {}
            brief = '' if tool == 'run_python' else compact_dict(args if isinstance(args, dict) else {}, 600)
            lines.append(f"Lookup {call} ({TOOL_NAMES.get(tool, tool)})" + (f': {brief}' if brief else ''))
        if tool == 'run_python':
            printed = ((calls.get(call) or {}).get('output') or {}).get('output') or ''
            by_line = {}
            for c in cs:
                found = output_lines(c['value'], printed)
                by_line.setdefault(found[0] if found else '(value not found in what the calculation printed)', []).append(c)
            for text, group in by_line.items():
                tags = ', '.join(f"[{c['id']}] {c['value']}" for c in group)
                lines.append(f"  {tags}  ← printed: {text}")
        else:
            src = cs[0].get('source')
            lines.append(f"  {tags} ← row {row}: " + (compact_dict(src) if src else '(row not found)'))
    for qt in r['quotes']:
        st = qt.get('statement') or {}
        where = ', '.join(x for x in (f"{st.get('platform', '')} statement".strip(), st.get('brand') or st.get('ingredient'),
                                     st.get('date')) if x)
        sent = ' '.join(st.get('evidence') or []) or '(sentence not found)'
        lines.append(f"Quote “{qt['text']}” ← {where}: {sent}")
    return '\n'.join(lines) or 'No evidence cited.'


def b2_evidence(trace):
    a = trace['result'].get('answer') or {}
    calls = {c['call']: c for c in trace.get('tool_calls') or []}
    lines, seen_calls = [], set()
    groups = {}
    for c in a.get('citations') or []:
        groups.setdefault((c['call'], c['row']), []).append(c['id'])
    for (call, row), ids in groups.items():
        tags = ''.join(f'[{i}]' for i in ids)
        tc = calls.get(call) or {}
        out = tc.get('output') or {}
        if call not in seen_calls:
            seen_calls.add(call)
            args = tc.get('arguments') or {}
            terms = ' AND '.join('(' + ' OR '.join(g) + ')' for g in (args.get('groups') or []) if g) if isinstance(args, dict) else ''
            m = out.get('matching_records') or {}
            lines.append(f"Lookup {call} (keyword search of the posts and reviews, platform {args.get('platform') if isinstance(args, dict) else '?'}): "
                         f"{terms} → matching records: Reddit {m.get('reddit', 0):,}, WebMD {m.get('webmd', 0):,}")
        if row == 0:
            lines.append(f'  {tags} ← the match counts of {call}')
            continue
        res = {x['row']: x for x in out.get('results') or []}.get(row)
        if not res:
            lines.append(f'  {tags} row {row}: (row not found)')
            continue
        meta = res.get('community') or res.get('profile') or ''
        lines.append(f"  {tags} ← row {row}: {res['record']}, {meta}, {res['date']}: {res['text']}")
    for qt in a.get('quotes') or []:
        lines.append(f"Quote “{qt['text']}” ← {qt['call']} row {qt['row']}")
    return '\n'.join(lines) or 'No evidence cited.'


def evidence_cell(system, trace):
    if system == 'askglp1':
        return askglp1_evidence(trace)
    if system == 'b2':
        return b2_evidence(trace)
    return 'No evidence cited.'


def reference_cell(ref, expected):
    if not ref:
        return f'(reference not written yet)\n\nA good answer {EXPECTED[expected]}.'
    facts = '\n'.join(f'- {f}' for f in ref.get('key_facts') or [])
    cannot = ref.get('cannot_answer') or ''
    concepts = '\n'.join(f"- {c.get('concept')}: {', '.join(c.get('members') or [])}" for c in ref.get('concepts') or [])
    calls = '\n'.join(f"- {c['debatable']}" for c in ref.get('concepts') or [] if c.get('debatable'))
    head = '' if re.match(r'(Should|Limited)', ref['reference']) else f'A good answer {EXPECTED[expected]}.\n\n'
    return (head + ref['reference'] + (f'\n\nKey facts:\n{facts}' if facts else '') +
            (f'\n\nThe data cannot answer: {cannot}' if cannot else '') + (f'\n\nCounted as:\n{concepts}' if concepts else '') +
            (f'\n\nJudgment calls (either reading is acceptable where the key facts give a range):\n{calls}' if calls else ''))


# ---------------------------------------------------------------------------------------------------- workbook
def guide_rows(path):
    """Markdown guide -> (cells, style) rows, style in title/heading/table_head/table/text (as in the other workbooks)."""
    rows = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line:
            rows.append(([], 'text'))
            continue
        if re.fullmatch(r'\|[-|\s]+\|', line):
            continue
        clean = lambda t: t.replace('**', '').strip()
        if line.startswith('# '):
            rows.append(([clean(line[2:])], 'title'))
        elif line.startswith('## '):
            rows.append(([clean(line[3:])], 'heading'))
        elif line.startswith('|'):
            cells = [clean(c) for c in line.strip('|').split('|')]
            head = rows and rows[-1][1] not in ('table_head', 'table')
            rows.append((cells, 'table_head' if head else 'table'))
        elif line.startswith('- '):
            rows.append((['• ' + clean(line[2:])], 'text'))
        else:
            rows.append(([clean(line)], 'heading' if line.startswith('**') and line.endswith('**') else 'text'))
    out = []
    for cells, style in rows:  # one blank line at most, none under a heading
        if not cells and out and (not out[-1][0] or out[-1][1] in ('heading', 'title')):
            continue
        out.append((cells, style))
    return out


def instructions(wb):
    ws = wb.active
    ws.title = 'Instructions'
    widths = (30, 105)
    for col, width in zip('AB', widths):
        ws.column_dimensions[col].width = width
    for r, (cells, style) in enumerate(guide_rows(GUIDE), 1):
        if style in ('table_head', 'table'):
            for c, value in enumerate(cells[:2], 1):
                cell = ws.cell(r, c, value)
                cell.alignment = TOP_WRAP
                cell.font = BOLD if style == 'table_head' else FONT
                if style == 'table_head':
                    cell.fill = HEAD_FILL
            continue
        if not cells:
            continue
        cell = ws.cell(r, 1, cells[0])
        cell.alignment = TOP_WRAP
        cell.font = TITLE if style == 'title' else BOLD if style == 'heading' else FONT
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
        lines = max(1, -(-len(cells[0]) // int(sum(widths) * 1.15)))  # merged cells do not grow by themselves
        ws.row_dimensions[r].height = (20 if style == 'title' else 14) * lines + 2
    ws.sheet_view.showGridLines = False


def rater_tab(wb, name, blocks):
    ws = wb.create_sheet(name)
    for c, (title, width, note) in enumerate(COLUMNS, 1):
        cell = ws.cell(1, c, title)
        cell.font, cell.fill, cell.alignment = BOLD, HEAD_FILL, TOP_WRAP
        if note:
            cell.comment = Comment(note, 'Instructions', width=320, height=140)
        ws.column_dimensions[cell.column_letter].width = width
    r = 2
    for block in blocks:
        first = r
        for row in block['rows']:
            no = block['no']
            values = [no, block['id_label'], row['letter'], block['question'], block['reference'], row['answer'],
                      row['evidence']] + row.get('ratings', [None] * 3)
            for c, value in enumerate(values, 1):
                cell = ws.cell(r, c, value)
                cell.font, cell.alignment = FONT, TOP_WRAP
                if c in (1, 2, 3):
                    cell.font = GRAY_FONT
                    cell.alignment = Alignment(horizontal='center' if c != 2 else None, vertical='top', wrap_text=True)
                if block['example']:
                    cell.fill = EXAMPLE_FILL
                elif c in (1, 2, 3):
                    cell.fill = ID_FILL
                if r == first and first > 2:
                    cell.border = SEPARATOR
            if row.get('why'):
                ws.cell(r, 8).comment = Comment(row['why'], 'Instructions', width=320, height=140)
            r += 1
        if len(block['rows']) > 1:
            for col in (1, 2, 4, 5):
                ws.merge_cells(start_row=first, start_column=col, end_row=r - 1, end_column=col)
    last = r - 1
    for col, (title, values) in zip('HI', (Q1, Q2)):
        dv = DataValidation(type='list', formula1='"' + ','.join(values) + '"', allow_blank=True, showDropDown=False,
                            showErrorMessage=True, error='Choose one of: ' + ', '.join(values) + '.')
        dv.add(f'{col}2:{col}{last}')
        ws.add_data_validation(dv)
    ws.freeze_panes = 'E2'
    return ws


def build_blocks(spec, runs, refs, rng, example=False, picks=None):
    blocks, key = [], {}
    for q in spec:
        traces = {s: runs[s][q['id']] for s in SYSTEMS}
        order = list(SYSTEMS)
        rng.shuffle(order)
        letters = dict(zip('ABC', order))
        if picks:
            letters = {k: v for k, v in letters.items() if v in picks}
        key[q['id']] = letters
        rows = []
        for letter, system in letters.items():
            t = traces[system]
            rows.append({'letter': letter, 'answer': answer_cell(t['result'].get('answer')), 'evidence': evidence_cell(system, t)})
        blocks.append({'qid': q['id'], 'question': q['question'], 'reference': reference_cell(refs.get(q['id']), q['expected']),
                       'rows': rows, 'example': example})
    return blocks, key


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    spec = json.loads(QUESTIONS.read_text())
    runs = {s: latest_traces(d) for s, d in RUNS.items()}
    missing = [(s, q['id']) for s in SYSTEMS for q in spec['questions'] if q['id'] not in runs[s]]
    if missing:
        raise SystemExit(f'missing answers: {missing[:10]}')
    refs = load_references()
    # Assignment and order look at the questions only: the 120 questions are shuffled within each kind and dealt
    # alternately to the two pairs, so both get the same mix; each pair's questions then appear in a random order.
    order_rng = random.Random(ORDER_SEED)
    kind = lambda q: q['area'] if q['area'] != 'boundary' else 'boundary-' + q['expected']
    halves, turn = {'A': [], 'B': []}, 0
    for k in sorted({kind(q) for q in spec['questions']}):
        pool = [q for q in spec['questions'] if kind(q) == k]
        order_rng.shuffle(pool)
        for q in pool:
            halves['AB'[turn % 2]].append(q)
            turn += 1
    for h in halves.values():
        order_rng.shuffle(h)
    sequence = halves['A'] + halves['B']
    blocks, letters = build_blocks(sequence, runs, refs, random.Random(SEED))
    group = {q['id']: p for p, h in halves.items() for q in h}
    for i, b in enumerate(blocks, 1):
        b['no'], b['id_label'], b['group'] = i, b['qid'], group[b['qid']]
    example_blocks = []
    ex = json.loads(EXAMPLES.read_text())
    ex_runs = {s: latest_traces(d) for s, d in EXAMPLE_RUNS.items()}
    v8 = {q['id']: q for q in json.loads((ROOT / 'src/weightloss/agent/eval_questions_test_v8.json').read_text())['questions']}
    for i, e in enumerate(ex['examples'], 1):
        b, k = build_blocks([v8[e['id']]], ex_runs, {e['id']: e['reference']}, random.Random(e['seed']), example=True,
                            picks=[r['system'] for r in e['rows']])
        ratings = {r['system']: r for r in e['rows']}
        for row in b[0]['rows']:
            er = ratings[k[e['id']][row['letter']]]
            row['ratings'], row['why'] = er['ratings'] + [er.get('note', '')], er.get('why')
        b[0]['no'], b[0]['id_label'] = f'Example {i}', f'example-{i}'
        example_blocks += b
    wb = Workbook()
    wb._fonts = IndexedList([FONT])  # workbook default font Arial 10, as in the other annotation sheets
    instructions(wb)
    tabs = {}
    for pair, names in PAIRS.items():
        mine = [b for b in blocks if b['group'] == pair]
        for name in names:
            rater_tab(wb, name, example_blocks + mine)
            tabs[name] = {'pair': pair, 'questions': [b['qid'] for b in mine]}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'answer-evaluation.xlsx'
    wb.save(path)
    numbers = {b['qid']: b['no'] for b in blocks}
    (OUT / 'key.json').write_text(json.dumps({
        'seed': SEED, 'order_seed': ORDER_SEED, 'letters': letters, 'numbers': numbers, 'pairs': PAIRS, 'tabs': tabs,
        'pair_questions': {p: [q['id'] for q in h] for p, h in halves.items()}}, indent=1))
    manifest = {'questions_sha256': hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(),
                'runs': {s: str(d.relative_to(ROOT)) for s, d in RUNS.items()}, 'references': len(refs), 'questions': len(blocks),
                'pair_questions': {p: len(h) for p, h in halves.items()},
                'answers_per_rater': {n: 3 * len(t['questions']) for n, t in tabs.items()},
                'example_rows': sum(len(b['rows']) for b in example_blocks),
                'xlsx_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
