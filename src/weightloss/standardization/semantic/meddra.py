"""Loader for the official MedDRA ASCII distribution (llt.asc, pt.asc; release file optional).

Format (MedDRA distribution file format, 29.1 English): every record ends with '$' and fields are '$'-separated.
llt.asc has 11 fields: llt_code, llt_name, pt_code, llt_whoart_code, llt_harts_code, llt_costart_sym, llt_icd9_code,
llt_icd9cm_code, llt_icd10_code, llt_currency (index 9, Y/N), llt_jart_code. pt.asc has 11 fields: pt_code, pt_name,
null_field, pt_soc_code (index 3, primary SOC), pt_whoart_code, pt_harts_code, pt_costart_sym, pt_icd9_code,
pt_icd9cm_code, pt_icd10_code, pt_jart_code. A line with a different field count is a format error, never guessed.

Only current LLTs are mapping targets. Names that fold to the same key under case folding (e.g. the 29.1 English Rh
antigen E/e terms) form a collision bucket: a folded lookup into such a bucket never yields an accepted mapping.
No directory given / no such directory raises ResourceBlocked (the run is resource_blocked, never mass unmapped); a
given directory that is incomplete or malformed fails (MeddraFormatError). meddra_release.asc holds exactly one
record of 5 fields (version, language, three empty fields); the language must equal the configured one.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import difflib
import hashlib
import json
from pathlib import Path
import re

from .textnorm import fold_key, loose_key, trigrams

LLT_FIELDS = ('llt_code', 'llt_name', 'pt_code', 'llt_whoart_code', 'llt_harts_code', 'llt_costart_sym', 'llt_icd9_code',
              'llt_icd9cm_code', 'llt_icd10_code', 'llt_currency', 'llt_jart_code')
PT_FIELDS = ('pt_code', 'pt_name', 'null_field', 'pt_soc_code', 'pt_whoart_code', 'pt_harts_code', 'pt_costart_sym',
             'pt_icd9_code', 'pt_icd9cm_code', 'pt_icd10_code', 'pt_jart_code')
REQUIRED = ('llt.asc', 'pt.asc')
RELEASE = 'meddra_release.asc'
RELEASE_FIELDS = ('version', 'language', 'null_field_1', 'null_field_2', 'null_field_3')
SYNTHETIC_MARKER = 'SYNTHETIC_TEST_DICTIONARY'
DICTIONARY_CLASSES = ('formal', 'unpinned', 'synthetic_test')
CODE = re.compile(r'^[0-9]{8}$')
VERSION_RE = re.compile(r'^[0-9]{1,2}\.[0-9]$')


class ResourceBlocked(Exception):
    """A required controlled resource is not available; the run is resource_blocked (not a semantic result)."""

    def __init__(self, reason, detail=''):
        super().__init__(f'{reason}: {detail}' if detail else reason)
        self.reason, self.detail = reason, detail


class MeddraFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Llt:
    code: str
    name: str
    pt_code: str
    current: bool


@dataclass(frozen=True)
class Pt:
    code: str
    name: str
    primary_soc_code: str


def parse_record(line, fields, filename, lineno):
    """One '$'-terminated record -> its field values; any other shape is a MeddraFormatError."""
    body = line.rstrip('\r\n')
    if not body.endswith('$'):
        raise MeddraFormatError(f'{filename}:{lineno}: record does not end with $')
    parts = body.split('$')
    if len(parts) != len(fields) + 1 or parts[-1] != '':
        raise MeddraFormatError(f'{filename}:{lineno}: {len(parts) - 1} fields, expected {len(fields)}')
    return dict(zip(fields, parts[:-1]))


def _find(directory, name):
    matches = [p for p in directory.iterdir() if p.is_file() and p.name.lower() == name]
    if len(matches) > 1:
        raise MeddraFormatError(f'several files named {name} (case-insensitive) in {directory}')
    return matches[0] if matches else None


def _read(path, encoding):
    raw = path.read_bytes()
    try:
        text = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError) as error:
        raise MeddraFormatError(f'{path.name} does not decode as {encoding}: {error}') from None
    if text.encode(encoding) != raw:
        raise MeddraFormatError(f'{path.name}: decoding as {encoding} is not lossless')
    return raw, text


class MeddraDictionary:
    """Read-only index of one frozen MedDRA release."""

    def __init__(self, *, version, language, encoding, files, llts, pts, version_source, integrity, dictionary_class):
        if dictionary_class not in DICTIONARY_CLASSES:
            raise ValueError(f'dictionary_class {dictionary_class!r}')
        self.dictionary_class = dictionary_class
        self.version, self.language, self.encoding = version, language, encoding
        self.files, self.llts, self.pts = files, llts, pts
        self.version_source, self.integrity = version_source, integrity
        self._exact = defaultdict(list)
        self._fold = defaultdict(set)
        for llt in llts.values():
            self._exact[llt.name].append(llt.code)
            self._fold[fold_key(llt.name)].add(llt.code)
        self.collisions = {key for key, codes in self._fold.items() if len(codes) > 1}
        self._trigram_index = None
        self.snapshot_sha256 = hashlib.sha256(json.dumps(
            {'version': version, 'language': language, 'encoding': encoding, 'dictionary_class': dictionary_class,
             'files': {name: {'sha256': f['sha256'], 'bytes': f['bytes']} for name, f in sorted(files.items())}},
            sort_keys=True).encode()).hexdigest()

    def snapshot(self):
        current = sum(1 for llt in self.llts.values() if llt.current)
        return {'vocabulary': 'MedDRA', 'dictionary_class': self.dictionary_class, 'version': self.version,
                'language': self.language, 'encoding': self.encoding,
                'version_source': self.version_source, 'files': self.files, 'llt_count': len(self.llts),
                'llt_current': current, 'pt_count': len(self.pts), 'case_collision_keys': len(self.collisions),
                'integrity': self.integrity, 'snapshot_sha256': self.snapshot_sha256}

    def exact(self, name):
        """LLTs whose name is exactly `name` (case-sensitive)."""
        return [self.llts[c] for c in sorted(self._exact.get(name, ()))]

    def folded(self, text):
        """(LLTs whose folded name equals the folded text, whether that key is a case-collision bucket)."""
        key = fold_key(text)
        return [self.llts[c] for c in sorted(self._fold.get(key, ()))], key in self.collisions

    def pt_of(self, llt):
        return self.pts[llt.pt_code]

    def fuzzy(self, text, limit=5, shortlist=200, cutoff=0.6):
        """Recall-only: current LLTs ranked by character similarity. Scores only order candidates."""
        if self._trigram_index is None:
            index = defaultdict(list)
            for llt in self.llts.values():
                if llt.current:
                    for gram in trigrams(llt.name):
                        index[gram].append(llt.code)
            self._trigram_index = index
        query = loose_key(text)
        if not query:
            return []
        overlap = Counter()
        for gram in trigrams(text):
            overlap.update(self._trigram_index.get(gram, ()))
        scored = []
        for code, _ in sorted(overlap.items(), key=lambda kv: (-kv[1], kv[0]))[:shortlist]:
            score = difflib.SequenceMatcher(None, query, loose_key(self.llts[code].name)).ratio()
            if score >= cutoff:
                scored.append((round(score, 4), code))
        scored.sort(key=lambda sc: (-sc[0], sc[1]))
        return [(self.llts[code], score) for score, code in scored[:limit]]


def load_meddra(directory, *, encoding, expected_version=None, expected_language='English', pinned_sha256=None):
    """Load and validate one MedDRA ASCII directory.

    Not given / no such directory -> ResourceBlocked (the run is resource_blocked). A directory that is given but
    incomplete or malformed fails with MeddraFormatError. `encoding` must be stated explicitly and decoding must be
    lossless. `pinned_sha256` ({file name: sha256}, from the resource freeze record) marks an authorized formal
    dictionary; a directory holding SYNTHETIC_MARKER is a synthetic test dictionary and can never be pinned.
    """
    if directory is None:
        raise ResourceBlocked('meddra_not_provided', 'no MedDRA ASCII directory was given')
    directory = Path(directory)
    if not directory.is_dir():
        raise ResourceBlocked('meddra_directory_missing', str(directory))
    if not encoding:
        raise MeddraFormatError('the file encoding must be given explicitly (it is recorded with the snapshot)')
    synthetic = _find(directory, SYNTHETIC_MARKER.lower()) is not None
    paths = {name: _find(directory, name) for name in REQUIRED + (RELEASE,)}
    missing = [name for name in REQUIRED + (RELEASE,) if paths[name] is None]
    if missing:
        raise MeddraFormatError(f'incomplete MedDRA ASCII directory, missing: {", ".join(missing)}')
    files, texts = {}, {}
    for name, path in paths.items():
        raw, text = _read(path, encoding)
        files[name] = {'name': path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        texts[name] = text
    lines = [line for line in texts[RELEASE].splitlines() if line.strip()]
    if len(lines) != 1:
        raise MeddraFormatError(f'{RELEASE}: {len(lines)} records, expected exactly one')
    rel = parse_record(lines[0], RELEASE_FIELDS, RELEASE, 1)
    if any(rel[f] for f in RELEASE_FIELDS[2:]):
        raise MeddraFormatError(f'{RELEASE}: the three trailing fields must be empty')
    version, language = rel['version'].strip(), rel['language'].strip()
    if not VERSION_RE.match(version):
        raise MeddraFormatError(f'{RELEASE}: invalid version {version!r}')
    if expected_version is not None and version != expected_version:
        raise MeddraFormatError(f'{RELEASE}: version {version}, expected {expected_version}')
    if not language or language.casefold() != (expected_language or '').casefold():
        raise MeddraFormatError(f'{RELEASE}: language {language!r}, expected {expected_language!r}')
    files[RELEASE]['rows'] = 1
    pts = {}
    for lineno, line in enumerate(texts['pt.asc'].splitlines(), 1):
        if not line.strip():
            continue
        rec = parse_record(line, PT_FIELDS, 'pt.asc', lineno)
        if not CODE.match(rec['pt_code']) or not CODE.match(rec['pt_soc_code']) or not rec['pt_name']:
            raise MeddraFormatError(f'pt.asc:{lineno}: invalid code or empty name')
        if rec['pt_code'] in pts:
            raise MeddraFormatError(f'pt.asc:{lineno}: duplicate pt_code {rec["pt_code"]}')
        pts[rec['pt_code']] = Pt(rec['pt_code'], rec['pt_name'], rec['pt_soc_code'])
    llts = {}
    for lineno, line in enumerate(texts['llt.asc'].splitlines(), 1):
        if not line.strip():
            continue
        rec = parse_record(line, LLT_FIELDS, 'llt.asc', lineno)
        if not CODE.match(rec['llt_code']) or not CODE.match(rec['pt_code']) or not rec['llt_name']:
            raise MeddraFormatError(f'llt.asc:{lineno}: invalid code or empty name')
        if rec['llt_currency'] not in ('Y', 'N'):
            raise MeddraFormatError(f'llt.asc:{lineno}: llt_currency {rec["llt_currency"]!r} is not Y or N')
        if rec['llt_code'] in llts:
            raise MeddraFormatError(f'llt.asc:{lineno}: duplicate llt_code {rec["llt_code"]}')
        if rec['pt_code'] not in pts:
            raise MeddraFormatError(f'llt.asc:{lineno}: pt_code {rec["pt_code"]} not in pt.asc')
        llts[rec['llt_code']] = Llt(rec['llt_code'], rec['llt_name'], rec['pt_code'], rec['llt_currency'] == 'Y')
    if not llts or not pts:
        raise MeddraFormatError(f'empty dictionary: {len(llts)} LLT and {len(pts)} PT records')
    # Every PT is also an LLT with the same code and name, linked to itself and current.
    bad = [code for code, pt in pts.items()
           if code not in llts or llts[code].pt_code != code or llts[code].name != pt.name or not llts[code].current]
    if bad:
        raise MeddraFormatError(f'{len(bad)} PT without a current self-LLT of the same name, e.g. {bad[:3]}')
    files['llt.asc']['rows'], files['pt.asc']['rows'] = len(llts), len(pts)
    if synthetic and pinned_sha256:
        raise MeddraFormatError('a synthetic test dictionary cannot be pinned as a formal dictionary')
    if pinned_sha256:
        observed = {name: files[name]['sha256'] for name in REQUIRED + (RELEASE,)}
        if observed != dict(pinned_sha256):
            raise MeddraFormatError('the dictionary files do not match the pinned sha256 values of the resource freeze')
    dictionary_class = 'synthetic_test' if synthetic else ('formal' if pinned_sha256 else 'unpinned')
    integrity = {'llt_name_exact_duplicates': sum(n - 1 for n in Counter(l.name for l in llts.values()).values() if n > 1)}
    return MeddraDictionary(version=version, language=language, encoding=encoding, files=files, llts=llts, pts=pts,
                            version_source='release_file', integrity=integrity, dictionary_class=dictionary_class)
