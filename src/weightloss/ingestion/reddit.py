"""Bounded, resumable Arctic Shift pilot. No clients or data reads at import.

Full pages retain the last timestamp second; query timeouts bisect intervals.
A saturated one-second interval stays unresolved. Completion describes the API
interval traversal, not proof that the archive contains every Reddit record.
"""
import fcntl
import hashlib
import math
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = 'https://arctic-shift.photon-reddit.com'
VERIFICATION_VERSION = 3
FIELDS = {
    'posts': 'id,created_utc,subreddit,retrieved_on,title,selftext,num_comments,score,crosspost_parent',
    'comments': 'id,created_utc,subreddit,retrieved_on,body,link_id,parent_id,score',
}


def normalize_reset_timestamp(value):
    """Absolute reset timestamps may be Unix seconds or Unix milliseconds."""
    return value / 1000 if value >= 100_000_000_000 else value


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def target_spec(config, subreddit):
    if not re.fullmatch(r'[A-Za-z0-9_]+', subreddit):
        raise ValueError('Use a subreddit name without r/ or a URL')
    if subreddit.casefold() in {n.casefold() for n in config.get('excluded_subreddits', [])}:
        raise ValueError('Subreddit was explicitly excluded from this project')
    matches = [names for name, names in config.get('targets', config.get('pilot_targets', {})).items()
               if name.casefold() == subreddit.casefold()]
    if len(matches) != 1:
        raise ValueError('Subreddit is not in the configured source inventory')
    brands = matches[0]
    start = min(config['brands'][brand] for brand in brands)
    return {'subreddit': subreddit.casefold(), 'brands': brands,
            'start': int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())}


class BudgetReached(Exception):
    pass


class QueryTimeout(Exception):
    """Source asks for a smaller query interval."""


class RateLimited(Exception):
    def __init__(self, until):
        self.until = until
        super().__init__('Source requested a cooldown')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ArcticClient:
    def __init__(self, budget=12, interval=1.0, timeout=40, direction='asc', auto_limit=False):
        self.budget, self.interval, self.timeout = budget, interval, timeout
        self.direction = direction
        self.auto_limit = auto_limit
        self.calls = 0
        self.last_call = 0.0
        self.history = []
        self.record_callback = None
        self.before_request = None
        self.on_rate_limit = None
        self.opener = None

    def request(self, path, params):
        url = f'{BASE_URL}/api/{path}?{urlencode(params)}'
        for attempt in range(3):
            if self.calls >= self.budget:
                raise BudgetReached()
            if self.before_request:
                self.before_request()
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_call)))
            self.last_call = time.monotonic()
            self.calls += 1
            record = {'url': url, 'requested_at': utc(time.time())}
            try:
                req = Request(url, headers={'User-Agent': 'WeightLoss-research-collector/0.2',
                                            'Accept': 'application/json'})
                with (self.opener or urlopen)(req, timeout=self.timeout) as response:
                    record['http_status'] = response.status
                    payload = json.load(response)
                rows = payload.get('data')
                if payload.get('error') or not isinstance(rows, list):
                    if 'timed out' in str(payload).lower() or 'timeout' in str(payload).lower():
                        raise QueryTimeout()
                    raise ValueError('API returned an error or invalid data envelope')
                record['rows'] = len(rows)
                return rows
            except HTTPError as error:
                record['http_status'] = error.code
                if error.code in (422, 500, 503, 504):
                    error_body = error.read(4096).decode('utf-8', errors='replace').lower()
                    if 'timed out' in error_body or 'timeout' in error_body:
                        record['error_type'] = 'QueryTimeout'
                        raise QueryTimeout() from error
                if error.code == 429 or (error.code == 503 and error.headers.get('Retry-After')):
                    now = time.time()
                    delays = [now + 60]  # Conservative fallback; persisted across resumes.
                    record['rate_limit_headers'] = {key: error.headers[key] for key in ('Retry-After', 'X-RateLimit-Reset', 'X-RateLimit-Reset-At') if error.headers.get(key)}
                    for key in ('Retry-After', 'X-RateLimit-Reset', 'X-RateLimit-Reset-At'):
                        value = error.headers.get(key)
                        if not value:
                            continue
                        try:
                            number = float(value)
                            if math.isfinite(number) and number >= 0:
                                delays.append(normalize_reset_timestamp(number) if key.endswith('-At') else now + number)
                        except ValueError:
                            if key == 'Retry-After':
                                try:
                                    delays.append(parsedate_to_datetime(value).timestamp())
                                except (ValueError, TypeError):
                                    pass
                    if self.on_rate_limit:
                        self.on_rate_limit(max(delays))
                    raise RateLimited(max(delays)) from error
                if error.code not in (422, 500, 502, 503, 504, 520, 521, 522, 523, 524) or attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
            except (URLError, TimeoutError) as error:
                record['error_type'] = type(error).__name__
                if attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
            finally:
                self.history.append(record)
                if self.record_callback:
                    self.record_callback(record)

    def __call__(self, kind, subreddit, lo, hi, limit):
        # Both bounds are exclusive in live API probes. Keep boundary seconds.
        return self.request(f'{kind}/search', dict(
            subreddit=subreddit, after=lo - 1, before=hi, sort=self.direction,
            limit='auto' if self.auto_limit else limit, fields=FIELDS[kind] if self.direction == 'asc'
            else 'id,created_utc,subreddit'))

    def count(self, kind, subreddit, lo, hi):
        rows = self.request(f'{kind}/search/aggregate', dict(
            subreddit=subreddit, after=lo - 1, before=hi, aggregate='subreddit'))
        if not rows:
            return 0
        if len(rows) != 1 or rows[0].get('key', rows[0].get('subreddit', '')).casefold() != subreddit.casefold():
            raise ValueError('Unexpected aggregate response')
        count = rows[0].get('count')
        if isinstance(count, str) and count.isdigit():
            count = int(count)
        if type(count) is not int or count < 0:
            raise ValueError('Invalid source count')
        return count


class Pilot:
    def __init__(self, directory, spec, end=None, page_size=100, direction='asc'):
        self.directory = Path(directory).expanduser().resolve()
        if not 1 <= page_size <= 100:
            raise ValueError('page_size must be in 1..100')
        if direction not in ('asc', 'desc'):
            raise ValueError('Invalid traversal direction')
        self.direction = direction
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / '.writer.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError('Another collector owns this output directory')
        self.db = sqlite3.connect(self.directory / 'checkpoint.sqlite')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS windows (
                kind TEXT, lo INTEGER, hi INTEGER, status TEXT, error TEXT,
                PRIMARY KEY(kind, lo, hi));
            CREATE TABLE IF NOT EXISTS items (
                kind TEXT, id TEXT, created_utc INTEGER, payload TEXT,
                PRIMARY KEY(kind, id));
            CREATE TABLE IF NOT EXISTS requests (record TEXT);
            CREATE INDEX IF NOT EXISTS chronological ON items(kind,created_utc,id);
            CREATE TABLE IF NOT EXISTS count_windows (
                kind TEXT,lo INTEGER,hi INTEGER,status TEXT,total INTEGER,checked_at TEXT,
                PRIMARY KEY(kind,lo,hi));
        ''')
        existing = self.db.execute("SELECT value FROM metadata WHERE key='spec'").fetchone()
        if existing:
            saved = json.loads(existing[0])
            if any(saved[k] != v for k, v in spec.items()) or saved['page_size'] != page_size or saved.get('direction', 'asc') != direction:
                self.close()
                raise ValueError('Resume configuration differs; use a new output directory')
            if end is not None and end != saved['end']:
                self.close()
                raise ValueError('Snapshot end differs; use a new output directory')
            self.spec = saved
        else:
            self.spec = dict(spec, end=int(time.time()) if end is None else end,
                             page_size=page_size, source=BASE_URL, direction=direction,
                             collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            if self.spec['end'] <= spec['start']:
                self.close()
                raise ValueError('Snapshot must end after FDA start')
            with self.db:
                self.db.execute('INSERT INTO metadata VALUES (?,?)', ('spec', json.dumps(self.spec)))
                for kind in FIELDS:
                    self.db.execute('INSERT INTO windows VALUES (?,?,?,?,?)',
                                    (kind, spec['start'], self.spec['end'], 'pending', None))

        # A transactional revision changes on ANY item mutation, including
        # same-count repairs. Request logs and verification metadata do not
        # invalidate otherwise identical content exports.
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS export_revision (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), generation TEXT, revision INTEGER);
            CREATE TRIGGER IF NOT EXISTS export_items_insert AFTER INSERT ON items BEGIN
                UPDATE export_revision SET revision=revision+1 WHERE singleton=1;
            END;
            CREATE TRIGGER IF NOT EXISTS export_items_update AFTER UPDATE ON items BEGIN
                UPDATE export_revision SET revision=revision+1 WHERE singleton=1;
            END;
            CREATE TRIGGER IF NOT EXISTS export_items_delete AFTER DELETE ON items BEGIN
                UPDATE export_revision SET revision=revision+1 WHERE singleton=1;
            END;
        """)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO export_revision VALUES (1,?,0)', (uuid.uuid4().hex,))
        old_reset = self.get_meta('not_before', 0)
        if old_reset >= 100_000_000_000:
            self.meta('not_before_migrated_from_milliseconds', old_reset)
            self.meta('not_before', normalize_reset_timestamp(old_reset))

    def close(self):
        self.db.close()
        self.lock.close()

    def meta(self, key, value):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, json.dumps(value)))

    def get_meta(self, key, default=None):
        row = self.db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def log_request(self, record):
        with self.db:
            self.db.execute('INSERT INTO requests VALUES (?)', (json.dumps(record),))

    def run(self, fetch, max_windows=12, export=True):
        if max_windows < 1:
            raise ValueError('max_windows must be positive')
        history_start = len(getattr(fetch, 'history', []))
        if self.get_meta('not_before', 0) > time.time():
            return self.export() if export else self.summary()
        if isinstance(fetch, ArcticClient):
            fetch.record_callback = self.log_request
        with self.db:
            self.db.execute("UPDATE windows SET status='pending' WHERE status='failed'")
        try:
            for _ in range(max_windows):
                window = self.db.execute("SELECT kind,lo,hi FROM windows WHERE status='pending' ORDER BY kind DESC,lo LIMIT 1").fetchone()
                if window is None:
                    break
                kind, lo, hi = window
                try:
                    rows = fetch(kind, self.spec['subreddit'], lo, hi, self.spec['page_size'])
                    max_returned = 1000 if getattr(fetch, 'auto_limit', False) else self.spec['page_size']
                    if len(rows) > max_returned:
                        raise ValueError('Response exceeds requested limit')
                    ids = set()
                    for row in rows:
                        ident, stamp = row.get('id'), row.get('created_utc')
                        if not isinstance(ident, str) or not ident or ident in ids:
                            raise ValueError('Missing or repeated ID in one response')
                        if not isinstance(stamp, int) or not lo <= stamp < hi:
                            raise ValueError('Timestamp outside requested half-open interval')
                        if row.get('subreddit', '').casefold() != self.spec['subreddit']:
                            raise ValueError('Response contains a different subreddit')
                        if self.direction == 'asc':
                            required = ('title', 'selftext') if kind == 'posts' else ('body','link_id','parent_id')
                            if any(key not in row for key in required):
                                raise ValueError('Source omitted required content or relationship fields')
                        ids.add(ident)
                    stamps = [r['created_utc'] for r in rows]
                    if stamps != sorted(stamps, reverse=self.direction == 'desc'):
                        raise ValueError('Source did not honor requested time ordering')
                    # Auto returns a server-selected cap in 100..1000. Treat any
                    # page >=100 as potentially truncated, even if it is <1000.
                    saturated = len(rows) >= self.spec['page_size']
                    with self.db:
                        for row in rows:
                            self.db.execute('INSERT OR IGNORE INTO items VALUES (?,?,?,?)',
                                            (kind, row['id'], row['created_utc'], json.dumps(row, ensure_ascii=False)))
                        status = 'complete'
                        if saturated and hi - lo > 1:
                            boundary = rows[-1]['created_utc']
                            # Only the final second can be incomplete. Fetch it again
                            # with the remaining interval, deduplicating by kind + ID.
                            if self.direction == 'asc':
                                mid = boundary if boundary > lo else lo + 1
                                pieces = ((lo, mid, 'complete' if boundary > lo else 'unresolved'),
                                          (mid, hi, 'pending'))
                            else:
                                mid = boundary + 1 if boundary < hi - 1 else hi - 1
                                pieces = ((lo, mid, 'pending'),
                                          (mid, hi, 'complete' if boundary < hi - 1 else 'unresolved'))
                            for a, b, state in pieces:
                                self.db.execute('INSERT OR IGNORE INTO windows VALUES (?,?,?,?,?)',
                                                (kind, a, b, state, None))
                            status = 'split'
                        elif saturated:
                            status = 'unresolved'
                        self.db.execute('UPDATE windows SET status=?,error=NULL WHERE kind=? AND lo=? AND hi=?',
                                        (status, kind, lo, hi))
                except RateLimited as error:
                    self.meta('not_before', error.until)
                    break
                except BudgetReached:
                    break
                except QueryTimeout:
                    with self.db:
                        if hi - lo <= 1:
                            self.db.execute("UPDATE windows SET status='failed',error='QueryTimeout' WHERE kind=? AND lo=? AND hi=?", (kind, lo, hi))
                            break
                        mid = (lo + hi) // 2
                        for a, b in ((lo, mid), (mid, hi)):
                            self.db.execute('INSERT OR IGNORE INTO windows VALUES (?,?,?,?,?)', (kind, a, b, 'pending', None))
                        self.db.execute("UPDATE windows SET status='split' WHERE kind=? AND lo=? AND hi=?", (kind, lo, hi))
                except Exception as error:
                    with self.db:
                        self.db.execute("UPDATE windows SET status='failed',error=? WHERE kind=? AND lo=? AND hi=?",
                                        (type(error).__name__, kind, lo, hi))
                    break
        finally:
            with self.db:
                for row in ([] if isinstance(fetch, ArcticClient) else getattr(fetch, 'history', [])[history_start:]):
                    self.db.execute('INSERT INTO requests VALUES (?)', (json.dumps(row),))
        return self.export() if export else self.summary()

    def summary(self):
        """Read checkpoint progress without rewriting content-addressed exports."""
        windows = dict(self.db.execute('SELECT status,count(*) FROM windows GROUP BY status'))
        done = not any(windows.get(x, 0) for x in ('pending', 'failed', 'unresolved'))
        snapshot = list(self.db.execute('SELECT generation,revision FROM export_revision WHERE singleton=1').fetchone())
        cached = self._cached_export(snapshot)
        if cached is not None:
            return {'status': 'api_traversal_complete' if done else 'partial',
                    'counts': cached['counts'], 'windows': windows}
        counts = {}
        for kind in FIELDS:
            count, earliest, latest = self.db.execute('SELECT count(*),min(created_utc),max(created_utc) FROM items WHERE kind=?', (kind,)).fetchone()
            counts[kind] = {'count': count, 'earliest_utc': utc(earliest) if earliest is not None else None,
                            'latest_utc': utc(latest) if latest is not None else None}
        return {'status': 'api_traversal_complete' if done else 'partial',
                'counts': counts, 'windows': windows}

    @staticmethod
    def _file_stamp(path):
        stat = path.stat()
        return [stat.st_size, stat.st_mtime_ns, stat.st_ino]

    def _cached_export(self, snapshot):
        """Reuse only our same-generation unchanged files; otherwise rebuild."""
        try:
            saved = json.loads((self.directory/'manifest.json').read_text())
            if saved.get('_content_snapshot') != snapshot or saved['spec'] != self.spec:
                return None
            files = []
            for kind in FIELDS:
                digest = saved['files_sha256'][kind+'.jsonl']
                if not re.fullmatch(r'[0-9a-f]{64}', digest):
                    return None
                expected = f'exports/{kind}-{digest}.jsonl'
                if saved['export_files'][kind] != expected:
                    return None
                files.extend([kind+'.jsonl', expected])
            if any(saved['_content_file_stats'][name] != self._file_stamp(self.directory/name) for name in files):
                return None
            for key in ('counts', 'local_month_counts'):
                if key not in saved:
                    return None
            return saved
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def export(self):
        windows = dict(self.db.execute('SELECT status,count(*) FROM windows GROUP BY status'))
        done = not any(windows.get(x, 0) for x in ('pending', 'failed', 'unresolved'))
        snapshot = list(self.db.execute('SELECT generation,revision FROM export_revision WHERE singleton=1').fetchone())
        cached = self._cached_export(snapshot)
        if cached is not None:
            stats, hashes, exports, months = (cached[k] for k in
                ('counts', 'files_sha256', 'export_files', 'local_month_counts'))
        else:
            stats, hashes, exports = {}, {}, {}
            (self.directory / 'exports').mkdir(exist_ok=True)
            for kind in FIELDS:
                count, earliest, latest = self.db.execute('SELECT count(*),min(created_utc),max(created_utc) FROM items WHERE kind=?', (kind,)).fetchone()
                stats[kind] = {'count': count, 'earliest_utc': utc(earliest) if earliest is not None else None,
                               'latest_utc': utc(latest) if latest is not None else None}
                path = self.directory / f'{kind}.jsonl'
                temp = path.with_suffix('.tmp')
                with temp.open('w', encoding='utf-8') as out:
                    for (payload,) in self.db.execute('SELECT payload FROM items WHERE kind=? ORDER BY created_utc,id', (kind,)):
                        out.write(payload + '\n')
                temp.replace(path)
                hashes[path.name] = sha256_file(path)
                # Manifest references immutable content-addressed files. A crash during
                # a later export cannot pair an old manifest with a new data file.
                immutable = self.directory / 'exports' / f'{kind}-{hashes[path.name]}.jsonl'
                if not immutable.exists() or sha256_file(immutable) != hashes[path.name]:
                    import shutil
                    staging = immutable.with_suffix('.tmp')
                    shutil.copyfile(path, staging)
                    staging.replace(immutable)
                exports[kind] = str(immutable.relative_to(self.directory))
            months = [dict(zip(('kind','month_utc','count'), row)) for row in self.db.execute("SELECT kind,strftime('%Y-%m',created_utc,'unixepoch'),count(*) FROM items GROUP BY 1,2 ORDER BY 1,2")]
        manifest = {'schema_version': 1, 'status': 'api_traversal_complete' if done else 'partial',
                    'spec': self.spec, 'start_utc': utc(self.spec['start']),
                    'snapshot_end_exclusive_utc': utc(self.spec['end']), 'exported_at': utc(time.time()),
                    'counts': stats, 'local_month_counts': months, 'windows': windows, 'files_sha256': hashes,
                    '_content_snapshot': snapshot,
                    '_content_file_stats': {name: self._file_stamp(self.directory/name)
                        for name in [*(kind+'.jsonl' for kind in FIELDS), *exports.values()]},
                    'export_files': exports,
                    'requests_total': self.db.execute('SELECT count(*) FROM requests').fetchone()[0],
                    'coverage_note': 'API interval traversal only; archive completeness, deletions and parent coverage require separate audit.',
                    'selected_fields': FIELDS if self.direction == 'asc' else {kind:'id,created_utc,subreddit' for kind in FIELDS}, 'author_fields_collected': False,
                    'not_before_utc': utc(self.get_meta('not_before')) if self.get_meta('not_before') else None,
                    'verification': self.get_meta('verification', {'status': 'not_verified'}),
                    'context_audit': self.get_meta('context_audit', {'status':'not_run'}),
                    'repair_audit': self.get_meta('repair_audit', {'status':'not_run'}),
                    'collector_current_sha256': sha256_file(Path(__file__))}
        path = self.directory / 'manifest.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        temp.replace(path)
        return manifest


def verification_passed(report):
    if report.get('verification_version') != VERIFICATION_VERSION:
        return False
    if report.get('status') == 'verified_against_source':
        return True
    if report.get('status') != 'verified_with_documented_source_absences':
        return False
    evidence = report.get('documented_source_absences', [])
    if not evidence or set(report.get('kinds', {})) != set(FIELDS):
        return False
    if len({(r['kind'], r['id']) for r in evidence}) != len(evidence):
        return False
    for kind, counts in report['kinds'].items():
        absent = sum(r['kind'] == kind for r in evidence)
        if (counts['missing_in_ascending'] or counts['timestamp_mismatches']
                or counts['missing_in_descending'] != absent
                or counts['ascending'] - absent != counts['descending']
                or counts['descending'] != counts['source_count']):
            return False
    return all(r.get('checks') == {'id_lookup': 0, 'ascending_second': 0,
                                  'descending_second': 0, 'second_aggregate': 0}
               and r.get('payload_sha256') and r.get('checked_at') for r in evidence)


def document_source_absences(pilot, reverse, client, report):
    """Explain isolated forward-only records without deleting or inventing IDs.

    Only accept four fresh successful empty probes, and only when all other
    IDs/timestamps and the reverse/source count already agree. Other cases
    stay mismatched. Evidence is never reused to skip the fresh probes.
    """
    kinds = report['kinds']
    extras = sum(v['missing_in_descending'] for v in kinds.values())
    if not sum(v['source_count'] for v in kinds.values()) or not 0 < extras <= 100 or any(v['missing_in_ascending'] or v['timestamp_mismatches']
            or v['descending'] != v['source_count'] for v in kinds.values()):
        return
    pilot.db.execute('ATTACH DATABASE ? AS absence_reverse', (str(reverse.directory/'checkpoint.sqlite'),))
    try:
        candidates = pilot.db.execute('''SELECT f.kind,f.id,f.created_utc,f.payload FROM main.items f
            WHERE NOT EXISTS (SELECT 1 FROM absence_reverse.items r WHERE r.kind=f.kind AND r.id=f.id)
            ORDER BY f.kind,f.id LIMIT 101''').fetchall()
    finally:
        pilot.db.execute('DETACH DATABASE absence_reverse')
    if len(candidates) != extras:
        return
    evidence = []
    for kind, ident, stamp, payload in candidates:
        if not pilot.spec['start'] <= stamp < pilot.spec['end']:
            return
        checks = {}
        checks['id_lookup'] = len(client.request(f'{kind}/ids',
            dict(ids=ident, fields='id,created_utc,subreddit')))
        for direction, label in [('asc', 'ascending_second'), ('desc', 'descending_second')]:
            checks[label] = len(client.request(f'{kind}/search', dict(
                subreddit=pilot.spec['subreddit'], after=stamp-1, before=stamp+1,
                sort=direction, limit=100, fields='id,created_utc,subreddit')))
        checks['second_aggregate'] = client.count(kind, pilot.spec['subreddit'], stamp, stamp+1)
        if any(checks.values()):
            return
        evidence.append(dict(kind=kind, id=ident, created_utc=stamp, checks=checks,
            checked_at=utc(time.time()), payload_sha256=hashlib.sha256(payload.encode()).hexdigest()))
    report['documented_source_absences'] = evidence
    report['status'] = 'verified_with_documented_source_absences'
    report['source_change_note'] = ('Original rows retained unchanged. These isolated records are no longer '
        'returned by current ID lookup, either search direction or same-second aggregate. '
        'All other IDs/timestamps and current reverse/source counts agree; not exact raw-count equality.')


def verify_collection(pilot, client, max_windows, defer_exports=False):
    """Independent descending ID traversal plus exact-bound source counts.

    Same-source agreement is explicitly weaker than independent archive coverage.
    A source mutation or cap never becomes a verified result on counts alone.
    """
    if pilot.summary()['status'] != 'api_traversal_complete':
        return
    spec = {k: pilot.spec[k] for k in ('subreddit', 'start', 'brands') if k in pilot.spec}
    reverse = Pilot(pilot.directory / 'reverse-audit', spec, end=pilot.spec['end'],
                    page_size=pilot.spec['page_size'], direction='desc')
    old_direction = client.direction
    client.direction = 'desc'
    report = {'verification_version': VERIFICATION_VERSION, 'status': 'in_progress', 'checked_at': utc(time.time()),
              'scope': 'same-source IDs and aggregate counts; not all historical Reddit', 'kinds': {}}
    try:
        reverse_result = reverse.run(client, max_windows=max_windows, export=not defer_exports)
        if defer_exports and reverse_result['status'] == 'api_traversal_complete':
            reverse.export()
        if reverse.get_meta('not_before', 0) > time.time():
            pilot.meta('not_before', reverse.get_meta('not_before'))
        if reverse_result['status'] != 'api_traversal_complete':
            report['reverse_status'] = reverse_result['status']
            report['reverse_counts'] = reverse_result['counts']
            return
        client.record_callback = pilot.log_request
        for kind in FIELDS:
            source_count = reconciled_count(pilot, client, kind)
            # SQLite joins keep memory bounded on very large communities.
            pilot.db.execute('ATTACH DATABASE ? AS reverse_db', (str(reverse.directory / 'checkpoint.sqlite'),))
            try:
                forward = pilot.db.execute('SELECT count(*) FROM items WHERE kind=?', (kind,)).fetchone()[0]
                backward = reverse.db.execute('SELECT count(*) FROM items WHERE kind=?', (kind,)).fetchone()[0]
                missing = pilot.db.execute('SELECT count(*) FROM reverse_db.items r WHERE kind=? AND NOT EXISTS (SELECT 1 FROM main.items f WHERE f.kind=r.kind AND f.id=r.id)', (kind,)).fetchone()[0]
                timestamp_mismatches = pilot.db.execute('SELECT count(*) FROM main.items f JOIN reverse_db.items r ON f.kind=r.kind AND f.id=r.id WHERE f.kind=? AND f.created_utc!=r.created_utc', (kind,)).fetchone()[0]
                extra = pilot.db.execute('SELECT count(*) FROM main.items f WHERE kind=? AND NOT EXISTS (SELECT 1 FROM reverse_db.items r WHERE r.kind=f.kind AND r.id=f.id)', (kind,)).fetchone()[0]
            finally:
                pilot.db.execute('DETACH DATABASE reverse_db')
            report['kinds'][kind] = {'ascending': forward, 'descending': backward,
                'source_count': source_count, 'missing_in_ascending': missing, 'missing_in_descending': extra,
                'timestamp_mismatches': timestamp_mismatches,
                'match': forward == backward == source_count and missing == extra == timestamp_mismatches == 0}
        report['status'] = 'verified_against_source' if all(v['match'] for v in report['kinds'].values()) else 'mismatch'
        if report['status'] == 'mismatch':
            document_source_absences(pilot, reverse, client, report)
        if report['status'] == 'verified_against_source' and not sum(v['source_count'] for v in report['kinds'].values()):
            report['status'] = 'empty_source_unconfirmed'
        report['count_query_span'] = dict(zip(('earliest_query_utc','latest_query_utc','completed_intervals'), pilot.db.execute("SELECT min(checked_at),max(checked_at),count(*) FROM count_windows WHERE status='complete'").fetchone()))
        report['relationships'] = relationship_audit(pilot)
    except BudgetReached:
        pass
    except RateLimited as error:
        pilot.meta('not_before', error.until)
    except Exception as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        report['error'] = str(error)[:250]
    finally:
        reverse.close()
        client.direction = old_direction
        client.record_callback = pilot.log_request
        report['finished_at'] = utc(time.time())
        pilot.meta('verification', report)


def reconciled_count(pilot, client, kind):
    """Split timed-out aggregates into persisted exact-bound intervals."""
    with pilot.db:
        pilot.db.execute('INSERT OR IGNORE INTO count_windows VALUES (?,?,?,?,?,?)',
            (kind,pilot.spec['start'],pilot.spec['end'],'pending',None,None))
    while True:
        window=pilot.db.execute("SELECT lo,hi FROM count_windows WHERE kind=? AND status='pending' ORDER BY lo LIMIT 1",(kind,)).fetchone()
        if window is None:
            break
        lo,hi=window
        try:
            total=client.count(kind,pilot.spec['subreddit'],lo,hi)
        except QueryTimeout:
            if hi-lo<=1:
                raise ValueError('Source cannot count an individual second')
            mid=(lo+hi)//2
            with pilot.db:
                for a,b in ((lo,mid),(mid,hi)):
                    pilot.db.execute('INSERT OR IGNORE INTO count_windows VALUES (?,?,?,?,?,?)',(kind,a,b,'pending',None,None))
                pilot.db.execute("UPDATE count_windows SET status='split' WHERE kind=? AND lo=? AND hi=?",(kind,lo,hi))
            continue
        with pilot.db:
            pilot.db.execute("UPDATE count_windows SET status='complete',total=?,checked_at=? WHERE kind=? AND lo=? AND hi=?",(total,utc(time.time()),kind,lo,hi))
    return pilot.db.execute("SELECT sum(total) FROM count_windows WHERE kind=? AND status='complete'",(kind,)).fetchone()[0]


def relationship_audit(pilot):
    counts = {'comments': 0, 'missing_link_posts': 0, 'missing_parent_comments': 0,
              'missing_parent_posts': 0, 'invalid_relationships': 0, 'unavailable_text': 0}
    for (payload,) in pilot.db.execute("SELECT payload FROM items WHERE kind='comments'"):
        row = json.loads(payload)
        counts['comments'] += 1
        if row.get('body') in (None, '', '[deleted]', '[removed]'):
            counts['unavailable_text'] += 1
        for field, expected in (('link_id', 't3_'), ('parent_id', None)):
            value = row.get(field, '')
            if not isinstance(value, str) or not value.startswith(('t1_', 't3_')) or (expected and not value.startswith(expected)):
                counts['invalid_relationships'] += 1
                continue
            kind = 'posts' if value.startswith('t3_') else 'comments'
            found = pilot.db.execute('SELECT 1 FROM items WHERE kind=? AND id=?', (kind, value[3:])).fetchone()
            if not found:
                key = 'missing_link_posts' if field == 'link_id' else 'missing_parent_' + kind
                counts[key] += 1
    counts['note'] = 'Missing parents may predate the FDA cutoff or be absent from the source; they remain explicit context gaps.'
    return counts


def collect_pilot(subreddit, output_dir, max_requests=12):
    from weightloss.settings import get_settings
    if max_requests < 1:
        raise ValueError('max_requests must be positive')
    config = json.loads((get_settings().root / 'configs/reddit.json').read_text())
    pilot = Pilot(output_dir, target_spec(config, subreddit))
    client = ArcticClient(budget=max_requests)
    try:
        pilot.run(client, max_windows=max_requests)
        if pilot.get_meta('not_before', 0) <= time.time() and not verification_passed(pilot.get_meta('verification', {})):
            verify_collection(pilot, client, max_requests)
        if pilot.get_meta('verification', {}).get('status') == 'mismatch' and pilot.get_meta('not_before',0)<=time.time():
            repair_from_reverse(pilot,client)
            if pilot.get_meta('not_before',0)<=time.time():
                verify_collection(pilot,client,max_requests)
        if verification_passed(pilot.get_meta('verification', {})):
            hydrate_context(pilot, client)
        return pilot.export()
    finally:
        pilot.close()


def collect_batch(output_dir, max_requests=100, subreddits=None):
    """One fixed snapshot and global request budget across the source inventory."""
    from weightloss.settings import get_settings
    config=json.loads((get_settings().root/'configs/reddit.json').read_text())
    names=sorted({s.casefold() for s in (subreddits or config['targets'])})
    specs=[target_spec(config,name) for name in names]
    directory=Path(output_dir).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'.batch.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError('Another batch owns this output directory')
    try:
        path=directory/'batch-spec.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['targets']!=specs:
                raise ValueError('Batch scope changed; use a new output directory')
        else:
            saved={'targets':specs,'snapshot_end_exclusive':int(time.time()),
                   'scope_note':'Complete selected communities since earliest associated brand FDA date; medication filtering is downstream.'}
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(saved,indent=2)+'\n');temp.replace(path)
        previous_path=directory/'batch-manifest.json'
        if previous_path.exists():
            previous=json.loads(previous_path.read_text())
            if previous.get('not_before',0)>time.time():
                previous['requests_this_invocation']=0
                return previous
        client=ArcticClient(budget=max_requests)
        not_before=0
        results={}
        for spec in specs:
            pilot=Pilot(directory/spec['subreddit'],spec,end=saved['snapshot_end_exclusive'])
            try:
                if client.calls<max_requests:
                    pilot.run(client,max_windows=max_requests)
                    if pilot.get_meta('not_before',0)<=time.time() and not verification_passed(pilot.get_meta('verification', {})):
                        verify_collection(pilot,client,max_requests)
                if pilot.get_meta('verification',{}).get('status')=='mismatch' and pilot.get_meta('not_before',0)<=time.time():
                    repair_from_reverse(pilot,client)
                    if pilot.get_meta('not_before',0)<=time.time():
                        verify_collection(pilot,client,max_requests)
                if verification_passed(pilot.get_meta('verification', {})):
                    hydrate_context(pilot,client)
                result=pilot.export()
                results[spec['subreddit']]={key:result[key] for key in ('status','counts','verification','not_before_utc')}
                # Cooldowns apply to this host/account, not just one community.
                if pilot.get_meta('not_before',0)>time.time():
                    not_before=pilot.get_meta('not_before')
                    break
            finally:
                pilot.close()
        done=len(results)==len(specs) and all(verification_passed(r['verification']) for r in results.values())
        qualified=any(r['verification']['status']=='verified_with_documented_source_absences' for r in results.values())
        report={'status':('verified_with_documented_source_absences' if qualified else 'verified_against_source') if done else 'partial',
                'snapshot_end_exclusive':saved['snapshot_end_exclusive'],'checked_at':utc(time.time()),
                'not_before':not_before,'requests_this_invocation':client.calls,'targets_total':len(specs),'communities':results}
        temp=directory/'batch-manifest.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(directory/'batch-manifest.json')
        return report
    finally:
        lock.close()


def hydrate_context(pilot, client):
    """Fetch explicit missing parent IDs into a separate context-only table.

    Never expands author histories or changes the FDA-defined cohort. Absence
    from ID lookup is recorded, not called a successfully downloaded record.
    """
    pilot.db.executescript('''
        CREATE TABLE IF NOT EXISTS context (
            kind TEXT,id TEXT,status TEXT,classification TEXT,payload TEXT,
            PRIMARY KEY(kind,id));
    ''')
    def queue(row):
        for field in ('parent_id','link_id'):
            value=row.get(field,'')
            if not isinstance(value,str) or not value.startswith(('t1_','t3_')):
                continue
            kind='posts' if value.startswith('t3_') else 'comments'
            ident=value[3:]
            if not pilot.db.execute('SELECT 1 FROM items WHERE kind=? AND id=?',(kind,ident)).fetchone():
                pilot.db.execute('INSERT OR IGNORE INTO context VALUES (?,?,?,?,?)',(kind,ident,'pending',None,None))
    with pilot.db:
        for (payload,) in pilot.db.execute("SELECT payload FROM items WHERE kind='comments'"):
            queue(json.loads(payload))
    client.record_callback=pilot.log_request
    if pilot.get_meta('not_before',0)>time.time():
        return
    context_error = None
    try:
        while True:
            first=pilot.db.execute("SELECT kind FROM context WHERE status='pending' LIMIT 1").fetchone()
            if not first: break
            kind=first[0]
            ids=[r[0] for r in pilot.db.execute("SELECT id FROM context WHERE kind=? AND status='pending' ORDER BY id LIMIT 100",(kind,))]
            rows=client.request(f'{kind}/ids',dict(ids=','.join(ids),fields=FIELDS[kind]))
            found={}
            for row in rows:
                ident=row.get('id')
                if ident not in ids or ident in found or type(row.get('created_utc')) is not int:
                    raise ValueError('Invalid ID lookup response')
                required=('title','selftext') if kind=='posts' else ('body','link_id','parent_id')
                if any(k not in row for k in required):
                    raise ValueError('ID lookup omitted content or relationships')
                found[ident]=row
            with pilot.db:
                for ident in ids:
                    row=found.get(ident)
                    if row is None:
                        pilot.db.execute("UPDATE context SET status='absent_from_source' WHERE kind=? AND id=?",(kind,ident))
                        continue
                    stamp=row['created_utc']
                    category=('different_community' if row.get('subreddit','').casefold()!=pilot.spec['subreddit']
                              else 'before_window' if stamp<pilot.spec['start']
                              else 'after_window' if stamp>=pilot.spec['end'] else 'in_window_gap')
                    pilot.db.execute("UPDATE context SET status='retrieved',classification=?,payload=? WHERE kind=? AND id=?",(category,json.dumps(row,ensure_ascii=False),kind,ident))
                    if kind=='comments': queue(row)
    except BudgetReached:
        pass
    except RateLimited as error:
        pilot.meta('not_before',error.until)
    except Exception as error:
        context_error = type(error).__name__
    finally:
        counts=[dict(zip(('status','classification','count'),r)) for r in pilot.db.execute('SELECT status,classification,count(*) FROM context GROUP BY status,classification')]
        if any(r['classification']=='in_window_gap' for r in counts):
            verification=pilot.get_meta('verification',{})
            verification['status']='source_search_gap'
            pilot.meta('verification',verification)
        pilot.meta('context_audit',{'status':'partial' if context_error or any(r['status']!='retrieved' for r in counts) else 'complete', 'error_type':context_error, 'checked_at':utc(time.time()),'counts':counts,
            'note':'Context stored separately; absent_from_source remains a gap. in_window_gap requires cohort reconciliation.'})


def repair_from_reverse(pilot, client):
    """Hydrate IDs found in reverse search but missed by forward search.

    Absent lookup results remain explicit. This never removes old records to
    force counts to match and never converts a failed lookup into completion.
    """
    reverse_path=pilot.directory/'reverse-audit'/'checkpoint.sqlite'
    if not reverse_path.exists(): return
    pilot.db.executescript('''CREATE TABLE IF NOT EXISTS repairs (
        kind TEXT,id TEXT,status TEXT,checked_at TEXT,PRIMARY KEY(kind,id));''')
    pilot.db.execute('ATTACH DATABASE ? AS repair_source',(str(reverse_path),))
    client.record_callback=pilot.log_request
    repair_error = None
    try:
        for kind in FIELDS:
            while True:
                candidates=pilot.db.execute('''SELECT r.id,r.created_utc FROM repair_source.items r
                    WHERE r.kind=? AND NOT EXISTS (SELECT 1 FROM main.items f WHERE f.kind=r.kind AND f.id=r.id)
                    AND NOT EXISTS (SELECT 1 FROM repairs a WHERE a.kind=r.kind AND a.id=r.id)
                    ORDER BY r.id LIMIT 100''',(kind,)).fetchall()
                if not candidates: break
                expected=dict(candidates)
                rows=client.request(f'{kind}/ids',dict(ids=','.join(expected),fields=FIELDS[kind]))
                found={}
                for row in rows:
                    ident=row.get('id')
                    if ident not in expected or ident in found:
                        raise ValueError('Unexpected or duplicate repair ID')
                    if row.get('created_utc')!=expected[ident] or row.get('subreddit','').casefold()!=pilot.spec['subreddit']:
                        raise ValueError('Repair response does not match expected source record')
                    required=('title','selftext') if kind=='posts' else ('body','link_id','parent_id')
                    if any(k not in row for k in required):
                        raise ValueError('Repair omitted required fields')
                    found[ident]=row
                with pilot.db:
                    for ident in expected:
                        row=found.get(ident)
                        if row:
                            pilot.db.execute('INSERT OR IGNORE INTO items VALUES (?,?,?,?)',(kind,ident,row['created_utc'],json.dumps(row,ensure_ascii=False)))
                        pilot.db.execute('INSERT INTO repairs VALUES (?,?,?,?)',(kind,ident,'retrieved' if row else 'absent_from_source',utc(time.time())))
        with pilot.db:
            pilot.db.execute('DELETE FROM count_windows')
    except BudgetReached:
        pass
    except RateLimited as error:
        pilot.meta('not_before',error.until)
    except Exception as error:
        repair_error = type(error).__name__
    finally:
        pilot.db.execute('DETACH DATABASE repair_source')
        pilot.meta('repair_audit',{'error_type':repair_error, 'checked_at':utc(time.time()),'counts':[dict(zip(('kind','status','count'),r)) for r in pilot.db.execute('SELECT kind,status,count(*) FROM repairs GROUP BY kind,status')]})
