"""Add covering summary indexes to an idle derived cleaning DB; no row edits."""
import argparse
import fcntl
import json
from pathlib import Path
import sqlite3

INDEXES = [
    'CREATE INDEX IF NOT EXISTS records_text_summary ON records(text_hash,text_length)',
    'CREATE INDEX IF NOT EXISTS records_input_summary ON records(input_key,decision)',
    'CREATE INDEX IF NOT EXISTS records_disposition_summary ON records(decision,reason)',
]


def optimize(directory):
    directory = Path(directory).resolve()
    spec = json.loads((directory/'run-spec.json').read_text())
    if 'cleaner_sha256' not in spec or not (directory/'cleaned.sqlite').is_file():
        raise ValueError('Not an initialized derived cleaning run')
    with (directory/'.cleaning.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('Cleaner is active; wait before indexing')
        with sqlite3.connect(directory/'cleaned.sqlite') as db:
            before = db.execute('SELECT count(*) FROM records').fetchone()[0]
            for sql in INDEXES: db.execute(sql)
            after = db.execute('SELECT count(*) FROM records').fetchone()[0]
            if before != after: raise ValueError('Unexpected row-count change')
        return {'records_before':before,'records_after':after,'indexes':INDEXES,
                'operation':'Covering indexes only; no text, disposition, checkpoint or run fingerprint changed.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True)
    print(json.dumps(optimize(parser.parse_args().output_dir),indent=2))
