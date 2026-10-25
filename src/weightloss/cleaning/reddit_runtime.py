"""Shared bounded execution, checkpoint, lock and progress infrastructure."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing
import shutil


def chunks(rows, size=250):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


@contextmanager
def worker_pool(workers, initializer, initargs):
    if not 1 <= workers <= 4:
        raise ValueError('Expected 1–4 workers')
    with ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=initializer, initargs=initargs) as pool:
        yield pool


def ordered_compute(pool, function, rows):
    return [item for batch in pool.map(function, chunks(rows)) for item in batch]


def guard_disk(directory, reserve_gib):
    if shutil.disk_usage(directory).free < reserve_gib * 2**30:
        raise ValueError('Disk reserve reached; checkpoint retained')


def checkpoint(db, key, offset, lines):
    # Caller holds the write transaction containing the corresponding records.
    db.execute('UPDATE inputs SET offset=?,lines=? WHERE key=?', (offset, lines, key))


def lock_file(handle, mode=fcntl.LOCK_EX):
    try:
        fcntl.flock(handle, mode | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise ValueError('Another cleaner owns this stage; checkpoint retained') from error


def progress(stage, event, **fields):
    print(json.dumps(dict(stage=stage, event=event,
        at=datetime.now(timezone.utc).isoformat(), **fields)), flush=True)
