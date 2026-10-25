"""Explicit, resumable removal of user-excluded production communities."""
from contextlib import ExitStack
import copy
import fcntl
import json
from pathlib import Path
import re
import shutil
import time

from .reddit import target_spec, utc
from .reddit_production import write_json


def remove_communities(output_dir, names, reason, *, retained=None):
    """Delete selected community folders while preserving retained checkpoints.

    Requires a stopped worker and stop marker. A write-ahead metadata journal
    makes interrupted configuration/state updates and deletions repeatable.
    The caller names either the removals or the exact retained inventory.
    Resolve a retained inventory under both locks, including on journal resume.
    """
    from weightloss.settings import get_settings
    directory = Path(output_dir).expanduser().resolve()
    if (names is None) == (retained is None):
        raise ValueError('Specify either removals or retained communities, not both')
    selected = sorted(set(retained if retained is not None else names))
    if not selected or not reason.strip() or any(not re.fullmatch(r'[a-z0-9_]+', n) for n in selected):
        raise ValueError('Explicit lowercase community names and reason required')
    names = selected if retained is None else None
    if not (directory/'.worker-stop').exists():
        raise ValueError('Stop the worker and create .worker-stop before removing communities')
    config_path = get_settings().root/'configs/reddit.json'
    pending = directory/'scope-removal-pending.json'
    with ExitStack() as stack:
        for name in ('.worker.lock', '.production.lock'):
            lock = stack.enter_context((directory/name).open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('Collector still active; no scope changes made') from error
        if retained is not None:
            # The original scope is needed if an earlier attempt already wrote
            # the narrowed config but was interrupted while deleting files.
            original = (json.loads(pending.read_text())['before']['config']
                        if pending.exists() else json.loads(config_path.read_text()))
            if not set(selected) < set(original['targets']):
                raise ValueError('Retained inventory must be a nonempty proper subset of the original scope')
            names = sorted(set(original['targets']) - set(selected))
        if pending.exists():
            journal = json.loads(pending.read_text())
            if journal['removed'] != names or journal['reason'] != reason:
                raise ValueError('Resume the existing removal before starting another')
            for path, key in ((config_path, 'config'), (directory/'production-spec.json', 'spec'),
                              (directory/'production-state.json', 'state')):
                current = json.loads(path.read_text())
                if current not in (journal['before'][key], journal['after'][key]):
                    raise ValueError('Scope files changed during pending removal')
        else:
            config = json.loads(config_path.read_text())
            spec = json.loads((directory/'production-spec.json').read_text())
            state = json.loads((directory/'production-state.json').read_text())
            expected = {n: target_spec(config, n) for n in sorted(config['targets'])}
            if expected != spec['targets'] or not set(names) < set(expected):
                raise ValueError('Removal must be a proper subset of the matching pinned scope')
            before = dict(config=config, spec=spec, state=state)
            after = copy.deepcopy(before)
            for key in ('config', 'spec'):
                for name in names:
                    del after[key]['targets'][name]
            after['config']['molecule_only_targets'] = [
                n for n in config.get('molecule_only_targets', []) if n not in names]
            after['config']['excluded_subreddits'] = sorted(set(config.get('excluded_subreddits', [])) | set(names))
            old_order = spec['order']
            order = [n for n in old_order if n not in names]
            cursor = state['cursor'] % len(old_order)
            next_name = next(n for n in old_order[cursor:]+old_order[:cursor] if n not in names)
            after['spec']['order'] = order
            after['spec']['scope_revision'] = spec.get('scope_revision', 1)+1
            new_state = after['state']
            new_state['communities'] = {n: v for n, v in state['communities'].items() if n in order}
            new_state.update(cursor=order.index(next_name), targets_total=len(order),
                             active_subreddit=None, active_subreddits=[], status='partial')
            new_state['rows_collected'] = sum(sum(v['count'] for v in c.get('counts', {}).values())
                                              for c in new_state['communities'].values())
            new_state['communities_verified'] = sum(c.get('source_done', False) for c in new_state['communities'].values())
            journal = dict(removed=names, reason=reason, created_at=utc(time.time()), before=before, after=after)
            write_json(pending, journal)
        for name in names:
            if (directory/name).is_symlink():
                raise ValueError('Refusing a symlink community directory')
        write_json(config_path, journal['after']['config'])
        write_json(directory/'production-spec.json', journal['after']['spec'])
        write_json(directory/'production-state.json', journal['after']['state'])
        for name in names:
            if (directory/name).exists():
                shutil.rmtree(directory/name)
        history = directory/'scope-history'
        history.mkdir(exist_ok=True)
        revision = journal['after']['spec']['scope_revision']
        write_json(history/f'removal-{revision}.json', dict(journal, completed_at=utc(time.time())))
        pending.unlink()
        return {'removed': names, 'targets_total': len(journal['after']['spec']['targets']),
                'scope_revision': revision, 'status': 'removed',
                'note': 'Stop marker retained; verify retained checkpoints before restarting.'}
