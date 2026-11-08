# [PATCH9] How long every stage and every step of a run takes, so that the time of later runs can be estimated from
# earlier ones (with the number of photos, cameras, GCPs and the options that drive it).
#
# A step is every program ODX runs (system.run: OpenSfM's detect_features, match_features, reconstruct, OpenMVS, PDAL,
# texrecon, GDAL ...) and our own Python steps (PATCH7, PATCH8). The figures go to odm_report/stats.json as
# "timing_statistics" (merged with those of an earlier run of the task: a restart from a later stage keeps the stages it
# does not repeat) and, per program, as startTime/endTime/seconds into log.json's process list.

import datetime
import os
import threading
import time
from contextlib import contextmanager

VERSION = 1


def step_name(cmd):
    """A short name for a command: the program, and for OpenSfM its sub-command (opensfm reconstruct)."""
    parts = [p.strip('"\'') for p in str(cmd).replace('&&', ' ').split() if p.strip('"\'')]
    if not parts:
        return '?'
    # skip environment exports and the interpreter
    i = 0
    while i < len(parts) and (parts[i] in ('export', 'env') or '=' in parts[i]):
        i += 1
    prog = os.path.basename(parts[i]) if i < len(parts) else os.path.basename(parts[0])
    rest = parts[i + 1:]
    if prog.startswith('python') and rest:
        prog = os.path.basename(rest[0])
        rest = rest[1:]
    if prog == 'opensfm' and rest:
        return 'opensfm ' + rest[0]
    if prog == 'pdal' and rest:
        return 'pdal ' + rest[0]
    return prog


def hms(seconds):
    seconds = int(round(seconds or 0))
    h, m = seconds // 3600, (seconds % 3600) // 60
    if h:
        return '%d h %02d min' % (h, m)
    if m:
        return '%d min %02d s' % (m, seconds % 60)
    return '%d s' % seconds


class RunTiming:
    def __init__(self):
        self._lock = threading.Lock()
        self._stages = {}
        self._order = []
        self._current = None

    def stage_started(self, name):
        with self._lock:
            self._stages[name] = {'name': name, 'start': datetime.datetime.now().isoformat(), '_t0': time.monotonic(),
                                  'steps': []}
            if name not in self._order:
                self._order.append(name)
            self._current = name

    def stage_finished(self, name, ok=True):
        with self._lock:
            s = self._stages.get(name)
            if s is None:
                return None
            s['end'] = datetime.datetime.now().isoformat()
            s['seconds'] = round(time.monotonic() - s.pop('_t0', time.monotonic()), 2)
            s['completed'] = bool(ok)
            if self._current == name:
                self._current = None
            return s['seconds']

    def add_step(self, name, start, seconds, exit_code=None, kind='program'):
        """A step of the current stage. start: datetime."""
        with self._lock:
            entry = {'name': name, 'kind': kind, 'start': start.isoformat(), 'seconds': round(seconds, 2)}
            if exit_code is not None:
                entry['exit_code'] = exit_code
            if self._current is not None:
                self._stages[self._current]['steps'].append(entry)

    def add_detail(self, key, value):
        """Extra figures of the current stage under 'details' (the parts of OpenSfM's reconstruction)."""
        with self._lock:
            if self._current is not None:
                self._stages[self._current].setdefault('details', {})[key] = value

    @contextmanager
    def measure(self, name):
        """Time one of our own steps (kind 'patch')."""
        start, t0 = datetime.datetime.now(), time.monotonic()
        try:
            yield
        finally:
            self.add_step(name, start, time.monotonic() - t0, kind='patch')

    def statistics(self, previous=None):
        """timing_statistics for stats.json. previous: the timing_statistics already there (an earlier run of this task);
        its stages are kept unless this run repeated them, then the new figures replace them."""
        with self._lock:
            stages, order = {}, []
            if previous and isinstance(previous.get('stages'), list):
                for s in previous['stages']:
                    if isinstance(s, dict) and s.get('name'):
                        stages[s['name']] = dict(s)
                        order.append(s['name'])
            repeated = False
            for name in self._order:
                mine = {k: v for k, v in self._stages[name].items() if not k.startswith('_')}
                if 'seconds' not in mine:
                    # the running stage (odm_report itself): what it took so far
                    t0 = self._stages[name].get('_t0')
                    mine['seconds'] = round(time.monotonic() - t0, 2) if t0 else None
                    mine['completed'] = False
                if name in stages:
                    repeated = True
                else:
                    order.append(name)
                stages[name] = mine
            listed = [stages[n] for n in order]
            if not listed:
                return previous
            return {
                'version': VERSION,
                'stages': listed,
                'total_seconds': round(sum(s.get('seconds') or 0 for s in listed), 2),
                'merged_with_earlier_run': bool(previous) and (repeated or len(order) > len(self._order)),
            }


timing = RunTiming()
