# [PATCH5] How much memory a run takes: the peak RAM, swap and RAM + swap of the whole run and of every stage, written to
# odm_report/stats.json as "memory_statistics" and printed after every stage (so a run stopped by the out-of-memory killer
# still leaves its figures in the console).
#
# Measured from the memory cgroup of the processing node's container (cgroup v2, or v1), so it covers every program the
# run starts. RAM is the memory the programs hold themselves (anonymous memory), not the file cache, which Linux gives back
# whenever it is needed; swap is what was moved to disk. Outside a container (no memory cgroup of our own) the figures are
# those of the whole computer, and say so. Sampled every few seconds: a peak shorter than that can be missed.

import datetime
import os
import threading

SAMPLE_SECONDS = 2.0
CGROUP = '/sys/fs/cgroup'
MEMINFO = '/proc/meminfo'
_NO_LIMIT = 1 << 60


def _read(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception:
        return None


def _read_int(path):
    text = _read(path)
    if text is None:
        return None
    text = text.strip()
    if text == 'max':
        return None
    try:
        value = int(text)
        return None if value >= _NO_LIMIT else value
    except ValueError:
        return None


def _stat(path):
    values = {}
    text = _read(path)
    if text:
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    values[parts[0]] = int(parts[1])
                except ValueError:
                    pass
    return values


class _Source:
    """Reads (ram, swap) in bytes, and the limits."""

    def __init__(self):
        self.kind = None
        self.ram_limit = None
        self.swap_limit = None
        if os.path.exists(CGROUP + '/memory.current') and os.path.exists(CGROUP + '/memory.stat'):
            self.kind = 'cgroup v2 (the processing node\'s container)'
            self.ram_limit = _read_int(CGROUP + '/memory.max')
            self.swap_limit = _read_int(CGROUP + '/memory.swap.max')
        elif os.path.exists(CGROUP + '/memory/memory.usage_in_bytes'):
            self.kind = 'cgroup v1 (the processing node\'s container)'
            self.ram_limit = _read_int(CGROUP + '/memory/memory.limit_in_bytes')
            memsw = _read_int(CGROUP + '/memory/memory.memsw.limit_in_bytes')
            if memsw is not None and self.ram_limit is not None:
                self.swap_limit = max(0, memsw - self.ram_limit)
        elif os.path.exists(MEMINFO):
            self.kind = 'the whole computer (/proc/meminfo; no memory cgroup of its own)'
            info = self._meminfo()
            self.ram_limit = info.get('MemTotal')
            self.swap_limit = info.get('SwapTotal')

    @staticmethod
    def _meminfo():
        values = {}
        text = _read(MEMINFO) or ''
        for line in text.splitlines():
            parts = line.replace(':', ' ').split()
            if len(parts) >= 2:
                try:
                    values[parts[0]] = int(parts[1]) * 1024
                except ValueError:
                    pass
        return values

    def sample(self):
        if self.kind is None:
            return None
        if self.kind.startswith('cgroup v2'):
            ram = _stat(CGROUP + '/memory.stat').get('anon')
            swap = _read_int(CGROUP + '/memory.swap.current') or 0
            return None if ram is None else (ram, swap)
        if self.kind.startswith('cgroup v1'):
            stat = _stat(CGROUP + '/memory/memory.stat')
            ram = stat.get('total_rss', stat.get('rss'))
            swap = stat.get('total_swap', stat.get('swap', 0))
            return None if ram is None else (ram, swap)
        info = self._meminfo()
        if 'MemTotal' not in info or 'MemAvailable' not in info:
            return None
        return (info['MemTotal'] - info['MemAvailable'], info.get('SwapTotal', 0) - info.get('SwapFree', 0))


def gib(value):
    return "%.1f GiB" % (value / 1024.0 ** 3) if value is not None else "no limit"


class MemoryMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._source = None
        self._thread = None
        self._stop = threading.Event()
        self._stage = None
        self._stages = {}  # name -> {'peak_ram_bytes', 'peak_swap_bytes', 'peak_total_bytes'}
        self._order = []

    @property
    def active(self):
        return self._source is not None and self._source.kind is not None

    def start(self):
        if self._thread is not None:
            return
        try:
            self._source = _Source()
        except Exception:
            self._source = None
        if not self.active:
            return
        self._thread = threading.Thread(target=self._run, name='memory-monitor', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            self._take_sample()
            self._stop.wait(SAMPLE_SECONDS)

    def _take_sample(self):
        try:
            values = self._source.sample()
        except Exception:
            values = None
        if values is None:
            return
        ram, swap = values
        with self._lock:
            if self._stage is None:
                return
            entry = self._stages[self._stage]
            entry['peak_ram_bytes'] = max(entry['peak_ram_bytes'], ram)
            entry['peak_swap_bytes'] = max(entry['peak_swap_bytes'], swap)
            entry['peak_total_bytes'] = max(entry['peak_total_bytes'], ram + swap)

    def stage_started(self, name):
        if not self.active:
            return
        with self._lock:
            self._stage = name
            if name not in self._stages:
                self._stages[name] = {'peak_ram_bytes': 0, 'peak_swap_bytes': 0, 'peak_total_bytes': 0,
                                      'measured_at': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
                self._order.append(name)
        self._take_sample()

    def stage_finished(self, name):
        """The [PATCH5] console line for the stage."""
        if not self.active:
            return None
        self._take_sample()
        with self._lock:
            entry = self._stages.get(name)
            if entry is None:
                return None
            run_peak = max(e['peak_total_bytes'] for e in self._stages.values())
            return "[PATCH5] Memory in the %s stage: peak %s RAM, %s swap, %s together (this run so far: %s; limit %s RAM, %s swap)" % (
                name, gib(entry['peak_ram_bytes']), gib(entry['peak_swap_bytes']), gib(entry['peak_total_bytes']),
                gib(run_peak), gib(self._source.ram_limit), gib(self._source.swap_limit))

    def statistics(self, previous=None):
        """memory_statistics for stats.json. previous: the memory_statistics already in stats.json (from an earlier run of this
        task): its stages are kept, and a stage measured again keeps the higher figure, so a restart from a later stage does not
        replace the figures of the whole run with those of the few stages it ran."""
        if not self.active:
            return previous
        self._take_sample()
        with self._lock:
            stages = {}
            order = []
            if previous and isinstance(previous.get('stages'), list):
                for s in previous['stages']:
                    if isinstance(s, dict) and s.get('name'):
                        stages[s['name']] = dict(s)
                        order.append(s['name'])
            for name in self._order:
                mine = self._stages[name]
                if name in stages:
                    old = stages[name]
                    for key in ('peak_ram_bytes', 'peak_swap_bytes', 'peak_total_bytes'):
                        old[key] = max(int(old.get(key) or 0), mine[key])
                    old['measured_at'] = mine['measured_at']
                else:
                    stages[name] = dict(mine, name=name)
                    order.append(name)
            listed = [dict(stages[n], name=n) for n in order]
            if not listed:
                return previous
            top = max(listed, key=lambda s: s.get('peak_total_bytes') or 0)
            return {
                'source': self._source.kind,
                'sample_interval_seconds': SAMPLE_SECONDS,
                'peak_ram_bytes': max(int(s.get('peak_ram_bytes') or 0) for s in listed),
                'peak_swap_bytes': max(int(s.get('peak_swap_bytes') or 0) for s in listed),
                'peak_total_bytes': int(top.get('peak_total_bytes') or 0),
                'peak_total_stage': top['name'],
                'ram_limit_bytes': self._source.ram_limit,
                'swap_limit_bytes': self._source.swap_limit,
                'stages': listed,
            }


monitor = MemoryMonitor()
