import time
import atexit
import threading
from collections import defaultdict
from functools import wraps

_lock = threading.Lock()
_stats = defaultdict(lambda: {"count": 0, "total": 0.0, "min": float("inf"), "max": 0.0})
_program_start = time.perf_counter()

def _record(label, dt):
    with _lock:
        s = _stats[label]
        s["count"] += 1
        s["total"] += dt
        s["min"] = dt if dt < s["min"] else s["min"]
        s["max"] = dt if dt > s["max"] else s["max"]

class section:
    def __init__(self, label):
        self.label = label
        self._t0 = None
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self._t0
        _record(self.label, dt)
        # 例外はそのまま伝播
        return False

def timeit(label=None):
    def deco(func):
        # メソッドやネスト関数まで分かりやすくしたいなら qualname を使う
        default_name = f"{func.__module__}.{func.__qualname__}"
        name = label or default_name
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                _record(name, time.perf_counter() - t0)
        return wrapper
    return deco

def report(limit=None):
    total_measured = sum(s["total"] for s in _stats.values())
    runtime = time.perf_counter() - _program_start
    rows = sorted(_stats.items(), key=lambda kv: kv[1]["total"], reverse=True)
    if limit is not None:
        rows = rows[:limit]
    print("=== Timing summary ===")
    print(f"Program runtime: {runtime:.3f}s, Measured total: {total_measured:.3f}s")
    for label, s in rows:
        avg = s["total"] / s["count"] if s["count"] else 0.0
        pct = (s["total"] / total_measured * 100.0) if total_measured > 0 else 0.0
        print(f"- {label}: total={s['total']:.3f}s  count={s['count']}  avg={avg*1000:.1f}ms  min={s['min']*1000:.1f}ms  max={s['max']*1000:.1f}ms  [{pct:.1f}%]")

@atexit.register
def _auto_report():
    if _stats:
        report()
