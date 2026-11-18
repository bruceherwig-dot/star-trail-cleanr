"""One standard place to record what ran, how long it took, and how often it fired.

WHY THIS EXISTS. Speed work kept finding time that no report showed: whole-frame
sweeps per detection, a mask merge, a pair-by-pair angle search. Each time it was
found with a throwaway profiler and then lost. Bruce's rule (2026-11-18): we
measure, and the measurement stays in the program. So every step that can take
real time records three things here:

  * WHEN it ran   (calls)
  * WHAT it did   (counts: "peeled", "kept_whole", "fell_back_to_whole_frame" ...)
  * HOW LONG it took (seconds)

and the engine prints them in the end-of-batch timing table, writes them into the
run log, and works out how much of each parent stage is still unaccounted for.

NAMES are "group.step" (for example "fit.peel_split"). Only two-part names count
toward a group's "unaccounted" line, so a nested detail ("fit.split_crossing.angles")
can be recorded without being counted twice.

TIMERS ONLY OBSERVE. They never change what the program computes. Cost is about a
microsecond a call, so they belong around steps, never around per-pixel work.
Thread safe (the stuck-pixel step runs several frames at once).
"""
import threading
import time

_LOCK = threading.Lock()
_STEPS = {}      # name -> [calls, seconds]
_COUNTS = {}     # name -> {key: n}


class _Step:
    """`with step("fit.props") as s:` times the block; s.count("kept") notes what it did."""
    __slots__ = ("name", "_t0")

    def __init__(self, name):
        self.name = name
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def count(self, key, n=1):
        count(self.name, key, n)

    def __exit__(self, *exc):
        add(self.name, time.perf_counter() - self._t0)


def step(name):
    """Context manager that records one call of `name` and its seconds."""
    return _Step(name)


def add(name, seconds, calls=1):
    """Record time already measured by hand (for code that keeps its own stopwatch)."""
    with _LOCK:
        rec = _STEPS.get(name)
        if rec is None:
            _STEPS[name] = [calls, float(seconds)]
        else:
            rec[0] += calls
            rec[1] += float(seconds)


def count(name, key, n=1):
    """Note that something fired: `count("fit.peel_split", "peeled")`."""
    with _LOCK:
        d = _COUNTS.setdefault(name, {})
        d[key] = d.get(key, 0) + n


def reset():
    with _LOCK:
        _STEPS.clear()
        _COUNTS.clear()


def snapshot():
    """A copy of everything recorded so far: {"steps": {name: [calls, seconds]}, "counts": {name: {key: n}}}."""
    with _LOCK:
        return {"steps": {k: list(v) for k, v in _STEPS.items()},
                "counts": {k: dict(v) for k, v in _COUNTS.items()}}


def group_of(name):
    """"fit.peel_split" -> "fit"; the part before the first dot."""
    return name.split(".", 1)[0]


def is_top_level(name):
    """True for two-part names ("fit.peel_split"), the ones summed against a parent."""
    return name.count(".") == 1


def unaccounted(parent_seconds, snap, group):
    """(seconds, percent) of a parent stage not explained by the group's top-level steps.

    Returns None when the group recorded nothing (so no line is printed for a stage that
    never ran)."""
    kids = [v[1] for k, v in snap["steps"].items()
            if group_of(k) == group and is_top_level(k)]
    if not kids or parent_seconds <= 0:
        return None
    gap = parent_seconds - sum(kids)
    return gap, 100.0 * gap / parent_seconds
