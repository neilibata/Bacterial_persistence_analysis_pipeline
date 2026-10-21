"""Cache what a figure needs, so re-formatting it does not re-run the analysis.

Two levels, because there are two costs.

*Reference cache.* Loading the breadth and depth matrices, resolving metadata
and building the depth-matched sets takes about ten minutes and produces the
same objects every time. Every format iteration paid it.

*Figure cache.* Some sections then spend real time on statistics before they
draw anything -- the axis-angle figures run a fresh ordination per depth bin.
A section can hand its computed frames here and read them back next time.

Neither cache is part of a pipeline run: `python -m persistence` recomputes
everything as before. These exist for the format-iteration loop, and a cache
older than the inputs it was built from is ignored rather than used.

    from persistence import plotdata
    data, dm, stats = plotdata.reference(ref)          # cached load
    df = plotdata.figure_frame("axis_angle_bins", ref, build=lambda: ...)
"""

from __future__ import annotations

import functools
import hashlib
import os
import pickle
import time
from typing import Callable

CACHE_DIRNAME = ".plotcache"
# Anything the loader reads. A cache older than any of these is stale.
_INPUT_GLOBS = ("*coverage_matrix*.csv", "*depth_matrix*.csv", "metadata_*.csv")


def _cache_dir(ref: str) -> str:
    d = os.path.join("results", ref, CACHE_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def _newest_input_mtime() -> float:
    import glob
    t = 0.0
    for pat in _INPUT_GLOBS:
        for p in glob.glob(pat):
            try:
                t = max(t, os.path.getmtime(p))
            except OSError:
                continue
    return t


def _fresh(path: str) -> bool:
    """Is this cache newer than every input it could have been built from?"""
    if not os.path.exists(path):
        return False
    return os.path.getmtime(path) >= _newest_input_mtime()


def load(name: str, ref: str):
    """Return a cached object, or None when there is no fresh one."""
    path = os.path.join(_cache_dir(ref), f"{name}.pkl")
    if not _fresh(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"    [cache] {name}: unreadable ({exc}); rebuilding")
        return None


def save(name: str, ref: str, obj) -> str:
    path = os.path.join(_cache_dir(ref), f"{name}.pkl")
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        mb = os.path.getsize(path) / 1e6
        print(f"    [cache] wrote {name} ({mb:.1f} MB)")
    except Exception as exc:
        print(f"    [cache] could not write {name}: {exc}")
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def cached(name: str, ref: str, build: Callable, use_cache: bool = True):
    """Read `name` from the cache, or build it and write it back."""
    if use_cache:
        got = load(name, ref)
        if got is not None:
            print(f"    [cache] {name}: reusing")
            return got
    t0 = time.time()
    obj = build()
    print(f"    [cache] {name}: built in {time.time()-t0:.0f}s")
    save(name, ref, obj)
    return obj


def reference(ref: str, use_cache: bool = True):
    """(data, dm, stats) with the ten-minute load paid at most once."""
    from .stats_report import StatsReport

    def _build():
        from . import config
        from .loading import load_reference
        from .reinstate_secondary import reinstate_distinct_secondaries
        from .depth_matching import build_depth_matched_sets
        s = StatsReport(ref)
        d = load_reference(ref)
        d = reinstate_distinct_secondaries(d, s)
        m = build_depth_matched_sets(
            d, frac_tol=config.DEPTH_MATCH_FRAC_TOL,
            health_frac_tol=config.HEALTH_TRIPLET_FRAC_TOL)
        return d, m

    data, dm = cached(f"reference_{ref}", ref, _build, use_cache)
    # StatsReport accumulates manuscript numbers and must not be reused across
    # runs -- a fresh one each time, so a cached run cannot inherit stale rows.
    return data, dm, StatsReport(ref)


# ── Function-level cache ───────────────────────────────────────────────────
# Off by default. A pipeline run must recompute everything, or the results stop
# being a product of the code that claims to produce them. The format-iteration
# tools turn it on explicitly.
_ENABLED = False


def enable(on: bool = True) -> None:
    """Turn the function cache on. Only the figure tooling should call this."""
    global _ENABLED
    _ENABLED = bool(on)
    print(f"    [cache] function cache {'ON' if _ENABLED else 'off'}")


def enabled() -> bool:
    return _ENABLED


def _ref_of(args, kwargs) -> str:
    """Find the reference name among a function's arguments."""
    for v in list(args) + list(kwargs.values()):
        r = getattr(v, "ref_name", None)
        if isinstance(r, str):
            return r
    for k in ("ref", "ref_name"):
        v = kwargs.get(k)
        if isinstance(v, str):
            return v
    for v in args:
        if isinstance(v, str) and v in ("UHGG", "lev", "chimp", "combined"):
            return v
    return "combined"


def _key_of(args, kwargs) -> str:
    """A short, stable key from the cheap, identifying arguments.

    Big frames are deliberately not hashed: they come from the reference, whose
    own cache is already invalidated by the input files. What varies between
    calls here is a threshold, a label, a spec -- small things that repr well.
    """
    def _fp(v):
        # Frames must be fingerprinted, not skipped. Several callers pass a
        # subset -- the Carter cohort, a depth-matched set -- with the same
        # spec as a full-cohort call, and a key built from the spec alone would
        # hand one the other's matrix. Shape plus the ends of the index is
        # cheap and separates them.
        shape = getattr(v, "shape", None)
        if shape is not None:
            head = tail = ""
            idx = getattr(v, "index", None)
            if idx is not None and len(idx):
                head, tail = str(idx[0]), str(idx[-1])
            cols = getattr(v, "columns", None)
            ncol = len(cols) if cols is not None else ""
            return f"arr{tuple(shape)}:{head}:{tail}:{ncol}"
        for attr in ("short_label", "description", "name", "ref_name"):
            a = getattr(v, attr, None)
            if isinstance(a, (str, int, float)):
                return f"{attr}={a}"
        if isinstance(v, (list, tuple, set)):
            return f"seq{len(v)}"
        return None

    bits = []
    for v in list(args) + [f"{k}={v!r}" for k, v in sorted(kwargs.items())]:
        if isinstance(v, (str, int, float, bool)) or v is None:
            bits.append(repr(v))
        else:
            f = _fp(v)
            if f is not None:
                bits.append(f)
            else:
                # Unrecognised and unhashable: refuse to cache rather than
                # risk returning another call's result.
                return None
    raw = "|".join(bits)
    return hashlib.sha1(raw.encode()).hexdigest()[:12] if raw else "noargs"


def memo(fn=None, *, name: str = None):
    """Cache a computation to disk, keyed on its cheap arguments.

    For the format-iteration loop: the numbers a figure plots do not change
    when its layout does, and recomputing them is what made every pass cost
    twenty minutes. Decorate the function that turns data into plot-ready
    values, not the one that draws.

    Inert unless plotdata.enable() has been called, so a pipeline run is
    unaffected.
    """
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not _ENABLED:
                return f(*args, **kwargs)
            k = _key_of(args, kwargs)
            if k is None:                       # cannot key it safely
                return f(*args, **kwargs)
            ref = _ref_of(args, kwargs)
            cname = f"fn_{name or f.__name__}_{k}"
            return cached(cname, ref, lambda: f(*args, **kwargs), use_cache=True)
        wrapper._uncached = f
        return wrapper
    return deco(fn) if fn is not None else deco
