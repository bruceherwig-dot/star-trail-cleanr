"""The run timing block for the Star Log: where a whole run's time went, in plain English.

WHY. A run is cut into batches and each batch prints its own timing table, but the saved Star Log
keeps only the first 50 and last 100 lines of the run window. Sean Parker's 572-frame run kept the
table for 1 batch of 29. So a user's log could not say why a run was slow. This module adds the
batches' numbers together and writes one short block into the Star Log's Run Summary section, which
is never cut.

HOW. The engine prints one BATCH_STEP_DETAIL line per batch (modules/step_timer.py numbers). The app
calls merge() for each line and render() once at the end. Both are plain functions with no window
code, so the tests can feed them made-up and real numbers.

PRIVACY. Only step names, seconds, counts and the machine's core count, memory and device kind. No
file names, folder names or pictures.
"""
from . import step_guide

# (timing key, parent group) used to say how much of a stage its recorded steps explain.
_GAP_PARENTS = (("phantom", "dp_prune_phantoms_s"), ("fit", "dp_fit_polygons_s"),
                ("hotpix", "hot_pixel_s"), ("write", "write_s"))


def new_accumulator():
    return {"batches": 0, "batch_seconds": 0.0, "timing": {}, "counts": {}}


def merge(acc, detail):
    """Add one batch's BATCH_STEP_DETAIL (a dict) into the run's totals. Never raises."""
    try:
        acc["batches"] += 1
        acc["batch_seconds"] += float(detail.get("batch_seconds", 0.0))
        for key, (calls, secs) in (detail.get("timing") or {}).items():
            cur = acc["timing"].setdefault(key, [0, 0.0])
            cur[0] += int(calls)
            cur[1] += float(secs)
        for name, kv in (detail.get("counts") or {}).items():
            cur = acc["counts"].setdefault(name, {})
            for k, v in kv.items():
                cur[k] = cur.get(k, 0) + int(v)
    except Exception:
        pass
    return acc


def _hms(sec):
    if 0 < sec < 10:
        return f"{sec:.1f}s"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _secs(acc, key):
    return acc["timing"].get(key, [0, 0.0])[1]


def _step_secs(acc, name):
    return acc["timing"].get("st_" + name + "_s", [0, 0.0])[1]


def _top_steps(acc, n, min_share_of_engine=0.01):
    """The n biggest recorded steps overall (two-part names only), each at least 1% of the engine time."""
    engine = acc.get("batch_seconds", 0.0) or 0.0
    rows = []
    for key, (_c, secs) in acc["timing"].items():
        if key.startswith("st_") and key.endswith("_s"):
            name = key[3:-2]
            if name.count(".") == 1 and name in step_guide.STEPS and engine and secs / engine >= min_share_of_engine:
                rows.append((secs, name))
    rows.sort(reverse=True)
    return rows[:n]


def render(acc, run_seconds=None, frames=None, machine=None):
    """The block's lines (a list of strings), or [] when nothing was recorded."""
    if not acc or not acc.get("batches") or not acc.get("timing"):
        return []
    engine = acc["batch_seconds"]
    lines = ["Run timing (all batches added together)"]
    if machine:
        lines.append(f"  Computer:  {machine}")
    head = f"  Batches:   {acc['batches']}"
    if frames:
        head += f"     Frames: {int(frames):,}"
    lines.append(head)
    if run_seconds:
        lines.append(f"  Whole run: {_hms(run_seconds)}   (cleaning engine, all batches: {_hms(engine)})")
    lines += ["", "  Where the engine's time went:"]
    listed = 0.0
    for key, label, purpose in step_guide.STAGES:
        secs = _secs(acc, key)
        if secs <= 0:
            continue
        listed += secs
        pct = 100.0 * secs / engine if engine else 0.0
        lines.append(f"    {label:<24} {_hms(secs):>9}  {pct:>3.0f}%   {purpose}")
    other = engine - listed
    if engine > 0 and other > 0:
        lines.append(f"    {'Everything else':<24} {_hms(other):>9}  {100.0 * other / engine:>3.0f}%   "
                     "Small steps between the stages above.")
    # how much each stage's recorded steps leave unexplained
    gaps = []
    for group, pkey in _GAP_PARENTS:
        parent = _secs(acc, pkey)
        kids = sum(secs for key, (_c, secs) in acc["timing"].items()
                   if key.startswith("st_" + group + ".") and key.endswith("_s")
                   and key[3:-2].count(".") == 1)
        if parent > 0 and kids > 0:
            gaps.append(f"{group} {100.0 * (parent - kids) / parent:.0f}%")
    if gaps:
        lines.append("  Share of each stage that the recorded steps do not explain: " + ", ".join(gaps))
    top = _top_steps(acc, 8)
    shown = {name for _s, name in top}
    if top:
        lines += ["", "  Biggest steps (each at least 1% of the engine's time):"]
    for secs, name in top:
        label, purpose, _kv = step_guide.STEPS[name]
        lines.append(f"    {label}: {_hms(secs)}")
        lines.append(f"      {purpose}")
    fired = []
    for name in sorted(acc["counts"]):
        if name not in shown or name == "repair.fill_method":
            continue
        label, _p, meanings = step_guide.STEPS[name]
        bits = []
        for k, v in sorted(acc["counts"][name].items()):
            bits.append(f"{v:,} {meanings.get(k, k.replace('_', ' '))}")
        if bits:
            fired.append(f"    {label}: " + "; ".join(bits))
    # how each trail was filled in (the keys are method names, so list them as they are)
    fm = acc["counts"].get("repair.fill_method")
    if fm:
        fired.append("    How each trail was filled in: "
                     + "; ".join(f"{v:,} {k.replace('_', ' ')}" for k, v in sorted(fm.items(), key=lambda kv: -kv[1])))
    if fired:
        lines += ["", "  What fired (what the steps did, all batches):"] + fired
    return lines


def machine_summary(device=None, cores=None, ram_gb=None):
    """"10 cores, 32 GB memory, Apple graphics chip": the kind of computer the run used."""
    kind = {"mps": "Apple graphics chip", "cuda": "Nvidia graphics card"}.get(device or "", "processor only")
    bits = []
    if cores:
        bits.append(f"{int(cores)} cores")
    if ram_gb:
        bits.append(f"{float(ram_gb):.0f} GB memory")
    bits.append(kind)
    return ", ".join(bits)
