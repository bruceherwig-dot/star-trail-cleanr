"""Time the processor-bound detection stages, so a slow machine can be diagnosed
without owning it.

WHY THIS EXISTS. Kari Tuomi's Windows PC ran the same 20 frames as Bruce's Mac
and took twice as long overall, but the gap is not spread evenly:

    RAW decode        1.3x slower      one big call into a C library
    AI inference      1.3x slower      the graphics card
    Repair            1.7x slower      a few operations on huge arrays
    Phantom pruning   4.0x slower      thousands of small operations
    Polygon fitting   4.9x slower      thousands of small operations

If his processor were simply slow, RAW decode would be slow too. If he were
paging, repair would suffer most, because it touches the biggest arrays. Neither
matches. What the two slow stages share is that they do many SMALL operations
rather than a few large ones, which points at per-operation overhead.

One candidate is OpenCV's thread dispatch: it spins up worker threads per call,
and on a many-core desktop that cost can swamp the work when the arrays are
small, while a laptop-class chip with fewer cores pays less. That is why this
benchmark runs everything TWICE, once with OpenCV threading as shipped and once
pinned to a single thread. If pinning makes the slow stages faster on Windows
and slower on the Mac, we have the answer.

The frames are synthesised, so this runs anywhere with no photographs and no
model: on a free GitHub Windows runner, the same trick that finally cracked the
Windows updater, and on any machine of Bruce's for the baseline.

Usage:  python3 scripts/bench_cpu_stages.py [--width 8152] [--height 5432]
                                            [--trails 24] [--repeats 3]

Read the numbers as RATIOS between machines, never as absolutes: a CI runner is
not a desktop, and the point is which stages diverge, not what any one costs.
"""
import argparse
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import detect_pipeline as dp                      # noqa: E402
from modules.detect_pipeline import _PredMaskWrap              # noqa: E402


class _FakePred:
    """A stand-in for a SAHI prediction. The real object exposes `.mask`, and
    that mask exposes `.bool_mask`; the pipeline reads it through both. Building
    the same shape here means the stages run their real code path, not a
    benchmark-only one."""
    __slots__ = ("mask",)

    def __init__(self, bool_mask):
        self.mask = _PredMaskWrap(bool_mask)


def _fresh_log(name):
    """A throwaway stage log. The stages want the same logging object the real
    pipeline hands them, and it needs a parent frame log to attach to."""
    return dp.StageLog(dp.FrameLog(frame_name="bench"), name)


def synth_frame(w, h, n_trails, seed=0):
    """A night sky with stars, real trails, and the dotted phantoms that phantom
    pruning exists to remove. Deterministic, so two machines measure the same work.

    The shape of the input matters more than its realism: the stages under test
    cost time per DETECTION and per COMPONENT, so the trail count and the amount
    of dotted noise are what must match a real frame, not the prettiness of it.
    """
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w, 3)) * 12).astype(np.uint8)          # dark sky floor

    # Stars: small bright dots, enough that the "is there light under this?"
    # test in phantom pruning has real work to do.
    for _ in range(4000):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (x, y), int(rng.integers(1, 3)), (200, 200, 200), -1)

    preds = []

    # Real trails: long, solid, bright. These survive pruning and go on to be
    # fitted into polygons, which is the work fit_polygons is timed on.
    for _ in range(n_trails):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        ang = rng.random() * np.pi
        length = int(rng.integers(w // 8, w // 3))
        x1 = int(np.clip(x0 + np.cos(ang) * length, 0, w - 1))
        y1 = int(np.clip(y0 + np.sin(ang) * length, 0, h - 1))
        cv2.line(img, (x0, y0), (x1, y1), (230, 230, 230), 4)
        m = np.zeros((h, w), np.uint8)
        cv2.line(m, (x0, y0), (x1, y1), 1, 5)
        preds.append(_FakePred(m.astype(bool)))

    # Phantoms: thin dotted lines with NOTHING under them in the image. This is
    # exactly the population prune_phantoms walks component by component, and the
    # per-component loop is where the time goes.
    for _ in range(n_trails // 2):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        ang = rng.random() * np.pi
        m = np.zeros((h, w), np.uint8)
        for step in range(0, 900, 14):                 # dotted, not continuous
            x = int(np.clip(x0 + np.cos(ang) * step, 0, w - 1))
            y = int(np.clip(y0 + np.sin(ang) * step, 0, h - 1))
            cv2.circle(m, (x, y), 1, 1, -1)
        preds.append(_FakePred(m.astype(bool)))

    return img, preds


def time_it(fn, repeats):
    """Best of N. The fastest run is the one least disturbed by whatever else the
    machine was doing, which matters on a shared CI runner."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run_suite(img, preds, repeats):
    """The two slow stages, plus two controls that isolate WHY.

    control_bigarray does one large operation over the whole frame; it is the
    memory-bandwidth reference. control_components does one connected-components
    scan; it is the single-big-OpenCV-call reference. If those two track between
    machines while the stages do not, the difference is per-call overhead rather
    than the machine being slow.
    """
    h, w = img.shape[:2]
    cfg = dp.StageConfig(prune_phantoms=True)
    out = {}

    def _prune():
        st = dp.PipelineState(image=img, raw_detections=list(preds))
        dp.stage_prune_phantoms(st, cfg, _fresh_log("prune_phantoms"))

    def _fit():
        st = dp.PipelineState(image=img, raw_detections=list(preds))
        dp.stage_fit_polygons(st, cfg, _fresh_log("fit_polygons"))

    mask = np.zeros((h, w), np.uint8)
    for p in preds[:8]:
        mask[p.mask.bool_mask] = 255

    out["prune_phantoms"] = time_it(_prune, repeats)
    out["fit_polygons"] = time_it(_fit, repeats)
    out["control_bigarray"] = time_it(lambda: img.max(2), repeats)
    out["control_components"] = time_it(
        lambda: cv2.connectedComponentsWithStats(mask, 8), repeats)
    return out


def main():
    """Print the machine's identity, then every timing twice: OpenCV threading as
    shipped, then pinned to one thread. The identity block matters as much as the
    numbers -- these results are only meaningful compared against another
    machine's run of the same frame size and trail count."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=8152, help="frame width (Kari's camera)")
    ap.add_argument("--height", type=int, default=5432, help="frame height")
    ap.add_argument("--trails", type=int, default=24, help="real trails per frame")
    ap.add_argument("--repeats", type=int, default=3, help="timed repeats, best wins")
    args = ap.parse_args()

    try:
        import psutil
        cores = f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads"
        ram = f"{psutil.virtual_memory().total / (1024 ** 3):.0f} GB RAM"
    except Exception:
        cores, ram = "cores unknown", "RAM unknown"

    print("=" * 68)
    print("  Star Trail CleanR - processor stage benchmark")
    print("=" * 68)
    print(f"  {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  {cores}, {ram}")
    print(f"  Python {platform.python_version()}, OpenCV {cv2.__version__}, "
          f"NumPy {np.__version__}")
    print(f"  frame {args.width}x{args.height} "
          f"({args.width * args.height / 1e6:.1f} MP), {args.trails} trails, "
          f"best of {args.repeats}")
    print(f"  OpenCV default threads: {cv2.getNumThreads()}")
    print()

    print("  building the test frame...", flush=True)
    img, preds = synth_frame(args.width, args.height, args.trails)
    print(f"  {len(preds)} detections to chew on\n", flush=True)

    results = {}
    default_threads = cv2.getNumThreads()
    for label, threads in (("as shipped", default_threads), ("pinned to 1 thread", 1)):
        cv2.setNumThreads(threads)
        print(f"  --- OpenCV {label} ({threads} thread(s)) ---", flush=True)
        r = run_suite(img, preds, args.repeats)
        results[label] = r
        for k, v in r.items():
            print(f"      {k:22s} {v:8.2f}s")
        print(flush=True)
    cv2.setNumThreads(default_threads)

    print("  --- does pinning help? ---")
    for k in results["as shipped"]:
        a = results["as shipped"][k]
        b = results["pinned to 1 thread"][k]
        verdict = "PINNING IS FASTER" if b < a * 0.9 else (
            "pinning is slower" if b > a * 1.1 else "no real difference")
        print(f"      {k:22s} {a:7.2f}s -> {b:7.2f}s   {verdict}")
    print()
    print("  Compare these numbers ACROSS machines, not against each other.")
    print("  The question is which rows diverge, not what any one row costs.")


if __name__ == "__main__":
    main()
