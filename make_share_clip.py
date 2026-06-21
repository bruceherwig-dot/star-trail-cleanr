"""make_share_clip.py — builds the post-run "share" outputs from a finished clean.

After Star Trail CleanR finishes a run, the user can ask for one or more keepsake
images/clips made from their frames. This one file produces ALL THREE of them.
It has no GUI: the app launches it as a separate process (one per requested
output) after a run, and it can also be run by hand from the command line.

THE THREE OUTPUTS
-----------------
1. STAR TRAIL  (make_star_trail, `--star-trail`)
   A full-resolution "quick and dirty" star trail: lighten-stack the CLEANED
   frames (trails already removed) into one JPG. Just the brightest pixel per
   location -- fast, not a comet-mode/gap-filled StarStaX stack. -> cleaned_star_trail.jpg

2. BEFORE/AFTER VIDEO  (make_share_clip, default mode)
   A short looping MP4 that wipes between the BEFORE (originals, trails visible)
   and the CLEANED (trails removed) so viewers see exactly what was removed.
   Details under that function.

3. RED TRAIL MAP  (make_red_trail_map, `--red-map`)  [dev-only in the app]
   The before-stack with every DETECTED trail painted solid red -- a visual of
   what the detector found. -> red_trail_map.jpg

THE SHARED IDEA (why all three look the way they do)
----------------------------------------------------
- "Lighten / maximum stack": for each pixel, keep the brightest value seen across
  the sequence. Stars sweep into arcs (a star trail); a removed trail leaves no
  bright streak. This one operation underlies all three outputs.
- "Before" = stack of the ORIGINAL frames (trails IN). "After"/"cleaned" = stack
  of the CLEANED frames (trails OUT). The words are used that precise way
  throughout this file.
- The first 3 and last 3 frames are always skipped (usually test shots) -- see
  _list_frames.
- NO SILENT DROPS: if a frame can't be read it is counted and reported LOUDLY at
  the end, because a short stack means short trails and a misleading result.

HOW TO RUN BY HAND
------------------
    python3 make_share_clip.py --star-trail --cleaned "<cleaned folder>" [--out <file.jpg>]
    python3 make_share_clip.py --original "<frames folder>" [--cleaned <dir>] [--out <file.mp4>]
    python3 make_share_clip.py --red-map --original "<frames folder>" [--out <file.jpg>]

cleaned_dir defaults to <original_dir>/cleaned. Each output's default filename is
noted in its function.

FILE LAYOUT (top to bottom)
---------------------------
  - Shared helpers: canvas sizing, frame listing, the two stack functions.
  - Video-only helpers: fonts, text box, slider grip, wipe timing, the encoder.
  - The three outputs: make_share_clip / make_red_trail_map / make_star_trail.
  - Command-line entry point.

The previous crossfade-boomerang version of the video is archived at
archive/make_share_clip_crossfade_2026_06_13.py.
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.io_safe import robust_imread          # noqa: E402  upright-safe read
from modules.frame_list import natural_key, IMAGE_EXTS  # noqa: E402

TAGLINE = "Remove the Trails. Keep the Stars."
URL = "www.StarTrailCleanR.com"
SKIP_FIRST = 3        # drop the first N frames (test shots) from the sequence
SKIP_LAST = 3         # drop the last N frames (test shots) from the sequence
FPS = 30
BOX_FRAC = 0.13       # bottom BLACK text box height, as a fraction of canvas height

# Wipe timing (seconds). Total = 10.0s loop.
MOVE_RIGHT_S = 1.5    # center -> right edge (reveals the full BEFORE)
HOLD_BEFORE_S = 2.0   # hold on the full before
MOVE_LEFT_S = 3.0     # right edge -> left edge (reveals the full CLEANED)
HOLD_AFTER_S = 2.0    # hold on the full cleaned
MOVE_CENTER_S = 1.5   # left edge -> center (loops back to the start)

_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")


# Facebook compresses 4K more aggressively than 1080p; the 1x canvas (1350x1080
# or 1080x1350) is the sweet spot for their re-encode pipeline.
SCALE = 1


# ════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS — canvas sizing, frame listing, and the two stack functions.
# (_canvas_size / _fill_canvas are video-only sizing; _list_frames and the two
#  _stack functions are the workhorses behind every output.)
# ════════════════════════════════════════════════════════════════════════════

def _canvas_size(w, h):
    """4:5 ratio in the orientation matching the source photo, rendered at SCALE x.
    Dimensions stay even (required by the H.264 yuv420p encode)."""
    bw, bh = (1350, 1080) if w >= h else (1080, 1350)
    return (bw * SCALE, bh * SCALE)   # (canvas_w, canvas_h)


def _fill_canvas(img, cw, ch):
    """Scale + center-crop so the image fully covers the canvas (no bars)."""
    h, w = img.shape[:2]
    scale = max(cw / w, ch / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (nw - cw) // 2, (nh - ch) // 2
    return r[y0:y0 + ch, x0:x0 + cw]


def _list_frames(folder):
    """Sorted image frames in `folder`, dropping the first/last few test shots
    (SKIP_FIRST / SKIP_LAST), in true capture order via natural_key."""
    exts = tuple(IMAGE_EXTS)
    fs = sorted(
        [f for f in os.listdir(folder)
         if os.path.splitext(f)[1].lower() in exts and os.path.isfile(os.path.join(folder, f))],
        key=natural_key,
    )
    end = len(fs) - SKIP_LAST                       # drop the trailing test shots
    return fs[SKIP_FIRST:end] if end > SKIP_FIRST else fs[SKIP_FIRST:]


def _stack(dirpath, names, cw, ch, label):
    """Lighten/maximum stack of `names` in `dirpath`, each filled to the canvas.
    Resizing to the small canvas BEFORE stacking keeps 900+ frames fast/light.

    No silent drops: any frame that is missing or unreadable is collected and
    reported LOUDLY at the end (an incomplete stack means shorter star trails,
    which silently breaks a before/after comparison)."""
    acc = None
    used = 0
    missing, unreadable = [], []
    for i, n in enumerate(names):
        p = os.path.join(dirpath, n)
        if not os.path.exists(p):
            missing.append(n)
            continue
        im = robust_imread(p, cv2.IMREAD_COLOR)
        if im is None:
            unreadable.append(n)
            continue
        c = _fill_canvas(im, cw, ch)
        acc = c if acc is None else np.maximum(acc, c)
        used += 1
        if i % 50 == 0:
            print(f"  {label}: {i + 1}/{len(names)}", flush=True)
    print(f"  {label}: stacked {used} of {len(names)} frames", flush=True)
    dropped = missing + unreadable
    if dropped:
        print("  " + "!" * 60, flush=True)
        print(f"  WARNING [{label}]: {len(dropped)} frame(s) NOT stacked "
              f"({len(missing)} missing, {len(unreadable)} unreadable) -- the "
              f"stack is INCOMPLETE.", flush=True)
        print(f"    skipped: {', '.join(dropped[:20])}"
              + (f" ... (+{len(dropped) - 20} more)" if len(dropped) > 20 else ""),
              flush=True)
        print("  " + "!" * 60, flush=True)
    return acc


def _stack_fullres(dirpath, names, label):
    """Lighten/maximum stack of `names` at FULL native resolution (no canvas
    resize). Only one frame plus the running accumulator are held in memory at a
    time (np.maximum writes back into the accumulator), so even hundreds of
    full-size frames stay light. Same no-silent-drop accounting as _stack(): any
    missing/unreadable frame is reported LOUDLY at the end. A frame whose size
    differs from the first is resized to match (a complete stack with one resized
    frame beats a silently shorter trail)."""
    acc = None
    used = 0
    missing, unreadable, resized = [], [], []
    for i, n in enumerate(names):
        p = os.path.join(dirpath, n)
        if not os.path.exists(p):
            missing.append(n)
            continue
        im = robust_imread(p, cv2.IMREAD_COLOR)     # upright 8-bit BGR (16-bit TIFF -> 8-bit)
        if im is None:
            unreadable.append(n)
            continue
        if acc is None:
            acc = im.copy()
        else:
            if im.shape != acc.shape:
                im = cv2.resize(im, (acc.shape[1], acc.shape[0]), interpolation=cv2.INTER_AREA)
                resized.append(n)
            np.maximum(acc, im, out=acc)
        used += 1
        if i % 50 == 0:
            print(f"  {label}: {i + 1}/{len(names)}", flush=True)
    print(f"  {label}: stacked {used} of {len(names)} frames", flush=True)
    dropped = missing + unreadable
    if dropped:
        print("  " + "!" * 60, flush=True)
        print(f"  WARNING [{label}]: {len(dropped)} frame(s) NOT stacked "
              f"({len(missing)} missing, {len(unreadable)} unreadable) -- the "
              f"stack is INCOMPLETE.", flush=True)
        print(f"    skipped: {', '.join(dropped[:20])}"
              + (f" ... (+{len(dropped) - 20} more)" if len(dropped) > 20 else ""),
              flush=True)
        print("  " + "!" * 60, flush=True)
    if resized:
        print(f"  NOTE [{label}]: {len(resized)} frame(s) had a different size and "
              f"were resized to match the first frame.", flush=True)
    return acc


class IncrementalStack:
    """A lighten/maximum stacker fed ONE frame at a time, so a star-trail or
    before/after stack can be built DURING a run instead of in a second full pass
    over every frame afterward (the slow tail the user waits on today).

    It is the exact same math as _stack / _stack_fullres -- np.maximum into a
    running accumulator -- just spread across many feed() calls, with the same
    no-silent-drop accounting. Lighten-max is order-independent, so the result is
    bit-identical to stacking the same frames all at once (proved in
    tests/test_incremental_stack.py).

    canvas=(cw, ch): resize each frame to the video canvas before stacking (matches
    _stack, used for the before/after video). canvas=None: keep full native
    resolution (matches _stack_fullres, used for the star trail), including resizing
    a stray odd-sized frame to match the first one."""

    def __init__(self, label, canvas=None):
        self.label = label
        self.canvas = canvas
        self.acc = None
        self.used = 0
        self.missing = []
        self.unreadable = []
        self.resized = []

    def feed_path(self, path):
        """Stack one frame by path. A missing or unreadable frame is RECORDED (not
        raised) so report() can warn loudly at the end, exactly like the batch stackers."""
        name = os.path.basename(path)
        if not os.path.exists(path):
            self.missing.append(name)
            return
        im = robust_imread(path, cv2.IMREAD_COLOR)
        if im is None:
            self.unreadable.append(name)
            return
        self.feed_image(im, name)

    def feed_image(self, im, name=""):
        """Stack one already-decoded upright BGR frame -- lets a caller fold in a
        frame it already holds in memory (zero extra disk read)."""
        if self.canvas is not None:
            cw, ch = self.canvas
            c = _fill_canvas(im, cw, ch)
            self.acc = c if self.acc is None else np.maximum(self.acc, c)
        else:
            if self.acc is None:
                self.acc = im.copy()
            else:
                if im.shape != self.acc.shape:
                    im = cv2.resize(im, (self.acc.shape[1], self.acc.shape[0]),
                                    interpolation=cv2.INTER_AREA)
                    if name:
                        self.resized.append(name)
                np.maximum(self.acc, im, out=self.acc)
        self.used += 1

    def report(self):
        """Print the same LOUD no-silent-drop summary the batch stackers print."""
        print(f"  {self.label}: stacked {self.used} frames", flush=True)
        dropped = self.missing + self.unreadable
        if dropped:
            print("  " + "!" * 60, flush=True)
            print(f"  WARNING [{self.label}]: {len(dropped)} frame(s) NOT stacked "
                  f"({len(self.missing)} missing, {len(self.unreadable)} unreadable) -- "
                  f"the stack is INCOMPLETE.", flush=True)
            print(f"    skipped: {', '.join(dropped[:20])}"
                  + (f" ... (+{len(dropped) - 20} more)" if len(dropped) > 20 else ""),
                  flush=True)
            print("  " + "!" * 60, flush=True)
        if self.resized:
            print(f"  NOTE [{self.label}]: {len(self.resized)} frame(s) had a different "
                  f"size and were resized to match the first frame.", flush=True)

    def result(self):
        """The finished stacked image (BGR uint8), or None if nothing was fed yet."""
        return self.acc


# ════════════════════════════════════════════════════════════════════════════
# VIDEO-ONLY HELPERS — everything below is used solely by the before/after wipe
# video: text fonts, the bottom branding box, the slider grip, the per-frame
# divider positions for the loop, and the H.264 encoder. The star trail and red
# map do not touch any of these.
# ════════════════════════════════════════════════════════════════════════════

def _font(size, bold=False):
    """Load the Inter font (bold or regular) at `size` from assets/fonts, with a
    Helvetica then PIL-default fallback if the bundled font isn't found."""
    f = os.path.join(_FONT_DIR, f"Inter-{'Bold' if bold else 'Regular'}.ttf")
    if os.path.exists(f):
        return ImageFont.truetype(f, size)
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def _compose(image_region, cw, ch, box_h):
    """Put the stacked image in the TOP of a cw x ch canvas and a SOLID BLACK
    box across the bottom holding the tagline + URL. The text lives in its own
    box BELOW the photo, never over it, so foreground content (trees, gear,
    horizon) is never covered no matter what the shot contains."""
    full = np.zeros((ch, cw, 3), np.uint8)        # black everywhere first
    full[:ch - box_h] = image_region              # photo on top; bottom stays black

    pil = Image.fromarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    tag_f = _font(int(box_h * 0.30), bold=True)
    url_f = _font(int(box_h * 0.23), bold=False)

    def center(text, font, y, fill):
        bb = d.textbbox((0, 0), text, font=font)
        d.text(((cw - (bb[2] - bb[0])) // 2, y), text, font=font, fill=fill)

    y0 = ch - box_h
    center(TAGLINE, tag_f, y0 + int(box_h * 0.20), (255, 255, 255))
    center(URL, url_f, y0 + int(box_h * 0.58), (122, 184, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _draw_handle(img, cx, cy, r):
    """Draw the round comparison-slider grip on the divider: a white disc with
    gray < > chevrons, like the before/after sliders on the web. Drawn straight
    onto the image so it clips to a half disc (one arrow showing) when the line
    reaches a frame edge, matching the reference screenshots."""
    cv2.circle(img, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)         # white disc
    cv2.circle(img, (cx, cy), r, (120, 120, 120), max(1, int(r * 0.05)),
               cv2.LINE_AA)                                                # soft border
    g = (90, 90, 90)
    th = max(2, int(r * 0.11))
    off = int(r * 0.34)        # how far each chevron sits from center
    aw = int(r * 0.20)         # chevron arm horizontal reach
    ah = int(r * 0.30)         # chevron arm vertical reach
    lx = cx - off              # left chevron  <  (apex on the left)
    cv2.line(img, (lx + aw, cy - ah), (lx - aw, cy), g, th, cv2.LINE_AA)
    cv2.line(img, (lx - aw, cy), (lx + aw, cy + ah), g, th, cv2.LINE_AA)
    rx = cx + off              # right chevron  >  (apex on the right)
    cv2.line(img, (rx - aw, cy - ah), (rx + aw, cy), g, th, cv2.LINE_AA)
    cv2.line(img, (rx + aw, cy), (rx - aw, cy + ah), g, th, cv2.LINE_AA)


def _wipe_positions(cw):
    """The divider column X for every frame of the 10-second loop, in order.
    Move phases interpolate at a steady speed and END exactly on the target so
    the following hold continues from there; the loop is seamless because the
    return-to-center ends where the next move-right begins."""
    center = cw // 2

    def ramp(x0, x1, secs):
        n = max(1, int(round(secs * FPS)))
        return [int(round(x0 + (x1 - x0) * (k + 1) / n)) for k in range(n)]

    def hold(x, secs):
        return [x] * max(1, int(round(secs * FPS)))

    xs = ramp(center, cw, MOVE_RIGHT_S)     # center -> right edge (full before)
    xs += hold(cw, HOLD_BEFORE_S)
    xs += ramp(cw, 0, MOVE_LEFT_S)          # right edge -> left edge (full cleaned)
    xs += hold(0, HOLD_AFTER_S)
    xs += ramp(0, center, MOVE_CENTER_S)    # left edge -> center (loops)
    return xs


def _open_writer(path, cw, ch):
    """Return a frame writer that takes BGR frames. Uses CBR 15 Mbps H.264 with
    dark-scene adaptive quantization (aq-mode=3) and fine grain injection — the
    evidence-backed combination for surviving Facebook's mandatory re-encode of
    near-black star-field content. Falls back to the OpenCV writer if the bundled
    ffmpeg isn't installed."""
    try:
        import imageio_ffmpeg  # noqa: F401  (ensures the bundled ffmpeg is present)
        import imageio
        w = imageio.get_writer(
            path, format="FFMPEG", mode="I", fps=FPS, codec="libx264",
            macro_block_size=None,
            ffmpeg_params=[
                "-b:v", "15M", "-maxrate", "15M", "-bufsize", "30M",
                "-preset", "medium",
                "-pix_fmt", "yuv420p", "-profile:v", "high",
                "-x264-params", "aq-mode=3",
                "-vf", "noise=alls=6:allf=u",
            ])

        class _W:
            backend = "imageio/libx264 CBR-15M aq-mode=3 +grain"

            def write(self, bgr):
                w.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            def close(self):
                w.close()
        return _W()
    except Exception as e:
        print(f"  (bundled ffmpeg unavailable: {e}; using OpenCV writer)", flush=True)
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), FPS, (cw, ch))
        if not vw.isOpened():
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (cw, ch))

        class _W:
            backend = "opencv (low bitrate fallback)"

            def write(self, bgr):
                vw.write(bgr)

            def close(self):
                vw.release()
        return _W()


# ════════════════════════════════════════════════════════════════════════════
# THE THREE OUTPUTS — one public function each. All three lighten-stack frames
# (see "THE SHARED IDEA" at the top); they differ only in WHICH frames they stack
# and how they present the result.
# ════════════════════════════════════════════════════════════════════════════

def make_share_clip(original_dir, cleaned_dir=None, out_path=None,
                    before=None, after=None):
    """OUTPUT 2 — the before/after wipe VIDEO (this is the default CLI mode).

    `before`/`after`: optional pre-built canvas stacks (photo-region size, BGR) from
    the in-run incremental stacker. When both are passed, the two frame folders are
    NOT re-read or re-stacked -- the expensive second pass is skipped and we go
    straight to the wipe encode. When they're None (the plain CLI path) the folders
    are stacked here exactly as before.

    Builds two lighten-stacks at video resolution -- BEFORE (original frames,
    trails in) and AFTER (cleaned frames, trails out) -- then writes an MP4 where
    a white divider line with a round slider grip sweeps across the frame: left of
    the line shows the before, right shows the cleaned. The line starts centered,
    slides right (reveals full before), holds, slides left (reveals full cleaned),
    holds, returns to center, and loops seamlessly. A 10-second loop:
        center -> right edge    1.5s   (reveals full BEFORE)
        hold                    2.0s
        right edge -> left edge 3.0s   (reveals full CLEANED)
        hold                    2.0s
        left edge -> center     1.5s   (loops)
    A black branding box along the bottom carries the tagline + website and the
    wipe never touches it. Canvas is 4:5 matched to the photo orientation (1350x1080
    landscape / 1080x1350 portrait) so it fills a social post with minimal crop.
    The before and after folders are stacked from their OWN file lists paired by
    sequence position (not by filename), so a zero-padded cleaned export can't
    silently drop frames. Output defaults to <original_dir>/share_clip.mp4.
    """
    if cleaned_dir is None:
        cleaned_dir = os.path.join(original_dir, "cleaned")

    names = _list_frames(original_dir)
    if not names:
        raise SystemExit("no frames left after skipping the first 3 and last 3")

    first = robust_imread(os.path.join(original_dir, names[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise SystemExit("could not read first frame")
    cw, ch = _canvas_size(first.shape[1], first.shape[0])
    box_h = int(ch * BOX_FRAC)
    img_h = ch - box_h                                          # photo region height

    if before is not None and after is not None:
        # Pre-built canvas stacks from the in-run incremental stacker -- skip the
        # whole second read pass over every frame and go straight to the encode.
        print(f"{len(names)} frames, canvas {cw}x{ch} (using pre-built in-run stacks)")
    else:
        # Stack each folder from its OWN file list, paired by sequence position, not
        # by matching filenames: a cleaned export may zero-pad early names (001.jpg)
        # while the originals don't (1.jpg), and name-matching would silently drop
        # every mismatch from the cleaned stack (shorter trails on one side).
        if not os.path.isdir(cleaned_dir):
            raise SystemExit(f"cleaned folder not found: {cleaned_dir}")
        clean_names = _list_frames(cleaned_dir)
        if not clean_names:
            raise SystemExit(f"no cleaned frames found in {cleaned_dir}")
        if len(clean_names) != len(names):
            print(f"  NOTE: {len(names)} before frames vs {len(clean_names)} cleaned "
                  f"frames; stacking each folder's full set.", flush=True)
        print(f"{len(names)} frames (first {SKIP_FIRST} and last {SKIP_LAST} skipped), "
              f"canvas {cw}x{ch}")
        before = _stack(original_dir, names, cw, img_h, "before")       # originals: trails in
        after = _stack(cleaned_dir, clean_names, cw, img_h, "after")    # cleaned: trails out
        if before is None or after is None:
            raise SystemExit(f"stacking failed (cleaned dir = {cleaned_dir})")

    # Base canvas with the black text box rendered once; each frame overwrites
    # only the photo region above it.
    base = _compose(np.zeros((img_h, cw, 3), np.uint8), cw, ch, box_h)
    r = max(20, int(cw * 0.035))      # grip radius scales with the canvas
    cy = img_h // 2                   # grip centered vertically in the photo
    lw = max(2, int(cw * 0.0035))     # divider line half-width

    if out_path is None:
        out_path = os.path.join(original_dir, "share_clip.mp4")
    writer = _open_writer(out_path, cw, ch)
    print(f"  encoder: {writer.backend}", flush=True)

    positions = _wipe_positions(cw)
    for x in positions:
        wiped = after.copy()                  # right of the line = cleaned
        if x > 0:
            wiped[:, :x] = before[:, :x]      # left of the line = before
        lo = max(0, x - lw)
        hi = min(cw, x + lw + 1)
        wiped[:, lo:hi] = (255, 255, 255)     # the white divider line
        _draw_handle(wiped, x, cy, r)         # the round grip with < > chevrons
        frame = base.copy()
        frame[:img_h] = wiped
        writer.write(frame)
    writer.close()

    total = len(positions)
    print(f"wrote {out_path}  ({total} frames, {total / FPS:.0f}s wipe loop: "
          f"{MOVE_RIGHT_S:.1f}s right / {HOLD_BEFORE_S:.0f}s / {MOVE_LEFT_S:.0f}s left / "
          f"{HOLD_AFTER_S:.0f}s / {MOVE_CENTER_S:.1f}s center)")
    return out_path


def make_red_trail_map(original_dir, out_path=None, masks_dir=None,
                       foreground_mask=None):
    """OUTPUT 3 — the RED TRAIL MAP (dev-only in the app, `--red-map`).

    The lighten-stacked BEFORE image with every DETECTED trail painted solid red:
    a picture of what the detector found. Detections come from the per-frame
    <stem>_polys.json files STC writes to cleanr_workspace/masks/ during a run,
    unioned across frames. The foreground is excluded via the foreground mask so
    red never lands on the landscape. Skips the first/last 3 like the video. No
    silent drops: frames missing detections are counted and reported."""
    import json
    if masks_dir is None or foreground_mask is None:
        from modules.workspace import find_workspace
        ws = find_workspace(original_dir) or os.path.join(original_dir, "cleanr_workspace")
        if masks_dir is None:
            masks_dir = os.path.join(ws, "masks")
        if foreground_mask is None:
            _fg = os.path.join(ws, "foreground_mask.png")
            foreground_mask = _fg if os.path.exists(_fg) else None
    exts = tuple(IMAGE_EXTS)

    names = sorted(
        [f for f in os.listdir(original_dir)
         if os.path.splitext(f)[1].lower() in exts and os.path.isfile(os.path.join(original_dir, f))],
        key=natural_key,
    )
    end = len(names) - SKIP_LAST
    names = names[SKIP_FIRST:end] if end > SKIP_FIRST else names[SKIP_FIRST:]
    if not names:
        raise SystemExit("no frames to map")

    first = robust_imread(os.path.join(original_dir, names[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise SystemExit("could not read first frame")
    cw, ch = _canvas_size(first.shape[1], first.shape[0])
    print(f"red map: canvas {cw}x{ch}  (detections from {masks_dir})")

    before = None
    red = np.zeros((ch, cw), np.uint8)
    used = 0
    n_poly = 0
    no_detect = []
    for nm in names:
        im = robust_imread(os.path.join(original_dir, nm), cv2.IMREAD_COLOR)
        if im is None:
            continue
        before = _fill_canvas(im, cw, ch) if before is None \
            else np.maximum(before, _fill_canvas(im, cw, ch))
        used += 1
        pj = os.path.join(masks_dir, os.path.splitext(nm)[0] + "_polys.json")
        if not os.path.exists(pj):
            no_detect.append(nm)
            continue
        try:
            d = json.load(open(pj))
        except Exception:
            no_detect.append(nm)
            continue
        W, H = d.get("width"), d.get("height")
        if not W or not H:
            continue
        scale = max(cw / W, ch / H)
        x0 = (W * scale - cw) / 2.0
        y0 = (H * scale - ch) / 2.0
        for p in d.get("polygons", []):
            pts = np.array(p.get("corners", []), np.float32)
            if len(pts) < 3:
                continue
            pts[:, 0] = pts[:, 0] * scale - x0
            pts[:, 1] = pts[:, 1] * scale - y0
            cv2.fillPoly(red, [pts.astype(np.int32)], 255)
            n_poly += 1
    if before is None:
        raise SystemExit("no readable frames")

    # Exclude the foreground so red never lands on the landscape (255 = foreground).
    if foreground_mask:
        fg = robust_imread(foreground_mask, cv2.IMREAD_GRAYSCALE)
        if fg is not None:
            fg = _fill_canvas(fg, cw, ch)
            red[fg > 127] = 0

    overlay = before.copy()
    overlay[red > 0] = (0, 0, 255)        # solid red (BGR) on detected trails

    if out_path is None:
        from modules.workspace import find_workspace
        ws = find_workspace(original_dir) or os.path.join(original_dir, "cleanr_workspace")
        out_path = os.path.join(ws, "red_trail_map.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    pct = 100.0 * (red > 0).mean()
    print(f"wrote {out_path}  ({used} frames, {n_poly} detections, "
          f"{pct:.1f}% red)")
    if no_detect:
        print(f"  NOTE: {len(no_detect)} frame(s) had no saved detections "
              f"(run with detection-saving on for the full map; e.g. {no_detect[0]})")
    return out_path


def make_star_trail(cleaned_dir, out_path=None, stack=None):
    """OUTPUT 1 — the quick-and-dirty full-resolution STAR TRAIL (`--star-trail`).

    `stack`: optional pre-built full-resolution lighten-max stack from the in-run
    incremental stacker. When given, the cleaned folder is NOT re-read -- the stack
    is saved straight to JPG. When None (the CLI path) the folder is stacked here.

    A lighten/maximum stack of the CLEANED frames (trails already removed) written
    as a JPG. This is NOT a comet-mode / gap-filled StarStaX stack -- just the
    brightest pixel per location across the cleaned sequence: fast, full-res, and
    good enough to share. Saved as cleaned_star_trail.jpg.

    STANDALONE: call make_star_trail(cleaned_dir, out_path) from anywhere, or run
        python3 make_share_clip.py --star-trail --cleaned "<cleaned folder>" [--out <file.jpg>]
    """
    if stack is None:
        if not os.path.isdir(cleaned_dir):
            raise SystemExit(f"cleaned folder not found: {cleaned_dir}")
        names = _list_frames(cleaned_dir)
        if not names:
            raise SystemExit(f"no cleaned frames found in {cleaned_dir}")
        print(f"{len(names)} cleaned frames (first {SKIP_FIRST} and last {SKIP_LAST} "
              f"skipped) -> full-res lighten-max star trail", flush=True)
        stack = _stack_fullres(cleaned_dir, names, "star trail")
        if stack is None:
            raise SystemExit(f"stacking failed (cleaned dir = {cleaned_dir})")
    if out_path is None:
        out_path = os.path.join(cleaned_dir, "cleaned_star_trail.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = cv2.imwrite(out_path, stack, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise SystemExit(f"could not write {out_path}")
    h, w = stack.shape[:2]
    print(f"wrote {out_path}  ({w}x{h} full-res lighten-max star trail)", flush=True)
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# COMMAND-LINE ENTRY POINT — picks ONE output by flag. The app calls this same
# script (one process per requested output); a person can run it the same way.
#   --star-trail  -> make_star_trail   (needs --cleaned, or --original to derive it)
#   --red-map     -> make_red_trail_map (needs --original)
#   (no flag)     -> make_share_clip    (the video; needs --original)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Make a shareable before/after star-trail MP4, Red Trail Map, or full-res cleaned star trail.")
    ap.add_argument("--original", default=None, help="folder of original frames (video / red map)")
    ap.add_argument("--cleaned", default=None, help="cleaned frames (default: <original>/cleaned)")
    ap.add_argument("--out", default=None, help="output file (default depends on mode)")
    ap.add_argument("--red-map", action="store_true",
                    help="make the Red Trail Map image instead of the wipe video")
    ap.add_argument("--star-trail", action="store_true",
                    help="make the full-res cleaned star trail (lighten-max of the cleaned frames)")
    ap.add_argument("--masks-dir", default=None,
                    help="folder of <stem>_polys.json detections (red map; default: resolved)")
    ap.add_argument("--foreground", default=None,
                    help="foreground mask PNG to exclude from the red map (default: resolved)")
    ap.add_argument("--prebuilt-before", default=None,
                    help="pre-built BEFORE canvas stack PNG (video; skips re-stacking the originals)")
    ap.add_argument("--prebuilt-after", default=None,
                    help="pre-built AFTER canvas stack PNG (video; skips re-stacking the cleaned frames)")
    args = ap.parse_args()
    if args.star_trail:
        st_cleaned = args.cleaned or (os.path.join(args.original, "cleaned") if args.original else None)
        if not st_cleaned:
            ap.error("--star-trail needs --cleaned (or --original to default to <original>/cleaned)")
        make_star_trail(st_cleaned, args.out)
    elif args.red_map:
        if not args.original:
            ap.error("--red-map needs --original")
        make_red_trail_map(args.original, args.out, args.masks_dir, args.foreground)
    else:
        if not args.original:
            ap.error("the wipe video needs --original")
        pb = pa = None
        if args.prebuilt_before and args.prebuilt_after:
            pb = robust_imread(args.prebuilt_before, cv2.IMREAD_COLOR)
            pa = robust_imread(args.prebuilt_after, cv2.IMREAD_COLOR)
            if pb is None or pa is None:
                ap.error("could not read the pre-built stack PNGs")
        make_share_clip(args.original, args.cleaned, args.out, before=pb, after=pa)
