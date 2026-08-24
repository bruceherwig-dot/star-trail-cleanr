"""ShareStacker — builds the share outputs' lighten-max stacks DURING a clean run
instead of in a second full pass over every frame afterward (the slow tail the user
waits on today).

No Qt here on purpose: the GUI wraps this in a QThread (so it stays off the UI thread
and out of the cleaning worker), but the logic is plain and unit-testable.

How it overlaps with the run:
- The BEFORE stack comes from the ORIGINAL frames, which are all present the moment a
  run starts, so build_before() can run immediately and finish early.
- The AFTER stack folds in CLEANED frames as the run produces them. scan_cleaned() is
  called at each batch boundary; it folds any newly-cleaned frames that aren't yet in.
- Each frame is read ONCE and fed into every stack that's enabled (full-res star trail,
  video canvas), so there is no duplicate reading.

Skip + no-silent-drop: _list_frames already drops the first/last few test shots from
whatever is present. During a run "present" is a growing prefix of the sorted order, so
its current last few are deferred naturally and folded once more frames arrive; at the
final scan the full list is known, so the true first/last few are the ones skipped and
folded == the batch path's kept set. The underlying IncrementalStack is proven
bit-identical to the batch stackers (tests/test_incremental_stack.py).
"""
import os
import time
import subprocess

import cv2

from modules.io_safe import robust_imread, robust_imwrite
# robust_imWRITE: cv2 cannot write non-ASCII paths on Windows.
import make_share_clip as msc


class ShareStacker:
    """Holds the running lighten-max stacks for the share outputs and lets a caller
    feed them frames over the life of a run: build_before() folds in the originals at
    the start, scan_cleaned() folds in cleaned frames at each batch boundary, and
    finalize() renders the star trail and video from the finished stacks. Builds only
    the stacks the enabled outputs need (full-res for the star trail, canvas for the
    video), so a clean-only run does no work here."""

    def __init__(self, original_dir, cleaned_dir, want_star=False, want_video=False,
                 video_cmd_prefix=None, comet_tail=0, thicken_px=0,
                 want_original_star=False):
        self.original_dir = original_dir
        self.cleaned_dir = cleaned_dir
        self.want_star = want_star
        self.want_video = want_video
        # Second star trail built from the ORIGINAL frames, so the Star Trail tab
        # offers the before picture next to the after one (the arrows page between
        # them). Free in reading time -- build_before already walks the originals
        # for the video -- but it holds one more full-res image in memory, so the
        # GUI turns it off when memory is tight rather than squeeze the cleaning.
        self.want_original_star = want_original_star
        # DEV-ONLY star-trail styling (0 = off): comet_tail fades trails into comet
        # tails over this many frames; thicken_px widens them. Only the star trail.
        self.comet_tail = comet_tail
        self.thicken_px = thicken_px
        # How to invoke make_share_clip.py as a SEPARATE process for the video encode
        # (e.g. [sys.executable, "-u", SHARE_SCRIPT]). The video runs in its own process
        # because ffmpeg-via-imageio stalls inside a Qt background thread; the star trail,
        # being just an image write, stays in-process. None disables the video.
        self._video_cmd_prefix = video_cmd_prefix
        # When this run started. A cleaned frame older than this is a leftover from a
        # PRIOR run (the worker overwrites cleaned/ in place, it doesn't clear it), so
        # we ignore it until this run rewrites it -- the stack only ever reflects the
        # current run's output. (2s slack absorbs filesystem mtime granularity.)
        self._run_start = time.time()
        self._cw = self._ch = self._img_h = None
        self.after_full = None        # full-res star-trail stack (cleaned)
        self.before_full = None       # full-res star-trail stack (ORIGINALS)
        self.before_vid = None        # canvas before-stack (video)
        self.after_vid = None         # canvas after-stack (video)
        self._after_folded = set()    # cleaned filenames already folded into the after-stacks
        self.before_built = False
        # Plain-English reasons an output could not be built, reported by finalize()
        # in its "skipped" dict so the GUI can tell the user instead of ending the
        # run with nothing and no explanation (the silent-empty path a v2.76 Windows
        # field report fell through).
        self._geom_fail = None        # why the canvas geometry never got established
        self._video_fail = None       # why the video encode subprocess failed

    # ── geometry ──────────────────────────────────────────────────────────────
    def _geometry(self):
        """Compute the video canvas geometry once from the first original frame.
        Returns False if there are no readable originals yet, recording why in
        _geom_fail: a geometry failure quietly disables EVERY stack (star trail
        and video), so the reason must survive to finalize()'s skipped report."""
        if self._cw is not None:
            return True
        names = msc._list_frames(self.original_dir)
        if not names:
            self._geom_fail = (f"no frames were found in the original folder "
                               f"({self.original_dir})")
            return False
        first = robust_imread(os.path.join(self.original_dir, names[0]), cv2.IMREAD_COLOR)
        if first is None:
            self._geom_fail = (f"the first original frame could not be read "
                               f"({os.path.join(self.original_dir, names[0])})")
            return False
        self._geom_fail = None
        self._cw, self._ch = msc._canvas_size(first.shape[1], first.shape[0])
        self._img_h = self._ch - int(self._ch * msc.BOX_FRAC)   # photo region (above the text box)
        return True

    # ── before stack (originals, available at run start) ───────────────────────
    def build_before(self, should_abort=None):
        """Read the ORIGINALS once and fold them into the before stacks the enabled
        outputs need. Only the video needs a before-stack; the star trail does not.
        Safe to call once at the very start of a run."""
        if not (self.want_video or self.want_original_star):
            self.before_built = True
            return
        if not self._geometry():
            return
        if self.want_video:
            self.before_vid = msc.IncrementalStack("before", canvas=(self._cw, self._img_h))
        if self.want_original_star:
            self.before_full = msc.IncrementalStack("before-full")            # full-res
        for n in msc._list_frames(self.original_dir):
            if should_abort and should_abort():
                return
            p = os.path.join(self.original_dir, n)
            if self.before_vid is not None:
                self.before_vid.feed_path(p)
            if self.before_full is not None:
                self.before_full.feed_path(p)
        self.before_built = True

    # ── after stack (cleaned, arrives during the run) ──────────────────────────
    def scan_cleaned(self, should_abort=None):
        """Fold any newly-cleaned frames into the after stacks. Call at each batch
        boundary and once more at the end. _list_frames already drops the current
        first/last test shots, so the most-recent few are deferred until more arrive
        and the true last few are skipped on the final scan."""
        if not (self.want_star or self.want_video):
            return
        if not self._geometry():
            return
        if self.want_star and self.after_full is None:
            self.after_full = msc.IncrementalStack("after-full")              # full-res
        if self.want_video and self.after_vid is None:
            self.after_vid = msc.IncrementalStack("after", canvas=(self._cw, self._img_h))
        for n in msc._list_frames(self.cleaned_dir):
            if n in self._after_folded:
                continue
            p = os.path.join(self.cleaned_dir, n)
            try:
                if os.path.getmtime(p) < self._run_start - 2.0:
                    continue   # leftover from a prior run; skip until THIS run rewrites it
            except OSError:
                continue
            if should_abort and should_abort():
                return
            im = robust_imread(p, cv2.IMREAD_COLOR)
            if im is None:
                # Record the miss on whichever stack exists; report() warns loudly later.
                (self.after_full or self.after_vid).unreadable.append(n)
                self._after_folded.add(n)
                continue
            if self.after_full is not None:
                self.after_full.feed_image(im, n)
            if self.after_vid is not None:
                self.after_vid.feed_image(im, n)
            self._after_folded.add(n)

    # ── finalize: render the enabled outputs from the finished stacks ──────────
    def finalize(self, star_out=None, video_out=None, should_abort=None,
                 original_star_out=None):
        """One last scan, then render each enabled output from the finished in-memory
        stacks (no re-read). The star trail is saved in-process (just an image write);
        the video is encoded in a SEPARATE process (ffmpeg via imageio stalls inside a
        Qt background thread). Each render is timed. Returns
        {"produced": {kind: path}, "timings": {kind: seconds}, "skipped": {kind: reason}}.
        Every enabled output lands in exactly one of produced or skipped: an output
        that quietly built nothing used to end the run with no video, no star trail,
        and no explanation (a v2.76 Windows field report), so the reason is now
        carried out in plain English for the GUI to show."""
        self.scan_cleaned(should_abort=should_abort)
        produced, timings, skipped = {}, {}, {}
        if should_abort and should_abort():
            return {"produced": produced, "timings": timings, "skipped": skipped}

        def _stack_reason(stack, what):
            """Why a stack has no usable result, in words a user can act on."""
            if self._geom_fail:
                return self._geom_fail
            if stack is None:
                return f"the {what} stack was never started (no frames were seen)"
            return f"no readable frames made it into the {what} stack"

        # Original-source trail first, so the CLEANED one is the newer file and the
        # Star Trail tab (newest first) opens on it, with this one a left-arrow away.
        if self.want_original_star and original_star_out:
            if self.before_full is not None and self.before_full.result() is not None:
                self.before_full.report()
                t0 = time.time()
                produced["original_star_trail"] = msc.make_star_trail(
                    self.original_dir, out_path=original_star_out,
                    stack=self.before_full.result())
                timings["original_star_trail"] = time.time() - t0
            else:
                skipped["original_star_trail"] = _stack_reason(
                    self.before_full, "original-frames")
        if self.want_star:
            if self.after_full is not None and self.after_full.result() is not None:
                self.after_full.report()
                t0 = time.time()
                produced["star_trail"] = msc.make_star_trail(
                    self.cleaned_dir, out_path=star_out, stack=self.after_full.result(),
                    comet_tail=self.comet_tail, thicken_px=self.thicken_px)
                timings["star_trail"] = time.time() - t0
            else:
                skipped["star_trail"] = _stack_reason(self.after_full, "cleaned-frames")
        if self.want_video:
            if (video_out and self._video_cmd_prefix
                    and self.before_vid is not None and self.after_vid is not None
                    and self.before_vid.result() is not None
                    and self.after_vid.result() is not None):
                self.before_vid.report()
                self.after_vid.report()
                t0 = time.time()
                vid = self._render_video_subprocess(video_out)
                if vid:
                    produced["video"] = vid
                    timings["video"] = time.time() - t0
                else:
                    skipped["video"] = (self._video_fail
                                        or "the video encoder did not finish")
            elif self.before_vid is None or (self.before_vid.result() is None):
                skipped["video"] = _stack_reason(self.before_vid, "original-frames")
            else:
                skipped["video"] = _stack_reason(self.after_vid, "cleaned-frames")
        return {"produced": produced, "timings": timings, "skipped": skipped}

    def _render_video_subprocess(self, video_out):
        """Save the two canvas stacks as temp PNGs and encode the wipe video in a
        SEPARATE process (make_share_clip.py --prebuilt-before/--prebuilt-after), so the
        ffmpeg encode never runs inside the Qt thread. Returns the video path on success,
        None on failure (logged). Temp PNGs are always cleaned up."""
        ws = os.path.dirname(video_out)
        before_png = os.path.join(ws, "_before_stack.png")
        after_png = os.path.join(ws, "_after_stack.png")
        try:
            robust_imwrite(before_png, self.before_vid.result())
            robust_imwrite(after_png, self.after_vid.result())
            cmd = list(self._video_cmd_prefix) + [
                "--original", self.original_dir, "--out", video_out,
                "--prebuilt-before", before_png, "--prebuilt-after", after_png]
            r = subprocess.run(cmd, capture_output=True, text=True)
            ok = r.returncode == 0 and os.path.exists(video_out)
            if not ok:
                self._video_fail = (f"the video encoder exited with code {r.returncode}: "
                                    f"{(r.stderr or r.stdout or 'no output')[-500:].strip()}")
                print(f"  video subprocess FAILED (rc={r.returncode}): "
                      f"{(r.stderr or '')[-500:]}", flush=True)
            return video_out if ok else None
        except Exception as e:
            self._video_fail = f"the video encoder could not be started: {type(e).__name__}: {e}"
            print(f"  video subprocess error: {type(e).__name__}: {e}", flush=True)
            return None
        finally:
            for p in (before_png, after_png):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
