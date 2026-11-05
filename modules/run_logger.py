"""
RunLogger -- appends one JSON record per pipeline event to a .jsonl file.

Written in real time (flush after every record) so partial runs are still
readable if the process crashes mid-batch.

Record types:
  detect  -- per-frame detection results: SAHI counts, filter stages, final
             trail component count. Written after _suppress_static_fps so
             static FP removal is already reflected in the final count.
  repair  -- per-frame repair results: per component area/bbox, split count,
             and per segment: tracking outcome, dx/dy, method, cleanup px.
  summary -- end-of-batch: elapsed time, frame/trail counts, run parameters.

Dev-only: only instantiated when running from live source (sys.frozen is False).
File lives at {input_dir}/cleanr_workspace/run_log_{timestamp}.jsonl.
"""

import json
from pathlib import Path


# Plain-English guide written as the FIRST record of every run log, so anyone
# (including a future Claude opening one of these cold) can interpret the file
# without reverse-engineering it. Update this whenever a field's meaning changes.
LOG_LEGEND = {
    "type": "legend",
    "_doc": "Plain-English guide to this run log. Read me first.",
    "foreground_mask_polarity": "GOTCHA, the foreground mask reads BACKWARDS from intuition: "
        "a pixel value of 0 (black) = OPEN SKY, and 255 (white) = FOREGROUND (ground, trees, "
        "buildings, gear). The mask marks what to IGNORE, not what to keep. So 'in_sky' is TRUE "
        "when the mask value is ZERO. This caught us out once, so: zero = sky, white = foreground.",
    "record_types": {
        "legend": "This explainer (always the first record).",
        "detect": "One per frame: what the AI detected and how it was filtered.",
        "repair": "One per frame: how each trail was repaired.",
        "summary": "End of batch: elapsed time, totals, run settings.",
        "harvest": "End of batch: plain-English tally of training examples this run produced.",
    },
    "key_detect_fields": {
        "sahi_raw_count": "How many raw detections the AI produced before any filtering.",
        "static_fp_suppressed": "A LIST (not a count) of detections REMOVED as static false "
            "positives: they sat in the SAME spot across neighboring frames, so they're fixed "
            "objects, not moving trails. Each item has area, cx, cy, bbox, in_sky, note. "
            "in_sky=true means it was over OPEN SKY (a genuine model false positive and good "
            "hard-negative training material); in_sky=false means it was on the foreground "
            "(wall/tree/building) -- expected, not training material.",
        "static_fp_kept_by_veto": "A LIST of detections that looked static but a veto KEPT "
            "(bright-pixel or star-motion veto judged them a real trail).",
        "final_trail_components": "How many separate trail blobs remained after all filtering.",
        "sky_mask_pixels_removed": "Detection pixels erased by the sky/foreground mask this frame.",
    },
    "decision_event_fields": "Every keep-or-toss decision in a detect record's stages carries the same "
        "four parts, so you can read what happened without opening any code: 'reason' (the rule's name), "
        "'action' (dropped, kept, thinned, added), 'bbox' ([x0, y0, x1, y1] in full-frame pixels: WHERE), "
        "'why' (plain words with the measured number and the cutoff it was judged against) and 'result' "
        "(what became of the piece). Older events (the static false-positive and phantom ones) carry a "
        "'detail' or 'note' in place of 'why'. A guard test fails if a new event leaves out the position.",
    "detect_stage_event_reasons": {
        "phantom_removed": "A THIN/FAINT hit pruned near sky. Usually a TRAIL-TIP TRIM hanging "
            "off a real trail, NOT a clean false positive. Do NOT use as a hard-negative without "
            "looking -- it often sits right on a real trail.",
        "bridge_gap_miss": "A real trail the model MISSED that the bridge reconnected. A "
            "hard-POSITIVE training candidate (label the trail at this spot).",
        "phantoms_pruned": "Count bookkeeping for the phantom-prune stage.",
        "kept_would_fail_fat_blob": "Informational only; the fat-blob gate is off.",
        "load_failed": "A frame's image could not be loaded for this stage.",
        "drop": "A detection removed in the first fitting step by a size or shape gate; the 'gate' field "
            "names it (area, star_size, aspect, fat_blob). The 'why' gives the measured size and the cutoff. "
            "star_size is the star-size rule: small, thin and not red means a star, not a trail.",
        "crossing_kept_whole": "The model outlined a crossing or tangle that the splitter could not cut "
            "into simple pieces, so it was kept as one outline. The crossing_bands event says what it "
            "became.",
        "crossing_bands": "What a kept-whole crossing became: cut into thin bands (one per line direction, "
            "plus one for any long outline piece no band covered) or painted as the model's exact outline. "
            "The F problem fix lives here.",
        "fat_thinned": "The large-square fix cut a fat piece (a junction blob) into thin bands along its "
            "line directions. 'result' says how many bands.",
        "fat_dropped_covered": "The large-square fix dropped a fat piece because thin pieces crossing it "
            "at 30 degrees or more already cover most of it.",
        "fat_dropped_redundant": "The large-square fix dropped a short wide piece that detections pointing a "
            "different way (8 degrees or more apart) already cover.",
        "group_fitted": "One group of touching, same-direction detections was fitted with polygon(s). "
            "'result' gives the polygon count and whether the fit was straight or curved. A detection "
            "that never appears in a group_fitted box was lost before fitting.",
        "leftover_polygon_added": "A long piece of a group's own pixels lay outside its fitted polygon "
            "(the Side lobe problem), so it got a box of its own.",
        "ground_mark_dropped": "The ground filter removed a polygon that looked like the outline of "
            "ground (trees, a roof) rather than a trail: dark ground along one side and no bright line "
            "inside. A mark the bright-line test cannot judge is kept, not dropped.",
        "fp_suppressed": "The static false-positive suppressor removed a detection whose pixels are nearly "
            "identical in neighbouring frames (a fixed object, not a moving trail).",
        "fp_veto_frame_edge": "A detection that looked static was KEPT because it touches the frame edge "
            "(trails exit the frame, static objects don't).",
        "fp_veto_bright_trail": "A detection that looked static was KEPT because it is much brighter than "
            "its surroundings (a real trail).",
        "regroup": "Seam second pass: detections re-grouped after new ones were folded in.",
        "ground_filter_skipped": "The ground filter did not run on this frame; 'reason' says why.",
        "skip": "A stage did not run; 'note' says why.",
    },
    "key_repair_fields": {
        "_structure": "A 'repair' record has a 'components' list, one entry per trail blob repaired "
            "on this frame. Each component has 'id', 'polygon' (which detected polygon/arm it came "
            "from), 'area' (px), 'bbox', 'split_into' (a long trail is cut into this many shorter "
            "segments so star motion is measured locally), and a 'segments' list. Each segment is one "
            "piece actually filled, described by the fields below. An EMPTY 'components' list means no "
            "trail was repaired on this frame.",
        "method": "HOW this piece was filled. 'blend' = borrowed BOTH neighbor frames (each shifted to "
            "follow the stars) and averaged them. 'single_shift' = borrowed ONE neighbor, shifted into "
            "place (keeps faint stars at full brightness). 'prev_shift'/'next_shift' = an edge frame "
            "(first/last of the run) with only one neighbor, shifted onto this frame. "
            "'prev_only'/'next_only' = a neighbor copied without a reliable shift. "
            "'raw_clean_track_failed' = star tracking failed, so a raw neighbor was pasted un-shifted. "
            "'crayon_sky_no_neighbors' = no neighbor to borrow, so painted with local sky colour + grain "
            "(the 'crayon' fill). 'black_no_sky' = the RARE last resort: no neighbor AND too little "
            "surrounding sky to sample, so pixels were set to black. Seeing 'black_no_sky' at all is "
            "notable and should essentially never happen.",
        "cascade": "WHICH tracker measured the star motion for this piece. 'agree' = star-detection and "
            "phase-correlation agreed (most trusted). 'phase' = phase-correlation confident on its own. "
            "'detect' = enough star votes on their own. 'fail' = neither confident, so no shift was used "
            "(the fill fell back to a raw neighbor paste). 'no_neighbors' = an edge frame with nothing to "
            "track against. 'none' = tracking was not run for this piece.",
        "tracking_ok": "True if a reliable star shift was measured; False = tracking failed and the sky "
            "was not slid.",
        "n_stars": "How many star streaks agreed on the measured shift (more = more trustworthy).",
        "dx / dy": "The measured star shift in pixels between the previous and next frame.",
        "still_trail_px": "Always 0 now (the old warm-pixel scrub that used this is disabled).",
        "edge_still_px": "First/last frame of the set only: pixels the edge-frame foreground "
            "protection kept UNSHIFTED (a static object like a tree trunk, detected by reaching "
            "to the second same-side neighbor because the normal two-neighbor routing can't run "
            "on an edge frame). >0 here means the fix spared foreground that would otherwise have "
            "been nicked by the single-neighbor slide. 0 on all interior frames.",
        "fg_darken_px": "Pixels of dark STATIC foreground (a tree spike, branch, rock, rooftop) that the "
            "trail crossed and that were restored by a darken (min) blend across a few neighbor frames, "
            "instead of being erased by the sky slide. On a fixed tripod that foreground is the same dark "
            "pixel every frame while the trail is bright, so the per-pixel minimum recovers it. Only pixels "
            "darker than a fraction of the local sky are touched; sky pixels keep the Star Bridge slide so "
            "moving stars stay put. >0 means foreground was saved on this frame.",
        "union_zeroed_px": "Pixels where BOTH immediate neighbors (N-1 and N+1) carried the trail (a "
            "crossing), so there was no clean sky next door. The repair reaches out to farther frames "
            "for clean sky where it can (see cross_reach_px); any pixels left over are filled from the "
            "colour-closest raw neighbor (crayon).",
        "cross_reach_px": "Of the crossing pixels (union_zeroed_px), how many were filled with REAL clean "
            "sky reached from a farther frame (N-2/N+2 or beyond, shifted to line the stars up on this "
            "frame) instead of pasting a still-dirty neighbor. Higher = more of the crossing recovered "
            "with genuine sky; the trail was moving, so a frame or two out is usually clear of it.",
        "ring_off": "The final per-channel [B,G,R] brightness nudge applied to match the patch to the sky "
            "right beside it (absorbs small frame-to-frame sky drift). null if it could not be measured.",
        "sky_filled_px": "No-neighbor case only: pixels the crayon sky-fill painted. 0 together with "
            "method 'black_no_sky' means the pure-black fallback fired.",
        "base": "Which neighbor's sky colour was the closer match here and was borrowed from ('prev' or "
            "'next').",
    },
}


class RunLogger:
    """Writes the run log: one open file, one JSON record per line.

    A RunLogger owns a single .jsonl file for one batch run. Each call to
    ``log()`` appends one record (one line of JSON) and flushes to disk right
    away, so even if the process crashes partway through a batch the file on
    disk is still complete up to the last event written.

    Usable as a context manager (``with RunLogger(path) as logger:``) so the
    file is always closed when the block exits, even on error.

    Only created when running from live Python source (not the frozen/shipped
    app), so this logging is a developer aid, not something testers ever see.
    """

    def __init__(self, log_path: str):
        """Open the log file for appending and write the legend as line one.

        ``log_path`` is the full path to the .jsonl file to write (typically
        under ``{input_dir}/cleanr_workspace/``). Any missing parent folders
        are created first. The file is opened in append mode, so re-running
        against the same path adds to it rather than wiping it. The very first
        thing written is LOG_LEGEND, so every log file explains its own format
        without needing this source code on hand.
        """
        # Make sure the workspace folder exists before opening the file for write.
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        # Append mode ("a"): never truncate an existing log; add to its end.
        self._f = open(log_path, "a", encoding="utf-8")
        self.path = log_path
        # Self-documenting header so the file is interpretable on its own.
        self.log(LOG_LEGEND)

    def log(self, record: dict) -> None:
        """Append one record to the log and flush it to disk immediately.

        ``record`` is any JSON-serializable dict (e.g. a detect/repair/summary
        event). It is written as a single line of JSON followed by a newline,
        which is what makes the file a valid .jsonl stream. The flush after
        every write is deliberate: it guarantees the record survives a crash,
        at the cost of some speed. Returns nothing.
        """
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self) -> None:
        """Close the underlying log file, swallowing any error.

        Called automatically at the end of a ``with`` block. The exception is
        intentionally ignored so that shutting down the logger can never itself
        crash the run (e.g. if the file was already closed).
        """
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        """Context-manager entry: return self so ``with RunLogger(...) as x`` works."""
        return self

    def __exit__(self, *args):
        """Context-manager exit: always close the file when the block ends."""
        self.close()
