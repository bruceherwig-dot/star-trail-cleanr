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
    "detect_stage_event_reasons": {
        "phantom_removed": "A THIN/FAINT hit pruned near sky. Usually a TRAIL-TIP TRIM hanging "
            "off a real trail, NOT a clean false positive. Do NOT use as a hard-negative without "
            "looking -- it often sits right on a real trail.",
        "bridge_gap_miss": "A real trail the model MISSED that the bridge reconnected. A "
            "hard-POSITIVE training candidate (label the trail at this spot).",
        "phantoms_pruned": "Count bookkeeping for the phantom-prune stage.",
        "kept_would_fail_fat_blob": "Informational only; the fat-blob gate is off.",
    },
}


class RunLogger:
    def __init__(self, log_path: str):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(log_path, "a", encoding="utf-8")
        self.path = log_path
        # Self-documenting header so the file is interpretable on its own.
        self.log(LOG_LEGEND)

    def log(self, record: dict) -> None:
        """Append one JSON record and flush immediately."""
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
