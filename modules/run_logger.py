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


class RunLogger:
    def __init__(self, log_path: str):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(log_path, "a", encoding="utf-8")
        self.path = log_path

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
