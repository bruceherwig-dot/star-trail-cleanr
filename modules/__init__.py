"""Package marker for the ``modules`` folder.

This file is intentionally empty of any real code. Its only job is to tell
Python that the surrounding ``modules/`` folder is a "package" — a named
collection of related code files that the rest of the app can import from.
Without a file named ``__init__.py`` sitting here, lines elsewhere in the app
like ``from modules.repair import repair_frame`` would fail, because Python
would not recognize ``modules`` as something it can pull pieces out of.

What lives in this package (the actual working code of the pipeline):
  - ``detect_pipeline.py`` / ``detect_trails.py`` — find airplane and
    satellite trails in each frame (the YOLO + SAHI tiled detection).
  - ``trail_grouper.py`` / ``crossing_splitter.py`` / ``slope_match.py`` —
    clean up and organize those raw detections into trail shapes.
  - ``repair.py`` / ``clean_sky.py`` — the "Star Bridge" repair that borrows
    clean sky and stars from neighbor frames, plus the sky-color fill.
  - ``hot_pixels.py`` — find and fix camera hot pixels.
  - ``align.py`` — frame alignment helpers.
  - ``io_safe.py`` / ``frame_list.py`` / ``user_folder.py`` — safe image
    reading/writing, building the ordered list of frames, and locating the
    user's app data folder.
  - ``gpu_pack.py`` / ``nvidia_detect.py`` — GPU detection and setup helpers.
  - ``model_update.py`` / ``update_check.py`` / ``sparkle_updater.py`` /
    ``winsparkle_updater.py`` — checking for and applying app and model updates.
  - ``run_logger.py`` — records what happened during a cleaning run.

The two entry points that drive these modules are the desktop app
(``star_trail_cleanr.py``) and the per-batch worker it launches
(``astro_clean_v5.py``), both of which live one folder up.

There is deliberately nothing else in this file: no functions, no classes, no
imports, and no package-wide setup. Each module is imported directly by name
(for example ``from modules.io_safe import robust_imread``) rather than being
re-exported here, so this marker file stays empty on purpose.
"""
