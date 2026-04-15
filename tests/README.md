# Star Trail CleanR tests

Regression safety net. Run before tagging any release.

```
python3 tests/run_all.py
```

All tests must pass green before shipping. No external dependencies — uses only
numpy/opencv/PIL already required by the app. Runs in under 30 seconds.

## What's tested
- `test_version.py` — `version.txt` exists, parses cleanly, in sane range
- `test_imports.py` — every module in `modules/` + `astro_clean_v5.py` imports without error
- `test_hot_pixels.py` — hot-pixel detector finds a planted defect and inpaints it
- `test_claude_md.py` — `CLAUDE.md` still contains the critical standing rules (sacred-data, version-bump, release checklist)

## What's NOT tested
- Visual ML quality (trail detection accuracy, repair aesthetics) — judged by eye, not by assert
- GUI event loop (PySide6) — too slow / flaky for a smoke suite
- Full pipeline end-to-end with real YOLO weights — tests must run without model files
