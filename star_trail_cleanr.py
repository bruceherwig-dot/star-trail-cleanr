import sys
import os
import re

# Prevent torch-on-Windows crash when another Python install has its own
# libiomp5md.dll on PATH. Must be set BEFORE any torch-touching import so it
# also propagates to the worker subprocess that re-runs this script.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Windows frozen app (--windowed) has no console: sys.stdout/stderr are None.
if sys.platform == 'win32' and getattr(sys, 'frozen', False):
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')

# Worker mode: frozen app re-invoked as subprocess for algorithm.
if len(sys.argv) > 1 and sys.argv[1] == '--cleanr-worker':
    script = sys.argv[2]
    sys.argv = [script] + sys.argv[3:]
    import runpy
    runpy.run_path(script, run_name='__main__')
    sys.exit(0)

import glob
import time
import subprocess
import cv2
import numpy as np
from collections import Counter
from PIL import Image

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QProgressBar,
    QTextEdit, QFileDialog, QStackedWidget, QCheckBox, QFrame,
    QSpinBox, QTabWidget, QTextBrowser,
)
from PySide6.QtCore import Qt, QThread, Signal, QSettings, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon, QPalette, QColor

from mask_painter import MaskPainterWidget

if getattr(sys, 'frozen', False):
    _base = sys._MEIPASS
else:
    _base = os.path.dirname(os.path.abspath(__file__))

# ── Theme system ─────────────────────────────────────────────────────────────
# One central place for every color the app uses. Brand colors (header navy,
# banner orange, button green/blue/red, heading blue) read fine on both light
# and dark backgrounds and stay constant. Surface/text colors swap per mode.
#
# Stylesheets pull from THE legacy globals below (MUTED_TEXT, CARD_BG, etc.)
# at widget-creation time. _apply_theme() repoints those globals to the right
# variant, sets an explicit Qt palette so plain QLabels also render correctly,
# and applies a window-level stylesheet.

# Brand colors — same in both modes
BRAND_HEADER_BG       = "#0a1e3f"   # header banner navy
BRAND_HEADER_TEXT     = "white"
BRAND_HEADER_SUB      = "#a8c0e0"
BRAND_TAB_INACTIVE_BG = "#142a4a"
BRAND_TAB_INACTIVE_FG = "#a8c0e0"
BRAND_TAB_ACTIVE_BG   = "#1a6fc4"
BRAND_TAB_ACTIVE_FG   = "white"
BRAND_TAB_HOVER_BG    = "#1d3a66"
BRAND_HEADING_BLUE    = "#1a6fc4"   # FAQ/About <h2>, detail label, stats border
BRAND_HEADING_HOVER   = "#1580e0"
BRAND_RUN_GREEN       = "#2a7a2a"
BRAND_RUN_GREEN_HOVER = "#339933"
BRAND_QUIT_RED        = "#d93025"
BRAND_QUIT_RED_HOVER  = "#b8271b"
BRAND_NOTICE_ORANGE   = "#e68a00"   # update banner, model card, NVIDIA banner
BRAND_NOTICE_HOVER    = "#fdf6e3"
BRAND_SUPPORT_BG      = "#d0e4f5"
BRAND_SUPPORT_FG      = "#1a3a5c"
BRAND_SUPPORT_BORDER  = "#a0c4e0"
BRAND_SUPPORT_HOVER   = "#b8d4ec"

# Surface / text colors — swap between light and dark
THEME = {
    "light": {
        "muted_text":     "#666",
        "hint_text":      "#888",
        "card_bg":        "#e0e0e0",
        "card_text":      "#000",
        "card_border":    "#ccc",
        "panel_bg":       "#f0f7ff",
        "browser_bg":     "white",
        "browser_text":   "#000",
        "disabled_btn":   "#999",
        "disabled_hover": "#888",
        "secondary_btn":  "#666",
        "window_bg":      "",        # empty = let Qt use system default
        "success_text":   "#2a7a2a",
    },
    "dark": {
        "muted_text":     "#aaaaaa",
        "hint_text":      "#9aa4b0",
        "card_bg":        "#2d3138",
        "card_text":      "#e6e6e6",
        "card_border":    "#3a3f4a",
        "panel_bg":       "#1c2733",
        "browser_bg":     "#1c1c1e",
        "browser_text":   "#e6e6e6",
        "disabled_btn":   "#4a4a4a",
        "disabled_hover": "#5a5a5a",
        "secondary_btn":  "#555555",
        "window_bg":      "#1c1c1e",
        "success_text":   "#5dd87a",
    },
}

_CURRENT_MODE = "light"

# Legacy globals — repointed by _apply_theme() so existing f-string stylesheets
# pick up the current mode's value at widget-creation time.
MUTED_TEXT          = THEME["light"]["muted_text"]
HINT_TEXT           = THEME["light"]["hint_text"]
CARD_BG             = THEME["light"]["card_bg"]
CARD_TEXT           = THEME["light"]["card_text"]
CARD_BORDER         = THEME["light"]["card_border"]
LIGHT_PANEL_BG      = THEME["light"]["panel_bg"]
DISABLED_BTN_BG     = THEME["light"]["disabled_btn"]
DISABLED_BTN_HOVER  = THEME["light"]["disabled_hover"]
SECONDARY_BTN_BG    = THEME["light"]["secondary_btn"]
BROWSER_BG          = THEME["light"]["browser_bg"]
BROWSER_TEXT        = THEME["light"]["browser_text"]
SUCCESS_TEXT        = THEME["light"]["success_text"]


def _detect_mode():
    """Return 'dark' or 'light' based on the current OS color scheme."""
    try:
        scheme = QApplication.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return "dark"
        if scheme == Qt.ColorScheme.Light:
            return "light"
    except Exception:
        pass
    # Fallback: read the system palette's window color and pick by lightness.
    try:
        bg = QApplication.palette().color(QPalette.Window)
        return "dark" if bg.lightness() < 128 else "light"
    except Exception:
        return "light"


def _apply_theme():
    """Detect OS appearance and rewire all theme globals + Qt palette.

    Run once at startup before any widget is built and again if the OS
    appearance changes mid-session. Sets an explicit Qt palette so plain
    QLabels (which inherit text color from the palette, not from any
    stylesheet) render correctly in dark mode regardless of any platform
    quirks in Qt's own auto-detection.
    """
    global _CURRENT_MODE
    global MUTED_TEXT, HINT_TEXT, CARD_BG, CARD_TEXT, CARD_BORDER
    global LIGHT_PANEL_BG, DISABLED_BTN_BG, DISABLED_BTN_HOVER, SECONDARY_BTN_BG
    global BROWSER_BG, BROWSER_TEXT, SUCCESS_TEXT

    mode = _detect_mode()
    _CURRENT_MODE = mode
    t = THEME[mode]

    MUTED_TEXT         = t["muted_text"]
    HINT_TEXT          = t["hint_text"]
    CARD_BG            = t["card_bg"]
    CARD_TEXT          = t["card_text"]
    CARD_BORDER        = t["card_border"]
    LIGHT_PANEL_BG     = t["panel_bg"]
    DISABLED_BTN_BG    = t["disabled_btn"]
    DISABLED_BTN_HOVER = t["disabled_hover"]
    SECONDARY_BTN_BG   = t["secondary_btn"]
    BROWSER_BG         = t["browser_bg"]
    BROWSER_TEXT       = t["browser_text"]
    SUCCESS_TEXT       = t["success_text"]

    app = QApplication.instance()
    if app is None:
        return

    # Set the QPalette explicitly. Plain QLabels and other widgets that
    # don't carry their own stylesheet read text color from the palette,
    # so this is what makes the step headings on the Main tab readable
    # in dark mode even though they have no setStyleSheet call.
    pal = app.style().standardPalette()
    if mode == "dark":
        body_bg   = QColor("#1c1c1e")
        body_text = QColor("#e6e6e6")
        base_bg   = QColor("#2a2c30")    # text-input fields
        button_bg = QColor("#3a3f4a")
        placeholder = QColor("#9aa4b0")
        pal.setColor(QPalette.Window,         body_bg)
        pal.setColor(QPalette.WindowText,     body_text)
        pal.setColor(QPalette.Base,           base_bg)
        pal.setColor(QPalette.AlternateBase,  body_bg)
        pal.setColor(QPalette.Text,           body_text)
        pal.setColor(QPalette.Button,         button_bg)
        pal.setColor(QPalette.ButtonText,     body_text)
        pal.setColor(QPalette.PlaceholderText, placeholder)
        pal.setColor(QPalette.ToolTipBase,    body_bg)
        pal.setColor(QPalette.ToolTipText,    body_text)
        pal.setColor(QPalette.Highlight,      QColor("#1a6fc4"))
        pal.setColor(QPalette.HighlightedText, QColor("white"))
        pal.setColor(QPalette.Link,           QColor("#5da9ff"))
        # Disabled state: dimmer button + faded text so Open Folder buttons
        # visibly gray out in dark mode just like they do in light mode.
        pal.setColor(QPalette.Disabled, QPalette.Button,     QColor("#2a2c30"))
        pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#6a6f78"))
        pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#6a6f78"))
        pal.setColor(QPalette.Disabled, QPalette.Text,       QColor("#6a6f78"))
    app.setPalette(pal)

    # Window-level stylesheet for QMainWindow / QStackedWidget background.
    # Empty string in light mode lets Qt use the system default.
    win_bg = t["window_bg"]
    if win_bg:
        app.setStyleSheet(
            f"QMainWindow {{ background-color: {win_bg}; }}"
            f"QStackedWidget {{ background-color: {win_bg}; }}"
        )
    else:
        app.setStyleSheet("")


SCRIPT = os.path.join(_base, "astro_clean_v5.py")
_bundled_model = os.path.join(_base, "best.pt")
_DEV_FALLBACK_MODEL = os.path.join(
    os.path.expanduser("~"),
    "Documents/yolo_runs/trail_detector_v11s_tiled/weights/best.pt")


def get_model_path():
    """Return the best available trail-detector model path for this session.

    Priority: user-folder download > bundled model > dev fallback.
    Re-evaluated on each call so a mid-session model install is picked up
    on the next processing run.
    """
    try:
        from modules.user_folder import (
            get_installed_model_path, get_installed_model_version,
        )
        user_model = get_installed_model_path()
        if user_model.is_file() and get_installed_model_version():
            return str(user_model)
    except Exception:
        pass
    if os.path.isfile(_bundled_model):
        return _bundled_model
    return _DEV_FALLBACK_MODEL

try:
    with open(os.path.join(_base, "version.txt")) as _f:
        VERSION = _f.read().strip()
except Exception:
    VERSION = "dev"

SETTINGS = QSettings("StarTrailCleanR", "StarTrailCleanR")


# Sentry DSN: baked at build time by CI from the SENTRY_DSN GitHub Secret.
# `_sentry_config.py` is gitignored and absent in dev, so Sentry stays
# inactive when running from source.
try:
    from _sentry_config import DSN as _SENTRY_DSN
except ImportError:
    _SENTRY_DSN = ""


def _maybe_init_sentry():
    """Initialize Sentry only if the user opted in AND a DSN is available.

    Privacy-safe defaults: no performance traces, no auto-collected personal
    info (file paths, env, etc). Wraps the import in try/except so a missing
    sentry-sdk in dev environments never breaks the app.
    """
    if not SETTINGS.value("crash_reporting_enabled", False, type=bool):
        return
    if not _SENTRY_DSN:
        return
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            traces_sample_rate=0,
            send_default_pii=False,
            release=f"star-trail-cleanr@{VERSION}",
        )
    except Exception:
        pass


WORKSPACE_DIR = "cleanr_workspace"


def workspace_path(input_folder, filename):
    """Return <input_folder>/cleanr_workspace/<filename>. Creates dir as needed."""
    ws = os.path.join(input_folder, WORKSPACE_DIR)
    os.makedirs(ws, exist_ok=True)
    return os.path.join(ws, filename)


def migrate_workspace(input_folder):
    """One-shot migration: move legacy files into cleanr_workspace/."""
    if not input_folder or not os.path.isdir(input_folder):
        return
    old_mask = os.path.join(input_folder, "masks", "foreground_mask.png")
    new_mask = os.path.join(input_folder, WORKSPACE_DIR, "foreground_mask.png")
    if os.path.isfile(old_mask) and not os.path.isfile(new_mask):
        os.makedirs(os.path.dirname(new_mask), exist_ok=True)
        try:
            os.rename(old_mask, new_mask)
            try:
                os.rmdir(os.path.dirname(old_mask))
            except OSError:
                pass
        except OSError:
            pass


def fmt_hms(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


class CleanerWorker(QThread):
    progress = Signal(int, int, str)   # pct, total, remaining_str
    status = Signal(str)               # log line
    batch_info = Signal(int, int)      # batch_num (1-based), n_batches
    step_progress = Signal(int, int, int)  # step (1 or 2), current, total
    step_detail = Signal(str)          # filename + detail text
    frame_count = Signal(int, int)     # frames_cleaned, total
    stats_ready = Signal(int, int)     # total_trails, total_frames_scanned
    timing_stats = Signal(float, float)  # initial_estimate_sec, actual_total_sec
    initial_estimate = Signal(float)   # initial estimate seconds (emitted once)
    error = Signal(str)
    done = Signal(str)

    def __init__(self, folder, output_folder, frame_limit, mask_path=None,
                 output_format="jpg", jpeg_quality=85):
        super().__init__()
        self.folder = folder
        self.output_folder = output_folder
        self.frame_limit = frame_limit
        self.mask_path = mask_path
        self.output_format = output_format
        self.jpeg_quality = jpeg_quality
        self._cancelled = False
        self._proc = None  # current subprocess

    def cancel(self):
        """Request cancellation — kills the running subprocess."""
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def run(self):
        folder = self.folder
        output_folder = self.output_folder

        try:
            os.makedirs(output_folder, exist_ok=True)

            exts = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff",
                    "*.JPG", "*.JPEG", "*.PNG", "*.TIF", "*.TIFF"]
            frames = sorted(set(
                f for e in exts for f in glob.glob(os.path.join(folder, e))
            ))
            if not frames:
                self.error.emit(f"No image files found in: {folder}")
                return

            total = len(frames)
            if self.frame_limit != "All Frames":
                total = min(total, int(self.frame_limit))
                frames = frames[:total]

            def _img_size(path):
                try:
                    with Image.open(path) as img:
                        return img.size
                except Exception:
                    return None

            sample_step = max(1, total // 10)
            sample = [frames[i] for i in range(0, total, sample_step)][:10]
            sizes = [s for s in (_img_size(f) for f in sample) if s is not None]
            if not sizes:
                self.error.emit("Could not read any image files to determine resolution.")
                return
            dominant = Counter(sizes).most_common(1)[0][0]
            filtered = [f for f in frames if _img_size(f) == dominant]
            skipped = total - len(filtered)
            frames = filtered
            total = len(frames)
            if not frames:
                self.error.emit("No image files matched the dominant resolution.")
                return

            MAX_BATCH = 20
            n_batches = (total + MAX_BATCH - 1) // MAX_BATCH
            batch_size = (total + n_batches - 1) // n_batches if n_batches else MAX_BATCH
            starts = list(range(0, total, batch_size))
            n_batches = len(starts)

            ref_pixels = 5472 * 3648
            img_pixels = dominant[0] * dominant[1]
            res_scale = img_pixels / ref_pixels
            frames_in_batch = min(batch_size, total)
            est_seconds = int(frames_in_batch * 5 * res_scale * n_batches)

            import re
            mask_note = " with foreground mask" if self.mask_path else ""
            header = (f"Processing {total} frames ({dominant[0]}\u00d7{dominant[1]}){mask_note}"
                      + (f" \u2014 skipped {skipped} file(s) with wrong resolution" if skipped else ""))
            header += f"\n{n_batches} batch{'es' if n_batches > 1 else ''} to run"
            self.status.emit(header + "\nStarting\u2026")
            self.progress.emit(0, 100, "")
            self.frame_count.emit(0, total)

            t0 = time.time()
            frames_cleaned = 0
            total_trails_run = 0
            total_frames_run = 0

            import uuid, socket, csv as _csv, datetime as _dt
            _run_id = str(uuid.uuid4())[:8]
            _machine = socket.gethostname()
            _log_dir = os.path.join(os.path.expanduser("~"), ".star_trail_cleanr")
            _log_path = os.path.join(_log_dir, "estimator_log.csv")
            _last_log_t = [0.0]
            _LOG_INTERVAL = 5.0
            _LOG_COLS = [
                "wall_time", "run_id", "machine", "elapsed_sec", "phase",
                "batch_idx", "n_batches", "frame_in_batch", "batch_size",
                "overall_pct", "estimate_remaining_sec",
                "det_ema", "rep_ema", "warm_batches_count", "note",
            ]
            try:
                os.makedirs(_log_dir, exist_ok=True)
                if not os.path.isfile(_log_path):
                    with open(_log_path, "w", newline="") as _lf:
                        _csv.writer(_lf).writerow(_LOG_COLS)
            except OSError:
                pass

            def _log_est(phase, batch_idx, frame_in_batch, batch_sz,
                         overall_pct, remaining, force=False, note=""):
                now = time.time()
                if not force and (now - _last_log_t[0]) < _LOG_INTERVAL:
                    return
                _last_log_t[0] = now
                try:
                    with open(_log_path, "a", newline="") as _lf:
                        _csv.writer(_lf).writerow([
                            _dt.datetime.now().isoformat(timespec="seconds"),
                            _run_id, _machine, round(now - t0, 2), phase,
                            batch_idx + 1 if batch_idx is not None else "",
                            n_batches,
                            frame_in_batch if frame_in_batch is not None else "",
                            batch_sz if batch_sz is not None else "",
                            overall_pct if overall_pct is not None else "",
                            round(remaining, 2) if remaining is not None else "",
                            round(est_det_ema, 3) if est_det_ema is not None else "",
                            round(est_rep_ema, 3) if est_rep_ema is not None else "",
                            len(est_warm_batch_dts),
                            note,
                        ])
                except OSError:
                    pass

            hot_map_file = workspace_path(folder, "hot_pixel_map.png")

            def _cleanup_hot_map():
                try:
                    if os.path.isfile(hot_map_file):
                        os.remove(hot_map_file)
                except OSError:
                    pass

            # Clear any stale hot-pixel map cache from a prior run so batch 1 always builds fresh
            if self.mask_path:
                _cleanup_hot_map()

            def _add_log(line):
                self.status.emit(line)

            # ── Cumulative-rate estimator state ──
            # Each frame contributes 1.0 work unit, split DETECT_FRAC / REPAIR_FRAC
            # between the two phases. rate = work_done / elapsed, constant across
            # batch boundaries, so no lurching when a new batch starts.
            DETECT_FRAC = 0.67
            REPAIR_FRAC = 0.33
            EST_MIN_WORK_FOR_MEASURED = 2.0  # frame-equivalents before trusting measured rate
            EST_PAD_FACTOR = 1.20  # under-promise: bias estimate 20% high

            # Load persisted timing from prior runs to seed the estimator per-machine
            import json
            _timing_path = os.path.join(os.path.expanduser("~"),
                                        ".star_trail_cleanr", "last_timing.json")
            seeded_sec_per_frame = None
            try:
                if os.path.isfile(_timing_path):
                    with open(_timing_path) as _tf:
                        _prior = json.load(_tf)
                    _prior_pixels = int(_prior.get("image_pixels", 0))
                    _prior_spf = float(_prior.get("sec_per_frame", 0))
                    if _prior_spf > 0 and _prior_pixels > 0:
                        cur_pixels = dominant[0] * dominant[1]
                        seeded_sec_per_frame = _prior_spf * (cur_pixels / _prior_pixels)
                    else:
                        # Backward compatibility with old warm_batch_mean format
                        _prior_per_batch = float(_prior.get("warm_batch_mean", 0))
                        if _prior_per_batch > 0 and _prior_pixels > 0:
                            cur_pixels = dominant[0] * dominant[1]
                            seeded_sec_per_frame = (_prior_per_batch / 20.0) * (cur_pixels / _prior_pixels)
            except (OSError, ValueError, KeyError):
                pass

            est_processing_start_t = None   # set when first frame tick fires
            est_batches_done_frames = 0     # sum of frame counts of fully-completed batches
            est_initial_shown = None        # first remaining-estimate shown to user
            final_sec_per_frame = None      # measured at run end for persistence

            # Shims kept so _log_est schema stays stable
            est_det_ema = None
            est_rep_ema = None
            est_warm_batch_dts = []
            est_completed_batch_dts = []

            def _estimate_remaining(now, this_batch_size, phase, frame_num, frame_total):
                if phase == "detect":
                    frac = (frame_num / frame_total) * DETECT_FRAC
                else:
                    frac = DETECT_FRAC + (frame_num / frame_total) * REPAIR_FRAC
                cur_batch_work = frac * this_batch_size
                work_done = est_batches_done_frames + cur_batch_work
                remaining_work = total - work_done
                if remaining_work <= 0:
                    return 0.0
                rate = None
                if (est_processing_start_t is not None
                        and work_done >= EST_MIN_WORK_FOR_MEASURED):
                    elapsed = now - est_processing_start_t
                    if elapsed > 0:
                        rate = work_done / elapsed  # frames/sec
                if rate is None or rate <= 0:
                    if seeded_sec_per_frame is None or seeded_sec_per_frame <= 0:
                        return None
                    return remaining_work * seeded_sec_per_frame * EST_PAD_FACTOR
                return (remaining_work / rate) * EST_PAD_FACTOR

            _log_est("run_start", None, None, None, 0, None, force=True,
                     note=f"n={total} batches={n_batches} res={dominant[0]}x{dominant[1]}")

            for i, start in enumerate(starts):
                if self._cancelled:
                    _log_est("cancelled", i, None, None, None, None, force=True)
                    return

                self.batch_info.emit(i + 1, n_batches)
                _add_log(f"Batch {i+1}/{n_batches}")

                this_batch = min(batch_size, total - start)
                _log_est("batch_start", i, 0, this_batch, None, None, force=True)
                if getattr(sys, 'frozen', False):
                    cmd = [sys.executable, '--cleanr-worker', SCRIPT, folder,
                           "-o", output_folder, "--model", get_model_path(),
                           "--start", str(start), "--batch", str(this_batch)]
                else:
                    cmd = [sys.executable, "-u", SCRIPT, folder,
                           "-o", output_folder, "--model", get_model_path(),
                           "--start", str(start), "--batch", str(this_batch)]

                if self.mask_path:
                    cmd.extend(["--foreground-mask", self.mask_path])
                    cmd.extend(["--hot-pixel-map", hot_map_file])
                cmd.extend(["--output-format", self.output_format,
                            "--jpeg-quality", str(self.jpeg_quality)])
                cmd.extend(["--expected-width", str(dominant[0]),
                            "--expected-height", str(dominant[1])])

                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, bufsize=1
                )
                cur_step = 0
                detect_count = 0
                repair_count = 0
                _sub_re = re.compile(r'(\d+)/(\d+)')

                for proc_line in self._proc.stdout:
                    if self._cancelled:
                        self._proc.kill()
                        return
                    proc_line = proc_line.strip()
                    if not proc_line:
                        continue

                    # Parse stat lines emitted by astro_clean_v5
                    if proc_line.startswith("BATCH_TRAIL_COUNT:"):
                        try:
                            total_trails_run += int(proc_line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                        continue
                    if proc_line.startswith("BATCH_FRAME_COUNT:"):
                        try:
                            total_frames_run += int(proc_line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                        continue

                    # Parse step transitions
                    if "Step 1" in proc_line and "detecting" in proc_line:
                        cur_step = 1
                        self.step_progress.emit(1, 0, this_batch)
                    elif "Step 2" in proc_line and "repairing" in proc_line:
                        cur_step = 2
                        self.step_progress.emit(1, this_batch, this_batch)
                        self.step_progress.emit(2, 0, this_batch)

                    # Parse frame progress within steps
                    sub_m = _sub_re.search(proc_line)
                    if sub_m:
                        frame_num = int(sub_m.group(1))
                        frame_total = int(sub_m.group(2))

                        if cur_step == 1 and "detecting " in proc_line:
                            detect_count = frame_num
                            now_t = time.time()
                            if est_processing_start_t is None:
                                est_processing_start_t = now_t
                            self.step_progress.emit(1, frame_num, frame_total)
                            self.step_detail.emit(proc_line)

                            remaining = _estimate_remaining(now_t, this_batch, "detect", frame_num, frame_total)
                            if remaining is not None:
                                batch_pct = (detect_count / frame_total) * 0.67
                                overall_pct = int(((i + batch_pct) / n_batches) * 100)
                                overall_pct = max(0, min(99, overall_pct))
                                self.progress.emit(overall_pct, 100, fmt_hms(remaining))
                                if est_initial_shown is None:
                                    est_initial_shown = remaining + (now_t - t0)
                                    self.initial_estimate.emit(float(est_initial_shown))
                                _log_est("detect", i, frame_num, frame_total,
                                         overall_pct, remaining)

                        elif cur_step == 2 and "repairing " in proc_line:
                            repair_count = frame_num
                            now_t = time.time()
                            if est_processing_start_t is None:
                                est_processing_start_t = now_t
                            self.step_progress.emit(2, frame_num, frame_total)
                            frames_cleaned = start + frame_num
                            self.frame_count.emit(frames_cleaned, total)
                            self.step_detail.emit(proc_line)

                            remaining = _estimate_remaining(now_t, this_batch, "repair", frame_num, frame_total)
                            if remaining is not None:
                                batch_pct = 0.67 + (repair_count / frame_total) * 0.33
                                overall_pct = int(((i + batch_pct) / n_batches) * 100)
                                overall_pct = max(0, min(99, overall_pct))
                                self.progress.emit(overall_pct, 100, fmt_hms(remaining))
                                if est_initial_shown is None:
                                    est_initial_shown = remaining + (now_t - t0)
                                    self.initial_estimate.emit(float(est_initial_shown))
                                _log_est("repair", i, frame_num, frame_total,
                                         overall_pct, remaining)

                        elif cur_step == 0:
                            self.step_detail.emit(proc_line)

                    _add_log(f"  {proc_line}")

                self._proc.wait()
                if self._cancelled:
                    return
                if self._proc.returncode != 0:
                    stderr_text = self._proc.stderr.read().strip()
                    err_lines = [l for l in stderr_text.splitlines() if l.strip()]
                    err_msg = err_lines[-1] if err_lines else "unknown error"
                    self.error.emit(f"Batch {i+1} failed: {err_msg}")
                    return
                self._proc = None

                # Cumulative-rate estimator: advance completed-frames counter
                est_batches_done_frames += this_batch
                _log_est("batch_end", i, this_batch, this_batch, None, None,
                         force=True, note=f"cum_frames={est_batches_done_frames}")

                # Mark both steps complete for this batch
                self.step_progress.emit(2, this_batch, this_batch)
                _add_log(f"Batch {i+1}/{n_batches} complete ({fmt_hms(time.time() - t0)} elapsed)")

            self.progress.emit(100, 100, "")
            _log_est("run_complete", None, None, None, 100, 0, force=True,
                     note=f"actual_total={round(time.time()-t0,2)}")
            done_msg = f"Done! {total} frames in {n_batches} batch{'es' if n_batches > 1 else ''} ({fmt_hms(time.time() - t0)})"
            _add_log(done_msg)
            self.step_detail.emit(done_msg)
            self.stats_ready.emit(total_trails_run, total_frames_run or total)
            if est_initial_shown is not None:
                self.timing_stats.emit(float(est_initial_shown),
                                       float(time.time() - t0))

            # Persist sec-per-frame for next run's estimator seed
            if est_processing_start_t is not None and est_batches_done_frames > 0:
                processing_elapsed = time.time() - est_processing_start_t
                final_sec_per_frame = processing_elapsed / est_batches_done_frames
                try:
                    _timing_dir = os.path.dirname(_timing_path)
                    os.makedirs(_timing_dir, exist_ok=True)
                    with open(_timing_path, "w") as _tf:
                        json.dump({
                            "sec_per_frame": final_sec_per_frame,
                            "image_pixels": dominant[0] * dominant[1],
                            "app_version": VERSION,
                        }, _tf)
                except OSError:
                    pass

            if self.mask_path:
                _cleanup_hot_map()

            self.done.emit(output_folder)

        except Exception as e:
            self.error.emit(str(e))


class UpdateCheckThread(QThread):
    """Background check for a newer app release on GitHub. Silent on any failure."""
    result_ready = Signal(dict)

    def run(self):
        from modules.update_check import check_for_update
        result = check_for_update(VERSION)
        if result:
            self.result_ready.emit(result)


class ModelUpdateCheckThread(QThread):
    """Background check for a newer trail-detector model release. Silent on any failure."""
    result_ready = Signal(dict)

    def run(self):
        from modules.model_update import check_for_model_update
        result = check_for_model_update()
        if result:
            self.result_ready.emit(result)


class NvidiaDetectThread(QThread):
    """Background NVIDIA GPU detection. Emits the outcome and a detail string."""
    result_ready = Signal(str, str)

    def run(self):
        from modules.nvidia_detect import detect_nvidia
        outcome, detail = detect_nvidia()
        self.result_ready.emit(outcome, detail)


class ModelDownloadThread(QThread):
    """Streams a model file into the user folder. Atomic via temp-then-rename.

    Writes the version note only after the rename succeeds, so a mid-download
    crash leaves the previous model in place and the note untouched.
    """
    progress = Signal(int, int)   # bytes_done, total_bytes (total=0 if unknown)
    finished_ok = Signal(str)     # version tag that was installed
    failed = Signal(str)          # short error string; not user-visible

    def __init__(self, url, target_path, version_tag, parent=None):
        super().__init__(parent)
        self.url = url
        self.target_path = target_path
        self.version_tag = version_tag

    def run(self):
        import os as _os
        import urllib.request
        tmp_path = self.target_path + ".tmp"
        try:
            req = urllib.request.Request(
                self.url,
                headers={"User-Agent": "StarTrailCleanR-ModelDownload"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        self.progress.emit(done, total)
            _os.replace(tmp_path, self.target_path)
            from modules.user_folder import save_installed_model_version
            save_installed_model_version(self.version_tag)
            self.finished_ok.emit(self.version_tag)
        except Exception as e:
            try:
                _os.unlink(tmp_path)
            except Exception:
                pass
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Star Trail CleanR (Beta v{VERSION})")
        self.setMinimumWidth(720)
        saved_geo = SETTINGS.value("window_geometry")
        if saved_geo:
            self.restoreGeometry(saved_geo)
        self.worker = None
        self._mask_path = None
        self._mask_window = None

        # Main stacked widget: page 0 = setup, page 1 = processing
        self._stack = QStackedWidget()

        # Tabs: Main / FAQ / About
        self._tabs = QTabWidget()
        self._tabs.tabBar().setExpanding(True)
        self._tabs.tabBar().setDocumentMode(True)
        self._tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: none; background: palette(window); }}"
            "QTabBar { qproperty-drawBase: 0; }"
            f"QTabBar::tab {{ background: {BRAND_TAB_INACTIVE_BG}; color: {BRAND_TAB_INACTIVE_FG}; padding: 14px 20px; "
            "font-size: 15px; font-weight: bold; border: none; min-width: 200px; }"
            f"QTabBar::tab:selected {{ background: {BRAND_TAB_ACTIVE_BG}; color: {BRAND_TAB_ACTIVE_FG}; }}"
            f"QTabBar::tab:hover:!selected {{ background: {BRAND_TAB_HOVER_BG}; color: {BRAND_TAB_ACTIVE_FG}; }}"
        )
        self._tabs.addTab(self._stack, "Main")
        self._tabs.addTab(self._build_faq_tab(), "FAQ")
        self._tabs.addTab(self._build_about_tab(), "About")

        # Container: banner on top, tabs below
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(self._build_banner())
        container_layout.addWidget(self._build_update_banner())
        container_layout.addWidget(self._build_model_update_card())
        container_layout.addWidget(self._build_nvidia_banner())
        container_layout.addWidget(self._tabs)
        self.setCentralWidget(container)

        self._build_setup_page()
        self._build_process_page()
        self._stack.setCurrentIndex(0)

        for lbl in self.findChildren(QLabel):
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self._start_update_check()
        self._start_model_update_check()
        self._start_nvidia_detect()

    # ── FAQ tab ──────────────────────────────────────────────────────────────

    def _build_faq_tab(self):
        wrap = QWidget()
        wrap_layout = QVBoxLayout(wrap)
        wrap_layout.setContentsMargins(24, 24, 24, 24)
        wrap_layout.setSpacing(0)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.document().setDocumentMargin(20)
        browser.setStyleSheet(
            f"QTextBrowser {{ background: {BROWSER_BG}; color: {BROWSER_TEXT}; border: none; font-size: 14px; }}"
        )
        browser.setHtml(f"""
        <html><body style='font-family: -apple-system, sans-serif; line-height: 1.5; margin:0; padding:0; color:{BROWSER_TEXT}; background-color:{BROWSER_BG};'>
        <p style='margin:0; padding:0; line-height:0; font-size:1px; height:0;'></p>
        <h2 style='color:{BRAND_HEADING_BLUE}; margin-top:0; margin-bottom:2px;'>Why Star Trail CleanR?</h2>
        <p style='margin-top:2px;'>Star Trail CleanR removes airplane and satellite trails
        from astrophotography sequences while preserving the real stars. The result is a
        clean set of frames you can stack into a perfect star trail composite.</p>

        <h2 style='color:{BRAND_HEADING_BLUE}; margin-bottom:2px;'>Trail Detection</h2>
        <p style='margin-top:2px;'>Each frame is run through a YOLO segmentation model
        trained on thousands of manually labeled airplane and satellite trails across many
        cameras, lenses, and sky conditions. The model produces pixel-accurate masks for
        every trail it finds.</p>

        <h2 style='color:{BRAND_HEADING_BLUE}; margin-bottom:2px;'>The Fix &mdash; Star Bridge Repair</h2>
        <p style='margin-top:2px;'>For each trail, Star Trail CleanR pulls clean pixels
        from the frame immediately before and after, blending them across the trail using
        a morphing technique called <i>Star Bridge</i>. This preserves the real stars
        underneath the trail and keeps the brightness and color natural &mdash; no smudges,
        no blank patches.</p>

        <h2 style='color:{BRAND_HEADING_BLUE}; margin-bottom:2px;'>Workflow</h2>
        <ol style='margin-top:2px;'>
        <li><b>Browse</b> &mdash; choose your folder of frames.</li>
        <li><b>Mask (optional)</b> &mdash; paint over ground, buildings, and rocks so
        the AI ignores them. Trees can be left unmasked.</li>
        <li><b>Format</b> &mdash; pick output format (JPG / TIFF 8-bit / TIFF 16-bit)
        and JPEG quality.</li>
        <li><b>Run</b> &mdash; sit back. Cleaned frames land in a <code>cleaned/</code>
        folder next to your originals.</li>
        <li><b>Stack</b> &mdash; load the cleaned frames into your favorite stacker
        (StarStaX, Sequator, Photoshop) for the final composite.</li>
        </ol>

        <h2 style='color:{BRAND_HEADING_BLUE}; margin-bottom:2px;'>Limitations</h2>
        <ul style='margin-top:2px;'>
        <li><b>Trail variety is bounded by the AI's training data.</b> If a type of
        trail isn't being detected well in your sequences, you can help train the next
        version: zip 300+ frames from that scene and send them to
        <a href='mailto:bruceherwig@gmail.com?subject=Star%20Trail%20CleanR%20training%20frames'>bruceherwig@gmail.com</a>.
        For large folders, share a Dropbox, Google Drive, or WeTransfer link instead
        of attaching directly. The model gets smarter every time the community
        contributes.</li>
        <li><b>Meteors will be removed too.</b> Their streaks look similar to airplane
        and satellite trails, so the detector can't tell them apart. If you want to
        keep them, use your originals to mask them back in.</li>
        <li><b>RAW files (.CR2, .NEF, .ARW, etc.) are not yet supported.</b> Convert
        your sequence to JPG or TIFF first, then run Star Trail CleanR on the converted
        frames.</li>
        <li><b>Not a one-click fix.</b> You'll still want to touch up the final
        composite in Photoshop or your editor of choice &mdash; but if we did our job
        right, it's a fraction of the time you used to spend.</li>
        <li><b>Designed for wide-field star trail sequences,</b> not deep-sky tracked exposures.</li>
        </ul>

        <p style='color:{HINT_TEXT}; margin-top:24px;'>Star Trail CleanR is free and offered as
        a gift to the astrophotography community.
        <a href='mailto:bruceherwig@gmail.com?subject=Star%20Trail%20CleanR%20feedback'>Feedback welcome.</a></p>
        </body></html>
        """)
        wrap_layout.addWidget(browser)
        return wrap

    # ── About tab ────────────────────────────────────────────────────────────

    def _build_about_tab(self):
        wrap = QWidget()
        layout = QHBoxLayout(wrap)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)

        # Photo on left
        photo_lbl = QLabel()
        img_path = os.path.join(_base, "assets", "bruce_silhouette.jpg")
        if os.path.exists(img_path):
            pix = QPixmap(img_path)
            photo_lbl.setPixmap(pix.scaledToWidth(220, Qt.SmoothTransformation))
        photo_lbl.setAlignment(Qt.AlignTop)
        layout.addWidget(photo_lbl)

        # Bio on right
        bio = QTextBrowser()
        bio.setOpenExternalLinks(True)
        bio.setStyleSheet(
            f"QTextBrowser {{ background: {BROWSER_BG}; color: {BROWSER_TEXT}; border: none; font-size: 14px; }}"
        )
        bio.document().setDocumentMargin(20)
        bio.setHtml(f"""
        <html><body style='font-family: -apple-system, sans-serif; line-height: 1.5; margin:0; padding:0; color:{BROWSER_TEXT}; background-color:{BROWSER_BG};'>
        <p style='margin:0; padding:0; line-height:0; font-size:1px; height:0;'></p>
        <h2 style='color:{BRAND_HEADING_BLUE}; margin-top:0; margin-bottom:2px;'>About the Authors</h2>
        <p style='margin-top:2px;'>Star Trail CleanR is a passion project. I've been
        shooting star trails for over a decade, and the whole time I kept thinking
        <i>somebody should really write a program that gets rid of all the airplane
        and satellite trails.</i> Nobody did. So I finally built one &mdash; with a
        lot of help.</p>

        <p>The "lot of help" is Claude, Anthropic's AI assistant. Countless hours of
        back-and-forth: I describe what I want, Claude writes the code, we test it,
        I push back, we try again. Star Trail CleanR wouldn't exist without that
        partnership.</p>

        <p>Star Trail CleanR is a free gift to the astrophotography community that
        has taught me so much.</p>

        <h3 style='color:{BRAND_HEADING_BLUE}; margin:12px 0 2px 0;'>Links</h3>
        <ul style='margin-top:2px;'>
        <li>Photos for sale: <a href='https://bruceherwig.com'>bruceherwig.com</a></li>
        <li>Blog: <a href='https://bruceherwig.wordpress.com'>bruceherwig.wordpress.com</a></li>
        </ul>

        <h3 style='color:{BRAND_HEADING_BLUE}; margin:12px 0 2px 0;'>Acknowledgments</h3>
        <p style='margin-top:2px;'>Star Trail CleanR exists because of the generosity of fellow astrophotographers
        who shared their image sequences for AI training, tested early builds, and offered
        feedback. Every detected trail is a thank-you to them.</p>
        <p><a href='https://bruceherwig.wordpress.com/star-trail-cleanr/#Thanks'>See the
        full list of contributors &rarr;</a></p>

        <h3 style='color:{BRAND_HEADING_BLUE}; margin:12px 0 2px 0;'>Version History</h3>
        <p style='margin-top:2px;'>See the full <a href='https://github.com/bruceherwig-dot/star-trail-cleanr/blob/main/CHANGELOG.md'>version history on GitHub</a>.</p>

        <h3 style='color:{BRAND_HEADING_BLUE}; margin:12px 0 2px 0;'>Share Your Work&hellip; Have a Suggestion?</h3>
        <p style='margin-top:2px;'>Got a before-and-after you'd like to share? I would love to see it!<br>
        Have an idea or feedback to make Star Trail CleanR even better? I want to hear it!<br>
        Email me at <a href='mailto:bruceherwig@gmail.com?subject=Star%20Trail%20CleanR'>bruceherwig@gmail.com</a></p>

        <p style='color:{HINT_TEXT}; margin-top:24px;'>&copy; 2026 Bruce Herwig</p>
        </body></html>
        """)
        layout.addWidget(bio, 1)

        return wrap

    # ── Banner ───────────────────────────────────────────────────────────────

    def _build_banner(self):
        banner = QWidget()
        banner.setFixedHeight(80)
        banner.setStyleSheet(f"background-color: {BRAND_HEADER_BG};")
        outer = QHBoxLayout(banner)
        outer.setContentsMargins(0, 0, 16, 0)
        outer.setSpacing(12)

        # Left: icon
        icon_lbl = QLabel()
        icon_path = os.path.join(_base, "assets", "icon_1024.png")
        if os.path.exists(icon_path):
            pix = QPixmap(icon_path)
            icon_lbl.setPixmap(pix.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        icon_lbl.setFixedSize(80, 80)
        icon_lbl.setStyleSheet("background: transparent;")
        outer.addWidget(icon_lbl)

        # Title block (vertically centered next to icon)
        text_wrap = QWidget()
        text_wrap.setStyleSheet("background: transparent;")
        text_col = QVBoxLayout(text_wrap)
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)
        text_col.addStretch()
        title = QLabel("Star Trail CleanR")
        title.setStyleSheet(f"color: {BRAND_HEADER_TEXT}; font-size: 26px; font-weight: bold; background: transparent;")
        text_col.addWidget(title)
        sub = QLabel(f"Beta v{VERSION}")
        sub.setStyleSheet(f"color: {BRAND_HEADER_SUB}; font-size: 12px; background: transparent;")
        text_col.addWidget(sub)
        self._header_model_label = QLabel(self._current_model_display_name())
        self._header_model_label.setStyleSheet(
            f"color: {BRAND_HEADER_SUB}; font-size: 12px; background: transparent;"
        )
        text_col.addWidget(self._header_model_label)
        text_col.addStretch()
        outer.addWidget(text_wrap)
        outer.addStretch()

        # Hidden relaunch button (invisible, to the left of Support)
        relaunch_btn = QPushButton("")
        relaunch_btn.setFixedSize(32, 32)
        relaunch_btn.setStyleSheet("QPushButton { background: transparent; border: none; }")
        relaunch_btn.clicked.connect(self._relaunch)
        outer.addWidget(relaunch_btn)

        # Right: Support button
        support_btn = QPushButton("\u2764  Support")
        support_btn.setFixedHeight(32)
        support_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BRAND_SUPPORT_BG}; color: {BRAND_SUPPORT_FG}; font-size: 13px; "
            f"font-weight: bold; border-radius: 16px; border: 1px solid {BRAND_SUPPORT_BORDER}; "
            f"padding: 0 16px; }}"
            f"QPushButton:hover {{ background-color: {BRAND_SUPPORT_HOVER}; }}"
        )
        support_btn.setToolTip("Support this project")
        support_btn.clicked.connect(lambda: __import__('webbrowser').open(
            "https://bruceherwigphotographer.square.site/product/tip-jar/WCQQP7HM4SGFWSNBSAFNX7QF"))
        outer.addWidget(support_btn)

        # Right: Close X
        quit_btn = QPushButton("\u2715")
        quit_btn.setFixedSize(32, 32)
        quit_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BRAND_QUIT_RED}; color: white; font-size: 22px; "
            f"font-weight: bold; border-radius: 4px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_QUIT_RED_HOVER}; }}"
        )
        quit_btn.setToolTip("Quit Star Trail CleanR")
        quit_btn.clicked.connect(self.close)
        outer.addWidget(quit_btn)

        return banner

    # ── Update banner (hidden until a newer release is found on GitHub) ──────

    def _build_update_banner(self):
        banner = QFrame()
        banner.setFixedHeight(44)
        banner.setStyleSheet(f"QFrame {{ background-color: {BRAND_NOTICE_ORANGE}; }}")
        banner.setVisible(False)
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(16, 0, 8, 0)
        layout.setSpacing(12)

        self._update_label = QLabel("")
        self._update_label.setStyleSheet(
            "color: white; font-size: 14px; font-weight: bold; background: transparent;"
        )
        layout.addWidget(self._update_label)
        layout.addStretch()

        download_btn = QPushButton("Download")
        download_btn.setFixedHeight(28)
        download_btn.setStyleSheet(
            f"QPushButton {{ background-color: white; color: {BRAND_NOTICE_ORANGE}; font-size: 13px; "
            f"font-weight: bold; border-radius: 4px; padding: 0 16px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_NOTICE_HOVER}; }}"
        )
        download_btn.clicked.connect(self._on_update_download)
        layout.addWidget(download_btn)

        dismiss_btn = QPushButton("✕")
        dismiss_btn.setFixedSize(28, 28)
        dismiss_btn.setToolTip("Dismiss for this session")
        dismiss_btn.setStyleSheet(
            "QPushButton { background: transparent; color: white; font-size: 16px; "
            "font-weight: bold; border: none; }"
            "QPushButton:hover { color: #ffdddd; }"
        )
        dismiss_btn.clicked.connect(lambda: self._update_banner.setVisible(False))
        layout.addWidget(dismiss_btn)

        self._update_banner = banner
        self._update_download_url = None
        return banner

    def _start_update_check(self):
        self._update_thread = UpdateCheckThread(self)
        self._update_thread.result_ready.connect(self._on_update_result)
        self._update_thread.start()

    def _on_update_result(self, result):
        tag = result.get("tag", "")
        self._update_download_url = result.get("download_url")
        self._update_label.setText(f"New version available: {tag}")
        self._update_banner.setVisible(True)

    def _on_update_download(self):
        if self._update_download_url:
            import webbrowser
            webbrowser.open(self._update_download_url)

    # ── Model update card (shows when GitHub has a newer trail detector) ─────

    def _build_model_update_card(self):
        card = QFrame()
        card.setVisible(False)
        card.setStyleSheet(f"QFrame {{ background-color: {BRAND_NOTICE_ORANGE}; }}")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(2)

        self._model_title = QLabel("")
        self._model_title.setStyleSheet(
            "color: white; font-size: 16px; font-weight: bold; background: transparent;"
        )
        layout.addWidget(self._model_title)

        self._model_summary = QLabel("")
        self._model_summary.setStyleSheet(
            "color: white; font-size: 14px; background: transparent;"
        )
        self._model_summary.setWordWrap(True)
        layout.addWidget(self._model_summary)

        self._model_credits = QLabel("")
        self._model_credits.setStyleSheet(
            "color: white; font-size: 13px; font-style: italic; background: transparent;"
        )
        self._model_credits.setWordWrap(True)
        layout.addWidget(self._model_credits)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 6, 0, 0)
        action_row.setSpacing(8)

        self._model_download_btn = QPushButton("Download now")
        self._model_download_btn.setFixedHeight(28)
        self._model_download_btn.setStyleSheet(
            f"QPushButton {{ background-color: white; color: {BRAND_NOTICE_ORANGE}; font-size: 13px; "
            f"font-weight: bold; border-radius: 4px; padding: 0 16px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_NOTICE_HOVER}; }}"
        )
        self._model_download_btn.clicked.connect(self._on_model_download_clicked)
        action_row.addWidget(self._model_download_btn)

        self._model_notnow_btn = QPushButton("Not right now")
        self._model_notnow_btn.setFixedHeight(28)
        self._model_notnow_btn.setStyleSheet(
            "QPushButton { background: transparent; color: white; font-size: 13px; "
            "font-weight: bold; border-radius: 4px; padding: 0 16px; "
            "border: 1px solid white; }"
            "QPushButton:hover { background-color: rgba(255,255,255,0.15); }"
        )
        self._model_notnow_btn.clicked.connect(self._on_model_notnow_clicked)
        action_row.addWidget(self._model_notnow_btn)

        self._model_progress = QProgressBar()
        self._model_progress.setFixedHeight(24)
        self._model_progress.setVisible(False)
        self._model_progress.setStyleSheet(
            "QProgressBar { background-color: rgba(255,255,255,0.25); border: 1px solid white; "
            "border-radius: 4px; color: white; text-align: center; font-weight: bold; }"
            "QProgressBar::chunk { background-color: white; border-radius: 3px; }"
        )
        action_row.addWidget(self._model_progress, 1)

        self._model_gotit_btn = QPushButton("Got it")
        self._model_gotit_btn.setFixedHeight(28)
        self._model_gotit_btn.setVisible(False)
        self._model_gotit_btn.setStyleSheet(
            f"QPushButton {{ background-color: white; color: {BRAND_NOTICE_ORANGE}; font-size: 13px; "
            f"font-weight: bold; border-radius: 4px; padding: 0 16px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_NOTICE_HOVER}; }}"
        )
        self._model_gotit_btn.clicked.connect(lambda: self._model_card.setVisible(False))
        action_row.addWidget(self._model_gotit_btn)

        action_row.addStretch()
        layout.addLayout(action_row)

        self._model_card = card
        self._model_download_url = None
        self._model_download_tag = None
        return card

    @staticmethod
    def _model_display_name(tag):
        """'model-v2' becomes 'Trail Detector v2'. Falls back to the raw tag on parse failure."""
        if not tag:
            return "New model"
        m = re.match(r"^model-v(\d+(?:\.\d+)?)", tag)
        if not m:
            return tag
        num = m.group(1)
        if "." in num:
            num = num.rstrip("0").rstrip(".")
        return f"Trail Detector v{num}"

    def _current_model_display_name(self):
        """Return 'Trail Detector N' for the currently-active model. Empty string on failure."""
        try:
            from modules.model_update import local_model_version
            return self._model_display_name(local_model_version())
        except Exception:
            return ""

    def _start_model_update_check(self):
        self._model_update_thread = ModelUpdateCheckThread(self)
        self._model_update_thread.result_ready.connect(self._on_model_update_result)
        self._model_update_thread.start()

    def _on_model_update_result(self, result):
        self._model_download_tag = result.get("tag", "")
        self._model_download_url = result.get("download_url")
        display = self._model_display_name(self._model_download_tag)
        self._model_title.setText(f"{display} available")
        summary = result.get("summary") or "A new trail detector has been released."
        self._model_summary.setText(summary)
        self._model_summary.setVisible(True)
        credits = result.get("credits") or ""
        if credits:
            self._model_credits.setText(f"Credits: {credits}")
            self._model_credits.setVisible(True)
        else:
            self._model_credits.setVisible(False)
        self._model_download_btn.setVisible(True)
        self._model_notnow_btn.setVisible(True)
        self._model_progress.setVisible(False)
        self._model_gotit_btn.setVisible(False)
        self._model_card.setVisible(True)

    def _on_model_download_clicked(self):
        if not self._model_download_url:
            return
        self._model_download_btn.setVisible(False)
        self._model_notnow_btn.setVisible(False)
        self._model_progress.setRange(0, 100)
        self._model_progress.setValue(0)
        self._model_progress.setFormat("Downloading %p%")
        self._model_progress.setVisible(True)
        from modules.user_folder import get_installed_model_path
        target = str(get_installed_model_path())
        self._model_download_thread = ModelDownloadThread(
            self._model_download_url, target, self._model_download_tag, self
        )
        self._model_download_thread.progress.connect(self._on_model_download_progress)
        self._model_download_thread.finished_ok.connect(self._on_model_download_finished)
        self._model_download_thread.failed.connect(self._on_model_download_failed)
        self._model_download_thread.start()

    def _on_model_notnow_clicked(self):
        self._model_card.setVisible(False)

    def _on_model_download_progress(self, done, total):
        if total > 0:
            pct = int(done * 100 / total)
            self._model_progress.setValue(pct)
        else:
            # Server didn't send Content-Length: fall back to an indeterminate pulse.
            if self._model_progress.minimum() != 0 or self._model_progress.maximum() != 0:
                self._model_progress.setRange(0, 0)

    def _on_model_download_finished(self, version):
        self._model_progress.setVisible(False)
        display = self._model_display_name(version)
        self._model_title.setText(f"{display} installed")
        self._model_summary.setVisible(False)
        self._model_credits.setVisible(False)
        self._model_gotit_btn.setVisible(True)
        self._header_model_label.setText(display)

    def _on_model_download_failed(self, err):
        # Silent fallback: hide the card, try again next launch.
        self._model_card.setVisible(False)

    # ── NVIDIA "coming soon" banner ──────────────────────────────────────────

    def _build_nvidia_banner(self):
        banner = QFrame()
        banner.setFixedHeight(44)
        banner.setStyleSheet(f"QFrame {{ background-color: {BRAND_NOTICE_ORANGE}; }}")
        banner.setVisible(False)
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(16, 0, 8, 0)
        layout.setSpacing(12)

        self._nvidia_label = QLabel(
            "NVIDIA GPU detected. Full GPU support is coming in a future update."
        )
        self._nvidia_label.setStyleSheet(
            "color: white; font-size: 14px; font-weight: bold; background: transparent;"
        )
        layout.addWidget(self._nvidia_label)
        layout.addStretch()

        gotit_btn = QPushButton("Got it")
        gotit_btn.setFixedHeight(28)
        gotit_btn.setStyleSheet(
            f"QPushButton {{ background-color: white; color: {BRAND_NOTICE_ORANGE}; font-size: 13px; "
            f"font-weight: bold; border-radius: 4px; padding: 0 16px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_NOTICE_HOVER}; }}"
        )
        gotit_btn.clicked.connect(self._on_nvidia_gotit_clicked)
        layout.addWidget(gotit_btn)

        self._nvidia_banner = banner
        return banner

    def _start_nvidia_detect(self):
        if SETTINGS.value("nvidia_coming_soon_dismissed", False, type=bool):
            return
        self._nvidia_thread = NvidiaDetectThread(self)
        self._nvidia_thread.result_ready.connect(self._on_nvidia_detect_result)
        self._nvidia_thread.start()

    def _on_nvidia_detect_result(self, outcome, detail):
        print(f"[nvidia-detect] outcome={outcome} detail={detail}", flush=True)
        if outcome == "yes":
            self._nvidia_banner.setVisible(True)

    def _on_nvidia_gotit_clicked(self):
        SETTINGS.setValue("nvidia_coming_soon_dismissed", True)
        self._nvidia_banner.setVisible(False)

    def _relaunch(self):
        """Close and reopen the app."""
        import subprocess
        if getattr(sys, 'frozen', False):
            subprocess.Popen([sys.executable])
        else:
            subprocess.Popen([sys.executable, os.path.abspath(__file__)])
        self.close()

    # ── Setup page ───────────────────────────────────────────────────────────

    def _build_setup_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(6)
        layout.setContentsMargins(24, 20, 24, 20)

        lbl_font = QFont()
        lbl_font.setPointSize(15)
        lbl_font.setBold(True)

        step_font = QFont()
        step_font.setPointSize(12)

        # Headline + subtitle
        headline = QLabel("Remove the Trails. Keep the Stars.")
        headline_font = QFont()
        headline_font.setPointSize(22)
        headline_font.setBold(True)
        headline.setFont(headline_font)
        layout.addWidget(headline)

        subtitle = QLabel(
            "Drop in a folder of star trail frames and let the AI scrub out "
            "airplane and satellite streaks."
        )
        sub_font = QFont()
        sub_font.setPointSize(13)
        subtitle.setFont(sub_font)
        subtitle.setStyleSheet(f"color: {MUTED_TEXT};")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(16)

        # ── Step 1: Select Images ────────────────────────────────────────────
        step1 = QLabel("1. Select Star Trail Images")
        step1.setFont(lbl_font)
        layout.addWidget(step1)

        hint1 = QLabel("(.JPG, .TIF \u2014 8 & 16 bit files accepted)")
        hint1.setFont(step_font)
        hint1.setStyleSheet(f"color: {MUTED_TEXT};")
        layout.addWidget(hint1)

        row_in = QHBoxLayout()
        self._folder_input = QLineEdit()
        self._folder_input.setPlaceholderText("Select folder using Browse\u2026")
        self._folder_input.textChanged.connect(self._auto_output)
        self._folder_input.textChanged.connect(self._update_input_open_btn_state)
        self._folder_input.editingFinished.connect(self._on_input_edited)
        row_in.addWidget(self._folder_input, 4)
        browse_in = QPushButton("Browse\u2026")
        browse_in.clicked.connect(self._browse_input)
        row_in.addWidget(browse_in, 1)
        self._input_open_btn = QPushButton("Open Folder")
        self._input_open_btn.setEnabled(False)
        self._input_open_btn.clicked.connect(self._open_setup_input_folder)
        row_in.addWidget(self._input_open_btn, 1)
        layout.addLayout(row_in)

        self._frame_count_label = QLabel("")
        self._frame_count_label.setFont(step_font)
        self._frame_count_label.setStyleSheet(f"color: {MUTED_TEXT};")
        layout.addWidget(self._frame_count_label)
        layout.addSpacing(10)

        # ── Step 2: Select Output ────────────────────────────────────────────
        step2 = QLabel("2. Select Output Folder")
        step2.setFont(lbl_font)
        layout.addWidget(step2)

        hint2 = QLabel("(default: a \u2018Cleaned\u2019 folder inside your originals)")
        hint2.setFont(step_font)
        hint2.setStyleSheet(f"color: {MUTED_TEXT};")
        layout.addWidget(hint2)

        row_out = QHBoxLayout()
        self._output_input = QLineEdit()
        self._output_input.setPlaceholderText("Auto-fills from input folder")
        self._output_input.textChanged.connect(self._update_output_open_btn_state)
        row_out.addWidget(self._output_input, 4)
        browse_out = QPushButton("Browse\u2026")
        browse_out.clicked.connect(self._browse_output)
        row_out.addWidget(browse_out, 1)
        self._output_open_btn = QPushButton("Open Folder")
        self._output_open_btn.setEnabled(False)
        self._output_open_btn.clicked.connect(self._open_setup_output_folder)
        row_out.addWidget(self._output_open_btn, 1)
        layout.addLayout(row_out)
        layout.addSpacing(10)

        # ── Step 3: Foreground Mask ──────────────────────────────────────────
        step3 = QLabel("3. Foreground Mask (optional)")
        step3.setFont(lbl_font)
        layout.addWidget(step3)

        hint3 = QLabel("Not required, but helpful \u2014 keeps the AI focused on the sky")
        hint3.setFont(step_font)
        hint3.setStyleSheet(f"color: {MUTED_TEXT};")
        layout.addWidget(hint3)

        mask_row = QHBoxLayout()
        self._mask_btn = QPushButton("Create Mask\u2026")
        self._mask_btn.setFixedHeight(34)
        self._mask_btn.setFixedWidth(160)
        self._mask_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BRAND_RUN_GREEN}; color: white; font-size: 13px; "
            f"font-weight: bold; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_RUN_GREEN_HOVER}; }}"
        )
        self._mask_btn.clicked.connect(self._open_mask_editor)
        mask_row.addWidget(self._mask_btn)

        self._mask_status = QLabel("No mask")
        self._mask_status.setStyleSheet(f"color: {HINT_TEXT}; font-size: 12px; margin-left: 8px;")
        mask_row.addWidget(self._mask_status)
        mask_row.addStretch()
        layout.addLayout(mask_row)
        layout.addSpacing(10)

        # ── Step 4: Number of Images ─────────────────────────────────────────
        step4 = QLabel("4. Number of Images to Process")
        step4.setFont(lbl_font)
        layout.addWidget(step4)

        hint4 = QLabel("Recommended: test a small batch before doing a full run")
        hint4.setFont(step_font)
        hint4.setStyleSheet(f"color: {MUTED_TEXT};")
        layout.addWidget(hint4)

        self._frame_limit = QComboBox()
        self._frame_limit.addItems(["20", "50", "100", "250", "All Frames"])
        self._frame_limit.setFixedWidth(140)
        layout.addWidget(self._frame_limit)
        layout.addSpacing(10)

        # ── Step 5: Output Options ───────────────────────────────────────────
        step5 = QLabel("5. Output Options")
        step5.setFont(lbl_font)
        layout.addWidget(step5)

        hint5 = QLabel("File format and quality")
        hint5.setFont(step_font)
        hint5.setStyleSheet(f"color: {MUTED_TEXT};")
        layout.addWidget(hint5)

        hp_row = QHBoxLayout()

        fmt_label = QLabel("Output format:")
        fmt_label.setFont(step_font)
        hp_row.addWidget(fmt_label)
        self._format_combo = QComboBox()
        self._format_combo.addItems(["JPG", "TIFF 8-bit", "TIFF 16-bit"])
        self._format_combo.setFixedWidth(130)
        hp_row.addWidget(self._format_combo)
        hp_row.addSpacing(12)

        self._jpeg_quality_label = QLabel("JPEG quality:")
        self._jpeg_quality_label.setFont(step_font)
        hp_row.addWidget(self._jpeg_quality_label)
        self._jpeg_quality = QSpinBox()
        self._jpeg_quality.setRange(60, 100)
        self._jpeg_quality.setSingleStep(5)
        self._jpeg_quality.setValue(95)
        self._jpeg_quality.setFixedWidth(55)
        hp_row.addWidget(self._jpeg_quality)
        hp_row.addStretch(1)

        self._format_combo.currentTextChanged.connect(self._on_format_changed)

        layout.addLayout(hp_row)
        layout.addSpacing(14)

        # ── Step 6: Run ──────────────────────────────────────────────────────
        step6 = QLabel("6. Remove airplane and satellite trails")
        step6.setFont(lbl_font)
        layout.addWidget(step6)

        hint6 = QLabel("Processing time depends on image count, resolution, and computer speed")
        hint6.setFont(step_font)
        hint6.setStyleSheet(f"color: {MUTED_TEXT};")
        layout.addWidget(hint6)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: red; font-size: 13px;")
        layout.addWidget(self._error_label)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignBottom)
        btn_row.setSpacing(16)

        self._run_btn = QPushButton("Clean My Stars!")
        self._run_btn.setFixedHeight(60)
        self._run_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BRAND_RUN_GREEN}; color: white; font-size: 22px; "
            f"font-weight: bold; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_RUN_GREEN_HOVER}; }}"
            f"QPushButton:disabled {{ background-color: {DISABLED_BTN_BG}; }}"
        )
        self._run_btn.clicked.connect(self._run)
        btn_row.addWidget(self._run_btn, 2)

        self._setup_open_btn = QPushButton("Open Cleaned Folder")
        self._setup_open_btn.setFixedHeight(48)
        self._setup_open_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BRAND_HEADING_BLUE}; color: white; font-size: 18px; "
            f"font-weight: bold; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_HEADING_HOVER}; }}"
            f"QPushButton:disabled {{ background-color: {DISABLED_BTN_BG}; }}"
        )
        self._setup_open_btn.setEnabled(False)
        self._setup_open_btn.clicked.connect(self._open_output_from_setup)
        btn_row.addWidget(self._setup_open_btn, 1)
        layout.addLayout(btn_row)

        self._stack.addWidget(page)

        # Restore last used folder
        last_dir = SETTINGS.value("last_input_dir", "")
        if last_dir and os.path.isdir(last_dir):
            self._folder_input.setText(last_dir)
            self._last_input_seen = last_dir

        # Restore persisted widget state
        fmt = SETTINGS.value("output_format", "JPG")
        idx = self._format_combo.findText(fmt)
        if idx >= 0:
            self._format_combo.setCurrentIndex(idx)
        self._jpeg_quality.setValue(
            int(SETTINGS.value("jpeg_quality", 95)))
        self._on_format_changed(self._format_combo.currentText())
        last_frame_limit = SETTINGS.value("frame_limit", "20")
        fli = self._frame_limit.findText(last_frame_limit)
        if fli >= 0:
            self._frame_limit.setCurrentIndex(fli)

        # Persist on change
        self._format_combo.currentTextChanged.connect(
            lambda t: SETTINGS.setValue("output_format", t))
        self._jpeg_quality.valueChanged.connect(
            lambda v: SETTINGS.setValue("jpeg_quality", int(v)))
        self._frame_limit.currentTextChanged.connect(
            lambda t: SETTINGS.setValue("frame_limit", t))

        # Check for existing mask
        self._update_mask_status()
        self._update_frame_count()

    # ── Process page ─────────────────────────────────────────────────────────

    def _build_process_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        title = QLabel("Cleaning in Progress")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        self._process_title = title
        layout.addWidget(title)

        # ── Overall progress bar (fat) ──
        frame_label_row = QHBoxLayout()
        self._frame_counter = QLabel("")
        self._frame_counter.setStyleSheet(f"font-size: 13px; color: {MUTED_TEXT};")
        self._frame_counter.setTextInteractionFlags(Qt.TextSelectableByMouse)
        frame_label_row.addWidget(self._frame_counter)
        frame_label_row.addSpacing(20)
        self._initial_est_label = QLabel("")
        self._initial_est_label.setStyleSheet(f"font-size: 13px; color: {MUTED_TEXT};")
        self._initial_est_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        frame_label_row.addWidget(self._initial_est_label)
        frame_label_row.addStretch()
        layout.addLayout(frame_label_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(36)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {CARD_BORDER}; border-radius: 10px; "
            f"background: {CARD_BG}; text-align: center; font-weight: bold; font-size: 14px; color: {CARD_TEXT}; }}"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #4a9eff, stop:1 #66b3ff); border-radius: 9px; }"
        )
        layout.addWidget(self._progress_bar)

        # ── Time estimate + elapsed ──
        time_row = QHBoxLayout()
        self._time_label = QLabel("")
        self._time_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        time_row.addWidget(self._time_label)
        time_row.addStretch()
        self._elapsed_label = QLabel("")
        self._elapsed_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        time_row.addWidget(self._elapsed_label)
        layout.addLayout(time_row)

        # ── Batch label with spinner ──
        self._batch_label = QLabel("")
        batch_font = QFont()
        batch_font.setPointSize(15)
        batch_font.setBold(True)
        self._batch_label.setFont(batch_font)
        self._batch_label.setAlignment(Qt.AlignCenter)
        self._batch_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._batch_label)

        # ── Divider ──
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)

        # ── Step 1: Detecting ──
        step1_row = QHBoxLayout()
        self._step1_label = QLabel("Detecting\nwaiting")
        self._step1_label.setFixedWidth(120)
        self._step1_label.setStyleSheet(f"font-size: 12px; color: {HINT_TEXT};")
        self._step1_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        step1_row.addWidget(self._step1_label)
        self._step1_bar = QProgressBar()
        self._step1_bar.setFixedHeight(30)
        self._step1_bar.setTextVisible(True)
        self._step1_bar.setValue(0)
        self._step1_bar.setFormat("0%")
        self._step1_bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {CARD_BORDER}; border-radius: 8px; "
            f"background: {CARD_BG}; text-align: center; font-weight: bold; font-size: 13px; color: {CARD_TEXT}; }}"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #4a9eff, stop:1 #66b3ff); border-radius: 7px; }"
        )
        step1_row.addWidget(self._step1_bar, 1)
        layout.addLayout(step1_row)

        # ── Step 2: Repair ──
        step2_row = QHBoxLayout()
        self._step2_label = QLabel("Repairing\nwaiting")
        self._step2_label.setFixedWidth(120)
        self._step2_label.setStyleSheet(f"font-size: 12px; color: {HINT_TEXT};")
        self._step2_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        step2_row.addWidget(self._step2_label)
        self._step2_bar = QProgressBar()
        self._step2_bar.setFixedHeight(30)
        self._step2_bar.setTextVisible(True)
        self._step2_bar.setValue(0)
        self._step2_bar.setFormat("0%")
        self._step2_bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {CARD_BORDER}; border-radius: 8px; "
            f"background: {CARD_BG}; text-align: center; font-weight: bold; font-size: 13px; color: {CARD_TEXT}; }}"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #4a9eff, stop:1 #66b3ff); border-radius: 7px; }"
        )
        step2_row.addWidget(self._step2_bar, 1)
        layout.addLayout(step2_row)

        # ── Detail / status line ──
        self._detail_label = QLabel("")
        self._detail_label.setAlignment(Qt.AlignCenter)
        self._detail_label.setStyleSheet(f"font-size: 12px; color: {BRAND_HEADING_BLUE};")
        self._detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._detail_label)

        # ── Post-run stats card (hidden until run completes) ──
        self._stats_label = QLabel("")
        self._stats_label.setAlignment(Qt.AlignCenter)
        self._stats_label.setWordWrap(True)
        self._stats_label.setTextFormat(Qt.RichText)
        self._stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._stats_label.setStyleSheet(
            f"QLabel {{ background-color: {LIGHT_PANEL_BG}; border: 1px solid {BRAND_HEADING_BLUE}; "
            f"border-radius: 6px; padding: 12px; color: {CARD_TEXT}; font-size: 15px; }}"
        )
        self._stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._stats_label.hide()
        layout.addWidget(self._stats_label)

        # ── Log area ──
        self._status_out = QTextEdit()
        self._status_out.setReadOnly(True)
        layout.addWidget(self._status_out, 1)

        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignBottom)
        btn_row.setSpacing(16)

        self._cancel_btn = QPushButton("Cancel Cleaning")
        self._cancel_btn.setFixedHeight(60)
        self._cancel_btn.setStyleSheet(
            f"QPushButton {{ background-color: {SECONDARY_BTN_BG}; color: white; font-size: 22px; "
            f"font-weight: bold; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: {DISABLED_BTN_HOVER}; }}"
        )
        self._cancel_btn.clicked.connect(self._cancel_run)
        btn_row.addWidget(self._cancel_btn, 2)

        self._open_folder_btn = QPushButton("Open Cleaned Folder")
        self._open_folder_btn.setFixedHeight(48)
        self._open_folder_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BRAND_HEADING_BLUE}; color: white; font-size: 18px; "
            f"font-weight: bold; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_HEADING_HOVER}; }}"
        )
        self._open_folder_btn.clicked.connect(self._open_output_folder)
        btn_row.addWidget(self._open_folder_btn, 1)

        layout.addLayout(btn_row)

        self._stack.addWidget(page)

    # ── Browse / validation ──────────────────────────────────────────────────

    def _browse_input(self):
        last_dir = SETTINGS.value("last_input_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder", last_dir)
        if folder:
            self._folder_input.setText(folder)
            SETTINGS.setValue("last_input_dir", folder)
            self._update_mask_status()
            self._update_frame_count()

    def _on_input_edited(self):
        self._update_frame_count()

    def _update_input_open_btn_state(self):
        path = self._folder_input.text().strip()
        self._input_open_btn.setEnabled(bool(path) and os.path.isdir(path))

    def _update_output_open_btn_state(self):
        path = self._output_input.text().strip()
        self._output_open_btn.setEnabled(bool(path) and os.path.isdir(path))

    def _open_setup_input_folder(self):
        path = self._folder_input.text().strip()
        if path and os.path.isdir(path):
            if sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.run(["open", path])

    def _open_setup_output_folder(self):
        path = self._output_input.text().strip()
        if path and os.path.isdir(path):
            if sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.run(["open", path])

    def _update_frame_count(self):
        folder = self._folder_input.text().strip()
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        count = None
        if not folder:
            self._frame_count_label.setText("")
        elif not os.path.isdir(folder):
            self._frame_count_label.setText("Folder not found")
        else:
            try:
                count = sum(1 for n in os.listdir(folder)
                            if os.path.splitext(n)[1].lower() in exts)
            except OSError:
                count = 0
            if count == 0:
                self._frame_count_label.setText("No images found")
            else:
                self._frame_count_label.setText(
                    f"<b>{count:,}</b> frame{'s' if count != 1 else ''} found")

        model = self._frame_limit.model()
        for i in range(self._frame_limit.count()):
            text = self._frame_limit.itemText(i)
            if text == "All Frames" or count is None:
                enabled = True
            else:
                try:
                    enabled = int(text) <= count
                except ValueError:
                    enabled = True
            item = model.item(i)
            if item is not None:
                flags = item.flags()
                if enabled:
                    item.setFlags(flags | Qt.ItemIsEnabled)
                else:
                    item.setFlags(flags & ~Qt.ItemIsEnabled)
            self._frame_limit.view().setRowHidden(i, not enabled)

    def _browse_output(self):
        last_dir = SETTINGS.value("last_output_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", last_dir)
        if folder:
            self._output_input.setText(folder)
            SETTINGS.setValue("last_output_dir", folder)

    def _auto_output(self, text):
        if text and text.strip():
            self._output_input.setText(os.path.join(text.strip(), "cleaned"))
            self._update_mask_status()
            self._update_open_btn_state()

    def _update_mask_status(self):
        """Check if a mask exists for the current input folder."""
        folder = self._folder_input.text().strip()
        if folder:
            migrate_workspace(folder)
            mask_path = os.path.join(folder, WORKSPACE_DIR, "foreground_mask.png")
        else:
            mask_path = ""
        if mask_path and os.path.exists(mask_path):
            self._mask_path = mask_path
            self._mask_status.setText("\u2705 Mask saved")
            self._mask_status.setStyleSheet(f"color: {SUCCESS_TEXT}; font-size: 12px; margin-left: 8px;")
            self._mask_btn.setText("Edit Mask\u2026")
        else:
            self._mask_path = None
            self._mask_status.setText("No mask")
            self._mask_status.setStyleSheet(f"color: {HINT_TEXT}; font-size: 12px; margin-left: 8px;")
            self._mask_btn.setText("Create Mask\u2026")

    # ── Mask editor ──────────────────────────────────────────────────────────

    def _open_mask_editor(self):
        folder = self._folder_input.text().strip()
        if not folder or not os.path.isdir(folder):
            self._error_label.setText("Select an input folder first (Step 1).")
            return
        self._error_label.setText("")

        # Find first image
        exts = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff",
                "*.JPG", "*.JPEG", "*.PNG", "*.TIF", "*.TIFF"]
        frames = sorted(set(
            f for e in exts for f in glob.glob(os.path.join(folder, e))
        ))
        if not frames:
            self._error_label.setText("No image files found in the selected folder.")
            return

        # Create mask window if needed, or reuse
        if self._mask_window is None:
            self._mask_window = MaskEditorWindow(self)
            self._mask_window.mask_saved.connect(self._on_mask_saved)

        self._mask_window.load_image(frames[0])

        # Load existing mask if available
        migrate_workspace(folder)
        mask_path = os.path.join(folder, WORKSPACE_DIR, "foreground_mask.png")
        if os.path.exists(mask_path):
            self._mask_window.load_existing_mask(mask_path)

        self._mask_window.show()
        self._mask_window.raise_()
        self._mask_window.activateWindow()

    def _on_mask_saved(self, mask_np):
        folder = self._folder_input.text().strip()
        if folder:
            if mask_np.any():
                mask_path = workspace_path(folder, "foreground_mask.png")
                cv2.imwrite(mask_path, mask_np)
                self._mask_path = mask_path
            else:
                mask_path = os.path.join(folder, WORKSPACE_DIR, "foreground_mask.png")
                if os.path.exists(mask_path):
                    os.remove(mask_path)
                self._mask_path = None
            self._update_mask_status()

    # ── Run ──────────────────────────────────────────────────────────────────

    def _validate(self):
        folder = self._folder_input.text().strip()
        if not folder:
            self._error_label.setText("Please select an input folder (Step 1).")
            return None
        if not os.path.isdir(folder):
            self._error_label.setText(f"Folder not found: {folder}")
            return None
        output = self._output_input.text().strip()
        if not output:
            self._error_label.setText("Please select an output folder (Step 2).")
            return None
        self._error_label.setText("")
        return folder, output

    def _run(self):
        result = self._validate()
        if not result:
            return
        folder, output = result
        SETTINGS.setValue("last_input_dir", folder)

        # Go to process page — reset all widgets
        self._process_title.setText("Cleaning in Progress")
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {CARD_BORDER}; border-radius: 10px; "
            f"background: {CARD_BG}; text-align: center; font-weight: bold; font-size: 14px; color: {CARD_TEXT}; }}"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #4a9eff, stop:1 #66b3ff); border-radius: 9px; }"
        )
        self._frame_counter.setText("Scrubbing the stars\u2026")
        self._initial_est_label.setText("")
        self._time_label.setText("")
        self._elapsed_label.setText("")
        self._batch_label.setText("")
        self._step1_bar.setValue(0)
        self._step1_bar.setFormat("0%")
        self._step1_label.setText("Detecting\nwaiting")
        self._step2_bar.setValue(0)
        self._step2_bar.setFormat("0%")
        self._step2_label.setText("Repairing\nwaiting")
        self._detail_label.setText("")
        self._stats_label.setText("")
        self._stats_label.hide()
        self._status_out.setText("")
        self._cancel_btn.setText("Cancel Cleaning")
        self._cancel_btn.setEnabled(True)
        try:
            self._cancel_btn.clicked.disconnect()
        except RuntimeError:
            pass
        self._cancel_btn.clicked.connect(self._cancel_run)
        self._cancel_btn.show()
        self._stack.setCurrentIndex(1)

        self._spinner_chars = "|/-\\"
        self._spinner_idx = 0
        self._has_estimate = False
        self._batch_text = ""

        # Start spinner + elapsed timer (handles both)
        self._run_start_time = time.time()
        self._spinner_timer = QTimer(self)
        self._spinner_timer.timeout.connect(self._update_spinner)
        self._spinner_timer.start(250)

        # Set output folder now so the "Open Cleaned Folder" button works during the run
        self._done_output_folder = output

        fmt_map = {"JPG": "jpg", "TIFF 8-bit": "tif8", "TIFF 16-bit": "tif16"}
        out_fmt = fmt_map.get(self._format_combo.currentText(), "jpg")
        self.worker = CleanerWorker(
            folder, output, self._frame_limit.currentText(), self._mask_path,
            output_format=out_fmt,
            jpeg_quality=self._jpeg_quality.value())
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self._on_status)
        self.worker.batch_info.connect(self._on_batch_info)
        self.worker.step_progress.connect(self._on_step_progress)
        self.worker.step_detail.connect(self._on_step_detail)
        self.worker.frame_count.connect(self._on_frame_count)
        self.worker.stats_ready.connect(self._on_stats_ready)
        self.worker.timing_stats.connect(self._on_timing_stats)
        self.worker.initial_estimate.connect(self._on_initial_estimate)
        self.worker.error.connect(self._on_error)
        self.worker.done.connect(self._on_done)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _cancel_run(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._cancel_btn.setText("Cancelling\u2026")
            # Stop progress bar animation immediately
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(0)
            self._progress_bar.setFormat("Cancelled")
            self._stop_elapsed_timer()
            # Wait briefly then force cleanup
            self.worker.wait(3000)
            if self.worker.isRunning():
                self.worker.terminate()
            self._cancel_btn.setText("Back to Setup")
            self._cancel_btn.setEnabled(True)
            try:
                self._cancel_btn.clicked.disconnect()
            except RuntimeError:
                pass
            self._cancel_btn.clicked.connect(self._go_to_setup)
            self._status_out.append("\nCleaning cancelled.")

    def _stop_elapsed_timer(self):
        if hasattr(self, '_spinner_timer') and self._spinner_timer.isActive():
            self._spinner_timer.stop()

    def _go_to_setup(self):
        self._stop_elapsed_timer()
        self._stack.setCurrentIndex(0)

    def _update_spinner(self):
        self._spinner_idx += 1
        ch = self._spinner_chars[self._spinner_idx % len(self._spinner_chars)]
        elapsed = time.time() - self._run_start_time if hasattr(self, '_run_start_time') else 0
        m, s = divmod(int(elapsed), 60)
        self._elapsed_label.setText(f"{ch}  Elapsed: {m}m {s:02d}s")


        # Pulse "Estimating..." before real estimate arrives
        if not getattr(self, '_has_estimate', False):
            dots = "." * ((self._spinner_idx % 3) + 1)
            self._time_label.setText(f"Estimating{dots}")

    def _on_progress(self, pct, total, remaining_str):
        self._progress_bar.setRange(0, 100)
        pct = max(0, min(100, pct))
        self._progress_bar.setValue(pct)

        if remaining_str:
            self._time_label.setText(f"\u23f1 ~{remaining_str} remaining")
            self._has_estimate = True

    def _on_batch_info(self, batch_num, n_batches):
        self._batch_text = f"Batch {batch_num} of {n_batches}"
        self._batch_label.setText(self._batch_text)
        # Reset step bars for new batch
        self._step1_bar.setValue(0)
        self._step1_bar.setFormat("0%")
        self._step1_label.setText("Detecting\nwaiting")
        step1_style = (
            f"QProgressBar {{ border: 1px solid {CARD_BORDER}; border-radius: 8px; "
            f"background: {CARD_BG}; text-align: center; font-weight: bold; font-size: 13px; color: {CARD_TEXT}; }}"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #4a9eff, stop:1 #66b3ff); border-radius: 7px; }"
        )
        self._step1_bar.setStyleSheet(step1_style)
        self._step2_bar.setValue(0)
        self._step2_bar.setFormat("0%")
        self._step2_label.setText("Repairing\nwaiting")
        self._step2_bar.setStyleSheet(step1_style)

    def _on_step_progress(self, step, current, total):
        green_style = (
            f"QProgressBar {{ border: 1px solid {CARD_BORDER}; border-radius: 8px; "
            f"background: {CARD_BG}; text-align: center; font-weight: bold; font-size: 13px; color: {CARD_TEXT}; }}"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #34c759, stop:1 #5dd87a); border-radius: 7px; }"
        )
        pct = int(current / total * 100) if total > 0 else 0
        if step == 1:
            self._step1_bar.setValue(pct)
            if pct >= 100:
                self._step1_bar.setFormat("100%")
                self._step1_label.setText("Detecting\ncomplete")
                self._step1_bar.setStyleSheet(green_style)
            else:
                self._step1_bar.setFormat(f"{pct}%")
                self._step1_label.setText(f"Detecting\nframe {current}/{total}")
        elif step == 2:
            self._step2_bar.setValue(pct)
            if pct >= 100:
                self._step2_bar.setFormat("100%")
                self._step2_label.setText("Repairing\ncomplete")
                self._step2_bar.setStyleSheet(green_style)
            else:
                self._step2_bar.setFormat(f"{pct}%")
                self._step2_label.setText(f"Repairing\nframe {current}/{total}")

    def _on_step_detail(self, text):
        self._detail_label.setText(text)

    def _on_frame_count(self, current, total):
        self._frame_counter.setText(f"Scrubbing the stars\u2026 {current} of {total}")

    def _on_initial_estimate(self, seconds):
        m, s = divmod(int(round(seconds)), 60)
        if m:
            self._initial_est_label.setText(f"Estimated Time: {m}m {s:02d}s")
        else:
            self._initial_est_label.setText(f"Estimated Time: {s}s")

    def _on_format_changed(self, text):
        is_jpg = text == "JPG"
        self._jpeg_quality.setEnabled(is_jpg)
        self._jpeg_quality_label.setEnabled(is_jpg)

    def _on_stats_ready(self, total_trails, total_frames):
        if total_trails <= 0:
            return
        SECONDS_PER_MANUAL_TRAIL = 20
        saved_sec = total_trails * SECONDS_PER_MANUAL_TRAIL
        if saved_sec >= 60:
            rounded_min = int(round(saved_sec / 900.0) * 15)
            if rounded_min < 15:
                rounded_min = 15
            h = rounded_min // 60
            m = rounded_min % 60
            if h >= 1 and m == 0:
                time_saved = f"~{h} hour{'s' if h != 1 else ''}"
            elif h >= 1:
                time_saved = f"~{h} hour{'s' if h != 1 else ''} {m} minute{'s' if m != 1 else ''}"
            else:
                time_saved = f"~{m} minute{'s' if m != 1 else ''}"
        else:
            time_saved = f"~{saved_sec} second{'s' if saved_sec != 1 else ''}"
        self._stats_trail_line = (
            f"Swept <b>{total_trails:,}</b> airplane and satellite trails from your skies "
            f"across <b>{total_frames:,}</b> frames.<br>"
            f"<i>Based on manual cleanup at 20 seconds per trail.</i><br><br>"
            f"<span style='font-size:20px; font-weight:bold;'>TIME SAVED: {time_saved}</span>"
            f"<br><br><b>Time to stack!</b><br>"
            f"Open the Cleaned Folder, then load the frames into your favorite "
            f"stacker (StarStaX, Sequator, Photoshop) for the final composite."
        )
        self._stats_label.setText(
            self._stats_trail_line + getattr(self, '_stats_timing_line', ''))
        self._stats_label.show()

    def _on_timing_stats(self, initial_est_sec, actual_sec):
        tail = "You're welcome." if actual_sec <= initial_est_sec else "My apologies."
        self._stats_timing_line = (
            f"<br><br><span style='font-size:14px; color:{MUTED_TEXT};'>"
            f"Thought it'd take <b>{fmt_hms(initial_est_sec)}</b>. "
            f"Took <b>{fmt_hms(actual_sec)}</b>. {tail}"
            f"</span>"
        )
        if hasattr(self, '_stats_trail_line'):
            self._stats_label.setText(self._stats_trail_line + self._stats_timing_line)
            self._stats_label.show()

    def _on_status(self, text):
        self._status_out.append(text)
        sb = self._status_out.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _switch_to_back_btn(self):
        self._cancel_btn.setText("Back to Setup")
        self._cancel_btn.setEnabled(True)
        try:
            self._cancel_btn.clicked.disconnect()
        except RuntimeError:
            pass
        self._cancel_btn.clicked.connect(self._go_to_setup)

    def _on_error(self, msg):
        self._stop_elapsed_timer()
        self._status_out.setText(f"ERROR: {msg}")
        self._switch_to_back_btn()

    def _on_done(self, output_folder):
        # Bounce the Dock icon (Mac) or flash the taskbar button (Windows) to
        # get the user's attention if they've switched to another app. No-op
        # if the window is currently in focus.
        QApplication.alert(self)
        self._stop_elapsed_timer()
        self._process_title.setText("Cleaning Complete")
        self._progress_bar.setValue(100)
        self._progress_bar.setFormat("Complete!")
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {CARD_BORDER}; border-radius: 10px; "
            f"background: {CARD_BG}; text-align: center; font-weight: bold; font-size: 14px; color: {CARD_TEXT}; }}"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #34c759, stop:1 #5dd87a); border-radius: 9px; }"
        )
        self._time_label.setText("")
        self._batch_label.setText(getattr(self, '_batch_text', ''))
        self._done_output_folder = output_folder
        self._update_open_btn_state()
        self._switch_to_back_btn()

    def _update_open_btn_state(self):
        folder = self._output_input.text().strip()
        has_files = False
        if folder and os.path.isdir(folder):
            has_files = any(os.scandir(folder))
        self._setup_open_btn.setEnabled(has_files)

    def _open_output_from_setup(self):
        folder = self._output_input.text().strip()
        if not folder:
            folder = getattr(self, '_done_output_folder', None)
        if not folder:
            self._error_label.setText("No output folder set \u2014 select an input folder first.")
            return
        if not os.path.isdir(folder):
            self._error_label.setText(f"Output folder doesn\u2019t exist yet \u2014 run cleaning first.")
            return
        if sys.platform == "win32":
            os.startfile(folder)
        else:
            subprocess.run(["open", folder])

    def _open_output_folder(self):
        folder = getattr(self, '_done_output_folder', None)
        if folder and os.path.isdir(folder):
            if sys.platform == "win32":
                os.startfile(folder)
            else:
                subprocess.run(["open", folder])

    def _on_finished(self):
        self._stop_elapsed_timer()
        self._switch_to_back_btn()

    def closeEvent(self, event):
        """Save window size and clean up worker thread before closing."""
        SETTINGS.setValue("window_geometry", self.saveGeometry())
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        event.accept()


class MaskEditorWindow(QMainWindow):
    """Separate window for mask painting — closes back to main setup."""
    mask_saved = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Star Trail CleanR \u2014 Foreground Mask")
        self._painter = MaskPainterWidget()
        self.setCentralWidget(self._painter)

        # Connect signals
        self._painter.mask_done.connect(self._on_done)
        self._painter.mask_skipped.connect(self.close)
        self._painter.go_back.connect(self.close)

        # Size to 90% of screen
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            w = int(geom.width() * 0.9)
            h = int(geom.height() * 0.9)
            self.resize(w, h)
            x = geom.x() + (geom.width() - w) // 2
            y = geom.y() + (geom.height() - h) // 2
            self.move(x, y)

    def load_image(self, img_path: str):
        self._painter.load_image(img_path)

    def load_existing_mask(self, mask_path: str):
        self._painter.load_existing_mask(mask_path)

    def _on_done(self, mask_np):
        self.mask_saved.emit(mask_np)
        self.close()


if __name__ == '__main__':
    # Cross-platform single-instance check (Mac, Windows, Linux)
    import socket
    _lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _lock_socket.bind(('127.0.0.1', 49173))
    except OSError:
        from PySide6.QtWidgets import QMessageBox
        app = QApplication(sys.argv)
        QMessageBox.warning(None, "Star Trail CleanR",
                            "Star Trail CleanR is already running.")
        sys.exit(1)

    app = QApplication(sys.argv)

    # Set the Dock / taskbar icon. Same icon file the frozen build embeds,
    # so dev-mode (live Python via the AppleScript wrapper) also gets the
    # proper Star Trail CleanR icon instead of the Python launcher's rocket.
    if getattr(sys, 'frozen', False):
        _icon_base = sys._MEIPASS
    else:
        _icon_base = os.path.dirname(os.path.abspath(__file__))
    _icon_ext = '.ico' if sys.platform == 'win32' else '.icns'
    _icon_path = os.path.join(_icon_base, 'assets', 'StarTrailCleanR' + _icon_ext)
    if os.path.exists(_icon_path):
        app.setWindowIcon(QIcon(_icon_path))

    _apply_theme()

    # First-run crash-reporting opt-in. Asked once; choice persists in
    # QSettings. Only shown when a DSN is actually present (CI builds), so
    # dev runs don't see the prompt at all.
    if _SENTRY_DSN and not SETTINGS.contains("crash_reporting_choice_made"):
        from PySide6.QtWidgets import QMessageBox
        prompt = QMessageBox()
        prompt.setWindowTitle("Star Trail CleanR")
        prompt.setIcon(QMessageBox.Question)
        prompt.setText("Help improve Star Trail CleanR by sending anonymous crash reports?")
        prompt.setInformativeText(
            "If the app ever crashes, an automatic error report is sent so the bug "
            "can be fixed.\n\nThe report contains a stack trace, your operating "
            "system, and the app version. No images, no folder paths, no personal "
            "information."
        )
        prompt.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        prompt.setDefaultButton(QMessageBox.Yes)
        choice = prompt.exec()
        SETTINGS.setValue("crash_reporting_enabled", choice == QMessageBox.Yes)
        SETTINGS.setValue("crash_reporting_choice_made", True)

    _maybe_init_sentry()

    window = MainWindow()
    window.show()

    # Live OS appearance switching: when the user toggles macOS Light/Dark
    # mid-session, relaunch so every themed widget rebuilds with the new
    # palette. QSettings preserves folder selections and options, so the
    # user lands right back where they were.
    def _on_color_scheme_changed(_scheme):
        try:
            window._relaunch()
        except Exception:
            pass
    try:
        app.styleHints().colorSchemeChanged.connect(_on_color_scheme_changed)
    except Exception:
        pass

    sys.exit(app.exec())
