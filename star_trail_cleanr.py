import sys
import os
import re

# Prevent torch-on-Windows crash when another Python install has its own
# libiomp5md.dll on PATH. Must be set BEFORE any torch-touching import so it
# also propagates to the worker subprocess that re-runs this script.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Apple Silicon: torchvision::nms is not implemented for the MPS (GPU)
# device in the PyTorch version we ship, so YOLO warmup crashes during
# inference. Falling back to CPU for unimplemented MPS ops is invisible
# to the user (negligible perf hit on small ops) and fixes the crash.
# Set in the GUI process AND the worker (see astro_clean_v5.py top).
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

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

"""
star_trail_cleanr.py — Star Trail CleanR desktop application (GUI)

WHAT THIS APP DOES
------------------
Star Trail CleanR removes airplane and satellite trails from astrophotography
image sequences. Astrophotographers capture hundreds of frames of the night sky
over hours, then stack them into a single "star trail" composite. Any aircraft or
satellite that crosses the field of view during those hours leaves a bright streak
in the final image. This app finds those streaks and removes them — frame by frame,
automatically — before the user stacks.

WHO USES IT
-----------
Amateur and semi-professional astrophotographers shooting star trail sequences on
fixed tripods. The foreground (landscape, buildings, trees) is perfectly static
across every frame. The stars move in arcs. Anything else moving through the frame
is a trail to be removed.

THE TWO-STEP PIPELINE
---------------------
1. Detect: YOLO AI model (Trail DetectoR) finds trail pixels in each frame via
   tiled inference (SAHI). The sky/foreground mask limits false positives.
   Static false positives (objects at the same location in every frame) are
   suppressed by comparing detections across neighboring frames.

2. Repair: Star Bridge fills removed pixels by tracking star motion from the
   frame before and after, then blending those two neighbor frames together to
   synthesize what the frame would look like without the trail. Any trail pixel
   that can't be repaired via star tracking is filled with the local sky color
   plus matching grain (a feathered patch that blends into the surrounding sky),
   falling back to pure black only when there isn't enough nearby sky to sample.
   Black fill is invisible in a lighten-max stack because real star pixels in
   other frames always win.

HOW THIS FILE WORKS
-------------------
This file is both the GUI and the algorithm launcher. The PySide6 desktop app
lives entirely in this file. When the user clicks Run, the app spawns itself as
a subprocess with the --cleanr-worker flag, which causes the re-invoked process
to immediately load and execute astro_clean_v5.py (the algorithm) instead of
showing a window. This worker-subprocess model keeps the GUI responsive during
long batch jobs and isolates model loading from the GUI process.

The worker re-launch block at the top of this file handles that early re-dispatch.
Everything below that block (imports, classes, main()) is GUI-only code that never
runs in the worker.

BATCH SIZE AND STAR ROTATION
-----------------------------
Frames are processed in batches of up to 20 at a time. This limit exists because
star motion between frames accumulates over time — a batch that spans too many
minutes of real time will have stars that have moved far enough to confuse the
repair step. The GUI passes each batch's start index and frame count to the
worker; the worker itself pulls in one extra frame before and after each batch
(reading the neighboring files from the same folder) so the repair can stitch
across batch edges.

KEY FILES
---------
- astro_clean_v5.py: worker subprocess — detection + repair algorithm
- modules/detect_trails.py: YOLO/SAHI inference, sky mask, per-frame detection
- modules/trail_grouper.py: fragments → groups → polygons
- modules/repair.py: Star Bridge morph repair per trail
- assets/best.pt: shipped YOLO segmentation weights (Trail DetectoR)
"""

import glob
import json
import threading
from pathlib import Path
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
    QSpinBox, QTabWidget, QTextBrowser, QScrollArea, QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QSettings, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon, QPalette, QColor, QPainter, QIntValidator

from mask_painter import MaskPainterWidget

if getattr(sys, 'frozen', False):
    _base = sys._MEIPASS
else:
    _base = os.path.dirname(os.path.abspath(__file__))


def _open_folder_in_file_manager(path):
    """Open a folder in the OS file manager. Windows: Explorer.
    Mac: Finder via 'open'. Linux: whichever file manager is wired to xdg-open."""
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", path])
    else:
        subprocess.run(["xdg-open", path])

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
_GPU_BUILD_URL = "https://github.com/bruceherwig-dot/star-trail-cleanr/blob/main/docs/nvidia_gpu_setup.md"
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


def _secondary_btn_css():
    """Return the stylesheet string for the app's grey "secondary" buttons
    (Browse, Open Folder, etc.). Pulls its colors from the current theme
    globals, so it must be called at widget-creation time (after _apply_theme
    has set the mode) for the button to match light/dark mode."""
    return (
        f"QPushButton {{ background-color: {SECONDARY_BTN_BG}; color: white; "
        f"font-size: 15px; border-radius: 6px; border: none; padding: 0 8px; }}"
        f"QPushButton:hover {{ background-color: {DISABLED_BTN_HOVER}; }}"
        f"QPushButton:disabled {{ background-color: {DISABLED_BTN_BG}; color: {MUTED_TEXT}; }}"
    )


SCRIPT = os.path.join(_base, "astro_clean_v5.py")
_bundled_model = os.path.join(_base, "best.pt")
_DEV_FALLBACK_MODEL = os.path.join(
    os.path.expanduser("~"),
    "Documents/yolo_runs/trail_detector_v13s_tiled/weights/best.pt")

_DEV_SWITCHER_ENABLED = Path.home().joinpath(
    ".star_trail_cleanr", ".dev_model_switcher").is_file()
_YOLO_RUNS_DIR = Path.home() / "Documents" / "yolo_runs"


def _get_dev_model_choices():
    """Return list of (folder_name, best.pt path) from ~/Documents/yolo_runs, newest first."""
    choices = []
    if _YOLO_RUNS_DIR.is_dir():
        for folder in sorted(_YOLO_RUNS_DIR.iterdir(), reverse=True):
            pt = folder / "weights" / "best.pt"
            if pt.is_file():
                choices.append((folder.name, str(pt)))
    return choices


def get_model_path():
    """Return the best available trail-detector model path for this session.

    Priority: dev override (if switcher enabled) > user-folder download > bundled model > dev fallback.
    Re-evaluated on each call so a mid-session model install is picked up
    on the next processing run.
    """
    if _DEV_SWITCHER_ENABLED:
        override = SETTINGS.value("dev_model_override", "", type=str)
        if override and os.path.isfile(override):
            return override
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


def _pre_window_update_check():
    """Tight-budget GitHub poll BEFORE MainWindow is constructed.

    If a newer release exists, show a small dialog with an Update Now button.
    Update Now opens the platform download URL in the user's browser and
    exits the app. Continue proceeds to normal launch.

    Respects per-version dismissal: once the user clicks Continue on a given
    release tag, the dialog stays quiet for that tag until a newer one ships.

    Runs entirely outside MainWindow's setup code, so a launch-class crash
    inside MainWindow.__init__ (the v1.97-beta failure mode) cannot block
    this prompt. Hard 1.5s network timeout so a slow/down GitHub never
    visibly delays startup.
    """
    try:
        from modules.update_check import check_for_update
        result = check_for_update(VERSION, timeout_s=1.5)
    except Exception:
        return
    if not result:
        return
    dismissed = SETTINGS.value("dismissed_update_tag", "", type=str) or ""
    if dismissed == result["tag"]:
        return
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox()
        box.setWindowTitle("Star Trail CleanR")
        box.setIcon(QMessageBox.Information)
        box.setText(f"A newer version of Star Trail CleanR ({result['tag']}) is available.")
        box.setInformativeText(
            f"You're on v{VERSION}. Update now to make sure you have the latest fixes."
        )
        update_btn = box.addButton("Update Now", QMessageBox.AcceptRole)
        box.addButton("Continue", QMessageBox.RejectRole)
        box.setDefaultButton(update_btn)
        box.exec()
        if box.clickedButton() is update_btn:
            QDesktopServices.openUrl(QUrl(result["download_url"]))
            sys.exit(0)
        # User chose Continue — remember this release tag so we don't ask
        # again until something newer ships.
        SETTINGS.setValue("dismissed_update_tag", result["tag"])
    except Exception:
        return


def _handle_launch_failure(exc):
    """Last-resort recovery dialog when MainWindow construction or show()
    raises. Captures to Sentry (if available) and offers a Download Latest
    button. Module-scope so the bundle-readiness test treats the sentry_sdk
    import here as lazy.
    """
    try:
        import sentry_sdk
        sentry_sdk.capture_exception(exc)
    except Exception:
        pass
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QMessageBox
        from modules.update_check import get_download_url
        box = QMessageBox()
        box.setWindowTitle("Star Trail CleanR")
        box.setIcon(QMessageBox.Critical)
        box.setText("Star Trail CleanR ran into a problem launching.")
        box.setInformativeText(
            "Click Download Latest below to get the newest build for your "
            "computer. If the problem repeats, email "
            "bruceherwig+startrailcleanr@gmail.com."
        )
        download_btn = box.addButton("Download Latest", QMessageBox.AcceptRole)
        box.addButton("Quit", QMessageBox.RejectRole)
        box.setDefaultButton(download_btn)
        box.exec()
        if box.clickedButton() is download_btn:
            QDesktopServices.openUrl(QUrl(get_download_url()))
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
    """Format a duration in seconds as "1h 07m 30s" (or "7m 30s" under an
    hour), with zero-padded minutes and seconds. Used for elapsed/total times
    in the run summary so columns line up. Negative inputs are clamped to 0."""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def fmt_estimate(seconds):
    """Format a duration like fmt_hms but WITHOUT zero-padding ("1h 7m 30s").
    Used for the live "remaining" / "Estimated Time" labels where a compact,
    non-padded look reads better. Negative inputs are clamped to 0."""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def _windows_release_label():
    """Return '11' on Windows 11, '10' on Windows 10, etc.

    platform.release() reports '10' on both Windows 10 and Windows 11 because
    Microsoft kept the kernel version at 10.0. The build number from
    platform.version() is the reliable signal: build >= 22000 = Windows 11.
    """
    import platform as _p
    try:
        build = int(_p.version().split('.')[-1])
        if build >= 22000:
            return "11"
    except Exception:
        pass
    return _p.release()


class CleanerWorker(QThread):
    """Background thread that drives an entire cleaning run, end to end.

    Why it exists: the actual detection + repair work runs in separate
    subprocesses (one per batch of frames), and reading their output would
    freeze the GUI if done on the main thread. This QThread does all the
    orchestration off the main thread and talks back to the window only
    through Qt signals (the GUI thread can safely react to those).

    What it does, in order:
      1. Globs the input folder, removes RAW/JPG/TIFF duplicate "twins",
         applies any frame-range/limit, and checks every frame's resolution.
      2. If frames would be skipped (wrong size, unreadable), it pauses and
         asks the main thread to show a modal before continuing.
      3. Splits the surviving frames into batches (size capped by available
         memory) and launches the worker subprocess once per batch.
      4. Parses each subprocess's stdout line-by-line to drive the progress
         bars, the time estimate, the running trail counter, and the log.
      5. Handles a worker asking what to do about an unreadable file
         (the bad-file prompt) and crash reporting on non-zero exit.
      6. Persists timing so the next run can seed its estimate.

    The signals below are the ONLY way it communicates with the GUI; each
    one is wired to a MainWindow handler in MainWindow._run.
    """
    progress = Signal(int, int, str)   # pct, total, remaining_str
    status = Signal(str)               # log line
    batch_info = Signal(int, int)      # batch_num (1-based), n_batches
    step_progress = Signal(int, int, int, int, int)  # step, batch_current, batch_total, global_current, global_total
    step_detail = Signal(str)          # filename + detail text
    frame_count = Signal(int, int)     # frames_cleaned, total
    stats_ready = Signal(int, int)     # total_trails, total_frames_scanned
    trail_count_update = Signal(int)   # running total trails after each batch
    timing_stats = Signal(float, float)  # initial_estimate_sec, actual_total_sec
    initial_estimate = Signal(float)   # initial estimate seconds (emitted once)
    warmup_active = Signal(bool)       # True = AI loading window, False = real per-frame progress kicked in
    bad_file_prompt = Signal(str, str)  # path, diagnosis — main thread shows the modal
    too_many_bad_files = Signal(int)   # count — main thread shows the final notice
    frames_filter_prompt = Signal(dict)  # info dict — main thread shows the modal explaining skipped frames
    error = Signal(str)
    done = Signal(str)

    # Cap on how many "skip and continue" decisions we'll accept in a single
    # run before auto-stopping. Bruce: "if it continues to fail, then
    # gracefully exit." After the user has been notified of one bad file and
    # chosen to continue, a second failure means the problem isn't a single
    # frame — exit gracefully instead of pelting the user with more popups.
    BAD_FILE_SKIP_CAP = 1

    def __init__(self, folder, output_folder, frame_limit, mask_path=None,
                 output_format="jpg", jpeg_quality=85, frame_start=0, frame_end=0,
                 max_batch=20, mem_note="", twin_prefer="raw"):
        """Capture the run's settings; the actual work happens in run().

        Inputs:
          folder         - input folder of source frames
          output_folder  - where cleaned frames are written
          frame_limit    - how many frames to process ("All Frames", "" or a
                           number-as-string from the Step 4 dropdown)
          mask_path      - optional foreground-mask PNG to keep the AI off the
                           ground/buildings (None = no mask)
          output_format  - "jpg" / "tif8" / "tif16"
          jpeg_quality   - 60-100, only used for JPG output
          frame_start/frame_end - dev-only sub-range into the sorted frame list
                           (0/0 = whole list)
          max_batch      - largest batch the machine's RAM can hold (5-20),
                           chosen by the GUI's memory check
          mem_note       - human-readable record of how max_batch was decided,
                           logged at run start
          twin_prefer    - "raw" or "nonraw": which to keep when a frame exists
                           as both a RAW and a JPG/TIFF
        """
        super().__init__()
        self.folder = folder
        self.output_folder = output_folder
        self.frame_limit = frame_limit
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.mask_path = mask_path
        self.output_format = output_format
        self.jpeg_quality = jpeg_quality
        # "raw" or "nonraw": when a frame exists as both a RAW and a JPG/TIFF,
        # which to process. Chosen by the user in _validate's one-time prompt;
        # passed through to each worker subprocess so both sides dedup alike.
        self.twin_prefer = twin_prefer
        # Largest batch the machine's memory can hold, chosen in _validate from
        # free RAM + the frames' bit depth (falls back to 20). mem_note is a
        # human-readable record of that decision, logged at run start.
        self.max_batch = max_batch
        self.mem_note = mem_note
        self._cancelled = False
        self._proc = None  # current subprocess
        # Bad-file dialog plumbing. The QThread blocks reading subprocess
        # stdout when it sees a prompt sentinel; the main thread shows the
        # modal and calls set_bad_file_decision() to release it.
        self._bad_file_event = threading.Event()
        self._bad_file_response = None
        self._run_skip_count = 0          # cumulative across all batches
        self._graceful_stop_requested = False
        # Frame-filter dialog plumbing. Fires before any subprocess is
        # launched, when the GUI's pre-flight scan finds frames it'll have
        # to drop (different resolution, unreadable header). Same blocking
        # pattern as bad_file but lives entirely in the QThread (no stdin
        # to a subprocess needed).
        self._frames_filter_event = threading.Event()
        self._frames_filter_response = None

    def cancel(self):
        """Request cancellation — kills the running subprocess."""
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
        # Unblock anything waiting on a modal dialog so the thread can exit
        # cleanly even mid-prompt.
        self._bad_file_response = "STOP"
        self._bad_file_event.set()
        self._frames_filter_response = "CANCEL"
        self._frames_filter_event.set()

    def set_frames_filter_decision(self, decision):
        """Called from the main thread after the user answers the
        frames-filter modal. `decision` is "CONTINUE" or "CANCEL"."""
        self._frames_filter_response = decision if decision in ("CONTINUE", "CANCEL") else "CANCEL"
        self._frames_filter_event.set()

    def set_bad_file_decision(self, decision):
        """Called from the main thread after the user answers the bad-file
        modal. `decision` is "CONTINUE" (skip this frame) or "STOP" (end the
        run gracefully)."""
        self._bad_file_response = decision if decision in ("CONTINUE", "STOP") else "STOP"
        self._bad_file_event.set()

    def request_graceful_stop(self):
        """Tell the run loop to stop after the current batch instead of
        starting the next one. Set when the user clicked Stop Run on the
        bad-file modal or when the run-wide skip cap was hit."""
        self._graceful_stop_requested = True

    def run(self):
        """The thread body — runs the whole job and emits signals as it goes.

        This is QThread's entry point: it executes on the worker thread when
        the GUI calls .start(). It builds the frame list, runs the pre-flight
        resolution check, loops over batches launching one subprocess each,
        parses their stdout to drive progress/estimates/counters, and emits
        done() (or error()) at the end. Any uncaught exception is funneled to
        the error signal so the GUI always learns the run ended.
        """
        folder = self.folder
        output_folder = self.output_folder

        try:
            os.makedirs(output_folder, exist_ok=True)

            from modules.frame_list import (
                dedupe_frames, glob_patterns, frame_too_small, MIN_FRAME_SHORT_SIDE)
            frames = sorted(set(
                f for e in glob_patterns() for f in glob.glob(os.path.join(folder, e))
            ))
            if not frames:
                self.error.emit(f"No image files found in: {folder}")
                return

            # Remove duplicate twins (JPG/TIFF/RAW of the same frame) ONCE,
            # before counting or splitting into batches, so the frame count is
            # the true number of unique photos and the worker (which applies the
            # identical rule and the same RAW-vs-JPG/TIFF preference) stays in
            # lockstep. Doing this here is what keeps a final batch from
            # collapsing below the 3-frame minimum after the worker drops twins.
            _pre_dedup = len(frames)
            frames = dedupe_frames(frames, prefer_raw=(self.twin_prefer == "raw"))
            self._deduped_pairs_count = _pre_dedup - len(frames)

            if self.frame_end > 0:
                frames = frames[self.frame_start : self.frame_end + 1]
            elif self.frame_start > 0:
                frames = frames[self.frame_start:]
            total = len(frames)
            if self.frame_limit not in ("All Frames", ""):
                try:
                    total = min(total, int(self.frame_limit))
                except ValueError:
                    pass
                frames = frames[:total]

            from modules.io_safe import image_size as _img_size
            # image_size returns (width, height) for JPEG/PNG (Pillow), TIFF
            # (Pillow with tifffile fallback for BigTIFF/odd compression), and
            # RAW (rawpy .sizes), mirroring the worker's read coverage so the
            # GUI scan never rejects a file the worker could actually process.

            # Inspect every frame's size up front (not just a 10-sample) so
            # we know exactly what's in the folder. The pre-flight modal
            # needs the full picture so it can tell the user what's being
            # dropped and why.
            frame_sizes = {f: _img_size(f) for f in frames}
            unreadable = [f for f, s in frame_sizes.items() if s is None]
            readable = {f: s for f, s in frame_sizes.items() if s is not None}

            if not readable:
                self.error.emit(
                    "Couldn't read any image files in this folder. "
                    "The files may be damaged or in an unsupported format."
                )
                return

            size_counts = Counter(readable.values())
            dominant = size_counts.most_common(1)[0][0]
            if frame_too_small(dominant[0], dominant[1]):
                self.error.emit(
                    f"These images are too small for trail detection "
                    f"(minimum {MIN_FRAME_SHORT_SIDE} pixels on the shorter side). "
                    f"They look like downsized previews. Run Star Trail CleanR on "
                    f"your full-size original images."
                )
                return
            matching = sorted(f for f, s in readable.items() if s == dominant)
            mismatched = sorted(f for f, s in readable.items() if s != dominant)
            unreadable_sorted = sorted(unreadable)

            skipped_total = len(mismatched) + len(unreadable_sorted)

            if skipped_total > 0:
                # Build the breakdown for the modal: every distinct size and
                # how many files have it, plus example filenames per bucket.
                breakdown = []
                for size, count in sorted(size_counts.items(), key=lambda kv: -kv[1]):
                    is_dominant = (size == dominant)
                    breakdown.append({
                        "size": f"{size[0]} × {size[1]}",
                        "count": count,
                        "is_dominant": is_dominant,
                    })

                self._frames_filter_response = None
                self._frames_filter_event.clear()
                self.frames_filter_prompt.emit({
                    "total_found": total,
                    "matching": len(matching),
                    "mismatched": len(mismatched),
                    "unreadable": len(unreadable_sorted),
                    "dominant_size": f"{dominant[0]} × {dominant[1]}",
                    "breakdown": breakdown,
                    "mismatched_sample": [os.path.basename(p) for p in mismatched[:5]],
                    "unreadable_sample": [os.path.basename(p) for p in unreadable_sorted[:5]],
                })
                self._frames_filter_event.wait(timeout=300)
                if self._frames_filter_response != "CONTINUE" or self._cancelled:
                    self.error.emit(
                        "Run cancelled because some frames in this folder "
                        "would have been skipped. No frames processed."
                    )
                    return

            frames = matching
            total = len(frames)
            if not frames:
                self.error.emit("No image files matched the dominant resolution.")
                return

            # Stash skipped counts so the run summary can show them.
            self._skipped_resolution_count = len(mismatched)
            self._skipped_unreadable_count = len(unreadable_sorted)
            self._dominant_size_str = f"{dominant[0]} × {dominant[1]}"

            # Bit-depth scan (header-only). A folder can hold a mix of 8-bit and
            # 16-bit frames -- e.g. most frames have a 16-bit TIFF twin we keep,
            # but a few tail frames are JPG-only (8-bit). Rather than fail on a
            # late batch, the worker evens every batch out to one depth; here we
            # pick that target (the majority across the whole sequence) and count
            # how many frames it will touch, for an honest one-line note.
            from modules.io_safe import image_bitdepth as _img_depth
            _depths = [d for d in (_img_depth(f) for f in frames) if d is not None]
            if _depths:
                _n16 = sum(1 for d in _depths if d == 16)
                _n8 = len(_depths) - _n16
                # Majority wins; a tie favors 16-bit so the higher-precision
                # frames are never the ones degraded.
                self._dominant_bitdepth = 16 if _n16 >= _n8 else 8
                self._normalized_depth_count = sum(
                    1 for d in _depths if d != self._dominant_bitdepth)
            else:
                self._dominant_bitdepth = None
                self._normalized_depth_count = 0

            MAX_BATCH = self.max_batch  # memory-aware cap (20 unless RAM is tight)
            n_batches = (total + MAX_BATCH - 1) // MAX_BATCH
            batch_size = (total + n_batches - 1) // n_batches if n_batches else MAX_BATCH
            starts = list(range(0, total, batch_size))
            if len(starts) > 1 and (total - starts[-1]) < 3:
                starts.pop()
            n_batches = len(starts)

            ref_pixels = 5472 * 3648
            img_pixels = dominant[0] * dominant[1]
            res_scale = img_pixels / ref_pixels
            frames_in_batch = min(batch_size, total)
            est_seconds = int(frames_in_batch * 5 * res_scale * n_batches)

            import re
            mask_note = " with foreground mask" if self.mask_path else ""
            skipped_total = self._skipped_resolution_count + self._skipped_unreadable_count
            header = (f"Processing {total} frames ({dominant[0]}\u00d7{dominant[1]}){mask_note}"
                      + (f" \u2014 skipped {skipped_total} file(s)" if skipped_total else ""))
            _dups = getattr(self, "_deduped_pairs_count", 0)
            if _dups:
                header += (f"\nMerged {_dups} duplicate JPG/TIFF pair"
                           f"{'s' if _dups != 1 else ''} (kept the TIFF)")
            _normed = getattr(self, "_normalized_depth_count", 0)
            if _normed:
                _depth_label = "16-bit" if self._dominant_bitdepth == 16 else "8-bit"
                header += (f"\nEvened out {_normed} frame"
                           f"{'s' if _normed != 1 else ''} to {_depth_label} "
                           f"(folder mixes 8-bit and 16-bit)")
            header += f"\n{n_batches} batch{'es' if n_batches > 1 else ''} to run"
            self.status.emit(header + "\nStarting\u2026")

            # Run settings summary \u2014 always logged before the first subprocess
            # launches so it survives silent worker crashes and lands in emails.
            try:
                from modules.detect_trails import best_device as _best_device
                _dev = _best_device()
                _device_str = {"mps": "GPU (Apple)", "cuda": "GPU (NVIDIA)"}.get(_dev, "CPU")
            except Exception:
                _device_str = "unknown device"
            try:
                from modules.user_folder import get_installed_model_version as _gmv
                _mv = _gmv()
                _mm = re.match(r"model-v(\d+(?:\.\d+)?)", _mv or "")
                _model_str = f"Trail DetectoR v{_mm.group(1)}" if _mm else (_mv or "bundled model")
            except Exception:
                _model_str = "unknown model"
            _scrub_str = ("Second ScrubbeR on"
                          if SETTINGS.value("second_scrub_enabled", False, type=bool)
                          else "Second ScrubbeR off")
            self.status.emit(
                f"{_model_str}  |  {_device_str}  |  {self.output_format.upper()} output"
                f"  |  {_scrub_str}"
            )

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
                """Append one row to the per-machine estimator CSV
                (~/.star_trail_cleanr/estimator_log.csv). Used only to tune the
                time-estimate model offline; never shown to the user. Throttled
                to once every _LOG_INTERVAL seconds unless force=True. `phase`
                is a stage tag ("detect"/"repair"/"batch_start"/etc.)."""
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
                """Delete the cached hot-pixel map file if present (ignored if
                it can't be removed). Used to clear a stale cache before batch 1
                and to tidy up after the run."""
                try:
                    if os.path.isfile(hot_map_file):
                        os.remove(hot_map_file)
                except OSError:
                    pass

            # Clear any stale hot-pixel map cache from a prior run so batch 1 always builds fresh
            if self.mask_path:
                _cleanup_hot_map()

            def _add_log(line):
                """Emit one line to the GUI's Star Log via the status signal."""
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
                """Estimate seconds left in the whole run using a cumulative
                work-rate model. Each frame is 1 work unit split DETECT_FRAC /
                REPAIR_FRAC across the two phases, so the rate stays steady
                across batch boundaries (no lurching). Returns
                (remaining_seconds_or_None, measured); see the comment below for
                why `measured` matters."""
                # Returns (remaining_seconds_or_None, measured) where `measured`
                # is True only when the number comes from the rate we've actually
                # timed this run -- False while we're still falling back to the
                # cold-start seed from the previous run. The headline "Estimated
                # Time" locks only on a measured value, so a wrong seed (e.g. a
                # slow-TIFF prior run seeding a fast-JPG run) can't freeze a
                # bogus 3-hour headline next to an accurate live estimate.
                if phase == "detect":
                    frac = (frame_num / frame_total) * DETECT_FRAC
                else:
                    frac = DETECT_FRAC + (frame_num / frame_total) * REPAIR_FRAC
                cur_batch_work = frac * this_batch_size
                work_done = est_batches_done_frames + cur_batch_work
                remaining_work = total - work_done
                if remaining_work <= 0:
                    return 0.0, True
                rate = None
                if (est_processing_start_t is not None
                        and work_done >= EST_MIN_WORK_FOR_MEASURED):
                    elapsed = now - est_processing_start_t
                    if elapsed > 0:
                        rate = work_done / elapsed  # frames/sec
                if rate is None or rate <= 0:
                    if seeded_sec_per_frame is None or seeded_sec_per_frame <= 0:
                        return None, False
                    return remaining_work * seeded_sec_per_frame * EST_PAD_FACTOR, False
                return (remaining_work / rate) * EST_PAD_FACTOR, True

            _log_est("run_start", None, None, None, 0, None, force=True,
                     note=f"n={total} batches={n_batches} res={dominant[0]}x{dominant[1]}"
                          + (f" {self.mem_note}" if self.mem_note else ""))

            for i, start in enumerate(starts):
                if self._cancelled:
                    _log_est("cancelled", i, None, None, None, None, force=True)
                    return
                if self._graceful_stop_requested:
                    _log_est("graceful_stop", i, None, None, None, None,
                             force=True,
                             note=f"skipped_count={self._run_skip_count}")
                    self.done.emit(
                        f"Run stopped after {self._run_skip_count} unreadable "
                        f"file(s). Cleaned {i} of {n_batches} batch(es)."
                    )
                    return

                self.batch_info.emit(i + 1, n_batches)
                _add_log(f"Batch {i+1}/{n_batches}")

                # The last batch absorbs any remainder: a tiny tail batch
                # (<3 frames) is merged into the previous one above, so the
                # final batch must cover everything left or those frames would
                # be silently dropped -- which happens once batches are small.
                if i == len(starts) - 1:
                    this_batch = total - start
                else:
                    this_batch = min(batch_size, total - start)
                _log_est("batch_start", i, 0, this_batch, None, None, force=True)
                abs_start = start + self.frame_start
                if getattr(sys, 'frozen', False):
                    cmd = [sys.executable, '--cleanr-worker', SCRIPT, folder,
                           "-o", output_folder, "--model", get_model_path(),
                           "--start", str(abs_start), "--batch", str(this_batch)]
                else:
                    cmd = [sys.executable, "-u", SCRIPT, folder,
                           "-o", output_folder, "--model", get_model_path(),
                           "--start", str(abs_start), "--batch", str(this_batch)]

                if self.mask_path:
                    cmd.extend(["--foreground-mask", self.mask_path])
                    cmd.extend(["--hot-pixel-map", hot_map_file])
                cmd.extend(["--output-format", self.output_format,
                            "--jpeg-quality", str(self.jpeg_quality)])
                cmd.extend(["--twin-prefer", self.twin_prefer])
                cmd.extend(["--expected-width", str(dominant[0]),
                            "--expected-height", str(dominant[1])])
                if getattr(self, "_dominant_bitdepth", None):
                    cmd.extend(["--expected-bitdepth", str(self._dominant_bitdepth)])
                if SETTINGS.value("second_scrub_enabled", False, type=bool):
                    cmd.append("--second-scrub")

                worker_env = os.environ.copy()
                if (SETTINGS.value("crash_reporting_enabled", False, type=bool)
                        and _SENTRY_DSN):
                    worker_env["STC_SENTRY_DSN"] = _SENTRY_DSN

                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    env=worker_env,
                )
                cur_step = 0
                detect_count = 0
                repair_count = 0
                _sub_re = re.compile(r'(\d+)/(\d+)')
                # Capture worker stdout for crash reports. Limit memory by
                # only keeping the first 50 and last 50 lines — that's
                # enough context for triage without retaining huge logs.
                proc_stdout_lines = []

                for proc_line in self._proc.stdout:
                    if self._cancelled:
                        self._proc.kill()
                        return
                    proc_line = proc_line.strip()
                    if not proc_line:
                        continue
                    proc_stdout_lines.append(proc_line)

                    # Bad-file prompt: worker is asking what to do about an
                    # unreadable image. Show the modal on the main thread and
                    # write back the user's decision via worker stdin.
                    if proc_line.startswith("STC_BAD_FILE_PROMPT:"):
                        try:
                            payload = json.loads(
                                proc_line.split(":", 1)[1].strip()
                            )
                        except Exception:
                            payload = {"path": "(unknown)", "diagnosis": "(unknown)"}
                        path = payload.get("path", "(unknown)")
                        diag = payload.get("diagnosis", "(none)")

                        if self._run_skip_count >= self.BAD_FILE_SKIP_CAP:
                            # Cap exceeded: don't prompt again. Tell worker to
                            # stop, mark the run for graceful exit, and let
                            # the main thread show the final "too many" notice.
                            decision = "STOP"
                            self._graceful_stop_requested = True
                            self.too_many_bad_files.emit(self._run_skip_count + 1)
                        else:
                            # Block until the main thread answers the modal.
                            self._bad_file_response = None
                            self._bad_file_event.clear()
                            self.bad_file_prompt.emit(path, diag)
                            self._bad_file_event.wait(timeout=300)
                            decision = self._bad_file_response or "STOP"
                            if decision == "CONTINUE":
                                self._run_skip_count += 1
                            else:
                                self._graceful_stop_requested = True

                        try:
                            self._proc.stdin.write(decision + "\n")
                            self._proc.stdin.flush()
                        except Exception:
                            pass
                        continue

                    # Parse stat lines emitted by astro_clean_v5
                    if proc_line.startswith("FRAME_TRAIL_COUNT:"):
                        try:
                            self.trail_count_update.emit(total_trails_run + int(proc_line.split(":", 1)[1].strip()))
                        except ValueError:
                            pass
                        continue
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
                    if "frames loaded" in proc_line:
                        # Frame loading finished. Start the heartbeat NOW so the
                        # silent hot-pixel + AI-load windows that follow show motion.
                        self.warmup_active.emit(True)
                    if "Step 1" in proc_line and "detecting" in proc_line:
                        cur_step = 1
                        self.step_progress.emit(1, 0, this_batch, start, total)
                        # Idempotent on the GUI side; harmless if already running.
                        self.warmup_active.emit(True)
                    elif "Step 2" in proc_line and "cleaning" in proc_line:
                        cur_step = 2
                        self.step_progress.emit(1, this_batch, this_batch, start + this_batch, total)
                        self.step_progress.emit(2, 0, this_batch, start, total)

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
                            self.warmup_active.emit(False)
                            self.step_progress.emit(1, frame_num, frame_total, start + frame_num, total)
                            self.step_detail.emit(proc_line)

                            remaining, est_measured = _estimate_remaining(now_t, this_batch, "detect", frame_num, frame_total)
                            if remaining is not None:
                                batch_pct = (detect_count / frame_total) * 0.67
                                overall_pct = int(((i + batch_pct) / n_batches) * 100)
                                overall_pct = max(0, min(99, overall_pct))
                                self.progress.emit(overall_pct, 100, fmt_estimate(remaining))
                                if est_initial_shown is None and est_measured:
                                    est_initial_shown = remaining + (now_t - t0)
                                    self.initial_estimate.emit(float(est_initial_shown))
                                _log_est("detect", i, frame_num, frame_total,
                                         overall_pct, remaining)

                        elif cur_step == 2 and "cleaning " in proc_line:
                            repair_count = frame_num
                            now_t = time.time()
                            if est_processing_start_t is None:
                                est_processing_start_t = now_t
                            self.step_progress.emit(2, frame_num, frame_total, start + frame_num, total)
                            frames_cleaned = start + frame_num
                            self.frame_count.emit(frames_cleaned, total)
                            self.step_detail.emit(proc_line)

                            remaining, est_measured = _estimate_remaining(now_t, this_batch, "repair", frame_num, frame_total)
                            if remaining is not None:
                                batch_pct = 0.67 + (repair_count / frame_total) * 0.33
                                overall_pct = int(((i + batch_pct) / n_batches) * 100)
                                overall_pct = max(0, min(99, overall_pct))
                                self.progress.emit(overall_pct, 100, fmt_estimate(remaining))
                                if est_initial_shown is None and est_measured:
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
                    if err_lines:
                        err_msg = err_lines[-1]
                    else:
                        # Error was printed to stdout (e.g. mixed bit-depth check).
                        # Find the last ERROR: line, or fall back to last stdout line.
                        stdout_err = [l for l in proc_stdout_lines if l.startswith("ERROR:")]
                        err_msg = stdout_err[-1] if stdout_err else (
                            proc_stdout_lines[-1] if proc_stdout_lines else "unknown error"
                        )

                    def _head_tail(lines, n=50):
                        """Return first n + last n lines joined, with a marker
                        if the middle was elided. Caps memory + email size."""
                        if len(lines) <= 2 * n:
                            return "\n".join(lines)
                        head = "\n".join(lines[:n])
                        tail = "\n".join(lines[-n:])
                        omitted = len(lines) - 2 * n
                        return f"{head}\n... ({omitted} lines elided) ...\n{tail}"

                    stderr_lines = [l for l in stderr_text.splitlines() if l.strip()]
                    stderr_preview = _head_tail(stderr_lines)
                    stdout_preview = _head_tail(proc_stdout_lines)

                    # Safety net: forward worker stderr + stdout to Sentry.
                    # The worker has its own Sentry init for unhandled
                    # exceptions, but crashes that die before that init runs
                    # (missing DLLs, bundle import failures, OS-level kills)
                    # only surface here.
                    if (SETTINGS.value("crash_reporting_enabled", False, type=bool)
                            and _SENTRY_DSN):
                        try:
                            import sentry_sdk
                            import platform as _plat
                            sysname = _plat.system()
                            rel = _windows_release_label() if sysname == "Windows" else _plat.release()
                            os_tag = f"{sysname} {rel} ({_plat.machine()})"
                            with sentry_sdk.push_scope() as scope:
                                scope.set_tag("component", "gui_worker_capture")
                                scope.set_tag("app_version", VERSION)
                                scope.set_tag("batch_index", str(i + 1))
                                scope.set_tag("n_batches", str(n_batches))
                                scope.set_tag("image_w", str(dominant[0]))
                                scope.set_tag("image_h", str(dominant[1]))
                                scope.set_tag("output_format", str(self.output_format))
                                scope.set_tag("os", os_tag)
                                scope.set_extra("stderr_preview", stderr_preview or "")
                                scope.set_extra("stdout_preview", stdout_preview or "")
                                scope.set_extra("stderr_full", stderr_text or "")
                                sentry_sdk.capture_message(
                                    f"Worker exited {self._proc.returncode}: {err_msg}",
                                    level="error",
                                )
                        except Exception:
                            pass
                    self.error.emit(f"Batch {i+1} failed: {err_msg}")
                    return
                self._proc = None

                # Cumulative-rate estimator: advance completed-frames counter
                est_batches_done_frames += this_batch
                _log_est("batch_end", i, this_batch, this_batch, None, None,
                         force=True, note=f"cum_frames={est_batches_done_frames}")

                # Mark both steps complete for this batch
                self.step_progress.emit(2, this_batch, this_batch, start + this_batch, total)
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
        """Query GitHub for a newer app release; emit the result dict if one
        exists. Any failure is swallowed silently."""
        from modules.update_check import check_for_update
        result = check_for_update(VERSION)
        if result:
            self.result_ready.emit(result)


class ModelUpdateCheckThread(QThread):
    """Background check for a newer trail-detector model release. Silent on any failure."""
    result_ready = Signal(dict)

    def run(self):
        """Query GitHub for a newer trail-detector model release; emit the
        result dict if one exists. Any failure is swallowed silently."""
        from modules.model_update import check_for_model_update
        result = check_for_model_update()
        if result:
            self.result_ready.emit(result)


class NvidiaDetectThread(QThread):
    """Background NVIDIA GPU detection. Emits the outcome and a detail string."""
    result_ready = Signal(str, str)

    def run(self):
        """Probe for an NVIDIA GPU and emit (outcome, detail) — outcome is
        "yes"/"no"/etc., detail is a human-readable note."""
        from modules.nvidia_detect import detect_nvidia
        outcome, detail = detect_nvidia()
        self.result_ready.emit(outcome, detail)


class BestDeviceThread(QThread):
    """Background torch device detection. Emits 'cuda', 'mps', or 'cpu'."""
    result_ready = Signal(str)

    def run(self):
        """Determine the torch compute device and emit it ("cuda"/"mps"/"cpu"),
        defaulting to "cpu" on any failure."""
        try:
            from modules.detect_trails import best_device
            self.result_ready.emit(best_device())
        except Exception:
            self.result_ready.emit("cpu")


class ModelDownloadThread(QThread):
    """Streams a model file into the user folder. Atomic via temp-then-rename.

    Writes the version note only after the rename succeeds, so a mid-download
    crash leaves the previous model in place and the note untouched.
    """
    progress = Signal(int, int)   # bytes_done, total_bytes (total=0 if unknown)
    finished_ok = Signal(str)     # version tag that was installed
    failed = Signal(str)          # short error string; not user-visible

    def __init__(self, url, target_path, version_tag, parent=None):
        """Store the download `url`, the final `target_path` for the weights,
        and the `version_tag` to record once the install succeeds."""
        super().__init__(parent)
        self.url = url
        self.target_path = target_path
        self.version_tag = version_tag

    def run(self):
        """Stream the model file to a .tmp path emitting progress, then
        atomically rename it into place and save the version tag. On any
        failure, delete the partial .tmp and emit failed()."""
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


class GpuPackInstallThread(QThread):
    """Downloads the CUDA PyTorch wheels and extracts them into the GPU override folder.

    Emits progress(label, bytes_done, total_bytes) during download.
    label changes per step; total_bytes=0 signals an indeterminate phase.
    """
    progress = Signal(str, float, float)
    finished_ok = Signal()
    failed = Signal(str)

    def _download(self, label, urls, dest_path):
        """Try each URL in order. On HTTP 403 move to the next mirror silently.
        Raises RuntimeError with a __blocked__ sentinel if every mirror returns 403."""
        import urllib.request
        import urllib.error
        last_403 = None
        for idx, url in enumerate(urls):
            if idx > 0:
                self.progress.emit(f"{label} — trying backup server...", 0, 0)
            req = urllib.request.Request(url, headers={"User-Agent": "StarTrailCleanR-GpuPack"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    total = float(resp.headers.get("Content-Length") or 0)
                    done = 0.0
                    with open(str(dest_path), "wb") as f:
                        while True:
                            chunk = resp.read(131072)
                            if not chunk:
                                break
                            f.write(chunk)
                            done += len(chunk)
                            self.progress.emit(label, done, total)
                return
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    last_403 = e
                    continue
                raise RuntimeError(f"{label} failed: {e}") from e
            except Exception as e:
                raise RuntimeError(f"{label} failed: {e}") from e
        raise RuntimeError(
            f"{label} blocked: all servers returned 403 Forbidden\n__blocked__"
        ) from last_403

    def run(self):
        """Download and install the CUDA torch + torchvision wheels into the
        GPU override folder. Clears any stale prior install first, downloads
        each wheel (trying mirrors on 403), extracts it, fixes file
        permissions, then writes the version tag. Emits finished_ok() on
        success or failed() with a user-friendly message (permission, 403, or
        connection errors get tailored text)."""
        import zipfile
        from modules.gpu_pack import (get_all_download_url_sets, get_override_dir,
                                       write_version_tag, clear_gpu_files,
                                       chmod_extracted_files)

        url_sets = get_all_download_url_sets()
        if not url_sets:
            self.failed.emit(
                "Cannot determine download URLs for this build.\n"
                "Try updating Star Trail CleanR first, then install GPU support again."
            )
            return

        torch_urls = [s[0] for s in url_sets]
        tv_urls    = [s[1] for s in url_sets]
        torch_ver  = url_sets[0][2]
        override_dir = get_override_dir()

        try:
            override_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.failed.emit(
                f"Cannot create the GPU support folder:\n{override_dir}\n\n{e}\n\n"
                "Check that you have permission to write to your AppData folder."
            )
            return

        torch_whl = override_dir / "torch_pack.whl"
        tv_whl = override_dir / "torchvision_pack.whl"

        # Robust cleanup: onerror handler + shell fallback + 3-retry loop.
        # Handles read-only files from zip extraction and transient AV locks.
        self.progress.emit("Preparing...", 0, 0)
        _ok, _err = clear_gpu_files()
        for _stale in (override_dir / "torch", override_dir / "torchvision"):
            if _stale.is_dir():
                detail = f"\n\nDetails: {_err}" if _err else ""
                self.failed.emit(
                    "Installation blocked: GPU support files from a previous install "
                    f"could not be removed. Windows is holding them open.{detail}\n\n"
                    "Reboot your computer, then reopen Star Trail CleanR and click "
                    "Install GPU Support again. The reboot will release the locked files."
                )
                return

        try:
            self._download("Downloading GPU support (1 of 2)", torch_urls, torch_whl)
            self.progress.emit("Installing GPU support (1 of 2)...", 0, 0)
            with zipfile.ZipFile(str(torch_whl), "r") as zf:
                zf.extractall(str(override_dir))
            torch_whl.unlink(missing_ok=True)
            chmod_extracted_files(override_dir)

            self._download("Downloading GPU support (2 of 2)", tv_urls, tv_whl)
            self.progress.emit("Installing GPU support (2 of 2)...", 0, 0)
            with zipfile.ZipFile(str(tv_whl), "r") as zf:
                zf.extractall(str(override_dir))
            tv_whl.unlink(missing_ok=True)
            chmod_extracted_files(override_dir)

            if not write_version_tag(torch_ver):
                raise RuntimeError(
                    "Files downloaded successfully but could not write the version tag.\n"
                    "GPU support may not activate on restart. Try installing again."
                )
            self.finished_ok.emit()

        except Exception as e:
            for whl in (torch_whl, tv_whl):
                try:
                    whl.unlink(missing_ok=True)
                except Exception:
                    pass
            msg = str(e)
            if "Errno 13" in msg or "Permission denied" in msg or "Access is denied" in msg:
                msg = (
                    "Installation failed: Windows denied access to a file.\n\n"
                    "Reboot your computer, then reopen Star Trail CleanR and click "
                    "Install GPU Support again. The reboot will release the locked files.\n\n"
                    f"Details: {e}"
                )
            elif "__blocked__" in msg or ("403" in msg and "Forbidden" in msg):
                msg = (
                    "Download blocked (HTTP 403).\n\n"
                    "PyTorch's download servers are blocking requests from your network. "
                    "We automatically tried an alternative server, but it was also blocked.\n\n"
                    "The most reliable fix is to use a VPN — connect to any US or European "
                    "server, then click Install GPU Support again.\n\n"
                    "Click More Info for step-by-step instructions.\n\n"
                    f"Details: {e}"
                )
            elif "urlopen error" in msg or "ConnectionReset" in msg or "timed out" in msg:
                msg = (
                    "Download failed. Check your internet connection and try again.\n\n"
                    f"Details: {e}"
                )
            self.failed.emit(msg)


class _XCloseButton(QPushButton):
    """Close button that paints two diagonal white lines via QPainter for
    pixel-perfect centering at any size. Avoids font-metric drift that left
    the unicode multiplication sign sitting visibly off-center across
    platforms. Background color and hover state come from the stylesheet."""

    def paintEvent(self, event):
        """Draw the button's background (via the base class) then paint the two
        white diagonal strokes of the X by hand, with a margin scaled to the
        button size so the glyph stays centered at any size."""
        from PySide6.QtGui import QPainter, QPen
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(Qt.white, 3.5)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        w, h = self.width(), self.height()
        margin = max(10, int(min(w, h) * 0.34))
        painter.drawLine(margin, margin, w - margin, h - margin)
        painter.drawLine(w - margin, margin, margin, h - margin)


class MainWindow(QMainWindow):
    """The app's single main window — everything the user sees and clicks.

    It hosts a banner across the top and a four-tab area below it
    (Main / FAQ / About / Settings). The "Main" tab is itself a two-page
    stack: page 0 is the Setup form (pick folders, mask, format, then
    "Clean My Stars!") and page 1 is the live processing view (progress bars,
    Star Log, run-complete dialog). It owns the CleanerWorker thread for a
    run and the separate MaskEditorWindow, and it runs three small background
    threads at startup (app-update check, model-update check, GPU detection)
    whose results populate the orange notice banners.

    The bulk of this class is _build_* methods that construct each piece of
    UI and _on_* methods that handle button clicks and worker signals.
    """
    def __init__(self):
        """Build the whole window: restore saved geometry (or open at a
        sensible first-launch size), assemble the banner + tabs + setup and
        process pages, make every QLabel text-selectable for easy copy, and
        kick off the three startup background checks."""
        super().__init__()
        self.setWindowTitle(f"Star Trail CleanR (Beta v{VERSION})")
        self.setMinimumWidth(720)
        # Min height is computed dynamically from the Setup page's actual
        # layout after the window first shows — see _lock_min_height. No
        # hardcoded numbers here so the floor self-corrects whenever the
        # layout changes (more steps, padding tweaks, font swap, etc.).
        self._min_height_locked = False
        saved_geo = SETTINGS.value("window_geometry")
        if saved_geo:
            self.restoreGeometry(saved_geo)
        else:
            # First-time launch: open at the natural content height so all
            # six steps AND the Clean My Stars button are visible without
            # scrolling. Capped at 90% of available screen.
            screen = QApplication.primaryScreen()
            if screen:
                geom = screen.availableGeometry()
                w = min(1300, int(geom.width() * 0.9))
                h = min(1300, int(geom.height() * 0.9))
                self.resize(w, h)
                # Center on the active screen.
                x = geom.x() + (geom.width() - w) // 2
                y = geom.y() + (geom.height() - h) // 2
                self.move(x, y)
            else:
                self.resize(1300, 1300)
        self.worker = None
        self._mask_path = None
        self._mask_window = None
        self._nvidia_outcome = None
        self._compute_device = None
        self._gpu_install_via_banner = False

        # Main stacked widget: page 0 = setup, page 1 = processing
        self._stack = QStackedWidget()

        # Tabs: Main / FAQ / About / Settings
        self._tabs = QTabWidget()
        self._tabs.tabBar().setExpanding(True)
        self._tabs.tabBar().setDocumentMode(True)
        self._tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: none; background: palette(window); }}"
            "QTabBar { qproperty-drawBase: 0; }"
            f"QTabBar::tab {{ background: {BRAND_TAB_INACTIVE_BG}; color: {BRAND_TAB_INACTIVE_FG}; padding: 14px 20px; "
            "font-size: 19px; font-weight: bold; border: none; min-width: 200px; }}"
            f"QTabBar::tab:selected {{ background: {BRAND_TAB_ACTIVE_BG}; color: {BRAND_TAB_ACTIVE_FG}; }}"
            f"QTabBar::tab:hover:!selected {{ background: {BRAND_TAB_HOVER_BG}; color: {BRAND_TAB_ACTIVE_FG}; }}"
        )
        self._tabs.addTab(self._stack, "Main")
        self._tabs.addTab(self._build_faq_tab(), "FAQ")
        self._tabs.addTab(self._build_about_tab(), "About")
        self._tabs.addTab(self._build_settings_tab(), "Settings")
        self._tabs.tabBar().setUsesScrollButtons(True)

        # Container: banner on top, tabs below
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(self._build_banner())
        container_layout.addWidget(self._build_update_banner())
        container_layout.addWidget(self._build_model_update_card())
        container_layout.addWidget(self._build_nvidia_banner())
        # Stretch factor 1 on tabs so extra vertical space (when the user
        # resizes or maximizes the window) goes into the tab area and
        # through to the Run-tab Star Log + Setup-tab scroll area, rather
        # than empty space at top or bottom. Banners stay their fixed
        # heights; chrome stays the same size; only content grows.
        container_layout.addWidget(self._tabs, 1)
        self.setCentralWidget(container)

        self._build_setup_page()
        self._build_process_page()
        self._stack.setCurrentIndex(0)

        for lbl in self.findChildren(QLabel):
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self._start_update_check()
        self._start_model_update_check()
        self._start_nvidia_detect()

    def showEvent(self, event):
        """First show: defer one tick to let the layout settle, then lock
        the window's minimum height to whatever the Setup page's actual
        layout requires. Avoids hardcoding heights that drift out of sync
        when the layout changes."""
        super().showEvent(event)
        if not self._min_height_locked:
            QTimer.singleShot(0, self._lock_min_height)
            self._min_height_locked = True
        QTimer.singleShot(500, self._maybe_ask_crash_reporting)

    def _maybe_ask_crash_reporting(self):
        """First-run crash-reporting opt-in. Fires once after the main window
        is visible so it never blocks startup. Only shown in CI builds where
        the Sentry DSN is present."""
        if not _SENTRY_DSN:
            return
        if SETTINGS.contains("crash_reporting_choice_made"):
            return
        prompt = QMessageBox(self)
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

    def _lock_min_height(self):
        """Set the window's minimum vertical size to the Setup tab's
        natural content height so Setup never clips below the Clean
        button. Maximum is intentionally unbounded so users can maximize
        the window on Windows (where the previous min==max lock left
        empty desktop below the app) or resize larger if they want.
        Run's right panel may show extra space at the bottom when the
        window is taller than Setup's natural height — accepted
        trade-off vs the maximize-doesn't-fill bug Warren reported."""
        if not hasattr(self, "_setup_inner") or self._setup_inner is None:
            return
        setup_natural = self._setup_inner.sizeHint().height()
        chrome = self.height() - self._stack.height()
        target = setup_natural + chrome
        screen = QApplication.primaryScreen()
        if screen:
            target = min(target, int(screen.availableGeometry().height() * 0.9))
        target = max(target, 600)
        self.setMinimumHeight(target)
        if self.height() != target:
            self.resize(self.width(), target)

    # ── FAQ tab ──────────────────────────────────────────────────────────────

    def _build_faq_tab(self):
        """Build the FAQ tab: a single read-only HTML browser explaining what
        the app does, the Detect/Repair pipeline, the workflow, and the known
        limitations. Returns the wrapper widget added to the tab bar."""
        wrap = QWidget()
        wrap_layout = QVBoxLayout(wrap)
        wrap_layout.setContentsMargins(16, 16, 16, 16)
        wrap_layout.setSpacing(0)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.document().setDocumentMargin(16)
        browser.setStyleSheet(
            f"QTextBrowser {{ background: {BROWSER_BG}; color: {BROWSER_TEXT}; border: none; font-size: 13px; }}"
        )
        browser.setHtml(f"""
        <html><body style='font-family: Inter, -apple-system, Segoe UI, sans-serif; line-height: 1.5; margin:0; padding:0; color:{BROWSER_TEXT}; background-color:{BROWSER_BG};'>
        <p style='margin:0; padding:0; line-height:0; font-size:1px; height:0;'></p>
        <h2 style='color:{BRAND_HEADING_BLUE}; margin-top:0; margin-bottom:2px;'>Why Star Trail CleanR?</h2>
        <p style='margin-top:2px;'>Star Trail CleanR removes airplane and satellite trails
        from astrophotography sequences while preserving the real stars. The result is a
        clean set of frames you can stack into a star trail composite. (That's the goal, anyway.)</p>

        <h2 style='color:{BRAND_HEADING_BLUE}; margin-bottom:2px;'>Trail Detection</h2>
        <p style='margin-top:2px;'>Each frame is run through a YOLO segmentation model
        trained on thousands of manually labeled airplane and satellite trails across many
        cameras, lenses, and sky conditions. The model produces pixel-accurate masks for
        every trail it finds.</p>

        <h2 style='color:{BRAND_HEADING_BLUE}; margin-bottom:2px;'>The Fix: Star Bridge Repair</h2>
        <p style='margin-top:2px;'>For each trail, Star Trail CleanR pulls clean pixels
        from the frame immediately before and after, blending them across the trail using
        a morphing technique called <i>Star Bridge</i>. This preserves the real stars
        underneath the trail and keeps the brightness and color natural. No smudges,
        no blank patches.</p>

        <h2 style='color:{BRAND_HEADING_BLUE}; margin-bottom:2px;'>Workflow</h2>
        <ol style='margin-top:2px;'>
        <li><b>Browse.</b> Choose your folder of frames.</li>
        <li><b>Mask (optional).</b> Paint over ground, buildings, and rocks so
        the AI ignores them. Trees can be left unmasked.</li>
        <li><b>Format.</b> Pick output format (JPG / TIFF 8-bit / TIFF 16-bit)
        and JPEG quality.</li>
        <li><b>Run.</b> Sit back. Cleaned frames land in a <code>cleaned/</code>
        folder next to your originals.</li>
        <li><b>Stack.</b> Load the cleaned frames into your favorite stacker
        (StarStaX, Sequator, Photoshop, etc.) for the final composite.</li>
        </ol>

        <h2 style='color:{BRAND_HEADING_BLUE}; margin-bottom:2px;'>Limitations</h2>
        <ul style='margin-top:2px;'>
        <li><b>Trail variety is bounded by the AI's training data.</b> If a type of
        trail isn't being detected well in your sequences, you can help train the next
        version: zip 300+ frames from that scene and send them to
        <a href='mailto:bruceherwig+startrailcleanr@gmail.com?subject=Star%20Trail%20CleanR%20training%20frames'>bruceherwig+startrailcleanr@gmail.com</a>.
        For large folders, share a Dropbox, Google Drive, or WeTransfer link instead
        of attaching directly. The model gets smarter every time the community
        contributes.</li>
        <li><b>Meteors will be removed too.</b> Their streaks look similar to airplane
        and satellite trails, so the detector can't tell them apart. If you want to
        keep them, use your originals to mask them back in.</li>
        <li><b>RAW files are supported</b> (.CR2, .CR3, .NEF, .ARW, .RAF, .DNG, and most
        others). Just drop the folder in. If a frame has both a RAW and a JPG/TIFF,
        Star Trail CleanR asks once which to use (RAW by default). Keep your output
        format set to TIFF 16-bit if you want to preserve the RAW's full bit depth.</li>
        <li><b>Not a one-click fix.</b> You'll still want to touch up the final
        composite in Photoshop or your editor of choice. But if we did our job
        right, it's a fraction of the time you used to spend.</li>
        <li><b>Designed for wide-field star trail sequences,</b> not deep-sky tracked exposures.</li>
        </ul>

        <p style='color:{HINT_TEXT}; margin-top:24px;'>Star Trail CleanR is free and offered as
        a gift to the astrophotography community.
        <a href='mailto:bruceherwig+startrailcleanr@gmail.com?subject=Star%20Trail%20CleanR%20feedback'>Feedback welcome.</a></p>
        </body></html>
        """)
        wrap_layout.addWidget(browser)
        return wrap

    # ── Settings tab ─────────────────────────────────────────────────────────

    def _build_settings_tab(self):
        """Build the Settings tab. Four sections, top to bottom: Updates
        (manual check button), GPU Acceleration (status line + the Windows-only
        NVIDIA GPU-pack install/clear/help controls and progress bar), Second
        ScrubbeR (a 180-degree second detection pass toggle), and Crash
        Reporting (anonymous Sentry opt-in). Stashes the widgets it later
        toggles on self. Returns the wrapper widget for the tab bar."""
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignTop)

        _h = QLabel("Updates")
        _h.setStyleSheet(f"color: {BRAND_HEADING_BLUE}; font-size: 18px; font-weight: bold;")
        layout.addWidget(_h)
        layout.addSpacing(4)
        _d = QLabel("Star Trail CleanR checks for a new version every time you open it. Use this to check right now.")
        _d.setStyleSheet(f"color: {BROWSER_TEXT}; font-size: 13px;")
        _d.setWordWrap(True)
        layout.addWidget(_d)

        check_btn = QPushButton("Check for Updates")
        check_btn.setFixedHeight(34)
        check_btn.setFixedWidth(200)
        check_btn.setCursor(Qt.PointingHandCursor)
        check_btn.setStyleSheet(
            f"QPushButton {{ background-color: {SECONDARY_BTN_BG}; color: white; "
            f"font-size: 13px; font-weight: bold; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: {DISABLED_BTN_HOVER}; }}"
            f"QPushButton:disabled {{ background-color: {DISABLED_BTN_BG}; color: {MUTED_TEXT}; }}"
        )
        check_btn.clicked.connect(self._on_check_for_updates)
        self._check_updates_btn = check_btn

        run_hint = QLabel("A run is in progress. Updates are paused until it finishes.")
        run_hint.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px;")
        run_hint.setVisible(False)
        self._check_updates_run_hint = run_hint

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(16, 0, 0, 0)
        btn_row.addWidget(check_btn)
        btn_row.addSpacing(12)
        btn_row.addWidget(run_hint)
        btn_row.addStretch()
        layout.addSpacing(4)
        layout.addLayout(btn_row)

        layout.addSpacing(14)

        _h = QLabel("GPU Acceleration")
        _h.setStyleSheet(f"color: {BRAND_HEADING_BLUE}; font-size: 18px; font-weight: bold;")
        layout.addWidget(_h)

        compute_status = QLabel("Detecting...")
        compute_status.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 13px; margin-left: 16px;")
        compute_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._compute_status_label = compute_status
        layout.addSpacing(4)
        layout.addWidget(compute_status)

        # GPU install controls wrapped in a container so when hidden they
        # leave no dead spacing in the layout.
        gpu_install_widget = QWidget()
        gpu_install_layout = QVBoxLayout(gpu_install_widget)
        gpu_install_layout.setContentsMargins(0, 0, 0, 0)
        gpu_install_layout.setSpacing(0)
        gpu_install_widget.setVisible(False)
        self._gpu_install_widget = gpu_install_widget

        gpu_upgrade_browser = QTextBrowser()
        gpu_upgrade_browser.setOpenExternalLinks(False)
        gpu_upgrade_browser.document().setDocumentMargin(4)
        gpu_upgrade_browser.setStyleSheet(
            f"QTextBrowser {{ background: {BROWSER_BG}; color: {BROWSER_TEXT}; border: none; font-size: 13px; }}"
        )
        gpu_upgrade_browser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        gpu_upgrade_browser.setHtml(f"""
        <html><body style='font-family: Inter, -apple-system, Segoe UI, sans-serif; line-height: 1.5; margin:0; padding:0; color:{BROWSER_TEXT}; background-color:{BROWSER_BG};'>
        <p style='margin-top:8px;'>An NVIDIA GPU is available. Installing GPU support downloads approximately 3-4 GB from pytorch.org. This is a one-time download that survives app updates automatically.</p>
        </body></html>
        """)
        gpu_upgrade_browser.setFixedHeight(50)
        self._gpu_upgrade_browser = gpu_upgrade_browser
        gpu_install_layout.addSpacing(4)
        gpu_install_layout.addWidget(gpu_upgrade_browser)

        gpu_btn = QPushButton("Install GPU Support")
        gpu_btn.setFixedHeight(34)
        gpu_btn.setFixedWidth(200)
        gpu_btn.setCursor(Qt.PointingHandCursor)
        gpu_btn.setStyleSheet(
            f"QPushButton {{ background-color: {SECONDARY_BTN_BG}; color: white; "
            f"font-size: 13px; font-weight: bold; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: {DISABLED_BTN_HOVER}; }}"
        )
        gpu_btn.clicked.connect(self._on_nvidia_download_clicked)
        self._gpu_download_btn = gpu_btn

        gpu_btn_row = QHBoxLayout()
        gpu_btn_row.setContentsMargins(16, 0, 0, 0)
        gpu_btn_row.addWidget(gpu_btn)
        gpu_btn_row.addStretch()
        gpu_install_layout.addSpacing(4)
        gpu_install_layout.addLayout(gpu_btn_row)

        gpu_clear_btn = QPushButton("Clear GPU Support Files")
        gpu_clear_btn.setFixedHeight(28)
        gpu_clear_btn.setFixedWidth(200)
        gpu_clear_btn.setCursor(Qt.PointingHandCursor)
        gpu_clear_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {MUTED_TEXT}; "
            f"font-size: 12px; border: none; text-decoration: underline; }}"
            f"QPushButton:hover {{ color: {BROWSER_TEXT}; }}"
        )
        gpu_clear_btn.clicked.connect(self._on_gpu_clear_clicked)
        self._gpu_clear_btn = gpu_clear_btn
        gpu_clear_row = QHBoxLayout()
        gpu_clear_row.setContentsMargins(16, 0, 0, 0)
        gpu_clear_row.addWidget(gpu_clear_btn)
        gpu_clear_row.addStretch()
        gpu_install_layout.addSpacing(2)
        gpu_install_layout.addLayout(gpu_clear_row)

        gpu_help_btn = QPushButton("GPU installation troubleshooting guide")
        gpu_help_btn.setFixedHeight(28)
        gpu_help_btn.setCursor(Qt.PointingHandCursor)
        gpu_help_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {MUTED_TEXT}; "
            f"font-size: 12px; border: none; text-decoration: underline; }}"
            f"QPushButton:hover {{ color: {BROWSER_TEXT}; }}"
        )
        gpu_help_btn.clicked.connect(self._on_gpu_help_clicked)
        self._gpu_help_btn = gpu_help_btn
        gpu_help_row = QHBoxLayout()
        gpu_help_row.setContentsMargins(16, 0, 0, 0)
        gpu_help_row.addWidget(gpu_help_btn)
        gpu_help_row.addStretch()
        gpu_install_layout.addSpacing(0)
        gpu_install_layout.addLayout(gpu_help_row)

        from PySide6.QtWidgets import QProgressBar
        gpu_progress = QProgressBar()
        gpu_progress.setRange(0, 100)
        gpu_progress.setValue(0)
        gpu_progress.setFixedHeight(20)
        gpu_progress.setVisible(False)
        self._gpu_progress = gpu_progress
        gpu_install_layout.addSpacing(8)
        gpu_install_layout.addWidget(gpu_progress)

        gpu_progress_label = QLabel("")
        gpu_progress_label.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px; margin-left: 4px;")
        gpu_progress_label.setVisible(False)
        self._gpu_progress_label = gpu_progress_label
        gpu_install_layout.addSpacing(2)
        gpu_install_layout.addWidget(gpu_progress_label)

        gpu_restart_btn = QPushButton("Restart Now to Activate GPU Support")
        gpu_restart_btn.setFixedHeight(34)
        gpu_restart_btn.setFixedWidth(290)
        gpu_restart_btn.setCursor(Qt.PointingHandCursor)
        gpu_restart_btn.setStyleSheet(
            f"QPushButton {{ background-color: {SECONDARY_BTN_BG}; color: white; "
            f"font-size: 13px; font-weight: bold; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: {DISABLED_BTN_HOVER}; }}"
        )
        gpu_restart_btn.clicked.connect(self._relaunch)
        gpu_restart_btn.setVisible(False)
        self._gpu_restart_btn = gpu_restart_btn
        restart_row = QHBoxLayout()
        restart_row.setContentsMargins(16, 0, 0, 0)
        restart_row.addWidget(gpu_restart_btn)
        restart_row.addStretch()
        gpu_install_layout.addSpacing(8)
        gpu_install_layout.addLayout(restart_row)

        layout.addSpacing(4)
        layout.addWidget(gpu_install_widget)

        layout.addSpacing(14)

        _h = QLabel("Second ScrubbeR")
        _h.setStyleSheet(f"color: {BRAND_HEADING_BLUE}; font-size: 18px; font-weight: bold;")
        layout.addWidget(_h)
        layout.addSpacing(4)
        _d = QLabel("Runs the trail detector a second time on each frame after rotating it 180°. Catches trails the first pass tends to miss. Detection takes roughly twice as long.")
        _d.setStyleSheet(f"color: {BROWSER_TEXT}; font-size: 13px;")
        _d.setWordWrap(True)
        layout.addWidget(_d)
        _d2 = QLabel("Most helpful with earlier detection models. Less necessary now that the AI trains on rotated trail images at multiple angles.")
        _d2.setStyleSheet(f"color: {BROWSER_TEXT}; font-size: 13px;")
        _d2.setWordWrap(True)
        layout.addWidget(_d2)

        scrub_chk = QCheckBox("Enable Second ScrubbeR")
        scrub_chk.setStyleSheet(f"QCheckBox {{ font-size: 13px; color: {BROWSER_TEXT}; margin-left: 16px; }}")
        scrub_chk.setChecked(SETTINGS.value("second_scrub_enabled", False, type=bool))
        scrub_chk.toggled.connect(lambda v: SETTINGS.setValue("second_scrub_enabled", v))
        self._scrub_chk = scrub_chk

        scrub_run_hint = QLabel("A run is in progress. Setting locked until it finishes.")
        scrub_run_hint.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 12px;")
        scrub_run_hint.setVisible(False)
        self._scrub_run_hint = scrub_run_hint

        scrub_row = QHBoxLayout()
        scrub_row.setContentsMargins(16, 0, 0, 0)
        scrub_row.addWidget(scrub_chk)
        scrub_row.addSpacing(12)
        scrub_row.addWidget(scrub_run_hint)
        scrub_row.addStretch()
        layout.addSpacing(4)
        layout.addLayout(scrub_row)

        layout.addSpacing(14)

        _h = QLabel("Crash Reporting")
        _h.setStyleSheet(f"color: {BRAND_HEADING_BLUE}; font-size: 18px; font-weight: bold;")
        layout.addWidget(_h)
        layout.addSpacing(4)
        _d = QLabel("Sends anonymous crash reports to help find and fix bugs. Reports contain technical details like error messages and basic image dimensions.")
        _d.setStyleSheet(f"color: {BROWSER_TEXT}; font-size: 13px;")
        _d.setWordWrap(True)
        layout.addWidget(_d)

        crash_chk = QCheckBox("Send anonymous crash reports")
        crash_chk.setStyleSheet(f"QCheckBox {{ font-size: 13px; color: {BROWSER_TEXT}; margin-left: 16px; }}")
        crash_chk.setChecked(SETTINGS.value("crash_reporting_enabled", False, type=bool))

        def _on_crash_chk_toggled(v):
            """Settings crash-reporting checkbox handler. `v` is the new
            checked state."""
            # Persist the choice and, if newly enabled, initialize Sentry right
            # away so reporting starts this session without a restart.
            SETTINGS.setValue("crash_reporting_enabled", v)
            if v:
                _maybe_init_sentry()

        crash_chk.toggled.connect(_on_crash_chk_toggled)

        crash_row = QHBoxLayout()
        crash_row.setContentsMargins(16, 0, 0, 0)
        crash_row.addWidget(crash_chk)
        crash_row.addStretch()
        layout.addSpacing(4)
        layout.addLayout(crash_row)

        layout.addStretch()
        return wrap

    def _set_updates_run_state(self, running):
        """Enable or disable the Settings controls that must not change mid-run.
        `running` True = a cleaning run is active, so grey out the "Check for
        Updates" button and the Second ScrubbeR checkbox and show their "locked
        until it finishes" hints; False restores them."""
        if hasattr(self, '_check_updates_btn'):
            self._check_updates_btn.setEnabled(not running)
            self._check_updates_run_hint.setVisible(running)
        if hasattr(self, '_scrub_chk'):
            self._scrub_chk.setEnabled(not running)
            self._scrub_run_hint.setVisible(running)

    def _updater_unavailable_fallback(self):
        """The built-in updater engine is not running, most often because
        macOS disabled it (app launched from the disk image or another
        non-Applications location). NEVER fail silently -- a mute Check for
        Updates button is exactly what stranded a real Mac user on 2026-06-10.
        Explain the cause in plain language, then open the download page so
        the user can still update by hand."""
        QMessageBox.information(
            self, "Updater not available",
            "The built-in updater isn't available in this session.\n\n"
            "On a Mac this usually means the app isn't running from the "
            "Applications folder: open the downloaded disk image, drag "
            "Star Trail CleanR into Applications, and launch it from there. "
            "Updates will then install themselves with one click.\n\n"
            "Opening the download page so you can get the latest version "
            "now.")
        import webbrowser
        webbrowser.open("https://startrailcleanr.com")

    def _on_check_for_updates(self):
        """Settings tab "Check for Updates" button. In a frozen build, trigger
        the platform's in-app updater (Sparkle on Mac, WinSparkle on Windows);
        on Linux or when running from source, just open the project website.
        If the updater engine isn't running, show the visible fallback instead
        of doing nothing."""
        if getattr(sys, 'frozen', False):
            if sys.platform == "darwin":
                from modules.sparkle_updater import check_for_updates
                if not check_for_updates():
                    self._updater_unavailable_fallback()
            elif sys.platform == "win32":
                from modules.winsparkle_updater import check_for_updates
                if not check_for_updates():
                    self._updater_unavailable_fallback()
            else:
                import webbrowser
                webbrowser.open("https://startrailcleanr.com")
        else:
            import webbrowser
            webbrowser.open("https://startrailcleanr.com")

    # ── About tab ────────────────────────────────────────────────────────────

    def _build_about_tab(self):
        """Build the About tab: Bruce's silhouette photo on the left and an
        HTML bio + links + acknowledgments on the right. Returns the wrapper
        widget for the tab bar."""
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
            f"QTextBrowser {{ background: {BROWSER_BG}; color: {BROWSER_TEXT}; border: none; font-size: 13px; }}"
        )
        bio.document().setDocumentMargin(16)
        bio.setHtml(f"""
        <html><body style='font-family: Inter, -apple-system, Segoe UI, sans-serif; line-height: 1.5; margin:0; padding:0; color:{BROWSER_TEXT}; background-color:{BROWSER_BG};'>
        <p style='margin:0; padding:0; line-height:0; font-size:1px; height:0;'></p>
        <h2 style='color:{BRAND_HEADING_BLUE}; margin-top:0; margin-bottom:2px;'>About the Authors</h2>
        <p style='margin-top:2px;'>Star Trail CleanR is a passion project. I've been
        shooting star trails for over a decade, and the whole time I kept thinking
        <i>somebody should really write a program that gets rid of all the airplane
        and satellite trails.</i> Nobody did. So I finally built one, with a
        lot of help.</p>

        <p>After countless hours of back-and-forth with Claude Code, I described what
        I wanted, Claude wrote the code, we tested it, I pushed back, we tried again.
        Star Trail CleanR wouldn't exist without that partnership.</p>

        <p>Star Trail CleanR is my free gift to the astrophotography community that
        has taught me so much.</p>

        <h3 style='color:{BRAND_HEADING_BLUE}; margin:12px 0 2px 0;'>Links</h3>
        <ul style='margin-top:2px;'>
        <li>Project site: <a href='https://startrailcleanr.com'>StarTrailCleanR.com</a></li>
        <li>Photos for sale: <a href='https://bruceherwigphotographer.square.site/shop/astrophotography/3?page=1&amp;limit=30&amp;sort_by=category_order&amp;sort_order=asc'>bruceherwig.com</a></li>
        <li>Blog: <a href='https://bruceherwig.wordpress.com'>bruceherwig.wordpress.com</a></li>
        </ul>

        <h3 style='color:{BRAND_HEADING_BLUE}; margin:12px 0 2px 0;'>Acknowledgments</h3>
        <p style='margin-top:2px;'>Star Trail CleanR exists because of the generosity of fellow astrophotographers
        who shared their image sequences for AI training, tested early builds, and offered
        feedback. Thank you, all of you.</p>
        <p><a href='https://bruceherwig.wordpress.com/star-trail-cleanr/#Thanks'>See the
        full list of contributors &rarr;</a></p>

        <h3 style='color:{BRAND_HEADING_BLUE}; margin:12px 0 2px 0;'>Version History</h3>
        <p style='margin-top:2px;'>See the full <a href='https://github.com/bruceherwig-dot/star-trail-cleanr/blob/v2-auto-update/CHANGELOG.md'>version history on GitHub</a>.</p>

        <h3 style='color:{BRAND_HEADING_BLUE}; margin:12px 0 2px 0;'>Share Your Work&hellip; Have a Suggestion?</h3>
        <p style='margin-top:2px;'>Got a before-and-after you'd like to share? I would love to see it!<br>
        Have an idea or feedback to make Star Trail CleanR even better? I want to hear it!<br>
        Email me at <a href='mailto:bruceherwig+startrailcleanr@gmail.com?subject=Star%20Trail%20CleanR'>bruceherwig+startrailcleanr@gmail.com</a></p>

        <p style='color:{HINT_TEXT}; margin-top:24px;'>&copy; 2026 Bruce Herwig</p>
        </body></html>
        """)
        layout.addWidget(bio, 1)

        return wrap

    # ── Banner ───────────────────────────────────────────────────────────────

    def _build_banner(self):
        """Build the fixed navy header bar at the very top of the window:
        app icon, the title + a selectable "Beta vN / Trail DetectoR vN"
        subline, a heart-shaped Support (tip jar) button, and the red close X.
        Includes an invisible relaunch button used during development. Returns
        the banner widget."""
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
        # Single QLabel holding both subtitle lines so the user can select
        # and copy the version + detector together as one block. Two separate
        # QLabels forced one-line-at-a-time copy.
        self._header_subline = QLabel(
            f"Beta v{VERSION}\n{self._current_model_display_name()}"
        )
        self._header_subline.setStyleSheet(
            f"color: {BRAND_HEADER_SUB}; font-size: 12px; background: transparent;"
        )
        self._header_subline.setTextInteractionFlags(Qt.TextSelectableByMouse)
        text_col.addWidget(self._header_subline)
        text_col.addStretch()
        outer.addWidget(text_wrap)
        outer.addStretch()

        # Hidden relaunch button (invisible, to the left of Support)
        relaunch_btn = QPushButton("")
        relaunch_btn.setFixedSize(32, 32)
        relaunch_btn.setStyleSheet("QPushButton { background: transparent; border: none; }")
        relaunch_btn.clicked.connect(self._relaunch)
        outer.addWidget(relaunch_btn)

        # Right: Support button. Heart and "Support" use different font sizes
        # so the heart can be visually larger than the word. QPushButton's
        # own text would force one shared size, so we put two QLabels inside
        # the button. WA_TransparentForMouseEvents makes the labels pass
        # clicks through to the button.
        support_btn = QPushButton()
        support_btn.setFixedHeight(36)
        # Width is fixed because a QPushButton with an internal layout
        # sizes to its (empty) text by default, clipping the children. This
        # value comfortably fits heart (22px) + spacing + "Support" (15px)
        # + 18px padding on each side.
        support_btn.setFixedWidth(144)
        support_btn.setCursor(Qt.PointingHandCursor)
        support_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BRAND_SUPPORT_BG}; "
            f"border-radius: 18px; border: 1px solid {BRAND_SUPPORT_BORDER}; "
            f"padding: 0 18px; }}"
            f"QPushButton:hover {{ background-color: {BRAND_SUPPORT_HOVER}; }}"
        )
        _support_inner = QHBoxLayout(support_btn)
        _support_inner.setContentsMargins(0, 0, 0, 0)
        _support_inner.setSpacing(8)
        _support_inner.setAlignment(Qt.AlignCenter)
        _heart_lbl = QLabel("\u2764")
        _heart_lbl.setStyleSheet(
            f"color: {BRAND_SUPPORT_FG}; font-size: 22px; background: transparent;"
        )
        _heart_lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        _support_text = QLabel("Support")
        _support_text.setStyleSheet(
            f"color: {BRAND_SUPPORT_FG}; font-size: 15px; font-weight: bold; background: transparent;"
        )
        _support_text.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        _support_inner.addWidget(_heart_lbl)
        _support_inner.addWidget(_support_text)
        support_btn.setToolTip("Support this project")
        support_btn.clicked.connect(lambda: __import__('webbrowser').open(
            "https://bruceherwigphotographer.square.site/product/tip-jar/WCQQP7HM4SGFWSNBSAFNX7QF"))
        outer.addWidget(support_btn)

        # Right: Close X. Painted via QPainter (see _XCloseButton) so the
        # glyph is centered to the pixel regardless of font metrics.
        quit_btn = _XCloseButton()
        quit_btn.setFixedSize(32, 32)
        quit_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BRAND_QUIT_RED}; "
            f"border-radius: 4px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_QUIT_RED_HOVER}; }}"
        )
        quit_btn.setToolTip("Quit Star Trail CleanR")
        quit_btn.clicked.connect(self.close)
        outer.addWidget(quit_btn)

        return banner

    # ── Update banner (hidden until a newer release is found on GitHub) ──────

    def _build_update_banner(self):
        """Build the orange "new app version available" banner. Hidden until
        the background UpdateCheckThread finds a newer release on GitHub
        (_on_update_result reveals it). Has a Download button and a dismiss X.
        Returns the banner widget."""
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
        dismiss_btn.clicked.connect(self._on_update_banner_dismissed)
        layout.addWidget(dismiss_btn)

        self._update_banner = banner
        self._update_download_url = None
        self._update_banner_tag = ""
        return banner

    def _start_update_check(self):
        """Launch the background app-update check. Its result_ready signal is
        wired to _on_update_result, which shows the orange banner if needed."""
        self._update_thread = UpdateCheckThread(self)
        self._update_thread.result_ready.connect(self._on_update_result)
        self._update_thread.start()

    def _on_update_result(self, result):
        """Handle the background update check's result. `result` carries the
        release "tag" and "download_url". Shows the orange update banner unless
        the user already dismissed this exact tag (via banner or pre-window
        popup)."""
        tag = result.get("tag", "")
        # Stay quiet if the user has already dismissed this release tag
        # (either via the pre-window popup or a previous banner click).
        dismissed = SETTINGS.value("dismissed_update_tag", "", type=str) or ""
        if dismissed == tag:
            return
        self._update_banner_tag = tag
        self._update_download_url = result.get("download_url")
        self._update_label.setText(f"New version available: {tag}")
        self._update_banner.setVisible(True)

    def _on_update_banner_dismissed(self):
        """Hide the banner and remember the user's dismissal so we don't show
        the banner OR the pre-window popup again for this release tag."""
        self._update_banner.setVisible(False)
        if self._update_banner_tag:
            SETTINGS.setValue("dismissed_update_tag", self._update_banner_tag)

    def _on_update_download(self):
        """The orange update banner's Download button.

        Plain English: on Mac and Windows this runs the built-in one-click
        installer (Sparkle / WinSparkle). It downloads the new version,
        installs it in place, and restarts the app for the user. No website,
        no manual download, no reinstall. Only on Linux, which has no built-in
        in-place updater, does this fall back to opening the GitHub download
        page in the browser."""
        if sys.platform == "darwin":
            from modules.sparkle_updater import check_for_updates
            if not check_for_updates():
                self._updater_unavailable_fallback()
        elif sys.platform == "win32":
            from modules.winsparkle_updater import check_for_updates
            if not check_for_updates():
                self._updater_unavailable_fallback()
        elif self._update_download_url:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl(self._update_download_url))

    # ── Model update card (shows when GitHub has a newer trail detector) ─────

    def _build_model_update_card(self):
        """Build the orange "new Trail DetectoR available" card. Hidden until
        ModelUpdateCheckThread finds a newer model release. Carries a title,
        summary, optional credits, a Download-now button (which swaps to an
        inline progress bar then a "Got it"), and a "Not right now" button.
        Unlike an app update, the model file downloads in-place into the user
        folder. Returns the card widget."""
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
        """'model-v2' becomes 'Trail DetectoR v2'. Falls back to the raw tag on parse failure."""
        if not tag:
            return "New model"
        m = re.match(r"^model-v(\d+(?:\.\d+)?)", tag)
        if not m:
            return tag
        num = m.group(1)
        if "." in num:
            num = num.rstrip("0").rstrip(".")
        return f"Trail DetectoR v{num}"

    def _current_model_display_name(self):
        """Return 'Trail DetectoR N' for the currently-active model. Empty string on failure."""
        try:
            from modules.model_update import local_model_version
            return self._model_display_name(local_model_version())
        except Exception:
            return ""

    def _start_model_update_check(self):
        """Launch the background trail-detector model-update check. Its result
        feeds _on_model_update_result, which reveals the orange model card."""
        self._model_update_thread = ModelUpdateCheckThread(self)
        self._model_update_thread.result_ready.connect(self._on_model_update_result)
        self._model_update_thread.start()

    def _on_model_update_result(self, result):
        """Populate and show the model-update card from the check's result
        dict ("tag", "download_url", optional "summary"/"credits")."""
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
        """Model card "Download now" button. Swap the buttons for a progress
        bar and start a ModelDownloadThread streaming the new weights into the
        user model folder. No-op if no download URL was set."""
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
        """Model card "Not right now" button — just hide the card for now;
        the check runs again on the next launch."""
        self._model_card.setVisible(False)

    def _on_model_download_progress(self, done, total):
        """Update the model card's progress bar from the download thread.
        `done`/`total` are bytes; total==0 means the server gave no
        Content-Length, so switch the bar to an indeterminate pulse."""
        if total > 0:
            pct = int(done * 100 / total)
            self._model_progress.setValue(pct)
        else:
            # Server didn't send Content-Length: fall back to an indeterminate pulse.
            if self._model_progress.minimum() != 0 or self._model_progress.maximum() != 0:
                self._model_progress.setRange(0, 0)

    def _on_model_download_finished(self, version):
        """Model download succeeded. Show "<model> installed" with a Got it
        button and refresh the header subline to the newly active detector.
        `version` is the installed release tag."""
        self._model_progress.setVisible(False)
        display = self._model_display_name(version)
        self._model_title.setText(f"{display} installed")
        self._model_summary.setVisible(False)
        self._model_credits.setVisible(False)
        self._model_gotit_btn.setVisible(True)
        self._header_subline.setText(f"Beta v{VERSION}\n{display}")

    def _on_model_download_failed(self, err):
        """Model download failed. Stay quiet (the existing model still works):
        hide the card and let the check retry on the next launch. `err` is a
        short error string, not shown to the user."""
        # Silent fallback: hide the card, try again next launch.
        self._model_card.setVisible(False)

    # ── NVIDIA "coming soon" banner ──────────────────────────────────────────

    def _build_nvidia_banner(self):
        """Build the orange "NVIDIA GPU detected — install GPU support" banner
        (Windows only). Hidden until NvidiaDetectThread reports a usable card
        and the GPU pack isn't already installed. Has Install / Later buttons
        and an inline progress bar shared with the Settings-tab installer.
        Returns the banner widget."""
        banner = QFrame()
        banner.setFixedHeight(44)
        banner.setStyleSheet(f"QFrame {{ background-color: {BRAND_NOTICE_ORANGE}; }}")
        banner.setVisible(False)
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(16, 0, 8, 0)
        layout.setSpacing(12)

        self._nvidia_label = QLabel(
            "NVIDIA GPU detected. Install GPU support for faster processing."
        )
        self._nvidia_label.setStyleSheet(
            "color: white; font-size: 14px; font-weight: bold; background: transparent;"
        )
        layout.addWidget(self._nvidia_label)
        layout.addStretch()

        download_btn = QPushButton("Install")
        download_btn.setFixedHeight(28)
        download_btn.setStyleSheet(
            f"QPushButton {{ background-color: white; color: {BRAND_NOTICE_ORANGE}; font-size: 13px; "
            f"font-weight: bold; border-radius: 4px; padding: 0 16px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_NOTICE_HOVER}; }}"
        )
        download_btn.clicked.connect(self._on_nvidia_download_clicked)
        layout.addWidget(download_btn)
        self._nvidia_install_btn = download_btn

        later_btn = QPushButton("Later")
        later_btn.setFixedHeight(28)
        later_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: white; font-size: 13px; "
            f"border-radius: 4px; padding: 0 12px; border: 1px solid white; }}"
            f"QPushButton:hover {{ background-color: rgba(255,255,255,0.15); }}"
        )
        later_btn.clicked.connect(self._on_nvidia_later_clicked)
        layout.addWidget(later_btn)
        self._nvidia_later_btn = later_btn

        from PySide6.QtWidgets import QProgressBar
        nvidia_progress = QProgressBar()
        nvidia_progress.setFixedHeight(16)
        nvidia_progress.setFixedWidth(220)
        nvidia_progress.setRange(0, 100)
        nvidia_progress.setValue(0)
        nvidia_progress.setVisible(False)
        nvidia_progress.setStyleSheet(
            "QProgressBar { background: rgba(255,255,255,0.3); border-radius: 4px; border: none; }"
            "QProgressBar::chunk { background: white; border-radius: 4px; }"
        )
        layout.addWidget(nvidia_progress)
        self._nvidia_banner_progress = nvidia_progress

        nvidia_progress_label = QLabel("")
        nvidia_progress_label.setStyleSheet(
            "color: white; font-size: 12px; background: transparent;"
        )
        nvidia_progress_label.setVisible(False)
        layout.addWidget(nvidia_progress_label)
        self._nvidia_banner_label = nvidia_progress_label

        self._nvidia_banner = banner
        return banner

    def _start_nvidia_detect(self):
        """Launch two background probes at startup: NvidiaDetectThread (is
        there an NVIDIA card?) and BestDeviceThread (which torch device will
        actually be used: cuda/mps/cpu). Both feed the Settings GPU status
        line and the NVIDIA banner."""
        self._nvidia_thread = NvidiaDetectThread(self)
        self._nvidia_thread.result_ready.connect(self._on_nvidia_detect_result)
        self._nvidia_thread.start()
        self._best_device_thread = BestDeviceThread(self)
        self._best_device_thread.result_ready.connect(self._on_best_device_result)
        self._best_device_thread.start()

    def _on_nvidia_detect_result(self, outcome, detail):
        """Handle the NVIDIA-detection result. `outcome` is "yes"/"no"/etc.
        Show the install banner when a card is present, the GPU pack isn't
        installed, and the user hasn't dismissed the banner before. Then
        refresh the Settings compute-status section."""
        print(f"[nvidia-detect] outcome={outcome} detail={detail}", flush=True)
        self._nvidia_outcome = outcome
        from modules.gpu_pack import is_installed as _gpu_installed
        if (outcome == "yes"
                and not _gpu_installed()
                and not SETTINGS.value("nvidia_banner_dismissed", False, type=bool)):
            self._nvidia_banner.setVisible(True)
        self._refresh_compute_section()

    def _on_best_device_result(self, device):
        """Record which compute device torch will use ("cuda"/"mps"/"cpu")
        and refresh the Settings compute-status line accordingly."""
        self._compute_device = device
        self._refresh_compute_section()

    def _on_nvidia_download_clicked(self):
        """Install GPU Support button (fired from either the banner or the
        Settings tab). Confirms the ~3-4 GB download with the user, then starts
        a GpuPackInstallThread and wires the UI (banner or Settings progress
        bar) to its progress/finished/failed signals."""
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("Install GPU Support")
        msg.setText(
            "GPU support requires a one-time download of approximately 3-4 GB from pytorch.org.\n\n"
            "Once installed, it survives Star Trail CleanR updates automatically.\n\n"
            "Star Trail CleanR will need to restart after installation to activate GPU support."
        )
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.button(QMessageBox.Ok).setText("Install")
        msg.button(QMessageBox.Cancel).setText("Not Now")
        if msg.exec() != QMessageBox.Ok:
            return

        self._gpu_install_via_banner = self._nvidia_banner.isVisible()
        if self._gpu_install_via_banner:
            self._nvidia_label.setVisible(False)
            self._nvidia_install_btn.setVisible(False)
            self._nvidia_later_btn.setVisible(False)
            self._nvidia_banner_progress.setRange(0, 100)
            self._nvidia_banner_progress.setValue(0)
            self._nvidia_banner_progress.setVisible(True)
            self._nvidia_banner_label.setText("Starting download...")
            self._nvidia_banner_label.setVisible(True)
        self._gpu_download_btn.setVisible(False)
        self._gpu_upgrade_browser.setVisible(False)
        self._gpu_progress.setRange(0, 100)
        self._gpu_progress.setValue(0)
        self._gpu_progress.setFormat("%p%")
        self._gpu_progress.setVisible(True)
        self._gpu_progress_label.setText("Starting download...")
        self._gpu_progress_label.setVisible(True)

        self._gpu_install_thread = GpuPackInstallThread(self)
        self._gpu_install_thread.progress.connect(self._on_gpu_install_progress)
        self._gpu_install_thread.finished_ok.connect(self._on_gpu_install_finished)
        self._gpu_install_thread.failed.connect(self._on_gpu_install_failed)
        self._gpu_install_thread.start()

    def _on_gpu_install_progress(self, label, done, total):
        """Update the GPU-install progress bar(s) from the install thread.
        `label` is the current step text; `done`/`total` are bytes (total==0 =
        indeterminate phase). Mirrors progress to the banner bar too when the
        install was launched from the banner."""
        if total > 0:
            pct = int(done * 100 / total)
            self._gpu_progress.setRange(0, 100)
            self._gpu_progress.setValue(pct)
            if self._gpu_install_via_banner:
                self._nvidia_banner_progress.setRange(0, 100)
                self._nvidia_banner_progress.setValue(pct)
        else:
            self._gpu_progress.setRange(0, 0)
            if self._gpu_install_via_banner:
                self._nvidia_banner_progress.setRange(0, 0)
        self._gpu_progress_label.setText(label)
        if self._gpu_install_via_banner:
            self._nvidia_banner_label.setText(label)

    def _on_gpu_install_finished(self):
        """GPU pack installed. Hide progress UI and reveal the "Restart Now to
        Activate GPU Support" button (the new torch wheels only take effect on
        a fresh process)."""
        if self._gpu_install_via_banner:
            self._nvidia_banner.setVisible(False)
            self._nvidia_label.setVisible(True)
            self._nvidia_install_btn.setVisible(True)
            self._nvidia_later_btn.setVisible(True)
            self._nvidia_banner_progress.setVisible(False)
            self._nvidia_banner_label.setVisible(False)
        self._gpu_progress.setVisible(False)
        self._gpu_progress_label.setVisible(False)
        self._gpu_restart_btn.setVisible(True)
        self._compute_status_label.setText("GPU support installed. Restart to activate.")

    def _on_gpu_install_failed(self, err):
        """GPU install failed. Restore the install controls and show an error
        dialog with the message `err` (already user-friendly from the install
        thread) plus a "More Info" link to the setup guide."""
        from PySide6.QtWidgets import QMessageBox
        if self._gpu_install_via_banner:
            self._nvidia_banner_progress.setVisible(False)
            self._nvidia_banner_label.setVisible(False)
            self._nvidia_label.setVisible(True)
            self._nvidia_install_btn.setVisible(True)
            self._nvidia_later_btn.setVisible(True)
        self._gpu_progress.setVisible(False)
        self._gpu_progress_label.setVisible(False)
        self._gpu_download_btn.setVisible(True)
        self._gpu_upgrade_browser.setVisible(True)
        box = QMessageBox(self)
        box.setWindowTitle("GPU Installation Failed")
        box.setIcon(QMessageBox.Critical)
        box.setText(f"GPU support could not be installed.\n\n{err}")
        box.addButton(QMessageBox.Ok)
        info_btn = box.addButton("More Info", QMessageBox.HelpRole)
        box.exec()
        if box.clickedButton() is info_btn:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl(_GPU_BUILD_URL))

    def _on_gpu_help_clicked(self):
        """Open the NVIDIA GPU-setup troubleshooting guide in the browser."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(_GPU_BUILD_URL))

    def _on_gpu_clear_clicked(self):
        """Settings "Clear GPU Support Files" link. After confirming, delete
        the installed GPU pack so the user can start a clean reinstall; report
        success or, if files are locked, point them at the folder to delete
        manually."""
        from PySide6.QtWidgets import QMessageBox
        from modules.gpu_pack import clear_gpu_files, get_override_dir
        confirm = QMessageBox.question(
            self,
            "Clear GPU Support Files",
            "This will delete the GPU support files so you can start a fresh install.\n\n"
            "You can reinstall them at any time from this Settings page.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        ok, detail = clear_gpu_files()
        if ok:
            QMessageBox.information(
                self,
                "GPU Support Files Cleared",
                "GPU support files have been removed. Click Install GPU Support to start fresh."
            )
        else:
            override_dir = get_override_dir()
            QMessageBox.critical(
                self,
                "Could Not Clear GPU Files",
                f"Some files could not be removed. You can delete this folder manually:\n\n"
                f"{override_dir}\n\n"
                f"Details: {detail}"
            )

    def _on_nvidia_later_clicked(self):
        """NVIDIA banner "Later" button. Remember the dismissal so the banner
        stays hidden on future launches, and hide it now."""
        SETTINGS.setValue("nvidia_banner_dismissed", True)
        self._nvidia_banner.setVisible(False)

    def _refresh_compute_section(self):
        """Rebuild the Settings tab's GPU-status line from the latest device
        and NVIDIA-detection results. Picks one of several status strings
        (Apple MPS active, NVIDIA CUDA active, CPU with various GPU-pack
        hints) and shows the Windows GPU-install controls only when an NVIDIA
        card is present but running on CPU. Safe to call before the Settings
        widgets exist (returns early)."""
        if not hasattr(self, "_compute_status_label"):
            return
        import platform as _pl
        device = self._compute_device
        outcome = self._nvidia_outcome
        gpu_mismatch = bool(os.environ.get('STC_GPU_VERSION_MISMATCH'))

        if device == "mps":
            status = "Apple MPS — GPU acceleration active"
        elif device == "cuda":
            status = "NVIDIA CUDA — GPU acceleration active"
        elif device == "cpu" and outcome == "yes" and gpu_mismatch:
            status = ("CPU — GPU pack version mismatch. "
                      "Reinstall the GPU pack for this version of Star Trail CleanR "
                      "to re-enable acceleration.")
        elif device == "cpu" and outcome == "yes" and os.environ.get('STC_CUDA_UNSUPPORTED'):
            status = ("NVIDIA GPU detected but your card isn't supported by the current "
                      "GPU pack — running on CPU.")
        elif device == "cpu" and outcome == "yes":
            status = "CPU — NVIDIA GPU detected. Install the GPU pack for faster processing."
        elif device == "cpu" and _pl.system() == "Windows":
            status = "CPU — no GPU acceleration"
        elif device == "cpu":
            status = "CPU processing only — GPU acceleration not available on this device"
        else:
            return

        self._compute_status_label.setText(status)
        show_upgrade = (
            _pl.system() == "Windows"
            and outcome == "yes"
            and device == "cpu"
            and not gpu_mismatch
            and not os.environ.get('STC_CUDA_UNSUPPORTED')
        )
        self._gpu_install_widget.setVisible(show_upgrade)
        self._gpu_upgrade_browser.setVisible(show_upgrade)
        self._gpu_download_btn.setVisible(show_upgrade)
        if hasattr(self, '_gpu_clear_btn'):
            self._gpu_clear_btn.setVisible(show_upgrade)
        if hasattr(self, '_gpu_help_btn'):
            self._gpu_help_btn.setVisible(show_upgrade)

    def _relaunch(self):
        """Close and reopen the app."""
        import subprocess
        from PySide6.QtWidgets import QApplication as _QApp
        _sock = getattr(_QApp.instance(), '_lock_socket', None)
        if _sock is not None:
            try:
                _sock.close()
            except Exception:
                pass
        if getattr(sys, 'frozen', False):
            subprocess.Popen([sys.executable, '--cleanr-relaunch'])
        else:
            subprocess.Popen([sys.executable, os.path.abspath(__file__), '--cleanr-relaunch'])
        self.close()

    # ── Setup page ───────────────────────────────────────────────────────────

    def _build_setup_page(self):
        """Build page 0 of the Main tab: the six-step setup form the user fills
        in before a run. Step 1 input folder + live frame count, Step 2 output
        folder (auto-filled to a "cleaned" subfolder), Step 3 optional
        foreground mask, Step 4 number of frames (plus dev-only start/end), Step
        5 output format + JPEG quality (plus dev-only model picker), Step 6 the
        big "Clean My Stars!" button. Wraps it all in a scroll area, restores
        persisted choices from QSettings, and wires change handlers. Added to
        the page stack as index 0."""
        page = QWidget()
        # Hold onto the inner widget so the window can lock its minimum
        # height to whatever the layout actually needs. See _lock_min_height.
        self._setup_inner = page
        layout = QVBoxLayout(page)
        layout.setSpacing(4)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setAlignment(Qt.AlignTop)

        # Sizes bumped 2026-04-28 to match Inter's smaller x-height vs the
        # platform default fonts. Old 15/12 pt rendered too small once we
        # forced Inter app-wide.
        lbl_font = QFont()
        lbl_font.setPointSize(19)
        lbl_font.setBold(True)

        step_font = QFont()
        step_font.setPointSize(15)

        # Headline + subtitle
        headline = QLabel("Remove the Trails. Keep the Stars.")
        headline_font = QFont()
        headline_font.setPointSize(24)
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

        layout.addSpacing(8)

        # ── Step 1: Select Images ────────────────────────────────────────────
        # Heading + format hint are ONE QLabel (rich text) so layout never
        # has to align two QLabels of different fonts on the same line —
        # that produced overlap. Frame count is dynamic, lives in its own
        # label to the right.
        # Option A: heading + frame count share one row. Heading on the left
        # at its sizeHint width; count on the far right, pushed there by a
        # stretch. No separate count row, no extra vertical space.
        step1_row = QHBoxLayout()
        step1_row.setSpacing(12)
        step1 = QLabel(
            "<span style='font-size:19pt; font-weight:bold;'>1. Select Folder with Your Star Trail Images</span>"
            f"&nbsp;&nbsp;<span style='font-size:14pt; color:{MUTED_TEXT}; vertical-align:baseline;'>(.JPG, .TIF 8/16-bit, and RAW)</span>"
        )
        step1.setTextFormat(Qt.RichText)
        step1_row.addWidget(step1)
        step1_row.addStretch(1)
        layout.addLayout(step1_row)

        # Frame count label sits in its own row above the input field, with
        # stretch proportions matching row_in below (4 empty : 2 label).
        # That keeps the label horizontally centered above the Browse +
        # Open Folder buttons no matter how wide the window is. Replaces
        # the prior "padding-right: 100px" workaround which only worked at
        # one fixed window width.
        count_row = QHBoxLayout()
        count_row.setContentsMargins(0, 0, 0, 0)
        count_row.addStretch(4)
        self._frame_count_label = QLabel("")
        self._frame_count_label.setAlignment(Qt.AlignCenter)
        self._frame_count_label.setStyleSheet(
            "color: #5b9bd5; font-size: 15pt;"
        )
        count_row.addWidget(self._frame_count_label, 2)
        layout.addLayout(count_row)

        row_in = QHBoxLayout()
        self._folder_input = QLineEdit()
        self._folder_input.setPlaceholderText("Select folder using Browse\u2026")
        self._folder_input.textChanged.connect(self._auto_output)
        self._folder_input.textChanged.connect(self._update_input_open_btn_state)
        self._folder_input.editingFinished.connect(self._on_input_edited)
        row_in.addWidget(self._folder_input, 4)
        browse_in = QPushButton("Browse\u2026")
        browse_in.setFixedHeight(34)
        browse_in.setStyleSheet(_secondary_btn_css())
        browse_in.clicked.connect(self._browse_input)
        row_in.addWidget(browse_in, 1)
        self._input_open_btn = QPushButton("Open Folder")
        self._input_open_btn.setFixedHeight(34)
        self._input_open_btn.setStyleSheet(_secondary_btn_css())
        self._input_open_btn.setEnabled(False)
        self._input_open_btn.clicked.connect(self._open_setup_input_folder)
        row_in.addWidget(self._input_open_btn, 1)
        layout.addLayout(row_in)
        layout.addSpacing(4)

        # ── Step 2: Select Output ────────────────────────────────────────────
        step2 = QLabel(
            "<span style='font-size:19pt; font-weight:bold;'>2. Select Output Folder</span>"
            f"&nbsp;&nbsp;<span style='font-size:14pt; color:{MUTED_TEXT}; vertical-align:baseline;'>(default: a \u2018Cleaned\u2019 folder inside your originals)</span>"
        )
        step2.setTextFormat(Qt.RichText)
        layout.addWidget(step2)

        row_out = QHBoxLayout()
        self._output_input = QLineEdit()
        self._output_input.setPlaceholderText("Auto-fills from input folder")
        self._output_input.textChanged.connect(self._update_output_open_btn_state)
        row_out.addWidget(self._output_input, 4)
        browse_out = QPushButton("Browse\u2026")
        browse_out.setFixedHeight(34)
        browse_out.setStyleSheet(_secondary_btn_css())
        browse_out.clicked.connect(self._browse_output)
        row_out.addWidget(browse_out, 1)
        self._output_open_btn = QPushButton("Open Folder")
        self._output_open_btn.setFixedHeight(34)
        self._output_open_btn.setStyleSheet(_secondary_btn_css())
        self._output_open_btn.setEnabled(False)
        self._output_open_btn.clicked.connect(self._open_setup_output_folder)
        row_out.addWidget(self._output_open_btn, 1)
        layout.addLayout(row_out)
        layout.addSpacing(4)

        # ── Step 3: Foreground Mask ──────────────────────────────────────────
        step3 = QLabel(
            "<span style='font-size:19pt; font-weight:bold;'>3. Foreground Mask (optional)</span>"
            f"&nbsp;&nbsp;<span style='font-size:14pt; color:{MUTED_TEXT}; vertical-align:baseline;'>Not required, but helpful \u2014 keeps the AI focused on the sky</span>"
        )
        step3.setTextFormat(Qt.RichText)
        layout.addWidget(step3)

        mask_row = QHBoxLayout()
        self._mask_btn = QPushButton("Create Mask\u2026")
        self._mask_btn.setFixedHeight(34)
        self._mask_btn.setFixedWidth(160)
        self._mask_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BRAND_RUN_GREEN}; color: white; font-size: 16px; "
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
        layout.addSpacing(4)

        # ── Step 4: Number of Images ─────────────────────────────────────────
        step4 = QLabel(
            "<span style='font-size:19pt; font-weight:bold;'>4. Number of Images to Process</span>"
            f"&nbsp;&nbsp;<span style='font-size:14pt; color:{MUTED_TEXT}; vertical-align:baseline;'>Recommended: test a small batch before doing a full run</span>"
        )
        step4.setTextFormat(Qt.RichText)
        layout.addWidget(step4)

        self._frame_limit = QComboBox()
        self._frame_limit.setEditable(True)
        self._frame_limit.addItems(["20", "50", "100", "250", "500", "1000", "All Frames"])
        self._frame_limit.lineEdit().setPlaceholderText("All Frames")
        self._frame_limit.lineEdit().setValidator(QIntValidator(1, 999999))
        self._frame_limit.setFixedWidth(160)
        layout.addWidget(self._frame_limit)

        if not getattr(sys, 'frozen', False):
            dev_row = QHBoxLayout()
            dev_label = QLabel("Dev: Start:")
            dev_label.setFont(step_font)
            dev_row.addWidget(dev_label)
            self._dev_start_frame = QSpinBox()
            self._dev_start_frame.setRange(0, 99999)
            self._dev_start_frame.setValue(0)
            self._dev_start_frame.setFixedWidth(90)
            dev_row.addWidget(self._dev_start_frame)
            dev_end_label = QLabel("End (0 = last):")
            dev_end_label.setFont(step_font)
            dev_row.addWidget(dev_end_label)
            self._dev_end_frame = QSpinBox()
            self._dev_end_frame.setRange(0, 99999)
            self._dev_end_frame.setValue(0)
            self._dev_end_frame.setFixedWidth(90)
            dev_row.addWidget(self._dev_end_frame)
            dev_row.addStretch()
            layout.addLayout(dev_row)
        else:
            self._dev_start_frame = None
            self._dev_end_frame = None

        layout.addSpacing(4)

        # ── Step 5: Output Options ───────────────────────────────────────────
        step5 = QLabel(
            "<span style='font-size:19pt; font-weight:bold;'>5. Output Options</span>"
            f"&nbsp;&nbsp;<span style='font-size:14pt; color:{MUTED_TEXT}; vertical-align:baseline;'>File format and quality</span>"
        )
        step5.setTextFormat(Qt.RichText)
        layout.addWidget(step5)

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
        # Don't hardcode a pixel width. Qt's sizeHint already computes
        # the right size from the current font's metrics + the native
        # spinbox chrome (arrows + frame padding) — which differs by
        # platform. The previous setFixedWidth(60) was tuned on Mac
        # against Inter and silently hid the digits on Windows where
        # native arrows take up more horizontal room.
        hp_row.addWidget(self._jpeg_quality)
        hp_row.addStretch(1)

        self._format_combo.currentTextChanged.connect(self._on_format_changed)

        layout.addLayout(hp_row)
        layout.addSpacing(6)

        if _DEV_SWITCHER_ENABLED:
            dev_row = QHBoxLayout()
            dev_label = QLabel("Model (dev):")
            dev_label.setFont(step_font)
            dev_row.addWidget(dev_label)
            self._dev_model_combo = QComboBox()
            self._dev_model_combo.setFixedWidth(300)
            _choices = _get_dev_model_choices()
            _saved = SETTINGS.value("dev_model_override", "", type=str)
            _saved_idx = 0
            for i, (name, pt_path) in enumerate(_choices):
                self._dev_model_combo.addItem(name, pt_path)
                if pt_path == _saved:
                    _saved_idx = i
            if _choices:
                self._dev_model_combo.setCurrentIndex(_saved_idx)
                SETTINGS.setValue("dev_model_override", _choices[_saved_idx][1])
            self._dev_model_combo.currentIndexChanged.connect(
                lambda idx: SETTINGS.setValue(
                    "dev_model_override", self._dev_model_combo.itemData(idx)))
            dev_row.addWidget(self._dev_model_combo)
            dev_row.addStretch(1)
            layout.addLayout(dev_row)
            layout.addSpacing(6)

        # ── Step 6: Run ──────────────────────────────────────────────────────
        step6 = QLabel(
            "<span style='font-size:19pt; font-weight:bold;'>6. Remove airplane and satellite trails</span>"
            f"&nbsp;&nbsp;<span style='font-size:14pt; color:{MUTED_TEXT}; vertical-align:baseline;'>Processing time depends on frame count, pixel count of your images, and computer speed</span>"
        )
        step6.setTextFormat(Qt.RichText)
        layout.addWidget(step6)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: red; font-size: 13px;")
        layout.addWidget(self._error_label)

        layout.addSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(16)

        self._run_btn = QPushButton("Clean My Stars!")
        self._run_btn.setFixedHeight(60)
        self._run_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BRAND_RUN_GREEN}; color: white; font-size: 26px; "
            f"font-weight: bold; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_RUN_GREEN_HOVER}; }}"
            f"QPushButton:disabled {{ background-color: {DISABLED_BTN_BG}; }}"
        )
        self._run_btn.clicked.connect(self._run)
        btn_row.addWidget(self._run_btn, 2)

        self._setup_open_btn = QPushButton("Open Cleaned Folder")
        self._setup_open_btn.setFixedHeight(48)
        self._setup_open_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BRAND_HEADING_BLUE}; color: white; font-size: 22px; "
            f"font-weight: bold; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_HEADING_HOVER}; }}"
            f"QPushButton:disabled {{ background-color: {DISABLED_BTN_BG}; }}"
        )
        self._setup_open_btn.setEnabled(False)
        self._setup_open_btn.clicked.connect(self._open_output_from_setup)
        btn_row.addWidget(self._setup_open_btn, 1)
        layout.addLayout(btn_row)

        scroll = QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._stack.addWidget(scroll)

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
        else:
            self._frame_limit.setCurrentText(last_frame_limit)

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
        """Build page 1 of the Main tab: the live processing view shown during
        a run. Holds the running trail counter and per-frame stats, the fat
        overall progress bar with time estimate/elapsed, the batch label, the
        two per-step (Detecting / Repairing) bars, and a 50/50 split of the
        scrolling Star Log on the left and a community/error-contact panel on
        the right. Bottom row has Cancel and Open-Cleaned-Folder buttons.
        Added to the page stack as index 1."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        title = QLabel("Cleaning in Progress")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        self._process_title = title

        self._trail_counter_label = QLabel("")
        self._trail_counter_label.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {MUTED_TEXT};"
        )
        self._trail_counter_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._trail_counter_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Secondary run stats under the trail counter: avg trails/frame and
        # avg seconds/frame. Running averages over the whole run so far,
        # refreshed once per frame (not on the 250ms clock).
        self._run_stats_label = QLabel("")
        self._run_stats_label.setStyleSheet(f"font-size: 14px; color: {MUTED_TEXT};")
        self._run_stats_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._run_stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._stats_frames_done = 0

        self._run_source_label = QLabel("")
        self._run_source_label.setStyleSheet(f"font-size: 19px; color: {MUTED_TEXT};")
        self._run_source_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self._run_source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        title_row = QHBoxLayout()
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self._run_source_label)
        title_row.addStretch()
        counter_col = QVBoxLayout()
        counter_col.setSpacing(2)
        counter_col.addWidget(self._trail_counter_label)
        counter_col.addWidget(self._run_stats_label)
        title_row.addLayout(counter_col)
        layout.addLayout(title_row)

        # ── Overall progress bar (fat) ──
        frame_label_row = QHBoxLayout()
        self._frame_counter = QLabel("")
        self._frame_counter.setStyleSheet(f"font-size: 15px; color: {MUTED_TEXT};")
        self._frame_counter.setTextInteractionFlags(Qt.TextSelectableByMouse)
        frame_label_row.addWidget(self._frame_counter)
        frame_label_row.addSpacing(20)
        self._initial_est_label = QLabel("")
        self._initial_est_label.setStyleSheet(f"font-size: 15px; color: {MUTED_TEXT};")
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
            f"background: {CARD_BG}; text-align: center; font-weight: bold; font-size: 16px; color: {CARD_TEXT}; }}"
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
        self._elapsed_label.setStyleSheet(f"font-size: 14px; color: {MUTED_TEXT};")
        self._elapsed_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        time_row.addWidget(self._elapsed_label)
        layout.addLayout(time_row)

        # ── Batch label with spinner ──
        self._batch_label = QLabel("")
        batch_font = QFont()
        batch_font.setPointSize(18)
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
        self._step1_label.setWordWrap(True)
        self._step1_label.setStyleSheet(f"font-size: 14px; color: {HINT_TEXT};")
        self._step1_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        step1_row.addWidget(self._step1_label)
        self._step1_bar = QProgressBar()
        self._step1_bar.setFixedHeight(30)
        self._step1_bar.setTextVisible(True)
        self._step1_bar.setValue(0)
        self._step1_bar.setFormat("0%")
        self._step1_bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {CARD_BORDER}; border-radius: 8px; "
            f"background: {CARD_BG}; text-align: center; font-weight: bold; font-size: 15px; color: {CARD_TEXT}; }}"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #4a9eff, stop:1 #66b3ff); border-radius: 7px; }"
        )
        step1_row.addWidget(self._step1_bar, 1)
        layout.addLayout(step1_row)

        # ── Step 2: Repair ──
        step2_row = QHBoxLayout()
        self._step2_label = QLabel("Repairing\nwaiting")
        self._step2_label.setFixedWidth(120)
        self._step2_label.setWordWrap(True)
        self._step2_label.setStyleSheet(f"font-size: 14px; color: {HINT_TEXT};")
        self._step2_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        step2_row.addWidget(self._step2_label)
        self._step2_bar = QProgressBar()
        self._step2_bar.setFixedHeight(30)
        self._step2_bar.setTextVisible(True)
        self._step2_bar.setValue(0)
        self._step2_bar.setFormat("0%")
        self._step2_bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {CARD_BORDER}; border-radius: 8px; "
            f"background: {CARD_BG}; text-align: center; font-weight: bold; font-size: 15px; color: {CARD_TEXT}; }}"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #4a9eff, stop:1 #66b3ff); border-radius: 7px; }"
        )
        step2_row.addWidget(self._step2_bar, 1)
        layout.addLayout(step2_row)

        # The detail / status line lives at the top of the right-side
        # community panel under the "Star Log" header. See
        # _build_run_community_panel for its creation.

        # Post-run stats are shown in a centered modal dialog, not inline,
        # so they can't fight with the log area for vertical space or hide
        # the Back to Setup button. See _show_run_complete_dialog.

        # ── Log area + community panel (50/50 horizontal split) ──
        log_row = QHBoxLayout()
        log_row.setSpacing(20)

        # Left column: "Star Log" title + scrolling log box.
        log_col = QVBoxLayout()
        log_col.setSpacing(8)
        log_col.setContentsMargins(0, 0, 0, 0)

        # Title row: centered "Star Log" heading with a "View Star Log" link to
        # its right. The link opens the saved run-log text file; it stays hidden
        # during a run and appears only once the run ends (finished/stopped/error).
        _title_row = QHBoxLayout()
        _title_row.setContentsMargins(0, 0, 0, 0)
        _title_row.addStretch(1)
        self._star_log_title = QLabel("Star Log")
        self._star_log_title.setAlignment(Qt.AlignCenter)
        self._star_log_title.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {CARD_TEXT};"
        )
        _title_row.addWidget(self._star_log_title)
        # A flat QPushButton styled to look like a link. QLabel's <a> link did
        # not reliably fire linkActivated in this layout; a button's clicked
        # signal always fires.
        self._view_log_link = QPushButton("View Star Log (with run detail)")
        self._view_log_link.setFlat(True)
        self._view_log_link.setCursor(Qt.PointingHandCursor)
        _vlf = self._view_log_link.font()
        _vlf.setPointSize(14)
        _vlf.setUnderline(True)
        self._view_log_link.setFont(_vlf)
        self._view_log_link.setStyleSheet(
            f"QPushButton {{ border: none; background: transparent; "
            f"color: {BRAND_HEADING_BLUE}; padding: 0 0 0 10px; }}"
        )
        self._view_log_link.clicked.connect(self._view_star_log)
        self._view_log_link.setVisible(False)
        _title_row.addWidget(self._view_log_link)
        _title_row.addStretch(1)
        log_col.addLayout(_title_row)

        self._status_out = QTextEdit()
        self._status_out.setReadOnly(True)
        # Center every line of the log within the box.
        from PySide6.QtGui import QTextOption as _QTextOpt
        _opt = self._status_out.document().defaultTextOption()
        _opt.setAlignment(Qt.AlignCenter)
        self._status_out.document().setDefaultTextOption(_opt)
        log_col.addWidget(self._status_out, 1)

        log_row.addLayout(log_col, 1)

        log_row.addWidget(self._build_run_community_panel(), 1)
        layout.addLayout(log_row, 1)

        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignBottom)
        btn_row.setSpacing(16)

        self._cancel_btn = QPushButton("Cancel Cleaning")
        self._cancel_btn.setFixedHeight(60)
        self._cancel_btn.setStyleSheet(
            f"QPushButton {{ background-color: {SECONDARY_BTN_BG}; color: white; font-size: 26px; "
            f"font-weight: bold; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: {DISABLED_BTN_HOVER}; }}"
        )
        self._cancel_btn.clicked.connect(self._cancel_run)
        btn_row.addWidget(self._cancel_btn, 2)

        self._open_folder_btn = QPushButton("Open Cleaned Folder")
        self._open_folder_btn.setFixedHeight(48)
        self._open_folder_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BRAND_HEADING_BLUE}; color: white; font-size: 22px; "
            f"font-weight: bold; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: {BRAND_HEADING_HOVER}; }}"
        )
        self._open_folder_btn.clicked.connect(self._open_output_folder)
        btn_row.addWidget(self._open_folder_btn, 1)

        layout.addLayout(btn_row)

        self._stack.addWidget(page)

    def _build_run_community_panel(self):
        """Build the right-hand panel of the processing page: a "tag
        @bruceherwig #StarTrailCleanR" social nudge and an error-report email
        link that pre-fills app version + OS. Stashes the support email and a
        pre-built mailto URL on self. Returns the panel widget."""
        import platform as _plat
        import urllib.parse as _urlp
        import html as _html

        sysname = _plat.system()
        if sysname == "Darwin":
            machine = _plat.machine()
            os_line = "Mac (Apple Silicon)" if machine == "arm64" else "Mac (Intel)"
        elif sysname == "Windows":
            os_line = f"Windows {_windows_release_label()}"
        elif sysname == "Linux":
            os_line = "Linux"
        else:
            os_line = sysname

        self._support_email = "bruceherwig+startrailcleanr@gmail.com"
        subject = "Star Trail CleanR error report"
        body = (
            "Hi Bruce,\n\n"
            "[Describe what happened]\n\n"
            f"App version: Beta v{VERSION}\n"
            f"OS: {os_line}\n"
        )
        self._support_mail_url = (
            f"mailto:{self._support_email}"
            f"?subject={_urlp.quote(subject)}"
            f"&body={_urlp.quote(body)}"
        )
        # HTML-escape the URL so the & in subject=...&body=... doesn't trip
        # Qt's HTML parser (a literal & inside an href value can silently
        # truncate the URL).
        safe_url = _html.escape(self._support_mail_url, quote=True)

        panel = QWidget()
        # Expanding vertical policy: panel fills the full log_row height
        # instead of being sized to sizeHint and centered. With the layout
        # below using a bottom stretch, content sits at the top — aligned
        # with the top edge of the log box on the left.
        from PySide6.QtWidgets import QSizePolicy as _QSP
        panel.setSizePolicy(_QSP.Preferred, _QSP.Expanding)
        v = QVBoxLayout(panel)
        v.setContentsMargins(24, 0, 24, 12)
        v.setSpacing(0)
        # Top stretch paired with the existing bottom stretch below so the
        # content (community message + status flash) sits vertically
        # centered in the panel at any window size, instead of pinned to
        # the top.
        v.addStretch(1)

        # Two separate QLabels — one per paragraph. Single-paragraph
        # plain word-wrapped QLabels size themselves cleanly via
        # Qt's natural sizeHint. The earlier single-QLabel-with-HTML
        # approach had buggy heightForWidth interactions that clipped
        # the email link.
        # ONE text box. Both messages, blank line between. Width is
        # whatever the layout gives it; height is set explicitly to the
        # rendered document height after the HTML is laid out, so the
        # box is exactly tall enough to hold all the text — no clipping,
        # no scrollbar, no per-line sizing tricks.
        from PySide6.QtWidgets import QTextBrowser
        from PySide6.QtCore import QUrl as _QUrl

        content_html = (
            "<div style='text-align:center;'>"
            f"<p style='margin:0; font-size:16px; color:{CARD_TEXT};'>"
            "Help spread the word! When you share on social media, "
            "tag <b>@bruceherwig #StarTrailCleanR</b>"
            "</p>"
            f"<p style='margin:24px 0 0 0; font-size:16px; color:{CARD_TEXT};'>"
            "Did you get an error message? Take a screenshot and email "
            f"<a href=\"{safe_url}\" style='color:{BRAND_HEADING_BLUE};'>"
            f"{self._support_email}</a>"
            "</p>"
            "</div>"
        )
        self._community_lbl = QTextBrowser()
        self._community_lbl.setOpenLinks(False)
        self._community_lbl.setFrameShape(QFrame.NoFrame)
        self._community_lbl.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._community_lbl.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._community_lbl.setStyleSheet("background: transparent;")
        self._community_lbl.setHtml(content_html)
        self._community_lbl.anchorClicked.connect(
            lambda url: self._on_support_link_clicked(url.toString())
        )
        # Fixed height in LOGICAL pixels. Qt scales logical to device
        # pixels automatically, and content inside the box uses the same
        # logical pixel space, so 160 holds the same wrapped lines at
        # 100% / 125% / 150% DPI. Tuned to fit at Setup-natural-locked
        # window size on Bruce's Mac at 100% — verified live.
        self._community_lbl.setFixedHeight(160)
        v.addWidget(self._community_lbl)

        v.addSpacing(10)

        # Hidden status label that flashes "Email copied" briefly on click.
        self._community_status = QLabel("")
        self._community_status.setAlignment(Qt.AlignCenter)
        self._community_status.setStyleSheet(
            f"font-size: 13px; color: {MUTED_TEXT};"
        )
        v.addWidget(self._community_status)

        v.addStretch(1)

        return panel

    def _on_support_link_clicked(self, url):
        """Click handler for the support email link. Always copies the
        address to the clipboard (works regardless of mail-app setup) and
        also tries to open the user's mail app with the message
        pre-filled. A short status line confirms the click registered."""
        from PySide6.QtCore import QUrl as _QUrl
        from PySide6.QtGui import QDesktopServices as _QDS
        import urllib.parse as _urlp
        import platform as _plat

        QApplication.clipboard().setText(self._support_email)

        log_text = self._status_out.toPlainText() if hasattr(self, '_status_out') else ""
        if len(log_text) > 1500:
            log_text = "...(truncated)\n" + log_text[-1500:]

        sysname = _plat.system()
        machine = _plat.machine()
        subject = "Star Trail CleanR error report"
        body = (
            "Hi Bruce,\n\n"
            "[Describe what happened]\n\n"
            f"App version: Beta v{VERSION}\n"
            f"OS: {sysname} ({machine})\n\n"
            "--- Star Log ---\n"
            f"{log_text}\n"
        )
        mailto_url = (
            f"mailto:{self._support_email}"
            f"?subject={_urlp.quote(subject)}"
            f"&body={_urlp.quote(body)}"
        )
        _QDS.openUrl(_QUrl(mailto_url))
        self._community_status.setText("Email copied. Paste it anywhere.")
        QTimer.singleShot(
            3000, lambda: self._community_status.setText("")
        )

    # ── Browse / validation ──────────────────────────────────────────────────

    def _browse_input(self):
        """Step 1 Browse button. Open a folder picker (starting at the last
        used input dir), set the chosen path, remember it, and refresh the mask
        status and frame count."""
        last_dir = SETTINGS.value("last_input_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder", last_dir)
        if folder:
            self._folder_input.setText(folder)
            SETTINGS.setValue("last_input_dir", folder)
            self._update_mask_status()
            self._update_frame_count()

    def _on_input_edited(self):
        """Fired when the user finishes typing in the input-folder field;
        refresh the live frame count for the typed path."""
        self._update_frame_count()

    def _update_input_open_btn_state(self):
        """Enable the Step 1 "Open Folder" button only when the input field
        holds a real existing directory."""
        path = self._folder_input.text().strip()
        self._input_open_btn.setEnabled(bool(path) and os.path.isdir(path))

    def _update_output_open_btn_state(self):
        """Enable the Step 2 "Open Folder" button only when the output field
        holds a real existing directory."""
        path = self._output_input.text().strip()
        self._output_open_btn.setEnabled(bool(path) and os.path.isdir(path))

    def _open_setup_input_folder(self):
        """Open the Step 1 input folder in the OS file manager (if it exists)."""
        path = self._folder_input.text().strip()
        if path and os.path.isdir(path):
            _open_folder_in_file_manager(path)

    def _open_setup_output_folder(self):
        """Open the Step 2 output folder in the OS file manager (if it exists)."""
        path = self._output_input.text().strip()
        if path and os.path.isdir(path):
            _open_folder_in_file_manager(path)

    def _update_frame_count(self):
        """Refresh the blue "N frames found (WxH)" label under Step 1 for the
        current input folder, and grey out frame-limit dropdown choices that
        exceed the available count. Counts files by image extension and reads
        the first frame's dimensions for the size hint. Handles empty / missing
        / no-image folders with their own messages."""
        folder = self._folder_input.text().strip()
        from modules.frame_list import IMAGE_EXTS as exts
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
                dim_str = ""
                try:
                    first = next(
                        os.path.join(folder, n) for n in sorted(os.listdir(folder))
                        if os.path.splitext(n)[1].lower() in exts
                    )
                    from modules.io_safe import image_size as _image_size
                    _sz = _image_size(first)
                    if _sz is not None:
                        _w, _h = _sz
                        dim_str = f"  ({_w:,}px x {_h:,}px)"
                except Exception:
                    pass
                self._frame_count_label.setText(
                    f"<b>{count:,}</b> frame{'s' if count != 1 else ''} found{dim_str}")

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
        """Step 2 Browse button. Open a folder picker (starting at the last
        used output dir), set and remember the chosen path."""
        last_dir = SETTINGS.value("last_output_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", last_dir)
        if folder:
            self._output_input.setText(folder)
            SETTINGS.setValue("last_output_dir", folder)

    def _auto_output(self, text):
        """Auto-fill the output field as the user sets the input folder:
        default to a "cleaned" subfolder inside the input. `text` is the new
        input path. Normalizes to forward slashes so Windows doesn't show a
        mixed-slash path."""
        if text and text.strip():
            # Normalize to forward slashes for display: Qt's QFileDialog
            # returns forward slashes on every platform, but os.path.join
            # uses os.sep, so on Windows the joined string ends up mixing
            # forward and back slashes ("L:/foo/bar\cleaned"). Replace makes
            # the displayed path consistent.
            joined = os.path.join(text.strip(), "cleaned").replace("\\", "/")
            self._output_input.setText(joined)
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
        """Step 3 Create/Edit Mask button. Open the separate MaskEditorWindow
        on the first frame of the input folder so the user can paint over the
        foreground (ground, buildings) the AI should ignore. Loads any existing
        saved mask to edit. Shows an inline error if no input folder/images are
        set yet."""
        folder = self._folder_input.text().strip()
        if not folder or not os.path.isdir(folder):
            self._error_label.setText("Select an input folder first (Step 1).")
            return
        self._error_label.setText("")

        # Find first image
        from modules.frame_list import glob_patterns
        frames = sorted(set(
            f for e in glob_patterns() for f in glob.glob(os.path.join(folder, e))
        ))
        if not frames:
            self._error_label.setText("No image files found in the selected folder.")
            return

        # Create mask window if needed, or reuse
        if self._mask_window is None:
            self._mask_window = MaskEditorWindow(self)
            self._mask_window.mask_saved.connect(self._on_mask_saved)

        self._mask_window.load_frames(frames, 0)

        # Load existing mask if available
        migrate_workspace(folder)
        mask_path = os.path.join(folder, WORKSPACE_DIR, "foreground_mask.png")
        if os.path.exists(mask_path):
            self._mask_window.load_existing_mask(mask_path)

        self._mask_window._painter._set_mode(False)
        self._mask_window.show()
        self._mask_window.raise_()
        self._mask_window.activateWindow()

    def _on_mask_saved(self, mask_np):
        """Receive the painted mask from the mask editor. `mask_np` is the
        mask image array. If it has any painted pixels, save it as the
        foreground mask PNG in the workspace; if it's blank, delete any
        existing mask. Then refresh the Step 3 status label."""
        folder = self._folder_input.text().strip()
        if folder:
            if mask_np.any():
                mask_path = workspace_path(folder, "foreground_mask.png")
                from modules.io_safe import robust_imwrite
                robust_imwrite(mask_path, mask_np)
                self._mask_path = mask_path
            else:
                mask_path = os.path.join(folder, WORKSPACE_DIR, "foreground_mask.png")
                if os.path.exists(mask_path):
                    os.remove(mask_path)
                self._mask_path = None
            self._update_mask_status()

    # ── Run ──────────────────────────────────────────────────────────────────

    def _validate(self):
        """Run every pre-flight check before a job starts and gather a few
        run parameters as side effects. Returns (input_folder, output_folder)
        if the run may proceed, or None if any check failed or the user
        cancelled at a prompt.

        Checks, in order: input/output folders are set and the input exists;
        the output folder is writable (probe write); enough disk space and at
        least 3 frames; a memory-aware batch cap (stored on self._max_batch /
        self._mem_note); a saved mask is still readable; and, if any frame has
        both a RAW and a JPG/TIFF, a one-time prompt for which to use (stored on
        self._twin_prefer). The resource checks are best-effort and never block
        a run on their own failure."""
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

        # Pre-flight write check: try to create the output folder + write a small
        # probe file. Catches read-only drives, OneDrive sync conflicts, locked-down
        # folders, and antivirus blocks BEFORE the worker starts so the user sees
        # a clear message instead of a mid-run crash.
        try:
            os.makedirs(output, exist_ok=True)
            _probe_path = os.path.join(output, ".star_trail_cleanr_probe.tmp")
            with open(_probe_path, "w") as _f:
                _f.write("probe")
            os.remove(_probe_path)
        except (PermissionError, OSError) as _err:
            import errno as _errno
            from PySide6.QtWidgets import QMessageBox as _QMB
            if getattr(_err, "errno", None) == _errno.ENOSPC:
                _QMB.warning(
                    self,
                    "Drive is full",
                    f"The output drive is full.\n\n{output}\n\n"
                    "Free up space on that drive, or pick a different output folder."
                )
            else:
                _QMB.warning(
                    self,
                    "Cannot write to output folder",
                    f"Star Trail CleanR cannot write to:\n\n{output}\n\n"
                    "Pick a different folder, or check that it isn't on a read-only "
                    "drive, a OneDrive synced location, or a folder where files are "
                    "open in another app.\n\n"
                    f"(Detail: {type(_err).__name__}: {_err})"
                )
            return None

        # Disk space + memory check: estimate resources needed before the run starts.
        try:
            import shutil as _shutil
            from modules.frame_list import glob_patterns as _glob_patterns
            _frames = sorted(set(
                f for e in _glob_patterns() for f in glob.glob(os.path.join(folder, e))
            ))
            if _frames:
                _lim_text = self._frame_limit.currentText().strip()
                try:
                    _total_frames = (len(_frames) if _lim_text in ("All Frames", "")
                                     else min(int(_lim_text), len(_frames)))
                except ValueError:
                    _total_frames = len(_frames)

                if _total_frames < 3:
                    from PySide6.QtWidgets import QMessageBox as _QMB
                    _QMB.warning(
                        self,
                        "Not enough frames",
                        f"Star Trail CleanR needs at least 3 frames to run.\n\n"
                        f"Your folder has {_total_frames} image(s). "
                        "Add more frames and try again.\n\n"
                        "Star Trail CleanR works on individual frames before "
                        "stacking, not on a finished star trail image.",
                        _QMB.Ok,
                    )
                    return None

                from modules.io_safe import image_size as _image_size
                _sz0 = _image_size(_frames[0])
                if _sz0 is None:
                    raise ValueError("could not read first frame's size")
                _w, _h = _sz0

                _out_fmt = self._format_combo.currentText()
                if _out_fmt == "TIFF 16-bit":
                    _bpp = 6.0
                elif _out_fmt == "TIFF 8-bit":
                    _bpp = 3.0
                else:
                    _bpp = 0.6  # conservative for JPG quality 95
                _estimated_bytes = int(_w * _h * _bpp * _total_frames)

                def _fmt_gb(b):
                    """Format a byte count as a one-decimal "N.N GB" string."""
                    return f"{b / 1_073_741_824:.1f} GB"

                _free_bytes = _shutil.disk_usage(output).free
                if _free_bytes < _estimated_bytes:
                    from PySide6.QtWidgets import QMessageBox as _QMB
                    _dlg = _QMB(self)
                    _dlg.setIcon(_QMB.Warning)
                    _dlg.setWindowTitle("Low disk space")
                    _dlg.setText(
                        f"The output drive may not have enough space for this run.\n\n"
                        f"Estimated space needed:  {_fmt_gb(_estimated_bytes)}\n"
                        f"Free space available:      {_fmt_gb(_free_bytes)}\n\n"
                        "You can continue anyway or cancel and pick a different output folder."
                    )
                    _cont_btn = _dlg.addButton("Continue", _QMB.AcceptRole)
                    _dlg.addButton("Cancel", _QMB.RejectRole)
                    _dlg.setDefaultButton(_cont_btn)
                    _dlg.exec()
                    if _dlg.clickedButton().text() == "Cancel":
                        return None

                # Memory-aware batch sizing. Loading frames plus the AI model
                # can peak well past a modest Mac's RAM on big images, so the OS
                # kills the worker (SIGBUS / SIGKILL). Predict the peak from the
                # frames' real size and bit depth, then pick the largest batch
                # that fits -- from 20 down to a floor of 5 (Star Bridge repair
                # needs neighbor frames; fewer than ~5 hurts quality, and the
                # worker hard-refuses below 3). Measured model on 6000x4000:
                #     peak GB = 4.4 (model+torch) + 5 x (one decoded frame) x frames
                # so per frame ~0.36 GB at 8-bit, ~0.72 GB at 16-bit.
                self._max_batch = 20
                self._mem_note = ""
                try:
                    import psutil as _psutil
                    # Decoded in-memory bytes for ONE frame = w x h x channels x
                    # bytes-per-sample. The file size on disk is NOT this (a
                    # 12 MB JPG decodes to 72 MB). Read the real bit depth from
                    # the header. PIL mislabels 16-bit RGB TIFFs, so use
                    # tifffile for TIFFs; anything unreadable falls back to the
                    # heavier 16-bit guess so we never over-size the batch.
                    _suf = os.path.splitext(_frames[0])[1].lower()
                    if _suf in (".tif", ".tiff"):
                        try:
                            import tifffile as _tf
                            _pg = _tf.TiffFile(_frames[0]).pages[0]
                            _bytes_per = max(1, _pg.dtype.itemsize)
                            _spp = getattr(_pg, "samplesperpixel", 3) or 3
                            _channels = 3 if _spp >= 3 else 1
                        except Exception:
                            _bytes_per, _channels = 2, 3  # conservative (16-bit RGB)
                    else:
                        _bytes_per, _channels = 1, 3  # jpg / png are 8-bit
                    _decoded_frame = _w * _h * _channels * _bytes_per
                    _BASE = int(4.4 * 1_073_741_824)   # model + torch floor
                    _PER_FRAME = 5 * _decoded_frame    # measured peak per frame
                    _SAFETY = 0.8                      # leave 20% for OS + our window
                    _available = _psutil.virtual_memory().available
                    _usable = _available * _SAFETY
                    _fit = int((_usable - _BASE) / _PER_FRAME) if _PER_FRAME else 20
                    _bit = _bytes_per * 8
                    if _fit < 5:
                        # Even the smallest safe batch won't fit. Don't shrink
                        # below 5 -- tell the user honestly and let them decide.
                        from PySide6.QtWidgets import QMessageBox as _QMB
                        _need = _fmt_gb(_BASE + _PER_FRAME * 5)
                        resp = _QMB.warning(
                            self,
                            "Not enough memory",
                            f"These photos are large for this computer's available "
                            f"memory, so the run may fail partway through.\n\n"
                            f"Memory needed:           {_need}\n"
                            f"Memory available now:  {_fmt_gb(_available)}\n\n"
                            "Close any other open programs to free up memory, then "
                            "try again.\n\n"
                            "You can continue anyway or cancel.",
                            _QMB.Ok | _QMB.Cancel,
                            _QMB.Cancel,
                        )
                        if resp == _QMB.Cancel:
                            return None
                        self._max_batch = 5
                    else:
                        self._max_batch = min(20, _fit)
                    self._mem_note = (
                        f"mem_avail={_available // (1024 * 1024)}MB "
                        f"bit={_bit} frame={_decoded_frame // (1024 * 1024)}MB "
                        f"peak={_fmt_gb(_BASE + _PER_FRAME * self._max_batch)} "
                        f"max_batch={self._max_batch}"
                    )
                except ImportError:
                    pass
        except Exception:
            pass  # best-effort, never block a run on a failed resource check

        # Pre-flight mask check: if a mask was saved but can't be read now,
        # ask the user whether to proceed without it rather than crashing mid-run.
        if self._mask_path:
            try:
                import cv2 as _cv2
                _test = _cv2.imread(self._mask_path, _cv2.IMREAD_GRAYSCALE)
            except Exception:
                _test = None
            if _test is None:
                from PySide6.QtWidgets import QMessageBox as _QMB
                resp = _QMB.warning(
                    self,
                    "Saved mask can't be opened",
                    "The saved foreground mask couldn't be read:\n\n"
                    f"{self._mask_path}\n\n"
                    "The file may be corrupted or missing. "
                    "Proceed without the mask (trails in the foreground area may be flagged), "
                    "or cancel and re-draw it.",
                    _QMB.Ok | _QMB.Cancel,
                    _QMB.Cancel,
                )
                if resp == _QMB.Cancel:
                    return None
                self._mask_path = None

        # RAW vs JPG/TIFF twin choice. If any frame exists as BOTH a RAW and a
        # JPG/TIFF, ask once which to process (default RAW). When there are no
        # such pairs, no prompt appears and the default is harmless. The choice
        # is stored on self and handed to the worker so both sides dedup alike.
        self._twin_prefer = "raw"
        try:
            from modules.frame_list import gather_frames, count_raw_twins
            _all_files = gather_frames(folder)
            _twins = count_raw_twins(_all_files)
        except Exception:
            _twins = 0
        if _twins > 0:
            from PySide6.QtWidgets import QMessageBox as _QMB
            _dlg = _QMB(self)
            _dlg.setIcon(_QMB.Question)
            _dlg.setWindowTitle("RAW and JPEG/TIFF found")
            _dlg.setText(
                f"{_twins} of your frames have BOTH a RAW file and a "
                f"JPEG/TIFF version.\n\n"
                "Which should Star Trail CleanR process?"
            )
            _raw_btn = _dlg.addButton("Use RAW files", _QMB.AcceptRole)
            _other_btn = _dlg.addButton("Use JPEG/TIFF", _QMB.RejectRole)
            _dlg.setDefaultButton(_raw_btn)
            _dlg.exec()
            self._twin_prefer = "nonraw" if _dlg.clickedButton() is _other_btn else "raw"

        self._error_label.setText("")
        return folder, output

    def _run(self):
        """"Clean My Stars!" button handler — start a cleaning run.

        Validates via _validate (bails on failure), resets all the
        processing-page widgets, switches to that page, starts the spinner and
        AI-warmup heartbeat timers, then constructs the CleanerWorker with the
        chosen settings, wires every worker signal to its handler, and starts
        the thread. Also seeds the run-summary fields that the stats/timing
        handlers fill in for the end-of-run log."""
        result = self._validate()
        if not result:
            return
        folder, output = result
        SETTINGS.setValue("last_input_dir", folder)

        # Run summary fields — captured by the stats/timing handlers, then
        # written to disk by _write_run_summary() at the end of _on_done.
        import datetime as _dt
        self._run_start_time = _dt.datetime.now()
        self._run_total_trails = 0
        self._run_total_frames = 0
        self._run_initial_est_sec = 0
        self._run_actual_sec = 0
        self._run_cancelled = False        # set True by _cancel_run
        self._stats_last_trail_count = 0   # live trail count for cancelled-run logs

        # Go to process page — reset all widgets
        self._process_title.setText("Cleaning in Progress")
        _p = Path(folder)
        _display = f"{_p.parent.name}/{_p.name}" if _p.parent.name else _p.name
        self._run_source_label.setText(_display)
        self._trail_counter_label.setText("")
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {CARD_BORDER}; border-radius: 10px; "
            f"background: {CARD_BG}; text-align: center; font-weight: bold; font-size: 14px; color: {CARD_TEXT}; }}"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #4a9eff, stop:1 #66b3ff); border-radius: 9px; }"
        )
        self._frame_counter.setText("Scrubbing the stars\u2026")
        self._initial_est_label.setText("Estimated Time: estimating\u2026")
        self._time_label.setText("")
        self._elapsed_label.setText("")
        self._run_stats_label.setText("")
        self._stats_frames_done = 0
        self._batch_label.setText("")
        self._step1_bar.setValue(0)
        self._step1_bar.setFormat("0%")
        self._step1_label.setText("Detecting\nwaiting")
        self._step2_bar.setValue(0)
        self._step2_bar.setFormat("0%")
        self._step2_label.setText("Repairing\nwaiting")
        # Stats are now shown in a modal dialog (see _show_run_complete_dialog),
        # not in an inline label. No widget to reset here.
        self._status_out.setText("")
        self._view_log_link.setVisible(False)  # reappears when the run ends
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
        self._spinner_timer = QTimer(self)
        self._spinner_timer.timeout.connect(self._update_spinner)
        self._spinner_timer.start(250)

        # Warmup heartbeat: rotates astro-flavored phrases through the Star
        # Log detail label during the silent AI-load gap (15-30s on first
        # batch) so the run never looks frozen. Stops when the first real
        # per-frame detect line arrives.
        self._warmup_phrases = [
            "Studying your stars",
            "Scanning the sky",
            "Hunting for trails",
            "Tracing starlight",
            "Sweeping the sky",
            "Spotting trails",
            "Squinting at the sky",
            "Eyeing your stars",
            "Watching for streaks",
            "Reading the sky",
            "Peering into your night",
            "Combing the night sky",
        ]
        self._warmup_counter = 0
        self._warmup_timer = QTimer(self)
        self._warmup_timer.timeout.connect(self._warmup_tick)

        # Set output folder now so the "Open Cleaned Folder" button works during the run
        self._done_output_folder = output

        fmt_map = {"JPG": "jpg", "TIFF 8-bit": "tif8", "TIFF 16-bit": "tif16"}
        out_fmt = fmt_map.get(self._format_combo.currentText(), "jpg")
        self._run_cancelled = False
        frame_start = self._dev_start_frame.value() if self._dev_start_frame is not None else 0
        frame_end = self._dev_end_frame.value() if self._dev_end_frame is not None else 0
        self.worker = CleanerWorker(
            folder, output, self._frame_limit.currentText(), self._mask_path,
            output_format=out_fmt,
            jpeg_quality=self._jpeg_quality.value(),
            frame_start=frame_start,
            frame_end=frame_end,
            max_batch=getattr(self, "_max_batch", 20),
            mem_note=getattr(self, "_mem_note", ""),
            twin_prefer=getattr(self, "_twin_prefer", "raw"))
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self._on_status)
        self.worker.batch_info.connect(self._on_batch_info)
        self.worker.step_progress.connect(self._on_step_progress)
        self.worker.warmup_active.connect(self._on_warmup_active)
        self.worker.frame_count.connect(self._on_frame_count)
        self.worker.stats_ready.connect(self._on_stats_ready)
        self.worker.timing_stats.connect(self._on_timing_stats)
        self.worker.initial_estimate.connect(self._on_initial_estimate)
        self.worker.error.connect(self._on_error)
        self.worker.done.connect(self._on_done)
        self.worker.finished.connect(self._on_finished)
        self.worker.bad_file_prompt.connect(self._on_bad_file_prompt)
        self.worker.too_many_bad_files.connect(self._on_too_many_bad_files)
        self.worker.frames_filter_prompt.connect(self._on_frames_filter_prompt)
        self.worker.trail_count_update.connect(self._on_trail_count_update)
        self._trail_counter_label.setText("0 Trails Cleaned")
        self._trail_counter_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #5b9bd5;"
        )
        self.worker.start()
        self._set_updates_run_state(True)

    def _cancel_run(self):
        """Cancel Cleaning button. Tell the worker to stop (which kills its
        current subprocess), freeze the progress bar at "Cancelled", stop the
        timers, wait briefly for the thread to exit (force-terminate if it
        won't), then turn the button into "Back to Setup"."""
        if self.worker and self.worker.isRunning():
            self._run_cancelled = True
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
        """Stop the spinner/elapsed and AI-warmup timers and drop the spinner
        glyph from the "Star Log" heading. Called whenever a run ends (finish,
        cancel, or error)."""
        if hasattr(self, '_spinner_timer') and self._spinner_timer.isActive():
            self._spinner_timer.stop()
        if hasattr(self, '_warmup_timer') and self._warmup_timer.isActive():
            self._warmup_timer.stop()
        if hasattr(self, '_star_log_title'):
            self._star_log_title.setText("Star Log")

    def _go_to_setup(self):
        """Stop the timers and switch the Main tab back to the Setup page
        (page 0). Wired to the "Back to Setup" button after a run ends."""
        self._stop_elapsed_timer()
        self._stack.setCurrentIndex(0)

    def _update_spinner(self):
        """Fired every 250 ms during a run: advance the |/-\\ spinner glyph,
        update the "Elapsed: ..." label and the spinner in the Star Log title,
        and pulse "Estimating..." until the first real time estimate arrives."""
        self._spinner_idx += 1
        ch = self._spinner_chars[self._spinner_idx % len(self._spinner_chars)]
        start = getattr(self, '_run_start_time', None)
        elapsed = (time.time() - start.timestamp()) if start is not None else 0
        self._elapsed_label.setText(f"{ch}  Elapsed: {fmt_estimate(elapsed)}")
        if hasattr(self, '_star_log_title'):
            self._star_log_title.setText(f"Star Log  {ch}")


        # Pulse "Estimating..." before real estimate arrives
        if not getattr(self, '_has_estimate', False):
            dots = "." * ((self._spinner_idx % 3) + 1)
            self._time_label.setText(f"Estimating{dots}")

    def _on_progress(self, pct, total, remaining_str):
        """Worker progress signal handler. Set the fat overall progress bar to
        `pct` (0-100) and, when present, show "~<remaining_str> remaining".
        Ignored after the user cancelled."""
        if getattr(self, "_run_cancelled", False):
            return
        self._progress_bar.setRange(0, 100)
        pct = max(0, min(100, pct))
        self._progress_bar.setValue(pct)

        if remaining_str:
            self._time_label.setText(f"\u23f1 ~{remaining_str} remaining")
            self._has_estimate = True

    def _on_batch_info(self, batch_num, n_batches):
        """Worker signal at the start of each batch. Update the "Batch X of Y"
        label and reset both per-step (Detecting / Repairing) bars back to
        0% / "waiting" / blue for the new batch."""
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

    def _on_step_progress(self, step, current, total, global_current, global_total):
        """Worker signal driving the two per-step bars. `step` is 1 (Detecting)
        or 2 (Repairing); `current`/`total` are this batch's frame progress;
        `global_current`/`global_total` place it in the whole job. Sets the
        bar percent and a "Detecting/Repairing frames 21-40 (of 450)" style
        label, turning the bar green at 100%. Ignored after cancellation."""
        if getattr(self, "_run_cancelled", False):
            return
        green_style = (
            f"QProgressBar {{ border: 1px solid {CARD_BORDER}; border-radius: 8px; "
            f"background: {CARD_BG}; text-align: center; font-weight: bold; font-size: 13px; color: {CARD_TEXT}; }}"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #34c759, stop:1 #5dd87a); border-radius: 7px; }"
        )
        pct = int(current / total * 100) if total > 0 else 0
        # Name the frames this batch is actually working, as a global range with
        # whole-job context, e.g. "frames 21-40 (of 450)". The bar % is progress
        # through just those frames, so the label and the % refer to the same set.
        range_start = global_current - current + 1
        range_end = range_start + total - 1
        # Action word + "frame(s)" on line 1, just the numbers on line 2. Keeps the
        # number line short ("21-40 (of 450)") so it fits the fixed label column at
        # any frame count; the bar never moves. Word-wrap on the label is the safety.
        if range_start == range_end:
            head, nums = "frame", f"{range_start} (of {global_total})"
        else:
            head, nums = "frames", f"{range_start}-{range_end} (of {global_total})"
        if step == 1:
            self._step1_bar.setValue(pct)
            if pct >= 100:
                self._step1_bar.setFormat("100%")
                self._step1_label.setText("Detecting\ncomplete")
                self._step1_bar.setStyleSheet(green_style)
            else:
                self._step1_bar.setFormat(f"{pct}%")
                self._step1_label.setText(f"Detecting {head}\n{nums}")
        elif step == 2:
            self._step2_bar.setValue(pct)
            if pct >= 100:
                self._step2_bar.setFormat("100%")
                self._step2_label.setText("Repairing\ncomplete")
                self._step2_bar.setStyleSheet(green_style)
            else:
                self._step2_bar.setFormat(f"{pct}%")
                self._step2_label.setText(f"Repairing {head}\n{nums}")

    def _on_warmup_active(self, active):
        """Worker signal toggling the AI-warmup heartbeat. `active` True starts
        the rotating "Studying your stars..." log messages that fill the silent
        15-30 s while the model loads; False stops them once real per-frame
        progress begins. Idempotent so a repeat True won't restart the rotation."""
        if active:
            # Idempotent: if the heartbeat is already running (because we triggered
            # it at "frames loaded"), don't reset the rotation when "Step 1" arrives.
            if not self._warmup_timer.isActive():
                last_phrase = self._warmup_counter // 4
                next_phrase = (last_phrase + 1) % len(self._warmup_phrases)
                self._warmup_counter = next_phrase * 4 if self._warmup_counter > 0 else 0
                self._warmup_tick()
                self._warmup_timer.start(500)
        else:
            self._warmup_timer.stop()

    def _warmup_tick(self):
        """Fired every 500 ms while the warmup heartbeat is active. Appends one
        rotating astro-flavored phrase to the Star Log every 2 seconds (4 ticks)
        so a long model-load never looks frozen; after the first cycle it
        settles on a steady "Warming up the AI trail detector" line."""
        # Append once per phrase, no in-place dot animation. The previous
        # "append on tick 0, replace last block on ticks 1-3" approach
        # caused doubling when worker messages interleaved with warmup
        # ticks (the QTextEdit replace-last-block cursor logic targeted
        # the wrong block in some flows). One line per phrase, held for
        # 2 seconds, gives the same "still working" signal without the
        # bug surface.
        phrase_step = self._warmup_counter // 4
        sub_step = self._warmup_counter % 4
        if phrase_step < len(self._warmup_phrases):
            text = self._warmup_phrases[phrase_step]
        else:
            # Past the first cycle (AI taking longer than 24 seconds to
            # warm up): show a steady "still warming up" line so the
            # rotation doesn't loop back to "Studying your stars" looking
            # like the run restarted.
            text = "Warming up the AI trail detector"
        if sub_step == 0:
            self._status_out.append("  " + text + "...")
            sb = self._status_out.verticalScrollBar()
            sb.setValue(sb.maximum())
        self._warmup_counter += 1

    def _on_frame_count(self, current, total):
        """Worker signal updating the "Scrubbing the stars... N of M" line and
        the frames-done counter used by the running per-frame averages."""
        self._frame_counter.setText(f"Scrubbing the stars\u2026 {current} of {total}")
        self._stats_frames_done = current

    def _on_initial_estimate(self, seconds):
        """Worker signal carrying the first measured total-time estimate.
        Locks it into the "Estimated Time: ..." label (formatted compactly).
        Emitted once, only when the estimate is based on real measured speed."""
        secs = int(round(seconds))
        if secs < 60:
            self._initial_est_label.setText(f"Estimated Time: {secs}s")
        else:
            # Hour-aware (1h 7m 50s, or 7m 50s under an hour) -- matches the
            # live "remaining" line instead of running minutes past 60.
            self._initial_est_label.setText(f"Estimated Time: {fmt_estimate(secs)}")

    def _on_format_changed(self, text):
        """Step 5 output-format change handler. Enable the JPEG-quality spinbox
        only when the format is JPG (it's meaningless for TIFF output)."""
        is_jpg = text == "JPG"
        self._jpeg_quality.setEnabled(is_jpg)
        self._jpeg_quality_label.setEnabled(is_jpg)

    def _on_trail_count_update(self, count):
        """Worker signal with the running total of trails cleaned so far.
        Update the big "N Trails Cleaned" counter and the secondary
        trails/frame and seconds/frame averages."""
        self._stats_last_trail_count = count  # kept for cancelled-run summaries
        self._trail_counter_label.setText(f"{count:,} Trails Cleaned")
        # Running averages over the whole run so far. frame_count fires just
        # before this on each frame, so _stats_frames_done is current.
        frames = self._stats_frames_done
        if frames > 0:
            start = getattr(self, '_run_start_time', None)
            elapsed = (time.time() - start.timestamp()) if start is not None else 0
            tpf = count / frames
            spf = elapsed / frames
            self._run_stats_label.setText(
                f"{tpf:.0f} trails/frame · {spf:.1f}s/frame"
            )

    def _on_stats_ready(self, total_trails, total_frames):
        """Worker signal at run end with the final totals. Stores them for the
        log and builds the HTML summary line (self._stats_trail_line) shown in
        the run-complete dialog: either a "sky was clean" message or a
        "swept N trails ... TIME SAVED" message estimating time saved at 30 s
        per manually-cleaned trail."""
        # Capture for the run-summary file and the run-complete dialog.
        self._run_total_trails = total_trails
        self._run_total_frames = total_frames
        color = SUCCESS_TEXT if total_trails > 0 else MUTED_TEXT
        self._trail_counter_label.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {color};"
        )
        if total_trails <= 0:
            self._stats_trail_line = (
                f"Sky was clean — no airplane or satellite trails found<br>"
                f"in your <b>{total_frames:,}</b> frames.<br><br>"
                f"<b>Time to stack!</b><br>"
                f"Open the Cleaned Folder, then load the frames into your favorite "
                f"stacker (StarStaX, Sequator, Photoshop, etc.) for the final composite."
            )
            return
        SECONDS_PER_MANUAL_TRAIL = 30
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
            f"Swept <b>{total_trails:,}</b> airplane and satellite trails from your stars<br>"
            f"across <b>{total_frames:,}</b> twinkling frames.<br>"
            f"<i>Based on manual cleanup at 30 seconds per trail.</i><br><br>"
            f"<span style='font-size:20px; font-weight:bold;'>TIME SAVED: {time_saved}</span>"
            f"<br><br><b>Time to stack!</b><br>"
            f"Open the Cleaned Folder, then load the frames into your favorite "
            f"stacker (StarStaX, Sequator, Photoshop, etc.) for the final composite."
        )
        # Stats HTML stored on self; the modal dialog renders it on _on_done.

    def _on_timing_stats(self, initial_est_sec, actual_sec):
        """Worker signal at run end comparing the original estimate to the
        actual time. Builds the small "Thought it'd take X. Took Y." HTML line
        (self._stats_timing_line) appended in the run-complete dialog, with a
        cheeky "You're welcome." / "My apologies." depending on which was
        faster."""
        self._run_initial_est_sec = initial_est_sec
        self._run_actual_sec = actual_sec
        tail = "You're welcome." if actual_sec <= initial_est_sec else "My apologies."
        frames = getattr(self, '_run_total_frames', 0)
        pf = f"  ({actual_sec / frames:.1f}s/frame)" if frames > 0 else ""
        self._stats_timing_line = (
            f"<br><br><span style='font-size:14px; color:{MUTED_TEXT};'>"
            f"Thought it'd take <b>{fmt_hms(initial_est_sec)}</b>. "
            f"Took <b>{fmt_hms(actual_sec)}</b>{pf}. {tail}"
            f"</span>"
        )
        # Modal dialog will read this on _on_done.

    def _on_status(self, text):
        """Worker status signal: append `text` to the scrolling Star Log and
        keep the view scrolled to the bottom."""
        self._status_out.append(text)
        sb = self._status_out.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _switch_to_back_btn(self):
        """Repurpose the Cancel button into "Back to Setup" once a run has
        ended (rewires its click to _go_to_setup)."""
        self._cancel_btn.setText("Back to Setup")
        self._cancel_btn.setEnabled(True)
        try:
            self._cancel_btn.clicked.disconnect()
        except RuntimeError:
            pass
        self._cancel_btn.clicked.connect(self._go_to_setup)

    def _on_error(self, msg):
        """Worker error signal. Stop the timers, append the error plus an
        app-version/OS line to the Star Log, and flip the button to "Back to
        Setup". The run summary is still written by _on_finished afterward."""
        import platform as _plat
        self._stop_elapsed_timer()
        self._status_out.append(f"\nERROR: {msg}")
        self._status_out.append(
            f"App: Beta v{VERSION}  |  {_plat.system()} {_plat.machine()}"
        )
        self._switch_to_back_btn()

    def _on_bad_file_prompt(self, path, diagnosis):
        """Worker hit a file that no reader could decode. Block the run with
        a modal so the user knows about it and can choose to skip the frame
        or stop the run. The worker is paused waiting for our answer."""
        from pathlib import Path as _Path
        QApplication.alert(self)
        name = _Path(path).name if path else "(unknown)"

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Couldn't read a file")
        msg.setTextFormat(Qt.RichText)
        msg.setText(
            "<b>Could not read this file</b>"
            f"<p style='font-family: monospace;'>{name}</p>"
        )
        diagnostic_sent = bool(
            SETTINGS.value("crash_reporting_enabled", False, type=bool)
            and _SENTRY_DSN
        )
        diagnostic_line = (
            " Diagnostic data has already been sent automatically."
            if diagnostic_sent
            else ""
        )
        from modules.frame_list import RAW_EXTS as _RAW_EXTS
        _is_raw = _Path(path).suffix.lower() in _RAW_EXTS if path else False
        if _is_raw:
            # RAW-specific: the file is a camera RAW we couldn't debayer (an
            # unusual/newer RAW variant, or a corrupted file). Point the user at
            # the reliable fix — export that sequence to TIFF or JPEG first.
            _problem = (
                "<p>This is a camera RAW file, and Star Trail CleanR couldn't "
                "decode it. That usually means it's a newer or unusual RAW "
                "variant, or the file is damaged.</p>"
            )
            _workaround = (
                "<p><b>The fix:</b> export your sequence from Lightroom, Camera "
                "Raw, DPP, or your editor to <b>16-bit TIFF</b> (keeps the full "
                "quality) or <b>JPEG</b>, then run Star Trail CleanR on that "
                "folder.</p>"
            )
        else:
            _problem = (
                "<p>We tried our image readers three times. None could read "
                "the file. It may be damaged, or it may have a format our "
                "readers can't handle.</p>"
            )
            _workaround = (
                "<p><b>As a workaround:</b> export your entire sequence as "
                "JPEGs and run Star Trail CleanR on the JPEG folder.</p>"
            )
        msg.setInformativeText(
            _problem +
            "<p>If you continue, Star Trail CleanR will skip this frame. "
            "There may be a small gap in your final star trail where this "
            "image would have been.</p>"
            + _workaround +
            "<p>If you'd like to help us improve, please email this file to "
            "<a href='mailto:bruceherwig+startrailcleanr@gmail.com"
            "?subject=Star%20Trail%20CleanR%20unreadable%20file'>"
            f"bruceherwig+startrailcleanr@gmail.com</a>.{diagnostic_line}</p>"
        )
        msg.setDetailedText(
            f"Path:\n  {path}\n\nReader output:\n{diagnosis or '(none)'}"
        )
        skip_btn = msg.addButton(
            "Skip this frame and continue", QMessageBox.AcceptRole
        )
        msg.addButton("Stop Run", QMessageBox.RejectRole)
        msg.setDefaultButton(skip_btn)
        msg.exec()

        decision = "CONTINUE" if msg.clickedButton() is skip_btn else "STOP"
        if hasattr(self, "worker") and self.worker is not None:
            if decision == "STOP":
                self.worker.request_graceful_stop()
            self.worker.set_bad_file_decision(decision)

    def _on_frames_filter_prompt(self, info):
        """Pre-flight scan found frames that won't process (mismatched
        resolution or unreadable header). Show the user what's about to be
        skipped and let them choose Continue or Cancel before the run
        actually starts."""
        QApplication.alert(self)
        self._run_filter_info = dict(info)  # remember for run summary

        n_mismatched = info.get("mismatched", 0)
        n_unreadable = info.get("unreadable", 0)
        total_found = info.get("total_found", 0)
        matching = info.get("matching", 0)
        dominant_size = info.get("dominant_size", "")
        breakdown = info.get("breakdown", [])
        mismatched_sample = info.get("mismatched_sample", [])
        unreadable_sample = info.get("unreadable_sample", [])

        # Build the body in plain English.
        parts = [
            "<b>Some frames in this folder will be skipped.</b>",
            f"<p>Star Trail CleanR found <b>{total_found}</b> image files "
            f"in this folder. Of those, <b>{matching}</b> will be processed "
            f"and <b>{n_mismatched + n_unreadable}</b> will be skipped.</p>",
        ]
        if breakdown:
            parts.append("<p><b>Resolutions in this folder:</b></p><ul>")
            for entry in breakdown:
                marker = "  (will process)" if entry["is_dominant"] else "  (will skip)"
                parts.append(
                    f"<li>{entry['size']} &mdash; {entry['count']} frame(s){marker}</li>"
                )
            parts.append("</ul>")
        if n_unreadable > 0:
            parts.append(
                f"<p><b>{n_unreadable}</b> file(s) couldn't be opened to "
                "check their resolution; they'll be skipped too.</p>"
            )
            if unreadable_sample:
                parts.append("<p>Examples: " +
                             ", ".join(f"<code>{n}</code>" for n in unreadable_sample) +
                             "</p>")
        if mismatched_sample:
            parts.append("<p>Examples of skipped frames: " +
                         ", ".join(f"<code>{n}</code>" for n in mismatched_sample) +
                         "</p>")
        parts.append(
            "<p>Common causes: some frames are portrait-orientation "
            "(rotated 90&deg;), some came from a different camera, or "
            "the RAW converter produced different sizes. Cancel if you "
            "want to check the folder first; Continue to process the "
            f"{matching} matching frame(s) only.</p>"
        )

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Some frames will be skipped")
        msg.setTextFormat(Qt.RichText)
        msg.setText("".join(parts))
        cont_btn = msg.addButton(
            f"Continue with {matching} frames", QMessageBox.AcceptRole
        )
        msg.addButton("Cancel run", QMessageBox.RejectRole)
        msg.setDefaultButton(cont_btn)
        msg.exec()

        decision = "CONTINUE" if msg.clickedButton() is cont_btn else "CANCEL"
        if hasattr(self, "worker") and self.worker is not None:
            self.worker.set_frames_filter_decision(decision)

    def _on_too_many_bad_files(self, count):
        """Run-wide cap hit. The worker has already been told to stop; this
        modal is purely informational so the user understands why."""
        QApplication.alert(self)
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Multiple unreadable files")
        msg.setTextFormat(Qt.RichText)
        msg.setText(
            "<b>Star Trail CleanR is stopping this run.</b>"
        )
        msg.setInformativeText(
            f"<p>{count} files in your input folder couldn't be read. "
            "Something may be wrong with the source.</p>"
            "<p><b>As a workaround:</b> export your entire sequence as "
            "JPEGs and run Star Trail CleanR on the JPEG folder.</p>"
            "<p>If your input folder is on a USB or network drive, you "
            "can also try copying it to your main hard drive and running "
            "again &mdash; external drives sometimes drop reads.</p>"
            "<p>The frames cleaned so far are preserved in the output "
            "folder.</p>"
        )
        msg.exec()

    def _on_done(self, output_folder):
        """Worker done signal — a run finished successfully. `output_folder` is
        the cleaned-frames folder. Marks the UI complete (green 100% bar,
        "Cleaning Complete"), enables the Open-Folder buttons, flips to "Back
        to Setup", and pops the run-complete summary dialog. The run-summary
        text file is written later in _on_finished (which also covers
        stop/error)."""
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
        # The run log is written in _on_finished (covers stop/error too), which
        # always fires after this handler.
        # Run-complete dialog fires for every finished run, including
        # zero-trail runs (the dialog message branches on trail count).
        self._show_run_complete_dialog()

    def _show_run_complete_dialog(self):
        """Centered modal showing run summary. Replaces the old inline card
        which fought the log area for space and hid the Back to Setup button.
        Reads HTML lines built in _on_stats_ready and _on_timing_stats."""
        trail_html = getattr(self, '_stats_trail_line', '')
        timing_html = getattr(self, '_stats_timing_line', '')
        if not trail_html:
            return
        from PySide6.QtWidgets import QDialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Run Complete")
        dlg.setModal(True)
        dlg.setMinimumWidth(560)
        dlg.setStyleSheet(f"QDialog {{ background-color: {LIGHT_PANEL_BG}; }}")

        v = QVBoxLayout(dlg)
        v.setContentsMargins(28, 24, 28, 20)
        v.setSpacing(14)

        # Big header
        header = QLabel("Your skies are scrubbed!")
        hf = QFont()
        hf.setPointSize(24)
        hf.setBold(True)
        header.setFont(hf)
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(f"color: {BRAND_HEADING_BLUE}; background: transparent;")
        v.addWidget(header)

        # Body — same HTML the inline card used to render
        body = QLabel(trail_html + timing_html)
        body.setTextFormat(Qt.RichText)
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignCenter)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setStyleSheet(
            f"color: {CARD_TEXT}; font-size: 17px; background: transparent;"
        )
        v.addWidget(body)

        # Social-media nudge under the body, separated by a divider.
        v.addSpacing(8)
        share = QLabel(
            "Help spread the word! When you share on social media, "
            "tag <b>@bruceherwig #StarTrailCleanR</b>"
        )
        share.setTextFormat(Qt.RichText)
        share.setWordWrap(True)
        share.setAlignment(Qt.AlignCenter)
        share.setTextInteractionFlags(Qt.TextSelectableByMouse)
        share.setStyleSheet(
            f"color: {CARD_TEXT}; font-size: 15px; background: transparent;"
            f"padding: 12px 24px; border-top: 1px solid {CARD_BORDER};"
        )
        v.addWidget(share)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        open_btn = QPushButton("Open Cleaned Folder")
        open_btn.setFixedHeight(44)
        open_btn.setStyleSheet(
            f"QPushButton {{ background-color: {BRAND_HEADING_BLUE}; color: white; "
            f"font-size: 18px; font-weight: bold; border-radius: 6px; border: none; "
            f"padding: 0 18px; }}"
            f"QPushButton:hover {{ background-color: {BRAND_HEADING_HOVER}; }}"
        )
        open_btn.clicked.connect(self._open_output_folder)
        btn_row.addWidget(open_btn)

        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(44)
        close_btn.setStyleSheet(
            f"QPushButton {{ background-color: {SECONDARY_BTN_BG}; color: white; "
            f"font-size: 18px; font-weight: bold; border-radius: 6px; border: none; "
            f"padding: 0 24px; }}"
            f"QPushButton:hover {{ background-color: {DISABLED_BTN_HOVER}; }}"
        )
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn)
        v.addLayout(btn_row)

        # Center over the main window. The centering must run AFTER the
        # dialog's layout has fully rendered — adjustSize() before exec()
        # under-reports height when the body has wrapped text or a newly
        # added section, so we use a deferred QTimer.singleShot(0, ...)
        # which fires once the dialog is on screen and dlg.size() returns
        # real values. frameGeometry() on the parent includes the title
        # bar so the vertical math is what the user actually sees.
        def _center_dialog():
            """Move the run-complete dialog to the center of the main window.
            Deferred via QTimer so it runs after the dialog is on screen and
            reports real width/height (see the comment above)."""
            parent_frame = self.frameGeometry()
            if not (parent_frame.isValid() and parent_frame.width() > 0):
                return
            dx = parent_frame.x() + (parent_frame.width() - dlg.width()) // 2
            dy = parent_frame.y() + (parent_frame.height() - dlg.height()) // 2
            dlg.move(dx, dy)

        QTimer.singleShot(0, _center_dialog)
        dlg.exec()

    def _write_run_summary(self):
        """Write a plain-text run summary into <input>/cleanr_workspace/."""
        import datetime as _dt
        self._last_log_path = None  # set on success; stays None if we can't write
        input_folder = self._folder_input.text().strip()
        if not input_folder or not os.path.isdir(input_folder):
            return
        start = getattr(self, '_run_start_time', None)
        if start is None:
            return
        end = _dt.datetime.now()
        output_folder = (getattr(self, '_done_output_folder', '')
                         or self._output_input.text().strip())
        trails = getattr(self, '_run_total_trails', 0)
        frames = getattr(self, '_run_total_frames', 0)
        est_sec = getattr(self, '_run_initial_est_sec', 0)
        actual_sec = getattr(self, '_run_actual_sec', 0)

        # A cancelled or errored run never emits its final stats, so the totals
        # above stay 0. Fall back to the live running values (frames cleaned so
        # far, the last trail count, wall-clock elapsed) and flag the summary as
        # an incomplete run. A normal finish always has _run_total_frames > 0.
        incomplete = getattr(self, '_run_cancelled', False) or frames == 0
        if frames == 0:
            frames = getattr(self, '_stats_frames_done', 0)
        if trails == 0:
            trails = getattr(self, '_stats_last_trail_count', 0)
        if actual_sec == 0:
            actual_sec = max(0.0, (end - start).total_seconds())

        # Mirror the on-screen "TIME SAVED" formatting.
        SECONDS_PER_MANUAL_TRAIL = 30
        saved_sec = trails * SECONDS_PER_MANUAL_TRAIL
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

        tail = "You're welcome." if actual_sec <= est_sec else "My apologies."
        detector = self._current_model_display_name() or "(unknown)"

        # Technical details — gathered fresh each run for the debug section.
        import platform as _platform
        try:
            from modules import detect_trails as _dt
            hybrid_on = bool(getattr(_dt, "HYBRID_AXIS_EXTEND_ENABLED", False))
            slope_on = bool(getattr(_dt, "SLOPE_MATCH_ENABLED", False))
            try:
                compute = _dt.best_device()
            except Exception:
                compute = "(unknown)"
        except Exception:
            hybrid_on = False
            slope_on = False
            compute = "(unknown)"

        hybrid_label = "active (v6 guarded)" if hybrid_on else "off"
        slope_label = "active" if slope_on else "off"
        compute_pretty = {"cuda": "NVIDIA CUDA", "mps": "Apple MPS",
                          "cpu": "CPU"}.get(compute, compute)

        sysname = _platform.system()
        if sysname == "Darwin":
            mac_ver = _platform.mac_ver()[0] or _platform.release()
            os_line = f"macOS {mac_ver}"
        elif sysname == "Windows":
            os_line = f"Windows {_windows_release_label()}"
        else:
            os_line = f"{sysname} {_platform.release()}"

        machine = _platform.machine()
        hw_line = {"arm64": "Apple Silicon (arm64)",
                   "x86_64": "Intel/AMD 64-bit (x86_64)",
                   "AMD64": "Intel/AMD 64-bit (AMD64)"}.get(machine, machine)

        out_fmt = self._format_combo.currentText() if hasattr(self, "_format_combo") else "(unknown)"
        jpeg_q = self._jpeg_quality.value() if hasattr(self, "_jpeg_quality") else None
        if out_fmt == "JPG" and jpeg_q is not None:
            output_line = f"JPG (quality {jpeg_q})"
        else:
            output_line = out_fmt

        # Pipeline settings — these are the script defaults the GUI runs with.
        # If a future GUI exposes them, surface the live values here instead.
        dilate_val = 1
        conf_val = 0.25
        tile_val = 640
        overlap_val = 0.2

        # Read EXIF from the first image in the input folder.
        def _read_exif_summary(folder):
            """Return a dict of camera/lens/date/f-stop/ISO read from the first
            image's EXIF, for the "Camera Info" block of the run summary. Every
            field defaults to "Unknown" and any read failure is swallowed."""
            from PIL import Image as _PILImage
            from PIL.ExifTags import TAGS
            from modules.frame_list import IMAGE_EXTS as exts
            first = next(
                (p for p in sorted(os.listdir(folder))
                 if os.path.splitext(p)[1].lower() in exts),
                None
            )
            fields = {
                "camera": "Unknown",
                "lens":   "Unknown",
                "taken":  "Unknown",
                "fstop":  "Unknown",
                "iso":    "Unknown",
            }
            if first is None:
                return fields
            try:
                with _PILImage.open(os.path.join(folder, first)) as _im:
                    raw = _im.getexif()
                    if not raw:
                        return fields
                    tag_map = {TAGS.get(k, k): v for k, v in raw.items()}
                    # get_ifd() must be called while the file is still open.
                    sub_map = {TAGS.get(k, k): v for k, v in raw.get_ifd(0x8769).items()}
                make  = str(tag_map.get("Make",  "")).strip()
                model = str(tag_map.get("Model", "")).strip()
                if make or model:
                    if make and model.startswith(make):
                        fields["camera"] = model
                    else:
                        fields["camera"] = f"{make} {model}".strip() or "Unknown"
                lens = str(sub_map.get("LensModel", "")).strip()
                fields["lens"] = lens or "Unknown"
                taken = str(sub_map.get("DateTimeOriginal", "")).strip()
                fields["taken"] = taken or "Unknown"
                fnum = sub_map.get("FNumber")
                if fnum is not None:
                    try:
                        fields["fstop"] = f"f/{float(fnum):.1f}"
                    except Exception:
                        fields["fstop"] = str(fnum)
                iso = sub_map.get("ISOSpeedRatings")
                if iso is not None:
                    fields["iso"] = str(iso)
            except Exception:
                pass
            return fields

        exif = _read_exif_summary(input_folder)

        lines = [
            "================================================",
            "  Star Trail CleanR",
            "  Remove the Trails. Keep the Stars.",
            "  www.startrailcleanr.com",
            "================================================",
            "",
            "Run Summary",
            "",
            f"Date:                  {start.strftime('%Y-%m-%d')}",
            f"Started:               {start.strftime('%H:%M:%S')}",
            f"Finished:              {end.strftime('%H:%M:%S')}",
            "",
            f"App version:           Beta v{VERSION}",
            f"Trail DetectoR:        {detector}",
            "",
            "Camera Info",
            f"  Camera:              {exif['camera']}",
            f"  Lens:                {exif['lens']}",
            f"  Date/Time taken:     {exif['taken']}",
            f"  F-stop:              {exif['fstop']}",
            f"  ISO:                 {exif['iso']}",
            "",
            "Input folder:",
            f"  {input_folder}",
            "",
            "Output folder:",
            f"  {output_folder}",
            "",
            f"Frames processed:      {frames:,}"
            + ("   (run cancelled before it finished)" if incomplete else ""),
            f"Trails found:          {trails:,}",
        ]

        # Skipped-frames detail (only present if pre-flight filter dropped any).
        filter_info = getattr(self, '_run_filter_info', None)
        if filter_info:
            n_mismatched = filter_info.get("mismatched", 0)
            n_unreadable = filter_info.get("unreadable", 0)
            n_total = filter_info.get("total_found", 0)
            dominant = filter_info.get("dominant_size", "?")
            if n_mismatched or n_unreadable:
                lines.append(
                    f"Frames skipped:        {n_mismatched + n_unreadable} "
                    f"(of {n_total} found)"
                )
                if n_mismatched:
                    lines.append(
                        f"  - {n_mismatched} had a different resolution from "
                        f"the dominant {dominant}"
                    )
                if n_unreadable:
                    lines.append(
                        f"  - {n_unreadable} couldn't be opened to check "
                        f"resolution"
                    )

        if incomplete:
            lines += [
                "",
                f"Time before cancel:    {fmt_hms(actual_sec)}",
                "",
                "Run cancelled before completion — the numbers above are the "
                "progress made so far, not a finished run.",
                "",
                f"Time saved so far (at ~30 sec per trail):  {time_saved}",
            ]
        else:
            lines += [
                "",
                f"Estimated time:        {fmt_hms(est_sec)}",
                f"Actual time:           {fmt_hms(actual_sec)}",
                "",
                f"Thought it'd take {fmt_hms(est_sec)}. "
                f"Took {fmt_hms(actual_sec)}. {tail}",
                "",
                f"Time saved vs cleaning manually (at ~30 sec per trail):  {time_saved}",
            ]
        lines += [
            "",
            "================================================",
            "",
            "Technical Details",
            "",
            f"Mask edge expansion:   dilate={dilate_val}",
            f"AI confidence cutoff:  {conf_val}",
            f"Tile size / overlap:   {tile_val} px / {int(overlap_val*100)}%",
            f"Hybrid axis-extend:    {hybrid_label}",
            f"Slope-match merge:     {slope_label}",
            f"Output:                {output_line}",
            f"Operating system:      {os_line}",
            f"Hardware:              {hw_line}",
            f"Compute device:        {compute_pretty}",
            f"Python:                {_platform.python_version()}",
            "",
            "================================================",
        ]

        # Append the Star Log content (what scrolled in the run window) so
        # this single file is enough to diagnose any run issue without
        # asking the user for screenshots. For very large runs (1000+
        # frames) the log can grow into hundreds of KB of repetitive
        # per-frame progress lines; collapse the middle so the file stays
        # human-scrollable. Useful info clusters at the head (resolution,
        # batch count, skipped-files notice) and tail (errors, completion
        # summary), so head+tail captures the diagnosis-relevant lines.
        try:
            log_text = self._status_out.toPlainText() if hasattr(self, '_status_out') else ""
        except Exception:
            log_text = ""
        if log_text.strip():
            log_lines = log_text.rstrip().splitlines()
            HEAD_LINES = 50
            TAIL_LINES = 100
            ELIDE_THRESHOLD = HEAD_LINES + TAIL_LINES + 20
            if len(log_lines) > ELIDE_THRESHOLD:
                head = log_lines[:HEAD_LINES]
                tail = log_lines[-TAIL_LINES:]
                omitted = len(log_lines) - HEAD_LINES - TAIL_LINES
                rendered_log = "\n".join(head) + (
                    f"\n\n... ({omitted:,} lines elided — repetitive per-frame "
                    f"progress; first {HEAD_LINES} and last {TAIL_LINES} lines kept) ...\n\n"
                ) + "\n".join(tail)
            else:
                rendered_log = "\n".join(log_lines)
            lines += [
                "",
                "Star Log (what scrolled in the run window)",
                "================================================",
                rendered_log,
                "================================================",
            ]

        workspace = os.path.join(input_folder, WORKSPACE_DIR)
        try:
            os.makedirs(workspace, exist_ok=True)
            fname = f"star_trail_cleanr_log_{start.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
            _full = os.path.join(workspace, fname)
            with open(_full, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            self._last_log_path = _full  # opened by the "View Star Log" link
        except OSError:
            pass

    def _update_open_btn_state(self):
        """Enable the Setup page's big "Open Cleaned Folder" button only when
        the output folder exists and already contains at least one file."""
        folder = self._output_input.text().strip()
        has_files = False
        if folder and os.path.isdir(folder):
            has_files = any(os.scandir(folder))
        self._setup_open_btn.setEnabled(has_files)

    def _open_output_from_setup(self):
        """Setup page "Open Cleaned Folder" button. Open the output folder
        (falling back to the last completed run's output) in the file manager,
        with inline errors if it isn't set or doesn't exist yet."""
        folder = self._output_input.text().strip()
        if not folder:
            folder = getattr(self, '_done_output_folder', None)
        if not folder:
            self._error_label.setText("No output folder set \u2014 select an input folder first.")
            return
        if not os.path.isdir(folder):
            self._error_label.setText(f"Output folder doesn\u2019t exist yet \u2014 run cleaning first.")
            return
        _open_folder_in_file_manager(folder)

    def _open_output_folder(self):
        """Processing page / run-complete dialog "Open Cleaned Folder" button.
        Open the just-finished run's output folder in the file manager."""
        folder = getattr(self, '_done_output_folder', None)
        if folder and os.path.isdir(folder):
            _open_folder_in_file_manager(folder)

    def _on_finished(self):
        """QThread.finished handler — fires after every run end (clean finish,
        cancel, or error). Stops timers, restores the Settings run-locked
        controls, writes the run-summary text file, and reveals the "View Star
        Log" link if that file was written."""
        self._stop_elapsed_timer()
        self._switch_to_back_btn()
        self._set_updates_run_state(False)
        # Write this run's log (covers a clean finish, a stop/cancel, and an
        # error) and reveal the "View Star Log" link only if the file exists.
        self._write_run_summary()
        if getattr(self, "_last_log_path", None) and os.path.isfile(self._last_log_path):
            self._view_log_link.setVisible(True)

    def _view_star_log(self, *args):
        """Open this run's saved Star Log text file in the system text viewer."""
        path = getattr(self, "_last_log_path", None)
        if path and os.path.isfile(path):
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def closeEvent(self, event):
        """Save window size and clean up worker thread before closing."""
        SETTINGS.setValue("window_geometry", self.saveGeometry())
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        event.accept()


class MaskEditorWindow(QMainWindow):
    """Separate window for mask painting — closes back to main setup.

    A thin host around MaskPainterWidget: it sizes itself to 90% of the screen,
    forwards frame/mask loading to the painter, and re-emits the painter's
    finished mask as the mask_saved signal (which MainWindow saves to disk).
    """
    mask_saved = Signal(np.ndarray)

    def __init__(self, parent=None):
        """Create the mask-editor window: embed a MaskPainterWidget, wire its
        done/skip/back signals, and center it at 90% of the screen size."""
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
        """Show a single image in the painter as the mask backdrop."""
        self._painter.load_image(img_path)

    def load_frames(self, paths, index=0):
        """Load the full frame list into the painter (so the user can flip
        between frames while masking) and show the one at `index`."""
        self._painter.load_frames(paths, index)

    def load_existing_mask(self, mask_path: str):
        """Load a previously saved mask PNG into the painter for editing."""
        self._painter.load_existing_mask(mask_path)

    def _on_done(self, mask_np):
        """Painter finished. Re-emit the painted mask array via mask_saved
        (MainWindow writes it to disk) and close this window."""
        self.mask_saved.emit(mask_np)
        self.close()


if __name__ == '__main__':
    # Cross-platform single-instance check (Mac, Windows, Linux)
    import socket
    _lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _lock_socket.bind(('127.0.0.1', 49173))
    except OSError:
        app = QApplication(sys.argv)
        QMessageBox.warning(None, "Star Trail CleanR",
                            "Star Trail CleanR is already running.")
        sys.exit(1)

    app = QApplication(sys.argv)
    app._lock_socket = _lock_socket  # exposed so _relaunch can close it before spawning

    # Bundle our own font so widgets render at the same widths on every OS.
    # Without this, Mac uses San Francisco, Windows uses Segoe UI, Linux uses
    # whatever the desktop theme gives us — same point size renders different
    # widths, which clipped controls (e.g. the JPEG quality spinbox at 55 px
    # was fine on Mac, hid the number on Windows). Inter is OFL-licensed and
    # bundled in assets/fonts/. Falls back to system default if files missing.
    if getattr(sys, 'frozen', False):
        _font_base = sys._MEIPASS
    else:
        _font_base = os.path.dirname(os.path.abspath(__file__))
    _fonts_dir = os.path.join(_font_base, 'assets', 'fonts')
    from PySide6.QtGui import QFontDatabase
    for _fname in ("Inter-Regular.ttf", "Inter-Bold.ttf"):
        _fpath = os.path.join(_fonts_dir, _fname)
        if os.path.exists(_fpath):
            QFontDatabase.addApplicationFont(_fpath)
    if "Inter" in QFontDatabase.families():
        _app_font = QFont("Inter")
        # 13pt for Inter ≈ visual weight of 10pt San Francisco on Mac.
        # Inter has a smaller x-height than the platform defaults, so the
        # raw point size doesn't translate 1:1 between fonts.
        _app_font.setPointSize(13)
        app.setFont(_app_font)
        # Baseline sizes for input controls + buttons that don't carry their
        # own stylesheet. Buttons with inline styles (Run, Support, X, notice
        # cards) override these. Ensures Browse, Open Folder, spin/combo
        # boxes etc. don't render tiny in Inter.
        app.setStyleSheet(
            "QPushButton { font-size: 16px; }"
            "QComboBox { font-size: 15px; }"
            "QSpinBox  { font-size: 15px; }"
            "QLineEdit { font-size: 14px; }"
        )

    # Set the Dock / taskbar icon. Same icon file the frozen build embeds,
    # so dev-mode (live Python via the AppleScript wrapper) also gets the
    # proper Star Trail CleanR icon instead of the Python launcher's rocket.
    if getattr(sys, 'frozen', False):
        _icon_base = sys._MEIPASS
    else:
        _icon_base = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == 'win32':
        _icon_ext = '.ico'
    elif sys.platform == 'darwin':
        _icon_ext = '.icns'
    else:
        _icon_ext = '.png'
    _icon_path = os.path.join(_icon_base, 'assets', 'StarTrailCleanR' + _icon_ext)
    if os.path.exists(_icon_path):
        app.setWindowIcon(QIcon(_icon_path))

    _apply_theme()

    # Startup splash: visible while the slow first-launch work runs (theme
    # detection, Sentry/Sparkle init, MainWindow construction, font setup).
    # Without this, users see a frozen-looking app for 1-3 seconds on first
    # launch, since the GUI renders before the event loop is fully unblocked.
    # Frameless+StaysOnTop combo per feedback_qt_splash_flags.md (Qt's
    # SplashScreen flag is unreliable on macOS).
    from PySide6.QtWidgets import QProgressBar
    _splash = QWidget()
    _splash.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    _splash.setAttribute(Qt.WA_TranslucentBackground)
    _splash.setFixedSize(600, 338)
    _splash_outer = QVBoxLayout(_splash)
    _splash_outer.setContentsMargins(0, 0, 0, 0)
    _splash_card = QFrame()
    _splash_card.setStyleSheet(
        "QFrame#stcSplashCard { background: #f4f6f9; border-radius: 12px; border: 1px solid #d6dde6; }"
    )
    _splash_card.setObjectName("stcSplashCard")
    _splash_outer.addWidget(_splash_card)

    # Card body: top bar, content row, bottom bar
    _splash_body = QVBoxLayout(_splash_card)
    _splash_body.setContentsMargins(0, 0, 0, 0)
    _splash_body.setSpacing(0)

    _splash_top_bar = QFrame()
    _splash_top_bar.setFixedHeight(64)
    _splash_top_bar.setStyleSheet(
        f"QFrame {{ background: {BRAND_HEADER_BG}; border: none; "
        "border-top-left-radius: 11px; border-top-right-radius: 11px; }"
    )
    _splash_body.addWidget(_splash_top_bar)

    _splash_content = QWidget()
    _splash_content.setStyleSheet("background: transparent;")
    _splash_row = QHBoxLayout(_splash_content)
    _splash_row.setContentsMargins(32, 20, 32, 20)
    _splash_row.setSpacing(24)
    _splash_body.addWidget(_splash_content, 1)

    _splash_bottom_bar = QFrame()
    _splash_bottom_bar.setFixedHeight(64)
    _splash_bottom_bar.setStyleSheet(
        f"QFrame {{ background: {BRAND_HEADER_BG}; border: none; "
        "border-bottom-left-radius: 11px; border-bottom-right-radius: 11px; }"
    )
    _splash_body.addWidget(_splash_bottom_bar)
    _splash_icon = QLabel()
    if os.path.exists(_icon_path):
        _splash_icon.setPixmap(QIcon(_icon_path).pixmap(140, 140))
    _splash_icon.setFixedSize(140, 140)
    # No frame around the icon graphic; the icon file already has its own
    # rounded shape baked in.
    _splash_icon.setStyleSheet("background: transparent; border: none;")
    _splash_row.addWidget(_splash_icon, 0, Qt.AlignVCenter)
    _splash_text_col = QVBoxLayout()
    _splash_text_col.setContentsMargins(0, 0, 0, 0)
    _splash_text_col.setSpacing(8)
    _splash_text_col.addStretch(1)
    _splash_title = QLabel("Star Trail CleanR")
    _splash_title.setStyleSheet("font-size: 24pt; font-weight: bold; color: #1a1f2c; background: transparent; border: none;")
    _splash_text_col.addWidget(_splash_title)
    _splash_sub = QLabel("Remove the Trails. Keep the Stars.")
    _splash_sub.setStyleSheet("font-size: 14pt; color: #4a5568; background: transparent; border: none;")
    _splash_text_col.addWidget(_splash_sub)
    _splash_hashtag = QLabel("#StarTrailCleanR")
    _splash_hashtag.setStyleSheet("font-size: 12pt; color: #2d3748; background: transparent; border: none;")
    _splash_text_col.addWidget(_splash_hashtag)
    _splash_text_col.addSpacing(14)
    _splash_bar = QProgressBar()
    _splash_bar.setRange(0, 0)
    _splash_bar.setTextVisible(False)
    _splash_bar.setFixedHeight(8)
    _splash_bar.setStyleSheet(
        "QProgressBar { background: #e2e8f0; border: none; border-radius: 4px; }"
        "QProgressBar::chunk { background: #4a9eff; border-radius: 4px; }"
    )
    _splash_text_col.addWidget(_splash_bar)
    _splash_status = QLabel("Initializing…")
    _splash_status.setStyleSheet("font-size: 18pt; color: #6b7280; background: transparent; border: none;")
    _splash_text_col.addWidget(_splash_status)
    _splash_text_col.addStretch(1)
    _splash_row.addLayout(_splash_text_col, 1)
    _screen = QApplication.primaryScreen()
    if _screen:
        _g = _screen.availableGeometry()
        _splash.move(_g.x() + (_g.width() - 600) // 2, _g.y() + (_g.height() - 338) // 2)
    _maybe_init_sentry()

    import time as _time
    _cleanr_relaunch = '--cleanr-relaunch' in sys.argv
    if _cleanr_relaunch:
        _splash_shown_at = 0.0
    else:
        _splash.show()
        app.processEvents()
        _splash_shown_at = _time.monotonic()

    # Pre-window launch recovery (added v1.99-beta after v1.97-beta shipped a
    # NameError that crashed MainWindow.__init__ before the in-app update
    # banner could render — users had no signal that a fix existed). Two
    # safety nets, both running entirely outside MainWindow's setup code so
    # a future launch-class crash cannot block them:
    #   (1) Sparkle (Mac) / WinSparkle (Windows) auto-update infrastructure.
    #       Native popup appears IN-APP when an update is available; the
    #       app downloads only the changed bytes and restarts itself.
    #       Replaces the v1.x GitHub-Releases-API poll. Linux falls through
    #       to a no-op for now; banner-style notification can be added
    #       later if Linux usage warrants.
    #   (2) try/except around MainWindow construction + show, routed to
    #       _handle_launch_failure (defined at module scope so the test
    #       suite treats its imports as lazy). Sentry still gets the
    #       report; a fallback dialog points the user at the download page.
    _splash_status.setText("Checking for updates…")
    app.processEvents()
    if sys.platform == "darwin":
        from modules.sparkle_updater import init_sparkle

        def _dismiss_splash_for_update():
            """Close the startup splash so Sparkle's update popup isn't hidden
            behind it. Passed to init_sparkle as the on-update-found callback."""
            try:
                _splash.close()
            except Exception:
                pass

        init_sparkle(on_update_found=_dismiss_splash_for_update)
        # Check for an update on EVERY launch. Sparkle shows its one-click
        # "install now" popup ONLY if a newer version exists; if the user is
        # current, nothing appears. So a new release reaches people the moment
        # they open the app, not on the once-a-day timer. (Must be called here,
        # right after the updater starts and before the Qt event loop spins.)
        from modules.sparkle_updater import check_for_updates_in_background
        check_for_updates_in_background()
    elif sys.platform == "win32":
        from modules.winsparkle_updater import (
            init_winsparkle,
            check_for_updates_in_background as _winsparkle_check_on_launch,
        )
        init_winsparkle(
            appcast_url="https://bruceherwig-dot.github.io/star-trail-cleanr/appcast-windows.xml",
            app_name="Star Trail CleanR",
            app_version=VERSION,
            company_name="Star Trail CleanR",
        )
        # Same on Windows: check on every launch via WinSparkle. The native
        # "update available" window appears only if there's a newer version.
        _winsparkle_check_on_launch()
    _splash_status.setText("Warming up the trail detector…")
    app.processEvents()
    try:
        window = MainWindow()
        window.show()
    except Exception as _launch_exc:
        _handle_launch_failure(_launch_exc)
        sys.exit(1)

    # Mac location guard: if the app is running off the mounted disk image or
    # from the quarantine sandbox ("App Translocation"), macOS silently
    # disables the built-in Sparkle updater -- the user sees "Checking for
    # updates" at launch but no update ever arrives and the Check for Updates
    # button stays mute. Stranded a real user on 2026-06-10 (they reinstalled
    # from the website every release, never knowing why). Tell them how to fix
    # it the moment the window is up.
    if sys.platform == "darwin" and getattr(sys, "frozen", False):
        _exe_path = sys.executable
        if "/AppTranslocation/" in _exe_path or _exe_path.startswith("/Volumes/"):
            _where = ("the downloaded disk image"
                      if _exe_path.startswith("/Volumes/")
                      else "a temporary location macOS uses for newly "
                           "downloaded apps")
            def _show_move_to_applications_notice():
                """Deferred so the dialog appears over the main window, after
                the splash is gone."""
                QMessageBox.information(
                    window, "Move Star Trail CleanR to Applications",
                    f"Star Trail CleanR is running from {_where}, so macOS "
                    "has turned off automatic updates for this session.\n\n"
                    "To fix it: drag Star Trail CleanR into your Applications "
                    "folder, then launch it from there. Updates will then "
                    "install themselves with one click.")
            QTimer.singleShot(3500, _show_move_to_applications_notice)

    # Dismiss the startup splash. Minimum on-screen duration of 3000 ms (3
    # seconds) so the user gets to see the title + tagline + progress bar even
    # on a cached relaunch where the rest of startup is fast. On a true cold
    # first launch, the slow imports keep the splash up longer than the
    # minimum and this delay is effectively zero.
    _elapsed_ms = int((_time.monotonic() - _splash_shown_at) * 1000)
    _remaining_ms = max(0, 3000 - _elapsed_ms)
    QTimer.singleShot(_remaining_ms, _splash.close)

    # Live OS appearance switching: when the user toggles macOS Light/Dark
    # mid-session, relaunch so every themed widget rebuilds with the new
    # palette. QSettings preserves folder selections and options, so the
    # user lands right back where they were.
    def _on_color_scheme_changed(_scheme):
        """OS light/dark toggle handler. Relaunch the app so every themed
        widget rebuilds with the new palette (QSettings preserves the user's
        in-progress choices). Skipped while a run is active to avoid
        interrupting it."""
        try:
            if window.worker and window.worker.isRunning():
                return
            window._relaunch()
        except Exception:
            pass
    try:
        app.styleHints().colorSchemeChanged.connect(_on_color_scheme_changed)
    except Exception:
        pass

    sys.exit(app.exec())
