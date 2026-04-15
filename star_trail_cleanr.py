import sys
import os

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
from PySide6.QtGui import QFont, QPixmap

from mask_painter import MaskPainterWidget

if getattr(sys, 'frozen', False):
    _base = sys._MEIPASS
else:
    _base = os.path.dirname(os.path.abspath(__file__))

# Theme colors — initialized to light-mode defaults; _apply_theme() updates
# these at startup based on the OS color scheme. Read by stylesheets via
# f-string interpolation, so any widget built after main() runs gets the
# right values.
MUTED_TEXT = "#666"
HINT_TEXT = "#888"
CARD_BG = "#e0e0e0"
CARD_TEXT = "#000"
CARD_BORDER = "#ccc"
LIGHT_PANEL_BG = "#f0f7ff"
DISABLED_BTN_BG = "#999"
DISABLED_BTN_HOVER = "#888"
SECONDARY_BTN_BG = "#666"


def _apply_theme():
    """Detect OS color scheme and update the theme color globals.

    Called once from main() before any widget is built. Widget stylesheets
    interpolate these globals via f-strings, so they pick up the right
    values automatically. The mode is detected once at startup; toggling
    the OS theme mid-session won't take effect until restart.
    """
    global MUTED_TEXT, HINT_TEXT, CARD_BG, CARD_TEXT, CARD_BORDER
    global LIGHT_PANEL_BG, DISABLED_BTN_BG, DISABLED_BTN_HOVER, SECONDARY_BTN_BG
    try:
        from PySide6.QtCore import Qt as _Qt
        scheme = QApplication.styleHints().colorScheme()
        is_dark = (scheme == _Qt.ColorScheme.Dark)
    except Exception:
        is_dark = False
    if is_dark:
        MUTED_TEXT = "#aaaaaa"
        HINT_TEXT = "#9aa4b0"
        CARD_BG = "#2d3138"
        CARD_TEXT = "#e6e6e6"
        CARD_BORDER = "#3a3f4a"
        LIGHT_PANEL_BG = "#1c2733"
        DISABLED_BTN_BG = "#4a4a4a"
        DISABLED_BTN_HOVER = "#5a5a5a"
        SECONDARY_BTN_BG = "#555555"


SCRIPT = os.path.join(_base, "astro_clean_v5.py")
MODEL = os.path.join(os.path.expanduser("~"),
                     "Documents/yolo_runs/trail_detector_v11s_tiled/weights/best.pt")

try:
    with open(os.path.join(_base, "version.txt")) as _f:
        VERSION = _f.read().strip()
except Exception:
    VERSION = "dev"

SETTINGS = QSettings("StarTrailCleanR", "StarTrailCleanR")


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
                           "-o", output_folder, "--model", MODEL,
                           "--start", str(start), "--batch", str(this_batch)]
                else:
                    cmd = [sys.executable, "-u", SCRIPT, folder,
                           "-o", output_folder, "--model", MODEL,
                           "--start", str(start), "--batch", str(this_batch)]

                if self.mask_path:
                    cmd.extend(["--foreground-mask", self.mask_path])
                    cmd.extend(["--hot-pixel-map", hot_map_file])
                cmd.extend(["--output-format", self.output_format,
                            "--jpeg-quality", str(self.jpeg_quality)])

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
            "QTabWidget::pane { border: none; background: palette(window); }"
            "QTabBar { qproperty-drawBase: 0; }"
            "QTabBar::tab { background: #142a4a; color: #a8c0e0; padding: 14px 20px; "
            "font-size: 15px; font-weight: bold; border: none; min-width: 200px; }"
            "QTabBar::tab:selected { background: #1a6fc4; color: white; }"
            "QTabBar::tab:hover:!selected { background: #1d3a66; color: white; }"
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
        container_layout.addWidget(self._tabs)
        self.setCentralWidget(container)

        self._build_setup_page()
        self._build_process_page()
        self._stack.setCurrentIndex(0)

        for lbl in self.findChildren(QLabel):
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

    # ── FAQ tab ──────────────────────────────────────────────────────────────

    def _build_faq_tab(self):
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet(
            "QTextBrowser { background: palette(window); border: none; padding: 24px; font-size: 14px; }"
        )
        browser.setHtml(f"""
        <html><body style='font-family: -apple-system, sans-serif; line-height: 1.5;'>
        <h2 style='color:#1a6fc4; margin-top:0; margin-bottom:2px;'>Why Star Trail CleanR?</h2>
        <p style='margin-top:2px;'>Star Trail CleanR removes airplane and satellite trails
        from astrophotography sequences while preserving the real stars. The result is a
        clean set of frames you can stack into a perfect star trail composite.</p>

        <h2 style='color:#1a6fc4; margin-bottom:2px;'>Trail Detection</h2>
        <p style='margin-top:2px;'>Each frame is run through a YOLO segmentation model
        trained on thousands of manually labeled airplane and satellite trails across many
        cameras, lenses, and sky conditions. The model produces pixel-accurate masks for
        every trail it finds.</p>

        <h2 style='color:#1a6fc4; margin-bottom:2px;'>The Fix &mdash; Star Bridge Repair</h2>
        <p style='margin-top:2px;'>For each trail, Star Trail CleanR pulls clean pixels
        from the frame immediately before and after, blending them across the trail using
        a morphing technique called <i>Star Bridge</i>. This preserves the real stars
        underneath the trail and keeps the brightness and color natural &mdash; no smudges,
        no blank patches.</p>

        <h2 style='color:#1a6fc4; margin-bottom:2px;'>Workflow</h2>
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

        <h2 style='color:#1a6fc4; margin-bottom:2px;'>Limitations</h2>
        <ul style='margin-top:2px;'>
        <li>Designed for wide-field star trail sequences, not deep-sky tracked exposures.</li>
        <li><b>Trail variety is bounded by the AI's training data.</b> If a type of
        trail isn't being detected well in your sequences, you can help train the next
        version: zip 300+ frames from that scene and send them to
        <a href='mailto:bruceherwig@gmail.com?subject=Star%20Trail%20CleanR%20training%20frames'>bruceherwig@gmail.com</a>.
        For large folders, share a Dropbox, Google Drive, or WeTransfer link instead
        of attaching directly. The model gets smarter every time the community
        contributes.</li>
        <li><b>RAW files (.CR2, .NEF, .ARW, etc.) are not yet supported.</b> Convert
        your sequence to JPG or TIFF first, then run Star Trail CleanR on the converted
        frames.</li>
        <li><b>Not a one-click fix.</b> You'll still want to touch up the final
        composite in Photoshop or your editor of choice &mdash; but if we did our job
        right, it's a fraction of the time you used to spend.</li>
        </ul>

        <p style='color:{HINT_TEXT}; margin-top:24px;'>Star Trail CleanR is free and offered as
        a gift to the astrophotography community.
        <a href='mailto:bruceherwig@gmail.com?subject=Star%20Trail%20CleanR%20feedback'>Feedback welcome.</a></p>
        </body></html>
        """)
        return browser

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
            "QTextBrowser { background: palette(window); border: none; font-size: 14px; }"
        )
        bio.setHtml(f"""
        <html><body style='font-family: -apple-system, sans-serif; line-height: 1.5;'>
        <h2 style='color:#1a6fc4; margin-top:0; margin-bottom:2px;'>About the Authors</h2>
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

        <h3 style='color:#1a6fc4;'>Links</h3>
        <ul>
        <li>Photos for sale: <a href='https://bruceherwig.com'>bruceherwig.com</a></li>
        <li>Blog: <a href='https://bruceherwig.wordpress.com'>bruceherwig.wordpress.com</a></li>
        </ul>

        <h3 style='color:#1a6fc4;'>Acknowledgments</h3>
        <p>Star Trail CleanR exists because of the generosity of fellow astrophotographers
        who shared their image sequences for AI training, tested early builds, and offered
        feedback. Every detected trail is a thank-you to them.</p>
        <p><a href='https://bruceherwig.wordpress.com/star-trail-cleanr/#Thanks'>See the
        full list of contributors &rarr;</a></p>

        <h3 style='color:#1a6fc4;'>Version History</h3>
        <p>See the full <a href='https://github.com/bruceherwig-dot/star-trail-cleanr/blob/main/CHANGELOG.md'>version history on GitHub</a>.</p>

        <h3 style='color:#1a6fc4;'>Share Your Work&hellip; Have a Suggestion?</h3>
        <p>Got a before-and-after you'd like to share? I would love to see it!<br>
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
        banner.setStyleSheet("background-color: #0a1e3f;")
        outer = QHBoxLayout(banner)
        outer.setContentsMargins(24, 10, 16, 10)
        outer.setSpacing(12)

        # Left: title block
        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        title = QLabel("Star Trail CleanR")
        title.setStyleSheet("color: white; font-size: 26px; font-weight: bold; background: transparent;")
        text_col.addWidget(title)
        sub = QLabel(f"Beta v{VERSION}")
        sub.setStyleSheet("color: #a8c0e0; font-size: 12px; background: transparent;")
        text_col.addWidget(sub)
        outer.addLayout(text_col)
        outer.addStretch()

        # Right: Support button
        support_btn = QPushButton("\u2764  Support")
        support_btn.setFixedHeight(32)
        support_btn.setStyleSheet(
            "QPushButton { background-color: #d0e4f5; color: #1a3a5c; font-size: 13px; "
            "font-weight: bold; border-radius: 16px; border: 1px solid #a0c4e0; "
            "padding: 0 16px; }"
            "QPushButton:hover { background-color: #b8d4ec; }"
        )
        support_btn.setToolTip("Support this project")
        support_btn.clicked.connect(lambda: __import__('webbrowser').open(
            "https://bruceherwigphotographer.square.site/product/tip-jar/WCQQP7HM4SGFWSNBSAFNX7QF"))
        outer.addWidget(support_btn)

        # Right: Close X
        quit_btn = QPushButton("\u2715")
        quit_btn.setFixedSize(32, 32)
        quit_btn.setStyleSheet(
            "QPushButton { background-color: #d93025; color: white; font-size: 22px; "
            "font-weight: bold; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #b8271b; }"
        )
        quit_btn.setToolTip("Quit Star Trail CleanR")
        quit_btn.clicked.connect(self.close)
        outer.addWidget(quit_btn)

        return banner

    # ── Setup page ───────────────────────────────────────────────────────────

    def _build_setup_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(6)
        layout.setContentsMargins(24, 20, 24, 20)

        lbl_font = QFont()
        lbl_font.setPointSize(13)
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
        self._folder_input.editingFinished.connect(self._on_input_edited)
        row_in.addWidget(self._folder_input, 4)
        browse_in = QPushButton("Browse\u2026")
        browse_in.clicked.connect(self._browse_input)
        row_in.addWidget(browse_in, 1)
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
        row_out.addWidget(self._output_input, 4)
        browse_out = QPushButton("Browse\u2026")
        browse_out.clicked.connect(self._browse_output)
        row_out.addWidget(browse_out, 1)
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
            "QPushButton { background-color: #2a7a2a; color: white; font-size: 13px; "
            "font-weight: bold; border-radius: 6px; border: none; }"
            "QPushButton:hover { background-color: #339933; }"
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
        self._jpeg_quality.setValue(80)
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

        hint6 = QLabel("Processing time depends on image count and resolution")
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
            "QPushButton { background-color: #2a7a2a; color: white; font-size: 22px; "
            "font-weight: bold; border-radius: 6px; border: none; }"
            "QPushButton:hover { background-color: #339933; }"
            f"QPushButton:disabled {{ background-color: {DISABLED_BTN_BG}; }}"
        )
        self._run_btn.clicked.connect(self._run)
        btn_row.addWidget(self._run_btn, 2)

        self._setup_open_btn = QPushButton("Open Cleaned Folder")
        self._setup_open_btn.setFixedHeight(48)
        self._setup_open_btn.setStyleSheet(
            "QPushButton { background-color: #1a6fc4; color: white; font-size: 18px; "
            "font-weight: bold; border-radius: 6px; border: none; }"
            "QPushButton:hover { background-color: #1580e0; }"
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
            int(SETTINGS.value("jpeg_quality", 80)))
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
        self._detail_label.setStyleSheet("font-size: 12px; color: #1a6fc4;")
        self._detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._detail_label)

        # ── Post-run stats card (hidden until run completes) ──
        self._stats_label = QLabel("")
        self._stats_label.setAlignment(Qt.AlignCenter)
        self._stats_label.setWordWrap(True)
        self._stats_label.setTextFormat(Qt.RichText)
        self._stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._stats_label.setStyleSheet(
            f"QLabel {{ background-color: {LIGHT_PANEL_BG}; border: 1px solid #1a6fc4; "
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
            "QPushButton { background-color: #1a6fc4; color: white; font-size: 18px; "
            "font-weight: bold; border-radius: 6px; border: none; }"
            "QPushButton:hover { background-color: #1580e0; }"
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
            is_new = folder != self._folder_input.text().strip()
            self._folder_input.setText(folder)
            SETTINGS.setValue("last_input_dir", folder)
            if is_new:
                self._frame_limit.setCurrentText("20")
            self._update_mask_status()
            self._update_frame_count()

    def _on_input_edited(self):
        folder = self._folder_input.text().strip()
        if folder and folder != getattr(self, "_last_input_seen", None):
            self._last_input_seen = folder
            self._frame_limit.setCurrentText("20")
        self._update_frame_count()

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

        if count is not None and count > 0:
            cur = self._frame_limit.currentText()
            if cur != "All Frames":
                try:
                    if int(cur) > count:
                        for i in range(self._frame_limit.count()):
                            t = self._frame_limit.itemText(i)
                            if t == "All Frames":
                                continue
                            try:
                                if int(t) <= count:
                                    self._frame_limit.setCurrentIndex(i)
                                    break
                            except ValueError:
                                pass
                        else:
                            idx = self._frame_limit.findText("All Frames")
                            if idx >= 0:
                                self._frame_limit.setCurrentIndex(idx)
                except ValueError:
                    pass

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
            self._mask_status.setStyleSheet("color: #2a7a2a; font-size: 12px; margin-left: 8px;")
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
            f"Swept <b>{total_trails:,}</b> airplane trails from your skies "
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
    _apply_theme()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
