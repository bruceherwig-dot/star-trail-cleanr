#!/usr/bin/env python3
"""mask_checkr.py — Mask CheckR: a developer tool to spot trails that slipped through.

WHAT IT IS
  A small PySide6 desktop app that answers one question visually: "Did we miss a
  trail?" It does that the same way StarStaX builds a final star-trail image —
  by lighten-max stacking a sequence of frames (each output pixel is the
  brightest value seen across all frames). If a trail was correctly removed from
  every frame, it cannot show up in the stack. So any trail still visible in the
  stacked result is a trail that escaped — either a missed CVAT annotation or a
  missed detection in the actual cleaned run.

TWO MODES (chosen with a radio button in the picker dialog)
  1. "CVAT annotations" — Pull Bruce's reviewed polygons for a chosen CVAT task
     and frame range straight from the CVAT API (never from the stale local
     labelme JSON files). Black-fill every polygon on its source frame, then
     lighten-max stack the frames. Black is the darkest possible value, so a
     correctly-annotated trail is erased from the stack; anything still visible
     means a polygon was missing in CVAT. This audits annotation coverage
     BEFORE training.
  2. "Last STC run" — Point at a dataset folder that contains a cleaned/
     subfolder (the real Star Trail CleanR output). Lighten-max stack those
     already-cleaned JPEGs exactly as StarStaX would. Any trail still visible is
     a real problem in the shipped/cleaned result — a missed or under-cleaned
     detection. (Note: this mode stacks the real cleaned frames; it does NOT
     re-derive masks. The older "load saved masks and black-fill" behavior was
     replaced by stacking the actual cleaned output.)

OUTPUT (both modes, written to a mask_checkr_output/ folder and opened in Finder)
  - a plain lighten-max stack JPEG, and
  - a second JPEG with the 640x640 detection tile grid drawn on top and each
    tile labeled A1/B2-style, so a trail can be reported by its tile address
    and jumped to in TileFixR.

HOW TO RUN
  python3 tools/mask_checkr.py
  Requires CVAT running at http://localhost:8080 for the CVAT mode (the app
  still launches without CVAT and falls back to "Last STC run" mode). Single
  instance only — a lock file prevents a second copy from running.

WHERE IT FITS
  Part of Bruce's annotation-quality loop (Trail ScreenR -> TileFixR ->
  Mask CheckR -> fix -> repeat). It is a diagnostic/review tool, not part of the
  shipped Star Trail CleanR pipeline.
"""

import atexit
import json
import os
import sys
import subprocess
from pathlib import Path

import cv2
import numpy as np
import requests

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
    QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget,
)

# ── Paths / CVAT ──────────────────────────────────────────────────────────────

CVAT_URL  = "http://localhost:8080"
CVAT_USER = "bherwig2"
TRAILS_ROOT = Path("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images")
GKYLE_STAGING = Path("/Volumes/T7 Shield/AI Projects/Star Trail CleanR/external_datasets/gkyle_startrails/cvat_staging")
TASK_FOLDER_ALIASES = {
    "My First Star Trail": "Bruce Herwig - first star trail data",
    "Thomas Jackson - Borrego": "Thomas Jackson Star Trails Borrego",
    "Greg Meyer - Arizona": "Greg Meyer Arizona",
    "Bruce Herwig - Pioneertown Fisheye": "Pioneertown 6mm Fisheye Training",
    "Bruce Herwig - Borrego Springs 1": "borrego_springs_1",
}
STATE_DIR = Path.home() / ".star_trail_cleanr"
LAST_PICK = STATE_DIR / "mask_checkr_last.json"
LOCK_FILE = STATE_DIR / "mask_checkr.lock"

# ── Lock ──────────────────────────────────────────────────────────────────────

def acquire_lock():
    """Enforce single-instance: claim the lock file or refuse to start.

    Returns True if this process now owns the lock, False if another live
    Mask CheckR process already holds it. A stale lock (the recorded PID is no
    longer running) is treated as free and overwritten. The lock is removed
    automatically at process exit via atexit.
    """
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)
            return False
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))
    atexit.register(lambda: LOCK_FILE.unlink(missing_ok=True))
    return True

# ── Theme ─────────────────────────────────────────────────────────────────────

MUTED_TEXT = "#666"

def _apply_theme():
    """Adjust the muted-text color for dark mode.

    Detects the OS color scheme via Qt and, if dark, lightens the gray used for
    secondary labels so it stays legible. Silently leaves the light-mode default
    if the scheme can't be read.
    """
    global MUTED_TEXT
    try:
        is_dark = (QApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark)
    except Exception:
        is_dark = False
    if is_dark:
        MUTED_TEXT = "#aaaaaa"

# ── CVAT helpers ──────────────────────────────────────────────────────────────

def read_cvat_password():
    """Read the CVAT password from the credentials file in the home directory.

    The password lives at ~/.star_trail_cleanr/cvat_credentials (outside the
    repo, never hardcoded). Returns the trimmed contents.
    """
    return (Path.home() / ".star_trail_cleanr" / "cvat_credentials").read_text().strip()

def fetch_cvat_tasks(auth):
    """Fetch all CVAT tasks via the API, following pagination.

    Walks every page of GET /api/tasks and returns a list of small dicts
    {id, name, size} (size = frame count), sorted case-insensitively by name.
    On any request error it prints the error and returns whatever was gathered
    so far (possibly empty), so the app can still launch in STC-only mode.
    """
    out = []
    url = f"{CVAT_URL}/api/tasks"
    while url:
        try:
            r = requests.get(url, auth=auth).json()
        except Exception as e:
            print(f"fetch_cvat_tasks error: {e}")
            break
        for t in r.get("results", []):
            out.append({"id": t["id"], "name": t["name"], "size": t.get("size", 0)})
        url = r.get("next")
    out.sort(key=lambda t: t["name"].lower())
    return out

def _norm_name(s):
    """Lowercase and treat underscores/hyphens as spaces, collapsing runs of
    whitespace, so 'GoPro_G0037688' and 'GoPro G0037688' compare equal. CVAT
    task names and disk folder names drift on exactly this punctuation."""
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())


def resolve_image_dir(task_name):
    """Map a CVAT task name to the on-disk folder of its source images.

    Tries, in order: the special gkyle staging folder; an explicit alias from
    TASK_FOLDER_ALIASES; a folder under TRAILS_ROOT named exactly like the task
    (also trying the part before a " - v..." version suffix); then a match
    against the children of TRAILS_ROOT that ignores case AND treats
    underscores/hyphens/spaces as the same character (exact-normalized first,
    substring second). Returns the matching Path, or None if nothing is found.
    """
    if "gkyle" in task_name.lower() and GKYLE_STAGING.exists():
        return GKYLE_STAGING
    if task_name in TASK_FOLDER_ALIASES:
        p = TRAILS_ROOT / TASK_FOLDER_ALIASES[task_name]
        if p.exists():
            return p
    candidates = [task_name]
    if " - v" in task_name:
        candidates.append(task_name.split(" - v")[0].rstrip())
    for c in candidates:
        p = TRAILS_ROOT / c
        if p.exists():
            return p
    base = candidates[-1]
    if TRAILS_ROOT.exists():
        norm_base = _norm_name(base)
        children = [c for c in TRAILS_ROOT.iterdir() if c.is_dir()]
        for child in children:
            if _norm_name(child.name) == norm_base:
                return child
        for child in children:
            if norm_base in _norm_name(child.name):
                return child
    return None

# ── Persistence ───────────────────────────────────────────────────────────────

def load_last_pick():
    """Load the user's previous picker selections so the dialog reopens pre-filled.

    Reads the saved JSON (task id, frame range, source mode, last STC dataset
    folder) and returns it as a dict with safe defaults for any missing key. On
    any error (file missing or corrupt) returns a fully-defaulted dict.
    """
    try:
        d = json.loads(LAST_PICK.read_text())
        return {
            "task_id":        d.get("task_id", -1),
            "first_frame":    d.get("first_frame", 1),
            "last_frame":     d.get("last_frame", 9999),
            "source_mode":    d.get("source_mode", "cvat"),
            "stc_images_dir": d.get("stc_images_dir", ""),
            "stc_masks_dir":  d.get("stc_masks_dir", ""),
        }
    except Exception:
        return {"task_id": -1, "first_frame": 1, "last_frame": 9999,
                "source_mode": "cvat", "stc_images_dir": "", "stc_masks_dir": ""}

def save_last_pick(data: dict):
    """Persist the current picker selections to the state JSON for next launch."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LAST_PICK.write_text(json.dumps(data, indent=2))

# ── Task picker ───────────────────────────────────────────────────────────────

SELECTABLE = Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard

class TaskPickerDialog(QDialog):
    """The startup dialog where the user chooses what to stack.

    Presents two mutually-exclusive sources via radio buttons:
      - "CVAT annotations": pick a CVAT task and a 1-based frame range.
      - "Last STC run": browse to a dataset folder that has a cleaned/
        subfolder, with its own frame range.
    Only one source section is visible at a time. The Run button enables only
    when the active source has a valid selection (a resolvable CVAT task, or a
    dataset folder whose cleaned/ subfolder exists). Selections are remembered
    between launches via load_last_pick / save_last_pick.
    """
    def __init__(self, tasks, last_pick: dict, auth=None):
        """Build the dialog from the loaded CVAT task list and saved selections.

        tasks: list of {id, name, size} task dicts (empty if CVAT is offline).
        last_pick: previously saved choices used to pre-select the task, frame
        range, source mode, and last STC folder. auth: CVAT (user, password)
        tuple, used to query per-task metadata (deleted-frame counts).
        """
        super().__init__()
        self.tasks = tasks
        self.auth  = auth
        self.setWindowTitle("Mask CheckR")
        self.setMinimumWidth(580)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Mask CheckR")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        title.setTextInteractionFlags(SELECTABLE)
        layout.addWidget(title)

        # ── Source radio buttons ──
        src_row = QHBoxLayout()
        src_row.setSpacing(20)
        self._src_group = QButtonGroup(self)
        self._radio_cvat = QRadioButton("CVAT annotations")
        self._radio_stc  = QRadioButton("Last STC run")
        self._src_group.addButton(self._radio_cvat, 0)
        self._src_group.addButton(self._radio_stc,  1)
        src_row.addWidget(self._radio_cvat)
        src_row.addWidget(self._radio_stc)
        src_row.addStretch()
        layout.addLayout(src_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #ccc;")
        layout.addWidget(sep)

        # ── CVAT section ──
        self._cvat_widget = QWidget()
        cvat_lay = QVBoxLayout(self._cvat_widget)
        cvat_lay.setContentsMargins(0, 0, 0, 0)
        cvat_lay.setSpacing(8)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("CVAT task:"))
        self.combo = QComboBox()
        selected_idx = 0
        for i, t in enumerate(tasks):
            self.combo.addItem(f"{t['id']:>3}  —  {t['name']}  ({t['size']} frames)", i)
            if t["id"] == last_pick["task_id"]:
                selected_idx = i
        self.combo.setCurrentIndex(selected_idx)
        self.combo.currentIndexChanged.connect(self._on_task_changed)
        row1.addWidget(self.combo, stretch=1)
        cvat_lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("First frame:"))
        self.first_spin = QSpinBox()
        self.first_spin.setMinimum(1)
        self.first_spin.setMaximum(99999)
        self.first_spin.setValue(max(1, int(last_pick["first_frame"])))
        self.first_spin.valueChanged.connect(self._on_range_changed)
        row2.addWidget(self.first_spin)
        row2.addSpacing(20)
        row2.addWidget(QLabel("Last frame:"))
        self.last_spin = QSpinBox()
        self.last_spin.setMinimum(1)
        self.last_spin.setMaximum(99999)
        self.last_spin.setValue(max(1, int(last_pick["last_frame"])))
        self.last_spin.valueChanged.connect(self._on_range_changed)
        row2.addWidget(self.last_spin)
        row2.addStretch()
        cvat_lay.addLayout(row2)

        self.range_label = QLabel()
        self.range_label.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 11px;")
        self.range_label.setTextInteractionFlags(SELECTABLE)
        cvat_lay.addWidget(self.range_label)

        self.folder_label = QLabel()
        self.folder_label.setStyleSheet("font-size: 11px; font-family: monospace;")
        self.folder_label.setWordWrap(True)
        self.folder_label.setTextInteractionFlags(SELECTABLE)
        cvat_lay.addWidget(self.folder_label)

        layout.addWidget(self._cvat_widget)

        # ── STC section ──
        self._stc_widget = QWidget()
        stc_lay = QVBoxLayout(self._stc_widget)
        stc_lay.setContentsMargins(0, 0, 0, 0)
        stc_lay.setSpacing(8)

        _si = last_pick["stc_images_dir"]
        self._stc_images_dir = Path(_si) if _si and Path(_si).is_dir() else None
        self._stc_masks_dir  = None
        self._stc_cleaned_dir = None

        images_row = QHBoxLayout()
        img_lbl = QLabel("Source images:")
        img_lbl.setFixedWidth(106)
        images_row.addWidget(img_lbl)
        self._stc_images_path_lbl = QLabel(
            str(self._stc_images_dir) if self._stc_images_dir else "Not set")
        self._stc_images_path_lbl.setStyleSheet(
            "font-size: 11px; font-family: monospace; color: #666;")
        self._stc_images_path_lbl.setTextInteractionFlags(SELECTABLE)
        images_row.addWidget(self._stc_images_path_lbl, stretch=1)
        img_browse = QPushButton("Browse...")
        img_browse.setFixedHeight(26)
        img_browse.clicked.connect(self._browse_stc_images)
        images_row.addWidget(img_browse)
        stc_lay.addLayout(images_row)

        self._stc_status_lbl = QLabel()
        self._stc_status_lbl.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 11px;")
        self._stc_status_lbl.setTextInteractionFlags(SELECTABLE)
        stc_lay.addWidget(self._stc_status_lbl)

        # Frame range for STC mode (defaults to the entire batch). Same control
        # as the CVAT section and TileFixR.
        stc_range_row = QHBoxLayout()
        stc_range_row.addWidget(QLabel("First frame:"))
        self.stc_first_spin = QSpinBox()
        self.stc_first_spin.setMinimum(1)
        self.stc_first_spin.setMaximum(99999)
        self.stc_first_spin.setValue(1)
        self.stc_first_spin.valueChanged.connect(self._on_stc_range_changed)
        stc_range_row.addWidget(self.stc_first_spin)
        stc_range_row.addSpacing(20)
        stc_range_row.addWidget(QLabel("Last frame:"))
        self.stc_last_spin = QSpinBox()
        self.stc_last_spin.setMinimum(1)
        self.stc_last_spin.setMaximum(99999)
        self.stc_last_spin.setValue(1)
        self.stc_last_spin.valueChanged.connect(self._on_stc_range_changed)
        stc_range_row.addWidget(self.stc_last_spin)
        stc_range_row.addStretch()
        stc_lay.addLayout(stc_range_row)

        self.stc_range_label = QLabel()
        self.stc_range_label.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 11px;")
        self.stc_range_label.setTextInteractionFlags(SELECTABLE)
        stc_lay.addWidget(self.stc_range_label)

        if self._stc_images_dir:
            self._stc_cleaned_dir = self._detect_cleaned(self._stc_images_dir)
            self._set_stc_range_to_full()

        layout.addWidget(self._stc_widget)

        # ── Buttons ──
        self.btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btns.button(QDialogButtonBox.Ok).setText("Run")
        self.btns.button(QDialogButtonBox.Cancel).setText("Quit")
        self.btns.accepted.connect(self.accept)
        self.btns.rejected.connect(self.reject)
        layout.addWidget(self.btns)

        # Wire radio buttons after all widgets exist
        self._radio_cvat.toggled.connect(self._on_source_changed)
        self._radio_stc.toggled.connect(self._on_source_changed)

        # Disable CVAT radio if no tasks loaded
        if not tasks:
            self._radio_cvat.setEnabled(False)

        # Set initial source (triggers _on_source_changed)
        if last_pick["source_mode"] == "stc" or not tasks:
            self._radio_stc.setChecked(True)
        else:
            self._radio_cvat.setChecked(True)

        self._on_task_changed(self.combo.currentIndex())
        self._on_range_changed()

    # ── Source switching ──

    def _on_source_changed(self):
        """Show the CVAT or STC section depending on the selected radio button."""
        is_cvat = self._radio_cvat.isChecked()
        self._cvat_widget.setVisible(is_cvat)
        self._stc_widget.setVisible(not is_cvat)
        self._update_run_button()

    def _detect_cleaned(self, img_dir: Path):
        """Locate the cleaned/ output folder for an STC dataset selection.

        Returns img_dir itself if the user pointed straight at a folder named
        cleaned/, otherwise its cleaned/ subfolder if that exists, otherwise
        None.
        """
        # The cleaned output normally lives in a cleaned/ subfolder of the
        # dataset. If the user pointed straight at a cleaned/ folder, accept it.
        if img_dir.name == "cleaned" and img_dir.is_dir():
            return img_dir
        candidate = img_dir / "cleaned"
        return candidate if candidate.is_dir() else None

    def _update_run_button(self):
        """Enable or disable the Run button based on the active source's validity.

        CVAT mode requires a selected task whose image folder resolves on disk.
        STC mode requires both the dataset folder and its cleaned/ subfolder to
        exist, and also refreshes the STC status line.
        """
        if self._radio_cvat.isChecked():
            task = self.selected_task()
            ok = task is not None and resolve_image_dir(task["name"]) is not None
            self.btns.button(QDialogButtonBox.Ok).setEnabled(ok)
        else:
            images_ok = self._stc_images_dir is not None and self._stc_images_dir.is_dir()
            cleaned_ok = self._stc_cleaned_dir is not None and self._stc_cleaned_dir.is_dir()
            self.btns.button(QDialogButtonBox.Ok).setEnabled(images_ok and cleaned_ok)
            self._refresh_stc_status(images_ok, cleaned_ok)

    def _refresh_stc_status(self, images_ok, cleaned_ok):
        """Update the STC status line with a colored hint about the current folder.

        Gray prompt if no dataset folder is chosen, red warning if no cleaned/
        subfolder is found, green confirmation with the cleaned-frame count when
        everything is ready.
        """
        if not images_ok:
            self._stc_status_lbl.setText("Select the dataset folder (the one containing the cleaned/ output).")
            self._stc_status_lbl.setStyleSheet(f"color: {MUTED_TEXT}; font-size: 11px;")
        elif not cleaned_ok:
            self._stc_status_lbl.setText(
                "No cleaned images found. Expected a cleaned/ subfolder from the STC run.")
            self._stc_status_lbl.setStyleSheet("color: #c0392b; font-size: 11px;")
        else:
            n = len(list(self._stc_cleaned_dir.glob("*.jpg"))
                    + list(self._stc_cleaned_dir.glob("*.JPG")))
            self._stc_status_lbl.setText(
                f"cleaned/  —  {n} cleaned frames (stacked exactly as StarStaX would)")
            self._stc_status_lbl.setStyleSheet("color: #2a7a2a; font-size: 11px;")

    def _browse_stc_images(self):
        """Open a folder picker for the STC dataset and update state from the choice.

        On selection, records the dataset folder, re-detects its cleaned/
        subfolder, resets the STC frame range to the full batch, and re-evaluates
        the Run button.
        """
        start = str(self._stc_images_dir) if self._stc_images_dir else str(TRAILS_ROOT)
        chosen = QFileDialog.getExistingDirectory(self, "Select dataset folder (with cleaned/ output)", start)
        if chosen:
            self._stc_images_dir = Path(chosen)
            self._stc_images_path_lbl.setText(chosen)
            self._stc_images_path_lbl.setStyleSheet(
                "font-size: 11px; font-family: monospace; color: #2a7a2a;")
            self._stc_cleaned_dir = self._detect_cleaned(self._stc_images_dir)
            self._set_stc_range_to_full()
            self._update_run_button()

    def _set_stc_range_to_full(self):
        """Default the STC frame range to the entire batch (1..N cleaned frames)."""
        n = 0
        if self._stc_cleaned_dir and self._stc_cleaned_dir.is_dir():
            n = len(list(self._stc_cleaned_dir.glob("*.jpg"))
                    + list(self._stc_cleaned_dir.glob("*.JPG")))
        n = max(1, n)
        self.stc_first_spin.setMaximum(n)
        self.stc_last_spin.setMaximum(n)
        self.stc_first_spin.setValue(1)
        self.stc_last_spin.setValue(n)
        self._on_stc_range_changed()

    def _on_stc_range_changed(self):
        """Keep the STC last-frame >= first-frame and refresh the count label.

        If "last" drops below "first" (and isn't the field being actively
        edited), it snaps back up to "first". Then updates the "N frames (a - b)"
        summary label.
        """
        if (self.stc_last_spin.value() < self.stc_first_spin.value()
                and not self.stc_last_spin.hasFocus()):
            self.stc_last_spin.blockSignals(True)
            self.stc_last_spin.setValue(self.stc_first_spin.value())
            self.stc_last_spin.blockSignals(False)
        n = max(1, self.stc_last_spin.value() - self.stc_first_spin.value() + 1)
        self.stc_range_label.setText(
            f"  {n} frame{'s' if n != 1 else ''} "
            f"({self.stc_first_spin.value()} - {self.stc_last_spin.value()})")

    # ── CVAT section handlers ──

    def _on_task_changed(self, idx):
        """React to a new CVAT task selection in the combo box.

        Queries the task's data metadata to subtract deleted frames from the
        reported size (the "effective" frame count), updates the combo label and
        the frame-range spin maxima/defaults, and shows the resolved source-image
        folder in green or a red "NOT FOUND" if it can't be located.
        """
        task = self.selected_task()
        if not task:
            return
        effective_size = task["size"]
        if self.auth and task["size"] > 0:
            try:
                meta = requests.get(
                    f"{CVAT_URL}/api/tasks/{task['id']}/data/meta",
                    auth=self.auth).json()
                deleted = meta.get("deleted_frames", [])
                effective_size = task["size"] - len(deleted)
                self.combo.setItemText(idx,
                    f"{task['id']:>3}  —  {task['name']}  ({effective_size} frames)")
            except Exception:
                pass
        self.first_spin.setMaximum(effective_size)
        self.last_spin.setMaximum(effective_size)
        self.first_spin.setValue(1)
        self.last_spin.setValue(effective_size)
        img_dir = resolve_image_dir(task["name"])
        if img_dir is None:
            self.folder_label.setText(f"image folder NOT FOUND for '{task['name']}'")
            self.folder_label.setStyleSheet(
                "color: #c0392b; font-size: 11px; font-family: monospace;")
        else:
            self.folder_label.setText(str(img_dir))
            self.folder_label.setStyleSheet(
                "color: #2a7a2a; font-size: 11px; font-family: monospace;")
        self._on_range_changed()
        self._update_run_button()

    def _on_range_changed(self):
        """Keep the CVAT last-frame >= first-frame and refresh the count label.

        Same snap-back behavior as the STC range handler, applied to the CVAT
        frame-range spin boxes and their "N frames (a - b)" summary label.
        """
        if (self.last_spin.value() < self.first_spin.value()
                and not self.last_spin.hasFocus()):
            self.last_spin.blockSignals(True)
            self.last_spin.setValue(self.first_spin.value())
            self.last_spin.blockSignals(False)
        n = max(1, self.last_spin.value() - self.first_spin.value() + 1)
        self.range_label.setText(
            f"  {n} frame{'s' if n != 1 else ''} "
            f"({self.first_spin.value()} - {self.last_spin.value()})")

    # ── Accessors ──

    def source_mode(self):
        """Return "cvat" or "stc" for the currently selected source."""
        return "cvat" if self._radio_cvat.isChecked() else "stc"

    def selected_task(self):
        """Return the chosen {id, name, size} task dict, or None if none valid."""
        idx = self.combo.currentData()
        if idx is not None and 0 <= idx < len(self.tasks):
            return self.tasks[idx]
        return None

    def selected_first_frame(self):
        """Return the CVAT-mode first frame (1-based) as an int."""
        return int(self.first_spin.value())

    def selected_last_frame(self):
        """Return the CVAT-mode last frame (1-based, inclusive) as an int."""
        return int(self.last_spin.value())

    def selected_stc_images_dir(self):
        """Return the chosen STC dataset folder Path, or None."""
        return self._stc_images_dir

    def selected_stc_cleaned_dir(self):
        """Return the auto-detected cleaned/ folder Path for STC mode, or None."""
        return self._stc_cleaned_dir  # auto-detected in _browse_stc_images

    def selected_stc_first_frame(self):
        """Return the STC-mode first frame (1-based) as an int."""
        return int(self.stc_first_spin.value())

    def selected_stc_last_frame(self):
        """Return the STC-mode last frame (1-based, inclusive) as an int."""
        return int(self.stc_last_spin.value())


# ── Tile grid ─────────────────────────────────────────────────────────────────

TILE_SIZE = 640
TILE_OVERLAP = 0.2

def compute_tile_positions(img_w, img_h, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    """Compute the top-left x and y offsets of the 640x640 detection tiles.

    Mirrors the overlapping SAHI tiling used by the detector: tiles step by
    stride = tile_size * (1 - overlap), and the final tile in each axis is
    snapped flush to the right/bottom edge so the whole image is covered.
    Returns (xs, ys) lists of pixel offsets. A dimension smaller than one tile
    yields a single offset of 0.
    """
    stride = max(1, int(tile_size * (1 - overlap)))

    def positions(dim):
        """Return the tile start offsets along one axis of length `dim`."""
        if dim <= tile_size:
            return [0]
        pos = []
        x = 0
        while x + tile_size < dim:
            pos.append(x)
            x += stride
        pos.append(dim - tile_size)
        return pos

    return positions(img_w), positions(img_h)


def draw_tile_grid(img_arr, xs, ys, tile_size=TILE_SIZE):
    """Draw the labeled tile grid over a stacked image and return the overlaid copy.

    Each tile gets a faint white rectangle (blended at 35% so the image stays
    readable) and a centered A1/B2-style label (rows = letters A, B, C...;
    columns = 1, 2, 3...) on a black background chip. The label address lets a
    trail spotted in the stack be reported by tile and opened in TileFixR. Does
    not modify the input array.
    """
    out = img_arr.copy()
    h, w = out.shape[:2]

    overlay = out.copy()
    for ty in ys:
        for tx in xs:
            x2 = min(tx + tile_size, w) - 1
            y2 = min(ty + tile_size, h) - 1
            cv2.rectangle(overlay, (tx, ty), (x2, y2), (255, 255, 255), 1)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, tile_size / 400.0)
    thickness  = max(1, int(tile_size / 320))

    for row_idx, ty in enumerate(ys):
        row_letter = chr(ord('A') + row_idx) if row_idx < 26 else '?'
        for col_idx, tx in enumerate(xs):
            label = f"{row_letter}{col_idx + 1}"
            (tw, th), bl = cv2.getTextSize(label, font, font_scale, thickness)
            cx = tx + min(tile_size, w - tx) // 2
            cy = ty + min(tile_size, h - ty) // 2
            lx = cx - tw // 2
            ly = cy + th // 2
            cv2.rectangle(out, (lx - 4, ly - th - 4), (lx + tw + 4, ly + bl + 4), (0, 0, 0), -1)
            cv2.putText(out, label, (lx, ly), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out

# ── Splash window ─────────────────────────────────────────────────────────────

class SplashWindow(QWidget):
    """A frameless progress splash shown while a stack worker runs.

    Displays the tool name, a status line, an indeterminate-or-determinate
    progress bar, and a Cancel button that quits the whole app. Driven by the
    worker's progress signal via update_progress.
    """
    def __init__(self):
        """Build the splash window layout and styling."""
        super().__init__()
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setFixedSize(420, 185)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)
        layout.setSpacing(8)

        title = QLabel("Mask CheckR")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("Building annotation stack")
        sub.setStyleSheet("font-size: 12px; color: #a8c0e0;")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        self.status_label = QLabel("Starting...")
        self.status_label.setStyleSheet("font-size: 12px; color: #e6e6e6;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #1a3a5c; border-radius: 7px; "
            "background: #0a1e3f; }"
            "QProgressBar::chunk { background: #1a6fc4; border-radius: 6px; }"
        )
        layout.addWidget(self.progress_bar)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(28)
        cancel_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #a8c0e0; font-size: 12px; "
            "border: 1px solid #1a3a5c; border-radius: 4px; padding: 0 16px; }"
            "QPushButton:hover { color: white; border-color: #4a7cbf; }"
        )
        cancel_btn.clicked.connect(lambda: QApplication.instance().quit())
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.setStyleSheet("background-color: #0a1e3f; border-radius: 12px;")

    def update_progress(self, text, current=0, total=0):
        """Update the status line and progress bar, then pump the event loop.

        With total > 0 the bar shows determinate current/total progress;
        otherwise it shows an indeterminate "busy" animation. Calls
        processEvents so the splash repaints while the worker thread runs.
        """
        self.status_label.setText(text)
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        else:
            self.progress_bar.setRange(0, 0)
        QApplication.processEvents()

# ── CVAT stack worker ─────────────────────────────────────────────────────────

class StackWorker(QThread):
    """Background thread for CVAT mode: black-fill annotations, then lighten-max stack.

    Fetches the task's reviewed polygons from CVAT for the chosen frame range,
    paints each polygon black on its source JPEG, lighten-max stacks the frames,
    and writes both a plain stack and a tile-grid-labeled stack. Communicates
    with the UI via Qt signals (progress / finished / error) so the GUI stays
    responsive.
    """
    progress = Signal(str, int, int)
    finished = Signal(str, str, int, int, int)  # out_path, tiled_path, total, annotated, polys
    error    = Signal(str)

    def __init__(self, task, first_frame, last_frame, auth, img_dir):
        """Store the task, 1-based frame range, CVAT auth, and source image folder."""
        super().__init__()
        self.task        = task
        self.first_frame = first_frame
        self.last_frame  = last_frame
        self.auth        = auth
        self.img_dir     = img_dir

    def run(self):
        """Do the CVAT-mode work on the worker thread.

        Steps: find the task's job and its frame-name list; pull all polygon/mask
        shapes within the requested frame range and group them by image stem;
        for each source JPEG, fill its polygons black and lighten-max accumulate;
        write the plain and tile-grid stacks into
        <img_dir>/cleanr_workspace/masks/mask_checkr_output/; emit finished with
        counts (frames stacked, frames annotated, polygons filled). Emits error
        on any failure or if the expected masks folder is missing.
        """
        try:
            self.progress.emit("Fetching CVAT annotations...", 0, 0)
            task_id    = self.task["id"]
            cvat_start = self.first_frame - 1
            cvat_end   = self.last_frame

            jobs_resp = requests.get(
                f"{CVAT_URL}/api/jobs",
                params={"task_id": task_id}, auth=self.auth).json()
            job_id = jobs_resp["results"][0]["id"]

            meta_resp = requests.get(
                f"{CVAT_URL}/api/jobs/{job_id}/data/meta",
                auth=self.auth).json()
            frame_names = [f["name"] for f in meta_resp["frames"]]

            ann_resp = requests.get(
                f"{CVAT_URL}/api/jobs/{job_id}/annotations",
                auth=self.auth).json()

            poly_map = {}
            for shape in ann_resp.get("shapes", []):
                if shape.get("type") not in ("polygon", "mask"):
                    continue
                fidx = shape["frame"]
                if fidx < cvat_start or fidx >= cvat_end:
                    continue
                pts_flat = shape.get("points", [])
                if len(pts_flat) < 6:
                    continue
                pts = np.array(pts_flat, dtype=np.float32).reshape(-1, 2)
                if fidx < len(frame_names):
                    stem = Path(frame_names[fidx]).stem
                    poly_map.setdefault(stem, []).append(pts)

            n_polys_total = sum(len(v) for v in poly_map.values())
            print(f"  {n_polys_total} polygons across {len(poly_map)} frames "
                  f"(task {task_id}, frames {self.first_frame}-{self.last_frame})",
                  flush=True)

            img_by_name = {p.name: p for p in (
                list(self.img_dir.glob("*.jpg")) +
                list(self.img_dir.glob("*.JPG")))}
            jpgs = [
                img_by_name[frame_names[fidx]]
                for fidx in range(cvat_start, cvat_end)
                if fidx < len(frame_names) and frame_names[fidx] in img_by_name
            ]
            total = len(jpgs)

            acc = None
            annotated_count = 0
            poly_count = 0

            for i, jpg_path in enumerate(jpgs):
                self.progress.emit(
                    f"Frame {i + 1} of {total}  —  {jpg_path.name}", i + 1, total)

                img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                polys = poly_map.get(jpg_path.stem)
                if polys:
                    h, w = img.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    for pts in polys:
                        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2).astype(np.int32)], 255)
                    img[mask == 255] = 0
                    annotated_count += 1
                    poly_count += len(polys)

                if acc is None:
                    acc = img.astype(np.float32)
                else:
                    np.maximum(acc, img, out=acc)

            if acc is None:
                self.error.emit("No frames could be loaded.")
                return

            masks_dir = self.img_dir / "cleanr_workspace" / "masks"
            if not masks_dir.is_dir():
                self.error.emit(
                    f"Masks folder not found.\n"
                    f"Expected cleanr_workspace/masks/ inside:\n{self.img_dir}"
                )
                return
            out_dir  = masks_dir / "mask_checkr_output"
            out_dir.mkdir(exist_ok=True)
            out_path = str(out_dir / f"{self.img_dir.name}_maskcheckr.jpg")
            stack_img = acc.astype(np.uint8)
            cv2.imwrite(out_path, stack_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            h, w = stack_img.shape[:2]
            xs, ys = compute_tile_positions(w, h)
            tiled_img  = draw_tile_grid(stack_img, xs, ys)
            tiled_path = str(out_dir / f"{self.img_dir.name}_maskcheckr_tiled.jpg")
            cv2.imwrite(tiled_path, tiled_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            self.finished.emit(out_path, tiled_path, total, annotated_count, poly_count)

        except Exception as e:
            self.error.emit(str(e))

# ── STC run stack worker ──────────────────────────────────────────────────────

class STCRunWorker(QThread):
    """Background thread for "Last STC run" mode: lighten-max stack the cleaned frames.

    Stacks the actual cleaned/ JPEGs (exactly what StarStaX would produce) over
    the chosen frame range, and writes both a plain stack and a tile-grid-labeled
    stack. Unlike StackWorker it applies no masks: any trail still visible is a
    real defect in the cleaned output. Communicates via Qt signals.
    """
    progress = Signal(str, int, int)
    finished = Signal(str, str, int, int)  # out_path, tiled_path, total, masked_count
    error    = Signal(str)

    def __init__(self, cleaned_dir: Path, first_frame=1, last_frame=None):
        """Store the cleaned/ folder and the 1-based inclusive frame range.

        last_frame=None means stack through the final frame.
        """
        super().__init__()
        self.cleaned_dir = cleaned_dir
        self.first_frame = int(first_frame)
        self.last_frame  = last_frame  # None = to the end

    def run(self):
        """Do the STC-mode work on the worker thread.

        Sorts the cleaned JPEGs, slices to the selected 1-based inclusive range,
        lighten-max accumulates them, and writes the plain and tile-grid stacks
        into a mask_checkr_output/ folder beside cleaned/. Emits finished with the
        frame count, or error if no cleaned files or no frames in range.
        """
        try:
            # Stack the ACTUAL cleaned frames -- exactly what StarStaX would
            # output -- not source-with-masks-blackfilled. A trail visible here
            # is a real problem in the cleaned result.
            img_files = sorted(
                list(self.cleaned_dir.glob("*.jpg")) + list(self.cleaned_dir.glob("*.JPG"))
            )
            if not img_files:
                self.error.emit(f"No cleaned JPG files found in:\n{self.cleaned_dir}")
                return
            # Apply the selected frame range (1-based, inclusive).
            lo = max(1, self.first_frame) - 1
            hi = self.last_frame if self.last_frame else len(img_files)
            img_files = img_files[lo:hi]
            total = len(img_files)
            if total == 0:
                self.error.emit("No frames in the selected range.")
                return

            acc = None
            for i, jpg_path in enumerate(img_files):
                self.progress.emit(
                    f"Stacking {i + 1} of {total}  —  {jpg_path.name}", i + 1, total)
                img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if acc is None:
                    acc = img.astype(np.float32)
                else:
                    np.maximum(acc, img, out=acc)

            if acc is None:
                self.error.emit("No frames could be loaded.")
                return

            # Write to the top-level dataset folder (alongside cleaned/), where
            # the other Mask CheckR output already lives -- easy to find.
            dataset_dir = self.cleaned_dir.parent
            out_dir = dataset_dir / "mask_checkr_output"
            out_dir.mkdir(exist_ok=True)
            stack_img = acc.astype(np.uint8)
            out_path  = str(out_dir / f"{dataset_dir.name}_stcrun.jpg")
            cv2.imwrite(out_path, stack_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            h, w = stack_img.shape[:2]
            xs, ys = compute_tile_positions(w, h)
            tiled_img  = draw_tile_grid(stack_img, xs, ys)
            tiled_path = str(out_dir / f"{dataset_dir.name}_stcrun_tiled.jpg")
            cv2.imwrite(tiled_path, tiled_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            self.finished.emit(out_path, tiled_path, total, total)

        except Exception as e:
            self.error.emit(str(e))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """App entry point: lock, connect to CVAT, show the picker, run the chosen mode.

    Acquires the single-instance lock, tries to load CVAT tasks (continuing in
    STC-only mode if CVAT is offline), shows the picker dialog, and on "Run"
    launches the matching worker (StackWorker for CVAT, STCRunWorker for STC)
    behind a splash. When the worker finishes it shows a summary message box,
    opens the output folder in Finder, and quits. Saves the selection for next
    launch.
    """
    print("Mask CheckR")
    print("=" * 60, flush=True)

    app = QApplication(sys.argv)
    _apply_theme()

    if not acquire_lock():
        QMessageBox.warning(None, "Mask CheckR", "Mask CheckR is already running.")
        return

    # Try to connect to CVAT -- not required if user picks STC mode
    auth  = None
    tasks = []
    try:
        auth = (CVAT_USER, read_cvat_password())
        print("Fetching CVAT tasks...", flush=True)
        tasks = fetch_cvat_tasks(auth)
        if tasks:
            print(f"  {len(tasks)} tasks loaded", flush=True)
        else:
            print("  No tasks returned (CVAT offline?)", flush=True)
    except Exception as e:
        print(f"  CVAT unavailable: {e}", flush=True)

    last_pick = load_last_pick()

    picker = TaskPickerDialog(tasks, last_pick, auth=auth)
    if picker.exec() != QDialog.Accepted:
        print("Cancelled.")
        return

    mode = picker.source_mode()

    def on_error(msg):
        """Worker error handler: close the splash, show the message, quit the app."""
        splash.close()
        mb = QMessageBox(QMessageBox.Critical, "Mask CheckR — Error", msg)
        mb.setTextInteractionFlags(SELECTABLE)
        mb.exec()
        QApplication.quit()

    if mode == "cvat":
        task        = picker.selected_task()
        first_frame = picker.selected_first_frame()
        last_frame  = picker.selected_last_frame()
        if task is None:
            return

        img_dir = resolve_image_dir(task["name"])
        if img_dir is None:
            QMessageBox.critical(None, "Mask CheckR",
                f"Could not find image folder for task '{task['name']}'.")
            return

        save_last_pick({
            "task_id":        task["id"],
            "first_frame":    first_frame,
            "last_frame":     last_frame,
            "source_mode":    "cvat",
            "stc_images_dir": str(picker.selected_stc_images_dir() or ""),
        })
        print(f"  mode: CVAT", flush=True)
        print(f"  task {task['id']}: {task['name']}", flush=True)
        print(f"  frames {first_frame}-{last_frame}", flush=True)
        print(f"  images: {img_dir}", flush=True)

        splash = SplashWindow()
        splash.show()
        splash.update_progress("Fetching CVAT annotations...")

        worker = StackWorker(task, first_frame, last_frame, auth, img_dir)
        worker.progress.connect(lambda msg, cur, tot: splash.update_progress(msg, cur, tot))

        def on_cvat_finished(out_path, tiled_path, total, annotated, polys):
            """CVAT worker done: show the coverage summary, open output, quit."""
            splash.close()
            mb = QMessageBox(QMessageBox.Information, "Mask CheckR",
                f"Done.\n\n"
                f"Frames stacked: {total}\n"
                f"Frames with CVAT annotations applied: {annotated}\n"
                f"Polygons black-filled: {polys}\n\n"
                f"Any trail visible in the result is a missed annotation in CVAT.\n\n"
                f"Stack:  {out_path}\n"
                f"Tiled:  {tiled_path}")
            mb.setTextInteractionFlags(SELECTABLE)
            mb.exec()
            subprocess.run(["open", str(Path(out_path).parent)])
            QApplication.quit()

        worker.finished.connect(on_cvat_finished)
        worker.error.connect(on_error)
        worker.start()

    else:  # stc mode
        img_dir     = picker.selected_stc_images_dir()
        cleaned_dir = picker.selected_stc_cleaned_dir()
        first_frame = picker.selected_stc_first_frame()
        last_frame  = picker.selected_stc_last_frame()

        save_last_pick({
            "task_id":        last_pick["task_id"],
            "first_frame":    first_frame,
            "last_frame":     last_frame,
            "source_mode":    "stc",
            "stc_images_dir": str(img_dir),
        })
        print(f"  mode: Last STC run (stacking cleaned frames)", flush=True)
        print(f"  cleaned: {cleaned_dir}", flush=True)
        print(f"  frames {first_frame}-{last_frame}", flush=True)

        splash = SplashWindow()
        splash.show()
        splash.update_progress("Stacking cleaned frames...")

        worker = STCRunWorker(cleaned_dir, first_frame, last_frame)
        worker.progress.connect(lambda msg, cur, tot: splash.update_progress(msg, cur, tot))

        def on_stc_finished(out_path, tiled_path, total, _unused):
            """STC worker done: show the cleaned-stack summary, open output, quit."""
            splash.close()
            mb = QMessageBox(QMessageBox.Information, "Mask CheckR",
                f"Done.\n\n"
                f"Cleaned frames stacked: {total}\n\n"
                f"This is exactly what StarStaX would output. Any trail visible "
                f"in the result is a real problem in the cleaned output — read its "
                f"tile (e.g. B10) and open that frame in TileFixR.\n\n"
                f"Stack:  {out_path}\n"
                f"Tiled:  {tiled_path}")
            mb.setTextInteractionFlags(SELECTABLE)
            mb.exec()
            subprocess.run(["open", str(Path(out_path).parent)])
            QApplication.quit()

        worker.finished.connect(on_stc_finished)
        worker.error.connect(on_error)
        worker.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
