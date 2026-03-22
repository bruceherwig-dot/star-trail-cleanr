import sys
import os

# Windows frozen app (--windowed) has no console: sys.stdout/stderr are None.
# Must fix before anything else — both the GUI and worker subprocess need this.
if sys.platform == 'win32' and getattr(sys, 'frozen', False):
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')

# Worker mode: when the frozen app is re-invoked as a subprocess to run the algorithm.
# sys.executable in a frozen app is the app binary, not a Python interpreter —
# so we re-invoke ourselves with this flag and run the algorithm script instead of the GUI.
if len(sys.argv) > 1 and sys.argv[1] == '--cleanr-worker':
    script = sys.argv[2]
    sys.argv = [script] + sys.argv[3:]
    import runpy
    runpy.run_path(script, run_name='__main__')
    sys.exit(0)

import gradio as gr
import glob
import time
import subprocess
from collections import Counter
from PIL import Image

if getattr(sys, 'frozen', False):
    _base = sys._MEIPASS
else:
    _base = os.path.dirname(os.path.abspath(__file__))

SCRIPT = os.path.join(_base, "astro_clean_v4.py")

try:
    with open(os.path.join(_base, "version.txt")) as _f:
        VERSION = _f.read().strip()
except Exception:
    VERSION = "dev"

def default_output(folder):
    if folder and folder.strip():
        return os.path.join(folder.strip(), "cleaned")
    return gr.update()

def pick_folder():
    if sys.platform == "win32":
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        folder = filedialog.askdirectory()
        root.destroy()
        return folder or gr.update()
    else:
        result = subprocess.run(
            ["osascript", "-e", "POSIX path of (choose folder)"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return gr.update()

def fmt_hms(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"

def make_bar(pct, remaining_str, frames_done, total):
    pct = min(100, max(0, pct))
    color = "#1a6fc4"
    return f"""
    <div style="padding:12px 0">
      <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
        <span style="font-weight:600; font-size:1.05em;">{frames_done} / {total} frames ({pct:.1f}%)</span>
        <span style="font-size:1.05em; color:{color}; font-weight:600;">⏱ {remaining_str}</span>
      </div>
      <div style="background:#e0e0e0; border-radius:8px; height:28px; overflow:hidden;">
        <div style="width:{pct:.1f}%; background:{color}; height:100%; border-radius:8px;
                    transition:width 0.4s ease; display:flex; align-items:center; justify-content:center;">
          <span style="color:white; font-size:0.85em; font-weight:bold; padding:0 8px;">{pct:.0f}%</span>
        </div>
      </div>
    </div>
    """

def run_cleaner(folder, output_folder, frame_limit, progress=gr.Progress()):
    if not folder or not folder.strip():
        raise gr.Error("Please enter an input frames folder path.")
    folder = folder.strip()
    if not os.path.isdir(folder):
        raise gr.Error(f"Folder not found: {folder}")

    if not output_folder or not output_folder.strip():
        raise gr.Error("Please enter an output folder path.")
    output_folder = output_folder.strip()
    os.makedirs(output_folder, exist_ok=True)

    exts = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff",
            "*.JPG", "*.JPEG", "*.PNG", "*.TIF", "*.TIFF"]
    frames = sorted(set(f for e in exts for f in glob.glob(os.path.join(folder, e))))
    if not frames:
        raise gr.Error(f"No image files found in: {folder}")

    total = len(frames)
    if frame_limit != "All Trails":
        total = min(total, int(frame_limit))
        frames = frames[:total]

    # Majority-vote resolution: sample 10 evenly-spaced files, find dominant size,
    # then filter out any files that don't match.
    def _img_size(path):
        try:
            with Image.open(path) as img:
                return img.size  # (width, height)
        except Exception:
            return None

    sample_step = max(1, total // 10)
    sample = [frames[i] for i in range(0, total, sample_step)][:10]
    sizes = [s for s in (_img_size(f) for f in sample) if s is not None]
    if not sizes:
        raise gr.Error("Could not read any image files to determine resolution.")
    dominant = Counter(sizes).most_common(1)[0][0]
    filtered = [f for f in frames if _img_size(f) == dominant]
    skipped = total - len(filtered)
    frames = filtered
    total = len(frames)
    if not frames:
        raise gr.Error("No image files matched the dominant resolution.")

    starts = list(range(0, total - 20 + 1, 16))
    if not starts or starts[-1] + 20 < total:
        starts.append(total - 20)
    n_batches = len(starts)
    # Scale time estimate by image resolution relative to 20MP reference camera
    ref_pixels  = 5472 * 3648
    img_pixels  = dominant[0] * dominant[1]
    res_scale   = img_pixels / ref_pixels
    est_seconds = int(n_batches * 40 * res_scale)

    status_lines = [
        f"Found {total} frames to process ({dominant[0]}×{dominant[1]})"
        + (f" — skipped {skipped} file(s) with wrong resolution" if skipped else ""),
        f"Est. batches: {n_batches}  |  Est. time: {fmt_hms(est_seconds)}",
        "Starting...",
    ]
    yield "\n".join(status_lines), make_bar(0, fmt_hms(est_seconds), 0, total)

    t0 = time.time()
    frames_done = 0

    for i, start in enumerate(starts):
        elapsed = time.time() - t0
        if i > 0:
            avg = elapsed / i
            remaining = avg * (n_batches - i)
        else:
            remaining = n_batches * 40

        pct = frames_done / total * 100
        progress(pct / 100, desc=f"Batch {i+1}/{n_batches}")
        status_lines[-1] = f"Batch {i+1}/{n_batches} — ~{fmt_hms(remaining)} remaining"
        yield "\n".join(status_lines), make_bar(pct, fmt_hms(remaining), frames_done, total)

        if getattr(sys, 'frozen', False):
            cmd = [sys.executable, '--cleanr-worker', SCRIPT, folder,
                   "-o", output_folder, "--start", str(start), "--batch", "20"]
        else:
            cmd = [sys.executable, SCRIPT, folder,
                   "-o", output_folder, "--start", str(start), "--batch", "20"]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in proc.stdout:
            line = line.strip()
            if line:
                elapsed = time.time() - t0
                if i > 0:
                    avg = elapsed / i
                    remaining = avg * (n_batches - i)
                status_lines[-1] = f"Batch {i+1}/{n_batches}: {line}"
                yield "\n".join(status_lines), make_bar(pct, fmt_hms(remaining), frames_done, total)
        proc.wait()
        if proc.returncode != 0:
            raise gr.Error(f"Batch {i+1} failed — check terminal for details")

        # Each batch covers 20 frames; step=16 so last batch may overlap
        frames_done = min(total, start + 20)

    status_lines[-1] = f"Done. Processed {total} frames in {n_batches} batches."
    yield "\n".join(status_lines), make_bar(100, "Complete!", total, total)

    if sys.platform == "win32":
        os.startfile(output_folder)
    else:
        subprocess.run(["open", output_folder])


css = """
#run-btn { background: #1a6fc4; border-color: #1a6fc4; color: white; }
"""

with gr.Blocks(title=f"Star Trail CleanR (Beta {VERSION})", css=css) as demo:
    gr.Markdown(f"# **Star Trail CleanR (Beta {VERSION})**")
    gr.Markdown("#### Easily remove airplane trails from your star trail images at the touch of a button!")
    gr.Markdown("<br>")

    gr.Markdown("### **Original Star Trail Images Live Here** (.JPG, .TIF 8 & 16 bit files accepted)")
    with gr.Row():
        folder_input = gr.Textbox(label="", placeholder="Select folder using Browse…", scale=4)
        browse_in_btn = gr.Button("Browse…", scale=1)

    gr.Markdown("### **Output Folder**")
    with gr.Row():
        output_input = gr.Textbox(label="", placeholder="Select folder using Browse… (auto-fills from input folder)", scale=4)
        browse_out_btn = gr.Button("Browse…", scale=1)

    gr.Markdown("### **Number of Images to Process** <span style='font-size:0.85em; font-weight:normal'>(highly recommended to do a small batch before a full run)</span>")
    with gr.Row():
        frame_limit = gr.Dropdown(
            choices=["All Trails", "20", "50", "100", "250"],
            value="All Trails",
            label="",
            scale=1,
        )
        gr.Column(scale=3)

    gr.Markdown("<br>")
    with gr.Row():
        run_btn = gr.Button("Clean My Stars", variant="primary", size="lg", scale=0, elem_id="run-btn")

    progress_bar = gr.HTML(make_bar(0, "--", 0, 1))

    status_out = gr.Textbox(label="Status", lines=3, interactive=False)

    browse_in_btn.click(fn=pick_folder, inputs=[], outputs=folder_input).then(
        fn=default_output, inputs=folder_input, outputs=output_input
    )
    folder_input.change(fn=default_output, inputs=folder_input, outputs=output_input)
    browse_out_btn.click(fn=pick_folder, inputs=[], outputs=output_input)
    run_btn.click(fn=run_cleaner, inputs=[folder_input, output_input, frame_limit], outputs=[status_out, progress_bar])

import socket as _socket
import atexit as _atexit
import webbrowser as _webbrowser

def _free_port():
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# Single-instance guard: if a prior instance is still running, reopen its
# browser tab and exit rather than launching a second copy (which macOS blocks).
_PID_FILE = os.path.expanduser("~/.startrailcleanr.pid")

def _check_existing():
    """Return port of running instance, or None."""
    try:
        with open(_PID_FILE) as f:
            pid, port = f.read().strip().split(":")
            pid, port = int(pid), int(port)
        os.kill(pid, 0)   # raises if process is gone
        return port
    except Exception:
        try:
            os.remove(_PID_FILE)
        except Exception:
            pass
        return None

def _write_pid(port):
    with open(_PID_FILE, "w") as f:
        f.write(f"{os.getpid()}:{port}")

def _cleanup_pid():
    try:
        os.remove(_PID_FILE)
    except Exception:
        pass

existing_port = _check_existing()
if existing_port:
    _webbrowser.open(f"http://localhost:{existing_port}")
    # Sleep before exiting so macOS doesn't show "not open anymore" —
    # the dialog appears when an app quits within ~1s of launch.
    time.sleep(2)
    sys.exit(0)

port = _free_port()
_write_pid(port)
_atexit.register(_cleanup_pid)
demo.launch(inbrowser=True, server_port=port)
