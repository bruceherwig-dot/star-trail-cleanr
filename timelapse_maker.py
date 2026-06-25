"""
Timelapse Maker -- a Star Trail CleanR companion module with its OWN version.

Like the Foreground Mask editor, this is a self-contained piece that consumes
STC's output (a folder of cleaned frames) and is driven by its own window in
the app. THIS file is the render ENGINE; it runs as a subprocess because video
encoding must happen out of the app's own process (encoding in-process
truncates the file -- the same reason the share video runs separately).

Versioned independently of the app: bump TIMELAPSE_VERSION as this tool grows
(future styles like the growing star trail and comet view).

Styles
------
plain    -- one cleaned frame per movie frame (a straight timelapse).
blended  -- each movie frame is a Lighten (brightest-pixel) stack of the last
            N frames (default 3), the same blend used stacking a star-trail
            still. Smooths per-frame artifacts and flicker at the cost of the
            stars trailing slightly.

Frames always play in TRUE capture order (EXIF time, with filename fallback),
matching the cleaning pipeline, so a camera file-number rollover can't scramble
the movie.

CLI
---
    python3 timelapse_maker.py "<cleaned folder>" -o out.mp4 \
        --size 4k --fps 30 --style plain
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.io_safe import robust_imread, image_size, capture_time
from modules.frame_list import IMAGE_EXTS, order_by_capture_time, natural_key

TIMELAPSE_VERSION = "1.0"

# Target LONG-EDGE pixel counts for each preset. "full" keeps the native size.
# We never upscale: a preset bigger than the source falls back to full.
SIZE_PRESETS = {"1080p": 1920, "2k": 2560, "4k": 3840}

# Encoder settings mirror the before/after share clip (make_share_clip.py) so a
# timelapse uploaded to Facebook/YouTube survives their re-encode of near-black
# skies: constant 15 Mbps, dark-scene adaptive quantization (aq-mode=3), and fine
# grain that dithers away banding in smooth sky gradients. Because it is constant
# bitrate, output size tracks DURATION, not resolution.
VIDEO_BITRATE_BPS = 15_000_000
VIDEO_FFMPEG_PARAMS = [
    "-b:v", "15M", "-maxrate", "15M", "-bufsize", "30M",
    "-preset", "medium",
    "-pix_fmt", "yuv420p", "-profile:v", "high",
    "-x264-params", "aq-mode=3",
    "-vf", "noise=alls=6:allf=u",
]


def ordered_frames(frames_dir):
    """Every image in the folder, in true capture order (EXIF time, filename
    fallback) -- the same ordering rule the cleaning pipeline uses."""
    files = [os.path.join(frames_dir, n) for n in os.listdir(frames_dir)
             if os.path.splitext(n)[1].lower() in IMAGE_EXTS
             and os.path.isfile(os.path.join(frames_dir, n))]
    files = sorted(files, key=natural_key)
    times = {f: capture_time(f) for f in files}
    return order_by_capture_time(files, times)


def target_size(native_w, native_h, size_key):
    """Output (width, height) for a size preset, preserving aspect, never
    upscaling, and forced to even numbers (H.264 needs even dimensions)."""
    if size_key == "full" or size_key not in SIZE_PRESETS:
        return (native_w // 2 * 2, native_h // 2 * 2)
    long_edge = SIZE_PRESETS[size_key]
    longest = max(native_w, native_h)
    if long_edge >= longest:
        return (native_w // 2 * 2, native_h // 2 * 2)
    scale = long_edge / longest
    return (int(round(native_w * scale)) // 2 * 2,
            int(round(native_h * scale)) // 2 * 2)


def estimate_output_bytes(n_frames, fps, width, height):
    """Output-size estimate for the space check. The encoder is constant bitrate
    (VIDEO_BITRATE_BPS), so size tracks DURATION, not resolution: a 1080p and a 4K
    clip of the same length come out about the same size. width/height stay in the
    signature for callers but no longer change the estimate. +5% for container
    and grain overhead."""
    duration_s = max(1.0, n_frames / max(1, fps))
    return int(VIDEO_BITRATE_BPS / 8 * duration_s * 1.05)


def render(frames_dir, out_path, size_key="4k", fps=30, style="plain",
           blend_window=3, limit=0):
    """Encode the timelapse. Prints TIMELAPSE_PROGRESS lines the window reads to
    drive its progress bar. Returns 0 on success."""
    frames = ordered_frames(frames_dir)
    if limit and limit > 0:
        frames = frames[:limit]
    if len(frames) < 2:
        print("ERROR: need at least 2 frames to build a timelapse", flush=True)
        return 2

    nw, nh = image_size(frames[0])
    tw, th = target_size(nw, nh, size_key)
    print(f"Timelapse Maker v{TIMELAPSE_VERSION}: {len(frames)} frames -> "
          f"{tw}x{th} @ {fps}fps, style={style}", flush=True)

    import cv2
    import imageio
    writer = imageio.get_writer(
        out_path, format="FFMPEG", mode="I", fps=fps, codec="libx264",
        macro_block_size=None, ffmpeg_params=VIDEO_FFMPEG_PARAMS)

    buf = []
    n = len(frames)
    written = 0
    for i, f in enumerate(frames):
        img = robust_imread(f)  # BGR uint8/uint16
        if img is None:
            continue
        if img.dtype != np.uint8:
            img = np.clip(img / 256.0, 0, 255).astype(np.uint8) if img.max() > 255 else img.astype(np.uint8)
        if style == "blended":
            buf.append(img)
            if len(buf) > max(1, blend_window):
                buf.pop(0)
            frame = buf[0]
            for b in buf[1:]:
                frame = np.maximum(frame, b)
        else:
            frame = img
        if (frame.shape[1], frame.shape[0]) != (tw, th):
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        written += 1
        if i % 5 == 0 or i == n - 1:
            print(f"TIMELAPSE_PROGRESS: {i + 1}/{n}", flush=True)
    writer.close()
    if written < 2:
        print("ERROR: too few readable frames to build a timelapse", flush=True)
        return 2
    print(f"TIMELAPSE_DONE: {out_path}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=f"Timelapse Maker v{TIMELAPSE_VERSION}")
    ap.add_argument("frames_dir", help="Folder of cleaned frames")
    ap.add_argument("-o", "--out", required=True, help="Output video path (.mp4 or .mov)")
    ap.add_argument("--size", default="4k", help="full | 4k | 2k | 1080p")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--style", choices=["plain", "blended"], default="plain")
    ap.add_argument("--blend-window", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="Only render the first N frames (0 = all)")
    args = ap.parse_args()
    sys.exit(render(args.frames_dir, args.out, args.size, args.fps,
                    args.style, args.blend_window, args.limit))


if __name__ == "__main__":
    main()
