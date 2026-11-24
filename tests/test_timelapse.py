"""Timelapse Maker: output-size math, estimate, and version (pure, no ffmpeg)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from timelapse_maker import (target_size, estimate_output_bytes,
                             SIZE_PRESETS, TIMELAPSE_VERSION)


def test_full_keeps_native_and_forces_even():
    assert target_size(6000, 4000, "full") == (6000, 4000)
    assert target_size(6001, 4001, "full") == (6000, 4000)  # odd -> even


def test_4k_downscales_preserving_3to2_aspect():
    w, h = target_size(6000, 4000, "4k")
    assert (w, h) == (3840, 2560)  # long edge 3840, 3:2 preserved
    assert w % 2 == 0 and h % 2 == 0


def test_presets_long_edge():
    assert target_size(6000, 4000, "1080p")[0] == 1920
    assert target_size(6000, 4000, "2k")[0] == 2560
    assert set(SIZE_PRESETS) == {"1080p", "2k", "4k"}


def test_never_upscales():
    assert target_size(1920, 1280, "4k") == (1920, 1280)


def test_estimate_scales_with_duration():
    short = estimate_output_bytes(100, 30, 1920, 1080)
    long = estimate_output_bytes(900, 30, 1920, 1080)
    assert long > short > 0


def test_estimate_is_resolution_independent():
    # Constant-bitrate encode: size depends on length, not resolution.
    hd = estimate_output_bytes(300, 30, 1920, 1080)
    uhd = estimate_output_bytes(300, 30, 3840, 2160)
    assert hd == uhd


def test_version_is_current():
    # Bumped to 1.1 when the Blend pulldown was added (2026-07-03).
    # Bumped to 1.2 with the encoder fallback (2026-07-20): a bundle missing
    # imageio-ffmpeg now degrades to the OpenCV writer instead of crashing.
    assert TIMELAPSE_VERSION == "1.2"


def test_quality_levels_and_default():
    # Quality menu (2026-11-24): Standard is the default and matches the
    # module-level settings; each higher level allows more bits.
    from timelapse_maker import (QUALITY_PRESETS, DEFAULT_QUALITY, video_params,
                                 VIDEO_FFMPEG_PARAMS)
    assert DEFAULT_QUALITY == "standard"
    assert list(QUALITY_PRESETS) == ["standard", "high", "maximum"]
    assert VIDEO_FFMPEG_PARAMS == video_params("standard")
    caps = [QUALITY_PRESETS[k][1] for k in QUALITY_PRESETS]
    crfs = [QUALITY_PRESETS[k][0] for k in QUALITY_PRESETS]
    assert caps == sorted(caps) and crfs == sorted(crfs, reverse=True)
    p = video_params("maximum")
    assert p[p.index("-maxrate") + 1] == "100M" and "-crf" in p
    assert "noise" not in " ".join(p)          # the grain filter is gone
    assert video_params("nonsense") == video_params("standard")


def test_estimate_follows_quality_cap():
    std = estimate_output_bytes(240, 24, 3840, 2560)
    mx = estimate_output_bytes(240, 24, 3840, 2560, "maximum")
    assert std == estimate_output_bytes(240, 24, 3840, 2560, "standard")
    assert mx > 3 * std


def test_gui_passes_quality_to_the_encoder():
    # The Timelapse tab must hand the chosen level to the encoder and size check.
    src = (Path(__file__).parent.parent / "star_trail_cleanr.py").read_text(encoding="utf-8")
    assert '"--quality", quality' in src
    assert 'self._quality_cb.currentData() or "standard"' in src
    for label in ("Standard (about 30 Mbps)", "High (about 60 Mbps)",
                  "Maximum (about 100 Mbps)"):
        assert label in src
