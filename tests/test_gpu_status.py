"""Smoke tests for the "why are we on the CPU?" verdict (modules/gpu_pack.py).

These lock the class of bug that cost a user weeks of slow runs: a Windows
machine with a working NVIDIA card quietly fell back to the processor and NOTHING
in the app said so. The one thing that must never regress is that every
CPU-with-a-card case produces a status code AND a plain-English sentence for the
run log. All offline (no network, no GPU required).
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _clear_flags():
    """Drop the two environment flags the runtime hook and CUDA probe set, so a
    test never inherits a verdict from the machine it runs on."""
    for k in ("STC_GPU_VERSION_MISMATCH", "STC_CUDA_UNSUPPORTED"):
        os.environ.pop(k, None)


class _on_platform:
    """Pin sys.platform for the length of a with-block. The GPU pack exists only
    on Windows, so every test about pack codes must say which computer it is
    pretending to be: otherwise it passes or fails depending on whether it runs on
    the Mac, the Linux build machine or the Windows one."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self._real = sys.platform
        sys.platform = self.name

    def __exit__(self, *exc):
        sys.platform = self._real


def test_gpu_status_exports():
    import modules.gpu_pack as g
    for name in ("gpu_status", "status_message", "run_note", "GPU_STATUS_CODES"):
        assert hasattr(g, name), f"missing {name}"


def test_gpu_in_use_wins():
    """An active GPU is reported as such regardless of anything else."""
    from modules.gpu_pack import gpu_status
    _clear_flags()
    assert gpu_status("cuda", "yes") == "gpu_nvidia"
    assert gpu_status("mps", None) == "gpu_apple"
    # Even with a stale mismatch flag set, an active card still wins.
    os.environ["STC_GPU_VERSION_MISMATCH"] = "1"
    try:
        assert gpu_status("cuda", "yes") == "gpu_nvidia"
    finally:
        _clear_flags()


def test_no_card_says_so():
    """No usable NVIDIA card means there is nothing to fix, so neither of these
    codes may produce a run-log nag."""
    from modules.gpu_pack import gpu_status, run_note
    _clear_flags()
    code = gpu_status("cpu", "no_driver_or_card")
    assert code in ("cpu_no_card", "cpu_only")
    assert run_note(code) == ""


def test_card_present_but_on_cpu_always_explains_itself():
    """THE regression net. A machine with a working card that lands on the
    processor must always come back with a specific cause, a Settings line and a
    run-log sentence naming the fix."""
    from modules.gpu_pack import gpu_status, status_message, run_note
    _clear_flags()

    # The pack codes are Windows codes: pin the platform (see _on_platform).
    with _on_platform("win32"):
        os.environ["STC_GPU_VERSION_MISMATCH"] = "1"
        try:
            assert gpu_status("cpu", "yes") == "cpu_pack_mismatch"
        finally:
            _clear_flags()

        os.environ["STC_CUDA_UNSUPPORTED"] = "1"
        try:
            assert gpu_status("cpu", "yes") == "cpu_card_unsupported"
        finally:
            _clear_flags()

        # With no flags set the answer depends on whether a pack is on disk; both
        # outcomes are valid, and both must explain themselves.
        code = gpu_status("cpu", "yes")
        assert code in ("cpu_pack_missing", "cpu_pack_unused")

    for c in ("cpu_pack_mismatch", "cpu_card_unsupported",
              "cpu_pack_missing", "cpu_pack_unused", "cpu_os_unsupported"):
        assert status_message(c), f"{c} has no Settings line"
        assert run_note(c), f"{c} says nothing in the run log"


def test_every_code_has_a_settings_line():
    from modules.gpu_pack import GPU_STATUS_CODES, status_message
    for c in GPU_STATUS_CODES:
        assert status_message(c), f"{c} has no Settings line"


def test_run_note_is_written_for_photographers():
    """The run log goes to users, so its sentences carry no jargon. The Settings
    line may still name the pack; this is only about the run-log wording."""
    from modules.gpu_pack import GPU_STATUS_CODES, run_note
    banned = ("cuda", "mps", "torch", "wheel", "sys.path")
    for c in GPU_STATUS_CODES:
        note = run_note(c).lower()
        for word in banned:
            assert word not in note, f"{c} run note leaks jargon: {word}"


def test_unknown_code_is_harmless():
    """A future code must never blank the Settings tab or raise."""
    from modules.gpu_pack import (status_message, run_note, header_badge,
                                  summary_line)
    assert status_message("something_new") == ""
    assert run_note("something_new") == ""
    assert header_badge("something_new") == ("", "neutral")
    assert summary_line("something_new") == ""


def test_a_working_gpu_says_so_without_being_asked():
    """The gap a user reported: the app only ever spoke up when the graphics
    card was NOT being used, so there was no way to confirm before starting an
    hours-long run, or afterwards, that it had been used."""
    from modules.gpu_pack import header_badge, summary_line, run_note
    for code in ("gpu_nvidia", "gpu_apple"):
        text, tone = header_badge(code)
        assert text and tone == "ok", f"{code} must show a positive badge"
        assert summary_line(code), f"{code} must confirm on the run summary"
        # Nothing is wrong, so the run log stays quiet -- that part is unchanged.
        assert run_note(code) == ""


def test_an_unused_card_is_flagged_and_actionable():
    """A card going unused must read as fixable, not as a plain fact."""
    from modules.gpu_pack import header_badge, summary_line
    for code in ("cpu_pack_missing", "cpu_pack_mismatch", "cpu_pack_unused"):
        text, tone = header_badge(code)
        assert text and tone == "warn", f"{code} must warn in the header"
        assert "Settings" in summary_line(code), (
            f"{code} must point at where the fix lives")


def test_no_card_is_stated_not_scolded():
    """With no graphics card there is nothing to fix, so the header states it
    plainly and the run summary says nothing at all."""
    from modules.gpu_pack import header_badge, summary_line
    for code in ("cpu_no_card", "cpu_only", "cpu_card_unsupported",
                 "cpu_os_unsupported"):
        text, tone = header_badge(code)
        assert tone == "neutral", f"{code} must not read as a warning"
        assert text, f"{code} should still state what is doing the work"
        assert summary_line(code) == "", (
            f"{code} must not nag on the run summary")


def test_header_badge_stays_short():
    """It sits beside the version in a tight header. 'GPU: Apple Silicon' is the
    longest we allow; anything past that starts crowding the tab strip."""
    from modules.gpu_pack import GPU_STATUS_CODES, header_badge
    for code in GPU_STATUS_CODES:
        text, _ = header_badge(code)
        assert len(text) <= 18, f"{code} badge too long for the header: {text!r}"


def test_apple_hardware_is_named_only_when_it_is_accurate():
    """Apple's PyTorch requirements are "Apple silicon OR AMD GPUs", so a Mac
    using its GPU is not necessarily an M-series machine. Naming an Intel Mac
    with an AMD card "Apple Silicon" would be visibly wrong to its owner, so
    the specific name appears only on arm64 and the general wording otherwise.
    """
    import modules.gpu_pack as g
    real = g.is_apple_silicon
    try:
        g.is_apple_silicon = lambda: True
        assert g.header_badge("gpu_apple") == ("GPU: Apple Silicon", "ok")
        assert "Apple Silicon" in g.status_message("gpu_apple")
        assert "Apple Silicon" in g.summary_line("gpu_apple")

        g.is_apple_silicon = lambda: False
        text, tone = g.header_badge("gpu_apple")
        assert tone == "ok" and "Silicon" not in text, text
        assert "Silicon" not in g.status_message("gpu_apple")
        assert "Silicon" not in g.summary_line("gpu_apple")
        # The fallback must still say something on all three surfaces.
        assert text and g.status_message("gpu_apple") and g.summary_line("gpu_apple")
    finally:
        g.is_apple_silicon = real


def test_apple_silicon_check_is_specific_to_m_series():
    """It must key off the processor, not merely off being a Mac."""
    import platform
    from modules.gpu_pack import is_apple_silicon
    expected = sys.platform == "darwin" and platform.machine() == "arm64"
    assert is_apple_silicon() is expected


def test_the_engine_announces_the_device_it_really_used():
    """The app and the engine are separate processes, and only one of them knows.

    Field case, 2026-08-23: a user with an RTX 3080 Ti sat through a nine-hour
    run unable to tell whether his card was doing anything, because Windows Task
    Manager does not show this kind of GPU work in its default view. Meanwhile
    the app's badge and the anonymous usage report were BOTH filled in by asking
    the app's own process what device it could reach -- not the engine that
    actually loads the model, which silently drops to the processor if the model
    will not load on the card. A run could therefore report "GPU" while every
    frame was cleaned on the processor.
    """
    from pathlib import Path
    repo = Path(__file__).parent.parent

    engine = (repo / "astro_clean_v5.py").read_text(encoding="utf-8")
    assert "STC_DEVICE:" in engine, \
        "the engine must report the device it actually loaded the model on"
    assert "Using your NVIDIA graphics card" in engine, \
        "the engine must confirm success in plain words, not only warn on failure"

    detect = (repo / "modules" / "detect_trails.py").read_text(encoding="utf-8")
    assert "return model, device" in detect, \
        "load_model must hand back the device it ended up on, including after a " \
        "silent fallback to the processor"

    app = (repo / "star_trail_cleanr.py").read_text(encoding="utf-8")
    assert "_worker_device" in app, \
        "the app must record what the engine reported"
    assert "STC_DEVICE:" in app, \
        "the app must read the engine's device line"


def test_the_usage_report_states_the_engines_device_not_a_guess():
    """Otherwise the numbers we make decisions from are about the wrong process."""
    from pathlib import Path
    app = (Path(__file__).parent.parent / "star_trail_cleanr.py").read_text(encoding="utf-8")
    i = app.find('_ugpu = ')
    assert i > 0, "the usage report's gpu field vanished"
    window = app[i - 400:i + 400]
    assert "_worker_device" in window, (
        "the gpu field must come from the engine's reported device, falling back "
        "to a local guess only when no batch ever reported one")


def test_the_header_badge_is_corrected_by_the_engine():
    """The thing the user is actually looking at must not keep a stale answer.

    The badge beside the version is decided once at launch, in the app's own
    process, by asking what device IT could reach. The cleaning runs in a
    separate process that picks its own and silently falls back to the processor
    if the model will not load on the card. Kari Tuomi (2026-08-23) watched a
    nine-hour run on an RTX 3080 Ti with no way to tell whether the card was
    doing anything; a green badge sourced from a guess is worse than no badge,
    because he would have believed it.
    """
    from pathlib import Path
    app = (Path(__file__).parent.parent / "star_trail_cleanr.py").read_text(encoding="utf-8")

    assert "device_in_use = Signal(str)" in app, \
        "the worker must be able to tell the window which device it really used"
    assert "self.worker.device_in_use.connect" in app, \
        "the window must listen for it"

    i = app.find("def _on_device_in_use")
    assert i > 0, "the correction handler is gone"
    body = app[i:i + 1600]
    assert "_refresh_compute_section" in body, \
        "the correction must actually refresh the badge, not just record a value"
    assert "_compute_device" in body, \
        "the engine's answer must replace the startup guess"


# ── Linux: an NVIDIA card, but no GPU pack exists for this operating system ──

def test_linux_with_an_nvidia_card_is_told_the_truth_not_to_install_a_pack():
    """Field case (2026-10-05, Rainer Herkenrath, RTX 5060, Tuxedo OS / Linux Mint):
    the Linux build ships a CPU-only PyTorch and the GPU pack is Windows-only, yet
    the app offered "Install GPU support" on Linux. He installed it; it downloaded
    Windows files into a folder nothing reads; nothing changed. Everywhere the app
    talks about the GPU, Linux must now say there is nothing to install."""
    from modules.gpu_pack import (gpu_status, status_message, run_note, header_badge,
                                  summary_line)
    _clear_flags()
    with _on_platform("linux"):
        code = gpu_status("cpu", "yes")
        # no flag and no stray pack on disk may change the answer on Linux
        os.environ["STC_GPU_VERSION_MISMATCH"] = "1"
        try:
            assert gpu_status("cpu", "yes") == "cpu_os_unsupported"
        finally:
            _clear_flags()
    assert code == "cpu_os_unsupported"
    for text in (status_message(code), run_note(code)):
        low = text.lower()
        assert text, "it must still say something"
        assert "settings" not in low and "reinstall" not in low, text
        assert "install gpu" not in low and "install the gpu" not in low, text
        assert "nothing to install" in low or "not available" in low, text
    text, tone = header_badge(code)
    assert tone == "neutral", "it must not read as a fixable warning in the header"
    assert summary_line(code) == "", "it must not nag on the finished-run summary"


def test_linux_and_mac_never_see_a_gpu_offer_but_windows_still_does():
    """The three codes that make the app offer or point at a pack must be reachable
    on Windows only. This is the guard on the banner: it appears only for them."""
    from modules.gpu_pack import gpu_status
    _clear_flags()
    pack_codes = ("cpu_pack_missing", "cpu_pack_mismatch", "cpu_pack_unused")
    for name in ("linux", "darwin", "linux2"):
        with _on_platform(name):
            assert gpu_status("cpu", "yes") not in pack_codes, name
    with _on_platform("win32"):
        assert gpu_status("cpu", "yes") in pack_codes, "Windows must be unchanged"
    # and a working GPU, on any platform, is still reported as working
    with _on_platform("linux"):
        assert gpu_status("cuda", "yes") == "gpu_nvidia"


def test_the_banner_only_ever_appears_for_the_pack_codes():
    """The orange 'NVIDIA GPU detected. Install' banner is shown by
    _maybe_show_nvidia_banner, which has no platform check of its own: it relies on
    the status code. If a new code were ever added to its list the Linux fix would
    silently break, so lock its list to exactly the three Windows pack codes."""
    app = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    i = app.find("def _maybe_show_nvidia_banner")
    assert i > 0, "the banner function is gone"
    body = app[i:i + 2200]
    assert 'code == "cpu_pack_missing"' in body
    assert 'code in ("cpu_pack_mismatch", "cpu_pack_unused")' in body
    assert "cpu_os_unsupported" not in body, \
        "the banner must never be shown for the unsupported-OS code"

