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
              "cpu_pack_missing", "cpu_pack_unused"):
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
    from modules.gpu_pack import status_message, run_note
    assert status_message("something_new") == ""
    assert run_note("something_new") == ""
