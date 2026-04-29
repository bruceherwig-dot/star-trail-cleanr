"""Sentry subprocess wiring must stay intact.

The GUI process initializes Sentry only for itself. Worker subprocesses
(astro_clean_v5.py, run once per 20-frame batch) need their own init or
crashes inside processing never reach the dashboard. This file locks the
wiring on both sides:

  1. Worker initializes Sentry only when STC_SENTRY_DSN env var is set.
     Privacy: env var is the opt-in signal — if absent, the worker has
     no DSN and stays silent.

  2. GUI sets STC_SENTRY_DSN in the worker env ONLY when both
     `crash_reporting_enabled` is True AND `_SENTRY_DSN` is non-empty.
     Privacy: opt-out users never have the DSN passed to the worker.

  3. GUI captures non-zero worker exit and forwards stderr to Sentry as
     a safety net for crashes that die before the worker's own Sentry
     init runs (DLL miss, bundle import failure, etc).
"""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _read(p: str) -> str:
    return (REPO / p).read_text()


def test_worker_initializes_sentry_only_when_dsn_env_var_present():
    src = _read("astro_clean_v5.py")
    assert 'os.environ.get("STC_SENTRY_DSN"' in src, (
        "Worker must read STC_SENTRY_DSN from env to know whether to "
        "initialize Sentry. Without this gate, the worker would either "
        "always-init (privacy violation) or never-init (no reporting)."
    )
    assert "sentry_sdk.init(" in src, \
        "Worker must call sentry_sdk.init when DSN is provided."
    assert "_init_worker_sentry()" in src, (
        "Worker must actually CALL the init helper at module load — "
        "defining it without invoking it leaves Sentry silent."
    )


def test_worker_sentry_init_uses_privacy_safe_defaults():
    src = _read("astro_clean_v5.py")
    init_block_start = src.find("sentry_sdk.init(")
    assert init_block_start != -1
    init_block = src[init_block_start:init_block_start + 600]
    assert "traces_sample_rate=0" in init_block, \
        "Worker Sentry must NOT collect performance traces"
    assert "send_default_pii=False" in init_block, \
        "Worker Sentry must NOT send personally identifiable info"


def test_gui_passes_dsn_env_var_only_when_opted_in_and_dsn_baked():
    src = _read("star_trail_cleanr.py")
    assert 'worker_env["STC_SENTRY_DSN"]' in src, \
        "GUI must propagate the DSN to the worker via STC_SENTRY_DSN env var"
    assert 'crash_reporting_enabled' in src
    assert '_SENTRY_DSN' in src

    spawn_idx = src.find("self._proc = subprocess.Popen(")
    assert spawn_idx != -1
    setup_block = src[max(0, spawn_idx - 600):spawn_idx]
    assert 'worker_env["STC_SENTRY_DSN"]' in setup_block, (
        "DSN env var must be set in the same block that builds worker_env, "
        "right before the Popen call. Otherwise opted-out users could leak "
        "the DSN into the worker."
    )
    assert "crash_reporting_enabled" in setup_block, \
        "DSN must be gated on the user's opt-in setting at the spawn site"


def test_gui_passes_env_dict_to_subprocess_popen():
    src = _read("star_trail_cleanr.py")
    spawn_idx = src.find("self._proc = subprocess.Popen(")
    assert spawn_idx != -1
    popen_call = src[spawn_idx:spawn_idx + 600]
    assert "env=worker_env" in popen_call, (
        "Popen must receive env=worker_env. Without env=, the OS gives the "
        "worker a copy of the GUI's environment automatically — but our "
        "STC_SENTRY_DSN addition would not propagate."
    )


def test_gui_forwards_worker_stderr_to_sentry_on_failure():
    src = _read("star_trail_cleanr.py")
    fail_idx = src.find("if self._proc.returncode != 0:")
    assert fail_idx != -1
    fail_block = src[fail_idx:fail_idx + 1500]
    assert "sentry_sdk.capture_message" in fail_block, (
        "When the worker exits non-zero, the GUI must forward the captured "
        "stderr to Sentry. This is the only path that catches crashes that "
        "die before the worker's own Sentry init runs."
    )
    assert "stderr_full" in fail_block, \
        "GUI must attach the full worker stderr text to the Sentry event"


def test_worker_sentry_init_handles_missing_sdk_gracefully():
    """If sentry_sdk isn't bundled (dev environment, accidental build miss),
    the worker must NOT crash at import. The init must be wrapped in a
    try/except that swallows ImportError."""
    src = _read("astro_clean_v5.py")
    init_def_idx = src.find("def _init_worker_sentry():")
    assert init_def_idx != -1
    body = src[init_def_idx:init_def_idx + 1500]
    assert "try:" in body and "except" in body, (
        "_init_worker_sentry must wrap its sentry_sdk import + init in "
        "try/except so a missing SDK never breaks the worker."
    )
