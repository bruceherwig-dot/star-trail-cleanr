"""Tests for the anonymous usage-report module (ET Phones Home, opt-in send)."""
import os
import tempfile
import threading

from modules import usage_report


def test_install_id_stable_and_created():
    with tempfile.TemporaryDirectory() as d:
        orig = usage_report._INSTALL_ID_FILE
        usage_report._INSTALL_ID_FILE = os.path.join(d, "install_id.txt")
        try:
            a = usage_report.get_install_id()
            b = usage_report.get_install_id()
            assert a and a == b, "install id must be stable across calls"
            assert os.path.exists(usage_report._INSTALL_ID_FILE), "install id file must be created"
            assert len(a) >= 16, "install id should be a long random value"
        finally:
            usage_report._INSTALL_ID_FILE = orig


def test_is_dev_true_from_source():
    # The suite runs from source (not a frozen build), so dev must be True --
    # this is what keeps our own runs out of the published stats.
    assert usage_report._is_dev() is True


def test_build_payload_has_identity_and_facts():
    with tempfile.TemporaryDirectory() as d:
        orig = usage_report._INSTALL_ID_FILE
        usage_report._INSTALL_ID_FILE = os.path.join(d, "id.txt")
        try:
            p = usage_report.build_payload({"trails": 5, "frames": 20})
            assert p["trails"] == 5 and p["frames"] == 20
            assert p["schema"] == usage_report.SCHEMA_VERSION
            assert p["install_id"]
            assert p["dev"] is True
        finally:
            usage_report._INSTALL_ID_FILE = orig


def test_send_no_secret_is_silent_noop():
    # Without a configured secret, send() must do nothing and never raise.
    orig = usage_report._get_secret
    usage_report._get_secret = lambda: ""
    try:
        usage_report.send({"trails": 1})  # must not raise
    finally:
        usage_report._get_secret = orig


def test_send_with_secret_dispatches_payload():
    # With a secret, send() builds a payload and hands it to _post. Stub the
    # network so nothing touches the wire, and run the thread synchronously.
    captured = {}
    orig_secret = usage_report._get_secret
    orig_post = usage_report._post
    orig_thread = threading.Thread

    class _SyncThread:
        def __init__(self, target=None, args=(), daemon=None):
            self._t, self._a = target, args

        def start(self):
            self._t(*self._a)

    usage_report._get_secret = lambda: "TESTSECRET"
    usage_report._post = lambda secret, payload: captured.update(secret=secret, payload=payload)
    threading.Thread = _SyncThread
    try:
        usage_report.send({"trails": 3})
        assert captured.get("secret") == "TESTSECRET"
        assert captured["payload"]["trails"] == 3
        assert captured["payload"]["install_id"]
        assert captured["payload"]["schema"] == usage_report.SCHEMA_VERSION
        assert captured["payload"]["dev"] is True
    finally:
        usage_report._get_secret = orig_secret
        usage_report._post = orig_post
        threading.Thread = orig_thread


def _isolate(d):
    """Point every file usage_report touches at a temp dir, and give the caller
    back a restore function. The module writes into the user's own app folder,
    which a test must never touch."""
    saved = (usage_report._APP_DIR, usage_report._INSTALL_ID_FILE,
             usage_report._LAST_VERSION_FILE, usage_report._UPDATER_MARKER)
    usage_report._APP_DIR = d
    usage_report._INSTALL_ID_FILE = os.path.join(d, "id.txt")
    usage_report._LAST_VERSION_FILE = os.path.join(d, "last_version.txt")
    usage_report._UPDATER_MARKER = os.path.join(d, "updater_engaged.txt")

    def restore():
        (usage_report._APP_DIR, usage_report._INSTALL_ID_FILE,
         usage_report._LAST_VERSION_FILE, usage_report._UPDATER_MARKER) = saved
    return restore


def test_a_fresh_install_reports_no_previous_version():
    """A first run is not an upgrade and must not be counted as one."""
    with tempfile.TemporaryDirectory() as d:
        restore = _isolate(d)
        try:
            p = usage_report.build_payload({"app_version": "2.85"})
            assert "previous_version" not in p
            assert "updated_via" not in p
        finally:
            restore()


def test_an_ordinary_run_reports_no_previous_version():
    """The fields appear on the first run after a change, not on every run."""
    with tempfile.TemporaryDirectory() as d:
        restore = _isolate(d)
        try:
            usage_report.build_payload({"app_version": "2.85"})
            p = usage_report.build_payload({"app_version": "2.85"})
            assert "previous_version" not in p
        finally:
            restore()


def test_a_hand_downloaded_upgrade_is_reported_as_manual():
    with tempfile.TemporaryDirectory() as d:
        restore = _isolate(d)
        try:
            usage_report.build_payload({"app_version": "2.85"})
            p = usage_report.build_payload({"app_version": "2.86"})
            assert p["previous_version"] == "2.85"
            assert p["updated_via"] == "manual"
        finally:
            restore()


def test_a_one_click_update_is_reported_as_in_app():
    """THE question this exists to answer: did the in-app updater actually work?

    As of 2026-08-18 no Windows machine had ever been observed updating in
    place, and the data could not say whether the updater was broken or whether
    people simply re-download. This makes the two distinguishable.
    """
    with tempfile.TemporaryDirectory() as d:
        restore = _isolate(d)
        try:
            usage_report.build_payload({"app_version": "2.85"})
            usage_report.note_updater_engaged()          # user clicked update
            p = usage_report.build_payload({"app_version": "2.86"})
            assert p["previous_version"] == "2.85"
            assert p["updated_via"] == "in_app"
            # The marker is spent: the NEXT upgrade must not inherit it.
            p2 = usage_report.build_payload({"app_version": "2.87"})
            assert p2["updated_via"] == "manual", \
                "the updater marker must be cleared once it has been reported"
        finally:
            restore()


def test_telemetry_cannot_break_an_update():
    """note_updater_engaged is called from inside the updater path, so it must
    never raise, even when the folder cannot be written."""
    restore = _isolate("/nonexistent-path-that-cannot-be-created\x00/x")
    try:
        usage_report.note_updater_engaged()      # must not raise
    finally:
        restore()
