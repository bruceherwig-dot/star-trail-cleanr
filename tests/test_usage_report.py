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
