"""The in-app update checks must verify SSL against certifi's bundled CA roots.

Locks the 2026-06-20 fix: the orange update banner silently never fired on the
frozen Mac app because the GitHub check hit CERTIFICATE_VERIFY_FAILED (the frozen
app's Python could not reach a CA bundle). The fix pins certifi.where() in both
update_check and model_update, and build_helper packs certifi. If any of those
regress, the banner (and the model-update card) go dark with no visible error.
"""
import ssl
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def test_update_check_context_verifies_against_certifi():
    from modules.update_check import _verified_ssl_context
    ctx = _verified_ssl_context()
    # Must STILL verify (never disable it) and must have real CA certs loaded.
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.cert_store_stats()["x509_ca"] > 0, "no CA certs loaded (certifi missing?)"


def test_model_update_context_verifies_against_certifi():
    from modules.model_update import _verified_ssl_context
    ctx = _verified_ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.cert_store_stats()["x509_ca"] > 0, "no CA certs loaded (certifi missing?)"


def test_both_checks_pass_the_context_to_urlopen():
    uc = (REPO / "modules" / "update_check.py").read_text()
    mu = (REPO / "modules" / "model_update.py").read_text()
    for src, name in ((uc, "update_check.py"), (mu, "model_update.py")):
        assert "context=_verified_ssl_context()" in src, \
            f"{name} must pass the verified context to urlopen"


def test_certifi_is_collected_in_the_frozen_build():
    bh = (REPO / "build_helper.py").read_text()
    assert "'--collect-all', 'certifi'" in bh, \
        "build_helper must --collect-all certifi so the CA bundle ships in the frozen app"
