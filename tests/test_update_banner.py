"""Regression guard for the in-app update banner (the Option B notification).

The amber banner is the PRIMARY, can't-hide-behind-the-window update notice. If it
ever stops showing -- e.g. someone re-adds the old "the engine owns the
notification" suppression -- users silently stop being told about updates. That
class of failure has bitten this project before, so these tests lock the behavior:

  * the banner shows whenever a newer release exists,
  * it shows EVEN when the one-click engine is alive AND the app is frozen (the
    exact condition the removed suppression keyed on -- this is the guard that
    fails if that suppression comes back),
  * a dismissed tag still hides it,
  * a remembered (sticky) update shows instantly with no live check, so a slow or
    failed GitHub check on a given launch can never blank a known update.

Lightweight on purpose: it binds the real MainWindow banner methods onto a tiny
stub carrying only the widgets/settings they touch -- no QApplication, no network,
no heavy window construction -- so it runs in milliseconds inside run_all.py.
"""
import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class _FakeWidget:
    def __init__(self):
        self._hidden = True

    def setVisible(self, v):
        self._hidden = not bool(v)

    def isHidden(self):
        return self._hidden


class _FakeLabel:
    def __init__(self):
        self._text = ""

    def setText(self, t):
        self._text = t

    def text(self):
        return self._text


class _FakeSettings:
    """Stand-in for the global QSettings, isolated per test (no real plist writes)."""
    def __init__(self, initial=None):
        self._d = dict(initial or {})

    def value(self, key, default="", type=str):
        return self._d.get(key, default)

    def setValue(self, key, val):
        self._d[key] = val


def _load():
    """Import the app module; return (stc, True) or (None, False) if Qt is absent
    (so the suite skips rather than errors on a headless box without PySide6)."""
    try:
        sys.argv = ["test"]
        import star_trail_cleanr as stc
        return stc, True
    except Exception:
        return None, False


def _target(stc, settings):
    """A bare object carrying just what the banner methods touch, with the real
    MainWindow methods bound to it. Swaps in an isolated settings store."""
    stc.SETTINGS = settings
    t = types.SimpleNamespace(
        _update_banner=_FakeWidget(),
        _update_label=_FakeLabel(),
        _update_banner_tag="",
        _update_download_url=None,
    )
    for name in ("_reveal_update_banner", "_on_update_result", "_show_cached_update_banner"):
        setattr(t, name, types.MethodType(getattr(stc.MainWindow, name), t))
    return t


def test_banner_shows_on_update():
    stc, ok = _load()
    if not ok:
        return  # Qt unavailable -> skip
    orig = stc.SETTINGS
    try:
        t = _target(stc, _FakeSettings())
        assert t._update_banner.isHidden()
        t._on_update_result({"tag": "v99.0-beta", "download_url": "x"})
        assert not t._update_banner.isHidden(), "banner must show when an update exists"
        assert "v99.0-beta" in t._update_label.text()
        assert stc.SETTINGS.value("last_seen_update_tag") == "v99.0-beta", \
            "the found tag must be remembered (sticky)"
    finally:
        stc.SETTINGS = orig


def test_banner_not_suppressed_when_engine_alive_and_frozen():
    """The exact Option B regression: re-adding the old suppression would hide the
    banner here. Simulate frozen + engine-alive and require it STILL shows."""
    stc, ok = _load()
    if not ok:
        return
    orig = stc.SETTINGS
    had_frozen, old_frozen = hasattr(sys, "frozen"), getattr(sys, "frozen", None)
    import modules.sparkle_updater as su
    old_alive = su.updater_alive
    sys.frozen = True
    su.updater_alive = lambda: True
    try:
        t = _target(stc, _FakeSettings())
        t._on_update_result({"tag": "v99.0-beta", "download_url": "x"})
        assert not t._update_banner.isHidden(), \
            "banner must show even when the engine is alive AND frozen (no suppression)"
    finally:
        su.updater_alive = old_alive
        if had_frozen:
            sys.frozen = old_frozen
        else:
            try:
                delattr(sys, "frozen")
            except AttributeError:
                pass
        stc.SETTINGS = orig


def test_dismissed_tag_hides_banner():
    stc, ok = _load()
    if not ok:
        return
    orig = stc.SETTINGS
    try:
        t = _target(stc, _FakeSettings({"dismissed_update_tag": "v99.0-beta"}))
        t._on_update_result({"tag": "v99.0-beta", "download_url": "x"})
        assert t._update_banner.isHidden(), "a dismissed tag must keep the banner hidden"
    finally:
        stc.SETTINGS = orig


def test_sticky_banner_shows_remembered_update():
    """A remembered newer version must show with NO live check -- this is what
    keeps a transient timeout from silently blanking a known update."""
    stc, ok = _load()
    if not ok:
        return
    orig = stc.SETTINGS
    try:
        t = _target(stc, _FakeSettings({"last_seen_update_tag": "v99.0-beta"}))
        shown = t._show_cached_update_banner()
        assert shown and not t._update_banner.isHidden(), \
            "a remembered update must show instantly (survives a failed live check)"
        assert "v99.0-beta" in t._update_label.text()
    finally:
        stc.SETTINGS = orig
