"""Teardown helper for offscreen GUI checks that build the full MainWindow.

MainWindow's constructor starts background threads (NVIDIA detection, device
pick, update checks). A test script that exits while any of them runs makes Qt
abort at teardown, and macOS shows "Python quit unexpectedly" on Bruce's
screen -- indistinguishable from the real app crashing. That happened twice
(2026-07-27, 2026-08-02) despite a written rule, so a settings hook now blocks
any offscreen MainWindow command that doesn't reference join_qt_threads.

Usage in a throwaway check:

    import qt_offscreen
    w = S.MainWindow()
    try:
        ...assertions...
    finally:
        qt_offscreen.join_qt_threads(w, app)

Prefer panel-level tests (StarTrailPanel, TimelapsePanel, SummaryPanel):
they start no threads and need none of this.
"""


def join_qt_threads(window, app=None, timeout_ms=10000):
    """Close `window` and wait for every QThread it started before returning.

    Scans the window's attributes for QThread instances, asks each to quit,
    and waits (hard cap `timeout_ms` each) so the interpreter never exits with
    a live Qt thread. Safe to call twice; swallows nothing silently -- a thread
    that refuses to stop is reported so the test fails loudly instead of
    crashing the process at exit.
    """
    from PySide6.QtCore import QThread

    stubborn = []
    try:
        window.close()
    except Exception:
        pass
    for name in dir(window):
        if not name.startswith("_"):
            continue
        try:
            obj = getattr(window, name)
        except Exception:
            continue
        if isinstance(obj, QThread) and obj.isRunning():
            obj.quit()
            if not obj.wait(timeout_ms):
                obj.terminate()
                obj.wait(2000)
                stubborn.append(name)
    if app is not None:
        app.processEvents()
    if stubborn:
        raise RuntimeError(
            f"threads had to be force-stopped: {stubborn} -- fix the test or "
            f"the thread before shipping anything that relies on this run")
