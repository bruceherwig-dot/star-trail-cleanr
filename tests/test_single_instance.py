"""A stranger holding the port must not lock the user out of the app.

Field report 2026-09-07 (Steve): "it says it is already running but I cannot seem
to use it", and reinstalling changed nothing. The old check claimed port 49173 and
read ANY bind failure as proof a second copy was running.

49173 is inside the operating system's dynamic port range (49152-65535 on Windows,
macOS and Linux), which is the pool the OS hands out at random for outgoing
connections. So any program could be given that port by chance and make Star Trail
CleanR permanently unopenable, with nothing the user could do about it.

No unit test could have caught the old code, because it is correct in isolation and
wrong only when another process exists. That is exactly what this file supplies:
a real second process holding the port.
"""
import socket
import sys
import threading
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import star_trail_cleanr as S


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_the_port_is_below_the_os_dynamic_range():
    """The OS assigns 49152 and above at random. Anything we claim in there can be
    taken from us by an unrelated program, which is the whole bug."""
    assert S.SINGLE_INSTANCE_PORT < 49152, (
        f"port {S.SINGLE_INSTANCE_PORT} is inside the dynamic range (49152-65535); "
        "the OS can hand it to any other program and lock the user out")


def test_a_stranger_on_the_port_does_not_block_startup():
    """THE regression. Something else holds the port and never answers. The app
    must start anyway, simply without the lock."""
    port = _free_port()
    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    squatter.bind(('127.0.0.1', port))
    squatter.listen(1)
    try:
        lock, already = S.acquire_single_instance(port)
        assert already is False, (
            "a program that is not Star Trail CleanR must never be reported as "
            "'already running' -- this is the bug that locked Steve out")
        assert lock is None, "we cannot hold a port someone else has"
    finally:
        squatter.close()


def test_a_silent_squatter_that_never_replies_also_does_not_block():
    """Same again, but the squatter accepts the connection and says nothing. The
    probe must time out and let the app start, not hang or refuse."""
    port = _free_port()
    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    squatter.bind(('127.0.0.1', port))
    squatter.listen(1)
    held = []

    def _accept_and_stay_silent():
        try:
            conn, _ = squatter.accept()
            held.append(conn)
        except OSError:
            pass

    t = threading.Thread(target=_accept_and_stay_silent, daemon=True)
    t.start()
    try:
        lock, already = S.acquire_single_instance(port)
        assert already is False, "silence is not proof of a second copy"
        assert lock is None
    finally:
        for c in held:
            c.close()
        squatter.close()


def test_a_real_second_copy_is_detected():
    """The feature still has to work: when the port is held by an actual Star
    Trail CleanR, a second launch must be refused."""
    port = _free_port()
    first, already = S.acquire_single_instance(port)
    assert first is not None and already is False, "the first copy should take the lock"
    try:
        second, already2 = S.acquire_single_instance(port)
        assert already2 is True, (
            "a genuine second copy must be recognised through the handshake")
        assert second is None
    finally:
        first.close()


def test_the_lock_is_released_when_the_first_copy_goes():
    """After the holder closes, the next launch takes the lock cleanly."""
    port = _free_port()
    first, _ = S.acquire_single_instance(port)
    assert first is not None
    first.close()
    second, already = S.acquire_single_instance(port)
    try:
        assert already is False and second is not None, \
            "the port must be reusable once the first copy has gone"
    finally:
        if second is not None:
            second.close()
