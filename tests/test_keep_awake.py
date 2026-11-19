"""Keep-awake helper: import health, and the contract that acquire()/release()
are exception-safe and idempotent and never leave the helper stuck 'active'.

The helper is best-effort by design (if the platform tool is missing it becomes a
no-op), so these tests assert it never RAISES and always ends inactive, not that a
real assertion was taken (that depends on the OS).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.keep_awake import KeepAwake


def test_release_without_acquire_is_safe():
    KeepAwake().release()   # must not raise


def test_acquire_then_release():
    k = KeepAwake()
    assert k.active is False
    k.acquire()             # may start a helper or be a no-op; must not raise
    k.release()
    assert k.active is False


def test_double_acquire_and_release_are_idempotent():
    k = KeepAwake()
    k.acquire()
    k.acquire()             # second acquire is a no-op
    k.release()
    k.release()             # second release is safe
    assert k.active is False
