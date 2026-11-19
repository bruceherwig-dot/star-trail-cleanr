"""The test runner's Windows verdict: judged by what the suite printed, not by a shutdown exit code."""
import importlib.util
from pathlib import Path


def _runner():
    p = Path(__file__).parent / "run_all.py"
    spec = importlib.util.spec_from_file_location("run_all_for_verdict", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_all_passed_with_bad_exit_code_is_pass():
    assert _runner().verdict_from_output("x\n\033[92m\033[1mAll 646 tests passed\033[0m  (9.0s)\n", 1) == 0


def test_printed_failure_always_fails():
    assert _runner().verdict_from_output("\033[91m2 failed, 644 passed\033[0m\n", 0) == 1


def test_failure_wins_over_pass_line():
    assert _runner().verdict_from_output("All 5 tests passed\n3 failed, 2 passed\n", 0) == 1


def test_no_verdict_line_means_failure():
    r = _runner()
    assert r.verdict_from_output("Traceback ...\n", 0) == 1
    assert r.verdict_from_output("Traceback ...\n", 3) == 3
