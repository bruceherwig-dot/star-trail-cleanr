#!/usr/bin/env python3
"""Run every test_*.py in this folder, report pass/fail, exit non-zero on any failure."""
import importlib.util
import os
import re
import sys
import time
import traceback
from pathlib import Path

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_test_file(path: Path):
    mod = load_module(path)
    results = []
    for name in dir(mod):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        t0 = time.time()
        try:
            fn()
            results.append((name, True, None, time.time() - t0))
        except Exception:
            results.append((name, False, traceback.format_exc(), time.time() - t0))
    return results


def main():
    tests_dir = Path(__file__).parent
    repo_root = tests_dir.parent
    sys.path.insert(0, str(repo_root))

    test_files = sorted(tests_dir.glob("test_*.py"))
    if not test_files:
        print(f"{RED}No test_*.py files found in {tests_dir}{RESET}")
        return 1

    print(f"{BOLD}Star Trail CleanR smoke tests{RESET}")
    print(f"Running {len(test_files)} test files...\n")

    total_pass = 0
    total_fail = 0
    t_start = time.time()

    for test_file in test_files:
        print(f"  {test_file.name}")
        results = run_test_file(test_file)
        for name, ok, err, dur in results:
            if ok:
                print(f"    {GREEN}PASS{RESET}  {name}  ({dur*1000:.0f}ms)")
                total_pass += 1
            else:
                print(f"    {RED}FAIL{RESET}  {name}  ({dur*1000:.0f}ms)")
                for line in err.splitlines():
                    print(f"        {line}")
                total_fail += 1

    elapsed = time.time() - t_start
    print()
    if total_fail == 0:
        print(f"{GREEN}{BOLD}All {total_pass} tests passed{RESET}  ({elapsed:.1f}s)")
        return 0
    print(f"{RED}{BOLD}{total_fail} failed, {total_pass} passed{RESET}  ({elapsed:.1f}s)")
    return 1


_ALL_PASSED = re.compile(r"All (\d+) tests passed")
_SOME_FAILED = re.compile(r"(\d+) failed, (\d+) passed")


def verdict_from_output(text: str, child_code: int) -> int:
    """The suite's real result from what it printed. A printed failure count always fails; the
    'All N tests passed' line passes even when the process then died on the way out; with neither
    line (the suite itself crashed) the child's own exit code stands."""
    plain = re.sub(r"\033\[[0-9;]*m", "", text)
    if _SOME_FAILED.search(plain):
        return 1
    if _ALL_PASSED.search(plain):
        return 0
    return child_code or 1


def run_in_child_on_windows():
    """On Windows the process that builds the Qt windows can exit 1 during its own shutdown, after
    every test passed (the v3.01 Windows build did this on 10 of 10 runs). Run the suite in a child
    process and judge it by what it printed; this parent never loads Qt, so it exits cleanly."""
    import subprocess
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = proc.stdout.decode("utf-8", errors="replace")
    sys.stdout.write(text)
    sys.stdout.flush()
    code = verdict_from_output(text, proc.returncode)
    if proc.returncode != code:
        print(f"(child exited {proc.returncode}; its printed result says {code}, so exiting {code})")
        sys.stdout.flush()
    return code


if __name__ == "__main__":
    if sys.platform == "win32" and "--child" not in sys.argv:
        os._exit(run_in_child_on_windows())
    code = main()
    # Leave without the normal interpreter teardown. A few tests build real Qt windows, and
    # on the CI machines Python's shutdown sometimes destroys those windows in the wrong
    # order and the process dies ("QObject: shared QObject was deleted directly", exit 134
    # on Mac, 139 on Linux, 1 on Windows) AFTER every test has passed and been printed,
    # which failed the v2.99 build twice (a different machine each time). The exit code
    # below is the tests' real result.
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32":
        # On Windows even os._exit runs the DLL shutdown, and Qt's libraries can fail
        # there (the Windows build still exited 1 after all 563 tests passed, with
        # the runner already exiting 0). Ending the process directly skips that step.
        try:
            import ctypes
            ctypes.windll.kernel32.TerminateProcess(ctypes.windll.kernel32.GetCurrentProcess(), code)
        except Exception:
            pass
    os._exit(code)
