#!/usr/bin/env python3
"""Run every test_*.py in this folder, report pass/fail, exit non-zero on any failure."""
import importlib.util
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


if __name__ == "__main__":
    sys.exit(main())
