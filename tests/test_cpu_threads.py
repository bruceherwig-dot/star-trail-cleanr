"""A run on the processor must use several threads for the AI step.

Before 2026-11-17 the AI library left it at one thread, which made processor
runs about 4 times slower in the AI step than they need to be.
"""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.cpu_threads import (MAX_AI_THREADS, choose_thread_count,   # noqa: E402
                                 use_cores_for_cpu_run)


def test_thread_counts_by_machine_size():
    assert choose_thread_count(1) == 1
    assert choose_thread_count(2) == 1
    assert choose_thread_count(4) == 2
    assert choose_thread_count(8) == 6
    assert choose_thread_count(32) == MAX_AI_THREADS


def test_graphics_runs_are_left_alone():
    assert use_cores_for_cpu_run("cuda") is None
    assert use_cores_for_cpu_run("mps") is None


def test_processor_run_sets_the_threads():
    import torch
    before = torch.get_num_threads()
    try:
        n = use_cores_for_cpu_run("cpu")
        assert n == choose_thread_count()
        assert torch.get_num_threads() == n
    finally:
        torch.set_num_threads(before)


def test_engine_calls_it_after_loading_the_model():
    src = (REPO / "astro_clean_v5.py").read_text(encoding="utf-8")
    assert "use_cores_for_cpu_run(_device_used)" in src
    assert src.index("load_model(str(args.model)") < src.index("use_cores_for_cpu_run(_device_used)")
