"""Let the AI step use several processor threads when it runs on the processor.

Measured 2026-11-17: the AI library sets "use one processor thread" the moment
it is loaded, so a run on the processor (Windows without GPU support, Linux,
Intel Macs) used a single core for the AI step. Six threads made the AI step
about 4 times faster and the whole run about 3 times faster, with the same
answers. More than six did not help.

Only used when the chosen device is the processor. Graphics cards (Nvidia, Apple)
do the heavy work on the card and are left alone.
"""
import os

MAX_AI_THREADS = 6      # measured: 6 was the best, 12 was a little slower
SPARE_CORES = 2         # left free for the window and the rest of the machine


def choose_thread_count(logical_cores=None):
    """How many threads the AI step should use on the processor."""
    if logical_cores is None:
        logical_cores = os.cpu_count() or 1
    return max(1, min(MAX_AI_THREADS, int(logical_cores) - SPARE_CORES))


def use_cores_for_cpu_run(device_used):
    """Call after the model is loaded. Returns the thread count it set, or None
    when it did nothing (not a processor run, or anything went wrong)."""
    if device_used != "cpu":
        return None
    try:
        import torch
        n = choose_thread_count()
        torch.set_num_threads(n)
        return n
    except Exception:
        return None


def restore_opencv_threads():
    """Turn OpenCV's own threading back on after the AI library has switched it off.

    Importing ultralytics runs cv2.setNumThreads(0) (ultralytics/utils/__init__.py,
    "prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)"). The
    engine uses no DataLoader, so that only slowed it: OpenCV ran on one thread with
    the other cores idle. Measured 2026-11-18 inside a real 20-frame batch, labelling
    connected components cost 120 ms a call (70 calls, 8.2 s a batch) against 4.7 ms
    with threads on, and the labels come out identical. Gaussian blur and colour
    conversion are 7 and 4 times faster too.

    Must be called AFTER the AI library is imported (the model load does that), or the
    import switches the threads off again. Returns the thread count OpenCV ends up
    with, or None if OpenCV could not be asked. Never raises.
    """
    try:
        import cv2
        n = int(cv2.getNumberOfCPUs())
        if n > 1 and cv2.getNumThreads() < n:
            cv2.setNumThreads(n)
        return int(cv2.getNumThreads())
    except Exception:
        return None
