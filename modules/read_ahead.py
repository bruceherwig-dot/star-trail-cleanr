"""Read frames a few steps ahead of the code that uses them.

WHY (2026-11-19). On 200 real 20 MP frames the hot-pixel step took 95 s, and the step
timers showed most of that was waiting for the disk to hand over frames one at a time:
the ground check spent 22 s on work that takes 4 s once the frames are already in
memory. Reading is mostly decoding a JPEG, which several threads can do at once while
the main thread works on the frame it already has.

The frames come back in EXACTLY the order they were asked for and each is read exactly
once, so whatever the caller computes is unchanged. Only the waiting changes. At most
`ahead` frames are in flight or waiting at a time, which bounds the memory.
"""
from concurrent.futures import ThreadPoolExecutor

DEFAULT_THREADS = 4


def read_ahead(read, items, threads=DEFAULT_THREADS, ahead=None):
    """Yield (item, read(item)) for each item, in order, reading several at once."""
    items = list(items)
    threads = max(1, int(threads))
    ahead = max(1, int(ahead if ahead is not None else 2 * threads))
    pool = ThreadPoolExecutor(max_workers=threads)
    futs = {}
    try:
        for pos, item in enumerate(items):
            for j in range(pos, min(len(items), pos + ahead)):
                if j not in futs:
                    futs[j] = pool.submit(read, items[j])
            yield item, futs.pop(pos).result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
