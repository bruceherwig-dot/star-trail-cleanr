"""Read the frames of a batch a few at a time, but hand them back in file order.

Frame files are read one after another before detection starts. Each read is
independent of the others, so a few can run at once. The caller still sees the
frames in the same order as the file list, one at a time, so everything it does
per frame (the bad-file report and prompt, the progress lines, the counters) is
unchanged.

OpenCV's console log level is one setting for the whole process. The reader
mutes it and restores it on every call, so overlapping calls could restore the
wrong level and leave it muted for good. It is muted once here, around the whole
batch, and restored once at the end.
"""
from concurrent.futures import ThreadPoolExecutor

DEFAULT_WORKERS = 3     # a few at once; more only adds memory and disk contention


def read_frames_in_order(paths, reader, workers=DEFAULT_WORKERS):
    """Yield (path, image, diagnosis) for each path, in the order given.

    `reader(path)` must return (image_or_None, diagnosis_or_None), as
    robust_imread_diag does. An exception inside `reader` is raised here at that
    frame's turn, like the one-at-a-time loop did. Stopping early (closing the
    generator, or an exit while a prompt is open) cancels reads that have not
    started and restores the log level.
    """
    from modules.io_safe import _restore_cv2_logs, _silence_cv2_logs
    paths = list(paths)
    prev = _silence_cv2_logs()
    pool = None
    try:
        if workers > 1 and len(paths) > 1:
            try:
                pool = ThreadPoolExecutor(max_workers=min(int(workers), len(paths)))
            except Exception:
                pool = None            # cannot start threads: read one at a time
        if pool is None:
            for p in paths:
                img, diag = reader(p)
                yield p, img, diag
            return
        futures = [pool.submit(reader, p) for p in paths]
        for p, fut in zip(paths, futures):
            img, diag = fut.result()
            yield p, img, diag
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        _restore_cv2_logs(prev)
