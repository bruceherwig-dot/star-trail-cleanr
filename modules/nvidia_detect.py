"""
NVIDIA GPU detection via NVIDIA's official Python binding (nvidia-ml-py).

Returns a three-state result so the caller can decide what to do:
  - "yes"                 NVIDIA card + working driver -> offer CUDA download
  - "driver_problem"      NVML initialized but something is off
  - "no_driver_or_card"   library not found, no card, or no driver

The import is intentionally lazy so the module loads even when the
nvidia-ml-py package isn't installed (Mac dev builds, for example).
"""


def detect_nvidia():
    """
    Query NVML once. Returns (outcome, detail) where outcome is one of
    'yes', 'driver_problem', 'no_driver_or_card', 'library_not_installed'.
    """
    try:
        import pynvml
    except ImportError as e:
        return ("library_not_installed", str(e))

    try:
        pynvml.nvmlInit()
    except Exception as e:
        err = str(e)
        low = err.lower()
        if ("libraryNotFound".lower() in low
                or "not found" in low
                or "driver not loaded" in low):
            return ("no_driver_or_card", err)
        return ("driver_problem", err)

    try:
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return ("no_driver_or_card", "NVML initialized but device count is 0")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8", errors="replace")
        return ("yes", f"{name} (driver {driver}, {count} device(s))")
    except Exception as e:
        return ("driver_problem", str(e))
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
