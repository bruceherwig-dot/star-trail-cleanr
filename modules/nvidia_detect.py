"""
NVIDIA GPU detection via NVIDIA's official Python binding (nvidia-ml-py).

WHAT THIS FILE IS
-----------------
A tiny helper that answers one question: "Does this computer have a working
NVIDIA graphics card?" It asks NVIDIA's own management library (NVML, exposed
to Python through the `pynvml` package from `nvidia-ml-py`) and reports back.

WHY IT EXISTS / HOW IT FITS THE APP
-----------------------------------
Star Trail CleanR runs much faster when it can hand the heavy AI work to an
NVIDIA GPU using CUDA. But the CUDA build is a large, NVIDIA-only download, so
the app should only offer it to people who can actually use it. This module is
the check that decides whether to offer that download: if a real NVIDIA card
with a working driver is present, the caller can point the user at the CUDA
build; otherwise it stays on the standard build.

WHAT THE RESULT MEANS
---------------------
The single function here returns a three-state answer (plus one extra state for
"the checking library itself isn't even installed"), so the caller can decide
what to do:
  - "yes"                  NVIDIA card + working driver -> offer CUDA download
  - "driver_problem"       NVML started up but something else went wrong
  - "no_driver_or_card"    no card present, or the driver isn't loaded
  - "library_not_installed"  the pynvml package isn't installed at all

The import of pynvml is intentionally lazy (done inside the function, not at
the top of the file) so this module still loads cleanly on machines where the
nvidia-ml-py package isn't installed at all -- for example Mac development
builds, where there is no NVIDIA hardware to talk to.
"""


def detect_nvidia():
    """
    Ask NVIDIA's management library, exactly once, whether a usable NVIDIA GPU
    is present, and report the outcome.

    Inputs: none.

    Returns: a (outcome, detail) pair.
      - outcome is one of the four strings:
          'yes'                   a working NVIDIA card was found
          'driver_problem'        NVML initialized but a later call failed
          'no_driver_or_card'     no card / driver-not-loaded
          'library_not_installed' the pynvml package is missing
      - detail is a human-readable string: a short description of the GPU on
        success, or the underlying error message on any failure. It is meant
        for logging / display, not for programmatic decisions.

    Why it exists: see the module docstring -- it gates the optional CUDA
    download so only machines that can actually use it get offered it.
    """
    # Lazy import: if the nvidia-ml-py package isn't installed (e.g. a Mac dev
    # build with no NVIDIA hardware), bail out cleanly instead of crashing.
    try:
        import pynvml
    except ImportError as e:
        return ("library_not_installed", str(e))

    # Start up NVML. This is the call that fails when there's no driver/card.
    try:
        pynvml.nvmlInit()
    except Exception as e:
        err = str(e)
        low = err.lower()
        # NVML reports many problems only as a text message, so we sniff the
        # message to tell "no card / no driver" apart from other failures.
        # These phrases catch NVML errors for a missing library
        # ("libraryNotFound"), any message containing the generic words
        # "not found", or an explicitly unloaded driver ("driver not loaded")
        # -- all treated as "you simply don't have a usable NVIDIA GPU" cases
        # rather than a misconfiguration. Note the middle check is a broad
        # "not found" substring match, not a device-specific phrase.
        if ("libraryNotFound".lower() in low
                or "not found" in low
                or "driver not loaded" in low):
            return ("no_driver_or_card", err)
        # Anything else: NVML failed to start for a reason we can't classify
        # as "no card", so flag it as a driver problem worth surfacing.
        return ("driver_problem", err)

    # NVML started successfully. Now confirm there's at least one device and
    # read its name + driver version for the detail string.
    try:
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            # NVML works but reports zero GPUs -> treat as no card present.
            return ("no_driver_or_card", "NVML initialized but device count is 0")
        # Look at the first GPU (index 0). For multi-GPU machines we only need
        # one working card to justify offering the CUDA build.
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        # Older pynvml versions return bytes instead of str; normalize both
        # the GPU name and the driver version to plain text either way.
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8", errors="replace")
        return ("yes", f"{name} (driver {driver}, {count} device(s))")
    except Exception as e:
        # NVML was up but a query failed -> something is off with the driver.
        return ("driver_problem", str(e))
    finally:
        # Always release NVML, even on the success path. Wrapped in its own
        # try/except so a shutdown hiccup never masks the real result above.
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
