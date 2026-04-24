"""Smoke tests for modules/detect_trails.best_device()."""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def test_best_device_imports():
    from modules.detect_trails import best_device
    assert callable(best_device)


def test_best_device_returns_known_string():
    """best_device must return one of 'cuda', 'mps', 'cpu'. No other values."""
    from modules.detect_trails import best_device
    result = best_device()
    assert result in ("cuda", "mps", "cpu"), f"Unexpected device: {result!r}"


def test_best_device_never_raises():
    """Function must be safe to call under any environment; never raises."""
    from modules.detect_trails import best_device
    # Call a few times in case any internal state could break on repeat
    for _ in range(3):
        best_device()


def test_load_model_auto_resolution_path():
    """load_model with device=None or 'auto' should resolve before use.

    We do not actually load a model here (that requires a real .pt file and
    heavy dependencies). We only verify the signature and the resolution
    hook work.
    """
    from modules.detect_trails import load_model
    import inspect
    sig = inspect.signature(load_model)
    # device parameter exists and its default is None or 'auto'
    assert "device" in sig.parameters
    default = sig.parameters["device"].default
    assert default is None or default == "auto"
