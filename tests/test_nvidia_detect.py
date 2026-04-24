"""Smoke tests for modules/nvidia_detect.py."""
import sys
import types
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _install_fake_pynvml(fake):
    """Install a fake pynvml module into sys.modules. Caller should pop it after."""
    sys.modules["pynvml"] = fake


def _make_fake_pynvml(*, init_raises=None, device_count=1, device_name="NVIDIA GeForce RTX 5090",
                     driver_version="555.00", get_count_raises=None, get_name_raises=None):
    fake = types.ModuleType("pynvml")

    class FakeNVMLError(Exception):
        pass

    fake.NVMLError = FakeNVMLError

    def nvmlInit():
        if init_raises is not None:
            raise init_raises

    def nvmlShutdown():
        pass

    def nvmlDeviceGetCount():
        if get_count_raises is not None:
            raise get_count_raises
        return device_count

    def nvmlDeviceGetHandleByIndex(i):
        return i

    def nvmlDeviceGetName(h):
        if get_name_raises is not None:
            raise get_name_raises
        return device_name

    def nvmlSystemGetDriverVersion():
        return driver_version

    fake.nvmlInit = nvmlInit
    fake.nvmlShutdown = nvmlShutdown
    fake.nvmlDeviceGetCount = nvmlDeviceGetCount
    fake.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
    fake.nvmlDeviceGetName = nvmlDeviceGetName
    fake.nvmlSystemGetDriverVersion = nvmlSystemGetDriverVersion
    return fake


def test_nvidia_detect_imports():
    import modules.nvidia_detect as m
    assert hasattr(m, "detect_nvidia")


def test_outcome_yes_with_working_driver():
    from modules.nvidia_detect import detect_nvidia
    fake = _make_fake_pynvml()
    _install_fake_pynvml(fake)
    try:
        outcome, detail = detect_nvidia()
    finally:
        sys.modules.pop("pynvml", None)
    assert outcome == "yes", f"expected yes, got {outcome} ({detail})"
    assert "RTX 5090" in detail
    assert "555.00" in detail


def test_outcome_no_driver_when_library_missing():
    from modules.nvidia_detect import detect_nvidia
    err = Exception("NVML Shared Library Not Found")
    fake = _make_fake_pynvml(init_raises=err)
    _install_fake_pynvml(fake)
    try:
        outcome, detail = detect_nvidia()
    finally:
        sys.modules.pop("pynvml", None)
    assert outcome == "no_driver_or_card", f"expected no_driver_or_card, got {outcome}"


def test_outcome_driver_problem_on_other_nvml_error():
    from modules.nvidia_detect import detect_nvidia
    err = Exception("Driver Version Mismatch")
    fake = _make_fake_pynvml(init_raises=err)
    _install_fake_pynvml(fake)
    try:
        outcome, detail = detect_nvidia()
    finally:
        sys.modules.pop("pynvml", None)
    assert outcome == "driver_problem", f"expected driver_problem, got {outcome}"


def test_outcome_no_driver_when_device_count_zero():
    from modules.nvidia_detect import detect_nvidia
    fake = _make_fake_pynvml(device_count=0)
    _install_fake_pynvml(fake)
    try:
        outcome, detail = detect_nvidia()
    finally:
        sys.modules.pop("pynvml", None)
    assert outcome == "no_driver_or_card"


def test_outcome_library_not_installed_when_import_fails():
    """If nvidia-ml-py isn't installed at all, we should get library_not_installed."""
    from modules.nvidia_detect import detect_nvidia
    sys.modules.pop("pynvml", None)

    class BlockPynvml:
        def find_module(self, name, path=None):
            if name == "pynvml":
                return self
            return None

        def load_module(self, name):
            raise ImportError("blocked by test")

        # For Python 3.4+ finder protocol
        def find_spec(self, name, path=None, target=None):
            if name == "pynvml":
                raise ImportError("blocked by test")
            return None

    blocker = BlockPynvml()
    sys.meta_path.insert(0, blocker)
    try:
        outcome, detail = detect_nvidia()
    finally:
        sys.meta_path.remove(blocker)
    assert outcome == "library_not_installed", f"expected library_not_installed, got {outcome}"
