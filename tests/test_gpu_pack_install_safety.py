"""Smoke tests for the safety rules around installing the Windows GPU pack.

The rule these lock down: a failed or incomplete GPU install must never cost
someone the working GPU support they already had. The old installer deleted the
working pack BEFORE downloading, so a blocked or dropped download left a user
with no GPU acceleration and nothing on screen to explain it.

All offline: no network, no NVIDIA hardware, no real wheels. The pack folders are
redirected to a temporary directory via LOCALAPPDATA.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


class _TempPackHome:
    """Point the GPU-pack folders at a throwaway directory for one test."""

    def __enter__(self):
        self._prev = os.environ.get("LOCALAPPDATA")
        self.root = tempfile.mkdtemp(prefix="stc_gpu_test_")
        os.environ["LOCALAPPDATA"] = self.root
        return self

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = self._prev
        shutil.rmtree(self.root, ignore_errors=True)
        return False


def _make_pack(folder, marker):
    """Create a stand-in for an extracted pack: both package folders, each with a
    file naming which pack it is so a test can tell them apart."""
    for part in ("torch", "torchvision"):
        d = Path(folder) / part
        d.mkdir(parents=True, exist_ok=True)
        (d / "which.txt").write_text(marker)


def test_install_safety_exports():
    import modules.gpu_pack as g
    for name in ("get_staging_dir", "get_backup_dir", "swap_staged_into_place"):
        assert hasattr(g, name), f"missing {name}"


def test_staging_is_not_the_live_folder():
    """Staging and backup must be separate folders from the one the runtime hook
    loads, or downloading into them would clobber the live pack."""
    with _TempPackHome():
        from modules.gpu_pack import (get_override_dir, get_staging_dir,
                                      get_backup_dir)
        live, staging, backup = get_override_dir(), get_staging_dir(), get_backup_dir()
        assert len({live, staging, backup}) == 3, "folders must all differ"
        assert staging.parent == live.parent, "staging must sit beside the live pack"


def test_incomplete_download_leaves_the_working_pack_alone():
    """THE regression net. A staged folder missing part of the download must be
    refused, and the pack already in service must survive untouched."""
    with _TempPackHome():
        from modules.gpu_pack import (get_override_dir, get_staging_dir,
                                      swap_staged_into_place)
        live = get_override_dir()
        _make_pack(live, "working")
        (live / "torch_version.txt").write_text("2.11.0")

        # Only half the download arrived.
        staging = get_staging_dir()
        (staging / "torch").mkdir(parents=True)

        ok, err = swap_staged_into_place()
        assert not ok, "an incomplete download must never be swapped in"
        assert err, "a refusal must come with a reason"
        assert (live / "torch" / "which.txt").read_text() == "working"
        assert (live / "torchvision" / "which.txt").read_text() == "working"
        assert (live / "torch_version.txt").is_file()


def test_complete_download_replaces_the_old_pack():
    with _TempPackHome():
        from modules.gpu_pack import (get_override_dir, get_staging_dir,
                                      get_backup_dir, swap_staged_into_place)
        live = get_override_dir()
        _make_pack(live, "old")
        _make_pack(get_staging_dir(), "new")

        ok, err = swap_staged_into_place()
        assert ok, f"a complete download should swap in cleanly: {err}"
        assert (live / "torch" / "which.txt").read_text() == "new"
        assert (live / "torchvision" / "which.txt").read_text() == "new"
        assert not get_staging_dir().exists(), "staging should be consumed by the swap"
        assert not get_backup_dir().exists(), "the old pack should be cleaned up"


def test_first_ever_install_needs_no_existing_pack():
    with _TempPackHome():
        from modules.gpu_pack import (get_override_dir, get_staging_dir,
                                      swap_staged_into_place)
        _make_pack(get_staging_dir(), "first")
        ok, err = swap_staged_into_place()
        assert ok, f"a first install should succeed: {err}"
        assert (get_override_dir() / "torch" / "which.txt").read_text() == "first"


def test_installer_refuses_a_version_the_app_wont_load():
    """The installer must not fall back to an older torch version. The runtime
    hook only loads a pack matching the running build, so a substituted version
    would mean a ~4 GB download the app then ignores at every launch."""
    src = (REPO / "star_trail_cleanr.py").read_text()
    start = src.index("class GpuPackInstallThread")
    end = src.index("class _XCloseButton")
    body = src[start:end]
    assert "healed" in body, "the install flow no longer checks the self-heal flag"
    healed_at = body.index("if healed:")
    after = body[healed_at:healed_at + 400]
    assert "self.failed.emit" in after, (
        "a healed (substituted) version must stop the install, not proceed")
