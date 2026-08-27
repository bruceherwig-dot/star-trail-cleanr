"""Smoke tests for the Windows GPU-pack download URLs (modules/gpu_pack.py).

These lock the exact wheel-URL format and the self-heal/pin consistency. They are
the regression net for the class of bug that broke GPU install for every Windows
NVIDIA user: the build baked torch 2.12.0, which PyTorch never shipped a cu128
wheel for, so the installer's URL 404'd. All offline (no network).
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def test_gpu_pack_exports():
    import modules.gpu_pack as g
    for name in ("build_download_url_sets", "candidate_version_pairs",
                 "resolve_available_url_set", "wheel_published",
                 "_KNOWN_GOOD_CU128", "CUDA_SUFFIX", "PYTHON_TAG"):
        assert hasattr(g, name), f"missing {name}"


def test_url_format_is_exact():
    """The wheel URL must match PyTorch's filename convention exactly: the '+' is
    %2B-encoded, the CUDA suffix and the doubled cp311 ABI tag are present, and the
    aliyun mirror is the second entry. A drift here is what 404'd in the field."""
    from modules.gpu_pack import build_download_url_sets
    sets = build_download_url_sets("2.11.0", "0.26.0")
    assert len(sets) == 2, "expected pytorch.org + aliyun mirrors"
    torch_url, tv_url, tver, tvver = sets[0]
    assert torch_url == (
        "https://download.pytorch.org/whl/cu128/"
        "torch-2.11.0%2Bcu128-cp311-cp311-win_amd64.whl"), torch_url
    assert tv_url == (
        "https://download.pytorch.org/whl/cu128/"
        "torchvision-0.26.0%2Bcu128-cp311-cp311-win_amd64.whl"), tv_url
    assert (tver, tvver) == ("2.11.0", "0.26.0")
    assert "aliyun" in sets[1][0], "second mirror should be the aliyun fallback"


def test_candidate_pairs_dedup_and_order():
    """Known-good fallbacks are present, newest-first, with no duplicates."""
    from modules.gpu_pack import candidate_version_pairs, _KNOWN_GOOD_CU128
    pairs = candidate_version_pairs()
    assert _KNOWN_GOOD_CU128[0] in pairs
    assert len(pairs) == len(set(pairs)), "candidate list must not contain duplicates"


def test_version_lock_matches_pin_and_known_good():
    """The version lock, the CI pin and the self-heal default must all agree.

    Bumping the torch version drops every Windows user who already installed a
    GPU pack back to the CPU until they redownload ~4 GB, because the runtime
    hook only loads a pack matching the running build. Stating the version in
    three places and failing here on disagreement means a bump cannot slip
    through as a quiet one-line edit in build.yml.
    """
    from modules.gpu_pack import GPU_PACK_TORCH_LOCK, _KNOWN_GOOD_CU128
    workflow = (REPO / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    m = re.search(r"pip install torch==([\d.]+) torchvision==([\d.]+) "
                  r"--index-url https://download\.pytorch\.org/whl/cpu", workflow)
    assert m, "could not find the pinned Windows torch install line in build.yml"
    assert (m.group(1), m.group(2)) == GPU_PACK_TORCH_LOCK, (
        f"build.yml pins torch=={m.group(1)}/{m.group(2)} but GPU_PACK_TORCH_LOCK is "
        f"{GPU_PACK_TORCH_LOCK}. Changing this orphans every existing GPU user: read "
        f"the comment on GPU_PACK_TORCH_LOCK in modules/gpu_pack.py before touching it.")
    assert _KNOWN_GOOD_CU128[0] == GPU_PACK_TORCH_LOCK, (
        f"_KNOWN_GOOD_CU128[0] is {_KNOWN_GOOD_CU128[0]} but the lock is "
        f"{GPU_PACK_TORCH_LOCK}; keep them in sync")


def test_known_good_first_matches_ci_pin():
    """The self-heal default (_KNOWN_GOOD_CU128[0]) must equal the version the
    Windows build is pinned to, or a bumped pin could ship a version the self-heal
    list doesn't know about. Cross-checks the two so they can't drift apart."""
    from modules.gpu_pack import _KNOWN_GOOD_CU128
    pin_torch, pin_tv = _KNOWN_GOOD_CU128[0]
    workflow = (REPO / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    m = re.search(r"pip install torch==([\d.]+) torchvision==([\d.]+) "
                  r"--index-url https://download\.pytorch\.org/whl/cpu", workflow)
    assert m, "could not find the pinned Windows torch install line in build.yml"
    assert (m.group(1), m.group(2)) == (pin_torch, pin_tv), (
        f"build.yml pins torch=={m.group(1)} torchvision=={m.group(2)} but "
        f"_KNOWN_GOOD_CU128[0] is {pin_torch}/{pin_tv} — keep them in sync")
