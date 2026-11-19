"""check_gpu_wheels.py — release gate for the Windows GPU pack download links.

WHY THIS EXISTS
---------------
Windows NVIDIA users can install an optional "GPU pack": the CUDA (cu128) PyTorch
and torchvision wheels, downloaded at runtime from pytorch.org. The build bakes in
whatever torch version it compiled against, and the in-app installer asks for the
cu128 flavour of that exact version. PyTorch does not publish a cu128 wheel for
every release (2.12.0, for example, shipped cu126 + cu130 but NOT cu128), so a
build can ship an installer that points at a wheel that was never built. That is
invisible until a user clicks "Install GPU support" and gets a download failure.

This gate runs in CI on the Windows build. It takes the torch + torchvision
versions this build will ship (the installed versions, the same source
build_helper.py bakes in) and confirms, via pytorch.org's package index, that the
matching cu128 Windows wheels are actually published. If either is absent it exits
non-zero and FAILS the build, so a broken GPU installer can never reach users.

(pytorch.org returns HTTP 403 for both a missing file and a region block, so a
plain HEAD can't tell them apart. The package index page lists every published
wheel, so it is the authoritative existence check — that is what gpu_pack uses too.)

Usage:
    python scripts/check_gpu_wheels.py
    python scripts/check_gpu_wheels.py --torch 2.11.0 --torchvision 0.26.0
The explicit-version form is for testing the gate itself (prove it passes on a
known-good pair and fails on a known-bad one).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.gpu_pack import wheel_published, build_download_url_sets  # noqa: E402


def _installed_version(mod_name):
    """Return the bare version of an installed module (drops any +build suffix)."""
    try:
        mod = __import__(mod_name)
        return mod.__version__.split("+")[0]
    except Exception:
        return None


def _published_with_retry(package, version, attempts=3):
    """wheel_published() but retry while the index itself is unreachable (None),
    so a transient CI network blip doesn't fail the build. Returns True/False/None."""
    for i in range(attempts):
        result = wheel_published(package, version)
        if result is not None:
            return result
        if i < attempts - 1:
            time.sleep(3)
    return None


def main():
    ap = argparse.ArgumentParser(description="Verify GPU pack download links resolve.")
    ap.add_argument("--torch", default=None, help="override torch version to check")
    ap.add_argument("--torchvision", default=None, help="override torchvision version")
    args = ap.parse_args()

    torch_ver = args.torch or _installed_version("torch")
    tv_ver = args.torchvision or _installed_version("torchvision")
    if not torch_ver or not tv_ver:
        print("GPU WHEEL GATE: FAIL — could not determine torch/torchvision versions "
              f"(torch={torch_ver}, torchvision={tv_ver}).")
        return 1

    sets = build_download_url_sets(torch_ver, tv_ver)
    print(f"GPU WHEEL GATE: checking cu128 wheels for torch {torch_ver} / "
          f"torchvision {tv_ver}")
    print(f"  torch:       {sets[0][0]}")
    print(f"  torchvision: {sets[0][1]}")

    torch_pub = _published_with_retry("torch", torch_ver)
    tv_pub = _published_with_retry("torchvision", tv_ver)
    print(f"  torch published:       {torch_pub}")
    print(f"  torchvision published: {tv_pub}")

    if torch_pub is True and tv_pub is True:
        print("GPU WHEEL GATE: PASS — both GPU wheels are published.")
        return 0

    if torch_pub is None or tv_pub is None:
        print("GPU WHEEL GATE: FAIL — could not reach pytorch.org's index to verify "
              "the GPU wheels. Not shipping an unverified GPU installer.")
        return 1

    print("GPU WHEEL GATE: FAIL — a GPU wheel this build would ship is NOT published "
          "on pytorch.org. The in-app GPU installer would fail. Pin the Windows build "
          "to a torch version that has cu128 Windows wheels (see modules/gpu_pack.py "
          "_KNOWN_GOOD_CU128), or update the CUDA suffix.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
