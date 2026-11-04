"""Smoke tests for the doubled-photo guard (the v4 jpg+tif contamination).

A photo uploaded twice (same name, .jpg and .tif) must be caught by the preflight
check and refused by both CVAT upload tools. No CVAT needed; runs in milliseconds.
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "dataset_pipeline" / "scripts"))

from cvat_preflight import duplicate_stems


def test_preflight_finds_jpg_tif_twins():
    d = duplicate_stems(["_DSC1.jpg", "_DSC1.tif", "_DSC2.jpg", "_DSC3.JPG", "_DSC3.tiff"])
    assert sorted(d) == ["_dsc1", "_dsc3"], d


def test_preflight_clean_set_passes():
    assert duplicate_stems(["a.jpg", "b.jpg", "c.jpg"]) == {}


def test_upload_from_polys_refuses_doubled_folder():
    import cvat_upload_from_polys as u
    with tempfile.TemporaryDirectory() as t:
        for n in ("x1.jpg", "x1.tif", "x2.jpg"):
            (Path(t) / n).write_bytes(b"")
        try:
            u.list_frames(t)
        except SystemExit as e:
            assert "REFUSING" in str(e)
            return
        raise AssertionError("doubled folder was accepted")


def test_upload_from_polys_accepts_clean_folder():
    import cvat_upload_from_polys as u
    with tempfile.TemporaryDirectory() as t:
        for n in ("x1.jpg", "x2.jpg"):
            (Path(t) / n).write_bytes(b"")
        assert u.list_frames(t) == ["x1.jpg", "x2.jpg"]
