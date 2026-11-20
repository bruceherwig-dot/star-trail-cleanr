"""A file with an accidental-looking name is flagged before the run, never skipped.

2026-11-19: a hand-made IMG_6616.jpg.jpg sat in a 1,500-frame folder, was stacked like
any other frame, and left rectangles and dots in a smoothed star trail. Bruce's call:
no rule to skip such files, but tell the person before the run. Offline.
"""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules.frame_list import odd_file_names, odd_name_notice, ODD_MAX_NAMED


def test_double_extensions_are_flagged():
    got = odd_file_names(["/x/IMG_1.jpg", "/x/IMG_6616.jpg.jpg", "/x/IMG_2.TIF.TIF",
                          "/x/IMG_3.jpeg.jpg", "/x/IMG_4.cr2.jpg"])
    assert got == ["IMG_6616.jpg.jpg", "IMG_2.TIF.TIF", "IMG_3.jpeg.jpg", "IMG_4.cr2.jpg"]


def test_copy_and_edited_names_are_flagged():
    got = odd_file_names(["/x/IMG_5 copy.jpg", "/x/IMG_6 (1).jpg", "/x/IMG_7_edited.tif",
                          "/x/IMG_8-edited.jpg", "/x/IMG_9 copy 2.jpg", "/x/IMG_10.jpg"])
    assert got == ["IMG_5 copy.jpg", "IMG_6 (1).jpg", "IMG_7_edited.tif", "IMG_8-edited.jpg",
                   "IMG_9 copy 2.jpg"]


def test_normal_names_are_not_flagged():
    normal = ["/x/IMG_0001.jpg", "/x/_DSC0733.tiff", "/x/143A8732.jpg", "/x/IMG_6616.jpg",
              "/x/Copyright_test_01.jpg", "/x/frame.001.jpg", "/x/photo_(final)_01.jpg"]
    assert odd_file_names(normal) == []


def test_the_notice_names_a_few_and_counts_the_rest():
    one = odd_name_notice(["IMG_6616.jpg.jpg"])
    assert "IMG_6616.jpg.jpg" in one and "will be included" in one and "take it out" in one
    many = odd_name_notice([f"IMG_{i}.jpg.jpg" for i in range(7)])
    assert "7 files" in many and f"and {7 - ODD_MAX_NAMED} more" in many
    assert odd_name_notice([]) == ""


def test_checking_names_never_changes_the_frame_list():
    frames = ["/x/IMG_1.jpg", "/x/IMG_2.jpg.jpg", "/x/IMG_3.jpg"]
    before = list(frames)
    odd_file_names(frames)
    assert frames == before, "nothing is skipped, renamed or reordered"


def test_the_gui_shows_it_next_to_the_pause_notice():
    src = (REPO / "star_trail_cleanr.py").read_text(encoding="utf-8")
    assert "odd_file_names(frames)" in src and "odd_name_notice(odd)" in src
