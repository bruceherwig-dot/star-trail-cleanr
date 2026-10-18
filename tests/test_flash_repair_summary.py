"""The plain-English summary and the counts for the usage report."""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from modules import flash_repair as fr


def _sky(bail=None, bursts=0):
    r = fr.FlashResult()
    r.bail, r.burst_frames = bail, bursts
    return r


def _ground(bail=None):
    g = fr.GroundResult()
    g.bail = bail
    return g


def test_a_normal_run_reads_in_plain_english_and_counts_what_was_done():
    lines, counts = fr.plain_summary(
        _sky(bursts=2), {"flash_spots": 12, "defect_spots": 1, "hidden_not_repaired": 5, "pixels_changed": 300},
        _ground(), {"ground_spots": 40, "on_a_trail": 3, "pixels_changed": 90})
    text = "\n".join(lines)
    assert "12 one-frame flashes and 1 blinking sensor defect" in text
    assert "2 frames" in text and "lighting events" in text
    assert "removed 40 specks and left 3 alone that sat on a trail" in text
    assert counts["flash_spots"] == 12 and counts["defect_spots"] == 1 and counts["ground_spots"] == 40
    assert counts["pixels_changed"] == 390 and counts["left_alone"] == 8 and counts["lighting_event_frames"] == 2
    assert not counts["sky_stopped"] and not counts["ground_stopped"]


def test_a_stopped_pass_says_why_and_is_counted_as_stopped():
    lines, counts = fr.plain_summary(_sky(bail="the sky's motion could not be measured"), {}, _ground(bail="no foreground mask"), {})
    text = "\n".join(lines)
    assert "nothing was changed because the sky's motion could not be measured" in text
    assert "left alone because no foreground mask" in text
    assert counts["sky_stopped"] and counts["ground_stopped"]
    assert counts["flash_spots"] == 0 and counts["ground_spots"] == 0


def test_a_quiet_night_says_nothing_was_found():
    lines, counts = fr.plain_summary(_sky(), {}, _ground(), {})
    assert "no single-frame flashes or blinking defects found" in "\n".join(lines)
    assert "removed 0 specks" in "\n".join(lines)


def test_the_counts_hold_only_numbers_and_flags_never_text():
    _, counts = fr.plain_summary(_sky(bail="a path /Users/someone/pics"), {}, _ground(), {})
    assert all(isinstance(v, (int, bool)) for v in counts.values()), counts


def test_the_marker_lines_round_trip():
    lines, counts = fr.plain_summary(_sky(), {"flash_spots": 3}, _ground(), {"ground_spots": 1})
    got_lines, got_counts = [], None
    for m in fr.marker_lines(lines, counts):
        kind, val = fr.parse_marker(m)
        if kind == "line":
            got_lines.append(val)
        else:
            got_counts = val
    assert got_lines == lines and got_counts == counts
    assert fr.parse_marker("  speck removal done") is None
    assert fr.parse_marker(fr.MARKER_COUNTS + " not json") is None


def test_the_wording_has_no_banned_style():
    lines, _ = fr.plain_summary(_sky(bursts=1), {"flash_spots": 1, "hidden_not_repaired": 1}, _ground(), {"ground_spots": 1, "on_a_trail": 1})
    for t in lines:
        assert "—" not in t and "settled" not in t.lower()
