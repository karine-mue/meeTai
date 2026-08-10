"""A. 暦・数値層（spec/verification.md §2 A1..A8）"""

from life import BUILD_DIR, parse
from render import build_all
import validate


def test_a1_weekdays_match_real_calendar(record):
    assert validate.check_weekdays(record) == []


def test_a2_screenshot_is_after_every_record_on_it(record):
    assert validate.check_causality(record) == []


def test_a3_records_ascend_within_a_screen(record):
    assert validate.check_intra_screen_order(record) == []


def test_a4_screens_do_not_go_back_in_time(record):
    assert validate.check_inter_screen_order(record) == []


def test_a5_file_sequence_increases_with_capture_order(record):
    assert validate.check_sequence(record) == []


def test_a6_filenames_are_consecutive_from_01(record):
    assert validate.check_filenames(record) == []


def test_a7_device_values_are_in_range(record):
    assert validate.check_ranges(record) == []


def test_a8_status_bar_clock_comes_from_shot_at(record):
    """レンダラが shot_at から時刻を導出していること。"""
    build_all(record)
    for s in record["screens"]:
        if "device" not in s:
            continue
        html = (BUILD_DIR / s["file"].replace(".png", ".html")).read_text(encoding="utf-8")
        clock = parse(s["device"]["shot_at"]).strftime("%H:%M")
        assert f"<span>{clock}</span>" in html, f'画面{s["index"]}: {clock} が出ていない'


def test_span_covers_eight_years(record):
    """記録が発注の8年間（2015-04 〜 2023-03）に収まっている。"""
    span = record["work"]["span"]
    lo = parse(span["from"] + "T00:00")
    hi = parse(span["to"] + "T23:59")
    for s in record["screens"]:
        for e in (s.get("posts") or []) + (s.get("messages") or []):
            assert lo <= parse(e["at"]) <= hi, f'{e["at"]} が8年の外'
