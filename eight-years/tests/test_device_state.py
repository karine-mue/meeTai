"""B. 端末状態と行動の整合（spec/design.md §5.3 の R1..R12）"""

from life import parse, shot_screens
import validate


def test_all_device_rules_hold(record):
    assert validate.check_device_rules(record) == []


def test_every_screenshot_declares_a_context(record):
    for s in shot_screens(record):
        assert s["device"]["context"], f'画面{s["index"]}: context がない'
    for s in record["screens"]:
        for p in s.get("posts") or []:
            assert p["context"], f'画面{s["index"]}: 投稿 {p["at"]} に context がない'


def test_charging_never_happens_outdoors(record):
    """R1/R2 を独立に書き下した形でも押さえる。"""
    for s in shot_screens(record):
        d = s["device"]
        if d["context"] in ("out", "transit"):
            assert not d["charging"], f'画面{s["index"]}: 外で充電している'


def test_wifi_only_where_wifi_exists(record):
    for s in shot_screens(record):
        d = s["device"]
        if d["network"] == "wifi":
            assert d["context"] in ("home", "work"), f'画面{s["index"]}: 外で Wi-Fi'


def test_battery_is_plausible_for_time_of_day(record):
    for s in shot_screens(record):
        d = s["device"]
        h = parse(d["shot_at"]).hour
        if not d["charging"] and h <= 8:
            assert d["battery"] >= 60, f'画面{s["index"]}: 朝なのに電池が減りすぎ'
        if d["context"] == "out" and h >= 19:
            assert d["battery"] <= 55, f'画面{s["index"]}: 夜の外出で電池が多すぎ'
