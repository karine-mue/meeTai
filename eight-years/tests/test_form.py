"""D. 構成・禁止事項（spec/verification.md §2 D1..D9）"""

import validate


def test_all_form_rules_hold(record):
    assert validate.check_form(record) == []


def test_authors_words_exist_only_at_the_head(record):
    notices = [s for s in record["screens"] if s["kind"] == "notice"]
    assert len(notices) == 1 and notices[0]["index"] == 1
    for s in record["screens"][1:]:
        assert "text" not in s and "footer" not in s, f'画面{s["index"]} に地の文がある'


def test_body_is_ten_to_twelve_screens(record):
    screens = record["screens"]
    assert 10 <= len(screens) <= 12
    body = screens[1:-1]
    assert 10 <= len(body) <= 12, f"本体が {len(body)} 枚"


def test_medium_changes_spec_exactly_once(record):
    uis = [s["ui"] for s in record["screens"] if "ui" in s]
    changes = [(a, b) for a, b in zip(uis, uis[1:]) if a != b]
    assert changes == [("v1", "v2")], f"UI の変化が {changes}"


def test_account_migrates_at_the_same_point(record):
    accs = [s["account"] for s in record["screens"] if "account" in s]
    uis = [s["ui"] for s in record["screens"] if "ui" in s]
    acc_at = [i for i, (a, b) in enumerate(zip(accs, accs[1:])) if a != b]
    ui_at = [i for i, (a, b) in enumerate(zip(uis, uis[1:])) if a != b]
    assert acc_at == ui_at and len(acc_at) == 1


def test_ending_is_spoken_by_someone_else(record):
    last = record["screens"][-1]
    me = record["work"]["self"]
    assert last["kind"] == "dm"
    assert last.get("messages")
    assert not last.get("posts")
    assert all(m["from"] != me for m in last["messages"])


def test_ending_is_not_total_silence(record):
    """完全な無反応である必要はない ── 既読だけが残る。"""
    assert "read_at" in record["screens"][-1]


def test_nothing_follows_the_ending(record):
    assert record["screens"][-1]["index"] == len(record["screens"])


def test_no_sentimental_vocabulary(record):
    for s in record["screens"]:
        if s["kind"] == "notice":
            continue
        for e in (s.get("posts") or []) + (s.get("messages") or []):
            for w in validate.SENTIMENTAL:
                assert w not in e["text"], f'画面{s["index"]}: 「{w}」'


def test_no_real_world_proper_nouns(record):
    for s in record["screens"]:
        if s["kind"] == "notice":
            continue
        fields = [e["text"] for e in (s.get("posts") or []) + (s.get("messages") or [])]
        fields += [s[k] for k in ("month_label", "date_label", "peer", "account") if k in s]
        for t in fields:
            for w in validate.REAL_WORLD:
                assert w not in t, f'画面{s["index"]}: 「{w}」'


def test_no_image_generation_assets(record):
    """写真はすべて CSS のプレースホルダ種別で表現されている。"""
    allowed = {None, "plain", "glove"}
    for s in record["screens"]:
        for e in (s.get("posts") or []) + (s.get("messages") or []):
            assert e.get("photo") in allowed, f'未知の写真種別 {e.get("photo")}'
