"""F. テストが弱くないことの検査。

全件が通る状態は、検査が甘いことの証拠にもなりうる。
そこで record.json のコピーに破壊を1件ずつ注入し、
対応する検査が確かに違反を報告することを確かめる。
"""

import pytest

import validate
from conftest import has

# (名前, 破壊関数, 期待する検査ID接頭辞, 呼ぶ検査関数名)
MUTATIONS = []


def mutation(name, prefix, check):
    def deco(fn):
        MUTATIONS.append((name, fn, prefix, check))
        return fn
    return deco


@mutation("曜日を1日ずらす", "A1", "check_weekdays")
def m_weekday(r):
    r["screens"][1]["posts"][0]["weekday"] = "金"


@mutation("スクショを記録より前にする", "A2", "check_causality")
def m_causality(r):
    r["screens"][1]["device"]["shot_at"] = "2015-05-07T07:00"


@mutation("画面内の投稿順を逆にする", "A3", "check_intra_screen_order")
def m_intra(r):
    r["screens"][1]["posts"].reverse()


@mutation("画面の順序を入れ替える", "A4", "check_inter_screen_order")
def m_inter(r):
    r["screens"][2], r["screens"][3] = r["screens"][3], r["screens"][2]


@mutation("連番を減らす", "A5", "check_sequence")
def m_seq(r):
    r["screens"][5]["device"]["seq"] = 700


@mutation("電池を140%にする", "A7", "check_ranges")
def m_battery_range(r):
    r["screens"][1]["device"]["battery"] = 140


@mutation("移動中に Wi-Fi を掴ませる", "R7", "check_device_rules")
def m_wifi_outside(r):
    r["screens"][1]["device"]["context"] = "transit"


@mutation("外出中に充電する", "R1", "check_device_rules")
def m_charging_outside(r):
    r["screens"][9]["device"]["charging"] = True


@mutation("深夜に外にいることにする", "R3", "check_device_rules")
def m_night_outside(r):
    r["screens"][2]["device"]["context"] = "out"


@mutation("夜の外出で電池を満タンにする", "R5", "check_device_rules")
def m_battery_narrative(r):
    s = r["screens"][9]
    s["device"]["shot_at"] = "2021-05-29T21:31"
    s["device"]["battery"] = 96
    s["device"]["context"] = "out"


@mutation("縦糸を全年度に置く", "C2", "check_thread")
def m_thread_everywhere(r):
    for s in r["screens"]:
        for p in s.get("posts") or []:
            p["thread"] = True


@mutation("欠落への言及を2回にする", "C3", "check_thread")
def m_two_mentions(r):
    r["screens"][8]["posts"][0]["thread_mention"] = True


@mutation("言及を縦糸のある年度に移す", "C4", "check_thread")
def m_mention_wrong_year(r):
    for s in r["screens"]:
        for p in s.get("posts") or []:
            p.pop("thread_mention", None)
    r["screens"][10]["posts"][2]["thread_mention"] = True


@mutation("本体に地の文を足す", "D2", "check_form")
def m_narration(r):
    r["screens"][4]["text"] = ["この年、彼女は学校を出た。"]


@mutation("媒体の仕様を2回変える", "D4", "check_form")
def m_two_ui_changes(r):
    r["screens"][4]["ui"] = "v2"


@mutation("終端に本人の発話を足す", "D6", "check_form")
def m_self_replies(r):
    r["screens"][-1]["messages"].append(
        {"from": "あお", "at": "2023-03-05T12:20", "weekday": "日", "text": "見た", "photo": None}
    )


@mutation("実在の固有名詞を混ぜる", "D9", "check_form")
def m_real_world(r):
    r["screens"][3]["posts"][0]["text"] = "渋谷まで出た"


@mutation("感動的な語彙で締める", "D8", "check_form")
def m_sentimental(r):
    r["screens"][10]["posts"][3]["text"] = "かけがえのない八年だった"


@pytest.mark.parametrize(
    "name,break_it,prefix,check",
    MUTATIONS,
    ids=[m[0] for m in MUTATIONS],
)
def test_broken_input_is_detected(clone, name, break_it, prefix, check):
    broken = clone()
    break_it(broken)
    errs = getattr(validate, check)(broken)
    assert has(errs, prefix), f"「{name}」を {prefix} が検出できていない。報告: {errs}"


def test_intact_record_reports_nothing(record):
    """破壊していない入力では、当然どの検査も黙っていること。"""
    assert validate.all_checks(record) == []


def test_every_mutation_is_caught_by_the_full_suite(clone):
    """個別検査だけでなく、all_checks でも必ず何か落ちること。"""
    for name, break_it, _, _ in MUTATIONS:
        broken = clone()
        break_it(broken)
        assert validate.all_checks(broken), f"「{name}」が all_checks を素通りした"
