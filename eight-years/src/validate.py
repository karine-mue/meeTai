"""検証関数群。

すべて `record`（record.json を読んだ dict）を受け取り、
違反メッセージの list を返す純粋関数。空リストなら合格。

テストからも、意図的に壊した入力からも同じ関数を呼べるようにしてある。
"""

from __future__ import annotations

from life import (
    entries,
    fiscal_year,
    fiscal_years,
    own_posts,
    parse,
    shot_screens,
    weekday_ja,
)

# --------------------------------------------------------------------------
# ブラックリスト
# --------------------------------------------------------------------------

# 集団的に典型な反復要素。縦糸の題材がここに入っていたら失格。
TYPICAL_MOTIFS = [
    "桜", "初詣", "誕生日", "花火", "クリスマス", "正月", "成人式",
    "卒業式", "入学式", "七夕", "紅葉", "初日の出", "大晦日", "バレンタイン",
]

# 感動的な語彙。本文に出てはいけない。
SENTIMENTAL = [
    "かけがえ", "あの日", "きらめ", "永遠", "奇跡", "運命", "青春",
    "感動", "旅立", "忘れない", "大切な", "宝物", "ありがとう", "さようなら",
    "輝い", "涙", "泣い", "だいじょうぶ", "がんばろう",
]

# 実在の企業・サービス・施設・路線。画面上のどの文字列にも出てはいけない。
REAL_WORLD = [
    "Twitter", "ツイッター", "LINE", "Instagram", "インスタ", "TikTok",
    "Facebook", "mixi", "Amazon", "Google", "Apple", "iPhone", "Android",
    "YouTube", "ディズニー", "マクドナルド", "スターバックス", "スタバ",
    "セブンイレブン", "ローソン", "ファミリーマート", "ユニクロ", "ドコモ",
    "ソフトバンク", "au", "楽天", "渋谷", "新宿", "池袋", "原宿", "秋葉原",
    "東京駅", "山手線", "JR", "京王", "小田急", "東急",
]


def _fmt(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------
# A. 暦・数値層
# --------------------------------------------------------------------------

def check_weekdays(record) -> list[str]:
    """A1: 記録された曜日が実在の暦と一致する。"""
    errs = []
    for s in record["screens"]:
        for e in entries(s):
            dt = parse(e["at"])
            real = weekday_ja(dt)
            if e["weekday"] != real:
                errs.append(
                    f"A1 画面{s['index']}: {e['at']} の曜日が「{e['weekday']}」だが実際は「{real}」"
                )
        dev = s.get("device")
        if dev:
            dt = parse(dev["shot_at"])
            real = weekday_ja(dt)
            if dev["weekday"] != real:
                errs.append(
                    f"A1 画面{s['index']}: スクショ {dev['shot_at']} の曜日が"
                    f"「{dev['weekday']}」だが実際は「{real}」"
                )
    return errs


def check_causality(record) -> list[str]:
    """A2: スクリーンショットは、その画面に写るどの記録よりも後に撮られている。"""
    errs = []
    for s in shot_screens(record):
        shot = parse(s["device"]["shot_at"])
        for e in entries(s):
            if parse(e["at"]) >= shot:
                errs.append(
                    f"A2 画面{s['index']}: 記録 {e['at']} がスクショ "
                    f"{s['device']['shot_at']} より後（または同時）"
                )
        if "read_at" in s and parse(s["read_at"]) >= shot:
            errs.append(f"A2 画面{s['index']}: 既読 {s['read_at']} がスクショより後")
    return errs


def check_intra_screen_order(record) -> list[str]:
    """A3: 画面内の記録が時系列昇順（UI の「古い順」表示と一致）。"""
    errs = []
    for s in record["screens"]:
        es = entries(s)
        for a, b in zip(es, es[1:]):
            if parse(a["at"]) >= parse(b["at"]):
                errs.append(
                    f"A3 画面{s['index']}: {a['at']} の次に {b['at']} が来ている（昇順でない）"
                )
        if es and s.get("sort") not in (None, "古い順"):
            errs.append(f"A3 画面{s['index']}: 並び替えが「{s['sort']}」なのに昇順で並んでいる")
    return errs


def check_inter_screen_order(record) -> list[str]:
    """A4: 画面間で時刻が戻らない。画面の期間が重ならない。"""
    errs = []
    shots = shot_screens(record)
    for a, b in zip(shots, shots[1:]):
        if parse(a["device"]["shot_at"]) >= parse(b["device"]["shot_at"]):
            errs.append(
                f"A4 画面{a['index']}→{b['index']}: スクショ時刻が進んでいない"
            )
        ea, eb = entries(a), entries(b)
        if ea and eb and parse(ea[-1]["at"]) >= parse(eb[0]["at"]):
            errs.append(
                f"A4 画面{a['index']}→{b['index']}: 期間が重なっている"
                f"（{ea[-1]['at']} ≥ {eb[0]['at']}）"
            )
    return errs


def check_sequence(record) -> list[str]:
    """A5: ファイル連番が撮影順に厳密増加し、ファイル名と一致する。"""
    errs = []
    shots = shot_screens(record)
    seen = set()
    prev = None
    for s in shots:
        seq = s["device"]["seq"]
        if seq in seen:
            errs.append(f"A5 画面{s['index']}: 連番 {seq} が重複")
        seen.add(seq)
        if prev is not None and seq <= prev:
            errs.append(f"A5 画面{s['index']}: 連番が {prev} → {seq} と減少/停滞")
        prev = seq
        if f"IMG_{seq:04d}" not in s["file"]:
            errs.append(f"A5 画面{s['index']}: ファイル名 {s['file']} が連番 {seq} と不一致")
    return errs


def check_filenames(record) -> list[str]:
    """A6: ファイル名の先頭連番が画面 index と一致し、01 から連続する。"""
    errs = []
    for i, s in enumerate(record["screens"], start=1):
        if s["index"] != i:
            errs.append(f"A6 {i}番目の画面の index が {s['index']}")
        head = s["file"].split("_")[0]
        if head != f"{i:02d}":
            errs.append(f"A6 画面{s['index']}: ファイル名 {s['file']} の連番が {i:02d} でない")
        if not s["file"].endswith(".png"):
            errs.append(f"A6 画面{s['index']}: {s['file']} が png でない")
    return errs


def check_ranges(record) -> list[str]:
    """A7: 端末層の数値が取りうる範囲に収まっている。"""
    errs = []
    for s in shot_screens(record):
        d = s["device"]
        if not isinstance(d["battery"], int) or not 1 <= d["battery"] <= 100:
            errs.append(f"A7 画面{s['index']}: 電池 {d['battery']} が 1..100 の整数でない")
        if not isinstance(d["bars"], int) or not 0 <= d["bars"] <= 4:
            errs.append(f"A7 画面{s['index']}: アンテナ {d['bars']} が 0..4 の整数でない")
        if d["network"] not in ("wifi", "mobile", "weak", "none"):
            errs.append(f"A7 画面{s['index']}: 通信 {d['network']} が未定義")
        if d["context"] not in ("home", "school", "work", "out", "transit"):
            errs.append(f"A7 画面{s['index']}: context {d['context']} が未定義")
    return errs


# --------------------------------------------------------------------------
# B. 端末状態と行動の整合（spec/design.md §5.3 の R1..R12）
# --------------------------------------------------------------------------

def check_device_rules(record) -> list[str]:
    errs = []
    for s in shot_screens(record):
        d = s["device"]
        tag = f"画面{s['index']}"
        dt = parse(d["shot_at"])
        h = dt.hour
        ctx, net, bat, chg, bars = (
            d["context"], d["network"], d["battery"], d["charging"], d["bars"],
        )

        if chg and ctx not in ("home", "work"):
            errs.append(f"R1 {tag}: {ctx} なのに充電中")
        if ctx in ("out", "transit") and chg:
            errs.append(f"R2 {tag}: 外出中（{ctx}）なのに充電中")
        if h <= 5 and ctx != "home":
            errs.append(f"R3 {tag}: {h}時なのに context が {ctx}")
        if ctx == "school" and not (8 <= h <= 18 and dt.weekday() <= 5):
            errs.append(f"R4 {tag}: 学校にいる時刻/曜日として不自然（{_fmt(dt)}）")
        if ctx == "out" and h >= 19 and bat > 55:
            errs.append(f"R5 {tag}: 19時以降の外出で電池 {bat}% は高すぎる")
        if not chg and h <= 8 and bat < 60:
            errs.append(f"R6 {tag}: 朝{h}時・非充電で電池 {bat}% は低すぎる")
        if net == "wifi" and ctx not in ("home", "work"):
            errs.append(f"R7 {tag}: {ctx} なのに Wi-Fi")
        if ctx == "transit" and net not in ("mobile", "weak", "none"):
            errs.append(f"R8 {tag}: 移動中なのに通信が {net}")
        if net == "none" and bars != 0:
            errs.append(f"R9 {tag}: 圏外なのにアンテナ {bars}")
        if net == "weak" and bars > 1:
            errs.append(f"R9 {tag}: 弱電波なのにアンテナ {bars}")
        if net == "mobile" and bars < 2:
            errs.append(f"R9 {tag}: モバイル通信でアンテナ {bars}")
        if net == "wifi" and bars < 1:
            errs.append(f"R9 {tag}: Wi-Fi でアンテナ {bars}")
        if bat == 100 and not chg:
            errs.append(f"R10 {tag}: 電池 100% なのに非充電")
        if chg and ctx == "home" and h <= 6 and bat < 70:
            errs.append(f"R11 {tag}: 未明の自宅充電で電池 {bat}% は低すぎる")

        es = entries(s)
        if es:
            last = es[-1]
            ldt = parse(last["at"])
            same_day = ldt.date() == dt.date()
            within = (dt - ldt).total_seconds() <= 90 * 60
            if same_day and within and "context" in last and last["context"] != ctx:
                errs.append(
                    f"R12 {tag}: 直近の記録が {last['context']} なのにスクショは {ctx}"
                )

    # 投稿側には R3・R4 のみ適用する（投稿は電池・通信を持たないため）
    for s in record["screens"]:
        for e in s.get("posts") or []:
            dt = parse(e["at"])
            h = dt.hour
            ctx = e["context"]
            if h <= 5 and ctx != "home":
                errs.append(f"R3 画面{s['index']}: 投稿 {e['at']} が {h}時で context {ctx}")
            if ctx == "school" and not (8 <= h <= 18 and dt.weekday() <= 5):
                errs.append(f"R4 画面{s['index']}: 投稿 {e['at']} が学校の時間帯でない")
    return errs


# --------------------------------------------------------------------------
# C. 縦糸
# --------------------------------------------------------------------------

def thread_by_year(record) -> dict[int, int]:
    """年度ごとの、本人による縦糸投稿の件数。"""
    counts = {y: 0 for y in fiscal_years(record)}
    for p in own_posts(record):
        if p.get("thread"):
            y = fiscal_year(parse(p["at"]))
            counts[y] = counts.get(y, 0) + 1
    return counts


def screens_by_year(record) -> dict[int, int]:
    counts = {y: 0 for y in fiscal_years(record)}
    for s in record["screens"]:
        es = entries(s)
        if not es:
            continue
        for e in es:
            counts[fiscal_year(parse(e["at"]))] = (
                counts.get(fiscal_year(parse(e["at"])), 0) + 1
            )
    return counts


def check_thread(record) -> list[str]:
    errs = []
    counts = thread_by_year(record)
    years = fiscal_years(record)

    # C1 年に1回程度
    for y, n in counts.items():
        if n > 1:
            errs.append(f"C1 年度{y}: 縦糸が {n} 回（年1回程度を超えている）")

    # C2 欠落が2回以上（= 全年度に存在してはならない、を含む）
    absent = [y for y in years if counts.get(y, 0) == 0]
    if len(absent) < 2:
        errs.append(f"C2 縦糸の欠落が {len(absent)} 年度しかない（2以上必要）")

    # C3 欠落への言及はちょうど1回
    mentions = [p for p in own_posts(record) if p.get("thread_mention")]
    if len(mentions) != 1:
        errs.append(f"C3 欠落への言及が {len(mentions)} 回（ちょうど1回であること）")

    # C4 言及が置かれた年度は、実際に欠落している年度である
    for m in mentions:
        y = fiscal_year(parse(m["at"]))
        if counts.get(y, 0) != 0:
            errs.append(f"C4 言及が年度{y}にあるが、その年度には縦糸が存在する")

    # C5 全年度に最低1件の記録がある（欠落が「記録が無いだけ」に退化していない）
    per_year = screens_by_year(record)
    for y in years:
        if per_year.get(y, 0) == 0:
            errs.append(f"C5 年度{y}: 記録が1件もない")

    # C6 縦糸の題材が集団的に典型な語でない
    motif = record["work"]["thread_motif"]
    for t in TYPICAL_MOTIFS:
        if t in motif:
            errs.append(f"C6 縦糸の題材「{motif}」が典型語「{t}」を含む")

    # C7 縦糸投稿は本人のものだけ（DM の相手の写真は数えない）
    for s in record["screens"]:
        for m in s.get("messages") or []:
            if m.get("thread"):
                errs.append(f"C7 画面{s['index']}: 本人以外の発話に thread が付いている")
    return errs


# --------------------------------------------------------------------------
# D. 構成・禁止事項
# --------------------------------------------------------------------------

def check_form(record) -> list[str]:
    errs = []
    ss = record["screens"]

    # D1 冒頭はちょうど1枚で、先頭
    notices = [s for s in ss if s["kind"] == "notice"]
    if len(notices) != 1:
        errs.append(f"D1 notice 画面が {len(notices)} 枚（1枚であること）")
    elif notices[0]["index"] != 1:
        errs.append(f"D1 notice が先頭でない（index {notices[0]['index']}）")

    # D2 冒頭以外に作者のテキストがない
    for s in ss:
        if s["kind"] != "notice" and ("text" in s or "footer" in s):
            errs.append(f"D2 画面{s['index']}: 本体に作者テキスト欄がある")

    # D3 枚数
    if not 10 <= len(ss) <= 12:
        errs.append(f"D3 画面総数が {len(ss)}（10..12 であること）")
    body = [s for s in ss if s["kind"] != "notice"][:-1]
    if not 10 <= len(body) <= 12:
        errs.append(f"D3 本体が {len(body)} 枚（10..12 であること）")

    # D4 媒体の仕様変更がちょうど1回
    uis = [s["ui"] for s in ss if "ui" in s]
    changes = [(a, b) for a, b in zip(uis, uis[1:]) if a != b]
    if len(changes) != 1:
        errs.append(f"D4 UI の変化が {len(changes)} 回（ちょうど1回であること）")
    elif changes[0] != ("v1", "v2"):
        errs.append(f"D4 UI の変化が {changes[0]}（v1→v2 であること）")

    # D5 アカウント表記も同じ位置で1回だけ変わる
    accs = [s["account"] for s in ss if "account" in s]
    acc_changes = [i for i, (a, b) in enumerate(zip(accs, accs[1:])) if a != b]
    ui_changes = [i for i, (a, b) in enumerate(zip(uis, uis[1:])) if a != b]
    if len(acc_changes) != 1:
        errs.append(f"D5 アカウント表記の変化が {len(acc_changes)} 回")
    elif acc_changes != ui_changes:
        errs.append("D5 アカウント表記の変化位置が UI の変化位置と違う")

    # D6 終端は本人以外の発話だけ
    last = ss[-1]
    me = record["work"]["self"]
    if last["kind"] != "dm":
        errs.append(f"D6 終端の kind が {last['kind']}（dm であること）")
    if last.get("posts"):
        errs.append("D6 終端に本人の投稿がある")
    for m in last.get("messages") or []:
        if m["from"] == me:
            errs.append(f"D6 終端に本人（{me}）の発話がある: {m['at']}")
    if not (last.get("messages") or []):
        errs.append("D6 終端に発話が1件もない")

    # D7 終端の後に画面がない
    if ss[-1]["index"] != len(ss):
        errs.append("D7 終端が配列の最後でない")

    # D8/D9 語彙
    texts = []
    for s in ss:
        if s["kind"] == "notice":
            continue
        for e in entries(s):
            texts.append((s["index"], e["text"]))
        for key in ("month_label", "date_label", "sort", "peer", "account"):
            if key in s:
                texts.append((s["index"], s[key]))
    for idx, t in texts:
        for w in SENTIMENTAL:
            if w in t:
                errs.append(f"D8 画面{idx}: 感動的語彙「{w}」が本文にある")
        for w in REAL_WORLD:
            if w in t:
                errs.append(f"D9 画面{idx}: 実在固有名詞「{w}」が画面にある")
    return errs


# --------------------------------------------------------------------------

CHECKS = {
    "A1_weekdays": check_weekdays,
    "A2_causality": check_causality,
    "A3_intra_order": check_intra_screen_order,
    "A4_inter_order": check_inter_screen_order,
    "A5_sequence": check_sequence,
    "A6_filenames": check_filenames,
    "A7_ranges": check_ranges,
    "B_device_rules": check_device_rules,
    "C_thread": check_thread,
    "D_form": check_form,
}


def all_checks(record) -> list[str]:
    errs = []
    for fn in CHECKS.values():
        errs.extend(fn(record))
    return errs
