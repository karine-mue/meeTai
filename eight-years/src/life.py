"""record.json の読み込みと、暦まわりの小さなユーティリティ。"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RECORD_PATH = ROOT / "src" / "record.json"
BUILD_DIR = ROOT / "src" / "build"
OUTPUT_DIR = ROOT / "output"

WEEKDAYS_JA = "月火水木金土日"

VIEWPORT_W = 390
VIEWPORT_H = 844
SCALE = 2


def load(path: pathlib.Path | None = None) -> dict:
    with open(path or RECORD_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def parse(ts: str) -> _dt.datetime:
    return _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M")


def weekday_ja(dt: _dt.datetime) -> str:
    return WEEKDAYS_JA[dt.weekday()]


def fiscal_year(dt: _dt.datetime) -> int:
    """年度。4月始まりで、2015-05-21 も 2016-02-28 も年度2015。"""
    return dt.year if dt.month >= 4 else dt.year - 1


def fiscal_years(record: dict) -> list[int]:
    span = record["work"]["span"]
    first = fiscal_year(_dt.datetime.strptime(span["from"], "%Y-%m-%d"))
    last = fiscal_year(_dt.datetime.strptime(span["to"], "%Y-%m-%d"))
    return list(range(first, last + 1))


def screens(record: dict) -> list[dict]:
    return record["screens"]


def shot_screens(record: dict) -> list[dict]:
    """スクリーンショット（端末層を持つ画面）だけ。冒頭の notice を除く。"""
    return [s for s in record["screens"] if "device" in s]


def entries(screen: dict) -> list[dict]:
    """画面が抱える記録（投稿 or メッセージ）を、表示順のまま返す。"""
    return screen.get("posts") or screen.get("messages") or []


def own_posts(record: dict) -> list[dict]:
    """本人の投稿だけ。DM の相手の発話は含まない。"""
    out = []
    for s in record["screens"]:
        out.extend(s.get("posts") or [])
    return out


def all_texts(record: dict) -> list[str]:
    """画面上に出る本文テキスト（冒頭の作者の言葉を除く）。"""
    out = []
    for s in record["screens"]:
        if s["kind"] == "notice":
            continue
        for e in entries(s):
            out.append(e["text"])
    return out
