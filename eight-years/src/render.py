"""record.json から src/build/*.html を生成する。

日付・曜日・時刻・電池・連番はすべて record.json の値をそのまま描く。
レンダラは値を作らない（作ると検証が恒真になる）。唯一の例外は
ステータスバーの時刻で、これは shot_at から HH:MM を切り出す。
"""

from __future__ import annotations

import html
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from life import BUILD_DIR, load, parse  # noqa: E402

ASSETS = pathlib.Path(__file__).resolve().parent / "assets"

WIFI_SVG = (
    '<svg class="wifi" viewBox="0 0 16 12" fill="none" stroke="currentColor" '
    'stroke-width="1.5" stroke-linecap="round">'
    '<path d="M1 4.1C3 2.3 5.4 1.4 8 1.4s5 .9 7 2.7" opacity="{o1}"/>'
    '<path d="M3.4 6.9C4.7 5.7 6.3 5.1 8 5.1s3.3 .6 4.6 1.8" opacity="{o2}"/>'
    '<path d="M6 9.7c.6-.6 1.3-.9 2-.9s1.4.3 2 .9" opacity="{o3}"/>'
    "</svg>"
)

BOLT_SVG = (
    '<svg class="bolt" viewBox="0 0 9 12" fill="currentColor">'
    '<path d="M5.3 0 .6 6.7h3L3.2 12 8.4 5.1h-3L5.3 0z"/></svg>'
)

GEAR_SVG = (
    '<svg class="gear" viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.4">'
    '<circle cx="9" cy="9" r="2.6"/><circle cx="9" cy="9" r="6.4"/></svg>'
)


def _css(*names: str) -> str:
    return "\n".join((ASSETS / n).read_text(encoding="utf-8") for n in names)


def _esc(s: str) -> str:
    return html.escape(s, quote=False)


def status_bar(dev: dict) -> str:
    """ステータスバー。時刻は shot_at の HH:MM をそのまま出す。"""
    clock = parse(dev["shot_at"]).strftime("%H:%M")
    net, bars = dev["network"], dev["bars"]

    if net == "wifi":
        o = [1 if bars >= n else 0.25 for n in (1, 2, 3)]
        signal = f'<span class="sb-net">Wi-Fi</span>' + WIFI_SVG.format(
            o1=o[0], o2=o[1], o3=o[2]
        )
    else:
        label = {"mobile": "4G", "weak": "4G", "none": "圏外"}[net]
        pips = "".join(
            f'<i class="{"on" if n <= bars else ""}"></i>' for n in (1, 2, 3, 4)
        )
        signal = f'<div class="bars">{pips}</div><span class="sb-net">{label}</span>'

    fill = max(2.0, 22.4 * dev["battery"] / 100)
    bolt = BOLT_SVG if dev["charging"] else ""
    battery = (
        f'<div class="batt">{bolt}<span class="batt-num">{dev["battery"]}</span>'
        f'<div class="batt-shell"><div class="batt-fill" style="width:{fill:.1f}px"></div></div></div>'
    )
    return (
        f'<div class="statusbar"><span>{clock}</span>'
        f'<div class="sb-right">{signal}{battery}</div></div>'
    )


def _photo(kind: str | None) -> str:
    return f'<div class="photo {kind}"></div>' if kind else ""


def _when_v1(post: dict) -> str:
    dt = parse(post["at"])
    return f'{dt.strftime("%Y/%m/%d")} ({post["weekday"]}) {dt.strftime("%H:%M")}'


def _when_v2(post: dict) -> str:
    dt = parse(post["at"])
    return f'{dt.strftime("%Y.%m.%d")} {post["weekday"]} · {dt.strftime("%H:%M")}'


def render_notice(s: dict) -> str:
    lines = "".join(f"<p>{_esc(t)}</p>" for t in s["text"])
    return f"""<div class="device">
  <div class="notice">{lines}<div class="rule"></div>
    <div class="stamp">{_esc(s["footer"])}</div>
  </div>
</div>"""


def render_feed(s: dict) -> str:
    v1 = s["ui"] == "v1"
    star = "☆" if v1 else "◇"
    when = _when_v1 if v1 else _when_v2

    posts = "".join(
        f'<div class="post"><div class="when">{_esc(when(p))}</div>'
        f'<div class="body">{_esc(p["text"])}</div>'
        f'{_photo(p.get("photo"))}'
        f'<div class="react">{star} {p["reactions"]}</div></div>'
        for p in s["posts"]
    )

    if v1:
        head = f"""
  <div class="appbar"><span class="logo">ヌイシロ</span>{GEAR_SVG}</div>
  <div class="profile"><div class="av"></div>
    <div><div class="nm">あお</div><div class="id">{_esc(s["account"])}</div></div>
  </div>
  <div class="sortbar"><span>新しい順</span><span class="on">古い順</span></div>
  <div class="monthbar">{_esc(s["month_label"])}</div>"""
        tabs = ["記録", "さがす", "通知", "じぶん"]
        on = 3
    else:
        head = f"""
  <div class="appbar"><span class="logo">nuishiro</span>{GEAR_SVG}</div>
  <div class="profile"><div class="av"></div>
    <div><div class="nm">あお</div><div class="id">{_esc(s["account"])}</div></div>
  </div>
  <div class="seg"><span>新しい順</span><span class="on">古い順</span></div>
  <div class="monthbar">{_esc(s["month_label"])}</div>"""
        tabs = ["きろく", "つながり", "じぶん"]
        on = 2

    tabbar = '<div class="tabbar">' + "".join(
        f'<div class="{"on" if i == on else ""}"><i></i>{t}</div>'
        for i, t in enumerate(tabs)
    ) + "</div>"

    return f"""<div class="device">
  {status_bar(s["device"])}{head}
  <div class="feed">{posts}</div>
  {tabbar}
</div>"""


def render_dm(s: dict) -> str:
    rows = []
    for i, m in enumerate(s["messages"]):
        t = parse(m["at"]).strftime("%H:%M")
        av = '<div class="av"></div>' if i == 0 else ""
        cls = "row" if i == 0 else "row cont"
        if m.get("photo"):
            inner = f'<div class="bubble pic">{_photo(m["photo"])}</div>'
        else:
            inner = f'<div class="bubble">{_esc(m["text"])}</div>'
        rows.append(f'<div class="{cls}">{av}{inner}<span class="t">{t}</span></div>')
        if m.get("photo"):
            rows.append(
                f'<div class="row cont"><div class="bubble">{_esc(m["text"])}</div>'
                f'<span class="t">{t}</span></div>'
            )

    read = parse(s["read_at"]).strftime("%H:%M")
    return f"""<div class="device">
  {status_bar(s["device"])}
  <div class="dmbar"><div class="back"></div><span class="peer">{_esc(s["peer"])}</span></div>
  <div class="thread">
    <div class="daydiv">{_esc(s["date_label"])}</div>
    {"".join(rows)}
    <div class="read">既読 {read}</div>
  </div>
  <div class="composer"><div class="field">メッセージ</div><div class="send"></div></div>
</div>"""


def page(s: dict) -> str:
    if s["kind"] == "notice":
        css, body = _css("base.css", "notice.css"), render_notice(s)
    elif s["kind"] == "dm":
        css, body = _css("base.css", "v2.css"), render_dm(s)
    else:
        css, body = _css("base.css", f'{s["ui"]}.css'), render_feed(s)
    return (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        f"<style>{css}</style></head><body>{body}</body></html>"
    )


def build_all(record: dict | None = None) -> list[pathlib.Path]:
    record = record or load()
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for s in record["screens"]:
        path = BUILD_DIR / s["file"].replace(".png", ".html")
        path.write_text(page(s), encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    for p in build_all():
        print(p.name)
