"""src/build/*.html を output/*.png に落とす。

同時に、各画面の実測寸法と溢れの有無を output/manifest.json に書く。
「レイアウトが 844px に収まっているか」はテストから機械的に検査したいので、
ここでしか取れない値（body の scrollHeight）を残しておく。
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from life import BUILD_DIR, OUTPUT_DIR, SCALE, VIEWPORT_H, VIEWPORT_W, load  # noqa: E402
from render import build_all  # noqa: E402

CHROMIUM = "/opt/pw-browsers/chromium"


def _executable() -> str | None:
    for cand in pathlib.Path("/opt/pw-browsers").glob("chromium*/chrome-linux/chrome"):
        return str(cand)
    return None


def shoot(record: dict | None = None) -> dict:
    from playwright.sync_api import sync_playwright

    record = record or load()
    build_all(record)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    with sync_playwright() as pw:
        kwargs = {}
        exe = _executable()
        if exe:
            kwargs["executable_path"] = exe
        browser = pw.chromium.launch(**kwargs)
        ctx = browser.new_context(
            viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
            device_scale_factor=SCALE,
        )
        page = ctx.new_page()
        for s in record["screens"]:
            src = BUILD_DIR / s["file"].replace(".png", ".html")
            page.goto(src.as_uri())
            page.wait_for_load_state("networkidle")
            metrics = page.evaluate(
                """() => ({
                    scroll_h: document.body.scrollHeight,
                    scroll_w: document.body.scrollWidth,
                    device_h: document.querySelector('.device').getBoundingClientRect().height,
                })"""
            )
            dst = OUTPUT_DIR / s["file"]
            page.screenshot(
                path=str(dst),
                clip={"x": 0, "y": 0, "width": VIEWPORT_W, "height": VIEWPORT_H},
            )
            entries.append(
                {
                    "index": s["index"],
                    "file": s["file"],
                    "css_width": VIEWPORT_W,
                    "css_height": VIEWPORT_H,
                    "px_width": VIEWPORT_W * SCALE,
                    "px_height": VIEWPORT_H * SCALE,
                    "body_scroll_height": metrics["scroll_h"],
                    "body_scroll_width": metrics["scroll_w"],
                    "device_height": round(metrics["device_h"], 2),
                    "overflow_y": metrics["scroll_h"] > VIEWPORT_H,
                    "overflow_x": metrics["scroll_w"] > VIEWPORT_W,
                }
            )
            print(
                f'{s["file"]:>22}  scrollH={metrics["scroll_h"]:>5}  '
                f'{"OVERFLOW" if metrics["scroll_h"] > VIEWPORT_H else "ok"}'
            )
        browser.close()

    manifest = {"viewport": [VIEWPORT_W, VIEWPORT_H], "scale": SCALE, "screens": entries}
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    m = shoot()
    bad = [e["file"] for e in m["screens"] if e["overflow_y"] or e["overflow_x"]]
    print(f'\n{len(m["screens"])} 枚。溢れ: {bad or "なし"}')
