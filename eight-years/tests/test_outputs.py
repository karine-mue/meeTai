"""E. 成果物（spec/verification.md §2 E1..E3）"""

import json
import struct

import pytest

from life import OUTPUT_DIR, SCALE, VIEWPORT_H, VIEWPORT_W

MANIFEST = OUTPUT_DIR / "manifest.json"


def png_size(path):
    """PNG の IHDR を直接読む。画像ライブラリに依存しない。"""
    with open(path, "rb") as fh:
        head = fh.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} が PNG でない"
    return struct.unpack(">II", head[16:24])


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.exists():
        pytest.fail("output/manifest.json がない。src/shoot.py を先に実行すること")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_e1_one_png_per_screen(record):
    pngs = sorted(p.name for p in OUTPUT_DIR.glob("*.png"))
    expected = sorted(s["file"] for s in record["screens"])
    assert pngs == expected
    assert 10 <= len(pngs) <= 12, f"{len(pngs)} 枚"


def test_e2_every_png_has_the_declared_size(record):
    for s in record["screens"]:
        w, h = png_size(OUTPUT_DIR / s["file"])
        assert (w, h) == (VIEWPORT_W * SCALE, VIEWPORT_H * SCALE), f'{s["file"]} が {w}x{h}'


def test_e3_no_screen_overflows_the_viewport(manifest):
    bad = [e["file"] for e in manifest["screens"] if e["overflow_y"] or e["overflow_x"]]
    assert bad == [], f"画面が溢れて切れている: {bad}"


def test_manifest_covers_every_screen(record, manifest):
    assert [e["file"] for e in manifest["screens"]] == [s["file"] for s in record["screens"]]


def test_pngs_are_not_blank(record):
    """真っ白/真っ黒な失敗出力を弾く粗い下限。"""
    for s in record["screens"]:
        size = (OUTPUT_DIR / s["file"]).stat().st_size
        assert size > 4000, f'{s["file"]} が {size} バイトしかない'
