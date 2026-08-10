"""C. 縦糸（spec/verification.md §2 C1..C7）"""

from life import fiscal_year, fiscal_years, own_posts, parse
import validate


def test_all_thread_rules_hold(record):
    assert validate.check_thread(record) == []


def test_thread_appears_at_most_once_a_year(record):
    for year, n in validate.thread_by_year(record).items():
        assert n <= 1, f"年度{year} に縦糸が {n} 回"


def test_thread_is_absent_in_at_least_two_years(record):
    counts = validate.thread_by_year(record)
    absent = [y for y in fiscal_years(record) if counts[y] == 0]
    assert len(absent) >= 2, f"欠落が {absent} だけ"


def test_thread_is_not_present_in_every_year(record):
    counts = validate.thread_by_year(record)
    assert any(counts[y] == 0 for y in fiscal_years(record))


def test_exactly_one_absence_is_spoken_of(record):
    mentions = [p for p in own_posts(record) if p.get("thread_mention")]
    assert len(mentions) == 1, f"言及が {len(mentions)} 回"
    year = fiscal_year(parse(mentions[0]["at"]))
    assert validate.thread_by_year(record)[year] == 0, "言及の年度に縦糸が存在する"


def test_other_absences_are_never_mentioned(record):
    """言及されていない欠落が1つ以上残っていること。"""
    counts = validate.thread_by_year(record)
    absent = {y for y in fiscal_years(record) if counts[y] == 0}
    spoken = {
        fiscal_year(parse(p["at"]))
        for p in own_posts(record)
        if p.get("thread_mention")
    }
    assert len(absent - spoken) >= 1, "全ての欠落が言及されている"


def test_every_year_has_records(record):
    """欠落が『記録が無いだけ』に退化していないこと。"""
    per_year = validate.screens_by_year(record)
    for y in fiscal_years(record):
        assert per_year[y] > 0, f"年度{y} に記録がない"


def test_motif_is_not_a_collective_cliche(record):
    motif = record["work"]["thread_motif"]
    for t in validate.TYPICAL_MOTIFS:
        assert t not in motif, f"縦糸「{motif}」が典型語「{t}」を含む"
