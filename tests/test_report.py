"""HTML comparison report and report history (PROMPTS step 20, PRD F8.5, F8.6)."""

import base64
import re
from datetime import datetime

import pytest

from defect_inspection.core import compare as C
from defect_inspection.core import report as R
from test_compare import blob_factory, make_project_with_runs


@pytest.fixture
def compared(tmp_path):
    p, runs = make_project_with_runs(tmp_path)
    results = C.compare_runs(p, runs, make_predictor=blob_factory(), devices=[("CPU", "cpu")])
    return p, runs, results


NOW = datetime(2026, 10, 19, 14, 5, 33)


def test_report_file_name_and_sections(compared):
    p, runs, results = compared
    path = R.build_report(p, results, runs, memo="라인 A 결론: Crop 채택 <b>?</b>", now=NOW)
    assert path == p.reports_dir / "20261019_1405_full_vs_crop.html"
    html = path.read_text(encoding="utf-8")
    for section in ("① 요약", "② 실험 조건", "③ 지표 비교", "④ 그래프", "⑤ 미검 / 과검 사례", "⑥ 결론 / 메모"):
        assert section in html
    assert "Full vs Crop 비교 리포트" in html
    assert "라인 A 결론: Crop 채택 &lt;b&gt;?&lt;/b&gt;" in html  # memo is escaped
    assert "타일링 추론" in html and "128 px, overlap 26 px" in html
    # self-contained: figures inline, nothing loaded from elsewhere
    pngs = re.findall(r'src="data:image/png;base64,([^"]+)"', html)
    assert len(pngs) >= 4
    assert base64.b64decode(pngs[0])[:8] == b"\x89PNG\r\n\x1a\n"
    assert not re.search(r'(src|href)="(http|file|/)', html)
    # a second report in the same minute does not overwrite the first
    assert R.build_report(p, results, runs, now=NOW).name == "20261019_1405_2_full_vs_crop.html"


def test_metric_values_match_comparison(compared):
    p, runs, results = compared
    html = R.build_report(p, results, runs, now=NOW).read_text(encoding="utf-8")
    for row in C.metric_rows(results):
        row_html = re.search(r"<tr><th>" + re.escape(row.label.replace("&", "&amp;").replace("<", "&lt;")
                                                     .replace(">", "&gt;")) + r"</th>(.*?)</tr>", html)
        assert row_html, row.label
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html.group(1))
        assert [c.replace("&gt;", ">") for c in cells] == row.texts, row.label


def test_recommendation_and_priority(compared):
    p, runs, results = compared
    results[1].metrics.recall = results[0].metrics.recall + 0.2
    results[0].metrics.map50_95 = results[1].metrics.map50_95 + 0.2
    html = R.build_report(p, results, runs, priority="Recall", now=NOW).read_text(encoding="utf-8")
    data = R.read_report_data(R.list_reports(p)[0].path)
    assert data["recommended"] == 1 and data["priority"] == "Recall" and "Recall가 가장 높음" in data["reason"]
    assert f"추천: <b>B · {results[1].label}</b>" in html


def test_cases_show_errors(compared):
    p, runs, results = compared
    im = results[0].images[0]
    im.predictions = []  # model A misses everything on this image
    html = R.build_report(p, results, runs, now=NOW).read_text(encoding="utf-8")
    assert f"<b>{im.path}</b>" in html and "미검 1" in html
    assert len(re.findall(r'src="data:image/jpeg;base64,', html)) == 2  # A and B side by side


def test_history_list_read_delete(compared):
    p, runs, results = compared
    first = R.build_report(p, results, runs, now=NOW)
    second = R.build_report(p, results, runs, now=datetime(2026, 10, 20, 9, 0))
    (p.reports_dir / "notes_full_vs_crop.html").write_text("<html>사람이 만든 파일</html>", encoding="utf-8")
    reports = R.list_reports(p)
    assert [r.path for r in reports[:2]] == [second, first]
    info = reports[0]
    assert [r["run_id"] for r in info.runs] == [C.load_spec(d).run_id for d in runs]
    configs = info.configs
    assert [c.dataset for c in configs] == ["full", "crop"] and configs[0].epochs == 3
    assert R.read_report_data(p.reports_dir / "notes_full_vs_crop.html") is None
    R.delete_report(first)
    assert first not in [r.path for r in R.list_reports(p)]


def test_needs_two_results(compared):
    p, runs, results = compared
    with pytest.raises(ValueError):
        R.build_report(p, results[:1], runs[:1])


def test_korean_font_chosen_per_os(monkeypatch):
    import defect_inspection.platform_utils as pu

    monkeypatch.setattr(pu, "IS_WINDOWS", True)
    assert pu.korean_font_candidates()[0] == "Malgun Gothic"
    monkeypatch.setattr(pu, "IS_WINDOWS", False)
    assert "Noto Sans CJK KR" in pu.korean_font_candidates()
    from matplotlib import rcParams

    font = R.setup_matplotlib()
    if font is not None:
        assert rcParams["font.family"][0] == font
