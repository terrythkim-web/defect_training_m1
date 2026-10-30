"""Full vs Crop comparison report (PRD F8.5, F8.6): ONE self-contained HTML file. Qt-free.

- reports/<YYYYMMDD_HHMM>_full_vs_crop.html, figures (matplotlib) and example images as base64,
  so it opens in any browser on another PC. HTML only: no PDF/Markdown/CSV output.
- Sections: ① summary (recommendation + reason) ② conditions ③ metrics table ④ figures
  ⑤ missed / false detection cases ⑥ conclusion / notes.
- The experiment (run dirs, train configs, results without per-image predictions) is embedded as
  JSON (<script id="defectlab-data">), so the report list can show it and "same settings again"
  can re-run it without anything but the HTML file.
"""

from __future__ import annotations

import base64
import io
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from defect_inspection import __version__
from defect_inspection.core.compare import (
    BAR_LABELS,
    RunComparison,
    best_cells,
    match,
    metric_rows,
    recommend,
)
from defect_inspection.core.evaluate import load_training_image
from defect_inspection.core.preview import draw_comparison
from defect_inspection.core.project import Project
from defect_inspection.core.trainer import TrainConfig, load_spec
from defect_inspection.platform_utils import atomic_write_text, detect_device, pick_korean_font

REPORT_SUFFIX = "_full_vs_crop.html"
DATA_ID = "defectlab-data"
# Same order as the app's charts (ui/widgets/training_charts.SERIES): run A, B, C...
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
CASE_SIDE = 520  # px of each example image
MAX_CASES = 6


# ---- figures ---------------------------------------------------------------------------------

def setup_matplotlib() -> str | None:
    """Agg backend + a Korean font picked per OS (Malgun Gothic on Windows, Noto/Nanum elsewhere)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib import rcParams

    font = pick_korean_font()
    if font:
        rcParams["font.family"] = [font, "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False  # the Korean fonts have no U+2212
    return font


def _figure(w: float = 5.2, h: float = 3.0):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(w, h), dpi=110)
    ax = fig.add_subplot()
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    return fig, ax


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _letter(i: int) -> str:
    return chr(65 + i)


def metric_bars(results: list[RunComparison]) -> str:
    fig, ax = _figure()
    n = len(results)
    width = 0.8 / max(n, 1)
    for i, r in enumerate(results):
        xs = [k - 0.4 + width * (i + 0.5) for k in range(len(BAR_LABELS))]
        ax.bar(xs, [getattr(r.metrics, a) for a, _ in BAR_LABELS], width * 0.9, color=COLORS[i % len(COLORS)])
    ax.set_xticks(range(len(BAR_LABELS)), [name for _, name in BAR_LABELS])
    ax.set_ylim(0, 1)
    ax.set_title("지표 (같은 Val 원본, 같은 conf)", fontsize=10)
    return _png(fig)


def size_bars(results: list[RunComparison]) -> str:
    fig, ax = _figure()
    bins = [("small", "작음"), ("medium", "중간"), ("large", "큼")]
    width = 0.8 / max(len(results), 1)
    for i, r in enumerate(results):
        xs = [k - 0.4 + width * (i + 0.5) for k in range(3)]
        ax.bar(xs, [r.metrics.size_recall[b] or 0.0 for b, _ in bins], width * 0.9, color=COLORS[i % len(COLORS)])
    counts = results[0].metrics.size_counts if results else {}
    ax.set_xticks(range(3), [f"{name}\n({counts.get(b, 0)}개)" for b, name in bins])
    ax.set_ylim(0, 1)
    ax.set_title("결함 크기별 Recall", fontsize=10)
    return _png(fig)


def pr_figure(results: list[RunComparison]) -> str:
    fig, ax = _figure(4.2, 3.4)
    for i, r in enumerate(results):
        rec, prec = r.metrics.pr_curve
        ax.plot(rec, prec, color=COLORS[i % len(COLORS)], linewidth=2, marker="o" if len(rec) < 30 else None,
                markersize=3)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall", color=TEXT_SECONDARY)
    ax.set_ylabel("precision", color=TEXT_SECONDARY)
    ax.set_title("PR curve (IoU 0.5)", fontsize=10)
    return _png(fig)


def series_figure(results: list[RunComparison], keys: list[str], title: str, y_label: str) -> str | None:
    fig, ax = _figure()
    drawn = False
    for i, r in enumerate(results):
        values = r.series.get("values", {})
        key = next((k for k in keys if k in values), None)
        if key is None:
            continue
        pts = [(e, v) for e, v in zip(r.series["epochs"], values[key]) if v is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS[i % len(COLORS)], linewidth=2)
        drawn = True
    if not drawn:
        return None
    ax.set_xlabel("epoch", color=TEXT_SECONDARY)
    ax.set_ylabel(y_label, color=TEXT_SECONDARY)
    ax.set_title(title, fontsize=10)
    return _png(fig)


# ---- cases (⑤) ------------------------------------------------------------------------------------

def _jpeg(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def pick_cases(results: list[RunComparison], limit: int = MAX_CASES) -> list[str]:
    """Images where some model missed or falsely detected something; images where the models
    disagree come first, then the ones with the most errors."""
    paths = [im.path for im in results[0].images] if results else []
    score = {}
    for path in paths:
        errors, found = 0, []
        for r in results:
            im = next((x for x in r.images if x.path == path), None)
            if im is None:
                continue
            matched, missed = match(im.ground_truth, im.above(r.conf))
            errors += sum(missed) + len(matched) - sum(matched)
            found.append((sum(matched), len(matched) - sum(matched)))
        if errors:
            score[path] = (len(set(found)) > 1, errors)
    return sorted(score, key=lambda p: score[p], reverse=True)[:limit]


def case_images(project: Project, results: list[RunComparison], paths: list[str]) -> list[dict[str, Any]]:
    cases = []
    for path in paths:
        image = load_training_image(project, path)
        if image is None:
            continue
        cells = []
        for i, r in enumerate(results):
            im = next((x for x in r.images if x.path == path), None)
            if im is None:
                continue
            preds = im.above(r.conf)
            matched, missed = match(im.ground_truth, preds)
            drawn = draw_comparison(image, im.ground_truth, [p.shape for p in preds], max_side=CASE_SIDE)
            cells.append({"letter": _letter(i), "color": COLORS[i % len(COLORS)], "img": _jpeg(drawn),
                          "tp": sum(matched), "fp": len(matched) - sum(matched), "fn": sum(missed)})
        cases.append({"path": path, "cells": cells})
    return cases


# ---- conditions (②) ---------------------------------------------------------------------------------

def environment_info() -> dict[str, str]:
    info = {"OS": f"{platform.system()} {platform.release()}", "Python": sys.version.split()[0],
            "DefectLab": __version__, "장치": detect_device()}
    for name in ("torch", "ultralytics", "onnxruntime", "albumentations"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "?")
        except ImportError:
            info[name] = "-"
    return info


def run_conditions(project: Project, r: RunComparison, run_dir: Path) -> dict[str, Any]:
    spec = load_spec(run_dir)
    t = spec.train
    rows = {
        "데이터셋": t.dataset, "모델": t.model, "epochs": t.epochs, "입력 크기": t.imgsz,
        "batch": "자동" if t.batch == -1 else t.batch, "lr0": t.lr0 or "기본", "patience": t.patience,
        "seed": t.seed, "온라인 증강": "켬" if t.online_augment else "끔", "사전학습 가중치": "사용" if t.pretrained else "안 함",
        "장치": t.device,
    }
    tiling = spec.tiling()
    if tiling is not None:
        crop = tiling.crop
        kinds = [name for key, name in (("use_roi", "수동 ROI"), ("use_defect", "결함 중심"),
                                        ("use_background", "배경"), ("use_tiles", "타일")) if crop.get(key)]
        rows["crop 방식"] = ", ".join(kinds) or "-"
        if crop.get("use_defect") or crop.get("use_background"):
            rows["crop 크기"] = f"{crop.get('crop_size')} px · 라벨당 {crop.get('per_defect')} · 흔들기 " \
                              f"{round(100 * (crop.get('jitter') or 0))}% · 배경 비율 {crop.get('background_ratio')}"
        rows["타일링 추론"] = f"{tiling.tile_size} px, overlap {tiling.overlap_px} px"
        rows["crop 증강 수"] = crop.get("multiplier")
    if r.edge_filter:
        from defect_inspection.core.edge_filter import settings_text
        from defect_inspection.core.edge_filter_config import EdgeFilterConfig

        rows["테두리 필터"] = settings_text(EdgeFilterConfig.model_validate(r.edge_filter))
    return rows


def augment_summary(project: Project) -> str:
    aug = project.config.augment
    parts = [f"{k} (p {aug.setting(k).p:.2f}, 강도 {aug.setting(k).strength:.2f})" for k in aug.enabled_keys()]
    text = f"원본당 {aug.multiplier}장 · 라벨 유지 ≥ {aug.min_visibility:.0%}, ≥ {aug.min_area_px:.0f} px²"
    if aug.copy_paste.enabled:
        text += f" · Copy-Paste p {aug.copy_paste.p}"
    return text + (" · " + ", ".join(parts) if parts else " · 변환 없음")


# ---- HTML -----------------------------------------------------------------------------------------

TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
:root { --text:#0b0b0b; --muted:#52514e; --line:#e6e5e1; --bg:#fcfcfb; --accent:#2a78d6; }
body { font-family: "Malgun Gothic","Apple SD Gothic Neo","Noto Sans CJK KR","Noto Sans KR",sans-serif;
       color:var(--text); background:var(--bg); margin:0; padding:24px 16px; line-height:1.5; }
main { max-width:1100px; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 4px; } h2 { font-size:1.15rem; margin:32px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px; }
.muted { color:var(--muted); font-size:.9rem; }
.rec { border-left:4px solid var(--accent); background:#fff; padding:12px 16px; margin:16px 0; }
.rec b { font-size:1.1rem; }
table { border-collapse:collapse; width:100%; background:#fff; font-size:.9rem; margin:8px 0; }
th, td { border:1px solid var(--line); padding:4px 8px; text-align:left; vertical-align:top; }
td.num { text-align:right; font-variant-numeric: tabular-nums; } td.best { font-weight:700; }
.wrap { overflow-x:auto; }
.chip { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }
.figs { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; }
.figs img, .case img { max-width:100%; height:auto; background:#fff; border:1px solid var(--line); }
.case { margin:12px 0 20px; } .case .cells { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:8px; }
.memo { min-height:80px; background:#fff; border:1px solid var(--line); padding:8px; white-space:pre-wrap; }
@media print { .noprint { display:none; } }
</style></head>
<body><main>
<h1>{{ title }}</h1>
<div class="muted">{{ project }} · 작성 {{ created }} · DefectLab {{ version }}</div>

<h2>① 요약</h2>
<div class="rec">추천: <b>{{ runs[recommended].letter }} · {{ runs[recommended].label }}</b><br>{{ reason }}</div>
<div class="muted">우선 지표 {{ priority }} (차이가 1%p 이내면 CPU 추론이 빠른 쪽) · Val 원본 {{ n_images }}장 (NG {{ n_ng }}, OK {{ n_ok }})
 · conf {{ conf }} · NMS IoU {{ iou }} · 박스 기준 IoU ≥ 0.5, 같은 클래스 · Crop 모델은 타일링 추론</div>
<p>{% for r in runs %}<span class="chip" style="background:{{ r.color }}"></span>{{ r.letter }}: {{ r.label }}&nbsp;&nbsp; {% endfor %}</p>

<h2>② 실험 조건</h2>
<div class="wrap"><table>
<tr><th></th>{% for r in runs %}<th><span class="chip" style="background:{{ r.color }}"></span>{{ r.letter }} · {{ r.short }}</th>{% endfor %}</tr>
{% for key in condition_keys %}<tr><th>{{ key }}</th>{% for r in runs %}<td>{{ r.conditions.get(key, "-") }}</td>{% endfor %}</tr>
{% endfor %}</table></div>
<p class="muted">증강(프로젝트 현재 설정): {{ augment }}</p>
<table>{% for k, v in environment.items() %}<tr><th style="width:160px">{{ k }}</th><td>{{ v }}</td></tr>{% endfor %}</table>

<h2>③ 지표 비교</h2>
<div class="wrap"><table>
<tr><th></th>{% for r in runs %}<th><span class="chip" style="background:{{ r.color }}"></span>{{ r.letter }} · {{ r.short }}</th>{% endfor %}</tr>
{% for row in rows %}<tr><th>{{ row.label }}</th>{% for t in row.texts %}<td class="num{% if loop.index0 in row.best %} best{% endif %}">{{ t }}</td>{% endfor %}</tr>
{% endfor %}</table></div>
<p class="muted">굵게 = 가장 좋은 값. 참고 행의 자체 val 지표는 run마다 평가 이미지가 달라(전체 vs crop) 직접 비교하지 않습니다.</p>

<h2>④ 그래프</h2>
<div class="figs">{% for f in figures %}<img alt="{{ f.alt }}" src="data:image/png;base64,{{ f.png }}">{% endfor %}</div>

<h2>⑤ 미검 / 과검 사례</h2>
{% if cases %}<p class="muted">파랑 = 정답 라벨, 주황 = 예측(점수). 모델 간 결과가 다른 이미지부터.</p>
{% for c in cases %}<div class="case"><b>{{ c.path }}</b><div class="cells">
{% for cell in c.cells %}<div><span class="chip" style="background:{{ cell.color }}"></span>{{ cell.letter }} — 맞음 {{ cell.tp }} · 과검 {{ cell.fp }} · 미검 {{ cell.fn }}<br>
<img alt="{{ c.path }} {{ cell.letter }}" src="data:image/jpeg;base64,{{ cell.img }}"></div>{% endfor %}
</div></div>{% endfor %}
{% else %}<p>모든 모델이 Val 이미지에서 미검·과검 없이 맞혔습니다.</p>{% endif %}

<h2>⑥ 결론 / 메모</h2>
<div class="memo" contenteditable="true">{{ memo }}</div>
<p class="muted noprint">위 칸은 브라우저에서 바로 고칠 수 있습니다. 고친 내용을 남기려면 브라우저의 "다른 이름으로 저장"(Ctrl+S, Mac은 ⌘S)으로 저장하세요.</p>

<script type="application/json" id="{{ data_id }}">{{ data_json }}</script>
</main></body></html>
"""


def report_path(project: Project, now: datetime | None = None) -> Path:
    """reports/<YYYYMMDD_HHMM>_full_vs_crop.html; a second report in the same minute gets _2, _3…"""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M")
    path = project.reports_dir / f"{stamp}{REPORT_SUFFIX}"
    n = 1
    while path.exists():
        n += 1
        path = project.reports_dir / f"{stamp}_{n}{REPORT_SUFFIX}"
    return path


def _safe_json(data: dict[str, Any]) -> str:
    """JSON that cannot end the <script> element early."""
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def build_report(project: Project, results: list[RunComparison], run_dirs: list[Path], priority: str = "mAP50-95",
                 memo: str = "", now: datetime | None = None, out_path: Path | None = None) -> Path:
    """Write the HTML report and return its path."""
    from jinja2 import Environment

    if len(results) < 2 or len(results) != len(run_dirs):
        raise ValueError("비교 결과가 2개 이상 필요합니다")
    setup_matplotlib()
    now = now or datetime.now()
    recommended, reason = recommend(results, priority)
    runs = []
    for i, (r, d) in enumerate(zip(results, run_dirs)):
        runs.append({"letter": _letter(i), "color": COLORS[i % len(COLORS)], "label": r.label,
                     "short": f"{r.training.dataset.capitalize()} · {r.training.model}",
                     "conditions": run_conditions(project, r, Path(d))})
    keys: list[str] = []
    for r in runs:
        keys += [k for k in r["conditions"] if k not in keys]
    rows = [{"label": row.label, "texts": row.texts, "best": best_cells(row)} for row in metric_rows(results)]
    figures = [{"alt": "지표 막대", "png": metric_bars(results)}, {"alt": "PR curve", "png": pr_figure(results)},
               {"alt": "크기별 Recall", "png": size_bars(results)}]
    for keys_, title, label in ((["metrics/mAP50-95(B)", "metrics/mAP50-95(M)"], "mAP50-95 (각 run 자체 val)", "mAP50-95"),
                                (["val/box_loss", "train/box_loss"], "box loss", "loss")):
        png = series_figure(results, keys_, title, label)
        if png:
            figures.append({"alt": title, "png": png})
    m0 = results[0].metrics
    data = {
        "version": 1, "app": "DefectLab", "created": now.isoformat(timespec="seconds"), "project": project.config.name,
        "priority": priority, "recommended": recommended, "reason": reason, "conf": results[0].conf,
        "iou": results[0].iou, "memo": memo,
        "runs": [{"run_id": r.run_id, "run_dir": str(Path(d)), "label": r.label,
                  "train": load_spec(Path(d)).train.model_dump(mode="json"), "summary": r.summary()}
                 for r, d in zip(results, run_dirs)],
    }
    title = "Full vs Crop 비교 리포트" if {r.training.dataset for r in results} == {"full", "crop"} else "모델 비교 리포트"
    page = Environment(autoescape=True).from_string(TEMPLATE).render(
        title=title, project=project.config.name, created=now.strftime("%Y-%m-%d %H:%M"), version=__version__,
        runs=runs, recommended=recommended, reason=reason, priority=priority, n_images=m0.images,
        n_ng=m0.ng_images, n_ok=m0.ok_images, conf=f"{results[0].conf:.2f}", iou=f"{results[0].iou:.2f}",
        condition_keys=keys, augment=augment_summary(project), environment=environment_info(), rows=rows,
        figures=figures, cases=case_images(project, results, pick_cases(results)), memo=memo,
        data_id=DATA_ID, data_json=_Markup(_safe_json(data)),
    )
    path = out_path or report_path(project, now)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, page)
    return path


class _Markup(str):
    """Pre-escaped text for the template (the JSON must not be HTML-escaped inside <script>)."""

    def __html__(self) -> str:
        return str(self)


# ---- history (F8.6) -------------------------------------------------------------------------------

@dataclass
class ReportInfo:
    path: Path
    created: str
    title: str
    runs: list[dict[str, Any]]  # run_id, run_dir, label, train
    recommended: int | None
    priority: str

    @property
    def configs(self) -> list[TrainConfig]:
        """Training settings of every compared run, to run the same experiment again."""
        return [TrainConfig.model_validate(r["train"]) for r in self.runs if "train" in r]


def read_report_data(path: Path) -> dict[str, Any] | None:
    """The JSON embedded by build_report, or None for files that are not DefectLab reports."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    marker = f'<script type="application/json" id="{DATA_ID}">'
    start = text.find(marker)
    if start < 0:
        return None
    end = text.find("</script>", start)
    try:
        return json.loads(text[start + len(marker):end])
    except ValueError:
        return None


def list_reports(project: Project) -> list[ReportInfo]:
    """Reports of a project, newest first."""
    out = []
    if project.reports_dir.is_dir():
        for path in project.reports_dir.glob(f"*{REPORT_SUFFIX}"):
            data = read_report_data(path) or {}
            title_runs = data.get("runs", [])
            out.append(ReportInfo(path, data.get("created", ""), path.stem, title_runs,
                                  data.get("recommended"), data.get("priority", "")))
    return sorted(out, key=lambda r: (r.created, r.path.name), reverse=True)


def delete_report(path: Path) -> None:
    Path(path).unlink(missing_ok=True)
