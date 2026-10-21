"""Live training charts (PRD F4.6): one loss chart and one metrics chart, never a dual axis.

Colour follows the entity in a fixed order (validated reference palette, light mode):
- Loss chart: colour = loss component (box, cls, dfl, seg), line style = train (solid) / val (dashed).
- Metrics chart: precision, recall, mAP50, mAP50-95; segmentation mask metrics reuse the box
  metric's colour, dashed.
A hover crosshair shows every value at the nearest epoch. Legends sit in a text row above each
chart (never inside the plot, where they covered line ends); text is in text colours, the colour
lives only in the line sample.
"""

from __future__ import annotations

import re

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from defect_inspection.core.trainer import EpochSeries

# Reference categorical palette (dataviz skill), slots 1-8 in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_ALPHA = 0.15

LOSS_COMPONENTS = ["box", "cls", "dfl", "seg"]  # colour slots 1-4; others follow
METRICS = [("precision", "Precision"), ("recall", "Recall"), ("mAP50", "mAP50"), ("mAP50-95", "mAP50-95")]
_LOSS_RE = re.compile(r"^(train|val)/(\w+?)_loss$")
_METRIC_RE = re.compile(r"^metrics/(precision|recall|mAP50|mAP50-95)\((B|M)\)$")


def loss_style(key: str) -> tuple[str, str, Qt.PenStyle] | None:
    """(legend name, colour, line style) for a loss key like 'val/cls_loss'."""
    m = _LOSS_RE.match(key)
    if not m:
        return None
    split, comp = m.groups()
    idx = LOSS_COMPONENTS.index(comp) if comp in LOSS_COMPONENTS else len(LOSS_COMPONENTS)
    style = Qt.PenStyle.SolidLine if split == "train" else Qt.PenStyle.DashLine
    return f"{comp} ({split})", SERIES[idx % len(SERIES)], style


def metric_style(key: str) -> tuple[str, str, Qt.PenStyle] | None:
    """(legend name, colour, line style) for 'metrics/mAP50(B)' etc.; mask (M) metrics dashed."""
    m = _METRIC_RE.match(key)
    if not m:
        return None
    name, kind = m.groups()
    idx = [k for k, _ in METRICS].index(name)
    label = dict(METRICS)[name] + (" (mask)" if kind == "M" else "")
    return label, SERIES[idx], Qt.PenStyle.SolidLine if kind == "B" else Qt.PenStyle.DashLine


def _sort_key(key: str) -> tuple:
    m = _LOSS_RE.match(key)
    if m:
        split, comp = m.groups()
        return (0, LOSS_COMPONENTS.index(comp) if comp in LOSS_COMPONENTS else 9, split != "train")
    m = _METRIC_RE.match(key)
    if m:
        return (1, [k for k, _ in METRICS].index(m.group(1)), m.group(2) != "B")
    return (2, key)


class _Chart(pg.PlotWidget):
    def __init__(self, title: str, y_label: str) -> None:
        super().__init__(background=SURFACE)
        self.setTitle(title, color=TEXT_PRIMARY, size="10pt")
        self.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        for axis in ("left", "bottom"):
            ax = self.getAxis(axis)
            ax.setPen(pg.mkPen(TEXT_SECONDARY, width=1))
            ax.setTextPen(pg.mkPen(TEXT_SECONDARY))
        self.setLabel("bottom", "epoch", color=TEXT_SECONDARY)
        self.setLabel("left", y_label, color=TEXT_SECONDARY)
        self.curves: dict[str, pg.PlotDataItem] = {}
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()

        self.crosshair = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(TEXT_SECONDARY, width=1, style=Qt.PenStyle.DotLine))
        self.readout = pg.TextItem(color=TEXT_PRIMARY, fill=pg.mkBrush(SURFACE + "f0"), border=pg.mkPen(TEXT_SECONDARY), anchor=(0, 0))
        for item in (self.crosshair, self.readout):
            item.setZValue(100)
            item.hide()
            self.addItem(item, ignoreBounds=True)

    def set_curve(self, key: str, name: str, color: str, style: Qt.PenStyle, x: list, y: list) -> None:
        pts = [(a, b) for a, b in zip(x, y) if b is not None]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if key not in self.curves:
            pen = pg.mkPen(color, width=2, style=style)
            self.curves[key] = self.plot(xs, ys, pen=pen)
            self.curves[key].series_name = name
            self.curves[key].legend_style = (color, style)
        else:
            self.curves[key].setData(xs, ys)

    def clear_curves(self) -> None:
        for curve in self.curves.values():
            self.removeItem(curve)
        self.curves = {}
        self.crosshair.hide()
        self.readout.hide()

    def legend_html(self) -> str:
        parts = []
        for curve in self.curves.values():
            color, style = curve.legend_style
            sample = "━━" if style == Qt.PenStyle.SolidLine else "╺╺╺"
            parts.append(f'<span style="color:{color}; font-weight:bold">{sample}</span>'
                         f'&nbsp;<span style="color:{TEXT_SECONDARY}">{curve.series_name}</span>')
        return "&nbsp;&nbsp;&nbsp; ".join(parts)

    def hover_text(self, epoch: int, series: EpochSeries) -> str:
        if epoch not in series.epochs:
            return ""
        i = series.epochs.index(epoch)
        lines = [f"epoch {epoch}"]
        for key, curve in self.curves.items():
            v = series.values.get(key, [None] * (i + 1))[i]
            if v is not None:
                lines.append(f"{curve.series_name}: {v:.4f}")
        return "\n".join(lines)


class TrainingCharts(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=True)
        self.loss = _Chart("Loss  (실선 train · 점선 val)", "loss")
        self.metrics = _Chart("검증 지표", "")
        self.metrics.setYRange(0, 1, padding=0.02)
        self.series = EpochSeries()
        self.loss_legend, self.metrics_legend = QLabel(), QLabel()
        for lbl in (self.loss_legend, self.metrics_legend):
            lbl.setWordWrap(True)
            lbl.setTextFormat(Qt.TextFormat.RichText)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.loss_legend)
        layout.addWidget(self.loss, 1)
        layout.addWidget(self.metrics_legend)
        layout.addWidget(self.metrics, 1)
        self._proxies = [
            pg.SignalProxy(chart.scene().sigMouseMoved, rateLimit=30, slot=lambda ev, c=chart: self._on_hover(c, ev[0]))
            for chart in (self.loss, self.metrics)
        ]

    def clear(self) -> None:
        self.series = EpochSeries()
        self.loss.clear_curves()
        self.metrics.clear_curves()
        self.loss_legend.clear()
        self.metrics_legend.clear()

    def show_series(self, series: EpochSeries) -> None:
        self.series = series
        self._redraw()

    def add_event(self, event: dict) -> None:
        self.series.add(event)
        if event.get("type") == "epoch":
            self._redraw()

    def _redraw(self) -> None:
        x = self.series.epochs
        for key in sorted(self.series.values, key=_sort_key):
            y = self.series.values[key]
            if (s := loss_style(key)) is not None:
                self.loss.set_curve(key, *s, x, y)
            elif (s := metric_style(key)) is not None:
                self.metrics.set_curve(key, *s, x, y)
        self.loss_legend.setText(self.loss.legend_html())
        self.metrics_legend.setText(self.metrics.legend_html())

    def _on_hover(self, chart: _Chart, pos) -> None:
        if not self.series.epochs or not chart.sceneBoundingRect().contains(pos):
            chart.crosshair.hide()
            chart.readout.hide()
            return
        x = chart.getPlotItem().vb.mapSceneToView(pos).x()
        epoch = min(self.series.epochs, key=lambda e: abs(e - x))
        text = chart.hover_text(epoch, self.series)
        if not text:
            return
        chart.crosshair.setPos(epoch)
        chart.readout.setText(text)
        (x0, x1), (y0, y1) = chart.getPlotItem().vb.viewRange()
        chart.readout.setAnchor((1, 0) if epoch > (x0 + x1) / 2 else (0, 0))
        chart.readout.setPos(epoch, y1)
        chart.crosshair.show()
        chart.readout.show()
