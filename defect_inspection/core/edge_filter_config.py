"""Edge filter settings (PRD F9): pure data, stored in project.yaml as `edge_filter`.
The per-box edge check lives in core/edge_band.py (imports OpenCV)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EdgeFilterConfig(BaseModel):
    version: int = 1
    enabled: bool = False
    # Length of the strip along the expected edge on each side of the box (F9.1). Across the edge the
    # strip reaches the image border: the edge must be the outermost line (no line beyond it).
    along_px: int = Field(default=200, ge=40, le=4096)
    # A box may stop this far short of the line (a crack hanging into the object from the edge).
    touch_px: int = Field(default=12, ge=0, le=256)
    # The line: thinner than ~5 px, darker (or brighter, see `line_polarity`) than both sides by
    # `dark_contrast` grey levels, covering at least `min_length` of the strip outside the box, tilted
    # at most `max_tilt_deg`. Bright lines: a display panel's edge on a dark background.
    line_polarity: Literal["dark", "bright", "both"] = "both"
    min_length: float = Field(default=0.5, gt=0.0, le=1.0)
    dark_contrast: float = Field(default=20.0, gt=0.0, le=255.0)
    max_tilt_deg: float = Field(default=5.0, ge=0.0, le=20.0)  # steeper: the line blurs into "beyond"
    # Beyond the line (background side) at most this share of the strip may be textured.
    flat_max: float = Field(default=0.05, ge=0.0, le=1.0)
