"""Edge filter settings (PRD F9): pure data, stored in project.yaml as `edge_filter`.
The per-box edge check lives in core/edge_band.py (imports OpenCV)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EdgeFilterConfig(BaseModel):
    version: int = 1
    enabled: bool = False
    # Strip around each predicted box in which the edge is looked for (F9.1): `along_px` along the
    # expected edge, `across_px` across it. Narrow across, so a parallel line just outside shows up
    # but the object's interior far away does not decide.
    along_px: int = Field(default=200, ge=40, le=4096)
    across_px: int = Field(default=30, ge=8, le=1024)
    # A box may stop this far short of the line (a crack hanging into the object from the edge).
    # The strip is widened when `across_px` leaves no room beyond a line this far away.
    touch_px: int = Field(default=12, ge=0, le=256)
    # The line: thinner than ~5 px, darker than both sides by `dark_contrast` grey levels, covering at
    # least `min_length` of the strip outside the box, tilted at most `max_tilt_deg`.
    min_length: float = Field(default=0.5, gt=0.0, le=1.0)
    dark_contrast: float = Field(default=20.0, gt=0.0, le=255.0)
    max_tilt_deg: float = Field(default=5.0, ge=0.0, le=20.0)  # steeper: the line blurs into "beyond"
    # Beyond the line (background side) at most this share of the strip may be textured.
    flat_max: float = Field(default=0.05, ge=0.0, le=1.0)
