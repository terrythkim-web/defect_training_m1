from enum import Enum


class Tool(Enum):
    SELECT = "select"
    BBOX = "bbox"
    POLYGON = "polygon"
    BRUSH = "brush"
    CROP_ROI = "crop_roi"
    SAM = "sam"
    ERASER = "eraser"
