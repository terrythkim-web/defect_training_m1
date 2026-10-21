from enum import Enum


class Tool(Enum):
    SELECT = "select"
    BBOX = "bbox"
    POLYGON = "polygon"
    CROP_ROI = "crop_roi"
    SAM = "sam"
    ERASER = "eraser"
