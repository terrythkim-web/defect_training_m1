"""Augmentation pipeline (PROMPTS step 8, PRD F3.1/F3.3/F3.4/F3.6)."""

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from defect_inspection.core.annotation import BBox, ImageAnnotations, Polygon
from defect_inspection.core.augment import (
    BUILTIN_PRESETS,
    TRANSFORM_KEYS,
    TRANSFORMS,
    AugmentConfig,
    Augmenter,
    TransformSetting,
    available_transforms,
    load_presets,
    save_preset,
)

W, H = 320, 240
GEOMETRIC = [t.key for t in TRANSFORMS if t.geometric]
PHOTOMETRIC = [t.key for t in TRANSFORMS if not t.geometric]


def only(key, p=1.0, strength=1.0, **kw):
    return AugmentConfig(transforms={key: TransformSetting(enabled=True, p=p, strength=strength)}, **kw)


def ann(*shapes, w=W, h=H):
    return ImageAnnotations(image="a.png", width=w, height=h, shapes=list(shapes))


def rasterize(shape, w=W, h=H):
    """Filled white shape on black, pixel-exact for continuous coordinates."""
    img = np.zeros((h, w), np.uint8)
    pts = np.array(shape.to_points(), np.float32) - 0.5  # continuous -> pixel centres
    cv2.fillPoly(img, [np.round(pts * 16).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=4)
    return img


def mask_bounds(img):
    ys, xs = np.nonzero(img > 127)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


BOX = BBox(class_id=0, x1=100, y1=70, x2=180, y2=130)
POLY = Polygon(class_id=1, points=[(110, 60), (210, 90), (170, 170), (120, 150)])


# ---- label / pixel agreement for every geometric transform --------------------------

@pytest.mark.parametrize("key", GEOMETRIC)
@pytest.mark.parametrize("shape", [BOX, POLY], ids=["bbox", "polygon"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_labels_follow_pixels(key, shape, seed):
    aug = Augmenter(only(key, strength=0.6, min_visibility=0.0, min_area_px=0), channels=1, seed=seed)
    result = aug(rasterize(shape), ann(shape))
    if not result.annotations.shapes:
        assert result.dropped == 1 and (result.image > 127).sum() == 0  # moved fully off-image
        return
    out = result.annotations.shapes[0]
    assert type(out) is type(shape)
    assert out.bounds() == pytest.approx(mask_bounds(result.image), abs=2.0)
    if isinstance(shape, Polygon):
        drawn = rasterize(out, *result.image.shape[::-1]) > 127
        actual = result.image > 127
        iou = (drawn & actual).sum() / (drawn | actual).sum()
        assert iou > 0.93


def test_hflip_is_exact():
    aug = Augmenter(only("hflip"), seed=0)
    r = aug(np.zeros((H, W), np.uint8), ann(BBox(class_id=0, x1=0, y1=0, x2=10, y2=10)))
    assert r.annotations.shapes[0].bounds() == pytest.approx((W - 10, 0, W, 10))


def test_rotate90_swaps_size():
    aug = Augmenter(only("rotate90"), seed=3)
    r = aug(np.zeros((H, W, 3), np.uint8), ann(BOX))
    assert {(r.annotations.width, r.annotations.height), r.image.shape[1::-1]} <= {(W, H), (H, W)}
    assert (r.annotations.width, r.annotations.height) == (r.image.shape[1], r.image.shape[0])


@pytest.mark.parametrize("key", PHOTOMETRIC)
def test_photometric_keeps_labels_exactly(key):
    img = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    r = Augmenter(only(key), channels=3, seed=0)(img, ann(BOX, POLY))
    # one approx per shape: pytest ≥ 9.2 refuses nested sequences in approx
    bounds = [s.bounds() for s in r.annotations.shapes]
    assert len(bounds) == 2
    assert bounds[0] == pytest.approx(BOX.bounds()) and bounds[1] == pytest.approx(POLY.bounds())
    assert r.image.shape == img.shape and r.image.dtype == np.uint8


# ---- F3.6 keep / drop rule ------------------------------------------------------------

def test_mostly_hidden_label_dropped():
    edge = BBox(class_id=0, x1=290, y1=100, x2=320, y2=140)
    cfg = AugmentConfig(transforms={"translate": TransformSetting(enabled=True, p=1, strength=1)},
                        min_visibility=0.4)
    kept = dropped = 0
    for seed in range(40):
        r = Augmenter(cfg, seed=seed)(np.zeros((H, W), np.uint8), ann(edge))
        kept += len(r.annotations.shapes)
        dropped += r.dropped
        for s in r.annotations.shapes:
            x1, y1, x2, y2 = s.bounds()
            assert 0 <= x1 and x2 <= W and 0 <= y1 and y2 <= H  # clipped to the image
    assert kept and dropped  # both outcomes happen across seeds


def test_visibility_threshold():
    # Shift 70% of the box outside: kept at 0.25, dropped at 0.4.
    half_out = BBox(class_id=0, x1=-70, y1=10, x2=30, y2=50)
    for vis, expected in ((0.25, 1), (0.4, 0)):
        aug = Augmenter(AugmentConfig(min_visibility=vis), seed=0)  # identity pipeline
        assert len(aug(np.zeros((H, W), np.uint8), ann(half_out)).annotations.shapes) == expected


def test_min_area_drops_tiny_labels():
    tiny = BBox(class_id=0, x1=10, y1=10, x2=13, y2=13)  # 9 px²
    for min_area, expected in ((5, 1), (16, 0)):
        r = Augmenter(AugmentConfig(min_area_px=min_area), seed=0)(np.zeros((H, W), np.uint8), ann(tiny))
        assert len(r.annotations.shapes) == expected and r.dropped == 1 - expected


def test_suggestions_are_not_augmented():
    a = ann(BOX)
    a.add_suggestions([BBox(class_id=0, x1=5, y1=5, x2=50, y2=50, score=0.9)], source="model")
    r = Augmenter(only("hflip"), seed=0)(np.zeros((H, W), np.uint8), a)
    assert len(r.annotations.shapes) == 1 and r.annotations.suggestions() == []


# ---- grayscale (F1.6) -----------------------------------------------------------------

def test_grayscale_stays_single_channel_with_every_transform():
    cfg = AugmentConfig(transforms={k: TransformSetting(enabled=True, p=1, strength=0.7) for k in TRANSFORM_KEYS})
    img = np.random.default_rng(1).integers(0, 255, (H, W), dtype=np.uint8)
    for seed in range(5):
        r = Augmenter(cfg, channels=1, seed=seed)(img, ann(BOX))
        assert r.image.ndim == 2 and r.image.dtype == np.uint8


def test_color_only_transforms_hidden_for_grayscale():
    gray = {t.key for t in available_transforms(1)}
    assert "hue_saturation" not in gray and "brightness_contrast" in gray
    assert "hue_saturation" in Augmenter(BUILTIN_PRESETS["강함"], channels=3).keys
    assert "hue_saturation" not in Augmenter(BUILTIN_PRESETS["강함"], channels=1).keys


def test_grayscale_project_rejects_color_image():
    with pytest.raises(ValueError):
        Augmenter(AugmentConfig(), channels=1)(np.zeros((H, W, 3), np.uint8), ann())


# ---- inputs / reproducibility -------------------------------------------------------------

def test_rejects_non_uint8_and_size_mismatch():
    aug = Augmenter(AugmentConfig())
    with pytest.raises(ValueError):
        aug(np.zeros((H, W), np.uint16), ann())
    with pytest.raises(ValueError):
        aug(np.zeros((H + 1, W), np.uint8), ann())


def test_seed_reproducible():
    img = np.random.default_rng(2).integers(0, 255, (H, W, 3), dtype=np.uint8)
    runs = []
    for _ in range(2):
        aug = Augmenter(BUILTIN_PRESETS["강함"], seed=42)
        runs.append([aug(img, ann(BOX, POLY)) for _ in range(3)])
    for a, b in zip(*runs):
        assert np.array_equal(a.image, b.image)
        assert [s.bounds() for s in a.annotations.shapes] == [s.bounds() for s in b.annotations.shapes]


def test_output_image_name():
    r = Augmenter(AugmentConfig())(np.zeros((H, W), np.uint8), ann(BOX), image_name="a_aug003.png")
    assert r.annotations.image == "a_aug003.png"


# ---- configuration and presets (F3.4) -------------------------------------------------

def test_builtin_presets_valid():
    assert list(BUILTIN_PRESETS) == ["약함", "보통", "강함", "금속 표면", "PCB"]
    for cfg in BUILTIN_PRESETS.values():
        assert set(cfg.transforms) <= set(TRANSFORM_KEYS)
        assert cfg.enabled_keys()
    assert "hue_saturation" not in BUILTIN_PRESETS["금속 표면"].enabled_keys()


def test_config_validation():
    with pytest.raises(ValidationError):
        TransformSetting(p=1.5)
    with pytest.raises(ValidationError):
        AugmentConfig(multiplier=0)
    with pytest.raises(KeyError):
        AugmentConfig().setting("swirl")
    assert AugmentConfig().multiplier == 20 and AugmentConfig().min_visibility == 0.4


def test_user_preset_roundtrip(tmp_path):
    d = tmp_path / "프리셋 폴더"
    cfg = only("rotate", p=0.3, strength=0.8, multiplier=7)
    path = save_preset(d, "우리 라인", cfg)
    assert path.name == "우리 라인.yaml" and "rotate" in path.read_text(encoding="utf-8")
    presets = load_presets(d)
    assert list(presets)[:5] == list(BUILTIN_PRESETS)
    assert presets["우리 라인"] == cfg


def test_user_preset_rules(tmp_path):
    with pytest.raises(ValueError):
        save_preset(tmp_path, "보통", AugmentConfig())  # built-in name
    with pytest.raises(ValueError):
        save_preset(tmp_path, "a/b", AugmentConfig())
    (tmp_path / "broken.yaml").write_text("transforms: [1, 2", encoding="utf-8")
    (tmp_path / "bad.yaml").write_text("multiplier: -3\n", encoding="utf-8")
    assert set(load_presets(tmp_path)) == set(BUILTIN_PRESETS)


def test_builtin_presets_are_copies():
    presets = load_presets()
    presets["보통"].multiplier = 99
    assert BUILTIN_PRESETS["보통"].multiplier == 20


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_every_transform_builds_across_strengths(strength):
    # Albumentations validates ranges (e.g. Illumination intensity <= 0.2); the slider ends must work.
    for spec in TRANSFORMS:
        spec.build(strength)
