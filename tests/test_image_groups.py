"""Slices / copies of other project images (core/image_groups), group-aware split and the Val
reliability warnings (dataset.val_warnings)."""

import time

import cv2
import numpy as np
import pytest
import yaml
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox

from defect_inspection.core.annotation import BBox
from defect_inspection.core.augment_config import AugmentConfig
from defect_inspection.core.crop import build_crop_dataset
from defect_inspection.core.dataset import build_full_dataset, select_images, split_images, val_warnings
from defect_inspection.core.image_groups import contains, find_source_groups, image_signature
from defect_inspection.core.project import Project, compute_split
from defect_inspection.platform_utils import imwrite_unicode
from defect_inspection.ui.main_window import MainWindow


def texture(seed, h=512, w=1536):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w), dtype=np.uint8)
    blurred = cv2.GaussianBlur(img, (0, 0), 3).astype(np.int32)
    return ((blurred - blurred.min()) * 4 % 256).astype(np.uint8)  # smooth but not flat


def sig(path, name=None):
    return image_signature(name or path.name, path)


def test_slices_jpeg_copies_and_duplicates_are_found(tmp_path):
    big = texture(1)
    imwrite_unicode(tmp_path / "wide.png", big)
    for k in range(6):  # six 256 px slices, like a line-scan photo cut into pieces
        imwrite_unicode(tmp_path / f"slice{k}.png", big[:, k * 256:(k + 1) * 256])
    imwrite_unicode(tmp_path / "slice_jpeg.jpg", big[:, 300:556], [cv2.IMWRITE_JPEG_QUALITY, 95])  # re-encoded
    other = texture(2)
    imwrite_unicode(tmp_path / "other.png", other)
    imwrite_unicode(tmp_path / "other_copy.png", other)
    imwrite_unicode(tmp_path / "other_part.png", texture(3)[:, :256])  # same size as a slice, unrelated
    imwrite_unicode(tmp_path / "flat.png", np.full((512, 256), 128, np.uint8))  # matches anything flat
    sigs = [sig(p) for p in sorted(tmp_path.iterdir())]
    groups = find_source_groups(sigs)
    assert {m for m, s in groups.items() if s == "wide.png"} == {f"slice{k}.png" for k in range(6)} | {"slice_jpeg.jpg"}
    assert groups["other_copy.png"] == "other.png" or groups.get("other.png") == "other_copy.png"
    assert "other_part.png" not in groups and "flat.png" not in groups


def test_contains_needs_the_same_pixels(tmp_path):
    big = texture(4)
    imwrite_unicode(tmp_path / "big.png", big)
    piece = big[:, 512:768].copy()
    piece[200:260, 50:200] = 255  # an edited slice is not the same picture
    imwrite_unicode(tmp_path / "edited.png", piece)
    assert not contains(sig(tmp_path / "edited.png"), sig(tmp_path / "big.png"))


def test_find_can_be_cancelled(tmp_path):
    imwrite_unicode(tmp_path / "a.png", texture(5))
    imwrite_unicode(tmp_path / "b.png", texture(6))
    assert find_source_groups([sig(tmp_path / "a.png"), sig(tmp_path / "b.png")], cancel=lambda: True) is None


def make_project(tmp_path, with_slices=True):
    """Four wide photos with one defect each, plus (optionally) their slices; one defect in a
    slice is labeled again; slices without the defect are OK images."""
    src = tmp_path / "원본"
    src.mkdir(parents=True)
    for n in range(4):
        wide = texture(10 + n)
        imwrite_unicode(src / f"wide{n}.png", wide)
        if with_slices and n < 2:
            for k in range(6):
                imwrite_unicode(src / f"wide{n}_{k}.png", wide[:, k * 256:(k + 1) * 256])
    p = Project.create(tmp_path / "그룹 프로젝트", color_mode="grayscale")
    p.add_class("broken")
    p.add_class("crack")
    p.add_images([src])
    for e in p.config.images:
        name = e.path.removesuffix(".png")
        if "_" not in name:  # wide photo: one defect at x 600..650 (slice 2)
            n = int(name[-1])
            ann = p.load_labels(e)
            ann.shapes.append(BBox(class_id=n % 2, x1=600, y1=100, x2=650, y2=160))
            p.save_labels(ann)
        elif name.endswith("_2"):
            ann = p.load_labels(e)
            ann.shapes.append(BBox(class_id=int(name[4]) % 2, x1=88, y1=100, x2=138, y2=160))
            p.save_labels(ann)
        else:
            p.set_ok(e.path)
    p.config.augment = AugmentConfig(multiplier=1, transforms={})
    p.save()
    return p


def detect_groups(p):
    sigs = [image_signature(e.path, p.image_abspath(e)) for e in p.config.images]
    p.config.source_groups = find_source_groups(sigs)
    return p.config.source_groups


@pytest.mark.parametrize("seed", range(6))
def test_group_members_share_a_split(tmp_path, seed):
    p = make_project(tmp_path)
    groups = detect_groups(p)
    assert len(groups) == 12  # every slice -> its wide photo
    p.config.split.seed = seed
    defect, ok, _ = select_images(p)
    split = split_images(p, defect, ok)
    for member, source in groups.items():
        assert split[member] == split[source], (member, source)
    assert any(split[e.path] == "val" for e in defect)


def test_manual_split_of_a_member_moves_the_group(tmp_path):
    p = make_project(tmp_path)
    detect_groups(p)
    p.set_split("wide0_4.png", "val")
    defect, ok, _ = select_images(p)
    split = split_images(p, defect, ok)
    assert all(split[path] == "val" for path in ["wide0.png"] + [f"wide0_{k}.png" for k in range(6)])


def test_without_groups_split_is_unchanged(tmp_path):
    p = make_project(tmp_path)
    defect, ok, _ = select_images(p)
    expected = compute_split(defect, p.config.split)
    expected.update(compute_split(ok, p.config.split))
    assert split_images(p, defect, ok) == expected


def test_val_warnings(tmp_path):
    p = make_project(tmp_path)
    detect_groups(p)
    defect, ok, _ = select_images(p)
    split = split_images(p, defect, ok)
    warnings = val_warnings(p, split, defect)
    text = "\n".join(warnings)
    val_classes = {p.load_labels(e).confirmed()[0].class_id for e in defect if split[e.path] == "val"}
    for cid, name in enumerate(["broken", "crack"]):
        assert (f"Val에 '{name}' 결함이 없습니다" in text) == (cid not in val_classes)
    assert "Val 결함이" in text and "recall이 약" in text
    # a defect labeled in the photo and again in its slice counts once
    assert "서로 다른 결함 수가 적습니다: broken 2, crack 2" in text
    assert "같은 원본에서 잘라낸 이미지 12장" in text


def test_build_reports_store_warnings(tmp_path):
    p = make_project(tmp_path, with_slices=False)
    full = build_full_dataset(p)
    assert full.warnings and any("서로 다른 결함 수가 적습니다" in w for w in full.warnings)
    info = yaml.safe_load((p.dataset_dir / "full" / "build_info.yaml").read_text(encoding="utf-8"))
    assert info["report"]["warnings"] == full.warnings
    crop = build_crop_dataset(p)
    assert crop.warnings == full.warnings  # counted on the original images, not on crops


# ---- UI ---------------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return cond()


@pytest.fixture
def window(qapp, tmp_path):
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), cache_root=tmp_path / "cache")
    w.show()
    yield w
    w.train_panel.shutdown()
    w.project_dirty = False
    w.close()


def test_window_finds_slices_and_warns(window, tmp_path, qapp):
    p = make_project(tmp_path)
    window.load_project(p)
    assert wait_until(qapp, lambda: len(p.config.source_groups) == 12)
    assert window.project_dirty
    notice = window._group_notice
    assert notice is not None and notice.isVisible() and not notice.isModal()
    assert "12장" in notice.text() and "wide0.png ← wide0_0.png" in notice.detailedText()
    tip = window.image_panel._item("wide0_3.png").toolTip()
    assert "원본: wide0.png" in tip
    assert "6장" in window.image_panel._item("wide1.png").toolTip()
    notice.close()

    # unchanged result (e.g. reopening): no new notice
    window._group_notice = None
    window._start_group_check()
    assert window.group_checker.wait() is not None
    qapp.processEvents()
    assert window._group_notice is None


def test_menu_check_lists_groups_even_without_new_ones(window, tmp_path, qapp):
    p = make_project(tmp_path, with_slices=False)
    window.load_project(p)
    assert window.group_checker.wait() is not None
    qapp.processEvents()
    assert window._group_notice is None  # nothing found, nothing shown
    window.act_check_groups.trigger()
    assert wait_until(qapp, lambda: window._group_notice is not None)
    assert "없습니다" in window._group_notice.text()
    window._group_notice.close()


def test_build_waits_for_a_running_check(window, tmp_path, qapp, monkeypatch):
    p = make_project(tmp_path)
    window.load_project(p)
    monkeypatch.setattr(QMessageBox, "show", lambda self: None)
    window._before_build()  # blocks until the check is done, then applies it
    assert len(p.config.source_groups) == 12


# ---- synthetic images, Train/Val by hand -----------------------------------------------------

def test_synthetic_unit_always_train_and_not_counted(tmp_path):
    p = make_project(tmp_path)
    detect_groups(p)
    defect, ok, _ = select_images(p)
    for seed in range(6):  # whatever the seed, the marked image's group stays in Train
        p.config.split.seed = seed
        p.set_synthetic("wide1_2.png")
        split = split_images(p, defect, ok)
        assert all(split[path] == "train" for path in ["wide1.png"] + [f"wide1_{k}.png" for k in range(6)])
    text = "\n".join(val_warnings(p, split_images(p, defect, ok), defect))
    assert "합성 이미지 1장" in text
    q = make_project(tmp_path / "b", with_slices=False)  # wide0..3: broken, crack, broken, crack
    q.set_synthetic("wide3.png")
    defect, ok, _ = select_images(q)
    text = "\n".join(val_warnings(q, split_images(q, defect, ok), defect))
    assert "broken 2, crack 1" in text  # wide3's crack is synthetic: only wide1's crack counts


def test_window_split_and_synthetic_actions(window, tmp_path, qapp):
    p = make_project(tmp_path, with_slices=False)
    window.load_project(p)
    window.image_panel.select("wide2.png")
    assert wait_until(qapp, lambda: window.current_image == "wide2.png")
    assert window.act_split_val in window.image_list.actions() and window.act_synthetic.isCheckable()

    window.act_split_val.trigger()
    assert p.image("wide2.png").split == "val" and window.project_dirty
    assert "Val 지정" in window.image_panel._item("wide2.png").text()
    window.act_split_auto.trigger()
    assert p.image("wide2.png").split == "auto" and "지정" not in window.image_panel._item("wide2.png").text()

    window.act_split_val.trigger()
    window.act_synthetic.trigger()
    e = p.image("wide2.png")
    assert e.synthetic and e.split == "auto" and window.act_synthetic.isChecked()  # never measured on
    assert "합성" in window.image_panel._item("wide2.png").text()
    window.act_split_val.trigger()
    assert p.image("wide2.png").split == "auto" and "Val로 지정할 수 없습니다" in window.statusBar().currentMessage()
    window.image_panel.select("wide1.png")
    assert wait_until(qapp, lambda: window.current_image == "wide1.png") and not window.act_synthetic.isChecked()


def test_split_by_hand_moves_the_whole_group(window, tmp_path, qapp, monkeypatch):
    p = make_project(tmp_path)
    monkeypatch.setattr(QMessageBox, "show", lambda self: None)
    window.load_project(p)
    window._before_build()  # groups known
    window.image_panel.select("wide0_3.png")
    assert wait_until(qapp, lambda: window.current_image == "wide0_3.png")
    window.act_split_train.trigger()
    assert all(p.image(path).split == "train" for path in ["wide0.png"] + [f"wide0_{k}.png" for k in range(6)])
    assert "함께" in window.statusBar().currentMessage()
    project_reloaded = window.project
    project_reloaded.save()
    assert Project.open(p.root).image("wide0_5.png").split == "train"
