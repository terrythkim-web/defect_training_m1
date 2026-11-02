"""Stale dataset warning: build_info.yaml records what the dataset was built from, and training
asks to rebuild when the project changed since (labels, Train/Val, settings)."""

import yaml

from defect_inspection.core.annotation import BBox
from defect_inspection.core.crop import build_crop_dataset
from defect_inspection.core.dataset import build_full_dataset
from defect_inspection.core.dataset_inputs import dataset_changes, dataset_inputs
from defect_inspection.core.roi import Roi, RoiStore
from test_dataset import make_project


def built(tmp_path, mode="full"):
    p = make_project(tmp_path)
    p.config.augment.multiplier = 1
    p.save()
    build_full_dataset(p) if mode == "full" else build_crop_dataset(p)
    return p


def labeled(p):
    return next(e.path for e in p.config.images if p.load_labels(e).confirmed())


def test_no_dataset_means_nothing_to_compare(tmp_path):
    p = make_project(tmp_path)
    assert dataset_changes(p, "full") is None


def test_fresh_after_build(tmp_path):
    p = built(tmp_path)
    assert dataset_changes(p, "full") == []
    info = yaml.safe_load((p.dataset_dir / "full" / "build_info.yaml").read_text(encoding="utf-8"))
    assert info["inputs"] == dataset_inputs(p, "full")


def test_label_change_is_found(tmp_path):
    p = built(tmp_path)
    path = labeled(p)
    ann = p.load_labels(path)
    ann.shapes.append(BBox(class_id=0, x1=1, y1=1, x2=9, y2=9))
    p.save_labels(ann)
    assert dataset_changes(p, "full") == ["라벨이 바뀐 이미지 1장"]


def test_suggestions_and_shape_ids_do_not_count(tmp_path):
    p = built(tmp_path)
    path = labeled(p)
    ann = p.load_labels(path)
    ann.add_suggestions([BBox(class_id=0, x1=1, y1=1, x2=9, y2=9, score=0.9)], source="model")
    ann.shapes[0] = ann.shapes[0].model_copy(update={"id": "another-id"})
    p.save_labels(ann)
    assert dataset_changes(p, "full") == []


def test_split_change_is_found(tmp_path):
    p = built(tmp_path)
    full_split = dataset_inputs(p, "full")["split"]
    path = next(k for k, v in full_split.items() if v == "train")
    p.set_split(path, "val")
    changes = dataset_changes(p, "full")  # the Val quota may move another image back to Train
    assert len(changes) == 1 and changes[0].startswith("Train/Val이 바뀐 이미지")


def test_added_and_removed_images_are_found(tmp_path):
    p = built(tmp_path)
    p.set_ok(next(e.path for e in p.config.images if not p.load_labels(e).confirmed() and not e.is_ok))
    changes = dataset_changes(p, "full")  # a new OK image may also move the OK split
    assert changes[0] == "추가되거나 빠진 이미지 1장" and all("라벨" not in c for c in changes)


def test_settings_changes_are_found(tmp_path):
    p = built(tmp_path)
    p.config.augment.multiplier = 5
    assert dataset_changes(p, "full") == ["증강 설정"]
    p.config.augment.multiplier = 1
    p.add_class("hole")
    assert dataset_changes(p, "full") == ["클래스·작업 종류·색 모드"]


def test_crop_settings_and_rois_only_matter_for_crop(tmp_path):
    p = built(tmp_path, "crop")
    assert dataset_changes(p, "crop") == []
    store = RoiStore(p.root)
    path = p.config.images[0].path
    store.set_rois(path, [Roi(x1=0, y1=0, x2=50, y2=50)], store.common)
    assert dataset_changes(p, "crop") == ["Crop 설정·ROI"]
    p.config.crop = p.config.crop.model_copy(update={"per_defect": p.config.crop.per_defect + 1})
    assert dataset_changes(p, "crop") == ["Crop 설정·ROI"]


def test_dataset_from_before_the_check_is_reported(tmp_path):
    p = built(tmp_path)
    info_path = p.dataset_dir / "full" / "build_info.yaml"
    info = yaml.safe_load(info_path.read_text(encoding="utf-8"))
    info.pop("inputs")
    info_path.write_text(yaml.safe_dump(info, allow_unicode=True), encoding="utf-8")
    assert dataset_changes(p, "full") == ["이전 버전에서 만든 데이터셋이라 비교할 수 없음"]


def test_hand_made_dataset_without_build_info_is_not_judged(tmp_path):
    p = built(tmp_path)
    (p.dataset_dir / "full" / "build_info.yaml").unlink()
    assert dataset_changes(p, "full") is None


# ---- asking before training on an outdated dataset ---------------------------------------------------

import pytest  # noqa: E402

from test_train_panel import qapp, wait_until, window  # noqa: E402,F401 (fixtures)


def change_a_label(p):
    path = labeled(p)
    ann = p.load_labels(path)
    ann.shapes.append(BBox(class_id=0, x1=1, y1=1, x2=9, y2=9))
    p.save_labels(ann)


def asked(window, monkeypatch, answer):
    seen = []

    def ask(mode, changes):
        seen.append((mode, changes))
        return answer

    monkeypatch.setattr(window, "_ask_rebuild", ask)
    return seen


def test_up_to_date_dataset_trains_without_asking(window, qapp, monkeypatch):
    seen = asked(window, monkeypatch, "cancel")
    assert window.train_panel.start() and seen == []


def test_cancel_does_not_train(window, qapp, monkeypatch):
    change_a_label(window.project)
    seen = asked(window, monkeypatch, "cancel")
    assert not window.train_panel.start()
    assert seen == [("full", ["라벨이 바뀐 이미지 1장"])] and not window.train_panel.running


def test_train_anyway(window, qapp, monkeypatch):
    change_a_label(window.project)
    asked(window, monkeypatch, "train")
    assert window.train_panel.start() and window.train_panel.running


def test_rebuild_then_train(window, qapp, monkeypatch):
    change_a_label(window.project)
    seen = asked(window, monkeypatch, "rebuild")
    assert not window.train_panel.start()  # the build runs first
    assert wait_until(qapp, lambda: window.train_panel.running, timeout=60)
    assert len(seen) == 1  # after the rebuild the dataset is up to date: no second question
    from defect_inspection.core.dataset_inputs import dataset_changes
    assert dataset_changes(window.project, "full") == []


def test_resume_is_not_checked(window, qapp, monkeypatch):
    seen = asked(window, monkeypatch, "cancel")
    assert window._before_training(None) and seen == []
