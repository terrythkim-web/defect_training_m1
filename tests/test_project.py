import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from defect_inspection.core.project import (
    PROJECT_FILE,
    ImageEntry,
    Project,
    ProjectConfig,
    SplitConfig,
    compute_split,
)
from defect_inspection.platform_utils import imread_unicode, imwrite_unicode


@pytest.fixture
def project_root(tmp_path):
    # Spaces and Korean characters, as in real user folders.
    return tmp_path / "검사 프로젝트"


@pytest.fixture
def source_images(tmp_path):
    src = tmp_path / "원본 이미지"
    src.mkdir()
    img = np.zeros((8, 8, 3), np.uint8)
    for name in ["스크래치 01.png", "dent.JPG", "b.jpeg", "c.bmp", "d.tif"]:
        assert imwrite_unicode(src / name, img)
    (src / "notes.txt").write_text("not an image", encoding="utf-8")
    return src


def entries(n):
    return [ImageEntry(path=f"img_{i:03d}.png") for i in range(n)]


# ---- create / open / save ----------------------------------------------------

def test_create_makes_folders_and_yaml(project_root):
    project = Project.create(project_root, task="seg")

    assert (project_root / PROJECT_FILE).is_file()
    assert project.images_dir.is_dir() and project.labels_dir.is_dir()
    assert project.config.name == "검사 프로젝트"
    assert project.config.task == "seg"
    assert project.config.split.val_ratio == 0.2  # default 8:2


def test_create_refuses_existing_project(project_root):
    Project.create(project_root)
    with pytest.raises(FileExistsError):
        Project.create(project_root)


def test_save_open_roundtrip(project_root, source_images):
    project = Project.create(project_root, name="PCB 라인")
    project.add_class("스크래치")
    project.add_class("dent", color="#00ff00")
    project.add_images([source_images])
    project.set_ok("b.jpeg")
    project.set_split("c.bmp", "val")
    project.config.split.seed = 7
    project.save()

    reopened = Project.open(project_root)

    assert reopened.config == project.config
    assert reopened.config.classes[1].color == "#00FF00"


def test_yaml_is_utf8_readable_and_posix(project_root, source_images):
    project = Project.create(project_root)
    project.add_class("스크래치")
    project.add_images([source_images])
    project.save()

    text = (project_root / PROJECT_FILE).read_text(encoding="utf-8")
    assert "스크래치" in text  # not escaped as \uXXXX
    assert "\\" not in text
    data = yaml.safe_load(text)
    assert data["version"] == 1


def test_open_rejects_invalid_yaml_values(project_root):
    Project.create(project_root)
    path = project_root / PROJECT_FILE
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["task"] = "classify"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValidationError):
        Project.open(project_root)


# ---- validation ----------------------------------------------------------------

@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
def test_val_ratio_bounds(ratio):
    with pytest.raises(ValidationError):
        SplitConfig(val_ratio=ratio)


@pytest.mark.parametrize("path", ["/abs/a.png", "C:/x/a.png", "../a.png", "a/../../b.png", ""])
def test_image_path_must_be_relative(path):
    with pytest.raises(ValidationError):
        ImageEntry(path=path)


def test_image_path_normalized_to_posix():
    assert ImageEntry(path="sub\\a.png").path == "sub/a.png"


def test_split_assignment_validated():
    entry = ImageEntry(path="a.png")
    with pytest.raises(ValidationError):
        entry.split = "test"


def test_duplicate_class_names_rejected(project_root):
    project = Project.create(project_root)
    project.add_class("scratch")
    with pytest.raises(ValueError):
        project.add_class(" scratch ")
    with pytest.raises(ValidationError):
        ProjectConfig(name="x", classes=[{"name": "a"}, {"name": "a"}])


def test_class_color_and_rename(project_root):
    project = Project.create(project_root)
    project.add_class("a")
    project.add_class("b")
    assert project.config.classes[0].color != project.config.classes[1].color  # palette default

    project.rename_class(0, "찍힘")
    project.set_class_color(0, "#123abc")
    assert project.config.classes[0].name == "찍힘"
    assert project.config.classes[0].color == "#123ABC"
    with pytest.raises(ValueError):
        project.rename_class(0, "b")
    with pytest.raises(ValidationError):
        project.set_class_color(0, "red")


# ---- images ------------------------------------------------------------------

def test_add_images_filters_unsupported_and_copies(project_root, source_images):
    project = Project.create(project_root)
    added = project.add_images([source_images])

    assert sorted(e.path for e in added) == sorted(
        ["스크래치 01.png", "dent.JPG", "b.jpeg", "c.bmp", "d.tif"]
    )
    for entry in added:
        assert project.image_abspath(entry).is_file()
    assert not (project.images_dir / "notes.txt").exists()


def test_add_images_renames_on_clash(project_root, source_images):
    project = Project.create(project_root)
    project.add_images([source_images / "b.jpeg"])
    second = project.add_images([source_images / "b.jpeg"])

    assert second[0].path == "b_1.jpeg"
    assert (project.images_dir / "b_1.jpeg").is_file()


def test_add_images_clash_is_case_insensitive(project_root, source_images, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    imwrite_unicode(other / "DENT.jpg", np.zeros((4, 4, 3), np.uint8))
    project = Project.create(project_root)
    project.add_images([source_images / "dent.JPG"])
    added = project.add_images([other / "DENT.jpg"])

    assert added[0].path == "DENT_1.jpg"


def test_unknown_image_raises(project_root):
    project = Project.create(project_root)
    with pytest.raises(KeyError):
        project.set_ok("missing.png")


# ---- split -------------------------------------------------------------------

def test_split_default_ratio_8_2():
    result = compute_split(entries(20), SplitConfig())
    assert list(result.values()).count("val") == 4
    assert list(result.values()).count("train") == 16


def test_split_is_reproducible_and_order_independent():
    imgs = entries(30)
    a = compute_split(imgs, SplitConfig(seed=1))
    b = compute_split(list(reversed(imgs)), SplitConfig(seed=1))
    assert a == b


def test_split_changes_with_seed():
    imgs = entries(30)
    assert compute_split(imgs, SplitConfig(seed=1)) != compute_split(imgs, SplitConfig(seed=2))


def test_split_respects_manual_assignment():
    imgs = entries(10)
    imgs[0].split = "val"
    imgs[1].split = "val"
    imgs[2].split = "train"
    result = compute_split(imgs, SplitConfig(val_ratio=0.2))

    assert result["img_000.png"] == "val"
    assert result["img_001.png"] == "val"
    assert result["img_002.png"] == "train"
    # Manual vals already fill the 20% quota.
    assert list(result.values()).count("val") == 2


def test_split_small_sets_keep_one_val_and_one_train():
    assert sorted(compute_split(entries(2), SplitConfig()).values()) == ["train", "val"]
    assert list(compute_split(entries(3), SplitConfig(val_ratio=0.01)).values()).count("val") == 1
    assert list(compute_split(entries(3), SplitConfig(val_ratio=0.99)).values()).count("train") == 1
    assert compute_split(entries(1), SplitConfig()) == {"img_000.png": "train"}


def test_split_stable_when_adding_image():
    imgs = entries(20)
    before = compute_split(imgs, SplitConfig())
    after = compute_split(imgs + [ImageEntry(path="new.png")], SplitConfig())
    moved = [p for p in before if before[p] != after[p]]
    assert len(moved) <= 1


# ---- grayscale mode (F1.6) ----------------------------------------------------

@pytest.fixture
def color_images(tmp_path):
    src = tmp_path / "컬러 원본"
    src.mkdir()
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    imwrite_unicode(src / "noisy.png", noisy)
    imwrite_unicode(src / "photo.jpg", noisy)
    imwrite_unicode(src / "gray.png", noisy[..., 0])
    imwrite_unicode(src / "deep.tif", np.full((16, 16, 3), 40000, np.uint16))
    return src


def test_default_color_mode_keeps_color(project_root, color_images):
    project = Project.create(project_root)
    project.add_images([color_images / "noisy.png"])

    assert project.channels == 3
    assert imread_unicode(project.image_abspath("noisy.png")).shape == (64, 64, 3)


def test_grayscale_project_converts_on_import(project_root, color_images):
    project = Project.create(project_root, color_mode="grayscale")
    project.add_images([color_images])

    assert project.channels == 1
    for entry in project.config.images:
        assert imread_unicode(project.image_abspath(entry)).ndim == 2, entry.path
    assert imread_unicode(project.image_abspath("deep.tif")).dtype == np.uint16
    # Smaller than the color source.
    assert (project.images_dir / "noisy.png").stat().st_size < (color_images / "noisy.png").stat().st_size
    # Sources are untouched.
    assert imread_unicode(color_images / "noisy.png").ndim == 3


def test_grayscale_import_skips_undecodable(project_root, tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not a png")
    project = Project.create(project_root, color_mode="grayscale")
    with pytest.warns(UserWarning, match="cannot decode"):
        added = project.add_images([bad])
    assert added == [] and project.config.images == []


def test_convert_existing_project_to_grayscale(project_root, color_images):
    project = Project.create(project_root)
    project.add_images([color_images])

    result = project.convert_to_grayscale()
    project.save()

    assert result.converted == 3 and result.already_gray == 1 and result.failed == []
    assert result.bytes_after < result.bytes_before
    for entry in project.config.images:
        assert imread_unicode(project.image_abspath(entry)).ndim == 2
    assert not list(project.images_dir.glob("*.tmp.*"))
    assert Project.open(project_root).config.color_mode == "grayscale"


def test_ok_images_are_split_too(project_root, source_images):
    project = Project.create(project_root)
    project.add_images([source_images])
    project.set_ok("c.bmp")
    assert set(project.splits()) == {e.path for e in project.config.images}
