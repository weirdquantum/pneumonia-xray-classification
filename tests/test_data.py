import pandas as pd
import pytest

from pneumonia.data import assign_val_split, parse_filename


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("person1_virus_6.jpeg", ("person1", "virus")),
        ("person1_bacteria_1.jpeg", ("person1", "bacteria")),
        ("person1180_virus_2010_1.jpeg", ("person1180", "virus")),
        ("IM-0115-0001.jpeg", ("IM-0115", "normal")),
        ("IM-0511-0001-0002.jpeg", ("IM-0511", "normal")),
        ("NORMAL2-IM-0373-0001.jpeg", ("NORMAL2-IM-0373", "normal")),
    ],
)
def test_parse_filename(filename, expected):
    assert parse_filename(filename) == expected


def test_parse_filename_rejects_unknown():
    with pytest.raises(ValueError):
        parse_filename("scan.jpeg")


def test_val_split_keeps_patients_together_and_stratifies():
    rows = []
    for p in range(200):
        label = int(p % 4 != 0)  # ~3:1 pneumonia:normal, like the real data
        for k in range(1 + p % 3):  # 1-3 images per patient
            rows.append({"path": f"{p}_{k}", "split": "train", "label": label, "patient_id": f"p{p}"})
    rows += [{"path": "t", "split": "test", "label": 0, "patient_id": "t0"}]
    df = assign_val_split(pd.DataFrame(rows), val_frac=0.2, seed=0)

    train_p = set(df.loc[df.split == "train", "patient_id"])
    val_p = set(df.loc[df.split == "val", "patient_id"])
    assert train_p.isdisjoint(val_p)
    assert (df.loc[df.path == "t", "split"] == "test").all()
    val_frac = (df.split == "val").sum() / (df.split != "test").sum()
    assert 0.12 < val_frac < 0.28
    assert abs(df.loc[df.split == "val", "label"].mean() - df.loc[df.split == "train", "label"].mean()) < 0.1


def test_eval_transform_keeps_field_of_view_at_any_size():
    import torch

    from pneumonia.data import build_transforms

    img = torch.zeros(1, 256, 256, dtype=torch.uint8)
    img[:, 20:30, 20:30] = 255  # marker near the corner, inside the 224/256 field of view
    for size in (224, 112):
        out = build_transforms(size, augment=False)(img)
        assert out.shape == (1, size, size)
        assert out.max() > 0.5, f"marker cropped away at image_size={size}"


def test_load_split_rejects_cache_manifest_mismatch(tmp_path):
    import numpy as np

    from pneumonia.data import load_split

    pd.DataFrame({"split": ["train", "test"], "label": [0, 1], "patient_id": ["a", "b"]}).to_csv(tmp_path / "manifest.csv", index=False)
    np.save(tmp_path / "images.npy", np.zeros((3, 8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="rerun scripts/prepare_data.py"):
        load_split(tmp_path, "train", augment=False)
