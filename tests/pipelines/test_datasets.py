import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipelines.datasets import FaceDataset


def test_face_dataset_load(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_a = (np.random.rand(120, 120, 3) * 255).astype("uint8")
    path_a = images_dir / "a.png"
    Image.fromarray(image_a, mode="RGB").save(path_a)

    image_b = (np.random.rand(120, 120, 3) * 255).astype("uint8")
    path_b = images_dir / "b.png"
    Image.fromarray(image_b, mode="RGB").save(path_b)

    manifest = tmp_path / "manifest.jsonl"
    entries = [
        {"path": str(path_a), "label": 1},
        {"path": str(path_b), "label": 2},
    ]
    with manifest.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    dataset = FaceDataset(manifest)
    tensor, label = dataset[0]
    assert tensor.shape == (3, 112, 112)
    assert label == 1
    distribution = dataset.class_distribution()
    assert len(distribution) == 2
