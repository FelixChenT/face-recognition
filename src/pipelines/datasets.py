from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .preprocess import preprocess_image


@dataclass(frozen=True)
class Sample:
    """Manifest entry describing an image sample."""

    path: Path
    label: int


class FaceDataset(Dataset[tuple[Tensor, int]]):
    """Dataset loading faces from a manifest file.

    The manifest file is expected to contain one JSON object per line and requires
    the following keys:

    - `path`: absolute or relative path to the image.
    - `label`: integer class index.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.transform = transform or preprocess_image
        self.samples: List[Sample] = self._load_manifest(self.manifest_path)

    @staticmethod
    def _load_manifest(path: Path) -> List[Sample]:
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")
        samples: List[Sample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                sample_path = Path(payload["path"]).expanduser()
                if not sample_path.is_absolute():
                    sample_path = path.parent / sample_path
                samples.append(Sample(path=sample_path, label=int(payload["label"])))
        if not samples:
            raise ValueError(f"Manifest file {path} is empty.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        tensor = self.transform(image)
        return tensor, sample.label

    def class_distribution(self) -> Sequence[int]:
        """Return the class histogram for the dataset."""
        histogram: dict[int, int] = {}
        for sample in self.samples:
            histogram[sample.label] = histogram.get(sample.label, 0) + 1
        return [histogram[label] for label in sorted(histogram.keys())]
