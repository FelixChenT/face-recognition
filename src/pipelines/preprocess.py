from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def load_image(path: str | Path) -> Image.Image:
    """Load an RGB image from disk."""
    image = Image.open(path).convert("RGB")
    return image


def preprocess_image(image: Image.Image, image_size: int = 112) -> Tensor:
    """Prepare an image for inference.

    Parameters
    ----------
    image:
        Input PIL image assumed to be in RGB format.
    image_size:
        Target resolution for the model input.

    Returns
    -------
    torch.Tensor
        Normalised tensor of shape `(3, image_size, image_size)`.
    """

    resized = image.resize((image_size, image_size), resample=Image.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def batch_from_images(images: Sequence[Image.Image], image_size: int = 112, device: torch.device | None = None) -> Tensor:
    """Create a batch tensor from a list of images."""
    processed = [preprocess_image(img, image_size=image_size) for img in images]
    batch = torch.stack(processed, dim=0)
    if device is not None:
        batch = batch.to(device)
    return batch


def denormalise(tensor: Tensor) -> Tensor:
    """Reverse the normalisation applied during preprocessing."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    return tensor * std + mean


def preprocess_paths(paths: Iterable[str | Path], image_size: int = 112, device: torch.device | None = None) -> Tensor:
    """Load and preprocess a list of image paths into a batch tensor."""
    images = [load_image(path) for path in paths]
    return batch_from_images(images, image_size=image_size, device=device)
