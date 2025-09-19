from __future__ import annotations

from typing import Callable

from PIL import Image
from torchvision import transforms
from torch import Tensor

from .preprocess import IMAGENET_MEAN, IMAGENET_STD


def build_train_transform(image_size: int, grayscale_probability: float = 0.0) -> Callable[[Image.Image], Tensor]:
    """Construct training augmentations producing normalised tensors."""
    ops = [transforms.Resize((image_size, image_size))]
    if grayscale_probability > 0.0:
        ops.append(transforms.RandomGrayscale(p=grayscale_probability))
    ops.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )
    return transforms.Compose(ops)


def build_eval_transform(image_size: int) -> Callable[[Image.Image], Tensor]:
    """Create deterministic preprocessing for validation/inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )
