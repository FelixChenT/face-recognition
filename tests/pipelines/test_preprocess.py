import numpy as np
import torch
from PIL import Image

from src.pipelines.preprocess import preprocess_image


def test_preprocess_image_shape_and_range() -> None:
    data = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    image = Image.fromarray(data, mode="RGB")
    tensor = preprocess_image(image, image_size=112)
    assert tensor.shape == (3, 112, 112)
    assert torch.isfinite(tensor).all()
    # Values should be roughly standardised
    assert tensor.mean().abs().item() < 1.0
