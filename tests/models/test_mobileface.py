import torch

from src.models import MobileFaceNet, build_mobileface


def test_mobileface_forward_shape() -> None:
    model = MobileFaceNet(embedding_dim=128, width_multiplier=0.5)
    inputs = torch.randn(2, 3, 112, 112)
    outputs = model(inputs)
    assert outputs.shape == (2, 128)
    norms = outputs.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_build_mobileface_parameter_budget() -> None:
    model = build_mobileface(embedding_dim=128, width_multiplier=0.75)
    total_params = sum(p.numel() for p in model.parameters())
    # 10MB budget -> ~2.6 million float32 parameters. Enforce tighter cap for margin.
    assert total_params < 2_000_000
