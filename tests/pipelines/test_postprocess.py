import torch

from src.pipelines.postprocess import cosine_similarity, l2_normalise, pairwise_distances, verify


def test_cosine_similarity_unit_norm() -> None:
    embeddings = torch.randn(4, 128)
    normalized = l2_normalise(embeddings)
    sims = cosine_similarity(normalized, normalized)
    assert torch.allclose(sims, torch.ones(4), atol=1e-5)


def test_pairwise_distances_symmetric() -> None:
    embeddings = torch.randn(3, 64)
    distances = pairwise_distances(embeddings)
    assert torch.allclose(distances, distances.t(), atol=1e-6)
    assert torch.diagonal(distances).abs().max().item() < 1e-5


def test_verify_threshold() -> None:
    a = torch.ones(128)
    b = torch.ones(128) * 0.99
    match, score = verify(a, b, threshold=0.95)
    assert match is True
    assert score > 0.95
