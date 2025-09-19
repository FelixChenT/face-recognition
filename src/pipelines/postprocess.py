from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def l2_normalise(embeddings: Tensor, eps: float = 1e-5) -> Tensor:
    """L2 normalise embeddings along the channel dimension."""
    norms = embeddings.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
    return embeddings / norms


def cosine_similarity(embeddings_a: Tensor, embeddings_b: Tensor) -> Tensor:
    """Compute cosine similarity between two batches of embeddings."""
    a = l2_normalise(embeddings_a)
    b = l2_normalise(embeddings_b)
    return (a * b).sum(dim=1)


def pairwise_distances(embeddings: Tensor) -> Tensor:
    """Compute pairwise cosine distances for a batch of embeddings."""
    normalized = l2_normalise(embeddings)
    similarity = normalized @ normalized.t()
    distances = 1.0 - similarity
    return distances


def verify(embedding_a: Tensor, embedding_b: Tensor, threshold: float = 0.5) -> Tuple[bool, float]:
    """Verify whether two embeddings belong to the same identity."""
    score = cosine_similarity(embedding_a.unsqueeze(0), embedding_b.unsqueeze(0)).item()
    return score >= threshold, score
