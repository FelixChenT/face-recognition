from __future__ import annotations

import torch
from torch import Tensor


def accuracy(logits: Tensor, targets: Tensor) -> float:
    """Compute top-1 accuracy for classification logits."""
    if logits.ndim != 2:
        raise ValueError("Logits tensor must be of shape (batch, num_classes).")
    if targets.ndim != 1:
        raise ValueError("Targets tensor must be 1D.")
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return float(correct) / float(targets.numel())


def cosine_accuracy(embeddings: Tensor, labels: Tensor, prototypes: Tensor) -> float:
    """Measure accuracy by nearest-prototype cosine similarity."""
    normalized_embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    normalized_prototypes = torch.nn.functional.normalize(prototypes, dim=1)
    similarities = normalized_embeddings @ normalized_prototypes.t()
    preds = similarities.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return float(correct) / float(labels.numel())
