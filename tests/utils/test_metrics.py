import pytest
import torch

from src.utils.metrics import accuracy, cosine_accuracy


def test_accuracy_basic() -> None:
    logits = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    targets = torch.tensor([1, 1, 0])
    assert accuracy(logits, targets) == pytest.approx(2 / 3)


def test_cosine_accuracy_matches_prototypes() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    assert cosine_accuracy(embeddings, labels, prototypes) == pytest.approx(1.0)
