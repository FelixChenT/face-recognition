from pathlib import Path

import torch

from src.models import MobileFaceNet
from src.pipelines.trainer import (
    CheckpointConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingRunner,
)


class TinyDataset(torch.utils.data.Dataset[tuple[torch.Tensor, int]]):
    def __init__(self) -> None:
        self.inputs = torch.randn(8, 3, 112, 112)
        self.targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.inputs[index], int(self.targets[index].item())


def test_training_runner_single_epoch(tmp_path: Path) -> None:
    dataset = TinyDataset()
    model = MobileFaceNet(embedding_dim=32, width_multiplier=0.5)
    trainer_config = TrainerConfig(
        epochs=1,
        batch_size=4,
        num_workers=0,
        precision="fp32",
        device="cpu",
        optimizer=OptimizerConfig(lr=1e-3, weight_decay=0.0),
        scheduler=SchedulerConfig(name="none"),
    )
    checkpoint_dir = tmp_path / "checkpoints"
    runner = TrainingRunner(
        model=model,
        train_dataset=dataset,
        val_dataset=None,
        trainer_config=trainer_config,
        num_classes=2,
        checkpoint_config=CheckpointConfig(dir=checkpoint_dir, keep_last=1, save_best_metric="accuracy"),
    )
    runner.fit()
    saved_files = list(checkpoint_dir.glob("*.pt"))
    assert saved_files
