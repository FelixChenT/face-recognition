import argparse
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from src.models import build_mobileface
from src.pipelines import FaceDataset, build_eval_transform, build_train_transform
from src.pipelines.trainer import (
    CheckpointConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingRunner,
)
from src.utils.config import ConfigError, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight face recognition model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training YAML config.")
    parser.add_argument("--device", type=str, default=None, help="Override the compute device (e.g. cuda:0).")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(base_dir: Path, candidate: str | Path) -> Path:
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        return candidate_path
    return (base_dir / candidate_path).resolve()


def build_trainer_config(config: Dict[str, Any], device_override: str | None) -> TrainerConfig:
    optimizer_cfg = config.get("optimizer", {})
    scheduler_cfg = config.get("scheduler", {})
    trainer = TrainerConfig(
        epochs=int(config.get("epochs", scheduler_cfg.get("total_epochs", 1))),
        batch_size=int(config.get("batch_size", 32)),
        num_workers=int(config.get("num_workers", 4)),
        precision=str(config.get("precision", "fp32")),
        device=device_override or str(config.get("device", "cpu")),
        optimizer=OptimizerConfig(
            name=str(optimizer_cfg.get("name", "adamw")),
            lr=float(optimizer_cfg.get("lr", 1e-3)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
            betas=tuple(float(x) for x in optimizer_cfg.get("betas", (0.9, 0.999))),
        ),
        scheduler=SchedulerConfig(
            name=str(scheduler_cfg.get("name", "cosine_annealing_warmup")),
            warmup_epochs=int(scheduler_cfg.get("warmup_epochs", 0)),
            total_epochs=int(scheduler_cfg.get("total_epochs", config.get("epochs", 1))),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-5)),
        ),
    )
    return trainer


def build_checkpoint_config(base_dir: Path, config: Dict[str, Any] | None) -> CheckpointConfig | None:
    if not config:
        return None
    checkpoint_dir = resolve_path(base_dir, config.get("dir", "artifacts/checkpoints"))
    return CheckpointConfig(
        dir=checkpoint_dir,
        keep_last=int(config.get("keep_last", 3)),
        save_best_metric=str(config.get("save_best_metric", "val_accuracy")),
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    config_base = config_path.parent

    experiment_cfg = config.get("experiment", {})
    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)

    model_cfg = config.get("model", {})
    model = build_mobileface(
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        width_multiplier=float(model_cfg.get("width_multiplier", 1.0)),
        dropout=float(model_cfg.get("backbone_dropout", 0.1)),
    )

    training_cfg = config.get("training")
    if training_cfg is None:
        raise ConfigError("Training configuration is missing from the YAML file.")

    dataset_cfg = training_cfg.get("dataset", {})
    train_manifest = dataset_cfg.get("train_manifest")
    if train_manifest is None:
        raise ConfigError("	raining.dataset.train_manifest must be provided in the config.")

    train_manifest_path = resolve_path(config_base, train_manifest)
    if not train_manifest_path.exists():
        raise FileNotFoundError(f"Training manifest not found: {train_manifest_path}")

    image_size = int(dataset_cfg.get("image_size", 112))
    grayscale_probability = float(dataset_cfg.get("grayscale_probability", 0.0))
    train_transform = build_train_transform(image_size=image_size, grayscale_probability=grayscale_probability)
    eval_transform = build_eval_transform(image_size=image_size)

    train_dataset = FaceDataset(train_manifest_path, transform=train_transform)

    val_manifest = dataset_cfg.get("val_manifest")
    val_dataset = None
    if val_manifest:
        val_path = resolve_path(config_base, val_manifest)
        if val_path.exists():
            val_dataset = FaceDataset(val_path, transform=eval_transform)
        else:
            print(f"Validation manifest not found at {val_path}, skipping validation phase.")

    labels = {sample.label for sample in train_dataset.samples}
    num_classes = len(labels)
    if num_classes == 0:
        raise RuntimeError("Training dataset must contain at least one class.")

    trainer_config = build_trainer_config(training_cfg, args.device)
    checkpoint_config = build_checkpoint_config(config_base, config.get("checkpointing"))

    runner = TrainingRunner(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        trainer_config=trainer_config,
        num_classes=num_classes,
        checkpoint_config=checkpoint_config,
    )
    runner.fit()


if __name__ == "__main__":
    main()
