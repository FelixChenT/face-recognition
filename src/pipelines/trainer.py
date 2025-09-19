from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional

import torch
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from ..utils.metrics import accuracy


@dataclass
class OptimizerConfig:
    """Hyper-parameters for the optimiser."""

    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: Iterable[float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler hyper-parameters."""

    name: str = "cosine_annealing_warmup"
    warmup_epochs: int = 0
    total_epochs: int = 1
    min_lr: float = 1e-5


@dataclass
class TrainerConfig:
    """Wraps all training-related settings."""

    epochs: int = 1
    batch_size: int = 32
    num_workers: int = 4
    precision: str = "fp32"
    device: str = "cpu"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint persistence."""

    dir: Path = Path("artifacts/checkpoints")
    keep_last: int = 1
    save_best_metric: str = "val_accuracy"


def _build_dataloader(dataset: Dataset[tuple[Tensor, int]], config: TrainerConfig, shuffle: bool) -> DataLoader[tuple[Tensor, int]]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )


def _build_optimizer(parameters: Iterable[Tensor], config: OptimizerConfig) -> Optimizer:
    betas = tuple(config.betas)
    if config.name.lower() == "adamw":
        return torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay, betas=betas)
    if config.name.lower() == "sgd":
        return torch.optim.SGD(parameters, lr=config.lr, momentum=betas[0], weight_decay=config.weight_decay, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {config.name}")


def _build_scheduler(optimizer: Optimizer, config: SchedulerConfig, steps_per_epoch: int) -> Optional[LambdaLR]:
    if config.name.lower() != "cosine_annealing_warmup":
        return None
    total_steps = max(1, config.total_epochs * steps_per_epoch)
    warmup_steps = max(0, config.warmup_epochs * steps_per_epoch)
    base_lr = optimizer.param_groups[0]["lr"]
    min_lr = config.min_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_scale = min_lr / base_lr if base_lr > 0 else 0.0
        return min_scale + (1.0 - min_scale) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class TrainingRunner:
    """Handles model optimisation and evaluation loops."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset[tuple[Tensor, int]],
        val_dataset: Dataset[tuple[Tensor, int]] | None,
        trainer_config: TrainerConfig,
        num_classes: int,
        checkpoint_config: CheckpointConfig | None = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = trainer_config
        self.device = torch.device(trainer_config.device)
        self.checkpoint_config = checkpoint_config
        self._best_metric: float | None = None
        self._checkpoint_paths: Deque[Path] = deque()

        self.model.to(self.device)

        embedding_layer = getattr(self.model, "embedding", None)
        embedding_dim = getattr(embedding_layer, "out_features", None)
        if embedding_dim is None:
            raise AttributeError("Model must expose an embedding linear layer with out_features metadata.")
        self.classifier = nn.Linear(embedding_dim, num_classes).to(self.device)

        params = list(self.model.parameters()) + list(self.classifier.parameters())
        self.optimizer = _build_optimizer(params, trainer_config.optimizer)
        steps_per_epoch = math.ceil(len(train_dataset) / trainer_config.batch_size)
        self.scheduler = _build_scheduler(self.optimizer, trainer_config.scheduler, steps_per_epoch)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler(device=self.device.type, enabled=self._use_mixed_precision())

    def _use_mixed_precision(self) -> bool:
        wants_fp16 = self.config.precision.lower() == "fp16"
        return wants_fp16 and self.device.type == "cuda"

    def fit(self) -> None:
        """Run the training loop for the configured number of epochs."""
        train_loader = _build_dataloader(self.train_dataset, self.config, shuffle=True)
        val_loader = _build_dataloader(self.val_dataset, self.config, shuffle=False) if self.val_dataset else None
        global_step = 0
        for epoch in range(1, self.config.epochs + 1):
            metrics = self._train_one_epoch(train_loader, global_step)
            global_step = int(metrics["global_step"])
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            self._maybe_save_checkpoint(epoch, metrics)
            self._log_epoch(epoch, metrics)

    def _train_one_epoch(self, loader: DataLoader[tuple[Tensor, int]], global_step: int) -> Dict[str, float]:
        self.model.train()
        self.classifier.train()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0

        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(self.device, non_blocking=True)
            batch_targets = batch_targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self._use_mixed_precision()):
                embeddings = self.model(batch_inputs)
                logits = self.classifier(embeddings)
                loss = self.criterion(logits, batch_targets)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()
            global_step += 1

            batch_size = batch_inputs.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy(logits.detach(), batch_targets) * batch_size
            total_samples += batch_size

        return {
            "loss": running_loss / max(1, total_samples),
            "accuracy": running_acc / max(1, total_samples),
            "global_step": global_step,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader[tuple[Tensor, int]]) -> Dict[str, float]:
        self.model.eval()
        self.classifier.eval()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0

        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(self.device, non_blocking=True)
            batch_targets = batch_targets.to(self.device, non_blocking=True)
            embeddings = self.model(batch_inputs)
            logits = self.classifier(embeddings)
            loss = self.criterion(logits, batch_targets)

            batch_size = batch_inputs.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy(logits, batch_targets) * batch_size
            total_samples += batch_size

        return {
            "loss": running_loss / max(1, total_samples),
            "accuracy": running_acc / max(1, total_samples),
        }

    def _maybe_save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        if not self.checkpoint_config:
            return
        metric_name = self.checkpoint_config.save_best_metric
        if not metric_name.startswith("val_"):
            if f"val_{metric_name}" in metrics:
                metric_key = f"val_{metric_name}"
            else:
                metric_key = metric_name
        else:
            metric_key = metric_name
        metric_value = metrics.get(metric_key)
        if metric_value is None:
            return
        mode = "min" if "loss" in metric_key else "max"
        improved = False
        if self._best_metric is None:
            improved = True
        elif mode == "min":
            improved = metric_value < self._best_metric
        else:
            improved = metric_value > self._best_metric
        if improved:
            self._best_metric = metric_value
            path = self._save_checkpoint(epoch, metric_value)
            self._register_checkpoint(path)

    def _save_checkpoint(self, epoch: int, metric_value: float) -> Path:
        assert self.checkpoint_config is not None
        ckpt_dir = Path(self.checkpoint_config.dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"epoch_{epoch:03d}.pt"
        path = ckpt_dir / file_name
        payload = {
            "epoch": epoch,
            "metric": metric_value,
            "model_state": self.model.state_dict(),
            "classifier_state": self.classifier.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(payload, path)
        print(f"Saved checkpoint: {path} ({metric_value:.4f})")
        return path

    def _register_checkpoint(self, path: Path) -> None:
        assert self.checkpoint_config is not None
        self._checkpoint_paths.append(path)
        keep_last = max(1, self.checkpoint_config.keep_last)
        while len(self._checkpoint_paths) > keep_last:
            stale = self._checkpoint_paths.popleft()
            if stale.exists():
                stale.unlink()
                print(f"Removed stale checkpoint: {stale}")

    def _log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        metric_items = [f"{k}={v:.4f}" for k, v in metrics.items() if k != "global_step"]
        log_message = f"Epoch {epoch}: " + ", ".join(metric_items)
        print(log_message)
