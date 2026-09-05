"""Binary segmentation metrics implemented without third-party metric packages."""

import torch


def _validate(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    pred = pred.float().reshape(pred.shape[0], -1)
    target = target.float().reshape(target.shape[0], -1)
    return pred, target


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    pred, target = _validate(pred, target)
    pred = (pred >= threshold).float()
    intersection = (pred * target).sum(dim=1)
    return ((2 * intersection + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)).mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    pred, target = _validate(pred, target)
    pred = (pred >= threshold).float()
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def precision_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    pred, target = _validate(pred, target)
    pred = (pred >= threshold).float()
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    return ((tp + eps) / (tp + fp + eps)).mean()


def recall_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    pred, target = _validate(pred, target)
    pred = (pred >= threshold).float()
    tp = (pred * target).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)
    return ((tp + eps) / (tp + fn + eps)).mean()
