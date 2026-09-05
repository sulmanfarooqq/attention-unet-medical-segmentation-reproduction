import torch

from src.evaluation.metrics import dice_score, iou_score, precision_score, recall_score


def test_perfect_prediction() -> None:
    target = torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.float32)
    assert torch.isclose(dice_score(target, target), torch.tensor(1.0))
    assert torch.isclose(iou_score(target, target), torch.tensor(1.0))
    assert torch.isclose(precision_score(target, target), torch.tensor(1.0))
    assert torch.isclose(recall_score(target, target), torch.tensor(1.0))
