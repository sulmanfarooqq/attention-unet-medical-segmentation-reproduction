"""Utilities for model parameter and inference measurements."""

import time

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_latency(model: nn.Module, sample: torch.Tensor, warmup: int = 10, runs: int = 50) -> float:
    """Return mean single-forward latency in milliseconds after warm-up."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)
        if sample.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(sample)
        if sample.is_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / runs
