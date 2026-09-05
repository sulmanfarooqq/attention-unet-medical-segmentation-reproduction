# E4 — Attention-Gate Ablation

## Question
Are observed differences attributable to the attention gates, rather than unrelated implementation or training differences?

## Design

Compare:

1. Standard U-Net.
2. Complete Attention U-Net.
3. Attention architecture with gates disabled or bypassed while retaining the surrounding implementation where possible.

Use the same data split, preprocessing, optimizer, learning-rate policy, training budget and evaluation code.

## Interpretation

Ablation results should be interpreted together with variance, computational cost and qualitative examples. A single favorable run is insufficient evidence.