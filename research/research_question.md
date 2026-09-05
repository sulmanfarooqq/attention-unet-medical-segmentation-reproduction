# Research Question

## Main Question

Can Attention U-Net reproduce the reported advantages of attention gates for biomedical image segmentation, and what trade-offs do the gates introduce in segmentation quality and computational cost compared with a standard U-Net?

## Sub-questions

1. Does Attention U-Net achieve higher Dice and IoU than a matched U-Net baseline?
2. Do attention gates change precision/recall behavior?
3. How much additional computation and latency do attention gates introduce?
4. Which types of image regions benefit most from attention gating?
5. Does removing the gates reduce the observed performance, supporting a causal interpretation of their contribution?
6. How closely do the reproduced trends agree with the original publication?

## Hypotheses

**H1:** Attention U-Net will improve segmentation quality relative to a matched U-Net baseline.

**H2:** Attention gates will introduce additional parameters and computational overhead.

**H3:** The ablated model without attention gates will perform differently from the complete Attention U-Net, allowing the contribution of the gates to be examined directly.

These hypotheses are prospective. They must not be converted into conclusions until experiments are completed.