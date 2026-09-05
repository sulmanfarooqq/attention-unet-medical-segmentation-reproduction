# Literature Review

## 1. U-Net

Ronneberger, Fischer and Brox introduced U-Net for biomedical image segmentation. Its encoder extracts contextual features while a decoder restores spatial resolution, with skip connections transferring high-resolution information between corresponding levels.

Reference: Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, arXiv:1505.04597.

## 2. Attention U-Net

Oktay et al. proposed Attention U-Net by inserting attention gates into the U-Net architecture. The gates are intended to suppress irrelevant regions and emphasize task-relevant features before information is passed through skip connections.

Reference: Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*, arXiv:1804.03999.

## 3. Research Gap for This Project

The purpose of this repository is not to claim a new architecture. It is to perform a transparent reproduction study: reconstruct the architectures, control experimental variables, measure performance and computational cost, and investigate whether the reported qualitative advantage of attention gating appears under the selected experimental setting.

## 4. Evidence Extraction Table

| Source | Architecture | Dataset | Metrics | Main finding | Reproduced? |
|---|---|---|---|---|---|
| Ronneberger et al. | U-Net | Biomedical segmentation | Reported in paper | Baseline architecture | Pending |
| Oktay et al. | Attention U-Net | Abdominal CT / pancreas segmentation | Reported in paper | Attention gates improve focus on salient regions | Pending |

## 5. Additional Literature To Add

Before final conclusions, review at least 8–12 additional peer-reviewed or preprint sources covering:

- Biomedical image segmentation
- Attention mechanisms in CNNs
- Attention U-Net variants
- Segmentation evaluation metrics
- Reproducibility in deep learning
- Computational efficiency of segmentation networks

Each source should be recorded with its citation, research question, dataset, method, result and relevance to this study.