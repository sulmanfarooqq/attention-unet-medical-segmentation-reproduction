# Methodology

## Study Type

Controlled deep-learning reproducibility and ablation study.

## Models

### Baseline

A clean PyTorch implementation of the original U-Net architecture.

### Intervention

A PyTorch implementation of Attention U-Net using attention gates on the encoder-to-decoder skip pathways.

### Ablation

An otherwise matched model with the attention mechanism disabled. The purpose is to estimate the contribution of the attention mechanism rather than merely compare two independently tuned systems.

## Controlled Variables

Where practical, the following will remain fixed across comparisons:

- Dataset and split
- Image preprocessing
- Image dimensions
- Batch size
- Optimizer
- Learning-rate policy
- Loss function
- Training epochs / stopping rule
- Data augmentation
- Random-seed policy
- Hardware environment
- Evaluation code

Any deviation must be recorded in `research/experimental_protocol.md`.

## Primary Metrics

- Dice similarity coefficient
- Intersection over Union (IoU)
- Precision
- Recall

## Efficiency Metrics

- Number of trainable parameters
- Approximate computational cost where tooling permits
- Single-image inference latency
- Throughput where measurable
- Peak GPU memory where measurable

Latency must be measured with a documented hardware/software configuration and a warm-up procedure. CPU and GPU measurements must not be mixed.

## Statistical Practice

Where computational resources permit, repeat the principal comparison across multiple seeds. Report mean and standard deviation rather than selecting the most favorable run. Preserve per-run results in machine-readable files.

## Qualitative Analysis

Store representative input images, ground-truth masks and predictions. Include both successful and failure cases. Selection criteria must be stated rather than choosing examples only because they look favorable.

## Reproducibility

Record Python version, PyTorch version, CUDA version, hardware, package versions, seed, dataset version/source, preprocessing settings and exact training configuration.