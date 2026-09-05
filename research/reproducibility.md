# Reproducibility Checklist

## Environment

- [ ] Python version recorded
- [ ] PyTorch version recorded
- [ ] CUDA version recorded when applicable
- [ ] GPU/CPU model recorded
- [ ] Dependencies pinned

## Data

- [ ] Dataset source recorded
- [ ] Dataset version/date recorded
- [ ] License/terms reviewed
- [ ] Split generation documented
- [ ] Leakage checks performed
- [ ] Preprocessing documented

## Training

- [ ] Random seed recorded
- [ ] Batch size recorded
- [ ] Optimizer recorded
- [ ] Learning rate recorded
- [ ] Scheduler recorded
- [ ] Loss function recorded
- [ ] Epoch/stopping policy recorded
- [ ] Augmentation recorded

## Evaluation

- [ ] Test set held out until evaluation
- [ ] Metric implementation tested
- [ ] Per-run metrics saved
- [ ] Mean/std reported when repeated runs are used
- [ ] Timing procedure documented

## Artifacts

- [ ] Configuration files saved
- [ ] Checkpoints identified
- [ ] Training curves saved
- [ ] Representative predictions saved
- [ ] Final tables generated from raw results

## Reproduction Quality

A reproduction is considered complete only when another researcher can reconstruct the environment, understand the data pipeline, reproduce the experiment configuration and trace reported metrics to stored outputs.