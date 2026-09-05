# Attention U-Net Medical Segmentation Reproduction

A reproducibility-focused research project studying whether the reported benefits of attention gates in Attention U-Net can be reproduced under a controlled experimental protocol.

## Why This Project Matters for a Research Internship

This repository is being developed as a **research portfolio project**. Its purpose is to demonstrate research ability through a complete process: literature review, research-question formulation, experimental design, implementation, controlled experiments, analysis, limitations, and reproducibility.

It is intended to support—not replace—the broader evidence required in a competitive research-internship application.

See [`research/mitacs_alignment.md`](research/mitacs_alignment.md) for the evidence this project is designed to produce.

## Research Question

**Can Attention U-Net reproduce the reported advantages of attention gates for biomedical image segmentation, and what trade-offs do the gates introduce in segmentation quality and computational cost compared with a standard U-Net?**

## Research Objectives

1. Study the original U-Net and Attention U-Net architectures.
2. Re-implement both models in PyTorch from the published descriptions.
3. Use a documented public biomedical segmentation dataset.
4. Establish a controlled U-Net baseline.
5. Evaluate Attention U-Net under matched training conditions.
6. Run an attention-gate ablation study.
7. Measure Dice, IoU, precision, recall, parameter count and inference latency.
8. Analyze qualitative segmentation behavior.
9. Compare reproduced findings with the original paper's claims.
10. Document limitations, deviations and reproducibility details.
11. Produce a research-style technical report suitable for portfolio review.
12. Preserve experiment configurations, seeds, results and research notes so another researcher can reproduce the study.

## Hypothesis

Attention gates may improve segmentation quality by suppressing irrelevant image regions and emphasizing salient structures, while introducing a measurable but potentially modest computational overhead.

This is a hypothesis. **No experimental result is claimed until it has been measured and recorded.**

## Primary Study

Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas* (2018), arXiv:1804.03999.

Baseline: Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015), arXiv:1505.04597.

## Experimental Design

| Experiment | Purpose |
|---|---|
| E0 | Environment and reproducibility validation |
| E1 | Dataset analysis and preprocessing validation |
| E2 | Standard U-Net baseline |
| E3 | Attention U-Net |
| E4 | Attention-gate ablation |
| E5 | Efficiency and inference analysis |
| E6 | Error and qualitative analysis |
| E7 | Reproduction comparison and final analysis |

All model comparisons should use the same dataset split, preprocessing, random-seed policy, optimizer family, training budget and evaluation protocol unless an experiment explicitly studies one of these variables.

## Research Evidence Roadmap

A finished portfolio-quality study should contain:

- Literature review with source extraction.
- Explicit research question and hypotheses.
- Dataset provenance and licensing/access notes.
- Reproducible environment and seed policy.
- Baseline U-Net implementation and experiment.
- Attention U-Net implementation and experiment.
- Attention-gate ablation study.
- Quantitative metrics and computational measurements.
- Repeated-run/statistical analysis where feasible.
- Qualitative error analysis and figures.
- Comparison against the published study, including documented deviations.
- Final research report and reproducibility instructions.
- Honest research diary and hour log.

## Repository Structure

```text
research/
  research_question.md
  literature_review.md
  methodology.md
  experimental_protocol.md
  reproducibility.md
  limitations.md
  mitacs_alignment.md
  hour_log.md

src/
  models/
    unet.py
    attention_unet.py
  data/
    dataset.py
    preprocessing.py
  training/
    train.py
  evaluation/
    metrics.py
    evaluate.py
  utils/
    seed.py

experiments/
  baseline_unet/README.md
  attention_unet/README.md
  ablation/README.md
  results/README.md

notebooks/
  README.md

reports/
  final_report.md

tests/
  test_models.py
  test_metrics.py

.github/
  workflows/ci.yml

requirements.txt
.gitignore
LICENSE
```

## 200-Hour Research Program

The target is a **planned 200-hour research program**, not a fabricated claim of completed work.

| Phase | Hours | Status |
|---|---:|---|
| Literature review | 20 | Planned |
| Research design | 12 | Planned |
| Dataset investigation | 15 | Planned |
| Data pipeline | 18 | Planned |
| U-Net implementation | 18 | Planned |
| Attention U-Net implementation | 22 | Planned |
| Baseline experiments | 22 | Planned |
| Attention experiments | 22 | Planned |
| Ablation study | 15 | Planned |
| Efficiency/error analysis | 12 | Planned |
| Statistical/result analysis | 10 | Planned |
| Final report/reproducibility | 14 | Planned |
| **Total** | **200** | **Planned** |

Actual hours must be entered in `research/hour_log.md` as the work is performed.

## Status

**Phase 0 — Research framework initialized.**

No performance numbers are currently reported. Results folders contain templates only until experiments are actually executed.

## Ethics and Scope

This project is an academic research/reproducibility study. It does not provide clinical diagnoses, treatment recommendations, or validated medical decision support. Dataset licenses and usage conditions must be checked before downloading or redistributing data.

## Research Integrity

This repository deliberately does not fabricate research hours, experimental improvements, publications, awards, supervisor endorsements, or clinical claims. Any result included in the final study must be traceable to an executed experiment and recorded methodology.

## License

MIT License. See `LICENSE`.
