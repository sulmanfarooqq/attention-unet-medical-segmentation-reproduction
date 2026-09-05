# Notebook Workflow

Use notebooks for exploration and visualization, not as the only location for the final experiment logic.

1. `01_dataset_analysis.ipynb` — inspect dimensions, masks, class balance and sample cases.
2. `02_baseline_experiment.ipynb` — inspect U-Net training and outputs.
3. `03_attention_unet.ipynb` — inspect Attention U-Net behavior.
4. `04_results_analysis.ipynb` — aggregate saved results and generate figures.

Reusable training/evaluation logic belongs in `src/` so experiments remain scriptable and reproducible.