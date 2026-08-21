---
date: 2026-08-06
task: pooled North Sea diagnostics for Figures 5 and 6
dataset: FORCE 2020 / North Sea test split
missing_rate: 0.80
seed: 20260806
---

# Figure 5 and Figure 6 pooled diagnostics

## Evaluation contract

- Test split: North Sea (`data/beihai/test_processed.csv`) with 16 wells and 153,634 depth samples.
- Windowing: 80-sample non-overlapping windows within each well; final windows were edge-padded for inference and padded positions were excluded from all summaries.
- Artificial masking: one deterministic 80% mask per well, shared by D2C-GAN, Bi-LSTM, Bi-GRU and Transformer.
- Target rule: only artificially masked cells that were native-valid before masking were scored or plotted.
- Model checkpoints: frozen checkpoints under `ablation_checkpoints`; the covariance-matching D2C-GAN checkpoint is `Ours_Full/best_model.pth`.

## Figure 5 and covariance companion figure

- Pooled paired RHOB-NPHI target cells: 63,777.
- Common four-curve target cells for covariance matrices: 38,075.
- Covariance matrices are computed in standardized model space, matching the scale of the covariance regularizer.
- Absolute covariance-difference Frobenius norm: 0.0105767 (displayed as 0.011).
- Source files: `figure_05_crossplot_pooled.npz`, `figure_05_covariance_ground_truth.csv`, `figure_05_covariance_d2cgan.csv`, `figure_05_covariance_absolute_difference.csv`.
- The joint-distribution plot and covariance matrix are exported as separate figure assets because they require different aspect ratios and visual scales.

## Figure 6

- Welch PSD uses 64-point segments on complete 80-point evaluation windows.
- Each window is reconstructed by retaining observed native-valid values and replacing only artificial targets with the model prediction.
- PSDs are area-normalized per window. Curves and bands use the geometric mean and +/-1 standard deviation in log-PSD across windows, which is appropriate for the logarithmic ordinate.
- Valid windows: GR 1,917; RHOB 1,772; NPHI 1,251; DTC 1,632.
- Source file: `figure_06_psd_summary.csv`.

## Outputs

- `figures/crossplot_rhob_nphi_pooled.{png,pdf,svg,tiff}`
- `figures/covariance_matrix_pooled.{png,pdf,svg,tiff}`
- `figures/psd_multi_baseline_pooled.{png,pdf,svg,tiff}`
- `figures/redraw_figures_5_6_pooled.py`
- `figures/export_pooled_test_arrays.py`

The Nature figure preflight reported 13 passes, 0 failures and one conservative warning for the logarithmic PSD transform; all PSD values are clipped to a positive floor before log transformation and the plotted lower limits are set from positive band values. Local LaTeX compilation was not available because neither `pdflatex` nor `xelatex` is installed in the workstation environment.
