---
date: 2026-08-07
task: well-level uncertainty support for pooled diagnostic figures
dataset: FORCE 2020 / North Sea test split
missing_rate: 0.80
independent_unit: well
bootstrap_replicates: 5000
---

# Well-level uncertainty support for diagnostic figures

## Scope

The existing saved test predictions and ground-truth arrays were reused. No model inference or training changes were required for this figure-statistics revision.

## Figure 5: pooled RHOB-NPHI diagnostic

- 63,777 paired artificially masked, native-valid cells were pooled across 16 test wells.
- A well-level source-data table was added with paired Pearson correlations for the ground truth and D2C-GAN reconstruction.
- Mean Pearson correlation and percentile bootstrap intervals were calculated by resampling wells, not individual cells.
- Ground truth: mean -0.254, 95% CI [-0.458, -0.043].
- D2C-GAN: mean -0.277, 95% CI [-0.480, -0.076].

## Figure 6: covariance diagnostic

- The pooled covariance matrices use 38,075 common four-curve target cells across 16 wells in standardized space.
- Panel d reports the per-well Frobenius norm of the covariance difference.
- Mean well-level Frobenius error: 0.0142.
- 95% percentile bootstrap interval: [0.0098, 0.0197].

## Figure 7: PSD diagnostic

- Welch PSDs were computed on the same valid 80-sample evaluation windows.
- Windows were first averaged within each well and then summarized across wells.
- Shaded bands show 95% percentile bootstrap intervals across wells for ground truth and D2C-GAN.
- Valid window counts are GR 1,917, RHOB 1,772, NPHI 1,251 and DTC 1,632; 16 wells contribute to each curve-level summary.

## Outputs

- `figures/crossplot_rhob_nphi_pooled.{png,pdf,svg,tiff}`
- `figures/covariance_matrix_pooled.{png,pdf,svg,tiff}`
- `figures/psd_multi_baseline_pooled.{png,pdf,svg,tiff}`
- `figures/figure5_well_joint_statistics.csv`
- `figures/figure6_well_covariance_statistics.csv`
- `figures/figure7_psd_summary.csv`
- `figures/redraw_figures_5_6_pooled.py`

The figure source and vector exports use editable text, and the raster exports use 600 dpi. The figures distinguish pooled cell-level diagnostics from well-level uncertainty summaries.

## Layout revision on 2026-08-08

- The Figure 5 density colorbar is placed in a dedicated external column and no longer overlaps the reconstruction panel.
- Figure 5 uses the NPHI y-axis title only on panel a because panels a-b share the same y-scale; panel b retains the tick labels without repeating the title.
- Figure 5 panel labels a-b were moved closer to the upper-left corners of their respective axes to avoid excessive whitespace.
- Figure 6 is exported as a single-column 2 x 2 layout. The covariance heatmaps use a unified low-saturation palette; the well-level panel uses star markers for individual wells, a diamond for the mean and a dotted line for the pooled reference.
- All four Figure 6 panels use a square box aspect, and the vertical spacing between rows is reduced to improve single-column compactness.
- Figure 6 panels a-c now share the same signed diverging map (blue-gray for negative and terracotta for positive values). Panel c is non-negative by definition, so it occupies the neutral-to-terracotta half of the same map; darker cells indicate larger absolute differences.
- Figure 7 no longer uses arbitrary low- and high-frequency background shading. The frequency axis is shown on a single neutral background; the PSD curves and well-level confidence bands carry the statistical information.
