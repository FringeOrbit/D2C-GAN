# D2C-GAN

Reference implementation and paper-level result files for D2C-GAN, a structured
adversarial framework for multivariate well-log reconstruction.

The implementation reconstructs four logs: GR, RHOB, NPHI and DTC. The model
combines depth-domain and frequency-domain features, a dual-stream PatchGAN
discriminator, masked reconstruction loss, cross-curve covariance consistency,
and adjacent-difference regularisation.

## Repository scope

This repository contains the D2C-GAN implementation and the result files used
in the manuscript. Raw well-log data and baseline implementations are not
included. Dataset sources and access information are documented in the paper.

## Source layout

```text
models/
  generator.py
  discriminator.py
training/
  cgan.py
config.py
evaluate_paper_style.py
results/
  figures/
  metrics/
  external_validation/
  logs/
```

`config.py` contains the covariance-version model and training settings used by
the supplied source. `evaluate_paper_style.py` loads a generator checkpoint,
applies the manuscript missing-rate protocol and reports RMSE and R2 results.
Raw datasets are intentionally not included; configure the documented data
paths locally before running training or evaluation.

Model weights are distributed separately. See `weights/README.md` for the
external download location.

## Curves and metrics

The reported numerical metrics are RMSE, MAE, R2 and PCC. The repository also
contains the manuscript-level covariance, cross-plot and PSD summaries without
including the underlying raw wells.

## Citation

Please cite the accompanying manuscript when using this code or the derived
result files.
