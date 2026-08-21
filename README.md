# D2C-GAN

[English](README.md) | [中文](README_zh.md)

## Overview

D2C-GAN is a structured adversarial framework for reconstructing incomplete
multivariate well logs. The implementation targets four commonly used curves:
gamma ray (GR), bulk density (RHOB), neutron porosity (NPHI) and compressional
sonic slowness (DTC).

The accompanying paper studies reconstruction under artificial missingness and
focuses on a central practical problem: recovering numerical values while
preserving the local shape of long depth sequences and the statistical
dependence between different curves. The repository contains the model source,
the evaluation helper, and the paper-level result files used to prepare the
reported analyses.

## Model design

The implementation follows four complementary design choices:

- A masked-input sequence generator processes the depth-domain signal with
  dilated residual blocks. The dilation schedule is configurable and is used
  to capture both local variations and longer depth-range context.
- Frequency-aware feature blocks transform sequence features into the Fourier
  domain and return them to the sequence representation, providing an
  additional description of broad trends and oscillatory structure.
- A Patch Hybrid Sequence Discriminator contains separate structure and
  texture streams. It evaluates the reconstructed multivariate sequence and
  the GR texture stream at different levels of detail.
- The generator objective combines point-wise reconstruction, cross-curve
  covariance consistency and adjacent-difference regularisation. The latter
  covers boundary matching and suppression of unnecessary local changes.

Together, these components are intended to make reconstruction quality depend
not only on point-wise error, but also on depth-wise signal structure and
cross-curve relationships.

## Repository scope

This repository contains:

- the D2C-GAN generator and discriminator;
- the training script and covariance-version configuration;
- the paper-style evaluation helper;
- manuscript figures, compact source data and summary metrics;
- logs describing the reported diagnostic and external-validation analyses.

Raw well-log files, baseline implementations and trained model weights are not
included in the source tree. The public result files are derived from the
analyses reported in the paper and do not replace the original well-log data.
Model weights are distributed separately; see [`weights/README.md`](weights/README.md).

## Repository layout

```text
models/
  generator.py              # sequence and frequency-aware generator blocks
  discriminator.py          # structure/texture PatchGAN discriminator
training/
  cgan.py                   # dataset wrapper, losses and training loop
config.py                   # covariance-version model and training settings
evaluate_paper_style.py     # paper-style RMSE/R² evaluation helper
results/
  figures/                  # manuscript figures and compact figure data
  metrics/                  # summary metrics and bootstrap files
  external_validation/      # independent-basin summary files
  logs/                     # figure and diagnostic experiment records
weights/
  README.md                 # external checkpoint instructions
```

## Installation

Python dependencies are listed in [`requirements.txt`](requirements.txt).
From the repository root:

```bash
pip install -r requirements.txt
```

The implementation uses PyTorch and automatically selects a CUDA device when
one is available. Otherwise, it falls back to CPU execution.

## Data and configuration

The default paths in `config.py` expect processed CSV files and a fitted
standardisation object under `data/`:

```text
data/
  train_processed.csv
  val_processed.csv
  test_processed.csv
  scaler.pkl
```

The input schema is configured through `well_column`, `depth_column` and
`feature_names`. Before training or evaluation, place the locally available
data and matching standardisation file at the configured paths, or edit the
paths in `config.py`. The supplied configuration uses an 80-sample sequence
window and the four target curves listed above.

## Training

After preparing the local data and standardisation file, run:

```bash
python training/cgan.py
```

The training script constructs the well-level datasets, creates the generator
and Patch Hybrid Sequence Discriminator, evaluates the validation reconstruction
loss, applies early stopping when enabled, and saves the best generator
checkpoint under the configured checkpoint directory.

The covariance-version setting is controlled by `weight_petro` in `config.py`.
The same configuration also exposes the frequency block, dilation and CBAM
switches used for component-wise experiments.

## Evaluation

`evaluate_paper_style.py` loads a generator checkpoint and a fitted scaler,
then evaluates the four target curves at 20%, 40%, 60% and 80% missing rates.
The helper reports per-curve RMSE and R² values and can generate depth-curve
visualisations. Run it after placing the checkpoint and scaler at the paths
specified in `config.py`:

```bash
python evaluate_paper_style.py
```

The manuscript reports RMSE, MAE, R² and PCC. The compact result files under
`results/` provide the corresponding paper-level summaries, figure source data
and external well-level metrics.

## Paper-level results

The result files are organised as follows:

| Manuscript material | Repository location |
| --- | --- |
| Figures 1–7 | `results/figures/` |
| Covariance and cross-plot diagnostics | `results/figures/` |
| PSD comparison summaries | `results/figures/` |
| Independent Espírito Santo evaluation | `results/external_validation/` |
| Well-level bootstrap summary | `results/metrics/` |
| Figure-generation and diagnostic records | `results/logs/` |

The external-validation files contain summary outputs for the frozen covariance
version of D2C-GAN. They do not contain the original external well-log files.

## Citation

If you use the implementation or the derived result files, please cite the
accompanying manuscript and acknowledge this repository:

```text
D2C-GAN: structured adversarial reconstruction of incomplete multivariate
well logs. Source code and paper-level results:
https://github.com/FringeOrbit/D2C-GAN
```

## License and use

Please check the repository license and the terms of the original data sources
before redistributing data or derived materials. The result files in this
repository are provided to support inspection and reproduction of the reported
analyses.
