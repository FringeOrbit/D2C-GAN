# Model weights

The trained checkpoint is distributed separately from this source repository.
This directory intentionally contains no weight file.

After downloading the checkpoint, place it at the path configured by
`config.py`, together with the matching standardisation file. The checkpoint
must correspond to the covariance version of D2C-GAN.

The corrected default generator is the parallel/Softmax architecture of the
original `Ours_Full` checkpoints, not the sequential generator from `93b2e61`.
For checkpoint hashes and protocol limitations, see
[`docs/RELEASE_CORRECTION.md`](../docs/RELEASE_CORRECTION.md).
Check compatibility before evaluation:

```bash
python scripts/check_checkpoint.py path/to/best_model.pth
```
