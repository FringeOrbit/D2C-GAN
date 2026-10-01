# Release correction and provenance — 2026-10-01

## What is corrected

The generator published at commit `93b2e61dbb3e6a194646b66359f02200ba0ff954`
was a sequential implementation. The original North Sea, Kansas and NW-Otway
`Ours_Full` checkpoint parameter names instead contain five `time_branch`,
`freq_branch` and `weight_net` groups. The corrected default generator is the
parallel implementation recovered from the audited server source. Its two-way
Softmax is enabled by `use_learned_branch_weights=True`.

The old sequential and DAPA variants are retained in `models/legacy_generator.py`
for explicit historical use, not silently substituted for the original model.
The original four-channel first convolution and unused optional `mask` argument
are preserved. Changing the input to eight channels would create a different
architecture and break original checkpoint compatibility.

Configuration now uses the audited server's 200-epoch default. This is not a
claim that every historical run's full configuration was recovered. Data paths
remain portable and local. The training entry point now resolves repository-root
imports and honours `config.val_path`.

## Covariance is retained, not replaced with Wyllie

The published `petrophysical_consistency_loss` formula matches the recovered
pre-Wyllie Python 3.13 trainer cache:

1. Compute each curve's mean on its own artificially missing, native-valid set.
2. Center and mask each curve separately.
3. Divide each cross-product by the corresponding pair-overlap count plus epsilon.
4. Match the two matrices with elementwise MSE.

That formula is deliberately unchanged. The recovered cache's
`train_step` calls it with `artificial_missing.float()`.
The old cache is evidence of a historical implementation, not a signed source
snapshot of the original May training run. Its SHA256 is
`021b04547eda7d043974c0985361de566b5888d8513aaa2725d9f60bc82fa371`.

The common-four artificial-target intersection and `np.cov(..., ddof=1)`
were found in the saved figure-diagnostic workflow. They do not establish that
the same formula was used in training. The later Wyllie trial has its own
checkpoint; it is not substituted for the original covariance checkpoint here.

## Evaluation protocols are explicit

| Entry point | Windows | Artificial masks |
| --- | --- | --- |
| `evaluate_d2cgan.py` | Legacy full-table reshape; may cross well boundaries | Pointwise random |
| `evaluation/external_espirito_santo.py` | Non-overlapping 80-point windows within a well; padding excluded from targets | Pointwise random with stable well/rate seeds |

The external evaluator is ported from the audited PCC external-evaluation
script. Changes are repository-root imports, restricted/strict checkpoint
loading, diagnostic-only defaults for Wyllie constants and explicit protocol
metadata. Wyllie residual outputs are diagnostics, not a training objective.
The port preserves the existing dataset preprocessing and missingness semantics;
it is not a new cleaned dataset or a newly evaluated comparison.

The training dataset includes both pointwise and contiguous-block corruption.
This does not mean the saved test results used contiguous blocks. The audited
saved figure masks had more than one missing run in 6,588 of 6,593 fully
native-valid window/curve pairs. No continuous-block result is newly claimed.

## Original checkpoint identity

| Variant | SHA256 |
| --- | --- |
| North Sea `Ours_Full` | `e7046ac0940578487990fdf71ffc682b8534e15e3e1751f87007f0a425049234` |
| Kansas `Ours_Full_kansasi` | `2cd7381a18d7c0bf3e4b53269e04e99ad29bfe6f98ba988d987102bde591b230` |
| NW-Otway `Ours_Full_outewei` | `d101a0c97d34ba54cf456875cd9c8c7bb4687fc515b73120b4bd77f0232f3c1c` |

The North Sea covariance-recheck checkpoint is byte-for-byte identical to the
North Sea original. The three originals contain `Geo_Petro` loss histories,
whereas the later seed trial contains `Physics_Wyllie`. These names and hashes
help identify artifacts; weights alone cannot prove the exact historical loss.
The retained North Sea training log's best validation value also differs from
the original checkpoint's recorded value, so that log is not treated as a
complete checkpoint-to-run provenance record.

Use `python scripts/check_checkpoint.py path/to/best_model.pth` for a read-only
strict check. Arbitrary checkpoint globals are rejected; shape/key mismatches
are not hidden by `strict=False`. Weights, raw data and private server paths
are not added to this source release.

## Boundaries of this correction

- Saved manuscript figures, summary tables and reported metric values are unchanged.
- No model is retrained and no paper comparison is regenerated.
- Original checkpoints remain four-channel models, not the manuscript's eight-channel description.
- Common-four sample covariance and continuous-block testing remain manuscript/implementation discrepancies to resolve with provenance or new experiments.
- Shared-target consistency of the internal RMSE/R² tables is not fixed by changing the generator.
- This release is a checkpoint-compatible architecture correction and evidence-based clarification, not a claim of complete reproduction of all paper experiments.

## 中文摘要

此次修正默认生成器，使其匹配原模型的五个时频并行块及 Softmax 门控；保留误发布的串联版本作为 legacy，不改动历史协方差损失公式。输入仍为四通道，实际外部测试仍为井内窗口＋逐点随机掩码。共同四曲线样本协方差属于图件诊断，连续块测试与稿件的一致性尚待进一步解决。此次未改动论文结果或原始权重，也未声称全部实验已经复现。
