"""Regression checks for the corrected historical architecture and protocols."""

import ast
from datetime import date
import io
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from checkpoint_io import load_generator_state
from config import OptimizedConfig
from evaluation.external_espirito_santo import evaluate_well, metric_row, pooled_summary
from models.generator import AdvancedSeqGenerator, ParallelTimeFrequencyBlock
from models.legacy_generator import AdvancedSeqGenerator as SequentialGenerator


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.config = OptimizedConfig()
        self.config.device = torch.device("cpu")

    def test_parallel_softmax_branches_receive_same_input(self):
        block = ParallelTimeFrequencyBlock(8, 2, config=self.config).eval()
        x = torch.randn(2, 8, 80)
        inputs = []
        handles = [branch.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
                   for branch in (block.time_branch, block.freq_branch)]
        with torch.no_grad():
            actual = block(x)
        for handle in handles:
            handle.remove()
        self.assertTrue(all(value is x for value in inputs))
        weights = block.weight_net(x)
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(2)))
        expected = weights[:, 0, None, None] * block.time_branch(x)
        expected += weights[:, 1, None, None] * block.freq_branch(x)
        self.assertTrue(torch.allclose(actual, expected))

    def test_five_blocks_four_channels_and_mask_compatibility(self):
        model = AdvancedSeqGenerator(self.config).eval()
        blocks = [item for item in model.modules() if isinstance(item, ParallelTimeFrequencyBlock)]
        self.assertEqual(len(blocks), 5)
        self.assertEqual(tuple(model.feature_extractor[0].weight.shape), (128, 4, 1))
        x = torch.randn(2, 4, 80)
        with torch.no_grad():
            first = model(x, mask=torch.ones_like(x))
            second = model(x, mask=torch.zeros_like(x))
        self.assertEqual(tuple(first.shape), (2, 4, 80))
        self.assertTrue(torch.equal(first, second))  # historical mask argument is unused

    def test_fft_disabled(self):
        block = ParallelTimeFrequencyBlock(8, 1, use_fft=False, config=self.config).eval()
        x = torch.randn(2, 8, 80)
        self.assertTrue(torch.equal(block(x), block.time_branch(x)))

    def test_restricted_checkpoint_round_trip(self):
        model = AdvancedSeqGenerator(self.config)
        buffer = io.BytesIO()
        torch.save({"generator": model.state_dict(), "config": self.config}, buffer)
        buffer.seek(0)
        restored = AdvancedSeqGenerator(self.config)
        restored.load_state_dict(load_generator_state(buffer), strict=True)

    def test_unknown_checkpoint_global_rejected(self):
        buffer = io.BytesIO()
        torch.save({"generator": {"weight": torch.ones(1)}, "unexpected": date(2026, 1, 1)}, buffer)
        buffer.seek(0)
        with self.assertRaises(Exception):
            load_generator_state(buffer)

    def test_sequential_checkpoint_rejected(self):
        state = SequentialGenerator(self.config).state_dict()
        with self.assertRaises(RuntimeError):
            AdvancedSeqGenerator(self.config).load_state_dict(state, strict=True)

    def test_covariance_formula_retains_pair_overlap(self):
        # Compile just the loss method, without importing the plotting/training entry point.
        path = Path(__file__).resolve().parents[1] / "training" / "train_d2cgan.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CGANTrainer")
        method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                      and node.name == "petrophysical_consistency_loss")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        holder = type("Holder", (), {"device": torch.device("cpu"), "mse_for_cov": torch.nn.MSELoss()})()
        real = torch.tensor([[[1., 2., 4., 8.], [2., 3., 5., 9.],
                              [0., 1., 2., 3.], [3., 2., 1., 0.]]])
        fake = real + torch.tensor([[[0., 2., 0., 0.], [0., 0., 1., 0.],
                                    [0., 0., 0., 0.], [0., 0., 0., 0.]]])
        mask = torch.tensor([[[1., 1., 0., 0.], [0., 1., 1., 0.],
                              [0., 0., 1., 1.], [1., 0., 0., 1.]]])
        self.assertFalse(mask.bool().all(dim=1).any())
        def covariance(value):
            mean = (value * mask).sum(2, keepdim=True) / (mask.sum(2, keepdim=True) + 1e-8)
            centered = (value - mean) * mask
            return torch.bmm(centered, centered.transpose(1, 2)) / (
                torch.bmm(mask, mask.transpose(1, 2)) + 1e-8)
        actual = namespace[method.name](holder, fake, real, mask)
        expected = torch.nn.functional.mse_loss(covariance(fake), covariance(real))
        self.assertTrue(torch.allclose(actual, expected))
        self.assertGreater(actual.item(), 0)
        self.assertEqual(namespace[method.name](holder, fake, real, torch.zeros_like(mask)).item(), 0)

    def test_per_well_evaluator_targets_and_padding(self):
        class Identity(torch.nn.Module):
            def forward(self, x, mask=None):
                return x
        raw = np.arange(81 * 4, dtype=np.float32).reshape(81, 4) / 100
        frame = pd.DataFrame(raw, columns=self.config.feature_names)
        frame["WELL"] = "synthetic-well"
        frame["LITH"] = "onshore"
        frame["DEPTH_MD"] = np.arange(81)
        frame.loc[5, "GR"] = np.nan
        scaler = StandardScaler().fit(raw)
        rows, _, _, chunks = evaluate_well(Identity(), scaler, self.config, frame, 20260804)
        rows_again, _, _, chunks_again = evaluate_well(Identity(), scaler, self.config, frame, 20260804)
        self.assertEqual([row["n"] for row in rows], [row["n"] for row in rows_again])
        self.assertTrue(all(row["n"] <= 81 for row in rows))
        for first, second in zip(chunks, chunks_again):
            self.assertTrue(np.array_equal(first["depth"], second["depth"]))
            self.assertTrue(np.isfinite(first["y_true"]).all())
            self.assertTrue((first["depth"] < 81).all())
            if first["feature"] == "GR":
                self.assertFalse((first["depth"] == 5).any())

    def test_pooled_metrics_match_pooled_targets(self):
        ys = [np.array([1., 2., 5.]), np.array([9., 10.])]
        preds = [np.array([1.2, 2.1, 4.]), np.array([8., 11.])]
        frame = pd.DataFrame([metric_row(str(i), .8, "GR", y, p)
                              for i, (y, p) in enumerate(zip(ys, preds))])
        row = pooled_summary(frame, ["missing_rate", "feature"]).iloc[0]
        self.assertAlmostEqual(row.rmse ** 2 / (1 - row.r2), np.var(np.concatenate(ys)))


if __name__ == "__main__":
    unittest.main()
