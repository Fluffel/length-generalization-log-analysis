#!/usr/bin/env python3
"""Tests for single-run weighted-max selection used by --max-aggregation max."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_utils import (  # noqa: E402
    bin_weight,
    model_size_key,
    select_max_run,
    select_weighted_max_run,
    single_run_line_xy,
    weighted_run_score,
)

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None  # type: ignore[assignment]


BUCKETS = ["0-49", "50-99", "100-149"]
BINS_COL = "0-49|50-99|100-149"


def _dcv(*runs: tuple[str, float, list[float]]) -> dict[tuple[str, float, str], list[float]]:
    out: dict[tuple[str, float, str], list[float]] = {}
    for model, lr, accs in runs:
        for bucket, acc in zip(BUCKETS, accs):
            out[(model, lr, bucket)] = [acc]
    return out


class BinWeightTests(unittest.TestCase):
    def test_default_factor_is_arithmetic_sequence(self) -> None:
        self.assertAlmostEqual(bin_weight(0), 1.0)
        self.assertAlmostEqual(bin_weight(1), 1.1)
        self.assertAlmostEqual(bin_weight(2), 1.2)
        self.assertAlmostEqual(bin_weight(3), 1.3)

    def test_factor_one_is_uniform(self) -> None:
        self.assertEqual([bin_weight(i, 1.0) for i in range(4)], [1.0, 1.0, 1.0, 1.0])


class WeightedScoreTests(unittest.TestCase):
    def test_score_is_weighted_sum(self) -> None:
        dp = ("m", 1e-4)
        dcv = _dcv((dp[0], dp[1], [1.0, 0.5, 0.0]))
        # 1.0*1.0 + 1.1*0.5 + 1.2*0.0 = 1.55
        self.assertAlmostEqual(weighted_run_score(dp, BUCKETS, dcv), 1.55)

    def test_missing_bin_counts_as_zero(self) -> None:
        dp = ("m", 1e-4)
        dcv = {(dp[0], dp[1], "0-49"): [1.0]}
        self.assertAlmostEqual(weighted_run_score(dp, BUCKETS, dcv), 1.0)


class SelectWeightedMaxRunTests(unittest.TestCase):
    def test_later_bins_outweigh_early_bins(self) -> None:
        strong_early = ("early", 1e-3)
        strong_late = ("late", 1e-4)
        dcv = _dcv(
            (strong_early[0], strong_early[1], [1.0, 0.5, 0.5]),
            (strong_late[0], strong_late[1], [0.5, 0.5, 1.0]),
        )
        # early: 1.0 + 0.55 + 0.60 = 2.15
        # late:  0.5 + 0.55 + 1.20 = 2.25
        winner = select_weighted_max_run(
            {strong_early, strong_late}, dcv, BUCKETS
        )
        self.assertEqual(winner, strong_late)

    def test_returns_exactly_one_run(self) -> None:
        a = ("a", 1e-3)
        b = ("b", 1e-4)
        dcv = _dcv(
            (a[0], a[1], [0.9, 0.8, 0.7]),
            (b[0], b[1], [0.1, 0.1, 0.1]),
        )
        winner = select_weighted_max_run({a, b}, dcv, BUCKETS)
        self.assertEqual(winner, a)

    def test_tie_breaks_toward_later_bins(self) -> None:
        # Same weighted sum is unlikely with 1.1, so use factor=1 (equal weights).
        a = ("a", 1e-3)
        b = ("b", 1e-4)
        dcv = _dcv(
            (a[0], a[1], [1.0, 0.0, 0.5]),
            (b[0], b[1], [0.0, 0.5, 1.0]),
        )
        winner = select_weighted_max_run({a, b}, dcv, BUCKETS, weight_factor=1.0)
        self.assertEqual(winner, b)

    def test_empty_inputs(self) -> None:
        self.assertIsNone(select_weighted_max_run(set(), {}, BUCKETS))
        self.assertIsNone(select_weighted_max_run({("m", 1e-4)}, {}, []))


class SingleRunLineTests(unittest.TestCase):
    def test_plots_that_run_only(self) -> None:
        dp = ("m", 1e-4)
        dcv = _dcv((dp[0], dp[1], [0.2, 0.4, 0.6]))
        xs, ys, stds = single_run_line_xy(
            dp, BUCKETS, dcv, x_of_bucket=lambda b: float(BUCKETS.index(b))
        )
        self.assertEqual(xs, [0.0, 1.0, 2.0])
        self.assertEqual(ys, [20.0, 40.0, 60.0])
        self.assertEqual(stds, [0.0, 0.0, 0.0])


class ModelSizeKeyTests(unittest.TestCase):
    def test_layers_then_d_model_then_heads(self) -> None:
        self.assertEqual(model_size_key("ssm2l64d"), (2, 64, 0))
        self.assertEqual(model_size_key("ssm4l64d"), (4, 64, 0))
        self.assertEqual(model_size_key("ssm2l128d"), (2, 128, 0))
        self.assertEqual(model_size_key("lm2l4h64d"), (2, 64, 4))
        self.assertLess(model_size_key("ssm2l64d"), model_size_key("ssm4l64d"))
        self.assertLess(model_size_key("ssm2l64d"), model_size_key("ssm2l128d"))
        self.assertLess(model_size_key("lm2l4h64d"), model_size_key("lm2l8h64d"))
        # layers dominate d_model: 2l/128d is smaller than 4l/64d
        self.assertLess(model_size_key("ssm2l128d"), model_size_key("ssm4l64d"))


class SelectMaxRunTests(unittest.TestCase):
    def test_prefers_fewer_layers_when_both_pass_threshold(self) -> None:
        small = ("ssm2l64d", 1e-4)
        large = ("ssm4l64d", 1e-4)
        dcv = _dcv(
            (small[0], small[1], [0.99, 0.98, 0.98]),
            (large[0], large[1], [1.0, 1.0, 1.0]),
        )
        winner = select_max_run({small, large}, dcv, BUCKETS)
        self.assertEqual(winner, small)

    def test_layers_outrank_d_model(self) -> None:
        few_layers = ("ssm2l128d", 1e-4)
        many_layers = ("ssm4l64d", 1e-4)
        dcv = _dcv(
            (few_layers[0], few_layers[1], [0.99, 0.99, 0.99]),
            (many_layers[0], many_layers[1], [1.0, 1.0, 1.0]),
        )
        winner = select_max_run({few_layers, many_layers}, dcv, BUCKETS)
        self.assertEqual(winner, few_layers)

    def test_same_layers_prefers_smaller_d_model(self) -> None:
        narrow = ("ssm2l64d", 1e-4)
        wide = ("ssm2l128d", 1e-4)
        dcv = _dcv(
            (narrow[0], narrow[1], [0.98, 0.98, 0.98]),
            (wide[0], wide[1], [1.0, 1.0, 1.0]),
        )
        winner = select_max_run({narrow, wide}, dcv, BUCKETS)
        self.assertEqual(winner, narrow)

    def test_same_layers_and_d_model_prefers_fewer_heads(self) -> None:
        few_h = ("lm2l4h64d", 1e-4)
        many_h = ("lm2l8h64d", 1e-4)
        dcv = _dcv(
            (few_h[0], few_h[1], [0.98, 0.98, 0.98]),
            (many_h[0], many_h[1], [1.0, 1.0, 1.0]),
        )
        winner = select_max_run({few_h, many_h}, dcv, BUCKETS)
        self.assertEqual(winner, few_h)

    def test_falls_back_to_weighted_max_if_none_pass(self) -> None:
        strong_early = ("ssm2l64d", 1e-3)
        strong_late = ("ssm8l128d", 1e-4)
        dcv = _dcv(
            (strong_early[0], strong_early[1], [1.0, 0.5, 0.5]),
            (strong_late[0], strong_late[1], [0.5, 0.5, 1.0]),
        )
        winner = select_max_run({strong_early, strong_late}, dcv, BUCKETS)
        self.assertEqual(winner, strong_late)

    def test_run_below_threshold_on_one_bin_is_ignored(self) -> None:
        small_fail = ("ssm2l64d", 1e-4)
        large_pass = ("ssm4l64d", 1e-4)
        dcv = _dcv(
            (small_fail[0], small_fail[1], [1.0, 1.0, 0.97]),
            (large_pass[0], large_pass[1], [0.98, 0.98, 0.98]),
        )
        winner = select_max_run({small_fail, large_pass}, dcv, BUCKETS)
        self.assertEqual(winner, large_pass)


def _task_rows(model: str, lr: float, accs: list[float]) -> list[dict]:
    return [
        {
            "task": "toy",
            "model": model,
            "learning_rate": lr,
            "bucket": bucket,
            "accuracy": acc,
            "arch": "ssm",
            "bins": BINS_COL,
        }
        for bucket, acc in zip(BUCKETS, accs)
    ]


@unittest.skipUnless(pd is not None, "pandas is required")
class PrepareMaxAggregationTests(unittest.TestCase):
    def _prepare(self, max_aggregation: str):
        from generate_plot_df import _prepare_task_plot

        df = pd.DataFrame(
            [
                *_task_rows("early", 1e-3, [1.0, 0.5, 0.5]),
                *_task_rows("late", 1e-4, [0.5, 0.5, 1.0]),
            ]
        )
        return _prepare_task_plot(
            df,
            task="toy",
            group_by=["arch"],
            group_label_mode="group",
            group_custom_labels=[],
            max_aggregation=max_aggregation,
            max_bin_weight=1.1,
            max_acc_threshold=0.98,
            x_ticks_mode="bins",
            x_tick_step=10,
            x_axis_break=None,
            num_bins=None,
            merge_bins=False,
        )

    def test_max_plots_single_weighted_winner(self) -> None:
        prepared = self._prepare("max")
        self.assertEqual(len(prepared["max_series"]), 1)
        _label, _mx, means, stds = next(iter(prepared["max_series"].values()))
        self.assertEqual(means, [50.0, 50.0, 100.0])
        self.assertEqual(stds, [0.0, 0.0, 0.0])

    def test_bin_max_may_mix_runs(self) -> None:
        prepared = self._prepare("bin_max")
        _label, _mx, means, _stds = next(iter(prepared["max_series"].values()))
        self.assertEqual(means, [100.0, 50.0, 100.0])


if __name__ == "__main__":
    unittest.main()
