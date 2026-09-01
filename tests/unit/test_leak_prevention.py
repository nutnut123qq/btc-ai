import inspect
import unittest
import numpy as np
from unittest.mock import patch

from prediction_service import predict_from_vector
from backtest_strategy import simulate_trades
from run_blind_oos_audit import simulate_engine_b_windows


class MockPredictor:
    """Mock predictor with predict_proba to test simulate_trades deterministically without disk dependencies."""
    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.weights
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class TestLeakPreventionAndPermutationInvariance(unittest.TestCase):
    """
    Genuine production-linked regression tests verifying:
    1. Production prediction_service.predict_from_vector signature and inference do not accept or depend on true_label.
    2. Production backtest_strategy.simulate_trades trade decisions are 100% invariant to true_label / forward return permutations.
    3. Production Engine B decisions are invariant to labels and target returns.
    4. The Engine B comparison oracle rejects a deliberate label-derived mutation.
    """

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 30
        self.n_features = 175
        self.horizon_ms = 3600_000

        # Construct deterministic synthetic feature vectors & timestamps
        self.times = [1700000000000 + i * self.horizon_ms for i in range(self.n_samples)]
        self.vecs = [list(np.random.randn(self.n_features).astype(np.float32)) for _ in range(self.n_samples)]
        self.true_labels_orig = list(np.random.choice([-1, 0, 1], size=self.n_samples))
        self.target_returns_orig = [float(np.random.uniform(-0.05, 0.05)) for _ in range(self.n_samples)]

        # Synthetic klines spanning all required entry and exit timestamps
        all_times = sorted(set(self.times + [t + self.horizon_ms for t in self.times]))
        self.klines = [
            (t, 100.0 + (i * 0.5), 105.0 + (i * 0.5), 95.0 + (i * 0.5), 100.0 + (i * 0.5), 1000.0)
            for i, t in enumerate(all_times)
        ]

        # Model with deterministic weights for 3 classes
        weights = np.random.randn(self.n_features, 3).astype(np.float32)
        self.mock_model = MockPredictor(weights)
        self.meta = {"model_name": "BTCUSDT_1h_ws5_h1h_XGB_calibrated"}

    def test_production_predict_signature_is_leak_free_and_deterministic(self):
        """
        Verify that inference does not accept outcomes and the promoted artifact is
        deterministic. Binary compatibility is covered separately.
        """
        sig = inspect.signature(predict_from_vector)
        param_names = list(sig.parameters.keys())
        expected_params = ["feature_vector", "symbol", "timeframe", "window_size", "horizon", "model_name"]
        self.assertEqual(param_names, expected_params)
        self.assertNotIn("true_label", param_names)
        self.assertNotIn("label", param_names)
        self.assertNotIn("target_return", param_names)

        feature_vec = list(np.random.randn(175).astype(np.float32))
        manifest = {
            "feature_dim": 175,
            "class_mapping": {"0": -1, "1": 0, "2": 1},
            "version": "test",
        }
        with patch("prediction_service.load_model", return_value=(self.mock_model, manifest)):
            first = predict_from_vector(feature_vec, "BTCUSDT", "4h", 5, "4h")
            second = predict_from_vector(feature_vec, "BTCUSDT", "4h", 5, "4h")
        self.assertEqual(first["label"], second["label"])
        self.assertEqual(first["confidence"], second["confidence"])
        self.assertAlmostEqual(
            first["prob_down"] + first["prob_sideways"] + first["prob_up"], 1.0
        )

    def test_production_simulate_trades_permutation_invariance(self):
        """
        Verify that holding observable feature vectors constant while permuting or inverting
        true labels and forward target returns yields 100% identical trades in simulate_trades.
        """
        rows_orig = [
            (t, v, l, r)
            for t, v, l, r in zip(self.times, self.vecs, self.true_labels_orig, self.target_returns_orig)
        ]

        # Inverted labels and returns
        true_labels_inv = [-1 * l for l in self.true_labels_orig]
        target_returns_inv = [-1.0 * r for r in self.target_returns_orig]
        rows_inv = [
            (t, v, l, r)
            for t, v, l, r in zip(self.times, self.vecs, true_labels_inv, target_returns_inv)
        ]

        # Permuted labels and returns
        perm = np.random.permutation(self.n_samples)
        true_labels_perm = [self.true_labels_orig[i] for i in perm]
        target_returns_perm = [self.target_returns_orig[i] for i in perm]
        rows_perm = [
            (t, v, l, r)
            for t, v, l, r in zip(self.times, self.vecs, true_labels_perm, target_returns_perm)
        ]

        # Execute production simulate_trades across all 3 variants
        trades_orig = simulate_trades(rows_orig, self.klines, self.horizon_ms, model=self.mock_model, meta=self.meta)
        trades_inv = simulate_trades(rows_inv, self.klines, self.horizon_ms, model=self.mock_model, meta=self.meta)
        trades_perm = simulate_trades(rows_perm, self.klines, self.horizon_ms, model=self.mock_model, meta=self.meta)

        self.assertGreater(len(trades_orig), 0, "Expected non-zero trades generated")
        self.assertEqual(len(trades_orig), len(trades_inv), "Trade count changed under inverted labels!")
        self.assertEqual(len(trades_orig), len(trades_perm), "Trade count changed under permuted labels!")

        # Verify trade-by-trade decision invariance
        for i in range(len(trades_orig)):
            t_orig = trades_orig[i]
            t_inv = trades_inv[i]
            t_perm = trades_perm[i]

            self.assertEqual(t_orig["entry_time"], t_inv["entry_time"])
            self.assertEqual(t_orig["entry_time"], t_perm["entry_time"])

            self.assertEqual(t_orig["exit_time"], t_inv["exit_time"])
            self.assertEqual(t_orig["exit_time"], t_perm["exit_time"])

            self.assertEqual(t_orig["side"], t_inv["side"], "Trade direction leaked label under inversion!")
            self.assertEqual(t_orig["side"], t_perm["side"], "Trade direction leaked label under permutation!")

            self.assertAlmostEqual(t_orig["confidence"], t_inv["confidence"], places=6)
            self.assertAlmostEqual(t_orig["confidence"], t_perm["confidence"], places=6)

            self.assertAlmostEqual(t_orig["entry_price"], t_inv["entry_price"], places=6)
            self.assertAlmostEqual(t_orig["exit_price"], t_inv["exit_price"], places=6)

            self.assertAlmostEqual(t_orig["gross_return"], t_inv["gross_return"], places=6)
            self.assertAlmostEqual(t_orig["net_return"], t_inv["net_return"], places=6)

    def _engine_b_rows(self, labels, returns):
        return [
            (t, v, label, target_return)
            for t, v, label, target_return in zip(self.times, self.vecs, labels, returns)
        ]

    def _run_engine_b(self, rows):
        kline_map = {
            int(t): {
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            }
            for t, open_price, high, low, close, volume in self.klines
        }
        return simulate_engine_b_windows(
            rows,
            kline_map,
            self.horizon_ms,
            confidence_threshold=0.0,
            ml_model=self.mock_model,
        )

    def _assert_same_engine_b_decisions(self, left, right):
        decision_fields = ("entry_time", "exit_time", "side", "confidence")
        self.assertEqual(
            [tuple(trade[field] for field in decision_fields) for trade in left],
            [tuple(trade[field] for field in decision_fields) for trade in right],
        )

    def test_production_engine_b_permutation_invariance(self):
        """Permuting hidden outcomes cannot alter Engine B's production decisions."""
        rows_orig = self._engine_b_rows(self.true_labels_orig, self.target_returns_orig)
        perm = np.random.permutation(self.n_samples)
        rows_permuted = self._engine_b_rows(
            [self.true_labels_orig[i] for i in perm],
            [self.target_returns_orig[i] for i in perm],
        )
        rows_inverted = self._engine_b_rows(
            [-label for label in self.true_labels_orig],
            [-target_return for target_return in self.target_returns_orig],
        )

        trades_orig, probabilities_orig, times_orig = self._run_engine_b(rows_orig)
        trades_permuted, probabilities_permuted, times_permuted = self._run_engine_b(rows_permuted)
        trades_inverted, probabilities_inverted, times_inverted = self._run_engine_b(rows_inverted)

        self.assertGreater(len(trades_orig), 0)
        self.assertEqual(probabilities_orig, probabilities_permuted)
        self.assertEqual(probabilities_orig, probabilities_inverted)
        self.assertEqual(times_orig, times_permuted)
        self.assertEqual(times_orig, times_inverted)
        self.assertEqual(trades_orig, trades_permuted)
        self.assertEqual(trades_orig, trades_inverted)

    def test_engine_b_uses_first_complete_five_bar_window(self):
        """Index 4 is the first valid five-close momentum window (0 through 4)."""
        times = [1700000000000 + i * self.horizon_ms for i in range(6)]
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        kline_map = {
            timestamp: {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1.0,
            }
            for timestamp, close in zip(times, closes)
        }
        rows = [(times[4], [0.0] * self.n_features, 1, 0.01)]

        trades, probabilities, _times = simulate_engine_b_windows(
            rows,
            kline_map,
            self.horizon_ms,
            confidence_threshold=0.0,
        )

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "long")
        self.assertAlmostEqual(probabilities[0], 0.405)

    def test_engine_b_oracle_kills_label_leak_mutation(self):
        """The decision comparison fails if production output is mutated from labels."""
        rows = self._engine_b_rows(self.true_labels_orig, self.target_returns_orig)
        trades, _probabilities, _times = self._run_engine_b(rows)
        label_by_time = dict(zip(self.times, self.true_labels_orig))
        mutated = [dict(trade) for trade in trades]
        for trade in mutated:
            label = label_by_time[trade["entry_time"]]
            trade["side"] = "long" if label == 1 else "short" if label == -1 else "flat"

        with self.assertRaises(AssertionError):
            self._assert_same_engine_b_decisions(trades, mutated)


if __name__ == "__main__":
    unittest.main()
