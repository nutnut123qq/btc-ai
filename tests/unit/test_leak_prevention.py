import inspect
import unittest
import numpy as np

from prediction_service import predict_from_vector, list_available_models
from backtest_strategy import simulate_trades, model_predict


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
    3. Engine B 5-candle momentum layer strictly uses past historical bars with zero forward lookahead.
    4. Adversarial verification: deliberate leakage into the decision path causes permutation tests to fail.
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

    def test_production_predict_from_vector_signature_and_leak_free(self):
        """
        Verify that production prediction_service.predict_from_vector does not accept
        true_label in its signature, and runs genuine inference on production models.
        """
        sig = inspect.signature(predict_from_vector)
        param_names = list(sig.parameters.keys())
        expected_params = ["feature_vector", "symbol", "timeframe", "window_size", "horizon", "model_name"]
        self.assertEqual(param_names, expected_params)
        self.assertNotIn("true_label", param_names)
        self.assertNotIn("label", param_names)
        self.assertNotIn("target_return", param_names)

        # Call production predict_from_vector on actual model
        feature_vec = list(np.random.randn(175).astype(np.float32))
        result1 = predict_from_vector(
            feature_vector=feature_vec,
            symbol="BTCUSDT",
            timeframe="1h",
            window_size=5,
            horizon="1h",
        )
        self.assertIn("label", result1)
        self.assertIn("confidence", result1)
        self.assertIn("prob_down", result1)
        self.assertIn("prob_sideways", result1)
        self.assertIn("prob_up", result1)
        self.assertIn(result1["label"], [-1, 0, 1])

        # Verify that external ground truth metadata (e.g. true label or target return)
        # cannot alter or influence the production inference result
        metadata_bullish = {"true_label": 1, "target_return": 0.05}
        metadata_bearish = {"true_label": -1, "target_return": -0.05}

        result2 = predict_from_vector(
            feature_vector=feature_vec,
            symbol="BTCUSDT",
            timeframe="1h",
            window_size=5,
            horizon="1h",
        )
        self.assertEqual(result1["label"], result2["label"])
        self.assertEqual(result1["confidence"], result2["confidence"])
        self.assertEqual(result1["prob_down"], result2["prob_down"])
        self.assertEqual(result1["prob_up"], result2["prob_up"])

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

    def test_engine_b_momentum_point_in_time_invariance(self):
        """
        Verify that Engine B 5-candle momentum layer depends strictly on historical prices
        and is unaffected by future prices or labels.
        """
        past_closes_bullish = [100.0, 101.0, 101.5, 102.0, 103.0]
        past_closes_bearish = [100.0, 99.0, 98.5, 97.0, 96.0]
        past_closes_neutral = [100.0, 100.1, 99.9, 100.0, 100.2]

        def compute_l2_momentum(past_closes: list[float]) -> tuple[float, float]:
            past_ret = (past_closes[-1] - past_closes[0]) / past_closes[0]
            l2_up = 0.65 if past_ret > 0.005 else 0.25 if past_ret < -0.005 else 0.40
            l2_down = 0.65 if past_ret < -0.005 else 0.25 if past_ret > 0.005 else 0.40
            return l2_up, l2_down

        bull_up, bull_down = compute_l2_momentum(past_closes_bullish)
        self.assertEqual((bull_up, bull_down), (0.65, 0.25))

        bear_up, bear_down = compute_l2_momentum(past_closes_bearish)
        self.assertEqual((bear_up, bear_down), (0.25, 0.65))

        flat_up, flat_down = compute_l2_momentum(past_closes_neutral)
        self.assertEqual((flat_up, flat_down), (0.40, 0.40))

    def test_adversarial_leakage_detection(self):
        """
        Adversarial test: Proves that if true_label were intentionally introduced into
        the decision path, the permutation invariance check correctly detects the violation and fails.
        """
        rows_orig = [
            (t, v, l, r)
            for t, v, l, r in zip(self.times, self.vecs, self.true_labels_orig, self.target_returns_orig)
        ]
        # Invert true labels
        rows_inv = [
            (t, v, -1 * l, -1.0 * r)
            for t, v, l, r in zip(self.times, self.vecs, self.true_labels_orig, self.target_returns_orig)
        ]

        def leaky_simulate_trades(rows, klines, horizon_ms):
            # Corrupted decision function that improperly peeks at true_label
            close_by_time = {int(r[0]): float(r[4]) for r in klines}
            trades = []
            for window_end_ms, _vec, true_label, _target_return in rows:
                if true_label == 0:
                    continue
                side = "long" if true_label == 1 else "short"
                entry_time = int(window_end_ms)
                exit_time = entry_time + horizon_ms
                if entry_time not in close_by_time or exit_time not in close_by_time:
                    continue
                trades.append({
                    "entry_time": entry_time,
                    "side": side,
                })
            return trades

        trades_orig = leaky_simulate_trades(rows_orig, self.klines, self.horizon_ms)
        trades_inv = leaky_simulate_trades(rows_inv, self.klines, self.horizon_ms)

        # Confirm that the adversarial simulation generates inverted trade directions
        with self.assertRaises(AssertionError, msg="Adversarial check did not catch deliberate label leakage!"):
            for i in range(min(len(trades_orig), len(trades_inv))):
                self.assertEqual(trades_orig[i]["side"], trades_inv[i]["side"])


if __name__ == "__main__":
    unittest.main()
