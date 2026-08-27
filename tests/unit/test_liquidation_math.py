import unittest
import sys
import os
from pathlib import Path

# Add btc-ai to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from liquidation_engine import LiquidationEngine


class TestLiquidationMath(unittest.TestCase):
    """
    Hermetic unit tests for liquidation math and swept tranche filtering.
    Requires zero external database or network dependencies.
    """

    def setUp(self):
        self.engine = LiquidationEngine(bin_step_pct=0.003)

    def test_long_liquidation_price_formula(self):
        # Entry = 100,000, Lev = 100, MMR = 0.004 => 100,000 * (1 - 0.01 + 0.004) = 99,400.0
        liq_100x = self.engine.calculate_long_liq_price(100000.0, 100, 0.004)
        self.assertAlmostEqual(liq_100x, 99400.0, places=4)

        # Entry = 100,000, Lev = 50, MMR = 0.005 => 100,000 * (1 - 0.02 + 0.005) = 98,500.0
        liq_50x = self.engine.calculate_long_liq_price(100000.0, 50, 0.005)
        self.assertAlmostEqual(liq_50x, 98500.0, places=4)

    def test_short_liquidation_price_formula(self):
        # Entry = 100,000, Lev = 100, MMR = 0.004 => 100,000 * (1 + 0.01 - 0.004) = 100,600.0
        liq_100x = self.engine.calculate_short_liq_price(100000.0, 100, 0.004)
        self.assertAlmostEqual(liq_100x, 100600.0, places=4)

        # Entry = 100,000, Lev = 50, MMR = 0.005 => 100,000 * (1 + 0.02 - 0.005) = 101,500.0
        liq_50x = self.engine.calculate_short_liq_price(100000.0, 50, 0.005)
        self.assertAlmostEqual(liq_50x, 101500.0, places=4)

    def test_swept_tranche_filtering(self):
        bars = [
            {"open_time_ms": 1000, "open": 100, "high": 102, "low": 98, "close": 100, "volume": 10, "volume_usdt": 1000, "delta_oi_usdt": 1000, "ls_ratio": 1.0},
            {"open_time_ms": 2000, "open": 100, "high": 101, "low": 90, "close": 94, "volume": 20, "volume_usdt": 2000, "delta_oi_usdt": 0, "ls_ratio": 1.0},
            {"open_time_ms": 3000, "open": 94, "high": 96, "low": 93, "close": 95, "volume": 10, "volume_usdt": 1000, "delta_oi_usdt": 500, "ls_ratio": 1.0},
        ]
        long_tranches, short_tranches = self.engine.compute_liquidation_tranches(bars, current_price=95.0)

        # Long tranches at bar 0 that had liq_price >= 90 must have been swept by bar 1's low of 90
        for t in long_tranches:
            if t["bar_idx"] == 0:
                self.assertLess(t["liq_price"], 90.0)


if __name__ == "__main__":
    unittest.main()
