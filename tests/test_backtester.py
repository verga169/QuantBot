import unittest
import numpy as np
import pandas as pd
from config import Config
from backtester import Backtester

class TestBacktester(unittest.TestCase):
    def setUp(self):
        self.rules = {
            "tick_size": 0.1,
            "qty_step": 0.001,
            "min_qty": 0.001,
            "min_order_amount": 10.0,
            "max_order_amount": 50000.0,
        }
        self.bt = Backtester(self.rules, fee_rate=0.001, slippage_pct=0.0005, eur_to_quote_rate=1.0)

    def test_execution_lag_realistic_open_fill(self):
        """
        Verifica che il segnale generato alla chiusura della barra t
        venga eseguito all'OPEN della barra t+1 e mai al CLOSE trascorso di t.
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [50000.0, 50100.0, 50200.0, 50300.0, 50400.0],
            "high": [50050.0, 50150.0, 50250.0, 50350.0, 50450.0],
            "low": [49950.0, 50050.0, 50150.0, 50250.0, 50350.0],
            "close": [50000.0, 50120.0, 50220.0, 50320.0, 50420.0],
            "signal": [0, 1, 0, 0, 0],  # Signal generated on bar 1 (close 50120)
            "setup_type": ["", "TEST_SETUP", "", "", ""],
            "atr": [500.0] * 5,
            "atr_pct": [1.0] * 5,
        })

        trades, m = self.bt.run(df, initial_capital=1000.0, verbose=False, use_rl=False)
        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        # Entry timestamp must be bar 2 (2024-01-01 02:00:00), NOT bar 1
        self.assertEqual(trade["entry_time"], pd.Timestamp("2024-01-01 02:00:00"))
        # Entry price must be based on open of bar 2 (50200.0) + slippage, NOT close of bar 1 (50120.0)
        expected_entry = round(50200.0 * (1 + 0.0005), 1)
        self.assertAlmostEqual(trade["entry_price"], expected_entry, places=1)

    def test_fee_deduction(self):
        """Verifica che le fee Maker/Taker (0.1% entry + 0.1% exit) siano integralmente dedotte dal PnL netto."""
        dates = pd.date_range("2024-01-01", periods=4, freq="h")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "signal": [0, 1, 0, 0],
            "setup_type": ["", "SETUP", "", ""],
            "atr": [1.0] * 4,
            "atr_pct": [1.0] * 4,
        })
        bt_no_slip = Backtester(self.rules, fee_rate=0.001, slippage_pct=0.0)
        trades, m = bt_no_slip.run(df, initial_capital=1000.0, verbose=False, use_rl=False)
        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        # Since price didn't move, gross PnL is 0, so net PnL must be strictly negative due to roundtrip fees (0.2%)
        self.assertLess(trade["net_pnl"], 0.0)
        expected_fee_pct = 0.002
        approx_fee = trade["entry_price"] * trade["qty"] * expected_fee_pct
        self.assertAlmostEqual(abs(trade["net_pnl"]), approx_fee, places=2)

    def test_trailing_stop_tiers(self):
        """Verifica l'attivazione del multi-tier trailing stop (Tier 1 e Tier 2)."""
        dates = pd.date_range("2024-01-01", periods=6, freq="h")
        # Bar 1 generates signal -> Entry at Bar 2 Open (100.0)
        # Bar 3 spikes to 101.6 (+1.6% -> triggers Tier 1 lock at +0.4% = 100.4)
        # Bar 4 drops to 100.2 (hits trailing stop 100.4)
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0, 100.0, 100.0, 101.0, 100.5, 100.0],
            "high": [100.0, 100.0, 100.5, 101.6, 100.5, 100.0],
            "low": [100.0, 100.0, 99.5, 100.8, 100.1, 99.5],
            "close": [100.0, 100.0, 100.2, 101.5, 100.2, 100.0],
            "signal": [0, 1, 0, 0, 0, 0],
            "setup_type": ["", "SETUP", "", "", "", ""],
            "atr": [1.0] * 6,
            "atr_pct": [1.0] * 6,
        })
        bt_zero = Backtester(self.rules, fee_rate=0.0, slippage_pct=0.0)
        trades, m = bt_zero.run(
            df, initial_capital=1000.0, sl_pct=0.018, tp_pct=0.038,
            trail_tier1_trigger=0.012, trail_tier1_lock=0.004, verbose=False, use_rl=False
        )
        self.assertEqual(len(trades), 1)
        self.assertIn(trades.iloc[0]["exit_reason"], ["TRAIL_TIER1", "TRAIL_TIER2"])
        self.assertGreater(trades.iloc[0]["net_pnl"], 0.0)

if __name__ == "__main__":
    unittest.main()
