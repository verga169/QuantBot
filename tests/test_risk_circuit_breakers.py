import unittest
import pandas as pd
from backtester import Backtester

class TestRiskCircuitBreakers(unittest.TestCase):
    def setUp(self):
        self.rules = {
            "tick_size": 0.1,
            "qty_step": 0.001,
            "min_qty": 0.001,
            "min_order_amount": 10.0,
            "max_order_amount": 50000.0,
        }

    def test_consecutive_losses_circuit_breaker(self):
        """Verifica che il circuit breaker blocchi nuovi trade dopo N perdite consecutive."""
        # 3 signals in a row, all will hit stop loss
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0, 100.0, 95.0, 95.0, 90.0, 90.0, 85.0, 85.0, 85.0, 85.0],
            "high": [100.0, 100.0, 95.0, 95.0, 90.0, 90.0, 85.0, 85.0, 85.0, 85.0],
            "low": [100.0, 95.0, 90.0, 90.0, 85.0, 85.0, 80.0, 80.0, 80.0, 80.0],
            "close": [100.0, 95.0, 90.0, 90.0, 85.0, 85.0, 80.0, 80.0, 80.0, 80.0],
            "signal": [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],  # 3 candidate entries
            "setup_type": ["", "SETUP1", "", "SETUP2", "", "SETUP3", "", "", "", ""],
            "atr": [1.0] * 10,
            "atr_pct": [1.0] * 10,
        })

        bt = Backtester(self.rules, fee_rate=0.0, slippage_pct=0.0)

        # Senza circuit breaker: tutti e 3 i trade vengono eseguiti
        trades_all, _ = bt.run(df, initial_capital=1000.0, max_consecutive_losses=0, verbose=False, use_rl=False)
        self.assertEqual(len(trades_all), 3)

        # Con max_consecutive_losses = 2: il terzo trade deve essere categoricamente bloccato
        trades_blocked, _ = bt.run(df, initial_capital=1000.0, max_consecutive_losses=2, verbose=False, use_rl=False)
        self.assertEqual(len(trades_blocked), 2, "Il terzo trade doveva essere bloccato dal circuit breaker consecutivi!")

    def test_cooldown_bars_after_loss(self):
        """Verifica che dopo una perdita scatti il cooldown di N barre."""
        dates = pd.date_range("2024-01-01", periods=6, freq="h")
        # Bar 0 signal -> Entry Bar 1 (100.0), stops out at 98.2 (low 97.0).
        # Bar 2 signal -> Entry Bar 3 if no cooldown.
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0, 100.0, 98.0, 98.0, 98.0, 98.0],
            "high": [100.0, 100.0, 98.0, 98.0, 98.0, 98.0],
            "low": [100.0, 97.0, 98.0, 98.0, 98.0, 98.0],
            "close": [100.0, 98.0, 98.0, 98.0, 98.0, 98.0],
            "signal": [1, 0, 1, 0, 0, 0],
            "setup_type": ["SETUP1", "", "SETUP2", "", "", ""],
            "atr": [1.0] * 6,
            "atr_pct": [1.0] * 6,
        })
        bt = Backtester(self.rules, fee_rate=0.0, slippage_pct=0.0)

        # Senza cooldown: 2 trade eseguiti
        trades_no_cd, _ = bt.run(df, initial_capital=1000.0, cooldown_bars_after_loss=0, verbose=False, use_rl=False)
        self.assertEqual(len(trades_no_cd), 2)

        # Con cooldown di 3 barre: il secondo trade viene saltato
        trades_cd, _ = bt.run(df, initial_capital=1000.0, cooldown_bars_after_loss=3, verbose=False, use_rl=False)
        self.assertEqual(len(trades_cd), 1)

if __name__ == "__main__":
    unittest.main()
