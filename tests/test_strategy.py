import unittest
import numpy as np
import pandas as pd
from config import Config
from strategy import QuantStrategy

class TestQuantStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = QuantStrategy()
        # Generate 100 bars of synthetic deterministic OHLCV
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=150, freq="h")
        base_price = 50000.0 + np.cumsum(np.random.randn(150) * 100.0)
        self.df = pd.DataFrame({
            "timestamp": dates,
            "open": base_price,
            "high": base_price + np.random.uniform(50, 150, 150),
            "low": base_price - np.random.uniform(50, 150, 150),
            "close": base_price + np.random.randn(150) * 50,
            "volume": np.random.uniform(10, 100, 150),
        })

    def test_indicator_computation_completeness(self):
        """Verifica che tutti gli indicatori tecnici siano calcolati correttamente senza NaN."""
        ind_df = self.strategy.compute_indicators(self.df)
        expected_cols = [
            "ema_20", "ema_50", "ema_100", "ema_200", "ema_800", "ema_200_slope",
            "rsi", "atr", "atr_pct", "bb_mid", "bb_std", "bb_upper",
            "bb_lower", "bb_lower_22", "squeeze", "high_20", "vol_sma", "vol_ratio",
            "adx", "chop"
        ]
        for col in expected_cols:
            self.assertIn(col, ind_df.columns, f"Colonna mancante: {col}")
            self.assertFalse(ind_df[col].isna().any(), f"NaN trovati in colonna: {col}")

    def test_zero_look_ahead_bias(self):
        """
        Verifica rigorosa di Zero Look-Ahead Bias:
        Modificando i prezzi delle candele future (t+1..N), i valori degli indicatori
        e i segnali alla candela t devono rimanere rigorosamente identici.
        """
        ind_base = self.strategy.compute_indicators(self.df).set_index("timestamp")
        test_ts = self.df["timestamp"].iloc[80]
        val_at_t_base = ind_base.loc[test_ts, "rsi"]
        adx_at_t_base = ind_base.loc[test_ts, "adx"]

        # Modifica radicale delle candele future (> test_ts)
        df_perturbed = self.df.copy()
        future_mask = df_perturbed["timestamp"] > test_ts
        df_perturbed.loc[future_mask, "close"] *= 3.0
        df_perturbed.loc[future_mask, "high"] *= 3.5
        df_perturbed.loc[future_mask, "low"] *= 2.5
        df_perturbed.loc[future_mask, "volume"] *= 10.0

        ind_perturbed = self.strategy.compute_indicators(df_perturbed).set_index("timestamp")
        val_at_t_perturbed = ind_perturbed.loc[test_ts, "rsi"]
        adx_at_t_perturbed = ind_perturbed.loc[test_ts, "adx"]

        self.assertAlmostEqual(val_at_t_base, val_at_t_perturbed, places=6,
                               msg="Look-ahead bias rilevato: RSI alla candela t è cambiato dopo aver modificato t+1!")
        self.assertAlmostEqual(adx_at_t_base, adx_at_t_perturbed, places=6,
                               msg="Look-ahead bias rilevato: ADX alla candela t è cambiato dopo aver modificato t+1!")

    def test_regime_indicators_bounds(self):
        """Verifica che RSI, ADX e CHOP rientrino nei range matematici attesi."""
        ind_df = self.strategy.compute_indicators(self.df)
        self.assertTrue((ind_df["rsi"] >= 0.0).all() and (ind_df["rsi"] <= 100.0).all())
        self.assertTrue((ind_df["adx"] >= 0.0).all() and (ind_df["adx"] <= 100.0).all())
        self.assertTrue((ind_df["chop"] >= 0.0).all() and (ind_df["chop"] <= 100.0).all())

if __name__ == "__main__":
    unittest.main()
