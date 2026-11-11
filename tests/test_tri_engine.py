import unittest
import numpy as np
import pandas as pd
from config import Config
from strategy import QuantStrategy
from rl_experience_engine import RLExperienceEngine
from backtester import Backtester

class TestTriEngineMultiRegime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Config.TRI_ENGINE_MODE = True
        cls.df = pd.read_parquet("data/BTCUSDC_10Y_60m_eu.parquet")
        cls.fng = pd.read_parquet("data/fear_greed_history.parquet")
        cls.strategy = QuantStrategy()
        cls.signals = cls.strategy.generate_signals(cls.df, fng_df=cls.fng)

    def test_three_setups_generated(self):
        """Verifica che la strategia Tri-Engine generi attivamente tutti e 3 i setup."""
        setups_present = set(self.signals["setup_type"].unique())
        self.assertIn("PULLBACK_REATTIVO", setups_present, "Setup 1 (PULLBACK_REATTIVO) non generato")
        self.assertIn("MOMENTUM_BREAKOUT", setups_present, "Setup 2 (MOMENTUM_BREAKOUT) non generato")
        self.assertIn("MEAN_REVERSION_BOUNCE", setups_present, "Setup 3 (MEAN_REVERSION_BOUNCE) non generato")

        counts = self.signals["setup_type"].value_counts()
        self.assertGreater(counts.get("PULLBACK_REATTIVO", 0), 100, "Troppi pochi segnali Pullback")
        self.assertGreater(counts.get("MOMENTUM_BREAKOUT", 0), 500, "Troppi pochi segnali Breakout")
        self.assertGreater(counts.get("MEAN_REVERSION_BOUNCE", 0), 100, "Troppi pochi segnali Mean-Reversion")

    def test_setup3_active_in_bear_regime(self):
        """Verifica che il Setup 3 (Mean-Reversion) operi in Bear Market (Prezzo < EMA200 o EMA50 < EMA200) o Range (ADX < 24)."""
        s3_rows = self.signals[self.signals["setup_type"] == "MEAN_REVERSION_BOUNCE"]
        self.assertFalse(s3_rows.empty, "Nessun segnale per Setup 3")
        is_bear_or_range = (s3_rows["close"] < s3_rows["ema_200"]) | (s3_rows["ema_50"] < s3_rows["ema_200"]) | (s3_rows["adx"] < 24.0)
        self.assertTrue(is_bear_or_range.all(), "Setup 3 generato fuori dal regime Bear/Range!")

    def test_rl_error_decay_and_probed_recovery(self):
        """Verifica che il decadimento temporale (Half-Life Decay) sblocchi gli stati bloccati."""
        rl = RLExperienceEngine(memory_file="/tmp/test_tri_mem.json")
        rl.memory = {}
        # Simulate 2 consecutive errors on a state to block it
        key = "PULLBACK_REATTIVO|FEAR|LOW_VOL|FLAT_BEAR|MID"
        rl.record_feedback(key, is_win=False, net_return_pct=-1.8)
        rl.record_feedback(key, is_win=False, net_return_pct=-1.8)

        # Before decay: state must be blocked
        ok, reason, stats = rl.evaluate_proposal(key)
        self.assertFalse(ok, "Lo stato con 2 errori non e stato bloccato!")

        # Apply time decay
        unblocked = rl.apply_time_decay()
        self.assertGreaterEqual(unblocked, 1, "Il decadimento non ha sbloccato alcuno stato")

        # After decay: state must enter PROBED_RECOVERY with protected sizing
        ok_after, reason_after, stats_after = rl.evaluate_proposal(key)
        self.assertTrue(ok_after, "Lo stato non e stato sbloccato in PROBED_RECOVERY!")
        self.assertIn("PROBED_RECOVERY", reason_after)
        self.assertAlmostEqual(stats_after["sizing_multiplier"], 0.45, places=2)

    def test_continuous_compounding_scaling(self):
        """Verifica che con RL_COMPOUNDING=True la size degli ordini scali con il capitale oltre i 1000 USDC."""
        rules = {
            "tick_size": 0.1,
            "qty_step": 0.00001,
            "min_qty": 0.00001,
            "min_order_amount": 5.0,
            "max_order_amount": 100000.0,
        }
        bt = Backtester(rules=rules, fee_rate=0.001)
        # Verify that max_notional with compounding scales above $1000
        Config.RL_COMPOUNDING = True
        sample_df = self.signals.iloc[500:600].copy().reset_index(drop=True)
        # With $10,000 capital, order notional can scale up to 9500 USDC
        trades, m = bt.run(sample_df, initial_capital=10000.0, verbose=False, use_rl=False)
        if not trades.empty:
            notionals = trades["qty"] * trades["entry_price"]
            self.assertGreater(notionals.max(), 1000.0, "Compounding disabilitato: notional bloccato a 1000")

    def test_radar_table_zero_truncation(self):
        """Verifica che la tabella Radar della TUI renderizzi a 257, 250 e 190 caratteri SENZA alcun troncamento (...)"""
        from rich.console import Console
        from app_tui import TradingAppTUI
        from unittest.mock import MagicMock

        # Mock engine per TUI
        mock_engine = MagicMock()
        mock_engine._open_order.return_value = None
        mock_engine._get_state.side_effect = lambda k, default=None: "0" if "loss" in k or "tier" in k else ("0.0" if "score" in k else default)
        mock_engine.rl_engine = None

        tui = TradingAppTUI(mock_engine)
        tui.last_indicators = {
            "close": 73571.0, "open": 73410.0, "high": 74794.0, "low": 73299.0,
            "ema_20": 73500.0, "ema_50": 74336.0, "ema_100": 75000.0, "ema_200": 75980.0,
            "rsi": 48.2, "rsi_7": 35.1, "adx": 19.4, "vol_ratio": 0.95,
            "bb_lower_165": 72760.0, "bb_lower": 72500.0, "bb_mid": 73500.0,
            "squeeze": False, "prev_squeeze": False, "prev_high_12": 74794.0, "prev_high_20": 75000.0,
            "close_location": 0.55
        }
        tui.fng_data = {"value": 32, "classification": "Neutral"}

        for w in [257, 250, 190]:
            c = Console(width=w, height=80, record=True)
            tui.console = c
            panel = tui.make_radar_panel()
            c.print(panel)
            txt = c.export_text()
            has_ellipsis = "\u2026" in txt or "..." in txt
            self.assertFalse(has_ellipsis, f"Rilevato troncamento (...) nella tabella Radar a larghezza {w}!")

    def test_chart_panel_rendering(self):
        """Verifica che il pannello Grafico Candele Giapponesi 1H + Big Live Ticker renderizzi senza errori e senza troncamenti a 257 e 250."""
        from rich.console import Console
        from app_tui import TradingAppTUI
        from unittest.mock import MagicMock

        mock_engine = MagicMock()
        mock_engine._open_order.return_value = None
        mock_engine._get_state.side_effect = lambda k, default=None: "0" if "loss" in k or "tier" in k else ("0.0" if "score" in k else default)
        mock_engine.rl_engine = None

        tui = TradingAppTUI(mock_engine)
        tui.last_ticker = {
            "bid": 73643.8, "ask": 73643.9, "last": 73651.1,
            "high24h": 74907.6, "low24h": 72673.7, "change24h_pct": -1.46,
            "volume24h": 102.1, "turnover24h": 7500000.0
        }
        tui.last_indicators = {
            "close": 73643.8, "open": 73410.0, "high": 74794.0, "low": 73299.0,
            "ema_20": 73500.0, "ema_50": 74336.0, "ema_100": 75000.0, "ema_200": 75980.0,
            "rsi": 48.2, "rsi_7": 35.1, "adx": 19.4, "vol_ratio": 0.95,
            "bb_lower_165": 72760.0, "bb_lower": 72500.0, "bb_mid": 73500.0,
            "prev_high_12": 74794.0
        }
        tui.cached_chart_candles = [
            {"open": 73400.0 + i*10, "high": 73500.0 + i*10, "low": 73350.0 + i*10, "close": 73480.0 + i*10, "volume": 50.0 + i}
            for i in range(50)
        ]

        for w in [257, 250]:
            c = Console(width=w, height=80, record=True)
            tui.console = c
            panel = tui.make_chart_panel()
            c.print(panel)
            txt = c.export_text()
            has_ellipsis = "\u2026" in txt or "..." in txt
            self.assertFalse(has_ellipsis, f"Rilevato troncamento (...) nel Chart Panel a larghezza {w}!")
            self.assertIn("Grafico Candele Giapponesi", txt)
            self.assertIn("LIVE", txt)

if __name__ == "__main__":
    unittest.main()
