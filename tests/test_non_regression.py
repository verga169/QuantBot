import unittest
import pandas as pd
from config import Config
from strategy import QuantStrategy
from backtester import Backtester

class TestNonRegressionBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Config.TRI_ENGINE_MODE = False
        Config.RL_COMPOUNDING = False
        cls.df_10y = pd.read_parquet("data/BTCUSDC_10Y_60m_eu.parquet")
        cls.fng_df = pd.read_parquet("data/fear_greed_history.parquet")
        cls.strategy = QuantStrategy()
        cls.signals = cls.strategy.generate_signals(cls.df_10y, fng_df=cls.fng_df)
        cls.rules = {
            "tick_size": 0.1,
            "qty_step": 0.00001,
            "min_qty": 0.00001,
            "min_order_amount": 5.0,
            "max_order_amount": 100000.0,
        }

    def test_core_strategy_edge_preservation(self):
        """
        Non-Regression Test:
        Verifica che le ottimizzazioni strutturali mantengano o superino
        gli standard quantitativi di eccellenza quando in modalita baseline:
        - Win Rate >= 70%
        - Profit Factor >= 2.0
        - Max Drawdown <= 5.0% (limite max consentito 8.0%)
        - Expectancy positiva
        """
        from rl_experience_engine import RLExperienceEngine
        rl_engine = RLExperienceEngine(memory_file="data/is_validation_memory.json")
        bt = Backtester(rules=self.rules, fee_rate=Config.FEE_RATE, eur_to_quote_rate=1.1165)
        trades, m = bt.run(
            self.signals,
            initial_capital=1000.0,
            sl_pct=Config.DEFAULT_SL_PCT,
            tp_pct=Config.DEFAULT_TP_PCT,
            verbose=False,
            max_holding_hours=Config.MAX_HOLDING_HOURS,
            volatility_target_pct=Config.VOLATILITY_TARGET_PCT,
            trail_tier1_trigger=Config.TRAIL_TIER1_TRIGGER,
            trail_tier1_lock=Config.TRAIL_TIER1_LOCK,
            trail_tier2_trigger=Config.TRAIL_TIER2_TRIGGER,
            trail_tier2_lock=Config.TRAIL_TIER2_LOCK,
            use_rl=True,
            rl_engine=rl_engine,
        )

        self.assertGreaterEqual(m["win_rate"], 70.0, f"Win rate degradato: {m['win_rate']:.2f}% < 70%")
        self.assertGreaterEqual(m["profit_factor"], 2.0, f"Profit Factor degradato: {m['profit_factor']:.2f} < 2.0")
        self.assertLessEqual(m["max_dd"], 5.0, f"Max Drawdown aumentato oltre il target: {m['max_dd']:.2f}% > 5.0%")
        self.assertGreater(m["return_pct"], 15.0, f"Rendimento insufficiente: {m['return_pct']:+.2f}%")
        self.assertGreater(m["expectancy_usdc"], 0.0, "Expectancy per trade negativa!")

if __name__ == "__main__":
    unittest.main()
