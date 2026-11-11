"""
Automated Parameter Sensitivity & Robustness Suite.
Tests strategy behavior under ±10% perturbations on core parameters to verify
that performance sits on a robust, stable plateau and not on an overfitted peak.
"""

import math
from typing import Dict, Any, List
import pandas as pd
from config import Config
from strategy import QuantStrategy
from backtester import Backtester
from exchange_client import BybitClient
from rl_experience_engine import RLExperienceEngine

def run_sensitivity_tests(df_signals: pd.DataFrame, initial_capital: float = 1000.0) -> pd.DataFrame:
    rules = {
        "tick_size": 0.1,
        "qty_step": 0.00001,
        "min_qty": 0.00001,
        "min_order_amount": 5.0,
        "max_order_amount": 100000.0,
    }

    base_sl = Config.DEFAULT_SL_PCT
    base_tp = Config.DEFAULT_TP_PCT
    base_t1_trig = Config.TRAIL_TIER1_TRIGGER
    base_t1_lock = Config.TRAIL_TIER1_LOCK
    base_t2_trig = Config.TRAIL_TIER2_TRIGGER
    base_t2_lock = Config.TRAIL_TIER2_LOCK
    base_holding = Config.MAX_HOLDING_HOURS

    variations = [
        ("BASELINE", base_sl, base_tp, base_t1_trig, base_t1_lock, base_t2_trig, base_t2_lock, base_holding),
        ("SL -10% (1.62%)", base_sl * 0.90, base_tp, base_t1_trig, base_t1_lock, base_t2_trig, base_t2_lock, base_holding),
        ("SL +10% (1.98%)", base_sl * 1.10, base_tp, base_t1_trig, base_t1_lock, base_t2_trig, base_t2_lock, base_holding),
        ("TP -10% (3.42%)", base_sl, base_tp * 0.90, base_t1_trig, base_t1_lock, base_t2_trig, base_t2_lock, base_holding),
        ("TP +10% (4.18%)", base_sl, base_tp * 1.10, base_t1_trig, base_t1_lock, base_t2_trig, base_t2_lock, base_holding),
        ("Trail T1 -10% (1.08%)", base_sl, base_tp, base_t1_trig * 0.90, base_t1_lock, base_t2_trig, base_t2_lock, base_holding),
        ("Trail T1 +10% (1.32%)", base_sl, base_tp, base_t1_trig * 1.10, base_t1_lock, base_t2_trig, base_t2_lock, base_holding),
        ("Holding -10% (28.8h)", base_sl, base_tp, base_t1_trig, base_t1_lock, base_t2_trig, base_t2_lock, base_holding * 0.90),
        ("Holding +10% (35.2h)", base_sl, base_tp, base_t1_trig, base_t1_lock, base_t2_trig, base_t2_lock, base_holding * 1.10),
    ]

    results = []
    bt = Backtester(rules=rules, fee_rate=Config.FEE_RATE, eur_to_quote_rate=1.1165)

    for label, sl, tp, t1_tr, t1_lk, t2_tr, t2_lk, max_h in variations:
        trades_df, m = bt.run(
            df_signals,
            initial_capital=initial_capital,
            sl_pct=sl,
            tp_pct=tp,
            verbose=False,
            max_holding_hours=max_h,
            volatility_target_pct=Config.VOLATILITY_TARGET_PCT,
            trail_tier1_trigger=t1_tr,
            trail_tier1_lock=t1_lk,
            trail_tier2_trigger=t2_tr,
            trail_tier2_lock=t2_lk,
            use_rl=True,
        )
        is_robust = (m["win_rate"] >= 65.0) and (m["profit_factor"] >= 1.50) and (m["max_dd"] <= 8.0)
        results.append({
            "Perturbation": label,
            "Trades": m["trades"],
            "Win Rate %": round(m["win_rate"], 2),
            "Profit Factor": round(m["profit_factor"], 2),
            "Return %": round(m["return_pct"], 2),
            "Max DD %": round(m["max_dd"], 2),
            "Plateau Robustness": "✅ ROBUST PLATEAU" if is_robust else "⚠️ SENSITIVE",
        })

    report_df = pd.DataFrame(results)
    return report_df

if __name__ == "__main__":
    df_10y = pd.read_parquet("data/BTCUSDC_10Y_60m_eu.parquet")
    fng_df = pd.read_parquet("data/fear_greed_history.parquet")
    strat = QuantStrategy()
    signals = strat.generate_signals(df_10y, fng_df=fng_df)
    report = run_sensitivity_tests(signals)
    print("\n" + "=" * 75)
    print(" PARAMETER SENSITIVITY & ROBUSTNESS TEST (±10% PERTURBATION)")
    print("=" * 75)
    print(report.to_string(index=False))
    print("=" * 75)
