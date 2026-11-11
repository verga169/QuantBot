import argparse
import os

import pandas as pd

from backtester import Backtester
from config import Config
from data_manager import DataManager
from exchange_client import BybitClient
from sentiment_engine import SentimentEngine
from strategy import QuantStrategy
from rl_experience_engine import RLExperienceEngine


def run_walk_forward(
    window_months: int = 6,
    output_path: str | None = None,
    ema200_trend_filter: bool | None = None,
) -> pd.DataFrame:
    if window_months <= 0:
        raise ValueError("window_months deve essere maggiore di zero.")

    client = BybitClient()
    rules = client.get_instrument_rules(Config.SYMBOL)
    if rules["status"] != "Trading" or rules["quote_asset"] != Config.QUOTE_ASSET:
        raise ValueError(f"Strumento non negoziabile o quote asset non coerente: {rules}")
    eur_to_quote_rate = client.get_eur_usdc_rate() if Config.QUOTE_ASSET == "USDC" else 1.0

    candles = DataManager(client).load_or_fetch_10y(years=10)
    fear_and_greed = SentimentEngine().fetch_fear_and_greed_history()
    previous_filter = Config.EMA200_TREND_FILTER
    if ema200_trend_filter is not None:
        Config.EMA200_TREND_FILTER = ema200_trend_filter
    try:
        signals = QuantStrategy().generate_signals(candles, fng_df=fear_and_greed)
    finally:
        Config.EMA200_TREND_FILTER = previous_filter
    if signals.empty:
        raise ValueError("Nessun segnale disponibile per il walk-forward.")
    applied_filter = previous_filter if ema200_trend_filter is None else ema200_trend_filter

    backtester = Backtester(
        rules=rules,
        fee_rate=Config.FEE_RATE,
        eur_to_quote_rate=eur_to_quote_rate,
    )
    interval_ms = DataManager.TIMEFRAMES[str(Config.TIMEFRAME)][1]
    interval = pd.Timedelta(milliseconds=interval_ms)
    start = pd.Timestamp(signals["timestamp"].iloc[0])
    final = pd.Timestamp(signals["timestamp"].iloc[-1])
    capital = Config.BACKTEST_CAPITAL_EUR * eur_to_quote_rate
    continuous_rl_engine = RLExperienceEngine(
        min_experience_trials=2,
        min_state_win_rate=Config.RL_MIN_STATE_WR,
        max_consecutive_errors=Config.RL_MAX_CONSECUTIVE_ERRORS,
        learning_rate=Config.RL_LEARNING_RATE
    )
    rows = []

    while start <= final:
        end = min(start + pd.DateOffset(months=window_months), final + interval)
        period = signals.loc[
            (signals["timestamp"] >= start - interval) & (signals["timestamp"] < end)
        ].reset_index(drop=True)
        capital_at_start = capital

        if len(period) > 1:
            trades, metrics = backtester.run(
                period,
                initial_capital=capital,
                sl_pct=Config.DEFAULT_SL_PCT,
                tp_pct=Config.DEFAULT_TP_PCT,
                verbose=False,
                max_holding_hours=Config.MAX_HOLDING_HOURS,
                volatility_target_pct=Config.VOLATILITY_TARGET_PCT,
                trail_tier1_trigger=Config.TRAIL_TIER1_TRIGGER,
                trail_tier1_lock=Config.TRAIL_TIER1_LOCK,
                trail_tier2_trigger=Config.TRAIL_TIER2_TRIGGER,
                trail_tier2_lock=Config.TRAIL_TIER2_LOCK,
                use_rl=Config.RL_ACTIVE,
                rl_engine=continuous_rl_engine,
            )
            capital = float(metrics["final_capital"])
            if not trades.empty:
                notional_eur = trades["entry_price"] * trades["qty"] / eur_to_quote_rate
                notional_min = float(notional_eur.min())
                notional_max = float(notional_eur.max())
                data_gap_exits = int(trades["exit_reason"].eq("DATA_GAP_EXIT").sum())
            else:
                notional_min = notional_max = 0.0
                data_gap_exits = 0
            row = {
                "window_start_utc": start,
                "window_end_utc": min(end, final + interval) - interval,
                "trades": metrics["trades"],
                "win_rate_pct": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "return_pct": metrics["return_pct"],
                "max_drawdown_pct": metrics["max_dd"],
                "initial_capital_quote": capital_at_start,
                "final_capital_quote": capital,
                "data_gap_exits": data_gap_exits,
                "min_trade_notional_eur": notional_min,
                "max_trade_notional_eur": notional_max,
                "ema200_trend_filter": applied_filter,
            }
            rows.append(row)
        start = end

    report = pd.DataFrame(rows)
    if output_path is None:
        suffix = "_ema200_off" if not applied_filter else ""
        output_path = os.path.join(Config.DATA_DIR, f"walk_forward_{Config.QUOTE_ASSET.lower()}{suffix}.csv")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"Walk-forward a parametri fissi salvato: {output_path}")
    total_trades = int(report["trades"].sum()) if not report.empty else 0
    if total_trades < 100:
        print(f"[WARN] Campione ridotto: {total_trades} trade totali; non sufficiente per stimare un edge stabile.")
    return report


def run_is_oos_split(split_date: str = "2023-01-01") -> dict:
    """
    In-Sample vs Out-of-Sample Validation:
    Trains online experience on In-Sample (pre-2023), then tests the learned policy
    strictly out-of-sample on unseen post-2023 data.
    """
    client = BybitClient(public_only=True)
    rules = {
        "tick_size": 0.1,
        "qty_step": 0.00001,
        "min_qty": 0.00001,
        "min_order_amount": 5.0,
        "max_order_amount": 100000.0,
    }
    candles = DataManager(client).load_or_fetch_10y(years=10)
    fear_and_greed = SentimentEngine().fetch_fear_and_greed_history()
    signals = QuantStrategy().generate_signals(candles, fng_df=fear_and_greed)
    split_dt = pd.Timestamp(split_date)
    is_signals = signals[signals["timestamp"] < split_dt].reset_index(drop=True)
    oos_signals = signals[signals["timestamp"] >= split_dt].reset_index(drop=True)

    backtester = Backtester(rules=rules, fee_rate=Config.FEE_RATE, eur_to_quote_rate=1.1165)
    is_rl = RLExperienceEngine(
        memory_file=os.path.join(Config.DATA_DIR, "is_validation_memory.json"),
        min_experience_trials=2,
        min_state_win_rate=Config.RL_MIN_STATE_WR,
        max_consecutive_errors=Config.RL_MAX_CONSECUTIVE_ERRORS,
        learning_rate=Config.RL_LEARNING_RATE
    )
    is_trades, is_metrics = backtester.run(
        is_signals,
        initial_capital=1000.0,
        verbose=False,
        use_rl=True,
        rl_engine=is_rl,
    )

    oos_trades, oos_metrics = backtester.run(
        oos_signals,
        initial_capital=float(is_metrics["final_capital"]),
        verbose=False,
        use_rl=True,
        rl_engine=is_rl,
    )

    return {
        "in_sample": {
            "period": f"{is_signals['timestamp'].iloc[0]} -> {split_date}",
            "trades": is_metrics["trades"],
            "win_rate": is_metrics["win_rate"],
            "profit_factor": is_metrics["profit_factor"],
            "return_pct": is_metrics["return_pct"],
            "max_dd": is_metrics["max_dd"],
        },
        "out_of_sample": {
            "period": f"{split_date} -> {oos_signals['timestamp'].iloc[-1]}",
            "trades": oos_metrics["trades"],
            "win_rate": oos_metrics["win_rate"],
            "profit_factor": oos_metrics["profit_factor"],
            "return_pct": oos_metrics["return_pct"],
            "max_dd": oos_metrics["max_dd"],
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report cronologico a finestre fisse; non ottimizza i parametri e non invia ordini."
    )
    parser.add_argument("--window-months", type=int, default=6)
    parser.add_argument("--output", default=None)
    parser.add_argument("--ema200-filter", choices=("on", "off"), default=None)
    parser.add_argument("--is-oos", action="store_true", help="Esegue split In-Sample (pre-2023) vs Out-Of-Sample (post-2023)")
    args = parser.parse_args()

    if args.is_oos:
        print("\n=== VALIDAZIONE IN-SAMPLE VS OUT-OF-SAMPLE ===")
        res = run_is_oos_split()
        for k, v in res.items():
            print(f"\n[{k.upper()}] Periodo: {v['period']}")
            print(f"  Trades: {v['trades']} | Win Rate: {v['win_rate']:.2f}% | Profit Factor: {v['profit_factor']:.2f} | Return: {v['return_pct']:+.2f}% | Max DD: {v['max_dd']:.2f}%")
        return

    ema200_filter = None if args.ema200_filter is None else args.ema200_filter == "on"
    report = run_walk_forward(
        window_months=args.window_months,
        output_path=args.output,
        ema200_trend_filter=ema200_filter,
    )
    if report.empty:
        print("Nessuna finestra valutabile.")
    else:
        print(report.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()