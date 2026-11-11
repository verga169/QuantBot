import pandas as pd
from config import Config
Config.TRI_ENGINE_MODE = True
Config.RL_COMPOUNDING = True
from strategy import QuantStrategy
from backtester import Backtester
from rl_experience_engine import RLExperienceEngine

def run_dataset_validation(name: str, path: str):
    print("=" * 65)
    print(f" VALIDATION RUN: {name}")
    print("=" * 65)
    df = pd.read_parquet(path)
    fng = pd.read_parquet("data/fear_greed_history.parquet")

    strat = QuantStrategy()
    sig = strat.generate_signals(df, fng_df=fng)

    rules = {
        "tick_size": 0.1,
        "qty_step": 0.00001,
        "min_qty": 0.00001,
        "min_order_amount": 5.0,
        "max_order_amount": 100000.0,
    }

    bt = Backtester(rules=rules, fee_rate=Config.FEE_RATE, eur_to_quote_rate=1.1165)
    rl = RLExperienceEngine(memory_file=f"/tmp/rl_{name.lower().replace('/', '_')}.json")
    rl.memory = {}
    bt.run(sig, initial_capital=1000.0, verbose=False, use_rl=True, rl_engine=rl)

    # Pass 2: Exploitation
    trades, m = bt.run(sig, initial_capital=1000.0, verbose=False, use_rl=True, rl_engine=rl)
    trades["year"] = pd.to_datetime(trades["entry_time"]).dt.year

    print(f"Total Trades        : {len(trades)}")
    print(f"Win Rate            : {m['win_rate']:.2f}%")
    print(f"Profit Factor       : {m['profit_factor']:.2f}")
    print(f"Max Drawdown        : {m['max_dd']:.2f}%")
    print(f"Final Capital       : ${m['final_capital']:.2f} ({m['return_pct']:+.2f}%)")
    print(f"CAGR                : {m.get('cagr_pct', 0.0):.2f}%")
    print(f"Sharpe Ratio (Ann.) : {m.get('sharpe_ratio', 0.0):.2f}")
    print(f"Sortino Ratio       : {m.get('sortino_ratio', 0.0):.2f}")
    print(f"Calmar Ratio        : {m.get('calmar_ratio', 0.0):.2f}")
    print(f"Expectancy USDC     : ${m.get('expectancy_usdc', 0.0):.2f}")

    print("\n--- ANNUALE BREAKDOWN ---")
    by_year = trades.groupby("year").agg(
        trades=("net_pnl", "count"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
        net_pnl=("net_pnl", "sum"),
        profit_factor=("net_pnl", lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if len(x[x < 0]) > 0 else 999.0)
    )
    print(by_year.to_string())

    print("\n--- SETUP BREAKDOWN ---")
    by_setup = trades.groupby("setup").agg(
        trades=("net_pnl", "count"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
        net_pnl=("net_pnl", "sum"),
        profit_factor=("net_pnl", lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if len(x[x < 0]) > 0 else 999.0)
    )
    print(by_setup.to_string())
    print("\n")

if __name__ == "__main__":
    run_dataset_validation("BTC_USDC_8Y", "data/BTCUSDC_10Y_60m_eu.parquet")
    run_dataset_validation("BTC_USDT_10Y", "data/BTCUSDT_10Y_60m.parquet")
