import os
from config import Config
from exchange_client import BybitClient
from data_manager import DataManager
from sentiment_engine import SentimentEngine
from strategy import QuantStrategy
from backtester import Backtester

def main():
    print("=== BYBIT EU SPOT - BACKTEST INTRADAY BTCUSDC ===")
    client = BybitClient()
    rules = client.get_instrument_rules(Config.SYMBOL)
    if rules["status"] != "Trading":
        raise ValueError(f"Lo strumento {Config.SYMBOL} non e' attivo: {rules['status']}")
    if rules["quote_asset"] != Config.QUOTE_ASSET:
        raise ValueError(
            f"Quote asset non coerente: {Config.SYMBOL} usa {rules['quote_asset']}, "
            f"configurato {Config.QUOTE_ASSET}."
        )
    eur_to_quote_rate = client.get_eur_usdc_rate() if Config.QUOTE_ASSET == "USDC" else 1.0
    if Config.QUOTE_ASSET == "USDC":
        initial_capital_quote = Config.BACKTEST_CAPITAL_USDC
        capital_eur_label = f"{initial_capital_quote / eur_to_quote_rate:.2f} EUR equiv."
    else:
        initial_capital_quote = Config.BACKTEST_CAPITAL_EUR
        capital_eur_label = f"{Config.BACKTEST_CAPITAL_EUR:.2f} EUR"

    mode = "DEMO" if Config.DEMO_MODE else "PRODUZIONE (ordini non abilitati)"
    print(f"Endpoint: {client.session.endpoint} | Modalita API: {mode}")
    print(
        f"Capitale simulato: {initial_capital_quote:.2f} {Config.QUOTE_ASSET} ({capital_eur_label}) | "
        f"Cambio EUR/{Config.QUOTE_ASSET}: {eur_to_quote_rate:.6f} | "
        f"Notional ordine: {Config.MIN_ORDER_NOTIONAL_USDC:.2f}-{Config.MAX_ORDER_NOTIONAL_USDC:.2f} {Config.QUOTE_ASSET} | "
        f"Rischio/trade: {Config.RISK_PER_TRADE * 100:.2f}% | "
        f"Target ATR orario: {Config.VOLATILITY_TARGET_PCT:.2f}% | "
        f"Filtro EMA200: {'ON' if Config.EMA200_TREND_FILTER else 'OFF'}"
    )

    sent = SentimentEngine()
    fng_df = sent.fetch_fear_and_greed_history()
    news_report = sent.analyze_live_news()

    print(f"Fear & Greed Attuale: {int(fng_df.iloc[-1]['fng_value'])}/100" if not fng_df.empty else "")
    print(f"News Macro Sentiment live (non incluso nel backtest): {news_report['avg_score']} ({news_report['regime']})\n")

    dm = DataManager(client)
    df_10y = dm.load_or_fetch_10y(years=10)
    _, interval_ms = dm._timeframe()
    timestamp_ms = dm._timestamps_ms(df_10y["timestamp"])
    gaps = timestamp_ms[1:] - timestamp_ms[:-1]
    gap_mask = gaps > interval_ms
    missing_candles = int((((gaps[gap_mask] // interval_ms) - 1).sum()))
    print(
        f"Dataset: {len(df_10y)} barre {df_10y['timestamp'].iloc[0]} -> {df_10y['timestamp'].iloc[-1]} "
        f"({Config.QUOTE_ASSET}); mancanti: {missing_candles}; "
        f"fonti: {df_10y['data_source'].value_counts().to_dict()}"
    )

    strategy = QuantStrategy()
    df_signals = strategy.generate_signals(
        df_10y,
        fng_df=fng_df,
        live_news_score=float(news_report.get("avg_score", 0.0)),
    )

    bt = Backtester(
        rules=rules,
        fee_rate=Config.FEE_RATE,
        eur_to_quote_rate=eur_to_quote_rate,
    )
    trades_df, _ = bt.run(
        df_signals,
        initial_capital=initial_capital_quote,
        sl_pct=Config.DEFAULT_SL_PCT,
        tp_pct=Config.DEFAULT_TP_PCT,
        verbose=True,
        max_holding_hours=Config.MAX_HOLDING_HOURS,
        volatility_target_pct=Config.VOLATILITY_TARGET_PCT,
        trail_tier1_trigger=Config.TRAIL_TIER1_TRIGGER,
        trail_tier1_lock=Config.TRAIL_TIER1_LOCK,
        trail_tier2_trigger=Config.TRAIL_TIER2_TRIGGER,
        trail_tier2_lock=Config.TRAIL_TIER2_LOCK,
        use_rl=Config.RL_ACTIVE,
    )

    if not trades_df.empty:
        out_csv = os.path.join(Config.DATA_DIR, "backtest_spot_usdc_trades.csv")
        trades_df.to_csv(out_csv, index=False)
        print(f"\nDettaglio operazioni Spot salvato in: {out_csv}")

if __name__ == "__main__":
    main()
