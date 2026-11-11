# Market Data

## Collect Spot History

Install the project dependencies and run the collector from the project directory:

```bash
python -m pip install -r requirements.txt
python collect_market_data.py --years 10
```

The collector discovers active BTC spot markets in `USDT`, `USDC`, `USD`, and `EUR` on the configured CCXT venues. It keeps every exchange and quote currency separate; it does not stitch candles from different markets into one price series.

For Bybit EU, the adapter uses the `bybit.eu` hostname and stores files under `data/markets/bybit.eu/`, separate from the older global-Bybit cache. The default instrument is `BTCUSDC`. The USDC history uses Binance from the first available BTC/USDC candle, Bybit global for the 2021-2025 bridge, then Bybit EU BTC/USDC. Source venue changes remain marked on each row.

The default venue list is Bybit, Binance, OKX, Kraken, Coinbase, Bitstamp, KuCoin, Bitget, Gate, MEXC, HTX, Poloniex, Phemex, Crypto.com, WhiteBIT, Bitvavo, and Gemini. Availability, listing date, API retention, and geographic restrictions vary by venue.

Optional collection scope:

```bash
python collect_market_data.py --years 5 --timeframe 60 --exchanges bybit binance kraken --bases BTC ETH --quotes USDT USDC USD EUR
```

## Files And Provenance

- `data/markets/<exchange-host>/<BASE>_<QUOTE>_<timeframe>.parquet` stores OHLCV bars separately by venue and market. Rows include the market symbol, base/quote assets, timeframe, source, and ingestion timestamp.
- `data/market_manifest.parquet` reports row counts, actual date coverage, missing candles, gap intervals, and collection status for each market.
- `data/news_headlines.parquet` accumulates RSS headlines when the news engine runs, including source, feed URL, article ID/URL, sentiment score, publication time, and observation time. It starts collecting from the first run; it is not a historical news archive.
- `data/fear_greed_history.parquet` stores the available daily Fear & Greed history.

Manifest statuses may report limited history, stale data, or gaps. Exchanges do not all retain ten years of hourly candles. Small gaps are retried against the same exchange; unresolved gaps are reported, not interpolated. The old `BTCUSDT_10Y_60m.parquet` cache is a legacy composite with unknown per-row provenance and is not equivalent to a Bybit-only series.

## Bybit EU And EUR-USDC Sizing

The trading client defaults to `BTCUSDC` and the `api.bybit.eu` domain. Demo mode is on by default. The project includes a historical backtest and a continuous paper loop; neither sends real orders.

Optional `.env` settings:

```dotenv
BYBIT_TLD=eu
BYBIT_DEMO_MODE=true
BASE_ASSET=BTC
QUOTE_ASSET=USDC
BYBIT_SYMBOL=BTCUSDC
BACKTEST_CAPITAL_EUR=1000
MIN_ORDER_NOTIONAL_EUR=50
MAX_ORDER_NOTIONAL_EUR=1000
RISK_PER_TRADE=0.01
VOLATILITY_TARGET_PCT=1.0
EMA200_TREND_FILTER=true
FEE_RATE=0.001
SLIPPAGE_PCT=0.0005
MAX_HOLDING_HOURS=24
PAPER_DB_PATH=data/paper_trading_demo.sqlite3
PAPER_POLL_SECONDS=15
PAPER_NEWS_POLL_SECONDS=300
PAPER_INITIAL_CAPITAL_USDC=1000
PAPER_DAILY_LOSS_LIMIT_USDC=50
PAPER_MAX_CONSECUTIVE_LOSSES=2
PAPER_REVIEW_MIN_TRADES=30
LOCAL_TIMEZONE=Europe/Rome
```

The Bybit EU `EURC/USDC` mid-price converts EUR capital and order limits to USDC. The current rate is used as a single sizing snapshot throughout the historical backtest; it is not a historical FX series. The backtester sizes by risk first, then caps notional at the configured maximum. If the risk-sized order is below the configured minimum or available capital, it skips the trade instead of increasing risk. When hourly ATR is above `VOLATILITY_TARGET_PCT`, risk is scaled down; it is never scaled above the configured 1% base risk. The default backtest capital is a placeholder; set it to the intended account balance. Fee rates depend on the Bybit EU account tier and should be configured to match the account before interpreting results.

With `EMA200_TREND_FILTER=true`, long signals are allowed only while BTC closes above its 200-period EMA. This is a research filter, not a guarantee of profitability; the latest holdout contains only a small number of trades.

USDC-denominated BTC history starts on Binance on 2018-12-15, not ten years ago. The current composite uses Binance, Bybit global, and Bybit EU; Bybit EU's own history starts in June 2025. It has 59 isolated missing hourly candles before the Bybit bridge. They are reported, not synthesized; the backtester exits at the first available open if a position crosses a data gap and skips entries on that boundary. The Coinbase `BTC/USDC` cache was found to be numerically identical to `BTC/USD`, including before USDC existed, and is excluded from the USDC backtest.

## Paper And Walk-Forward

Start the continuous paper loop:

```bash
python paper_trading.py
```

For a single smoke-test cycle with an isolated database:

```bash
python paper_trading.py --once --db /tmp/bybit-paper.sqlite3
```

The paper process uses an unauthenticated public-data client; it does not load API credentials from `.env` and it never calls private wallet or order endpoints. It starts with a local-only 1,000 USDC balance, polls Bybit EU prices and closed candles, polls RSS news on its own interval (with time-decay sentiment scoring across CNBC, Investing.com, CoinDesk, Cointelegraph, Decrypt, and Bitcoin Magazine), and persists ticks, candles, headlines, simulated orders, fees, entry/exit slippage, equity, and events in `data/paper_trading_demo.sqlite3`. Entries simulate fills at ask plus configured slippage; exits use bid minus slippage. It enforces one open long, a 36-hour time exit, a 50 USDC intraday loss trigger, critical news defense exits on breaking negative spikes, and a lock after two consecutive losing trades, resetting the session on the next `Europe/Rome` day.

After each simulated close, a local review stores win rate, profit factor, losing streak, slippage, and performance grouped by setup/news/volatility/trend (`DIP_BUY_LONG`, `EMA100_PULLBACK`, `BULL_BREAKOUT`). It waits for at least 30 closed trades before issuing review flags.

Generate a chronological report with fixed strategy settings and continuous equity:

```bash
python walk_forward.py --window-months 6
python walk_forward.py --window-months 6 --ema200-filter off
```

The filtered report is saved to `data/walk_forward_usdc.csv`; the unfiltered comparison goes to `data/walk_forward_usdc_ema200_off.csv`. Across 16 rolling half-year windows from 2018 to 2026, the updated multi-setup strategy executes 270 trades (well above the 100-trade advisory threshold), achieving a Profit Factor of 1.25 and positive net capital growth (+29.14% in full backtest).