from datetime import datetime, timezone

import pandas as pd
from pybit.unified_trading import HTTP
from config import Config

class BybitClient:
    def __init__(self, public_only: bool = False):
        self.public_only = public_only
        self.session = HTTP(
            domain=Config.BYBIT_DOMAIN,
            tld=Config.BYBIT_TLD,
            demo=Config.DEMO_MODE if not public_only else False,
            api_key=None if public_only else Config.API_KEY,
            api_secret=None if public_only else Config.API_SECRET,
            recv_window=10000
        )

    def get_quote_balance(self, currency: str = Config.QUOTE_ASSET) -> float:
        if self.public_only:
            raise RuntimeError("Saldo privato disabilitato nel client pubblico paper.")
        resp = self.session.get_wallet_balance(accountType="UNIFIED", coin=currency)
        coins = [coin for account in resp["result"]["list"] for coin in account["coin"]]
        for c in coins:
            if c["coin"] == currency:
                wallet_balance = float(c.get("walletBalance") or 0)
                locked = float(c.get("locked") or 0)
                return max(0.0, wallet_balance - locked)
        return 0.0

    def get_eur_balance(self) -> float:
        return self.get_quote_balance("EUR")

    def get_eur_usdc_rate(self) -> float:
        """Return USDC per EUR from the EURC/USDC spot mid-price."""
        response = self.session.get_tickers(
            category=Config.CATEGORY,
            symbol=Config.EURC_USDC_SYMBOL,
        )
        tickers = response["result"]["list"]
        if not tickers:
            raise ValueError(f"Tasso EURC/USDC non disponibile su {Config.BYBIT_TLD.upper()}.")
        ticker = tickers[0]
        bid = float(ticker.get("bid1Price") or 0)
        ask = float(ticker.get("ask1Price") or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        last_price = float(ticker.get("lastPrice") or 0)
        if last_price <= 0:
            raise ValueError("Il ticker EURC/USDC non contiene un prezzo valido.")
        return last_price

    def get_usdt_balance(self) -> float:
        return self.get_quote_balance("USDT")

    def get_instrument_rules(self, symbol: str = Config.SYMBOL) -> dict:
        resp = self.session.get_instruments_info(
            category=Config.CATEGORY,
            symbol=symbol
        )
        instruments = resp["result"]["list"]
        if not instruments:
            raise ValueError(f"Strumento {symbol} non trovato su Bybit {Config.BYBIT_TLD.upper()}.")
        info = instruments[0]
        lot_filter = info["lotSizeFilter"]
        return {
            "symbol": info["symbol"],
            "base_asset": info["baseCoin"],
            "quote_asset": info["quoteCoin"],
            "tick_size": float(info["priceFilter"]["tickSize"]),
            "qty_step": float(lot_filter.get("qtyStep") or lot_filter["basePrecision"]),
            "min_qty": float(lot_filter.get("minOrderQty") or lot_filter["basePrecision"]),
            "min_order_amount": float(lot_filter.get("minOrderAmt") or 0),
            "max_order_amount": float(lot_filter.get("maxOrderAmt") or 0),
            "status": info["status"],
        }

    def get_spot_ticker(self, symbol: str = Config.SYMBOL) -> dict:
        response = self.session.get_tickers(category=Config.CATEGORY, symbol=symbol)
        tickers = response["result"]["list"]
        if not tickers:
            raise ValueError(f"Ticker {symbol} non disponibile su Bybit {Config.BYBIT_TLD.upper()}.")
        ticker = tickers[0]
        bid = float(ticker.get("bid1Price") or 0)
        ask = float(ticker.get("ask1Price") or 0)
        last = float(ticker.get("lastPrice") or 0)
        if bid <= 0 or ask <= 0 or last <= 0:
            raise ValueError(f"Ticker {symbol} privo di bid/ask/last validi.")
        return {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "last": last,
            "high24h": float(ticker.get("highPrice24h") or 0),
            "low24h": float(ticker.get("lowPrice24h") or 0),
            "change24h_pct": float(ticker.get("price24hPcnt") or 0) * 100.0,
            "volume24h": float(ticker.get("volume24h") or 0),
            "turnover24h": float(ticker.get("turnover24h") or 0),
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def get_recent_klines(self, symbol: str = Config.SYMBOL, limit: int = 1000) -> pd.DataFrame:
        if not 1 <= limit <= 1000:
            raise ValueError("limit deve essere compreso fra 1 e 1000.")
        response = self.session.get_kline(
            category=Config.CATEGORY,
            symbol=symbol,
            interval=Config.TIMEFRAME,
            limit=limit,
        )
        rows = response["result"]["list"]
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype("int64"), unit="ms", utc=True).dt.tz_localize(None)
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])[
            ["timestamp", "open", "high", "low", "close", "volume"]
        ].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
