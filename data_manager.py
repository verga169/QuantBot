import os
import time
from datetime import datetime, timezone
from typing import Optional
import ccxt
import pandas as pd
import numpy as np
from config import Config
from exchange_client import BybitClient

class DataManager:
    DEFAULT_MARKET_EXCHANGES = (
        "bybit", "bybit_global", "binance", "okx", "kraken", "coinbase", "bitstamp", "kucoin", "bitget",
        "gate", "mexc", "htx", "poloniex", "phemex", "cryptocom", "whitebit", "bitvavo", "gemini",
    )
    DEFAULT_QUOTE_ASSETS = ("USDT", "USDC", "USD", "EUR")
    TIMEFRAMES = {
        "15": ("15m", 15 * 60 * 1000),
        "60": ("1h", 60 * 60 * 1000),
        "240": ("4h", 4 * 60 * 60 * 1000),
        "D": ("1d", 24 * 60 * 60 * 1000),
    }

    def __init__(self, client: Optional[BybitClient] = None):
        self.client = client
        os.makedirs(Config.DATA_DIR, exist_ok=True)

    def _timeframe(self) -> tuple[str, int]:
        try:
            return self.TIMEFRAMES[str(Config.TIMEFRAME)]
        except KeyError as exc:
            raise ValueError(f"Timeframe non supportato: {Config.TIMEFRAME}") from exc

    def load_or_fetch_10y(self, years: int = 10) -> pd.DataFrame:
        """Carica o costruisce lo storico e verifica che sia continuo e aggiornato."""
        file_path = os.path.join(
            Config.DATA_DIR,
            f"{Config.SYMBOL}_10Y_{Config.TIMEFRAME}m_{Config.BYBIT_TLD}.parquet",
        )
        _, interval_ms = self._timeframe()
        now_ms = int(time.time() * 1000)
        target_start_ms = (now_ms - int(years * 365.25 * 86400 * 1000)) // interval_ms * interval_ms
        end_exclusive_ms = now_ms // interval_ms * interval_ms
        if Config.QUOTE_ASSET == "USDC":
            available_starts = [
                self._cached_market_start_ms(exchange_id, f"{Config.BASE_ASSET}/USDC")
                for exchange_id in ("binance", "bybit_global", "bybit")
            ]
            available_starts = [timestamp for timestamp in available_starts if timestamp is not None]
            if available_starts:
                target_start_ms = max(target_start_ms, min(available_starts))

        if os.path.exists(file_path):
            df = self._normalise(pd.read_parquet(file_path))
            if df.empty or df["timestamp"].iloc[0] > pd.to_datetime(target_start_ms, unit="ms"):
                historical_start = target_start_ms if df.empty else target_start_ms
                historical_end = end_exclusive_ms if df.empty else int(df["timestamp"].iloc[0].value // 1_000_000)
                df = pd.concat([self._fetch_historical_range(historical_start, historical_end), df])

            before_repair = len(df)
            df = self._repair_gaps(df)
            if len(df) > before_repair:
                print(f"Riparate {len(df) - before_repair} candele mancanti nello storico.")

            df = self._normalise(df)
            if Config.QUOTE_ASSET == "USDC" and not df.empty:
                target_start_ms = max(target_start_ms, int(df["timestamp"].iloc[0].value // 1_000_000))
            last_ts_ms = int(df["timestamp"].iloc[-1].value // 1_000_000) if not df.empty else target_start_ms
            if last_ts_ms < end_exclusive_ms - interval_ms:
                print("Aggiornamento incrementale delle ultime candele da Bybit...")
                recent = self._fetch_ccxt_range(
                    "bybit", f"{Config.BASE_ASSET}/{Config.QUOTE_ASSET}", last_ts_ms + interval_ms, end_exclusive_ms
                )
                df = self._normalise(pd.concat([df, recent], ignore_index=True))

            self._validate_coverage(df, target_start_ms, end_exclusive_ms, allow_short_gaps=Config.QUOTE_ASSET == "USDC")
            df.to_parquet(file_path, index=False)
            print(f"Database caricato: {len(df)} candele ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")
            return df

        print(f"=== COSTRUZIONE STORICO {years} ANNI (2016 -> OGGI) ===")
        df = self._fetch_historical_range(target_start_ms, end_exclusive_ms)
        df = self._repair_gaps(df)
        if Config.QUOTE_ASSET == "USDC" and not df.empty:
            target_start_ms = max(target_start_ms, int(df["timestamp"].iloc[0].value // 1_000_000))
        self._validate_coverage(df, target_start_ms, end_exclusive_ms, allow_short_gaps=Config.QUOTE_ASSET == "USDC")
        df.to_parquet(file_path, index=False)
        print(f"Storico verificato e salvato: {len(df)} candele totali.")
        return df

    def collect_exchange_histories(
        self,
        years: float = 10,
        exchange_ids: Optional[list[str]] = None,
        base_assets: Optional[list[str]] = None,
        quote_assets: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Collect spot OHLCV per venue and quote asset without stitching markets together."""
        if years <= 0:
            raise ValueError("years deve essere maggiore di zero.")

        timeframe, interval_ms = self._timeframe()
        now_ms = int(time.time() * 1000)
        start_ms = (now_ms - int(years * 365.25 * 86400 * 1000)) // interval_ms * interval_ms
        end_ms = now_ms // interval_ms * interval_ms
        requested_start = pd.to_datetime(start_ms, unit="ms")
        requested_end = pd.to_datetime(end_ms, unit="ms")
        collected_at = pd.Timestamp.now(tz="UTC").tz_localize(None)
        bases = set(base_assets or ["BTC"])
        quotes = set(quote_assets or self.DEFAULT_QUOTE_ASSETS)
        exchanges = exchange_ids or list(self.DEFAULT_MARKET_EXCHANGES)
        reports = []

        for exchange_id in exchanges:
            try:
                ccxt_id = "bybit" if exchange_id == "bybit_global" else exchange_id
                exchange = getattr(ccxt, ccxt_id)({"enableRateLimit": True})
                exchange_host = self._exchange_host(exchange_id)
                if exchange_id in {"bybit", "bybit_global"}:
                    exchange.hostname = "bybit.com" if exchange_id == "bybit_global" else exchange_host
                if not exchange.has.get("fetchOHLCV") or timeframe not in (exchange.timeframes or {}):
                    reports.append({
                        "exchange_id": exchange_id, "exchange_host": exchange_host, "timeframe": timeframe,
                        "requested_start_utc": requested_start, "requested_end_utc": requested_end,
                        "collected_at_utc": collected_at, "status": "unsupported_timeframe",
                    })
                    continue
                markets = exchange.load_markets()
                symbols = sorted({
                    market["symbol"]
                    for market in markets.values()
                    if market.get("base") in bases
                    and market.get("quote") in quotes
                    and market.get("spot")
                    and market.get("active") is not False
                })
            except Exception as exc:
                reports.append({
                    "exchange_id": exchange_id, "exchange_host": self._exchange_host(exchange_id), "timeframe": timeframe,
                    "requested_start_utc": requested_start, "requested_end_utc": requested_end,
                    "collected_at_utc": collected_at, "status": "exchange_error", "error": str(exc)[:300],
                })
                continue

            if not symbols:
                reports.append({
                    "exchange_id": exchange_id, "exchange_host": self._exchange_host(exchange_id), "timeframe": timeframe,
                    "requested_start_utc": requested_start, "requested_end_utc": requested_end,
                    "collected_at_utc": collected_at, "status": "no_matching_spot_market",
                })
                continue

            for symbol in symbols:
                try:
                    frame, file_path = self._collect_spot_pair_history(
                        exchange, exchange_id, exchange_host, symbol, start_ms, end_ms
                    )
                    if frame.empty:
                        reports.append({
                            "exchange_id": exchange_id, "exchange_host": exchange_host, "market_symbol": symbol,
                            "timeframe": timeframe,
                            "requested_start_utc": requested_start,
                            "requested_end_utc": requested_end,
                            "collected_at_utc": collected_at,
                            "status": "no_data",
                            "file": os.path.relpath(file_path, Config.DATA_DIR),
                        })
                        continue

                    timestamps = self._timestamps_ms(frame["timestamp"])
                    gaps = timestamps[1:] - timestamps[:-1]
                    gap_intervals = gaps > interval_ms
                    missing_candles = int(((gaps[gap_intervals] // interval_ms) - 1).sum())
                    issues = []
                    if timestamps[0] > start_ms + interval_ms:
                        issues.append("limited_history")
                    if timestamps[-1] < end_ms - interval_ms:
                        issues.append("stale")
                    if missing_candles:
                        issues.append("gaps")
                    reports.append({
                        "exchange_id": exchange_id,
                        "exchange_host": exchange_host,
                        "market_symbol": symbol,
                        "timeframe": timeframe,
                        "requested_start_utc": requested_start,
                        "requested_end_utc": requested_end,
                        "collected_at_utc": collected_at,
                        "rows": len(frame),
                        "coverage_start_utc": frame["timestamp"].iloc[0],
                        "coverage_end_utc": frame["timestamp"].iloc[-1],
                        "missing_candles": missing_candles,
                        "gap_intervals": int(gap_intervals.sum()),
                        "status": ";".join(issues) if issues else "ok",
                        "file": os.path.relpath(file_path, Config.DATA_DIR),
                    })
                except Exception as exc:
                    reports.append({
                        "exchange_id": exchange_id,
                        "exchange_host": exchange_host,
                        "market_symbol": symbol,
                        "timeframe": timeframe,
                        "requested_start_utc": requested_start,
                        "requested_end_utc": requested_end,
                        "collected_at_utc": collected_at,
                        "status": "market_error",
                        "error": str(exc)[:300],
                    })

        manifest = pd.DataFrame(reports)
        manifest_path = os.path.join(Config.DATA_DIR, "market_manifest.parquet")
        if os.path.exists(manifest_path):
            previous = pd.read_parquet(manifest_path)
            key_columns = ["exchange_id", "exchange_host", "market_symbol", "timeframe"]
            for column in key_columns:
                if column not in previous.columns:
                    previous[column] = pd.NA
            new_keys = manifest[key_columns].drop_duplicates().assign(_updated=True)
            previous = previous.merge(new_keys, on=key_columns, how="left")
            previous = previous[previous["_updated"].isna()].drop(columns="_updated")
            manifest = pd.concat([previous, manifest], ignore_index=True, sort=False)
        manifest = manifest.sort_values(
            ["exchange_id", "exchange_host", "market_symbol", "timeframe"], na_position="last"
        ).reset_index(drop=True)
        manifest.to_parquet(manifest_path, index=False)
        print(f"Manifest mercati salvato: {manifest_path}")
        return manifest

    def _collect_spot_pair_history(
        self, exchange, exchange_id: str, exchange_host: str, symbol: str, start_ms: int, end_ms: int
    ) -> tuple[pd.DataFrame, str]:
        timeframe, interval_ms = self._timeframe()
        base, quote = symbol.split("/")[:2]
        file_name = f"{base}_{quote}_{timeframe}.parquet"
        file_path = os.path.join(Config.DATA_DIR, "markets", exchange_host, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        frame = self._normalise(pd.read_parquet(file_path)) if os.path.exists(file_path) else self._empty_frame()
        new_frames = []

        if frame.empty:
            new_frames.append(self._fetch_ccxt_range(exchange_id, symbol, start_ms, end_ms, exchange=exchange))
        else:
            timestamps = self._timestamps_ms(frame["timestamp"])
            if timestamps[0] > start_ms:
                new_frames.append(self._fetch_ccxt_range(
                    exchange_id, symbol, start_ms, int(timestamps[0]), exchange=exchange
                ))
            if timestamps[-1] < end_ms - interval_ms:
                new_frames.append(self._fetch_ccxt_range(
                    exchange_id, symbol, int(timestamps[-1] + interval_ms), end_ms, exchange=exchange
                ))

        tagged_frames = [self._tag_market_rows(part, exchange_id, exchange_host, symbol) for part in new_frames if not part.empty]
        if tagged_frames:
            frame = self._normalise(pd.concat([frame, *tagged_frames], ignore_index=True))
        frame = self._repair_market_gaps(exchange, exchange_id, exchange_host, symbol, frame)
        frame = self._normalise(frame)
        frame.to_parquet(file_path, index=False)
        return frame, file_path

    def _repair_market_gaps(
        self, exchange, exchange_id: str, exchange_host: str, symbol: str, frame: pd.DataFrame
    ) -> pd.DataFrame:
        frame = self._normalise(frame)
        if len(frame) < 2:
            return frame

        _, interval_ms = self._timeframe()
        timestamps = self._timestamps_ms(frame["timestamp"])
        gaps = timestamps[1:] - timestamps[:-1]
        repairs = []
        for index in (gaps > interval_ms).nonzero()[0]:
            gap_start = int(timestamps[index] + interval_ms)
            gap_end = int(timestamps[index + 1])
            expected = set(range(gap_start, gap_end, interval_ms))
            if len(expected) > 24:
                continue
            candidates = self._fetch_ccxt_range(
                exchange_id, symbol, max(0, gap_start - interval_ms), gap_end + interval_ms,
                exchange=exchange,
            )
            candidate_ms = self._timestamps_ms(candidates["timestamp"])
            mask = np.isin(candidate_ms, list(expected))
            recovered = candidates.loc[mask]
            if not recovered.empty:
                repairs.append(self._tag_market_rows(recovered, exchange_id, exchange_host, symbol))
        if repairs:
            frame = self._normalise(pd.concat([frame, *repairs], ignore_index=True))
        return frame

    def _tag_market_rows(self, frame: pd.DataFrame, exchange_id: str, exchange_host: str, symbol: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        base, quote = symbol.split("/")[:2]
        frame = frame.copy()
        frame["exchange_id"] = exchange_id
        frame["exchange_host"] = exchange_host
        frame["market_symbol"] = symbol
        frame["base_asset"] = base
        frame["quote_asset"] = quote
        frame["timeframe"] = self._timeframe()[0]
        frame["data_source"] = f"{exchange_host}:{symbol}"
        frame["ingested_at_utc"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
        return frame

    @staticmethod
    def _timestamps_ms(timestamps: pd.Series) -> np.ndarray:
        return timestamps.astype("datetime64[ns]").astype("int64").to_numpy() // 1_000_000

    def _fetch_historical_range(self, start_ms: int, end_ms: int) -> pd.DataFrame:
        if Config.QUOTE_ASSET == "USDC":
            symbol = f"{Config.BASE_ASSET}/USDC"
            source_starts = [
                (self._cached_market_start_ms(exchange_id, symbol), exchange_id)
                for exchange_id in ("binance", "bybit_global", "bybit")
            ]
            source_starts = sorted(
                (timestamp, exchange_id)
                for timestamp, exchange_id in source_starts
                if timestamp is not None
            )
            if not source_starts:
                raise ValueError("Nessuno storico BTC/USDC locale. Eseguire prima la raccolta mercati.")

            frames = []
            for index, (source_start, exchange_id) in enumerate(source_starts):
                segment_start = max(start_ms, source_start)
                segment_end = min(end_ms, source_starts[index + 1][0]) if index + 1 < len(source_starts) else end_ms
                if segment_start < segment_end:
                    frames.append(self._load_or_fetch_market_range(exchange_id, symbol, segment_start, segment_end))
            return self._normalise(pd.concat(frames, ignore_index=True)) if frames else self._empty_frame()

        split_ms = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        frames = []
        if Config.QUOTE_ASSET == "EUR":
            old_exchange, old_symbol = "bitstamp", "BTC/EUR"
            recent_exchange, recent_symbol = "coinbase", "BTC/EUR"
        else:
            old_exchange, old_symbol = "bitstamp", "BTC/USD"
            recent_exchange, recent_symbol = "binance", f"BTC/{Config.QUOTE_ASSET}"
        if start_ms < split_ms:
            old_end = min(end_ms, split_ms)
            frames.append(self._load_or_fetch_market_range(old_exchange, old_symbol, start_ms, old_end))
        if end_ms > split_ms:
            modern_start = max(start_ms, split_ms)
            bybit_start = self._cached_market_start_ms("bybit", f"{Config.BASE_ASSET}/{Config.QUOTE_ASSET}")
            if Config.QUOTE_ASSET == "EUR" and bybit_start is not None and bybit_start < end_ms:
                recent_end = min(end_ms, bybit_start)
                if modern_start < recent_end:
                    frames.append(self._load_or_fetch_market_range(
                        recent_exchange, recent_symbol, modern_start, recent_end
                    ))
                bybit_begin = max(modern_start, bybit_start)
                if bybit_begin < end_ms:
                    frames.append(self._load_or_fetch_market_range(
                        "bybit", f"{Config.BASE_ASSET}/{Config.QUOTE_ASSET}", bybit_begin, end_ms
                    ))
            else:
                frames.append(self._load_or_fetch_market_range(recent_exchange, recent_symbol, modern_start, end_ms))
        return self._normalise(pd.concat(frames, ignore_index=True)) if frames else self._empty_frame()

    def _cached_market_start_ms(self, exchange_id: str, symbol: str) -> Optional[int]:
        cached = self._read_market_cache(exchange_id, symbol)
        if cached is None or cached.empty:
            return None
        return int(cached["timestamp"].iloc[0].value // 1_000_000)

    def _market_cache_path(self, exchange_id: str, symbol: str) -> str:
        timeframe, _ = self._timeframe()
        base, quote = symbol.split("/")[:2]
        exchange_host = self._exchange_host(exchange_id)
        return os.path.join(Config.DATA_DIR, "markets", exchange_host, f"{base}_{quote}_{timeframe}.parquet")

    def _read_market_cache(self, exchange_id: str, symbol: str) -> Optional[pd.DataFrame]:
        file_path = self._market_cache_path(exchange_id, symbol)
        if not os.path.exists(file_path):
            return None
        frame = self._normalise(pd.read_parquet(file_path))
        if exchange_id == "bybit_global" and not frame.empty:
            frame["exchange_id"] = "bybit"
            frame["exchange_host"] = "bybit.com"
            frame["market_symbol"] = symbol
            frame["base_asset"], frame["quote_asset"] = symbol.split("/")[:2]
            frame["timeframe"] = self._timeframe()[0]
            frame["data_source"] = f"bybit.com:{symbol}"
        return frame

    def _load_or_fetch_market_range(self, exchange_id: str, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        _, interval_ms = self._timeframe()
        cached = self._read_market_cache(exchange_id, symbol)
        if cached is None:
            return self._fetch_ccxt_range(exchange_id, symbol, start_ms, end_ms)

        if cached.empty:
            return self._fetch_ccxt_range(exchange_id, symbol, start_ms, end_ms)
        timestamps = self._timestamps_ms(cached["timestamp"])
        frames = [cached.loc[(timestamps >= start_ms) & (timestamps < end_ms)]]
        if timestamps[0] > start_ms:
            frames.append(self._fetch_ccxt_range(exchange_id, symbol, start_ms, min(int(timestamps[0]), end_ms)))
        if timestamps[-1] < end_ms - interval_ms:
            frames.append(self._fetch_ccxt_range(
                exchange_id, symbol, max(start_ms, int(timestamps[-1] + interval_ms)), end_ms
            ))
        return self._normalise(pd.concat(frames, ignore_index=True))

    def _repair_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._normalise(df)
        if len(df) < 2:
            return df

        _, interval_ms = self._timeframe()
        timestamps = (df["timestamp"].astype("datetime64[ns]").astype("int64") // 1_000_000).to_numpy()
        gap_indexes = (timestamps[1:] - timestamps[:-1] > interval_ms).nonzero()[0]
        repairs = []
        cached_sources = {}
        split_ms = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

        for index in gap_indexes:
            gap_start = int(timestamps[index] + interval_ms)
            gap_end = int(timestamps[index + 1])
            missing_timestamps = set(range(gap_start, gap_end, interval_ms))
            if Config.QUOTE_ASSET == "EUR":
                preferred_sources = (
                    [("bitstamp", "BTC/EUR"), ("coinbase", "BTC/EUR"), ("bybit", "BTC/EUR")]
                    if gap_start < split_ms
                    else [("coinbase", "BTC/EUR"), ("bybit", "BTC/EUR"), ("bitstamp", "BTC/EUR")]
                )
            elif Config.QUOTE_ASSET == "USDC":
                preferred_sources = [
                    ("binance", "BTC/USDC"),
                    ("bybit_global", "BTC/USDC"),
                    ("bybit", "BTC/USDC"),
                ]
            else:
                preferred_sources = (
                    [("bitstamp", "BTC/USD"), ("binance", f"BTC/{Config.QUOTE_ASSET}"), ("bybit", f"BTC/{Config.QUOTE_ASSET}")]
                    if gap_start < split_ms
                    else [("binance", f"BTC/{Config.QUOTE_ASSET}"), ("bybit", f"BTC/{Config.QUOTE_ASSET}"), ("bitstamp", "BTC/USD")]
                )

            for exchange_id, symbol in preferred_sources:
                if not missing_timestamps:
                    break
                cache_key = (exchange_id, symbol)
                if cache_key not in cached_sources:
                    cached_sources[cache_key] = self._read_market_cache(exchange_id, symbol)
                cached = cached_sources[cache_key]
                if cached is None:
                    candidates = self._fetch_ccxt_range(
                        exchange_id,
                        symbol,
                        max(0, gap_start - interval_ms),
                        gap_end + interval_ms,
                    )
                else:
                    cache_ms = self._timestamps_ms(cached["timestamp"])
                    cache_mask = (cache_ms >= gap_start - interval_ms) & (cache_ms < gap_end + interval_ms)
                    candidates = cached.loc[cache_mask]
                candidate_ms = candidates["timestamp"].astype("datetime64[ns]").astype("int64") // 1_000_000
                recovered = candidates[candidate_ms.isin(missing_timestamps)]
                if not recovered.empty:
                    repairs.append(recovered)
                    missing_timestamps.difference_update(
                        int(timestamp) for timestamp in candidate_ms[candidate_ms.isin(missing_timestamps)]
                    )

        if repairs:
            df = self._normalise(pd.concat([df, *repairs], ignore_index=True))
        return df

    def _validate_coverage(
        self, df: pd.DataFrame, start_ms: int, end_ms: int, allow_short_gaps: bool = False
    ) -> None:
        _, interval_ms = self._timeframe()
        if df.empty:
            raise ValueError("Storico OHLCV vuoto: impossibile eseguire il backtest.")

        timestamps = (df["timestamp"].astype("datetime64[ns]").astype("int64") // 1_000_000).to_numpy()
        expected_start = start_ms // interval_ms * interval_ms
        expected_last = end_ms - interval_ms
        if timestamps[0] > expected_start:
            raise ValueError(f"Storico incompleto all'inizio: prima candela {df['timestamp'].iloc[0]}.")
        if timestamps[-1] < expected_last:
            raise ValueError(f"Storico non aggiornato: ultima candela {df['timestamp'].iloc[-1]}.")

        gaps = timestamps[1:] - timestamps[:-1]
        missing = gaps > interval_ms
        if missing.any():
            gap_lengths = ((gaps[missing] // interval_ms) - 1).astype(int)
            count = int(gap_lengths.sum())
            if allow_short_gaps and int(gap_lengths.max()) <= 24:
                first_gap = pd.to_datetime(timestamps[missing.nonzero()[0][0]] + interval_ms, unit="ms")
                print(f"[WARN] {count} candele orarie mancanti; prima lacuna da {first_gap}.")
                return
            first_gap = pd.to_datetime(timestamps[missing.nonzero()[0][0]] + interval_ms, unit="ms")
            raise ValueError(f"Storico ancora incompleto: {count} candele mancanti, prima lacuna da {first_gap}.")

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "data_source"])

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return self._empty_frame()
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Dataset OHLCV senza colonne richieste: {', '.join(sorted(missing))}")
        optional = [
            column for column in (
                "data_source", "exchange_id", "exchange_host", "market_symbol", "base_asset",
                "quote_asset", "timeframe", "ingested_at_utc"
            ) if column in df.columns
        ]
        df = df[required + optional].copy()
        if "data_source" not in df.columns:
            df["data_source"] = "legacy_cache_unknown"
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        for column in required[1:]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["data_source"] = df["data_source"].fillna("legacy_cache_unknown")
        
        # Rigorous Data Sanitization: drop non-positive prices, negative volumes, and corrupt candles
        clean_mask = (
            (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) &
            (df["volume"] >= 0) & (df["high"] >= df["low"]) &
            (df["high"] >= df["open"]) & (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) & (df["low"] <= df["close"]) &
            np.isfinite(df["open"]) & np.isfinite(df["high"]) & np.isfinite(df["low"]) & np.isfinite(df["close"])
        )
        df = df[clean_mask]
        return df.dropna(subset=required).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    def _fetch_ccxt_range(self, exchange_id: str, symbol: str, start_ms: int, end_ms: int, exchange=None) -> pd.DataFrame:
        if exchange is None:
            ccxt_id = "bybit" if exchange_id == "bybit_global" else exchange_id
            exchange = getattr(ccxt, ccxt_id)({"enableRateLimit": True})
            if exchange_id in {"bybit", "bybit_global"}:
                exchange.hostname = "bybit.com" if exchange_id == "bybit_global" else self._exchange_host(exchange_id)
        timeframe, step_ms = self._timeframe()

        all_rows = []
        last_progress = 0
        cursor = start_ms // step_ms * step_ms

        while cursor < end_ms:
            for attempt in range(5):
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise RuntimeError(
                            f"Download OHLCV fallito da {exchange_id} dopo 5 tentativi: {exc}"
                        ) from exc
                    time.sleep(2)
            if not ohlcv:
                break

            all_rows.extend(row for row in ohlcv if cursor <= row[0] < end_ms)
            last_fetched = max(row[0] for row in ohlcv)
            next_cursor = last_fetched + step_ms
            if next_cursor <= cursor:
                raise RuntimeError(f"Paginazione OHLCV bloccata su {exchange_id} {symbol} a {cursor}.")
            cursor = next_cursor
            if len(all_rows) >= last_progress + 10000:
                dt_str = datetime.fromtimestamp(cursor / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                print(f"  [{exchange_id.upper()}] Avanzamento storico: {dt_str} ({len(all_rows)} candele)...")
                last_progress = len(all_rows) // 10000 * 10000

        if not all_rows:
            return self._empty_frame()

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        source_id = "bybit.com" if exchange_id == "bybit_global" else self._exchange_host(exchange_id)
        df["data_source"] = f"{source_id}:{symbol}"
        return self._normalise(df)

    def _exchange_host(self, exchange_id: str) -> str:
        if exchange_id == "bybit_global":
            return "bybit.com"
        if exchange_id == "bybit":
            return f"{Config.BYBIT_DOMAIN}.{Config.BYBIT_TLD}"
        return exchange_id
