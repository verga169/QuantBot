import argparse
import hashlib
import json
import math
import os
import sqlite3
import time
import threading
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from config import Config
from data_manager import DataManager
from exchange_client import BybitClient
from sentiment_engine import SentimentEngine
from strategy import QuantStrategy
from rl_experience_engine import RLExperienceEngine


class PaperTradingEngine:
    """Read-only market/news polling with simulated spot fills persisted in SQLite."""

    def __init__(
        self,
        database_path: str = Config.PAPER_DB_PATH,
        client: BybitClient | None = None,
        sentiment: SentimentEngine | None = None,
    ):
        os.makedirs(os.path.dirname(database_path) or ".", exist_ok=True)
        self.database_path = database_path
        self.db_lock = threading.RLock()
        self.db = sqlite3.connect(database_path, timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.client = client or BybitClient(public_only=True)
        self.sentiment = sentiment or SentimentEngine()
        self.strategy = QuantStrategy()
        self.rl_engine = RLExperienceEngine()
        if self.rl_engine is not None:
            self.rl_engine.apply_time_decay()
        if Config.QUOTE_ASSET != "USDC":
            raise ValueError("Il paper account locale richiede QUOTE_ASSET=USDC.")
        self.rules = self.client.get_instrument_rules(Config.SYMBOL)
        if self.rules["status"] != "Trading" or self.rules["quote_asset"] != Config.QUOTE_ASSET:
            raise ValueError(f"Strumento paper non negoziabile/coerente: {self.rules}")
        self._create_schema()
        rate = self._eur_to_quote_rate()
        self._initialize_account(rate)

    def _create_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS paper_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at_utc TEXT NOT NULL,
                symbol TEXT NOT NULL,
                bid REAL NOT NULL,
                ask REAL NOT NULL,
                last REAL NOT NULL,
                eur_to_quote_rate REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_paper_prices_time ON paper_prices(observed_at_utc);
            CREATE TABLE IF NOT EXISTS paper_candles (
                timestamp_utc TEXT PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_news (
                fingerprint TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                article_id TEXT,
                title TEXT NOT NULL,
                score REAL NOT NULL,
                published_at_utc TEXT NOT NULL,
                observed_at_utc TEXT NOT NULL,
                url TEXT,
                feed_url TEXT
            );
            CREATE TABLE IF NOT EXISTS paper_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_time_utc TEXT NOT NULL UNIQUE,
                signal_setup TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                status TEXT NOT NULL,
                qty REAL NOT NULL,
                entry_time_utc TEXT NOT NULL,
                entry_reference REAL NOT NULL,
                entry_fill REAL NOT NULL,
                entry_slippage_bps REAL NOT NULL,
                entry_fee_quote REAL NOT NULL,
                notional_quote REAL NOT NULL,
                notional_eur REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                news_score REAL NOT NULL DEFAULT 0,
                atr_pct REAL NOT NULL DEFAULT 0,
                fng_value REAL NOT NULL DEFAULT 50,
                ema200_gap_pct REAL NOT NULL DEFAULT 0,
                exit_time_utc TEXT,
                exit_reference REAL,
                exit_fill REAL,
                exit_slippage_bps REAL,
                exit_fee_quote REAL,
                pnl_quote REAL,
                pnl_eur REAL,
                exit_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_paper_orders_status ON paper_orders(status);
            CREATE TABLE IF NOT EXISTS paper_equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at_utc TEXT NOT NULL,
                local_session_date TEXT NOT NULL,
                cash_quote REAL NOT NULL,
                position_qty REAL NOT NULL,
                equity_quote REAL NOT NULL,
                equity_eur REAL NOT NULL,
                daily_pnl_pct REAL NOT NULL,
                daily_loss_quote REAL NOT NULL DEFAULT 0,
                consecutive_losses INTEGER NOT NULL DEFAULT 0,
                eur_to_quote_rate REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                closed_trade_count INTEGER NOT NULL,
                report_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details_json TEXT NOT NULL
            );
            """
        )
        self._ensure_column("paper_orders", "news_score", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("paper_orders", "atr_pct", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("paper_orders", "fng_value", "REAL NOT NULL DEFAULT 50")
        self._ensure_column("paper_orders", "ema200_gap_pct", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("paper_equity", "daily_loss_quote", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("paper_equity", "consecutive_losses", "INTEGER NOT NULL DEFAULT 0")
        self.db.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in self.db.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _get_state(self, key: str, default: str = "") -> str:
        row = self.db.execute("SELECT value FROM paper_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def _set_state(self, key: str, value: object) -> None:
        self.db.execute(
            "INSERT INTO paper_state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    def _initialize_account(self, rate: float) -> None:
        if not self._get_state("cash_quote"):
            initial_capital = Config.PAPER_INITIAL_CAPITAL_USDC
            self._set_state("cash_quote", initial_capital)
            self._set_state("starting_capital_quote", initial_capital)
            self._set_state("paper_mode", "LOCAL_SIMULATION_ONLY")
            self._set_state("initial_eur_to_quote_rate", rate)
            self._set_state("current_news_score", 0.0)
            self._set_state("consecutive_losses_today", 0)
            self.db.commit()

    def _eur_to_quote_rate(self) -> float:
        return self.client.get_eur_usdc_rate() if Config.QUOTE_ASSET == "USDC" else 1.0

    @staticmethod
    def _utc_now(now: datetime | None = None) -> datetime:
        if now is None:
            return datetime.now(timezone.utc)
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)

    @staticmethod
    def _timestamp_text(value: object) -> str:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.isoformat()

    def _log_event(self, event_type: str, details: dict, now: datetime) -> None:
        self.db.execute(
            "INSERT INTO paper_events(observed_at_utc, event_type, details_json) VALUES(?, ?, ?)",
            (now.isoformat(), event_type, json.dumps(details, ensure_ascii=True, default=str)),
        )

    def _record_news(self, articles: list[dict]) -> None:
        for article in articles:
            article_id = article.get("article_id", "") or ""
            url = article.get("url", "") or ""
            title = article.get("title", "") or ""
            source = article.get("source", "unknown") or "unknown"
            published = str(article.get("published_at_utc", ""))
            fingerprint = hashlib.sha256(
                "|".join((source, article_id, url, title, published)).encode("utf-8")
            ).hexdigest()
            self.db.execute(
                """INSERT OR IGNORE INTO paper_news
                (fingerprint, source, article_id, title, score, published_at_utc,
                 observed_at_utc, url, feed_url)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fingerprint,
                    source,
                    article_id,
                    title,
                    float(article.get("score", 0.0)),
                    published,
                    str(article.get("observed_at_utc", "")),
                    url,
                    article.get("feed_url", "") or "",
                ),
            )

    def _poll_news(self, now: datetime, news_poll_seconds: float) -> tuple[float, int]:
        last_poll = float(self._get_state("last_news_poll_epoch", "0") or 0)
        if news_poll_seconds > 0.0 and (now.timestamp() - last_poll < news_poll_seconds):
            return float(self._get_state("current_news_score", "0") or 0), 0

        report = self.sentiment.analyze_live_news()
        articles = report.get("articles", [])
        self._record_news(articles)
        self._set_state("last_news_poll_epoch", now.timestamp())
        self._set_state("current_news_score", report.get("avg_score", 0.0))
        self._log_event(
            "NEWS_POLL",
            {
                "score": report.get("avg_score", 0.0),
                "regime": report.get("regime", "NEUTRAL"),
                "articles": len(articles),
            },
            now,
        )
        return float(report.get("avg_score", 0.0)), len(articles)

    def _record_price(self, ticker: dict, rate: float, now: datetime) -> None:
        self.db.execute(
            """INSERT INTO paper_prices
            (observed_at_utc, symbol, bid, ask, last, eur_to_quote_rate)
            VALUES(?, ?, ?, ?, ?, ?)""",
            (
                now.isoformat(), ticker["symbol"], ticker["bid"], ticker["ask"],
                ticker["last"], rate,
            ),
        )

    def _record_new_candles(self, candles: pd.DataFrame) -> None:
        last_saved = self._get_state("last_candle_saved_utc")
        if last_saved:
            last_saved_timestamp = pd.Timestamp(last_saved).tz_convert(None)
            candles = candles[candles["timestamp"] > last_saved_timestamp]
        for _, row in candles.iterrows():
            timestamp = self._timestamp_text(row["timestamp"])
            self.db.execute(
                """INSERT OR IGNORE INTO paper_candles
                (timestamp_utc, open, high, low, close, volume)
                VALUES(?, ?, ?, ?, ?, ?)""",
                (timestamp, row["open"], row["high"], row["low"], row["close"], row["volume"]),
            )
            self._set_state("last_candle_saved_utc", timestamp)

    def _open_order(self):
        return self.db.execute(
            "SELECT * FROM paper_orders WHERE status = 'OPEN' ORDER BY id DESC LIMIT 1"
        ).fetchone()

    def _close_open_order(self, order, bid: float, now: datetime, reason: str) -> dict:
        exit_reference = bid
        exit_fill = bid * (1 - Config.SLIPPAGE_PCT)
        qty = float(order["qty"])
        exit_fee = qty * exit_fill * Config.FEE_RATE
        entry_fee = float(order["entry_fee_quote"])
        entry_fill = float(order["entry_fill"])
        pnl_quote = qty * (exit_fill - entry_fill) - entry_fee - exit_fee
        cash_quote = float(self._get_state("cash_quote")) + qty * exit_fill - exit_fee
        rate = self._eur_to_quote_rate()
        self._set_state("cash_quote", cash_quote)
        self.db.execute(
            """UPDATE paper_orders SET status = 'CLOSED', exit_time_utc = ?,
            exit_reference = ?, exit_fill = ?, exit_slippage_bps = ?,
            exit_fee_quote = ?, pnl_quote = ?, pnl_eur = ?, exit_reason = ?
            WHERE id = ?""",
            (
                now.isoformat(), exit_reference, exit_fill,
                (exit_reference - exit_fill) / exit_reference * 10000,
                exit_fee, pnl_quote, pnl_quote / rate, reason, order["id"],
            ),
        )
        local_date = now.astimezone(ZoneInfo(Config.LOCAL_TIMEZONE)).date().isoformat()
        if self._get_state("daily_session_date") != local_date:
            consecutive_losses = 0
            self._set_state("daily_session_date", local_date)
        else:
            consecutive_losses = int(self._get_state("consecutive_losses_today", "0") or 0)
        consecutive_losses = consecutive_losses + 1 if pnl_quote < 0 else 0
        self._set_state("consecutive_losses_today", consecutive_losses)

        # Online Reinforcement Learning Feedback
        if Config.RL_ACTIVE and self.rl_engine is not None:
            setup = str(order["signal_setup"])
            fng = float(order["fng_value"]) if "fng_value" in order.keys() else 50.0
            atr_pct = float(order["atr_pct"]) if "atr_pct" in order.keys() else 1.0
            slope = float(order["ema200_gap_pct"]) if "ema200_gap_pct" in order.keys() else 0.0
            state_key = self.rl_engine.get_state_key(setup, fng, atr_pct, slope, 50.0)
            is_win = (pnl_quote > 0)
            ret_pct = (pnl_quote / (entry_fill * qty)) * 100.0 if (entry_fill * qty) > 0 else 0.0

            # Institutional R-Multiple Normalization
            stop_price = float(order["stop_price"]) if "stop_price" in order.keys() and order["stop_price"] else entry_fill * (1 - Config.DEFAULT_SL_PCT)
            risk_per_unit = max(0.0001, abs(entry_fill - stop_price))
            r_multiple = (exit_fill - entry_fill) / risk_per_unit

            self.rl_engine.record_feedback(state_key, is_win, ret_pct, r_multiple=r_multiple, timestamp=now.isoformat())
            self.rl_engine.save_memory()

        details = {"order_id": order["id"], "reason": reason, "pnl_quote": pnl_quote, "pnl_eur": pnl_quote / rate}
        self._log_event("PAPER_EXIT", details, now)
        if pnl_quote < 0 and consecutive_losses >= Config.PAPER_MAX_CONSECUTIVE_LOSSES:
            self._set_state("daily_loss_locked", "1")
            self._log_event(
                "CONSECUTIVE_LOSS_LOCK",
                {"consecutive_losses": consecutive_losses, "limit": Config.PAPER_MAX_CONSECUTIVE_LOSSES},
                now,
            )
        details["consecutive_losses_today"] = consecutive_losses
        details["learning_review"] = self._record_learning_review(now)
        return details

    def _record_learning_review(self, now: datetime) -> dict:
        trades = pd.read_sql_query(
            """SELECT signal_setup, pnl_quote, entry_slippage_bps, exit_slippage_bps,
            news_score, atr_pct, fng_value, ema200_gap_pct FROM paper_orders
            WHERE status = 'CLOSED' ORDER BY exit_time_utc""",
            self.db,
        )
        count = len(trades)
        report = {
            "closed_trades": count,
            "min_trades_for_review": Config.PAPER_REVIEW_MIN_TRADES,
            "status": "insufficient_sample" if count < Config.PAPER_REVIEW_MIN_TRADES else "review_only",
            "automatic_strategy_changes": False,
        }
        if count:
            wins = trades.loc[trades["pnl_quote"] > 0, "pnl_quote"]
            losses = trades.loc[trades["pnl_quote"] <= 0, "pnl_quote"]
            gross_loss = abs(float(losses.sum()))
            loss_streak = max_loss_streak = 0
            for pnl in trades["pnl_quote"]:
                loss_streak = loss_streak + 1 if pnl < 0 else 0
                max_loss_streak = max(max_loss_streak, loss_streak)
            report.update({
                "win_rate_pct": float(len(wins) / count * 100),
                "profit_factor": float(wins.sum()) / gross_loss if gross_loss else None,
                "mean_pnl_quote": float(trades["pnl_quote"].mean()),
                "mean_entry_slippage_bps": float(trades["entry_slippage_bps"].mean()),
                "mean_exit_slippage_bps": float(trades["exit_slippage_bps"].mean()),
                "max_consecutive_losses": max_loss_streak,
                "setup_summary": trades.groupby("signal_setup")["pnl_quote"].agg(["count", "mean", "sum"]).round(4).to_dict("index"),
            })
        if count >= Config.PAPER_REVIEW_MIN_TRADES:
            trades["news_regime"] = pd.cut(
                trades["news_score"], [-float("inf"), -0.15, 0.15, float("inf")],
                labels=["risk_off", "neutral", "risk_on"],
            )
            trades["volatility_regime"] = pd.cut(
                trades["atr_pct"], [-float("inf"), Config.VOLATILITY_TARGET_PCT, float("inf")],
                labels=["below_target", "above_target"],
            )
            trades["trend_regime"] = pd.cut(
                trades["ema200_gap_pct"], [-float("inf"), 0, float("inf")],
                labels=["below_ema200", "above_ema200"],
                include_lowest=True,
            )
            weak_groups = []
            for dimension in ("signal_setup", "news_regime", "volatility_regime", "trend_regime"):
                summaries = []
                for label, group in trades.groupby(dimension, observed=True):
                    group_wins = group.loc[group["pnl_quote"] > 0, "pnl_quote"]
                    group_losses = abs(float(group.loc[group["pnl_quote"] <= 0, "pnl_quote"].sum()))
                    group_pf = float(group_wins.sum()) / group_losses if group_losses else None
                    summary = {"group": str(label), "trades": len(group), "mean_pnl_quote": float(group["pnl_quote"].mean()), "profit_factor": group_pf}
                    summaries.append(summary)
                    if len(group) >= 10 and group_pf is not None and group_pf < 1:
                        weak_groups.append({"dimension": dimension, **summary})
                report[f"performance_by_{dimension}"] = summaries
            report["weak_groups_to_review"] = weak_groups
            report["recommendation"] = "Review weak groups and execution costs manually; no strategy parameters were changed."
        self.db.execute(
            "INSERT INTO paper_reviews(created_at_utc, closed_trade_count, report_json) VALUES(?, ?, ?)",
            (now.isoformat(), count, json.dumps(report, ensure_ascii=True, default=str)),
        )
        self._log_event("LEARNING_REVIEW", report, now)
        return report

    def _paper_position_size(
        self,
        entry_price: float,
        atr_pct: float,
        cash_quote: float,
        rate: float,
        sizing_multiplier: float = 1.0
    ) -> tuple[float, float] | None:
        stop_distance = entry_price * Config.DEFAULT_SL_PCT
        risk_scale = 1.0
        if math.isfinite(atr_pct) and atr_pct > 0:
            risk_scale = min(1.0, Config.VOLATILITY_TARGET_PCT / atr_pct)
        # RL Dynamic Position Sizing multiplier
        effective_multiplier = max(0.2, min(2.0, sizing_multiplier)) if Config.RL_DYNAMIC_SIZING else 1.0
        max_exchange_notional = self.rules["max_order_amount"] or (Config.MAX_ORDER_NOTIONAL_EUR * rate)
        min_notional = max(Config.MIN_ORDER_NOTIONAL_EUR * rate, self.rules["min_order_amount"])
        qty_step = float(self.rules["qty_step"])
        qty_decimals = max(0, int(round(-math.log10(qty_step))))

        if getattr(Config, "RL_COMPOUNDING", True):
            base_alloc = getattr(Config, "BASE_POSITION_PCT", 0.75)
            target_notional = cash_quote * base_alloc * effective_multiplier
            max_notional = min(cash_quote * 0.98, max_exchange_notional)
            target_notional = min(target_notional, max_notional)
            qty = math.floor((target_notional / entry_price) / qty_step) * qty_step
        else:
            risk_amount = cash_quote * Config.RISK_PER_TRADE * risk_scale * effective_multiplier
            max_notional = min(Config.MAX_ORDER_NOTIONAL_EUR * rate, max_exchange_notional)
            qty = math.floor((risk_amount / stop_distance) / qty_step) * qty_step
            qty = min(qty, math.floor((max_notional / entry_price) / qty_step) * qty_step)

        qty = round(qty, qty_decimals)
        notional = qty * entry_price
        fee = notional * Config.FEE_RATE
        if qty < float(self.rules["min_qty"]) or notional < min_notional or notional + fee > cash_quote:
            return None
        return qty, notional

    def _open_paper_order(
        self,
        signal: pd.Series,
        ticker: dict,
        rate: float,
        news_score: float,
        now: datetime,
    ) -> dict | None:
        reference = (ticker["bid"] + ticker["ask"]) / 2
        fill = ticker["ask"] * (1 + Config.SLIPPAGE_PCT)
        cash_quote = float(self._get_state("cash_quote"))
        signal_time = self._timestamp_text(signal["timestamp"])

        # Spread spike protection on entry
        spread = ticker["ask"] - ticker["bid"]
        spread_bps = (spread / ticker["ask"] * 10000.0) if ticker["ask"] > 0 else 0.0
        if spread_bps > getattr(Config, "MAX_SPREAD_BPS", 15.0):
            self._log_event(
                "PAPER_ORDER_SKIPPED",
                {"signal_time_utc": signal_time, "reason": f"spread_too_wide ({spread_bps:.1f} bps > {Config.MAX_SPREAD_BPS} bps)"},
                now,
            )
            return None

        sizing_mult = float(signal.get("rl_sizing_multiplier", 1.0))
        size = self._paper_position_size(fill, float(signal["atr_pct"]), cash_quote, rate, sizing_multiplier=sizing_mult)
        if size is None:
            self._log_event(
                "PAPER_ORDER_SKIPPED",
                {"signal_time_utc": signal_time, "reason": "risk_size_below_minimum_or_cash"},
                now,
            )
            return None

        qty, notional = size
        entry_fee = notional * Config.FEE_RATE
        setup_name = str(signal.get("setup_type", ""))
        if setup_name == "MEAN_REVERSION_BOUNCE":
            stop_price = fill * (1 - getattr(Config, "MEAN_REV_SL_PCT", 0.015))
            target_price = fill * (1 + getattr(Config, "MEAN_REV_TP_PCT", 0.024))
        else:
            stop_price = fill * (1 - Config.DEFAULT_SL_PCT)
            target_price = fill * (1 + Config.DEFAULT_TP_PCT)
        cash_after = cash_quote - notional - entry_fee
        self._set_state("cash_quote", cash_after)
        self._set_state("paper_trailing_tier", "0")
        cursor = self.db.execute(
            """INSERT OR IGNORE INTO paper_orders
            (signal_time_utc, signal_setup, symbol, side, status, qty,
             entry_time_utc, entry_reference, entry_fill, entry_slippage_bps,
             entry_fee_quote, notional_quote, notional_eur, stop_price, target_price,
             news_score, atr_pct, fng_value, ema200_gap_pct)
            VALUES(?, ?, ?, 'LONG', 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal_time,
                str(signal["setup_type"]),
                Config.SYMBOL,
                qty,
                now.isoformat(),
                reference,
                fill,
                (fill - reference) / reference * 10000,
                entry_fee,
                notional,
                notional / rate,
                stop_price,
                target_price,
                news_score,
                float(signal.get("atr_pct", 0.0)),
                float(signal.get("fng_value", 50.0)),
                (float(signal.get("close", fill)) / float(signal.get("ema_200", fill)) - 1) * 100
                if float(signal.get("ema_200", 0) or 0) > 0 else 0.0,
            ),
        )
        if cursor.rowcount == 0:
            self._set_state("cash_quote", cash_quote)
            return None
        details = {
            "order_id": cursor.lastrowid,
            "setup": str(signal["setup_type"]),
            "rl_state": str(signal.get("rl_state", "")),
            "qty": qty,
            "sizing_multiplier": sizing_mult,
            "entry_fill": fill,
            "notional_quote": notional,
            "notional_eur": notional / rate,
            "stop_price": stop_price,
            "target_price": target_price,
            "entry_slippage_bps": (fill - reference) / reference * 10000,
        }
        self._log_event("PAPER_ENTRY", details, now)
        return details

    def get_dynamic_daily_loss_limit(self, daily_start_equity: float) -> float:
        """
        Calcola dinamicamente il limite di perdita giornaliera:
        - Percentuale fissa sul capitale di inizio giornata (default 3.5%, es. 35 USDC su 1000 USDC)
        - Vincolato a un minimo di sicurezza e al limite base configurato.
        """
        pct_limit = daily_start_equity * Config.PAPER_DAILY_LOSS_LIMIT_PCT
        return max(Config.PAPER_DAILY_LOSS_LIMIT_MIN_USDC, pct_limit)

    def _equity_and_daily_lock(
        self, ticker: dict, rate: float, now: datetime, record_snapshot: bool = True
    ) -> tuple[float, float, float, int, bool]:
        cash_quote = float(self._get_state("cash_quote"))
        order = self._open_order()
        position_qty = float(order["qty"]) if order else 0.0
        equity_quote = cash_quote + position_qty * ticker["bid"] * (1 - Config.FEE_RATE)
        local_date = now.astimezone(ZoneInfo(Config.LOCAL_TIMEZONE)).date().isoformat()
        stored_date = self._get_state("daily_session_date")
        if local_date != stored_date:
            self._set_state("daily_session_date", local_date)
            self._set_state("daily_start_equity_quote", equity_quote)
            self._set_state("daily_loss_locked", "0")
            self._set_state("consecutive_losses_today", "0")
            if self.rl_engine is not None:
                self.rl_engine.apply_time_decay(now)
                self.rl_engine.save_memory()
        daily_start = float(self._get_state("daily_start_equity_quote", str(equity_quote)))
        daily_loss_quote = max(0.0, daily_start - equity_quote)
        daily_pnl_pct = ((equity_quote / daily_start) - 1) if daily_start > 0 else 0.0
        consecutive_losses = int(self._get_state("consecutive_losses_today", "0") or 0)
        locked = self._get_state("daily_loss_locked") == "1"
        effective_daily_loss_limit = self.get_dynamic_daily_loss_limit(daily_start)
        if daily_loss_quote >= effective_daily_loss_limit and not locked:
            locked = True
            self._set_state("daily_loss_locked", "1")
            self._log_event(
                "DAILY_LOSS_LOCK",
                {"daily_loss_quote": daily_loss_quote, "limit_usdc": effective_daily_loss_limit},
                now,
            )
        if consecutive_losses >= Config.PAPER_MAX_CONSECUTIVE_LOSSES and not locked:
            locked = True
            self._set_state("daily_loss_locked", "1")
            self._log_event(
                "CONSECUTIVE_LOSS_LOCK",
                {"consecutive_losses": consecutive_losses, "limit": Config.PAPER_MAX_CONSECUTIVE_LOSSES},
                now,
            )
        if record_snapshot:
            self.db.execute(
                """INSERT INTO paper_equity
                (observed_at_utc, local_session_date, cash_quote, position_qty,
                 equity_quote, equity_eur, daily_pnl_pct, daily_loss_quote,
                 consecutive_losses, eur_to_quote_rate)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now.isoformat(), local_date, cash_quote, position_qty,
                    equity_quote, equity_quote / rate, daily_pnl_pct,
                    daily_loss_quote, consecutive_losses, rate,
                ),
            )
        return equity_quote, daily_loss_quote, daily_pnl_pct, consecutive_losses, locked

    def step(
        self,
        news_poll_seconds: float = Config.PAPER_NEWS_POLL_SECONDS,
        now: datetime | None = None,
        force_news_poll: bool = False
    ) -> dict:
        with self.db_lock:
            now = self._utc_now(now)
            ticker = self.client.get_spot_ticker()
            rate = self._eur_to_quote_rate()
            self._record_price(ticker, rate, now)
            effective_poll_seconds = 0.0 if force_news_poll else news_poll_seconds
            news_score, article_count = self._poll_news(now, effective_poll_seconds)

            interval = pd.Timedelta(minutes=DataManager.TIMEFRAMES[str(Config.TIMEFRAME)][1] / 60000)
            try:
                candles = self.client.get_recent_klines(symbol=Config.SYMBOL, limit=1000)
            except Exception as exc:
                self._log_event("KLINE_ERROR", {"error": str(exc)}, now)
                candles = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            now_naive_utc = pd.Timestamp(now).tz_convert(None)
            if candles.empty:
                closed = candles
            else:
                closed = candles[candles["timestamp"] + interval <= now_naive_utc].reset_index(drop=True)
            self._record_new_candles(closed)

            last_processed = self._get_state("last_processed_bar_utc")
            new_bar = not closed.empty and self._timestamp_text(closed["timestamp"].iloc[-1]) != last_processed
            has_gap = False
            if new_bar and last_processed:
                previous_time = pd.Timestamp(last_processed).tz_convert(None)
                current_time = pd.Timestamp(closed["timestamp"].iloc[-1])
                has_gap = current_time - previous_time > interval

            equity_quote, daily_loss_quote, daily_pnl_pct, consecutive_losses, daily_locked = (
                self._equity_and_daily_lock(ticker, rate, now, record_snapshot=False)
            )
            open_order = self._open_order()
            exit_details = None
            if open_order:
                entry_time = pd.Timestamp(open_order["entry_time_utc"]).to_pydatetime()
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                elapsed = now - entry_time.astimezone(timezone.utc)
                bid = ticker["bid"]
                entry_fill = float(open_order["entry_fill"])
                current_gain = (bid - entry_fill) / entry_fill
                trailing_tier = int(self._get_state("paper_trailing_tier", "0") or 0)

                # Dynamic multi-tier trailing stop
                if trailing_tier == 0 and current_gain >= Config.TRAIL_TIER1_TRIGGER:
                    new_stop = entry_fill * (1.0 + Config.TRAIL_TIER1_LOCK)
                    self.db.execute("UPDATE paper_orders SET stop_price = ? WHERE id = ?", (new_stop, open_order["id"]))
                    self._set_state("paper_trailing_tier", "1")
                    trailing_tier = 1
                    open_order = self._open_order()
                elif trailing_tier == 1 and current_gain >= Config.TRAIL_TIER2_TRIGGER:
                    new_stop = entry_fill * (1.0 + Config.TRAIL_TIER2_LOCK)
                    self.db.execute("UPDATE paper_orders SET stop_price = ? WHERE id = ?", (new_stop, open_order["id"]))
                    self._set_state("paper_trailing_tier", "2")
                    trailing_tier = 2
                    open_order = self._open_order()

                # Proactive News Defense: If negative news hits and position has modest gain, lock breakeven
                if trailing_tier == 0 and news_score <= -0.15 and current_gain >= 0.005:
                    be_stop = entry_fill * 1.0020
                    if be_stop > float(open_order["stop_price"]):
                        self.db.execute("UPDATE paper_orders SET stop_price = ? WHERE id = ?", (be_stop, open_order["id"]))
                        self._log_event("NEWS_DEFENSE_BREAKEVEN_LOCK", {"gain_pct": current_gain * 100, "news_score": news_score}, now)
                        open_order = self._open_order()

                daily_start = float(self._get_state("daily_start_equity_quote", str(equity_quote)))
                effective_daily_loss_limit = self.get_dynamic_daily_loss_limit(daily_start)
                if daily_loss_quote >= effective_daily_loss_limit:
                    exit_reason = "DAILY_LOSS_LIMIT"
                elif consecutive_losses >= Config.PAPER_MAX_CONSECUTIVE_LOSSES:
                    exit_reason = "CONSECUTIVE_LOSS_LIMIT"
                elif has_gap:
                    exit_reason = "DATA_GAP_EXIT"
                elif bid <= float(open_order["stop_price"]):
                    exit_reason = "TRAIL_TIER2" if trailing_tier == 2 else ("TRAIL_TIER1" if trailing_tier == 1 else "STOP_LOSS")
                elif bid >= float(open_order["target_price"]):
                    exit_reason = "TAKE_PROFIT"
                elif news_score <= -0.30 and (bid - entry_fill) / entry_fill < 0.005:
                    exit_reason = "CRITICAL_NEWS_DEFENSE_EXIT"
                elif elapsed >= pd.Timedelta(hours=getattr(Config, "MEAN_REV_MAX_HOLDING_HOURS", 12.0) if open_order["signal_setup"] == "MEAN_REVERSION_BOUNCE" else Config.MAX_HOLDING_HOURS):
                    exit_reason = "TIME_EXIT"
                else:
                    exit_reason = ""
                if exit_reason:
                    exit_details = self._close_open_order(open_order, bid, now, exit_reason)

            entry_details = None
            bar_signal = 0
            if new_bar:
                latest_bar = closed.iloc[-1]
                bar_time = self._timestamp_text(latest_bar["timestamp"])
                if has_gap:
                    self._log_event("SIGNAL_SKIPPED", {"bar_time_utc": bar_time, "reason": "data_gap"}, now)
                elif exit_details is None and not self._open_order():
                    fng = self.sentiment.fetch_fear_and_greed_history()
                    funding_rate = self.sentiment.fetch_market_funding_rate() if Config.FUNDING_RATE_FILTER else 0.0
                    signals = self.strategy.generate_signals(
                        closed,
                        fng_df=fng,
                        live_news_score=news_score,
                        live_funding_rate=funding_rate,
                        rl_engine=self.rl_engine
                    )
                    signal = signals.iloc[-1]
                    bar_signal = int(signal["signal"])
                    setup_type = str(signal.get("setup_type", ""))
                    if bar_signal and daily_locked:
                        self._log_event("SIGNAL_SKIPPED", {"bar_time_utc": bar_time, "reason": "daily_loss_limit", "setup": setup_type}, now)
                    elif bar_signal:
                        entry_details = self._open_paper_order(signal, ticker, rate, news_score, now)
                    elif "PRUNED_BY_RL" in setup_type:
                        self._log_event("SIGNAL_PRUNED", {"bar_time_utc": bar_time, "reason": setup_type, "setup": setup_type}, now)
                    elif "BLOCKED_BY" in setup_type:
                        self._log_event("SIGNAL_BLOCKED", {"bar_time_utc": bar_time, "reason": setup_type, "setup": setup_type}, now)
                self._set_state("last_processed_bar_utc", bar_time)

            equity_quote, daily_loss_quote, daily_pnl_pct, consecutive_losses, daily_locked = (
                self._equity_and_daily_lock(ticker, rate, now)
            )
            self._set_state("last_step_utc", now.isoformat())
            self.db.commit()
            result = {
                "observed_at_utc": now.isoformat(),
                "symbol": Config.SYMBOL,
                "bid": ticker["bid"],
                "ask": ticker["ask"],
                "last": ticker["last"],
                "high24h": ticker.get("high24h", 0.0),
                "low24h": ticker.get("low24h", 0.0),
                "change24h_pct": ticker.get("change24h_pct", 0.0),
                "volume24h": ticker.get("volume24h", 0.0),
                "turnover24h": ticker.get("turnover24h", 0.0),
                "news_score": news_score,
                "news_articles": article_count,
                "new_closed_bar": new_bar,
                "bar_signal": bar_signal,
                "entry": entry_details,
                "exit": exit_details,
                "equity_quote": equity_quote,
                "equity_eur": equity_quote / rate,
                "daily_loss_quote": daily_loss_quote,
                "daily_pnl_pct": daily_pnl_pct,
                "consecutive_losses_today": consecutive_losses,
                "daily_loss_locked": daily_locked,
            }
            return result

    def close(self) -> None:
        self.db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trading persistente Bybit EU; non invia ordini reali.")
    parser.add_argument("--once", action="store_true", help="Esegue un solo ciclo e termina.")
    parser.add_argument("--poll-seconds", type=float, default=Config.PAPER_POLL_SECONDS)
    parser.add_argument("--news-poll-seconds", type=float, default=Config.PAPER_NEWS_POLL_SECONDS)
    parser.add_argument("--db", default=Config.PAPER_DB_PATH, help="Percorso database SQLite.")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.news_poll_seconds <= 0:
        parser.error("Gli intervalli di polling devono essere maggiori di zero.")

    engine = PaperTradingEngine(database_path=args.db)
    simulated_cash = float(engine._get_state("cash_quote"))
    print(
        f"PAPER ONLY | feed pubblico {engine.client.session.endpoint} | {Config.SYMBOL} | "
        f"saldo simulato {simulated_cash:.2f} USDC | DB: {args.db} | "
        "chiavi API non usate; ordini reali disabilitati"
    )
    try:
        while True:
            try:
                result = engine.step(news_poll_seconds=args.news_poll_seconds)
                if args.once or result["new_closed_bar"] or result["entry"] or result["exit"]:
                    print(json.dumps(result, ensure_ascii=True, default=str))
            except Exception as exc:
                now = datetime.now(timezone.utc)
                engine._log_event("ERROR", {"error": str(exc)}, now)
                engine.db.commit()
                print(f"[PAPER ERROR] {exc}")
            if args.once:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Paper loop interrotto dall'utente.")
    finally:
        engine.close()


if __name__ == "__main__":
    main()