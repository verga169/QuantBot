#!/usr/bin/env python3
"""
BYBIT QUANT BOT - TERMINAL DASHBOARD (TUI)
Interfaccia Terminale Interattiva Avanzata per Trading Intraday con Reinforcement Learning
Supporta:
- Monitoraggio live del budget simulato (fondi realistici e persistenti)
- Monitoraggio della posizione aperta, trailing stop dinamico, PnL non realizzato
- Market Scanner 1H live con indicatori tecnici (EMA 50/100/200, RSI, ATR, Squeeze)
- Sentiment Intelligence: 10 feed RSS, ultime notizie in tempo reale, Fear & Greed Index
- Motore di Reinforcement Learning con classificazione vettoriale del regime live
- Visualizzazione e modifica interattiva delle credenziali API Bybit EU (.env)
- Registro eventi esteso senza troncamenti con motivazioni in evidenza
- Avvio e stop del bot tramite tasti da tastiera con controlli di sicurezza
"""

import os
import sys
import time
import json
import math
import select
import termios
import tty
import threading
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich.markup import escape

from config import Config
from exchange_client import BybitClient
from paper_trading import PaperTradingEngine
from rl_experience_engine import RLExperienceEngine


# Dizionario font Unicode a blocchi per rendering Big Price Ticker (3 righe fisse, 3 char/cifra)
BIG_DIGITS: Dict[str, List[str]] = {
    "0": ["█▀█", "█ █", "█▄█"],
    "1": [" █ ", " █ ", " █ "],
    "2": ["█▀█", " ▄▀", "█▄▄"],
    "3": ["▀▀█", "▀▀█", "▄▄█"],
    "4": ["█ █", "█▄█", "  █"],
    "5": ["█▀▀", "▀▀█", "▄▄█"],
    "6": ["█▀▀", "█▀█", "█▄█"],
    "7": ["▀▀█", "  █", "  █"],
    "8": ["█▀█", "█▀█", "█▄█"],
    "9": ["█▀█", "█▄█", "  █"],
    "$": ["▄█▀", " █ ", "▀█▄"],
    "B": ["█▀▄", "█▀▄", "▀▀ "],
    "T": ["▀█▀", " █ ", " ▀ "],
    "C": ["█▀█", "█  ", "█▄█"],
    ",": [" ", " ", "▄"],
    ".": [" ", " ", "▄"],
    "+": ["   ", " █ ", "   "],
    "-": ["   ", "▀▀▀", "   "],
    " ": ["   ", "   ", "   "],
}


def _render_big_digits(text: str) -> List[str]:
    """Renderizza una stringa numerica in caratteri grandi a 3 righe Unicode con allineamento rigoroso."""
    lines = ["", "", ""]
    for ch in text:
        b = BIG_DIGITS.get(ch, ["   "] * 3)
        for i in range(3):
            lines[i] += b[i] + " "
    return [l[:-1] for l in lines] if (lines and lines[0].endswith(" ")) else lines


class TradingAppTUI:
    def __init__(self, db_path: Optional[str] = None):
        self.console = Console()
        self.db_path = db_path or Config.PAPER_DB_PATH
        self.engine = PaperTradingEngine(database_path=self.db_path)
        
        # Stato applicazione
        self.is_running = True
        self.trading_active = True  # True: esegue ordini simulati | False: monitoraggio in pausa
        self.force_refresh_event = threading.Event()
        self.force_full_refresh = False
        self._last_funding_poll = 0.0
        self._last_fng_poll = 0.0
        self.lock = threading.Lock()
        self.start_time = datetime.now()
        
        # Cache dati per il rendering
        self.last_ticker: Dict[str, Any] = {
            "bid": 0.0, "ask": 0.0, "last": 0.0,
            "high24h": 0.0, "low24h": 0.0, "change24h_pct": 0.0,
            "volume24h": 0.0, "turnover24h": 0.0
        }
        self.last_step_data: Dict[str, Any] = {}
        self.fng_data: Dict[str, Any] = {"value": 50, "classification": "Neutral"}
        self.last_indicators: Dict[str, Any] = {}
        self.cached_chart_candles: List[Dict[str, Any]] = []
        self.last_news_item: Dict[str, Any] = {}
        self.last_funding_rate: float = 0.0001
        self.status_message = "Dashboard inizializzata con successo."
        self.status_timestamp = datetime.now()
        
        # Imposta titolo della finestra e icona nel pannello di sistema
        self._set_window_title("Quant Bot")
        
        # Carica dati iniziali in background per non rallentare l'avvio
        threading.Thread(target=self._initial_background_fetch, daemon=True).start()

    @staticmethod
    def _set_window_title(title: str = "Quant Bot") -> None:
        """Imposta il titolo della finestra esclusivamente per questo terminale via sequenza standard OSC."""
        try:
            sys.stdout.write(f"\033]0;{title}\007")
            sys.stdout.flush()
        except Exception:
            pass

    def _initial_background_fetch(self) -> None:
        """Caricamento asincrono iniziale di Fear & Greed, ultime news e indicatori tecnici."""
        try:
            fng_df = self.engine.sentiment.fetch_fear_and_greed_history()
            if not fng_df.empty:
                latest = fng_df.iloc[-1]
                with self.lock:
                    self.fng_data = {
                        "value": int(latest.get("fng_value", 50)),
                        "classification": str(latest.get("fng_classification", "Neutral"))
                    }
        except Exception:
            pass

        try:
            fr = self.engine.sentiment.fetch_market_funding_rate()
            with self.lock:
                self.last_funding_rate = fr
        except Exception:
            pass

        self._refresh_cached_news()
        self._refresh_cached_indicators()

    def _refresh_cached_news(self) -> None:
        """Recupera l'ultima notizia salvata dal database SQLite."""
        try:
            with self.engine.db_lock:
                row = self.engine.db.execute(
                    "SELECT source, title, score, published_at_utc FROM paper_news ORDER BY published_at_utc DESC LIMIT 1"
                ).fetchone()
            if row:
                with self.lock:
                    self.last_news_item = {
                        "source": str(row["source"]),
                        "title": str(row["title"]),
                        "score": float(row["score"]),
                        "published_at_utc": str(row["published_at_utc"])
                    }
        except Exception:
            pass

    def _refresh_cached_indicators(self) -> None:
        """Calcola e aggiorna gli indicatori tecnici quantitativi sulle candele storiche chiuse."""
        try:
            with self.engine.db_lock:
                df = pd.read_sql_query(
                    "SELECT timestamp_utc as timestamp, open, high, low, close, volume FROM paper_candles ORDER BY timestamp_utc ASC",
                    self.engine.db
                )
            if df.empty or len(df) < 30:
                return
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            ind_df = self.engine.strategy.compute_indicators(df)
            if ind_df.empty:
                return
            last_bar = ind_df.iloc[-1]
            prev_bar = ind_df.iloc[-2] if len(ind_df) >= 2 else last_bar
            with self.lock:
                rolling_low_3 = float(ind_df["low"].rolling(3).min().iloc[-1]) if len(ind_df) >= 3 else float(last_bar.get("low", 0.0))
                rolling_dist_3 = float(ind_df["dist_ema20_pct"].rolling(3).min().iloc[-1]) if ("dist_ema20_pct" in ind_df.columns and len(ind_df) >= 3) else 0.0
                rolling_rsi_3 = float(ind_df["rsi"].rolling(3).min().iloc[-1]) if len(ind_df) >= 3 else float(last_bar.get("rsi", 50.0))
                rolling_rsi7_3 = float(ind_df["rsi_7"].rolling(3).min().iloc[-1]) if ("rsi_7" in ind_df.columns and len(ind_df) >= 3) else float(last_bar.get("rsi", 50.0))
                prev_ema50_3 = float(ind_df["ema_50"].shift(3).iloc[-1]) if len(ind_df) >= 4 else float(last_bar.get("ema_50", 0.0))

                self.last_indicators = {
                    "timestamp": last_bar["timestamp"],
                    "open": float(last_bar.get("open", 0.0)),
                    "high": float(last_bar.get("high", 0.0)),
                    "low": float(last_bar.get("low", 0.0)),
                    "close": float(last_bar.get("close", 0.0)),
                    "ema_20": float(last_bar.get("ema_20", 0.0)),
                    "ema_50": float(last_bar.get("ema_50", 0.0)),
                    "ema_100": float(last_bar.get("ema_100", 0.0)),
                    "ema_200": float(last_bar.get("ema_200", 0.0)),
                    "rsi": float(last_bar.get("rsi", 50.0)),
                    "rsi_7": float(last_bar.get("rsi_7", last_bar.get("rsi", 50.0))),
                    "adx": float(last_bar.get("adx", 25.0)),
                    "chop": float(last_bar.get("chop", 50.0)),
                    "atr": float(last_bar.get("atr", 0.0)),
                    "atr_pct": float(last_bar.get("atr_pct", 0.0)),
                    "bb_mid": float(last_bar.get("bb_mid", 0.0)),
                    "bb_upper": float(last_bar.get("bb_upper", 0.0)),
                    "bb_lower": float(last_bar.get("bb_lower", 0.0)),
                    "bb_lower_165": float(last_bar.get("bb_lower_165", last_bar.get("bb_lower", 0.0))),
                    "bb_lower_18": float(last_bar.get("bb_lower_18", last_bar.get("bb_lower", 0.0))),
                    "bb_width": float(last_bar.get("bb_width", 0.0)),
                    "dist_ema20_pct": float(last_bar.get("dist_ema20_pct", 0.0)),
                    "close_location": float(last_bar.get("close_location", 0.5)),
                    "squeeze": bool(last_bar.get("squeeze", False)),
                    "prev_squeeze": bool(prev_bar.get("squeeze", False)),
                    "high_12": float(last_bar.get("high_12", 0.0)),
                    "prev_high_12": float(prev_bar.get("high_12", 0.0)),
                    "high_20": float(last_bar.get("high_20", 0.0)),
                    "prev_high_20": float(prev_bar.get("high_20", 0.0)),
                    "low_20": float(last_bar.get("low_20", 0.0)),
                    "prev_low_20": float(prev_bar.get("low_20", 0.0)),
                    "prev_low": float(prev_bar.get("low", 0.0)),
                    "prev_high": float(prev_bar.get("high", 0.0)),
                    "prev_rsi_7": float(prev_bar.get("rsi_7", 50.0)),
                    "prev_ema_50_3": prev_ema50_3,
                    "rolling_low_3": rolling_low_3,
                    "rolling_dist_ema20_3": rolling_dist_3,
                    "rolling_rsi_3": rolling_rsi_3,
                    "rolling_rsi7_3": rolling_rsi7_3,
                    "vol_ratio": float(last_bar.get("vol_ratio", 1.0)),
                    "ema_200_slope": float(last_bar.get("ema_200_slope", 0.0)),
                }
                chart_df = df.tail(130).copy()
                self.cached_chart_candles = chart_df.to_dict("records")
        except Exception:
            pass

    def _get_trade_stats(self) -> Dict[str, Any]:
        """Calcola le statistiche complessive e della sessione sui trade conclusi."""
        try:
            with self.engine.db_lock:
                row = self.engine.db.execute(
                    """SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN pnl_quote > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN pnl_quote <= 0 THEN 1 ELSE 0 END) as losses,
                        COALESCE(SUM(pnl_quote), 0.0) as sum_pnl
                    FROM paper_orders WHERE status = 'CLOSED'"""
                ).fetchone()
                total = int(row["total"] or 0)
                wins = int(row["wins"] or 0)
                losses = int(row["losses"] or 0)
                sum_pnl = float(row["sum_pnl"] or 0.0)
                wr = (wins / total * 100.0) if total > 0 else 0.0
                return {"total": total, "wins": wins, "losses": losses, "wr": wr, "sum_pnl": sum_pnl}
        except Exception:
            return {"total": 0, "wins": 0, "losses": 0, "wr": 0.0, "sum_pnl": 0.0}

    def _get_weekly_trades(self, days: int = 7) -> Dict[str, Any]:
        """
        Recupera le operazioni della settimana (ultimi 7 giorni o aperte)
        con dettagli formattati, durate, PnL e statistiche aggregate.
        """
        try:
            now_utc = datetime.now(timezone.utc)
            cutoff_utc = now_utc - timedelta(days=days)
            cutoff_iso = cutoff_utc.isoformat()
            
            with self.engine.db_lock:
                rows = self.engine.db.execute(
                    """
                    SELECT * FROM paper_orders
                    WHERE entry_time_utc >= ? OR status = 'OPEN'
                    ORDER BY id DESC
                    """,
                    (cutoff_iso,)
                ).fetchall()
            
            with self.lock:
                ticker_bid = float(self.last_ticker.get("bid", 0.0))
            
            trades = []
            closed_count = 0
            open_count = 0
            wins = 0
            losses = 0
            realized_pnl = 0.0
            unrealized_pnl = 0.0
            
            for r in rows:
                status = str(r["status"] or "CLOSED").upper()
                trade_id = int(r["id"])
                setup = str(r["signal_setup"] or "SETUP")
                side = str(r["side"] or "LONG").upper()
                qty = float(r["qty"] or 0.0)
                entry_fill = float(r["entry_fill"] or 0.0)
                notional = float(r["notional_quote"] or (qty * entry_fill))
                stop_price = float(r["stop_price"] or 0.0)
                target_price = float(r["target_price"] or 0.0)
                entry_time_str = str(r["entry_time_utc"] or "")
                
                # Conversione data/ora in locale
                entry_dt_local = None
                entry_dt_utc = None
                try:
                    entry_dt_utc = datetime.fromisoformat(entry_time_str)
                    entry_dt_local = entry_dt_utc.astimezone(ZoneInfo(Config.LOCAL_TIMEZONE))
                    entry_display = entry_dt_local.strftime("%d/%m %H:%M")
                except Exception:
                    entry_display = entry_time_str[:16] if entry_time_str else "--"
                
                if status == "OPEN":
                    open_count += 1
                    current_price = ticker_bid or entry_fill
                    exit_display = f"${current_price:,.2f}"
                    pnl = (current_price - entry_fill) * qty - float(r["entry_fee_quote"] or 0.0)
                    pnl_pct = ((current_price - entry_fill) / entry_fill * 100.0) if entry_fill > 0 else 0.0
                    unrealized_pnl += pnl
                    exit_reason = "IN_CORSO"
                    exit_reason_label = "[bold green]⚡ APERTA (In Corso)[/bold green]"
                    
                    if entry_dt_utc:
                        dur_sec = max(0, int((now_utc - entry_dt_utc.astimezone(timezone.utc)).total_seconds()))
                        dur_h = dur_sec // 3600
                        dur_m = (dur_sec % 3600) // 60
                        duration_display = f"{dur_h}h {dur_m:02d}m"
                    else:
                        duration_display = "--"
                else:
                    closed_count += 1
                    exit_fill = float(r["exit_fill"] or 0.0)
                    exit_display = f"${exit_fill:,.2f}" if exit_fill > 0 else "--"
                    pnl = float(r["pnl_quote"] or 0.0)
                    pnl_pct = (pnl / notional * 100.0) if notional > 0 else 0.0
                    realized_pnl += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    
                    exit_reason = str(r["exit_reason"] or "CLOSED")
                    reason_map = {
                        "TAKE_PROFIT": "[bold green]🎯 TAKE PROFIT[/bold green]",
                        "STOP_LOSS": "[bold red]🛑 STOP LOSS[/bold red]",
                        "TRAIL_TIER1": "[bold cyan]🛡️ TRAIL TIER 1[/bold cyan]",
                        "TRAIL_TIER2": "[bold cyan]🛡️ TRAIL TIER 2[/bold cyan]",
                        "TRAILING_STOP": "[bold cyan]🛡️ TRAIL STOP[/bold cyan]",
                        "TIME_EXIT": "[bold yellow]⏱️ TIME EXIT[/bold yellow]",
                        "DAILY_LOSS_LIMIT": "[bold red]⛔ MAX LOSS[/bold red]",
                        "CRITICAL_NEWS_DEFENSE_EXIT": "[bold magenta]🚨 NEWS EXIT[/bold magenta]",
                    }
                    exit_reason_label = reason_map.get(exit_reason, f"[white]{exit_reason}[/white]")
                    
                    exit_time_str = str(r["exit_time_utc"] or "")
                    if entry_dt_utc and exit_time_str:
                        try:
                            exit_dt_utc = datetime.fromisoformat(exit_time_str)
                            dur_sec = max(0, int((exit_dt_utc - entry_dt_utc).total_seconds()))
                            dur_h = dur_sec // 3600
                            dur_m = (dur_sec % 3600) // 60
                            duration_display = f"{dur_h}h {dur_m:02d}m"
                        except Exception:
                            duration_display = "--"
                    else:
                        duration_display = "--"
                
                trades.append({
                    "id": trade_id,
                    "signal_time": r["signal_time_utc"],
                    "setup": setup,
                    "side": side,
                    "status": status,
                    "qty": qty,
                    "notional": notional,
                    "entry_fill": entry_fill,
                    "entry_display": entry_display,
                    "exit_display": exit_display,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "duration_display": duration_display,
                    "exit_reason": exit_reason,
                    "exit_reason_label": exit_reason_label,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                })
            
            wr = (wins / closed_count * 100.0) if closed_count > 0 else 0.0
            
            return {
                "trades": trades,
                "total": len(trades),
                "closed_count": closed_count,
                "open_count": open_count,
                "wins": wins,
                "losses": losses,
                "win_rate": wr,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "net_pnl": realized_pnl + unrealized_pnl,
            }
        except Exception:
            return {
                "trades": [],
                "total": 0,
                "closed_count": 0,
                "open_count": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "net_pnl": 0.0,
            }

    @staticmethod
    def _make_mini_gauge(val: float, max_val: float, width: int = 8, fill_char: str = "█", empty_char: str = "░") -> str:
        """Genera una barra grafica compatta ASCII per indici percentuali."""
        if max_val <= 0:
            return f"[{empty_char * width}]"
        clamped = max(0.0, min(max_val, val))
        filled = int(round((clamped / max_val) * width))
        filled = min(width, max(0, filled))
        empty = width - filled
        return f"[{fill_char * filled}{empty_char * empty}]"

    @staticmethod
    def _make_rr_marker(stop: float, entry: float, target: float, current: float, width: int = 12) -> str:
        """Crea un indicatore visivo del percorso del prezzo tra Stop Loss e Take Profit."""
        if target <= stop:
            return ""
        ratio = (current - stop) / (target - stop)
        clamped = max(0.0, min(1.0, ratio))
        pos = int(round(clamped * (width - 1)))
        chars = ["─"] * width
        chars[pos] = "●"
        track = "".join(chars)
        return f"[dim]SL ${stop:,.0f}[/dim] ├{track}┤ [dim]TP ${target:,.0f}[/dim]"

    def _mask_key(self, key: str) -> str:
        if not key or len(key) < 6:
            return "[dim red]Non impostata[/dim red]"
        return f"{key[:4]}••••••••{key[-3:]}"

    def _format_event_details(self, etype: str, details: Dict[str, Any]) -> str:
        """
        Formatta i dettagli degli eventi in modo chiaro, colorato e senza troncamenti forzati.
        Mette sempre in primo piano il motivo (reason) e le informazioni operative essenziali.
        """
        if not isinstance(details, dict):
            return str(details)

        if etype == "SIGNAL_SKIPPED":
            reason = str(details.get("reason", "unknown"))
            reason_map = {
                "data_gap": "Gap storico candele exchange (candela saltata per salvaguardare l'integrità)",
                "daily_loss_limit": "Raggiunto limite perdita intraday (3.5% max del capitale)",
                "consecutive_loss_limit": "Raggiunto max errori consecutivi oggi (2 max consentiti)",
                "risk_size_below_minimum_or_cash": "Capitale o size ordine sotto il minimo exchange",
                "manual_user_confirmation": "Trading sospeso manualmente dall'operatore",
            }
            reason_label = reason_map.get(reason, reason)
            bar_time = details.get("bar_time_utc")
            time_info = ""
            if bar_time:
                try:
                    dt = datetime.fromisoformat(str(bar_time))
                    local_dt = dt.astimezone(ZoneInfo(Config.LOCAL_TIMEZONE))
                    time_info = f" [dim](Candela {local_dt.strftime('%H:%M')} Roma)[/dim]"
                except Exception:
                    time_info = f" [dim]({bar_time})[/dim]"
            return f"[bold yellow]Motivo: {reason_label}[/bold yellow]{time_info}"

        elif etype in ("SIGNAL_PRUNED", "SIGNAL_BLOCKED") or "PRUNED" in etype or "BLOCKED" in etype:
            reason = str(details.get("reason", "RL Filter"))
            setup = str(details.get("setup", ""))
            setup_str = f" [bold white]{setup}[/bold white]" if setup else ""
            if "PRUNED_BY_RL" in reason:
                desc = "Bloccato da Mistake Memory RL (Win Rate storico regime < 55%)"
            elif "CRITICAL_NEWS" in reason:
                desc = "Bloccato da Notizie Macro fortemente negative (Risk-Off Panic)"
            elif "NEGATIVE_NEWS" in reason:
                desc = "Bloccato da Notizie Macro sfavorevoli a breakout (Difesa Bull Trap)"
            else:
                desc = reason
            return f"[bold magenta]Motivo: {desc}[/bold magenta]{setup_str}"

        elif etype == "NEWS_POLL":
            score = float(details.get("score", 0.0))
            regime = str(details.get("regime", "NEUTRAL"))
            articles = details.get("articles", 0)
            score_color = "green" if score >= 0.15 else ("red" if score <= -0.15 else "cyan")
            regime_color = "bold green" if score >= 0.15 else ("bold red" if score <= -0.15 else "cyan")
            return f"Macro News Score: [{score_color}]{score:+.3f}[/{score_color}] ([{regime_color}]{regime}[/{regime_color}]) • [white]{articles} articoli analizzati[/white] (10 feed RSS)"

        elif etype == "PAPER_ENTRY":
            setup = str(details.get("setup", "SETUP"))
            qty = float(details.get("qty", 0.0))
            fill = float(details.get("entry_fill", 0.0))
            stop = float(details.get("stop_price", 0.0)) if "stop_price" in details else 0.0
            tp = float(details.get("target_price", 0.0)) if "target_price" in details else 0.0
            sl_tp = f" | SL: ${stop:,.1f} | TP: ${tp:,.1f}" if stop > 0 and tp > 0 else ""
            return f"[bold green]Acquisto {qty:.4f} BTC @ ${fill:,.2f}[/bold green] • Setup: [bold cyan]{setup}[/bold cyan]{sl_tp}"

        elif etype == "PAPER_EXIT":
            reason = str(details.get("reason", "EXIT"))
            pnl = float(details.get("pnl_quote", 0.0))
            pnl_color = "bold green" if pnl >= 0 else "bold red"
            sign = "+" if pnl >= 0 else ""
            reason_map = {
                "TAKE_PROFIT": "Target Take Profit raggiunto (+3.60%)",
                "STOP_LOSS": "Stop Loss iniziale colpito (-1.80%)",
                "TRAIL_TIER1": "Trailing Stop Tier 1 protetto (+0.40%)",
                "TRAIL_TIER2": "Trailing Stop Tier 2 protetto (+1.20%)",
                "TIME_EXIT": "Timeout durata max candele (6h)",
                "DAILY_LOSS_LIMIT": "Chiusura di emergenza: Limite perdita giornaliera superato",
                "CRITICAL_NEWS_DEFENSE_EXIT": "Chiusura di emergenza: Notizia critica macro negativa",
            }
            reason_str = reason_map.get(reason, reason)
            return f"PnL Realizzato: [{pnl_color}]{sign}{pnl:.2f} USDC[/{pnl_color}] • [bold white]Motivo: {reason_str}[/bold white]"

        elif etype == "DAILY_LOSS_LOCK":
            loss = float(details.get("daily_loss_quote", 0.0))
            limit = float(details.get("limit_usdc", 35.0))
            return f"[bold red]CIRCUITO BLOCCATO: Perdita odierna {loss:.2f} / {limit:.2f} USDC (Trading sospeso per protezione)[/bold red]"

        elif etype == "CONSECUTIVE_LOSS_LOCK":
            losses = details.get("consecutive_losses", 0)
            limit = details.get("limit", 2)
            return f"[bold red]CIRCUITO BLOCCATO: Raggiunti {losses} trade in perdita consecutivi (max {limit})[/bold red]"

        elif etype == "BOT_PAUSED":
            reason = details.get("reason", "")
            reason_str = " (conferma operatore)" if "manual" in str(reason) else ""
            return f"[bold yellow]Trading automatico in PAUSA{reason_str}[/bold yellow] • Monitoraggio dati attivo"

        elif etype == "BOT_RESUMED":
            return "[bold green]Trading automatico ATTIVO[/bold green] • Scanner pronto per nuovi setup"

        elif etype == "LEARNING_REVIEW":
            return "[bold magenta]RL Engine: Statistiche Q-learning aggiornate e salvate in memoria[/bold magenta]"

        elif etype in ("ERROR", "KLINE_ERROR"):
            err = details.get("error", "Errore sconosciuto")
            return f"[bold red]Errore: {err}[/bold red]"

        # Default per altri tipi di evento generici
        parts = []
        if "reason" in details:
            parts.append(f"[bold yellow]Motivo: {details['reason']}[/bold yellow]")
        for k, v in details.items():
            if k != "reason":
                parts.append(f"{k}: {v}")
        return " • ".join(parts) if parts else str(details)

    def _get_events(self, limit: int = 7) -> List[Dict[str, Any]]:
        try:
            with self.engine.db_lock:
                rows = self.engine.db.execute(
                    "SELECT observed_at_utc, event_type, details_json FROM paper_events ORDER BY id DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            events = []
            for r in rows:
                dt_utc = datetime.fromisoformat(r["observed_at_utc"])
                dt_local = dt_utc.astimezone(ZoneInfo(Config.LOCAL_TIMEZONE))
                time_str = dt_local.strftime("%H:%M:%S")
                events.append({
                    "time": time_str,
                    "type": r["event_type"],
                    "details": json.loads(r["details_json"])
                })
            return events
        except Exception:
            return []

    def make_header(self) -> Panel:
        now_dt = datetime.now(ZoneInfo(Config.LOCAL_TIMEZONE))
        now_rome = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        uptime_sec = max(0, int((datetime.now() - self.start_time).total_seconds()))
        hours = uptime_sec // 3600
        mins = (uptime_sec % 3600) // 60
        secs = uptime_sec % 60
        uptime_str = f"{hours}h {mins:02d}m {secs:02d}s" if hours > 0 else f"{mins}m {secs:02d}s"
        
        status_text = (
            "[bold green]● TRADING ATTIVO[/bold green]"
            if self.trading_active
            else "[bold yellow]⏸ IN PAUSA / MONITORAGGIO[/bold yellow]"
        )
        
        mode_text = "[bold cyan]BYBIT EU SPOT (DEMO USDC)[/bold cyan]"
        header_table = Table.grid(expand=True)
        header_table.add_column(justify="left", no_wrap=True)
        header_table.add_column(justify="center", ratio=1)
        header_table.add_column(justify="right", no_wrap=True)
        
        header_table.add_row(
            Text.from_markup(f"  ⚡ [bold white]QUANT BOT INTRADAY[/bold white] | {mode_text}"),
            Text.from_markup(f"STATO: {status_text} | [dim]Uptime: {uptime_str}[/dim]"),
            Text.from_markup(f"🕒 [cyan]{now_rome} (Rome)[/cyan]  ")
        )
        return Panel(header_table, style="bold blue", padding=(0, 0))

    def make_chart_panel(self) -> Panel:
        """Pannello grafico a candele giapponesi 1H (ultime 48-60 candele) con asse prezzi, volume e Big Live Ticker."""
        with self.lock:
            ticker = dict(self.last_ticker)
            indicators = dict(self.last_indicators)
            candles = list(getattr(self, "cached_chart_candles", []))

        bid = float(ticker.get("bid", 0.0))
        ask = float(ticker.get("ask", 0.0))
        last_p = float(ticker.get("last", 0.0))
        if bid > 0 and ask > 0 and (last_p < bid or last_p > ask or last_p <= 0):
            live_price = (bid + ask) / 2.0
        else:
            live_price = last_p or ((bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0)

        if not candles:
            if live_price > 0:
                candles = [{"open": live_price, "high": live_price, "low": live_price, "close": live_price, "volume": 1.0}]
            else:
                candles = [{"open": 73500.0, "high": 73600.0, "low": 73400.0, "close": 73550.0, "volume": 1.0}]

        ema_50 = float(indicators.get("ema_50", 0.0))
        ema_200 = float(indicators.get("ema_200", 0.0))
        prev_high_12 = float(indicators.get("prev_high_12", indicators.get("prev_high_20", 0.0)))
        bb_lower_165 = float(indicators.get("bb_lower_165", indicators.get("bb_lower", 0.0)))
        open_curr = float(indicators.get("open", candles[-1]["open"] if candles else live_price))

        chg_24 = float(ticker.get("change24h_pct", 0.0))
        low_24 = float(ticker.get("low24h", 0.0))
        high_24 = float(ticker.get("high24h", 0.0))
        spread = ask - bid if ask > bid else 0.0
        spread_bps = (spread / ask * 10000.0) if ask > 0 else 0.0

        is_green = (live_price >= open_curr) if open_curr > 0 else (chg_24 >= 0)
        p_color = "bold #00ff88" if is_green else "bold #ff3355"

        term_w = getattr(self.console, "width", 257)
        is_compact = term_w < 220
        w_right = 68 if not is_compact else 54
        w_chart_max = max(60, term_w - w_right - 6)
        n_candles = max(25, min(len(candles), (w_chart_max - 18) // 2))
        display_candles = candles[-n_candles:]
        w_chart = len(display_candles) * 2 + 18

        H = 10
        min_p = min(float(c.get("low", live_price)) for c in display_candles)
        max_p = max(float(c.get("high", live_price)) for c in display_candles)
        if live_price > 0:
            min_p = min(min_p, live_price)
            max_p = max(max_p, live_price)
        price_range = max_p - min_p if max_p > min_p else 1.0

        def p_to_r(p_val: float) -> int:
            frac = (p_val - min_p) / price_range
            r = int((1.0 - frac) * (H - 1))
            return max(0, min(H - 1, r))

        grid = [[(" ", "") for _ in range(len(display_candles) * 2)] for _ in range(H)]
        for i, c in enumerate(display_candles):
            col = i * 2
            c_open = float(c.get("open", live_price))
            c_high = float(c.get("high", live_price))
            c_low = float(c.get("low", live_price))
            c_close = float(c.get("close", live_price))
            r_high = p_to_r(c_high)
            r_low = p_to_r(c_low)
            r_open = p_to_r(c_open)
            r_close = p_to_r(c_close)
            r_top = min(r_open, r_close)
            r_bottom = max(r_open, r_close)

            is_bull = c_close >= c_open
            col_style = "bold #00ff88" if is_bull else "bold #ff3355"

            for r in range(r_high, r_low + 1):
                grid[r][col] = ("│", col_style)

            if r_top == r_bottom:
                grid[r_top][col] = ("┼" if (r_top > r_high or r_bottom < r_low) else "─", col_style)
            else:
                for r in range(r_top, r_bottom + 1):
                    grid[r][col] = ("█", col_style)

        r_ema50 = p_to_r(ema_50) if ema_50 > 0 else -1
        r_ema200 = p_to_r(ema_200) if ema_200 > 0 else -1
        r_live = p_to_r(live_price) if live_price > 0 else -1

        chart_table = Table.grid(expand=True, padding=(0, 0))
        chart_table.add_column(no_wrap=True)

        for r in range(H):
            t_row = Text()
            for ch, style in grid[r]:
                t_row.append(ch, style=style)
            p_val = max_p - (r / (H - 1)) * price_range
            if r == r_live:
                t_row.append(f" ◄ LIVE ${live_price:,.0f}", style="bold yellow")
            elif r == r_ema50:
                t_row.append(f" ── E50 ${ema_50:,.0f}", style="bold cyan")
            elif r == r_ema200:
                t_row.append(f" ── E200 ${ema_200:,.0f}", style="bold magenta")
            else:
                t_row.append(f" ┤ ${p_val:,.0f}", style="dim")
            chart_table.add_row(t_row)

        # Big Live Ticker (Right Column - solo numero senza simbolo $)
        big_price_str = f"{live_price:,.0f}" if live_price > 0 else "0"
        price_lines = _render_big_digits(big_price_str)
        big_t = Text("\n".join(price_lines), style=p_color, justify="right")

        right_table = Table.grid(expand=True, padding=(0, 1))
        lbl_w = 20 if is_compact else 25
        right_table.add_column(style="dim", justify="left", width=lbl_w, no_wrap=True)
        right_table.add_column(justify="right", style="bold", ratio=1, no_wrap=True)

        chg_sign = "+" if chg_24 >= 0 else ""
        chg_color = "bold green" if chg_24 >= 0 else "bold red"
        chg_usd = live_price * (chg_24 / 100.0) if live_price > 0 else 0.0

        now_rome = datetime.now(ZoneInfo(Config.LOCAL_TIMEZONE))
        mins_left = 59 - now_rome.minute
        secs_left = 59 - now_rome.second
        countdown_str = f"{mins_left:02d}m {secs_left:02d}s"

        dist_ema200 = ((live_price - ema_200) / ema_200 * 100.0) if ema_200 > 0 else 0.0
        dist_res = ((live_price - prev_high_12) / prev_high_12 * 100.0) if prev_high_12 > 0 else 0.0
        dist_bb = ((live_price - bb_lower_165) / bb_lower_165 * 100.0) if bb_lower_165 > 0 else 0.0

        d_e200_col = "bold green" if dist_ema200 >= 0 else "bold red"
        d_res_col = "bold green" if dist_res >= 0 else "bold yellow"
        d_bb_col = "bold green" if dist_bb >= 0 else "bold red"

        tag_e200 = f"vs EMA200 ${ema_200:,.0f}" if not is_compact else f"E200 ${ema_200/1e3:.1f}k"
        tag_res = f"vs Max ${prev_high_12:,.0f}" if not is_compact else f"Max ${prev_high_12/1e3:.1f}k"
        tag_bb = f"vs BB ${bb_lower_165:,.0f}" if not is_compact else f"BB ${bb_lower_165/1e3:.1f}k"

        lbl_24h = "Variazione 24H Bybit:" if not is_compact else "Var 24H Bybit:"
        lbl_spr = "Spread Bid / Ask:" if not is_compact else "Spread Bid/Ask:"
        lbl_close = "Chiusura Candela 1H:" if not is_compact else "Chiusura 1H:"
        lbl_d_e200 = "Distanza EMA200:" if not is_compact else "Dist EMA200:"
        lbl_d_res = "Distanza Resistenza 12H:" if not is_compact else "Dist Res 12H:"
        lbl_d_bb = "Distanza BB Inf 1.65σ:" if not is_compact else "Dist BB Inf:"

        stat_24h = f"[{chg_color}]{chg_sign}{chg_24:.2f}%[/{chg_color}] [dim]({chg_sign}${chg_usd:,.0f} • ${low_24:,.0f}-${high_24:,.0f})[/dim]" if not is_compact else f"[{chg_color}]{chg_sign}{chg_24:.2f}%[/{chg_color}] [dim](${low_24/1e3:.0f}k-${high_24/1e3:.0f}k)[/dim]"
        stat_spr = f"${bid:,.1f} / ${ask:,.1f} [dim](${spread:,.1f} • {spread_bps:.2f} bps)[/dim]" if not is_compact else f"${bid:,.0f}/${ask:,.0f} [dim]({spread_bps:.1f} bps)[/dim]"

        right_table.add_row(lbl_24h, stat_24h)
        right_table.add_row(lbl_spr, stat_spr)
        right_table.add_row(lbl_close, f"Tra [bold cyan]{countdown_str}[/bold cyan] [dim](countdown)[/dim]")
        right_table.add_row(lbl_d_e200, f"[{d_e200_col}]{dist_ema200:+.2f}%[/{d_e200_col}] [dim]({tag_e200})[/dim]")
        right_table.add_row(lbl_d_res, f"[{d_res_col}]{dist_res:+.2f}%[/{d_res_col}] [dim]({tag_res})[/dim]")
        right_table.add_row(lbl_d_bb, f"[{d_bb_col}]{dist_bb:+.2f}%[/{d_bb_col}] [dim]({tag_bb})[/dim]")

        right_box = Table.grid(expand=True)
        right_box.add_column(width=w_right, no_wrap=True)
        right_box.add_row(big_t)
        right_box.add_row(right_table)

        main_grid = Table.grid(expand=True, padding=(0, 1))
        main_grid.add_column(width=w_chart, no_wrap=True)
        main_grid.add_column(width=w_right, no_wrap=True)
        main_grid.add_row(chart_table, right_box)

        return Panel(
            main_grid,
            title=f"📈 [bold cyan]Grafico Candele Giapponesi BTC/USDC (1H - Ultime {n_candles} Candele) & Big Live Ticker[/bold cyan]",
            border_style="cyan"
        )

    def make_wallet_panel(self) -> Panel:
        with self.lock:
            with self.engine.db_lock:
                cash = float(self.engine._get_state("cash_quote", "1000.0"))
                initial_cap = float(self.engine._get_state("starting_capital_quote", "1000.0"))
                order = self.engine._open_order()
                daily_start = float(self.engine._get_state("daily_start_equity_quote", "1000.0"))
                daily_locked = self.engine._get_state("daily_loss_locked") == "1"
                consec_losses = int(self.engine._get_state("consecutive_losses_today", "0") or 0)
            
            ticker_bid = self.last_ticker.get("bid", 0.0)
            pos_qty = float(order["qty"]) if order else 0.0
            pos_notional = pos_qty * ticker_bid
            total_equity = cash + (pos_qty * ticker_bid * (1 - Config.FEE_RATE))
            total_pnl = total_equity - initial_cap
            total_pnl_pct = (total_pnl / initial_cap) * 100.0 if initial_cap > 0 else 0.0
            
            daily_pnl = total_equity - daily_start
            daily_pnl_pct = (daily_pnl / daily_start) * 100.0 if daily_start > 0 else 0.0
            
            daily_loss = max(0.0, daily_start - total_equity)
            daily_limit = self.engine.get_dynamic_daily_loss_limit(daily_start)
            remaining_risk_buffer = max(0.0, daily_limit - daily_loss)
            risk_gauge = self._make_mini_gauge(remaining_risk_buffer, daily_limit, width=8)

        stats = self._get_trade_stats()
        
        t = Table.grid(expand=True, padding=(0, 1))
        t.add_column(style="dim", justify="left")
        t.add_column(justify="right", style="bold")
        
        pnl_style = "green" if total_pnl >= 0 else "red"
        pnl_sign = "+" if total_pnl >= 0 else ""
        
        daily_pnl_style = "green" if daily_pnl >= 0 else "red"
        daily_pnl_sign = "+" if daily_pnl >= 0 else ""
        
        t.add_row("Capitale Iniziale:", f"{initial_cap:.2f} USDC")
        t.add_row("Saldo Cassa (Liquidità):", f"{cash:.2f} USDC")
        t.add_row("Controvalore Posizione:", f"{pos_notional:.2f} USDC")
        t.add_row("Equity Totale Portafoglio:", f"[bold white]{total_equity:.2f} USDC[/bold white]")
        t.add_row("PnL Netto da Inizio:", f"[{pnl_style}]{pnl_sign}{total_pnl:.2f} USDC ({pnl_sign}{total_pnl_pct:.2f}%)[/{pnl_style}]")
        t.add_row("PnL Sessione Odierna:", f"[{daily_pnl_style}]{daily_pnl_sign}{daily_pnl:.2f} USDC ({daily_pnl_sign}{daily_pnl_pct:.2f}%)[/{daily_pnl_style}]")
        
        if stats["total"] > 0:
            trades_str = f"{stats['total']} trade (WR: {stats['wr']:.1f}% | W: {stats['wins']} L: {stats['losses']})"
        else:
            trades_str = "0 conclusi (Win Rate: --%)"
        t.add_row("Trade Eseguiti:", f"[white]{trades_str}[/white]")
        
        buffer_color = "green" if remaining_risk_buffer > daily_limit * 0.5 else ("yellow" if remaining_risk_buffer > daily_limit * 0.2 else "red")
        t.add_row("Buffer Rischio Oggi:", f"[{buffer_color}]{risk_gauge} {remaining_risk_buffer:.1f} / {daily_limit:.1f} USDC[/{buffer_color}]")
        t.add_row("Perdite Consecutive Oggi:", f"{consec_losses} / {Config.PAPER_MAX_CONSECUTIVE_LOSSES} max")
        
        cb_color = "red" if daily_locked else "green"
        cb_text = "BLOCCATO (CIRCUIT BREAKER)" if daily_locked else "ATTIVO (OK)"
        t.add_row("Protezione Intraday:", f"[{cb_color}]{cb_text}[/{cb_color}]")
        
        return Panel(t, title="💰 [bold yellow]Portafoglio & Budget Simulato (Continuo)[/bold yellow]", border_style="yellow")

    def make_position_panel(self) -> Panel:
        with self.lock:
            with self.engine.db_lock:
                order = self.engine._open_order()
                trailing_tier = int(self.engine._get_state("paper_trailing_tier", "0") or 0)
            ticker_bid = self.last_ticker.get("bid", 0.0)
            indicators = dict(self.last_indicators)
            last_p = self.last_ticker.get("last", 0.0) or ticker_bid or indicators.get("close", 0.0)
        
        # Scenario 1: Nessuna operazione in corso -> Radar Tecnico & Scanner Live 1H
        if not order:
            now_rome = datetime.now(ZoneInfo(Config.LOCAL_TIMEZONE))
            mins_left = 59 - now_rome.minute
            secs_left = 59 - now_rome.second
            candle_countdown = f"[bold cyan]{mins_left:02d}m {secs_left:02d}s[/bold cyan]"
            
            t = Table.grid(expand=True, padding=(0, 1))
            t.add_column(style="dim", justify="left")
            t.add_column(justify="right", style="bold")
            
            # Trend Strutturale 1H
            ema_20 = indicators.get("ema_20", 0.0)
            ema_50 = indicators.get("ema_50", 0.0)
            ema_100 = indicators.get("ema_100", 0.0)
            ema_200 = indicators.get("ema_200", 0.0)
            rsi = indicators.get("rsi", 50.0)
            high_20 = indicators.get("prev_high_20", indicators.get("high_20", 0.0))
            bb_lower = indicators.get("bb_lower", 0.0)
            
            if ema_50 > 0 and ema_200 > 0:
                if last_p > ema_50 and ema_50 > ema_200:
                    trend_str = "[bold green]🟢 RIALZISTA (Bullish Trend - Setup 1 Pullback & Setup 2 Breakout)[/bold green]"
                elif last_p < ema_50 and ema_50 < ema_200:
                    trend_str = "[bold yellow]🟠 BEAR/RANGE ATTIVO (Setup 3 Mean-Reversion in Scanner)[/bold yellow]"
                else:
                    trend_str = "[bold yellow]🟡 CONSOLIDAMENTO / LATERALE (Tri-Engine Scanner Attivo)[/bold yellow]"
            else:
                trend_str = "[dim]Calcolo trend in corso...[/dim]"
            
            if ema_100 > 0 and last_p > 0:
                dist_ema100 = ((last_p - ema_100) / ema_100) * 100.0
                ema100_status = f"{dist_ema100:+.2f}% da EMA100 (${ema_100:,.0f})"
            else:
                ema100_status = "In attesa dati"
                
            if high_20 > 0 and last_p > 0:
                dist_high20 = ((last_p - high_20) / high_20) * 100.0
                breakout_status = f"{dist_high20:+.2f}% da Resistenza 20H (${high_20:,.0f})"
            else:
                breakout_status = "In attesa dati"

            if bb_lower > 0 and last_p > 0:
                dist_bb = ((last_p - bb_lower) / bb_lower) * 100.0
                mean_rev_status = f"{dist_bb:+.2f}% da BB Inf (${bb_lower:,.0f})"
            else:
                mean_rev_status = "In attesa dati"

            t.add_row("Stato Posizione:", "[bold green]🟢 NESSUNA POSIZIONE APERTA (Tri-Engine Scanner)[/bold green]")
            t.add_row("Valutazione Prossima Candela 1H:", f"Tra {candle_countdown} (chiusura candela)")
            t.add_row("Trend Strutturale 1H:", trend_str)
            t.add_row("Setup 1 (Pullback Reattivo):", f"[white]{ema100_status}[/white]")
            t.add_row("Setup 2 (Momentum Breakout):", f"[white]{breakout_status}[/white]")
            t.add_row("Setup 3 (Mean-Rev Bounce):", f"[white]{mean_rev_status}[/white]")
            
            with self.engine.db_lock:
                eq_val = float(self.engine._get_state("cash_quote", str(Config.BACKTEST_CAPITAL_USDC))) or 1000.0
            base_alloc = getattr(Config, "BASE_POSITION_PCT", 0.75)
            t.add_row(
                "Allocazione & Rischio Reale:",
                f"[bold white]Base {base_alloc*100:.0f}% (${eq_val*base_alloc:,.0f})[/bold white] • [dim]RL: Probed 34% (${eq_val*base_alloc*0.45:,.0f}) / Max 97.5% (${eq_val*base_alloc*1.3:,.0f}) | SL Rischio: ~1.35% (${eq_val*base_alloc*0.018:,.1f})[/dim]"
            )
            t.add_row("Trailing Stop Dinamico:", f"[white]Tier 1: +{Config.TRAIL_TIER1_TRIGGER*100:.2f}% (Lock +{Config.TRAIL_TIER1_LOCK*100:.2f}%) • Tier 2: +{Config.TRAIL_TIER2_TRIGGER*100:.2f}% (Lock +{Config.TRAIL_TIER2_LOCK*100:.2f}%)[/white]")
            t.add_row("Dettaglio Condizioni:", "[bold yellow]Vedi tabella Radar sottostante per checklist live dei 3 Setup[/bold yellow]")
            
            return Panel(t, title="📈 [bold cyan]Operazione in Corso & Monitor Posizione (BTC/USDC)[/bold cyan]", border_style="cyan")
        
        # Scenario 2: Operazione aperta presente
        entry_fill = float(order["entry_fill"])
        qty = float(order["qty"])
        entry_notional = qty * entry_fill
        unrealized_pnl = (ticker_bid - entry_fill) * qty - float(order["entry_fee_quote"])
        unrealized_pnl_pct = ((ticker_bid - entry_fill) / entry_fill) * 100.0 if entry_fill > 0 else 0.0
        
        stop_price = float(order["stop_price"])
        target_price = float(order["target_price"])
        setup_name = str(order["signal_setup"])
        entry_time_str = str(order["entry_time_utc"])
        
        try:
            entry_dt = datetime.fromisoformat(entry_time_str)
            elapsed_sec = max(0, int((datetime.now(timezone.utc) - entry_dt.astimezone(timezone.utc)).total_seconds()))
            hours = elapsed_sec // 3600
            mins = (elapsed_sec % 3600) // 60
            duration_str = f"{hours}h {mins}m / max {Config.MAX_HOLDING_HOURS:g}h"
        except Exception:
            duration_str = "n/d"
        
        # Calcolo R-multiple e indicatore di percorso SL / TP
        initial_risk = (entry_fill * Config.DEFAULT_SL_PCT) if entry_fill > 0 else 1.0
        r_multiple = (ticker_bid - entry_fill) / initial_risk if initial_risk > 0 else 0.0
        rr_track = self._make_rr_marker(stop_price, entry_fill, target_price, ticker_bid, width=12)
        
        t = Table.grid(expand=True, padding=(0, 1))
        t.add_column(style="dim", justify="left")
        t.add_column(justify="right", style="bold")
        
        t.add_row("Setup Operativo:", f"[bold magenta]{setup_name}[/bold magenta] (LONG)")
        t.add_row("Quantità BTC:", f"{qty:.5f} BTC ({entry_notional:.2f} USDC)")
        t.add_row("Prezzo d'Entrata:", f"${entry_fill:,.2f}")
        t.add_row("Prezzo Attuale (Bid):", f"${ticker_bid:,.2f}")
        
        pnl_style = "bold green" if unrealized_pnl >= 0 else "bold red"
        pnl_sign = "+" if unrealized_pnl >= 0 else ""
        t.add_row("PnL Non Realizzato:", f"[{pnl_style}]{pnl_sign}{unrealized_pnl:.2f} USDC ({pnl_sign}{unrealized_pnl_pct:.2f}%)[/{pnl_style}]")
        
        t1_lock_pct = getattr(Config, "TRAIL_TIER1_LOCK", 0.0080) * 100.0
        t2_lock_pct = getattr(Config, "TRAIL_TIER2_LOCK", 0.0150) * 100.0
        tier_label = {
            0: f"[dim]Stop Loss Iniziale ({getattr(Config, 'DEFAULT_SL_PCT', 0.018)*100:.1f}%)[/dim]",
            1: f"[bold cyan]Tier 1 Trailing Locked (+{t1_lock_pct:.2f}%)[/bold cyan]",
            2: f"[bold green]Tier 2 Trailing Locked (+{t2_lock_pct:.2f}%)[/bold green]"
        }.get(trailing_tier, "[yellow]Trailing Attivo[/yellow]")
        
        tp_pct = ((target_price - entry_fill) / entry_fill * 100.0) if entry_fill > 0 else 3.60
        t.add_row("Percorso Prezzo:", f"{rr_track} ([bold]{r_multiple:+.2f}R[/bold])")
        t.add_row("Stop Loss Attivo:", f"${stop_price:,.2f} ({tier_label})")
        t.add_row("Take Profit Target:", f"${target_price:,.2f} (+{tp_pct:.2f}%)")
        t.add_row("Tempo in Posizione:", f"{duration_str}")
        
        return Panel(t, title="📈 [bold green]Operazione Aperta (LONG ATTIVA)[/bold green]", border_style="green")

    def make_weekly_trades_panel(self) -> Panel:
        """Pannello riassuntivo delle ultime operazioni effettuate nella settimana (ultimi 7 giorni)."""
        data = self._get_weekly_trades(days=7)
        trades = data["trades"]
        tot = data["total"]
        closed = data["closed_count"]
        wins = data["wins"]
        losses = data["losses"]
        wr = data["win_rate"]
        realized_pnl = data["realized_pnl"]
        open_count = data["open_count"]
        unrealized_pnl = data["unrealized_pnl"]
        
        # Stili PnL
        pnl_style = "green" if realized_pnl >= 0 else "red"
        pnl_sign = "+" if realized_pnl >= 0 else ""
        wr_style = "green" if wr >= 50.0 else ("yellow" if wr > 0 else "dim")
        
        title_summary = f"[bold white]{tot}[/bold white] nella settimana [dim]•[/dim] Conclusi: [white]{closed}[/white] ([bold green]{wins}W[/bold green] / [bold red]{losses}L[/bold red] • WR: [bold {wr_style}]{wr:.1f}%[/bold {wr_style}]) [dim]|[/dim] PnL Realizzato: [{pnl_style}]{pnl_sign}{realized_pnl:.2f} USDC[/{pnl_style}]"
        if open_count > 0:
            unreal_style = "green" if unrealized_pnl >= 0 else "red"
            unreal_sign = "+" if unrealized_pnl >= 0 else ""
            title_summary += f" [dim]|[/dim] Aperte: [bold cyan]{open_count}[/bold cyan] (Non Realizzato: [{unreal_style}]{unreal_sign}{unrealized_pnl:.2f} USDC[/{unreal_style}])"

        if not trades:
            empty_msg = Text("\n  Nessuna operazione effettuata negli ultimi 7 giorni. Il bot è in monitoraggio attivo dei setup di mercato.", style="dim italic")
            return Panel(
                empty_msg,
                title=f"💼 [bold cyan]Ultime Operazioni Effettuate nella Settimana (BTC/USDC)[/bold cyan]  •  {title_summary}",
                border_style="cyan",
                padding=(0, 1),
            )
            
        t = Table(expand=True, box=None, padding=(0, 1), show_header=True, header_style="bold cyan")
        t.add_column("#", style="dim", width=4, justify="center", no_wrap=True)
        t.add_column("Data/Ora Ingr.", style="bold white", width=14, justify="center", no_wrap=True)
        t.add_column("Setup Operativo", width=16, justify="left", no_wrap=True)
        t.add_column("Side", width=6, justify="center", no_wrap=True)
        t.add_column("Quantità / Controvalore", width=22, justify="right", no_wrap=True)
        t.add_column("Prezzo Ingresso", width=16, justify="right", no_wrap=True)
        t.add_column("Prezzo Uscita / Live", width=20, justify="right", no_wrap=True)
        t.add_column("Target (SL / TP)", width=24, justify="center", no_wrap=True)
        t.add_column("Durata", width=10, justify="center", no_wrap=True)
        t.add_column("Esito / Motivo Uscita", width=22, justify="left", no_wrap=True)
        t.add_column("PnL Netto (USDC)", width=24, justify="right", no_wrap=True)
        t.add_column("Stato", width=10, justify="center", no_wrap=True)

        for tr in trades[:6]:
            tr_id = f"#{tr['id']}"
            setup_name = tr["setup"].replace("_BOUNCE", "").replace("_REACTIVE", "").replace("_BREAKOUT", "")
            if "MEAN_REV" in setup_name or "MEAN" in setup_name:
                setup_label = "[bold magenta]MEAN-REV[/bold magenta]"
            elif "PULLBACK" in setup_name:
                setup_label = "[bold cyan]PULLBACK[/bold cyan]"
            elif "MOMENTUM" in setup_name or "BREAKOUT" in setup_name:
                setup_label = "[bold yellow]BREAKOUT[/bold yellow]"
            else:
                setup_label = f"[bold white]{setup_name}[/bold white]"
                
            side_label = f"[bold green]{tr['side']}[/bold green]" if tr["side"] == "LONG" else f"[bold red]{tr['side']}[/bold red]"
            size_label = f"{tr['qty']:.5f} BTC [dim](${tr['notional']:.1f})[/dim]"
            entry_p = f"${tr['entry_fill']:,.2f}"
            exit_p = tr["exit_display"]
            sl_tp = f"SL: ${tr['stop_price']:,.0f} | TP: ${tr['target_price']:,.0f}"
            dur = tr["duration_display"]
            reason = tr["exit_reason_label"]
            
            pnl_val = tr["pnl"]
            pnl_pct_val = tr["pnl_pct"]
            p_style = "bold green" if pnl_val >= 0 else "bold red"
            p_sign = "+" if pnl_val >= 0 else ""
            pnl_str = f"[{p_style}]{p_sign}{pnl_val:.2f} USDC ({p_sign}{pnl_pct_val:.2f}%)[/{p_style}]"
            
            status_label = "[bold green]APERTA[/bold green]" if tr["status"] == "OPEN" else "[dim]CHIUSA[/dim]"
            
            t.add_row(
                tr_id,
                tr["entry_display"],
                setup_label,
                side_label,
                size_label,
                entry_p,
                exit_p,
                sl_tp,
                dur,
                reason,
                pnl_str,
                status_label,
            )

        return Panel(
            t,
            title=f"💼 [bold cyan]Ultime Operazioni Effettuate nella Settimana (BTC/USDC)[/bold cyan]  •  {title_summary}",
            border_style="cyan",
            padding=(0, 0),
        )

    def make_sentiment_panel(self) -> Panel:
        with self.lock:
            with self.engine.db_lock:
                news_score = float(self.engine._get_state("current_news_score", "0.0") or 0.0)
            ticker = self.last_ticker
            fng_val = self.fng_data.get("value", 50)
            fng_class = self.fng_data.get("classification", "Neutral")
            latest_news = dict(self.last_news_item)
        
        if news_score >= 0.15:
            news_regime = "[bold green]BULLISH REGIME[/bold green]"
        elif news_score <= -0.15:
            news_regime = "[bold red]RISK-OFF BEARISH[/bold red]"
        else:
            news_regime = "[cyan]NEUTRAL REGIME[/cyan]"
        
        fng_color = "red" if fng_val < 30 else ("yellow" if fng_val < 55 else "green")
        fng_gauge = self._make_mini_gauge(fng_val, 100, width=8)
        
        with self.lock:
            fr = getattr(self, "last_funding_rate", 0.0001)
        fr_color = "bold red" if fr >= 0.0003 else ("bold green" if fr < 0 else "cyan")
        fr_tag = "Surriscaldato (Leva Long)" if fr >= 0.0003 else ("Negativo (Squeeze Short)" if fr < 0 else "Equilibrato")

        # Dati 24h da Bybit
        chg_24 = ticker.get("change24h_pct", 0.0)
        chg_style = "bold green" if chg_24 >= 0 else "bold red"
        chg_sign = "+" if chg_24 >= 0 else ""
        low_24 = ticker.get("low24h", 0.0)
        high_24 = ticker.get("high24h", 0.0)
        vol_24 = ticker.get("volume24h", 0.0)
        turnover_24 = ticker.get("turnover24h", 0.0)
        
        bid = ticker.get("bid", 0.0)
        ask = ticker.get("ask", 0.0)
        spread = ask - bid if ask > bid else 0.0
        spread_bps = (spread / ask * 10000.0) if ask > 0 else 0.0
        
        last_price = ticker.get("last", 0.0)
        if bid > 0 and ask > 0 and (last_price < bid or last_price > ask or last_price <= 0):
            spot_price = (bid + ask) / 2.0
        else:
            spot_price = last_price or ((bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0)

        t = Table.grid(expand=True, padding=(0, 1))
        t.add_column(style="dim", justify="left")
        t.add_column(justify="right", style="bold")
        
        t.add_row("Prezzo BTC Spot:", f"${spot_price:,.2f}")
        t.add_row("Spread Bid / Ask:", f"${bid:,.1f} / ${ask:,.1f} (${spread:,.1f} • {spread_bps:.2f} bps)")
        t.add_row("Statistiche 24H Bybit:", f"[{chg_style}]{chg_sign}{chg_24:.2f}%[/{chg_style}] (Min: ${low_24:,.0f} | Max: ${high_24:,.0f})")
        t.add_row("Volume 24H:", f"{vol_24:,.1f} BTC (~${turnover_24/1e6:.2f}M USDC)")
        t.add_row("Funding Rate (Derivati):", f"[{fr_color}]{fr*100:+.4f}% / 8h[/{fr_color}] [dim]({fr_tag})[/dim]")
        t.add_row("Fear & Greed Index:", f"[{fng_color}]{fng_gauge} {fng_val}/100 ({fng_class})[/{fng_color}]")
        t.add_row("Macro News Score:", f"{news_score:+.3f} ({news_regime})")
        
        if latest_news and latest_news.get("title"):
            title = latest_news["title"]
            src = latest_news.get("source", "")
            pub = latest_news.get("published_at_utc", "")
            time_str = ""
            if pub:
                try:
                    dt = datetime.fromisoformat(pub)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    local_dt = dt.astimezone(ZoneInfo(Config.LOCAL_TIMEZONE))
                    time_str = f" • {local_dt.strftime('%H:%M')}"
                except Exception:
                    pass
            full_title = f"[{src}] {title}" if src else title
            n_score = latest_news.get("score", 0.0)
            sc_color = "green" if n_score > 0.1 else ("red" if n_score < -0.1 else "cyan")
            t.add_row("Ultima Notizia Live:", f"[{sc_color}]Sentiment Score: {n_score:+.2f}{time_str}[/{sc_color}]")
            news_line = Text.from_markup(f"  [dim italic]\"{escape(full_title)}\"[/dim italic]", overflow="fold")
            content = Group(t, news_line)
        else:
            t.add_row("Fonti Intelligence Connesse:", f"[white]{len(self.engine.sentiment.RSS_FEEDS)} feed attivi (Bloomberg, Reuters, FT, WSJ, FED, BCE, Sole24Ore, X/FastWire)[/white]")
            content = t

        return Panel(content, title="📰 [bold cyan]Sentiment & Intelligence Live[/bold cyan]", border_style="cyan")

    def make_rl_panel(self) -> Panel:
        summary = self.engine.rl_engine.get_summary() if self.engine.rl_engine else {}
        total = summary.get("total_states_experienced", 0)
        accepted = summary.get("high_conviction_states", 0)
        pruned = summary.get("pruned_error_states", 0)
        
        with self.lock:
            indicators = dict(self.last_indicators)
            fng_val = self.fng_data.get("value", 50)
        
        # Discretizzazione vettoriale del regime live
        if indicators and self.engine.rl_engine:
            slope = indicators.get("ema_200_slope", 0.0)
            atr_pct = indicators.get("atr_pct", 1.0)
            rsi = indicators.get("rsi", 50.0)
            sample_token = RLExperienceEngine.get_state_key("SCANNER", fng_val, atr_pct, slope, rsi)
            parts = sample_token.split("|")[1:]
            token_display = " • ".join(parts)
            
            adx_val = float(indicators.get("adx", 25.0))
            if indicators.get("close", 0) > indicators.get("ema_200", 0) and adx_val >= 22.0:
                active_setup = "PULLBACK_REATTIVO"
            elif adx_val < 24.0 or indicators.get("close", 0) < indicators.get("ema_200", 0):
                active_setup = "MEAN_REVERSION_BOUNCE"
            else:
                active_setup = "PULLBACK_REATTIVO"
            pb_key = RLExperienceEngine.get_state_key(active_setup, fng_val, atr_pct, slope, rsi)
            should_take, reason, stats = self.engine.rl_engine.evaluate_proposal(pb_key)
            sizing = stats.get("sizing_multiplier", 1.0)
            sizing_fmt = f"{sizing:.2f}x" if sizing == 0.45 else f"{sizing:.1f}x"
            if should_take:
                if "ULTRA" in reason:
                    policy_verdict = f"[bold green]● AMMESSO (Ultra Convinzione {sizing_fmt})[/bold green]"
                elif "PROBED" in reason:
                    policy_verdict = f"[bold yellow]● PROBED RECOVERY ({sizing_fmt} size protetta)[/bold yellow]"
                else:
                    policy_verdict = f"[bold green]● AMMESSO (Alta Convinzione {sizing_fmt})[/bold green]"
            else:
                policy_verdict = f"[bold red]● FILTRO ATTIVO ({reason})[/bold red]"
        else:
            token_display = "Inizializzazione token..."
            policy_verdict = "[dim]In attesa dati candela...[/dim]"
        
        t = Table.grid(expand=True, padding=(0, 1))
        t.add_column(style="dim", justify="left")
        t.add_column(justify="right", style="bold")
        
        total = summary.get("total_states_experienced", 0)
        accepted = summary.get("high_conviction_states", 0)
        probed = summary.get("probed_recovery_states", 0)
        pruned = summary.get("pruned_error_states", 0)

        t.add_row("Regimi Storici Appresi:", f"[bold white]{total} pattern (10Y Backtest + Live)[/bold white]")
        t.add_row("Regimi ad Alta Convinzione:", f"[bold green]{accepted} ammessi (Sizing 1.0x-1.3x)[/bold green]")
        if probed > 0:
            t.add_row("Regimi in Probed Recovery:", f"[bold yellow]{probed} sbloccati via Half-Life Decay (0.45x)[/bold yellow]")
        t.add_row("Errori Bloccati (Mistake Memory):", f"[bold red]{pruned} regimi bloccati[/bold red]")
        t.add_row("Regime Attuale di Mercato:", f"[bold yellow]{token_display}[/bold yellow]")
        t.add_row("Valutazione Policy RL:", policy_verdict)
        t.add_row("Penalità Asimmetrica:", f"{Config.RL_MAX_CONSECUTIVE_ERRORS} errori max • 1.8x (R-Multiples)")
        t.add_row("Apprendimento Continuo:", "[bold green]Online Feedback Attivo ad ogni chiusura[/bold green]")
        
        return Panel(t, title="🧠 [bold magenta]Reinforcement Learning Engine[/bold magenta]", border_style="magenta")

    def make_api_panel(self) -> Panel:
        masked_key = self._mask_key(Config.API_KEY)
        masked_secret = "••••••••••••••••••••" if Config.API_SECRET else "[dim red]Non impostata[/dim red]"
        endpoint = "api-demo.bybit.eu (Spot)" if Config.DEMO_MODE else "api.bybit.eu (Prod)"
        
        t = Table.grid(expand=True, padding=(0, 1))
        t.add_column(style="dim", justify="left")
        t.add_column(justify="right", style="bold")
        
        t.add_row("Endpoint:", f"[cyan]{endpoint}[/cyan]")
        t.add_row("Bybit Key:", f"[white]{masked_key}[/white]")
        t.add_row("Secret Key:", f"[white]{masked_secret}[/white]")
        t.add_row("Modalità:", "[bold green]PAPER TRADING LOCALE[/bold green]")
        t.add_row("Fee & Slippage:", f"Fee: {Config.FEE_RATE*100:.2f}% | Slip: {Config.SLIPPAGE_PCT*100:.2f}%")
        t.add_row("Persistenza:", "[dim]SQLite WAL (paper_demo)[/dim]")
        t.add_row("Config:", "[bold yellow]Premi [E] per modificare[/bold yellow]")
        
        return Panel(t, title="🔑 [bold blue]Credenziali & Exchange[/bold blue]", border_style="blue")

    def make_events_table(self) -> Panel:
        events = self._get_events(limit=7)
        table = Table(box=None, expand=True, show_header=True, header_style="bold dim", padding=(0, 1))
        table.add_column("Ora", style="dim", width=9, justify="center")
        table.add_column("Tipo Evento", width=18)
        table.add_column("Dettagli", ratio=1, overflow="fold", no_wrap=False)
        
        if not events:
            table.add_row("-", "NESSUN_EVENTO", "Inizializzazione completata. In ascolto su candele orarie.")
        else:
            for ev in events:
                etype = ev["type"]
                style = "white"
                if "ENTRY" in etype:
                    style = "bold green"
                elif "EXIT" in etype:
                    details = ev.get("details", {})
                    pnl = float(details.get("pnl_quote", 0.0)) if isinstance(details, dict) else 0.0
                    style = "bold green" if pnl >= 0 else "bold red"
                elif "PRUNED" in etype or "BLOCKED" in etype:
                    style = "bold magenta"
                elif "LOCK" in etype or "SKIPPED" in etype or "PAUSED" in etype:
                    style = "bold yellow"
                elif "RESUMED" in etype:
                    style = "bold green"
                elif "ERROR" in etype:
                    style = "bold red"
                elif "NEWS" in etype:
                    style = "dim cyan"
                
                det_formatted = self._format_event_details(etype, ev["details"])
                table.add_row(ev["time"], f"[{style}]{etype}[/{style}]", det_formatted)
        
        return Panel(table, title="📜 [bold white]Registro Ultime Attività ed Eventi[/bold white]", border_style="dim")

    def make_radar_panel(self) -> Panel:
        """
        Radar Quantitativo Real-Time: visualizza la checklist analitica di tutti i requisiti
        dei Setup Long Tri-Engine (PULLBACK_REATTIVO, MOMENTUM_BREAKOUT, MEAN_REVERSION_BOUNCE), evidenziando le condizioni soddisfatte (PASS)
        e quelle non soddisfatte per cui il bot è attualmente in attesa (WAIT).
        """
        with self.lock:
            indicators = dict(self.last_indicators)
            fng_val = int(self.fng_data.get("value", 50))
            fng_class = str(self.fng_data.get("classification", "Neutral"))
            ticker_bid = float(self.last_ticker.get("bid", 0.0))
            ticker_ask = float(self.last_ticker.get("ask", 0.0))
            live_last = float(self.last_ticker.get("last", 0.0)) or ticker_bid
            with self.engine.db_lock:
                news_score = float(self.engine._get_state("current_news_score", "0.0") or 0.0)
                daily_locked = self.engine._get_state("daily_loss_locked") == "1"
                consec_losses = int(self.engine._get_state("consecutive_losses_today", "0") or 0)
                order = self.engine._open_order()

        close = float(indicators.get("close", live_last or 0.0))
        open_p = float(indicators.get("open", close))
        high_p = float(indicators.get("high", close))
        low_p = float(indicators.get("low", close))
        ema_20 = float(indicators.get("ema_20", 0.0))
        ema_50 = float(indicators.get("ema_50", 0.0))
        ema_100 = float(indicators.get("ema_100", 0.0))
        ema_200 = float(indicators.get("ema_200", 0.0))
        rsi = float(indicators.get("rsi", 50.0))
        rsi_7 = float(indicators.get("rsi_7", rsi))
        vol_ratio = float(indicators.get("vol_ratio", 1.0))
        prev_sqz = bool(indicators.get("prev_squeeze", False))
        curr_sqz = bool(indicators.get("squeeze", False))
        prev_high_20 = float(indicators.get("prev_high_20", 0.0))
        atr_pct = float(indicators.get("atr_pct", 1.0))
        slope = float(indicators.get("ema_200_slope", 0.0))

        # Radar Tri-Engine Quantitativo Real-Time
        # Setup 1: PULLBACK REATTIVO
        s1_trend_ok = ((close > ema_200) or ((ema_20 > ema_50) and (close > ema_50))) if (ema_50 > 0) else False
        s1_touch_ema = (
            (ema_20 > 0 and low_p <= ema_20 * 1.0045 and close >= ema_20 * 0.9955) or
            (ema_50 > 0 and low_p <= ema_50 * 1.0045 and close >= ema_50 * 0.9955) or
            (ema_100 > 0 and low_p <= ema_100 * 1.0045 and close >= ema_100 * 0.9955)
        )
        prev_rsi_7 = float(indicators.get("prev_rsi_7", 50.0))
        prev_high = float(indicators.get("prev_high", 0.0))
        s1_micro_dip = (prev_rsi_7 < 38.0) and (close > prev_high)
        s1_touch_ok = s1_touch_ema or s1_micro_dip
        s1_rsi_ok = (rsi < getattr(Config, "PULLBACK_RSI_MAX", 48.0)) or (rsi_7 < getattr(Config, "PULLBACK_RSI7_MAX", 38.0))
        close_loc = float(indicators.get("close_location", 0.5))
        s1_candle_ok = (close > open_p) and (close_loc >= 0.45)
        s1_vol_ok = (vol_ratio >= getattr(Config, "PULLBACK_MIN_VOL_RATIO", 0.90))
        pullback_ok = s1_trend_ok and s1_touch_ok and s1_rsi_ok and s1_candle_ok and s1_vol_ok

        # Setup 2: MOMENTUM & VOLATILITY BREAKOUT
        adx_val = float(indicators.get("adx", 25.0))
        prev_ema_50_3 = float(indicators.get("prev_ema_50_3", ema_50))
        prev_high_12 = float(indicators.get("prev_high_12", prev_high_20))
        s2_trend_ok = (close > ema_50) and ((close > ema_200) or (ema_20 > ema_50)) if (ema_50 > 0) else False
        s2_brk_ok = (close > prev_high_12) if prev_high_12 > 0 else False
        s2_vol_ok = (vol_ratio >= getattr(Config, "BREAKOUT_MIN_VOL_RATIO", 1.15))
        s2_exp_ok = prev_sqz or curr_sqz
        s2_rsi_ok = (getattr(Config, "BREAKOUT_RSI_MIN", 52.0) <= rsi <= getattr(Config, "BREAKOUT_RSI_MAX", 76.0))
        s2_adx_ok = (adx_val >= getattr(Config, "ADX_MIN_TREND", 22.0)) and (ema_50 > prev_ema_50_3)
        score = int(s2_brk_ok) + int(s2_vol_ok) + int(s2_exp_ok) + int(s2_rsi_ok)
        breakout_ok = s2_trend_ok and s2_brk_ok and (score >= getattr(Config, "BREAKOUT_MIN_SCORE", 2)) and (close > open_p) and (close_loc >= 0.50) and s2_adx_ok

        # Setup 3: MEAN-REVERSION & BEAR/RANGE BOUNCE
        bb_lower = float(indicators.get("bb_lower", 0.0))
        bb_lower_165 = float(indicators.get("bb_lower_165", bb_lower))
        bb_mid = float(indicators.get("bb_mid", ema_20))
        dist_ema20 = float(indicators.get("dist_ema20_pct", ((close - ema_20) / ema_20 * 100.0) if ema_20 > 0 else 0.0))
        r_low_3 = float(indicators.get("rolling_low_3", low_p))
        r_dist_3 = float(indicators.get("rolling_dist_ema20_3", dist_ema20))
        r_rsi_3 = float(indicators.get("rolling_rsi_3", rsi))
        r_rsi7_3 = float(indicators.get("rolling_rsi7_3", rsi_7))
        prev_low = float(indicators.get("prev_low", low_p))

        s3_macro_ok = (close < ema_200) or (ema_50 < ema_200) or (adx_val < 24.0) if (ema_200 > 0 and ema_50 > 0) else False
        s3_oversold_ok = (bb_lower_165 > 0 and r_low_3 <= bb_lower_165) or (bb_lower > 0 and r_low_3 <= bb_lower) or (r_dist_3 <= getattr(Config, "MEAN_REV_DIST_EMA20_MIN", -1.25))
        s3_rsi_ok = (r_rsi_3 < getattr(Config, "MEAN_REV_RSI_MAX", 40.0)) or (r_rsi7_3 < getattr(Config, "MEAN_REV_RSI7_MAX", 28.0))
        s3_reversal_ok = (close > open_p) and (close > prev_low) and (close < bb_mid)
        mean_rev_ok = s3_macro_ok and s3_oversold_ok and s3_rsi_ok and s3_reversal_ok

        # Macro filters
        news_crit_blocked = (news_score <= -0.30)
        news_breakout_blocked = (news_score <= -0.15)
        rl_blocked = False
        rl_reason = "OK"
        if self.engine.rl_engine and Config.RL_ACTIVE and ema_200 > 0:
            active_setup = "PULLBACK_REATTIVO" if pullback_ok else ("MOMENTUM_BREAKOUT" if breakout_ok else "MEAN_REVERSION_BOUNCE")
            st_key = RLExperienceEngine.get_state_key(active_setup, fng_val, atr_pct, slope, rsi)
            rl_ok, rl_reason, _ = self.engine.rl_engine.evaluate_proposal(st_key)
            rl_blocked = not rl_ok

        circuit_blocked = daily_locked or (consec_losses >= Config.PAPER_MAX_CONSECUTIVE_LOSSES)

        # Diagnosi di attesa
        waiting_reasons = []
        if not s1_trend_ok and not s3_macro_ok:
            waiting_reasons.append("EMA50/200 in calcolo")
        if not s1_touch_ok and s1_trend_ok:
            waiting_reasons.append("Prezzo distante da supporto EMA20/50/100")
        if not s2_brk_ok and prev_high_12 > 0:
            waiting_reasons.append(f"Sotto Max 12H (${prev_high_12:,.0f})")
        if not s3_oversold_ok and s3_macro_ok:
            waiting_reasons.append("Nessun eccesso ipervenduto (BB1.65 o RSI<40)")
        if fng_val < 35:
            waiting_reasons.append(f"F&G={fng_val} (Mercato in Paura)")

        term_w = self.console.width
        max_reasons = 6 if term_w >= 210 else (4 if term_w >= 165 else 3)
        reasons_summary = " • ".join(waiting_reasons[:max_reasons]) if waiting_reasons else "In attesa chiusura barra 1H"

        if order:
            status_banner = "[bold green]🟢 POSIZIONE APERTA IN CORSO[/bold green] • Gestione trailing stop e target attiva"
            border_col = "green"
        elif pullback_ok:
            status_banner = "[bold green]🚀 TRIGGER PULLBACK REATTIVO SODDISFATTO![/bold green] • Pronto per acquisto su chiusura barra"
            border_col = "green"
        elif breakout_ok:
            status_banner = "[bold green]🚀 TRIGGER MOMENTUM BREAKOUT SODDISFATTO![/bold green] • Pronto per acquisto su chiusura barra"
            border_col = "green"
        elif mean_rev_ok:
            status_banner = "[bold green]🚀 TRIGGER MEAN-REVERSION BOUNCE SODDISFATTO![/bold green] • Pronto per acquisto su chiusura barra"
            border_col = "green"
        else:
            status_banner = f"[bold yellow]🟡 BOT IN ATTESA DEI TRIGGER (0/3 Setup pronti)[/bold yellow] • [yellow]{reasons_summary}[/yellow]"
            border_col = "yellow"

        is_compact = term_w < 220
        if not is_compact:
            w_col1 = 30
            w_col2 = 38
            w_col3 = 31
            w_col4 = 31
            w_col5 = 31
        else:
            w_col1 = 20
            w_col2 = 29
            w_col3 = 28
            w_col4 = 28
            w_col5 = 28

        t = Table(expand=True, box=None, padding=(0, 1), show_header=True, header_style="bold cyan")
        t.add_column("Condizione / Parametro" if not is_compact else "Condizione", style="bold white", width=w_col1, no_wrap=True)
        t.add_column("Valore Reale (Barra/Live)" if not is_compact else "Valore Reale", style="yellow", width=w_col2, no_wrap=True)
        t.add_column("PULLBACK (S1)" if not is_compact else "PULLBACK", width=w_col3, no_wrap=True)
        t.add_column("BREAKOUT (S2)" if not is_compact else "BREAKOUT", width=w_col4, no_wrap=True)
        t.add_column("MEAN-REV (S3)" if not is_compact else "MEAN-REV", width=w_col5, no_wrap=True)
        t.add_column("Stato Live & Dettaglio di Standby" if not is_compact else "Stato Live", style="dim", ratio=1, no_wrap=True)

        # 1. Trend Strutturale Macro
        v_trend = f"${close:,.0f} | E50:{ema_50:,.0f} | E200:{ema_200:,.0f}" if not is_compact else f"${close:,.0f} | E50:{ema_50/1e3:.0f}k | E200:{ema_200/1e3:.0f}k"
        s_p_tr = "[bold green]✅ PASS (Trend OK)[/bold green]" if s1_trend_ok else f"[bold red]❌ WAIT (> ${ema_200:,.0f} o E20>50)[/bold red]"
        s_b_tr = "[bold green]✅ PASS (ADX>=22)[/bold green]" if (s2_trend_ok and s2_adx_ok) else f"[bold red]❌ WAIT (> ${ema_50:,.0f} & ADX≥22)[/bold red]"
        s_m_tr = "[bold green]✅ PASS (Range/Bear)[/bold green]" if s3_macro_ok else "[bold red]❌ WAIT (ADX<24 o <E200)[/bold red]"
        d_tr = "[green]Trend favorevole[/green]" if s1_trend_ok else ("[green]Regime Range (ADX < 24 - Setup 3 ON)[/green]" if (adx_val < 24.0 and not is_compact) else ("[green]Range (ADX<24, S3 ON)[/green]" if adx_val < 24.0 else "[yellow]Sotto EMA200[/yellow]"))
        t.add_row("1. Regime & Struttura 1H" if not is_compact else "1. Regime & Trend", v_trend, s_p_tr, s_b_tr, s_m_tr, d_tr)

        # 2. Supporto / Resistenza / Canale
        nearest_ema = ema_20 if (ema_20 > 0 and abs(low_p - ema_20) <= abs(low_p - ema_50)) else (ema_50 if ema_50 > 0 else ema_100)
        dip_target = nearest_ema * 1.0045 if nearest_ema > 0 else (low_p * 1.0045)
        target_bb = bb_lower_165 if bb_lower_165 > 0 else (ema_20 * 0.9875 if ema_20 > 0 else low_p * 0.98)

        v_sr = f"L:{low_p:,.0f} | H12:{prev_high_12:,.0f} | BB:{bb_lower_165:,.0f}" if not is_compact else f"L:{low_p/1e3:.0f}k | H12:{prev_high_12/1e3:.0f}k | BB:{bb_lower_165/1e3:.0f}k"
        s_p_sr = "[bold green]✅ PASS (Touch/Dip)[/bold green]" if s1_touch_ok else f"[bold red]❌ WAIT (Dip ≤ ${dip_target:,.0f})[/bold red]"
        s_b_sr = "[bold green]✅ PASS (Break 12H)[/bold green]" if s2_brk_ok else f"[bold red]❌ WAIT (Close > ${prev_high_12:,.0f})[/bold red]"
        s_m_sr = "[bold green]✅ PASS (Ipervenduto)[/bold green]" if s3_oversold_ok else f"[bold red]❌ WAIT (Low ≤ ${target_bb:,.0f})[/bold red]"
        d_sr = "[green]Trigger confermato[/green]" if (s1_touch_ok or s2_brk_ok or s3_oversold_ok) else ("[dim]Prezzo tra canali in attesa[/dim]" if not is_compact else "[dim]Tra i canali[/dim]")
        t.add_row("2. Livello di Trigger Prezzo" if not is_compact else "2. Livello Trigger", v_sr, s_p_sr, s_b_sr, s_m_sr, d_sr)

        # 3. Oscillatore RSI(14/7)
        v_rsi = f"RSI14: {rsi:.1f} | RSI7: {rsi_7:.1f}"
        s_p_rsi = "[bold green]✅ PASS (<48/38)[/bold green]" if s1_rsi_ok else "[yellow]❌ WAIT (RSI14<48 o 7<38)[/yellow]"
        s_b_rsi = "[bold green]✅ PASS (52-76)[/bold green]" if s2_rsi_ok else "[yellow]❌ WAIT (Target: 52-76)[/yellow]"
        s_m_rsi = "[bold green]✅ PASS (<40/28)[/bold green]" if s3_rsi_ok else "[yellow]❌ WAIT (RSI14<40 o 7<28)[/yellow]"
        d_rsi = "[green]RSI allineato con il regime[/green]" if (s1_rsi_ok or s2_rsi_ok or s3_rsi_ok) else "[dim]Oscillatore neutro[/dim]"
        t.add_row("3. Oscillatore RSI(14/7)" if not is_compact else "3. Oscillatore RSI", v_rsi, s_p_rsi, s_b_rsi, s_m_rsi, d_rsi)

        # 4. Direzione Candela 1H
        is_green = (close > open_p)
        c_dir = "Verde" if is_green else "Rossa"
        v_cnd = f"C:${close:,.0f} | O:${open_p:,.0f} ({c_dir})" if not is_compact else f"C:${close:,.0f} O:${open_p:,.0f}"
        s_p_cnd = "[bold green]✅ PASS (Verde)[/bold green]" if is_green else f"[bold red]❌ WAIT (Serve > ${open_p:,.0f})[/bold red]"
        s_b_cnd = "[bold green]✅ PASS (Verde)[/bold green]" if is_green else f"[bold red]❌ WAIT (Serve > ${open_p:,.0f})[/bold red]"
        s_m_cnd = "[bold green]✅ PASS (Reversal)[/bold green]" if s3_reversal_ok else f"[bold red]❌ WAIT (Serve > ${open_p:,.0f})[/bold red]"
        d_cnd = "[green]Candela confermata[/green]" if is_green else "[yellow]Candela ribassista[/yellow]"
        t.add_row("4. Direzione Candela 1H" if not is_compact else "4. Direzione Candela", v_cnd, s_p_cnd, s_b_cnd, s_m_cnd, d_cnd)

        # 5. Volume Surge & Ratio
        s1_min_v = getattr(Config, "PULLBACK_MIN_VOL_RATIO", 0.90)
        v_vol = f"{vol_ratio:.2f}x media 20"
        s_p_vol = f"[bold green]✅ PASS (≥ {s1_min_v:.2f}x)[/bold green]" if s1_vol_ok else f"[yellow]❌ WAIT (Serve ≥ {s1_min_v:.2f}x)[/yellow]"
        s_b_vol = "[bold green]✅ PASS (≥ 1.15x)[/bold green]" if s2_vol_ok else "[yellow]❌ WAIT (Serve ≥ 1.15x)[/yellow]"
        s_m_vol = "[dim]— Non vincolante[/dim]"
        d_vol = "[green]Volume partecipativo[/green]" if (s1_vol_ok or s2_vol_ok) else "[dim]Volumi nella norma[/dim]"
        t.add_row("5. Volume Surge 1H" if not is_compact else "5. Volume Surge", v_vol, s_p_vol, s_b_vol, s_m_vol, d_vol)

        # 6. Compressione / Espansione Volatilità
        sqz_str = "Si" if curr_sqz else "No"
        prev_sqz_str = "Si" if prev_sqz else "No"
        v_sqz = f"Sqz: {sqz_str} (Prec:{prev_sqz_str}) | ADX:{adx_val:.1f}" if not is_compact else f"Sqz:{sqz_str} | ADX:{adx_val:.1f}"
        s_p_sqz = "[dim]— Non vincolante[/dim]"
        s_b_sqz = f"[bold green]✅ PASS (Score {score}/4)[/bold green]" if (score >= getattr(Config, "BREAKOUT_MIN_SCORE", 2) and s2_adx_ok) else f"[yellow]❌ WAIT (Score {score}/2 | ADX≥22)[/yellow]"
        s_m_sqz = "[dim]— Non vincolante[/dim]"
        d_sqz = "[green]Espansione attiva[/green]" if s2_exp_ok else "[dim]Bande standard[/dim]"
        t.add_row("6. Bollinger Squeeze / Score" if not is_compact else "6. Squeeze / Score", v_sqz, s_p_sqz, s_b_sqz, s_m_sqz, d_sqz)

        # 7. Sentiment & Fear & Greed
        v_fng = f"{fng_val}/100 ({fng_class})"
        s_p_fng = "[bold green]✅ PASS (F&G≥25)[/bold green]" if fng_val >= 25 else "[yellow]❌ WAIT (Serve ≥ 25)[/yellow]"
        s_b_fng = "[bold green]✅ PASS (F&G≥35)[/bold green]" if fng_val >= 35 else "[yellow]❌ WAIT (Serve ≥ 35)[/yellow]"
        s_m_fng = "[bold green]✅ PASS (Contro-Trend)[/bold green]"
        d_fng = f"[dim]Allocazione F&G ({fng_val}/100)[/dim]"
        t.add_row("7. Fear & Greed Index" if not is_compact else "7. Fear & Greed", v_fng, s_p_fng, s_b_fng, s_m_fng, d_fng)

        # 8. Filtri Macro News / RL / Circuit Breakers
        v_gates = f"News: {news_score:+.3f} | Stop: {consec_losses}/2"
        if news_crit_blocked:
            s_p_gt = "[bold red]❌ NEWS CRITICAL[/bold red]"
            s_b_gt = "[bold red]❌ NEWS CRITICAL[/bold red]"
            s_m_gt = "[bold red]❌ NEWS CRITICAL[/bold red]"
            d_gt = "[bold red]News macro critiche[/bold red]"
        elif news_breakout_blocked:
            s_p_gt = "[bold green]✅ PASS (Gate OK)[/bold green]"
            s_b_gt = "[bold red]❌ WAIT (News>-0.15)[/bold red]"
            s_m_gt = "[bold green]✅ PASS (Gate OK)[/bold green]"
            d_gt = "[yellow]News negative: breakout disattivato[/yellow]"
        elif circuit_blocked:
            s_p_gt = "[bold red]❌ STOP COOLDOWN[/bold red]"
            s_b_gt = "[bold red]❌ STOP COOLDOWN[/bold red]"
            s_m_gt = "[bold red]❌ STOP COOLDOWN[/bold red]"
            d_gt = "[bold red]Stop giornaliero raggiunto[/bold red]"
        elif rl_blocked:
            s_p_gt = f"[bold magenta]❌ RL VETO[/bold magenta]"
            s_b_gt = "[bold green]✅ PASS (Gate OK)[/bold green]"
            s_m_gt = "[bold green]✅ PASS (Gate OK)[/bold green]"
            d_gt = f"[magenta]RL Policy: veto regime ({rl_reason[:15]})[/magenta]"
        else:
            s_p_gt = "[bold green]✅ PASS (Gate OK)[/bold green]"
            s_b_gt = "[bold green]✅ PASS (Gate OK)[/bold green]"
            s_m_gt = "[bold green]✅ PASS (Gate OK)[/bold green]"
            d_gt = "[green]Nessun blocco macro/RL (Circuit OK)[/green]" if not is_compact else "[green]Circuit OK[/green]"
        t.add_row("8. News / RL / Circuit Gate" if not is_compact else "8. News / RL Gate", v_gates, s_p_gt, s_b_gt, s_m_gt, d_gt)

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_row(Align.center(Text.from_markup(f"  {status_banner}  \n")))
        grid.add_row(t)

        return Panel(
            grid,
            title="🎯 [bold yellow]Radar Condizioni di Ingresso Long in Tempo Reale: Cosa Soddisfa (✅) vs Cosa Aspetta (❌)[/bold yellow]",
            border_style=border_col,
            subtitle="[dim]Valutazione esecutiva alla chiusura della candela oraria (60m) per prevenire falsi segnali intraday[/dim]"
        )

    def make_footer(self) -> Panel:
        t = Table.grid(expand=True)
        t.add_column(justify="center", ratio=1)
        controls = (
            "[bold green][S][/bold green] Avvia/Pausa Trading  |  "
            "[bold cyan][O][/bold cyan] Dettaglio Operazioni Settimana  |  "
            "[bold yellow][E][/bold yellow] Modifica API Key  |  "
            "[bold cyan][R][/bold cyan] Forza Refresh Totale (Prezzo/News/Indicatori)  |  "
            "[bold red][Q][/bold red] Esci  |  "
            "💼 [bold cyan]Ultime Operazioni Settimana: Live[/bold cyan]"
        )
        with self.lock:
            msg = getattr(self, "status_message", "")
            ts = getattr(self, "status_timestamp", datetime.min)
        if msg and (datetime.now() - ts).total_seconds() < 8:
            controls = f"[bold cyan]ℹ️  {msg}[/bold cyan]  •  " + controls

        t.add_row(Text.from_markup(controls))
        return Panel(t, style="dim white", padding=(0, 0))

    def build_dashboard(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )
        
        layout["body"].split_column(
            Layout(name="chart", ratio=14),
            Layout(name="top", ratio=10),
            Layout(name="trades", ratio=9),
            Layout(name="radar", ratio=12),
            Layout(name="mid", ratio=9),
            Layout(name="bottom", ratio=8),
        )
        
        layout["body"]["top"].split_row(
            Layout(name="wallet", ratio=1),
            Layout(name="position", ratio=1),
        )
        
        layout["body"]["mid"].split_row(
            Layout(name="sentiment", ratio=1),
            Layout(name="rl", ratio=1),
        )
        
        # Ratio 2:5 dà il 71% dello spazio orizzontale al Registro Eventi e mantiene compatte le credenziali
        layout["body"]["bottom"].split_row(
            Layout(name="api", ratio=2),
            Layout(name="events", ratio=5),
        )
        
        layout["header"].update(self.make_header())
        layout["body"]["chart"].update(self.make_chart_panel())
        layout["body"]["top"]["wallet"].update(self.make_wallet_panel())
        layout["body"]["top"]["position"].update(self.make_position_panel())
        layout["body"]["trades"].update(self.make_weekly_trades_panel())
        layout["body"]["radar"].update(self.make_radar_panel())
        layout["body"]["mid"]["sentiment"].update(self.make_sentiment_panel())
        layout["body"]["mid"]["rl"].update(self.make_rl_panel())
        layout["body"]["bottom"]["api"].update(self.make_api_panel())
        layout["body"]["bottom"]["events"].update(self.make_events_table())
        layout["footer"].update(self.make_footer())
        
        return layout

    def _worker_loop(self) -> None:
        """Loop di background per l'esecuzione del trading e polling dati Bybit."""
        last_step_time = 0.0
        while self.is_running:
            now = time.time()
            do_step = False
            is_forced = False
            
            if self.force_refresh_event.is_set():
                self.force_refresh_event.clear()
                do_step = True
                is_forced = bool(getattr(self, "force_full_refresh", False))
                self.force_full_refresh = False
            elif now - last_step_time >= Config.PAPER_POLL_SECONDS:
                do_step = True
            
            if do_step:
                try:
                    now_utc = datetime.now(timezone.utc)
                    if self.trading_active:
                        step_res = self.engine.step(force_news_poll=is_forced)
                        with self.lock:
                            self.last_step_data = step_res
                            self.last_ticker = {
                                "bid": step_res.get("bid", 0.0),
                                "ask": step_res.get("ask", 0.0),
                                "last": step_res.get("last", 0.0),
                                "high24h": step_res.get("high24h", 0.0),
                                "low24h": step_res.get("low24h", 0.0),
                                "change24h_pct": step_res.get("change24h_pct", 0.0),
                                "volume24h": step_res.get("volume24h", 0.0),
                                "turnover24h": step_res.get("turnover24h", 0.0),
                            }
                    else:
                        ticker = self.engine.client.get_spot_ticker(Config.SYMBOL)
                        rate = self.engine._eur_to_quote_rate()
                        with self.engine.db_lock:
                            self.engine._record_price(ticker, rate, now_utc)
                            if is_forced:
                                try:
                                    candles = self.engine.client.get_recent_klines(symbol=Config.SYMBOL, limit=1000)
                                    interval = pd.Timedelta(minutes=DataManager.TIMEFRAMES[str(Config.TIMEFRAME)][1] / 60000)
                                    now_naive = pd.Timestamp(now_utc).tz_convert(None)
                                    closed = candles[candles["timestamp"] + interval <= now_naive].reset_index(drop=True)
                                    self.engine._record_new_candles(closed)
                                except Exception:
                                    pass
                                self.engine._poll_news(now_utc, 0.0)
                        with self.lock:
                            self.last_ticker = ticker
                    
                    # Refresh Funding Rate (per derivati/sentiment)
                    if is_forced or (now - getattr(self, "_last_funding_poll", 0.0) >= 300.0):
                        try:
                            fr = self.engine.sentiment.fetch_market_funding_rate()
                            with self.lock:
                                self.last_funding_rate = fr
                            self._last_funding_poll = now
                        except Exception:
                            pass

                    # Refresh Fear & Greed Index
                    if is_forced or (now - getattr(self, "_last_fng_poll", 0.0) >= 600.0):
                        try:
                            fng_df = self.engine.sentiment.fetch_fear_and_greed_history()
                            if not fng_df.empty:
                                latest = fng_df.iloc[-1]
                                with self.lock:
                                    self.fng_data = {
                                        "value": int(latest.get("fng_value", 50)),
                                        "classification": str(latest.get("fng_classification", "Neutral"))
                                    }
                            self._last_fng_poll = now
                        except Exception:
                            pass

                    self._refresh_cached_news()
                    self._refresh_cached_indicators()
                    last_step_time = time.time()

                    if is_forced:
                        with self.lock:
                            t_str = datetime.now().strftime("%H:%M:%S")
                            self.status_message = f"Refresh totale completato alle {t_str} (Prezzo spot, Indicatori, 28 News, Funding)"
                            self.status_timestamp = datetime.now()
                except Exception as exc:
                    with self.lock:
                        self.status_message = f"Errore polling: {str(exc)[:60]}"
                        self.status_timestamp = datetime.now()
            
            time.sleep(1.0)

    def edit_api_credentials_dialog(self, live: Live) -> None:
        """Modalità interattiva per modificare le chiavi API Bybit salvandole su .env."""
        live.stop()
        self.console.clear()
        self.console.print("\n" + "=" * 65, style="bold cyan")
        self.console.print(" 🔑 MODIFICA CHIAVI API BYBIT EU", style="bold yellow")
        self.console.print("=" * 65 + "\n", style="bold cyan")
        self.console.print(f"API Key attuale   : {self._mask_key(Config.API_KEY)}")
        self.console.print(f"API Secret attuale: {'••••••••••••••••' if Config.API_SECRET else 'Non impostata'}\n")
        self.console.print("[dim]Premi INVIO direttamente su un campo per lasciarlo invariato.[/dim]\n")
        
        try:
            new_key = input("Inserisci Nuova BYBIT_API_KEY: ").strip()
            new_secret = input("Inserisci Nuova BYBIT_API_SECRET: ").strip()
            
            updated = False
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            env_content = ""
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    env_content = f.read()
            
            if new_key:
                Config.API_KEY = new_key
                os.environ["BYBIT_API_KEY"] = new_key
                updated = True
                if "BYBIT_API_KEY=" in env_content:
                    lines = [line if not line.startswith("BYBIT_API_KEY=") else f"BYBIT_API_KEY={new_key}" for line in env_content.splitlines()]
                    env_content = "\n".join(lines)
                else:
                    env_content += f"\nBYBIT_API_KEY={new_key}"
            
            if new_secret:
                Config.API_SECRET = new_secret
                os.environ["BYBIT_API_SECRET"] = new_secret
                updated = True
                if "BYBIT_API_SECRET=" in env_content:
                    lines = [line if not line.startswith("BYBIT_API_SECRET=") else f"BYBIT_API_SECRET={new_secret}" for line in env_content.splitlines()]
                    env_content = "\n".join(lines)
                else:
                    env_content += f"\nBYBIT_API_SECRET={new_secret}"
            
            if updated:
                with open(env_path, "w", encoding="utf-8") as f:
                    f.write(env_content.strip() + "\n")
                # Re-inizializza il client con le nuove credenziali
                self.engine.client = BybitClient()
                self.console.print("\n[bold green]✅ Credenziali API aggiornate con successo e salvate in .env![/bold green]")
            else:
                self.console.print("\n[yellow]Nessuna modifica applicata.[/yellow]")
            
            self.console.print("[dim]Rientro nella dashboard in corso...[/dim]")
            time.sleep(1.5)
        except Exception as exc:
            self.console.print(f"\n[bold red]Errore durante il salvataggio: {exc}[/bold red]")
            time.sleep(2.0)
        finally:
            self.console.clear()
            live.start()

    def confirm_pause_dialog(self, live: Live) -> bool:
        """Richiede conferma esplicita prima di mettere in pausa il bot, evitando pause involontarie."""
        live.stop()
        self.console.clear()
        self.console.print("\n" + "=" * 65, style="bold yellow")
        self.console.print(" ⏸  CONFERMA RICHIESTA PAUSA TRADING", style="bold red")
        self.console.print("=" * 65 + "\n", style="bold yellow")
        self.console.print("[bold white]Vuoi davvero METTERE IN PAUSA il trading automatico?[/bold white]")
        self.console.print("[dim]In pausa, il bot non aprirà nuove posizioni finché non verrà riattivato manualmente.[/dim]\n")
        self.console.print("Premi [bold red][P][/bold red] o [bold red][S][/bold red] e poi [bold white]INVIO[/bold white] per CONFERMARE la pausa.")
        self.console.print("Premi semplicemente [bold green]INVIO[/bold green] (o qualsiasi altro tasto) per ANNULLARE e continuare a tradare.\n")
        
        try:
            choice = input("Confermi la pausa? [p/Annulla]: ").strip().lower()
            return choice in {"p", "s", "si", "yes", "y"}
        except Exception:
            return False
        finally:
            self.console.clear()
            live.start()

    def confirm_exit_dialog(self, live: Live) -> bool:
        """Richiede conferma prima di chiudere completamente l'applicazione."""
        live.stop()
        self.console.clear()
        self.console.print("\n" + "=" * 65, style="bold red")
        self.console.print(" 🛑 CONFERMA CHIUSURA APPLICAZIONE", style="bold red")
        self.console.print("=" * 65 + "\n", style="bold red")
        self.console.print("Sei sicuro di voler uscire dal Bot? [s/N]: ", end="", style="bold white")
        try:
            choice = input().strip().lower()
            return choice in {"s", "si", "y", "yes"}
        except Exception:
            return False
        finally:
            self.console.clear()
            live.start()

    def show_trades_history_dialog(self, live: Live) -> None:
        """Visualizza a schermo intero lo storico completo delle operazioni con audit approfondito."""
        live.stop()
        self.console.clear()
        
        data = self._get_weekly_trades(days=7)
        trades = data["trades"]
        
        self.console.print("\n" + "=" * 90, style="bold cyan")
        self.console.print(" 💼 STORICO OPERAZIONI DELLA SETTIMANA - AUDIT DETTAGLIATO (BTC/USDC)", style="bold white")
        self.console.print("=" * 90 + "\n", style="bold cyan")
        
        tot = data["total"]
        closed = data["closed_count"]
        wins = data["wins"]
        losses = data["losses"]
        wr = data["win_rate"]
        realized_pnl = data["realized_pnl"]
        open_count = data["open_count"]
        unrealized_pnl = data["unrealized_pnl"]
        
        pnl_style = "bold green" if realized_pnl >= 0 else "bold red"
        pnl_sign = "+" if realized_pnl >= 0 else ""
        
        summary_grid = Table.grid(expand=True, padding=(0, 2))
        summary_grid.add_column(style="dim", justify="left")
        summary_grid.add_column(style="bold white", justify="left")
        summary_grid.add_column(style="dim", justify="left")
        summary_grid.add_column(style="bold white", justify="left")
        
        summary_grid.add_row(
            "Operazioni Totali Settimana:", f"{tot} (Concluse: {closed} • Aperte: {open_count})",
            "Win Rate Settimanale:", f"{wr:.1f}% ({wins} Vincenti / {losses} Perdenti)"
        )
        summary_grid.add_row(
            "PnL Netto Realizzato:", f"[{pnl_style}]{pnl_sign}{realized_pnl:.2f} USDC[/{pnl_style}]",
            "PnL Non Realizzato (Posizione):", f"[{pnl_style if unrealized_pnl >= 0 else 'bold red'}]{unrealized_pnl:+.2f} USDC[/{pnl_style if unrealized_pnl >= 0 else 'bold red'}]" if open_count > 0 else "Nessuna posizione aperta"
        )
        
        self.console.print(Panel(summary_grid, title="📊 [bold yellow]Riepilogo Prestazionale Settimana[/bold yellow]", border_style="yellow"))
        self.console.print()
        
        if not trades:
            self.console.print("[dim italic]Nessuna operazione registrata negli ultimi 7 giorni.[/dim italic]\n")
        else:
            t = Table(expand=True, show_header=True, header_style="bold cyan", padding=(0, 1))
            t.add_column("#", style="dim", width=4, justify="center")
            t.add_column("Data/Ora Ingr.", style="bold white", width=14, justify="center")
            t.add_column("Setup Operativo", width=16, justify="left")
            t.add_column("Side", width=6, justify="center")
            t.add_column("Quantità / Controvalore", width=22, justify="right")
            t.add_column("Prezzo Ingresso", width=16, justify="right")
            t.add_column("Prezzo Uscita / Live", width=20, justify="right")
            t.add_column("Target (SL / TP)", width=24, justify="center")
            t.add_column("Durata", width=10, justify="center")
            t.add_column("Esito / Motivo Uscita", width=22, justify="left")
            t.add_column("PnL Netto (USDC)", width=24, justify="right")
            t.add_column("Stato", width=10, justify="center")
            
            for tr in trades:
                tr_id = f"#{tr['id']}"
                setup_name = tr["setup"].replace("_BOUNCE", "").replace("_REACTIVE", "").replace("_BREAKOUT", "")
                if "MEAN_REV" in setup_name or "MEAN" in setup_name:
                    setup_label = "[bold magenta]MEAN-REV[/bold magenta]"
                elif "PULLBACK" in setup_name:
                    setup_label = "[bold cyan]PULLBACK[/bold cyan]"
                elif "MOMENTUM" in setup_name or "BREAKOUT" in setup_name:
                    setup_label = "[bold yellow]BREAKOUT[/bold yellow]"
                else:
                    setup_label = f"[bold white]{setup_name}[/bold white]"
                    
                side_label = f"[bold green]{tr['side']}[/bold green]" if tr["side"] == "LONG" else f"[bold red]{tr['side']}[/bold red]"
                size_label = f"{tr['qty']:.5f} BTC [dim](${tr['notional']:.1f})[/dim]"
                entry_p = f"${tr['entry_fill']:,.2f}"
                exit_p = tr["exit_display"]
                sl_tp = f"SL: ${tr['stop_price']:,.0f} | TP: ${tr['target_price']:,.0f}"
                dur = tr["duration_display"]
                reason = tr["exit_reason_label"]
                
                pnl_val = tr["pnl"]
                pnl_pct_val = tr["pnl_pct"]
                p_style = "bold green" if pnl_val >= 0 else "bold red"
                p_sign = "+" if pnl_val >= 0 else ""
                pnl_str = f"[{p_style}]{p_sign}{pnl_val:.2f} USDC ({p_sign}{pnl_pct_val:.2f}%)[/{p_style}]"
                
                status_label = "[bold green]APERTA[/bold green]" if tr["status"] == "OPEN" else "[dim]CHIUSA[/dim]"
                
                t.add_row(
                    tr_id,
                    tr["entry_display"],
                    setup_label,
                    side_label,
                    size_label,
                    entry_p,
                    exit_p,
                    sl_tp,
                    dur,
                    reason,
                    pnl_str,
                    status_label,
                )
            self.console.print(t)
            
        self.console.print("\n[bold white]Premi INVIO per tornare alla Dashboard in tempo reale...[/bold white] ", end="")
        try:
            input()
        except Exception:
            pass
        finally:
            self.console.clear()
            live.start()

    def run(self) -> None:
        """Ciclo principale dell'applicazione con gestione TTY non bloccante."""
        # Avvia worker thread per il trading in background
        worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        worker_thread.start()
        
        is_tty = sys.stdin.isatty()
        old_term_settings = None
        if is_tty:
            old_term_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        
        self._set_window_title("Quant Bot")
        try:
            with Live(self.build_dashboard(), refresh_per_second=2, console=self.console, screen=True) as live:
                self._set_window_title("Quant Bot")
                while self.is_running:
                    live.update(self.build_dashboard())
                    
                    if is_tty:
                        rlist, _, _ = select.select([sys.stdin], [], [], 0.3)
                        if rlist:
                            char = sys.stdin.read(1)

                            # 1. Filtro di sicurezza per sequenze di escape ANSI (frecce, scroll mouse, resize)
                            if char == "\x1b":
                                while True:
                                    r_esc, _, _ = select.select([sys.stdin], [], [], 0.05)
                                    if r_esc:
                                        sys.stdin.read(1)
                                    else:
                                        break
                                continue

                            # 2. Uscita con conferma esplicita
                            if char.lower() == "q":
                                if old_term_settings:
                                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term_settings)
                                if self.confirm_exit_dialog(live):
                                    self.is_running = False
                                    break
                                if is_tty:
                                    tty.setcbreak(sys.stdin.fileno())

                            # 3. Pausa / Ripresa del trading con conferma
                            elif char.lower() in ("s", "p"):
                                if self.trading_active:
                                    if old_term_settings:
                                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term_settings)
                                    if self.confirm_pause_dialog(live):
                                        with self.lock:
                                            self.trading_active = False
                                            with self.engine.db_lock:
                                                self.engine._log_event(
                                                    "BOT_PAUSED",
                                                    {"active": False, "reason": "manual_user_confirmation"},
                                                    datetime.now(timezone.utc)
                                                )
                                                self.engine.db.commit()
                                    if is_tty:
                                        tty.setcbreak(sys.stdin.fileno())
                                else:
                                    with self.lock:
                                        self.trading_active = True
                                        with self.engine.db_lock:
                                            self.engine._log_event(
                                                "BOT_RESUMED",
                                                {"active": True, "reason": "manual_user_resume"},
                                                datetime.now(timezone.utc)
                                            )
                                            self.engine.db.commit()

                            # 4. Forza refresh immediato (Prezzo, Indicatori, News, Funding, F&G)
                            elif char.lower() == "r":
                                self.force_full_refresh = True
                                with self.lock:
                                    self.status_message = "Refresh globale in corso (Prezzo spot, Indicatori, 28 feed news, Funding)..."
                                    self.status_timestamp = datetime.now()
                                self.force_refresh_event.set()

                            # 5. Modifica credenziali API
                            elif char.lower() == "e":
                                if old_term_settings:
                                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term_settings)
                                self.edit_api_credentials_dialog(live)
                                if is_tty:
                                    tty.setcbreak(sys.stdin.fileno())

                            # 6. Storico completo operazioni della settimana
                            elif char.lower() in ("o", "t"):
                                if old_term_settings:
                                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term_settings)
                                self.show_trades_history_dialog(live)
                                if is_tty:
                                    tty.setcbreak(sys.stdin.fileno())
                    else:
                        time.sleep(0.5)
        finally:
            if is_tty and old_term_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term_settings)
            self.is_running = False
            self.engine.close()
            self.console.print("\n[bold cyan]Bot di trading terminato correttamente.[/bold cyan]\n")


def main() -> None:
    app = TradingAppTUI()
    app.run()


if __name__ == "__main__":
    main()
