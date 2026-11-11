import math
import pandas as pd
from config import Config

import math
from typing import Optional
import numpy as np
import pandas as pd
from config import Config
from rl_experience_engine import RLExperienceEngine

class Backtester:
    def __init__(self, rules: dict, fee_rate: float = 0.001, slippage_pct: float = 0.0005, eur_to_quote_rate: float = 1.0):
        self.tick_size = rules["tick_size"]
        self.qty_step = rules["lot_size_filter"]["qtyStep"] if "lot_size_filter" in rules else rules["qty_step"]
        self.min_qty = rules["lot_size_filter"]["minOrderQty"] if "lot_size_filter" in rules else rules["min_qty"]
        self.qty_decimals = max(0, int(round(-math.log10(self.qty_step))))
        self.min_order_amount = rules.get("min_order_amount", 0.0)
        self.max_order_amount = rules.get("max_order_amount", 0.0)
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct
        if eur_to_quote_rate <= 0:
            raise ValueError("eur_to_quote_rate deve essere maggiore di zero.")
        self.eur_to_quote_rate = eur_to_quote_rate

    def _round_step(self, value: float, step: float) -> float:
        decimals = max(0, int(round(-math.log10(step))))
        return round(math.floor(value / step + 1e-9) * step, decimals)

    def run(
        self,
        df: pd.DataFrame,
        initial_capital: float = Config.BACKTEST_CAPITAL_USDC,
        sl_pct: float = Config.DEFAULT_SL_PCT,
        tp_pct: float = Config.DEFAULT_TP_PCT,
        verbose: bool = True,
        max_holding_hours: float = Config.MAX_HOLDING_HOURS,
        use_atr_stops: bool = False,
        atr_stop_multiplier: float = Config.ATR_STOP_MULTIPLIER,
        risk_reward_ratio: float = Config.RISK_REWARD_RATIO,
        volatility_target_pct: float | None = None,
        trail_tier1_trigger: float = Config.TRAIL_TIER1_TRIGGER,
        trail_tier1_lock: float = Config.TRAIL_TIER1_LOCK,
        trail_tier2_trigger: float = Config.TRAIL_TIER2_TRIGGER,
        trail_tier2_lock: float = Config.TRAIL_TIER2_LOCK,
        use_rl: bool = Config.RL_ACTIVE,
        rl_engine: Optional[RLExperienceEngine] = None,
        cooldown_bars_after_loss: int = 0,
        max_consecutive_losses: int = 0,
        max_daily_loss_pct: float = 0.0,
        dynamic_slippage: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
        if max_holding_hours <= 0:
            raise ValueError("max_holding_hours deve essere maggiore di zero.")
        if use_atr_stops and (atr_stop_multiplier <= 0 or risk_reward_ratio <= 0):
            raise ValueError("I parametri ATR e risk/reward devono essere maggiori di zero.")
        if use_atr_stops and "atr" not in df.columns:
            raise ValueError("Gli stop ATR richiedono la colonna 'atr' nei segnali.")
        if volatility_target_pct is not None and (volatility_target_pct <= 0 or "atr_pct" not in df.columns):
            raise ValueError("Il sizing per volatilita richiede un target positivo e la colonna 'atr_pct'.")

        if use_rl and rl_engine is None:
            rl_engine = RLExperienceEngine(
                min_experience_trials=2,
                min_state_win_rate=Config.RL_MIN_STATE_WR,
                max_consecutive_errors=Config.RL_MAX_CONSECUTIVE_ERRORS,
                learning_rate=Config.RL_LEARNING_RATE
            )

        capital = initial_capital
        peak_capital = initial_capital
        max_drawdown = 0.0
        trades_log = []

        consecutive_losses = 0
        last_loss_exit_idx = -999999
        current_day_str = ""
        day_start_capital = initial_capital
        daily_locked = False

        max_holding_time = pd.Timedelta(hours=max_holding_hours)
        timeframe_minutes = {"15": 15, "60": 60, "240": 240, "D": 1440}[str(Config.TIMEFRAME)]
        candle_interval = pd.Timedelta(minutes=timeframe_minutes)

        in_pos = False
        entry_price = sl_price = tp_price = qty = initial_sl_price = 0.0
        entry_time = None
        entry_idx = 0
        setup_name = ""
        current_state_key = ""
        trailing_tier = 0
        cur_holding_time = max_holding_time

        # Fast NumPy arrays for 100x execution speed
        timestamps = df["timestamp"].values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        signals = df["signal"].values
        setups = df["setup_type"].values
        atrs = df["atr"].values if "atr" in df.columns else np.zeros(len(df))
        atr_pcts = df["atr_pct"].values if "atr_pct" in df.columns else np.zeros(len(df))
        rl_states = df["rl_state"].values if "rl_state" in df.columns else ["" for _ in range(len(df))]
        n = len(df)

        def close_position(exit_ts, exit_price: float, exit_reason: str) -> None:
            nonlocal capital, peak_capital, max_drawdown, in_pos, consecutive_losses, last_loss_exit_idx
            gross_pnl = (exit_price - entry_price) * qty
            fees = (entry_price * qty * self.fee_rate) + (exit_price * qty * self.fee_rate)
            net_pnl = gross_pnl - fees
            capital += net_pnl

            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)

            is_win = (net_pnl > 0)
            if is_win:
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                last_loss_exit_idx = entry_idx

            # Online Reinforcement Learning Feedback
            if use_rl and rl_engine is not None and current_state_key:
                ret_pct = (net_pnl / (entry_price * qty)) * 100.0 if (entry_price * qty) > 0 else 0.0
                initial_risk_per_unit = max(0.0001, abs(entry_price - initial_sl_price)) if initial_sl_price > 0 else (entry_price * Config.DEFAULT_SL_PCT)
                r_multiple = (exit_price - entry_price) / initial_risk_per_unit
                rl_engine.record_feedback(current_state_key, is_win, ret_pct, r_multiple=r_multiple, timestamp=exit_ts)

            trades_log.append({
                "entry_time": entry_time,
                "exit_time": pd.to_datetime(exit_ts),
                "side": "LONG",
                "setup": setup_name,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "qty": round(qty, self.qty_decimals),
                "exit_reason": exit_reason,
                "net_pnl": round(net_pnl, 2),
                "capital": round(capital, 2)
            })
            in_pos = False

        for i in range(1, n):
            has_data_gap = (timestamps[i] - timestamps[i - 1]) > np.timedelta64(int(candle_interval.total_seconds()), 's')
            o = opens[i]
            h = highs[i]
            l = lows[i]
            c = closes[i]

            # Day tracking for daily loss circuit breaker
            day_str = str(timestamps[i])[:10]
            if day_str != current_day_str:
                current_day_str = day_str
                day_start_capital = capital
                daily_locked = False

            if max_daily_loss_pct > 0.0 and day_start_capital > 0:
                current_day_loss = (day_start_capital - capital) / day_start_capital
                if current_day_loss >= max_daily_loss_pct:
                    daily_locked = True

            # Dynamic vs Static Slippage
            if dynamic_slippage:
                cur_atr = float(atr_pcts[i - 1]) if i > 0 and math.isfinite(atr_pcts[i - 1]) else 1.0
                eff_slippage = max(self.slippage_pct, min(0.003, 0.05 * (cur_atr / 100.0)))
            else:
                eff_slippage = self.slippage_pct

            if in_pos:
                exit_price = None
                exit_reason = ""

                # Multi-Tier Dynamic Trailing Stop (evaluated from previous bar)
                prev_high_gain = (highs[i - 1] - entry_price) / entry_price
                if trailing_tier == 0 and prev_high_gain >= trail_tier1_trigger:
                    sl_price = self._round_step(entry_price * (1.0 + trail_tier1_lock), self.tick_size)
                    trailing_tier = 1
                if trailing_tier == 1 and prev_high_gain >= trail_tier2_trigger:
                    sl_price = self._round_step(entry_price * (1.0 + trail_tier2_lock), self.tick_size)
                    trailing_tier = 2

                # Exit checks
                if has_data_gap:
                    exit_price = o * (1 - eff_slippage)
                    exit_reason = "DATA_GAP_EXIT"
                elif l <= sl_price:
                    exit_price = sl_price * (1 - eff_slippage)
                    exit_reason = "TRAIL_TIER2" if trailing_tier == 2 else ("TRAIL_TIER1" if trailing_tier == 1 else "STOP_LOSS")
                elif h >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TAKE_PROFIT"
                elif (timestamps[i] - timestamps[entry_idx]) >= np.timedelta64(int(cur_holding_time.total_seconds()), 's'):
                    exit_price = c * (1 - eff_slippage)
                    exit_reason = "TIME_EXIT"

                if exit_price is not None:
                    close_position(timestamps[i], exit_price, exit_reason)

            is_circuit_blocked = (
                daily_locked or
                (max_consecutive_losses > 0 and consecutive_losses >= max_consecutive_losses) or
                (cooldown_bars_after_loss > 0 and (i - last_loss_exit_idx) <= cooldown_bars_after_loss)
            )

            if not in_pos and not has_data_gap and not is_circuit_blocked and signals[i - 1] == 1:
                # RL Evaluator Gate: Learn from mistakes, reject low-conviction regimes
                st_key = rl_states[i - 1]
                if not st_key and setups[i - 1]:
                    st_key = RLExperienceEngine.get_state_key(
                        setups[i - 1], 50.0, float(atr_pcts[i - 1]), 0.0, 50.0
                    )

                sizing_mult = 1.0
                if use_rl and rl_engine is not None and st_key:
                    should_take, _, stats = rl_engine.evaluate_proposal(st_key)
                    if not should_take:
                        continue
                    if Config.RL_DYNAMIC_SIZING:
                        sizing_mult = float(stats.get("sizing_multiplier", 1.0))

                raw_entry = o * (1 + eff_slippage)
                entry_price = self._round_step(raw_entry, self.tick_size)

                if use_atr_stops:
                    atr_distance = float(atrs[i - 1]) * atr_stop_multiplier
                    stop_pct = min(
                        max(atr_distance / entry_price, Config.MIN_STOP_PCT),
                        Config.MAX_STOP_PCT,
                    )
                    sl_price = self._round_step(entry_price * (1 - stop_pct), self.tick_size)
                    tp_price = self._round_step(
                        entry_price + (entry_price - sl_price) * risk_reward_ratio,
                        self.tick_size,
                    )
                else:
                    if setups[i - 1] == "MEAN_REVERSION_BOUNCE":
                        setup_sl = getattr(Config, "MEAN_REV_SL_PCT", 0.015)
                        setup_tp = getattr(Config, "MEAN_REV_TP_PCT", 0.024)
                        cur_holding_time = pd.Timedelta(hours=getattr(Config, "MEAN_REV_MAX_HOLDING_HOURS", 12.0))
                    else:
                        setup_sl = sl_pct
                        setup_tp = tp_pct
                        cur_holding_time = max_holding_time
                    sl_price = self._round_step(entry_price * (1 - setup_sl), self.tick_size)
                    tp_price = self._round_step(entry_price * (1 + setup_tp), self.tick_size)

                initial_sl_price = sl_price
                trailing_tier = 0
                risk_scale = 1.0
                if volatility_target_pct is not None:
                    current_atr_pct = float(atr_pcts[i - 1])
                    if math.isfinite(current_atr_pct) and current_atr_pct > 0:
                        risk_scale = min(1.0, volatility_target_pct / current_atr_pct)
                risk_amount = capital * Config.RISK_PER_TRADE * risk_scale * sizing_mult
                sl_distance = entry_price - sl_price
                if sl_distance <= 0:
                    continue
                raw_qty = risk_amount / sl_distance
                exchange_max = self.max_order_amount if self.max_order_amount > 0 else (Config.MAX_ORDER_NOTIONAL_USDC if Config.QUOTE_ASSET == 'USDC' else Config.MAX_ORDER_NOTIONAL_EUR)
                min_notional = max(
                    Config.MIN_ORDER_NOTIONAL_USDC if Config.QUOTE_ASSET == 'USDC' else (Config.MIN_ORDER_NOTIONAL_EUR * self.eur_to_quote_rate),
                    self.min_order_amount,
                )
                if getattr(Config, "RL_COMPOUNDING", True):
                    base_alloc = getattr(Config, "BASE_POSITION_PCT", 0.75)
                    target_notional = capital * base_alloc * sizing_mult
                    max_notional = min(capital * 0.98, exchange_max)
                    target_notional = min(target_notional, max_notional)
                    raw_qty = target_notional / entry_price
                else:
                    max_notional = min(
                        Config.MAX_ORDER_NOTIONAL_USDC if Config.QUOTE_ASSET == 'USDC' else (Config.MAX_ORDER_NOTIONAL_EUR * self.eur_to_quote_rate),
                        exchange_max,
                    )
                    raw_qty = min(raw_qty, max_notional / entry_price)
                qty = self._round_step(raw_qty, self.qty_step)

                notional_quote = entry_price * qty
                if qty >= self.min_qty and notional_quote >= min_notional and notional_quote <= capital:
                    in_pos = True
                    entry_time = pd.to_datetime(timestamps[i])
                    entry_idx = i
                    setup_name = setups[i - 1]
                    current_state_key = st_key

            if in_pos:
                unrealized_fees = (entry_price * qty + c * qty) * self.fee_rate
                equity = capital + (c - entry_price) * qty - unrealized_fees
                peak_capital = max(peak_capital, equity)
                drawdown = (peak_capital - equity) / peak_capital
                max_drawdown = max(max_drawdown, drawdown)

        if in_pos:
            final_exit = closes[-1] * (1 - eff_slippage)
            close_position(timestamps[-1], final_exit, "END_OF_DATA")

        trades_df = pd.DataFrame(trades_log)
        metrics = self._compute_metrics(trades_df, initial_capital, capital, max_drawdown)
        if use_rl and rl_engine is not None:
            rl_engine.save_memory()
        if verbose:
            self._print_summary(
                trades_df, metrics, sl_pct, tp_pct, max_holding_hours,
                use_atr_stops, atr_stop_multiplier, risk_reward_ratio,
                rl_engine=rl_engine
            )
        return trades_df, metrics

    def _compute_metrics(self, trades_df: pd.DataFrame, initial_capital: float, final_capital: float, max_dd: float) -> dict:
        if trades_df.empty:
            return {
                "final_capital": initial_capital, "return_pct": 0.0, "profit_factor": 0.0,
                "win_rate": 0.0, "max_dd": 0.0, "trades": 0, "cagr_pct": 0.0,
                "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0,
                "max_underwater_days": 0, "expectancy_usdc": 0.0, "win_loss_ratio": 0.0,
                "market_exposure_pct": 0.0, "annual_breakdown": []
            }
        wins = trades_df[trades_df["net_pnl"] > 0]
        losses = trades_df[trades_df["net_pnl"] <= 0]
        gross_profit = float(wins["net_pnl"].sum())
        gross_loss = float(abs(losses["net_pnl"].sum()))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        entry_t = pd.to_datetime(trades_df["entry_time"])
        exit_t = pd.to_datetime(trades_df["exit_time"])
        start_date = entry_t.min()
        end_date = exit_t.max()
        years = max(0.01, (end_date - start_date).total_seconds() / (365.25 * 86400))
        cagr = ((final_capital / initial_capital) ** (1.0 / years) - 1.0) * 100.0

        daily_pnl = trades_df.set_index(exit_t).resample("D")["net_pnl"].sum()
        all_days = pd.date_range(start_date.floor("D"), end_date.ceil("D"), freq="D")
        daily_series = daily_pnl.reindex(all_days, fill_value=0.0)
        daily_equity = initial_capital + daily_series.cumsum()
        daily_returns = daily_equity.pct_change().dropna()

        mean_ret = float(daily_returns.mean()) * 365.0
        std_ret = float(daily_returns.std()) * np.sqrt(365.0)
        sharpe = (mean_ret / std_ret) if std_ret > 0 else 0.0

        downside = daily_returns[daily_returns < 0]
        std_downside = float(downside.std()) * np.sqrt(365.0)
        sortino = (mean_ret / std_downside) if std_downside > 0 else 0.0

        calmar = (cagr / (max_dd * 100.0)) if max_dd > 0 else 0.0

        running_max = daily_equity.cummax()
        is_underwater = daily_equity < running_max
        underwater_durations = []
        cur_dur = 0
        for uw in is_underwater:
            if uw:
                cur_dur += 1
            else:
                if cur_dur > 0:
                    underwater_durations.append(cur_dur)
                cur_dur = 0
        if cur_dur > 0:
            underwater_durations.append(cur_dur)
        max_underwater_days = max(underwater_durations) if underwater_durations else 0

        avg_win = float(wins["net_pnl"].mean()) if not wins.empty else 0.0
        avg_loss = float(abs(losses["net_pnl"].mean())) if not losses.empty else 0.0
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else float("inf")
        expectancy = float(trades_df["net_pnl"].mean())

        total_holding_hours = float((exit_t - entry_t).dt.total_seconds().sum() / 3600.0)
        total_period_hours = float((end_date - start_date).total_seconds() / 3600.0)
        exposure_pct = (total_holding_hours / total_period_hours * 100.0) if total_period_hours > 0 else 0.0

        # Annual breakdown
        temp_df = trades_df.copy()
        temp_df["year"] = exit_t.dt.year
        annual_records = []
        for yr, grp in temp_df.groupby("year"):
            w = grp[grp["net_pnl"] > 0]
            l = grp[grp["net_pnl"] <= 0]
            gp = float(w["net_pnl"].sum())
            gl = float(abs(l["net_pnl"].sum()))
            yr_pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0)
            yr_wr = (len(w) / len(grp) * 100.0) if len(grp) > 0 else 0.0
            annual_records.append({
                "year": int(yr),
                "trades": len(grp),
                "net_pnl": round(float(grp["net_pnl"].sum()), 2),
                "win_rate_pct": round(yr_wr, 1),
                "profit_factor": round(yr_pf, 2),
            })

        return {
            "final_capital": final_capital,
            "return_pct": ((final_capital / initial_capital) - 1.0) * 100.0,
            "profit_factor": pf,
            "win_rate": (len(wins) / len(trades_df)) * 100.0,
            "max_dd": max_dd * 100.0,
            "trades": len(trades_df),
            "cagr_pct": cagr,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_underwater_days": max_underwater_days,
            "expectancy_usdc": expectancy,
            "win_loss_ratio": win_loss_ratio,
            "market_exposure_pct": exposure_pct,
            "annual_breakdown": annual_records,
        }

    def _print_summary(
        self,
        trades_df: pd.DataFrame,
        m: dict,
        sl: float,
        tp: float,
        max_holding_hours: float,
        use_atr_stops: bool,
        atr_stop_multiplier: float,
        risk_reward_ratio: float,
        rl_engine: Optional[RLExperienceEngine] = None
    ):
        stop_target = (
            f"SL: {atr_stop_multiplier:g}x ATR | TP: {risk_reward_ratio:g}R"
            if use_atr_stops
            else f"SL: {sl*100:.1f}% | TP: {tp*100:.1f}% | Trailing Tier1: {Config.TRAIL_TIER1_TRIGGER*100:.1f}%->{Config.TRAIL_TIER1_LOCK*100:.2f}% | Tier2: {Config.TRAIL_TIER2_TRIGGER*100:.1f}%->{Config.TRAIL_TIER2_LOCK*100:.2f}%"
        )
        print("\n" + "=" * 65)
        print(f" QUANTITATIVE AUDIT & BACKTEST REPORT: {Config.SYMBOL}")
        print(f" {stop_target} | HOLD MAX: {max_holding_hours:g}h")
        print("=" * 65)
        if trades_df.empty:
            print("Nessuna operazione eseguita.")
            return
        print(f"Periodo Analizzato   : {trades_df['entry_time'].iloc[0]} -> {trades_df['exit_time'].iloc[-1]}")
        print(f"Totale Operazioni    : {m['trades']}")
        print(f"Win Rate             : {m['win_rate']:.2f}% (Target >= 70%: {'✅ RAGGIUNTO' if m['win_rate'] >= 70.0 else '❌ NON RAGGIUNTO'})")
        print(f"Profit Factor        : {m['profit_factor']:.2f}")
        print(f"Max Drawdown         : {m['max_dd']:.2f}% (Target <= 8%: {'✅ RAGGIUNTO' if m['max_dd'] <= 8.0 else '❌ NON RAGGIUNTO'})")
        print(f"Capitale Finale      : {m['final_capital']:.2f} {Config.QUOTE_ASSET} ({m['return_pct']:+.2f}%)")
        print(f"CAGR                 : {m.get('cagr_pct', 0.0):.2f}%")
        print(f"Sharpe Ratio (Ann.)  : {m.get('sharpe_ratio', 0.0):.2f}")
        print(f"Sortino Ratio (Ann.) : {m.get('sortino_ratio', 0.0):.2f}")
        print(f"Calmar Ratio         : {m.get('calmar_ratio', 0.0):.2f}")
        print(f"Max Underwater Days  : {m.get('max_underwater_days', 0)} giorni")
        print(f"Expectancy per trade : {m.get('expectancy_usdc', 0.0):+.2f} {Config.QUOTE_ASSET}")
        print(f"Win / Loss Ratio     : {m.get('win_loss_ratio', 0.0):.2f}")
        print(f"Esposizione a Mercato: {m.get('market_exposure_pct', 0.0):.2f}% del tempo")
        print("-" * 65)
        print("Rendimento per Tipologia di Setup:")
        setup_agg = trades_df.groupby("setup")["net_pnl"].agg(
            Conteggio="count",
            Totale_USDC="sum",
            Media_USDC="mean",
            WinRate_Pct=lambda s: (s > 0).mean() * 100,
        )
        print(setup_agg.to_string())
        print("-" * 65)
        print("Dettaglio Chiusure per Motivo di Uscita:")
        exit_agg = trades_df.groupby("exit_reason")["net_pnl"].agg(
            Conteggio="count",
            Totale_USDC="sum",
            Media_USDC="mean",
        )
        print(exit_agg.to_string())
        if m.get("annual_breakdown"):
            print("-" * 65)
            print("Performance Anno per Anno:")
            ann_df = pd.DataFrame(m["annual_breakdown"])
            print(ann_df.to_string(index=False))
        if rl_engine is not None:
            print("-" * 65)
            summary = rl_engine.get_summary()
            print(f"RL Experience Memory : {summary['total_states_experienced']} regimi memorizzati | {summary['high_conviction_states']} ammessi | {summary['pruned_error_states']} potati per errori")
        print("=" * 65)
