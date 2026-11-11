from typing import Optional
import numpy as np
import pandas as pd
from config import Config
from rl_experience_engine import RLExperienceEngine

class QuantStrategy:
    def compute_indicators(self, df: pd.DataFrame, fng_df: pd.DataFrame = None) -> pd.DataFrame:
        df = df.copy()
        df["date"] = df["timestamp"].dt.floor("D")
        if fng_df is not None and not fng_df.empty:
            df = df.merge(fng_df[["date", "fng_value"]], on="date", how="left")
            df["fng_value"] = df["fng_value"].ffill().fillna(50.0)
        else:
            df["fng_value"] = 50.0

        # Moving Averages: Fast, Medium, Institutional, Long-term
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_100"] = df["close"].ewm(span=100, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
        df["ema_800"] = df["close"].ewm(span=800, adjust=False).mean()
        df["ema_200_slope"] = df["ema_200"].pct_change(24) * 100

        # Relative Strength Index (RSI 14 & fast RSI 7)
        delta = df["close"].diff()
        gain_14 = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss_14 = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs_14 = gain_14 / loss_14.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs_14))

        gain_7 = delta.clip(lower=0).ewm(alpha=1/7, adjust=False).mean()
        loss_7 = (-delta.clip(upper=0)).ewm(alpha=1/7, adjust=False).mean()
        rs_7 = gain_7 / loss_7.replace(0, np.nan)
        df["rsi_7"] = 100 - (100 / (1 + rs_7))

        # Average True Range (ATR 14)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
        df["atr_pct"] = (df["atr"] / df["close"]) * 100

        # Bollinger Bands (20, 2.0 & 2.2 std & 1.8/1.65 std)
        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2.0 * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - 2.0 * df["bb_std"]
        df["bb_lower_18"] = df["bb_mid"] - 1.8 * df["bb_std"]
        df["bb_lower_165"] = df["bb_mid"] - 1.65 * df["bb_std"]
        df["bb_lower_22"] = df["bb_mid"] - 2.2 * df["bb_std"]

        # Volatility Squeeze Metric
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
        df["squeeze"] = df["bb_width"] < df["bb_width"].rolling(30).mean()

        # Rolling Extremes & Volume
        df["high_12"] = df["high"].rolling(12).max()
        df["high_20"] = df["high"].rolling(20).max()
        df["low_20"] = df["low"].rolling(20).min()
        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"].replace(0, np.nan)
        df["dist_ema20_pct"] = (df["close"] - df["ema_20"]) / df["ema_20"] * 100

        # Intraday Candle Morphology (Rejection / Absorption metrics)
        candle_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["close_location"] = ((df["close"] - df["low"]) / candle_range).fillna(0.5)
        df["lower_wick"] = ((np.minimum(df["open"], df["close"]) - df["low"]) / candle_range).fillna(0.0)

        # Average Directional Index (ADX 14) - Market Trend Strength
        up_move = df["high"] - df["high"].shift(1)
        down_move = df["low"].shift(1) - df["low"]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr_smooth = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr_smooth.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr_smooth.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0.0)

        # Choppiness Index (CHOP 14) - Consolidation vs Trending Market Regime
        tr_sum = tr.rolling(14).sum()
        high_14 = df["high"].rolling(14).max()
        low_14 = df["low"].rolling(14).min()
        range_14 = (high_14 - low_14).replace(0, np.nan)
        chop_ratio = (tr_sum / range_14).replace(0, np.nan)
        df["chop"] = (100 * np.log10(chop_ratio.clip(lower=1e-6)) / np.log10(14)).fillna(50.0)

        indicator_columns = [
            "ema_20", "ema_50", "ema_100", "ema_200", "ema_800", "ema_200_slope",
            "rsi", "atr", "atr_pct", "bb_mid", "bb_std", "bb_upper",
            "bb_lower", "bb_lower_18", "bb_lower_165", "bb_lower_22", "squeeze", "high_20", "vol_sma", "vol_ratio",
            "adx", "chop",
        ]
        return df.dropna(subset=indicator_columns).reset_index(drop=True)

    def generate_signals(
        self,
        df: pd.DataFrame,
        fng_df: pd.DataFrame = None,
        live_news_score: float = 0.0,
        live_funding_rate: float = 0.0,
        rl_engine: Optional[RLExperienceEngine] = None
    ) -> pd.DataFrame:
        df = self.compute_indicators(df, fng_df)
        df["signal"] = 0
        df["setup_type"] = ""
        df["rl_state"] = ""
        df["rl_sizing_multiplier"] = 1.0

        if getattr(Config, "TRI_ENGINE_MODE", True):
            # =========================================================================
            # TRI-ENGINE MULTI-REGIME HIGH-ROTATION ARCHITECTURE (Setups 1, 2, 3)
            # =========================================================================

            # SETUP 1: Trend Pullback Reattivo
            # Operates in macro bull or nascent uptrend (EMA20 > EMA50, close > EMA50 if under EMA200)
            s1_macro = (df["close"] > df["ema_200"]) | ((df["ema_20"] > df["ema_50"]) & (df["close"] > df["ema_50"]))
            s1_touch_ema = (
                ((df["low"] <= df["ema_20"] * 1.0045) & (df["close"] >= df["ema_20"] * 0.9955)) |
                ((df["low"] <= df["ema_50"] * 1.0045) & (df["close"] >= df["ema_50"] * 0.9955)) |
                ((df["low"] <= df["ema_100"] * 1.0045) & (df["close"] >= df["ema_100"] * 0.9955))
            )
            s1_micro_dip = (df["rsi_7"].shift(1) < 38.0) & (df["close"] > df["high"].shift(1))
            s1_touch = s1_touch_ema | s1_micro_dip
            s1_rsi = (df["rsi"] < getattr(Config, "PULLBACK_RSI_MAX", 48.0)) | (df["rsi_7"] < getattr(Config, "PULLBACK_RSI7_MAX", 38.0))
            s1_candle = (df["close"] > df["open"]) & (df["close_location"] >= 0.45)
            s1_vol = df["vol_ratio"] >= getattr(Config, "PULLBACK_MIN_VOL_RATIO", 0.90)
            setup1_active = s1_macro & s1_touch & s1_rsi & s1_candle & s1_vol

            # SETUP 2: Momentum & Volatility Breakout
            # Operates during structural trend or compression expansion (Anti-Choppy Filter: ADX >= 22.0 & EMA50 Rising)
            s2_macro = (df["close"] > df["ema_50"]) & ((df["close"] > df["ema_200"]) | (df["ema_20"] > df["ema_50"]))
            c_brk = (df["close"] > df["high_20"].shift(1)) | (df["close"] > df["high_12"].shift(1))
            c_vol = df["vol_ratio"] >= getattr(Config, "BREAKOUT_MIN_VOL_RATIO", 1.15)
            c_exp = df["squeeze"].shift(1) | (df["bb_width"] > df["bb_width"].rolling(10).mean())
            c_rsi = (df["rsi"] >= getattr(Config, "BREAKOUT_RSI_MIN", 52.0)) & (df["rsi"] <= getattr(Config, "BREAKOUT_RSI_MAX", 76.0))
            score = c_brk.astype(int) + c_vol.astype(int) + c_exp.astype(int) + c_rsi.astype(int)
            s2_adx_trend = (df["adx"] >= getattr(Config, "ADX_MIN_TREND", 22.0)) & (df["ema_50"] > df["ema_50"].shift(3))
            setup2_active = (
                s2_macro &
                (df["close"] > df["high_12"].shift(1)) &
                (score >= getattr(Config, "BREAKOUT_MIN_SCORE", 2)) &
                (df["close"] > df["open"]) &
                (df["close_location"] >= 0.50) &
                s2_adx_trend
            )

            # SETUP 3: Mean-Reversion & Bear/Range Bounce (Multi-Bar Rolling Oversold Window & Range ADX < 24)
            # Operates when price is below EMA200 or in range/consolidation market (ADX < 24)
            s3_macro = (df["close"] < df["ema_200"]) | (df["ema_50"] < df["ema_200"]) | (df["adx"] < 24.0)
            rolling_low = df["low"].rolling(3).min()
            rolling_dist_ema20 = df["dist_ema20_pct"].rolling(3).min()
            rolling_rsi = df["rsi"].rolling(3).min()
            rolling_rsi7 = df["rsi_7"].rolling(3).min()

            s3_oversold = (
                (rolling_low <= df["bb_lower_165"]) |
                (rolling_low <= df["bb_lower"]) |
                (rolling_dist_ema20 <= getattr(Config, "MEAN_REV_DIST_EMA20_MIN", -1.25))
            )
            s3_rsi = (
                (rolling_rsi < getattr(Config, "MEAN_REV_RSI_MAX", 40.0)) |
                (rolling_rsi7 < getattr(Config, "MEAN_REV_RSI7_MAX", 28.0))
            )
            s3_reversal = (
                (df["close"] > df["open"]) &
                (df["close"] > df["low"].shift(1)) &
                (df["close"] < df["bb_mid"])
            )
            # Exhaustion bounces happen on declining selling volume; no mandatory volume surge required
            setup3_active = s3_macro & s3_oversold & s3_rsi & s3_reversal

            # Optional Market Regime Gate (ADX & Choppiness Filter)
            if getattr(Config, "REGIME_FILTER_ACTIVE", False):
                regime_trend_ok = (df["adx"] >= getattr(Config, "ADX_MIN_TREND", 20.0)) & (df["chop"] <= getattr(Config, "CHOP_MAX_TREND", 61.8))
                setup2_active &= regime_trend_ok

            # Priority assignment of Tri-Engine setups
            df.loc[setup2_active, ["signal", "setup_type"]] = [1, "MOMENTUM_BREAKOUT"]
            df.loc[setup3_active, ["signal", "setup_type"]] = [1, "MEAN_REVERSION_BOUNCE"]
            df.loc[setup1_active, ["signal", "setup_type"]] = [1, "PULLBACK_REATTIVO"]

        else:
            # =========================================================================
            # BASELINE DUAL-SETUP BULL-ONLY SNIPER (Phase 0 Baseline Fallback)
            # =========================================================================
            pullback_vol_ok = (df["vol_ratio"] >= Config.PULLBACK_MIN_VOL_RATIO) if hasattr(Config, "PULLBACK_MIN_VOL_RATIO") else (df["vol_ratio"] >= 1.05)
            ema100_pullback = (
                (df["ema_50"] > df["ema_200"]) &
                (df["low"] <= df["ema_100"]) &
                (df["close"] > df["ema_100"]) &
                (df["close"] > df["open"]) &
                (df["rsi"] < 45) &
                (df["fng_value"] >= 35) &
                pullback_vol_ok
            )

            bull_breakout = (
                (df["fng_value"] >= 55) &
                (df["ema_50"] > df["ema_200"]) &
                (df["squeeze"].shift(1)) &
                (df["close"] > df["high_20"].shift(1)) &
                (df["vol_ratio"] >= 1.4) &
                (df["rsi"] >= 58) & (df["rsi"] <= 72)
            )

            if Config.EMA200_TREND_FILTER:
                trend_is_positive = (df["close"] > df["ema_200"]) & (df["close"] > df["ema_800"])
                ema100_pullback &= trend_is_positive
                bull_breakout &= trend_is_positive

            if getattr(Config, "REGIME_FILTER_ACTIVE", False):
                regime_trend_ok = (df["adx"] >= getattr(Config, "ADX_MIN_TREND", 20.0)) & (df["chop"] <= getattr(Config, "CHOP_MAX_TREND", 61.8))
                bull_breakout &= regime_trend_ok

            df.loc[ema100_pullback, ["signal", "setup_type"]] = [1, "EMA100_PULLBACK"]
            df.loc[bull_breakout & (df["signal"] == 0), ["signal", "setup_type"]] = [1, "BULL_BREAKOUT"]

        # Vectorized generation of state tokens for all active signals
        sig_mask = df["signal"] == 1
        if sig_mask.any():
            df.loc[sig_mask, "rl_state"] = [
                RLExperienceEngine.get_state_key(
                    row["setup_type"],
                    row["fng_value"],
                    row["atr_pct"],
                    row["ema_200_slope"],
                    row["rsi"]
                )
                for _, row in df[sig_mask].iterrows()
            ]

        # Adaptive Multi-Tier News & Sentiment Overlay on live bar:
        if len(df) > 0 and df.iloc[-1]["signal"] == 1:
            last_idx = df.index[-1]
            last_setup = df.at[last_idx, "setup_type"]
            last_st = df.at[last_idx, "rl_state"]

            # Tier 1: Severe Risk-Off (Crash/Hack/War panic) -> Complete Circuit Breaker
            if live_news_score <= -0.30:
                df.at[last_idx, "signal"] = 0
                df.at[last_idx, "setup_type"] = "BLOCKED_BY_CRITICAL_NEWS"

            # Tier 2: Moderate Risk-Off -> Block Breakouts (prevent bull traps), allow only deep value dips
            elif live_news_score <= -0.15 and last_setup in {"BULL_BREAKOUT", "MOMENTUM_BREAKOUT"}:
                df.at[last_idx, "signal"] = 0
                df.at[last_idx, "setup_type"] = "BLOCKED_BY_NEGATIVE_NEWS"

            # Tier 2.5: Extreme Long Euphoria / Funding Rate Trap (Overheated retail leverage)
            elif Config.FUNDING_RATE_FILTER and live_funding_rate >= Config.FUNDING_RATE_MAX_THRESHOLD and last_setup in {"BULL_BREAKOUT", "MOMENTUM_BREAKOUT"}:
                df.at[last_idx, "signal"] = 0
                df.at[last_idx, "setup_type"] = "BLOCKED_BY_FUNDING_RATE_OVERHEAT"

            # Tier 3: Adaptive Reinforcement Learning Policy Gate (Online Mistake Pruning & Dynamic Sizing)
            elif rl_engine is not None and Config.RL_ACTIVE:
                should_take, reason, stats = rl_engine.evaluate_proposal(last_st)
                if not should_take:
                    df.at[last_idx, "signal"] = 0
                    df.at[last_idx, "setup_type"] = f"PRUNED_BY_RL ({reason})"
                else:
                    sizing_mult = float(stats.get("sizing_multiplier", 1.0))
                    df.at[last_idx, "rl_sizing_multiplier"] = sizing_mult

        return df
