import os
import json
import logging
from typing import Optional, Tuple, Dict, Any
from config import Config

logger = logging.getLogger(__name__)

class RLExperienceEngine:
    """
    Adaptive Reinforcement Learning Experience Engine.
    Behaves like an adaptive LLM learning from execution feedback:
    - Maintains an episodic memory of market regimes and error patterns.
    - Updates value functions Q(s, a) and win rates per state cluster.
    - Rejects low-conviction or historically failing setups (mistake pruning).
    - Persists learned experience across backtest, paper trading, and live execution.
    """
    
    DEFAULT_STORAGE_PATH = os.path.join(Config.DATA_DIR, "rl_experience_memory.json")
    
    def __init__(
        self,
        memory_file: Optional[str] = None,
        min_experience_trials: int = 2,
        min_state_win_rate: float = 0.55,
        max_consecutive_errors: int = 2,
        learning_rate: float = 0.25,
        loss_penalty_multiplier: float = 1.8
    ):
        self.memory_file = memory_file or self.DEFAULT_STORAGE_PATH
        self.min_experience_trials = min_experience_trials
        self.min_state_win_rate = min_state_win_rate
        self.max_consecutive_errors = max_consecutive_errors
        self.learning_rate = learning_rate
        self.loss_penalty_multiplier = loss_penalty_multiplier
        
        # In-memory Q-Table / Mistake Memory: state_key -> dict
        self.memory: Dict[str, Dict[str, Any]] = {}
        self.load_memory()

    @staticmethod
    def get_state_key(
        setup_type: str,
        fng_value: float,
        atr_pct: float,
        ema_200_slope: float,
        rsi: float
    ) -> str:
        """Discretize continuous market indicators into context regime tokens."""
        # 1. Sentiment Regime
        if fng_value <= 25:
            fng_token = "EXT_FEAR"
        elif fng_value <= 45:
            fng_token = "FEAR"
        elif fng_value <= 60:
            fng_token = "NEUTRAL"
        elif fng_value <= 80:
            fng_token = "GREED"
        else:
            fng_token = "EXT_GREED"

        # 2. Volatility Regime (Hourly ATR %)
        if atr_pct < 0.8:
            vol_token = "LOW_VOL"
        elif atr_pct < 1.4:
            vol_token = "MED_VOL"
        else:
            vol_token = "HIGH_VOL"

        # 3. Macro Trend Momentum (EMA 200 Slope % over 24h)
        if ema_200_slope > 0.15:
            trend_token = "STRONG_BULL"
        elif ema_200_slope > 0.0:
            trend_token = "MILD_BULL"
        else:
            trend_token = "FLAT_BEAR"

        # 4. Local Oscillator (RSI 14)
        if rsi < 38:
            rsi_token = "OVERSOLD"
        elif rsi <= 60:
            rsi_token = "MID"
        else:
            rsi_token = "MOMENTUM"

        return f"{setup_type}|{fng_token}|{vol_token}|{trend_token}|{rsi_token}"

    def extract_state_from_row(self, row: Any) -> str:
        """Extracts the discrete state key from a Pandas Series or dict."""
        setup = row.get("setup_type", "UNKNOWN")
        fng = float(row.get("fng_value", 50.0))
        atr_pct = float(row.get("atr_pct", 1.0))
        slope = float(row.get("ema_200_slope", 0.0))
        rsi = float(row.get("rsi", 50.0))
        return self.get_state_key(setup, fng, atr_pct, slope, rsi)

    def evaluate_proposal(self, state_key: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Adaptive Policy Filter & Dynamic Sizing Evaluator:
        Decides whether to execute or reject a trade proposal based on mistake memory.
        Calculates a dynamic position sizing multiplier (0.45x to 1.3x) based on conviction.
        Includes an anti-deadlock probed recovery mechanism for high-WR states suffering from recency shock.
        Returns: (should_take, refusal_reason, state_stats_with_sizing)
        """
        st = self.memory.get(state_key)
        if st is None:
            # Exploration phase: unvisited state
            return True, "EXPLORE_NEW_REGIME", {
                "trials": 0, "win_rate": 1.0, "q_value": 0.0, "consecutive_losses": 0, "sizing_multiplier": 0.8
            }

        trials = st.get("wins", 0) + st.get("losses", 0)
        wr = st.get("win_rate", 0.0)
        streak = st.get("consecutive_losses", 0)
        q_val = st.get("q_value", 0.0)

        if trials < self.min_experience_trials:
            # Warm-up trials for this regime
            st["sizing_multiplier"] = 0.8
            return True, "WARMUP_TRIALS", st

        # 1. Ultra-High Conviction Regime: WR >= 70% and positive expectancy
        if wr >= 0.70 and (q_val >= 0.2 or wr >= 0.75) and streak < self.max_consecutive_errors:
            st["sizing_multiplier"] = 1.3
            return True, f"ACCEPTED_ULTRA_CONVICTION (WR={wr*100:.1f}%, Q={q_val:.2f}, Size=1.3x)", st

        # 2. Standard High Conviction: WR >= 55% and Q >= 0 and streak < max errors
        if wr >= self.min_state_win_rate and q_val >= 0.0 and streak < self.max_consecutive_errors:
            st["sizing_multiplier"] = 1.0
            return True, f"ACCEPTED_HIGH_CONVICTION (WR={wr*100:.1f}%, Q={q_val:.2f})", st

        # 3. Probed Recovery Gate (Anti-Deadlock & Half-Life Decay Mechanism):
        # Allows regimes suffering from recency loss shock or unlocked via time-decay to trade with protected 0.45x sizing!
        if st.get("decay_probed", False):
            st["sizing_multiplier"] = getattr(Config, "RL_PROBED_SIZING_MULTIPLIER", 0.45)
            return True, f"ACCEPTED_PROBED_RECOVERY (Decay Active, Size={st['sizing_multiplier']:.2f}x)", st

        if wr >= self.min_state_win_rate and trials >= 2 and streak <= 2:
            st["sizing_multiplier"] = getattr(Config, "RL_PROBED_SIZING_MULTIPLIER", 0.45)
            return True, f"ACCEPTED_PROBED_RECOVERY (WR={wr*100:.1f}%, Streak={streak}, Size={st['sizing_multiplier']:.2f}x)", st

        # 4. Mistake Pruning & Rejection Rules:
        if wr < self.min_state_win_rate:
            st["sizing_multiplier"] = 0.0
            return False, f"REJECTED_LOW_WIN_RATE ({wr*100:.1f}% < {self.min_state_win_rate*100:.0f}%)", st

        if streak > 3:
            st["sizing_multiplier"] = 0.0
            return False, f"REJECTED_EXTENDED_ERROR_STREAK ({streak} consecutive losses)", st

        if q_val < -1.0 and wr < 0.50:
            st["sizing_multiplier"] = 0.0
            return False, f"REJECTED_NEGATIVE_EXPECTANCY (Q={q_val:.2f})", st

        # Default fallback
        st["sizing_multiplier"] = 0.5
        return True, f"ACCEPTED_CAUTIOUS (WR={wr*100:.1f}%, Size=0.5x)", st

    def record_feedback(
        self,
        state_key: str,
        is_win: bool,
        net_return_pct: float,
        r_multiple: Optional[float] = None,
        timestamp: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Reinforcement Update:
        Incorporates trade execution feedback into episodic memory with point-in-time timestamping.
        Uses normalized R-Multiples when available, with asymmetric punishment on mistakes.
        """
        from datetime import datetime, timezone
        st = self.memory.setdefault(state_key, {
            "wins": 0,
            "losses": 0,
            "consecutive_losses": 0,
            "q_value": 0.0,
            "win_rate": 0.0,
            "r_sum": 0.0,
            "avg_r": 0.0,
            "expectancy_r": 0.0
        })

        ts_str = str(timestamp) if timestamp is not None else datetime.now(timezone.utc).isoformat()
        st["last_update_time"] = ts_str

        # Calculate normalized R-Multiple
        if r_multiple is None:
            default_sl_pct = Config.DEFAULT_SL_PCT * 100.0 if hasattr(Config, "DEFAULT_SL_PCT") else 1.8
            r_multiple = net_return_pct / default_sl_pct

        if is_win:
            st["wins"] = st.get("wins", 0) + 1
            st["consecutive_losses"] = 0
            st["decay_probed"] = False
            st["last_win_time"] = ts_str
            reward = r_multiple if Config.RL_USE_R_MULTIPLES else net_return_pct
        else:
            st["losses"] = st.get("losses", 0) + 1
            st["consecutive_losses"] = st.get("consecutive_losses", 0) + 1
            st["decay_probed"] = False
            st["last_loss_time"] = ts_str
            # Loss penalty: negative asymmetric reinforcement
            base_loss = r_multiple if Config.RL_USE_R_MULTIPLES else net_return_pct
            reward = -abs(base_loss) * self.loss_penalty_multiplier

        # Temporal Difference Q-Value Update
        st["q_value"] = st.get("q_value", 0.0) + self.learning_rate * (reward - st.get("q_value", 0.0))
        trials = st["wins"] + st["losses"]
        st["win_rate"] = st["wins"] / trials if trials > 0 else 0.0
        st["r_sum"] = st.get("r_sum", 0.0) + r_multiple
        st["avg_r"] = st["r_sum"] / trials if trials > 0 else 0.0
        st["expectancy_r"] = st["avg_r"]

        return st

    def apply_time_decay(
        self,
        current_time: Optional[Any] = None,
        half_life_hours: float = 72.0,
        max_idle_days: float = 7.0
    ) -> int:
        """
        Multi-Step Error Half-Life Decay & Probed Recovery:
        When current_time is passed, calculates the number of half-life steps (elapsed_hours // 72)
        since the last error:
        - Reduces consecutive_losses by 1 for each 72h half-life elapsed.
        - Attenuates negative Q-value towards baseline.
        - Regimes idle for > 7-14 days transition into PROBED_RECOVERY (0.45x).
        - Recent errors (< 72h) remain blocked under active protection.
        When called without explicit time (or legacy records without timestamp),
        advances 1 decay step for immediate policy relaxation.
        Returns the count of unblocked/decayed states.
        """
        from datetime import datetime, timezone
        import pandas as pd
        
        unblocked = 0
        explicit_time = current_time is not None
        if not explicit_time:
            curr_dt = datetime.now(timezone.utc)
        elif isinstance(current_time, str):
            curr_dt = pd.to_datetime(current_time).to_pydatetime()
            if curr_dt.tzinfo is None:
                curr_dt = curr_dt.replace(tzinfo=timezone.utc)
        elif hasattr(current_time, "tzinfo") and current_time.tzinfo is None:
            curr_dt = current_time.replace(tzinfo=timezone.utc)
        elif hasattr(current_time, "to_pydatetime"):
            curr_dt = current_time.to_pydatetime()
            if curr_dt.tzinfo is None:
                curr_dt = curr_dt.replace(tzinfo=timezone.utc)
        else:
            curr_dt = current_time

        for st in self.memory.values():
            streak = st.get("consecutive_losses", 0)
            wr = st.get("win_rate", 0.0)
            trials = st.get("wins", 0) + st.get("losses", 0)
            is_blocked = (streak >= self.max_consecutive_errors) or (trials >= self.min_experience_trials and wr < self.min_state_win_rate)
            
            if not is_blocked and not st.get("decay_probed", False):
                continue
                
            last_loss_str = st.get("last_loss_time") or st.get("last_update_time")
            if last_loss_str and explicit_time:
                try:
                    last_dt = pd.to_datetime(last_loss_str).to_pydatetime()
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    elapsed_hours = max(0.0, (curr_dt - last_dt).total_seconds() / 3600.0)
                    if elapsed_hours >= half_life_hours:
                        steps = max(1, int(elapsed_hours // half_life_hours))
                        st["consecutive_losses"] = max(0, streak - steps)
                        st["q_value"] = max(-0.4, st.get("q_value", 0.0) * (0.7 ** min(steps, 6)))
                        if st["consecutive_losses"] < self.max_consecutive_errors or elapsed_hours >= (max_idle_days * 24.0):
                            st["decay_probed"] = True
                            unblocked += 1
                except Exception:
                    st["decay_probed"] = True
                    st["consecutive_losses"] = max(0, streak - 1)
                    st["q_value"] = max(-0.4, st.get("q_value", 0.0) * 0.7)
                    unblocked += 1
            else:
                # Direct call without explicit time offset (or missing timestamp): apply 1 decay step
                st["decay_probed"] = True
                st["consecutive_losses"] = max(0, streak - 1)
                st["q_value"] = max(-0.4, st.get("q_value", 0.0) * 0.7)
                unblocked += 1

        return unblocked

    def save_memory(self, path: Optional[str] = None) -> None:
        """Persist memory to disk in JSON format."""
        file_path = path or self.memory_file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2)
            logger.info("RL Experience Memory salvata con %d stati in %s", len(self.memory), file_path)
        except Exception as exc:
            logger.error("Impossibile salvare memoria RL in %s: %s", file_path, exc)

    def load_memory(self, path: Optional[str] = None) -> None:
        """Load memory from disk if available and populate missing fields."""
        file_path = path or self.memory_file
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
                for st in self.memory.values():
                    trials = st.get("wins", 0) + st.get("losses", 0)
                    if "r_sum" not in st:
                        st["r_sum"] = (st.get("wins", 0) * 1.5) - (st.get("losses", 0) * 1.0)
                    st["avg_r"] = st["r_sum"] / trials if trials > 0 else 0.0
                    st["expectancy_r"] = st["avg_r"]
                logger.info("RL Experience Memory caricata con %d stati da %s", len(self.memory), file_path)
            except Exception as exc:
                logger.warning("Errore nel caricamento della memoria RL da %s: %s. Creazione nuova memoria.", file_path, exc)
                self.memory = {}
        else:
            self.memory = {}

    def get_summary(self) -> Dict[str, Any]:
        """Returns aggregate performance stats from the learned memory table."""
        if not self.memory:
            return {"total_states": 0, "active_winning_states": 0, "pruned_error_states": 0, "probed_recovery_states": 0}
            
        winning_states = 0
        pruned_states = 0
        probed_states = 0
        for k in self.memory.keys():
            should_take, reason, stats = self.evaluate_proposal(k)
            if should_take:
                if "PROBED" in reason:
                    probed_states += 1
                else:
                    winning_states += 1
            else:
                pruned_states += 1
                    
        return {
            "total_states_experienced": len(self.memory),
            "high_conviction_states": winning_states,
            "probed_recovery_states": probed_states,
            "pruned_error_states": pruned_states
        }
