import unittest
from rl_experience_engine import RLExperienceEngine

class TestRLExperienceEngine(unittest.TestCase):
    def setUp(self):
        # Inizializza con file di memoria non persistente per test isolati
        self.rl = RLExperienceEngine(memory_file="data/test_memory_tmp.json", min_experience_trials=2, min_state_win_rate=0.55)

    def tearDown(self):
        import os
        if os.path.exists("data/test_memory_tmp.json"):
            os.remove("data/test_memory_tmp.json")

    def test_state_key_discretization(self):
        """Verifica la discretizzazione coerente dei parametri continui in token di regime."""
        key1 = RLExperienceEngine.get_state_key("EMA100_PULLBACK", 20.0, 0.6, 0.20, 30.0)
        self.assertEqual(key1, "EMA100_PULLBACK|EXT_FEAR|LOW_VOL|STRONG_BULL|OVERSOLD")

        key2 = RLExperienceEngine.get_state_key("BULL_BREAKOUT", 75.0, 1.8, -0.05, 65.0)
        self.assertEqual(key2, "BULL_BREAKOUT|GREED|HIGH_VOL|FLAT_BEAR|MOMENTUM")

    def test_mistake_pruning(self):
        """Verifica che il motore blocchi (prune) un regime con 2 perdite consecutive."""
        st_key = "TEST_SETUP|FEAR|MED_VOL|MILD_BULL|MID"
        
        # Inizialmente sconosciuto: ammesso per esplorazione
        take, reason, _ = self.rl.evaluate_proposal(st_key)
        self.assertTrue(take)

        # Registra prima perdita
        self.rl.record_feedback(st_key, is_win=False, net_return_pct=-1.8, r_multiple=-1.0)
        take, reason, _ = self.rl.evaluate_proposal(st_key)
        self.assertTrue(take)  # Warmup (1 trial)

        # Registra seconda perdita consecutiva
        self.rl.record_feedback(st_key, is_win=False, net_return_pct=-1.8, r_multiple=-1.0)
        take, reason, stats = self.rl.evaluate_proposal(st_key)
        self.assertFalse(take, "Il regime doveva essere bloccato da Mistake Memory dopo 2 perdite consecutive!")
        self.assertIn("REJECTED", reason)
        self.assertEqual(stats["sizing_multiplier"], 0.0)

    def test_dynamic_sizing_for_high_conviction(self):
        """Verifica che regimi con alto win rate e reward ottengano un sizing multiplier fino a 1.3x."""
        st_key = "HIGH_CONVICTION|GREED|LOW_VOL|STRONG_BULL|MOMENTUM"
        # Registra 5 vincite consecutive con R-multiple elevato
        for _ in range(5):
            self.rl.record_feedback(st_key, is_win=True, net_return_pct=3.8, r_multiple=2.1)
        
        take, reason, stats = self.rl.evaluate_proposal(st_key)
        self.assertTrue(take)
        self.assertGreaterEqual(stats["sizing_multiplier"], 1.1)

    def test_multi_step_time_decay_with_timestamps(self):
        """Verifica che errori recenti (<72h) restino bloccati, mentre errori storici (>72h o >7d) vengano sbloccati."""
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        
        # 1. Errore recente (10 ore fa): deve restare bloccato
        key_recent = "SETUP_RECENT|FEAR|LOW_VOL|FLAT_BEAR|MID"
        t_recent = (now - timedelta(hours=10)).isoformat()
        self.rl.record_feedback(key_recent, is_win=False, net_return_pct=-1.8, timestamp=t_recent)
        self.rl.record_feedback(key_recent, is_win=False, net_return_pct=-1.8, timestamp=t_recent)
        
        # 2. Errore vecchio (30 giorni fa): deve sbloccarsi in PROBED_RECOVERY
        key_old = "SETUP_OLD|FEAR|LOW_VOL|FLAT_BEAR|MID"
        t_old = (now - timedelta(days=30)).isoformat()
        self.rl.record_feedback(key_old, is_win=False, net_return_pct=-1.8, timestamp=t_old)
        self.rl.record_feedback(key_old, is_win=False, net_return_pct=-1.8, timestamp=t_old)

        # Prima del decay: entrambi bloccati
        take_rec, _, _ = self.rl.evaluate_proposal(key_recent)
        take_old, _, _ = self.rl.evaluate_proposal(key_old)
        self.assertFalse(take_rec, "Stato recente non bloccato")
        self.assertFalse(take_old, "Stato vecchio non bloccato")

        # Applica decay temporale multi-step
        unblocked = self.rl.apply_time_decay(current_time=now)
        self.assertGreaterEqual(unblocked, 1, "Il decay non ha sbloccato lo stato storico")

        # Verifica: recente ancora bloccato, vecchio sbloccato in PROBED_RECOVERY
        take_rec_after, _, _ = self.rl.evaluate_proposal(key_recent)
        take_old_after, reason_old, stats_old = self.rl.evaluate_proposal(key_old)
        self.assertFalse(take_rec_after, "Errore recente (<72h) non doveva sbloccarsi!")
        self.assertTrue(take_old_after, "Errore vecchio (>7d) doveva sbloccarsi in PROBED_RECOVERY!")
        self.assertIn("PROBED_RECOVERY", reason_old)
        self.assertAlmostEqual(stats_old["sizing_multiplier"], 0.45, places=2)

    def test_production_memory_isolation(self):
        """Verifica che i test unitari operino su file temporanei e non sporchino data/rl_experience_memory.json."""
        self.assertNotEqual(self.rl.memory_file, "data/rl_experience_memory.json")
        self.assertTrue(self.rl.memory_file.endswith("test_memory_tmp.json"))

if __name__ == "__main__":
    unittest.main()
