import unittest
import pandas as pd
from data_manager import DataManager

class TestDataSanitization(unittest.TestCase):
    def setUp(self):
        self.dm = DataManager(client=None)

    def test_corrupt_candles_filtered_out(self):
        """Verifica che candele con prezzi negativi, high < low o volumi negativi vengano rimosse da _normalise."""
        corrupt_data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
            "open": [100.0, -50.0, 100.0, 100.0, 100.0, 100.0],  # row 1: negative open
            "high": [105.0, 105.0, 95.0, 105.0, 105.0, 105.0],   # row 2: high (95) < low (100)
            "low":  [95.0,  95.0,  100.0, 95.0,  95.0,  95.0],
            "close": [102.0, 102.0, 102.0, 102.0, 102.0, 102.0],
            "volume": [10.0, 10.0, 10.0, -5.0, 10.0, 10.0],      # row 3: negative volume
        })

        clean = self.dm._normalise(corrupt_data)
        self.assertEqual(len(clean), 3, "Le 3 righe corrotte dovevano essere scartate!")
        self.assertTrue((clean["open"] > 0).all())
        self.assertTrue((clean["high"] >= clean["low"]).all())
        self.assertTrue((clean["volume"] >= 0).all())

    def test_deduplication_and_sorting(self):
        """Verifica che timestamp duplicati vengano deduplicati e ordinati cronologicamente."""
        dates = [
            pd.Timestamp("2024-01-01 02:00:00"),
            pd.Timestamp("2024-01-01 01:00:00"),
            pd.Timestamp("2024-01-01 02:00:00"),  # Duplicate
        ]
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0, 100.0, 101.0],
            "high": [105.0, 105.0, 106.0],
            "low": [95.0, 95.0, 96.0],
            "close": [102.0, 102.0, 103.0],
            "volume": [10.0, 10.0, 12.0],
        })
        clean = self.dm._normalise(df)
        self.assertEqual(len(clean), 2)
        self.assertEqual(clean["timestamp"].iloc[0], pd.Timestamp("2024-01-01 01:00:00"))
        self.assertEqual(clean["timestamp"].iloc[1], pd.Timestamp("2024-01-01 02:00:00"))

if __name__ == "__main__":
    unittest.main()
