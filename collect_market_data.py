import argparse

from config import Config
from data_manager import DataManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Raccoglie OHLCV spot per exchange e coppia in archivi separati.")
    parser.add_argument("--years", type=float, default=10, help="Anni di storico richiesti (default: 10).")
    parser.add_argument("--timeframe", choices=DataManager.TIMEFRAMES, default=Config.TIMEFRAME)
    parser.add_argument("--exchanges", nargs="+", default=None, help="Exchange CCXT da interrogare.")
    parser.add_argument("--bases", nargs="+", default=["BTC"], help="Asset base, ad esempio BTC ETH SOL.")
    parser.add_argument("--quotes", nargs="+", default=list(DataManager.DEFAULT_QUOTE_ASSETS))
    args = parser.parse_args()

    Config.TIMEFRAME = args.timeframe
    manifest = DataManager().collect_exchange_histories(
        years=args.years,
        exchange_ids=args.exchanges,
        base_assets=args.bases,
        quote_assets=args.quotes,
    )
    if manifest.empty:
        print("Nessun mercato raccolto.")
    else:
        print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()