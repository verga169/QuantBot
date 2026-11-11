import os
import time
import calendar
import requests
import feedparser
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import Config

class SentimentEngine:
    RSS_FEEDS = {
        # 1. Bloomberg (Markets / Terminal / Wire)
        "Bloomberg_Wire": "https://news.google.com/rss/search?q=source:Bloomberg+when:1d&hl=en-US&gl=US&ceid=US:en",
        
        # 2. Reuters (World / Markets / Business)
        "Reuters_Wire": "https://news.google.com/rss/search?q=source:Reuters+when:1d&hl=en-US&gl=US&ceid=US:en",
        
        # 3. Financial Times (FT)
        "FT_Markets": "https://www.ft.com/markets?format=rss",
        
        # 4. The Wall Street Journal (WSJ)
        "WSJ_Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "WSJ_World": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
        
        # 5. Calendari Economici Macro (Forex Factory & Investing.com)
        "ForexFactory_Macro": "https://news.google.com/rss/search?q=site:forexfactory.com+when:2d&hl=en-US&gl=US&ceid=US:en",
        "Investing_Economy": "https://www.investing.com/rss/news_14.rss",
        
        # 6. CNBC (Breaking News, Squawk Box & Macro)
        "CNBC_TopNews": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "CNBC_Macro": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
        "CNBC_World": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
        
        # 7. Banche Centrali Istituzionali (FED, BCE, BoE)
        "FED_Press": "https://www.federalreserve.gov/feeds/press_all.xml",
        "ECB_Press": "https://www.ecb.europa.eu/rss/press.html",
        "BoE_News": "https://www.bankofengland.co.uk/rss/news",
        
        # 8. SEC EDGAR (Regolamentazione & Enforcement)
        "SEC_Press": "https://www.sec.gov/news/pressreleases.rss",
        
        # 9. Il Sole 24 Ore / Radiocor (Eurozona, BTP, Spread & Italia)
        "Sole24Ore_Economia": "https://www.ilsole24ore.com/rss/economia.xml",
        "Sole24Ore_Mondo": "https://www.ilsole24ore.com/rss/mondo.xml",
        
        # 10. Feed X / Fast Wire (Walter Bloomberg, Deltaone, Trump & Musk)
        "FastWire_Deltaone_WalterB": "https://news.google.com/rss/search?q=%22Walter+Bloomberg%22+OR+%22Deltaone%22+OR+%22FinancialJuice%22+when:2d&hl=en-US&gl=US&ceid=US:en",
        "GoogleNews_Trump_Musk_Geo": "https://news.google.com/rss/search?q=Trump+OR+Elon+Musk+OR+geopolitics+when:1d&hl=en-US&gl=US&ceid=US:en",
        
        # Geopolitica Globale & Top Media Mondiali
        "BBC_World": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "NYT_World": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "Politico_US": "https://rss.politico.com/politics-news.xml",
        "AlJazeera_World": "https://www.aljazeera.com/xml/rss/all.xml",
        
        # Crypto & Digital Assets
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "Cointelegraph": "https://cointelegraph.com/rss",
        "TheBlock": "https://www.theblock.co/rss.xml",
        "Decrypt": "https://decrypt.co/feed",
        "BitcoinMagazine": "https://bitcoinmagazine.com/.rss/full/",
        "CryptoSlate": "https://cryptoslate.com/feed/",
    }

    def __init__(self):
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        self.analyzer = SentimentIntensityAnalyzer()
        # Arricchimento lessico finanziario, macroeconomico, geopolitico e crypto per VADER
        self.analyzer.lexicon.update({
            "rate cut": 2.5, "cuts rates": 2.5, "dovish": 2.2, "inflows": 2.2,
            "etf approved": 3.0, "etf approval": 2.8, "bullish": 2.0, "stimulus": 2.2,
            "rally": 2.0, "breakout": 1.8, "all-time high": 2.5, "ath": 2.2,
            "institutional adoption": 2.5, "treasury buy": 2.2, "liquidity injection": 2.2,
            "rate hike": -2.5, "hawkish": -2.0, "inflation rises": -2.2,
            "war": -3.2, "invasion": -3.2, "missile": -2.8, "sanctions": -2.2,
            "hack": -3.2, "exploit": -3.0, "bankrupt": -3.5, "insolvency": -3.5,
            "lawsuit": -2.2, "sec charges": -2.8, "sec lawsuit": -2.8, "subpoena": -2.2,
            "outflows": -2.2, "crash": -3.2, "liquidation": -2.5, "whale dump": -2.8,
            "delisting": -2.5, "ban": -2.8, "contagion": -3.0,
            # Geopolitics, US Policy, Trump & Elon Musk keywords:
            "tariff": -2.2, "tariffs": -2.2, "trade war": -2.8, "embargo": -2.5,
            "ceasefire": 2.5, "peace deal": 2.8, "diplomacy": 2.0, "treaty": 2.0,
            "escalation": -2.8, "nuclear": -3.2, "strikes": -2.5, "airstrike": -2.8,
            "recession": -2.8, "stagflation": -2.8, "default": -3.2, "debt crisis": -2.8,
            "deregulation": 2.2, "tax cuts": 2.2, "tax cut": 2.2, "pro-crypto": 2.5,
            "executive order": 1.5, "elon musk": 1.5, "trump": 1.5, "doge": 1.8,
            "spacex": 1.5, "ai boom": 2.2, "peace": 2.2, "stability": 1.8,
            # Termini economici in italiano (Il Sole 24 Ore, Radiocor, BTP/Spread, BCE):
            "taglio tassi": 2.5, "taglia i tassi": 2.5, "rialzo tassi": -2.5, "alza i tassi": -2.5,
            "inflazione in calo": 2.2, "inflazione sale": -2.2, "spread in calo": 2.0, "spread sale": -2.0,
            "debito pubblico": -2.0, "crisi": -2.8, "guerra": -3.2, "dazi": -2.5, "accordo": 2.2,
            "tregua": 2.5, "pace": 2.5, "crescita": 2.0, "recessione": -2.8, "fallimento": -3.5,
            "bancarotta": -3.5, "utili record": 2.5, "crollo": -3.2,
        })

    def fetch_fear_and_greed_history(self) -> pd.DataFrame:
        """Scarica l'intero storico giornaliero del Fear & Greed Index (2018 -> oggi)."""
        cache_path = os.path.join(Config.DATA_DIR, "fear_greed_history.parquet")
        # Se aggiornato nelle ultime 12 ore, usa la cache
        if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path)) < 43200:
            return pd.read_parquet(cache_path)

        print("Scarico storico completo Fear & Greed Index...")
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=0", timeout=15)
            data = resp.json().get("data", [])
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.floor("D")
            df["fng_value"] = df["value"].astype(float)
            df = df[["date", "fng_value", "value_classification"]].sort_values("date").drop_duplicates("date")
            df.to_parquet(cache_path, index=False)
            return df
        except Exception as e:
            print(f"[WARN] Impossibile scaricare Fear & Greed: {e}")
            return pd.DataFrame(columns=["date", "fng_value", "value_classification"])

    def fetch_market_funding_rate(self) -> float:
        """Scarica il funding rate live di BTC perpetual da Bybit globale (sentiment di leva retail)."""
        try:
            resp = requests.get(
                "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=4
            )
            if resp.status_code == 200:
                tickers = resp.json().get("result", {}).get("list", [])
                if tickers:
                    return float(tickers[0].get("fundingRate") or 0.0)
        except Exception:
            pass
        return 0.0

    def analyze_live_news(self, max_per_feed: int = 5) -> dict:
        """Legge le ultime notizie mondiali/economiche/crypto in parallelo e calcola il sentiment score."""
        headlines = []
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        observed_at = pd.Timestamp.now(tz="UTC").tz_localize(None)

        def _fetch_single_feed(item):
            source, url = item
            feed_items = []
            try:
                resp = requests.get(url, headers=headers, timeout=5)
                feed = feedparser.parse(resp.content)
                for entry in feed.entries[:max_per_feed]:
                    title = entry.get("title", "").strip()
                    if title:
                        score = self.analyzer.polarity_scores(title)["compound"]
                        published_at = pd.to_datetime(entry.get("published"), utc=True, errors="coerce")
                        if pd.isna(published_at) and entry.get("published_parsed"):
                            published_at = pd.to_datetime(
                                calendar.timegm(entry.published_parsed), unit="s", utc=True
                            )
                        if pd.isna(published_at):
                            published_at = observed_at.tz_localize("UTC")
                        feed_items.append({
                            "source": source,
                            "title": title,
                            "score": score,
                            "published_at_utc": published_at.tz_convert("UTC").tz_localize(None),
                            "observed_at_utc": observed_at,
                            "article_id": entry.get("id", "").strip() or entry.get("link", "").strip(),
                            "feed_url": url,
                            "url": entry.get("link", "").strip(),
                        })
            except Exception:
                pass
            return feed_items

        with ThreadPoolExecutor(max_workers=14) as executor:
            results = executor.map(_fetch_single_feed, self.RSS_FEEDS.items())
            for feed_items in results:
                headlines.extend(feed_items)

        if not headlines:
            return {"avg_score": 0.0, "weighted_score": 0.0, "regime": "NEUTRAL", "headlines": [], "articles": []}

        self._archive_headlines(headlines)
        df_news = pd.DataFrame(headlines)

        # Calcolo peso decadimento temporale (notizie fresche < 2h hanno peso maggiore)
        age_hours = (observed_at - pd.to_datetime(df_news["published_at_utc"])).dt.total_seconds() / 3600.0
        age_hours = age_hours.clip(lower=0.0)
        # Decadimento esponenziale con dimezzamento ogni 6 ore
        weights = 2.0 ** (-age_hours / 6.0)
        weighted_score = float((df_news["score"] * weights).sum() / weights.sum()) if weights.sum() > 0 else float(df_news["score"].mean())
        avg_score = weighted_score

        if avg_score <= -0.30:
            regime = "CRITICAL_RISK_OFF (Crash/Hack/Panico Macro - Blocco Totale Long)"
        elif avg_score <= -0.15:
            regime = "RISK_OFF_BEARISH (Cautela Macro - Blocco Breakout, Solo Dip Difensivi)"
        elif avg_score >= 0.15:
            regime = "RISK_ON_BULLISH (Favorevole a Breakout e Trend Following)"
        else:
            regime = "NEUTRAL"

        return {
            "avg_score": round(avg_score, 3),
            "weighted_score": round(weighted_score, 3),
            "raw_mean_score": round(float(df_news["score"].mean()), 3),
            "regime": regime,
            "top_positive": df_news.sort_values("score", ascending=False).head(3).to_dict("records"),
            "top_negative": df_news.sort_values("score", ascending=True).head(3).to_dict("records"),
            "total_analyzed": len(df_news),
            "articles": headlines,
        }

    def _archive_headlines(self, headlines: list[dict]) -> None:
        archive_path = os.path.join(Config.DATA_DIR, "news_headlines.parquet")
        new_rows = pd.DataFrame(headlines)
        if os.path.exists(archive_path):
            old_rows = pd.read_parquet(archive_path)
            new_rows = pd.concat([old_rows, new_rows], ignore_index=True)
        for column in ("article_id", "feed_url", "url"):
            if column not in new_rows.columns:
                new_rows[column] = ""
            new_rows[column] = new_rows[column].fillna("")
        new_rows = new_rows.drop_duplicates(
            subset=["source", "article_id", "url", "title", "published_at_utc"], keep="last"
        ).sort_values(["published_at_utc", "observed_at_utc"]).reset_index(drop=True)
        new_rows.to_parquet(archive_path, index=False)
