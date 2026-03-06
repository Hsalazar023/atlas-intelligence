#!/usr/bin/env python3
"""
FinBERT-powered news sentiment pipeline for ATLAS.

Fetches Yahoo Finance RSS headlines per ticker, scores with ProsusAI/finbert,
and saves aggregated sentiment to data/news_sentiment.json.

Usage:
    python scripts/fetch_news_sentiment.py                # all tickers from brain_signals.json
    python scripts/fetch_news_sentiment.py AAPL MSFT TSLA  # specific tickers

Model (~400MB) downloads on first run, cached in ~/.cache/huggingface.
No API key required.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Lazy FinBERT model loading ───────────────────────────────────────────────
_sentiment_model = None


def get_model():
    global _sentiment_model
    if _sentiment_model is None:
        from transformers import pipeline as hf_pipeline
        _sentiment_model = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            return_all_scores=True,
        )
    return _sentiment_model


# ── RSS headline fetching ────────────────────────────────────────────────────

def _fetch_yahoo_headlines(ticker: str, days: int = 7, max_headlines: int = 10) -> list[str]:
    """Fetch recent headlines from Yahoo Finance RSS for a ticker."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        r = requests.get(url, timeout=10, headers={
            "User-Agent": "ATLAS Research Tool/1.0"
        })
        if r.status_code != 200:
            return []

        root = ElementTree.fromstring(r.content)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        headlines = []

        for item in root.iter("item"):
            title = item.findtext("title", "").strip()
            pub_date = item.findtext("pubDate", "")
            if not title:
                continue
            # Parse pubDate (RFC 822 format)
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(pub_date)
                if dt < cutoff:
                    continue
            except Exception:
                pass  # include headline if date unparseable
            headlines.append(title)
            if len(headlines) >= max_headlines:
                break

        return headlines
    except Exception:
        return []


# ── FinBERT scoring ──────────────────────────────────────────────────────────

def _score_headlines(headlines: list[str]) -> dict:
    """Score headlines with FinBERT and return aggregated sentiment."""
    if not headlines:
        return None

    model = get_model()
    pos_count = 0
    neg_count = 0
    neutral_count = 0
    confidences = []

    for headline in headlines:
        try:
            result = model(headline[:512])  # FinBERT max token safety
            scores = {s["label"]: s["score"] for s in result[0]}
            best_label = max(scores, key=scores.get)
            best_score = scores[best_label]

            if best_label == "positive":
                pos_count += 1
            elif best_label == "negative":
                neg_count += 1
            else:
                neutral_count += 1

            confidences.append(best_score)
        except Exception:
            continue

    total = pos_count + neg_count + neutral_count
    if total == 0:
        return None

    sentiment_score = round((pos_count - neg_count) / total, 4)
    sentiment_confidence = round(sum(confidences) / len(confidences), 4)

    return {
        "sentiment_score": sentiment_score,
        "sentiment_confidence": sentiment_confidence,
        "headline_count": total,
        "positive_pct": round(pos_count / total, 3),
        "negative_pct": round(neg_count / total, 3),
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "top_headlines": headlines[:3],
    }


# ── Main pipeline ────────────────────────────────────────────────────────────

def fetch_news_sentiment(tickers: list[str]) -> dict:
    """Fetch and score news sentiment for each ticker.

    Returns dict keyed by ticker with sentiment data.
    Saves to data/news_sentiment.json.
    """
    print(f"FinBERT sentiment: scoring {len(tickers)} tickers...")
    results = {}
    fetched = 0
    failed = 0

    for i, ticker in enumerate(tickers):
        try:
            headlines = _fetch_yahoo_headlines(ticker)
            if not headlines:
                failed += 1
                continue
            scored = _score_headlines(headlines)
            if scored:
                results[ticker] = scored
                fetched += 1
        except Exception as e:
            print(f"  {ticker}: error — {e}")
            failed += 1

        # Rate limit: 1 req per 2 sec
        if i < len(tickers) - 1:
            time.sleep(2)

        # Progress every 10 tickers
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(tickers)} ({fetched} scored)")

    output = {
        "updated": datetime.now(timezone.utc).isoformat() + "Z",
        "count": len(results),
        "method": "finbert",
        "model": "ProsusAI/finbert",
        "tickers": results,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = DATA_DIR / "news_sentiment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"FinBERT sentiment: {fetched} tickers scored, {failed} failed → {out_path}")
    return results


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    # Get tickers from CLI args or from brain_signals.json
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
    else:
        signals_path = DATA_DIR / "brain_signals.json"
        if signals_path.exists():
            with open(signals_path) as f:
                signals = json.load(f)
            tickers = list({s.get("ticker", "") for s in signals if s.get("ticker")})
            print(f"Loaded {len(tickers)} unique tickers from brain_signals.json")
        else:
            print("No tickers provided and brain_signals.json not found.")
            sys.exit(1)

    fetch_news_sentiment(tickers)


if __name__ == "__main__":
    main()
