from __future__ import annotations

import datetime as dt
import json
import hashlib
from pathlib import Path
from typing import Sequence

from cache_io import day_cache_path, ensure_dir
from sentiment_models import BaseSentimentScorer
from signal_utils import aggregate_scores, map_scores, texts_from_news


def load_or_score_day(
    *,
    cache_dir: Path,
    namespace: str,
    symbol: str,
    day: dt.date,
    texts: Sequence[str],
    scorer: BaseSentimentScorer | None = None,
    cache_only: bool = False,
) -> list[float]:
    """Cache per-title scores to avoid recomputation."""
    p = day_cache_path(cache_dir, f"{namespace}_sent", symbol, day, "json")
    if p.exists():
        arr = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(arr, list) and len(arr) == len(texts):
            return [float(x) for x in arr]

    if cache_only:
        raise FileNotFoundError(
            f"sentiment cache miss: namespace={namespace} symbol={symbol} day={day.isoformat()} items={len(texts)}"
        )
    if scorer is None:
        raise ValueError("scorer is required when cache_only is False")

    ensure_dir(p.parent)
    scores = map_scores(texts, scorer.score)
    p.write_text(json.dumps(scores, ensure_ascii=False), encoding="utf-8")
    return scores


def load_or_score_batch(
    *,
    cache_dir: Path,
    namespace: str,
    symbol: str,
    day: dt.date,
    batch_tag: str,
    texts: Sequence[str],
    scorer: BaseSentimentScorer | None = None,
    cache_only: bool = False,
) -> list[float]:
    """Cache scores for an arbitrary batch of texts.

    Cache key is derived from `batch_tag` and a hash of the texts.
    """
    h = hashlib.sha256((batch_tag + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    p = day_cache_path(cache_dir, f"{namespace}_sent", symbol, day, f"{h}.json")
    if p.exists():
        arr = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(arr, list) and len(arr) == len(texts):
            return [float(x) for x in arr]

    if cache_only:
        raise FileNotFoundError(
            f"sentiment batch cache miss: namespace={namespace} symbol={symbol} day={day.isoformat()} tag={batch_tag}"
        )
    if scorer is None:
        raise ValueError("scorer is required when cache_only is False")

    ensure_dir(p.parent)
    scores = map_scores(texts, scorer.score)
    p.write_text(json.dumps(scores, ensure_ascii=False), encoding="utf-8")
    return scores


def daily_sentiment_from_articles(
    *,
    cache_dir: Path,
    namespace: str,
    symbol: str,
    day: dt.date,
    articles: list[dict],
    scorer: BaseSentimentScorer,
    max_articles: int = 50,
    agg: str = "mean",
) -> float:
    texts = texts_from_news(articles, max_items=max_articles)
    if not texts:
        return 0.0
    scores = load_or_score_day(
        cache_dir=cache_dir,
        namespace=namespace,
        symbol=symbol,
        day=day,
        texts=texts,
        scorer=scorer,
    )
    return float(aggregate_scores(scores, method=agg))
