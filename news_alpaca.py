from __future__ import annotations

import dataclasses
import datetime as dt
import os
from typing import Any


@dataclasses.dataclass(frozen=True)
class AlpacaNewsItem:
    headline: str
    summary: str
    url: str
    created_at_utc: dt.datetime
    source: str | None = None
    symbols: list[str] | None = None
    extras: dict[str, Any] | None = None


def _resolve_alpaca_news_creds() -> tuple[str, str]:
    key = str(os.getenv("ALPACA_API_KEY_ID") or "").strip()
    secret = str(os.getenv("ALPACA_API_SECRET_KEY") or "").strip()
    if key and secret:
        return key, secret
    raise RuntimeError("Missing ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY for Alpaca news")


def fetch_alpaca_news(
    *,
    symbols: str,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    limit: int = 50,
) -> list[AlpacaNewsItem]:
    """Fetch news from Alpaca News API for symbols in [start_utc, end_utc).

    `symbols` accepts comma-separated list like "SPY,QQQ".
    """
    from alpaca.data.historical import NewsClient
    from alpaca.data.requests import NewsRequest

    key, secret = _resolve_alpaca_news_creds()

    max_retries = int(os.getenv("ALPACA_NEWS_MAX_RETRIES", "8"))
    backoff_s = float(os.getenv("ALPACA_NEWS_BACKOFF_S", "1.0"))

    client = NewsClient(key, secret)
    req = NewsRequest(symbols=symbols, start=start_utc, end=end_utc, limit=limit)

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            news_set = client.get_news(req)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Alpaca returns 429 "too many requests"; backoff and retry
            time_sleep = backoff_s * attempt
            import time

            time.sleep(time_sleep)
    else:
        raise RuntimeError(f"Alpaca news request failed after retries: {last_err}")

    items = news_set.data.get("news") if hasattr(news_set, "data") else []
    out: list[AlpacaNewsItem] = []
    for n in items or []:
        # n may be a News model
        headline = getattr(n, "headline", "") or ""
        summary = getattr(n, "summary", "") or ""
        url = getattr(n, "url", "") or ""
        created_at = getattr(n, "created_at", None)
        if created_at is None:
            continue
        created_at_utc = created_at.astimezone(dt.timezone.utc)
        out.append(
            AlpacaNewsItem(
                headline=headline.strip(),
                summary=summary.strip(),
                url=url.strip(),
                created_at_utc=created_at_utc,
                source=getattr(n, "source", None),
                symbols=list(getattr(n, "symbols", []) or []),
                extras=getattr(n, "model_dump", lambda: None)() or None,
            )
        )
    # Newest first is common; sort ascending to be stable
    out.sort(key=lambda x: x.created_at_utc)
    return out
