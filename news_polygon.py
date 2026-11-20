from __future__ import annotations

import dataclasses
import datetime as dt
import os
from typing import Any

import requests


@dataclasses.dataclass(frozen=True)
class PolygonNewsItem:
    headline: str
    summary: str
    url: str
    created_at_utc: dt.datetime
    source: str | None = None
    symbols: list[str] | None = None
    extras: dict[str, Any] | None = None


def fetch_polygon_news(
    *,
    symbol: str,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    limit: int = 50,
) -> list[PolygonNewsItem]:
    """Fetch news from Polygon reference news API for one symbol."""

    api_key = str(os.getenv("POLYGON_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing POLYGON_API_KEY for Polygon news")

    params = {
        "ticker": str(symbol or "").upper(),
        "published_utc.gte": start_utc.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "published_utc.lt": end_utc.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "limit": max(1, int(limit)),
        "sort": "published_utc",
        "order": "asc",
        "apiKey": api_key,
    }
    timeout_s = float(os.getenv("POLYGON_TIMEOUT_S", "10"))
    max_retries = int(os.getenv("POLYGON_MAX_RETRIES", "2"))
    backoff_s = float(os.getenv("POLYGON_BACKOFF_S", "1.0"))

    last_err: Exception | None = None
    payload: list[dict[str, Any]] | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                "https://api.polygon.io/v2/reference/news",
                params=params,
                timeout=timeout_s,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"Polygon HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            results = data.get("results") or []
            if not isinstance(results, list):
                raise RuntimeError(f"Unexpected Polygon payload: {type(results).__name__}")
            payload = results
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if "Polygon HTTP 429" in str(exc):
                break
            if attempt >= max_retries:
                raise RuntimeError(f"Polygon news request failed after retries: {last_err}") from exc
            import time

            time.sleep(backoff_s * attempt)
    if payload is None:
        if last_err is not None and "Polygon HTTP 429" in str(last_err):
            return []
        if last_err is not None:
            raise RuntimeError(f"Polygon news request failed after retries: {last_err}") from last_err
        return []

    out: list[PolygonNewsItem] = []
    for item in payload or []:
        try:
            published = str(item.get("published_utc") or "").strip()
            created_at_utc = dt.datetime.fromisoformat(published.replace("Z", "+00:00")).astimezone(
                dt.timezone.utc
            )
            tickers = [str(t).upper() for t in (item.get("tickers") or []) if str(t).strip()]
            if symbol and str(symbol).upper() not in tickers:
                tickers.append(str(symbol).upper())
            out.append(
                PolygonNewsItem(
                    headline=str(item.get("title") or "").strip(),
                    summary=str(item.get("description") or "").strip(),
                    url=str(item.get("article_url") or "").strip(),
                    created_at_utc=created_at_utc,
                    source=str((item.get("publisher") or {}).get("name") or "").strip() or None,
                    symbols=tickers,
                    extras=dict(item),
                )
            )
        except Exception:
            continue
    out.sort(key=lambda x: x.created_at_utc)
    return out
