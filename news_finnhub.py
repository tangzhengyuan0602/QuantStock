from __future__ import annotations

import dataclasses
import datetime as dt
import os
from typing import Any

import requests


@dataclasses.dataclass(frozen=True)
class FinnhubNewsItem:
    headline: str
    summary: str
    url: str
    created_at_utc: dt.datetime
    source: str | None = None
    symbols: list[str] | None = None
    extras: dict[str, Any] | None = None


def fetch_finnhub_news(
    *,
    symbol: str,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    limit: int = 50,
) -> list[FinnhubNewsItem]:
    """Fetch company news from Finnhub for one symbol in [start_utc, end_utc)."""

    api_key = str(os.getenv("FINNHUB_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing FINNHUB_API_KEY for Finnhub news")

    params = {
        "symbol": str(symbol or "").upper(),
        "from": start_utc.astimezone(dt.timezone.utc).date().isoformat(),
        "to": end_utc.astimezone(dt.timezone.utc).date().isoformat(),
        "token": api_key,
    }
    timeout_s = float(os.getenv("FINNHUB_TIMEOUT_S", "10"))
    max_retries = int(os.getenv("FINNHUB_MAX_RETRIES", "2"))
    backoff_s = float(os.getenv("FINNHUB_BACKOFF_S", "1.0"))

    last_err: Exception | None = None
    payload: list[dict[str, Any]] | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params=params,
                timeout=timeout_s,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"Finnhub HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected Finnhub payload: {type(data).__name__}")
            payload = data
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt >= max_retries:
                raise RuntimeError(f"Finnhub news request failed after retries: {last_err}") from exc
            import time

            time.sleep(backoff_s * attempt)

    out: list[FinnhubNewsItem] = []
    for item in payload or []:
        try:
            created_at_utc = dt.datetime.fromtimestamp(
                float(item.get("datetime") or 0),
                tz=dt.timezone.utc,
            )
            if not (start_utc <= created_at_utc < end_utc):
                continue
            out.append(
                FinnhubNewsItem(
                    headline=str(item.get("headline") or "").strip(),
                    summary=str(item.get("summary") or "").strip(),
                    url=str(item.get("url") or "").strip(),
                    created_at_utc=created_at_utc,
                    source=str(item.get("source") or "").strip() or None,
                    symbols=[str(symbol or "").upper()],
                    extras=dict(item),
                )
            )
        except Exception:
            continue
    out.sort(key=lambda x: x.created_at_utc)
    if limit > 0:
        out = out[-int(limit) :]
    return out
