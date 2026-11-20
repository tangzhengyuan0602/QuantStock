from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any

from cache_io import day_cache_path, ensure_dir
from news_finnhub import fetch_finnhub_news
from news_alpaca import fetch_alpaca_news
from news_polygon import fetch_polygon_news

MACRO_KEYWORDS = {
    "fed", "inflation", "cpi", "pce", "payroll", "jobs", "yield", "treasury",
    "rates", "tariff", "recession", "gdp", "geopolitical", "opec", "oil", "dollar",
}

AI_KEYWORDS = {
    "ai", "artificial intelligence", "semiconductor", "semiconductors", "chip",
    "chipmaker", "chipmakers", "gpu", "datacenter", "data center", "cloud",
    "nvidia", "amd", "micron", "tsmc",
}

SYMBOL_KEYWORDS = {
    "SPY": {"s&p 500", "spy", "wall street", "us stock market"},
    "QQQ": {"qqq", "nasdaq", "nasdaq 100", "big tech"},
    "XLK": {"xlk", "technology sector", "tech stocks", "software", "cloud"},
    "SMH": {"smh", "semiconductor", "semiconductors", "chipmakers", "nvidia", "amd", "tsmc", "micron"},
    "SOXX": {"soxx", "semiconductor", "semiconductors", "chipmakers", "nvidia", "amd", "tsmc", "micron"},
    "XLF": {"xlf", "banks", "financials", "credit"},
    "KRE": {"kre", "regional banks", "regional banking", "bank stocks", "commercial real estate", "deposits"},
    "XLE": {"xle", "energy sector", "oil", "gas"},
    "XLV": {"xlv", "healthcare", "biotech", "pharma"},
    "XBI": {"xbi", "biotech", "biotechnology", "fda", "clinical trial", "phase 3", "drug approval"},
    "IBB": {"ibb", "biotech", "biotechnology", "pharma", "drug approval", "clinical trial"},
    "XLI": {"xli", "industrials", "aerospace", "defense", "manufacturing"},
    "ITA": {"ita", "aerospace", "defense", "defense contractors", "military", "weapons", "aircraft"},
    "XLY": {"xly", "consumer discretionary", "retail", "e-commerce", "autos"},
    "XLP": {"xlp", "consumer staples", "staples", "groceries", "household products", "food"},
    "XLB": {"xlb", "materials", "chemicals", "metals", "mining", "copper", "industrial metals"},
    "XLU": {"xlu", "utilities", "utility", "power demand", "electricity", "grid"},
    "XLRE": {"xlre", "real estate", "reit", "commercial real estate", "office", "property"},
    "EEM": {"eem", "emerging markets", "china", "india", "brazil"},
    "HYG": {"hyg", "high yield", "junk bonds", "credit spreads"},
    "TLT": {"tlt", "treasuries", "long bonds", "yields"},
    "GLD": {"gld", "gold"},
    "DBC": {"dbc", "commodities", "oil", "metals", "agriculture"},
    "IEF": {"ief", "treasuries", "intermediate bonds", "yields"},
    "VTI": {"vti", "total stock market", "us equities"},
    "EFA": {"efa", "developed markets", "europe", "japan"},
    "VWO": {"vwo", "emerging markets", "china", "india", "brazil"},
    "TAN": {"tan", "solar", "clean energy"},
    "KWEB": {"kweb", "china internet", "alibaba", "tencent", "jd", "pdd"},
}

BROAD_BETA_SYMBOLS = {"SPY", "QQQ", "VTI", "IWM", "EFA", "VWO", "EEM", "TLT", "IEF", "GLD", "DBC", "HYG"}


def _article_key(item: dict[str, Any]) -> str:
    url = str(item.get("url") or "").strip().lower()
    if url:
        return url
    title = str(item.get("title") or "").strip().lower()
    return title


def _classify_article(symbol: str, item: dict[str, Any]) -> tuple[str, int]:
    sym = str(symbol or "").upper()
    title = str(item.get("title") or "")
    summary = str(item.get("summary") or "")
    source = str(item.get("source") or item.get("domain") or "")
    text = f"{title}\n{summary}\n{source}".lower()
    tags: list[str] = []
    if any(key in text for key in SYMBOL_KEYWORDS.get(sym, set())) or sym.lower() in text:
        tags.append("etf_specific")
    if any(key in text for key in AI_KEYWORDS):
        tags.append("theme_ai")
    if any(key in text for key in MACRO_KEYWORDS):
        tags.append("macro")
    sector_terms = SYMBOL_KEYWORDS.get(sym, set()) - {sym.lower()}
    if any(key in text for key in sector_terms):
        tags.append("sector")
    if not tags:
        tags.append("general")

    if "etf_specific" in tags:
        category = "etf_specific"
    elif "sector" in tags:
        category = "sector"
    elif "theme_ai" in tags:
        category = "theme_ai"
    elif "macro" in tags:
        category = "macro"
    else:
        category = "general"

    priority = 10
    if category == "etf_specific":
        priority = 40
    elif category == "sector":
        priority = 34
    elif category == "theme_ai":
        priority = 30
    elif category == "macro":
        priority = 18
    if sym in BROAD_BETA_SYMBOLS and category == "macro":
        priority = 14
    if sym in {"QQQ", "XLK", "SMH", "SOXX"} and category == "theme_ai":
        priority = max(priority, 36)
    if str(item.get("provider") or "") == "alpaca":
        priority += 2
    item["category"] = category
    item["relevance_priority"] = int(priority)
    return category, int(priority)


def load_or_fetch_alpaca_daily(
    *,
    cache_dir: Path,
    symbol: str,
    day: dt.date,
    limit: int = 50,
) -> list[dict]:
    """Fetch and cache one day's news items via Alpaca.

    Cache format: JSON under `{cache_dir}/alpaca_news/{symbol}/{YYYY-MM-DD}.json`.
    """
    p = day_cache_path(cache_dir, "alpaca_news", symbol, day, "json")
    if p.exists():
        import json

        return json.loads(p.read_text(encoding="utf-8"))

    ensure_dir(p.parent)
    start_utc = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
    end_utc = start_utc + dt.timedelta(days=1)
    items = fetch_alpaca_news(symbols=symbol, start_utc=start_utc, end_utc=end_utc, limit=limit)
    data = [
        {
            "title": it.headline,
            "summary": it.summary,
            "url": it.url,
            "created_at_utc": it.created_at_utc.isoformat(),
            "source": it.source,
            "symbols": it.symbols,
        }
        for it in items
    ]
    import json

    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def load_or_fetch_alpaca_daily_multi(
    *,
    cache_dir: Path,
    symbols: list[str],
    day: dt.date,
    limit: int = 200,
    offline_cache_only: bool = False,
) -> dict[str, list[dict]]:
    """Fetch one day's news for multiple symbols in one API call.

    Cache format: JSON under `{cache_dir}/alpaca_news_all/{YYYY-MM-DD}.json`.
    Returned dict maps symbol -> list[dict]. Each item keeps `symbols` field from provider.
    """
    import json

    p = day_cache_path(cache_dir, "alpaca_news_all", "ALL", day, "json")
    if p.exists():
        payload = json.loads(p.read_text(encoding="utf-8"))
    else:
        if offline_cache_only:
            payload = []
        else:
            ensure_dir(p.parent)
            start_utc = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
            end_utc = start_utc + dt.timedelta(days=1)
            items = fetch_alpaca_news(
                symbols=",".join(sorted({s.upper() for s in symbols})),
                start_utc=start_utc,
                end_utc=end_utc,
                limit=limit,
            )
            payload = [
                {
                    "title": it.headline,
                    "summary": it.summary,
                    "url": it.url,
                    "created_at_utc": it.created_at_utc.isoformat(),
                    "source": it.source,
                    "symbols": it.symbols,
                }
                for it in items
            ]
            p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    out: dict[str, list[dict]] = {s: [] for s in symbols}
    for item in payload:
        item_syms = {str(sym).upper() for sym in ((item.get("symbols") or []) or [])}
        for s in symbols:
            rec = dict(item)
            if s in item_syms:
                _classify_article(s, rec)
                out[s].append(rec)
                continue
            category, _ = _classify_article(s, rec)
            if category in {"etf_specific", "sector"} or (
                category == "theme_ai" and s in {"QQQ", "XLK", "SMH", "SOXX"}
            ):
                out[s].append(rec)
    return out


def load_or_fetch_finnhub_daily(
    *,
    cache_dir: Path,
    symbol: str,
    day: dt.date,
    limit: int = 50,
) -> list[dict]:
    """Fetch and cache one day's news items via Finnhub."""

    import json

    p = day_cache_path(cache_dir, "finnhub_news", symbol, day, "json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))

    ensure_dir(p.parent)
    start_utc = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
    end_utc = start_utc + dt.timedelta(days=1)
    items = fetch_finnhub_news(symbol=symbol, start_utc=start_utc, end_utc=end_utc, limit=limit)
    data = [
        {
            "title": it.headline,
            "summary": it.summary,
            "url": it.url,
            "created_at_utc": it.created_at_utc.isoformat(),
            "source": it.source,
            "symbols": it.symbols,
        }
        for it in items
    ]
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def load_or_fetch_polygon_daily(
    *,
    cache_dir: Path,
    symbol: str,
    day: dt.date,
    limit: int = 50,
) -> list[dict]:
    """Fetch and cache one day's news items via Polygon."""

    import json

    p = day_cache_path(cache_dir, "polygon_news", symbol, day, "json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))

    ensure_dir(p.parent)
    start_utc = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
    end_utc = start_utc + dt.timedelta(days=1)
    items = fetch_polygon_news(symbol=symbol, start_utc=start_utc, end_utc=end_utc, limit=limit)
    data = [
        {
            "title": it.headline,
            "summary": it.summary,
            "url": it.url,
            "created_at_utc": it.created_at_utc.isoformat(),
            "source": it.source,
            "symbols": it.symbols,
        }
        for it in items
    ]
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def load_or_fetch_multisource_daily_multi(
    *,
    cache_dir: Path,
    symbols: list[str],
    day: dt.date,
    limit: int = 200,
    gdelt_limit: int = 25,
    offline_cache_only: bool = False,
) -> dict[str, list[dict]]:
    """Fetch one day's news from Alpaca + Finnhub + Polygon, then dedupe and rank."""

    alpaca = load_or_fetch_alpaca_daily_multi(
        cache_dir=cache_dir,
        symbols=symbols,
        day=day,
        limit=limit,
        offline_cache_only=offline_cache_only,
    )
    out: dict[str, list[dict]] = {s: [] for s in symbols}
    for sym in symbols:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()

        for item in alpaca.get(sym) or []:
            rec = dict(item)
            rec["provider"] = "alpaca"
            _classify_article(sym, rec)
            key = _article_key(rec)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            merged.append(rec)

        if str(os.getenv("FINNHUB_API_KEY") or "").strip():
            try:
                finnhub_rows = load_or_fetch_finnhub_daily(
                    cache_dir=cache_dir,
                    symbol=sym,
                    day=day,
                    limit=limit,
                )
            except Exception:
                finnhub_rows = []
        else:
            finnhub_rows = []
        for item in finnhub_rows:
            rec = dict(item)
            rec["provider"] = "finnhub"
            _classify_article(sym, rec)
            key = _article_key(rec)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            merged.append(rec)

        if str(os.getenv("POLYGON_API_KEY") or "").strip():
            try:
                polygon_rows = load_or_fetch_polygon_daily(
                    cache_dir=cache_dir,
                    symbol=sym,
                    day=day,
                    limit=limit,
                )
            except Exception:
                polygon_rows = []
        else:
            polygon_rows = []
        for item in polygon_rows:
            rec = dict(item)
            rec["provider"] = "polygon"
            _classify_article(sym, rec)
            key = _article_key(rec)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            merged.append(rec)

        merged.sort(
            key=lambda item: (
                -int(item.get("relevance_priority") or 0),
                str(item.get("created_at_utc") or item.get("seendate_utc") or ""),
            )
        )
        out[sym] = merged
    return out
