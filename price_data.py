from __future__ import annotations

import datetime as dt

import pandas as pd


def _get_alpaca_creds() -> tuple[str | None, str | None]:
    """Return Alpaca (key, secret) from env.

    Supports both modern `ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY` and legacy
    `APCA_API_KEY_ID/APCA_API_SECRET_KEY`.
    """

    import json
    import os
    from pathlib import Path

    key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if key and secret:
        return key, secret

    # Fallback: try common local credential files without printing secrets.
    candidates = [
        Path.home() / ".alpaca" / "credentials.json",
        Path.home() / ".alpaca" / "credentials",
        Path.home() / ".config" / "alpaca" / "credentials.json",
        Path.home() / ".config" / "alpaca" / "credentials",
    ]

    def _from_mapping(m: dict) -> tuple[str | None, str | None]:
        k = (
            m.get("ALPACA_API_KEY_ID")
            or m.get("APCA_API_KEY_ID")
            or m.get("key")
            or m.get("key_id")
            or m.get("api_key")
            or m.get("api_key_id")
        )
        s = (
            m.get("ALPACA_API_SECRET_KEY")
            or m.get("APCA_API_SECRET_KEY")
            or m.get("secret")
            or m.get("secret_key")
            or m.get("api_secret")
            or m.get("api_secret_key")
        )
        return (str(k) if k else None), (str(s) if s else None)

    for p in candidates:
        try:
            if not p.exists() or not p.is_file():
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
            if not txt:
                continue
            if p.suffix.lower() == ".json":
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    k2, s2 = _from_mapping(obj)
                    if k2 and s2:
                        return k2, s2
                continue

            # .env-like / key=value
            m: dict[str, str] = {}
            for line in txt.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                m[k] = v
            k2 = m.get("ALPACA_API_KEY_ID") or m.get("APCA_API_KEY_ID")
            s2 = m.get("ALPACA_API_SECRET_KEY") or m.get("APCA_API_SECRET_KEY")
            if k2 and s2:
                return str(k2), str(s2)
        except Exception:
            continue

    return None, None


def _yfinance_download_with_retry(
    *,
    symbols: list[str],
    start: dt.date,
    end: dt.date,
    max_retries: int = 6,
    backoff_s: float = 15.0,
):
    """Download daily bars from yfinance with backoff.

    yfinance/Yahoo is frequently rate-limited on shared IPs; this retries when the
    returned dataframe is empty (common symptom of 429).
    """

    import time

    import yfinance as yf

    start_s = start.isoformat()
    end_s = (end + dt.timedelta(days=1)).isoformat()
    tickers = " ".join(symbols)

    last_df = None
    for attempt in range(1, max_retries + 1):
        df = yf.download(
            tickers=tickers,
            start=start_s,
            end=end_s,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
        last_df = df
        if df is not None and not df.empty:
            return df
        time.sleep(backoff_s * attempt)

    raise RuntimeError(
        f"yfinance returned empty dataframe after retries (likely rate-limited). "
        f"tickers={tickers} start={start_s} end={end_s} last_shape={getattr(last_df, 'shape', None)}"
    )


def fetch_daily_opens_stooq(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Fetch daily open prices from Stooq (free, no auth).

    URL format: https://stooq.com/q/d/l/?s=spy.us&i=d
    """
    import io
    import os
    import requests

    def stooq_symbol(s: str) -> str:
        s = s.lower()
        if "." in s:
            return s
        # US ETFs/stocks generally use .us
        return f"{s}.us"

    apikey = os.getenv("STOOQ_APIKEY")
    out = {}
    for sym in symbols:
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol(sym)}&i=d"
        if apikey:
            url = f"{url}&apikey={apikey}"
        r = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
            },
        )
        r.raise_for_status()
        if "get_apikey" in r.text.lower() or "Get your apikey" in r.text:
            raise RuntimeError("stooq requires apikey+captcha; cannot fetch without manual apikey")
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            raise RuntimeError(f"stooq returned empty data for {sym}")
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[(df["Date"] >= start) & (df["Date"] <= end)]
        out[sym] = df.set_index("Date")["Open"].astype(float)
    opens = pd.DataFrame(out).sort_index().dropna(how="all")
    return opens


def fetch_daily_closes_stooq(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Fetch daily close prices from Stooq (supports STOOQ_APIKEY)."""

    import io
    import os

    import requests

    def stooq_symbol(s: str) -> str:
        s = s.lower()
        if "." in s:
            return s
        return f"{s}.us"

    apikey = os.getenv("STOOQ_APIKEY")
    out = {}
    for sym in symbols:
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol(sym)}&i=d"
        if apikey:
            url = f"{url}&apikey={apikey}"
        r = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
            },
        )
        r.raise_for_status()
        if "get_apikey" in r.text.lower() or "Get your apikey" in r.text:
            raise RuntimeError("stooq requires apikey+captcha; cannot fetch without manual apikey")
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            raise RuntimeError(f"stooq returned empty data for {sym}")
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[(df["Date"] >= start) & (df["Date"] <= end)]
        out[sym] = df.set_index("Date")["Close"].astype(float)
    closes = pd.DataFrame(out).sort_index().dropna(how="all")
    return closes


def fetch_daily_opens_alpaca(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Fetch daily open prices from Alpaca Market Data.

    Requires env vars:
      - ALPACA_API_KEY_ID
      - ALPACA_API_SECRET_KEY
    Optional:
      - ALPACA_DATA_FEED (default: iex)
    """
    import os

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key, secret = _get_alpaca_creds()
    if not key or not secret:
        raise RuntimeError(
            "Missing ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY (or APCA_API_KEY_ID/APCA_API_SECRET_KEY) for Alpaca data"
        )
    feed = os.getenv("ALPACA_DATA_FEED", "iex")

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end + dt.timedelta(days=1)),
        feed=feed,
    )
    bars = client.get_stock_bars(req).df
    if bars is None or len(bars) == 0:
        raise RuntimeError("alpaca returned empty bars")

    # bars index: MultiIndex(symbol, timestamp)
    opens = (
        bars.reset_index()
        .assign(date=lambda x: pd.to_datetime(x["timestamp"]).dt.date)
        .pivot(index="date", columns="symbol", values="open")
        .sort_index()
    )
    opens = opens[(opens.index >= start) & (opens.index <= end)].dropna(how="all")
    # Ensure all requested symbols exist as columns
    for s in symbols:
        if s not in opens.columns:
            opens[s] = pd.NA
    opens = opens[symbols]
    return opens


def fetch_daily_ohlc_alpaca(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Fetch daily OHLC (open, close) for symbols from Alpaca."""
    import os

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key, secret = _get_alpaca_creds()
    if not key or not secret:
        raise RuntimeError(
            "Missing ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY (or APCA_API_KEY_ID/APCA_API_SECRET_KEY) for Alpaca data"
        )
    feed = os.getenv("ALPACA_DATA_FEED", "iex")

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end + dt.timedelta(days=1)),
        feed=feed,
    )
    bars = client.get_stock_bars(req).df
    if bars is None or len(bars) == 0:
        raise RuntimeError("alpaca returned empty bars")

    df = bars.reset_index().assign(date=lambda x: pd.to_datetime(x["timestamp"]).dt.date)
    # columns: symbol, date, open, close
    out = {}
    for field in ["open", "close"]:
        out[field] = df.pivot(index="date", columns="symbol", values=field).sort_index()
    # Build MultiIndex columns: (field, symbol)
    res = pd.concat(out, axis=1)
    # Ensure requested symbols exist
    for field in ["open", "close"]:
        for s in symbols:
            if (field, s) not in res.columns:
                res[(field, s)] = pd.NA
    res = res.loc[(res.index >= start) & (res.index <= end)]
    res = res.sort_index()
    return res


def fetch_daily_opens_yfinance(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Fetch daily open prices using yfinance.

    end is inclusive (we request end+1 to be safe).
    """
    df = _yfinance_download_with_retry(symbols=symbols, start=start, end=end)

    opens: pd.DataFrame
    if isinstance(df.columns, pd.MultiIndex):
        opens = pd.DataFrame({s: df[s]["Open"] for s in symbols})
    else:
        # single symbol
        opens = pd.DataFrame({symbols[0]: df["Open"]})
    opens.index = pd.to_datetime(opens.index).date
    opens = opens.sort_index().dropna(how="all")
    return opens


def fetch_daily_opens(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Fetch daily opens with fallback.

    yfinance is convenient but can be rate-limited; stooq is the fallback.
    """
    import os

    # Prefer Stooq when apikey is provided (good for long history).
    if os.getenv("STOOQ_APIKEY"):
        try:
            return fetch_daily_opens_stooq(symbols, start=start, end=end)
        except Exception:
            pass

    # Prefer Alpaca when credentials are available (Yahoo is often rate-limited).
    key, secret = _get_alpaca_creds()
    if key and secret:
        try:
            return fetch_daily_opens_alpaca(symbols, start=start, end=end)
        except Exception:
            return fetch_daily_opens_yfinance(symbols, start=start, end=end)

    try:
        return fetch_daily_opens_yfinance(symbols, start=start, end=end)
    except Exception as e_yf:  # noqa: BLE001
        # Fallback to Alpaca if available (requires keys), then try Stooq.
        try:
            return fetch_daily_opens_alpaca(symbols, start=start, end=end)
        except Exception as e_alp:  # noqa: BLE001
            try:
                return fetch_daily_opens_stooq(symbols, start=start, end=end)
            except Exception as e_stq:  # noqa: BLE001
                raise RuntimeError(
                    "No usable price source for daily opens. "
                    f"yfinance failed: {e_yf}; alpaca failed: {e_alp}; stooq failed: {e_stq}. "
                    "Set ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY, provide STOOQ_APIKEY, or retry later."
                )


def fetch_daily_closes(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Fetch daily close prices with fallback."""
    import os

    # Prefer Stooq when apikey is provided (good for long history).
    if os.getenv("STOOQ_APIKEY"):
        try:
            return fetch_daily_closes_stooq(symbols, start=start, end=end)
        except Exception:
            pass

    # Prefer Alpaca when credentials are available (Yahoo is often rate-limited).
    key, secret = _get_alpaca_creds()
    if key and secret:
        try:
            ohlc = fetch_daily_ohlc_alpaca(symbols, start=start, end=end)
            closes = ohlc["close"].copy()
            closes.index = closes.index
            return closes
        except Exception:
            df = _yfinance_download_with_retry(symbols=symbols, start=start, end=end)
            if isinstance(df.columns, pd.MultiIndex):
                closes = pd.DataFrame({s: df[s]["Close"] for s in symbols})
            else:
                closes = pd.DataFrame({symbols[0]: df["Close"]})
            closes.index = pd.to_datetime(closes.index).date
            return closes.sort_index().dropna(how="all")

    try:
        df = _yfinance_download_with_retry(symbols=symbols, start=start, end=end)
        if isinstance(df.columns, pd.MultiIndex):
            closes = pd.DataFrame({s: df[s]["Close"] for s in symbols})
        else:
            closes = pd.DataFrame({symbols[0]: df["Close"]})
        closes.index = pd.to_datetime(closes.index).date
        return closes.sort_index().dropna(how="all")
    except Exception as e_yf:  # noqa: BLE001
        # Fallback to Alpaca bars, then try Stooq.
        try:
            ohlc = fetch_daily_ohlc_alpaca(symbols, start=start, end=end)
            closes = ohlc["close"].copy()
            closes.index = closes.index
            return closes
        except Exception as e_alp:  # noqa: BLE001
            try:
                return fetch_daily_closes_stooq(symbols, start=start, end=end)
            except Exception as e_stq:  # noqa: BLE001
                raise RuntimeError(
                    "No usable price source for daily closes. "
                    f"yfinance failed: {e_yf}; alpaca failed: {e_alp}; stooq failed: {e_stq}. "
                    "Set ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY, provide STOOQ_APIKEY, or retry later."
                )
