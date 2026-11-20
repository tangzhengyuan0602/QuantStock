from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(s: pd.Series, win: int = 20) -> pd.Series:
    mu = s.rolling(win, min_periods=max(5, win // 4)).mean()
    sd = s.rolling(win, min_periods=max(5, win // 4)).std(ddof=0)
    z = (s - mu) / sd.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def momentum(close: pd.Series, lookback: int = 60) -> pd.Series:
    r = close / close.shift(lookback) - 1.0
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def trend_filter(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    ma_fast = close.rolling(fast, min_periods=max(10, fast // 2)).mean()
    ma_slow = close.rolling(slow, min_periods=max(20, slow // 2)).mean()
    ok = (ma_fast > ma_slow).astype(float)
    return ok.fillna(0.0)


def realized_vol(close: pd.Series, win: int = 20) -> pd.Series:
    # close-to-close vol (annualized)
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    vol = ret.rolling(win, min_periods=max(5, win // 4)).std(ddof=0) * np.sqrt(252.0)
    return vol.fillna(0.0)


def clip(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return x.clip(lower=lo, upper=hi)


def sigmoid(x: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x: pd.Series) -> pd.Series:
    return np.tanh(x)

