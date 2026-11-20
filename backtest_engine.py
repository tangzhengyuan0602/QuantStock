from __future__ import annotations

import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame  # index: date, columns: equity, ret, drawdown
    trades: pd.DataFrame  # index: date(open), columns: weights/positions
    metrics: dict


def _max_drawdown(equity: Sequence[float]) -> float:
    peak = -math.inf
    mdd = 0.0
    for x in equity:
        peak = max(peak, x)
        if peak > 0:
            mdd = min(mdd, x / peak - 1.0)
    return float(mdd)


def compute_metrics(daily_returns: pd.Series) -> dict:
    r = daily_returns.dropna().astype(float)
    if len(r) == 0:
        return {"cagr": 0.0, "sharpe": 0.0, "mdd": 0.0, "days": 0}
    equity = (1.0 + r).cumprod()
    years = len(r) / 252.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(r.std(ddof=0))
    sharpe = float((r.mean() / vol) * math.sqrt(252.0)) if vol > 0 else 0.0
    mdd = _max_drawdown(equity.values)
    return {
        "days": int(len(r)),
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": float(mdd),
        "total_return": float(equity.iloc[-1] - 1.0),
        "avg_daily": float(r.mean()),
        "vol_daily": vol,
    }


def run_open_to_open_backtest(
    *,
    opens: pd.DataFrame,
    positions: pd.DataFrame,
    weights: Dict[str, float] | None = None,
    cost_bps: float = 1.0,
    max_drawdown_stop: float | None = None,
) -> BacktestResult:
    """Backtest with trades at open and PnL to next open.

    opens: index=date, columns=symbol, values=open price
    positions: index=date (open trade date), columns=symbol, values in {-1,0,1}
      Position held from date open -> next date open.
    weights: per-symbol capital weight (default equal weights across provided symbols)
    cost_bps: transaction cost in basis points per 1.0 notional turnover
    """
    symbols = list(opens.columns)
    w = weights or {s: 1.0 / len(symbols) for s in symbols}

    px = opens[symbols].copy().sort_index()
    # open-to-next-open returns
    next_open = px.shift(-1)
    sym_ret = (next_open / px) - 1.0
    sym_ret = sym_ret.iloc[:-1]

    pos = positions.reindex(sym_ret.index).fillna(0.0).astype(float)

    # Optional: max drawdown stop. If equity drawdown breaches threshold,
    # set positions to 0 from the next day onward (stay in cash).
    if max_drawdown_stop is not None:
        stop = float(max_drawdown_stop)
        if stop >= 0:
            raise ValueError("max_drawdown_stop should be negative, e.g. -0.1")

    port_ret = pd.Series(0.0, index=sym_ret.index)

    stop_from: pd.Timestamp | dt.date | None = None
    if max_drawdown_stop is not None:
        equity = 1.0
        peak = 1.0
        for i, d in enumerate(sym_ret.index):
            if stop_from is not None and d >= stop_from:
                port_ret.loc[d] = 0.0
                continue
            r = 0.0
            for s in symbols:
                r += float(w.get(s, 0.0)) * float(pos.at[d, s]) * float(sym_ret.at[d, s])
            port_ret.loc[d] = r
            equity *= (1.0 + r)
            peak = max(peak, equity)
            dd = equity / peak - 1.0
            if dd <= stop:
                # Stop trading from next day (if exists).
                if i + 1 < len(sym_ret.index):
                    stop_from = sym_ret.index[i + 1]
                else:
                    stop_from = d
    else:
        for d in sym_ret.index:
            r = 0.0
            for s in symbols:
                r += float(w.get(s, 0.0)) * float(pos.at[d, s]) * float(sym_ret.at[d, s])
            port_ret.loc[d] = r

    if stop_from is not None:
        pos.loc[pos.index >= stop_from, :] = 0.0

    # transaction costs based on turnover in target weights at open
    target_w = pos.copy()
    for s in symbols:
        target_w[s] = target_w[s] * float(w.get(s, 0.0))
    turnover = (target_w.diff().abs().sum(axis=1)).fillna(target_w.abs().sum(axis=1))
    cost = turnover * (cost_bps / 10000.0)
    port_ret = port_ret - cost

    equity = (1.0 + port_ret).cumprod()
    dd = equity / equity.cummax() - 1.0
    daily = pd.DataFrame({"equity": equity, "ret": port_ret, "drawdown": dd})
    metrics = compute_metrics(port_ret)

    trades = pd.DataFrame(index=sym_ret.index)
    for s in symbols:
        trades[f"pos_{s}"] = pos[s]
        trades[f"w_{s}"] = target_w[s]
    trades["turnover"] = turnover
    trades["cost"] = cost
    return BacktestResult(daily=daily, trades=trades, metrics=metrics)
