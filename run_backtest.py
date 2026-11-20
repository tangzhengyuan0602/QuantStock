from __future__ import annotations

import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from backtest_engine import compute_metrics, run_open_to_open_backtest
from news_pipeline import load_or_fetch_alpaca_daily_multi
from price_data import fetch_daily_closes, fetch_daily_opens
from sentiment_models import build_scorer
from sentiment_pipeline import load_or_score_batch
from signal_utils import score_to_position
from signal_utils import texts_from_news
from feature_engine import momentum, realized_vol, rolling_zscore, tanh, trend_filter


def _prev_trading_day(trading_days: list[dt.date], i: int) -> dt.date:
    if i <= 0:
        raise ValueError("No previous trading day")
    return trading_days[i - 1]


def _date_range_inclusive(a: dt.date, b: dt.date) -> list[dt.date]:
    if b < a:
        return []
    out: list[dt.date] = []
    d = a
    while d <= b:
        out.append(d)
        d = d + dt.timedelta(days=1)
    return out


def _merge_news_over_days(
    *,
    cache_dir: Path,
    symbols: list[str],
    days: list[dt.date],
    limit: int,
    offline_cache_only: bool,
) -> dict[str, list[dict]]:
    """Merge Alpaca daily news across multiple calendar days.

    Used to include weekend/holiday news for the next trading day.
    """

    merged: dict[str, list[dict]] = {s: [] for s in symbols}
    seen_url: set[str] = set()
    for day in days:
        per = load_or_fetch_alpaca_daily_multi(
            cache_dir=cache_dir,
            symbols=symbols,
            day=day,
            limit=limit,
            offline_cache_only=offline_cache_only,
        )
        for s in symbols:
            for it in (per.get(s) or []):
                url = str(it.get("url") or "")
                if url and url in seen_url:
                    continue
                if url:
                    seen_url.add(url)
                merged[s].append(it)
    return merged


def _is_rebalance_day_from_trading_days(*, trading_days: list[dt.date], i: int, mode: str) -> bool:
    """Whether trading_days[i] is the first trading day of week/month in the series."""

    if i <= 0 or i >= len(trading_days):
        return False
    mode = (mode or "").lower().strip()
    if mode not in {"weekly", "month", "monthly"}:
        raise ValueError(f"Unknown rebalance mode: {mode}")

    d = trading_days[i]
    prev = trading_days[i - 1]
    if mode == "weekly":
        return d.isocalendar()[:2] != prev.isocalendar()[:2]
    return (d.year, d.month) != (prev.year, prev.month)


def build_positions_rotation(
    *,
    closes: pd.DataFrame,
    trading_days: list[dt.date],
    symbols: list[str],
    rebalance: str,
    mom_lookback: int,
    top_k: int,
    cash_if_negative: bool,
    riskoff_trend: bool,
    riskoff_fast: int,
    riskoff_slow: int,
) -> pd.DataFrame:
    """Strategy2: momentum rotation (weekly/monthly rebalance) backtest positions.

    Positions are target portfolio weights held from trade_day open -> next open.
    Signals are computed from closes as-of signal_day (previous trading day).
    """

    if mom_lookback < 5:
        raise ValueError("mom_lookback too small")
    top_k = max(1, int(top_k))

    pos = pd.DataFrame(0.0, index=trading_days[1:], columns=symbols, dtype=float)
    cur = pd.Series(0.0, index=symbols, dtype=float)
    bench = "SPY" if "SPY" in symbols else symbols[0]

    for i in range(1, len(trading_days)):
        trade_day = trading_days[i]
        signal_day = trading_days[i - 1]

        if _is_rebalance_day_from_trading_days(trading_days=trading_days, i=i, mode=rebalance):
            mom_last: dict[str, float] = {}
            for sym in symbols:
                s = closes[sym].astype(float).dropna()
                s = s[s.index <= signal_day]
                if len(s) <= mom_lookback:
                    mom_last[sym] = 0.0
                else:
                    mom_last[sym] = float(s.iloc[-1] / s.iloc[-1 - mom_lookback] - 1.0)
            ranked = sorted(mom_last.items(), key=lambda kv: kv[1], reverse=True)
            if cash_if_negative:
                picks = [sym for sym, m in ranked if float(m) > 0.0][:top_k]
            else:
                picks = [sym for sym, _ in ranked[:top_k]]

            # Trend risk-off gate (applied at rebalance decision).
            if riskoff_trend and bench in closes.columns:
                s_b = closes[bench].astype(float).dropna()
                s_b = s_b[s_b.index <= signal_day]
                tf = trend_filter(s_b, fast=riskoff_fast, slow=riskoff_slow)
                tf_v = float(tf.iloc[-1]) if len(tf) else 0.0
                if tf_v <= 0.0:
                    picks = []

            cur = pd.Series(0.0, index=symbols, dtype=float)
            if picks:
                for sym in picks:
                    cur[sym] = 1.0 / float(len(picks))

        pos.loc[trade_day] = cur

    return pos


def build_positions_from_sentiment(
    *,
    sentiments: pd.DataFrame,
    threshold: float,
    long_short: bool,
    allocation: str = "independent",
) -> pd.DataFrame:
    allocation = allocation.lower().strip()
    cols = list(sentiments.columns)
    pos = pd.DataFrame(0.0, index=sentiments.index, columns=cols, dtype=float)

    if allocation == "independent" or len(cols) == 1:
        for s in cols:
            pos[s] = sentiments[s].apply(
                lambda x: score_to_position(float(x), threshold=threshold, long_short=long_short)
            )
        return pos

    if allocation in {"winner", "winner_takes_all"}:
        # Each day pick the symbol with largest |score| and allocate position only to it.
        for d, row in sentiments.iterrows():
            best = max(cols, key=lambda s: abs(float(row[s])))
            pos.loc[d, best] = score_to_position(float(row[best]), threshold=threshold, long_short=long_short)
        return pos

    raise ValueError(f"Unknown allocation: {allocation}")


def build_positions_enhanced(
    *,
    sentiments: pd.DataFrame,
    closes: pd.DataFrame,
    threshold: float,
    long_short: bool,
    gate_mode: str = "hybrid",
    target_vol: float = 0.12,
    vol_lookback: int = 20,
    z_win: int = 20,
    mom_lb: int = 60,
    trend_fast: int = 50,
    trend_slow: int = 200,
    score_w: float = 1.0,
    z_w: float = 0.7,
    mom_w: float = 0.6,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    """Enhanced signal: sentiment z-score + momentum + trend filter + vol targeting.

    Output positions are continuous weights in [-max_leverage, +max_leverage].
    """
    idx = sentiments.index
    syms = list(sentiments.columns)
    pos = pd.DataFrame(0.0, index=idx, columns=syms, dtype=float)
    gate_mode = str(gate_mode).lower().strip()
    if gate_mode not in {"hybrid", "sentiment_only"}:
        raise ValueError(f"Unknown gate_mode: {gate_mode}")

    for sym in syms:
        s = sentiments[sym].astype(float)
        z = rolling_zscore(s, win=z_win)

        close_full = closes[sym].astype(float)
        mom_full = momentum(close_full, lookback=mom_lb)
        tf_full = trend_filter(close_full, fast=trend_fast, slow=trend_slow)
        vol_full = realized_vol(close_full, win=vol_lookback)

        mom = mom_full.reindex(idx).fillna(0.0)
        tf = tf_full.reindex(idx).fillna(0.0)
        vol = vol_full.reindex(idx).fillna(0.0)
        vol_scale = (target_vol / vol.replace(0.0, np.nan)).fillna(0.0).clip(0.0, 1.5)

        raw = score_w * s + z_w * z + mom_w * mom
        strength = tanh(raw)  # [-1,1]
        sign_ = np.sign(raw)
        if not long_short:
            sign_ = (raw > 0).astype(float)

        # gate by news signal strength (avoid trading on tiny/noisy scores)
        if gate_mode == "sentiment_only":
            if long_short:
                gate = (s.abs() >= threshold).astype(float)
            else:
                gate = (s >= threshold).astype(float)
        else:
            gate = ((z.abs() >= 0.5) | (s.abs() >= threshold)).astype(float)
        pos[sym] = (sign_ * strength.abs() * gate * tf * vol_scale).clip(-max_leverage, max_leverage)

    # Cross-sectional relative strength: allocate more to stronger momentum when both signals are same sign
    if len(syms) >= 2:
        # Use momentum ranks
        mom_df = pd.DataFrame({sym: momentum(closes[sym].astype(float), lookback=mom_lb) for sym in syms}).reindex(idx)
        rank = mom_df.rank(axis=1, method="average")
        # scale ranks to [-0.5,0.5]
        rs = (rank - rank.mean(axis=1).values.reshape(-1, 1))
        rs = rs / (rs.abs().sum(axis=1).replace(0.0, np.nan).values.reshape(-1, 1))
        rs = rs.fillna(0.0)
        pos = (pos + 0.25 * rs).clip(-max_leverage, max_leverage)

    return pos


def _compute_risk_mask(*, index: pd.Index, symbols: list[str], closes: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    """Risk-off mask (1=trade, 0=cash) aligned to the provided trade-day index."""

    mask = pd.Series(1.0, index=index)
    if (not bool(args.riskoff_trend)) and (args.riskoff_vol_max is None):
        return mask

    bench = "SPY" if "SPY" in symbols else symbols[0]
    bench_close = closes[bench].astype(float)
    if bool(args.riskoff_trend):
        tf_bench = trend_filter(bench_close, fast=args.riskoff_fast, slow=args.riskoff_slow).reindex(mask.index)
        mask = mask * tf_bench.fillna(0.0)
    if args.riskoff_vol_max is not None:
        vol_bench = realized_vol(bench_close, win=int(args.riskoff_vol_win)).reindex(mask.index)
        mask = mask * (vol_bench.fillna(0.0) <= float(args.riskoff_vol_max)).astype(float)
    return mask


def _opens_for_trade_days(opens: pd.DataFrame, trade_days: list[dt.date] | pd.Index) -> pd.DataFrame:
    """Slice opens so that open->next-open returns exist for all trade_days."""

    td = list(trade_days)
    if not td:
        raise ValueError("empty trade_days")

    open_days = list(opens.index)
    start = td[0]
    end_trade = td[-1]
    if start not in opens.index or end_trade not in opens.index:
        raise ValueError("trade_days must be within opens.index")

    j0 = open_days.index(start)
    j1 = open_days.index(end_trade)
    # include next day open for last trade day
    j_end = min(j1 + 1, len(open_days) - 1)
    need = open_days[j0 : j_end + 1]
    return opens.loc[need]


def _grid_search_best(
    *,
    sentiments: pd.DataFrame,
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    symbols: list[str],
    args: argparse.Namespace,
    max_dd_stop: float | None,
) -> tuple[dict, dict]:
    """Return (best_cfg, best_train_metrics) under optional mdd constraint."""

    risk_mask = _compute_risk_mask(index=sentiments.index, symbols=symbols, closes=closes, args=args)
    best = None
    best_any = None

    thr_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    ls_list = [False, True]

    trade_days = list(sentiments.index)
    opens_slice = _opens_for_trade_days(opens, trade_days)

    for thr in thr_list:
        for ls in ls_list:
            if args.signal_mode == "enhanced":
                for mom_lb in [40, 60, 90]:
                    for z_win in [10, 20, 40]:
                        pos2 = build_positions_enhanced(
                            sentiments=sentiments,
                            closes=closes,
                            threshold=thr,
                            long_short=ls,
                            gate_mode=args.enhanced_gate_mode,
                            target_vol=args.target_vol,
                            mom_lb=mom_lb,
                            z_win=z_win,
                            trend_fast=args.trend_fast,
                            trend_slow=args.trend_slow,
                        )
                        pos2 = pos2.mul(risk_mask, axis=0)
                        r2 = run_open_to_open_backtest(
                            opens=opens_slice,
                            positions=pos2,
                            cost_bps=args.cost_bps,
                            max_drawdown_stop=max_dd_stop,
                        )
                        m = r2.metrics
                        key = (m["cagr"], m["sharpe"])
                        cfg = {
                            "signal_mode": "enhanced",
                            "threshold": thr,
                            "long_short": ls,
                            "mom_lb": mom_lb,
                            "z_win": z_win,
                        }
                        if best_any is None or key > best_any[0]:
                            best_any = (key, cfg, m)
                        if (args.mdd_limit is None) or (abs(float(m["mdd"])) <= float(args.mdd_limit)):
                            if best is None or key > best[0]:
                                best = (key, cfg, m)
            else:
                for alloc in ["independent", "winner"]:
                    pos2 = build_positions_from_sentiment(
                        sentiments=sentiments,
                        threshold=thr,
                        long_short=ls,
                        allocation=alloc,
                    )
                    pos2 = pos2.mul(risk_mask, axis=0)
                    r2 = run_open_to_open_backtest(
                        opens=opens_slice,
                        positions=pos2,
                        cost_bps=args.cost_bps,
                        max_drawdown_stop=max_dd_stop,
                    )
                    m = r2.metrics
                    key = (m["cagr"], m["sharpe"])
                    cfg = {
                        "signal_mode": "baseline",
                        "threshold": thr,
                        "long_short": ls,
                        "allocation": alloc,
                    }
                    if best_any is None or key > best_any[0]:
                        best_any = (key, cfg, m)
                    if (args.mdd_limit is None) or (abs(float(m["mdd"])) <= float(args.mdd_limit)):
                        if best is None or key > best[0]:
                            best = (key, cfg, m)

    chosen = best if best is not None else best_any
    if chosen is None:
        return ({}, {})
    _, cfg, m = chosen
    return cfg, m


def _build_positions_from_cfg(
    *,
    cfg: dict,
    sentiments: pd.DataFrame,
    closes: pd.DataFrame,
    symbols: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if not cfg:
        return pd.DataFrame(0.0, index=sentiments.index, columns=symbols, dtype=float)

    if cfg.get("signal_mode") == "enhanced":
        pos = build_positions_enhanced(
            sentiments=sentiments,
            closes=closes,
            threshold=float(cfg["threshold"]),
            long_short=bool(cfg["long_short"]),
            gate_mode=args.enhanced_gate_mode,
            target_vol=args.target_vol,
            mom_lb=int(cfg["mom_lb"]),
            z_win=int(cfg["z_win"]),
            trend_fast=args.trend_fast,
            trend_slow=args.trend_slow,
        )
    else:
        pos = build_positions_from_sentiment(
            sentiments=sentiments,
            threshold=float(cfg["threshold"]),
            long_short=bool(cfg["long_short"]),
            allocation=str(cfg.get("allocation") or "independent"),
        )

    risk_mask = _compute_risk_mask(index=sentiments.index, symbols=symbols, closes=closes, args=args)
    return pos.mul(risk_mask, axis=0)


def _walk_forward_validate(
    *,
    sentiments: pd.DataFrame,
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    symbols: list[str],
    args: argparse.Namespace,
    max_dd_stop: float | None,
) -> None:
    """Walk-forward: re-fit params on rolling train window, evaluate on next test window."""

    trade_days = list(sentiments.index)
    train_n = int(float(args.wf_train_years) * 252)
    test_n = int(float(args.wf_test_years) * 252)
    step_n = int(float(args.wf_step_years) * 252)
    if train_n < 60 or test_n < 20 or step_n < 20:
        raise ValueError("walk-forward window too small")

    rows = []
    oos_pos = pd.DataFrame(0.0, index=sentiments.index, columns=symbols, dtype=float)

    i0 = 0
    while (i0 + train_n + test_n) <= len(trade_days):
        train_days = trade_days[i0 : i0 + train_n]
        test_days = trade_days[i0 + train_n : i0 + train_n + test_n]
        s_train = sentiments.loc[train_days]
        s_test = sentiments.loc[test_days]

        cfg, m_train = _grid_search_best(
            sentiments=s_train,
            closes=closes,
            opens=opens,
            symbols=symbols,
            args=args,
            max_dd_stop=max_dd_stop,
        )
        pos_test = _build_positions_from_cfg(cfg=cfg, sentiments=s_test, closes=closes, symbols=symbols, args=args)
        oos_pos.loc[test_days, :] = pos_test.reindex(oos_pos.index).loc[test_days, :]

        # Per-window test metrics (for diagnostics)
        opens_test = _opens_for_trade_days(opens, test_days)
        r_test = run_open_to_open_backtest(
            opens=opens_test,
            positions=pos_test,
            cost_bps=args.cost_bps,
            max_drawdown_stop=max_dd_stop,
        )
        rows.append(
            {
                "train_start": str(train_days[0]),
                "train_end": str(train_days[-1]),
                "test_start": str(test_days[0]),
                "test_end": str(test_days[-1]),
                **cfg,
                "train_cagr": float(m_train.get("cagr", 0.0)),
                "train_sharpe": float(m_train.get("sharpe", 0.0)),
                "train_mdd": float(m_train.get("mdd", 0.0)),
                "test_cagr": float(r_test.metrics.get("cagr", 0.0)),
                "test_sharpe": float(r_test.metrics.get("sharpe", 0.0)),
                "test_mdd": float(r_test.metrics.get("mdd", 0.0)),
            }
        )

        i0 += step_n

    if not rows:
        print("===== Walk-forward (OOS) =====")
        print({"error": "not enough data for walk-forward windows"})
        return

    # Overall OOS backtest on stitched positions (keeps turnover continuity)
    first_test = dt.date.fromisoformat(rows[0]["test_start"])
    last_test = dt.date.fromisoformat(rows[-1]["test_end"])
    # Slice sentiments/positions to contiguous OOS span
    oos_trade_days = [d for d in trade_days if (d >= first_test and d <= last_test)]
    pos_oos = oos_pos.loc[oos_trade_days]
    opens_oos = _opens_for_trade_days(opens, oos_trade_days)
    res_oos = run_open_to_open_backtest(
        opens=opens_oos,
        positions=pos_oos,
        cost_bps=args.cost_bps,
        max_drawdown_stop=max_dd_stop,
    )

    print("===== Walk-forward (OOS) =====")
    print({k: round(v, 6) if isinstance(v, float) else v for k, v in res_oos.metrics.items()})
    # Print compact window table
    preview = rows[: min(len(rows), 12)]
    print("===== Walk-forward Windows (first 12) =====")
    for r in preview:
        out = {
            "train": f"{r['train_start']}..{r['train_end']}",
            "test": f"{r['test_start']}..{r['test_end']}",
            "cfg": {k: r[k] for k in ["signal_mode", "threshold", "long_short", "allocation", "mom_lb", "z_win"] if k in r},
            "train": {"cagr": round(r["train_cagr"], 6), "sharpe": round(r["train_sharpe"], 6), "mdd": round(r["train_mdd"], 6)},
            "test": {"cagr": round(r["test_cagr"], 6), "sharpe": round(r["test_sharpe"], 6), "mdd": round(r["test_mdd"], 6)},
        }
        print(out)


def _holdout_validate(
    *,
    sentiments: pd.DataFrame,
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    symbols: list[str],
    args: argparse.Namespace,
    max_dd_stop: float | None,
) -> None:
    """Single holdout: fit on train window, test on last holdout window."""

    trade_days = list(sentiments.index)
    holdout_n = int(float(args.holdout_years) * 252)
    train_n = int(float(args.holdout_train_years) * 252)
    if holdout_n < 20 or train_n < 60:
        raise ValueError("holdout window too small")
    if (train_n + holdout_n) > len(trade_days):
        print("===== Holdout (OOS) =====")
        print({"error": "not enough data for holdout split"})
        return

    test_days = trade_days[-holdout_n:]
    train_days = trade_days[-(holdout_n + train_n) : -holdout_n]
    s_train = sentiments.loc[train_days]
    s_test = sentiments.loc[test_days]

    cfg, m_train = _grid_search_best(
        sentiments=s_train,
        closes=closes,
        opens=opens,
        symbols=symbols,
        args=args,
        max_dd_stop=max_dd_stop,
    )
    pos_test = _build_positions_from_cfg(cfg=cfg, sentiments=s_test, closes=closes, symbols=symbols, args=args)
    opens_test = _opens_for_trade_days(opens, test_days)
    res_test = run_open_to_open_backtest(
        opens=opens_test,
        positions=pos_test,
        cost_bps=args.cost_bps,
        max_drawdown_stop=max_dd_stop,
    )

    print("===== Holdout (OOS) =====")
    print(
        {
            "train": f"{train_days[0]}..{train_days[-1]}",
            "test": f"{test_days[0]}..{test_days[-1]}",
            "best_cfg": cfg,
            "train_metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in m_train.items()},
            "test_metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in res_test.metrics.items()},
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategy",
        default="strategy1",
        choices=["strategy1", "strategy2"],
        help="strategy1=overnight sentiment; strategy2=momentum rotation",
    )
    ap.add_argument("--symbols", default="SPY,QQQ", help="Comma-separated symbols")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional)")
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD (default today)")
    ap.add_argument("--max-articles", type=int, default=10)
    ap.add_argument(
        "--aggregate-mode",
        default="per_article",
        choices=["per_article", "per_symbol_summary"],
        help="情绪聚合方式：逐条新闻打分 vs 每日按标的汇总后再打分（更省调用、更利用模型总结能力）",
    )
    ap.add_argument(
        "--scorer",
        default="finbert",
        choices=["finbert", "openai", "ark", "ark_struct", "glm", "lexicon", "ensemble", "ensemble_agree"],
    )
    ap.add_argument(
        "--out-perf-csv",
        default=None,
        help="Optional path to save daily backtest perf (date,equity,ret,drawdown).",
    )
    ap.add_argument(
        "--hf-endpoint",
        default=None,
        help="HuggingFace 镜像站，例如 https://hf-mirror.com （会设置环境变量 HF_ENDPOINT）",
    )
    ap.add_argument("--news-source", default="alpaca", choices=["alpaca"], help="News source")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--long-short", action="store_true", help="Enable shorting when score <= -threshold")
    ap.add_argument("--allocation", default="independent", choices=["independent", "winner"], help="Capital allocation mode")
    ap.add_argument("--signal-mode", default="baseline", choices=["baseline", "enhanced"], help="Signal engine")
    ap.add_argument(
        "--enhanced-gate-mode",
        default="hybrid",
        choices=["hybrid", "sentiment_only"],
        help="增强信号门控：hybrid=允许z-score参与开仓，sentiment_only=必须由情绪阈值直接触发",
    )
    ap.add_argument("--target-vol", type=float, default=0.12)
    ap.add_argument("--mom-lb", type=int, default=60)
    ap.add_argument("--z-win", type=int, default=20)
    ap.add_argument("--trend-fast", type=int, default=50)
    ap.add_argument("--trend-slow", type=int, default=200)

    # Strategy2 (rotation) params
    ap.add_argument("--s2-rebalance", default="weekly", choices=["weekly", "monthly"], help="Strategy2: rebalance freq")
    ap.add_argument("--s2-mom-lookback", type=int, default=60, help="Strategy2: momentum lookback")
    ap.add_argument("--s2-top-k", type=int, default=3, help="Strategy2: pick top K")
    ap.add_argument(
        "--s2-cash-if-negative",
        action="store_true",
        help="Strategy2: absolute momentum filter (mom<=0 -> stay in cash)",
    )
    ap.add_argument(
        "--riskoff-trend",
        action="store_true",
        help="风险控制：当基准（默认 SPY）不满足趋势条件时清仓持币",
    )
    ap.add_argument("--riskoff-fast", type=int, default=50)
    ap.add_argument("--riskoff-slow", type=int, default=200)
    ap.add_argument(
        "--riskoff-vol-max",
        type=float,
        default=None,
        help="风险控制：当基准年化波动率超过该阈值时清仓持币（例如 0.25）",
    )
    ap.add_argument("--riskoff-vol-win", type=int, default=20)
    ap.add_argument(
        "--max-dd-stop",
        type=float,
        default=None,
        help="最大回撤熔断（例如 0.1 表示回撤超过 10% 后后续全程持币）",
    )
    ap.add_argument(
        "--mdd-limit",
        type=float,
        default=None,
        help="优化约束：只在 max drawdown <= 该阈值（例如 0.15）内挑选最优参数",
    )
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / "data_cache"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--optimize", action="store_true", help="Grid-search thresholds on computed sentiments")
    ap.add_argument(
        "--validate",
        action="store_true",
        help="做样本外验证：holdout + walk-forward（每个窗口在训练集上重新选参数）",
    )
    ap.add_argument("--holdout-years", type=float, default=1.0, help="holdout 测试集长度（年）")
    ap.add_argument("--holdout-train-years", type=float, default=3.0, help="holdout 训练集长度（年）")
    ap.add_argument("--wf-train-years", type=float, default=3.0, help="walk-forward 每窗训练长度（年）")
    ap.add_argument("--wf-test-years", type=float, default=1.0, help="walk-forward 每窗测试长度（年）")
    ap.add_argument("--wf-step-years", type=float, default=1.0, help="walk-forward 滚动步长（年）")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    ap.add_argument(
        "--offline-cache-only",
        action="store_true",
        help="只使用本地缓存（新闻/情绪），缓存缺失则当日视为无新闻（避免大量 API 调用）",
    )
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    start = dt.date.fromisoformat(args.start) if args.start else (end - dt.timedelta(days=365 * args.years))
    cache_dir = Path(args.cache_dir)

    if args.hf_endpoint:
        import os

        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    # Alpaca news is rate-limited; cap concurrency to reduce 429s.
    # NOTE: local transformers pipeline (FinBERT) is not reliably thread-safe on CPU.
    # Lexicon scorer is pure Python and safe to parallelize.
    if args.scorer in {"finbert"}:
        workers = 1
    else:
        workers = min(int(args.workers), 4)

    opens = fetch_daily_opens(symbols, start=start, end=end)

    # Fetch extra history for feature computation (MA/vol/momentum), then align to trade dates.
    feat_days = max(
        450,
        int(args.mom_lb) * 3,
        int(args.trend_slow) * 2,
        int(args.s2_mom_lookback) * 3,
        int(args.riskoff_slow) * 2,
    )
    feat_start = start - dt.timedelta(days=feat_days)
    closes = fetch_daily_closes(symbols, start=feat_start, end=end)
    trading_days = list(opens.index)
    if len(trading_days) < 10:
        raise RuntimeError("Not enough price history")

    max_dd_stop = None
    if args.max_dd_stop is not None:
        max_dd_stop = -abs(float(args.max_dd_stop))

    if args.strategy == "strategy2":
        positions = build_positions_rotation(
            closes=closes,
            trading_days=trading_days,
            symbols=symbols,
            rebalance=args.s2_rebalance,
            mom_lookback=int(args.s2_mom_lookback),
            top_k=int(args.s2_top_k),
            cash_if_negative=bool(args.s2_cash_if_negative),
            riskoff_trend=bool(args.riskoff_trend),
            riskoff_fast=int(args.riskoff_fast),
            riskoff_slow=int(args.riskoff_slow),
        )
        res = run_open_to_open_backtest(
            opens=opens,
            positions=positions,
            weights={s: 1.0 for s in symbols},
            cost_bps=args.cost_bps,
            max_drawdown_stop=max_dd_stop,
        )
        print("===== Backtest Summary =====")
        print({k: round(v, 6) if isinstance(v, float) else v for k, v in res.metrics.items()})
        if args.out_perf_csv:
            out_df = res.daily.copy()
            out_df.index.name = "date"
            out_df.to_csv(args.out_perf_csv)
            print(f"saved daily perf -> {args.out_perf_csv}")
        return 0

    scorer = build_scorer(args.scorer)

    # Compute sentiment per trading day t-1, then align to trade day t.
    sentiments = pd.DataFrame(index=trading_days[1:], columns=symbols, dtype=float)

    # Parallelize by trade_day (fetch news once per day, split by symbol)
    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(1, len(trading_days)):
            trade_day = trading_days[i]
            trade_day_prev = _prev_trading_day(trading_days, i)
            td = trade_day
            nd = trade_day_prev

            def _job(trade_day=td, news_day=nd):
                # Include weekend/holiday news: merge calendar days from signal day (previous
                # trading day) to (trade_day - 1). Example: Monday trade_day uses Fri+Sat+Sun.
                news_days = _date_range_inclusive(news_day, trade_day - dt.timedelta(days=1))
                span_tag = f":span={len(news_days)}" if len(news_days) != 1 else ""
                per_sym = _merge_news_over_days(
                    cache_dir=cache_dir,
                    symbols=symbols,
                    days=news_days,
                    limit=max(50, args.max_articles),
                    offline_cache_only=bool(args.offline_cache_only),
                )
                # Build one batched call for all symbols to cut LLM requests.
                if args.aggregate_mode == "per_symbol_summary":
                    # Build 1 combined text per symbol; ask model to summarize implicitly.
                    combined = []
                    for sym in symbols:
                        arts = (per_sym.get(sym) or [])[: args.max_articles]
                        texts = texts_from_news(arts, max_items=args.max_articles)
                        if texts:
                            blob = "\n".join([f"- {t}" for t in texts])
                            combined.append(f"Symbol={sym}\nNews:\n{blob}")
                        else:
                            combined.append(f"Symbol={sym}\nNews:\n(no news)")

                    packed_scores = load_or_score_batch(
                        cache_dir=cache_dir,
                        namespace=args.news_source,
                        symbol="DAILY",
                        day=news_day,
                        batch_tag=f"daily:{args.scorer}:{','.join(symbols)}:{args.max_articles}{span_tag}",
                        texts=combined,
                        scorer=scorer,
                    )
                    scores = {sym: float(sc) for sym, sc in zip(symbols, packed_scores, strict=False)}
                    for sym in symbols:
                        scores.setdefault(sym, 0.0)
                    return trade_day, scores

                # Default: per-article scoring then average.
                packed_texts = []
                index = []  # list of (sym, offset)
                for sym in symbols:
                    arts = (per_sym.get(sym) or [])[: args.max_articles]
                    texts = texts_from_news(arts, max_items=args.max_articles)
                    index.extend([(sym, i) for i in range(len(texts))])
                    packed_texts.extend(texts)

                if packed_texts:
                    packed_scores = load_or_score_batch(
                        cache_dir=cache_dir,
                        namespace=args.news_source,
                        symbol="BATCH",
                        day=news_day,
                        batch_tag=f"{args.scorer}:{','.join(symbols)}:{args.max_articles}{span_tag}",
                        texts=packed_texts,
                        scorer=scorer,
                    )
                else:
                    packed_scores = []

                per_sym_scores = {s: [] for s in symbols}
                for (sym, _), sc in zip(index, packed_scores, strict=False):
                    per_sym_scores[sym].append(float(sc))

                from signal_utils import aggregate_scores

                scores = {sym: float(aggregate_scores(per_sym_scores.get(sym) or [], method="mean")) for sym in symbols}
                return trade_day, scores

            tasks.append(ex.submit(_job))

        for fut in tqdm(
            as_completed(tasks),
            total=len(tasks),
            desc=f"news+sentiment (workers={workers})",
            disable=bool(args.no_progress),
        ):
            trade_day, scores = fut.result()
            for sym, score in scores.items():
                sentiments.loc[trade_day, sym] = score

    if args.signal_mode == "enhanced":
        positions = build_positions_enhanced(
            sentiments=sentiments,
            closes=closes,
            threshold=args.threshold,
            long_short=args.long_short,
            gate_mode=args.enhanced_gate_mode,
            target_vol=args.target_vol,
            mom_lb=args.mom_lb,
            z_win=args.z_win,
            trend_fast=args.trend_fast,
            trend_slow=args.trend_slow,
        )
    else:
        positions = build_positions_from_sentiment(
            sentiments=sentiments,
            threshold=args.threshold,
            long_short=args.long_short,
            allocation=args.allocation,
        )

    # Conservative risk controls (principal protection): go to cash in risk-off regime.
    risk_mask = _compute_risk_mask(index=positions.index, symbols=symbols, closes=closes, args=args)
    positions = positions.mul(risk_mask, axis=0)

    res = run_open_to_open_backtest(
        opens=opens,
        positions=positions,
        cost_bps=args.cost_bps,
        max_drawdown_stop=max_dd_stop,
    )

    print("===== Backtest Summary =====")
    print({k: round(v, 6) if isinstance(v, float) else v for k, v in res.metrics.items()})

    if args.out_perf_csv:
        out_df = res.daily.copy()
        out_df.index.name = "date"
        out_df.to_csv(args.out_perf_csv)
        print(f"saved daily perf -> {args.out_perf_csv}")

    if args.optimize:
        best = None
        best_any = None
        for thr in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            for ls in [False, True]:
                if args.signal_mode == "enhanced":
                    for mom_lb in [40, 60, 90]:
                        for z_win in [10, 20, 40]:
                            pos2 = build_positions_enhanced(
                                sentiments=sentiments,
                                closes=closes,
                                threshold=thr,
                                long_short=ls,
                                gate_mode=args.enhanced_gate_mode,
                                target_vol=args.target_vol,
                                mom_lb=mom_lb,
                                z_win=z_win,
                                trend_fast=args.trend_fast,
                                trend_slow=args.trend_slow,
                            )
                            pos2 = pos2.mul(risk_mask, axis=0)
                            r2 = run_open_to_open_backtest(
                                opens=opens,
                                positions=pos2,
                                cost_bps=args.cost_bps,
                                max_drawdown_stop=max_dd_stop,
                            )
                            m = r2.metrics
                            key_any = (m["cagr"], m["sharpe"])
                            if best_any is None or key_any > best_any[0]:
                                best_any = (key_any, thr, ls, f"enhanced:mom={mom_lb},z={z_win}", m)
                            if (args.mdd_limit is None) or (abs(float(m["mdd"])) <= float(args.mdd_limit)):
                                if best is None or key_any > best[0]:
                                    best = (key_any, thr, ls, f"enhanced:mom={mom_lb},z={z_win}", m)
                else:
                    for alloc in ["independent", "winner"]:
                        pos2 = build_positions_from_sentiment(
                            sentiments=sentiments, threshold=thr, long_short=ls, allocation=alloc
                        )
                        pos2 = pos2.mul(risk_mask, axis=0)
                        r2 = run_open_to_open_backtest(
                            opens=opens,
                            positions=pos2,
                            cost_bps=args.cost_bps,
                            max_drawdown_stop=max_dd_stop,
                        )
                        m = r2.metrics
                        key_any = (m["cagr"], m["sharpe"])
                        if best_any is None or key_any > best_any[0]:
                            best_any = (key_any, thr, ls, alloc, m)
                        if (args.mdd_limit is None) or (abs(float(m["mdd"])) <= float(args.mdd_limit)):
                            if best is None or key_any > best[0]:
                                best = (key_any, thr, ls, alloc, m)
        if best:
            _, thr, ls, alloc, m = best
            print("===== Best (grid) =====")
            print(
                {
                    "threshold": thr,
                    "long_short": ls,
                    "allocation": alloc,
                    **{k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()},
                }
            )
        elif best_any and (args.mdd_limit is not None):
            _, thr, ls, alloc, m = best_any
            print("===== Best (grid, ignoring mdd-limit) =====")
            print(
                {
                    "threshold": thr,
                    "long_short": ls,
                    "allocation": alloc,
                    "note": f"No config met mdd-limit={float(args.mdd_limit):.4f}",
                    **{k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()},
                }
            )

    if args.validate:
        _holdout_validate(
            sentiments=sentiments,
            closes=closes,
            opens=opens,
            symbols=symbols,
            args=args,
            max_dd_stop=max_dd_stop,
        )
        _walk_forward_validate(
            sentiments=sentiments,
            closes=closes,
            opens=opens,
            symbols=symbols,
            args=args,
            max_dd_stop=max_dd_stop,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
