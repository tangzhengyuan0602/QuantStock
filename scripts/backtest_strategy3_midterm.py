from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

from backtest_engine import compute_metrics, run_open_to_open_backtest
from feature_engine import momentum, trend_filter
from price_data import fetch_daily_closes, fetch_daily_opens
from run_backtest import _date_range_inclusive, _merge_news_over_days
from sentiment_pipeline import load_or_score_batch
from signal_utils import texts_from_news

WEIGHT_EPS = 1e-8


def _sanitize_weight(value: float) -> float:
    v = float(value)
    return 0.0 if abs(v) < WEIGHT_EPS else v


def _build_sentiments(
    *,
    cache_dir: Path,
    news_cache_dir: Path,
    symbols: list[str],
    trade_days: list[dt.date],
    scorer_name: str,
    max_articles: int,
    offline_cache_only: bool,
) -> pd.DataFrame:
    scorer = None if offline_cache_only else build_scorer(scorer_name)
    sentiments = pd.DataFrame(index=trade_days[1:], columns=symbols, dtype=float)

    for i in range(1, len(trade_days)):
        trade_day = trade_days[i]
        signal_day = trade_days[i - 1]
        news_days = _date_range_inclusive(signal_day, trade_day - dt.timedelta(days=1))
        per_sym_news = _merge_news_over_days(
            cache_dir=news_cache_dir,
            symbols=symbols,
            days=news_days,
            limit=max(50, max_articles),
            offline_cache_only=offline_cache_only,
        )
        packed_texts: list[str] = []
        index: list[tuple[str, int]] = []
        for sym in symbols:
            arts = (per_sym_news.get(sym) or [])[:max_articles]
            texts = texts_from_news(arts, max_items=max_articles)
            index.extend([(sym, j) for j in range(len(texts))])
            packed_texts.extend(texts)

        if packed_texts:
            span_tag = f":span={len(news_days)}" if len(news_days) != 1 else ""
            try:
                packed_scores = load_or_score_batch(
                    cache_dir=cache_dir,
                    namespace="alpaca",
                    symbol="BATCH",
                    day=signal_day,
                    batch_tag=f"{scorer_name}:{','.join(symbols)}:{max_articles}{span_tag}",
                    texts=packed_texts,
                    scorer=scorer,
                    cache_only=offline_cache_only,
                )
            except FileNotFoundError:
                packed_scores = [0.0] * len(packed_texts)
        else:
            packed_scores = []
        per_sym_scores = {s: [] for s in symbols}
        for (sym, _), sc in zip(index, packed_scores, strict=False):
            per_sym_scores[sym].append(float(sc))
        scores = {
            sym: (float(sum(per_sym_scores[sym])) / len(per_sym_scores[sym]) if per_sym_scores[sym] else 0.0)
            for sym in symbols
        }
        sentiments.loc[trade_day, symbols] = [float(scores[s]) for s in symbols]
    return sentiments.fillna(0.0)


def build_strategy3_positions(
    *,
    sentiments: pd.DataFrame,
    closes: pd.DataFrame,
    symbols: list[str],
    threshold_entry: float,
    threshold_exit: float,
    target_vol: float,
    mom_lb: int,
    z_win: int,
    news_ema_span: int,
    news_long_win: int,
    min_hold_days: int,
    rebalance_band: float,
    max_weight_step: float,
    trend_fast: int,
    trend_slow: int,
    riskoff_trend: bool,
) -> pd.DataFrame:
    positions = pd.DataFrame(0.0, index=sentiments.index, columns=symbols, dtype=float)
    prev_weights = {sym: 0.0 for sym in symbols}
    hold_days = {sym: 0 for sym in symbols}

    for trade_day in sentiments.index:
        cand: dict[str, float] = {}
        total_strength = 0.0
        bench = "SPY" if "SPY" in symbols else symbols[0]
        risk_mask = 1.0
        if riskoff_trend:
            tf = trend_filter(closes[bench].astype(float), fast=trend_fast, slow=trend_slow)
            tf_value = float(tf.reindex([trade_day], method="ffill").fillna(0.0).iloc[0])
            risk_mask = 1.0 if tf_value > 0.0 else 0.0

        for sym in symbols:
            hist = sentiments[sym].astype(float)
            hist = hist[hist.index <= trade_day]
            ema = hist.ewm(span=max(2, int(news_ema_span)), adjust=False).mean()
            long_mean = hist.rolling(window=max(3, int(news_long_win)), min_periods=3).mean()
            z = (
                (hist - hist.rolling(window=max(5, int(z_win)), min_periods=5).mean())
                / hist.rolling(window=max(5, int(z_win)), min_periods=5).std(ddof=0).replace(0.0, pd.NA)
            ).fillna(0.0)
            close_full = closes[sym].astype(float)
            mom = momentum(close_full, lookback=int(mom_lb)).reindex([trade_day], method="ffill").fillna(0.0)
            returns = close_full.pct_change()
            vol = (
                returns.rolling(window=20, min_periods=5).std(ddof=0).mul(252.0**0.5).reindex([trade_day], method="ffill")
            ).fillna(0.0)
            vol_value = float(vol.iloc[0]) if len(vol) else 0.0
            vol_scale = min(1.25, max(0.0, float(target_vol) / vol_value)) if vol_value > 0.0 else 0.0

            score_now = float(hist.iloc[-1]) if len(hist) else 0.0
            ema_now = float(ema.iloc[-1]) if len(ema) else 0.0
            long_now = float(long_mean.iloc[-1]) if len(long_mean) else 0.0
            z_now = float(z.iloc[-1]) if len(z) else 0.0
            mom_now = float(mom.iloc[0]) if len(mom) else 0.0
            composite = (0.35 * score_now) + (0.30 * ema_now) + (0.20 * long_now) + (0.15 * mom_now)

            prev_w = float(prev_weights.get(sym, 0.0))
            prev_hold = int(hold_days.get(sym, 0))
            enter_ok = composite >= float(threshold_entry)
            stay_ok = composite >= float(threshold_exit)
            if prev_w > 0.0 and prev_hold < int(min_hold_days) and risk_mask > 0.0:
                gate = 1.0
            elif prev_w > 0.0:
                gate = 1.0 if stay_ok else 0.0
            else:
                gate = 1.0 if enter_ok else 0.0
            strength = max(0.0, math.tanh(composite + 0.20 * z_now))
            strength = strength * max(0.0, vol_scale) * gate * risk_mask
            cand[sym] = float(max(0.0, strength))
            total_strength += cand[sym]

        if total_strength > 0.0:
            cand = {sym: float(cand[sym]) / total_strength for sym in symbols}
        else:
            cand = {sym: 0.0 for sym in symbols}

        new_weights: dict[str, float] = {}
        for sym in symbols:
            prev_w = float(prev_weights.get(sym, 0.0))
            target_w = float(cand.get(sym, 0.0))
            if abs(target_w - prev_w) < float(rebalance_band):
                w = prev_w
            else:
                step = max(-float(max_weight_step), min(float(max_weight_step), target_w - prev_w))
                w = max(0.0, prev_w + step)
            if prev_w > 0.0 and w <= 0.0 and int(hold_days.get(sym, 0)) < int(min_hold_days):
                w = prev_w
            new_weights[sym] = _sanitize_weight(w)

        total_w = float(sum(new_weights.values()))
        if total_w > 1.0 and total_w > 0.0:
            new_weights = {sym: _sanitize_weight(float(w) / total_w) for sym, w in new_weights.items()}

        for sym in symbols:
            positions.at[trade_day, sym] = _sanitize_weight(float(new_weights[sym]))
            if float(new_weights[sym]) > 0.0:
                hold_days[sym] = int(hold_days.get(sym, 0)) + 1
            else:
                hold_days[sym] = 0
        prev_weights = dict(new_weights)

    return positions


def _segment_metrics(daily: pd.DataFrame, start: dt.date, end: dt.date) -> dict[str, float | int | str]:
    seg = daily[(daily.index >= start) & (daily.index <= end)]
    metrics = compute_metrics(seg["ret"]) if len(seg) else {"days": 0, "cagr": 0.0, "sharpe": 0.0, "mdd": 0.0}
    metrics["start"] = str(start)
    metrics["end"] = str(end)
    return metrics


def _trade_stats(trades: pd.DataFrame, symbols: list[str]) -> dict[str, float]:
    active = trades[[f"w_{s}" for s in symbols]].sum(axis=1)
    invested_days = int((active > 0).sum())
    avg_turnover = float(trades["turnover"].mean()) if len(trades) else 0.0
    median_turnover = float(trades["turnover"].median()) if len(trades) else 0.0

    hold_lengths: list[int] = []
    for sym in symbols:
        ser = (trades[f"w_{sym}"] > 0).astype(int)
        run = 0
        for v in ser:
            if v:
                run += 1
            elif run:
                hold_lengths.append(run)
                run = 0
        if run:
            hold_lengths.append(run)
    return {
        "invested_days": invested_days,
        "avg_turnover": avg_turnover,
        "median_turnover": median_turnover,
        "avg_hold_days": (float(sum(hold_lengths)) / len(hold_lengths)) if hold_lengths else 0.0,
        "median_hold_days": float(pd.Series(hold_lengths).median()) if hold_lengths else 0.0,
        "hold_segments": int(len(hold_lengths)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ")
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default=str(dt.date.today()))
    ap.add_argument("--cache-dir", default="data_cache_strategy3")
    ap.add_argument("--news-cache-dir", default="data_cache")
    ap.add_argument("--scorer", default="ark_struct")
    ap.add_argument("--max-articles", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--threshold-entry", type=float, default=0.15)
    ap.add_argument("--threshold-exit", type=float, default=0.05)
    ap.add_argument("--target-vol", type=float, default=0.08)
    ap.add_argument("--mom-lb", type=int, default=60)
    ap.add_argument("--z-win", type=int, default=60)
    ap.add_argument("--news-ema-span", type=int, default=5)
    ap.add_argument("--news-long-win", type=int, default=20)
    ap.add_argument("--min-hold-days", type=int, default=5)
    ap.add_argument("--rebalance-band", type=float, default=0.08)
    ap.add_argument("--max-weight-step", type=float, default=0.20)
    ap.add_argument("--trend-fast", type=int, default=50)
    ap.add_argument("--trend-slow", type=int, default=200)
    ap.add_argument("--max-dd-stop", type=float, default=0.18)
    ap.add_argument("--out-prefix", default="")
    ap.add_argument("--offline-cache-only", action="store_true")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    cache_dir = Path(args.cache_dir)
    news_cache_dir = Path(args.news_cache_dir)

    opens = fetch_daily_opens(symbols, start=start, end=end)
    if len(opens.index) < 20:
        raise RuntimeError("not enough open price history")
    feat_start = start - dt.timedelta(days=800)
    closes = fetch_daily_closes(symbols, start=feat_start, end=end)
    trading_days = list(opens.index)

    sentiments = _build_sentiments(
        cache_dir=cache_dir,
        news_cache_dir=news_cache_dir,
        symbols=symbols,
        trade_days=trading_days,
        scorer_name=args.scorer,
        max_articles=int(args.max_articles),
        offline_cache_only=bool(args.offline_cache_only),
    )
    positions = build_strategy3_positions(
        sentiments=sentiments,
        closes=closes,
        symbols=symbols,
        threshold_entry=float(args.threshold_entry),
        threshold_exit=float(args.threshold_exit),
        target_vol=float(args.target_vol),
        mom_lb=int(args.mom_lb),
        z_win=int(args.z_win),
        news_ema_span=int(args.news_ema_span),
        news_long_win=int(args.news_long_win),
        min_hold_days=int(args.min_hold_days),
        rebalance_band=float(args.rebalance_band),
        max_weight_step=float(args.max_weight_step),
        trend_fast=int(args.trend_fast),
        trend_slow=int(args.trend_slow),
        riskoff_trend=True,
    )

    res = run_open_to_open_backtest(
        opens=opens,
        positions=positions,
        weights={s: 1.0 for s in symbols},
        cost_bps=float(args.cost_bps),
        max_drawdown_stop=-abs(float(args.max_dd_stop)),
    )

    trade_stats = _trade_stats(res.trades, symbols)
    segments = {
        "full": _segment_metrics(res.daily, res.daily.index.min(), res.daily.index.max()),
        "short": _segment_metrics(res.daily, dt.date(2025, 1, 1), res.daily.index.max()),
        "medium": _segment_metrics(res.daily, dt.date(2022, 1, 1), dt.date(2024, 12, 31)),
        "long": _segment_metrics(res.daily, res.daily.index.min(), dt.date(2021, 12, 31)),
    }

    out = {
        "params": {
            "symbols": symbols,
            "start": str(start),
            "end": str(end),
            "cache_dir": str(cache_dir),
            "news_cache_dir": str(news_cache_dir),
            "scorer": args.scorer,
            "max_articles": int(args.max_articles),
            "threshold_entry": float(args.threshold_entry),
            "threshold_exit": float(args.threshold_exit),
            "target_vol": float(args.target_vol),
            "mom_lb": int(args.mom_lb),
            "z_win": int(args.z_win),
            "news_ema_span": int(args.news_ema_span),
            "news_long_win": int(args.news_long_win),
            "min_hold_days": int(args.min_hold_days),
            "rebalance_band": float(args.rebalance_band),
            "max_weight_step": float(args.max_weight_step),
            "trend_fast": int(args.trend_fast),
            "trend_slow": int(args.trend_slow),
            "max_dd_stop": float(args.max_dd_stop),
        },
        "summary": {k: round(v, 6) if isinstance(v, float) else v for k, v in res.metrics.items()},
        "trade_stats": {k: round(v, 6) if isinstance(v, float) else v for k, v in trade_stats.items()},
        "segments": {
            name: {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}
            for name, metrics in segments.items()
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))

    if args.out_prefix:
        prefix = Path(args.out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        res.daily.to_csv(str(prefix) + "_daily.csv")
        res.trades.to_csv(str(prefix) + "_trades.csv")
        Path(str(prefix) + "_summary.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
