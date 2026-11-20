# QuantStock

A research and backtesting toolkit that combines **multi-source news, NLP sentiment scoring,
technical features, and a daily portfolio backtest** for US equities / ETFs.

> **Research project only.** This repository contains backtesting, feature engineering, and
> sentiment-pipeline code. It does **not** place live orders, and it ships no production trading
> or monitoring scripts. Nothing here is investment advice.

## What it does

1. **Fetch market prices** through a pluggable source chain (Yahoo Finance, Alpaca, Stooq) with
   per-source retry and fallback.
2. **Collect daily news** from Alpaca News, Finnhub, Polygon, and GDELT, deduplicate articles,
   and cache them on disk as JSON so repeated runs stay offline and fast.
3. **Score article sentiment** with interchangeable scorers: a local FinBERT model, OpenAI-compatible
   chat endpoints, Volcengine Ark, Zhipu GLM, a rule-based lexicon, an ensemble, or a self-hosted
   FinGPT endpoint.
4. **Engineer features** (rolling z-score, momentum, trend filter, realized volatility) and map a
   daily sentiment score into a portfolio position.
5. **Backtest** an open-to-open portfolio, report CAGR / Sharpe / max drawdown / total return, and
   evaluate on holdout and walk-forward out-of-sample windows.

## Pipeline overview

```
prices (yfinance / alpaca / stooq)
        │
news (alpaca / finnhub / polygon / gdelt) ──► daily JSON cache
        │
        ▼
sentiment scorer (finbert / ark / glm / openai / lexicon / ensemble / fingpt)
        │
        ▼
features (momentum, trend, vol, z-score) ──► position signal ──► backtest engine
```

## Module map

| File | Responsibility |
| --- | --- |
| `price_data.py` | Daily open/close prices with multi-source fallback and retry. |
| `news_alpaca.py` / `news_finnhub.py` / `news_polygon.py` / `news_gdelt.py` | Per-source news fetchers. |
| `news_pipeline.py` | Daily news aggregation, deduplication, and per-day JSON cache. |
| `sentiment_models.py` | Interchangeable sentiment scorers and `build_scorer(kind)` factory. |
| `sentiment_pipeline.py` | Score a cached day of articles and persist the results. |
| `feature_engine.py` | Technical indicators: z-score, momentum, trend filter, realized vol. |
| `signal_utils.py` | Score clipping, aggregation, news text extraction, score-to-position mapping. |
| `backtest_engine.py` | Open-to-open portfolio simulation and performance metrics. |
| `run_backtest.py` | Main CLI entry point: load data, score news, build signals, backtest. |
| `cache_io.py` | On-disk day-level cache paths and directory helpers. |

## Install

Requires Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras used only by the research scripts in `scripts/`:

```bash
# FinRL PPO script
pip install "gymnasium>=0.29" "stable-baselines3[extra]>=2.3"

# Qlib Alpha158 baseline script
pip install pyqlib lightgbm
```

## Configuration

Copy the example environment file and fill in the keys you need:

```bash
cp .env.example .env
```

All credentials are read from environment variables; **no secrets are stored in the repo.**
You only have to configure the data sources and scorers you actually use. GDELT and Stooq are
keyless; Yahoo Finance is the free price fallback.

## Quick start

### Run a backtest

```bash
python3 run_backtest.py \
  --symbols SPY,QQQ \
  --years 5 \
  --scorer finbert \
  --threshold 0.15 \
  --out-perf-csv perf.csv
```

Key flags:

- `--symbols` — comma-separated tickers.
- `--scorer` — one of `finbert`, `openai`, `ark`, `ark_struct`, `glm`, `lexicon`, `ensemble`,
  `ensemble_agree`.
- `--strategy` — `strategy1` (sentiment-driven) or `strategy2` (momentum rotation).
- `--long-short` — allow a short position when the score is below the negative threshold.
- `--cost-bps` — per-trade transaction cost in basis points.
- `--out-perf-csv` — write the daily equity curve to a CSV.

The summary prints CAGR, Sharpe, max drawdown, total return, average daily return, and daily
volatility, followed by holdout and walk-forward out-of-sample blocks.

### Compare sentiment scorers

Score the same cached articles with lexicon / FinBERT / Ark and report pairwise correlation,
sign agreement, and score spread:

```bash
python3 scripts/compare_scorers.py --days 30 --per-day 5
```

### Mid-term sentiment + momentum strategy

```bash
python3 scripts/backtest_strategy3_midterm.py --symbols SPY,QQQ --scorer ark_struct
```

### Research scripts

- `scripts/finrl_ppo_etf.py` — PPO actor-critic ETF portfolio with a custom Gym environment
  (`--train --eval`).
- `scripts/qlib_alpha158_baseline.py` — Qlib Alpha158 + LightGBM ranking baseline for IC / Rank-IC
  comparison.

## Sentiment scorers

| `--scorer` | Backing |
| --- | --- |
| `finbert` | Local FinBERT (offline after the HF model is cached). |
| `openai` | Any OpenAI-compatible chat endpoint (`OPENAI_BASE_URL` / `OPENAI_MODEL`). |
| `ark`, `ark_struct` | Volcengine Ark chat / structured output. |
| `glm` | ZhipuAI GLM chat endpoint. |
| `lexicon` | Rule-based lexical sentiment, no external calls. |
| `ensemble`, `ensemble_agree` | Aggregate several scorers by mean or strict agreement. |
| `fingpt` | Self-hosted FinGPT behind an OpenAI-compatible vLLM/TGI server. |

## Project layout

```
.
├── run_backtest.py            # main CLI
├── backtest_engine.py         # portfolio simulation + metrics
├── feature_engine.py          # technical indicators
├── news_pipeline.py           # news aggregation + cache
├── sentiment_pipeline.py      # daily scoring + cache
├── sentiment_models.py        # scorer implementations + factory
├── signal_utils.py            # score -> position helpers
├── price_data.py              # price fetchers + fallback chain
├── news_*.py                  # per-source news fetchers
├── cache_io.py                # cache path helpers
├── scripts/                   # standalone research / comparison experiments
├── .env.example               # environment variable template
└── requirements.txt
```

## Scope and limitations

- Backtests are **open-to-open** and assume the configured cost model; slippage and liquidity are
  not modeled in detail.
- News coverage depends on the configured sources and their free-tier rate limits.
- Results on historical data are not indicative of future performance.
- This snapshot intentionally excludes live order routing, account shadowing, daily scheduling, and
  operational monitoring; those belong to a deployment context, not a public research repo.

## License

MIT
