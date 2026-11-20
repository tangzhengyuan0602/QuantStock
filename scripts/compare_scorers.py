"""F-3' three-scorer agreement report.

For ~30 trading days from the news cache, score the same articles with
lexicon / finbert / ark_struct (DeepSeek-V3.1 via Ark) and emit:
  - per-day mean sentiment per scorer
  - pairwise Pearson correlation
  - sign agreement % (do they at least agree on +/-/0?)
  - distribution of |s_finbert - s_lexicon|, |s_ark - s_finbert| etc.

Usage:
    set -a; source .env; set +a
    python3 scripts/compare_scorers.py --days 30 --per-day 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentiment_models import build_scorer  # noqa: E402

NEWS_DIR_CANDIDATES = [
    ROOT / "data_cache" / "alpaca_news_all" / "ALL",
]


def resolve_news_dir(explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = ROOT / explicit
        return path
    dated_dirs: list[tuple[str, int, Path]] = []
    for idx, candidate in enumerate(NEWS_DIR_CANDIDATES):
        if not candidate.exists():
            continue
        files = sorted(candidate.glob("*.json"))
        if not files:
            continue
        # Prefer the freshest date first, and break same-day ties by the
        # candidate priority order above rather than falling back to the legacy
        # base cache.
        dated_dirs.append((files[-1].stem, -idx, candidate))
    if not dated_dirs:
        return NEWS_DIR_CANDIDATES[-1]
    dated_dirs.sort(key=lambda item: (item[0], item[1]))
    return dated_dirs[-1][2]


def collect_articles(
    days: int,
    per_day: int,
    news_dir: Path,
    *,
    seed: int = 7,
    latest_only: bool = False,
    end_date: str = "",
) -> pd.DataFrame:
    files = sorted(news_dir.glob("*.json"))
    if end_date:
        try:
            end_dt = dt.date.fromisoformat(end_date)
        except Exception:
            end_dt = None
        if end_dt is not None:
            filtered: list[Path] = []
            for f in files:
                try:
                    file_dt = dt.date.fromisoformat(f.stem)
                except Exception:
                    continue
                if file_dt <= end_dt:
                    filtered.append(f)
            files = filtered
    rng = random.Random(seed)
    if latest_only:
        sampled = files[-min(days, len(files)) :]
    else:
        sampled = rng.sample(files, k=min(days, len(files)))
        sampled.sort()
    rows = []
    for f in sampled:
        try:
            arr = json.loads(f.read_text())
        except Exception:
            continue
        if not arr:
            continue
        pick = rng.sample(arr, k=min(per_day, len(arr)))
        for art in pick:
            t = (art.get("title") or "").strip()
            s = (art.get("summary") or "").strip()
            text = (t + ". " + s)[:1024] if t else s[:1024]
            if not text:
                continue
            rows.append({"date": f.stem, "title": t[:120], "text": text})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--per-day", type=int, default=5)
    ap.add_argument("--news-dir", default="", help="Directory of cached Alpaca daily news JSON files")
    ap.add_argument("--out", default="finrl_outputs/scorer_comparison.json")
    ap.add_argument("--latest-only", action="store_true", help="Use the most recent consecutive cached trade days instead of random sampling.")
    ap.add_argument("--end-date", default="", help="Only sample cache files on or before YYYY-MM-DD.")
    args = ap.parse_args()

    news_dir = resolve_news_dir(str(args.news_dir or "").strip())
    df = collect_articles(
        args.days,
        args.per_day,
        news_dir,
        latest_only=bool(args.latest_only),
        end_date=str(args.end_date or "").strip(),
    )
    print(f"news_dir={news_dir}")
    print(f"sampled {len(df)} articles across {df['date'].nunique()} days")

    scorers: dict[str, object] = {}
    print("[lexicon]")
    scorers["lexicon"] = build_scorer("lexicon")
    print("[finbert] loading (first call may download ~440MB)...")
    try:
        # FinBERT can use HF mirror to avoid timeout
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        scorers["finbert"] = build_scorer("finbert")
    except Exception as e:
        print(f"  finbert unavailable: {e}")

    if os.getenv("ARK_API_KEY"):
        print("[ark_struct] (DeepSeek-V3.1 via Ark)")
        try:
            scorers["ark_struct"] = build_scorer("ark_struct")
        except Exception as e:
            print(f"  ark_struct unavailable: {e}")
    else:
        print("[ark_struct] skipped: ARK_API_KEY not set")

    scores: dict[str, list[float]] = {k: [] for k in scorers}
    texts = df["text"].tolist()
    for name, sc in scorers.items():
        print(f"scoring with {name} ({len(texts)} articles)...")
        try:
            res = sc.score(texts)
        except Exception as e:
            print(f"  {name} failed: {e}")
            res = [float("nan")] * len(texts)
        scores[name] = list(res)

    out_df = df.copy()
    for k, v in scores.items():
        out_df[f"s_{k}"] = v
    out_df.to_csv(ROOT / "finrl_outputs" / "scorer_comparison_raw.csv", index=False)

    cols = [f"s_{k}" for k in scorers]
    arr = out_df[cols].dropna()
    print(f"\nValid rows where all scorers agreed to score: {len(arr)} / {len(out_df)}")
    if len(arr) >= 5:
        corr = arr.corr().round(3)
        print("\nPearson correlation:")
        print(corr.to_string())

        # Sign agreement
        signs = arr.apply(np.sign)
        agree = {}
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                pct = (signs[a] == signs[b]).mean() * 100
                agree[f"{a[2:]}_vs_{b[2:]}"] = round(float(pct), 1)
        print(f"\nSign agreement (% rows with same sign):")
        for k, v in agree.items():
            print(f"  {k}: {v}%")

        # Daily means
        daily = out_df.groupby("date")[cols].mean()
        daily_corr = daily.corr().round(3)
        print(f"\nDaily-mean correlation (n={len(daily)} days):")
        print(daily_corr.to_string())

        report = {
            "n_articles_scored": int(len(arr)),
            "n_articles_total": int(len(out_df)),
            "news_dir": str(news_dir),
            "sample_mode": "latest_only" if bool(args.latest_only) else "random",
            "sample_end_date": str(args.end_date or "").strip(),
            "pearson_corr": corr.to_dict(),
            "sign_agreement_pct": agree,
            "daily_corr": daily_corr.to_dict(),
            "scorers": list(scorers.keys()),
        }
    else:
        report = {
            "error": "too few valid rows",
            "valid": int(len(arr)),
            "news_dir": str(news_dir),
            "sample_mode": "latest_only" if bool(args.latest_only) else "random",
            "sample_end_date": str(args.end_date or "").strip(),
        }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
