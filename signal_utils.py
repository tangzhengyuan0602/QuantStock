from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class ScoredItem:
    text: str
    score: float  # [-1, 1]


def clip_score(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(-1.0, min(1.0, float(x)))


def aggregate_scores(scores: Sequence[float], method: str = "mean") -> float:
    if not scores:
        return 0.0
    xs = [clip_score(s) for s in scores]
    if method == "mean":
        return sum(xs) / len(xs)
    if method == "median":
        xs2 = sorted(xs)
        mid = len(xs2) // 2
        return xs2[mid] if len(xs2) % 2 == 1 else 0.5 * (xs2[mid - 1] + xs2[mid])
    if method == "trimmed_mean_10":
        xs2 = sorted(xs)
        k = max(0, int(0.1 * len(xs2)))
        xs2 = xs2[k : len(xs2) - k] if len(xs2) - 2 * k > 0 else xs2
        return sum(xs2) / len(xs2)
    raise ValueError(f"Unknown aggregate method: {method}")


def score_to_position(score: float, threshold: float = 0.15, long_short: bool = True) -> int:
    """Convert daily sentiment score to position for next open.

    Returns:
      -1 short, 0 flat, +1 long
    """
    s = clip_score(score)
    if s >= threshold:
        return 1
    if long_short and s <= -threshold:
        return -1
    return 0


def texts_from_news(articles: Iterable[dict], max_items: int = 50) -> list[str]:
    out: list[str] = []
    for a in articles:
        if len(out) >= max_items:
            break
        title = (a.get("title") or "").strip()
        summary = (a.get("summary") or "").strip()
        if title and summary:
            out.append(f"{title}. {summary}")
        elif title:
            out.append(title)
    return out


def map_scores(texts: Sequence[str], scorer: Callable[[Sequence[str]], Sequence[float]]) -> list[float]:
    # scorer is expected to be batched
    res = scorer(texts)
    if len(res) != len(texts):
        raise ValueError(f"scorer returned {len(res)} scores for {len(texts)} texts")
    return [clip_score(float(x)) for x in res]
