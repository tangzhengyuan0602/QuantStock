from __future__ import annotations

import datetime as dt
from pathlib import Path


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def day_cache_path(cache_dir: Path, namespace: str, symbol: str, day: dt.date, ext: str) -> Path:
    # e.g. data_cache/gdelt/SPY/2024-01-01.jsonl
    return cache_dir / namespace / symbol / f"{day.isoformat()}.{ext.lstrip('.')}"

