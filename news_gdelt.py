from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import socket
import ssl
import time
import urllib.parse
from typing import Any, Iterable

import requests


@dataclasses.dataclass(frozen=True)
class GDELTArticle:
    title: str
    url: str
    seendate_utc: dt.datetime
    domain: str | None = None
    source_country: str | None = None
    language: str | None = None
    extras: dict[str, Any] | None = None


class GDELTRuntimeError(RuntimeError):
    pass


class GDELTDocClient:
    """Minimal client for GDELT 2.1 DOC API (free, historical).

    Docs: https://blog.gdeltproject.org/gdelt-2-1-api-debuts/
    Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
    """

    def __init__(
        self,
        session: requests.Session | None = None,
        base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc",
        timeout_s: float = 5.0,
        max_retries: int = 2,
        backoff_s: float = 0.5,
    ) -> None:
        self._session = session or requests.Session()
        self._base_url = base_url
        self._timeout_s = float(os.getenv("GDELT_TIMEOUT_S", timeout_s))
        self._max_retries = int(os.getenv("GDELT_MAX_RETRIES", max_retries))
        self._backoff_s = float(os.getenv("GDELT_BACKOFF_S", backoff_s))

    @staticmethod
    def _fmt_gdelt_dt(d: dt.datetime) -> str:
        d = d.astimezone(dt.timezone.utc)
        return d.strftime("%Y%m%d%H%M%S")

    def _transport_diagnostics(self) -> str:
        parsed = urllib.parse.urlparse(self._base_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return "diag=invalid_base_url"

        parts: list[str] = [f"host={host}", f"port={port}"]

        dns_t0 = time.perf_counter()
        try:
            addrinfo = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            dns_ms = (time.perf_counter() - dns_t0) * 1000.0
            addrs = sorted({item[4][0] for item in addrinfo if item[4]})
            parts.append(f"dns_ok_ms={dns_ms:.1f}")
            parts.append(f"dns_count={len(addrs)}")
            if addrs:
                parts.append(f"dns_first={addrs[0]}")
        except Exception as exc:  # noqa: BLE001
            dns_ms = (time.perf_counter() - dns_t0) * 1000.0
            parts.append(f"dns_fail_ms={dns_ms:.1f}")
            parts.append(f"dns_err={type(exc).__name__}:{exc}")
            return "diag=" + " ".join(parts)

        tcp_t0 = time.perf_counter()
        sock: socket.socket | None = None
        try:
            sock = socket.create_connection((host, port), timeout=self._timeout_s)
            tcp_ms = (time.perf_counter() - tcp_t0) * 1000.0
            parts.append(f"tcp_ok_ms={tcp_ms:.1f}")
        except Exception as exc:  # noqa: BLE001
            tcp_ms = (time.perf_counter() - tcp_t0) * 1000.0
            parts.append(f"tcp_fail_ms={tcp_ms:.1f}")
            parts.append(f"tcp_err={type(exc).__name__}:{exc}")
            return "diag=" + " ".join(parts)

        try:
            tls_t0 = time.perf_counter()
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(sock, server_hostname=host) as tls_sock:
                tls_ms = (time.perf_counter() - tls_t0) * 1000.0
                parts.append(f"tls_ok_ms={tls_ms:.1f}")
                parts.append(f"tls_ver={tls_sock.version() or 'unknown'}")
                cert = tls_sock.getpeercert() or {}
                subject = cert.get("subject") or ()
                if subject and subject[0] and len(subject[0][0]) >= 2:
                    parts.append(f"tls_cn={subject[0][0][1]}")
        except Exception as exc:  # noqa: BLE001
            tls_ms = (time.perf_counter() - tls_t0) * 1000.0
            parts.append(f"tls_fail_ms={tls_ms:.1f}")
            parts.append(f"tls_err={type(exc).__name__}:{exc}")
        finally:
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass
        return "diag=" + " ".join(parts)

    def _get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        last_err: Exception | None = None
        last_diag: str | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._session.get(self._base_url, params=params, timeout=self._timeout_s)
                if resp.status_code >= 400:
                    raise GDELTRuntimeError(f"GDELT HTTP {resp.status_code}: {resp.text[:500]}")
                return resp.json()
            except Exception as e:  # noqa: BLE001
                last_err = e
                try:
                    last_diag = self._transport_diagnostics()
                except Exception as diag_exc:  # noqa: BLE001
                    last_diag = f"diag_failed={type(diag_exc).__name__}:{diag_exc}"
                err_text = str(e)
                # 429s and malformed empty responses are common on this free
                # endpoint; fail fast so callers can degrade to Alpaca-only.
                if "GDELT HTTP 429" in err_text or "Expecting value:" in err_text:
                    break
                time.sleep(self._backoff_s * attempt)
        detail = f"GDELT request failed after retries: {last_err}"
        if last_diag:
            detail = f"{detail} [{last_diag}]"
        raise GDELTRuntimeError(detail)

    def fetch_articles(
        self,
        query: str,
        start_utc: dt.datetime,
        end_utc: dt.datetime,
        max_records: int = 50,
        sort: str = "HybridRel",
        format_: str = "json",
    ) -> list[GDELTArticle]:
        """Fetch up to `max_records` most relevant articles in [start_utc, end_utc)."""
        if start_utc.tzinfo is None or end_utc.tzinfo is None:
            raise ValueError("start_utc/end_utc must be timezone-aware (UTC)")

        params = {
            "query": query,
            "mode": "ArtList",
            "format": format_,
            "startdatetime": self._fmt_gdelt_dt(start_utc),
            "enddatetime": self._fmt_gdelt_dt(end_utc),
            "maxrecords": int(max_records),
            "sort": sort,
        }
        data = self._get_json(params)
        arts = data.get("articles") or []
        out: list[GDELTArticle] = []
        for a in arts:
            try:
                # Example: 20240101123000
                seendate = a.get("seendate")
                seendate_utc = dt.datetime.strptime(seendate, "%Y%m%d%H%M%S").replace(
                    tzinfo=dt.timezone.utc
                )
                out.append(
                    GDELTArticle(
                        title=(a.get("title") or "").strip(),
                        url=(a.get("url") or "").strip(),
                        seendate_utc=seendate_utc,
                        domain=a.get("domain"),
                        source_country=a.get("sourceCountry"),
                        language=a.get("language"),
                        extras=a,
                    )
                )
            except Exception:
                continue
        return out

    def iter_daily_articles(
        self,
        query: str,
        start_date: dt.date,
        end_date: dt.date,
        max_records_per_day: int = 50,
        tz: dt.tzinfo = dt.timezone.utc,
    ) -> Iterable[tuple[dt.date, list[GDELTArticle]]]:
        """Iterate by UTC day: [start_date, end_date] inclusive."""
        if start_date > end_date:
            return
        cur = start_date
        while cur <= end_date:
            day_start = dt.datetime(cur.year, cur.month, cur.day, 0, 0, 0, tzinfo=tz).astimezone(
                dt.timezone.utc
            )
            day_end = (day_start + dt.timedelta(days=1)).astimezone(dt.timezone.utc)
            arts = self.fetch_articles(
                query=query,
                start_utc=day_start,
                end_utc=day_end,
                max_records=max_records_per_day,
            )
            yield cur, arts
            cur = cur + dt.timedelta(days=1)


def default_queries() -> dict[str, str]:
    """Reasonable default queries for SPY/QQQ sentiment (English, US market).

    GDELT supports operators like OR, quoting, etc.
    Keep queries concise to control volume and avoid rate limits.
    """
    spy = '((SPY OR "S&P 500" OR "US stock market" OR "US equities" OR "Wall Street") AND (rally OR selloff OR earnings OR inflation OR fed OR recession OR outlook OR guidance))'
    qqq = '((QQQ OR Nasdaq OR "Nasdaq 100" OR "tech stocks" OR "US technology") AND (rally OR selloff OR earnings OR inflation OR fed OR recession OR outlook OR guidance))'
    # Filter to English to improve sentiment model quality
    spy += " language:english"
    qqq += " language:english"
    return {"SPY": spy, "QQQ": qqq}


def articles_to_jsonl(articles: Iterable[GDELTArticle]) -> str:
    lines = []
    for a in articles:
        lines.append(
            json.dumps(
                {
                    "title": a.title,
                    "url": a.url,
                    "seendate_utc": a.seendate_utc.isoformat(),
                    "domain": a.domain,
                    "source_country": a.source_country,
                    "language": a.language,
                    "extras": a.extras,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def articles_from_jsonl(s: str) -> list[GDELTArticle]:
    out: list[GDELTArticle] = []
    for line in s.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out.append(
            GDELTArticle(
                title=obj["title"],
                url=obj["url"],
                seendate_utc=dt.datetime.fromisoformat(obj["seendate_utc"]).astimezone(dt.timezone.utc),
                domain=obj.get("domain"),
                source_country=obj.get("source_country"),
                language=obj.get("language"),
                extras=obj.get("extras"),
            )
        )
    return out

