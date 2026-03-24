#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except Exception:  # pragma: no cover
    trafilatura = None


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,uk;q=0.8,ru;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": "https://www.pravda.com.ua/",
}

MONTHS_EN = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


@dataclass(frozen=True)
class SiteConfig:
    latest_url: str
    date_url_tpl: str
    article_regex: re.Pattern[str]
    language: str


SITE_CONFIGS = {
    "en": SiteConfig(
        latest_url="https://www.pravda.com.ua/eng/news/",
        date_url_tpl="https://www.pravda.com.ua/eng/news/date_{ddmmyyyy}/",
        article_regex=re.compile(
            r"^https?://www\.pravda\.com\.ua/eng/news/\d{4}/\d{2}/\d{2}/\d+/?$"
        ),
        language="en",
    ),
    "ua": SiteConfig(
        latest_url="https://www.pravda.com.ua/news/",
        date_url_tpl="https://www.pravda.com.ua/news/date_{ddmmyyyy}/",
        article_regex=re.compile(
            r"^https?://www\.pravda\.com\.ua/news/\d{4}/\d{2}/\d{2}/\d+/?$"
        ),
        language="uk",
    ),
    "ru": SiteConfig(
        latest_url="https://www.pravda.com.ua/rus/news/",
        date_url_tpl="https://www.pravda.com.ua/rus/news/date_{ddmmyyyy}/",
        article_regex=re.compile(
            r"^https?://www\.pravda\.com\.ua/rus/news/\d{4}/\d{2}/\d{2}/\d+/?$"
        ),
        language="ru",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape Ukrainska Pravda news into CSV")
    p.add_argument("--lang", choices=sorted(SITE_CONFIGS), default="en")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--output", default="pravda_news.csv")
    p.add_argument("--date-from", help="Start date DD.MM.YYYY or YYYY-MM-DD")
    p.add_argument("--date-to", help="End date DD.MM.YYYY or YYYY-MM-DD (default: today)")
    p.add_argument("--max-days-back", type=int, default=30)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument(
        "--fetch-mode",
        choices=["auto", "requests", "playwright"],
        default="auto",
        help="Use requests, Playwright, or auto fallback when blocked.",
    )
    p.add_argument("--headed", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def parse_date_arg(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise SystemExit(f"Unsupported date format: {value}")


class Fetcher:
    def __init__(self, mode: str, timeout: int, headed: bool = False, verbose: bool = False) -> None:
        self.mode = mode
        self.timeout = timeout
        self.headed = headed
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _ensure_playwright(self) -> None:
        if self._page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Playwright is not installed. Run: pip install playwright && playwright install chromium"
            ) from e
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=not self.headed)
        self._context = self._browser.new_context(
            user_agent=USER_AGENT,
            locale="en-US",
            extra_http_headers={
                "Accept-Language": HEADERS["Accept-Language"],
                "Referer": HEADERS["Referer"],
            },
            viewport={"width": 1400, "height": 1000},
        )
        self._page = self._context.new_page()

    def get(self, url: str) -> str:
        last_err: Optional[Exception] = None
        modes = [self.mode] if self.mode != "auto" else ["requests", "playwright"]
        for mode in modes:
            try:
                if mode == "requests":
                    r = self.session.get(url, timeout=self.timeout)
                    r.raise_for_status()
                    return r.text
                if mode == "playwright":
                    self._ensure_playwright()
                    assert self._page is not None
                    resp = self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                    if resp is None:
                        raise RuntimeError("no browser response")
                    if resp.status >= 400:
                        raise RuntimeError(f"browser HTTP {resp.status}")
                    self._page.wait_for_timeout(700)
                    return self._page.content()
            except Exception as e:  # pragma: no cover
                last_err = e
                self.log(f"  -> {mode} failed for {url}: {e}")
                continue
        raise RuntimeError(str(last_err) if last_err else f"failed to fetch {url}")

    def close(self) -> None:
        try:
            if self._context is not None:
                self._context.close()
            if self._browser is not None:
                self._browser.close()
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            pass


def clean_title(title: str) -> str:
    title = re.sub(r"\s+\|\s+(Ukrainska|Українська) Pravda.*$", "", title, flags=re.I).strip()
    title = re.sub(r"\s+\|\s+УП.*$", "", title).strip()
    return title


def soup_text_lines(soup: BeautifulSoup) -> list[str]:
    return [ln.strip() for ln in soup.get_text("\n", strip=True).splitlines() if ln.strip()]


def extract_list_urls(html: str, cfg: SiteConfig) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(cfg.latest_url, a["href"])
        href = href.split("#", 1)[0]
        if cfg.article_regex.match(href) and href not in seen:
            seen.add(href)
            urls.append(href)
    return urls


def extract_previous_date_link(html: str, cfg: SiteConfig) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = urljoin(cfg.latest_url, a["href"])
        if re.match(r"^https?://www\.pravda\.com\.ua/(eng/|rus/)?news/date_\d{8}/?$", href):
            return href
    return None


def extract_iso_from_html(html: str) -> Optional[str]:
    patterns = [
        r'"datePublished"\s*:\s*"([^"]+)"',
        r'property=["\']article:published_time["\']\s+content=["\']([^"\']+)["\']',
        r'name=["\']article:published_time["\']\s+content=["\']([^"\']+)["\']',
    ]
    for pattern in patterns:
        m = re.search(pattern, html, flags=re.I)
        if m:
            return m.group(1)
    return None


def extract_published_at(html: str, url: str) -> str:
    iso = extract_iso_from_html(html)
    if iso:
        return iso

    url_date_match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/\d+/?$", url)
    base_date = None
    if url_date_match:
        y, m, d = map(int, url_date_match.groups())
        base_date = dt.date(y, m, d)

    text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
    m = re.search(r"—\s*(\d{1,2})\s+([A-Za-z]+),\s*(\d{1,2}:\d{2})", text)
    if m and base_date:
        day = int(m.group(1))
        month_name = m.group(2).lower()
        hour_min = m.group(3)
        month = MONTHS_EN.get(month_name)
        if month:
            return f"{base_date.year:04d}-{month:02d}-{day:02d} {hour_min}"

    if base_date:
        return base_date.isoformat()
    return ""


def extract_title(soup: BeautifulSoup) -> str:
    meta = soup.find("meta", attrs={"property": "og:title"})
    if meta and meta.get("content"):
        return clean_title(meta["content"])
    h1 = soup.find("h1")
    if h1:
        return clean_title(h1.get_text(" ", strip=True))
    title_tag = soup.find("title")
    if title_tag:
        return clean_title(title_tag.get_text(" ", strip=True))
    return ""


def clean_article_text(text: str) -> str:
    bad_prefixes = (
        "Advertisement:",
        "Support Ukrainska Pravda",
        "Посилання скопійовано",
        "The use of site materials is allowed only",
        "© 2000-",
        "Project founder:",
        "Registered online media entity",
        "Address:",
    )
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(line.startswith(p) for p in bad_prefixes):
            continue
        if line in {"Latest news", "Top news of today", "All news", "Sections", "Topics"}:
            continue
        if re.fullmatch(r"\d{1,2}:\d{2}", line):
            continue
        out.append(line)
    # stop before footer/legal boilerplate if it slipped through
    cut_markers = [
        "The use of site materials is allowed only",
        "Materials marked as PROMOTED",
        "Project founder:",
    ]
    joined = "\n".join(out)
    lower_joined = joined.lower()
    cut_positions = [lower_joined.find(m.lower()) for m in cut_markers if lower_joined.find(m.lower()) != -1]
    if cut_positions:
        joined = joined[: min(cut_positions)].strip()
    return joined.strip()


def extract_text(html: str) -> str:
    if trafilatura is not None:
        try:
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
                no_fallback=False,
            )
            if text:
                return clean_article_text(text)
        except Exception:
            pass

    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for selector in ["article", "main", "div.post__text", "div.article", "div.news"]:
        node = soup.select_one(selector)
        if node:
            candidates.append(node)
    if not candidates:
        candidates = [soup]

    best = ""
    for node in candidates:
        parts = []
        for el in node.find_all(["p", "li", "blockquote"]):
            txt = el.get_text(" ", strip=True)
            if txt:
                parts.append(txt)
        candidate = clean_article_text("\n".join(parts))
        if len(candidate) > len(best):
            best = candidate

    if best:
        return best
    return clean_article_text("\n".join(soup_text_lines(soup)))


def iter_day_urls(cfg: SiteConfig, date_from: dt.date, date_to: dt.date) -> Iterable[str]:
    current = date_to
    first = True
    while current >= date_from:
        if first and current == dt.date.today():
            yield cfg.latest_url
        else:
            yield cfg.date_url_tpl.format(ddmmyyyy=current.strftime("%d%m%Y"))
        current -= dt.timedelta(days=1)
        first = False


def collect_article_urls(
    fetcher: Fetcher,
    cfg: SiteConfig,
    limit: int,
    date_from: dt.date,
    date_to: dt.date,
    delay: float,
) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for page_url in iter_day_urls(cfg, date_from, date_to):
        print(f"[list] {page_url}")
        try:
            html = fetcher.get(page_url)
        except Exception as e:
            print(f"  -> skipped list page: {e}")
            continue
        for url in extract_list_urls(html, cfg):
            if url not in seen:
                seen.add(url)
                urls.append(url)
                if len(urls) >= limit:
                    return urls
        time.sleep(delay)
    return urls


def scrape_article(fetcher: Fetcher, url: str, delay: float) -> Optional[dict[str, str]]:
    try:
        html = fetcher.get(url)
    except Exception as e:
        print(f"  -> skipped article: {e}")
        return None

    soup = BeautifulSoup(html, "html.parser")
    title = extract_title(soup)
    published_at = extract_published_at(html, url)
    text = extract_text(html)
    time.sleep(delay)

    if not title or not text:
        return None

    return {
        "source": "pravda.com.ua",
        "title": title,
        "published_at": published_at,
        "url": url,
        "text": text,
    }


def write_csv(rows: list[dict[str, str]], output_path: str) -> None:
    fieldnames = ["source", "title", "published_at", "url", "text"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = SITE_CONFIGS[args.lang]

    date_to = parse_date_arg(args.date_to) or dt.date.today()
    date_from = parse_date_arg(args.date_from)
    if date_from is None:
        date_from = date_to - dt.timedelta(days=max(args.max_days_back - 1, 0))
    if date_from > date_to:
        raise SystemExit("date-from cannot be after date-to")

    fetcher = Fetcher(
        mode=args.fetch_mode,
        timeout=args.timeout,
        headed=args.headed,
        verbose=args.verbose,
    )
    try:
        article_urls = collect_article_urls(
            fetcher=fetcher,
            cfg=cfg,
            limit=args.limit,
            date_from=date_from,
            date_to=date_to,
            delay=args.delay,
        )
        if not article_urls:
            print("[done] no article URLs collected")
            write_csv([], args.output)
            print(f"[done] wrote 0 records to {args.output}")
            return

        print(f"[collect] discovered {len(article_urls)} article URLs")
        rows: list[dict[str, str]] = []
        for idx, url in enumerate(article_urls, 1):
            print(f"[article {idx}/{len(article_urls)}] {url}")
            row = scrape_article(fetcher, url, args.delay)
            if row is not None:
                rows.append(row)

        write_csv(rows, args.output)
        print(f"[done] wrote {len(rows)} records to {args.output}")
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
