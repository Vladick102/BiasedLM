import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import trafilatura
except ImportError:
    trafilatura = None


SOURCE_NAME = "armyinform.com.ua"
ARCHIVE_BASE_URL = "https://armyinform.com.ua/en/category/latest-news/"
CSV_COLUMNS = ["source", "title", "published_at", "url", "text"]

ARTICLE_URL_RE = re.compile(
    r"^https://armyinform\.com\.ua/en/\d{4}/\d{2}/\d{2}/[^/?#]+/?$"
)
ENGLISH_DATE_RE = re.compile(
    r"\b(\d{1,2}\s+[A-Z][a-z]+\s+\d{4},\s+\d{1,2}:\d{2})\b"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,uk;q=0.5",
}


@dataclass
class ArticleRecord:
    source: str
    title: str
    published_at: Optional[str]
    url: str
    text: str


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)

    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def normalize_url(base_url: str, href: str) -> str:
    return urljoin(base_url, href.split("#", 1)[0].strip())


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def fetch_html(session: requests.Session, url: str, timeout: int = 30) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def archive_page_url(page_num: int) -> str:
    if page_num <= 1:
        return ARCHIVE_BASE_URL
    return f"{ARCHIVE_BASE_URL}page/{page_num}/"


def extract_article_urls_from_archive(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []

    for a in soup.find_all("a", href=True):
        full_url = normalize_url(base_url, a["href"])
        if ARTICLE_URL_RE.match(full_url):
            urls.append(full_url)

    return dedupe_preserve_order(urls)


def collect_article_urls(
    session: requests.Session,
    limit: int,
    delay_sec: float = 1.0,
    max_pages: int = 50,
) -> list[str]:
    collected: list[str] = []

    for page_num in range(1, max_pages + 1):
        page_url = archive_page_url(page_num)
        print(f"[archive] page {page_num}: {page_url}")

        try:
            html = fetch_html(session, page_url)
        except requests.HTTPError as e:
            print(f"[archive] stopping on HTTP error: {e}")
            break

        page_urls = extract_article_urls_from_archive(html, page_url)
        new_count = 0

        for url in page_urls:
            if url not in collected:
                collected.append(url)
                new_count += 1
                if len(collected) >= limit:
                    return collected[:limit]

        if new_count == 0:
            print("[archive] no new article URLs found, stopping")
            break

        time.sleep(delay_sec)

    return collected[:limit]


def clean_text(text: str) -> str:
    if not text:
        return ""

    cut_markers = [
        "Читайте нас в Telegram",
        "Читайте нас у Telegram",
        "Читайте нас у Facebook",
        "Read us on Telegram",
        "Read us on Facebook",
        "Creative Commons Attribution 4.0 International license",
        "© 2018 - 2026",
    ]
    for marker in cut_markers:
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    filtered_lines: list[str] = []
    skip_exact = {
        "Latest News",
        "Important news",
        "Reports",
        "Home",
        "About Us",
        "UA",
        "УКРАЇНСЬКА",
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in skip_exact:
            continue
        if line.startswith("Image:"):
            continue
        if line.startswith("Прочитаєте за:"):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


def fallback_extract_text(soup: BeautifulSoup) -> str:
    container = (
        soup.select_one("article")
        or soup.select_one("main article")
        or soup.select_one("div.td-post-content")
        or soup.select_one("div.entry-content")
        or soup.select_one("main")
    )
    scope = container if container is not None else soup

    paragraphs: list[str] = []
    for p in scope.find_all(["p", "li"]):
        t = p.get_text(" ", strip=True)
        if len(t) >= 40:
            paragraphs.append(t)

    return clean_text("\n".join(paragraphs))


def extract_article(session: requests.Session, url: str) -> Optional[ArticleRecord]:
    html = fetch_html(session, url)
    soup = BeautifulSoup(html, "html.parser")

    metadata = trafilatura.extract_metadata(html) if trafilatura else None
    extracted_text = None
    if trafilatura:
        extracted_text = trafilatura.extract(
            html,
            include_links=False,
            include_images=False,
            include_tables=False,
            favor_precision=True,
            deduplicate=True,
        )

    title: Optional[str] = None
    published_at: Optional[str] = None

    if metadata:
        title = metadata.title or None
        published_at = metadata.date or None

    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)

    full_text = soup.get_text("\n", strip=True)
    if not published_at:
        m = ENGLISH_DATE_RE.search(full_text)
        if m:
            published_at = m.group(1)

    text = clean_text(extracted_text or "")
    if len(text) < 200:
        text = fallback_extract_text(soup)

    if not title or not text or len(text) < 200:
        return None

    return ArticleRecord(
        source=SOURCE_NAME,
        title=title.strip(),
        published_at=published_at.strip() if isinstance(published_at, str) else published_at,
        url=url,
        text=text,
    )


def write_csv(path: str, records: list[ArticleRecord]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape English ArmyInform news articles into a normalized CSV."
    )
    parser.add_argument("--limit", type=int, default=25, help="How many articles to fetch.")
    parser.add_argument(
        "--output",
        type=str,
        default="armyinform_en.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum archive pages to scan while collecting article URLs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be > 0")
    if args.max_pages <= 0:
        raise ValueError("--max-pages must be > 0")

    session = build_session()

    article_urls = collect_article_urls(
        session=session,
        limit=args.limit,
        delay_sec=args.delay,
        max_pages=args.max_pages,
    )
    print(f"[info] collected {len(article_urls)} article URLs")

    records: list[ArticleRecord] = []

    for idx, url in enumerate(article_urls, start=1):
        print(f"[article {idx}/{len(article_urls)}] {url}")
        try:
            record = extract_article(session, url)
            if record:
                records.append(record)
                print(f"  -> saved: {record.title[:90]}")
            else:
                print("  -> skipped: could not extract clean article")
        except Exception as e:
            print(f"  -> skipped: {e}")

        time.sleep(args.delay)

    write_csv(args.output, records)
    print(f"[done] wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
