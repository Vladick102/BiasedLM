import argparse
import csv
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SOURCE_NAME = "ukrinform.ua"
SEED_URL = "https://www.ukrinform.ua/rubric-politics/"
BASE_URL = "https://www.ukrinform.ua"

CSV_COLUMNS = ["source", "title", "published_at", "url", "text"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "uk-UA,uk;q=0.9,en;q=0.8",
}

# Example:
# https://www.ukrinform.ua/rubric-ato/4105147-majze-tisaca-droniv-za-dobu-povitrani-sili-uprodovz-dna-zbili-541-bpla-rosian.html
ARTICLE_URL_RE = re.compile(
    r"^https://www\.ukrinform\.ua/rubric-[a-z0-9-]+/\d+-[^/?#]+\.html$"
)

DATETIME_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}\b")


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


def fetch_html(session: requests.Session, url: str, timeout: int = 30) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def normalize_url(href: str, base_url: str = BASE_URL) -> str:
    return urljoin(base_url, href.split("#")[0].strip())


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def is_same_domain(url: str) -> bool:
    return urlparse(url).netloc.endswith("ukrinform.ua")


def extract_article_urls(html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls = []

    for a in soup.find_all("a", href=True):
        full_url = normalize_url(a["href"], page_url)
        if is_same_domain(full_url) and ARTICLE_URL_RE.match(full_url):
            urls.append(full_url)

    return dedupe_preserve_order(urls)


def collect_article_urls(
    session: requests.Session,
    limit: int,
    delay_sec: float = 1.0,
    max_pages_to_scan: int = 100,
) -> list[str]:
    """
    Seed from the politics page, then recurse through discovered article pages.
    This avoids relying on fragile archive pagination.
    """
    collected: list[str] = []
    scanned_pages = set()
    queue = deque([SEED_URL])

    while queue and len(collected) < limit and len(scanned_pages) < max_pages_to_scan:
        page_url = queue.popleft()
        if page_url in scanned_pages:
            continue
        scanned_pages.add(page_url)

        print(f"[scan] {page_url}")

        try:
            html = fetch_html(session, page_url)
        except Exception as e:
            print(f"  -> skipped page: {e}")
            continue

        page_article_urls = extract_article_urls(html, page_url)

        # Save new article URLs
        for article_url in page_article_urls:
            if article_url not in collected:
                collected.append(article_url)
                if len(collected) >= limit:
                    break

        # Recurse through article pages too, because Ukrinform pages contain related/article blocks
        for article_url in page_article_urls:
            if article_url not in scanned_pages and article_url not in queue:
                queue.append(article_url)

        time.sleep(delay_sec)

    return collected[:limit]


def clean_text_lines(lines: list[str]) -> str:
    cleaned = [line.strip() for line in lines if line and line.strip()]
    return "\n".join(cleaned).strip()


def extract_article(session: requests.Session, url: str) -> Optional[ArticleRecord]:
    html = fetch_html(session, url)
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.find("h1")
    if not title_el:
        return None
    title = title_el.get_text(" ", strip=True)

    full_text = soup.get_text("\n", strip=True)
    dt_match = DATETIME_RE.search(full_text)
    published_at = dt_match.group(0) if dt_match else None

    # Turn page text into line list
    lines = [line.strip() for line in full_text.splitlines()]
    lines = [line for line in lines if line]

    # Find title line
    try:
        start_idx = lines.index(title)
    except ValueError:
        start_idx = None

    if start_idx is None:
        return None

    # Body starts after title, date, source label, short lead
    body_lines = lines[start_idx + 1 :]

    stop_markers = {
        "Приєднуйтесь до наших каналів",
        "Актуально",
        "Рубрики",
        "Агентство",
        "Розширений пошук",
        "Приховати розширений пошук",
    }

    final_lines = []
    for line in body_lines:
        # stop if line begins with one of the known promo/footer markers
        if any(line.startswith(marker) for marker in stop_markers):
            break
        final_lines.append(line)

    # Remove duplicated metadata at the start
    if final_lines and published_at and final_lines[0] == published_at:
        final_lines = final_lines[1:]

    if final_lines and final_lines[0] == "Укрінформ":
        final_lines = final_lines[1:]

    text = clean_text_lines(final_lines)

    if len(text) < 200:
        return None

    return ArticleRecord(
        source=SOURCE_NAME,
        title=title,
        published_at=published_at,
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
        description="Scrape Ukrinform political/news articles into a normalized CSV."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="How many articles to fetch.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ukrinform.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds.",
    )
    parser.add_argument(
        "--max-pages-to-scan",
        type=int,
        default=100,
        help="Maximum seed/article pages to scan while discovering article URLs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be > 0")

    session = build_session()

    article_urls = collect_article_urls(
        session=session,
        limit=args.limit,
        delay_sec=args.delay,
        max_pages_to_scan=args.max_pages_to_scan,
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