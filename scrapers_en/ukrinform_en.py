import argparse
import csv
import re
import time
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import trafilatura
except ImportError:
    trafilatura = None


SOURCE_NAME = "ukrinform.net"
SEED_URL = "https://www.ukrinform.net/rubric-polytics"
BASE_URL = "https://www.ukrinform.net"
CSV_COLUMNS = ["source", "title", "published_at", "url", "text"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,uk;q=0.5",
}

ARTICLE_URL_RE = re.compile(
    r"^https://www\.ukrinform\.net/rubric-[a-z0-9-]+/\d+-[^/?#]+\.html$"
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
    return urljoin(base_url, href.split("#", 1)[0].strip())


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def is_same_domain(url: str) -> bool:
    return urlparse(url).netloc.endswith("ukrinform.net")


def rubric_page_url(page_num: int) -> str:
    if page_num <= 1:
        return SEED_URL
    return f"{SEED_URL}?page={page_num}"


def extract_article_urls(html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []

    for a in soup.find_all("a", href=True):
        full_url = normalize_url(a["href"], page_url)
        if is_same_domain(full_url) and ARTICLE_URL_RE.match(full_url):
            urls.append(full_url)

    return dedupe_preserve_order(urls)


def collect_article_urls(
    session: requests.Session,
    limit: int,
    delay_sec: float = 1.0,
    max_pages_to_scan: int = 50,
) -> list[str]:
    collected: list[str] = []

    for page_num in range(1, max_pages_to_scan + 1):
        page_url = rubric_page_url(page_num)
        print(f"[list] page {page_num}: {page_url}")

        try:
            html = fetch_html(session, page_url)
        except Exception as e:
            print(f"  -> skipped page: {e}")
            continue

        page_article_urls = extract_article_urls(html, page_url)
        new_count = 0

        for article_url in page_article_urls:
            if article_url not in collected:
                collected.append(article_url)
                new_count += 1
                if len(collected) >= limit:
                    return collected[:limit]

        if new_count == 0:
            print("[list] no new article URLs found, stopping")
            break

        time.sleep(delay_sec)

    return collected[:limit]


def clean_text_lines(lines: list[str]) -> str:
    filtered: list[str] = []
    skip_exact = {
        "Ukrinform",
        "Image",
        "photos",
        "video",
        "Exclusive",
    }

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line in skip_exact:
            continue
        if line.startswith("Read also:"):
            continue
        if line.startswith("Photo:"):
            continue
        if line.startswith("Topics"):
            break
        if line.startswith("Agency"):
            break
        if line.startswith("While citing and using any materials"):
            break
        filtered.append(line)

    return "\n".join(filtered).strip()


def fallback_extract_text(soup: BeautifulSoup) -> str:
    body = (
        soup.select_one("article")
        or soup.select_one("main article")
        or soup.select_one("div[itemprop='articleBody']")
        or soup.select_one("main")
    )
    scope = body if body is not None else soup

    chunks: list[str] = []
    for node in scope.find_all(["p", "li"]):
        text = node.get_text(" ", strip=True)
        if len(text) >= 35:
            chunks.append(text)

    return clean_text_lines(chunks)


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

    title = None
    published_at = None

    if metadata:
        title = metadata.title or None
        published_at = metadata.date or None

    title_el = soup.find("h1")
    if not title and title_el:
        title = title_el.get_text(" ", strip=True)

    full_text = soup.get_text("\n", strip=True)
    if not published_at:
        dt_match = DATETIME_RE.search(full_text)
        published_at = dt_match.group(0) if dt_match else None

    text = clean_text_lines((extracted_text or "").splitlines())
    if len(text) < 200:
        text = fallback_extract_text(soup)

    if not title or not text or len(text) < 200:
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
        description="Scrape English Ukrinform politics articles into a normalized CSV."
    )
    parser.add_argument("--limit", type=int, default=25, help="How many articles to fetch.")
    parser.add_argument(
        "--output",
        type=str,
        default="ukrinform_en.csv",
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
        default=50,
        help="Maximum rubric pages to scan while discovering article URLs.",
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
