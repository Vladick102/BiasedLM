import argparse
import csv
import re
import time
import xml.etree.ElementTree as ET
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


SOURCE_NAME = "mod.gov.ua"
BASE_URL = "https://mod.gov.ua"
HOME_URL = "https://mod.gov.ua/en"
NEWS_URL = "https://mod.gov.ua/en/news"
SITEMAP_URL = "https://mod.gov.ua/sitemap.xml"

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
    r"^https://mod\.gov\.ua/en/news/"
    r"(?!tag-|category-|archive|search|page(?:/|\?|$))[^/?#]+/?$"
)
ENGLISH_DT_RE = re.compile(
    r"\b\d{1,2}\s+[A-Z][a-z]+,\s+\d{4},\s+\d{1,2}:\d{2}\s+(?:AM|PM)\s+EET\b"
)


@dataclass
class ArticleRecord:
    source: str
    title: str
    published_at: Optional[str]
    url: str
    text: str


@dataclass
class SitemapEntry:
    loc: str
    lastmod: Optional[str]


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


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def fetch_text(session: requests.Session, url: str, timeout: int = 30) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def parse_sitemap_entries(xml_text: str) -> tuple[list[SitemapEntry], list[str]]:
    root = ET.fromstring(xml_text)

    url_entries: list[SitemapEntry] = []
    child_sitemaps: list[str] = []

    for elem in root:
        tag = elem.tag.lower()
        if tag.endswith("url"):
            loc = None
            lastmod = None
            for child in elem:
                child_tag = child.tag.lower()
                if child_tag.endswith("loc") and child.text:
                    loc = child.text.strip()
                elif child_tag.endswith("lastmod") and child.text:
                    lastmod = child.text.strip()
            if loc:
                url_entries.append(SitemapEntry(loc=loc, lastmod=lastmod))
        elif tag.endswith("sitemap"):
            for child in elem:
                child_tag = child.tag.lower()
                if child_tag.endswith("loc") and child.text:
                    child_sitemaps.append(child.text.strip())

    return url_entries, child_sitemaps


def try_collect_from_sitemap(session: requests.Session, limit: int, max_child_sitemaps: int = 50) -> list[str]:
    collected_entries: list[SitemapEntry] = []
    visited_sitemaps = set()
    queue = [SITEMAP_URL]

    while queue and len(visited_sitemaps) < max_child_sitemaps:
        sitemap_url = queue.pop(0)
        if sitemap_url in visited_sitemaps:
            continue
        visited_sitemaps.add(sitemap_url)

        try:
            xml_text = fetch_text(session, sitemap_url, timeout=30)
            url_entries, child_sitemaps = parse_sitemap_entries(xml_text)
        except Exception as e:
            print(f"[sitemap] failed on {sitemap_url}: {e}")
            continue

        for entry in url_entries:
            if ARTICLE_URL_RE.match(entry.loc):
                collected_entries.append(entry)

        for child in child_sitemaps:
            if child not in visited_sitemaps:
                queue.append(child)

    # Key fix: sort by lastmod descending, instead of taking sitemap order as-is.
    collected_entries.sort(key=lambda x: (x.lastmod or ""), reverse=True)

    urls = dedupe_preserve_order([entry.loc for entry in collected_entries])
    return urls[:limit]


def discover_listing_pages(soup: BeautifulSoup, base_url: str) -> list[str]:
    candidates: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#", 1)[0].strip()
        full_url = urljoin(base_url, href)
        if full_url.startswith(NEWS_URL) and full_url not in {NEWS_URL, HOME_URL}:
            if "page" in full_url or full_url.endswith("/news"):
                candidates.append(full_url)
    return dedupe_preserve_order(candidates)


def collect_from_listing_pages(session: requests.Session, limit: int, max_pages: int = 20) -> list[str]:
    collected: list[str] = []
    visited = set()
    queue: list[str] = [HOME_URL, NEWS_URL]

    while queue and len(collected) < limit and len(visited) < max_pages:
        page_url = queue.pop(0)
        if page_url in visited:
            continue
        visited.add(page_url)

        print(f"[listing] {page_url}")
        try:
            html = fetch_text(session, page_url)
        except Exception as e:
            print(f"[listing] error on {page_url}: {e}")
            continue

        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            full_url = urljoin(BASE_URL, a["href"].split("#", 1)[0].strip())
            if ARTICLE_URL_RE.match(full_url) and full_url not in collected:
                collected.append(full_url)
                if len(collected) >= limit:
                    break

        for next_url in discover_listing_pages(soup, page_url):
            if next_url not in visited and next_url not in queue:
                queue.append(next_url)

    return collected[:limit]


def clean_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Image:"):
            continue
        if line in {"Skip to main content", "News", "Main page"}:
            continue
        if line.startswith("Hotline:"):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def fallback_extract_text(soup: BeautifulSoup) -> str:
    body = (
        soup.select_one("article")
        or soup.select_one("main article")
        or soup.select_one("div.article-body")
        or soup.select_one("div.entry-content")
        or soup.select_one("main")
    )
    scope = body if body is not None else soup

    chunks: list[str] = []
    for node in scope.find_all(["p", "li"]):
        text = node.get_text(" ", strip=True)
        if len(text) >= 35:
            chunks.append(text)

    return clean_text("\n".join(chunks))


def extract_article(session: requests.Session, url: str) -> Optional[ArticleRecord]:
    html = fetch_text(session, url)
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

    h1 = soup.find("h1")
    if not title and h1:
        title = h1.get_text(" ", strip=True)

    if not published_at:
        date_elem = soup.find("time")
        if date_elem:
            published_at = date_elem.get_text(" ", strip=True)

    if not published_at:
        full_text = soup.get_text("\n", strip=True)
        m = ENGLISH_DT_RE.search(full_text)
        if m:
            published_at = m.group(0)

    text = clean_text(extracted_text or "")
    if len(text) < 120:
        text = fallback_extract_text(soup)

    if not title or not text or len(text) < 120:
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
        description="Scrape English mod.gov.ua news articles into a normalized CSV."
    )
    parser.add_argument("--limit", type=int, default=25, help="How many articles to fetch.")
    parser.add_argument(
        "--output",
        type=str,
        default="mil_gov_ua_en.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between article requests in seconds.",
    )
    parser.add_argument(
        "--max-list-pages",
        type=int,
        default=20,
        help="Maximum listing pages to scan when falling back from sitemap.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be > 0")

    session = build_session()

    print("[collect] trying sitemap first...")
    article_urls = try_collect_from_sitemap(session, limit=args.limit)

    if article_urls:
        print(f"[collect] got {len(article_urls)} URLs from sitemap")
    else:
        print("[collect] sitemap path yielded nothing, falling back to English listing pages")
        article_urls = collect_from_listing_pages(
            session,
            limit=args.limit,
            max_pages=args.max_list_pages,
        )
        print(f"[collect] got {len(article_urls)} URLs from fallback")

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
