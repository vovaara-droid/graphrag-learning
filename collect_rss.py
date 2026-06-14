import csv
import os
from datetime import datetime
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET

import requests

RSS_FEEDS = {
    "Суспільне": "https://suspilne.media/rss/all.rss",
    "ТСН": "https://tsn.ua/rss/full.rss",
    "Українська правда": "https://www.pravda.com.ua/rss/view_news/",
    "Bihus.info": "https://bihus.info/feed",
    "Гордон": "https://gordonua.com/ukr/api/media/out/rss/lastnews.xml",
    "УНІАН": "https://rss.unian.net/site/news_ukr.rss",
    "НВ": "https://nv.ua/rss/all.xml",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "news.csv")


def parse_date(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return raw.strip()


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    import re
    return re.sub(r"<[^>]+>", "", text).strip()


def fetch_feed(source: str, url: str) -> list[dict]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [!] {source}: помилка завантаження — {e}")
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        print(f"  [!] {source}: помилка парсингу XML — {e}")
        return []

    items = root.findall(".//item")
    records = []
    for item in items:
        title = (item.findtext("title") or "").strip()
        description = strip_html(item.findtext("description"))
        pub_date = parse_date(item.findtext("pubDate"))
        records.append({
            "date": pub_date,
            "source": source,
            "title": title,
            "text": description,
        })
    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_records: list[dict] = []
    counts: dict[str, int] = {}

    for source, url in RSS_FEEDS.items():
        print(f"Завантажую: {source} ...")
        records = fetch_feed(source, url)
        counts[source] = len(records)
        all_records.extend(records)
        print(f"  -> {len(records)} новин")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "source", "title", "text"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nЗбережено {len(all_records)} новин у {OUTPUT_FILE}")
    print("\nПідсумок за джерелами:")
    for source, count in counts.items():
        print(f"  {source:<22} {count:>4} новин")


if __name__ == "__main__":
    main()
