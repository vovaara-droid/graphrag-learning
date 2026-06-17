"""
Витягує суб'єкт-предикат-об'єкт трійки з перших 100 новин через Anthropic API.
Зберігає результат у data/triples.csv.
"""

import json
import os
import sys
import time

import anthropic
import pandas as pd
from dotenv import load_dotenv

# Force UTF-8 stdout so Ukrainian text doesn't block on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY не знайдено у .env файлі")

MODEL = "claude-haiku-4-5-20251001"
NEWS_CSV = "data/news.csv"
OUT_CSV = "data/triples.csv"
N_ARTICLES = 100

PROMPT_TEMPLATE = """Витягни всі факти з цього тексту у вигляді трійок суб'єкт-предикат-об'єкт.
Повертай тільки JSON масив без пояснень:
[{{"subject": "...", "predicate": "...", "object": "...", "subject_type": "PER/ORG/LOC", "object_type": "PER/ORG/LOC/EVENT"}}]
Текст: {text}"""


def extract_triples(client: anthropic.Anthropic, text: str) -> list[dict]:
    prompt = PROMPT_TEMPLATE.format(text=text[:3000])
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Видаляємо можливі markdown-блоки ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        triples = json.loads(raw)
        if isinstance(triples, list):
            return triples
    except (json.JSONDecodeError, IndexError, anthropic.APIError) as e:
        print(f"  [помилка парсингу]: {e}")
    return []


def main() -> None:
    df = pd.read_csv(NEWS_CSV)
    articles = df.head(N_ARTICLES)

    client = anthropic.Anthropic(api_key=API_KEY)

    rows = []
    for i, (_, row) in enumerate(articles.iterrows(), start=1):
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))
        date = str(row.get("date", ""))

        triples = extract_triples(client, text)

        for t in triples:
            rows.append({
                "subject": t.get("subject", ""),
                "predicate": t.get("predicate", ""),
                "object": t.get("object", ""),
                "subject_type": t.get("subject_type", ""),
                "object_type": t.get("object_type", ""),
                "source_title": title,
                "source_date": date,
            })

        print(f"article {i}/{N_ARTICLES}, triples: {len(triples)}", flush=True)

        # Невелика пауза щоб не перевантажити API
        if i % 10 == 0:
            time.sleep(1)

    out_df = pd.DataFrame(rows, columns=[
        "subject", "predicate", "object",
        "subject_type", "object_type",
        "source_title", "source_date",
    ])
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nDone! Saved {len(rows)} triples to {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
