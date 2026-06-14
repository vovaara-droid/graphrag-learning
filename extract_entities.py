import csv
import os
from collections import Counter

import spacy

CATEGORIES = ["PER", "ORG", "LOC"]
TOP_N = 20

nlp = spacy.load("uk_core_news_sm")

rows = []
with open("data/news.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        text = f"{row['title']} {row['text']}".strip()
        if text:
            rows.append(text)

print(f"Обробляю {len(rows)} новин...")

counters = {cat: Counter() for cat in CATEGORIES}

for i, text in enumerate(rows, 1):
    doc = nlp(text[:10000])
    for ent in doc.ents:
        label = ent.label_
        if label in counters:
            # use lemma to merge inflected forms (Трампа/Трампом → Трамп)
            lemma = ent.lemma_.strip() or ent.text.strip()
            counters[label][lemma] += 1
    if i % 100 == 0:
        print(f"  {i}/{len(rows)}")

os.makedirs("data", exist_ok=True)
with open("data/entities.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "entity", "count"])
    for cat in CATEGORIES:
        for entity, count in counters[cat].most_common(TOP_N):
            writer.writerow([cat, entity, count])

print("\ndata/entities.csv збережено.\n")

labels = {"PER": "Персони (PER)", "ORG": "Організації (ORG)", "LOC": "Місця (LOC)"}
for cat in CATEGORIES:
    print("-" * 40)
    print(f"  {labels[cat]} - топ-10")
    print("-" * 40)
    for rank, (entity, count) in enumerate(counters[cat].most_common(10), 1):
        print(f"  {rank:>2}. {entity:<30} {count:>4}")
    print()
