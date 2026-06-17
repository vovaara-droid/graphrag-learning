"""
Будує граф співзустрічальності сутностей з новин:
- NER через spaCy (PER / ORG / LOC)
- Нормалізація через словник aliases
- Вузол = сутність, ребро = спільна поява в одній новині
- Вага ребра = кількість спільних новин
- Зберігає граф у output/knowledge_graph.png
"""

import csv
import os
from collections import defaultdict
from itertools import combinations

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import spacy

# ── Нормалізація ──────────────────────────────────────────────────────────────
ALIASES: dict[str, str] = {
    # Персони
    "дональд трамп":       "трамп",
    "д. трамп":            "трамп",
    "володимир зеленський": "зеленський",
    "в. зеленський":       "зеленський",
    "володимир путін":     "путін",
    "в. путін":            "путін",
    # Місця
    "рф":                  "росія",
    "російська федерація": "росія",
    "сполучені штати":     "сша",
    "сполучені штати америки": "сша",
    "америка":             "сша",
    "київ":                "київ",
    "кийів":               "київ",
    "кий":                 "київ",
    "львов":               "львів",
}

CATEGORIES = {"PER", "ORG", "LOC"}
MIN_ENTITY_FREQ = 3   # відкидаємо рідкісні сутності
TOP_NODES       = 40  # скільки вузлів візуалізувати


def normalize(text: str) -> str:
    key = text.lower().strip()
    return ALIASES.get(key, key)


def load_texts(path: str) -> list[str]:
    texts = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            combined = f"{row['title']} {row.get('text', '')}".strip()
            if combined:
                texts.append(combined)
    return texts


def extract_entities_per_doc(
    texts: list[str], nlp
) -> tuple[list[list[tuple[str, str]]], dict[str, str]]:
    """
    Повертає:
      doc_entities — список [[(норм_сутність, категорія), ...], ...]
      entity_category — словник сутність → категорія (остання зустріч)
    """
    doc_entities: list[list[tuple[str, str]]] = []
    entity_category: dict[str, str] = {}

    for i, text in enumerate(texts, 1):
        doc = nlp(text[:10_000])
        seen: set[str] = set()
        doc_ents: list[tuple[str, str]] = []
        for ent in doc.ents:
            if ent.label_ not in CATEGORIES:
                continue
            lemma = (ent.lemma_.strip() or ent.text.strip()).lower()
            normed = normalize(lemma)
            if normed and normed not in seen:
                seen.add(normed)
                doc_ents.append((normed, ent.label_))
                entity_category[normed] = ent.label_
        doc_entities.append(doc_ents)
        if i % 100 == 0:
            print(f"  {i}/{len(texts)}")

    return doc_entities, entity_category


def build_graph(
    doc_entities: list[list[tuple[str, str]]],
    entity_category: dict[str, str],
    min_freq: int,
) -> nx.Graph:
    freq: dict[str, int] = defaultdict(int)
    for doc in doc_entities:
        for ent, _ in doc:
            freq[ent] += 1

    allowed = {e for e, c in freq.items() if c >= min_freq}

    G = nx.Graph()
    for ent, cat in entity_category.items():
        if ent in allowed:
            G.add_node(ent, category=cat, freq=freq[ent])

    for doc in doc_entities:
        ents = [e for e, _ in doc if e in allowed]
        for a, b in combinations(ents, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    return G


def visualize(G: nx.Graph, top_n: int, out_path: str) -> None:
    # Залишаємо top_n вузлів за degree (сума ваг ребер)
    strengths = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True))
                 for n in G.nodes()}
    top_nodes = sorted(strengths, key=strengths.get, reverse=True)[:top_n]
    sub = G.subgraph(top_nodes).copy()

    COLOR_MAP = {"PER": "#4C72B0", "ORG": "#55A868", "LOC": "#C44E52"}
    node_colors = [COLOR_MAP.get(sub.nodes[n].get("category", ""), "#888") for n in sub.nodes()]
    node_sizes  = [300 + sub.nodes[n].get("freq", 1) * 60 for n in sub.nodes()]
    edge_widths = [0.5 + d["weight"] * 0.4 for _, _, d in sub.edges(data=True)]

    fig, ax = plt.subplots(figsize=(20, 16))
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")

    pos = nx.spring_layout(sub, k=2.2, seed=42, weight="weight")

    nx.draw_networkx_edges(
        sub, pos, ax=ax,
        width=edge_widths,
        edge_color="#ffffff",
        alpha=0.25,
    )
    nx.draw_networkx_nodes(
        sub, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
    )
    nx.draw_networkx_labels(
        sub, pos, ax=ax,
        font_size=8,
        font_color="white",
        font_weight="bold",
    )

    legend = [
        mpatches.Patch(color=COLOR_MAP["PER"], label="Персони (PER)"),
        mpatches.Patch(color=COLOR_MAP["ORG"], label="Організації (ORG)"),
        mpatches.Patch(color=COLOR_MAP["LOC"], label="Місця (LOC)"),
    ]
    ax.legend(handles=legend, loc="upper left", facecolor="#222", labelcolor="white", fontsize=11)
    ax.set_title(
        f"Граф співзустрічальності сутностей (топ-{top_n})",
        color="white", fontsize=16, pad=16,
    )
    ax.axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nГраф збережено: {out_path}")


def main() -> None:
    print("Завантажую spaCy...")
    nlp = spacy.load("uk_core_news_sm")

    print("Читаю новини...")
    texts = load_texts("data/news.csv")
    print(f"Новин: {len(texts)}")

    print("\nВитягую сутності...")
    doc_entities, entity_category = extract_entities_per_doc(texts, nlp)

    print("\nБудую граф...")
    G = build_graph(doc_entities, entity_category, MIN_ENTITY_FREQ)
    print(f"Вузлів: {G.number_of_nodes()}, Ребер: {G.number_of_edges()}")

    print("\nВізуалізую...")
    visualize(G, TOP_NODES, "output/knowledge_graph.png")

    # Топ-10 вузлів за силою зв'язків
    strengths = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True))
                 for n in G.nodes()}
    print("\nТоп-10 найпов'язаніших сутностей:")
    for rank, (node, strength) in enumerate(
        sorted(strengths.items(), key=lambda x: x[1], reverse=True)[:10], 1
    ):
        cat = G.nodes[node].get("category", "?")
        freq = G.nodes[node].get("freq", 0)
        print(f"  {rank:>2}. [{cat}] {node:<30} сила={strength}  згадувань={freq}")


if __name__ == "__main__":
    main()
