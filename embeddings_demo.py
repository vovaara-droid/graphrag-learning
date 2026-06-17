"""
Демонстрація роботи ембедингів на українських новинах.
- Модель: paraphrase-multilingual-mpnet-base-v2
- Метрика: cosine similarity
- Виводить теплову карту всіх пар + конкретні порівняння
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── 8 заголовків з news.csv — різні теми ──────────────────────────────────────
HEADLINES = [
    "Пожежа у художньому музеї Харкова після удару РФ: усе найцінніше вивезено",      # 0 — війна / культура
    "Зранку російські війська атакували авто на двох дорогах Донеччини, троє поранених",# 1 — бойові дії
    "Трамп і Путін домовились про візит до Москви Віткоффа та Кушнера",                # 2 — дипломатія
    "Льюїс Гемілтон вперше виграв гонку Формули-1 в статусі пілота Ferrari",           # 3 — спорт (Ф-1)
    "Олімпійська віцечемпіонка виграла перший титул WTA за три роки",                  # 4 — спорт (теніс)
    "Погода в Івано-Франківську та області 15 червня",                                  # 5 — погода
    "До 135-річчя Євгена Коновальця у Житомирі відкрили мурал на його честь",          # 6 — культура / історія
    "На подіум у 60 років: історія моделі зі Львова Тетяни Помірко",                   # 7 — людська історія
]

# Скорочені мітки для осей heatmap
LABELS = [
    "Музей Харків (удар)",
    "Атака Донеччина",
    "Трамп-Путін Москва",
    "Гемілтон Ferrari",
    "WTA чемпіонка",
    "Погода Франківськ",
    "Мурал Коновалець",
    "Модель 60 років",
]

# ── Конкретні порівняння для демонстрації «фізики» ────────────────────────────
COMPARISON_PAIRS = [
    (
        "Зеленський зустрівся з лідерами G7",
        "Президент провів переговори з G7",
        "схожі за змістом",
    ),
    (
        "Зеленський зустрівся з лідерами G7",
        "Росія завдала удару по Харкову",
        "пов'язані темою (війна), але різні події",
    ),
    (
        "Зеленський зустрівся з лідерами G7",
        "Матч НБА завершився перемогою Lakers",
        "абсолютно різні теми",
    ),
]


def short(text: str, n: int = 45) -> str:
    return text if len(text) <= n else text[:n] + "…"


def main() -> None:
    print("Завантажую модель paraphrase-multilingual-mpnet-base-v2...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    print("Модель готова.\n")

    # ── 1. Ембединги 8 заголовків ──────────────────────────────────────────────
    print("Кодую заголовки...")
    embeddings = model.encode(HEADLINES, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)

    # ── 2. Теплова карта ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Cosine similarity")

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(LABELS, fontsize=9)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            val = sim_matrix[i, j]
            color = "black" if val > 0.6 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_title("Cosine similarity між заголовками новин\n(paraphrase-multilingual-mpnet-base-v2)",
                 fontsize=13, pad=14)
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    out_path = "output/embeddings_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap збережено: {out_path}\n")

    # ── 3. Найближча та найдальша пара ────────────────────────────────────────
    n = len(HEADLINES)
    best_val, best_i, best_j = -1, 0, 0
    worst_val, worst_i, worst_j = 2, 0, 0

    for i in range(n):
        for j in range(i + 1, n):
            v = sim_matrix[i, j]
            if v > best_val:
                best_val, best_i, best_j = v, i, j
            if v < worst_val:
                worst_val, worst_i, worst_j = v, i, j

    print("=" * 60)
    print("НАЙБЛИЖЧА ПАРА (найвища similarity):")
    print(f"  {best_val:.4f}  |  {short(HEADLINES[best_i])}")
    print(f"          vs  |  {short(HEADLINES[best_j])}")
    print()
    print("НАЙДАЛЬША ПАРА (найнижча similarity):")
    print(f"  {worst_val:.4f}  |  {short(HEADLINES[worst_i])}")
    print(f"          vs  |  {short(HEADLINES[worst_j])}")
    print("=" * 60)

    # ── 4. Три контрольних порівняння ─────────────────────────────────────────
    print("\nКОНТРОЛЬНІ ПОРІВНЯННЯ — демонстрація фізики ембедингів:")
    print("-" * 60)
    all_sentences = [s for pair in COMPARISON_PAIRS for s in pair[:2]]
    unique = list(dict.fromkeys(all_sentences))
    comp_emb = {s: model.encode([s])[0] for s in unique}

    for a, b, comment in COMPARISON_PAIRS:
        ea = comp_emb[a].reshape(1, -1)
        eb = comp_emb[b].reshape(1, -1)
        score = cosine_similarity(ea, eb)[0][0]
        bar = "#" * int(score * 30)
        print(f"\n[{comment}]")
        print(f"  A: {a}")
        print(f"  B: {b}")
        print(f"  similarity = {score:.4f}  [{bar:<30}]")

    print()


if __name__ == "__main__":
    main()
