"""
Регіональний аналіз атак по Україні з 2023 року
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "d:/output"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "figure.dpi": 130,
    "axes.grid": False,
})

# ─────────────────────────────────────────────────────────────
# ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("d:/data/missile_attacks_daily.csv")
df["date"] = pd.to_datetime(df["time_start"].str[:10], errors="coerce")
df = df[df["date"].dt.year >= 2023].dropna(subset=["launched", "date"]).copy()
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month

print(f"Завантажено: {len(df)} рядків (2023–{df['year'].max()})")

# ─────────────────────────────────────────────────────────────
# ФУНКЦІЯ ВИЛУЧЕННЯ РЕГІОНІВ
# ─────────────────────────────────────────────────────────────
OBLAST_KEYWORDS = [
    "Cherkasy", "Chernihiv", "Chernivtsi", "Dnipropetrovsk", "Donetsk",
    "Ivano-Frankivsk", "Kharkiv", "Kherson", "Khmelnytskyi", "Kirovohrad",
    "Kyiv", "Luhansk", "Lviv", "Mykolaiv", "Odesa", "Poltava",
    "Rivne", "Sumy", "Ternopil", "Vinnytsia", "Volyn",
    "Zakarpattia", "Zaporizhzhia", "Zhytomyr",
]
CITY_TO_OBLAST = {
    "Dnipro": "Dnipropetrovsk", "Kryvyi Rih": "Dnipropetrovsk",
    "Kharkiv": "Kharkiv", "Kyiv": "Kyiv", "Odesa": "Odesa",
    "Kherson": "Kherson", "Sumy": "Sumy", "Zaporizhzhia": "Zaporizhzhia",
    "Mykolaiv": "Mykolaiv", "Lviv": "Lviv", "Chernihiv": "Chernihiv",
    "Kramatorsk": "Donetsk", "Starokostiantyniv": "Khmelnytskyi",
    "Kolomyia": "Ivano-Frankivsk", "Snake Island": "Odesa",
}
SKIP_TARGETS = {
    "ukraine", "south", "east", "north", "west", "south-east", "south-west",
    "north-east", "north-west", "front line", "south and east",
    "south and north", "south and east and center", "north and east",
    "north and center", "south and north and center",
}

def extract_oblasts(target):
    if pd.isna(target):
        return []
    parts = [p.strip() for p in str(target).split(" and ")]
    result = []
    for part in parts:
        # зрізаємо підрайон (після коми)
        main = part.split(",")[0].strip()
        if main.lower() in SKIP_TARGETS:
            continue
        # якщо це "X oblast"
        if " oblast" in main.lower():
            name = main.replace(" oblast", "").replace(" Oblast", "").strip()
            name = name.split()[0]  # беремо перше слово (назву)
            # нормалізуємо відомі скорочення
            for kw in OBLAST_KEYWORDS:
                if name.lower() == kw.lower():
                    result.append(kw)
                    break
            else:
                result.append(name)
        # якщо це відоме місто
        elif main in CITY_TO_OBLAST:
            result.append(CITY_TO_OBLAST[main])
        # якщо назва точно збігається з Oblast keyword
        else:
            for kw in OBLAST_KEYWORDS:
                if kw.lower() in main.lower():
                    result.append(kw)
                    break
    return list(dict.fromkeys(result))  # унікальні, зберігаємо порядок

# Explode: кожен регіон → окремий рядок
df["oblasts"] = df["target"].apply(extract_oblasts)
df_exploded = df.explode("oblasts").dropna(subset=["oblasts"])
df_exploded = df_exploded[df_exploded["oblasts"] != ""]
print(f"Рядків з визначеними регіонами: {len(df_exploded)}")
print(f"Унікальних регіонів: {df_exploded['oblasts'].nunique()}")

YEARS  = sorted(df["year"].unique())
MONTHS = list(range(1, 13))
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
YEAR_LABELS = [str(y) for y in YEARS]

# Топ-регіони за кількістю атак (для читабельності графіків)
top_oblasts = (
    df_exploded.groupby("oblasts")["date"]
    .count()
    .sort_values(ascending=False)
    .head(18)
    .index.tolist()
)
df_reg = df_exploded[df_exploded["oblasts"].isin(top_oblasts)]

# ─────────────────────────────────────────────────────────────
# CUSTOM COLORMAP (чорний → жовтий → червоний)
# ─────────────────────────────────────────────────────────────
FIRE_CMAP = LinearSegmentedColormap.from_list(
    "fire", ["#0d1117", "#1a3a5c", "#e65100", "#ff6f00", "#ffca28"]
)
RED_CMAP = LinearSegmentedColormap.from_list(
    "heat", ["#fff9f9", "#ffccbc", "#e64a19", "#b71c1c", "#4a0000"]
)

# ═════════════════════════════════════════════════════════════
# VIZ 1 — HEATMAP: КІЛЬКІСТЬ АТАК ПО РЕГІОНАХ × РОКАХ
# ═════════════════════════════════════════════════════════════
pivot_ry = (
    df_reg.groupby(["oblasts", "year"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=YEARS, fill_value=0)
)
pivot_ry = pivot_ry.loc[
    pivot_ry.sum(axis=1).sort_values(ascending=False).index
]

fig, ax = plt.subplots(figsize=(9, 9))
im = ax.imshow(pivot_ry.values, cmap=RED_CMAP, aspect="auto",
               interpolation="nearest")
plt.colorbar(im, ax=ax, label="Кількість атак", shrink=0.6, pad=0.02)

ax.set_xticks(range(len(YEARS)))
ax.set_xticklabels(YEAR_LABELS, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(pivot_ry)))
ax.set_yticklabels(pivot_ry.index, fontsize=10)
ax.set_title("Кількість атак по регіонах і роках (2023–2026)",
             fontweight="bold", fontsize=14, pad=14)
ax.set_xlabel("Рік", fontsize=11)
ax.set_ylabel("Регіон", fontsize=11)

for i in range(len(pivot_ry)):
    for j, yr in enumerate(YEARS):
        val = pivot_ry.values[i, j]
        if val > 0:
            col = "white" if val > pivot_ry.values.max() * 0.5 else "#222"
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    fontsize=9, color=col, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/r01_attacks_region_by_year.png", bbox_inches="tight")
plt.close()
print("[OK] r01_attacks_region_by_year.png")

# ═════════════════════════════════════════════════════════════
# VIZ 2 — HEATMAP: КІЛЬКІСТЬ АТАК ПО РЕГІОНАХ × МІСЯЦЯХ
# ═════════════════════════════════════════════════════════════
pivot_rm = (
    df_reg.groupby(["oblasts", "month"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=MONTHS, fill_value=0)
)
pivot_rm = pivot_rm.loc[pivot_ry.index]  # той самий порядок рядків

fig, ax = plt.subplots(figsize=(14, 9))
im = ax.imshow(pivot_rm.values, cmap=RED_CMAP, aspect="auto",
               interpolation="nearest")
plt.colorbar(im, ax=ax, label="Кількість атак", shrink=0.6, pad=0.02)

ax.set_xticks(range(12))
ax.set_xticklabels(MONTH_LABELS, fontsize=10, fontweight="bold")
ax.set_yticks(range(len(pivot_rm)))
ax.set_yticklabels(pivot_rm.index, fontsize=10)
ax.set_title("Кількість атак по регіонах і місяцях (2023–2026, всі роки разом)",
             fontweight="bold", fontsize=14, pad=14)
ax.set_xlabel("Місяць", fontsize=11)
ax.set_ylabel("Регіон", fontsize=11)

for i in range(len(pivot_rm)):
    for j in range(12):
        val = pivot_rm.values[i, j]
        if val > 0:
            col = "white" if val > pivot_rm.values.max() * 0.5 else "#222"
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    fontsize=8, color=col, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/r02_attacks_region_by_month.png", bbox_inches="tight")
plt.close()
print("[OK] r02_attacks_region_by_month.png")

# ═════════════════════════════════════════════════════════════
# VIZ 3 — BAR: КІЛЬКІСТЬ ЗАПУЩЕНИХ ПО РОКАХ (ВСЯ УКРАЇНА)
# ═════════════════════════════════════════════════════════════
launched_year = df.groupby("year")["launched"].sum().reindex(YEARS, fill_value=0)
attacks_year  = df.groupby("year").size().reindex(YEARS, fill_value=0)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#e53935" if y < 2026 else "#ff8f00" for y in YEARS]
bars = ax.bar(YEAR_LABELS, launched_year.values, color=colors,
              edgecolor="white", linewidth=0.8, width=0.55)

for bar, val in zip(bars, launched_year.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 800,
            f"{int(val):,}", ha="center", va="bottom", fontsize=11,
            fontweight="bold", color="#222")

ax.set_xlabel("Рік", fontsize=12)
ax.set_ylabel("Загальна кількість запущених", fontsize=12)
ax.set_title("Загальна кількість запущених снарядів/БПЛА по роках\n(вся Україна, 2023–2026)",
             fontweight="bold", fontsize=13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.set_ylim(0, launched_year.max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)

note = "* 2026 — неповний рік (дані до червня)"
ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="#888", style="italic")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/r03_launched_by_year.png", bbox_inches="tight")
plt.close()
print("[OK] r03_launched_by_year.png")

# ═════════════════════════════════════════════════════════════
# VIZ 4 — LINE: КІЛЬКІСТЬ ЗАПУЩЕНИХ ПО МІСЯЦЯХ (ЛІНІЇ ПО РОКАМ)
# ═════════════════════════════════════════════════════════════
launched_ym = (
    df.groupby(["year", "month"])["launched"]
    .sum()
    .unstack(level=0, fill_value=0)
    .reindex(MONTHS, fill_value=0)
)

YEAR_COLORS = {2023: "#1565C0", 2024: "#6A1B9A", 2025: "#BF360C", 2026: "#F9A825"}
YEAR_STYLES = {2023: "-o", 2024: "-s", 2025: "-^", 2026: "--D"}

fig, ax = plt.subplots(figsize=(14, 6))
for yr in YEARS:
    if yr not in launched_ym.columns:
        continue
    vals = launched_ym[yr].values
    valid_months = [m for m, v in zip(MONTHS, vals) if v > 0]
    valid_vals   = [v for v in vals if v > 0]
    style = YEAR_STYLES.get(yr, "-o")
    ax.plot(valid_months, valid_vals,
            style, label=str(yr), color=YEAR_COLORS.get(yr, "gray"),
            linewidth=2.2, markersize=7, markeredgecolor="white",
            markeredgewidth=1.2)

ax.set_xticks(MONTHS)
ax.set_xticklabels(MONTH_LABELS, fontsize=10)
ax.set_ylabel("Кількість запущених", fontsize=11)
ax.set_xlabel("Місяць", fontsize=11)
ax.set_title("Динаміка кількості запущених снарядів/БПЛА по місяцях і роках",
             fontweight="bold", fontsize=13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(title="Рік", fontsize=10, title_fontsize=10, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25, linestyle="--")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/r04_launched_by_month_per_year.png", bbox_inches="tight")
plt.close()
print("[OK] r04_launched_by_month_per_year.png")

# ═════════════════════════════════════════════════════════════
# VIZ 5 — SUMMARY: АТАКИ + ЗАПУЩЕНІ ПО РОКАХ (ВСЯ УКРАЇНА)
# ═════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(11, 6))
ax2 = ax1.twinx()

x = np.arange(len(YEARS))
w = 0.38

b1 = ax1.bar(x - w / 2, attacks_year.values, w,
             color="#1565C0", alpha=0.88, label="Кількість атак", zorder=3)
b2 = ax2.bar(x + w / 2, launched_year.values, w,
             color="#b71c1c", alpha=0.88, label="Запущено об'єктів", zorder=3)

for bar, val in zip(b1, attacks_year.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             f"{int(val):,}", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color="#1565C0")

for bar, val in zip(b2, launched_year.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
             f"{int(val):,}", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color="#b71c1c")

ax1.set_xticks(x)
ax1.set_xticklabels(YEAR_LABELS, fontsize=12, fontweight="bold")
ax1.set_ylabel("Кількість атак (записів)", fontsize=11, color="#1565C0")
ax2.set_ylabel("Кількість запущених об'єктів", fontsize=11, color="#b71c1c")
ax1.set_xlabel("Рік", fontsize=12)
ax1.set_title("Загальна картина: атаки і запущені об'єкти по роках\n(вся Україна, 2023–2026)",
              fontweight="bold", fontsize=14, pad=12)

ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.tick_params(axis="y", colors="#1565C0")
ax2.tick_params(axis="y", colors="#b71c1c")
ax1.spines[["top"]].set_visible(False)
ax2.spines[["top"]].set_visible(False)

lines = [b1, b2]
labels = ["Кількість атак", "Запущено об'єктів"]
ax1.legend(lines, labels, loc="upper left", fontsize=10, framealpha=0.9)

ax1.set_ylim(0, attacks_year.max() * 1.2)
ax2.set_ylim(0, launched_year.max() * 1.2)

ax1.grid(axis="y", alpha=0.2, linestyle="--", zorder=0)

note = "* 2026 — дані лише до червня"
ax1.text(0.99, 0.01, note, transform=ax1.transAxes, ha="right", va="bottom",
         fontsize=8, color="#888", style="italic")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/r05_overall_summary_by_year.png", bbox_inches="tight")
plt.close()
print("[OK] r05_overall_summary_by_year.png")

# ─────────────────────────────────────────────────────────────
# ТЕКСТОВИЙ ПІДСУМОК
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ПІДСУМКОВА СТАТИСТИКА")
print("=" * 55)
print(f"\nЗагальна кількість атак (записів) 2023-2026: {len(df):,}")
print(f"Всього запущено об'єктів:                     {int(df['launched'].sum()):,}")

print("\nПо роках:")
for yr in YEARS:
    a = attacks_year[yr]
    l = int(launched_year[yr])
    print(f"  {yr}: {a:>5} атак,  {l:>8,} запущених")

print(f"\nТоп-5 регіонів за кількістю атак (2023-2026):")
top5 = df_exploded.groupby("oblasts").size().sort_values(ascending=False).head(5)
for reg, cnt in top5.items():
    print(f"  {reg:<22} {cnt} атак")

print(f"\nМісяць з найбільшою кількістю запущених (за весь час):")
by_month_total = df.groupby("month")["launched"].sum()
peak_m = by_month_total.idxmax()
print(f"  {MONTH_LABELS[peak_m-1]}: {int(by_month_total[peak_m]):,} запущених")

print(f"\nВізуалізації збережено в {OUTPUT_DIR}/:")
for fn in ["r01_attacks_region_by_year.png",
           "r02_attacks_region_by_month.png",
           "r03_launched_by_year.png",
           "r04_launched_by_month_per_year.png",
           "r05_overall_summary_by_year.png"]:
    print(f"  {fn}")
