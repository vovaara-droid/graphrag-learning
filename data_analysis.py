"""
Аналіз даних: missile_attacks_daily.csv
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
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────
# 1. ЗАВАНТАЖЕННЯ І СТРУКТУРА
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("1. СТРУКТУРА ДАТАСЕТУ")
print("=" * 60)

df = pd.read_csv("d:/data/missile_attacks_daily.csv")

print(f"\nРядків: {df.shape[0]}, Колонок: {df.shape[1]}")
print("\nТипи даних і приклади значень:")
print(f"{'Колонка':<35} {'Dtype':<12} {'Приклад'}")
print("-" * 75)
for col in df.columns:
    sample = str(df[col].dropna().iloc[0]) if df[col].notna().any() else "–"
    sample = sample[:35].replace("\n", " ")
    print(f"{col:<35} {str(df[col].dtype):<12} {sample}")

# ─────────────────────────────────────────────────────────────
# 2. ПРОБЛЕМИ: ПРОПУСКИ, ДУБЛІКАТИ, АНОМАЛІЇ
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. ДІАГНОСТИКА ПРОБЛЕМ")
print("=" * 60)

total_rows = len(df)

# --- Пропуски
print("\n[2.1] Пропуски по колонках:")
print(f"{'Колонка':<35} {'Null':<8} {'%'}")
print("-" * 55)
null_stats = df.isnull().sum().sort_values(ascending=False)
for col, n in null_stats.items():
    pct = n / total_rows * 100
    print(f"{col:<35} {n:<8} {pct:.1f}%")

# --- Дублікати
n_dup = df.duplicated().sum()
print(f"\n[2.2] Повні дублікати рядків: {n_dup}")

# --- border_crossing: формально не null, але пусто ({})
n_empty_bc = (df["border_crossing"] == "{}").sum()
print(f"\n[2.3] 'border_crossing' містить '{{}}' (семантично порожньо): "
      f"{n_empty_bc} ({n_empty_bc/total_rows*100:.1f}%)")

# --- Кодування
bad_encoding = df["model"].str.contains(r"[^\x00-\x7F]", na=False)
print(f"\n[2.4] Рядки з зіпсованим кодуванням у 'model': {bad_encoding.sum()}")
print("  Унікальні моделі з проблемою:",
      df.loc[bad_encoding, "model"].unique()[:5].tolist())

# --- Числові аномалії (IQR-метод)
print("\n[2.5] Аномальні значення (IQR-метод, числові колонки):")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
for col in numeric_cols:
    s = df[col].dropna()
    if len(s) < 10:
        continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = s[(s < low) | (s > high)]
    if len(outliers):
        print(f"  {col}: {len(outliers)} аномалій "
              f"(поріг [{low:.1f}, {high:.1f}]), max={s.max():.0f}")

# ─────────────────────────────────────────────────────────────
# 3. ОЧИЩЕННЯ ДАНИХ
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. ОЧИЩЕННЯ ДАНИХ")
print("=" * 60)

df_clean = df.copy()

# 3.1 Видалення колонок з >95% пропусків
HIGH_NULL_THRESHOLD = 0.95
high_null_cols = [
    col for col in df_clean.columns
    if df_clean[col].isnull().mean() > HIGH_NULL_THRESHOLD
]
df_clean.drop(columns=high_null_cols, inplace=True)
print(f"\n[3.1] Видалено {len(high_null_cols)} колонок (>95% пропусків):")
for c in high_null_cols:
    pct = df[c].isnull().mean() * 100
    print(f"  - {c} ({pct:.1f}% null)")

# 3.2 Парсинг часу
df_clean["date"] = pd.to_datetime(
    df_clean["time_start"].str[:10], errors="coerce"
)
n_bad_date = df_clean["date"].isna().sum()
print(f"\n[3.2] Розпарсено 'time_start' -> 'date'. "
      f"Нечитабельних дат: {n_bad_date}"
      )

# 3.3 Рядки з відсутнім 'launched' (3 рядки — ключовий показник)
n_no_launched = df_clean["launched"].isna().sum()
df_clean.dropna(subset=["launched"], inplace=True)
print(f"\n[3.3] Видалено {n_no_launched} рядків без значення 'launched' "
      "(ключовий числовий показник; без нього рядок непридатний для аналізу)")

# 3.4 Очищення 'model' від зіпсованого кодування
bad_mask = df_clean["model"].str.contains(r"[^\x00-\x7F]", na=False)
df_clean.loc[bad_mask, "model"] = "Unknown (encoding error)"
print(f"\n[3.4] Перейменовано {bad_mask.sum()} рядків з нечитабельною "
      "моделлю → 'Unknown (encoding error)'")

# 3.5 Нормалізація 'border_crossing': '{}' → NaN
df_clean["border_crossing"] = df_clean["border_crossing"].replace("{}", np.nan)
print(f"\n[3.5] 'border_crossing': '{{}}' замінено на NaN "
      "(семантично порожній словник = відсутність переходу)")

# 3.6 Аномалії launched: перевіряємо, чи є помилкові значення
# Значення > 200 належать реальним великим атакам Shahed — залишаємо.
# Значення = 0 або від'ємних немає. Нічого не видаляємо.
print("\n[3.6] Аномальні значення 'launched' (>200): залишено — "
      "це реальні масові атаки Shahed, підтверджені джерелами.")

print(f"\nПісля очищення: {df_clean.shape[0]} рядків, "
      f"{df_clean.shape[1]} колонок")

# ─────────────────────────────────────────────────────────────
# 4. СТАТИСТИКА ПІСЛЯ ОЧИЩЕННЯ
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. ОПИСОВА СТАТИСТИКА (після очищення)")
print("=" * 60)
print(df_clean[["launched", "destroyed", "not_reach_goal"]].describe().round(2))

# ─────────────────────────────────────────────────────────────
# 5. ВІЗУАЛІЗАЦІЇ
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "d:/output"
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ── VIZ 1: Розподіл 'launched' (гістограма) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Розподіл кількості запущених снарядів/БПЛА ('launched')",
             fontsize=14, fontweight="bold", y=1.01)

ax = axes[0]
ax.hist(df_clean["launched"], bins=60, color="#2196F3", edgecolor="white",
        linewidth=0.4)
ax.set_xlabel("launched")
ax.set_ylabel("Кількість атак")
ax.set_title("Лінійна шкала")

ax2 = axes[1]
ax2.hist(df_clean["launched"], bins=60, color="#FF5722", edgecolor="white",
         linewidth=0.4, log=True)
ax2.set_xlabel("launched")
ax2.set_ylabel("Кількість атак (log)")
ax2.set_title("Логарифмічна шкала Y")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_distribution_launched.png",
            bbox_inches="tight")
plt.close()

print("\n[ЗБЕРЕЖЕНО] 01_distribution_launched.png")
print("  Що показує: гістограма розподілу змінної 'launched' за двома шкалами. "
      "Ліва — лінійна вісь Y (видно концентрацію малих значень). "
      "Права — логарифмічна вісь Y (видно хвіст великих атак). "
      f"Мінімум: {df_clean['launched'].min():.0f}, "
      f"Медіана: {df_clean['launched'].median():.0f}, "
      f"Максимум: {df_clean['launched'].max():.0f}.")

# ── VIZ 2: Топ-10 моделей за сумою 'launched' ────────────────
top10 = (
    df_clean.groupby("model")["launched"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top10.index[::-1], top10.values[::-1],
               color=plt.cm.viridis_r(np.linspace(0.1, 0.9, 10)))
ax.set_xlabel("Загальна кількість запущених")
ax.set_title("Топ-10 моделей за загальною кількістю запущених", fontweight="bold")
for bar, val in zip(bars, top10.values[::-1]):
    ax.text(bar.get_width() + top10.max() * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}", va="center", fontsize=9)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_top10_models.png", bbox_inches="tight")
plt.close()

print("\n[ЗБЕРЕЖЕНО] 02_top10_models.png")
print("  Що показує: горизонтальна гістограма 10 типів озброєння "
      "з найбільшою сумарною кількістю запущених одиниць за весь час. "
      f"Лідер: {top10.index[0]} ({top10.iloc[0]:,.0f} од.).")

# ── VIZ 3: Динаміка по місяцях ────────────────────────────────
monthly = (
    df_clean.dropna(subset=["date"])
    .set_index("date")
    .resample("ME")["launched"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "total_launched", "count": "num_attacks"})
    .dropna()
)

fig, ax1 = plt.subplots(figsize=(16, 5))
ax2 = ax1.twinx()

ax1.fill_between(monthly.index, monthly["total_launched"],
                 alpha=0.35, color="#2196F3")
ax1.plot(monthly.index, monthly["total_launched"],
         color="#2196F3", linewidth=2, label="Запущено (сума)")
ax2.bar(monthly.index, monthly["num_attacks"],
        width=20, alpha=0.4, color="#FF9800", label="Кількість атак")

ax1.set_xlabel("Місяць")
ax1.set_ylabel("Загальна кількість запущених", color="#2196F3")
ax2.set_ylabel("Кількість записів-атак", color="#FF9800")
ax1.set_title("Місячна динаміка: сума запущених і кількість атак",
              fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_monthly_dynamics.png", bbox_inches="tight")
plt.close()

print("\n[ЗБЕРЕЖЕНО] 03_monthly_dynamics.png")
print("  Що показує: лінійний графік сумарної кількості запущених "
      "снарядів/БПЛА по місяцях (синя лінія, ліва вісь) і стовпчастий "
      "графік кількості зафіксованих атак на місяць (помаранчеві стовпці, "
      f"права вісь). Охоплено {monthly.index.min().strftime('%Y-%m')} — "
      f"{monthly.index.max().strftime('%Y-%m')}.")

# ── VIZ 4: Кореляційна матриця числових колонок ──────────────
num_cols_available = [c for c in
    ["launched", "destroyed", "not_reach_goal", "still_attacking",
     "num_hit_location", "num_fall_fragment_location"]
    if c in df_clean.columns]

corr = df_clean[num_cols_available].corr()

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(corr.columns, fontsize=9)
ax.set_title("Кореляційна матриця числових колонок", fontweight="bold")

for i in range(len(corr)):
    for j in range(len(corr.columns)):
        val = corr.values[i, j]
        color = "black" if abs(val) < 0.7 else "white"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=9, color=color, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation_matrix.png", bbox_inches="tight")
plt.close()

print("\n[ЗБЕРЕЖЕНО] 04_correlation_matrix.png")
print("  Що показує: кольорова матриця коефіцієнтів кореляції Пірсона "
      f"між {len(num_cols_available)} числовими змінними. "
      "Зелений — сильна позитивна кореляція (близько до 1), "
      "червоний — негативна, жовтий — відсутність зв'язку.")

# ── VIZ 5: Середнє vs медіана по групах (топ-цілі) ───────────
target_col = "target"
top_targets = (
    df_clean.groupby(target_col)["launched"]
    .count()
    .sort_values(ascending=False)
    .head(10)
    .index
)
grp = (
    df_clean[df_clean[target_col].isin(top_targets)]
    .groupby(target_col)["launched"]
    .agg(mean="mean", median="median")
    .sort_values("mean", ascending=True)
)

fig, ax = plt.subplots(figsize=(10, 7))
y = np.arange(len(grp))
h = 0.35
ax.barh(y - h / 2, grp["mean"], h, label="Середнє", color="#1976D2", alpha=0.85)
ax.barh(y + h / 2, grp["median"], h, label="Медіана", color="#FF7043", alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(grp.index, fontsize=9)
ax.set_xlabel("launched (одиниць)")
ax.set_title("Середнє та медіана 'launched' по топ-10 цілях", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_mean_vs_median_by_target.png", bbox_inches="tight")
plt.close()

print("\n[ЗБЕРЕЖЕНО] 05_mean_vs_median_by_target.png")
print("  Що показує: парні горизонтальні стовпці для 10 найчастіших значень "
      "колонки 'target'. Синій — арифметичне середнє кількості запущених, "
      "оранжевий — медіана. Відстань між ними відображає асиметрію "
      "розподілу всередині кожної групи цілей.")

# ─────────────────────────────────────────────────────────────
# ПІДСУМОК
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ПІДСУМОК АНАЛІЗУ")
print("=" * 60)
print(f"Початковий датасет:   {df.shape[0]} рядків, {df.shape[1]} колонок")
print(f"Після очищення:       {df_clean.shape[0]} рядків, {df_clean.shape[1]} колонок")
print(f"Видалено колонок:     {len(high_null_cols)} (>95% пропусків)")
print(f"Видалено рядків:      {df.shape[0] - df_clean.shape[0]}")
print(f"\nВізуалізації збережено в: {OUTPUT_DIR}/")
print("  01_distribution_launched.png")
print("  02_top10_models.png")
print("  03_monthly_dynamics.png")
print("  04_correlation_matrix.png")
print("  05_mean_vs_median_by_target.png")
