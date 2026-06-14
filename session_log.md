# Session Log — 2026-06-14

## Що зроблено

### 1. RSS-колектор (`collect_rss.py`)
Створено скрипт для збору новин з 7 RSS-стрічок.

**Активні джерела:**
| Джерело | RSS URL |
|---|---|
| Суспільне | https://suspilne.media/rss/all.rss |
| ТСН | https://tsn.ua/rss/full.rss |
| Українська правда | https://www.pravda.com.ua/rss/view_news/ |
| Bihus.info | https://bihus.info/feed |
| Гордон | https://gordonua.com/ukr/api/media/out/rss/lastnews.xml |
| УНІАН | https://rss.unian.net/site/news_ukr.rss |
| НВ | https://nv.ua/rss/all.xml |

**Результат:** 560 новин у `data/news.csv` (колонки: date, source, title, text)

**Перевірені URL, що не працюють:**
- Укрінформ (`ukrinform.ua`, `ukrinform.net`) — 404 на всіх шляхах
- Цензор.НЕТ (`censor.net/ua/feed`) — 403 Cloudflare
- Радіо Свобода (`radiosvoboda.org`) — повертає HTML замість RSS

---

### 2. Витяг сутностей (`extract_entities.py`)
Встановлено spaCy + модель `uk_core_news_sm`.

Скрипт витягує NER-сутності трьох категорій з колонок `title` + `text`:
- **PER** — персони
- **ORG** — організації  
- **LOC** — місця

Використовується **лематизація** (`ent.lemma_`) для об'єднання відмінкових форм.

**Результат:** топ-20 по кожній категорії у `data/entities.csv`

**Топ-10 після лематизації:**

| PER | count | ORG | count | LOC | count |
|---|---|---|---|---|---|
| трамп | 38 | зсу | 13 | україна | 110 |
| путін | 33 | чс-2026 | 12 | рф | 72 |
| зеленський | 20 | g7 | 10 | росія | 45 |
| дональд трамп | 17 | bihus | 10 | сша | 42 |
| володимир зеленський | 10 | ліга нація з волейбол | 7 | польща | 21 |
| іран | 8 | чм-2026 | 7 | крим | 16 |
| людмила лузан | 5 | нба | 5 | київ | 13 |
| володимир путін | 5 | сбу | 5 | львов | 12 |
| євген коновалець | 4 | білий дім | 5 | кремль | 11 |
| експерт | 3 | нато | 4 | чернівці | 10 |

---

## Що далі (ідеї)

- [ ] Нормалізація лем: `трамп` + `дональд трамп` → об'єднати в один кластер (coreference або fuzzy match)
- [ ] Відстеження сутностей у часі — які персони/місця з'являються частіше в конкретні дні
- [ ] Побудова графу зв'язків сутностей (co-occurrence) для GraphRAG
- [ ] Фільтрація шумових сутностей (`експерт`, `суспільному`, `іран` у PER)
- [ ] Додати збір з Telegram через `collect_telegram.py` і об'єднати з RSS

## Файли проекту

```
collect_rss.py        — збір новин з RSS
collect_telegram.py   — збір новин з Telegram (Telethon)
extract_entities.py   — NER + лематизація через spaCy
data/news.csv         — 560 новин (date, source, title, text)
data/entities.csv     — топ-20 сутностей по PER/ORG/LOC
```
