"""
Збір постів із публічних Telegram-каналів через Telethon MTProto API.
Зберігає результат у data/telegram_news.csv.

Запуск:
    python collect_telegram.py

Перший запуск попросить SMS-код для авторизації (одноразово).
Сесія зберігається у telegram_session.session — до .gitignore вже додано.
"""

import asyncio
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import ChannelPrivateError, UsernameNotOccupiedError
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument

load_dotenv()

API_ID   = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
PHONE    = os.getenv("TELEGRAM_PHONE")

if not API_ID or not API_HASH:
    print("Помилка: TELEGRAM_API_ID та TELEGRAM_API_HASH не знайдено у .env")
    print("Скопіюй .env.example -> .env і заповни ключі з https://my.telegram.org")
    sys.exit(1)

CHANNELS = [
    # Загальноукраїнські
    "suspilne.news",
    "ukrinform_ua",
    "tsn_ua",
    "pravda_com_ua",
    "kyivindependent",
    # Запорізькі
    "zap_info",        # Запоріжжя Інфо
    "5kanal_zp",       # 5 канал Запоріжжя
]

POSTS_PER_CHANNEL = 500
OUTPUT_PATH = "data/telegram_news.csv"


def has_media(msg) -> str:
    if msg.media is None:
        return "none"
    if isinstance(msg.media, MessageMediaPhoto):
        return "photo"
    if isinstance(msg.media, MessageMediaDocument):
        return "document"
    return "other"


async def collect_channel(client: TelegramClient, channel: str) -> list[dict]:
    records = []
    print(f"\n[{channel}] Починаю збір...")
    try:
        entity = await client.get_entity(channel)
        async for msg in client.iter_messages(entity, limit=POSTS_PER_CHANNEL):
            if msg.text is None and msg.media is None:
                continue
            records.append({
                "date":    msg.date.strftime("%Y-%m-%d %H:%M:%S"),
                "channel": channel,
                "post_id": msg.id,
                "text":    (msg.text or "").strip(),
                "views":   msg.views if msg.views else 0,
                "forwards": msg.forwards if msg.forwards else 0,
                "media":   has_media(msg),
                "url":     f"https://t.me/{channel}/{msg.id}",
            })
        print(f"[{channel}] Зібрано {len(records)} постів")
    except (ChannelPrivateError, UsernameNotOccupiedError) as e:
        print(f"[{channel}] Недоступний канал: {e}")
    except Exception as e:
        print(f"[{channel}] Помилка: {e}")
    return records


async def main():
    print("=" * 55)
    print("Telegram News Collector")
    print("=" * 55)
    print(f"Канали: {len(CHANNELS)}")
    print(f"Постів з каналу: {POSTS_PER_CHANNEL}")
    print(f"Вихідний файл: {OUTPUT_PATH}")

    os.makedirs("data", exist_ok=True)

    async with TelegramClient("telegram_session", int(API_ID), API_HASH) as client:
        if PHONE:
            await client.start(phone=PHONE)
        else:
            await client.start()

        all_records = []
        for channel in CHANNELS:
            records = await collect_channel(client, channel)
            all_records.extend(records)

    if not all_records:
        print("\nНічого не зібрано.")
        return

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", ascending=False, inplace=True)
    df.drop_duplicates(subset=["channel", "post_id"], inplace=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 55)
    print(f"Готово! Збережено {len(df)} постів у {OUTPUT_PATH}")
    print("\nПо каналах:")
    for ch, cnt in df.groupby("channel").size().sort_values(ascending=False).items():
        print(f"  {ch:<30} {cnt} постів")
    print(f"\nДіапазон дат: {df['date'].min().date()} — {df['date'].max().date()}")


if __name__ == "__main__":
    asyncio.run(main())
