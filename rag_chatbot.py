"""
RAG-чатбот над новинами з data/news.csv
LlamaIndex + ChromaDB + sentence-transformers + Claude Haiku
"""

import os
import sys
import pandas as pd
import chromadb
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

# ── Конфігурація ──────────────────────────────────────────────────────────────

DATA_PATH = "data/news.csv"
CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION = "news"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "claude-haiku-4-5-20251001"
TOP_K = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

SYSTEM_PROMPT = (
    "Відповідай тільки на основі наданого контексту. "
    "Якщо відповіді немає в контексті — скажи "
    "'У наявних новинах немає інформації про це'. "
    "Наприкінці вкажи джерела з яких взята інформація."
)

# ── Ініціалізація моделей ────────────────────────────────────────────────────

def init_settings():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Помилка: встановіть змінну середовища ANTHROPIC_API_KEY")

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Anthropic(
        model=LLM_MODEL,
        api_key=api_key,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=1024,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

# ── Завантаження документів ──────────────────────────────────────────────────

def load_documents() -> list[Document]:
    df = pd.read_csv(DATA_PATH)
    documents = []
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()
        combined = f"{title}\n\n{text}" if title else text
        doc = Document(
            text=combined,
            metadata={
                "source": str(row.get("source", "")),
                "date": str(row.get("date", "")),
                "title": title,
            },
        )
        documents.append(doc)
    print(f"Завантажено {len(documents)} новин")
    return documents

# ── Побудова / завантаження індексу ─────────────────────────────────────────

def build_index() -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    # Якщо колекція вже заповнена — просто підключаємось
    if collection.count() > 0:
        print(f"Знайдено існуючий індекс ({collection.count()} фрагментів). Пропускаю індексацію.")
        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_ctx
        )

    print("Індексую документи… (перший запуск займе кілька хвилин)")
    documents = load_documents()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_ctx,
        show_progress=True,
    )
    print(f"Індексацію завершено. Збережено до {CHROMA_DIR}/")
    return index

# ── Інтерактивний чат ────────────────────────────────────────────────────────

def run_chat(index: VectorStoreIndex):
    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        response_mode="compact",
    )

    print("\n" + "=" * 60)
    print("RAG-чатбот над новинами готовий.")
    print("Введіть питання або 'вихід' для завершення.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Ви: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо побачення!")
            break

        if not question:
            continue
        if question.lower() in {"вихід", "exit", "quit"}:
            print("До побачення!")
            break

        response = query_engine.query(question)
        print(f"\nБот: {response}\n")

# ── Точка входу ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_settings()
    index = build_index()
    run_chat(index)
