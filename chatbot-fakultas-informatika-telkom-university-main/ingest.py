"""
Script ingest: memproses semua dokumen di cleaned_file/ menjadi vector database.
Jalankan sekali sebelum menjalankan app.py.

Urutan langkah:
  1. Muat dan potong dokumen menjadi chunk    ← src/chunking.py
  2. Buat embedding dan simpan ke FAISS       ← src/embedding.py
"""

from src.chunking import load_documents, split_documents
from src.embedding import embed_and_store


def main():
    print("=== Memulai proses ingest dokumen ===\n")

    # ── Langkah 1: Muat dan potong dokumen ───────────────────────────────────
    documents = load_documents()
    chunks = split_documents(documents)

    # ── Langkah 2: Buat embedding dan simpan ke ChromaDB ─────────────────────
    embed_and_store(chunks)

    print("\n=== Ingest selesai! Vector database siap digunakan. ===")


if __name__ == "__main__":
    main()
