# Mengambil chunk yang paling relevan dari FAISS berdasarkan pertanyaan user

from src.embedding import load_vectorstore

# ── Konfigurasi ───────────────────────────────────────────────────────────────
TOP_K = 10    # jumlah chunk yang diambil dari FAISS
RETURN_K = 10  # jumlah chunk yang dikirim ke LLM

# FIX BUG #2: Threshold similarity score (L2 distance — makin kecil makin relevan).
# Chunk dengan jarak L2 > ambang ini dianggap tidak relevan dan dibuang.
# FAISS pakai L2 distance: nilai 0 = identik, makin besar = makin jauh/tidak relevan.
# Nilai 1.2 dipilih empiris untuk multilingual-MiniLM; turunkan jika masih hallusinasi.
SIMILARITY_THRESHOLD = 18.0


def get_relevant_documents(question: str) -> list:
    """Mengambil chunk paling relevan dari FAISS menggunakan L2 similarity.

    Mengembalikan list kosong jika tidak ada chunk yang melewati threshold,
    sehingga chatbot akan menjawab 'informasi tidak tersedia' bukan hallusinasi.
    """
    print("[retriever] Memuat vector database...")
    vectorstore = load_vectorstore()

    # Gunakan similarity_search_with_score agar bisa filter berdasarkan skor
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=TOP_K)

    print(f"[retriever] Skor similarity (L2) untuk setiap kandidat:")
    for i, (doc, score) in enumerate(docs_with_scores):
        preview = doc.page_content.strip()[:70].replace("\n", " ")
        print(f"  [{i+1}] score={score:.4f} | {doc.metadata.get('source')} | {preview}")

    # FIX: filter chunk yang skornya melebihi threshold (tidak relevan)
    filtered = [
        (doc, score) for doc, score in docs_with_scores
        if score <= SIMILARITY_THRESHOLD
    ]

    if not filtered:
        print(f"[retriever] Tidak ada chunk yang lolos threshold ({SIMILARITY_THRESHOLD}). Pertanyaan di luar konteks dokumen.")
        return []

    result = [doc for doc, _ in filtered[:RETURN_K]]

    print(f"[retriever] Mengambil {len(result)} chunk setelah filter threshold.")
    return result


if __name__ == "__main__":
    # Testing manual: python -m src.retriever
    results = get_relevant_documents("apa itu TAK")
    for i, doc in enumerate(results):
        print(f"\n--- DOC {i+1} ({doc.metadata.get('source')}) ---")
        print(doc.page_content[:300])
