# Mengambil chunk yang paling relevan dari FAISS berdasarkan pertanyaan user

from src.embedding import load_vectorstore

# ── Konfigurasi ───────────────────────────────────────────────────────────────
TOP_K = 10   # jumlah chunk yang diambil dari FAISS
RETURN_K = 5 # jumlah chunk yang dikirim ke LLM


def get_relevant_documents(question: str) -> list:
    """Mengambil chunk paling relevan dari FAISS menggunakan L2 similarity.

    FAISS mengurutkan hasil dari yang paling mirip ke paling tidak mirip.
    Kita ambil TOP_K kandidat lalu potong ke RETURN_K teratas.
    """
    print("[retriever] Memuat vector database...")
    vectorstore = load_vectorstore()

    # similarity_search mengurutkan hasil dari paling mirip ke paling tidak mirip
    docs = vectorstore.similarity_search(question, k=TOP_K)

    result = docs[:RETURN_K]

    print(f"[retriever] Mengambil {len(result)} chunk teratas dari {TOP_K} kandidat.")
    for i, doc in enumerate(result):
        preview = doc.page_content.strip()[:70].replace("\n", " ")
        print(f"  [{i+1}] {doc.metadata.get('source')} | {preview}")

    return result


if __name__ == "__main__":
    # Testing manual: python -m src.retriever
    results = get_relevant_documents("apa itu TAK")
    for i, doc in enumerate(results):
        print(f"\n--- DOC {i+1} ({doc.metadata.get('source')}) ---")
        print(doc.page_content[:300])
