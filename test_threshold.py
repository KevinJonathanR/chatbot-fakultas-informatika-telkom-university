from src.embedding import load_vectorstore

THRESHOLD = 12.0
vs = load_vectorstore()

queries = [
    "Apa itu TAK",
    "bagaimana cara daftar mahasiswa baru",
    "syarat mengambil KP",
]

for q in queries:
    results = vs.similarity_search_with_score(q, k=3)
    print(f"\nQuery: '{q}'")
    for doc, score in results:
        status = "✓ LOLOS" if score <= THRESHOLD else "✗ DIBUANG"
        print(f"  {status} score={score:.4f} | {doc.metadata.get('source')}")