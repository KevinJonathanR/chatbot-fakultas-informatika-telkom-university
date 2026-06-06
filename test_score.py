from src.embedding import load_vectorstore

vs = load_vectorstore()
results = vs.similarity_search_with_score('Apa itu TAK', k=5)
for doc, score in results:
    print(f"score={score:.4f} | {doc.metadata.get('source')}")