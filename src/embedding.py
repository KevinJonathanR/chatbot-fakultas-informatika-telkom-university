# Mengubah teks chunk menjadi vektor angka dan menyimpannya ke FAISS

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Konfigurasi
FAISS_DB_DIR = str(Path("data/faiss_db"))
# Model dipilih karena cepat & ringan, bisa diganti jika akurasi dinilai rendah
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Menyiapkan model yang akan digunakan
def get_embedding_function() -> HuggingFaceEmbeddings:
    print(f"[embedding] Memuat model embedding: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Teks to Vector
def embed_and_store(chunks: list[Document]) -> FAISS:

    embedding_fn = get_embedding_function()

    print(f"[embedding] Membuat embedding untuk {len(chunks)} chunk dan menyimpan ke FAISS...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_fn,
    )

    # Simpan database FAISS ke folder
    vectorstore.save_local(FAISS_DB_DIR)

    print(f"[embedding] Selesai. {len(chunks)} chunk tersimpan di '{FAISS_DB_DIR}'.")

    return vectorstore

# Memanggil Vector database (dipakai saat query bukan ingest)
def load_vectorstore() -> FAISS:

    embedding_fn = get_embedding_function()

    vectorstore = FAISS.load_local(
        FAISS_DB_DIR,
        embedding_fn,
        allow_dangerous_deserialization=True
    )

    return vectorstore