from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ── Konfigurasi ───────────────────────────────────────────────────────────────
CLEANED_FILE_DIR = Path("cleaned_file")
CHUNK_SIZE = 500    # jumlah karakter per chunk
CHUNK_OVERLAP = 100  # karakter yang tumpang tindih antar chunk agar konteks tidak terputus


def load_documents() -> list[Document]:
    """Membaca semua file .md dari folder cleaned_file."""
    documents = []

    for file_path in sorted(CLEANED_FILE_DIR.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")

        # metadata "source" dipakai nanti untuk menampilkan asal dokumen di jawaban chatbot
        doc = Document(
            page_content=text,
            metadata={"source": file_path.name}
        )
        documents.append(doc)

    print(f"[chunking] {len(documents)} dokumen dimuat: {[d.metadata['source'] for d in documents]}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Memotong setiap dokumen menjadi chunk-chunk kecil."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # urutan pemisah: coba potong di paragraf dulu, lalu baris, lalu kalimat
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    filtered_chunks = []

    for chunk in chunks:

        content = chunk.page_content.strip()

        # Skip chunk kosong / terlalu pendek
        if len(content) < 80:
            continue

        # Skip chunk yang cuma simbol atau karakter markdown
        if content.replace("-", "").replace(".", "").replace("#", "").strip() == "":
            continue

        # Skip chunk yang hanya berisi satu baris pendek (biasanya cuma heading)
        lines = [l for l in content.splitlines() if l.strip()]
        if len(lines) == 1 and len(content) < 120:
            continue

        # Tambahkan nama dokumen sebagai prefix teks chunk.
        # Tujuan: query tentang "TAK" akan lebih mudah cocok ke chunk dari "Pedoman TAK.md"
        # karena setiap chunk-nya diawali dengan "[Pedoman TAK]".
        source = chunk.metadata.get("source", "")
        doc_label = source.replace(".md", "").replace("_", " ")
        chunk.page_content = f"[{doc_label}]\n{content}"

        filtered_chunks.append(chunk)

    print(f"[chunking] Chunk awal: {len(chunks)}")
    print(f"[chunking] Chunk setelah filter: {len(filtered_chunks)}")

    return filtered_chunks

