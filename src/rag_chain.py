import re
from src.retriever import get_relevant_documents
from src.llm import get_llm

# FIX #1: Mapping dari nama file .md ke nama PDF aslinya di folder raw_file/
# Digunakan untuk menyertakan link download PDF pada sumber jawaban.
MD_TO_PDF_MAP = {
    "Panduan TA FIF.md":      "Panduan TA FIF.pdf",
    "Panduan_KP.md":          "Panduan KP.pdf",
    "Pedoman Proposal TA.md": "Panduan Proposal TA FIF.pdf",
    "Pedoman TAK.md":         "Pedoman TAK.pdf",
}

# Label tampilan untuk setiap dokumen (lebih ramah dibanding nama file)
MD_TO_LABEL_MAP = {
    "Panduan TA FIF.md":      "Panduan Tugas Akhir (TA) FIF",
    "Panduan_KP.md":          "Panduan Kerja Praktik (KP)",
    "Pedoman Proposal TA.md": "Pedoman Proposal Tugas Akhir",
    "Pedoman TAK.md":         "Pedoman TAK (Transkrip Aktivitas Kemahasiswaan)",
}


def build_prompt(question: str, documents: list) -> str:
    """Membuat prompt yang memadukan konteks retrieval, pertanyaan user, dan instruksi follow-up."""
    sources_text = []
    for index, doc in enumerate(documents, start=1):
        source_name = doc.metadata.get("source", "unknown")
        label = MD_TO_LABEL_MAP.get(source_name, source_name)
        sources_text.append(
            f"[{index}] Sumber: {label}\n{doc.page_content.strip()}"
        )

    context = "\n\n".join(sources_text)

    return (
        "Kamu adalah asisten akademik resmi Fakultas Informatika Telkom University.\n\n"

        "Tugasmu HANYA menjawab pertanyaan yang berkaitan dengan dokumen akademik berikut:\n"
        "- Panduan Tugas Akhir (TA)\n"
        "- Panduan Kerja Praktik (KP)\n"
        "- Pedoman TAK (Transkrip Aktivitas Kemahasiswaan)\n"
        "- Pedoman Proposal Tugas Akhir\n\n"

        "PENTING — ATURAN KETAT:\n"
        "Jika pertanyaan user TIDAK berkaitan langsung dengan KP, TA, TAK, atau Proposal TA, "
        "WAJIB jawab dengan kalimat ini persis:\n"
        "'Maaf, saya hanya dapat menjawab pertanyaan seputar KP, TA, TAK, dan Proposal TA. "
        "Untuk informasi lainnya, silakan hubungi pihak fakultas.'\n"
        "JANGAN mencoba menjawab pertanyaan di luar topik tersebut meskipun "
        "ada informasi yang tampak relevan dalam konteks.\n\n"

        "Jawab berdasarkan informasi dalam konteks di bawah ini. "
        "Kamu boleh menjelaskan dan menguraikan informasi dari konteks agar mudah dipahami, "
        "tapi jangan menambahkan fakta spesifik yang tidak ada dalam konteks.\n\n"

        "Gunakan format markdown yang rapi:\n"
        "- Gunakan bullet point untuk daftar\n"
        "- Gunakan paragraf pendek\n\n"

        "Jangan menyebut nama file.\n\n"

        "Setelah memberikan jawaban, tambahkan tepat 3 pertanyaan lanjutan yang relevan "
        "dengan format PERSIS seperti ini (jangan tambahkan teks lain setelah ini):\n"
        "PERTANYAAN_LANJUTAN:\n"
        "1. [pertanyaan 1]\n"
        "2. [pertanyaan 2]\n"
        "3. [pertanyaan 3]\n\n"

        f"Konteks:\n{context}\n\n"

        f"Pertanyaan User:\n{question}\n\n"

        "Jawaban:"
    )


def parse_response(raw: str) -> tuple[str, list[str]]:
    """Memisahkan jawaban utama dan daftar pertanyaan lanjutan dari respons LLM."""
    if "PERTANYAAN_LANJUTAN:" not in raw:
        return raw.strip(), []

    parts = raw.split("PERTANYAAN_LANJUTAN:", 1)
    answer = parts[0].strip()
    follow_up_block = parts[1].strip()

    follow_ups = []
    for line in follow_up_block.splitlines():
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
        if cleaned and len(cleaned) > 5:
            follow_ups.append(cleaned)

    return answer, follow_ups[:3]


def ask_question(question: str):
    """Menjalankan pipeline RAG dan mengembalikan jawaban, sumber (+ PDF link), dan pertanyaan lanjutan."""
    llm = get_llm()

    documents = get_relevant_documents(question)

    # FIX BUG #2: jika tidak ada dokumen yang relevan (semua di bawah threshold),
    # langsung kembalikan pesan tidak tersedia tanpa memanggil LLM
    if not documents:
        return (
            "Informasi tersebut tidak tersedia dalam dokumen yang ada. "
            "Silakan hubungi pihak fakultas atau kunjungi website resmi Telkom University.",
            [],
            []
        )

    prompt = build_prompt(question, documents)
    response = llm.invoke(prompt)

    raw_text = response.content if hasattr(response, "content") else response
    raw_text = raw_text.replace("### ###", "###").replace("## ##", "##")

    answer_text, follow_ups = parse_response(raw_text)

    # FIX #1: Kumpulkan semua sumber unik dan sertakan info PDF
    seen_sources = set()
    sources = []
    for doc in documents:
        md_name = doc.metadata.get("source", "unknown")
        if md_name in seen_sources:
            continue
        seen_sources.add(md_name)
        top_doc = documents[0]
        md_name = top_doc.metadata.get("source", "unknown")
        pdf_name = MD_TO_PDF_MAP.get(md_name)
        label = MD_TO_LABEL_MAP.get(md_name, md_name)

        sources = [{
            "name": label,
            "snippet": top_doc.page_content.strip(),
            "pdf_path": f"raw_file/{pdf_name}" if pdf_name else None,
            "pdf_name": pdf_name,
        }]

    return answer_text, sources, follow_ups
