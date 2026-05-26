import re
from src.retriever import get_relevant_documents
from src.llm import get_llm


def build_prompt(question: str, documents: list) -> str:
    """Membuat prompt yang memadukan konteks retrieval, pertanyaan user, dan instruksi follow-up."""
    sources_text = []
    for index, doc in enumerate(documents, start=1):
        source_name = doc.metadata.get("source", "unknown")
        sources_text.append(
            f"[{index}] Sumber: {source_name}\n{doc.page_content.strip()}"
        )

    context = "\n\n".join(sources_text)

    return (
        "Kamu adalah asisten akademik Telkom University.\n\n"

        "Jawab pertanyaan mahasiswa dengan bahasa yang natural, jelas, dan profesional.\n\n"

        "Jawab berdasarkan informasi dalam konteks di bawah ini. "
        "Kamu boleh menjelaskan dan menguraikan informasi dari konteks agar mudah dipahami, "
        "tapi jangan menambahkan fakta spesifik yang tidak ada dalam konteks.\n\n"

        "Jika konteks tidak memuat informasi yang relevan sama sekali, "
        "katakan dengan jujur: 'Informasi tersebut tidak tersedia dalam dokumen yang ada.'\n\n"

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
    """Menjalankan pipeline RAG dan mengembalikan jawaban, sumber, dan pertanyaan lanjutan."""
    llm = get_llm()

    documents = get_relevant_documents(question)

    if not documents:
        return "Maaf, tidak ditemukan konteks yang relevan.", [], []

    prompt = build_prompt(question, documents)
    response = llm.invoke(prompt)

    raw_text = response.content if hasattr(response, "content") else response
    raw_text = raw_text.replace("### ###", "###").replace("## ##", "##")

    answer_text, follow_ups = parse_response(raw_text)

    top_doc = documents[0]
    sources = [{
        "name": top_doc.metadata.get("source", "unknown"),
        "snippet": top_doc.page_content.strip()
    }]

    return answer_text, sources, follow_ups
