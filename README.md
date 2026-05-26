# Chatbot Akademik RAG — Fakultas Informatika Telkom University

Chatbot berbasis **Retrieval-Augmented Generation (RAG)** yang menjawab pertanyaan seputar prosedur akademik Fakultas Informatika Telkom University. Dibangun sebagai tugas capstone.

![Pipeline](assets/pipeline_chatbot.png)

---

## Fitur

- Menjawab pertanyaan berdasarkan dokumen resmi akademik (KP, TA, Proposal TA, TAK)
- Jawaban akurat karena bersumber dari dokumen, bukan pengetahuan umum model
- Antarmuka web yang mudah digunakan via Streamlit

---

## Dokumen Sumber

| Dokumen | Isi |
|---|---|
| Panduan KP | Prosedur dan syarat Kerja Praktik |
| Panduan TA FIF | Panduan pelaksanaan Tugas Akhir |
| Pedoman Proposal TA | Format dan syarat proposal TA |
| Pedoman TAK | Pedoman Tugas Akhir Komprehensif |

---

## Tech Stack

| Komponen | Teknologi |
|---|---|
| Embedding | `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2) |
| Vector Database | ChromaDB |
| LLM | Google Gemini API (`gemini-1.5-flash`) |
| Orchestration | LangChain |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Struktur Proyek

```
chatbot-project/
├── raw_file/               # PDF dokumen asli
├── cleaned_file/           # Markdown hasil cleaning dari PDF
├── data/
│   └── chroma_db/          # Vector database (dibuat oleh ingest.py)
├── src/
│   ├── chunking.py         # Memotong dokumen menjadi chunk
│   ├── embedding.py        # Membuat embedding dan menyimpan ke ChromaDB
│   ├── retriever.py        # Mengambil chunk relevan saat ada query
│   ├── llm.py              # Koneksi ke Gemini API
│   └── rag_chain.py        # Pipeline RAG lengkap
├── assets/                 # Gambar dan aset lainnya
├── app.py                  # Aplikasi Streamlit
├── ingest.py               # Script pemrosesan dokumen (jalankan sekali)
├── requirements.txt
├── .env.example
└── CLAUDE.md               # Panduan pengembangan untuk AI assistant
```

---

## Setup Lokal

### Prasyarat
- Python 3.10+
- Akun Google (untuk Gemini API key gratis)

### Langkah Instalasi

**1. Clone repository**
```bash
git clone https://github.com/username/chatbot-project.git
cd chatbot-project
```

**2. Buat virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependensi**
```bash
pip install -r requirements.txt
```

**4. Konfigurasi API key**
```bash
cp .env.example .env
```
Buka file `.env` dan isi `GEMINI_API_KEY` dengan key dari [Google AI Studio](https://aistudio.google.com/).

**5. Proses dokumen ke vector database**
```bash
python ingest.py
```
> Cukup dijalankan sekali. Ulangi hanya jika dokumen di `cleaned_file/` berubah.

**6. Jalankan aplikasi**
```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## Deployment

Aplikasi ini dapat di-deploy gratis ke **Streamlit Community Cloud**:

1. Push repository ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io) dan hubungkan repo
3. Tambahkan `GEMINI_API_KEY` di bagian **Secrets** pada dashboard
4. Pastikan folder `data/chroma_db/` ikut ter-commit ke repository

---

## Lisensi

Proyek ini dibuat untuk keperluan akademik — Tugas Capstone Fakultas Informatika Telkom University.
