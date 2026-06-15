"""
evaluate.py — Evaluasi RAGAS untuk PANDAI Chat
Jalankan: python evaluate.py

Langkah sebelum menjalankan:
1. pip install ragas langchain-google-genai datasets
2. Isi setiap "ground_truth" di TEST_DATASET dengan jawaban dari dokumen asli
   (buka file di cleaned_file/ sebagai referensi)
3. python evaluate.py
"""

import os
import csv
import json
import time
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, context_precision, context_recall  # noqa: E402
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper

from src.rag_chain import ask_question
from src.retriever import get_relevant_documents

load_dotenv()

CHECKPOINT_FILE = "eval_checkpoint.json"
RAGAS_CHECKPOINT_FILE = "eval_ragas_checkpoint.json"

# ── Dataset Evaluasi ──────────────────────────────────────────────────────────
# Isi ground_truth dengan jawaban referensi yang diambil langsung dari dokumen
# di folder cleaned_file/. Semakin akurat ground_truth, semakin valid hasilnya.

TEST_DATASET = [
    # ── Panduan Kerja Praktik (KP) ───────────────────────────────────────────
    {
        "question": "Apa saja syarat untuk mahasiswa mendaftar Kerja Praktik?",
        "ground_truth": (
            "Syarat untuk mengikuti Mata Kuliah KP adalah: "
            "(1) Peserta adalah mahasiswa aktif program sarjana Fakultas Informatika yang telah menyelesaikan "
            "semua mata kuliah tahun pertama dengan IP ≥ 2,00 dan harus sudah dinyatakan lulus Tingkat I/TPB. "
            "(2) Kerja Praktik dapat dilaksanakan paling awal sesudah berakhirnya Semester IV."
        ),
    },
    {
        "question": "Berapa lama durasi pelaksanaan Kerja Praktik?",
        "ground_truth": (
            "KP diberikan bobot 3 SKS yang setara dengan 144 jam dalam satu semester. "
            "Apabila dikonversikan ke hari kerja (1 hari kerja = 6 jam), maka durasi minimum KP adalah "
            "144 jam ÷ 6 jam/hari = 24 hari kerja. KP minimal dilaksanakan selama 24 hari kerja."
        ),
    },
    {
        "question": "Dokumen apa saja yang harus dikumpulkan setelah KP selesai?",
        "ground_truth": (
            "Berkas yang harus diserahkan mahasiswa kepada Pembimbing Akademik setelah KP selesai adalah: "
            "(1) Form KP-02 (berita acara bimbingan), "
            "(2) BLP (Buku Laporan Pelaksanaan) KP yang telah ditandatangani Pembimbing Lapangan dan distempel instansi, "
            "(3) Form KP-03B dalam amplop tertutup, "
            "(4) Surat Keterangan Selesai KP yang disahkan tanda tangan dan cap dari instansi."
        ),
    },
    {
        "question": "Bagaimana prosedur pengajuan tempat Kerja Praktik?",
        "ground_truth": (
            "Prosedur pengajuan surat permohonan KP: "
            "(1) Mahasiswa mengisi Form KP-01 dengan melampirkan fotokopi KTM dan KSM. "
            "(2) Menyerahkan Form KP-01 beserta profil instansi dan CV ke Petugas Administrasi KP. "
            "(3) Petugas menyiapkan surat pengantar maksimal 3 hari setelah berkas diterima. "
            "(4) Mahasiswa mengirim surat pengantar ke perusahaan menggunakan amplop resmi institusi. "
            "(5) Mahasiswa memantau surat balasan di administrasi KP. "
            "(6) Mahasiswa dapat melaksanakan KP jika telah memperoleh surat balasan penerimaan dari instansi. "
            "Jika ditolak atau tidak ada jawaban dalam 1 bulan, mahasiswa dapat mengulang proses dari langkah ke-3."
        ),
    },
    {
        "question": "Siapa yang menilai laporan Kerja Praktik?",
        "ground_truth": (
            "Laporan KP dinilai oleh dua pihak: Pembimbing Lapangan (dari instansi tempat KP) dan Pembimbing Akademik (dosen FIF). "
            "Bobot penilaian berbeda per prodi: "
            "Untuk S1 Informatika dan S1 Sains Data: Pembimbing Lapangan 40%, Pembimbing Akademik 60%. "
            "Untuk S1 Teknologi Informasi dan S1 Rekayasa Perangkat Lunak: masing-masing 50%."
        ),
    },

    # ── Panduan Tugas Akhir (TA) ─────────────────────────────────────────────
    {
        "question": "Apa saja syarat untuk mengajukan Tugas Akhir?",
        "ground_truth": (
            "Syarat akademik untuk mengambil mata kuliah TA adalah: "
            "(1) Lulus MK Penulisan Proposal, "
            "(2) Diambil pada semester terakhir masa studi, "
            "(3) Sekurang-kurangnya sudah lulus 120 SKS pada semester tersebut, "
            "(4) Memiliki Surat Keputusan (SK) TA yang masih berlaku pada semester tersebut. "
            "Secara administrasi, mahasiswa harus terdaftar sebagai mahasiswa aktif di Fakultas Informatika."
        ),
    },
    {
        "question": "Berapa SKS minimal yang harus ditempuh sebelum mengambil TA?",
        "ground_truth": (
            "Mahasiswa harus sekurang-kurangnya sudah lulus 120 SKS sebelum dapat mengambil mata kuliah TA, "
            "kecuali untuk kasus khusus yang ditetapkan oleh Program Studi. "
            "Selain itu, mahasiswa juga harus sudah lulus MK Penulisan Proposal (yang mensyaratkan minimal 110 SKS) "
            "dan memiliki SK TA yang masih berlaku."
        ),
    },
    {
        "question": "Bagaimana format penulisan laporan Tugas Akhir?",
        "ground_truth": (
            "Format laporan TA terdiri dari dua alternatif: Laporan TA atau Makalah ilmiah yang dipublikasikan. "
            "Laporan TA harus mencantumkan nama mahasiswa sebagai Penulis 1 dan tim Pembimbing dengan afiliasi Fakultas Informatika. "
            "Laporan TA dapat berbentuk jurnal atau buku sesuai arahan pembimbing. "
            "Contoh format laporan TA dapat diunduh pada media publikasi resmi Fakultas Informatika."
        ),
    },
    {
        "question": "Apa yang dimaksud dengan sidang TA dan bagaimana prosedurnya?",
        "ground_truth": (
            "Sidang TA adalah sidang yang diselenggarakan untuk mengapresiasi, menjaga kualitas, diseminasi, "
            "menggali potensi pengetahuan mahasiswa, serta mengevaluasi validitas pemahaman mahasiswa tentang topik yang ditelitinya. "
            "Sidang dilaksanakan maksimum 90 menit per mahasiswa dalam 3 sesi: "
            "(1) Sesi Presentasi TA, (2) Sesi Diskusi/Tanya Jawab, (3) Sesi Sidang Tertutup. "
            "Terdapat 4 skema sidang: Non Terjadwal, Terjadwal, Khusus, dan Pengganti Sidang. "
            "Syarat umum: sudah lulus semua SKS MK selain TA, dan laporan TA diserahkan ke pembimbing dan penguji "
            "minimal 1 hari kerja sebelum jadwal sidang."
        ),
    },
    {
        "question": "Berapa lama batas waktu pengerjaan Tugas Akhir?",
        "ground_truth": (
            "SK TA berlaku selama 6 bulan sejak diterbitkan. "
            "SK TA dapat diperpanjang maksimal 2 kali, dengan masing-masing perpanjangan berlaku 3 bulan. "
            "Jika SK TA habis masa berlaku dan sudah diperpanjang 2 kali, mahasiswa harus mengajukan SK TA baru "
            "dengan judul dan pembimbing yang berbeda. "
            "Total waktu maksimal dengan perpanjangan adalah sekitar 12 bulan."
        ),
    },

    # ── Pedoman Proposal Tugas Akhir ─────────────────────────────────────────
    {
        "question": "Apa saja komponen yang harus ada dalam proposal TA?",
        "ground_truth": (
            "Isi proposal TA yang harus ada meliputi: "
            "(1) Sampul Muka, "
            "(2) Lembar Persetujuan yang ditandatangani calon pembimbing, "
            "(3) Abstrak, "
            "(4) Pendahuluan (latar belakang, perumusan masalah, tujuan, hipotesis opsional, rencana kegiatan, jadwal kegiatan), "
            "(5) Kajian Pustaka, "
            "(6) Perancangan Sistem atau Alur Pemodelan, "
            "(7) Daftar Pustaka (minimal 10 referensi, dari publikasi 5 tahun terakhir), "
            "(8) Lampiran."
        ),
    },
    {
        "question": "Bagaimana format dan struktur penulisan proposal TA?",
        "ground_truth": (
            "Format penulisan proposal TA: "
            "Kertas HVS ukuran A4 (210x297 mm), berat 80 g/m2. "
            "Margin: kiri 4 cm (1,58 inch), atas/kanan/bawah 3 cm (1,18 inch). "
            "Font: Times New Roman. Ukuran: Judul Bab 16pt, sub bab 14pt, sub-sub bab dan isi 12pt. "
            "Spasi 1,5 untuk seluruh isi proposal. "
            "Pengumpulan dapat berupa hardcopy (dicetak hitam) atau softcopy PDF."
        ),
    },
    {
        "question": "Apa persyaratan untuk mengambil mata kuliah Penulisan Proposal TA?",
        "ground_truth": (
            "Mahasiswa yang hendak mengambil Mata Kuliah Penulisan Proposal harus memenuhi persyaratan: "
            "(a) Sudah lulus Sidang Tingkat 2, dan "
            "(b) Sudah lulus minimal 110 SKS, kecuali untuk mahasiswa yang dapat lulus lebih cepat dari masa studi normal "
            "melalui keputusan Ketua Program Studi Sarjana."
        ),
    },
    {
        "question": "Bagaimana proses seminar proposal Tugas Akhir?",
        "ground_truth": (
            "Presentasi proposal dilaksanakan paling tidak kepada calon pembimbing TA. "
            "Penilaian presentasi menggunakan Formulir TA1-03 dengan aspek: "
            "(1) Penguasaan Materi Proposal: menjawab latar belakang, perumusan masalah, tujuan, dan metodologi secara terstruktur, "
            "menguasai teori pendukung dan tools pemodelan/implementasi. "
            "(2) Expert Judgement: cara pemaparan/menjawab dan interpersonal communications. "
            "Seminar Proposal dilaksanakan pada sekitar Minggu ke-14 perkuliahan."
        ),
    },
    {
        "question": "Siapa yang berhak menjadi pembimbing Tugas Akhir?",
        "ground_truth": (
            "Pembimbing TA terdiri dari: "
            "(a) Pembimbing Satu: dosen tetap minimal berpendidikan S-2 dengan jabatan akademik minimal Asisten Ahli (AA) "
            "pada bidang keahlian yang linier dengan program studi. "
            "(b) Pembimbing Dua: dosen minimal S-2 dengan jabatan akademik minimal AA dengan bidang keahlian linier/mendukung topik TA, "
            "atau praktisi berpengalaman di bidang yang sesuai dan disetujui pembimbing satu atau prodi. "
            "(c) Pembimbing Tunggal: Dosen Tetap Fakultas Informatika yang berpendidikan S3 (Doktor) "
            "dengan jabatan akademik minimal Lektor (L)."
        ),
    },

    # ── Pedoman TAK ──────────────────────────────────────────────────────────
    {
        "question": "Apa itu TAK dan apa tujuannya?",
        "ground_truth": (
            "Transkrip Aktivitas Kemahasiswaan (TAK) adalah rekap penilaian keaktifan mahasiswa dalam kegiatan kemahasiswaan "
            "selama menempuh pendidikan Sarjana dan Diploma di Universitas Telkom, yang dinyatakan dalam Indeks Keaktifan Kumulatif (IKK). "
            "Fungsi TAK antara lain: sebagai syarat pengajuan beasiswa, syarat seleksi mahasiswa berprestasi, "
            "syarat seleksi wisudawan berprestasi, bagian dari penilaian kelulusan sidang yudisium, "
            "meningkatkan soft skills dan kompetensi non-akademik mahasiswa, "
            "serta sebagai indikator performa pengembangan diri mahasiswa selama masa studi."
        ),
    },
    {
        "question": "Berapa poin TAK minimal yang harus dipenuhi mahasiswa?",
        "ground_truth": (
            "Nilai kumulatif minimal TAK berdasarkan jenjang pendidikan: "
            "Mahasiswa Sarjana/D4 Reguler: minimal 60 poin (maksimal 120 poin), IKK minimal 2,00. "
            "Mahasiswa D3 Reguler: minimal 45 poin (maksimal 90 poin). "
            "Mahasiswa S1 Pendidikan Jarak Jauh (PJJ): minimal 40 poin (maksimal 80 poin). "
            "Mahasiswa S1 Ekstensi: minimal 35 poin (maksimal 70 poin). "
            "Mahasiswa Asing S1/D4: minimal 40 poin (maksimal 80 poin)."
        ),
    },
    {
        "question": "Kegiatan apa saja yang bisa mendapatkan poin TAK?",
        "ground_truth": (
            "Kegiatan yang dapat menghasilkan poin TAK meliputi: "
            "(1) Orientasi Mahasiswa Baru (PKKMB dan Wawasan Kebangsaan), "
            "(2) Aktivitas Pengembangan Karakter (self-management, leadership, entrepreneur mindset, dll), "
            "(3) Aktivitas Pembelajaran: kompetisi (BELMAWA dan mandiri), penelitian, publikasi ilmiah, pengabdian masyarakat, entrepreneurship, pembelajaran luar kampus, kekayaan intelektual, "
            "(4) Aktivitas Kemahasiswaan: organisasi kemahasiswaan, kepanitiaan, duta kampus, "
            "(5) Aktivitas Perencanaan Karier (career preparation training, bimbingan karier), "
            "(6) Aktivitas Peminatan: seminar, pelatihan, sertifikasi, rekognisi, pameran karya."
        ),
    },
    {
        "question": "Bagaimana cara mengajukan poin TAK?",
        "ground_truth": (
            "Terdapat dua cara pelaporan TAK: "
            "(1) Pelaporan TAK Kolektif: untuk kegiatan yang diselenggarakan oleh Universitas Telkom, organisasi kemahasiswaan, "
            "atau unit resmi kampus, pelaporan dilakukan secara kolektif oleh panitia penyelenggara ke Direktorat/Bagian Kemahasiswaan. "
            "(2) Pelaporan TAK Mandiri: untuk kegiatan eksternal, mahasiswa melaporkan secara mandiri melalui Sistem Pelaporan TAK "
            "dengan mengunggah bukti keikutsertaan. "
            "Batas waktu pelaporan mandiri: maksimal 6 bulan setelah kegiatan berlangsung dalam tahun yang sama. "
            "Untuk kegiatan Oktober-Desember, batas pelaporan adalah akhir Februari tahun berikutnya."
        ),
    },
    {
        "question": "Apakah TAK berpengaruh pada kelulusan mahasiswa?",
        "ground_truth": (
            "Ya, TAK berpengaruh langsung pada kelulusan mahasiswa. "
            "TAK merupakan bagian dari penilaian kelulusan sidang yudisium. "
            "Mahasiswa yang tidak melaksanakan seluruh kegiatan wajib dan tidak memenuhi nilai kumulatif minimal TAK "
            "tidak dapat mengikuti sidang yudisium untuk kelulusan. "
            "Untuk mahasiswa Sarjana/D4, nilai minimal yang harus dipenuhi adalah 60 poin TAK. "
            "Nilai TAK dikonversikan menjadi Indeks Keaktifan Kumulatif (IKK) dengan rumus: IKK = (Ni / Nm) × 4, "
            "di mana IKK minimal yang harus dicapai adalah 2,00."
        ),
    },
]


# ── Pipeline Evaluasi ─────────────────────────────────────────────────────────

def collect_responses(dataset: list) -> list:
    """Jalankan setiap pertanyaan melalui pipeline RAG dan kumpulkan hasilnya.

    Mendukung checkpoint/resume: progres disimpan ke CHECKPOINT_FILE setelah
    setiap pertanyaan. Jika proses dihentikan (rate limit, dll), ganti API key
    di .env lalu jalankan ulang — proses akan melanjutkan dari pertanyaan terakhir.
    """
    # Muat checkpoint yang sudah ada (jika ada)
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Checkpoint ditemukan: {len(results)}/{len(dataset)} pertanyaan sudah selesai.")
        print("Melanjutkan dari pertanyaan yang belum dijawab...\n")
    else:
        results = []

    completed_questions = {r["question"] for r in results}
    total = len(dataset)

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]

        if question in completed_questions:
            print(f"[{i + 1}/{total}] SKIP (sudah ada di checkpoint): {question[:60]}")
            continue

        print(f"\n[{i + 1}/{total}] {question}")

        answer = "ERROR"
        contexts = ["ERROR"]

        for attempt in range(1, 4):  # maksimal 3 percobaan
            try:
                # Reload .env agar API key terbaru langsung terbaca
                load_dotenv(override=True)
                docs = get_relevant_documents(question)
                answer, _, _ = ask_question(question)
                contexts = [doc.page_content for doc in docs] if docs else ["Tidak ada konteks relevan."]
                break  # berhasil, keluar dari loop retry
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "resource" in err.lower():
                    wait_sec = 60 * attempt
                    print(f"  Rate limit (percobaan {attempt}/3). Tunggu {wait_sec}s lalu coba lagi...")
                    print(f"  Jika ingin ganti API key: edit .env sekarang, lalu tekan Ctrl+C dan jalankan ulang.")
                    time.sleep(wait_sec)
                else:
                    print(f"  ERROR tidak dikenal: {e}")
                    break

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

        # Simpan checkpoint setelah setiap pertanyaan
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"  Checkpoint disimpan ({len(results)}/{total})")
        if answer != "ERROR":
            print(f"  Jawaban (50 karakter pertama): {answer[:50]}...")
            print(f"  Jumlah konteks: {len(contexts)} chunk")

    return results


def _score(val):
    """Ekstrak nilai float dari hasil RAGAS (bisa berupa list atau float)."""
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    try:
        v = float(val)
        return None if v != v else v  # handle NaN
    except (TypeError, ValueError):
        return None


def run_ragas_per_question(responses: list) -> list:
    """Evaluasi RAGAS satu pertanyaan per satu, dengan checkpoint per pertanyaan.

    Setiap pertanyaan selesai langsung disimpan ke RAGAS_CHECKPOINT_FILE.
    Jika kena rate limit: ganti API key di .env, jalankan ulang — lanjut dari pertanyaan terakhir.
    """
    if os.path.exists(RAGAS_CHECKPOINT_FILE):
        with open(RAGAS_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            scored = json.load(f)
        # Hanya anggap selesai jika semua 3 skor bukan null
        scored = [r for r in scored
                  if all(_score(r.get(k)) is not None
                         for k in ["faithfulness", "context_precision", "context_recall"])]
        print(f"RAGAS checkpoint: {len(scored)}/{len(responses)} pertanyaan sudah dievaluasi (lengkap).")
    else:
        scored = []

    completed = {r["question"] for r in scored}
    total = len(responses)
    run_config = RunConfig(max_workers=1, max_wait=120, timeout=300)

    for i, response in enumerate(responses):
        question = response["question"]

        if question in completed:
            print(f"[{i + 1}/{total}] SKIP RAGAS (sudah ada): {question[:55]}")
            continue

        print(f"\n[{i + 1}/{total}] Evaluasi RAGAS: {question[:55]}")

        # Reload .env setiap pertanyaan — API key baru langsung dipakai
        load_dotenv(override=True)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY tidak ditemukan di .env. Isi dulu lalu jalankan ulang.")

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY tidak ditemukan di .env")
        evaluator_llm = LangchainLLMWrapper(
            ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0)
        )

        single_dataset = Dataset.from_dict({
            "question":     [response["question"]],
            "answer":       [response["answer"]],
            "contexts":     [response["contexts"]],
            "ground_truth": [response["ground_truth"]],
        })

        try:
            result = evaluate(
                single_dataset,
                metrics=[faithfulness, context_precision, context_recall],
                llm=evaluator_llm,
                run_config=run_config,
            )
            row = {
                "question":          question,
                "answer":            response["answer"],
                "ground_truth":      response["ground_truth"],
                "faithfulness":      _score(result["faithfulness"]),
                "context_precision": _score(result["context_precision"]),
                "context_recall":    _score(result["context_recall"]),
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            row = {
                "question":          question,
                "answer":            response["answer"],
                "ground_truth":      response["ground_truth"],
                "faithfulness":      None,
                "context_precision": None,
                "context_recall":    None,
            }

        scored.append(row)

        with open(RAGAS_CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(scored, f, ensure_ascii=False, indent=2)

        f_val  = f"{row['faithfulness']:.4f}"      if row["faithfulness"]      is not None else "ERROR"
        cp_val = f"{row['context_precision']:.4f}" if row["context_precision"] is not None else "ERROR"
        cr_val = f"{row['context_recall']:.4f}"    if row["context_recall"]    is not None else "ERROR"
        print(f"  F={f_val} | CP={cp_val} | CR={cr_val}")
        print(f"  Checkpoint disimpan ({len(scored)}/{total})")

    return scored


def run_evaluation():
    # FASE 1: Kumpulkan respons RAG (dengan checkpoint/resume)
    print("=" * 60)
    print("FASE 1: Menjalankan pipeline RAG untuk semua pertanyaan...")
    if os.path.exists(CHECKPOINT_FILE):
        print("(checkpoint ditemukan — melanjutkan progres sebelumnya)")
    print("Tip: jika kena rate limit, ganti GOOGLE_API_KEY di .env lalu jalankan ulang.")
    print("=" * 60)
    responses = collect_responses(TEST_DATASET)

    # FASE 2: Evaluasi RAGAS per pertanyaan (ada checkpoint per pertanyaan)
    print("\n" + "=" * 60)
    print("FASE 2: Menjalankan evaluasi RAGAS (checkpoint per pertanyaan)...")
    if os.path.exists(RAGAS_CHECKPOINT_FILE):
        print("(RAGAS checkpoint ditemukan — melanjutkan dari pertanyaan terakhir)")
    print("Jika kena rate limit: ganti API key di .env lalu jalankan ulang.")
    print("=" * 60)
    scored = run_ragas_per_question(responses)

    return scored


def save_results(scored: list):
    """Simpan hasil evaluasi ke CSV dan tampilkan ringkasan."""

    def avg(key):
        vals = [_score(r[key]) for r in scored if r.get(key) is not None]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n" + "=" * 60)
    print("HASIL EVALUASI RAGAS")
    print("=" * 60)
    print(f"  Faithfulness       : {avg('faithfulness'):.4f}")
    print(f"  Context Precision  : {avg('context_precision'):.4f}")
    print(f"  Context Recall     : {avg('context_recall'):.4f}")
    print("=" * 60)

    fields = ["question", "answer", "ground_truth", "faithfulness", "context_precision", "context_recall"]
    with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored)
    print("\nDetail hasil disimpan ke: eval_results.csv")

    summary = {
        "faithfulness":      avg("faithfulness"),
        "context_precision": avg("context_precision"),
        "context_recall":    avg("context_recall"),
    }
    with open("eval_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    print("Ringkasan skor disimpan ke: eval_summary.csv")


if __name__ == "__main__":
    scored = run_evaluation()
    save_results(scored)
    # Hapus kedua checkpoint hanya setelah semua fase selesai dan CSV tersimpan
    for ckpt in [CHECKPOINT_FILE, RAGAS_CHECKPOINT_FILE]:
        if os.path.exists(ckpt):
            os.remove(ckpt)
    print("\nSemua checkpoint dihapus. Evaluasi selesai sepenuhnya.")
