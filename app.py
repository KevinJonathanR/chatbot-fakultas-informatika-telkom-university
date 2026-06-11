import streamlit as st
import time
import re
import json
from datetime import datetime
from pathlib import Path
from src.rag_chain import ask_question

# ── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFMate",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Warna utama */
:root {
    --bg-main:       #3A2E42;
    --bg-card:       #4C3D55;
    --bg-hover:      #5C4A66;
    --sidebar-bg:    #322737;
    --accent:        #796475;
    --accent-hover:  #90788B;
    --accent-muted:   rgba(121,100,117,0.20);
    --accent-border: rgba(121,100,117,0.35);
    --border:        #8F7990;
    --text-primary:  #FFFFFF;
    --text-secondary:#E8E0E6;
    --text-muted:    #C8BDC5;
}

/* Background utama */
.stApp {
    background:
        radial-gradient(
            circle at top right,
            rgba(255,255,255,0.04),
            transparent 35%
        ),

        radial-gradient(
            circle at bottom left,
            rgba(255,255,255,0.03),
            transparent 40%
        ),

        linear-gradient(
            180deg,
            #4E4054 0%,
            #43374A 50%,
            #3A2E42 100%
        ) !important;
}

/* semua layer Streamlit transparan */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
[data-testid="stBottom"],
section.main,
.main,
.block-container {
    background: transparent !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Tombol utama */
.stButton > button {
    width: 100%;
    background: #796475 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    cursor: pointer;
    transition: background 0.2s ease;
}

.stButton > button:hover {
    background: #90788B !important;
}

/* Tombol history — sibling selector karena div wrapper tidak bisa wrap st.button */
[data-testid="stSidebar"] .history-btn + div button,
[data-testid="stSidebar"] .history-btn ~ div button {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    font-size: 16px !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 8px 12px !important;
}

[data-testid="stSidebar"] .history-btn ~ div button:hover {
    background-color: var(--bg-hover) !important;
    color: var(--text-primary) !important;
}

/* Tombol feedback — pakai :has() untuk target button setelah marker */
[data-testid="stChatMessage"] [data-testid="stMarkdown"]:has(.feedback-btn) ~ div button,
[data-testid="stChatMessage"] .feedback-btn ~ div button {
    background-color: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    padding: 4px 10px !important;
    width: auto !important;
    min-width: 42px !important;
}

/* Tombol follow-up */
[data-testid="stChatMessage"] [data-testid="stMarkdown"]:has(.followup-btn) ~ div button,
[data-testid="stChatMessage"] .followup-btn ~ div button {
    background-color: var(--accent-muted) !important;
    color: #D8D0FF  !important;
    border: 1px solid var(--accent-border) !important;
    border-radius: 20px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 5px 14px !important;
}

/* Chat messages */
.stChatMessage {
    background: rgba(36,19,84,0.95) !important;
    backdrop-filter: blur(8px);
    border: 1px solid var(--border-gray) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
}

[data-testid="stChatMessage-user"] {
    background: rgba(155,109,255,0.18) !important;
    border-color: rgba(155,109,255,0.4) !important;
}

.stMarkdown p, .stMarkdown li {
    font-size: 15px;
    line-height: 1.75;
    color: var(--text-primary);
}

/* Wrapper bawah */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
.stChatInputContainer,
div[class*="bottom"] {
    background-color: var(--bg-main) !important;
    border-top: none !important;
}

/* Input chat */
[data-testid="stChatInput"] {
    background: #5B4B63 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    color: white !important;
    font-size: 15px !important;
}

[data-testid="stChatInput"] > div {
    background: #5B4B63 !important;
}

[data-testid="stChatInput"] textarea {
    background: #5B4B63 !important;
    color: white !important;
}

[data-testid="stChatInput"] input {
    background: #5B4B63 !important;
    color: white !important;
}
[data-testid="stChatInput"] button {
    background: #796475 !important;
    border-radius: 10px !important;
}

[data-testid="stChatInput"] button:hover {
    background: #90788B !important;
}
/* Expander sumber */
.streamlit-expanderHeader {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-gray) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    color: var(--text-secondary) !important;
}

.streamlit-expanderContent {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-gray) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    font-size: 13px !important;
}

hr {
    border-color: var(--border-gray) !important;
    margin: 12px 0 !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }

[data-testid="stToolbar"] { display: none; }

/* Welcome card */
.welcome-card {
    background:
        linear-gradient(
            135deg,
            #5B4B63 0%,
            #4C3D55 50%,
            #43374A 100%
        );

    backdrop-filter: blur(10px);

    border: 1px solid rgba(255,255,255,0.12);

    border-radius: 16px;

    padding: 40px 45px;

    margin-bottom: 24px;

    box-shadow:
        0 8px 32px rgba(0,0,0,0.15);
}

.welcome-card h3 {
    font-size: 42px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 12px;
}

.welcome-card p, .welcome-card li {
    font-size: 20px;
    color: var(--text-secondary);
    line-height: 1.8;
}

.example-chip {
    display: inline-block;
    background-color: var(--accent-muted);
    border: 1px solid var(--accent-border);
    border-radius: 20px;
    padding: 12px  20px;
    font-size: 16px;
    font-weight: 500;
    color: #D8D0FF;
    margin: 4px 4px 4px 0;
}

/* Header */
.page-header {
    padding: 8px 0 20px 0;
    border-bottom: 1px solid var(--border-gray);
    margin-bottom: 24px;
}

.page-title {
    font-size: 42px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.5px;
    color: white;
}

.page-subtitle {
    font-size: 18px;
    line-height: 1.7;
    color: var(--text-secondary);
    margin-top: 4px;
    color: #E6DEFF;
}

/* Label sidebar */
.sidebar-label {
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    margin-bottom: 8px;
    margin-top: 20px;
}

/* Download button di sidebar */
[data-testid="stDownloadButton"] button {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
}

[data-testid="stDownloadButton"] button:hover {
    background-color: var(--bg-hover) !important;
    color: var(--text-primary) !important;
}

</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────

FEEDBACK_FILE = Path("data/feedback.json")


def save_feedback(question: str, answer: str, value: str):
    """Simpan feedback 👍👎 ke data/feedback.json untuk evaluasi kualitas model."""
    FEEDBACK_FILE.parent.mkdir(exist_ok=True)
    records = []
    if FEEDBACK_FILE.exists():
        try:
            records = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    records.append({
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer_preview": answer[:200],
        "feedback": value
    })
    FEEDBACK_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def highlight_snippet(snippet: str, question: str) -> str:
    """Highlight kata kunci dari pertanyaan di dalam potongan teks sumber."""
    stop_words = {
        "apa", "bagaimana", "berapa", "siapa", "kapan", "dimana", "apakah",
        "adalah", "yang", "untuk", "dengan", "dari", "dan", "atau",
        "di", "ke", "pada", "ini", "itu", "saya", "kamu", "anda",
        "nya", "ya", "juga", "sudah", "belum", "bisa", "mohon", "tolong"
    }
    keywords = [
        w.lower().strip("?!.,;:'\"()") for w in question.split()
        if len(w.strip("?!.,;:'\"()")) > 3
        and w.lower().strip("?!.,;:'\"()") not in stop_words
    ]
    if not keywords:
        return snippet
    result = snippet
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        result = pattern.sub(
            r'<mark style="background:rgba(220,38,38,0.22);padding:0 2px;border-radius:2px;">\g<0></mark>',
            result
        )
    return result


def generate_pdf(messages: list) -> bytes:
    from fpdf import FPDF

    def clean(text: str) -> str:
        for pattern, repl in [
            (r'\*\*(.*?)\*\*', r'\1'),
            (r'\*(.*?)\*', r'\1'),
            (r'#{1,6}\s+', ''),
            (r'`(.*?)`', r'\1'),
        ]:
            text = __import__('re').sub(pattern, repl, text)
        for old, new in [
            ('\u2013', '-'), ('\u2014', '-'),
            ('\u201c', '"'), ('\u201d', '"'),
            ('\u2018', "'"), ('\u2019', "'"),
            ('\u2026', '...'), ('\u2022', '-'),
        ]:
            text = text.replace(old, new)
        return text.encode('latin-1', errors='replace').decode('latin-1')

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    w = pdf.epw  # effective page width (fpdf2 2.7+ requires explicit width, not 0)

    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(w, 10, clean('FIFMate - Telkom University'), align='C', new_x='LMARGIN', new_y='NEXT')

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(130, 130, 130)
    timestamp = __import__('datetime').datetime.now().strftime('%d %B %Y, %H:%M')
    pdf.multi_cell(w, 6, clean(f'Diekspor: {timestamp}'), align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    for msg in messages:
        if msg['role'] == 'user':
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(w, 7, 'Pertanyaan:', new_x='LMARGIN', new_y='NEXT')
        else:
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(160, 40, 40)
            pdf.multi_cell(w, 7, 'Jawaban:', new_x='LMARGIN', new_y='NEXT')

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(w, 6, clean(msg.get('content', '')), new_x='LMARGIN', new_y='NEXT')
        pdf.ln(4)

    return bytes(pdf.output())


def clean_snippet(text: str) -> str:
    """Bersihkan snippet dari label dokumen di awal."""
    text = re.sub(r'^\[.*?\]\s*\n?', '', text.strip())
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def render_sources(sources: list, answer: str, question: str, key_prefix: str = "src"):
    """Render expander sumber dokumen dengan highlight dan tombol download PDF."""
    info_not_found = "tidak tersedia dalam dokumen" in answer.lower()
    if not sources or info_not_found:
        return

    with st.expander(f"Sumber dokumen ({len(sources)})"):
        for k, source in enumerate(sources):
            if k > 0:
                st.markdown("---")

            st.markdown(
                f'<div style="font-size:11px; font-weight:600; letter-spacing:0.8px; '
                f'text-transform:uppercase; color:#9A9A9A; margin-bottom:8px;">'
                f'{source["name"]}</div>',
                unsafe_allow_html=True
            )

            snippet = clean_snippet(source["snippet"])
            display = (snippet[:450] + " …") if len(snippet) > 450 else snippet
            highlighted = highlight_snippet(display, question)
            st.markdown(
                f'<div style="font-size:13px; color:#C8C8C8; line-height:1.7; '
                f'padding:12px 14px; background:#252525; border-radius:8px; '
                f'border-left:2px solid #DC2626;">{highlighted}</div>',
                unsafe_allow_html=True
            )

        # ── Tombol download PDF di LUAR loop, satu per PDF unik ───────────────
        st.markdown("---")
        st.markdown(
            '<div style="font-size:11px; color:#606060; margin-bottom:8px; '
            'letter-spacing:0.8px; text-transform:uppercase;">Unduh Dokumen Sumber</div>',
            unsafe_allow_html=True
        )
        seen_pdfs: set = set()
        for k, source in enumerate(sources):
            pdf_path = source.get("pdf_path")
            pdf_name = source.get("pdf_name")
            if pdf_path and pdf_name and pdf_name not in seen_pdfs:
                seen_pdfs.add(pdf_name)
                pdf_file = Path(pdf_path)
                if pdf_file.exists():
                    pdf_bytes = pdf_file.read_bytes()
                    st.download_button(
                        label=f"⬇  {pdf_name}",
                        data=pdf_bytes,
                        file_name=pdf_name,
                        mime="application/pdf",
                        key=f"{key_prefix}_pdf_{k}_{pdf_name.replace(' ', '_')}",
                        use_container_width=False,
                    )


def render_follow_ups(follow_ups: list, key_prefix: str):
    """Render tombol pertanyaan lanjutan."""
    if not follow_ups:
        return
    st.markdown(
        '<div style="font-size:11px; color:#606060; margin-top:14px; margin-bottom:8px; '
        'letter-spacing:0.8px; text-transform:uppercase;">Pertanyaan lanjutan</div>',
        unsafe_allow_html=True
    )
    for j, q in enumerate(follow_ups):
        st.markdown('<div class="followup-btn">', unsafe_allow_html=True)
        if st.button(q, key=f"{key_prefix}_fup_{j}"):
            st.session_state.pending_question = q
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversations" not in st.session_state:
    st.session_state.conversations = []

# None = percakapan baru; integer = indeks percakapan yang dimuat dari riwayat
if "active_conv_idx" not in st.session_state:
    st.session_state.active_conv_idx = None

# Pertanyaan follow-up yang diklik, diproses di input handling
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="page-title" style="font-size:38px;">FIFMate</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle" style="font-size:20px;">Fakultas Informatika · Telkom University</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Export PDF
    if st.session_state.messages:
        try:
            pdf_bytes = generate_pdf(st.session_state.messages)
            st.download_button(
                label="⬇  Export PDF",
                data=pdf_bytes,
                file_name=f"percakapan_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.markdown('<div style="margin-bottom:6px;"></div>', unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"PDF error: {e}")

    # Tombol percakapan baru
    if st.button("＋  Percakapan Baru"):
        if st.session_state.messages:
            if st.session_state.active_conv_idx is None:
                judul = st.session_state.messages[0]["content"]
                judul = (judul[:42] + "…") if len(judul) > 42 else judul
                st.session_state.conversations.insert(0, {
                    "title": judul,
                    "messages": st.session_state.messages.copy()
                })
            else:
                st.session_state.conversations[st.session_state.active_conv_idx]["messages"] = st.session_state.messages.copy()
        st.session_state.messages = []
        st.session_state.active_conv_idx = None
        st.rerun()

    # Riwayat percakapan
    st.markdown('<div class="sidebar-label">Riwayat</div>', unsafe_allow_html=True)

    if not st.session_state.conversations:
        st.markdown('<p style="font-size:13px; color:#444; margin-top:8px;">Belum ada riwayat.</p>', unsafe_allow_html=True)
    else:
        for i, conv in enumerate(st.session_state.conversations):
            with st.container():
                st.markdown('<div class="history-btn">', unsafe_allow_html=True)
                if st.button(conv["title"], key=f"conv_{i}"):
                    if st.session_state.messages:
                        if st.session_state.active_conv_idx is None:
                            judul = st.session_state.messages[0]["content"]
                            judul = (judul[:42] + "…") if len(judul) > 42 else judul
                            st.session_state.conversations.insert(0, {
                                "title": judul,
                                "messages": st.session_state.messages.copy()
                            })
                            i += 1
                        else:
                            st.session_state.conversations[st.session_state.active_conv_idx]["messages"] = st.session_state.messages.copy()
                    st.session_state.messages = st.session_state.conversations[i]["messages"].copy()
                    st.session_state.active_conv_idx = i
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">FIFMate</div>
    <div class="page-subtitle">Tanya seputar KP, Tugas Akhir, TAK, dan prosedur akademik lainnya.</div>
</div>
""", unsafe_allow_html=True)


# ── Welcome Card ─────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h3>Selamat datang</h3>
        <p>Ajukan pertanyaan mengenai dokumen akademik resmi Fakultas Informatika.<br>
        Jawaban disusun berdasarkan panduan KP, TA, TAK, dan proposal TA.</p>
        <br>
        <p style="font-size:14px; font-weight:600; color:#FFFFFF; margin-bottom:10px;letter-spacing:1px;">CONTOH PERTANYAAN</p>
        <span class="example-chip">Apa syarat mengambil TA?</span>
        <span class="example-chip">Berapa minimal SKS untuk KP?</span>
        <span class="example-chip">Bagaimana format proposal TA?</span>
        <span class="example-chip">Apa itu TAK?</span>
    </div>
    """, unsafe_allow_html=True)


# ── Riwayat Chat Aktif ────────────────────────────────────────────────────────
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            is_last = (idx == len(st.session_state.messages) - 1)

            # Feedback 👍 👎
            feedback = message.get("feedback")
            q_text = st.session_state.messages[idx - 1]["content"] if idx > 0 else ""

            col1, col2, _ = st.columns([1, 1, 10])
            with col1:
                st.markdown('<div class="feedback-btn">', unsafe_allow_html=True)
                up_label = "👍 ✓" if feedback == "up" else "👍"
                if st.button(up_label, key=f"up_{idx}", disabled=(feedback is not None)):
                    st.session_state.messages[idx]["feedback"] = "up"
                    save_feedback(q_text, message["content"], "up")
                    if st.session_state.active_conv_idx is not None:
                        st.session_state.conversations[st.session_state.active_conv_idx]["messages"] = st.session_state.messages.copy()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="feedback-btn">', unsafe_allow_html=True)
                down_label = "👎 ✓" if feedback == "down" else "👎"
                if st.button(down_label, key=f"down_{idx}", disabled=(feedback is not None)):
                    st.session_state.messages[idx]["feedback"] = "down"
                    save_feedback(q_text, message["content"], "down")
                    if st.session_state.active_conv_idx is not None:
                        st.session_state.conversations[st.session_state.active_conv_idx]["messages"] = st.session_state.messages.copy()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            # Sumber dengan highlight
            sources = message.get("sources", [])
            render_sources(sources, message["content"], q_text, key_prefix=f"hist_{idx}")

            # Pertanyaan lanjutan (hanya untuk pesan terakhir)
            if is_last:
                follow_ups = message.get("follow_ups", [])
                render_follow_ups(follow_ups, key_prefix=f"hist_{idx}")


# ── Input Pengguna ────────────────────────────────────────────────────────────
prompt = st.chat_input("Tulis pertanyaan Anda...")

# Gunakan pertanyaan follow-up jika ada yang diklik
if not prompt and st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            try:
                answer, sources, follow_ups = ask_question(prompt)

                placeholder = st.empty()
                full_response = ""
                for word in answer.split():
                    full_response += word + " "
                    placeholder.markdown(full_response)
                    time.sleep(0.02)

                # Sumber
                render_sources(sources, answer, prompt, key_prefix="new")

                # Pertanyaan lanjutan (ditampilkan langsung setelah jawaban)
                render_follow_ups(follow_ups, key_prefix="new")

                # Simpan ke session state
                new_msg = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "follow_ups": follow_ups,
                    "feedback": None
                }
                st.session_state.messages.append(new_msg)

                if st.session_state.active_conv_idx is not None:
                    st.session_state.conversations[st.session_state.active_conv_idx]["messages"] = st.session_state.messages.copy()

                st.rerun()

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")