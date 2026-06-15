import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")

    # Fallback: baca dari Streamlit Secrets saat deploy di Streamlit Cloud
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENROUTER_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise ValueError(
            "Kunci API tidak ditemukan. Isi OPENROUTER_API_KEY di file .env (lokal) "
            "atau di Secrets dashboard Streamlit Cloud."
        )

    # OpenRouter memakai format API yang kompatibel dengan OpenAI.
    # Model tetap Gemini 2.5 Flash — OpenRouter yang meneruskan ke Google.
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.5,
    )

    return llm