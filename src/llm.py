import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load file .env (lokal). Di Streamlit Cloud, API key dibaca dari st.secrets.
load_dotenv()


def get_llm():

    api_key = os.getenv("GOOGLE_API_KEY")

    # Fallback: baca dari Streamlit Secrets saat deploy di Streamlit Cloud
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise ValueError(
            "Kunci API tidak ditemukan. Isi GOOGLE_API_KEY di file .env (lokal) "
            "atau di Secrets dashboard Streamlit Cloud."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.5
    )

    return llm