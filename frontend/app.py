import streamlit as st
import sys
import os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from backend.app import process_document, answer_question
from backend.utils import validate_pdf
from backend.config import GROQ_API_KEY

from groq import Groq
groq_client=Groq(api_key=GROQ_API_KEY)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="✦",
    layout="wide",
)


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #080b16,
            #10152a
        );
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        color: white;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: white;
        margin: 25px 0 18px;
    }

    .feature-card {
        background: #151a2d;
        border: 1px solid #292f4a;
        border-radius: 16px;
        padding: 24px;
        min-height: 170px;
    }

    .feature-icon {
        font-size: 30px;
        margin-bottom: 12px;
    }

    .feature-title {
        color: white;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .feature-text {
        color: #9ca3af;
        font-size: 14px;
        line-height: 1.6;
    }

    .document-bar {
        background: #151a2d;
        border: 1px solid #303750;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 30px;
    }

    .document-name {
        color: white;
        font-weight: 600;
    }

    .document-status {
        color: #4ade80;
        font-size: 14px;
        margin-top: 4px;
    }

    div[data-testid="stTextInput"] input {
        background: white;
        color: #111827;
        border-radius: 10px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        background: #6d4aff;
        color: white;
        font-weight: 700;
        border: none;
    }

    .answer-box {
        background: #151a2d;
        border: 1px solid #303750;
        border-radius: 16px;
        padding: 24px;
        color: white;
        line-height: 1.7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📚 Knowledge Base")

    st.write(
        "Upload a PDF and ask questions about its content."
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
    )

    st.divider()

    st.subheader("How it works")

    st.write("1. Upload your document")
    st.write("2. Retrieve relevant sections")
    st.write("3. Generate an AI answer")


# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="hero-title">✦ AI Knowledge Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-subtitle">
        Ask questions about your documents and get
        AI-powered answers based on their content.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# NO DOCUMENT
# =========================================================

if uploaded_file is None:

    st.markdown(
        '<div class="section-title">Get started</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <div class="feature-title">
                    Upload Documents
                </div>
                <div class="feature-text">
                    Upload a PDF and create a searchable
                    knowledge base.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔎</div>
                <div class="feature-title">
                    Smart Retrieval
                </div>
                <div class="feature-text">
                    Find the most relevant sections of
                    your document.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">✦</div>
                <div class="feature-title">
                    AI Answers
                </div>
                <div class="feature-text">
                    Generate answers using the retrieved
                    document information.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if not validate_pdf(uploaded_file):

    st.error("Please upload a valid PDF file.")
    st.stop()


if (
    "file_name" not in st.session_state
    or st.session_state.file_name != uploaded_file.name
):

    with st.spinner("Processing document..."):

        chunks, vectorizer, index = process_document(
            uploaded_file
        )

        st.session_state.file_name = uploaded_file.name
        st.session_state.chunks = chunks
        st.session_state.vectorizer = vectorizer
        st.session_state.index = index


# =========================================================
# DOCUMENT READY
# =========================================================

st.success(f"📄 {uploaded_file.name} — Document Ready")

# =========================================================
# ASK DOCUMENT
# =========================================================

st.markdown(
    '<div class="section-title">✦ Ask Your Document</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([5, 1])

with col1:
    question = st.text_input(
        "Question",
        placeholder="Ask something about your document...",
        label_visibility="collapsed",
    )

with col2:
    audio = st.audio_input(
        "🎤",
        label_visibility="collapsed",
    )
if audio:
    audio_bytes = audio.getvalue()

    transcription = groq_client.audio.transcriptions.create(
        file=("question.wav", audio_bytes),
        model="whisper-large-v3-turbo",
    )

    question = transcription.text

    st.info(f"🎤 {question}")
ask = st.button("✦ Ask AI")
# =========================================================
# ANSWER
# =========================================================

if ask:

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            answer, relevant_chunks = answer_question(
                question,
                st.session_state.chunks,
                st.session_state.vectorizer,
                st.session_state.index,
            )

        st.markdown(
            '<div class="section-title">✦ AI Answer</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="answer-box">
                {answer}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">🔎 Retrieved Document Sections</div>',
            unsafe_allow_html=True,
        )

        for number, chunk in enumerate(
            relevant_chunks,
            start=1,
        ):

            with st.expander(
                f"Document Section {number}"
            ):
                st.write(chunk)