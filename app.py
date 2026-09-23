import streamlit as st
from pypdf import PdfReader
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------
load_dotenv()
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown("""
<style>
    /* Main application */
    .stApp {
        background-color: #0b0f19;
    }
    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    header {
        visibility: hidden;
    }
    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    /* Header */
    .hero {
        padding: 10px 0 25px 0;
    }
    .hero-title {
        font-size: 42px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 16px;
        color: #9ca3af;
    }
    /* Status */
    .status {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background-color: #13261d;
        color: #4ade80;
        font-size: 13px;
        margin-top: 10px;
    }
    /* Cards */
    .card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
    }
    .card-title {
        color: #ffffff;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .card-text {
        color: #9ca3af;
        font-size: 14px;
    }
    /* Answer */
    .answer-box {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 25px;
        margin-top: 10px;
        color: #e5e7eb;
        line-height: 1.7;
        font-size: 16px;
    }
    /* Metric */
    .metric {
        background-color: #0f172a;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .metric-number {
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 12px;
    }
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 46px;
        font-weight: 600;
    }
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #111827;
        border-radius: 14px;
        padding: 8px;
    }
    /* Text input */
    [data-testid="stTextInput"] input {
        background-color: #111827;
        color: white;
        border: 1px solid #374151;
        border-radius: 10px;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #080c14;
        border-right: 1px solid #1f2937;
    }
</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">
        ✦ AI Knowledge Assistant
    </div>
    <div class="hero-subtitle">
        Ask questions and get intelligent answers from your documents.
    </div>
    <div class="status">
        ● AI System Ready
    </div>
</div>
""", unsafe_allow_html=True)
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 📚 Knowledge Base")
    st.markdown(
        "Upload a PDF document and the AI will analyze its contents."
    )
    st.divider()
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Choose a PDF document to analyze."
    )
    st.divider()
    st.markdown("### How it works")
    st.markdown("""
    **01 — Upload**
    Add your PDF document.
    **02 — Process**
    The document is extracted and indexed.
    **03 — Ask**
    Ask questions about the document.
    **04 — Answer**
    AI retrieves relevant information and generates an answer.
    """)
    st.divider()
    st.caption("AI Knowledge Assistant")
    st.caption("Built with Python • Streamlit • FAISS • Groq")
# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------
if uploaded_file is None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">📄 Upload Documents</div>
            <div class="card-text">
                Add PDF files to create your personal knowledge base.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🔎 Smart Retrieval</div>
            <div class="card-text">
                Relevant document sections are retrieved for each question.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">✦ AI Answers</div>
            <div class="card-text">
                Get answers generated from your uploaded document.
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.info("Upload a PDF from the sidebar to get started.")
else:
    # --------------------------------------------------
    # PDF EXTRACTION
    # --------------------------------------------------
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    st.success("Document uploaded successfully.")
    # --------------------------------------------------
    # TEXT CHUNKING
    # --------------------------------------------------
    chunk_size = 1000
    overlap = 200
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    if not chunks:
        st.error(
            "No readable text could be extracted from this PDF."
        )
        st.stop()
    # --------------------------------------------------
    # DOCUMENT INFORMATION
    # --------------------------------------------------
    st.markdown("### Document Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-number">{len(reader.pages)}</div>
                <div class="metric-label">Pages</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-number">{len(chunks)}</div>
                <div class="metric-label">Text Chunks</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-number">{len(text):,}</div>
                <div class="metric-label">Characters</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    # --------------------------------------------------
    # TF-IDF
    # --------------------------------------------------
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )
    vectors = vectorizer.fit_transform(chunks)
    vectors = vectors.astype(
        "float32"
    ).toarray()
    # --------------------------------------------------
    # FAISS
    # --------------------------------------------------
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    # --------------------------------------------------
    # QUESTION AREA
    # --------------------------------------------------
    st.markdown("### Ask Your Document")
    question = st.text_input(
        "Question",
        placeholder="What would you like to know about this document?",
        label_visibility="collapsed"
    )
    ask_button = st.button(
        "✦  Ask AI",
        use_container_width=True
    )
    # --------------------------------------------------
    # ANSWER
    # --------------------------------------------------
    if ask_button:
        if not question.strip():
            st.warning(
                "Please enter a question first."
            )
            st.stop()
        # Question vector
        question_vector = vectorizer.transform(
            [question]
        ).astype(
            "float32"
        ).toarray()
        # Search
        k = min(
            3,
            len(chunks)
        )
        distances, indices = index.search(
            question_vector,
            k
        )
        # Relevant chunks
        relevant_chunks = []
        for i in indices[0]:
            relevant_chunks.append(
                chunks[i]
            )
        context = "\n\n".join(
            relevant_chunks
        )
        # --------------------------------------------------
        # GROQ PROMPT
        # --------------------------------------------------
        prompt = f"""
Answer the user's question using ONLY the information
provided in the document context below.
If the answer cannot be found in the context,
say that the information is not available in the document.
DOCUMENT CONTEXT:
{context}
USER QUESTION:
{question}
"""
        # --------------------------------------------------
        # GROQ
        # --------------------------------------------------
        with st.spinner(
            "Analyzing the document..."
        ):
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content":
                        "You answer questions based only on provided document context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2
            )
        answer = response.choices[0].message.content
        # --------------------------------------------------
        # DISPLAY ANSWER
        # --------------------------------------------------
        st.markdown("### AI Answer")
        st.markdown(
            f"""
            <div class="answer-box">
                {answer}
            </div>
            """,
            unsafe_allow_html=True
        )
        # --------------------------------------------------
        # SOURCES
        # --------------------------------------------------
        with st.expander(
            "📚 View Retrieved Document Sections"
        ):
            for number, chunk in enumerate(
                relevant_chunks,
                1
            ):
                st.markdown(
                    f"**Retrieved Section {number}**"
                )
                st.write(chunk)
                if number < len(relevant_chunks):
                    st.divider()

