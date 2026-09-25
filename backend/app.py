from .chunking import extract_text_from_pdf, split_text_into_chunks
from .embedding import create_vectorizer, create_vectors, vectorize_question
from .vector_store import create_faiss_index, search_similar_chunks
from .generator import generate_answer
from .config import GROQ_API_KEY

from groq import Groq


def process_document(uploaded_file):
    """Process a PDF and create the vector store."""

    text = extract_text_from_pdf(uploaded_file)

    chunks = split_text_into_chunks(text)

    vectorizer = create_vectorizer()
    vectors = create_vectors(vectorizer, chunks)

    index = create_faiss_index(vectors)

    return chunks, vectorizer, index


def answer_question(question, chunks, vectorizer, index):
    """Retrieve relevant chunks and generate an answer."""

    question_vector = vectorize_question(vectorizer, question)

    relevant_chunks = search_similar_chunks(
        index,
        question_vector,
        chunks
    )

    context = "\n\n".join(relevant_chunks)

    groq_client = Groq(api_key=GROQ_API_KEY)

    answer = generate_answer(
        groq_client,
        context,
        question
    )

    return answer, relevant_chunks