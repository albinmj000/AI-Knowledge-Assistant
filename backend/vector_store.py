import faiss
import numpy as np


def create_faiss_index(vectors):
    """Create a FAISS index and add document vectors."""
    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    return index


def search_similar_chunks(index, question_vector, chunks, k=3):
    """Find the most relevant document chunks."""
    k = min(k, len(chunks))

    distances, indices = index.search(question_vector, k)

    relevant_chunks = []

    for i in indices[0]:
        relevant_chunks.append(chunks[i])

    return relevant_chunks