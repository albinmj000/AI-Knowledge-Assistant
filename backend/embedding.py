from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer():
    """Create and return a TF-IDF vectorizer."""
    return TfidfVectorizer(stop_words="english")


def create_vectors(vectorizer, chunks):
    """Convert document chunks into TF-IDF vectors."""
    vectors = vectorizer.fit_transform(chunks)

    return vectors.astype("float32").toarray()


def vectorize_question(vectorizer, question):
    """Convert a user question into a TF-IDF vector."""
    return vectorizer.transform([question]).astype("float32").toarray()