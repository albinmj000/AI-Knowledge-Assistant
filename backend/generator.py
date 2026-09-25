from groq import Groq


def generate_answer(groq_client, context, question):
    """Generate an answer using the retrieved document context."""

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

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You answer questions based only on provided document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content