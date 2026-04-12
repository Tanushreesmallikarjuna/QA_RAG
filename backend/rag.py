import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VECTOR_PATH = "vectorstore"

index = faiss.read_index(os.path.join(VECTOR_PATH, "faiss.index"))
chunks = np.load(os.path.join(VECTOR_PATH, "chunks.npy"), allow_pickle=True)


# Convert query → embedding
def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(response.data[0].embedding, dtype="float32")


# Retrieve similar chunks
def retrieve(query, k=2):
    query_vector = embed_query(query).reshape(1, -1)
    distances, indices = index.search(query_vector, k)

    results = [chunks[i] for i in indices[0]]
    return results


# LLM answer generation
def ask_llm(question, context):
    prompt = f"""
Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# Main function
def ask_question(question):
    docs = retrieve(question)
    context = "\n".join(docs)
    answer = ask_llm(question, context)
    return answer