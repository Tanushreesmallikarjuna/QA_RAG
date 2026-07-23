import os
import faiss
import numpy as np
from fastembed import TextEmbedding
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

VECTOR_PATH = "vectorstore"


def load_index():
    index_path = os.path.join(VECTOR_PATH, "faiss.index")
    chunks_path = os.path.join(VECTOR_PATH, "chunks.npy")
    sources_path = os.path.join(VECTOR_PATH, "sources.npy")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError("No index found. Please upload a PDF first.")

    index = faiss.read_index(index_path)
    chunks = np.load(chunks_path, allow_pickle=True)
    sources = np.load(sources_path, allow_pickle=True)
    return index, chunks, sources


def embed_query(query):
    embedding = list(embed_model.embed([query]))[0]
    vector = np.array(embedding, dtype="float32").reshape(1, -1)
    return vector


def retrieve(query, k=5):
    index, chunks, sources = load_index()

    query_vector = embed_query(query)
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, k)

    results = [chunks[i] for i in indices[0]]
    result_sources = [sources[i] for i in indices[0]]

    return results, result_sources


def ask_llm(question, context, history=None):
    history_text = ""

    if history:
        for turn in history[-3:]:
            history_text += (
                f"Previous Q: {turn['question']}\n"
                f"Previous A: {turn['answer']}\n\n"
            )

    prompt = f"""
Use the following context and conversation history to answer the question.
If the answer is not present, say you don't know.

{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def ask_question(question, history=None):
    docs, doc_sources = retrieve(question)

    if not docs:
        return {
            "answer": "No relevant content found.",
            "sources": []
        }

    context = "\n\n".join(chunk["text"] for chunk in docs)

    answer = ask_llm(question, context, history)

    sources = []
    for chunk, file in zip(docs, doc_sources):
        sources.append(
            {
                "file": file,
                "page": chunk["page"],
                "text": chunk["text"]
            }
        )

    return {
        "answer": answer,
        "sources": sources
    }