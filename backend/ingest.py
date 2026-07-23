import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

VECTOR_PATH = "vectorstore"
UPLOAD_DIR = "uploaded_pdfs"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def read_pdf(file_path):
    reader = PdfReader(file_path)

    pages = []

    for i, page in enumerate(reader.pages):

        page_text = page.extract_text()

        if page_text:
            pages.append(
                {
                    "page": i + 1,
                    "text": page_text
                }
            )

    return pages


def chunk_text(pages, chunk_size=1000, overlap=200):
    chunks = []

    for page in pages:

        words = page["text"].split()
        i = 0

        while i < len(words):

            chunk = " ".join(words[i:i + chunk_size])

            chunks.append(
                {
                    "text": chunk,
                    "page": page["page"]
                }
            )

            i += chunk_size - overlap

    return chunks


def create_embeddings(chunks):
    if not chunks:
        raise ValueError("No chunks found to embed!")
    return embed_model.encode(chunks, convert_to_numpy=True).astype("float32")


def build_index(pdf_path):
    filename = os.path.basename(pdf_path)

    print("📄 Reading PDF...")
    pages = read_pdf(pdf_path)

    if not pages:
        raise ValueError("No readable text found in this PDF!")

    print("✂️ Chunking text...")
    new_chunks = chunk_text(pages)

    print(f"📦 New chunks: {len(new_chunks)}")

    print("🧠 Creating embeddings...")
    new_embeddings = create_embeddings(
        [chunk["text"] for chunk in new_chunks]
    )

    faiss.normalize_L2(new_embeddings)
    dimension = new_embeddings.shape[1]

    os.makedirs(VECTOR_PATH, exist_ok=True)

    index_path = os.path.join(VECTOR_PATH, "faiss.index")
    chunks_path = os.path.join(VECTOR_PATH, "chunks.npy")
    sources_path = os.path.join(VECTOR_PATH, "sources.npy")

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        existing_chunks = list(np.load(chunks_path, allow_pickle=True))
        existing_sources = list(np.load(sources_path, allow_pickle=True))
    else:
        index = faiss.IndexFlatIP(dimension)
        existing_chunks = []
        existing_sources = []

    index.add(new_embeddings)

    all_chunks = existing_chunks + new_chunks
    all_sources = existing_sources + [filename] * len(new_chunks)

    faiss.write_index(index, index_path)
    np.save(chunks_path, np.array(all_chunks, dtype=object))
    np.save(sources_path, np.array(all_sources, dtype=object))

    print(
        f"✅ Added {len(new_chunks)} chunks from {filename}. Total chunks: {len(all_chunks)}"
    )

    return len(new_chunks), filename


# ==========================
# Delete one PDF
# ==========================

def delete_pdf(filename):

    chunks = list(np.load(f"{VECTOR_PATH}/chunks.npy", allow_pickle=True))
    sources = list(np.load(f"{VECTOR_PATH}/sources.npy", allow_pickle=True))

    keep_chunks = [c for c, s in zip(chunks, sources) if s != filename]
    keep_sources = [s for s in sources if s != filename]

    if keep_chunks:
        embeddings = create_embeddings(
    [chunk["text"] for chunk in keep_chunks]
)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
    else:
        index = faiss.IndexFlatIP(384)

    faiss.write_index(index, f"{VECTOR_PATH}/faiss.index")

    np.save(f"{VECTOR_PATH}/chunks.npy", np.array(keep_chunks, dtype=object))
    np.save(f"{VECTOR_PATH}/sources.npy", np.array(keep_sources, dtype=object))

    pdf_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    print(f"✅ {filename} deleted successfully.")


# ==========================
# Clear all PDFs
# ==========================

def clear_all():

    for file in ["faiss.index", "chunks.npy", "sources.npy"]:
        path = os.path.join(VECTOR_PATH, file)
        if os.path.exists(path):
            os.remove(path)

    if os.path.exists(UPLOAD_DIR):
        for file in os.listdir(UPLOAD_DIR):
            if file.endswith(".pdf"):
                os.remove(os.path.join(UPLOAD_DIR, file))

    print("✅ All PDFs deleted.")