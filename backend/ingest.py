import os
import faiss
import numpy as np
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VECTOR_PATH = "vectorstore"


# ----------------------------
# 1. READ PDF SAFELY
# ----------------------------
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return text.strip()


# ----------------------------
# 2. CHUNK TEXT (WORD-BASED)
# ----------------------------
def chunk_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks


# ----------------------------
# 3. CREATE EMBEDDINGS (BATCHED)
# ----------------------------
def create_embeddings(chunks):
    if not chunks:
        raise ValueError("No chunks found to embed!")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )

    embeddings = np.array([item.embedding for item in response.data]).astype("float32")
    return embeddings


# ----------------------------
# 4. BUILD FAISS INDEX
# ----------------------------
def build_index(pdf_path):
    print("📄 Reading PDF...")
    text = read_pdf(pdf_path)

    print("✂️ Chunking text...")
    chunks = chunk_text(text)

    print(f"📦 Total chunks: {len(chunks)}")

    print("🧠 Creating embeddings...")
    embeddings = create_embeddings(chunks)

    if embeddings.shape[0] == 0:
        raise ValueError("Embedding failed!")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # Use cosine similarity index
    index = faiss.IndexFlatIP(dimension)

    # Add embeddings
    index.add(embeddings)

    # Save everything
    os.makedirs(VECTOR_PATH, exist_ok=True)

    faiss.write_index(index, f"{VECTOR_PATH}/faiss.index")
    np.save(f"{VECTOR_PATH}/chunks.npy", np.array(chunks))

    print("✅ Vector DB created successfully!")


# ----------------------------
# 5. RUN SCRIPT
# ----------------------------
if __name__ == "__main__":
    build_index(r"C:\Users\Tanushree\Desktop\QA_RAG\VK.pdf")