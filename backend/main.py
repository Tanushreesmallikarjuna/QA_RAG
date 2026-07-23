import os
import shutil
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import build_index, delete_pdf, clear_all
from rag import ask_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_pdfs"
VECTOR_PATH = "vectorstore"

os.makedirs(UPLOAD_DIR, exist_ok=True)


class Question(BaseModel):
    question: str
    history: list = []


# ----------------------------
# Upload PDF
# ----------------------------
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunk_count, filename = build_index(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )

    return {
        "message": f"{filename} processed successfully",
        "chunks_created": chunk_count
    }


# ----------------------------
# Ask Question
# ----------------------------
@app.post("/ask")
def ask(q: Question):
    try:
        return ask_question(
            q.question,
            history=q.history
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet."
        )


# ----------------------------
# Status
# ----------------------------
@app.get("/status")
def get_status():

    sources_path = os.path.join(VECTOR_PATH, "sources.npy")
    chunks_path = os.path.join(VECTOR_PATH, "chunks.npy")

    if not os.path.exists(sources_path):
        return {
            "total_pdfs": 0,
            "total_chunks": 0,
            "files": []
        }

    sources = np.load(sources_path, allow_pickle=True)
    chunks = np.load(chunks_path, allow_pickle=True)

    files = sorted(list(set(sources.tolist())))

    return {
        "total_pdfs": len(files),
        "total_chunks": len(chunks),
        "files": files
    }


# ----------------------------
# Delete One PDF
# ----------------------------
@app.delete("/delete/{filename}")
def remove_pdf(filename: str):

    try:
        delete_pdf(filename)
        return {
            "message": f"{filename} deleted successfully."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ----------------------------
# Clear All PDFs
# ----------------------------
@app.delete("/clear")
def clear_database():

    try:
        clear_all()
        return {
            "message": "All PDFs deleted successfully."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )