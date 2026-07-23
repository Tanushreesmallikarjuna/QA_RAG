# 📚 StudyMate AI — Multi-Document RAG Q&A System

A Retrieval-Augmented Generation (RAG) system that lets you upload multiple PDFs and ask natural-language questions across all of them — without needing to know which document contains the answer. Built end-to-end with a FastAPI backend, Streamlit frontend, local embeddings, and a free LLM API.

## 🎯 Problem It Solves

When studying from multiple source documents (module notes, resumes, reports, question banks), you often don't know *which* file has the answer to a specific question. StudyMate AI solves this by indexing all uploaded PDFs into a single searchable knowledge base — ask anything, and it automatically finds the right document, the right page, and gives a cited, grounded answer.

## ✨ Features

- **Multi-PDF upload** — upload documents one at a time; all become instantly searchable together
- **Cross-document semantic search** — ask a question without specifying which PDF it belongs to
- **Source citations with page numbers** — every answer shows exactly which file and page it came from, reducing hallucination risk
- **Conversation memory** — ask natural follow-up questions; the system remembers prior turns in the session
- **Document management** — delete individual PDFs or clear the entire knowledge base
- **Diversity-aware retrieval** — prevents one document from dominating results when multiple files are indexed
- **Free, local embeddings** — no per-query API cost for the retrieval step
- **Clean, custom-styled UI** — built with a professional color scheme, not default Streamlit styling

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI |
| Frontend | Streamlit |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) — runs locally, no API cost |
| Vector Search | FAISS (cosine similarity, IndexFlatIP) |
| LLM (answer generation) | Groq API (`llama-3.1-8b-instant`) |
| PDF Parsing | pypdf |

## 🧠 How It Works

1. **Ingestion** — uploaded PDFs are parsed, split into overlapping word-based chunks, and embedded locally using Sentence-Transformers
2. **Indexing** — embeddings are normalized and stored in a FAISS index, with parallel metadata tracking each chunk's source file and page number
3. **Retrieval** — a question is embedded and matched against all indexed chunks; results are filtered for diversity across documents (capping how many chunks a single file can contribute)
4. **Generation** — retrieved chunks + conversation history are passed to Groq's LLM to generate a grounded, context-aware answer
5. **Citation** — the response is returned alongside the exact source file(s) and page number(s) used

## 📁 Project Structure
