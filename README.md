# 🚀 QA_RAG – FastAPI-based Retrieval Augmented Q&A System

## 📌 Overview
QA_RAG is a Question Answering system built using **Retrieval Augmented Generation (RAG)**.  
It combines semantic search with LLMs to generate accurate, context-aware answers.

This project exposes a **FastAPI API endpoint** to handle user queries and return intelligent responses.

---

## ⚙️ Features
- 🔍 Context retrieval using embeddings  
- 🤖 AI-generated answers using RAG pipeline  
- ⚡ FastAPI backend for real-time responses  
- 🌐 CORS enabled for frontend integration  
- 📦 Clean API structure for scalability  

---

## 🛠️ Tech Stack
- Python  
- FastAPI  
- Pydantic  
- Vector Database (FAISS / ChromaDB)  
- LLM (OpenAI / HuggingFace)  

---

## 🧠 How It Works
1. User sends a POST request with a question  
2. FastAPI receives input using Pydantic model  
3. `ask_question()` processes the query  
4. Relevant documents are retrieved  
5. LLM generates a contextual answer  
6. API returns the response  

---

## 🔌 API Endpoint

### ➤ Ask Question
**POST** `/ask`

#### Request Body:
```json
{
  "question": "What is RAG?"
}
