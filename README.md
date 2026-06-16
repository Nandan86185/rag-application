# 🧠 RAG Knowledge Assistant

An intelligent, full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload documents and seamlessly query them using AI. The system chunks, embeds, and stores document data in a local vector database to provide context-aware, accurate answers.

## 🚀 Features
- **Document Ingestion:** Upload text/PDF documents and automatically parse and chunk the data.
- **Semantic Search:** Uses a local vector database (ChromaDB) and embedding models to perform high-accuracy similarity search.
- **AI Agent Integration:** Powered by an intelligent agent using CrewAI and Groq for orchestrating multi-step reasoning and answering complex queries.
- **Interactive UI:** A modern, responsive Angular frontend for chatting with the AI and uploading knowledge bases.
- **RESTful API:** A fast, asynchronous backend powered by FastAPI.

## 🛠️ Tech Stack
**Frontend:**
- Angular (TypeScript, SCSS)
- RxJS for state management and async data handling

**Backend:**
- Python 3.11+
- FastAPI (REST API Framework)
- ChromaDB (Local Vector Database)
- CrewAI & Groq (LLM Orchestration)
- HuggingFace / Local Embedding Models

## 📂 Project Structure
```text
rag-application/
│
├── backend/               # FastAPI Python Backend
│   ├── main.py            # API Entrypoint & Routing
│   ├── services/          # Core RAG Logic
│   │   ├── agent.py       # CrewAI Agent Orchestration
│   │   ├── chunker.py     # Semantic Text Chunking
│   │   ├── embedder.py    # Vector Embedding Generation
│   │   ├── llm.py         # LLM Connections (Groq)
│   │   ├── parser.py      # Document Parsing
│   │   └── vector_store.py# ChromaDB Interface
│   └── .env               # Environment variables (API Keys)
│
└── frontend/              # Angular Web Application
    ├── src/app/           # Components & Services
    └── package.json       # Node Dependencies
