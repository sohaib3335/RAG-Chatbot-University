# RAG Chatbot - University Knowledge Base Assistant

A Retrieval-Augmented Generation (RAG) chatbot implementation for university student services. This system combines semantic document retrieval with large language model generation to provide accurate, context-aware responses to student queries.

## 📋 Table of Contents

- [RAG Chatbot - University Knowledge Base Assistant](#rag-chatbot---university-knowledge-base-assistant)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Overview](#-overview)
  - [✨ Features](#-features)
  - [🏗️ System Architecture](#️-system-architecture)
  - [📦 Prerequisites](#-prerequisites)
  - [🚀 Installation](#-installation)
    - [Step 1: Clone or Download the Project](#step-1-clone-or-download-the-project)
    - [Step 2: Create a Virtual Environment](#step-2-create-a-virtual-environment)
    - [Step 3: Install Dependencies](#step-3-install-dependencies)
    - [Step 4: Configure Environment Variables](#step-4-configure-environment-variables)
  - [⚙️ Configuration](#️-configuration)
  - [📖 Usage](#-usage)
    - [1. Document Ingestion](#1-document-ingestion)
    - [2. Web Interface (Streamlit)](#2-web-interface-streamlit)
    - [3. Command Line Interface](#3-command-line-interface)
  - [📁 Project Structure](#-project-structure)
  - [🧪 Testing \& Evaluation](#-testing--evaluation)
    - [Run Unit Tests](#run-unit-tests)
    - [Run RAG Evaluation](#run-rag-evaluation)
  - [🔧 Troubleshooting](#-troubleshooting)
    - [Common Issues](#common-issues)
    - [Resetting the System](#resetting-the-system)
  - [📊 Technical Specifications](#-technical-specifications)
  - [📝 License](#-license)
  - [👤 Author](#-author)

---

## 🎯 Overview

This RAG chatbot is designed to assist university students by providing accurate answers to queries about:
- Academic policies and procedures
- Course registration and enrollment
- Financial aid and tuition
- Campus resources and services
- Student support services

The system uses a **Retrieval-Augmented Generation** approach:
1. **Retrieval**: Finds relevant documents from the knowledge base using semantic search
2. **Augmentation**: Combines retrieved context with the user's question
3. **Generation**: Uses an LLM to generate accurate, contextual responses

---

## ✨ Features

- **Multi-format Document Support**: PDF, TXT, DOCX, Markdown, CSV
- **Semantic Search**: Uses OpenAI's `text-embedding-3-large` (3072 dimensions)
- **Claude Sonnet 4**: Powered by Anthropic's latest LLM for high-quality responses
- **Persistent Vector Store**: ChromaDB with HNSW indexing for fast retrieval
- **Web Interface**: User-friendly Streamlit UI
- **Command Line Interface**: For scripting and automation
- **Source Citations**: Responses include relevant source documents
- **Configurable Parameters**: Customize chunk size, overlap, and retrieval settings

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   RAG-Based Question Answering System               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────── Offline Ingestion ──────────────────────┐ │
│  │                                                                 │ │
│  │  Documents → Loader → Chunker → Embeddings → Vector Database   │ │
│  │  (PDF)      PyMuPDF   1000/200   text-embed-3   ChromaDB       │ │
│  │                                                                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─────────────────────── Online Query ───────────────────────────┐ │
│  │                                                                 │ │
│  │  User Query → Embed → Search → Context → LLM → Response        │ │
│  │              ↓        k-NN              Claude Sonnet 4         │ │
│  │              └──────→ ChromaDB ────────→ ┘                      │ │
│  │                                                                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Prerequisites

- **Python 3.10+** (Recommended: Python 3.11)
- **API Keys**:
  - OpenAI API key (for embeddings)
  - Anthropic API key (for Claude Sonnet 4)
- **Git** (for cloning the repository)

---

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
cd "path/to/your/project/folder"
```

### Step 2: Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   copy .env.example .env
   ```
   
2. Edit `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

---

## ⚙️ Configuration

The system is configured via the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key for embeddings |
| `ANTHROPIC_API_KEY` | - | Anthropic API key for Claude |
| `USE_CLAUDE` | `true` | Use Claude Sonnet 4 (set `false` for GPT-4) |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model version |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Number of documents to retrieve |

---

## 📖 Usage

### 1. Document Ingestion

Before using the chatbot, you must ingest documents into the vector database.

**Run ingestion:**
```powershell
python main.py ingest
```

This will:
- Load all PDF documents from `university_knowledge_base_pdf/`
- Split them into chunks (1000 chars with 200 overlap)
- Generate embeddings using OpenAI's text-embedding-3-large
- Store vectors in ChromaDB (persisted to `chroma_db/`)

**Expected output:**
```
╔══════════════════════════════════════════════════════════╗
║                    RAG CHATBOT v1.0                      ║
║         Retrieval Augmented Generation System            ║
╚══════════════════════════════════════════════════════════╝

📂 Loading documents from: university_knowledge_base_pdf
📄 Found 45 documents
✂️  Splitting documents into chunks...
📊 Created 445 chunks
🔢 Generating embeddings...
💾 Storing in ChromaDB...
✅ Ingestion complete! 445 chunks indexed.
```

---

### 2. Web Interface (Streamlit)

The easiest way to interact with the chatbot is through the web interface.

**Start the web application:**
```powershell
streamlit run app.py
```

This will:
- Start a local web server
- Open your browser to `http://localhost:8501`

**Using the interface:**
1. Click **"🔄 Connect to Knowledge Base"** in the sidebar
2. Wait for the connection confirmation
3. Type your question in the chat input
4. View responses with source citations

**Example queries:**
- "What is the process for applying for financial aid?"
- "How do I appeal a grade?"
- "What are the library hours?"
- "Tell me about the counseling services available"

---

### 3. Command Line Interface

For scripting or terminal-based interaction:

**Interactive chat mode:**
```powershell
python main.py chat
```

**Single query:**
```powershell
python main.py query "What is the add/drop deadline?"
```

**CLI Commands (in interactive mode):**
| Command | Description |
|---------|-------------|
| `/stats` | Show system statistics |
| `/sources` | Toggle source display |
| `/help` | Show available commands |
| `quit` | Exit the chatbot |

---

## 📁 Project Structure

```
RAG-Chatbot/
├── .env                          # Environment variables (API keys)
├── .env.example                  # Example environment file
├── requirements.txt              # Python dependencies
├── main.py                       # CLI application entry point
├── app.py                        # Streamlit web interface
├── README.md                     # This file
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── document_loader.py        # PDF/document loading
│   ├── embeddings.py             # Embedding generation
│   ├── vector_store.py           # ChromaDB operations
│   ├── llm.py                    # LLM interface (Claude/GPT)
│   └── rag_chain.py              # RAG pipeline orchestration
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_rag_chatbot.py       # Unit tests
│   ├── evaluate_rag.py           # RAG evaluation script
│   ├── test_queries.json         # Test query set
│   └── expected_responses.json   # Expected response patterns
│
├── university_knowledge_base/    # Source documents (Markdown)
├── university_knowledge_base_pdf/ # Source documents (PDF)
│
├── chroma_db/                    # Persisted vector database
│
├── architecture_diagram_research.html   # System architecture diagram
└── ingestion_pipeline_research.html     # Ingestion pipeline diagram
```

---

## 🧪 Testing & Evaluation

### Run Unit Tests

```powershell
pytest tests/test_rag_chatbot.py -v
```

### Run RAG Evaluation

```powershell
python tests/evaluate_rag.py
```

This evaluates the system on predefined queries and measures:
- **Retrieval Accuracy**: Are relevant documents retrieved?
- **Response Quality**: Does the answer contain expected information?
- **Source Citation**: Are sources properly attributed?

---

## 🔧 Troubleshooting

### Common Issues

**1. "No API key found" error**
```
Solution: Ensure .env file exists and contains valid API keys
```

**2. "No documents in vector store"**
```
Solution: Run document ingestion first: python main.py ingest
```

**3. "ChromaDB collection not found"**
```
Solution: Delete chroma_db/ folder and re-run ingestion
```

**4. Streamlit port already in use**
```powershell
streamlit run app.py --server.port 8502
```

**5. Module import errors**
```powershell
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Resetting the System

To completely reset and start fresh:
```powershell
# Remove vector database
Remove-Item -Path chroma_db -Recurse -Force

# Re-run ingestion
python main.py ingest
```

---

## 📊 Technical Specifications

| Component | Technology | Details |
|-----------|------------|---------|
| **Embeddings** | OpenAI text-embedding-3-large | 3072 dimensions, L2 normalized |
| **Vector Store** | ChromaDB | HNSW index, persistent storage |
| **LLM** | Claude Sonnet 4 | Anthropic's latest model |
| **Document Parser** | PyMuPDF | High-fidelity PDF extraction |
| **Web Framework** | Streamlit | Interactive web interface |
| **Chunking** | RecursiveCharacterTextSplitter | 1000 chars, 200 overlap |

---

## 📝 License

This project is developed for academic purposes as part of MSc Computing coursework.

---

## 👤 Author

Developed for CBRM (Computer-Based Research Methods) Module by Sohaib Farooq

---

*Last updated: December 2025*
