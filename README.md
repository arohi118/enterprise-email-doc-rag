# Enterprise Email & Document Intelligence (RAG Engine)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2+-orange.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade **Retrieval-Augmented Generation (RAG)** pipeline designed for banking emails, regulatory memos, and investment documentation[cite: 1]. 

The system implements **access-aware metadata filtering**, vector similarity search, **cross-encoder reranking**, and context-window optimization to prevent hallucinations and enforce regulatory data access boundaries[cite: 1].

---

## 🏗️ Architecture & Pipeline Flow
Raw Documents & Emails
│
▼
[Pre-processor & Chunking]  ──► Metadata Tagging (Role, Department, Clearance)
│
▼
[Vector Store (FAISS)]      ──► Embedding Generation (text-embedding-3-small)
│
User Query + User Role
│
▼
[Role-Based Metadata Filter] ──► Drops chunks exceeding caller permissions
│
▼
[Cross-Encoder Reranker]    ──► High-precision rescoring (MiniLM-L-6-v2)
│
▼
[LLM Grounded Synthesis]    ──► Context-constrained prompt to prevent hallucination
│
▼
Grounded Answer + Citations


---

## ✨ Features

- **RBAC-Aware Retrieval:** Incorporates role-based access control directly into the retrieval pipeline, ensuring users only retrieve information authorized for their clearance level[cite: 1].
- **Two-Stage Retrieval (Bi-Encoder + Cross-Encoder):** Uses vector embeddings for fast candidate retrieval followed by a cross-encoder reranker for context relevance[cite: 1].
- **Citation & Source Attribution:** Returns document IDs and classification tags alongside generated answers to guarantee auditability[cite: 1].

---

## 🛠️ Tech Stack

- **Framework:** LangChain[cite: 1]
- **Vector Engine:** FAISS (Facebook AI Similarity Search)[cite: 1]
- **Reranker:** Sentence-Transformers (`ms-marco-MiniLM-L-6-v2`)[cite: 1]
- **Embeddings & LLM:** OpenAI API (`text-embedding-3-small`, `gpt-4o-mini`)[cite: 1]
- **Language:** Python 3.11+[cite: 1]

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
git clone [https://github.com/arohi118/enterprise-email-doc-rag.git](https://github.com/arohi118/enterprise-email-doc-rag.git)
cd enterprise-email-doc-rag

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env and insert your OPENAI_API_KEY
2. Run the Demo Engine
Bash
python -m src.engine
👤 Author
Arohi Rup

LinkedIn: linkedin.com/in/arohirup

GitHub: @arohi118

