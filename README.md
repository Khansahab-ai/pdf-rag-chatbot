---
title: PDF RAG QA Chatbot
emoji: 📑
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 📄 PDF RAG Q&A Chatbot

A production-oriented **Retrieval-Augmented Generation (RAG)** application for asking questions about PDF documents.

This project implements an end-to-end RAG pipeline that combines **structure-aware PDF processing, semantic retrieval, BM25 lexical retrieval, hybrid search, Reciprocal Rank Fusion (RRF), cross-encoder reranking, context construction, and Gemini-based answer generation**.

The application is exposed through a **FastAPI backend**, has a responsive web frontend, and can be containerized using **Docker**.

---

## 🚀 Project Overview

Traditional LLM applications can generate answers that are not grounded in a specific document.

This project addresses that problem by retrieving relevant information from a PDF before generating an answer.

The system follows this pipeline:

```text
                         PDF
                          │
                          ▼
                        MinerU
                          │
                          ▼
              Structured Extraction
                          │
                          ▼
                Cleaning & Validation
                          │
                          ▼
             Structure-Aware Chunking
                          │
                          ▼
                       Indexing
                    ┌─────┴─────┐
                    │           │
                    ▼           ▼
                  BGE-M3      BM25
                Embeddings   Retrieval
                    │           │
                    ▼           ▼
                 Chroma      BM25 Index
                    │           │
                    └─────┬─────┘
                          ▼
                  Hybrid Retrieval
                          │
                          ▼
                         RRF
                          │
                          ▼
                Cross-Encoder Reranker
                          │
                          ▼
                    Top Results
                          │
                          ▼
                   Context Builder
                          │
                          ▼
                       Gemini
                          │
                          ▼
                      FastAPI
                          │
                          ▼
                      Frontend
                          │
                          ▼
                       Docker
```

---

## ✨ Features

- 📄 PDF document processing
- 🧩 Structure-aware document extraction
- 🧹 Document cleaning and validation
- ✂️ Structure-aware chunking
- 🔎 Dense semantic retrieval
- 🔤 BM25 lexical retrieval
- 🔀 Hybrid retrieval
- 📊 Reciprocal Rank Fusion (RRF)
- 🎯 Cross-encoder reranking
- 🧠 Context-aware answer generation
- 🤖 Gemini LLM integration
- ⚡ FastAPI REST API
- 🌐 Responsive web frontend
- 📝 Markdown answer rendering
- 📐 LaTeX equation rendering using MathJax
- 🐳 Docker support
- ❤️ Health-check endpoint
- 📈 Retrieval evaluation using Recall@K and MRR
- 🔐 Environment-variable based API key management

---

# 🧠 RAG Architecture

The application is divided into several stages.

## 1. PDF Processing

The project uses **MinerU** to extract structured information from PDF documents.

Instead of treating a PDF as a simple block of text, the extraction process preserves document structure such as:

- Headings
- Paragraphs
- Tables
- Figures
- Equations
- Page numbers
- Section hierarchy

Example:

```text
3 Model Architecture
    3.1 Encoder and Decoder Stacks
    3.2 Attention
    3.3 Position-wise Feed-Forward Networks
    3.4 Embeddings and Softmax
    3.5 Positional Encoding
```

This structure is later used during chunking and retrieval.

---

## 2. Document Cleaning

The extracted elements are cleaned and validated before being converted into LangChain documents.

Important metadata is preserved:

```text
source
page
content_type
element_id
section_path
document_title
```

This allows the application to provide source information together with generated answers.

---

## 3. Structure-Aware Chunking

Instead of blindly splitting text using a fixed character count, the project uses the structure of the document.

The chunker understands elements such as:

- Text
- Headings
- Tables
- Figures
- Equations

Chunk metadata contains information such as:

```text
chunk_id
section_path
content_types
pages
element_ids
chunk_size
image_paths
```

This helps preserve the semantic relationship between content and its original document section.

---

## 4. Embeddings

The project uses:

```text
BAAI/bge-m3
```

for semantic embeddings.

The embedding model converts text into numerical vectors.

For example:

```text
"What is positional encoding?"
                │
                ▼
        BGE-M3 Embedding
                │
                ▼
       [0.021, -0.183, ...]
```

Semantically similar content will have similar vector representations.

---

## 5. Chroma Vector Database

The generated embeddings are stored in **Chroma**.

The vector database allows the system to perform semantic similarity search.

Example:

```text
User Query
    │
    ▼
Embedding
    │
    ▼
Chroma Similarity Search
    │
    ▼
Top-K Documents
```

---

## 6. BM25 Retrieval

The project also implements **BM25 lexical retrieval**.

BM25 is useful when exact words and terminology are important.

For example, if the document contains:

```text
"scaled dot-product attention"
```

and the user asks:

```text
"How does scaled dot-product attention work?"
```

BM25 can strongly match the important terms.

The system therefore uses both:

```text
Semantic Search
      +
Keyword Search
```

---

## 7. Hybrid Retrieval

The system combines:

```text
Dense Retrieval
      +
BM25 Retrieval
```

The purpose is to benefit from both semantic similarity and exact keyword matching.

The retrieval pipeline is:

```text
                 User Query
                     │
            ┌────────┴────────┐
            ▼                 ▼
      Dense Retrieval      BM25 Retrieval
            │                 │
            ▼                 ▼
       Ranked Results     Ranked Results
            │                 │
            └────────┬────────┘
                     ▼
              RRF Fusion
                     │
                     ▼
             Combined Ranking
```

---

## 8. Reciprocal Rank Fusion (RRF)

The project uses **Reciprocal Rank Fusion** to combine the rankings produced by different retrieval methods.

Conceptually:

```text
RRF Score = Σ 1 / (k + rank)
```

where `k` is the RRF constant.

A document that appears near the top of multiple retrieval systems receives a stronger combined ranking.

For example:

```text
Dense Retrieval       BM25
----------------      ----------------
Document A → Rank 1  Document B → Rank 1
Document B → Rank 3  Document A → Rank 2
Document C → Rank 4  Document C → Rank 4
```

RRF combines these rankings into a single candidate list.

---

## 9. Cross-Encoder Reranking

After hybrid retrieval, the candidate documents are passed to:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The model evaluates:

```text
(query, document)
```

pairs.

Example:

```text
Query:
"What is positional encoding?"

        +

Candidate Document:
"Positional encodings are added to the input
embeddings..."

        │
        ▼
Cross Encoder
        │
        ▼
Relevance Score
```

The candidates are then sorted according to their reranker scores.

The final top results are passed to the context builder.

---

## 10. Context Building

The selected documents are converted into a structured context.

Each source contains information such as:

```text
Source
Chunk ID
Page
Section
Content
```

Example:

```text
[SOURCE 1]
Chunk ID: 12
Page: 6
Section: 3 Model Architecture > 3.5 Positional Encoding

Positional encodings are added to the input embeddings...
```

This context is then supplied to the LLM.

---

## 11. Gemini Answer Generation

The project uses Gemini for answer generation.

The LLM receives:

```text
System Instructions
        +
User Question
        +
Retrieved Context
```

The system prompt instructs the model to:

- Answer using the supplied context
- Avoid unsupported information
- State when the context is insufficient
- Cite source blocks
- Preserve equations using LaTeX
- Avoid inventing source metadata

This helps keep answers grounded in the retrieved document.

---

# 📊 Retrieval Evaluation

A manually verified evaluation dataset containing **8 questions** was used to compare retrieval approaches.

## Dense Retrieval

```text
Recall@1 : 0.375
Recall@3 : 0.625
Recall@5 : 0.625
MRR      : 0.479
```

## BM25

```text
Recall@1 : 0.625
Recall@3 : 0.750
Recall@5 : 1.000
MRR      : 0.723
```

## Hybrid RRF

```text
Recall@1 : 0.375
Recall@3 : 0.875
Recall@5 : 0.875
MRR      : 0.583
```

> **Note:** The current evaluation dataset contains only 8 questions, so these results represent an initial retrieval experiment rather than a comprehensive benchmark.

---

# 🛠️ Technology Stack

## AI / Machine Learning

- Python
- LangChain
- Sentence Transformers
- BGE-M3
- BM25
- Cross-Encoder
- Gemini

## Document Processing

- MinerU

## Vector Database

- Chroma

## Backend

- FastAPI
- Uvicorn
- Pydantic

## Frontend

- HTML
- CSS
- JavaScript
- Marked.js
- DOMPurify
- MathJax

## Deployment

- Docker
- Docker Desktop

---

# 📁 Project Structure

```text
pdf-rag-chatbot/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── index_documents.json
│
├── frontend/
│   └── index.html
│
├── src/
│   ├── __init__.py
│   ├── bm25_retriever.py
│   ├── chunker.py
│   ├── cleaning.py
│   ├── context_builder.py
│   ├── documents.py
│   ├── embeddings.py
│   ├── evaluate.py
│   ├── evaluate_bm25.py
│   ├── evaluate_dense.py
│   ├── evaluate_hybrid.py
│   ├── evaluate_retrieval.py
│   ├── evaluation_dataset.py
│   ├── hybrid_retriever.py
│   ├── indexing.py
│   ├── ingestion.py
│   ├── llm.py
│   ├── reranker.py
│   ├── retriever.py
│   ├── vectorstore.py
│   └── verify_ground_truth.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Prerequisites

Make sure you have installed:

- Python 3.12
- Git
- Docker Desktop (optional)
- A Gemini API key

---

# 💻 Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Khansahab-ai/pdf-rag-chatbot.git
```

Move into the project:

```bash
cd pdf-rag-chatbot
```

---

## 2. Create Virtual Environment

Windows:

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a file named:

```text
.env
```

in the project root.

Add:

```env
GEMINI_API_KEY=your_gemini_api_key
```

**Never commit the `.env` file to GitHub.**

The repository's `.gitignore` excludes `.env`.

---

# ▶️ Run the Backend

Start FastAPI:

```powershell
uvicorn api.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

---

# 📚 FastAPI Documentation

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

OpenAPI schema:

```text
http://127.0.0.1:8000/openapi.json
```

---

# ❤️ Health Check

Request:

```http
GET /health
```

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok",
  "message": "RAG API is running"
}
```

---

# ❓ Ask a Question

Endpoint:

```http
POST /ask
```

Request:

```json
{
  "query": "What is positional encoding and why is it needed?"
}
```

Example PowerShell request:

```powershell
$body = @{
    query = "What is positional encoding and why is it needed?"
} | ConvertTo-Json

$response = Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/ask" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

$response | ConvertTo-Json -Depth 10
```

---

# 🌐 Run the Frontend

The frontend is located at:

```text
frontend/index.html
```

Start a local web server:

```powershell
python -m http.server 3000 --directory frontend
```

Open:

```text
http://127.0.0.1:3000
```

The frontend communicates with:

```text
http://127.0.0.1:8000/ask
```

---

# 🐳 Docker

## Build the Image

```powershell
docker build -t pdf-rag-api .
```

## Run the Container

```powershell
docker run --env-file .env --name pdf-rag-container -p 8000:8000 pdf-rag-api
```

The API will be available at:

```text
http://localhost:8000
```

---

# 🩺 Docker Health Check

The API exposes:

```text
GET /health
```

Example:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "message": "RAG API is running"
}
```

---

# 🔐 Security

The project intentionally does not commit sensitive or generated files.

The `.gitignore` excludes:

```text
.env
.venv/
chroma_db/
mineru_output/
*.ipynb
```

API keys should always be supplied through environment variables or a secure secret manager.

Never hard-code API keys inside Python files.

---

# 🧩 API Architecture

The FastAPI application contains two main endpoints:

```text
GET /health
     │
     ▼
Health Status


POST /ask
     │
     ▼
Hybrid Retrieval
     │
     ▼
Reranking
     │
     ▼
Context Building
     │
     ▼
Gemini
     │
     ▼
Answer + Sources
```

---

# 📌 Example Query

```text
What is positional encoding and why is it needed?
```

The system:

```text
1. Converts the query into an embedding
2. Performs dense retrieval
3. Performs BM25 retrieval
4. Combines rankings using RRF
5. Generates candidate documents
6. Reranks candidates using a cross-encoder
7. Selects the final context
8. Sends the context to Gemini
9. Generates a grounded answer
10. Returns the answer and source metadata
```

---

# 📈 Current System Configuration

| Component | Technology |
|---|---|
| PDF Processing | MinerU |
| Document Framework | LangChain |
| Embedding Model | BAAI/bge-m3 |
| Vector Database | Chroma |
| Lexical Retrieval | BM25 |
| Hybrid Fusion | RRF |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Gemini |
| Backend | FastAPI |
| Frontend | HTML / CSS / JavaScript |
| Containerization | Docker |

---

# ⚠️ Current Limitations

### 1. Evaluation Dataset

The current retrieval evaluation contains only 8 questions.

A larger benchmark would provide a more reliable evaluation.

### 2. CPU Inference

The embedding and reranking models currently run on CPU.

### 3. Startup Time

The RAG components are loaded when the FastAPI application starts.

This includes:

- Embedding model
- Chroma
- BM25
- Cross-encoder
- Gemini client

### 4. Document Ingestion

The current repository contains a prepared indexing pipeline.

A complete production version could expose PDF upload and background ingestion through the API.

### 5. Production CORS

The current local configuration allows broad CORS access.

For production deployment, CORS should be restricted to the actual frontend domain.

---

# 🔮 Future Improvements

- [ ] Multi-PDF upload
- [ ] User-specific document collections
- [ ] Streaming LLM responses
- [ ] Improved source citation interface
- [ ] Larger retrieval evaluation dataset
- [ ] Automated RAG evaluation
- [ ] Authentication and authorization
- [ ] Production vector database
- [ ] GPU inference
- [ ] Background document ingestion
- [ ] Async document processing
- [ ] Better table retrieval
- [ ] Better figure retrieval
- [ ] Conversation history
- [ ] Document management UI
- [ ] Observability and tracing
- [ ] Cloud deployment
- [ ] Production secret management

---

# 🎯 What This Project Demonstrates

This project demonstrates practical implementation of:

```text
PDF Processing
      ↓
Document Understanding
      ↓
Structure-Aware Chunking
      ↓
Embeddings
      ↓
Vector Search
      ↓
BM25
      ↓
Hybrid Retrieval
      ↓
RRF
      ↓
Cross-Encoder Reranking
      ↓
Context Engineering
      ↓
LLM Generation
      ↓
FastAPI
      ↓
Docker
```

It therefore covers the major components required to build an end-to-end RAG application rather than only calling an LLM API.

---

# 👨‍💻 Author

**Abuzar Khan**

GitHub:

https://github.com/Khansahab-ai

Project:

https://github.com/Khansahab-ai/pdf-rag-chatbot

---

# 📄 License

This project is currently provided for educational and portfolio purposes.