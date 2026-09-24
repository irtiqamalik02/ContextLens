# ContextLens 🔍

**ContextLens** is an AI-powered codebase assistant that helps developers understand and query repositories using natural language. It uses **AST-based code chunking**, **vector search with Qdrant**, and **role-aware RAG** to retrieve relevant code and generate contextual answers.

## Tech Stack

* **Python / FastAPI**
* **AST-based code parsing & chunking**
* **Qdrant** — vector database
* **Embeddings** — semantic code search
* **LLM** — contextual answer generation
* **RAG** — retrieval-augmented generation
* **Docker** — local infrastructure

## How It Works

```text
Codebase
   ↓
AST-based Chunking
   ↓
Embeddings
   ↓
Qdrant
   ↓
User Query + Role
   ↓
Relevant Context
   ↓
LLM
   ↓
Role-aware Answer
```

## 🚀 Getting Started

### Prerequisites

Make sure you have:

* Python 3.10+
* Git
* Docker & Docker Compose
* Required LLM/embedding API credentials

### 1. Clone the repository

```bash
git clone https://github.com/irtiqamalik02/ContextLens.git
cd ContextLens
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file from the provided example:

```bash
cp .env.example .env
```

Update the required values such as:

```env
LLM_API_KEY=<your-api-key>
LLM_MODEL=<your-model>
EMBEDDING_MODEL=<your-embedding-model>

QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 5. Start Qdrant

```bash
docker compose up -d
```

Verify:

```bash
docker ps
```

Qdrant should be available at:

```text
http://localhost:6333
```

### 6. Index a repository

Run the project's indexing command:

```bash
python <indexer-entrypoint> --repo /path/to/repository
```

This parses the repository using AST-based chunking, generates embeddings, and stores them in Qdrant.

### 7. Start ContextLens

```bash
uvicorn <app_module>:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

## 💡 Example Questions

Once a repository is indexed, you can ask questions such as:

* `Where is authentication implemented?`
* `Explain the order creation flow.`
* `Which services interact with the payment service?`
* `Where is the database connection configured?`
* `What functionality is available for my role?`

## 📌 Key Highlights

* **AST-aware chunking** for structurally meaningful code retrieval
* **Hybrid (semantic + lexical) search** over indexed code
* **Role-aware retrieval and responses (Dev/PM/Business)**
* **Qdrant-powered vector search**
* **LLM-based contextual code understanding**
