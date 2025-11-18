# RAG PDF Embedding Pipeline (Dockerized)

This project provides an end-to-end pipeline for processing PDF
documents, extracting text, generating embeddings using
SentenceTransformers, and storing them in a PostgreSQL database enabled
with the **pgvector** extension. The entire application is fully
**dockerized**, ensuring easy deployment and reproducibility.

------------------------------------------------------------------------

## 🚀 Features

-   Extract text from PDF documents
-   Chunk text optimally for embedding
-   Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
-   Insert embeddings into a PostgreSQL + pgvector database
-   Track processed PDFs to avoid duplicate insertion
-   Dockerized application for production-ready deployment
-   Uses Docker volumes for persistent storage of caches and tracking
    files

------------------------------------------------------------------------

## 📁 Project Structure

    .
    ├── app.py                    # FastAPI app (if applicable)
    ├── jac_functions.py          # PDF processing & embedding functions
    ├── insert_pg_2.py            # Script to insert embeddings into pgvector
    ├── data/                     # Directory containing PDFs
    ├── storage/                  # Persistent folder for processed_pdfs.json
    ├── requirements.txt          # Python dependencies
    ├── Dockerfile                # Container setup
    └── docker-compose.yml        # Multi-service orchestration

------------------------------------------------------------------------

## 🐳 Dockerized Workflow

### 1. Build the Image

``` bash
docker compose build
```

### 2. Run the Container

``` bash
docker compose up -d
```

### 3. Persistent Volumes

The app uses volumes to: - Persist HuggingFace and SentenceTransformer
model caches - Persist `processed_pdfs.json` so PDFs are not
reprocessed - Map `data/` to the container so PDFs can be scanned inside
Docker

Example from `docker-compose.yml`:

``` yaml
volumes:
  huggingface_cache:
  sentence_transformers_cache:
```

### 4. Environment Variables

The database connection values are passed into Docker:

``` yaml
environment:
  - DB_CONNECTION_STRING=${DB_CONNECTION_STRING}
  - DB_COLLECTION_NAME=${DB_COLLECTION_NAME}
```

These should be defined in your local `.env` file.

------------------------------------------------------------------------

## 🗄 How Processed PDFs Are Tracked

A file named `processed_pdfs.json` is stored inside the container under:

    /app/storage/processed_pdfs.json

This prevents duplicate insertions. This folder is mounted to your host
for persistence:

    ./storage:/app/storage

------------------------------------------------------------------------

## 💾 Database Requirements

Ensure PostgreSQL has the pgvector extension installed:

``` sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Your table must include a `vector` column.

------------------------------------------------------------------------

## ▶️ Running the Embedding Insert Script Manually

``` bash
docker exec -it rag-app python insert_pg_2.py
```

------------------------------------------------------------------------

## 📦 Rebuilding After Code Changes

Anytime you modify source code:

``` bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

------------------------------------------------------------------------

## 📝 Summary

This project provides a production-ready, Dockerized pipeline for
embedding PDFs into pgvector. It handles caching, tracking, chunking,
embeddings, and persistence cleanly using Docker volumes and a robust
architecture.

------------------------------------------------------------------------

## 🧑‍💻 Author

Ugochukwu Febechukwu
