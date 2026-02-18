"""
Funtions for Building Chioma AI Model
Author: Ugochukwu Febechukwu

"""
#Imported libraries
# import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.document_loaders import TextLoader
import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
import re
# from langchain.schema import Document

# 1. Load the embedding model (same one used during ingestion)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
    )


from dotenv import load_dotenv
load_dotenv()
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
DB_COLLECTION_NAME = os.getenv("DB_COLLECTION_NAME")


def clean_text(text: str) -> str:
    """
    Clean text before insertion to remove invalid or binary characters.
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace('\x00', '')

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove other non-printable characters
    text = ''.join(ch for ch in text if ch.isprintable())

    return text.strip()

# Chunk files
def chunk_research_paper(paper_content, title):
    """Break a research paper into searchable chunks"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # ~200 words per chunk
        chunk_overlap=200,        # Overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(paper_content)
    
    # Add metadata to each chunk
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "content": chunk,
            "title": title,
            "chunk_id": f"{title}_{i}",
        })
    
    return chunk_data

# Embed chunked files


# def embed_documents(documents: list[str]) -> list[list[float]]:
#     """
#     Embed documents using a model.
#     """
#     device = (
#         "cuda"
#         if torch.cuda.is_available()
#         else "mps" if torch.backends.mps.is_available() else "cpu"
#     )
#     model = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": device},
#     )

#     embeddings = model.embed_documents(documents)
#     return embeddings

# from langchain.vectorstores.pgvector import PGVector
# from langchain.embeddings import HuggingFaceEmbeddings
# import torch

def embed_documents(documents: list[str]) -> list[list[float]]:
    """
    Embed documents using a model.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    embeddings = model.embed_documents(documents)
    return embeddings, model  # Return model for LangChain integration


def store_in_pgvector(documents: list[str], collection_name: str):
    """
    Store embedded documents into PostgreSQL with pgvector.
    """
    # Generate embeddings and the embedding model
    embeddings, model = embed_documents(documents)

    # Initialize PGVector and store
    db = PGVector.from_documents(
        embedding=model,          # Pass the embedding model, not the list of embeddings
        documents=documents,      # Texts to store
        collection_name=DB_COLLECTION_NAME,
        connection_string=DB_CONNECTION_STRING,
    )

    print(f"Stored {len(documents)} documents in collection '{collection_name}'")

    return db

# -------------------------------
# 1. Setup Embedding Model (no cache)
# -------------------------------
def embedder():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="/tmp/no_cache",
        model_kwargs={"device": device},
    )

# -------------------------------
# 2. Former Insert Documents into PGVector function
# -------------------------------
# from langchain.vectorstores import PGVector
# from langchain.docstore.document import Document

# def insert_publications(publications: list[dict]):
#     """
#     Insert documents into a pgvector collection (table).
#     Each publication is chunked and stored with metadata.
#     """

#     # Initialize embeddings and connection
#     embeddings = embedder()


#     all_docs = []
#     for publication in publications:
#         title = publication["title"]
#         content = publication["content"]
#         chunked_publication = chunk_research_paper(content, title)

#         # Convert each chunk into a LangChain Document
#         for chunk in chunked_publication:
#             doc = Document(
#                 page_content=chunk["content"],
#                 metadata={"title": title}
#             )
#             all_docs.append(doc)
    

#     # Clean all text and metadata before insertion
#     cleaned_docs = []
#     for pub in publications:
#         content = clean_text(pub.get("content", ""))
#         title = clean_text(pub.get("title", "Untitled"))

#         # Convert all metadata values to clean strings
#         metadata = {k: clean_text(str(v)) for k, v in pub.items() if k != "content"}

#         cleaned_docs.append(Document(page_content=content, metadata=metadata))

#     # ✅ Correct way to initialize PGVector and store documents
#     db = PGVector.from_documents(
#         documents=all_docs,
#         embedding=embeddings,
#         connection_string=DB_CONNECTION_STRING,
#         collection_name=DB_COLLECTION_NAME
#     )

#     print(f"✅ Inserted {len(all_docs)} documents into pgvector collection '{DB_COLLECTION_NAME}'")
#     return db


def generate_content_hash(text: str, title: str) -> str:
    """
    Generate a stable hash based on title + content.
    Prevents duplicate chunks across re-ingestion.
    """
    normalized = f"{title.strip().lower()}::{text.strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

import hashlib
from sqlalchemy import create_engine, text
from langchain_community.vectorstores import PGVector
from langchain.schema import Document

def insert_publications(publications: list[dict]):
    """
    Insert documents into a pgvector collection (table) without duplicates.
    Each publication is chunked and stored with metadata.
    """

    embeddings = embedder()
    
    # Create engine for database operations
    engine = create_engine(DB_CONNECTION_STRING)
    
    # Get existing content hashes - using JSONB-compatible approach for JSON column
    existing_hashes = set()
    try:
        with engine.connect() as conn:
            # Check if collection exists
            collection = conn.execute(
                text("SELECT uuid FROM langchain_pg_collection WHERE name = :name"),
                {"name": DB_COLLECTION_NAME}
            ).first()
            
            if collection:
                collection_id = collection[0]
                print(f"✅ Found existing collection with ID: {collection_id}")
                
                # Count existing documents
                count_result = conn.execute(
                    text("SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = :collection_id"),
                    {"collection_id": collection_id}
                ).scalar()
                print(f"📊 Collection has {count_result} existing documents")
                
                # For JSON column, we need to use different operators
                # Extract content_hash from metadata using ->> operator (works with both JSON and JSONB)
                hash_results = conn.execute(
                    text("""
                        SELECT cmetadata->>'content_hash' 
                        FROM langchain_pg_embedding 
                        WHERE collection_id = :collection_id
                        AND cmetadata->>'content_hash' IS NOT NULL
                    """),
                    {"collection_id": collection_id}
                ).fetchall()
                
                existing_hashes = set(row[0] for row in hash_results if row[0])
                print(f"✅ Loaded {len(existing_hashes)} existing content hashes from database")
                
                # Show sample if any exist
                if existing_hashes:
                    sample = list(existing_hashes)[:3]
                    print(f"🔍 Sample existing hashes: {[h[:8] + '...' for h in sample]}")
    except Exception as e:
        print(f"⚠️ Error loading existing hashes (likely first run or no content_hash field): {e}")
        # If error occurs (like column doesn't have content_hash yet), proceed with empty set
    
    # Prepare documents
    all_docs = []
    new_chunks = 0
    skipped_chunks = 0

    for publication in publications:
        title = clean_text(publication.get("title", "Untitled"))
        content = clean_text(publication.get("content", ""))

        # Skip empty or unreadable content
        if not content.strip():
            print(f"⚠️ Skipping empty document: {title}")
            continue

        chunked_publication = chunk_research_paper(content, title)
        print(f"📄 Processing '{title}': {len(chunked_publication)} chunks")

        for chunk in chunked_publication:
            clean_chunk = clean_text(chunk["content"])
            
            # Generate a unique ID based on content hash
            content_hash = hashlib.sha256(clean_chunk.encode()).hexdigest()
            
            # Skip if already exists
            if content_hash in existing_hashes:
                skipped_chunks += 1
                continue
            
            doc = Document(
                page_content=clean_chunk,
                metadata={
                    "title": title,
                    "content_hash": content_hash
                }
            )
            all_docs.append(doc)
            new_chunks += 1
            existing_hashes.add(content_hash)  # Prevent duplicates within same batch

    print(f"\n📊 Summary:")
    print(f"  - Total chunks processed: {new_chunks + skipped_chunks}")
    print(f"  - New chunks to insert: {new_chunks}")
    print(f"  - Existing chunks skipped: {skipped_chunks}")

    if not all_docs:
        print("✅ No new documents to insert")
        return None

    print(f"\n📤 Inserting {new_chunks} new chunks into database...")
    
    # Create the vector store and add documents
    try:
        db = PGVector.from_documents(
            documents=all_docs,
            embedding=embeddings,
            connection_string=DB_CONNECTION_STRING,
            collection_name=DB_COLLECTION_NAME,
            pre_delete_collection=False
        )
        print(f"✅ Successfully inserted {new_chunks} documents into pgvector collection '{DB_COLLECTION_NAME}'")
        return db
    except Exception as e:
        print(f"❌ Error inserting documents: {e}")
        return None

# -------------------------------
# 3. Search the pgvector Database
# -------------------------------
def search_research_db(query: str, db: PGVector, top_k=5):
    """
    Search pgvector for top-k most similar chunks.
    """
    results = db.similarity_search(query, k=top_k)

    relevant_chunks = []
    for r in results:
        relevant_chunks.append({
            "content": r.page_content,
            "title": r.metadata.get("title", "Unknown"),
            # pgvector doesn’t return cosine distance directly, so we skip similarity calc
        })

    return relevant_chunks


# -------------------------------
# 4. Generate Answer Based on Retrieved Research
# -------------------------------
def answer_research_question(query: str, db: PGVector, llm):
    """
    Generate an answer using the top 3 most relevant research chunks.
    """
    relevant_chunks = search_research_db(query, db, top_k=3)

    # Build context for LLM
    context = "\n\n".join([
        f"From {chunk['title']}:\n{chunk['content']}"
        for chunk in relevant_chunks
    ])

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Based on the following research findings, answer the student's question:

        Research Context:
        {context}

        Researcher's Question: {question}

        Answer: Provide a comprehensive answer based on the findings above.
        """
    )

    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content, relevant_chunks
