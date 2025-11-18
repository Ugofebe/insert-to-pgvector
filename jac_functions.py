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

# from langchain_postgres import PGVector

# 2. Connect to your persistent ChromaDB folder
# client = chromadb.PersistentClient(path="./research_db")
# collection = client.get_or_create_collection(
#     name="ml_publications",
#     metadata={"hnsw:space": "cosine"}
#)
from dotenv import load_dotenv
load_dotenv()
# CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
DB_COLLECTION_NAME = os.getenv("ml_publications")

# DB_CONNECTION_STRING = DB_CONNECTION_STRING
# DB_COLLECTION_NAME = DB_COLLECTION_NAME

# db = PGVector.from_documents(
#     embedding=embeddings,
#     documents=texts,
#     collection_name=COLLECTION_NAME,
#     connection_string=CONNECTION_STRING,
# )
# 4. Get retriever for RAG
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# # Set up our embedding model
# embeddings = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2"
#     )
# def clean_text(text):
#     if not isinstance(text, str):
#         return text
#     # Remove null bytes and other non-printable characters
#     return text.replace('\x00', '').strip()
# import re

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
# import torch
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores.pgvector import PGVector
# from langchain.docstore.document import Document
# from langchain.prompts import PromptTemplate

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
# 2. Insert Documents into PGVector
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



def insert_publications(publications: list[dict]):
    """
    Insert documents into a pgvector collection (table).
    Each publication is chunked and stored with metadata.
    """

    embeddings = embedder()
    all_docs = []

    for publication in publications:
        title = clean_text(publication.get("title", "Untitled"))
        content = clean_text(publication.get("content", ""))

        # Skip empty or unreadable content
        if not content.strip():
            print(f"⚠️ Skipping empty document: {title}")
            continue

        chunked_publication = chunk_research_paper(content, title)

        for chunk in chunked_publication:
            clean_chunk = clean_text(chunk["content"])

            doc = Document(
                page_content=clean_chunk,
                metadata={"title": title}
            )
            all_docs.append(doc)

    if not all_docs:
        print("⚠️ No valid documents to insert after cleaning.")
        return None

    db = PGVector.from_documents(
        documents=all_docs,
        embedding=embeddings,
        connection_string=DB_CONNECTION_STRING,
        collection_name=DB_COLLECTION_NAME
    )

    print(f"✅ Inserted {len(all_docs)} cleaned documents into pgvector collection '{DB_COLLECTION_NAME}'")
    return db


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
