import time
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from rag.config import DEFAULT_K, CHUNK_SIZE, CHUNK_OVERLAP
from rag.store import LanceDBStore
from rag.embedding import EmbeddingClient
from rag.ingest import ingest_directory, ingest_file
from rag.retriever import Retriever
from rag.generator import Generator

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG_Service")

app = FastAPI(title="Cost-Efficient RAG Service", version="1.0.0")

# Lazy load store, embedding client, retriever, and generator
store = LanceDBStore()
emb_client = EmbeddingClient()
retriever = Retriever(store, emb_client)
generator = Generator()

class IngestRequest(BaseModel):
    directory_path: Optional[str] = Field(None, description="Absolute or relative path to directory to ingest.")
    file_path: Optional[str] = Field(None, description="Absolute or relative path to single file to ingest.")
    chunk_size: int = Field(CHUNK_SIZE, description="Overriding chunk size.")
    chunk_overlap: int = Field(CHUNK_OVERLAP, description="Overriding chunk overlap.")

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user question.")
    k: int = Field(DEFAULT_K, description="Number of retrieved contexts to feed into LLM.")
    filter_expr: Optional[str] = Field(None, alias="filter", description="SQL-like metadata filter query.")

class CitationModel(BaseModel):
    file_name: str
    file_path: str
    chunk_index: int
    id: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationModel]
    retrieved_chunks: List[Dict[str, Any]]
    token_usage: Dict[str, int]
    retrieval_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "vector_count": store.count_vectors(),
        "embedding_provider": emb_client.provider,
        "llm_provider": generator.provider
    }

@app.post("/ingest")
def ingest(req: IngestRequest):
    """
    Ingests documents from a file or folder into the vector store.
    """
    start_time = time.time()
    initial_count = store.count_vectors()
    
    if req.directory_path:
        dir_path = Path(req.directory_path)
        if not dir_path.exists():
            raise HTTPException(status_code=404, detail=f"Directory '{req.directory_path}' not found.")
        ingest_directory(dir_path, store, emb_client, req.chunk_size, req.chunk_overlap)
    elif req.file_path:
        file_path = Path(req.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File '{req.file_path}' not found.")
        ingest_file(file_path, store, emb_client, req.chunk_size, req.chunk_overlap)
    else:
        # Default ingest data/corpus/ if exists
        default_dir = Path("./data/corpus")
        if default_dir.exists():
            ingest_directory(default_dir, store, emb_client, req.chunk_size, req.chunk_overlap)
        else:
            raise HTTPException(status_code=400, detail="Please specify directory_path or file_path in request.")
            
    final_count = store.count_vectors()
    elapsed_time = time.time() - start_time
    
    return {
        "message": "Ingestion completed successfully",
        "duration_seconds": round(elapsed_time, 2),
        "initial_vector_count": initial_count,
        "final_vector_count": final_count,
        "added_vectors": final_count - initial_count
    }

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Retrieves contexts and generates a grounded response.
    Logs latency, chunk count, and token usage.
    """
    start_time = time.time()
    
    # 1. Retrieval
    retrieval_start = time.time()
    retrieved_chunks = retriever.retrieve(req.query, k=req.k, filter_expr=req.filter_expr)
    retrieval_latency = int((time.time() - retrieval_start) * 1000)
    
    # 2. Generation
    generation_start = time.time()
    gen_result = generator.generate_answer(req.query, retrieved_chunks)
    generation_latency = gen_result["latency_ms"]
    
    total_latency_ms = int((time.time() - start_time) * 1000)
    
    # Structure retrieved chunks metadata to return to client safely (convert float similarity, etc)
    sanitized_chunks = []
    for chunk in retrieved_chunks:
        sanitized_chunks.append({
            "id": chunk.get("id"),
            "text": chunk.get("text"),
            "file_name": chunk.get("file_name"),
            "file_path": chunk.get("file_path"),
            "chunk_index": int(chunk.get("chunk_index", 0)),
            "similarity": float(chunk.get("similarity", 1.0))
        })
        
    # Log query metrics
    logger.info(
        f"Query: '{req.query}' | "
        f"Retrieval Latency: {retrieval_latency}ms | "
        f"Generation Latency: {generation_latency}ms | "
        f"Total Latency: {total_latency_ms}ms | "
        f"Chunks Retrieved: {len(retrieved_chunks)} | "
        f"Token Usage: {gen_result['token_usage']}"
    )
    
    return QueryResponse(
        query=req.query,
        answer=gen_result["answer"],
        citations=[
            CitationModel(
                file_name=c["file_name"],
                file_path=c["file_path"],
                chunk_index=c["chunk_index"],
                id=c["id"]
            ) for c in gen_result["citations"]
        ],
        retrieved_chunks=sanitized_chunks,
        token_usage=gen_result["token_usage"],
        retrieval_latency_ms=retrieval_latency,
        generation_latency_ms=generation_latency,
        total_latency_ms=total_latency_ms
    )
