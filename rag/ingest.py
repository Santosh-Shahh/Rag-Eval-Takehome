import os
import argparse
import hashlib
import time
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP
from rag.store import LanceDBStore
from rag.embedding import EmbeddingClient

def recursive_character_splitter(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Splits text into chunks recursively, trying to break on paragraphs, 
    sentences, words, and finally characters, to keep semantic structure intact.
    """
    if not text:
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def split_recurse(text_to_split, separators):
        # Base case: if text is small enough, return it
        if len(text_to_split) <= chunk_size:
            return [text_to_split]
            
        if not separators:
            # Fallback: hard split by chunk_size
            return [text_to_split[i:i+chunk_size] for i in range(0, len(text_to_split), chunk_size)]
            
        separator = separators[0]
        splits = text_to_split.split(separator)
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for part in splits:
            part_len = len(part)
            
            # If the part itself is larger than chunk_size, split it with sub-separators
            if part_len > chunk_size:
                # First, flush current chunk
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                # Split the oversized part recursively
                sub_parts = split_recurse(part, separators[1:])
                chunks.extend(sub_parts)
                continue
                
            # If adding this part exceeds chunk_size, flush the current chunk
            if current_len + part_len + (len(separator) if current_chunk else 0) > chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                # To handle overlap, backtrack from current chunk if possible
                # Simple overlap implementation: take last few parts that fit within overlap
                overlap_chunk = []
                overlap_len = 0
                for old_part in reversed(current_chunk):
                    if overlap_len + len(old_part) + (len(separator) if overlap_chunk else 0) <= chunk_overlap:
                        overlap_chunk.insert(0, old_part)
                        overlap_len += len(old_part) + len(separator)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_len = overlap_len
            
            current_chunk.append(part)
            current_len += part_len + (len(separator) if len(current_chunk) > 1 else 0)
            
        if current_chunk:
            chunks.append(separator.join(current_chunk))
            
        return chunks

    return split_recurse(text, separators)

def parse_file(file_path: Path) -> tuple[str, dict]:
    """
    Parses a PDF, HTML, or MD file and returns its raw text content 
    along with basic metadata.
    """
    resolved_path = file_path.resolve()
    suffix = resolved_path.suffix.lower()
    metadata = {
        "file_name": resolved_path.name,
        "file_path": str(resolved_path),
        "file_type": suffix[1:] if suffix else "unknown",
        "file_size": resolved_path.stat().st_size,
    }
    
    text = ""
    
    if suffix == ".pdf":
        reader = PdfReader(file_path)
        pages_text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        text = "\n\n--- Page Break ---\n\n".join(pages_text)
        
    elif suffix in (".html", ".htm"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n")
            # Basic cleanup of extra whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            
    elif suffix in (".md", ".markdown", ".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
        
    return text, metadata

def ingest_file(file_path: Path, store: LanceDBStore, embedding_client: EmbeddingClient, 
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Ingests a single file: parses it, chunks it, computes embeddings, 
    and saves to LanceDB store. Operates idempotently by deleting old 
    chunks for this file path before adding new ones.
    """
    print(f"Parsing: {file_path.name}")
    text, metadata = parse_file(file_path)
    
    chunks = recursive_character_splitter(text, chunk_size, chunk_overlap)
    print(f"Generated {len(chunks)} chunks from {file_path.name}")
    
    if not chunks:
        return
        
    # Delete existing entries for this file_path to support idempotency and clean re-ingests
    store.delete_by_file_path(metadata["file_path"])
    
    records = []
    # To speed up embedding calls, batch embed them
    embeddings = embedding_client.get_embeddings(chunks)
    
    for i, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
        # Generate a unique hash based on file path and chunk text to guarantee idempotency
        hasher = hashlib.sha256()
        hasher.update(metadata["file_path"].encode('utf-8'))
        hasher.update(chunk_text.encode('utf-8'))
        chunk_id = hasher.hexdigest()
        
        record = {
            "id": chunk_id,
            "vector": vector,
            "text": chunk_text,
            "file_path": metadata["file_path"],
            "file_name": metadata["file_name"],
            "chunk_index": i,
            "file_type": metadata["file_type"],
            "created_at": time.time()
        }
        records.append(record)
        
    store.add_chunks(records)
    print(f"Ingested {len(records)} chunks from {file_path.name} into vector store.")

def ingest_directory(dir_path: Path, store: LanceDBStore, embedding_client: EmbeddingClient, 
                     chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Recursively scans directory and ingests supported files."""
    supported_extensions = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt"}
    
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Directory {dir_path} does not exist.")
        return
        
    files_to_ingest = []
    for ext in supported_extensions:
        files_to_ingest.extend(dir_path.rglob(f"*{ext}"))
        
    print(f"Found {len(files_to_ingest)} files to ingest.")
    
    for file_path in files_to_ingest:
        try:
            ingest_file(file_path, store, embedding_client, chunk_size, chunk_overlap)
        except Exception as e:
            print(f"Failed to ingest {file_path.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest files into RAG vector store.")
    parser.add_argument("--dir", type=str, help="Directory of documents to ingest.")
    parser.add_argument("--file", type=str, help="Single file to ingest.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Size of each text chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Overlap between chunks.")
    
    args = parser.parse_args()
    
    store = LanceDBStore()
    emb_client = EmbeddingClient()
    
    if args.dir:
        ingest_directory(Path(args.dir), store, emb_client, args.chunk_size, args.chunk_overlap)
    elif args.file:
        ingest_file(Path(args.file), store, emb_client, args.chunk_size, args.chunk_overlap)
    else:
        print("Please provide --dir or --file path.")
