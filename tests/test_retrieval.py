import pytest
import shutil
from pathlib import Path
from rag.store import LanceDBStore
from rag.embedding import EmbeddingClient
from rag.retriever import Retriever

@pytest.fixture
def temp_db(tmp_path):
    db_dir = tmp_path / "lancedb_test"
    store = LanceDBStore(db_path=str(db_dir), table_name="test_chunks")
    yield store
    # Cleanup lancedb files
    if db_dir.exists():
        shutil.rmtree(db_dir)

def test_store_add_and_search(temp_db):
    emb = EmbeddingClient() # Uses local offline fallback in test env (no GEMINI_API_KEY)
    retriever = Retriever(temp_db, emb)

    chunks = [
        {
            "id": "chunk1",
            "vector": emb.get_embedding("The default replication factor of AetherDB is 3."),
            "text": "The default replication factor of AetherDB is 3.",
            "file_path": "doc1.md",
            "file_name": "doc1.md",
            "chunk_index": 0,
            "file_type": "md"
        },
        {
            "id": "chunk2",
            "vector": emb.get_embedding("AetherDB nodes require minimum 8 GB of RAM and 4 cores."),
            "text": "AetherDB nodes require minimum 8 GB of RAM and 4 cores.",
            "file_path": "doc2.md",
            "file_name": "doc2.md",
            "chunk_index": 0,
            "file_type": "md"
        }
    ]

    # Add chunks
    temp_db.add_chunks(chunks)
    assert temp_db.count_vectors() == 2

    # Query 1
    results = retriever.retrieve("What is the replication factor?", k=1)
    assert len(results) == 1
    assert "replication" in results[0]["text"]
    assert results[0]["file_name"] == "doc1.md"

    # Query 2 with filter
    results_filtered = retriever.retrieve("minimum RAM", k=2, filter_expr="file_name = 'doc2.md'")
    assert len(results_filtered) == 1
    assert "RAM" in results_filtered[0]["text"]
    assert results_filtered[0]["file_name"] == "doc2.md"
