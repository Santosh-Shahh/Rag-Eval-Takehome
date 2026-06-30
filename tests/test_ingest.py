import pytest
from pathlib import Path
from rag.ingest import recursive_character_splitter, parse_file

def test_recursive_character_splitter_basic():
    text = "Hello World. This is a simple test text for the recursive character splitter."
    chunks = recursive_character_splitter(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 0
    # Reconstructed text should cover all words (ignoring spacing artifacts)
    reconstructed = " ".join(chunks)
    assert "Hello" in reconstructed
    assert "splitter" in reconstructed

def test_recursive_character_splitter_large():
    text = "\n\n".join([f"This is paragraph {i}. It contains some sentences." for i in range(10)])
    chunks = recursive_character_splitter(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 5
    for c in chunks:
        assert len(c) <= 50

def test_parse_file_markdown(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.md"
    content = "# Test Document\nThis is a sample markdown file."
    p.write_text(content)
    
    text, meta = parse_file(p)
    assert "# Test Document" in text
    assert meta["file_name"] == "test.md"
    assert meta["file_type"] == "md"
